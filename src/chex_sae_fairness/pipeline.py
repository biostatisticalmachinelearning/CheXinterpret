from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from chex_sae_fairness.config import ExperimentConfig
from chex_sae_fairness.data.feature_cache import load_or_create_feature_bundle
from chex_sae_fairness.evaluation.disentanglement import (
    evaluate_disentanglement,
    reconstruction_metrics,
    summarize_latent_correlations,
)
from chex_sae_fairness.evaluation.fairness import evaluate_group_fairness, evaluate_multilabel_predictions
from chex_sae_fairness.mitigation.concept_debias import (
    fit_concept_residualizer,
    rank_age_associated_concepts,
)
from chex_sae_fairness.models.sae import SparseAutoencoder
from chex_sae_fairness.training.train_probe import fit_multilabel_probe
from chex_sae_fairness.training.train_sae import encode_features, train_sae_model
from chex_sae_fairness.utils.io import write_json
from chex_sae_fairness.utils.repro import seed_everything

logger = logging.getLogger(__name__)


def run_full_study(config_path: str) -> dict[str, object]:
    start_time = time.perf_counter()
    cfg = ExperimentConfig.from_yaml(config_path)
    cfg.ensure_output_dirs()
    seed_everything(cfg.seed)

    logger.info("Loading or creating CheXagent feature bundle (cache-first).")
    feature_result = load_or_create_feature_bundle(cfg, force_recompute=cfg.features.force_recompute)
    bundle = feature_result.bundle
    splits = bundle["split"].astype(str)
    pathologies = bundle["y_pathology"].astype(np.float32)
    x = bundle["features"].astype(np.float32)
    logger.info(
        "Feature bundle ready: n_samples=%d, feature_dim=%d, used_cache=%s",
        x.shape[0],
        x.shape[1] if x.ndim == 2 else -1,
        feature_result.used_cache,
    )

    path_cols = [str(c) for c in bundle["pathology_cols"].tolist()]
    metadata_cols = [str(c) for c in bundle["metadata_cols"].tolist()]
    metadata_df = pd.DataFrame(bundle["metadata"], columns=metadata_cols)

    train_mask, valid_mask, test_mask = _split_masks(
        splits,
        valid_name=cfg.data.validation_split_name,
        test_name=cfg.data.test_split_name,
    )

    x_train, x_valid, x_test = x[train_mask], x[valid_mask], x[test_mask]
    y_train, y_test = pathologies[train_mask], pathologies[test_mask]

    metadata_train = metadata_df.loc[train_mask].reset_index(drop=True)
    metadata_test = metadata_df.loc[test_mask].reset_index(drop=True)

    age_groups_train = bundle["age_group"][train_mask].astype(str)
    age_groups_test = bundle["age_group"][test_mask].astype(str)

    resolved_device = _resolve_device(cfg.features.device)
    if resolved_device != cfg.features.device:
        logger.warning(
            "Requested device '%s' unavailable; using '%s' instead.",
            cfg.features.device,
            resolved_device,
        )
    logger.info(
        "Split sizes: train=%d, valid=%d, test=%d",
        int(train_mask.sum()),
        int(valid_mask.sum()),
        int(test_mask.sum()),
    )
    logger.info(
        "Training SAE (variant=%s, latent_dim=%d, epochs=%d, batch_size=%d, device=%s).",
        cfg.sae.variant,
        cfg.sae.latent_dim,
        cfg.sae.epochs,
        cfg.sae.batch_size,
        resolved_device,
    )
    sae_output = train_sae_model(
        train_features=x_train,
        valid_features=x_valid,
        cfg=cfg.sae,
        device=resolved_device,
    )
    logger.info("SAE training complete. Saving checkpoint to %s", cfg.sae_checkpoint_path)

    torch.save(
        {
            "state_dict": sae_output.model.state_dict(),
            "input_dim": int(x.shape[1]),
            "latent_dim": int(cfg.sae.latent_dim),
            "variant": str(cfg.sae.variant),
            "topk_k": int(cfg.sae.topk_k),
        },
        cfg.sae_checkpoint_path,
    )

    logger.info("Encoding train/test features into SAE latent space.")
    z_train = encode_features(
        model=sae_output.model,
        features=x_train,
        batch_size=max(256, cfg.sae.batch_size),
        device=resolved_device,
    )
    z_test = encode_features(
        model=sae_output.model,
        features=x_test,
        batch_size=max(256, cfg.sae.batch_size),
        device=resolved_device,
    )

    logger.info("Reconstructing test features for reconstruction metrics.")
    x_hat_test = _reconstruct_features(
        model=sae_output.model,
        features=x_test,
        batch_size=max(256, cfg.sae.batch_size),
        device=resolved_device,
    )

    logger.info("Training baseline probe on raw CheXagent features.")
    baseline_probe = fit_multilabel_probe(x_train, y_train)
    baseline_scores = baseline_probe.predict_proba(x_test)

    logger.info("Training probe on SAE concept features.")
    concept_probe = fit_multilabel_probe(z_train, y_train)
    concept_scores = concept_probe.predict_proba(z_test)

    logger.info("Fitting concept residualizer for age-group debiasing.")
    residualizer = fit_concept_residualizer(
        z_train,
        age_groups=age_groups_train,
        strength=cfg.fairness.debias_strength,
    )
    z_train_debiased = residualizer.transform(z_train, age_groups_train)
    z_test_debiased = residualizer.transform(z_test, age_groups_test)

    debiased_probe = fit_multilabel_probe(z_train_debiased, y_train)
    debiased_scores = debiased_probe.predict_proba(z_test_debiased)

    logger.info("Evaluating pathology performance and age-group fairness metrics.")
    baseline_perf = evaluate_multilabel_predictions(y_test, baseline_scores, path_cols)
    concept_perf = evaluate_multilabel_predictions(y_test, concept_scores, path_cols)
    debiased_perf = evaluate_multilabel_predictions(y_test, debiased_scores, path_cols)

    baseline_fairness = evaluate_group_fairness(
        y_true=y_test,
        y_score=baseline_scores,
        groups=age_groups_test,
        label_names=path_cols,
        threshold=cfg.fairness.threshold,
        bootstrap_samples=cfg.fairness.bootstrap_samples,
    )
    concept_fairness = evaluate_group_fairness(
        y_true=y_test,
        y_score=concept_scores,
        groups=age_groups_test,
        label_names=path_cols,
        threshold=cfg.fairness.threshold,
        bootstrap_samples=cfg.fairness.bootstrap_samples,
    )
    debiased_fairness = evaluate_group_fairness(
        y_true=y_test,
        y_score=debiased_scores,
        groups=age_groups_test,
        label_names=path_cols,
        threshold=cfg.fairness.threshold,
        bootstrap_samples=cfg.fairness.bootstrap_samples,
    )

    logger.info("Running disentanglement and latent-correlation analyses.")
    disentanglement = evaluate_disentanglement(
        z_train=z_train,
        z_test=z_test,
        y_path_train=y_train,
        y_path_test=y_test,
        pathology_cols=path_cols,
        metadata_train=metadata_train,
        metadata_test=metadata_test,
        metadata_cols=metadata_cols,
    )
    latent_correlations = summarize_latent_correlations(
        z=z_test,
        y_pathology=y_test,
        pathology_cols=path_cols,
        metadata=metadata_test,
        metadata_cols=metadata_cols,
    )

    recon = reconstruction_metrics(x_test, x_hat_test)
    age_assoc = rank_age_associated_concepts(z_train, age_groups_train, top_k=25)

    report = {
        "config_path": str(Path(config_path).resolve()),
        "counts": {
            "n_total": int(len(splits)),
            "n_train": int(train_mask.sum()),
            "n_valid": int(valid_mask.sum()),
            "n_test": int(test_mask.sum()),
            "manifest_rows_dropped": int(feature_result.manifest_rows_dropped),
            "used_cached_features": bool(feature_result.used_cache),
        },
        "sae": {
            "train_curve": sae_output.train_curve,
            "valid_curve": sae_output.valid_curve,
            "reconstruction": recon,
        },
        "baseline_feature_probe": {
            "performance": baseline_perf,
            "fairness": baseline_fairness,
        },
        "sae_concept_probe": {
            "performance": concept_perf,
            "fairness": concept_fairness,
        },
        "sae_concept_probe_debiased": {
            "performance": debiased_perf,
            "fairness": debiased_fairness,
        },
        "disentanglement": disentanglement,
        "latent_correlations": latent_correlations,
        "age_associated_latents": age_assoc,
    }

    write_json(report, cfg.study_metrics_path)
    logger.info(
        "Saved study report to %s (elapsed %.1fs).",
        cfg.study_metrics_path,
        time.perf_counter() - start_time,
    )
    return report


def _split_masks(
    splits: np.ndarray,
    valid_name: str,
    test_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_mask = splits == test_name
    valid_mask = splits == valid_name
    train_mask = ~(test_mask | valid_mask)

    if valid_mask.sum() == 0:
        valid_mask = train_mask.copy()

    if test_mask.sum() == 0:
        raise ValueError(
            f"No rows found for test split '{test_name}'. Check `schema.split_col` and split values."
        )

    if train_mask.sum() == 0:
        raise ValueError("No rows left for train split after removing valid/test rows.")

    return train_mask, valid_mask, test_mask


def _resolve_device(requested: str) -> str:
    return requested if requested == "cpu" or torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _reconstruct_features(
    model: SparseAutoencoder,
    features: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    tensor = torch.tensor(features, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    model.eval().to(device)
    outputs: list[np.ndarray] = []

    for (batch,) in loader:
        batch = batch.to(device)
        x_hat, _ = model(batch)
        outputs.append(x_hat.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(outputs, axis=0)
