from __future__ import annotations

from dataclasses import asdict
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from chex_sae_fairness.config import ExperimentConfig
from chex_sae_fairness.data.feature_cache import load_or_create_feature_bundle
from chex_sae_fairness.data.splits import build_split_masks
from chex_sae_fairness.evaluation.disentanglement import (
    evaluate_disentanglement,
    reconstruction_metrics,
    summarize_latent_correlations,
)
from chex_sae_fairness.evaluation.fairness import evaluate_group_fairness, evaluate_multilabel_predictions
from chex_sae_fairness.mitigation.concept_debias import (
    apply_age_residualization,
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

    split_masks = build_split_masks(
        splits,
        valid_name=cfg.data.validation_split_name,
        test_name=cfg.data.test_split_name,
        context="feature bundle",
    )
    if split_masks.used_valid_as_test:
        logger.warning(
            "No '%s' split found. Using '%s' split as test and training SAE/probes on remaining rows.",
            cfg.data.test_split_name,
            cfg.data.validation_split_name,
        )

    x_train, x_valid, x_test = x[split_masks.train], x[split_masks.valid], x[split_masks.test]
    y_train, y_test = pathologies[split_masks.train], pathologies[split_masks.test]
    test_indices = np.where(split_masks.test)[0].astype(np.int64)

    metadata_train = metadata_df.loc[split_masks.train].reset_index(drop=True)
    metadata_test = metadata_df.loc[split_masks.test].reset_index(drop=True)

    age_groups_train = bundle["age_group"][split_masks.train].astype(str)
    age_groups_test = bundle["age_group"][split_masks.test].astype(str)

    resolved_device = _resolve_device(cfg.features.device)
    if resolved_device != cfg.features.device:
        logger.warning(
            "Requested device '%s' unavailable; using '%s' instead.",
            cfg.features.device,
            resolved_device,
        )
    logger.info(
        "Split sizes: train=%d, valid=%d, test=%d",
        int(split_masks.train.sum()),
        int(split_masks.valid.sum()),
        int(split_masks.test.sum()),
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
    baseline_scores, baseline_perf, baseline_fairness = _fit_and_evaluate_probe(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        age_groups_test=age_groups_test,
        pathology_cols=path_cols,
        cfg=cfg,
    )

    logger.info("Training probe on SAE concept features.")
    concept_scores, concept_perf, concept_fairness = _fit_and_evaluate_probe(
        x_train=z_train,
        x_test=z_test,
        y_train=y_train,
        y_test=y_test,
        age_groups_test=age_groups_test,
        pathology_cols=path_cols,
        cfg=cfg,
    )

    logger.info("Fitting concept residualizer for age-group debiasing.")
    residualizer = fit_concept_residualizer(
        z_train,
        age_groups=age_groups_train,
        strength=cfg.fairness.debias_strength,
    )
    z_train_debiased, z_test_debiased = apply_age_residualization(
        residualizer=residualizer,
        z_train=z_train,
        z_test=z_test,
        age_groups_train=age_groups_train,
        age_groups_test=age_groups_test,
        mode=cfg.fairness.debias_mode,
    )

    logger.info("Training probe on debiased SAE concepts (%s).", cfg.fairness.debias_mode)
    debiased_scores, debiased_perf, debiased_fairness = _fit_and_evaluate_probe(
        x_train=z_train_debiased,
        x_test=z_test_debiased,
        y_train=y_train,
        y_test=y_test,
        age_groups_test=age_groups_test,
        pathology_cols=path_cols,
        cfg=cfg,
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
    latent_activity = _summarize_latent_activity(z_train, latent_correlations)

    recon = reconstruction_metrics(x_test, x_hat_test)
    age_assoc = rank_age_associated_concepts(z_train, age_groups_train, top_k=25)

    report = {
        "config_path": str(Path(config_path).resolve()),
        "config": asdict(cfg),
        "counts": {
            "n_total": int(len(splits)),
            "n_train": int(split_masks.train.sum()),
            "n_valid": int(split_masks.valid.sum()),
            "n_test": int(split_masks.test.sum()),
            "used_valid_as_test": bool(split_masks.used_valid_as_test),
            "manifest_rows_dropped": int(feature_result.manifest_rows_dropped),
            "used_cached_features": bool(feature_result.used_cache),
        },
        "debiasing": {
            "method": "age_group_residualization",
            "mode": cfg.fairness.debias_mode,
            "strength": cfg.fairness.debias_strength,
            "fairness_threshold": cfg.fairness.threshold,
        },
        "sae": {
            "train_curve": sae_output.train_curve,
            "valid_curve": sae_output.valid_curve,
            "reconstruction": recon,
            "latent_activity": latent_activity,
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

    np.savez_compressed(
        cfg.study_predictions_path,
        y_true=y_test.astype(np.float32),
        age_groups=age_groups_test.astype(str),
        pathology_cols=np.array(path_cols, dtype=object),
        baseline_scores=baseline_scores.astype(np.float32),
        concept_scores=concept_scores.astype(np.float32),
        debiased_scores=debiased_scores.astype(np.float32),
        test_indices=test_indices,
        test_view_type=(
            bundle["view_type"][split_masks.test].astype(str)
            if "view_type" in bundle
            else np.array([], dtype=object)
        ),
        test_patient_id=(
            bundle["patient_id"][split_masks.test].astype(str)
            if "patient_id" in bundle
            else np.array([], dtype=object)
        ),
        test_image_path=(
            bundle["image_path"][split_masks.test].astype(str)
            if "image_path" in bundle
            else np.array([], dtype=object)
        ),
    )

    write_json(report, cfg.study_metrics_path)
    logger.info(
        "Saved study report to %s and prediction bundle to %s (elapsed %.1fs).",
        cfg.study_metrics_path,
        cfg.study_predictions_path,
        time.perf_counter() - start_time,
    )
    return report


def _fit_and_evaluate_probe(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    age_groups_test: np.ndarray,
    pathology_cols: list[str],
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    probe = fit_multilabel_probe(
        x_train,
        y_train,
        max_iter=cfg.probes.max_iter,
        c_value=cfg.probes.c_value,
    )
    scores = probe.predict_proba(x_test)
    performance = evaluate_multilabel_predictions(
        y_test,
        scores,
        pathology_cols,
        threshold=cfg.fairness.threshold,
    )
    fairness = evaluate_group_fairness(
        y_true=y_test,
        y_score=scores,
        groups=age_groups_test,
        label_names=pathology_cols,
        threshold=cfg.fairness.threshold,
        bootstrap_samples=cfg.fairness.bootstrap_samples,
    )
    return scores.astype(np.float32), performance, fairness


def _summarize_latent_activity(z_train: np.ndarray, correlations: dict[str, object]) -> dict[str, float]:
    if z_train.size == 0:
        return {
            "death_rate": float("nan"),
            "mean_active_per_sample": float("nan"),
            "reuse_ratio": float("nan"),
            "max_concepts_per_feature": float("nan"),
        }

    active_mask = z_train > 1e-6
    activity_rate = active_mask.mean(axis=0)
    death_rate = float(np.mean(activity_rate <= 1e-4))
    mean_active_per_sample = float(np.mean(active_mask.sum(axis=1)))

    concept_rows = correlations.get("concepts", []) if isinstance(correlations, dict) else []
    latent_counts: dict[int, int] = {}
    if isinstance(concept_rows, list):
        for row in concept_rows:
            if not isinstance(row, dict):
                continue
            idx = row.get("latent_index")
            if isinstance(idx, (int, np.integer)) and int(idx) >= 0:
                latent_counts[int(idx)] = latent_counts.get(int(idx), 0) + 1

    if latent_counts:
        reused = [count for count in latent_counts.values() if count > 1]
        reuse_ratio = float(len(reused) / len(latent_counts))
        max_concepts_per_feature = float(max(latent_counts.values()))
    else:
        reuse_ratio = float("nan")
        max_concepts_per_feature = float("nan")

    return {
        "death_rate": death_rate,
        "mean_active_per_sample": mean_active_per_sample,
        "reuse_ratio": reuse_ratio,
        "max_concepts_per_feature": max_concepts_per_feature,
    }


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
