from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from chex_sae_fairness.config import ExperimentConfig
from chex_sae_fairness.data.chexpert_plus import build_manifest, save_manifest
from chex_sae_fairness.evaluation.disentanglement import evaluate_disentanglement, reconstruction_metrics
from chex_sae_fairness.evaluation.fairness import evaluate_group_fairness, evaluate_multilabel_predictions
from chex_sae_fairness.mitigation.concept_debias import (
    fit_concept_residualizer,
    rank_age_associated_concepts,
)
from chex_sae_fairness.models.chexagent_features import (
    CheXagentVisionFeatureExtractor,
    FeatureExtractionConfig,
    load_feature_bundle,
    save_feature_bundle,
)
from chex_sae_fairness.models.sae import SparseAutoencoder
from chex_sae_fairness.training.train_probe import fit_multilabel_probe
from chex_sae_fairness.training.train_sae import encode_features, train_sae_model
from chex_sae_fairness.utils.io import write_json
from chex_sae_fairness.utils.repro import seed_everything


def run_full_study(config_path: str) -> dict[str, object]:
    cfg = ExperimentConfig.from_yaml(config_path)
    cfg.ensure_output_dirs()
    seed_everything(cfg.seed)

    manifest_result = build_manifest(cfg)
    save_manifest(manifest_result.manifest, cfg.manifest_path)

    extractor = CheXagentVisionFeatureExtractor(
        FeatureExtractionConfig(
            model_name=cfg.features.model_name,
            device=cfg.features.device,
            batch_size=cfg.features.batch_size,
            num_workers=cfg.features.num_workers,
            precision=cfg.features.precision,
            pooling=cfg.features.pooling,
        )
    )
    features = extractor.extract_from_manifest(manifest_result.manifest)

    save_feature_bundle(
        output_path=str(cfg.feature_path),
        features=features,
        manifest=manifest_result.manifest,
        split_col=cfg.schema.split_col,
        pathology_cols=cfg.schema.pathology_cols,
        metadata_cols=cfg.schema.metadata_cols,
        age_col=cfg.schema.age_col,
    )

    bundle = load_feature_bundle(str(cfg.feature_path))
    splits = bundle["split"].astype(str)
    pathologies = bundle["y_pathology"].astype(np.float32)
    x = bundle["features"].astype(np.float32)

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

    sae_output = train_sae_model(
        train_features=x_train,
        valid_features=x_valid,
        cfg=cfg.sae,
        device=_resolve_device(cfg.features.device),
    )

    torch.save(
        {
            "state_dict": sae_output.model.state_dict(),
            "input_dim": int(x.shape[1]),
            "latent_dim": int(cfg.sae.latent_dim),
        },
        cfg.sae_checkpoint_path,
    )

    z_train = encode_features(
        model=sae_output.model,
        features=x_train,
        batch_size=max(256, cfg.sae.batch_size),
        device=_resolve_device(cfg.features.device),
    )
    z_test = encode_features(
        model=sae_output.model,
        features=x_test,
        batch_size=max(256, cfg.sae.batch_size),
        device=_resolve_device(cfg.features.device),
    )

    x_hat_test = _reconstruct_features(
        model=sae_output.model,
        features=x_test,
        batch_size=max(256, cfg.sae.batch_size),
        device=_resolve_device(cfg.features.device),
    )

    baseline_probe = fit_multilabel_probe(x_train, y_train)
    baseline_scores = baseline_probe.predict_proba(x_test)

    concept_probe = fit_multilabel_probe(z_train, y_train)
    concept_scores = concept_probe.predict_proba(z_test)

    residualizer = fit_concept_residualizer(
        z_train,
        age_groups=age_groups_train,
        strength=cfg.fairness.debias_strength,
    )
    z_train_debiased = residualizer.transform(z_train, age_groups_train)
    z_test_debiased = residualizer.transform(z_test, age_groups_test)

    debiased_probe = fit_multilabel_probe(z_train_debiased, y_train)
    debiased_scores = debiased_probe.predict_proba(z_test_debiased)

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

    recon = reconstruction_metrics(x_test, x_hat_test)
    age_assoc = rank_age_associated_concepts(z_train, age_groups_train, top_k=25)

    report = {
        "config_path": str(Path(config_path).resolve()),
        "counts": {
            "n_total": int(len(manifest_result.manifest)),
            "n_train": int(train_mask.sum()),
            "n_valid": int(valid_mask.sum()),
            "n_test": int(test_mask.sum()),
            "manifest_rows_dropped": int(manifest_result.dropped_rows),
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
        "age_associated_latents": age_assoc,
    }

    write_json(report, cfg.study_metrics_path)
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
