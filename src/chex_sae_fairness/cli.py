from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from chex_sae_fairness.config import ExperimentConfig
from chex_sae_fairness.data.chexpert_plus import build_manifest, load_manifest, save_manifest
from chex_sae_fairness.models.chexagent_features import (
    CheXagentVisionFeatureExtractor,
    FeatureExtractionConfig,
    load_feature_bundle,
    save_feature_bundle,
)
from chex_sae_fairness.pipeline import run_full_study
from chex_sae_fairness.training.train_sae import train_sae_model
from chex_sae_fairness.utils.io import write_json
from chex_sae_fairness.utils.repro import seed_everything


def prepare_manifest_cli() -> None:
    parser = argparse.ArgumentParser(description="Build a cleaned manifest from CheXpert Plus metadata")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.ensure_output_dirs()

    result = build_manifest(cfg)
    save_manifest(result.manifest, cfg.manifest_path)

    print(f"Manifest saved: {cfg.manifest_path}")
    print(f"Rows: {len(result.manifest)}, dropped due to cleaning: {result.dropped_rows}")


def extract_features_cli() -> None:
    parser = argparse.ArgumentParser(description="Extract CheXagent vision features")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--build-manifest-if-missing",
        action="store_true",
        help="Build manifest automatically when missing",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.ensure_output_dirs()

    manifest_path = cfg.manifest_path
    if not manifest_path.exists():
        if not args.build_manifest_if_missing:
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. Run chex-prepare-manifest first "
                "or pass --build-manifest-if-missing."
            )
        result = build_manifest(cfg)
        save_manifest(result.manifest, manifest_path)

    manifest = load_manifest(manifest_path)

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

    features = extractor.extract_from_manifest(manifest)
    save_feature_bundle(
        output_path=str(cfg.feature_path),
        features=features,
        manifest=manifest,
        split_col=cfg.schema.split_col,
        pathology_cols=cfg.schema.pathology_cols,
        metadata_cols=cfg.schema.metadata_cols,
        age_col=cfg.schema.age_col,
    )

    print(f"Features saved: {cfg.feature_path}")
    print(f"Feature shape: {features.shape}")


def train_sae_cli() -> None:
    parser = argparse.ArgumentParser(description="Train sparse autoencoder from extracted features")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    bundle = load_feature_bundle(str(cfg.feature_path))
    x = bundle["features"].astype(np.float32)
    splits = bundle["split"].astype(str)

    test_mask = splits == cfg.data.test_split_name
    valid_mask = splits == cfg.data.validation_split_name
    train_mask = ~(test_mask | valid_mask)
    if valid_mask.sum() == 0:
        valid_mask = train_mask.copy()

    device = cfg.features.device if cfg.features.device == "cpu" or torch.cuda.is_available() else "cpu"

    output = train_sae_model(
        train_features=x[train_mask],
        valid_features=x[valid_mask],
        cfg=cfg.sae,
        device=device,
    )

    torch.save(
        {
            "state_dict": output.model.state_dict(),
            "input_dim": int(x.shape[1]),
            "latent_dim": int(cfg.sae.latent_dim),
        },
        cfg.sae_checkpoint_path,
    )

    curve_path = Path(cfg.paths.output_root) / "sae_curve.json"
    write_json({"train": output.train_curve, "valid": output.valid_curve}, curve_path)

    print(f"SAE checkpoint saved: {cfg.sae_checkpoint_path}")
    print(f"Training curve saved: {curve_path}")


def run_study_cli() -> None:
    parser = argparse.ArgumentParser(description="Run full CheXagent SAE fairness study")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    run_full_study(args.config)
    print("Study finished.")
    print(f"Report path: {cfg.study_metrics_path}")


if __name__ == "__main__":
    run_study_cli()
