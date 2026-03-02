from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from chex_sae_fairness.config import ExperimentConfig
from chex_sae_fairness.data.chexpert_plus import audit_png_layout, build_manifest, save_manifest
from chex_sae_fairness.data.feature_cache import load_or_create_feature_bundle
from chex_sae_fairness.models.chexagent_features import load_feature_bundle
from chex_sae_fairness.pipeline import run_full_study
from chex_sae_fairness.sweep import run_sae_sweep
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
    parser.add_argument("--force", action="store_true", help="Recompute features even if cache exists")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.ensure_output_dirs()

    feature_result = load_or_create_feature_bundle(
        cfg,
        force_recompute=args.force or cfg.features.force_recompute,
    )
    features = feature_result.bundle["features"]

    print(f"Features saved: {cfg.feature_path}")
    print(f"Feature shape: {features.shape}")
    print(f"Used cache: {feature_result.used_cache}")


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
            "variant": str(cfg.sae.variant),
            "topk_k": int(cfg.sae.topk_k),
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


def run_sae_sweep_cli() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-run SAE benchmark and comparison plots")
    parser.add_argument("--base-config", required=True, help="Path to base experiment YAML config")
    parser.add_argument("--sweep-config", required=True, help="Path to sweep YAML with SAE runs")
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Recompute CheXagent features even if cache exists",
    )
    args = parser.parse_args()

    summary = run_sae_sweep(
        base_config_path=args.base_config,
        sweep_config_path=args.sweep_config,
        force_recompute_features=args.force_features,
    )
    print("SAE sweep finished.")
    print(f"Summary JSON: {summary['output_dir']}/summary.json")
    print(f"Summary CSV: {summary['summary_csv']}")


def audit_data_cli() -> None:
    parser = argparse.ArgumentParser(description="Audit PNG path resolution and metadata wiring")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--sample-size", type=int, default=2000, help="Number of rows to sample")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    report = audit_png_layout(cfg, sample_size=args.sample_size)

    print("Dataset audit summary:")
    print(f"  Total metadata rows: {report['n_total_rows']}")
    print(f"  Sampled rows: {report['n_sampled']}")
    print(f"  Resolved image paths: {report['n_resolved']}")
    print(f"  Resolve rate: {report['resolve_rate']:.4f}")
    print(f"  image_root: {report['image_root']}")
    print(f"  has train/: {report['has_train_dir']}")
    print(f"  has PNG/train/: {report['has_png_train_dir']}")
    if report["png_chunk_zips_in_root"] or report["png_chunk_zips_in_png_dir"]:
        print("  Detected PNG zip chunks (extract before training):")
        for name in report["png_chunk_zips_in_root"]:
            print(f"    root/{name}")
        for name in report["png_chunk_zips_in_png_dir"]:
            print(f"    PNG/{name}")
    if report["unresolved_examples"]:
        print("  Example unresolved metadata paths:")
        for value in report["unresolved_examples"]:
            print(f"    - {value}")


if __name__ == "__main__":
    run_study_cli()
