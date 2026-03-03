from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from chex_sae_fairness.config import ExperimentConfig
from chex_sae_fairness.data.chexpert_plus import audit_png_layout, build_manifest, save_manifest
from chex_sae_fairness.data.feature_cache import load_or_create_feature_bundle
from chex_sae_fairness.data.splits import build_split_masks
from chex_sae_fairness.models.chexagent_features import load_feature_bundle
from chex_sae_fairness.pipeline import run_full_study
from chex_sae_fairness.sweep import run_sae_sweep
from chex_sae_fairness.training.train_sae import train_sae_model
from chex_sae_fairness.utils.io import write_json
from chex_sae_fairness.utils.logging import configure_logging
from chex_sae_fairness.utils.repro import seed_everything

logger = logging.getLogger(__name__)


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console/file logging level",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to <output_root>/logs/<command>.log",
    )


def _init_logging(args: argparse.Namespace, cfg: ExperimentConfig, command_name: str) -> Path | None:
    cfg.ensure_output_dirs()
    log_path = (
        Path(args.log_file).expanduser()
        if args.log_file
        else cfg.output_root / "logs" / f"{command_name}.log"
    )
    resolved_path = configure_logging(level=args.log_level, log_file=log_path)
    logger.info("Starting %s", command_name)
    logger.info("Config: %s", Path(args.config).resolve())
    if resolved_path is not None:
        logger.info("Log file: %s", resolved_path)
    return resolved_path


def prepare_manifest_cli() -> None:
    parser = argparse.ArgumentParser(description="Build a cleaned manifest from CheXpert Plus metadata")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    _add_logging_args(parser)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    _init_logging(args, cfg, "prepare_manifest")

    result = build_manifest(cfg)
    save_manifest(result.manifest, cfg.manifest_path)

    logger.info("Manifest saved: %s", cfg.manifest_path)
    logger.info("Rows retained: %d, dropped during cleaning: %d", len(result.manifest), result.dropped_rows)


def extract_features_cli() -> None:
    parser = argparse.ArgumentParser(description="Extract CheXagent vision features")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--force", action="store_true", help="Recompute features even if cache exists")
    _add_logging_args(parser)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    _init_logging(args, cfg, "extract_features")

    feature_result = load_or_create_feature_bundle(
        cfg,
        force_recompute=args.force or cfg.features.force_recompute,
    )
    features = feature_result.bundle["features"]

    logger.info("Features saved: %s", cfg.feature_path)
    logger.info("Feature shape: %s", tuple(features.shape))
    logger.info("Used cache: %s", feature_result.used_cache)


def train_sae_cli() -> None:
    parser = argparse.ArgumentParser(description="Train sparse autoencoder from extracted features")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    _add_logging_args(parser)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    _init_logging(args, cfg, "train_sae")
    seed_everything(cfg.seed)

    bundle = load_feature_bundle(str(cfg.feature_path))
    x = bundle["features"].astype(np.float32)
    splits = bundle["split"].astype(str)
    has_valid_split = bool((splits == cfg.data.validation_split_name).any())

    split_masks = build_split_masks(
        splits,
        valid_name=cfg.data.validation_split_name,
        test_name=cfg.data.test_split_name,
        context="feature bundle",
        require_test=False,
    )
    if not has_valid_split:
        logger.warning(
            "No '%s' split found; using train rows for SAE validation metrics.",
            cfg.data.validation_split_name,
        )

    device = cfg.features.device if cfg.features.device == "cpu" or torch.cuda.is_available() else "cpu"
    logger.info(
        "Training SAE on %d samples (valid=%d) with feature_dim=%d on device=%s",
        int(split_masks.train.sum()),
        int(split_masks.valid.sum()),
        int(x.shape[1]),
        device,
    )

    output = train_sae_model(
        train_features=x[split_masks.train],
        valid_features=x[split_masks.valid],
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

    logger.info("SAE checkpoint saved: %s", cfg.sae_checkpoint_path)
    logger.info("Training curve saved: %s", curve_path)


def run_study_cli() -> None:
    parser = argparse.ArgumentParser(description="Run full CheXagent SAE fairness study")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    _add_logging_args(parser)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    _init_logging(args, cfg, "run_study")
    report = run_full_study(args.config)
    logger.info("Study finished.")
    logger.info("Report path: %s", cfg.study_metrics_path)
    counts = report.get("counts", {})
    if isinstance(counts, dict):
        logger.info(
            "Split counts: total=%s train=%s valid=%s test=%s",
            counts.get("n_total"),
            counts.get("n_train"),
            counts.get("n_valid"),
            counts.get("n_test"),
        )
    baseline = report.get("baseline_feature_probe", {})
    debiased = report.get("sae_concept_probe_debiased", {})
    if isinstance(baseline, dict) and isinstance(debiased, dict):
        baseline_perf = baseline.get("performance", {})
        debiased_perf = debiased.get("performance", {})
        baseline_fair = baseline.get("fairness", {})
        debiased_fair = debiased.get("fairness", {})
        logger.info(
            "Baseline macro AUROC=%.4f | Debiased macro AUROC=%.4f",
            float(baseline_perf.get("macro_auroc", float("nan"))),
            float(debiased_perf.get("macro_auroc", float("nan"))),
        )
        logger.info(
            "Baseline worst-group macro AUROC=%s | Debiased worst-group macro AUROC=%s",
            _format_worst_group_metric(baseline_fair.get("worst_group_macro_auroc")),
            _format_worst_group_metric(debiased_fair.get("worst_group_macro_auroc")),
        )
        logger.info(
            "Baseline worst-group macro accuracy=%s | Debiased worst-group macro accuracy=%s",
            _format_worst_group_metric(baseline_fair.get("worst_group_macro_accuracy")),
            _format_worst_group_metric(debiased_fair.get("worst_group_macro_accuracy")),
        )


def run_sae_sweep_cli() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-run SAE benchmark and comparison plots")
    parser.add_argument("--base-config", required=True, help="Path to base experiment YAML config")
    parser.add_argument("--sweep-config", required=True, help="Path to sweep YAML with SAE runs")
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Recompute CheXagent features even if cache exists",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console/file logging level",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to <output_root>/logs/run_sae_sweep.log",
    )
    args = parser.parse_args()

    base_cfg = ExperimentConfig.from_yaml(args.base_config)
    base_cfg.ensure_output_dirs()
    log_path = (
        Path(args.log_file).expanduser()
        if args.log_file
        else base_cfg.output_root / "logs" / "run_sae_sweep.log"
    )
    resolved_log_path = configure_logging(level=args.log_level, log_file=log_path)
    logger.info("Starting run_sae_sweep")
    logger.info("Base config: %s", Path(args.base_config).resolve())
    logger.info("Sweep config: %s", Path(args.sweep_config).resolve())
    if resolved_log_path is not None:
        logger.info("Log file: %s", resolved_log_path)

    summary = run_sae_sweep(
        base_config_path=args.base_config,
        sweep_config_path=args.sweep_config,
        force_recompute_features=args.force_features,
    )
    logger.info("SAE sweep finished.")
    logger.info("Summary JSON: %s/summary.json", summary["output_dir"])
    logger.info("Summary CSV: %s", summary["summary_csv"])


def audit_data_cli() -> None:
    parser = argparse.ArgumentParser(description="Audit PNG path resolution and metadata wiring")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--sample-size", type=int, default=2000, help="Number of rows to sample")
    _add_logging_args(parser)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    _init_logging(args, cfg, "audit_data")
    report = audit_png_layout(cfg, sample_size=args.sample_size)

    logger.info("Dataset audit summary:")
    logger.info("  Total metadata rows: %s", report["n_total_rows"])
    logger.info("  Sampled rows: %s", report["n_sampled"])
    logger.info("  Resolved image paths: %s", report["n_resolved"])
    logger.info("  Resolve rate: %.4f", report["resolve_rate"])
    logger.info("  image_root: %s", report["image_root"])
    logger.info("  has train/: %s", report["has_train_dir"])
    logger.info("  has val/: %s", report["has_val_dir"])
    logger.info("  has PNG/train/: %s", report["has_png_train_dir"])
    logger.info("  has PNG/val/: %s", report["has_png_val_dir"])
    if report["chunk_dirs_detected"]:
        logger.info("  Detected extracted chunk directories:")
        for name in report["chunk_dirs_detected"][:20]:
            logger.info("    - %s", name)
    if report["png_chunk_zips_in_root"] or report["png_chunk_zips_in_png_dir"]:
        logger.info("  Detected PNG zip chunks (extract before training):")
        for name in report["png_chunk_zips_in_root"]:
            logger.info("    root/%s", name)
        for name in report["png_chunk_zips_in_png_dir"]:
            logger.info("    PNG/%s", name)
    if report["unresolved_examples"]:
        logger.info("  Example unresolved metadata paths:")
        for value in report["unresolved_examples"]:
            logger.info("    - %s", value)


def _format_worst_group_metric(metric: object) -> str:
    if isinstance(metric, dict):
        group = metric.get("group")
        value = metric.get("value")
        if group is not None and isinstance(value, (int, float, np.floating)):
            return f"{group}:{float(value):.4f}"
    return "n/a"


if __name__ == "__main__":
    run_study_cli()
