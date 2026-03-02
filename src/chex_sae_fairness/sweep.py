from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from chex_sae_fairness.config import ExperimentConfig, SAEConfig
from chex_sae_fairness.data.feature_cache import load_or_create_feature_bundle
from chex_sae_fairness.evaluation.disentanglement import (
    evaluate_disentanglement,
    reconstruction_metrics,
    summarize_latent_correlations,
)
from chex_sae_fairness.models.sae import SparseAutoencoder
from chex_sae_fairness.training.train_sae import encode_features, train_sae_model
from chex_sae_fairness.utils.io import write_json
from chex_sae_fairness.utils.repro import seed_everything


def run_sae_sweep(
    base_config_path: str,
    sweep_config_path: str,
    force_recompute_features: bool = False,
) -> dict[str, object]:
    base_cfg = ExperimentConfig.from_yaml(base_config_path)
    seed_everything(base_cfg.seed)

    feature_result = load_or_create_feature_bundle(
        base_cfg,
        force_recompute=force_recompute_features or base_cfg.features.force_recompute,
    )
    bundle = feature_result.bundle

    splits = bundle["split"].astype(str)
    x = bundle["features"].astype(np.float32)
    y_path = bundle["y_pathology"].astype(np.float32)
    path_cols = [str(c) for c in bundle["pathology_cols"].tolist()]
    metadata_cols = [str(c) for c in bundle["metadata_cols"].tolist()]
    metadata = pd.DataFrame(bundle["metadata"], columns=metadata_cols)

    train_mask, valid_mask, test_mask = _split_masks(
        splits,
        valid_name=base_cfg.data.validation_split_name,
        test_name=base_cfg.data.test_split_name,
    )

    x_train, x_valid, x_test = x[train_mask], x[valid_mask], x[test_mask]
    y_train, y_test = y_path[train_mask], y_path[test_mask]
    metadata_train = metadata.loc[train_mask].reset_index(drop=True)
    metadata_test = metadata.loc[test_mask].reset_index(drop=True)

    sweep_cfg = _read_yaml(sweep_config_path)
    runs = sweep_cfg.get("runs", [])
    if not runs:
        raise ValueError("Sweep config must include at least one run under `runs`.")

    output_dir = Path(
        sweep_cfg.get("output_dir", str(base_cfg.output_root / "sae_sweep"))
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    run_metrics: list[dict[str, object]] = []

    for run in runs:
        run_name = str(run["name"])
        sae_overrides = run.get("sae", {})
        sae_cfg = _build_sae_config(base_cfg.sae, sae_overrides)

        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        device = _resolve_device(base_cfg.features.device)
        trained = train_sae_model(
            train_features=x_train,
            valid_features=x_valid,
            cfg=sae_cfg,
            device=device,
        )

        checkpoint_path = run_dir / "sae.pt"
        torch.save(
            {
                "state_dict": trained.model.state_dict(),
                "input_dim": int(x.shape[1]),
                "latent_dim": int(sae_cfg.latent_dim),
                "variant": str(sae_cfg.variant),
                "topk_k": int(sae_cfg.topk_k),
            },
            checkpoint_path,
        )

        z_train = encode_features(
            model=trained.model,
            features=x_train,
            batch_size=max(256, sae_cfg.batch_size),
            device=device,
        )
        z_test = encode_features(
            model=trained.model,
            features=x_test,
            batch_size=max(256, sae_cfg.batch_size),
            device=device,
        )
        x_hat_test = _reconstruct_features(
            model=trained.model,
            features=x_test,
            batch_size=max(256, sae_cfg.batch_size),
            device=device,
        )

        recon = reconstruction_metrics(x_test, x_hat_test)
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
        correlations = summarize_latent_correlations(
            z=z_test,
            y_pathology=y_test,
            pathology_cols=path_cols,
            metadata=metadata_test,
            metadata_cols=metadata_cols,
        )

        run_report = {
            "run_name": run_name,
            "sae": asdict(sae_cfg),
            "train_curve": trained.train_curve,
            "valid_curve": trained.valid_curve,
            "reconstruction": recon,
            "disentanglement": disentanglement,
            "latent_correlations": correlations,
            "paths": {
                "checkpoint": str(checkpoint_path),
            },
        }
        write_json(run_report, run_dir / "metrics.json")

        _write_yaml(
            {
                "base_config": str(Path(base_config_path).resolve()),
                "sweep_config": str(Path(sweep_config_path).resolve()),
                "sae": asdict(sae_cfg),
            },
            run_dir / "run_config.yaml",
        )

        summary_rows.append(
            {
                "run_name": run_name,
                "variant": sae_cfg.variant,
                "latent_dim": sae_cfg.latent_dim,
                "topk_k": sae_cfg.topk_k,
                "l1_lambda": sae_cfg.l1_lambda,
                "epochs": sae_cfg.epochs,
                "reconstruction_mse": recon["mse"],
                "reconstruction_explained_variance": recon["explained_variance"],
                "mean_disentanglement": disentanglement["mean_disentanglement"],
                "mean_predictive_performance": disentanglement["mean_performance"],
                "mean_pathology_max_abs_corr": correlations["mean_pathology_max_abs_corr"],
                "mean_metadata_max_abs_corr": correlations["mean_metadata_max_abs_corr"],
            }
        )
        run_metrics.append(run_report)

    summary_frame = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "summary.csv"
    summary_frame.to_csv(summary_csv_path, index=False)

    plot_paths = build_sae_sweep_plots(summary_frame, output_dir / "plots")

    summary = {
        "base_config": str(Path(base_config_path).resolve()),
        "sweep_config": str(Path(sweep_config_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "used_cached_features": bool(feature_result.used_cache),
        "summary_csv": str(summary_csv_path.resolve()),
        "plots": [str(path.resolve()) for path in plot_paths],
        "runs": run_metrics,
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def build_sae_sweep_plots(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []

    if summary.empty:
        return plots

    # Plot 1: reconstruction error comparison.
    sorted_mse = summary.sort_values("reconstruction_mse", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sorted_mse["run_name"], sorted_mse["reconstruction_mse"], color="#4C78A8")
    ax.set_title("Reconstruction MSE Across SAE Runs")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Run")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = output_dir / "reconstruction_mse.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots.append(path)

    # Plot 2: pathology correlation comparison.
    sorted_corr = summary.sort_values("mean_pathology_max_abs_corr", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sorted_corr["run_name"], sorted_corr["mean_pathology_max_abs_corr"], color="#F58518")
    ax.set_title("Mean Max |corr| with Pathology Concepts")
    ax.set_ylabel("Mean max absolute correlation")
    ax.set_xlabel("Run")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = output_dir / "pathology_correlations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots.append(path)

    # Plot 3: tradeoff between reconstruction and concept correlation.
    fig, ax = plt.subplots(figsize=(7, 5))
    variants = summary["variant"].astype(str).unique().tolist()
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    color_map = {variant: palette[idx % len(palette)] for idx, variant in enumerate(variants)}
    for _, row in summary.iterrows():
        variant = str(row["variant"])
        ax.scatter(
            row["reconstruction_mse"],
            row["mean_pathology_max_abs_corr"],
            color=color_map[variant],
            label=variant,
            s=70,
            alpha=0.9,
        )
        ax.text(
            row["reconstruction_mse"],
            row["mean_pathology_max_abs_corr"],
            str(row["run_name"]),
            fontsize=8,
            ha="left",
            va="bottom",
        )

    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    ax.legend(dedup.values(), dedup.keys(), title="Variant")
    ax.set_title("Reconstruction vs Pathology Correlation")
    ax.set_xlabel("Reconstruction MSE (lower is better)")
    ax.set_ylabel("Mean max absolute correlation (higher is better)")
    fig.tight_layout()
    path = output_dir / "recon_vs_correlation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots.append(path)

    return plots


def _build_sae_config(base: SAEConfig, overrides: dict[str, Any]) -> SAEConfig:
    valid_keys = set(SAEConfig.__dataclass_fields__.keys())
    unknown = sorted(set(overrides.keys()) - valid_keys)
    if unknown:
        unknown_str = ", ".join(unknown)
        raise ValueError(f"Unknown SAE override keys: {unknown_str}")
    return replace(base, **overrides)


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
            f"No rows found for test split '{test_name}'. Check split column values in your manifest."
        )

    if train_mask.sum() == 0:
        raise ValueError("No rows left for train split after reserving valid/test.")

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


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Sweep config must be a YAML object.")
    return payload


def _write_yaml(payload: dict[str, Any], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
