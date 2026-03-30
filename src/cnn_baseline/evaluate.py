"""Evaluation: per-pathology AUROC, per-group AUROC, TPR disparity.

Output CSV schemas match the main project's presentation_pipeline.py exactly,
enabling direct comparison between CNN and SAE-based results.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

MIN_POSITIVES = 5
logger = logging.getLogger(__name__)


# ── Main entry point ──────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    pathology_cols: list[str],
    device: torch.device,
    output_dir: Path,
    amp: bool = True,
    split_name: str = "valid",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs inference, computes all metrics, saves CSVs + plots.
    Returns (auroc_df, tpr_disparity_df, per_group_auroc_df).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_dir = output_dir / "roc_curves"
    roc_dir.mkdir(exist_ok=True)

    labels, probs, attr_arrays = _collect_predictions(model, loader, device, amp)

    auroc_df = _build_auroc_df(labels, probs, pathology_cols)
    tpr_df = _build_tpr_table(labels, probs, pathology_cols, attr_arrays)
    disparity_df = _build_tpr_disparity_df(tpr_df)
    per_group_auroc_df = _build_per_group_auroc_df(labels, probs, pathology_cols, attr_arrays)

    auroc_df.to_csv(output_dir / "linear_separability.csv", index=False)
    tpr_df.to_csv(output_dir / "tpr_by_group.csv", index=False)
    disparity_df.to_csv(output_dir / "tpr_disparity.csv", index=False)
    per_group_auroc_df.to_csv(output_dir / "per_group_auroc.csv", index=False)

    logger.info("Macro AUROC (%s): %.4f", split_name, auroc_df["auroc"].dropna().mean())

    _plot_auroc_bar(auroc_df, output_dir)
    _plot_tpr_disparity_heatmap(disparity_df, output_dir)
    for attr in attr_arrays:
        _plot_tpr_by_group(tpr_df, attr, output_dir)
    for i, path in enumerate(pathology_cols):
        _plot_roc_for_pathology(labels[:, i], probs[:, i], path, attr_arrays, roc_dir)

    return auroc_df, disparity_df, per_group_auroc_df


# ── Inference ─────────────────────────────────────────────────────────────────

def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Returns (labels[N,C], probs[N,C], attr_arrays{name: array[N]})."""
    model.eval()
    all_labels, all_probs = [], []
    all_attrs: dict[str, list[str]] = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].cpu().numpy()

            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)

            for attr, vals in batch["attrs"].items():
                all_attrs.setdefault(attr, []).extend(vals)

    labels_arr = np.concatenate(all_labels, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)
    attr_arrays = {k: np.array(v) for k, v in all_attrs.items()}
    return labels_arr, probs_arr, attr_arrays


# ── Metric tables ─────────────────────────────────────────────────────────────

def _build_auroc_df(
    labels: np.ndarray,
    probs: np.ndarray,
    pathology_cols: list[str],
) -> pd.DataFrame:
    """Schema: pathology, auroc, n_positive — matches linear_separability.csv."""
    rows = []
    for i, path in enumerate(pathology_cols):
        y, p = labels[:, i], probs[:, i]
        n_pos = int(y.sum())
        if n_pos < MIN_POSITIVES or (len(y) - n_pos) < MIN_POSITIVES:
            auc = float("nan")
        else:
            try:
                auc = float(roc_auc_score(y, p))
            except Exception:
                auc = float("nan")
        rows.append({"pathology": path, "auroc": auc, "n_positive": n_pos})
    return pd.DataFrame(rows)


def _build_tpr_table(
    labels: np.ndarray,
    probs: np.ndarray,
    pathology_cols: list[str],
    attr_arrays: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Schema: pathology, attribute, group, tpr, n_positive, n_total."""
    rows = []
    for i, path in enumerate(pathology_cols):
        y, p = labels[:, i], probs[:, i]
        preds = (p >= threshold).astype(float)
        for attr, groups in attr_arrays.items():
            for grp in np.unique(groups):
                mask = groups == grp
                y_g, pred_g = y[mask], preds[mask]
                n_pos = int(y_g.sum())
                n_total = int(mask.sum())
                tpr = float(pred_g[y_g == 1].mean()) if n_pos >= MIN_POSITIVES else float("nan")
                rows.append({
                    "pathology": path, "attribute": attr, "group": grp,
                    "tpr": tpr, "n_positive": n_pos, "n_total": n_total,
                })
    return pd.DataFrame(rows)


def _build_tpr_disparity_df(tpr_df: pd.DataFrame) -> pd.DataFrame:
    """Schema: pathology, attribute, tpr_max, tpr_min, tpr_disparity."""
    rows = []
    for (path, attr), g in tpr_df.groupby(["pathology", "attribute"]):
        valid = g["tpr"].dropna()
        if len(valid) < 2:
            continue
        tpr_max, tpr_min = float(valid.max()), float(valid.min())
        rows.append({
            "pathology": path, "attribute": attr,
            "tpr_max": tpr_max, "tpr_min": tpr_min,
            "tpr_disparity": tpr_max - tpr_min,
        })
    return pd.DataFrame(rows)


def _build_per_group_auroc_df(
    labels: np.ndarray,
    probs: np.ndarray,
    pathology_cols: list[str],
    attr_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Per-group AUROC: pathology, attribute, group, auroc, n_positive, n_total."""
    rows = []
    for i, path in enumerate(pathology_cols):
        y, p = labels[:, i], probs[:, i]
        for attr, groups in attr_arrays.items():
            for grp in np.unique(groups):
                mask = groups == grp
                y_g, p_g = y[mask], p[mask]
                n_pos = int(y_g.sum())
                n_total = int(mask.sum())
                if n_pos < MIN_POSITIVES or (n_total - n_pos) < MIN_POSITIVES:
                    auc = float("nan")
                else:
                    try:
                        auc = float(roc_auc_score(y_g, p_g))
                    except Exception:
                        auc = float("nan")
                rows.append({
                    "pathology": path, "attribute": attr, "group": grp,
                    "auroc": auc, "n_positive": n_pos, "n_total": n_total,
                })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_auroc_bar(auroc_df: pd.DataFrame, output_dir: Path) -> None:
    valid = auroc_df.dropna(subset=["auroc"])
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#3a7bd5" if v >= 0.75 else "#e88b2a" if v >= 0.65 else "#d93b3b"
              for v in valid["auroc"]]
    ax.bar(valid["pathology"], valid["auroc"], color=colors)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Pathology AUROC (CNN, validation split)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(output_dir / "auroc_per_pathology.png", dpi=150)
    plt.close(fig)


def _plot_tpr_disparity_heatmap(disparity_df: pd.DataFrame, output_dir: Path) -> None:
    if disparity_df.empty:
        return
    pivot = disparity_df.pivot(index="pathology", columns="attribute", values="tpr_disparity")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax, vmin=0, vmax=0.5,
                linewidths=0.5)
    ax.set_title("TPR Disparity (CNN, validation split)")
    plt.tight_layout()
    fig.savefig(output_dir / "tpr_disparity_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_tpr_by_group(tpr_df: pd.DataFrame, attribute: str, output_dir: Path) -> None:
    sub = tpr_df[tpr_df["attribute"] == attribute].dropna(subset=["tpr"])
    if sub.empty:
        return
    pathologies = sub["pathology"].unique()
    groups = sorted(sub["group"].unique())
    n_path = len(pathologies)

    fig, axes = plt.subplots(1, n_path, figsize=(max(n_path * 2.5, 12), 5), sharey=False)
    if n_path == 1:
        axes = [axes]
    palette = plt.cm.Set2(np.linspace(0, 1, len(groups)))

    for ax, path in zip(axes, pathologies):
        g = sub[sub["pathology"] == path].set_index("group")["tpr"]
        vals = [g.get(grp, float("nan")) for grp in groups]
        ax.bar(groups, vals, color=palette[:len(groups)])
        ax.set_title(path, fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle(f"TPR by {attribute} (CNN)", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / f"tpr_by_{attribute}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_for_pathology(
    y: np.ndarray,
    p: np.ndarray,
    pathology: str,
    attr_arrays: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    n_pos = int(y.sum())
    if n_pos < MIN_POSITIVES:
        return

    attrs = list(attr_arrays.keys())
    n_attrs = len(attrs)
    ncols = 2
    nrows = (n_attrs + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 4))
    axes = np.array(axes).flatten()

    for ai, attr in enumerate(attrs):
        ax = axes[ai]
        groups = attr_arrays[attr]
        for grp in np.unique(groups):
            mask = groups == grp
            y_g, p_g = y[mask], p[mask]
            if y_g.sum() < MIN_POSITIVES:
                continue
            fpr, tpr, _ = roc_curve(y_g, p_g)
            auc = roc_auc_score(y_g, p_g)
            ax.plot(fpr, tpr, label=f"{grp} (AUC={auc:.2f})", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"{pathology} — by {attr}")
        ax.legend(fontsize=7)

    for ai in range(n_attrs, len(axes)):
        axes[ai].set_visible(False)

    fig.suptitle(f"ROC Curves: {pathology}", fontsize=11)
    plt.tight_layout()
    safe_path = pathology.replace(" ", "_").replace("/", "-")
    fig.savefig(output_dir / f"roc_{safe_path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
