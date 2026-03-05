#!/usr/bin/env python3
"""Fairness intervention pipeline — concept mean-ablation via SAE decoder.

Selects the best SAE for a given (demographic, pathology) pair (from the concept
analysis grid), identifies concepts whose pair-specificity exceeds a threshold,
replaces those concept activations with their training-set means, decodes back to
CheXagent embedding space, and evaluates the fairness impact.

Three evaluation conditions are compared:
  A  baseline  — original CheXagent embeddings, original LR
  B  ablated   — ablated embeddings, original LR  (effect of ablation alone)
  C  retrained — ablated embeddings, LR retrained on ablated training embeddings
                 (full intervention; only computed when --lr-mode retrained|both)

Produces for each condition:
  Separate evaluation plots (AUROC, TPR disparity, TPR by group, ROC curves)
and in comparison/:
  Side-by-side / overlaid plots for all conditions

Usage:
  python scripts/fairness_intervention.py \\
      --config configs/test.yaml \\
      --attr sex \\
      --pathology Cardiomegaly \\
      --threshold 0.05 \\
      --lr-mode both
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

MIN_POSITIVES = 5
SAE_BATCH_SIZE = 512


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",     required=True, help="Path to YAML config")
    p.add_argument("--attr",       required=True,
                   help="Demographic attribute to target (sex, age_group, race, insurance_type)")
    p.add_argument("--pathology",  required=True,
                   help="Pathology label to protect (e.g. 'Cardiomegaly')")
    p.add_argument("--threshold",  type=float, default=0.05,
                   help="Pair-specificity threshold for concept ablation (default: 0.05)")
    p.add_argument("--lr-mode",    choices=["original", "retrained", "both"], default="both",
                   help="LR training mode (default: both)")
    p.add_argument("--ablation",   choices=["mean", "zero"], default="mean",
                   help="Replacement value for ablated concepts (default: mean)")
    p.add_argument("--sae-dir",    default=None,
                   help="Path to sae-eval directory. Default: <output_root>/presentation/sae-eval/")
    p.add_argument("--output-dir", default=None,
                   help="Where to write results. Default: <output_root>/presentation/intervention/")
    p.add_argument("--max-iter",   type=int, default=1000,
                   help="Max iterations for logistic regression (default: 1000)")
    p.add_argument("--device",     default=None,
                   help="torch device (default: cuda if available, else cpu)")
    return p.parse_args()


# ── SAE (matches concept_analysis_pipeline.py) ────────────────────────────────

class TopKSAE(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Linear(input_dim, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, input_dim)
        self.activation = torch.nn.ReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.encoder(x))
        if self.k >= z.shape[-1]:
            return z
        _, indices = torch.topk(z, k=self.k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(dim=-1, index=indices, value=1.0)
        return z * mask

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def load_sae(checkpoint_path: Path, device: str) -> TopKSAE:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TopKSAE(ckpt["input_dim"], ckpt["latent_dim"], ckpt["k"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device)


@torch.no_grad()
def encode_features(model: TopKSAE, x: np.ndarray, device: str) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32)),
        batch_size=SAE_BATCH_SIZE * 4, shuffle=False,
    )
    parts: list[np.ndarray] = []
    for (batch,) in loader:
        parts.append(model.encode(batch.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def decode_features(model: TopKSAE, z: np.ndarray, device: str) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(z, dtype=torch.float32)),
        batch_size=SAE_BATCH_SIZE * 4, shuffle=False,
    )
    parts: list[np.ndarray] = []
    for (batch,) in loader:
        parts.append(model.decode(batch.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


# ── Ablation ──────────────────────────────────────────────────────────────────

def ablate_concepts(
    z: np.ndarray,
    concept_indices: list[int],
    replacement_values: np.ndarray,
) -> np.ndarray:
    """Return a copy of z with selected concept dimensions replaced."""
    z_out = z.copy()
    for c_idx in concept_indices:
        z_out[:, c_idx] = replacement_values[c_idx]
    return z_out


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _clean_attr(values: np.ndarray) -> np.ndarray:
    out = values.astype(object)
    out[np.isin(values.astype(str), {"nan", "None", "none", ""})] = np.nan
    return out


def compute_tpr(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float | None:
    pos = y_true == 1
    if pos.sum() < MIN_POSITIVES:
        return None
    return float((y_score[pos] >= threshold).mean())


def _train_lr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    pathology_cols: list[str],
    max_iter: int,
) -> tuple[np.ndarray, StandardScaler]:
    """Fit one LR per pathology; return score matrix [n_eval, n_path] and scaler."""
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_train)
    x_ev_s = scaler.transform(x_eval)

    scores = np.zeros((len(x_eval), len(pathology_cols)), dtype=np.float32)
    for p_idx, pathology in enumerate(pathology_cols):
        y_tr = y_train[:, p_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter, random_state=13)
        clf.fit(x_tr_s, y_tr)
        scores[:, p_idx] = clf.predict_proba(x_ev_s)[:, 1]
    return scores, scaler


def build_auroc_df(
    y_valid: np.ndarray,
    y_scores: np.ndarray,
    pathology_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for p_idx, pathology in enumerate(pathology_cols):
        y_va = y_valid[:, p_idx]
        n_pos = int((y_va == 1).sum())
        if len(np.unique(y_va)) < 2 or n_pos < MIN_POSITIVES:
            rows.append({"pathology": pathology, "auroc": float("nan"), "n_positive": n_pos})
            continue
        rows.append({
            "pathology": pathology,
            "auroc": float(roc_auc_score(y_va, y_scores[:, p_idx])),
            "n_positive": n_pos,
        })
    return pd.DataFrame(rows)


def build_tpr_table(
    y_valid: np.ndarray,
    y_scores: np.ndarray,
    pathology_cols: list[str],
    attr_arrays: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for p_idx, pathology in enumerate(pathology_cols):
        y_true = y_valid[:, p_idx]
        y_score = y_scores[:, p_idx]
        for attr_name, attr_values in attr_arrays.items():
            for group_val in sorted(np.unique(attr_values[~pd.isna(attr_values)])):
                mask = attr_values == group_val
                tpr = compute_tpr(y_true[mask], y_score[mask], threshold)
                rows.append({
                    "pathology": pathology, "attribute": attr_name,
                    "group": group_val, "tpr": tpr,
                    "n_positive": int((y_true[mask] == 1).sum()),
                    "n_total": int(mask.sum()),
                })
    return pd.DataFrame(rows)


@dataclass
class EvalResult:
    name: str
    color: str
    linestyle: str
    auroc_df: pd.DataFrame
    tpr_df: pd.DataFrame
    disparity_df: pd.DataFrame
    y_valid: np.ndarray
    y_scores: np.ndarray
    overall_tpr: dict[str, float | None]


def evaluate(
    x_train_for_lr: np.ndarray,
    x_eval: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    attr_arrays: dict[str, np.ndarray],
    pathology_cols: list[str],
    threshold: float,
    max_iter: int,
    name: str,
    color: str,
    linestyle: str,
) -> EvalResult:
    logger.info("  Evaluating condition '%s'…", name)
    y_scores, _ = _train_lr(x_train_for_lr, y_train, x_eval, pathology_cols, max_iter)

    auroc_df = build_auroc_df(y_valid, y_scores, pathology_cols)
    tpr_df = build_tpr_table(y_valid, y_scores, pathology_cols, attr_arrays, threshold)
    disparity_df = (
        tpr_df.dropna(subset=["tpr"])
        .groupby(["pathology", "attribute"])["tpr"]
        .agg(tpr_max="max", tpr_min="min")
        .assign(tpr_disparity=lambda d: d["tpr_max"] - d["tpr_min"])
        .reset_index()
    )
    overall_tpr = {
        pathology: compute_tpr(y_valid[:, p_i], y_scores[:, p_i], threshold)
        for p_i, pathology in enumerate(pathology_cols)
    }
    return EvalResult(
        name=name, color=color, linestyle=linestyle,
        auroc_df=auroc_df, tpr_df=tpr_df, disparity_df=disparity_df,
        y_valid=y_valid, y_scores=y_scores, overall_tpr=overall_tpr,
    )


# ── Separate per-condition plots ───────────────────────────────────────────────

def _save_separate(result: EvalResult, pathology_cols: list[str],
                   attr_arrays: dict, out_dir: Path, tpr_threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.auroc_df.to_csv(out_dir / "linear_separability.csv", index=False)
    result.tpr_df.to_csv(out_dir / "tpr_by_group.csv", index=False)
    result.disparity_df.to_csv(out_dir / "tpr_disparity.csv", index=False)

    _plot_auroc(result.auroc_df, result.name, out_dir)
    _plot_tpr_disparity_heatmap(result.disparity_df, pathology_cols, result.name, out_dir)
    for attr in attr_arrays:
        _plot_tpr_by_group(result.tpr_df, attr, pathology_cols, result.overall_tpr, result.name, out_dir)

    # Persist scores so post-hoc summary plots can load them without re-running.
    np.savez_compressed(
        out_dir / "scores.npz",
        y_valid=result.y_valid.astype(np.float32),
        y_scores=result.y_scores.astype(np.float32),
    )

    roc_dir = out_dir / "roc_curves"
    roc_dir.mkdir(exist_ok=True)
    for p_idx, pathology in enumerate(pathology_cols):
        _plot_roc_single(
            result.y_valid[:, p_idx], result.y_scores[:, p_idx],
            pathology, attr_arrays, result.name, roc_dir,
        )
    logger.info("  Separate plots → %s", out_dir)


def _plot_auroc(auroc_df: pd.DataFrame, title_tag: str, out_dir: Path) -> None:
    df = auroc_df.dropna(subset=["auroc"]).sort_values("auroc", ascending=True)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.45)))
    bars = ax.barh(df["pathology"], df["auroc"], color="#4C78A8", height=0.6)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_xlabel("AUROC (validation set)")
    ax.set_title(f"AUROC per Pathology — {title_tag}")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, df["auroc"]):
        ax.text(min(val + 0.01, 1.0), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "auroc_per_pathology.png", dpi=150)
    plt.close(fig)


def _plot_tpr_disparity_heatmap(
    disparity_df: pd.DataFrame, pathology_cols: list[str], title_tag: str, out_dir: Path
) -> None:
    if disparity_df.empty:
        return
    pivot = disparity_df.pivot(index="pathology", columns="attribute", values="tpr_disparity")
    pivot = pivot.reindex([p for p in pathology_cols if p in pivot.index])
    fig, ax = plt.subplots(figsize=(len(pivot.columns) * 2.2 + 1.5, len(pivot) * 0.55 + 2))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.4,
                cbar_kws={"label": "TPR disparity (max − min)  ↓ lower = more fair"})
    ax.set_title(
        f"TPR Disparity — {title_tag}\n"
        "Lower values = more equitable sensitivity across groups  (target: 0)",
        fontsize=9,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "tpr_disparity_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_tpr_by_group(
    tpr_df: pd.DataFrame, attribute: str, pathology_cols: list[str],
    overall_tpr: dict, title_tag: str, out_dir: Path,
) -> None:
    df = tpr_df[(tpr_df["attribute"] == attribute) & tpr_df["tpr"].notna()].copy()
    if df.empty:
        return
    groups = sorted(df["group"].unique())
    pathologies = [p for p in pathology_cols if p in df["pathology"].unique()]
    if not pathologies:
        return

    x = np.arange(len(pathologies))
    n_slots = len(groups) + 1
    width = min(0.8 / n_slots, 0.22)
    palette = sns.color_palette("tab10", len(groups))

    fig, ax = plt.subplots(figsize=(max(10, len(pathologies) * 0.9), 5))
    overall_vals = [_nan(overall_tpr.get(p)) for p in pathologies]
    ax.bar(x + (-(n_slots - 1) / 2) * width, overall_vals, width,
           label="Overall", color="#888888", alpha=0.9, hatch="//", edgecolor="white")
    for i, group in enumerate(groups):
        grp = df[df["group"] == group].set_index("pathology")["tpr"]
        tprs = [_nan(grp.get(p, float("nan"))) for p in pathologies]
        ax.bar(x + (i + 1 - (n_slots - 1) / 2) * width, tprs, width,
               label=str(group), color=palette[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(pathologies, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"TPR by {attribute.replace('_',' ').title()} — {title_tag}")
    ax.legend(title=attribute.replace("_", " ").title(),
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"tpr_by_{attribute}.png", dpi=150)
    plt.close(fig)


def _plot_roc_single(
    y_true: np.ndarray, y_score: np.ndarray, pathology: str,
    attr_arrays: dict, title_tag: str, out_dir: Path,
) -> None:
    if len(np.unique(y_true)) < 2:
        return
    attrs = list(attr_arrays.items())
    n_cols = 2
    n_rows = (len(attrs) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5), squeeze=False)
    fig.suptitle(f"ROC — {pathology} ({title_tag})", fontsize=11, fontweight="bold", y=1.01)

    overall_fpr, overall_tpr_curve, _ = roc_curve(y_true, y_score)
    overall_auc = float(roc_auc_score(y_true, y_score))

    for ax_idx, (attr_name, attr_values) in enumerate(attrs):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        ax.plot(overall_fpr, overall_tpr_curve, color="black", linewidth=2, linestyle="--",
                label=f"Overall (AUC={overall_auc:.3f})", zorder=5)
        groups = sorted(np.unique(attr_values[~pd.isna(attr_values)]))
        for color, g in zip(sns.color_palette("tab10", len(groups)), groups):
            mask = attr_values == g
            y_g, s_g = y_true[mask], y_score[mask]
            if len(np.unique(y_g)) < 2 or (y_g == 1).sum() < MIN_POSITIVES:
                continue
            fpr_g, tpr_g, _ = roc_curve(y_g, s_g)
            ax.plot(fpr_g, tpr_g, color=color, linewidth=1.5,
                    label=f"{g} (AUC={float(roc_auc_score(y_g, s_g)):.3f})")
        ax.plot([0, 1], [0, 1], color="lightgrey", linewidth=0.8, linestyle=":")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(attr_name.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    for ax_idx in range(len(attrs), n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)
    fig.tight_layout()
    safe = pathology.replace(" ", "_").replace("/", "-")
    fig.savefig(out_dir / f"roc_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Comparison plots ───────────────────────────────────────────────────────────

def _plot_auroc_comparison(results: list[EvalResult], pathology_cols: list[str], out_dir: Path) -> None:
    """Grouped horizontal bars: one cluster per pathology, one bar per condition."""
    valid_paths = [p for p in pathology_cols
                   if any(not r.auroc_df[r.auroc_df.pathology == p]["auroc"].isna().all()
                          for r in results)]
    if not valid_paths:
        return

    n_conditions = len(results)
    x = np.arange(len(valid_paths))
    width = min(0.8 / n_conditions, 0.25)

    fig, ax = plt.subplots(figsize=(max(10, len(valid_paths) * 0.8), 5))
    for i, res in enumerate(results):
        auroc_vals = [
            float(res.auroc_df[res.auroc_df.pathology == p]["auroc"].iloc[0])
            if not res.auroc_df[res.auroc_df.pathology == p]["auroc"].isna().all()
            else float("nan")
            for p in valid_paths
        ]
        offset = (i - (n_conditions - 1) / 2) * width
        ax.bar(x + offset, auroc_vals, width, label=res.name, color=res.color, alpha=0.85)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_paths, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    intervention_name = results[-1].name if len(results) > 1 else "intervention"
    ax.set_title(f"AUROC Comparison — baseline vs {intervention_name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "auroc_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved → auroc_comparison.png")


def _plot_tpr_disparity_comparison(
    results: list[EvalResult], pathology_cols: list[str], out_dir: Path
) -> None:
    """Side-by-side TPR disparity heatmaps, one per condition."""
    n = len(results)
    if n == 0:
        return

    # Build pivot per condition
    pivots = []
    for res in results:
        if res.disparity_df.empty:
            pivots.append(None)
            continue
        pivot = res.disparity_df.pivot(index="pathology", columns="attribute", values="tpr_disparity")
        pivot = pivot.reindex([p for p in pathology_cols if p in pivot.index])
        pivots.append(pivot)

    attrs = [c for p in pivots if p is not None for c in p.columns]
    attrs = sorted(set(attrs))
    n_attrs = len(attrs) if attrs else 1

    fig, axes = plt.subplots(1, n, figsize=(n_attrs * 2.0 * n + 1.5, len(pathology_cols) * 0.5 + 2),
                             squeeze=False)
    for col_idx, (res, pivot) in enumerate(zip(results, pivots)):
        ax = axes[0][col_idx]
        if pivot is None or pivot.empty:
            ax.set_visible(False)
            continue
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, linewidths=0.4,
                    cbar=(col_idx == n - 1),
                    cbar_kws={"label": "TPR disparity  ↓ lower = more fair"} if col_idx == n - 1 else {})
        ax.set_title(res.name, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)

    intervention_name = results[-1].name if len(results) > 1 else "intervention"
    fig.suptitle(
        f"TPR Disparity: baseline vs {intervention_name}\n"
        "Lower values = more equitable sensitivity across groups  (target: 0)",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "tpr_disparity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → tpr_disparity_comparison.png")


def _plot_tpr_by_group_comparison(
    results: list[EvalResult], attribute: str, pathology_cols: list[str], out_dir: Path
) -> None:
    """Grouped bar chart: paired bars (baseline | intervention) for overall + each group.

    Layout per pathology cluster:
      [Overall_A | Overall_B | Group1_A | Group1_B | Group2_A | Group2_B | ...]
    where A = results[0] (baseline), B = results[1] (intervention).
    Always called with exactly 2 conditions.
    """
    all_dfs = [r.tpr_df[r.tpr_df["attribute"] == attribute] for r in results]
    all_groups = sorted(set(g for df in all_dfs for g in df["group"].dropna().unique()))
    pathologies = [p for p in pathology_cols
                   if any(p in df["pathology"].values for df in all_dfs)]
    if not pathologies or not all_groups:
        return

    n_groups = len(all_groups)
    # Slots per pathology: 1 pair (overall) + n_groups pairs = 2 * (1 + n_groups)
    n_pairs = 1 + n_groups          # overall pair + one pair per group
    n_slots = 2 * n_pairs
    width = min(0.85 / n_slots, 0.15)
    x = np.arange(len(pathologies))

    group_palette = sns.color_palette("tab10", n_groups)

    fig, ax = plt.subplots(figsize=(max(12, len(pathologies) * 1.2), 5))

    # Build slot indices:
    #   slots 0,1 → overall A, overall B
    #   slots 2,3 → group0 A, group0 B
    #   slots 4,5 → group1 A, group1 B  ...
    # offset = (slot - (n_slots - 1) / 2) * width

    def _slot_offset(slot: int) -> float:
        return (slot - (n_slots - 1) / 2) * width

    # Overall pair (hatched, semi-transparent, using each condition's color)
    for ci, res in enumerate(results):
        overall_vals = [_nan(res.overall_tpr.get(p)) for p in pathologies]
        slot = ci  # 0 or 1
        ax.bar(x + _slot_offset(slot), overall_vals, width,
               color=res.color, alpha=0.55, hatch="///", edgecolor="grey", linewidth=0.5)

    # Per-group pairs (solid, colored by group)
    for gi, group in enumerate(all_groups):
        for ci, res in enumerate(results):
            grp_df = all_dfs[ci][all_dfs[ci]["group"] == group].set_index("pathology")["tpr"]
            tprs = [_nan(grp_df.get(p, float("nan"))) for p in pathologies]
            slot = 2 + gi * 2 + ci
            hatch = "///" if ci == 1 else ""
            ax.bar(x + _slot_offset(slot), tprs, width,
                   color=group_palette[gi], alpha=0.85,
                   hatch=hatch, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(pathologies, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_ylim(0, 1.15)
    intervention_name = results[-1].name if len(results) > 1 else "intervention"
    ax.set_title(f"TPR by {attribute.replace('_',' ').title()} — baseline vs {intervention_name}")

    # Legend: condition indicators + group colors
    res_a, res_b = results[0], results[1]
    legend_handles = [
        mpatches.Patch(facecolor="grey", alpha=0.55, hatch="///", edgecolor="grey",
                       label=f"Overall ({res_a.name})"),
        mpatches.Patch(facecolor="grey", alpha=0.55, hatch="///", edgecolor="grey",
                       label=f"Overall ({res_b.name})"),
        mpatches.Patch(facecolor="white", edgecolor="black", label=f"Group bar: {res_a.name} (solid)"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///",
                       label=f"Group bar: {res_b.name} (hatched)"),
    ]
    # Color key per group
    for gi, group in enumerate(all_groups):
        legend_handles.append(mpatches.Patch(facecolor=group_palette[gi], label=str(group)))

    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out_dir / f"tpr_by_{attribute}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → tpr_by_%s_comparison.png", attribute)


def _plot_roc_comparison(
    results: list[EvalResult],
    target_pathology: str,
    target_attr: str,
    p_idx: int,
    attr_arrays: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Overlay ROC curves for all conditions on a single (attr × row) figure.

    One subplot per attribute; each subplot shows one overall curve per condition
    plus coloured per-group curves for the targeted attribute.
    """
    attrs = list(attr_arrays.items())
    n_cols = 2
    n_rows = (len(attrs) + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 5), squeeze=False)
    fig.suptitle(
        f"ROC Curves — {target_pathology}  ·  comparison across conditions",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for ax_idx, (attr_name, attr_values) in enumerate(attrs):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]

        for res in results:
            y_true = res.y_valid[:, p_idx]
            y_score = res.y_scores[:, p_idx]
            if len(np.unique(y_true)) < 2:
                continue

            # Overall curve — thick, styled per condition
            fpr_all, tpr_all, _ = roc_curve(y_true, y_score)
            auc_all = float(roc_auc_score(y_true, y_score))
            ax.plot(fpr_all, tpr_all,
                    color=res.color, linewidth=2.5, linestyle=res.linestyle,
                    label=f"{res.name} overall (AUC={auc_all:.3f})", zorder=5)

            # Per-group curves — thin, drawn for every demographic subplot
            groups = sorted(np.unique(attr_values[~pd.isna(attr_values)]))
            group_palette = sns.color_palette("tab10", len(groups))
            for color, g in zip(group_palette, groups):
                mask = attr_values == g
                y_g, s_g = y_true[mask], y_score[mask]
                if len(np.unique(y_g)) < 2 or (y_g == 1).sum() < MIN_POSITIVES:
                    continue
                fpr_g, tpr_g, _ = roc_curve(y_g, s_g)
                auc_g = float(roc_auc_score(y_g, s_g))
                ax.plot(fpr_g, tpr_g, color=color, linewidth=1.2,
                        linestyle=res.linestyle, alpha=0.7,
                        label=f"{res.name} · {g} (AUC={auc_g:.3f})")

        ax.plot([0, 1], [0, 1], color="lightgrey", linewidth=0.8, linestyle=":")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(attr_name.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=6.5, loc="lower right")

    for ax_idx in range(len(attrs), n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

    fig.tight_layout()
    safe = target_pathology.replace(" ", "_").replace("/", "-")
    fig.savefig(out_dir / f"roc_{safe}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → roc_%s_comparison.png", safe)


def _plot_roc_focused_comparison(
    res_baseline: EvalResult,
    res_intervention: EvalResult,
    target_pathology: str,
    target_attr: str,
    p_idx: int,
    attr_arrays: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Single-panel ROC overlay: baseline (solid) vs intervention (dashed).

    Shows the overall ROC curve plus one ROC curve per demographic group of the
    targeted attribute.  Same colour per group; linestyle distinguishes conditions.
    A successful intervention will show the per-group curves converging toward
    the overall curve in the intervention condition.
    """
    attr_values = attr_arrays.get(target_attr)
    if attr_values is None:
        return

    y_true = res_baseline.y_valid[:, p_idx]
    if len(np.unique(y_true)) < 2:
        return

    groups = sorted(np.unique(attr_values[~pd.isna(attr_values)]))
    group_palette = sns.color_palette("tab10", len(groups))
    intervention_name = res_intervention.name

    fig, ax = plt.subplots(figsize=(7, 6))

    for res, ls, lw in [
        (res_baseline,     "-",  2.5),
        (res_intervention, "--", 2.0),
    ]:
        label_suffix = res.name
        y_score = res.y_scores[:, p_idx]

        # Overall ROC — black, thick
        fpr_all, tpr_all, _ = roc_curve(y_true, y_score)
        auc_all = float(roc_auc_score(y_true, y_score))
        ax.plot(fpr_all, tpr_all,
                color="black", linewidth=lw, linestyle=ls,
                label=f"Overall — {label_suffix} (AUC={auc_all:.3f})", zorder=5)

        # Per-group ROC — colour per group, linestyle per condition
        for color, g in zip(group_palette, groups):
            mask = attr_values == g
            y_g, s_g = y_true[mask], y_score[mask]
            if len(np.unique(y_g)) < 2 or (y_g == 1).sum() < MIN_POSITIVES:
                continue
            fpr_g, tpr_g, _ = roc_curve(y_g, s_g)
            auc_g = float(roc_auc_score(y_g, s_g))
            ax.plot(fpr_g, tpr_g, color=color, linewidth=1.4, linestyle=ls, alpha=0.85,
                    label=f"{g} — {label_suffix} (AUC={auc_g:.3f})")

    ax.plot([0, 1], [0, 1], color="lightgrey", linewidth=0.8, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(
        f"ROC Curves — {target_pathology}  ×  {target_attr.replace('_', ' ')}\n"
        f"Solid = baseline  ·  Dashed = {intervention_name}\n"
        f"Converging group curves → successful fairness intervention",
        fontsize=9,
    )
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    safe = target_pathology.replace(" ", "_").replace("/", "-")
    safe_a = target_attr.replace(" ", "_")
    fig.savefig(out_dir / f"roc_focused_{safe}_{safe_a}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → roc_focused_%s_%s.png", safe, safe_a)


# ── Main ───────────────────────────────────────────────────────────────────────

def _safe(s: str) -> str:
    return s.replace(" ", "_").replace("/", "-")


def _nan(v: object) -> float:
    """Convert None (and any non-finite sentinel) to float nan for matplotlib."""
    return float("nan") if v is None else float(v)


def main() -> None:
    args = parse_args()

    try:
        from chex_sae_fairness.config import ExperimentConfig
    except ImportError:
        sys.exit("Package not installed. Run:  pip install -e .")

    cfg = ExperimentConfig.from_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sae_dir = Path(args.sae_dir) if args.sae_dir else cfg.output_root / "presentation" / "sae-eval"
    if not sae_dir.exists():
        sys.exit(f"SAE-eval directory not found: {sae_dir}\nRun concept_analysis_pipeline.py first.")

    # ── Load feature bundle ────────────────────────────────────────────────────
    with np.load(cfg.feature_path, allow_pickle=True) as data:
        bundle = {key: data[key] for key in data.files}

    splits = bundle["split"].astype(str)
    x_all = bundle["features"].astype(np.float32)
    y_all = bundle["y_pathology"].astype(np.float32)
    pathology_cols = [str(c) for c in bundle["pathology_cols"]]
    metadata_cols = [str(c) for c in bundle["metadata_cols"]]
    metadata = pd.DataFrame(bundle["metadata"], columns=metadata_cols)
    age_group = bundle["age_group"].astype(str)

    train_mask = splits == "train"
    valid_mask = splits == cfg.data.validation_split_name
    x_train = x_all[train_mask]
    x_valid = x_all[valid_mask]
    y_train = y_all[train_mask]
    y_valid = y_all[valid_mask]
    meta_val = metadata[valid_mask].reset_index(drop=True)
    age_group_val = age_group[valid_mask]

    # Sensitive attributes on validation set
    attr_arrays: dict[str, np.ndarray] = {"age_group": _clean_attr(age_group_val)}
    for a in ["sex", "race", "insurance_type"]:
        if a in meta_val.columns:
            attr_arrays[a] = _clean_attr(meta_val[a].to_numpy())

    # ── Validate (attr, pathology) arguments ──────────────────────────────────
    if args.attr not in attr_arrays:
        sys.exit(
            f"Attribute '{args.attr}' not available. Choose from: {list(attr_arrays)}"
        )
    if args.pathology not in pathology_cols:
        sys.exit(
            f"Pathology '{args.pathology}' not found. Choose from:\n  " +
            "\n  ".join(pathology_cols)
        )

    target_p_idx = pathology_cols.index(args.pathology)
    safe_attr = _safe(args.attr)
    safe_path = _safe(args.pathology)

    # ── Find best SAE for this (attr, pathology) pair ─────────────────────────
    best_csv = sae_dir / "grid" / "best_sae_per_pair.csv"
    if not best_csv.exists():
        sys.exit(f"Grid summary not found: {best_csv}\nRun concept_analysis_pipeline.py first.")

    best_df = pd.read_csv(best_csv)
    row = best_df[(best_df["attr"] == args.attr) & (best_df["pathology"] == args.pathology)]
    if row.empty:
        sys.exit(f"No entry for ({args.attr}, {args.pathology}) in {best_csv}")

    best_k = int(row.iloc[0]["best_k"])
    best_dim = int(row.iloc[0]["best_dim"])
    best_spec = float(row.iloc[0]["best_spec"])
    run_label = f"k{best_k}_d{best_dim}"
    logger.info(
        "Best SAE for (%s, %s): %s  (max pair specificity = %.4f)",
        args.attr, args.pathology, run_label, best_spec,
    )

    # ── Load concept scores and select concepts to ablate ─────────────────────
    scores_csv = sae_dir / run_label / "concept_scores.csv"
    checkpoint_path = sae_dir / run_label / "sae_checkpoint.pt"
    for p in [scores_csv, checkpoint_path]:
        if not p.exists():
            sys.exit(f"Required file not found: {p}\nRe-run concept_analysis_pipeline.py.")

    scores_df = pd.read_csv(scores_csv)
    spec_col = f"spec_{args.attr}_{safe_path}"

    if spec_col not in scores_df.columns:
        sys.exit(
            f"Column '{spec_col}' not in concept_scores.csv.\n"
            "Re-run concept_analysis_pipeline.py with the current code."
        )

    ablate_mask = scores_df[spec_col] > args.threshold
    ablated_indices = scores_df.loc[ablate_mask, "concept_idx"].astype(int).tolist()

    logger.info(
        "Threshold=%.4f → %d / %d concepts selected for ablation",
        args.threshold, len(ablated_indices), len(scores_df),
    )
    if not ablated_indices:
        logger.warning(
            "No concepts exceed the threshold. Try a lower --threshold value.\n"
            "Max pair specificity in this SAE: %.4f",
            float(scores_df[spec_col].max()),
        )
        sys.exit(1)

    # Save list of ablated concepts
    ablated_concepts_df = scores_df.loc[ablate_mask, ["concept_idx", f"demo_eta2_{args.attr}",
                                                        f"path_eta2_{safe_path}", spec_col]]
    ablated_concepts_df = ablated_concepts_df.sort_values(spec_col, ascending=False)

    # ── Load SAE and apply ablation ───────────────────────────────────────────
    logger.info("Loading SAE checkpoint from %s", checkpoint_path)
    sae = load_sae(checkpoint_path, device)

    logger.info("Encoding training set through SAE (computing mean activations)…")
    z_train = encode_features(sae, x_train, device)

    # Replacement values: training-set mean or zero
    if args.ablation == "mean":
        replacement = z_train.mean(axis=0)   # shape [latent_dim]
    else:
        replacement = np.zeros(z_train.shape[1], dtype=np.float32)

    logger.info("Encoding validation set and applying ablation…")
    z_valid = encode_features(sae, x_valid, device)
    z_valid_ablated = ablate_concepts(z_valid, ablated_indices, replacement)

    logger.info("Decoding ablated latents back to embedding space…")
    x_ablated_valid = decode_features(sae, z_valid_ablated, device)

    if args.lr_mode in ("retrained", "both"):
        logger.info("Encoding training set with same ablation for retrained LR…")
        z_train_ablated = ablate_concepts(z_train, ablated_indices, replacement)
        x_ablated_train = decode_features(sae, z_train_ablated, device)

    # ── Set up output directories ──────────────────────────────────────────────
    tag = f"{safe_attr}_{safe_path}_t{args.threshold:.3f}"
    base_out = (
        Path(args.output_dir) if args.output_dir
        else cfg.output_root / "presentation" / "intervention" / tag
    )
    base_out.mkdir(parents=True, exist_ok=True)
    logger.info("Outputs → %s", base_out.resolve())

    # Save ablated concept list and attr_arrays (for post-hoc summary plots).
    ablated_concepts_df.to_csv(base_out / "ablated_concepts.csv", index=False)
    np.savez_compressed(
        base_out / "attr_arrays.npz",
        **{k: v.astype(str) for k, v in attr_arrays.items()},
    )
    logger.info(
        "Ablated concepts saved (%d concepts, top spec=%.4f)",
        len(ablated_indices), float(ablated_concepts_df[spec_col].iloc[0]),
    )

    # ── Evaluate conditions ────────────────────────────────────────────────────
    CONDITION_COLORS = {"baseline": "#4C78A8", "ablated": "#F58518", "retrained": "#54A24B"}
    CONDITION_LS     = {"baseline": "-",       "ablated": "--",      "retrained": ":"}

    results: list[EvalResult] = []

    # Condition A: original embeddings, original LR
    res_a = evaluate(
        x_train_for_lr=x_train, x_eval=x_valid,
        y_train=y_train, y_valid=y_valid,
        attr_arrays=attr_arrays, pathology_cols=pathology_cols,
        threshold=0.5, max_iter=args.max_iter,
        name="baseline", color=CONDITION_COLORS["baseline"], linestyle=CONDITION_LS["baseline"],
    )
    results.append(res_a)
    _save_separate(res_a, pathology_cols, attr_arrays, base_out / "baseline", 0.5)

    # Condition B: ablated embeddings, original LR (trained on x_train, not ablated x_train)
    res_b = evaluate(
        x_train_for_lr=x_train, x_eval=x_ablated_valid,
        y_train=y_train, y_valid=y_valid,
        attr_arrays=attr_arrays, pathology_cols=pathology_cols,
        threshold=0.5, max_iter=args.max_iter,
        name="ablated", color=CONDITION_COLORS["ablated"], linestyle=CONDITION_LS["ablated"],
    )
    results.append(res_b)
    _save_separate(res_b, pathology_cols, attr_arrays, base_out / "ablated", 0.5)

    # Condition C: ablated embeddings, retrained LR
    if args.lr_mode in ("retrained", "both"):
        res_c = evaluate(
            x_train_for_lr=x_ablated_train, x_eval=x_ablated_valid,
            y_train=y_train, y_valid=y_valid,
            attr_arrays=attr_arrays, pathology_cols=pathology_cols,
            threshold=0.5, max_iter=args.max_iter,
            name="retrained", color=CONDITION_COLORS["retrained"], linestyle=CONDITION_LS["retrained"],
        )
        results.append(res_c)
        _save_separate(res_c, pathology_cols, attr_arrays, base_out / "retrained", 0.5)

    # ── Comparison plots (baseline vs each non-baseline condition) ────────────
    # Each condition's folder contains a vs_baseline/ sub-directory so plots
    # are self-contained and easy to navigate.
    logger.info("Generating comparison plots…")
    non_baseline = [r for r in results if r.name != "baseline"]
    for res in non_baseline:
        comp_dir = base_out / res.name / "vs_baseline"
        comp_dir.mkdir(parents=True, exist_ok=True)
        pair = [res_a, res]

        _plot_auroc_comparison(pair, pathology_cols, comp_dir)
        _plot_tpr_disparity_comparison(pair, pathology_cols, comp_dir)
        for attr in attr_arrays:
            _plot_tpr_by_group_comparison(pair, attr, pathology_cols, comp_dir)

        # ROC comparison plots for every pathology, not just the target.
        roc_comp_dir = comp_dir / "roc_curves"
        roc_comp_dir.mkdir(exist_ok=True)
        for p_idx, pathology in enumerate(pathology_cols):
            _plot_roc_comparison(pair, pathology, args.attr, p_idx, attr_arrays, roc_comp_dir)
            _plot_roc_focused_comparison(
                res_a, res, pathology, args.attr, p_idx, attr_arrays, roc_comp_dir
            )
        logger.info("  Comparison plots → %s/vs_baseline/", res.name)

    # ── Summary print ──────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("SUMMARY  pair=(%s, %s)  SAE=%s  ablated=%d concepts",
                args.attr, args.pathology, run_label, len(ablated_indices))
    logger.info("%-20s  %8s  %8s", "condition", "macro AUROC", "disp max")
    for res in results:
        macro = float(res.auroc_df["auroc"].dropna().mean())
        disp_vals = (
            res.disparity_df[res.disparity_df.attribute == args.attr]["tpr_disparity"]
            .dropna()
        )
        max_disp = float(disp_vals.max()) if not disp_vals.empty else float("nan")
        logger.info("%-20s  %8.3f  %8.3f", res.name, macro, max_disp)
    logger.info("=" * 60)
    logger.info("Done. All outputs in %s", base_out.resolve())


if __name__ == "__main__":
    main()
