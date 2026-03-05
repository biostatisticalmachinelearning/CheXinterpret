#!/usr/bin/env python3
"""Presentation pipeline: linear separability and TPR disparity of CheXagent embeddings.

Trains a binary logistic regression per pathology on the training split and
evaluates on the validation split, reporting:

  - Per-pathology AUROC  (does a linear probe separate each diagnosis?)
  - TPR disparity across sex, age group, race, and insurance type
    (is sensitivity equal across subgroups?)

Outputs (written to <output_root>/presentation/):
  linear_separability.csv   — per-pathology AUROC + prevalence on validation set
  tpr_by_group.csv          — TPR per pathology × attribute × group value
  tpr_disparity.csv         — TPR disparity (max − min) per pathology × attribute
  auroc_per_pathology.png
  tpr_disparity_heatmap.png
  tpr_by_<attribute>.png    — one figure per sensitive attribute

Usage:
  python scripts/presentation_pipeline.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Sensitive attributes to analyse — must match metadata_cols in the config.
SENSITIVE_ATTRS = ["sex", "age_group", "race", "insurance_type"]

# Minimum number of positive validation examples required to report AUROC/TPR.
MIN_POSITIVES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/default.yaml)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write results. Defaults to <output_root>/presentation/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for computing TPR (default: 0.5)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max iterations for logistic regression solver (default: 1000)",
    )
    return parser.parse_args()


def load_bundle(feature_path: Path) -> dict[str, np.ndarray]:
    if not feature_path.exists():
        sys.exit(
            f"\nFeature bundle not found at {feature_path}\n"
            "Wait for `chex-run-study` to finish feature extraction, then re-run.\n"
        )
    logger.info("Loading feature bundle from %s", feature_path)
    with np.load(feature_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _clean_attr(values: np.ndarray) -> np.ndarray:
    """Replace 'nan', 'None', '' with np.nan for consistent missing-value handling."""
    out = values.astype(object)
    out[np.isin(values.astype(str), {"nan", "None", "none", ""})] = np.nan
    return out


def compute_tpr(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float | None:
    """Sensitivity (TPR) at threshold. Returns None if fewer than MIN_POSITIVES."""
    pos = y_true == 1
    if pos.sum() < MIN_POSITIVES:
        return None
    return float((y_score[pos] >= threshold).mean())


def build_tpr_table(
    y_valid: np.ndarray,
    y_scores: np.ndarray,
    pathology_cols: list[str],
    attr_arrays: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    """Per-pathology × attribute × group TPR table."""
    rows = []
    for p_idx, pathology in enumerate(pathology_cols):
        y_true = y_valid[:, p_idx]
        y_score = y_scores[:, p_idx]
        for attr_name, attr_values in attr_arrays.items():
            unique_groups = np.unique(attr_values[~pd.isna(attr_values)])
            for group_val in sorted(unique_groups):
                mask = attr_values == group_val
                tpr = compute_tpr(y_true[mask], y_score[mask], threshold)
                rows.append({
                    "pathology": pathology,
                    "attribute": attr_name,
                    "group": group_val,
                    "tpr": tpr,
                    "n_positive": int((y_true[mask] == 1).sum()),
                    "n_total": int(mask.sum()),
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    try:
        from chex_sae_fairness.config import ExperimentConfig
    except ImportError:
        sys.exit("Package not installed. Run:  pip install -e .")

    cfg = ExperimentConfig.from_yaml(args.config)
    bundle = load_bundle(cfg.feature_path)

    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_root / "presentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Outputs → %s", output_dir.resolve())

    # ── Unpack bundle ──────────────────────────────────────────────────────────
    splits = bundle["split"].astype(str)
    x = bundle["features"].astype(np.float32)
    y = bundle["y_pathology"].astype(np.float32)
    pathology_cols = [str(c) for c in bundle["pathology_cols"]]
    metadata_cols = [str(c) for c in bundle["metadata_cols"]]
    metadata = pd.DataFrame(bundle["metadata"], columns=metadata_cols)
    age_group = bundle["age_group"].astype(str)

    train_mask = splits == cfg.data.validation_split_name.replace("valid", "train").replace("train", "train")
    # Rebuild masks from split names in config
    train_mask = splits == "train"
    valid_mask = splits == cfg.data.validation_split_name

    logger.info(
        "Split sizes — train: %d  valid: %d  (threshold=%.2f)",
        train_mask.sum(), valid_mask.sum(), args.threshold,
    )

    x_train, y_train = x[train_mask], y[train_mask]
    x_valid, y_valid = x[valid_mask], y[valid_mask]

    # ── Sensitive attributes (validation set only) ─────────────────────────────
    meta_valid = metadata[valid_mask].reset_index(drop=True)
    age_group_valid = age_group[valid_mask]

    attr_arrays: dict[str, np.ndarray] = {"age_group": _clean_attr(age_group_valid)}
    for attr in ["sex", "race", "insurance_type"]:
        if attr in meta_valid.columns:
            attr_arrays[attr] = _clean_attr(meta_valid[attr].to_numpy())
        else:
            logger.warning("Attribute '%s' not found in metadata; skipping.", attr)

    # ── Scale features ────────────────────────────────────────────────────────
    logger.info("Fitting StandardScaler on training features.")
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_valid_s = scaler.transform(x_valid)

    # ── Train one logistic regression per pathology ───────────────────────────
    y_scores = np.full_like(y_valid, fill_value=0.0)
    auroc_rows = []

    logger.info("Training %d binary logistic regression classifiers…", len(pathology_cols))
    for p_idx, pathology in enumerate(pathology_cols):
        y_tr = y_train[:, p_idx]
        y_va = y_valid[:, p_idx]
        n_pos_valid = int((y_va == 1).sum())
        n_pos_train = int((y_tr == 1).sum())

        if len(np.unique(y_tr)) < 2 or n_pos_valid < MIN_POSITIVES:
            logger.warning(
                "  %-35s  SKIPPED (train_pos=%d, valid_pos=%d)",
                pathology, n_pos_train, n_pos_valid,
            )
            auroc_rows.append({
                "pathology": pathology,
                "auroc": float("nan"),
                "prevalence_valid": float(y_va.mean()),
                "n_positive_valid": n_pos_valid,
                "n_valid": int(valid_mask.sum()),
            })
            continue

        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=args.max_iter,
            random_state=13,
        )
        clf.fit(x_train_s, y_tr)
        prob = clf.predict_proba(x_valid_s)[:, 1]
        y_scores[:, p_idx] = prob

        auroc = float(roc_auc_score(y_va, prob))
        logger.info(
            "  %-35s  AUROC=%.3f  prevalence=%.1f%%  n_pos=%d",
            pathology, auroc, 100 * y_va.mean(), n_pos_valid,
        )
        auroc_rows.append({
            "pathology": pathology,
            "auroc": auroc,
            "prevalence_valid": float(y_va.mean()),
            "n_positive_valid": n_pos_valid,
            "n_valid": int(valid_mask.sum()),
        })

    auroc_df = pd.DataFrame(auroc_rows).sort_values("auroc", ascending=False)
    auroc_path = output_dir / "linear_separability.csv"
    auroc_df.to_csv(auroc_path, index=False)
    logger.info("Saved → %s", auroc_path)

    macro_auroc = auroc_df["auroc"].dropna().mean()
    logger.info("Macro AUROC across %d pathologies: %.3f", auroc_df["auroc"].notna().sum(), macro_auroc)

    # ── TPR by group ──────────────────────────────────────────────────────────
    logger.info("Computing TPR per subgroup…")
    tpr_df = build_tpr_table(y_valid, y_scores, pathology_cols, attr_arrays, args.threshold)
    tpr_path = output_dir / "tpr_by_group.csv"
    tpr_df.to_csv(tpr_path, index=False)
    logger.info("Saved → %s", tpr_path)

    disparity_df = (
        tpr_df.dropna(subset=["tpr"])
        .groupby(["pathology", "attribute"])["tpr"]
        .agg(tpr_max="max", tpr_min="min")
        .assign(tpr_disparity=lambda d: d["tpr_max"] - d["tpr_min"])
        .reset_index()
    )
    disparity_path = output_dir / "tpr_disparity.csv"
    disparity_df.to_csv(disparity_path, index=False)
    logger.info("Saved → %s", disparity_path)

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_auroc(auroc_df, output_dir)
    _plot_tpr_disparity_heatmap(disparity_df, pathology_cols, output_dir)
    for attr in attr_arrays:
        _plot_tpr_by_group(tpr_df, attr, pathology_cols, output_dir)

    logger.info("Plotting ROC curves (one figure per pathology)…")
    roc_dir = output_dir / "roc_curves"
    roc_dir.mkdir(exist_ok=True)
    for p_idx, pathology in enumerate(pathology_cols):
        _plot_roc_for_pathology(
            y_valid=y_valid[:, p_idx],
            y_score=y_scores[:, p_idx],
            pathology=pathology,
            attr_arrays=attr_arrays,
            output_dir=roc_dir,
        )

    logger.info("Done. All outputs in %s", output_dir.resolve())


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_auroc(auroc_df: pd.DataFrame, output_dir: Path) -> None:
    df = auroc_df.dropna(subset=["auroc"]).sort_values("auroc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.45)))
    bars = ax.barh(df["pathology"], df["auroc"], color="#4C78A8", height=0.6)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.axvline(0.7, color="#F58518", linestyle=":", linewidth=0.8, label="AUROC = 0.70")
    ax.set_xlabel("AUROC (validation set)")
    ax.set_title(
        "Linear Separability of CheXagent Embeddings\n"
        "Binary Logistic Regression per Pathology"
    )
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, df["auroc"]):
        ax.text(
            min(val + 0.01, 1.0), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8,
        )
    fig.tight_layout()
    path = output_dir / "auroc_per_pathology.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", path)


def _plot_tpr_disparity_heatmap(
    disparity_df: pd.DataFrame, pathology_cols: list[str], output_dir: Path
) -> None:
    if disparity_df.empty:
        return
    pivot = disparity_df.pivot(index="pathology", columns="attribute", values="tpr_disparity")
    pivot = pivot.reindex([p for p in pathology_cols if p in pivot.index])
    n_cols = len(pivot.columns)
    n_rows = len(pivot)
    fig, ax = plt.subplots(figsize=(n_cols * 2.2 + 1.5, n_rows * 0.55 + 2))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        cbar_kws={"label": "TPR disparity (max − min)"},
    )
    ax.set_title("TPR Disparity Across Subgroups per Pathology\n(higher = larger gap in sensitivity)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    path = output_dir / "tpr_disparity_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", path)


def _plot_tpr_by_group(
    tpr_df: pd.DataFrame,
    attribute: str,
    pathology_cols: list[str],
    output_dir: Path,
) -> None:
    df = tpr_df[(tpr_df["attribute"] == attribute) & tpr_df["tpr"].notna()].copy()
    if df.empty:
        return

    groups = sorted(df["group"].unique())
    n_groups = len(groups)
    # Keep pathology order consistent with pathology_cols
    pathologies = [p for p in pathology_cols if p in df["pathology"].unique()]
    n_path = len(pathologies)

    fig, ax = plt.subplots(figsize=(max(10, n_path * 0.9), 5))
    x = np.arange(n_path)
    width = min(0.8 / n_groups, 0.25)
    palette = sns.color_palette("tab10", n_groups)

    for i, group in enumerate(groups):
        grp = df[df["group"] == group].set_index("pathology")["tpr"]
        tprs = [grp.get(p, float("nan")) for p in pathologies]
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, tprs, width, label=str(group), color=palette[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(pathologies, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Sensitivity by {attribute.replace('_', ' ').title()} per Pathology")
    ax.legend(
        title=attribute.replace("_", " ").title(),
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout()
    path = output_dir / f"tpr_by_{attribute}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", path)


def _plot_roc_for_pathology(
    y_valid: np.ndarray,
    y_score: np.ndarray,
    pathology: str,
    attr_arrays: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """One figure per pathology: 2×2 subplots, one per sensitive attribute.

    Each subplot shows the overall ROC curve (black) plus per-group ROC curves
    (coloured), with AUROC values in the legend.
    """
    from sklearn.metrics import roc_curve

    if len(np.unique(y_valid)) < 2:
        logger.debug("Skipping ROC plot for %s — only one class in validation set.", pathology)
        return

    attrs = list(attr_arrays.items())
    n_attrs = len(attrs)
    n_cols = 2
    n_rows = (n_attrs + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5), squeeze=False)
    fig.suptitle(f"ROC Curves — {pathology}", fontsize=13, fontweight="bold", y=1.01)

    overall_fpr, overall_tpr, _ = roc_curve(y_valid, y_score)
    overall_auroc = float(roc_auc_score(y_valid, y_score))

    for ax_idx, (attr_name, attr_values) in enumerate(attrs):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]

        # Overall ROC
        ax.plot(
            overall_fpr, overall_tpr,
            color="black", linewidth=2, linestyle="--",
            label=f"Overall (AUC={overall_auroc:.3f})",
            zorder=5,
        )

        # Per-group ROC curves
        groups = sorted(np.unique(attr_values[~pd.isna(attr_values)]))
        palette = sns.color_palette("tab10", len(groups))
        for color, group_val in zip(palette, groups):
            mask = attr_values == group_val
            y_g = y_valid[mask]
            s_g = y_score[mask]
            if len(np.unique(y_g)) < 2 or (y_g == 1).sum() < MIN_POSITIVES:
                continue
            fpr_g, tpr_g, _ = roc_curve(y_g, s_g)
            auc_g = float(roc_auc_score(y_g, s_g))
            ax.plot(fpr_g, tpr_g, color=color, linewidth=1.5,
                    label=f"{group_val} (AUC={auc_g:.3f})")

        ax.plot([0, 1], [0, 1], color="lightgrey", linewidth=0.8, linestyle=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(attr_name.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=7, loc="lower right")

    # Hide any unused subplots
    for ax_idx in range(n_attrs, n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

    fig.tight_layout()
    safe_name = pathology.replace(" ", "_").replace("/", "-")
    path = output_dir / f"roc_{safe_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved ROC → %s", path.name)


if __name__ == "__main__":
    main()
