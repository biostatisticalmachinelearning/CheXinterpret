from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_sweep_figures(summary: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return []

    _set_pub_style()
    figures: list[Path] = []

    ranking = summary.copy()
    ranking["composite_score"] = _compute_composite_score(ranking)
    ranking = ranking.sort_values("composite_score", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=ranking,
        y="run_name",
        x="composite_score",
        hue="variant",
        dodge=False,
        palette="Set2",
        ax=ax,
    )
    ax.set_title("SAE Hyperparameter Ranking (Composite Score)")
    ax.set_xlabel("Composite score (higher is better)")
    ax.set_ylabel("Run")
    ax.legend(title="Variant")
    figures.append(_save(fig, out / "sae_hyperparam_ranking.png"))

    scatter_cols = {"mean_disentanglement", "mean_pathology_max_abs_corr", "latent_dim", "variant"}
    if scatter_cols.issubset(summary.columns):
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.scatterplot(
            data=summary,
            x="mean_disentanglement",
            y="mean_pathology_max_abs_corr",
            hue="variant",
            size="latent_dim",
            sizes=(80, 280),
            alpha=0.9,
            palette="Set2",
            ax=ax,
        )
        for _, row in summary.iterrows():
            ax.text(
                float(row["mean_disentanglement"]),
                float(row["mean_pathology_max_abs_corr"]),
                str(row["run_name"]),
                fontsize=8,
                ha="left",
                va="bottom",
            )
        ax.set_title("Disentanglement vs Pathology Correlation")
        ax.set_xlabel("Mean disentanglement")
        ax.set_ylabel("Mean max |corr| with pathology annotations")
        figures.append(_save(fig, out / "disentanglement_vs_pathology_correlation.png"))

    fairness_cols = {"concept_probe_macro_auroc", "concept_probe_worst_group_macro_auroc", "variant"}
    if fairness_cols.issubset(summary.columns):
        fig, ax = plt.subplots(figsize=(9, 7))
        hue_col = (
            "concept_probe_macro_auroc_gap"
            if "concept_probe_macro_auroc_gap" in summary.columns
            else "variant"
        )
        sns.scatterplot(
            data=summary,
            x="concept_probe_macro_auroc",
            y="concept_probe_worst_group_macro_auroc",
            hue=hue_col,
            style="variant",
            s=150,
            alpha=0.92,
            palette="viridis",
            ax=ax,
        )
        for _, row in summary.iterrows():
            ax.text(
                float(row["concept_probe_macro_auroc"]),
                float(row["concept_probe_worst_group_macro_auroc"]),
                str(row["run_name"]),
                fontsize=8,
                ha="left",
                va="bottom",
            )
        ax.set_title("Concept-Probe Performance vs Worst-Group AUROC")
        ax.set_xlabel("Macro AUROC")
        ax.set_ylabel("Worst-group macro AUROC")
        figures.append(_save(fig, out / "fairness_performance_tradeoff.png"))

    heatmap_cols = [
        "reconstruction_mse",
        "mean_disentanglement",
        "mean_pathology_max_abs_corr",
        "concept_probe_macro_auroc",
        "concept_probe_worst_group_macro_auroc",
        "concept_probe_macro_auroc_gap",
    ]
    available = [col for col in heatmap_cols if col in summary.columns]
    if available:
        metric_frame = summary.set_index("run_name")[available].copy()
        for col in metric_frame.columns:
            if col.endswith("_mse") or col.endswith("_gap"):
                metric_frame[col] = -metric_frame[col]
        standardized = metric_frame.apply(_zscore, axis=0).fillna(0.0)
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(available)), max(4.5, 0.48 * len(standardized))))
        sns.heatmap(
            standardized,
            cmap="coolwarm",
            center=0.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "z-score (higher is better)"},
            ax=ax,
        )
        ax.set_title("Run-by-Metric Scorecard")
        ax.set_xlabel("Metric")
        ax.set_ylabel("SAE run")
        figures.append(_save(fig, out / "run_metric_scorecard_heatmap.png"))

    return figures


def generate_study_figures(report: dict[str, Any], output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_pub_style()

    method_order = [
        ("baseline_feature_probe", "Baseline"),
        ("sae_concept_probe", "SAE Concepts"),
        ("sae_concept_probe_debiased", "SAE Concepts (Debiased)"),
    ]
    rows: list[dict[str, Any]] = []
    for key, label in method_order:
        entry = report.get(key, {})
        performance = entry.get("performance", {}) if isinstance(entry, dict) else {}
        fairness = entry.get("fairness", {}) if isinstance(entry, dict) else {}
        worst_auroc = fairness.get("worst_group_macro_auroc")
        worst_acc = fairness.get("worst_group_macro_accuracy")
        rows.append(
            {
                "method_key": key,
                "method": label,
                "macro_auroc": _as_float(performance.get("macro_auroc")),
                "macro_accuracy": _as_float(performance.get("macro_accuracy")),
                "micro_accuracy": _as_float(performance.get("micro_accuracy")),
                "worst_group_macro_auroc": _extract_worst_group_value(worst_auroc),
                "worst_group_macro_accuracy": _extract_worst_group_value(worst_acc),
                "macro_auroc_gap": _as_float(fairness.get("macro_auroc_gap")),
                "macro_accuracy_gap": _as_float(fairness.get("macro_accuracy_gap")),
                "equalized_odds_tpr_gap": _as_float(fairness.get("equalized_odds_tpr_gap")),
                "equalized_odds_fpr_gap": _as_float(fairness.get("equalized_odds_fpr_gap")),
            }
        )

    summary = pd.DataFrame(rows)
    figures: list[Path] = []

    perf_cols = ["macro_auroc", "macro_accuracy", "worst_group_macro_auroc", "worst_group_macro_accuracy"]
    perf_plot = (
        summary.melt(id_vars=["method"], value_vars=perf_cols, var_name="metric", value_name="value")
        .dropna(subset=["value"])
    )
    if not perf_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=perf_plot,
            x="metric",
            y="value",
            hue="method",
            palette="Set2",
            ax=ax,
        )
        ax.set_title("Classifier Performance and Worst-Group Performance")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Model")
        figures.append(_save(fig, out / "classifier_performance_summary.png"))

    gap_cols = ["macro_auroc_gap", "macro_accuracy_gap", "equalized_odds_tpr_gap", "equalized_odds_fpr_gap"]
    gap_plot = (
        summary.melt(id_vars=["method"], value_vars=gap_cols, var_name="metric", value_name="value")
        .dropna(subset=["value"])
    )
    if not gap_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=gap_plot,
            x="metric",
            y="value",
            hue="method",
            palette="Set2",
            ax=ax,
        )
        ax.set_title("Fairness Gaps (Lower Is Better)")
        ax.set_xlabel("")
        ax.set_ylabel("Gap")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Model")
        figures.append(_save(fig, out / "fairness_gap_summary.png"))

    group_rows: list[dict[str, Any]] = []
    for key, label in method_order:
        entry = report.get(key, {})
        fairness = entry.get("fairness", {}) if isinstance(entry, dict) else {}
        groups = fairness.get("groups", {}) if isinstance(fairness, dict) else {}
        if isinstance(groups, dict):
            for group_name, group_values in groups.items():
                if not isinstance(group_values, dict):
                    continue
                group_rows.append(
                    {
                        "method": label,
                        "group": str(group_name),
                        "macro_auroc": _as_float(group_values.get("macro_auroc")),
                        "macro_accuracy": _as_float(group_values.get("macro_accuracy")),
                    }
                )
    group_df = pd.DataFrame(group_rows)
    if not group_df.empty:
        for metric in ["macro_auroc", "macro_accuracy"]:
            plot_df = group_df.dropna(subset=[metric])
            if plot_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(11, 6))
            sns.barplot(
                data=plot_df,
                x="group",
                y=metric,
                hue="method",
                palette="Set2",
                ax=ax,
            )
            ax.set_title(f"Per-Group {metric.replace('_', ' ').title()}")
            ax.set_xlabel("Age group")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="Model")
            figures.append(_save(fig, out / f"group_{metric}.png"))

    latent_rows = report.get("age_associated_latents", [])
    if isinstance(latent_rows, list) and latent_rows:
        latent_df = pd.DataFrame(latent_rows).head(20)
        if {"latent_index", "age_assoc_score"}.issubset(latent_df.columns):
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.barplot(
                data=latent_df,
                y="latent_index",
                x="age_assoc_score",
                orient="h",
                color="#4C78A8",
                ax=ax,
            )
            ax.set_title("Top Age-Associated SAE Latents")
            ax.set_xlabel("Association score")
            ax.set_ylabel("Latent index")
            figures.append(_save(fig, out / "top_age_associated_latents.png"))

    train_curve = report.get("sae", {}).get("train_curve", [])
    valid_curve = report.get("sae", {}).get("valid_curve", [])
    if isinstance(train_curve, list) and train_curve:
        curve_rows = []
        for row in train_curve:
            if isinstance(row, dict):
                curve_rows.append({"split": "train", **row})
        if isinstance(valid_curve, list):
            for row in valid_curve:
                if isinstance(row, dict):
                    curve_rows.append({"split": "valid", **row})
        curve_df = pd.DataFrame(curve_rows)
        if {"epoch", "loss", "split"}.issubset(curve_df.columns):
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.lineplot(data=curve_df, x="epoch", y="loss", hue="split", marker="o", ax=ax)
            ax.set_title("SAE Training Dynamics")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            figures.append(_save(fig, out / "sae_training_curve.png"))

    return figures


def _compute_composite_score(frame: pd.DataFrame) -> pd.Series:
    positive = [
        col
        for col in [
            "mean_disentanglement",
            "mean_pathology_max_abs_corr",
            "concept_probe_macro_auroc",
            "concept_probe_worst_group_macro_auroc",
        ]
        if col in frame.columns
    ]
    negative = [
        col
        for col in [
            "reconstruction_mse",
            "concept_probe_macro_auroc_gap",
            "concept_probe_macro_accuracy_gap",
        ]
        if col in frame.columns
    ]
    if not positive and not negative:
        return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)

    score = pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
    for col in positive:
        score += _zscore(frame[col].astype(float))
    for col in negative:
        score -= _zscore(frame[col].astype(float))

    total = float(len(positive) + len(negative))
    return score / total if total > 0 else score


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - values.mean()) / std


def _extract_worst_group_value(value: Any) -> float:
    if isinstance(value, dict):
        return _as_float(value.get("value"))
    return float("nan")


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float, np.floating, np.integer)):
        out = float(value)
        return out if math.isfinite(out) else float("nan")
    return float("nan")


def _set_pub_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "font.family": "DejaVu Sans",
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
