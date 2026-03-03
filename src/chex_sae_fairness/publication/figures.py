from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chex_sae_fairness.reporting.figures import generate_study_figures, generate_sweep_figures


def generate_core_publication_figures(
    sweep_summary: pd.DataFrame,
    best_report: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    generated.extend(generate_sweep_figures(sweep_summary, out / "sweep"))
    generated.extend(generate_study_figures(best_report, out / "best_model"))
    return generated


def generate_supplement_figures(
    output_dir: str | Path,
    seed_stability: pd.DataFrame | None = None,
    uncertain_policy: pd.DataFrame | None = None,
    debias_ablation: pd.DataFrame | None = None,
    age_bin_sensitivity: pd.DataFrame | None = None,
    baseline_comparison: pd.DataFrame | None = None,
    threshold_sensitivity: pd.DataFrame | None = None,
    missingness_sensitivity: pd.DataFrame | None = None,
    permutation_control: pd.DataFrame | None = None,
    concept_precision_recall: pd.DataFrame | None = None,
    concept_permutation: pd.DataFrame | None = None,
    view_sensitivity: pd.DataFrame | None = None,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_style()
    generated: list[Path] = []

    if seed_stability is not None and not seed_stability.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=seed_stability, x="method", y="macro_auroc", ax=ax, color="#8DA0CB")
        sns.stripplot(data=seed_stability, x="method", y="macro_auroc", ax=ax, color="#1B3C73", alpha=0.75)
        ax.set_title("Seed Stability: Macro AUROC")
        ax.set_xlabel("Method")
        ax.set_ylabel("Macro AUROC")
        generated.append(_save(fig, out / "seed_stability_macro_auroc.png"))

    if uncertain_policy is not None and not uncertain_policy.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=uncertain_policy,
            x="uncertain_policy",
            y="macro_auroc",
            hue="method",
            palette="Set2",
            ax=ax,
        )
        ax.set_title("Uncertain Label Policy Sensitivity")
        ax.set_xlabel("Uncertain label policy")
        ax.set_ylabel("Macro AUROC")
        generated.append(_save(fig, out / "uncertain_policy_sensitivity.png"))

    if debias_ablation is not None and not debias_ablation.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=debias_ablation,
            x="debias_strength",
            y="worst_group_macro_auroc",
            hue="debias_mode",
            marker="o",
            ax=ax,
        )
        ax.set_title("Debias Strength/Mode Sensitivity")
        ax.set_xlabel("Debias strength")
        ax.set_ylabel("Worst-group macro AUROC")
        generated.append(_save(fig, out / "debias_mode_strength_sensitivity.png"))

    if age_bin_sensitivity is not None and not age_bin_sensitivity.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=age_bin_sensitivity,
            x="age_bins",
            y="macro_auroc_gap",
            hue="method",
            palette="Set2",
            ax=ax,
        )
        ax.set_title("Age Bin Sensitivity (Macro AUROC Gap)")
        ax.set_xlabel("Age bin scheme")
        ax.set_ylabel("Macro AUROC gap")
        ax.tick_params(axis="x", rotation=25)
        generated.append(_save(fig, out / "age_bin_sensitivity.png"))

    if baseline_comparison is not None and not baseline_comparison.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=baseline_comparison, x="method", y="macro_auroc", palette="Set2", ax=ax)
        ax.set_title("Alternative Baselines: Macro AUROC")
        ax.set_xlabel("Method")
        ax.set_ylabel("Macro AUROC")
        ax.tick_params(axis="x", rotation=20)
        generated.append(_save(fig, out / "alternative_baselines_macro_auroc.png"))
        if {"macro_auroc", "macro_auroc_gap", "method"}.issubset(baseline_comparison.columns):
            raw = baseline_comparison.loc[baseline_comparison["method"] == "raw"]
            if not raw.empty:
                ref_auroc = float(raw["macro_auroc"].iloc[0])
                ref_gap = float(raw["macro_auroc_gap"].iloc[0])
                pareto_df = baseline_comparison.copy()
                pareto_df["performance_drop"] = ref_auroc - pareto_df["macro_auroc"].astype(float)
                pareto_df["fairness_gain"] = ref_gap - pareto_df["macro_auroc_gap"].astype(float)
                pareto_df = pareto_df.dropna(subset=["performance_drop", "fairness_gain"])
                if not pareto_df.empty:
                    fig, ax = plt.subplots(figsize=(9, 7))
                    sns.scatterplot(
                        data=pareto_df,
                        x="performance_drop",
                        y="fairness_gain",
                        hue="method",
                        s=130,
                        palette="tab10",
                        ax=ax,
                    )
                    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
                    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
                    ax.set_title("Baseline Pareto: Fairness Gain vs AUROC Drop")
                    ax.set_xlabel("Macro AUROC drop vs raw baseline")
                    ax.set_ylabel("Macro AUROC-gap improvement")
                    generated.append(_save(fig, out / "baseline_pareto_front.png"))

    if threshold_sensitivity is not None and not threshold_sensitivity.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=threshold_sensitivity,
            x="threshold",
            y="macro_auroc_gap",
            hue="method",
            marker="o",
            ax=ax,
        )
        ax.set_title("Fairness Threshold Sensitivity")
        ax.set_xlabel("Classification threshold")
        ax.set_ylabel("Macro AUROC gap")
        generated.append(_save(fig, out / "threshold_sensitivity.png"))

    if missingness_sensitivity is not None and not missingness_sensitivity.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        style = "missing_type" if "missing_type" in missingness_sensitivity.columns else None
        sns.lineplot(
            data=missingness_sensitivity,
            x="missing_fraction",
            y="macro_auroc",
            hue="method",
            style=style,
            marker="o",
            ax=ax,
        )
        ax.set_title("Missingness Sensitivity")
        ax.set_xlabel("Simulated missing fraction")
        ax.set_ylabel("Macro AUROC")
        generated.append(_save(fig, out / "missingness_sensitivity.png"))

    if permutation_control is not None and not permutation_control.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(
            permutation_control["null_mean_pathology_corr"],
            bins=20,
            color="#9ECAE1",
            edgecolor="white",
            ax=ax,
        )
        observed = permutation_control["observed_mean_pathology_corr"].iloc[0]
        ax.axvline(observed, color="#C0392B", linestyle="--", linewidth=2)
        ax.set_title("Permutation Control: Pathology Correlation Null")
        ax.set_xlabel("Mean pathology max |corr| under permutation")
        ax.set_ylabel("Count")
        generated.append(_save(fig, out / "permutation_control_histogram.png"))

    if concept_precision_recall is not None and not concept_precision_recall.empty:
        plot_df = concept_precision_recall.copy()
        plot_df = plot_df.dropna(subset=["f1"])
        if not plot_df.empty:
            top = plot_df.sort_values("f1", ascending=False).head(25)
            fig, ax = plt.subplots(figsize=(11, 8))
            sns.barplot(data=top, y="concept", x="f1", hue="concept_type", dodge=False, palette="Set2", ax=ax)
            ax.set_title("Top Concept F1 Scores")
            ax.set_xlabel("F1")
            ax.set_ylabel("Concept")
            generated.append(_save(fig, out / "concept_precision_recall_top_f1.png"))

    if concept_permutation is not None and not concept_permutation.empty and "p_adj_bh" in concept_permutation.columns:
        plot_df = concept_permutation.copy().dropna(subset=["p_adj_bh"])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.histplot(plot_df["p_adj_bh"], bins=20, color="#A3BE8C", edgecolor="white", ax=ax)
            ax.set_title("Concept Permutation Control (BH-adjusted p-values)")
            ax.set_xlabel("Adjusted p-value")
            ax.set_ylabel("Count")
            generated.append(_save(fig, out / "concept_permutation_padj_hist.png"))

    if view_sensitivity is not None and not view_sensitivity.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=view_sensitivity, x="view_type", y="macro_auroc", hue="method", palette="Set2", ax=ax)
        ax.set_title("View-Type Sensitivity")
        ax.set_xlabel("View type")
        ax.set_ylabel("Macro AUROC")
        generated.append(_save(fig, out / "view_type_sensitivity.png"))

    return generated


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
