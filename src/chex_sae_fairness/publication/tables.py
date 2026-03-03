from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chex_sae_fairness.publication.statistics import bootstrap_core_metrics


def write_table(df: pd.DataFrame, output_path_no_ext: str | Path) -> dict[str, str]:
    base = Path(output_path_no_ext)
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    md_path = base.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    return {"csv": str(csv_path.resolve()), "markdown": str(md_path.resolve())}


def build_core_table_cohort(
    report: dict[str, Any],
    prediction_bundle: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    counts = report.get("counts", {})
    rows: list[dict[str, Any]] = []
    if isinstance(counts, dict):
        rows.extend(
            [
                {"section": "split_counts", "item": "n_total", "value": counts.get("n_total")},
                {"section": "split_counts", "item": "n_train", "value": counts.get("n_train")},
                {"section": "split_counts", "item": "n_valid", "value": counts.get("n_valid")},
                {"section": "split_counts", "item": "n_test", "value": counts.get("n_test")},
            ]
        )
    if prediction_bundle is not None:
        groups = prediction_bundle.get("age_groups")
        y_true = prediction_bundle.get("y_true")
        if groups is not None:
            unique, freq = np.unique(groups.astype(str), return_counts=True)
            for group, count in zip(unique.tolist(), freq.tolist()):
                rows.append({"section": "test_age_groups", "item": str(group), "value": int(count)})
        if y_true is not None:
            prevalence = np.mean(y_true.astype(float), axis=0)
            labels = prediction_bundle.get("pathology_cols")
            if labels is not None and len(labels) == len(prevalence):
                for label, prev in zip(labels.tolist(), prevalence.tolist()):
                    rows.append(
                        {
                            "section": "test_pathology_prevalence",
                            "item": str(label),
                            "value": float(prev),
                        }
                    )
    return pd.DataFrame(rows)


def build_core_table_main_results(
    report: dict[str, Any],
    prediction_bundle: dict[str, np.ndarray],
    threshold: float,
    bootstrap_samples: int = 300,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = [str(v) for v in prediction_bundle["pathology_cols"].tolist()]
    y_true = prediction_bundle["y_true"].astype(np.float32)
    groups = prediction_bundle["age_groups"].astype(str)
    method_specs = [
        ("baseline_feature_probe", "Baseline", "baseline_scores"),
        ("sae_concept_probe", "SAE Concepts", "concept_scores"),
        ("sae_concept_probe_debiased", "SAE Concepts (Debiased)", "debiased_scores"),
    ]
    for report_key, method_label, score_key in method_specs:
        perf = report.get(report_key, {}).get("performance", {})
        fair = report.get(report_key, {}).get("fairness", {})
        scores = prediction_bundle[score_key].astype(np.float32)
        ci = bootstrap_core_metrics(
            y_true=y_true,
            y_score=scores,
            age_groups=groups,
            label_names=labels,
            threshold=threshold,
            n_bootstrap=bootstrap_samples,
        )
        rows.append(
            {
                "method": method_label,
                "macro_auroc": perf.get("macro_auroc"),
                "macro_auroc_ci_low": ci.get("macro_auroc", {}).get("ci_low"),
                "macro_auroc_ci_high": ci.get("macro_auroc", {}).get("ci_high"),
                "macro_accuracy": perf.get("macro_accuracy"),
                "macro_accuracy_ci_low": ci.get("macro_accuracy", {}).get("ci_low"),
                "macro_accuracy_ci_high": ci.get("macro_accuracy", {}).get("ci_high"),
                "worst_group_macro_auroc": _worst_group_value(fair.get("worst_group_macro_auroc")),
                "worst_group_macro_auroc_ci_low": ci.get("worst_group_macro_auroc", {}).get("ci_low"),
                "worst_group_macro_auroc_ci_high": ci.get("worst_group_macro_auroc", {}).get("ci_high"),
                "macro_auroc_gap": fair.get("macro_auroc_gap"),
                "macro_auroc_gap_ci_low": ci.get("macro_auroc_gap", {}).get("ci_low"),
                "macro_auroc_gap_ci_high": ci.get("macro_auroc_gap", {}).get("ci_high"),
            }
        )
    return pd.DataFrame(rows)


def build_core_table_intervention_ablation(ablations: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in ablations:
        row = dict(result)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        by=["debias_mode", "debias_strength"],
        ascending=[True, True],
    )


def build_supplement_table_seed_stability(seed_runs: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(seed_runs).sort_values(by=["seed"])


def build_supplement_table_ablations(entries: list[dict[str, Any]], sort_by: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(entries)
    if frame.empty:
        return frame
    cols = [col for col in sort_by if col in frame.columns]
    if cols:
        return frame.sort_values(by=cols)
    return frame


def _worst_group_value(entry: Any) -> float:
    if isinstance(entry, dict):
        value = entry.get("value")
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    return float("nan")
