from __future__ import annotations

from typing import Any

import numpy as np

from chex_sae_fairness.evaluation.fairness import evaluate_group_fairness, evaluate_multilabel_predictions


def evaluate_prediction_bundle(
    y_true: np.ndarray,
    y_score: np.ndarray,
    age_groups: np.ndarray,
    label_names: list[str],
    threshold: float,
    bootstrap_samples: int,
) -> dict[str, object]:
    performance = evaluate_multilabel_predictions(
        y_true=y_true,
        y_score=y_score,
        label_names=label_names,
        threshold=threshold,
    )
    fairness = evaluate_group_fairness(
        y_true=y_true,
        y_score=y_score,
        groups=age_groups,
        label_names=label_names,
        threshold=threshold,
        bootstrap_samples=bootstrap_samples,
    )
    return {"performance": performance, "fairness": fairness}


def bootstrap_core_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    age_groups: np.ndarray,
    label_names: list[str],
    threshold: float,
    n_bootstrap: int = 400,
    seed: int = 13,
) -> dict[str, dict[str, float]]:
    if n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "macro_auroc": [],
        "macro_accuracy": [],
        "worst_group_macro_auroc": [],
        "macro_auroc_gap": [],
    }
    n = len(y_true)
    if n == 0:
        return {}

    indices = np.arange(n)
    for _ in range(n_bootstrap):
        sampled = rng.choice(indices, size=n, replace=True)
        out = evaluate_prediction_bundle(
            y_true=y_true[sampled],
            y_score=y_score[sampled],
            age_groups=age_groups[sampled],
            label_names=label_names,
            threshold=threshold,
            bootstrap_samples=0,
        )
        perf = out["performance"]
        fair = out["fairness"]
        if isinstance(perf, dict):
            _append_if_finite(samples["macro_auroc"], perf.get("macro_auroc"))
            _append_if_finite(samples["macro_accuracy"], perf.get("macro_accuracy"))
        if isinstance(fair, dict):
            _append_if_finite(samples["macro_auroc_gap"], fair.get("macro_auroc_gap"))
            worst = fair.get("worst_group_macro_auroc")
            if isinstance(worst, dict):
                _append_if_finite(samples["worst_group_macro_auroc"], worst.get("value"))

    return {
        key: _summarize_distribution(values)
        for key, values in samples.items()
        if len(values) > 0
    }


def _append_if_finite(values: list[float], candidate: Any) -> None:
    if isinstance(candidate, (int, float, np.integer, np.floating)):
        value = float(candidate)
        if np.isfinite(value):
            values.append(value)


def _summarize_distribution(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }
