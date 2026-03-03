from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass(slots=True)
class PairedBootstrapResult:
    metric: str
    method_a: str
    method_b: str
    observed_delta: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap_method_tests(
    y_true: np.ndarray,
    age_groups: np.ndarray,
    label_names: list[str],
    threshold: float,
    method_scores: dict[str, np.ndarray],
    method_pairs: list[tuple[str, str]],
    metrics: list[str],
    n_bootstrap: int = 1000,
    seed: int = 13,
) -> list[PairedBootstrapResult]:
    if n_bootstrap <= 0 or len(y_true) == 0:
        return []

    metric_getters = _endpoint_metric_getters()
    unknown = [m for m in metrics if m not in metric_getters]
    if unknown:
        raise ValueError(f"Unknown metrics requested for paired bootstrap: {unknown}")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = np.arange(n)

    # Cache observed endpoint values per method.
    observed_eval: dict[str, dict[str, object]] = {}
    for method, scores in method_scores.items():
        observed_eval[method] = evaluate_prediction_bundle(
            y_true=y_true,
            y_score=scores,
            age_groups=age_groups,
            label_names=label_names,
            threshold=threshold,
            bootstrap_samples=0,
        )

    results: list[PairedBootstrapResult] = []
    for method_a, method_b in method_pairs:
        if method_a not in method_scores or method_b not in method_scores:
            continue
        scores_a = method_scores[method_a]
        scores_b = method_scores[method_b]

        # Bootstrap deltas keyed by metric.
        deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
        for _ in range(n_bootstrap):
            sampled = rng.choice(indices, size=n, replace=True)
            eval_a = evaluate_prediction_bundle(
                y_true=y_true[sampled],
                y_score=scores_a[sampled],
                age_groups=age_groups[sampled],
                label_names=label_names,
                threshold=threshold,
                bootstrap_samples=0,
            )
            eval_b = evaluate_prediction_bundle(
                y_true=y_true[sampled],
                y_score=scores_b[sampled],
                age_groups=age_groups[sampled],
                label_names=label_names,
                threshold=threshold,
                bootstrap_samples=0,
            )
            for metric in metrics:
                value_a = metric_getters[metric](eval_a)
                value_b = metric_getters[metric](eval_b)
                if np.isfinite(value_a) and np.isfinite(value_b):
                    deltas[metric].append(float(value_b - value_a))

        for metric in metrics:
            observed_a = metric_getters[metric](observed_eval[method_a])
            observed_b = metric_getters[metric](observed_eval[method_b])
            observed_delta = float(observed_b - observed_a) if np.isfinite(observed_a) and np.isfinite(observed_b) else float("nan")
            delta_values = deltas[metric]
            if not delta_values:
                results.append(
                    PairedBootstrapResult(
                        metric=metric,
                        method_a=method_a,
                        method_b=method_b,
                        observed_delta=observed_delta,
                        ci_low=float("nan"),
                        ci_high=float("nan"),
                        p_value=float("nan"),
                    )
                )
                continue
            arr = np.asarray(delta_values, dtype=float)
            ci_low = float(np.quantile(arr, 0.025))
            ci_high = float(np.quantile(arr, 0.975))
            p_value = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
            p_value = min(1.0, p_value)
            results.append(
                PairedBootstrapResult(
                    metric=metric,
                    method_a=method_a,
                    method_b=method_b,
                    observed_delta=observed_delta,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=p_value,
                )
            )
    return results


def benjamini_hochberg_correction(p_values: list[float]) -> list[float]:
    valid = [(idx, float(p)) for idx, p in enumerate(p_values) if np.isfinite(p)]
    corrected = [float("nan")] * len(p_values)
    if not valid:
        return corrected

    sorted_pairs = sorted(valid, key=lambda item: item[1])
    m = len(sorted_pairs)
    adjusted = [0.0] * m
    running = 1.0
    for rev_idx, (_, p) in enumerate(reversed(sorted_pairs), start=1):
        rank = m - rev_idx + 1
        value = min(running, (m / rank) * p)
        running = value
        adjusted[m - rev_idx] = value

    for (idx, _), value in zip(sorted_pairs, adjusted):
        corrected[idx] = float(min(1.0, value))
    return corrected


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    valid = [(idx, float(p)) for idx, p in enumerate(p_values) if np.isfinite(p)]
    corrected = [float("nan")] * len(p_values)
    if not valid:
        return corrected

    sorted_pairs = sorted(valid, key=lambda item: item[1])
    m = len(sorted_pairs)
    prev = 0.0
    for rank, (idx, p) in enumerate(sorted_pairs, start=1):
        value = (m - rank + 1) * p
        value = max(prev, value)
        prev = value
        corrected[idx] = float(min(1.0, value))
    return corrected


def attach_multiple_testing_corrections(
    rows: list[dict[str, Any]],
    p_key: str = "p_value",
    by: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return rows

    if not by:
        p_vals = [float(row.get(p_key, float("nan"))) for row in rows]
        bh = benjamini_hochberg_correction(p_vals)
        holm = holm_bonferroni_correction(p_vals)
        for row, q_bh, q_holm in zip(rows, bh, holm):
            row["p_adj_bh"] = q_bh
            row["p_adj_holm"] = q_holm
        return rows

    grouped_indices: dict[tuple[Any, ...], list[int]] = {}
    for idx, row in enumerate(rows):
        key = tuple(row.get(token) for token in by)
        grouped_indices.setdefault(key, []).append(idx)

    for indices in grouped_indices.values():
        p_vals = [float(rows[idx].get(p_key, float("nan"))) for idx in indices]
        bh = benjamini_hochberg_correction(p_vals)
        holm = holm_bonferroni_correction(p_vals)
        for local_idx, global_idx in enumerate(indices):
            rows[global_idx]["p_adj_bh"] = bh[local_idx]
            rows[global_idx]["p_adj_holm"] = holm[local_idx]
    return rows


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


def _endpoint_metric_getters() -> dict[str, Callable[[dict[str, object]], float]]:
    def _safe_get(bundle: dict[str, object], path: list[str]) -> float:
        cursor: Any = bundle
        for key in path:
            if not isinstance(cursor, dict):
                return float("nan")
            cursor = cursor.get(key)
        if isinstance(cursor, dict):
            value = cursor.get("value")
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
            return float("nan")
        if isinstance(cursor, (int, float, np.integer, np.floating)):
            return float(cursor)
        return float("nan")

    return {
        "macro_auroc": lambda m: _safe_get(m, ["performance", "macro_auroc"]),
        "macro_accuracy": lambda m: _safe_get(m, ["performance", "macro_accuracy"]),
        "macro_brier": lambda m: _safe_get(m, ["performance", "macro_brier"]),
        "macro_ece": lambda m: _safe_get(m, ["performance", "macro_ece"]),
        "worst_group_macro_auroc": lambda m: _safe_get(m, ["fairness", "worst_group_macro_auroc"]),
        "worst_group_macro_accuracy": lambda m: _safe_get(m, ["fairness", "worst_group_macro_accuracy"]),
        "macro_auroc_gap": lambda m: _safe_get(m, ["fairness", "macro_auroc_gap"]),
        "macro_accuracy_gap": lambda m: _safe_get(m, ["fairness", "macro_accuracy_gap"]),
    }
