import numpy as np

from chex_sae_fairness.publication.tables import (
    build_core_table_cohort,
    build_core_table_main_results,
)


def test_build_core_table_cohort_has_split_counts() -> None:
    report = {"counts": {"n_total": 10, "n_train": 6, "n_valid": 2, "n_test": 2}}
    frame = build_core_table_cohort(report, prediction_bundle=None)
    assert "split_counts" in frame["section"].tolist()


def test_build_core_table_main_results_returns_three_methods() -> None:
    report = {
        "baseline_feature_probe": {"performance": {"macro_auroc": 0.6, "macro_accuracy": 0.5}, "fairness": {"macro_auroc_gap": 0.1}},
        "sae_concept_probe": {"performance": {"macro_auroc": 0.65, "macro_accuracy": 0.55}, "fairness": {"macro_auroc_gap": 0.09}},
        "sae_concept_probe_debiased": {
            "performance": {"macro_auroc": 0.66, "macro_accuracy": 0.56},
            "fairness": {"macro_auroc_gap": 0.05, "worst_group_macro_auroc": {"group": "old", "value": 0.55}},
        },
        "debiasing": {"fairness_threshold": 0.5},
    }
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    bundle = {
        "y_true": y_true,
        "age_groups": np.array(["young", "young", "old", "old"]),
        "pathology_cols": np.array(["a", "b"], dtype=object),
        "baseline_scores": np.array([[0.8, 0.2], [0.4, 0.7], [0.9, 0.6], [0.2, 0.3]], dtype=np.float32),
        "concept_scores": np.array([[0.82, 0.22], [0.45, 0.72], [0.88, 0.62], [0.25, 0.35]], dtype=np.float32),
        "debiased_scores": np.array([[0.81, 0.24], [0.44, 0.71], [0.87, 0.61], [0.24, 0.34]], dtype=np.float32),
    }
    frame = build_core_table_main_results(report, bundle, threshold=0.5, bootstrap_samples=20)
    assert len(frame) == 3
