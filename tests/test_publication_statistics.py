import numpy as np

from chex_sae_fairness.publication.statistics import (
    attach_multiple_testing_corrections,
    paired_bootstrap_method_tests,
)


def test_paired_bootstrap_method_tests_returns_rows() -> None:
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=np.int64)
    age_groups = np.array(["young", "young", "old", "old", "young", "old"])
    baseline = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.7, 0.6], [0.2, 0.3], [0.75, 0.25], [0.3, 0.7]],
        dtype=np.float32,
    )
    improved = np.array(
        [[0.85, 0.2], [0.2, 0.8], [0.8, 0.7], [0.1, 0.2], [0.8, 0.2], [0.2, 0.8]],
        dtype=np.float32,
    )

    rows = paired_bootstrap_method_tests(
        y_true=y_true,
        age_groups=age_groups,
        label_names=["a", "b"],
        threshold=0.5,
        method_scores={"baseline": baseline, "improved": improved},
        method_pairs=[("baseline", "improved")],
        metrics=["macro_auroc", "macro_auroc_gap"],
        n_bootstrap=50,
        seed=7,
    )

    assert len(rows) == 2
    assert {row.metric for row in rows} == {"macro_auroc", "macro_auroc_gap"}


def test_attach_multiple_testing_corrections_adds_adjusted_columns() -> None:
    rows = [
        {"metric": "a", "p_value": 0.01},
        {"metric": "b", "p_value": 0.04},
        {"metric": "c", "p_value": 0.2},
    ]
    out = attach_multiple_testing_corrections(rows, p_key="p_value")
    assert all("p_adj_bh" in row and "p_adj_holm" in row for row in out)
