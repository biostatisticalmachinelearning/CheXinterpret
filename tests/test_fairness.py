import numpy as np

from chex_sae_fairness.evaluation.fairness import evaluate_group_fairness


def test_group_fairness_zero_gap_for_equal_performance() -> None:
    y_true = np.array(
        [
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )
    y_score = np.array(
        [
            [0.1, 0.9],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    groups = np.array(["young", "young", "young", "young", "old", "old", "old", "old"])

    result = evaluate_group_fairness(
        y_true=y_true,
        y_score=y_score,
        groups=groups,
        label_names=["a", "b"],
        threshold=0.5,
        bootstrap_samples=0,
    )

    assert result["macro_auroc_gap"] == 0.0
    assert result["equalized_odds_tpr_gap"] == 0.0
    assert result["equalized_odds_fpr_gap"] == 0.0
