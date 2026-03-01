import numpy as np
import pandas as pd

from chex_sae_fairness.evaluation.disentanglement import evaluate_disentanglement


def test_disentanglement_runs_on_mixed_targets() -> None:
    rng = np.random.default_rng(3)
    n_train = 120
    n_test = 60

    concept = rng.integers(0, 2, size=n_train + n_test)
    age = rng.normal(60, 10, size=n_train + n_test)
    sex = np.where(concept == 1, "F", "M")

    z = np.stack(
        [
            concept + rng.normal(0, 0.1, size=n_train + n_test),
            age / 100 + rng.normal(0, 0.1, size=n_train + n_test),
            rng.normal(0, 1, size=n_train + n_test),
        ],
        axis=1,
    ).astype(np.float32)

    y_path = np.stack([concept, 1 - concept], axis=1).astype(np.float32)

    z_train, z_test = z[:n_train], z[n_train:]
    y_train, y_test = y_path[:n_train], y_path[n_train:]

    metadata_train = pd.DataFrame({"age": age[:n_train], "sex": sex[:n_train]})
    metadata_test = pd.DataFrame({"age": age[n_train:], "sex": sex[n_train:]})

    result = evaluate_disentanglement(
        z_train=z_train,
        z_test=z_test,
        y_path_train=y_train,
        y_path_test=y_test,
        pathology_cols=["p1", "p2"],
        metadata_train=metadata_train,
        metadata_test=metadata_test,
        metadata_cols=["age", "sex"],
    )

    assert result["num_concepts_evaluated"] >= 3
    assert "concept_metrics" in result
