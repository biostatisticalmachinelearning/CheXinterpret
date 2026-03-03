import numpy as np

from chex_sae_fairness.mitigation.concept_debias import (
    apply_age_residualization,
    fit_concept_residualizer,
)


def test_apply_age_residualization_train_and_test_mode() -> None:
    z_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    z_test = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    age_train = np.array(["young", "old"])
    age_test = np.array(["young", "old"])

    residualizer = fit_concept_residualizer(z_train, age_train, strength=1.0)
    z_train_out, z_test_out = apply_age_residualization(
        residualizer=residualizer,
        z_train=z_train,
        z_test=z_test,
        age_groups_train=age_train,
        age_groups_test=age_test,
        mode="train_and_test",
    )

    assert not np.allclose(z_train_out, z_train)
    assert not np.allclose(z_test_out, z_test)


def test_apply_age_residualization_test_only_mode() -> None:
    z_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    z_test = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    age_train = np.array(["young", "old"])
    age_test = np.array(["young", "old"])

    residualizer = fit_concept_residualizer(z_train, age_train, strength=1.0)
    z_train_out, z_test_out = apply_age_residualization(
        residualizer=residualizer,
        z_train=z_train,
        z_test=z_test,
        age_groups_train=age_train,
        age_groups_test=age_test,
        mode="test_only",
    )

    assert np.allclose(z_train_out, z_train)
    assert not np.allclose(z_test_out, z_test)


def test_apply_age_residualization_train_only_mode() -> None:
    z_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    z_test = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    age_train = np.array(["young", "old"])
    age_test = np.array(["young", "old"])

    residualizer = fit_concept_residualizer(z_train, age_train, strength=1.0)
    z_train_out, z_test_out = apply_age_residualization(
        residualizer=residualizer,
        z_train=z_train,
        z_test=z_test,
        age_groups_train=age_train,
        age_groups_test=age_test,
        mode="train_only",
    )

    assert not np.allclose(z_train_out, z_train)
    assert np.allclose(z_test_out, z_test)
