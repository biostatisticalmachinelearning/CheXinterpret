import numpy as np

from chex_sae_fairness.data.splits import build_split_masks


def test_build_split_masks_uses_valid_as_test_when_test_missing() -> None:
    splits = np.array(["train", "train", "valid", "valid"])
    masks = build_split_masks(splits, valid_name="valid", test_name="test", require_test=True)

    assert masks.used_valid_as_test is True
    assert int(masks.train.sum()) == 2
    assert int(masks.test.sum()) == 2
    assert int(masks.valid.sum()) == 2


def test_build_split_masks_allows_missing_test_for_train_only_flow() -> None:
    splits = np.array(["train", "train", "valid", "valid"])
    masks = build_split_masks(splits, valid_name="valid", test_name="test", require_test=False)

    assert masks.used_valid_as_test is False
    assert int(masks.train.sum()) == 2
    assert int(masks.valid.sum()) == 2
    assert int(masks.test.sum()) == 0
