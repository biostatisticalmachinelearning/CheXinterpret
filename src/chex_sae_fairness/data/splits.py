from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SplitMasks:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    used_valid_as_test: bool = False


def build_split_masks(
    splits: np.ndarray,
    valid_name: str,
    test_name: str,
    context: str = "dataset",
    require_test: bool = True,
) -> SplitMasks:
    test_mask = splits == test_name
    valid_mask = splits == valid_name
    train_mask = ~(test_mask | valid_mask)

    # Some releases expose only train+valid. In that case, use valid as test.
    used_valid_as_test = False
    if require_test and test_mask.sum() == 0 and valid_mask.sum() > 0:
        used_valid_as_test = True
        test_mask = valid_mask.copy()
        train_mask = ~test_mask
        valid_mask = train_mask.copy()

    if valid_mask.sum() == 0:
        valid_mask = train_mask.copy()

    if require_test and test_mask.sum() == 0:
        raise ValueError(
            f"No rows found for test split '{test_name}' in {context}. "
            f"Available split values: {sorted(np.unique(splits).tolist())}."
        )

    if train_mask.sum() == 0:
        raise ValueError(f"No rows left for train split in {context} after excluding valid/test.")

    return SplitMasks(
        train=train_mask,
        valid=valid_mask,
        test=test_mask,
        used_valid_as_test=used_valid_as_test,
    )
