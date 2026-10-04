from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def author_disjoint_split(X, y, groups=None, test_size=0.2, seed=42):
    idx = np.arange(len(X))
    if groups is None:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=y
        )
        return train_idx, test_idx

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(idx, y, groups=groups))
    return train_idx, test_idx
