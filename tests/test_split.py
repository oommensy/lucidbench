import numpy as np

from lucidbench.split import author_disjoint_split


def test_author_disjoint_split_keeps_each_author_in_one_partition():
    authors = np.repeat([f"author-{i}" for i in range(10)], 2)
    labels = np.tile([0, 1], 10)
    texts = np.arange(len(authors))

    train_idx, test_idx = author_disjoint_split(
        texts, labels, groups=authors, test_size=0.3, seed=42
    )

    assert set(authors[train_idx]).isdisjoint(set(authors[test_idx]))
    assert set(labels[test_idx]) == {0, 1}
