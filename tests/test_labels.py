import pandas as pd
from lucidbench.data import normalize_binary_labels


def test_normalize_binary_labels():
    s = pd.Series(["lucid", "non-lucid", "nightmare"])
    out = normalize_binary_labels(s, ["lucid"], ["non-lucid"])
    assert out.iloc[0] == 1
    assert out.iloc[1] == 0
    assert pd.isna(out.iloc[2])
