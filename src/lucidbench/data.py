from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd


@dataclass(frozen=True)
class Columns:
    text: str
    label: str
    author: str | None


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def infer_columns(df: pd.DataFrame, config: dict) -> Columns:
    text = _first_present(df.columns, config["text_column_candidates"])
    label = _first_present(df.columns, config["label_column_candidates"])
    author = _first_present(df.columns, config["author_column_candidates"])
    if text is None or label is None:
        raise ValueError(
            f"Could not infer required text/label columns. Found: {list(df.columns)}"
        )
    return Columns(text=text, label=label, author=author)


def load_reports(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run scripts/download_data.py")
    return pd.read_csv(path, sep="\t", low_memory=False)


def normalize_binary_labels(series: pd.Series, positive: list[str], negative: list[str]) -> pd.Series:
    pos = {x.strip().lower() for x in positive}
    neg = {x.strip().lower() for x in negative}

    def convert(value):
        v = str(value).strip().lower()
        if v in pos:
            return 1
        if v in neg:
            return 0
        return pd.NA

    return series.map(convert).astype("Int64")
