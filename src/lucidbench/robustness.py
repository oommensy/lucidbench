from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import infer_columns, load_reports, normalize_binary_labels
from .experiments import (
    _build_model,
    calculate_metrics,
    remove_cues,
)
from .split import author_disjoint_split

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<\s*/?\s*[a-z][^>]*>", re.IGNORECASE)
QUOTE_PATTERN = re.compile(
    r"(?im)^\s*(?:>|(?:\[/?quote[^\]]*\])|(?:on .{1,120}wrote:))"
)
SIGNATURE_PATTERN = re.compile(
    r"(?im)^\s*(?:--\s*$|sent from my |posted by |dream journal:?|"
    r"thanks for reading|end of (?:dream|post))"
)
FORUM_ABBREVIATIONS = (
    "ld",
    "dild",
    "mild",
    "wbtb",
    "rc",
    "lol",
    "omg",
    "afaik",
    "imo",
    "imho",
)
STRUCTURAL_FEATURES = (
    "character_count",
    "word_count",
    "sentence_count",
    "newline_count",
    "mean_word_length",
    "uppercase_ratio",
    "digit_ratio",
    "url_count",
    "html_tag_count",
    "quote_marker_count",
    "quoted_line_count",
    "signature_marker_count",
    "exclamation_count",
    "question_count",
    "comma_count",
    "period_count",
    "semicolon_count",
    "colon_count",
    "parenthesis_count",
    "bracket_count",
    "asterisk_count",
    "underscore_count",
    "bullet_line_count",
    "repeated_punctuation_count",
)


def author_balanced_metrics(y_true, predictions, probabilities, authors) -> dict:
    """Weight each report by 1 / reports by author, so each author sums to weight 1."""
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    authors = np.asarray(authors).astype(str)
    _, inverse, counts = np.unique(authors, return_inverse=True, return_counts=True)
    sample_weights = 1.0 / counts[inverse]
    matrix = np.zeros((2, 2), dtype=float)
    for actual, predicted, weight in zip(y_true, predictions, sample_weights):
        matrix[actual, predicted] += weight
    recalls = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(2, dtype=float),
        where=matrix.sum(axis=1) > 0,
    )
    tp, fp = matrix[1, 1], matrix[0, 1]
    fn = matrix[1, 0]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "n_reports": len(y_true),
        "n_authors": len(np.unique(authors)),
        "weighting": "each report weight = 1 / number of held-out reports for its author",
        "accuracy": float(np.average(y_true == predictions, weights=sample_weights)),
        "balanced_accuracy": float(recalls.mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "author_weighted_confusion_matrix": matrix.tolist(),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities, sample_weight=sample_weights))
            if np.unique(y_true).size == 2
            else None
        ),
    }
    return result


def cap_reports_per_author(authors, *, maximum: int, seed: int) -> np.ndarray:
    """Select at most maximum report indices per author, reproducibly."""
    authors = np.asarray(authors).astype(str)
    rng = np.random.default_rng(seed)
    selected = []
    for author in np.unique(authors):
        indices = np.flatnonzero(authors == author)
        selected.extend(
            rng.choice(indices, size=min(maximum, len(indices)), replace=False)
        )
    return np.sort(np.asarray(selected, dtype=int))


def _normalized_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _shingle_hashes(text: str, *, size: int = 5) -> set[int]:
    words = TOKEN_PATTERN.findall(str(text).lower())
    if len(words) < size:
        return set()
    return {
        int.from_bytes(
            hashlib.blake2b(
                " ".join(words[index : index + size]).encode("utf-8"),
                digest_size=8,
            ).digest(),
            byteorder="big",
        )
        for index in range(len(words) - size + 1)
    }


def _simhash(shingles: set[int]) -> int:
    if not shingles:
        return 0
    values = np.asarray(list(shingles), dtype=">u8")
    bit_values = np.unpackbits(values.view(np.uint8)).reshape(-1, 64)[:, ::-1]
    bit_sums = bit_values.sum(axis=0)
    return sum(
        (1 << bit) for bit, score in enumerate(bit_sums) if score * 2 >= len(shingles)
    )


def find_cross_author_duplicates(
    texts,
    authors,
    *,
    near_duplicate_jaccard: float = 0.85,
    shingle_size: int = 5,
) -> list[dict]:
    """Find normalized exact and LSH-candidate near duplicates across authors."""
    texts = np.asarray(texts, dtype=object)
    authors = np.asarray(authors).astype(str)
    if len(texts) != len(authors):
        raise ValueError("texts and authors must have the same number of rows.")

    exact_groups: dict[str, list[int]] = defaultdict(list)
    shingle_sets = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, text in enumerate(texts):
        exact_groups[_normalized_text(text)].append(index)
        shingles = _shingle_hashes(str(text), size=shingle_size)
        shingle_sets.append(shingles)
        if shingles:
            signature = _simhash(shingles)
            for band in range(4):
                key = (band, (signature >> (band * 16)) & 0xFFFF)
                buckets[key].append(index)

    pair_types: dict[tuple[int, int], dict] = {}
    for indices in exact_groups.values():
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                if authors[left] != authors[right]:
                    pair_types[(left, right)] = {
                        "kind": "exact",
                        "jaccard": 1.0,
                    }

    candidates = set()
    for bucket_indices in buckets.values():
        if len(bucket_indices) < 2:
            continue
        for position, left in enumerate(bucket_indices):
            for right in bucket_indices[position + 1 :]:
                if authors[left] != authors[right]:
                    candidates.add((min(left, right), max(left, right)))

    for left, right in candidates:
        if (left, right) in pair_types:
            continue
        left_shingles, right_shingles = shingle_sets[left], shingle_sets[right]
        if not left_shingles or not right_shingles:
            continue
        union_size = len(left_shingles | right_shingles)
        similarity = len(left_shingles & right_shingles) / union_size
        if similarity >= near_duplicate_jaccard:
            pair_types[(left, right)] = {
                "kind": "near",
                "jaccard": float(similarity),
            }

    return [
        {"left": left, "right": right, **details}
        for (left, right), details in sorted(pair_types.items())
    ]


def remove_duplicate_reports(n_reports: int, duplicate_pairs: list[dict]) -> np.ndarray:
    """Return row indices not involved in any detected cross-author duplicate pair."""
    excluded = {
        index
        for pair in duplicate_pairs
        for index in (int(pair["left"]), int(pair["right"]))
    }
    return np.asarray(
        [index for index in range(n_reports) if index not in excluded], dtype=int
    )


def structural_features(text: str) -> dict[str, float]:
    """Extract formatting/shape only; no lexical content enters the artifact model."""
    text = str(text)
    words = re.findall(r"\b[\w']+\b", text)
    chars = max(len(text), 1)
    lines = text.splitlines()
    features = {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len(re.findall(r"[.!?]+", text)),
        "newline_count": text.count("\n"),
        "mean_word_length": float(np.mean([len(word) for word in words]))
        if words
        else 0.0,
        "uppercase_ratio": sum(character.isupper() for character in text) / chars,
        "digit_ratio": sum(character.isdigit() for character in text) / chars,
        "url_count": len(URL_PATTERN.findall(text)),
        "html_tag_count": len(HTML_PATTERN.findall(text)),
        "quote_marker_count": len(QUOTE_PATTERN.findall(text)),
        "quoted_line_count": sum(bool(re.match(r"^\s*>", line)) for line in lines),
        "signature_marker_count": len(SIGNATURE_PATTERN.findall(text)),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "comma_count": text.count(","),
        "period_count": text.count("."),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "parenthesis_count": text.count("(") + text.count(")"),
        "bracket_count": text.count("[") + text.count("]"),
        "asterisk_count": text.count("*"),
        "underscore_count": text.count("_"),
        "bullet_line_count": sum(
            bool(re.match(r"^\s*[-*+]\s+", line)) for line in lines
        ),
        "repeated_punctuation_count": len(re.findall(r"[!?.,;:]{2,}", text)),
    }
    return {key: float(value) for key, value in features.items()}


def forum_abbreviation_count(text: str) -> int:
    """Count forum shorthand for audit only, not as a structural feature."""
    return len(
        re.findall(
            rf"\b(?:{'|'.join(FORUM_ABBREVIATIONS)})\b",
            str(text),
            flags=re.IGNORECASE,
        )
    )


def _fit_text_model(cfg, train_texts, train_labels, test_texts):
    model = _build_model(cfg)
    model.fit(train_texts, train_labels)
    return model.predict(test_texts), model.predict_proba(test_texts)[:, 1]


def _metric_subset(y_true, predictions, probabilities) -> dict:
    values = calculate_metrics(y_true, predictions, probabilities)
    return {
        key: values[key] for key in ("f1", "balanced_accuracy", "roc_auc", "accuracy")
    }


def _distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, metric), values in frame.melt(
        id_vars=["seed", "experiment"],
        value_vars=["f1", "balanced_accuracy", "roc_auc"],
        var_name="metric",
        value_name="value",
    ).groupby(["experiment", "metric"])["value"]:
        rows.append(
            {
                "experiment": experiment,
                "metric": metric,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "standard_deviation": float(values.std(ddof=1)),
                "percentile_2_5": float(values.quantile(0.025)),
                "percentile_97_5": float(values.quantile(0.975)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _write_repeated_splits(frame, texts, labels, authors, cfg, output_dir, seeds):
    rows = []
    for seed in seeds:
        train_indices, test_indices = author_disjoint_split(
            texts,
            labels,
            groups=authors,
            test_size=float(cfg["test_size"]),
            seed=int(seed),
        )
        if not set(authors[train_indices]).isdisjoint(set(authors[test_indices])):
            raise RuntimeError(f"Author leakage detected for seed {seed}.")
        for experiment in ("baseline", "lexical_ablation"):
            train_texts = texts[train_indices]
            test_texts = texts[test_indices]
            if experiment == "lexical_ablation":
                train_texts = [
                    remove_cues(text, cfg["ablation_terms"]) for text in train_texts
                ]
                test_texts = [
                    remove_cues(text, cfg["ablation_terms"]) for text in test_texts
                ]
            predictions, probabilities = _fit_text_model(
                cfg, train_texts, labels[train_indices], test_texts
            )
            rows.append(
                {
                    "seed": int(seed),
                    "experiment": experiment,
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "n_train_authors": int(pd.Series(authors[train_indices]).nunique()),
                    "n_test_authors": int(pd.Series(authors[test_indices]).nunique()),
                    **_metric_subset(labels[test_indices], predictions, probabilities),
                }
            )
    repeated = pd.DataFrame(rows)
    repeated.to_csv(output_dir / "repeated_split_metrics.csv", index=False)
    _distribution_summary(repeated).to_csv(
        output_dir / "repeated_split_summary.csv", index=False
    )
    return repeated


def _write_capped_runs(texts, labels, authors, cfg, output_dir, seeds, maximum):
    rows = []
    for seed in seeds:
        selected = cap_reports_per_author(authors, maximum=maximum, seed=int(seed))
        capped_texts, capped_labels, capped_authors = (
            texts[selected],
            labels[selected],
            authors[selected],
        )
        train_indices, test_indices = author_disjoint_split(
            capped_texts,
            capped_labels,
            groups=capped_authors,
            test_size=float(cfg["test_size"]),
            seed=int(seed),
        )
        if not set(capped_authors[train_indices]).isdisjoint(
            set(capped_authors[test_indices])
        ):
            raise RuntimeError(f"Author leakage detected in capped run seed {seed}.")
        for experiment in ("baseline", "lexical_ablation"):
            train_texts = capped_texts[train_indices]
            test_texts = capped_texts[test_indices]
            if experiment == "lexical_ablation":
                train_texts = [
                    remove_cues(text, cfg["ablation_terms"]) for text in train_texts
                ]
                test_texts = [
                    remove_cues(text, cfg["ablation_terms"]) for text in test_texts
                ]
            predictions, probabilities = _fit_text_model(
                cfg, train_texts, capped_labels[train_indices], test_texts
            )
            rows.append(
                {
                    "seed": int(seed),
                    "experiment": experiment,
                    "max_reports_per_author": int(maximum),
                    "n_reports_after_cap": len(selected),
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "n_test_authors": int(
                        pd.Series(capped_authors[test_indices]).nunique()
                    ),
                    **_metric_subset(
                        capped_labels[test_indices], predictions, probabilities
                    ),
                }
            )
    capped = pd.DataFrame(rows)
    capped.to_csv(output_dir / "capped_author_metrics.csv", index=False)
    _distribution_summary(capped).to_csv(
        output_dir / "capped_author_summary.csv", index=False
    )
    return capped


def _audit_feature_summary(features: pd.DataFrame, labels) -> list[dict]:
    labels = np.asarray(labels, dtype=int)
    output = []
    for feature in features.columns:
        negative = features.loc[labels == 0, feature].astype(float)
        positive = features.loc[labels == 1, feature].astype(float)
        pooled_sd = float(np.sqrt((negative.var(ddof=0) + positive.var(ddof=0)) / 2))
        output.append(
            {
                "feature": feature,
                "nonlucid_mean": float(negative.mean()),
                "lucid_mean": float(positive.mean()),
                "nonlucid_prevalence": float((negative > 0).mean()),
                "lucid_prevalence": float((positive > 0).mean()),
                "standardized_mean_difference": (
                    float((positive.mean() - negative.mean()) / pooled_sd)
                    if pooled_sd
                    else 0.0
                ),
            }
        )
    return output


def _audit_artifacts(
    frame, columns, eligible, labels, output_dir
) -> tuple[dict, pd.DataFrame]:
    analytic = frame.loc[eligible].reset_index(drop=True)
    labels = np.asarray(labels, dtype=int)
    texts = analytic[columns.text].astype(str).to_numpy()
    structural = pd.DataFrame([structural_features(text) for text in texts])
    audit_features = structural.copy()
    audit_features["forum_abbreviation_count"] = [
        forum_abbreviation_count(text) for text in texts
    ]
    report = {
        "text_artifact_and_abbreviation_audit_features": _audit_feature_summary(
            audit_features, labels
        ),
        "forum_abbreviation_count_excluded_from_artifact_model": True,
        "metadata_completeness_by_label": {},
        "metadata_distribution_by_label": {},
        "author_label_distribution": {},
        "text": {
            "url_pattern": URL_PATTERN.pattern,
            "html_tag_pattern": HTML_PATTERN.pattern,
            "quote_pattern": QUOTE_PATTERN.pattern,
            "signature_pattern": SIGNATURE_PATTERN.pattern,
            "forum_abbreviations_audited": list(FORUM_ABBREVIATIONS),
        },
        "interpretation": (
            "Descriptive class differences and artifact-only metrics are a negative "
            "control, not proof that formatting caused classification performance."
        ),
    }
    metadata_columns = [
        column
        for column in (
            "title",
            "tags",
            "categories",
            "nth_post",
            "timestamp",
            "wordcount",
        )
        if column in analytic.columns
    ]
    for label, label_name in ((0, "nonlucid"), (1, "lucid")):
        subset = analytic.loc[labels == label]
        report["metadata_completeness_by_label"][label_name] = {
            column: {
                "missing_count": int(subset[column].isna().sum()),
                "missing_fraction": float(subset[column].isna().mean()),
                "unique_nonmissing_values": int(subset[column].nunique(dropna=True)),
            }
            for column in metadata_columns
        }
        report["metadata_distribution_by_label"][label_name] = {}
        for column in ("nth_post", "wordcount"):
            if column in subset:
                values = pd.to_numeric(subset[column], errors="coerce").dropna()
                report["metadata_distribution_by_label"][label_name][column] = {
                    "median": float(values.median()),
                    "p25": float(values.quantile(0.25)),
                    "p75": float(values.quantile(0.75)),
                }
        if "title" in subset:
            report["metadata_distribution_by_label"][label_name][
                "title_character_count_median"
            ] = float(subset["title"].fillna("").astype(str).str.len().median())
        if "timestamp" in subset:
            timestamp_years = pd.to_datetime(
                subset["timestamp"], errors="coerce", utc=True
            ).dt.year.dropna()
            report["metadata_distribution_by_label"][label_name][
                "timestamp_year_counts"
            ] = {
                str(year): int(count)
                for year, count in timestamp_years.value_counts().sort_index().items()
            }
        if "categories" in subset:
            categories = subset["categories"].fillna("").astype(str)
            report["metadata_distribution_by_label"][label_name][
                "categories_with_lucid_term_fraction"
            ] = float(
                categories.str.contains(
                    r"(?<!non[-_])(?<![a-z])lucid(?![a-z])",
                    case=False,
                    regex=True,
                ).mean()
            )
            report["metadata_distribution_by_label"][label_name][
                "categories_with_nonlucid_term_fraction"
            ] = float(
                categories.str.contains(r"non[-_]?lucid", case=False, regex=True).mean()
            )
        if "tags" in subset:
            tags = subset["tags"].fillna("").astype(str)
            tagged = tags.str.strip().ne("")
            report["metadata_distribution_by_label"][label_name][
                "tagged_report_fraction"
            ] = float(tagged.mean())
            report["metadata_distribution_by_label"][label_name][
                "tags_with_lucid_term_fraction"
            ] = (
                float(
                    tags[tagged]
                    .str.contains(
                        r"(?<!non[-_])(?<![a-z])lucid(?![a-z])",
                        case=False,
                        regex=True,
                    )
                    .mean()
                )
                if tagged.any()
                else 0.0
            )
            report["metadata_distribution_by_label"][label_name][
                "tags_with_nonlucid_term_fraction"
            ] = (
                float(
                    tags[tagged]
                    .str.contains(r"non[-_]?lucid", case=False, regex=True)
                    .mean()
                )
                if tagged.any()
                else 0.0
            )
        if columns.author:
            author_ids = subset[columns.author].astype(str)
            report_texts = subset[columns.text].astype(str)
            own_id_mention_count = 0
            pattern_cache = {}
            for author_id, text in zip(author_ids, report_texts):
                pattern = pattern_cache.get(author_id)
                if pattern is None:
                    pattern = re.compile(
                        rf"(?<!\w){re.escape(author_id)}(?!\w)", re.IGNORECASE
                    )
                    pattern_cache[author_id] = pattern
                own_id_mention_count += bool(pattern.search(text))
            report["metadata_distribution_by_label"][label_name][
                "own_user_id_mentioned_in_text_count"
            ] = int(own_id_mention_count)
            report["metadata_distribution_by_label"][label_name][
                "own_user_id_mention_caveat"
            ] = (
                "Counts literal user_id token occurrences in each author's report; "
                "common/numeric IDs may produce false positives. IDs are not exported."
            )
    if columns.author:
        author_label_counts = (
            analytic.assign(_target=labels)
            .groupby(columns.author)["_target"]
            .agg(["min", "max", "size"])
        )
        report["author_label_distribution"] = {
            "unique_authors": len(author_label_counts),
            "authors_with_only_nonlucid_reports": int(
                (author_label_counts["max"] == 0).sum()
            ),
            "authors_with_only_lucid_reports": int(
                (author_label_counts["min"] == 1).sum()
            ),
            "authors_with_both_labels": int(
                (
                    (author_label_counts["min"] == 0)
                    & (author_label_counts["max"] == 1)
                ).sum()
            ),
            "authors_with_at_least_10_binary_reports": int(
                (author_label_counts["size"] >= 10).sum()
            ),
            "note": "Author IDs are not exported; counts are descriptive only.",
        }
    (output_dir / "artifact_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(report["text_artifact_and_abbreviation_audit_features"]).to_csv(
        output_dir / "artifact_feature_summary.csv", index=False
    )
    return report, structural


def _artifact_only_experiment(structural, labels, authors, cfg, output_dir) -> dict:
    train_indices, test_indices = author_disjoint_split(
        structural,
        labels,
        groups=authors,
        test_size=float(cfg["test_size"]),
        seed=int(cfg["seed"]),
    )
    vectorizer = DictVectorizer(sparse=False)
    train_matrix = vectorizer.fit_transform(
        structural.iloc[train_indices].to_dict(orient="records")
    )
    test_matrix = vectorizer.transform(
        structural.iloc[test_indices].to_dict(orient="records")
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=int(cfg["seed"]),
        ),
    )
    model.fit(train_matrix, labels[train_indices])
    predictions = model.predict(test_matrix)
    probabilities = model.predict_proba(test_matrix)[:, 1]
    result = {
        "model": "StandardScaler + class-weighted LogisticRegression",
        "features": list(structural.columns),
        "excluded_from_model": [
            "forum abbreviations (reported only as an artifact-audit variable)"
        ],
        "split": "GroupShuffleSplit by user_id, seed 42, same held-out authors as TF-IDF",
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "n_test_authors": int(pd.Series(authors[test_indices]).nunique()),
        "author_disjoint": True,
        "metrics": calculate_metrics(labels[test_indices], predictions, probabilities),
        "author_balanced_metrics": author_balanced_metrics(
            labels[test_indices],
            predictions,
            probabilities,
            authors[test_indices],
        ),
    }
    (output_dir / "artifact_only_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def _duplicate_audit(frame, columns, eligible, labels, cfg, output_dir) -> dict:
    author_column = columns.author
    if author_column is None or frame[author_column].isna().any():
        raise ValueError(
            "Duplicate audit requires complete author IDs in the full corpus."
        )
    all_pairs = find_cross_author_duplicates(
        frame[columns.text].astype(str).to_numpy(),
        frame[author_column].astype(str).to_numpy(),
        near_duplicate_jaccard=float(cfg["near_duplicate_jaccard"]),
        shingle_size=int(cfg["duplicate_shingle_size"]),
    )
    exact_pairs = [pair for pair in all_pairs if pair["kind"] == "exact"]
    near_pairs = [pair for pair in all_pairs if pair["kind"] == "near"]
    excluded_all = {
        index for pair in all_pairs for index in (pair["left"], pair["right"])
    }
    eligible_row_indices = np.flatnonzero(eligible.to_numpy())
    index_to_analytic = {
        int(source_index): analytic_index
        for analytic_index, source_index in enumerate(eligible_row_indices)
    }
    analytic_duplicate_pairs = [
        {
            **pair,
            "left": index_to_analytic[pair["left"]],
            "right": index_to_analytic[pair["right"]],
        }
        for pair in all_pairs
        if pair["left"] in index_to_analytic and pair["right"] in index_to_analytic
    ]
    retained_analytic = np.asarray(
        [
            analytic_index
            for analytic_index, source_index in enumerate(eligible_row_indices)
            if int(source_index) not in excluded_all
        ],
        dtype=int,
    )
    labels_array = np.asarray(labels, dtype=int)
    if (
        len(retained_analytic) == 0
        or np.unique(labels_array[retained_analytic]).size < 2
    ):
        raise ValueError("Duplicate removal left fewer than two target classes.")

    summary = {
        "method": {
            "exact": "case-fold text and collapse whitespace, then match identical normalized full reports across different user_id values",
            "near": (
                "5-word shingle 64-bit SimHash; four 16-bit locality-sensitive "
                "bands generate candidate pairs; verify distinct-author pairs by "
                f"shingle-set Jaccard >= {cfg['near_duplicate_jaccard']}"
            ),
            "note": "LSH candidate generation is approximate and may miss some near duplicates; reports are not published.",
        },
        "n_reports_scanned": len(frame),
        "n_cross_author_exact_pairs": len(exact_pairs),
        "n_cross_author_near_pairs": len(near_pairs),
        "n_reports_in_any_cross_author_pair": len(excluded_all),
        "n_binary_labeled_pairs": len(analytic_duplicate_pairs),
        "n_binary_labeled_reports_removed": len(eligible_row_indices)
        - len(retained_analytic),
        "n_binary_labeled_reports_retained": len(retained_analytic),
        "removed_pair_counts_by_kind": {
            kind: sum(pair["kind"] == kind for pair in analytic_duplicate_pairs)
            for kind in ("exact", "near")
        },
        "rerun_baseline": None,
    }

    texts = frame.loc[eligible, columns.text].astype(str).to_numpy()
    authors = frame.loc[eligible, author_column].astype(str).to_numpy()
    train_indices, test_indices = author_disjoint_split(
        texts[retained_analytic],
        labels_array[retained_analytic],
        groups=authors[retained_analytic],
        test_size=float(cfg["test_size"]),
        seed=int(cfg["seed"]),
    )
    predictions, probabilities = _fit_text_model(
        cfg,
        texts[retained_analytic][train_indices],
        labels_array[retained_analytic][train_indices],
        texts[retained_analytic][test_indices],
    )
    summary["rerun_baseline"] = {
        "seed": int(cfg["seed"]),
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "n_train_authors": int(
            pd.Series(authors[retained_analytic][train_indices]).nunique()
        ),
        "n_test_authors": int(
            pd.Series(authors[retained_analytic][test_indices]).nunique()
        ),
        "metrics": calculate_metrics(
            labels_array[retained_analytic][test_indices],
            predictions,
            probabilities,
        ),
    }
    (output_dir / "duplicate_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def run_robustness(config_path: str | Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    data_path = Path(cfg["data_path"])
    frame = load_reports(data_path)
    columns = infer_columns(frame, cfg)
    if columns.author is None:
        raise ValueError(
            "LucidBench robustness analyses require an author identity column."
        )
    labels_series = normalize_binary_labels(
        frame[columns.label], cfg["positive_labels"], cfg["negative_labels"]
    )
    eligible = labels_series.notna() & frame[columns.text].notna()
    analytic = frame.loc[eligible].reset_index(drop=True)
    if analytic[columns.author].isna().any():
        raise ValueError("Author IDs must be complete for robustness analyses.")
    labels = labels_series.loc[eligible].astype(int).to_numpy()
    texts = analytic[columns.text].astype(str).to_numpy()
    authors = analytic[columns.author].astype(str).to_numpy()
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    repeated_seeds = list(
        range(
            int(cfg["repeated_seed_start"]),
            int(cfg["repeated_seed_start"]) + int(cfg["repeated_split_count"]),
        )
    )
    capped_seeds = list(
        range(
            int(cfg["capped_seed_start"]),
            int(cfg["capped_seed_start"]) + int(cfg["capped_split_count"]),
        )
    )
    repeated = _write_repeated_splits(
        frame, texts, labels, authors, cfg, output_dir, repeated_seeds
    )
    capped = _write_capped_runs(
        texts,
        labels,
        authors,
        cfg,
        output_dir,
        capped_seeds,
        maximum=int(cfg["max_reports_per_author"]),
    )

    train_indices, test_indices = author_disjoint_split(
        texts,
        labels,
        groups=authors,
        test_size=float(cfg["test_size"]),
        seed=int(cfg["seed"]),
    )
    train_predictions, train_probabilities = _fit_text_model(
        cfg, texts[train_indices], labels[train_indices], texts[test_indices]
    )
    # Refit to obtain a separate ablation prediction on the identical held-out authors.
    ablated_train = [
        remove_cues(text, cfg["ablation_terms"]) for text in texts[train_indices]
    ]
    ablated_test = [
        remove_cues(text, cfg["ablation_terms"]) for text in texts[test_indices]
    ]
    ablated_predictions, ablated_probabilities = _fit_text_model(
        cfg, ablated_train, labels[train_indices], ablated_test
    )
    author_balanced = {
        "method": (
            "Each report is weighted by 1 / its author's held-out report count, "
            "so each author contributes total weight one. Precision, recall, F1, "
            "accuracy, balanced accuracy, and weighted ROC-AUC are computed from "
            "these report weights; this is author-equal weighting, not an average "
            "of per-author F1 values."
        ),
        "split_seed": int(cfg["seed"]),
        "n_test_reports": len(test_indices),
        "n_test_authors": int(pd.Series(authors[test_indices]).nunique()),
        "baseline": author_balanced_metrics(
            labels[test_indices],
            train_predictions,
            train_probabilities,
            authors[test_indices],
        ),
        "lexical_ablation": author_balanced_metrics(
            labels[test_indices],
            ablated_predictions,
            ablated_probabilities,
            authors[test_indices],
        ),
        "report_weighted_baseline": _metric_subset(
            labels[test_indices], train_predictions, train_probabilities
        ),
        "report_weighted_lexical_ablation": _metric_subset(
            labels[test_indices], ablated_predictions, ablated_probabilities
        ),
    }
    (output_dir / "author_balanced_metrics.json").write_text(
        json.dumps(author_balanced, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"experiment": name, **author_balanced[name]}
            for name in ("baseline", "lexical_ablation")
        ]
    ).to_csv(output_dir / "author_balanced_metrics.csv", index=False)

    _, structural = _audit_artifacts(frame, columns, eligible, labels, output_dir)
    artifact_only = _artifact_only_experiment(
        structural, labels, authors, cfg, output_dir
    )
    duplicate_audit = _duplicate_audit(
        frame, columns, eligible, labels, cfg, output_dir
    )
    summary = {
        "repeated_splits": {
            "count": len(repeated_seeds),
            "seed_start": repeated_seeds[0],
            "seed_end": repeated_seeds[-1],
            "summary_file": "repeated_split_summary.csv",
        },
        "author_balanced": author_balanced,
        "capped_author": {
            "maximum_reports_per_author": int(cfg["max_reports_per_author"]),
            "count": len(capped_seeds),
            "seed_start": capped_seeds[0],
            "seed_end": capped_seeds[-1],
            "summary_file": "capped_author_summary.csv",
        },
        "duplicate_audit": duplicate_audit,
        "artifact_only": artifact_only,
        "artifact_audit": {
            "summary_file": "artifact_audit.json",
            "n_structural_features": len(STRUCTURAL_FEATURES),
        },
    }
    (output_dir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "repeated_splits": _distribution_summary(repeated).to_dict(
                    orient="records"
                ),
                "capped_author": _distribution_summary(capped).to_dict(
                    orient="records"
                ),
                "author_balanced": author_balanced,
                "duplicate_audit": duplicate_audit,
                "artifact_only_metrics": artifact_only["metrics"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.2-robustness.yaml")
    args = parser.parse_args()
    run_robustness(args.config)


if __name__ == "__main__":
    main()
