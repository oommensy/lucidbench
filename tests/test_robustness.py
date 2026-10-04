import json

import numpy as np
import pandas as pd
import pytest
import yaml

from lucidbench.experiments import run_experiments
from lucidbench.robustness import (
    STRUCTURAL_FEATURES,
    author_balanced_metrics,
    cap_reports_per_author,
    find_cross_author_duplicates,
    forum_abbreviation_count,
    remove_duplicate_reports,
    structural_features,
)


def test_cap_reports_per_author_is_reproducible_and_respects_limit():
    authors = np.repeat(["one", "two", "three"], [25, 7, 2])

    first = cap_reports_per_author(authors, maximum=10, seed=25)
    second = cap_reports_per_author(authors, maximum=10, seed=25)

    assert np.array_equal(first, second)
    assert np.all(np.diff(first) > 0)
    assert all(
        np.count_nonzero(authors[first] == author) <= 10 for author in set(authors)
    )
    assert len(first) == 19


def test_cross_author_exact_and_near_duplicate_detection_and_removal():
    words = [f"token{index}" for index in range(60)]
    original = " ".join(words)
    near_duplicate = " ".join(["changedword"] + words[1:])
    texts = [
        original,
        f"  {original.upper()}  ",
        near_duplicate,
        "an unrelated dream report with other words and no matching sequence",
    ]
    authors = ["author-a", "author-b", "author-c", "author-a"]

    pairs = find_cross_author_duplicates(
        texts,
        authors,
        near_duplicate_jaccard=0.85,
        shingle_size=5,
    )

    exact_pairs = {
        (pair["left"], pair["right"]) for pair in pairs if pair["kind"] == "exact"
    }
    near_pairs = {
        (pair["left"], pair["right"]) for pair in pairs if pair["kind"] == "near"
    }
    assert (0, 1) in exact_pairs
    assert (0, 2) in near_pairs
    assert all(authors[pair["left"]] != authors[pair["right"]] for pair in pairs)

    retained = remove_duplicate_reports(len(texts), pairs)
    assert set(retained).isdisjoint({0, 1, 2})
    assert 3 in retained


def test_structural_features_exclude_report_lexical_content():
    features = structural_features(
        "I realized I was dreaming! https://example.org\n> quoted line\n--\n"
    )

    assert set(features) == set(STRUCTURAL_FEATURES)
    assert features["url_count"] == 1
    assert features["quoted_line_count"] == 1
    assert features["signature_marker_count"] == 1
    assert "realized" not in features
    assert "text" not in features
    assert "forum_abbreviation_count" not in features
    assert forum_abbreviation_count("LD DILD mild") == 3


def test_author_balanced_metrics_weight_each_author_equally():
    labels = np.array([1, 1, 1, 1, 0, 0])
    predictions = np.array([1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    authors = np.array(["prolific"] * 4 + ["single-a", "single-b"])

    report_level = (predictions == labels).mean()
    author_weighted = author_balanced_metrics(
        labels, predictions, probabilities, authors
    )

    assert report_level == pytest.approx(4 / 6)
    assert author_weighted["accuracy"] == pytest.approx(5 / 6)
    assert author_weighted["n_authors"] == 3
    assert author_weighted["weighting"].startswith("each report weight")


def test_main_experiment_requires_authors_unless_explicitly_opted_in(tmp_path):
    reports = [
        {
            "post_text": f"lucid report shared phrase for sample {index}",
            "lucidity": "lucid",
        }
        for index in range(12)
    ] + [
        {
            "post_text": f"ordinary report shared phrase for sample {index}",
            "lucidity": "nonlucid",
        }
        for index in range(12)
    ]
    data_path = tmp_path / "reports.tsv"
    pd.DataFrame(reports).to_csv(data_path, sep="\t", index=False)
    output_dir = tmp_path / "results"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "data_path": str(data_path),
                "output_dir": str(output_dir),
                "text_column_candidates": ["post_text"],
                "label_column_candidates": ["lucidity"],
                "author_column_candidates": ["user_id"],
                "positive_labels": ["lucid"],
                "negative_labels": ["nonlucid"],
                "test_size": 0.25,
                "min_df": 1,
                "max_features": 500,
                "ngram_range": [1, 2],
                "C": 1.0,
                "bootstrap_replicates": 20,
                "length_bins": 2,
                "coefficient_terms": 3,
                "minimum_coefficient_author_support": 1,
                "ablation_terms": ["lucid", "dream"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="--allow-report-level-split"):
        run_experiments(config_path)

    result = run_experiments(config_path, allow_report_level_split=True)
    saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert result["author_disjoint"] is False
    assert result["leakage_sensitive"] is True
    assert saved["leakage_sensitive"] is True
