import json

import numpy as np
import pandas as pd
import yaml

from lucidbench.experiments import (
    METRIC_NAMES,
    bootstrap_author_intervals,
    calculate_metrics,
    length_match_indices,
    remove_cues,
    run_experiments,
)
from lucidbench.split import author_disjoint_split


def test_author_split_prevents_leakage_and_is_reproducible():
    authors = np.repeat([f"author-{index}" for index in range(20)], 2)
    labels = np.tile([0, 1], 20)
    texts = np.arange(len(authors))

    first_train, first_test = author_disjoint_split(
        texts, labels, groups=authors, test_size=0.25, seed=42
    )
    second_train, second_test = author_disjoint_split(
        texts, labels, groups=authors, test_size=0.25, seed=42
    )

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_test, second_test)
    assert set(authors[first_train]).isdisjoint(set(authors[first_test]))


def test_lexical_ablation_removes_only_configured_whole_words():
    text = "Lucid dreaming: I realized it was real, not a dreamer story."
    result = remove_cues(
        text,
        [
            "lucid",
            "dreaming",
            "realized",
            "real",
        ],
    )

    assert result == ": I it was, not a dreamer story."
    assert remove_cues("LD DILD RC wild wildness", ["ld", "dild", "rc"]) == (
        "wild wildness"
    )


def test_metric_output_schema_and_author_bootstrap_intervals():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.8, 0.9])
    authors = np.array(["a", "b", "c", "d"])

    metrics = calculate_metrics(labels, predictions, probabilities)
    intervals = bootstrap_author_intervals(
        labels,
        predictions,
        probabilities,
        authors,
        replicates=100,
        seed=13,
    )

    assert set(metrics) == {*METRIC_NAMES, "confusion_matrix"}
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert set(intervals) == set(METRIC_NAMES)
    assert all(set(interval) == {"lower", "upper"} for interval in intervals.values())


def test_length_matching_balances_both_labels_within_bins():
    lengths = np.array([1, 2, 3, 4, 11, 12, 13, 14, 15])
    labels = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1])
    selected = length_match_indices(lengths, labels, bins=[0, 5, 10, 20], seed=42)

    selected_lengths = lengths[selected]
    selected_labels = labels[selected]
    for lower, upper in ((0, 5), (10, 20)):
        in_bin = (selected_lengths > lower) & (selected_lengths <= upper)
        assert np.count_nonzero(in_bin & (selected_labels == 0)) == np.count_nonzero(
            in_bin & (selected_labels == 1)
        )


def test_experiment_writes_metric_output_schema(tmp_path):
    reports = []
    for author_index in range(12):
        reports.extend(
            [
                {
                    "post_text": f"I knowingly control the flying scene lucid marker {author_index}.",
                    "lucidity": "lucid",
                    "user_id": f"author-{author_index}",
                },
                {
                    "post_text": f"I walk through the quiet garden nonlucid marker {author_index}.",
                    "lucidity": "nonlucid",
                    "user_id": f"author-{author_index}",
                },
            ]
        )
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
                "ablation_terms": ["lucid", "dream", "dreaming", "real"],
            }
        ),
        encoding="utf-8",
    )

    run_experiments(config_path)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["author_disjoint"] is True
    assert set(metrics["experiments"]) == {
        "baseline",
        "lexical_ablation",
        "length_matched",
    }
    for experiment in metrics["experiments"].values():
        assert set(experiment["metrics"]) == {*METRIC_NAMES, "confusion_matrix"}
        assert set(experiment["confidence_intervals"]) == set(METRIC_NAMES)
    assert (output_dir / "confusion_matrices.csv").is_file()
    assert (output_dir / "length_matching.csv").is_file()
    assert (output_dir / "performance_comparison.png").is_file()
