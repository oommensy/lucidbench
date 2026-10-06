import pandas as pd
import pytest

from lucidbench.annotation_sampling import sample_annotation_reports


def _sample_config():
    return {
        "seed": 303,
        "pilot_seed": 30303,
        "text_column_candidates": ["post_text"],
        "label_column_candidates": ["lucidity"],
        "author_column_candidates": ["user_id"],
        "positive_labels": ["lucid"],
        "negative_labels": ["nonlucid"],
        "reports_per_class": 40,
        "pilot_reports_per_class": 8,
        "max_reports_per_author": 2,
        "length_strata": 4,
    }


def _corpus():
    rows = []
    for label, prefix in (("lucid", "L"), ("nonlucid", "N")):
        for author_index in range(30):
            for report_index in range(2):
                word_count = 30 + ((author_index * 7 + report_index * 13) % 80)
                rows.append(
                    {
                        "post_id": f"{prefix}-{author_index}-{report_index}",
                        "user_id": f"{prefix}-author-{author_index}",
                        "lucidity": label,
                        "post_text": (
                            f"{prefix} report {author_index} {report_index} "
                            + ("dreamcontent " * word_count)
                        ),
                    }
                )
    for label in ("lucid", "nonlucid"):
        rows.append(
            {
                "post_id": f"mixed-{label}",
                "user_id": "mixed-author",
                "lucidity": label,
                "post_text": f"mixed author report labeled {label} " * 20,
            }
        )
    return pd.DataFrame(rows)


def test_sample_balances_classes_caps_authors_and_nests_pilot():
    frame = _corpus()
    manifest, template, generated = sample_annotation_reports(
        frame, _sample_config(), excluded_duplicate_indices={0}
    )

    assert len(manifest) == 80
    assert manifest["report_id"].is_unique
    assert manifest["sampling_label"].value_counts().to_dict() == {
        "lucid": 40,
        "nonlucid": 40,
    }
    selected_authors = frame.iloc[manifest["source_row_index"]]["user_id"]
    assert selected_authors.value_counts().max() <= 2
    sampled_labels_by_author = (
        pd.DataFrame(
            {"author": selected_authors.to_numpy(), "label": manifest["sampling_label"]}
        )
        .groupby("author")["label"]
        .nunique()
    )
    assert sampled_labels_by_author.max() == 1

    pilot = manifest.loc[manifest["pilot_double_annotate"]]
    assert len(pilot) == 16
    assert set(pilot["report_id"]).issubset(set(manifest["report_id"]))
    assert pilot["sampling_label"].value_counts().to_dict() == {
        "lucid": 8,
        "nonlucid": 8,
    }
    pilot_authors = frame.iloc[pilot["source_row_index"]]["user_id"]
    assert pilot_authors.value_counts().max() <= 2

    assert len(template) == 80
    assert "dream_text" not in manifest.columns
    assert "post_text" not in manifest.columns
    assert "dream_text" not in template.columns
    assert "dream_text" in generated["packet"].columns
    assert (
        generated["summary"]["reports_excluded_for_detected_cross_author_duplicates"]
        == 1
    )


def test_sample_is_reproducible_with_fixed_seeds():
    frame = _corpus()
    first, _, _ = sample_annotation_reports(frame, _sample_config())
    second, _, _ = sample_annotation_reports(frame, _sample_config())

    pd.testing.assert_frame_equal(first, second)


def test_sampling_fails_when_author_cap_makes_requested_size_impossible():
    frame = _corpus()
    config = _sample_config()
    config["reports_per_class"] = 100

    with pytest.raises(ValueError, match="Cannot select"):
        sample_annotation_reports(frame, config)


def test_sampling_requires_complete_author_identity():
    frame = _corpus()
    frame.loc[0, "user_id"] = None

    with pytest.raises(ValueError, match="complete author IDs"):
        sample_annotation_reports(frame, _sample_config())
