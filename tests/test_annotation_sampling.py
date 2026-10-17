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
        "paired_authors": 10,
        "pilot_paired_authors": 4,
        "max_reports_per_author": 2,
        "length_strata": 4,
    }


def _corpus():
    rows = []
    for label in ("lucid", "nonlucid"):
        for author_index in range(30):
            for report_index in range(2):
                word_count = 30 + ((author_index * 7 + report_index * 13) % 80)
                rows.append(
                    {
                        "post_id": f"exclusive-{label}-{author_index}-{report_index}",
                        "user_id": f"exclusive-{label}-{author_index}",
                        "lucidity": label,
                        "post_text": (
                            f"Report text {author_index} {report_index} "
                            + ("dreamcontent " * word_count)
                        ),
                    }
                )
    for author_index in range(30):
        for label, report_index in (("lucid", 0), ("nonlucid", 1)):
            word_count = 40 + ((author_index * 11 + report_index * 3) % 70)
            rows.append(
                {
                    "post_id": f"mixed-{author_index}-{label}",
                    "user_id": f"mixed-author-{author_index}",
                    "lucidity": label,
                    "post_text": (
                        f"Report text shared author {author_index} "
                        + ("dreamcontent " * word_count)
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_sample_has_paired_pilot_and_balanced_full_sample():
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
    sampled_authors = frame.iloc[manifest["source_row_index"]]["user_id"]
    assert sampled_authors.value_counts().max() <= 2
    assert generated["summary"]["maximum_sampled_reports_per_author"] == 2
    assert generated["summary"]["n_unique_authors"] == sampled_authors.nunique()
    assert generated["summary"]["paired_mixed_label_authors"] == 10
    assert generated["summary"]["independently_sampled_reports"] == 60

    paired = manifest.loc[manifest["sampling_component"] == "paired_mixed_label"]
    paired_source_authors = frame.iloc[paired["source_row_index"]]["user_id"]
    paired_labels = pd.DataFrame(
        {
            "author": paired_source_authors.to_numpy(),
            "label": paired["sampling_label"].to_numpy(),
        }
    )
    assert len(paired_labels.groupby("author")) == 10
    assert (paired_labels.groupby("author")["label"].nunique() == 2).all()
    assert (paired_labels.groupby(["author", "label"]).size() == 1).all()

    pilot = manifest.loc[manifest["pilot_double_annotate"]]
    assert len(pilot) == 16
    assert set(pilot["report_id"]).issubset(set(manifest["report_id"]))
    assert pilot["sampling_label"].value_counts().to_dict() == {
        "lucid": 8,
        "nonlucid": 8,
    }
    pilot_source_authors = frame.iloc[pilot["source_row_index"]]["user_id"]
    assert pilot_source_authors.value_counts().max() <= 2
    assert generated["summary"]["pilot_maximum_reports_per_author"] <= 2
    pilot_paired = pilot.loc[pilot["sampling_component"] == "paired_mixed_label"]
    pilot_authors = frame.iloc[pilot_paired["source_row_index"]]["user_id"]
    pilot_pairs = pd.DataFrame(
        {
            "author": pilot_authors.to_numpy(),
            "label": pilot_paired["sampling_label"].to_numpy(),
        }
    )
    assert pilot_pairs["author"].nunique() == 4
    assert (pilot_pairs.groupby("author")["label"].nunique() == 2).all()
    assert (pilot_pairs.groupby(["author", "label"]).size() == 1).all()
    assert generated["summary"]["pilot_paired_mixed_label_authors"] == 4
    assert generated["summary"]["pilot_independently_sampled_reports"] == 8

    assert len(template) == 80
    assert "dream_text" not in manifest.columns
    assert "post_text" not in manifest.columns
    assert "dream_text" not in template.columns
    _assert_annotator_files_blinded(generated, frame)
    assert (
        generated["summary"]["reports_excluded_for_detected_cross_author_duplicates"]
        == 1
    )


def _assert_annotator_files_blinded(generated, source_frame):
    forbidden_columns = {
        "sampling_label",
        "user_id",
        "author",
        "author_cluster_id",
        "paired_author_id",
        "sampling_component",
        "source_row_index",
        "source_post_id",
    }
    files = [
        generated["packet"],
        generated["pilot_template"],
        *generated["pilot_packets"].values(),
    ]
    for file in files:
        assert forbidden_columns.isdisjoint(file.columns)
        serialized = file.astype(str).to_numpy().ravel().tolist()
        content = "\n".join(serialized).lower()
        assert all(author_id.lower() not in content for author_id in source_frame["user_id"])
        assert "lucid" not in content
        assert "nonlucid" not in content
    assert list(generated["packet"].columns) == ["report_id", "dream_text"]
    assert all(
        list(packet.columns) == ["report_id", "dream_text"]
        for packet in generated["pilot_packets"].values()
    )
    assert generated["summary"]["source_labels_in_annotator_files"] is False
    assert generated["summary"]["author_identities_in_annotator_files"] is False
    assert generated["summary"]["pairing_information_in_annotator_files"] is False


def test_sample_is_reproducible_with_fixed_seeds():
    frame = _corpus()
    first, _, first_generated = sample_annotation_reports(frame, _sample_config())
    second, _, second_generated = sample_annotation_reports(frame, _sample_config())

    pd.testing.assert_frame_equal(first, second)
    assert first_generated["summary"] == second_generated["summary"]
    for annotator in first_generated["pilot_packets"]:
        pd.testing.assert_frame_equal(
            first_generated["pilot_packets"][annotator],
            second_generated["pilot_packets"][annotator],
        )


def test_sampling_fails_when_author_cap_makes_requested_size_impossible():
    frame = _corpus()
    config = _sample_config()
    config["reports_per_class"] = 100

    with pytest.raises(ValueError, match="Cannot select"):
        sample_annotation_reports(frame, config)


def test_sampling_does_not_silently_reduce_paired_author_target():
    frame = _corpus()
    config = _sample_config()
    config["paired_authors"] = 31

    with pytest.raises(ValueError, match="only 30 are feasible"):
        sample_annotation_reports(frame, config)


def test_sampling_requires_complete_author_identity():
    frame = _corpus()
    frame.loc[0, "user_id"] = None

    with pytest.raises(ValueError, match="complete author IDs"):
        sample_annotation_reports(frame, _sample_config())
