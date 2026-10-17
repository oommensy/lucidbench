import pandas as pd

from lucidbench.annotation_schema import (
    ANNOTATION_COLUMNS,
    BINARY_FIELDS,
    ORDINAL_FIELDS,
    derive_sensory_count,
    paired_agreement,
    validate_annotation_frame,
    validate_annotation_row,
)
from lucidbench.robustness import _simhash


def _valid_row(report_id="report-1", annotator_id="annotator-1"):
    row = {
        "report_id": report_id,
        "annotator_id": annotator_id,
        "emotional_valence": "neutral_mixed",
        "confidence": "high",
        "notes": "Synthetic rationale only.",
    }
    row.update({field: 1 for field in ORDINAL_FIELDS})
    row.update({field: 0 for field in BINARY_FIELDS})
    return row


def test_schema_accepts_valid_scores_and_rejects_out_of_range_values():
    row = _valid_row()
    assert validate_annotation_row(row) == []

    row["awareness"] = 3
    row["sensory_visual"] = 2
    row["emotional_valence"] = "very_positive"
    row["confidence"] = "certain"
    errors = validate_annotation_row(row)

    assert any("awareness" in error for error in errors)
    assert any("sensory_visual" in error for error in errors)
    assert any("emotional_valence" in error for error in errors)
    assert any("confidence" in error for error in errors)


def test_annotation_frame_rejects_duplicate_report_annotator_pair():
    frame = pd.DataFrame(
        [_valid_row(), _valid_row(annotator_id="annotator-2"), _valid_row()]
    )

    errors = validate_annotation_frame(frame)

    assert any("duplicate report_id/annotator_id" in error for error in errors)


def test_sensory_count_is_derived_and_requires_all_modality_scores():
    row = _valid_row()
    row.update(
        {
            "sensory_visual": 1,
            "sensory_auditory": 0,
            "sensory_tactile": 1,
            "sensory_proprioceptive": 1,
            "sensory_smell_taste": 0,
        }
    )
    assert derive_sensory_count(row) == 3
    row["sensory_smell_taste"] = ""
    assert derive_sensory_count(row) is None


def test_agreement_is_computed_only_from_supplied_paired_ratings():
    first = _valid_row("r1", "a1")
    second = _valid_row("r1", "a2")
    frame = pd.DataFrame([first, second], columns=ANNOTATION_COLUMNS)

    result = paired_agreement(frame, annotator_ids=["a1", "a2"])

    assert result["n_reports_with_both_annotators"] == 1
    assert result["fields"]["awareness"]["n_paired"] == 1
    assert result["fields"]["awareness"]["percent_agreement"] == 1.0
    assert result["fields"]["awareness"]["cohens_kappa"] is None
    assert result["fields"]["sensory_visual"]["percent_agreement"] == 1.0
    assert result["fields"]["emotional_valence"]["krippendorff_alpha"] is None
    assert result["fields"]["emotional_valence"]["cohens_kappa"] is None


def test_blank_pilot_template_returns_null_not_fabricated_agreement():
    first = _valid_row("r1", "a1")
    second = _valid_row("r1", "a2")
    for field in (*ORDINAL_FIELDS, *BINARY_FIELDS, "emotional_valence", "confidence"):
        first[field] = ""
        second[field] = ""
    frame = pd.DataFrame([first, second], columns=ANNOTATION_COLUMNS)

    result = paired_agreement(frame, annotator_ids=["a1", "a2"])

    assert result["n_reports_with_both_annotators"] == 1
    assert result["fields"]["awareness"]["n_paired"] == 0
    assert result["fields"]["awareness"]["percent_agreement"] is None
    assert result["fields"]["awareness"]["cohens_kappa"] is None
    assert result["fields"]["awareness"]["krippendorff_alpha"] is None


def test_optimized_simhash_matches_bitwise_reference():
    shingles = {1, 2, 4, 9, 2**63, 2**64 - 1}
    expected = 0
    for bit in range(64):
        if sum(bool(value & (1 << bit)) for value in shingles) * 2 >= len(shingles):
            expected |= 1 << bit

    assert _simhash(shingles) == expected
