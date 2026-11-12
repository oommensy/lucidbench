import pandas as pd

from lucidbench.annotation_schema import (
    ANNOTATION_COLUMNS,
    BINARY_FIELDS,
    NOMINAL_FIELDS,
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
        "control_success": "partial_uncertain",
        "control_domain": "environment",
        "route_to_lucidity": "reasoning_based",
        "transition_outcome": "lucid",
        "confidence": "high",
        "notes": "Synthetic rationale only.",
    }
    row.update({field: 1 for field in ORDINAL_FIELDS})
    row.update({field: 0 for field in BINARY_FIELDS})
    row["control_attempt"] = 1
    return row


def test_schema_accepts_valid_scores_and_rejects_out_of_range_values():
    row = _valid_row()
    assert validate_annotation_row(row) == []

    row["lucid_insight"] = 3
    row["sensory_visual"] = 2
    row["emotional_valence"] = "very_positive"
    row["confidence"] = "certain"
    errors = validate_annotation_row(row)

    assert any("lucid_insight" in error for error in errors)
    assert any("sensory_visual" in error for error in errors)
    assert any("emotional_valence" in error for error in errors)
    assert any("confidence" in error for error in errors)


def test_control_dependency_rules_are_enforced():
    row = _valid_row()
    row["control_attempt"] = 0
    row["control_success"] = "successful"
    row["control_domain"] = "physics"
    errors = validate_annotation_row(row)
    assert any("control_success" in error for error in errors)
    assert any("control_domain" in error for error in errors)

    row["control_success"] = "not_applicable"
    row["control_domain"] = "none"
    assert validate_annotation_row(row) == []


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


def test_agreement_is_computed_for_ordinal_binary_and_nominal_fields():
    first = _valid_row("r1", "a1")
    second = _valid_row("r1", "a2")
    frame = pd.DataFrame([first, second], columns=ANNOTATION_COLUMNS)

    result = paired_agreement(frame, annotator_ids=["a1", "a2"])

    assert result["n_reports_with_both_annotators"] == 1
    assert result["fields"]["lucid_insight"]["percent_agreement"] == 1.0
    assert result["fields"]["anomaly_noticed"]["percent_agreement"] == 1.0
    assert result["fields"]["control_domain"]["percent_agreement"] == 1.0
    assert result["fields"]["transition_outcome"]["percent_agreement"] == 1.0


def test_blank_pilot_template_returns_null_not_fabricated_agreement():
    first = _valid_row("r1", "a1")
    second = _valid_row("r1", "a2")
    for field in (*ORDINAL_FIELDS, *BINARY_FIELDS, *NOMINAL_FIELDS, "confidence"):
        first[field] = ""
        second[field] = ""
    frame = pd.DataFrame([first, second], columns=ANNOTATION_COLUMNS)

    result = paired_agreement(frame, annotator_ids=["a1", "a2"])

    assert result["fields"]["lucid_insight"]["n_paired"] == 0
    assert result["fields"]["lucid_insight"]["cohens_kappa"] is None
    assert result["fields"]["transition_outcome"]["krippendorff_alpha"] is None



def test_percent_agreement_is_elementwise_across_multiple_reports():
    rows = [
        _valid_row("r1", "a1"),
        _valid_row("r1", "a2"),
        _valid_row("r2", "a1"),
        _valid_row("r2", "a2"),
    ]
    rows[3]["lucid_insight"] = 2
    frame = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)

    result = paired_agreement(frame, annotator_ids=["a1", "a2"])

    assert result["fields"]["lucid_insight"]["n_paired"] == 2
    assert result["fields"]["lucid_insight"]["percent_agreement"] == 0.5

def test_optimized_simhash_matches_bitwise_reference():
    shingles = {1, 2, 4, 9, 2**63, 2**64 - 1}
    expected = 0
    for bit in range(64):
        if sum(bool(value & (1 << bit)) for value in shingles) * 2 >= len(shingles):
            expected |= 1 << bit

    assert _simhash(shingles) == expected
