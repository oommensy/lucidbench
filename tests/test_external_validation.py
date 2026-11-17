import pandas as pd
import pytest

from lucidbench.external_validation import prepare_external_reports


def _reports():
    return pd.DataFrame(
        {
            "rid": ["r1", "r2", "r3", "r4"],
            "pid": ["p1", "p1", "p2", "p2"],
            "group": ["lucid", "nonlucid", "recent_lucid", "recent_nonlucid"],
            "text": ["one", "two", "three", "four"],
            "human_score": [4, 1, 5, 2],
        }
    )


def test_prepare_external_reports_separates_model_packet_from_linkage():
    packet, linkage, summary = prepare_external_reports(
        _reports(),
        report_id_column="rid",
        participant_id_column="pid",
        condition_column="group",
        text_column="text",
        allowed_conditions=[
            "lucid",
            "nonlucid",
            "recent_lucid",
            "recent_nonlucid",
        ],
    )

    assert packet.columns.tolist() == ["report_id", "dream_text"]
    assert linkage.columns.tolist() == ["report_id", "participant_id", "condition"]
    assert "human_score" not in packet.columns
    assert "participant_id" not in packet.columns
    assert "condition" not in packet.columns
    assert summary["n_reports"] == 4
    assert summary["n_participants"] == 2
    assert summary["human_ratings_in_model_packet"] is False


def test_prepare_external_reports_rejects_duplicate_report_ids():
    frame = _reports()
    frame.loc[1, "rid"] = "r1"

    with pytest.raises(ValueError, match="report_id must be unique"):
        prepare_external_reports(
            frame,
            report_id_column="rid",
            participant_id_column="pid",
            condition_column="group",
            text_column="text",
        )


def test_prepare_external_reports_rejects_unexpected_condition():
    frame = _reports()
    frame.loc[0, "group"] = "unknown"

    with pytest.raises(ValueError, match="Unexpected condition values"):
        prepare_external_reports(
            frame,
            report_id_column="rid",
            participant_id_column="pid",
            condition_column="group",
            text_column="text",
            allowed_conditions=["lucid", "nonlucid"],
        )
