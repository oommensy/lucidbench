from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from lucidbench.annotation_schema import (
    ANNOTATION_COLUMNS,
    BINARY_FIELDS,
    NOMINAL_FIELDS,
    ORDINAL_FIELDS,
)
from lucidbench.pilot_annotation import (
    PROMPT_FIELDS,
    _prompt_value,
    completion_summary,
    load_session,
    parse_field_value,
    run_session,
)

ROOT = Path(__file__).resolve().parents[1]


def _packet(count: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "report_id": [f"opaque-{index}" for index in range(count)],
            "dream_text": ["synthetic fixture text" for _ in range(count)],
        }
    )


def _template(packet: pd.DataFrame, annotator: str = "annotator_1") -> pd.DataFrame:
    frame = pd.DataFrame(
        {"report_id": packet["report_id"], "annotator_id": annotator}
    )
    for column in ANNOTATION_COLUMNS[2:]:
        frame[column] = ""
    return frame.loc[:, ANNOTATION_COLUMNS]


def _answers() -> list[str]:
    values = {
        **{field: "1" for field in ORDINAL_FIELDS},
        **{field: "0" for field in BINARY_FIELDS},
        **{field: NOMINAL_FIELDS[field][0] for field in NOMINAL_FIELDS},
        "confidence": "medium",
        "notes": "",
    }
    values["control_attempt"] = "0"
    values["control_success"] = "not_applicable"
    values["control_domain"] = "none"
    return [values[field] for field in PROMPT_FIELDS]


def test_parse_field_value_accepts_valid_and_rejects_invalid_inputs():
    assert parse_field_value("lucid_insight", "2") == ("2", None)
    assert parse_field_value("anomaly_noticed", "1") == ("1", None)
    assert parse_field_value("transition_outcome", "LUCID") == ("lucid", None)
    assert parse_field_value("control_success", "successful") == ("successful", None)
    assert parse_field_value("confidence", "high") == ("high", None)
    assert parse_field_value("notes", "") == ("", None)

    assert parse_field_value("lucid_insight", "3")[1] is not None
    assert parse_field_value("anomaly_noticed", "2")[1] is not None
    assert parse_field_value("transition_outcome", "maybe")[1] is not None
    assert parse_field_value("notes", "x" * 301)[1] is not None


def test_load_session_requires_blinded_packet_and_aligned_template(tmp_path):
    packet = _packet()
    template = _template(packet)
    packet_path = tmp_path / "packet.csv"
    template_path = tmp_path / "template.csv"
    output_path = tmp_path / "annotations.csv"
    packet.to_csv(packet_path, index=False)
    template.to_csv(template_path, index=False)

    loaded_packet, loaded_template = load_session(
        packet_path, template_path, output_path, "annotator_1"
    )
    assert loaded_packet.columns.tolist() == ["report_id", "dream_text"]
    assert loaded_template["report_id"].tolist() == packet["report_id"].tolist()

    unblinded = packet.assign(sampling_label="lucid")
    unblinded.to_csv(packet_path, index=False)
    with pytest.raises(ValueError, match="only report_id and dream_text"):
        load_session(packet_path, template_path, output_path, "annotator_1")


def test_load_session_rejects_old_schema_saved_annotations(tmp_path):
    packet = _packet()
    template = _template(packet)
    packet_path = tmp_path / "packet.csv"
    template_path = tmp_path / "template.csv"
    output_path = tmp_path / "annotations.csv"
    packet.to_csv(packet_path, index=False)
    template.to_csv(template_path, index=False)
    pd.DataFrame(
        {
            "report_id": packet["report_id"],
            "annotator_id": ["annotator_1"] * len(packet),
            "awareness": [""] * len(packet),
        }
    ).to_csv(output_path, index=False)

    with pytest.raises(ValueError, match="older schema"):
        load_session(packet_path, template_path, output_path, "annotator_1")


def test_prompt_retries_invalid_input_without_accepting_it():
    prompts = iter(["9", "2"])
    messages: list[str] = []
    value, command = _prompt_value(
        "lucid_insight", "", lambda _: next(prompts), messages.append
    )
    assert value == "2"
    assert command is None
    assert messages


def test_session_autosaves_each_value_and_quits_safely(tmp_path):
    packet = _packet()
    annotations = _template(packet)
    output_path = tmp_path / "annotator_1_annotations.csv"
    values = iter([*_answers(), "quit"])
    saved_during_inputs = []

    def next_value(_prompt):
        if output_path.exists():
            saved_during_inputs.append(
                pd.read_csv(output_path, dtype=str, keep_default_na=False)
            )
        return next(values)

    run_session(
        packet,
        annotations,
        output_path,
        input_fn=next_value,
        output_fn=lambda _: None,
    )

    saved = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    assert saved_during_inputs
    assert saved["report_id"].tolist() == packet["report_id"].tolist()
    assert saved.loc[0, list(ORDINAL_FIELDS)].eq("1").all()
    assert saved.loc[0, "control_attempt"] == "0"
    assert saved.loc[0, "control_success"] == "not_applicable"
    assert saved.loc[0, "control_domain"] == "none"
    assert saved.loc[0, "confidence"] == "medium"
    assert saved.loc[1, list(ORDINAL_FIELDS)].eq("").all()


def test_completion_summary_reports_missing_fields_and_duplicate_ids():
    packet = pd.DataFrame(
        {"report_id": ["opaque-1", "opaque-1"], "dream_text": ["", ""]}
    )
    annotations = _template(_packet(2))
    summary = completion_summary(packet, annotations)

    assert summary["total_completed"] == 0
    assert summary["total_remaining"] == 2
    assert summary["missing_fields"]
    assert summary["duplicate_report_ids"] == ["opaque-1"]


def test_default_completed_annotation_path_is_git_ignored():
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "data/processed/v0.3/annotator_1_annotations.csv",
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
