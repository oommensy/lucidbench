from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from lucidbench.annotation_schema import (
    ANNOTATION_COLUMNS,
    BINARY_FIELDS,
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
        {
            "report_id": packet["report_id"],
            "annotator_id": annotator,
        }
    )
    for column in ANNOTATION_COLUMNS[2:]:
        frame[column] = ""
    return frame.loc[:, ANNOTATION_COLUMNS]


def _answers() -> list[str]:
    values = {
        **{field: "1" for field in ORDINAL_FIELDS},
        **{field: "0" for field in BINARY_FIELDS},
        "emotional_valence": "neutral_mixed",
        "confidence": "medium",
        "notes": "",
    }
    return [values[field] for field in PROMPT_FIELDS]


def test_parse_field_value_accepts_valid_and_rejects_invalid_inputs():
    assert parse_field_value("awareness", "2") == ("2", None)
    assert parse_field_value("sensory_visual", "1") == ("1", None)
    assert parse_field_value("emotional_valence", "POSITIVE") == ("positive", None)
    assert parse_field_value("confidence", "high") == ("high", None)
    assert parse_field_value("notes", "") == ("", None)

    assert parse_field_value("awareness", "3")[1] is not None
    assert parse_field_value("sensory_visual", "2")[1] is not None
    assert parse_field_value("emotional_valence", "unknown")[1] is not None
    assert parse_field_value("confidence", "certain")[1] is not None
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


def test_resume_starts_at_first_incomplete_report_and_keeps_blind_output(tmp_path):
    packet = _packet(3)
    annotations = _template(packet)
    annotations.loc[0, list(ORDINAL_FIELDS)] = "1"
    annotations.loc[0, list(BINARY_FIELDS)] = "0"
    annotations.loc[0, "emotional_valence"] = "neutral_mixed"
    annotations.loc[0, "confidence"] = "medium"
    annotations.loc[1, list(ORDINAL_FIELDS)] = "1"
    annotations.loc[1, list(BINARY_FIELDS)] = "0"
    annotations.loc[1, "emotional_valence"] = "neutral_mixed"
    annotations.loc[1, "confidence"] = "medium"
    output_path = tmp_path / "annotator_1_annotations.csv"
    packet_path = tmp_path / "annotator_1_pilot_packet.csv"
    template_path = tmp_path / "annotator_1_annotation_template.csv"
    packet.to_csv(packet_path, index=False)
    _template(packet).to_csv(template_path, index=False)
    annotations.to_csv(output_path, index=False)
    packet, annotations = load_session(
        packet_path, template_path, output_path, "annotator_1"
    )
    prompts = iter(["quit"])
    displayed: list[str] = []

    run_session(
        packet,
        annotations,
        output_path,
        input_fn=lambda _: next(prompts),
        output_fn=displayed.append,
    )

    assert "Report ID: opaque-2" in displayed
    assert "Report ID: opaque-1" not in displayed
    forbidden = (
        "lucid",
        "nonlucid",
        "sampling_label",
        "source_row",
        "source_post",
        "user_id",
        "username",
        "author_cluster",
        "paired_author",
        "sampling_component",
        "tags",
        "categories",
    )
    output = "\n".join(displayed).lower()
    assert all(value not in output for value in forbidden)


def test_prompt_retries_invalid_input_without_accepting_it():
    prompts = iter(["9", "2"])
    messages: list[str] = []

    value, command = _prompt_value(
        "awareness", "", lambda _: next(prompts), messages.append
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

    run_session(packet, annotations, output_path, input_fn=next_value, output_fn=lambda _: None)

    saved = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    assert saved_during_inputs
    assert saved["report_id"].tolist() == packet["report_id"].tolist()
    assert saved.loc[0, list(ORDINAL_FIELDS)].eq("1").all()
    assert saved.loc[0, list(BINARY_FIELDS)].eq("0").all()
    assert saved.loc[0, "emotional_valence"] == "neutral_mixed"
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
