from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from lucidbench.annotation_schema import (
    ANNOTATION_COLUMNS,
    BINARY_FIELDS,
    CONFIDENCE_VALUES,
    NOMINAL_FIELDS,
    ORDINAL_FIELDS,
    SCORE_FIELDS,
    validate_annotation_frame,
    validate_annotation_row,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data/processed/v0.3"
RUBRIC_PATH = ROOT / "docs/PHENOMENOLOGY_RUBRIC.md"

PROMPT_FIELDS = (
    "lucid_insight",
    "intentional_agency",
    "internal_commentary",
    "self_reflection",
    "state_reasoning",
    "waking_memory",
    "prospective_intention_memory",
    "anomaly_present",
    "anomaly_noticed",
    "reality_testing",
    "dream_rationalization",
    "route_to_lucidity",
    "transition_outcome",
    "control_attempt",
    "control_success",
    "control_domain",
    "sensory_visual",
    "sensory_auditory",
    "sensory_tactile",
    "sensory_proprioceptive",
    "sensory_smell_taste",
    "emotional_intensity",
    "emotional_valence",
    "dream_instability",
    "loss_of_lucidity",
    "awakening",
    "confidence",
    "notes",
)

FIELD_GUIDANCE = {
    "lucid_insight": "Dream-state insight: 0 not evidenced, 1 questioning/partial, 2 explicit recognition that this is a dream.",
    "intentional_agency": "Intentional action: 0 not evidenced, 1 brief/reactive choice, 2 sustained goal-directed action.",
    "internal_commentary": "In-dream wondering/commentary about events: 0 not evidenced, 1 evidenced.",
    "self_reflection": "Reflection on own thoughts, feelings, motives, or behavior: 0 not evidenced, 1 evidenced.",
    "state_reasoning": "Reasoning about current state/reality: 0 not evidenced, 1 limited, 2 explicit/sustained.",
    "waking_memory": "Waking-life retrieval: 0 not evidenced, 1 vague, 2 explicit fact/episode access.",
    "prospective_intention_memory": "Remembered waking intention/plan while dreaming: 0 not evidenced, 1 evidenced.",
    "anomaly_present": "Bizarre/impossible/inconsistent event is described: 0 no/not evidenced, 1 yes.",
    "anomaly_noticed": "Dreamer notices an anomaly as unusual: 0 not evidenced, 1 evidenced.",
    "reality_testing": "Dreamer deliberately tests whether experience is dream/waking reality: 0 not evidenced, 1 evidenced.",
    "dream_rationalization": "An anomaly is explained away within the dream instead of producing lucid insight: 0 not evidenced, 1 evidenced.",
    "route_to_lucidity": "none, reasoning_based, spontaneous_intuitive, wake_initiated, or other_unclear.",
    "transition_outcome": "no_transition, near_lucid, lucid, or unclear.",
    "control_attempt": "Deliberate attempt to influence dream content/state: 0 not evidenced, 1 evidenced.",
    "control_success": "not_applicable, failed, partial_uncertain, or successful.",
    "control_domain": "none, cognition, emotion, body, movement, characters, environment, physics, or multiple.",
    "sensory_visual": "Visual evidence described: 0 not evidenced, 1 present.",
    "sensory_auditory": "Auditory evidence described: 0 not evidenced, 1 present.",
    "sensory_tactile": "Tactile evidence described: 0 not evidenced, 1 present.",
    "sensory_proprioceptive": "Body movement/position/balance evidence: 0 not evidenced, 1 present.",
    "sensory_smell_taste": "Smell or taste evidence described: 0 not evidenced, 1 present.",
    "emotional_intensity": "Emotional strength: 0 none/not evidenced, 1 mild/moderate, 2 strong/intense.",
    "emotional_valence": "Predominant valence: negative, neutral_mixed, or positive.",
    "dream_instability": "Fading/instability: 0 not evidenced/stable, 1 threatened/limited, 2 clear collapse or major instability.",
    "loss_of_lucidity": "Explicit transition from lucid to non-lucid: 0 not evidenced, 1 evidenced.",
    "awakening": "Explicit awakening/termination into wakefulness: 0 not evidenced, 1 evidenced.",
    "confidence": "Coding clarity: low, medium, or high.",
    "notes": "Optional short paraphrase only; no quotations or identifying details.",
}

COMMANDS = {
    "next": "next",
    "next report": "next",
    "previous": "previous",
    "previous report": "previous",
    "save": "save",
    "quit": "quit",
    "q": "quit",
    "rubric": "rubric",
    "show rubric summary": "rubric",
    "progress": "progress",
    "show progress count": "progress",
}


def parse_field_value(field: str, value: str) -> tuple[str | None, str | None]:
    """Return a normalized valid value or an error without changing the input row."""
    value = value.strip()
    if field == "notes":
        if len(value) > 300:
            return None, "Notes must be 300 characters or fewer."
        return value, None
    if not value:
        return "", None

    if field in ORDINAL_FIELDS:
        allowed = {str(option) for option in ORDINAL_FIELDS[field]}
    elif field in BINARY_FIELDS:
        allowed = {"0", "1"}
    elif field in NOMINAL_FIELDS:
        allowed = set(NOMINAL_FIELDS[field])
        value = value.lower()
    elif field == "confidence":
        allowed = set(CONFIDENCE_VALUES)
        value = value.lower()
    else:
        raise ValueError(f"Unknown annotation field: {field}")

    if value not in allowed:
        return None, f"Enter one of: {', '.join(sorted(allowed))}."

    candidate = {
        "report_id": "validation",
        "annotator_id": "validation",
        field: value,
    }
    errors = validate_annotation_row(candidate)
    if errors:
        return None, errors[0]
    return value, None


def load_session(
    packet_path: Path,
    template_path: Path,
    annotations_path: Path,
    annotator_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    packet = pd.read_csv(packet_path, dtype=str, keep_default_na=False)
    if packet.columns.tolist() != ["report_id", "dream_text"]:
        raise ValueError("The packet must contain only report_id and dream_text.")
    packet_ids = packet["report_id"]
    if packet_ids.eq("").any() or packet_ids.duplicated().any():
        raise ValueError("The packet contains missing or duplicate report IDs.")

    template = pd.read_csv(template_path, dtype=str, keep_default_na=False)
    if template.columns.tolist() != list(ANNOTATION_COLUMNS):
        raise ValueError(
            "The annotation template does not match the v0.3.1 schema. "
            "Regenerate annotator templates before continuing."
        )
    if template["report_id"].tolist() != packet_ids.tolist():
        raise ValueError("The annotation template IDs do not align with the packet.")
    if set(template["annotator_id"]) != {annotator_id}:
        raise ValueError("The annotation template has an unexpected annotator ID.")

    if annotations_path.exists():
        annotations = pd.read_csv(
            annotations_path, dtype=str, keep_default_na=False
        )
        if annotations.columns.tolist() != list(ANNOTATION_COLUMNS):
            raise ValueError(
                "The saved annotation file uses an older schema. Preserve it as an "
                "archive and regenerate a v0.3.1 annotation file; do not silently migrate scores."
            )
        if annotations["report_id"].duplicated().any():
            raise ValueError("The saved annotation file contains duplicate report IDs.")
        if set(annotations["report_id"]) != set(packet_ids):
            raise ValueError("The saved annotations do not align with the packet.")
        if set(annotations["annotator_id"]) != {annotator_id}:
            raise ValueError("The saved file has an unexpected annotator ID.")
        schema_errors = validate_annotation_frame(annotations, require_complete=False)
        if schema_errors:
            raise ValueError("The saved annotation file contains invalid values.")
        annotations = (
            annotations.set_index("report_id")
            .loc[packet_ids.tolist()]
            .reset_index()
        )
    else:
        annotations = template.copy()
    return packet, annotations


def save_annotations(annotations: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            annotations.to_csv(temporary, index=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def completion_summary(
    packet: pd.DataFrame, annotations: pd.DataFrame
) -> dict[str, object]:
    duplicate_ids = sorted(
        report_id
        for report_id, count in Counter(packet["report_id"]).items()
        if count > 1
    )
    duplicate_ids.extend(
        report_id
        for report_id, count in Counter(annotations["report_id"]).items()
        if count > 1 and report_id not in duplicate_ids
    )
    missing_by_field = {
        field: int(annotations[field].map(_is_blank).sum())
        for field in SCORE_FIELDS
    }
    scored_mask = annotations[list(SCORE_FIELDS)].apply(
        lambda column: ~column.map(_is_blank)
    ).all(axis=1)
    valid_mask = annotations.apply(
        lambda row: not validate_annotation_row(row.to_dict()), axis=1
    )
    completed_mask = scored_mask & valid_mask
    schema_errors = validate_annotation_frame(annotations, require_complete=False)
    return {
        "total_completed": int(completed_mask.sum()),
        "total_remaining": int((~completed_mask).sum()),
        "missing_fields": {
            field: count for field, count in missing_by_field.items() if count
        },
        "duplicate_report_ids": duplicate_ids,
        "schema_validation": "passed" if not schema_errors else "failed",
        "schema_errors": schema_errors,
    }


def _is_blank(value: object) -> bool:
    return value is None or not str(value).strip()


def _first_incomplete(annotations: pd.DataFrame) -> int | None:
    scored = annotations[list(SCORE_FIELDS)].apply(
        lambda column: ~column.map(_is_blank)
    ).all(axis=1)
    valid = annotations.apply(
        lambda row: not validate_annotation_row(row.to_dict()), axis=1
    )
    complete = scored & valid
    incomplete = complete[~complete]
    return int(incomplete.index[0]) if not incomplete.empty else None


def _print_rubric_summary() -> None:
    print("Rubric summary: annotate only evidence in the report; do not infer unreported experience.")
    print("A 0 usually means 'not evidenced in the narrative', not proof the experience was absent.")
    print("Lucid insight, agency, reasoning, and control are distinct.")
    for field in PROMPT_FIELDS:
        if field != "notes":
            print(f"- {field}: {FIELD_GUIDANCE[field]}")
    print("- notes: optional short paraphrase, no quotations or identifying details.")
    print(f"Full rubric: {RUBRIC_PATH.relative_to(ROOT)}")


def _print_progress(annotations: pd.DataFrame) -> None:
    summary = completion_summary(
        pd.DataFrame({"report_id": annotations["report_id"]}), annotations
    )
    print(
        f"Progress: {summary['total_completed']} completed; "
        f"{summary['total_remaining']} remaining."
    )


def _print_completion_summary(packet: pd.DataFrame, annotations: pd.DataFrame) -> None:
    summary = completion_summary(packet, annotations)
    print("Completion summary")
    print(f"Total completed: {summary['total_completed']}")
    print(f"Total remaining: {summary['total_remaining']}")
    print(f"Missing fields: {summary['missing_fields']}")
    print(f"Duplicate report IDs: {summary['duplicate_report_ids']}")
    print(f"Schema validation: {summary['schema_validation']}")
    if summary["schema_errors"]:
        print(f"Schema errors: {summary['schema_errors']}")


def _prompt_value(
    field: str,
    current: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> tuple[str | None, str | None]:
    if field in ORDINAL_FIELDS:
        allowed = "/".join(str(value) for value in ORDINAL_FIELDS[field])
    elif field in BINARY_FIELDS:
        allowed = "0/1"
    elif field in NOMINAL_FIELDS:
        allowed = "/".join(NOMINAL_FIELDS[field])
    elif field == "confidence":
        allowed = "/".join(CONFIDENCE_VALUES)
    else:
        allowed = "optional short paraphrase"

    while True:
        prompt = f"{field} ({allowed})"
        if current:
            prompt += f" [current: {current}]"
        value = input_fn(
            f"{prompt}; commands: next, previous, save, quit, rubric, progress > "
        )
        command = COMMANDS.get(value.strip().lower())
        if command:
            return None, command
        normalized, error = parse_field_value(field, value)
        if error:
            output_fn(error)
            continue
        return normalized, None


def run_session(
    packet: pd.DataFrame,
    annotations: pd.DataFrame,
    output_path: Path,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    if not packet.empty and _first_incomplete(annotations) is None:
        _print_completion_summary(packet, annotations)
        return
    index = _first_incomplete(annotations) or 0
    while index < len(packet):
        report_id = packet.iloc[index]["report_id"]
        report_text = packet.iloc[index]["dream_text"]
        output_fn(f"\nReport {index + 1} of {len(packet)}")
        output_fn(f"Report ID: {report_id}")
        output_fn(str(report_text))
        _print_rubric_summary()

        navigation = None
        for field in PROMPT_FIELDS:
            while True:
                current = str(annotations.at[index, field])
                value, command = _prompt_value(field, current, input_fn, output_fn)

                if command == "rubric":
                    _print_rubric_summary()
                    continue
                if command == "progress":
                    _print_progress(annotations)
                    continue
                if command == "save":
                    save_annotations(annotations, output_path)
                    output_fn("Progress saved.")
                    continue
                if command == "quit":
                    save_annotations(annotations, output_path)
                    output_fn("Progress saved. Quitting.")
                    _print_completion_summary(packet, annotations)
                    return
                if command in {"next", "previous"}:
                    save_annotations(annotations, output_path)
                    navigation = command
                    break

                assert value is not None
                annotations.at[index, field] = value
                row_errors = validate_annotation_row(
                    annotations.iloc[index].to_dict()
                )
                if row_errors:
                    annotations.at[index, field] = ""
                    output_fn(
                        "The value failed schema validation; please re-enter it."
                    )
                    output_fn("; ".join(row_errors))
                    continue
                save_annotations(annotations, output_path)
                break

            if navigation is not None:
                break

        if navigation == "previous":
            index = max(index - 1, 0)
            continue
        if navigation == "next":
            index += 1
            continue

        save_annotations(annotations, output_path)
        next_incomplete = _first_incomplete(
            annotations.iloc[index + 1 :].reset_index(drop=True)
        )
        if next_incomplete is None:
            index = len(packet)
        else:
            index = index + 1 + next_incomplete

    _print_completion_summary(packet, annotations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-only local v0.3.1 pilot annotation; no scores are inferred."
    )
    parser.add_argument(
        "--annotator", required=True, choices=("annotator_1", "annotator_2")
    )
    parser.add_argument("--check", action="store_true", help="Show progress only.")
    args = parser.parse_args()
    annotator_id = args.annotator
    packet_path = OUTPUT_DIR / f"{annotator_id}_pilot_packet.csv"
    template_path = OUTPUT_DIR / f"{annotator_id}_annotation_template.csv"
    annotations_path = OUTPUT_DIR / f"{annotator_id}_annotations.csv"
    try:
        packet, annotations = load_session(
            packet_path, template_path, annotations_path, annotator_id
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Cannot start annotation: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if args.check:
        _print_completion_summary(packet, annotations)
        return
    print("Human annotation only. No scores are suggested or generated.")
    print("v0.3.1 schema: regenerate local templates before starting if you used the older rubric.")
    run_session(packet, annotations, annotations_path)


if __name__ == "__main__":
    main()
