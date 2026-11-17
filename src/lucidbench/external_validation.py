from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


REQUIRED_CONFIG_KEYS = (
    "reports_path",
    "output_dir",
    "report_id_column",
    "participant_id_column",
    "condition_column",
    "text_column",
)


def prepare_external_reports(
    reports: pd.DataFrame,
    *,
    report_id_column: str,
    participant_id_column: str,
    condition_column: str,
    text_column: str,
    allowed_conditions: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create a ratings-free model packet and a private linkage manifest."""
    required = {
        report_id_column,
        participant_id_column,
        condition_column,
        text_column,
    }
    missing = sorted(required - set(reports.columns))
    if missing:
        raise ValueError(f"Missing required report columns: {', '.join(missing)}.")

    frame = reports.loc[
        :, [report_id_column, participant_id_column, condition_column, text_column]
    ].copy()
    frame.columns = ["report_id", "participant_id", "condition", "dream_text"]

    for column in ("report_id", "participant_id", "condition", "dream_text"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be complete and non-empty.")

    if frame["report_id"].duplicated().any():
        raise ValueError("report_id must be unique.")

    if allowed_conditions:
        allowed = {str(value).strip() for value in allowed_conditions}
        observed = set(frame["condition"].astype(str).str.strip())
        unknown = sorted(observed - allowed)
        if unknown:
            raise ValueError(
                "Unexpected condition values: " + ", ".join(unknown) + "."
            )

    model_packet = frame.loc[:, ["report_id", "dream_text"]].copy()
    linkage_manifest = frame.loc[
        :, ["report_id", "participant_id", "condition"]
    ].copy()

    reports_per_participant = (
        linkage_manifest.groupby("participant_id")["report_id"].count()
    )
    summary = {
        "n_reports": int(len(frame)),
        "n_participants": int(linkage_manifest["participant_id"].nunique()),
        "conditions": {
            str(key): int(value)
            for key, value in linkage_manifest["condition"].value_counts().items()
        },
        "reports_per_participant_min": int(reports_per_participant.min()),
        "reports_per_participant_median": float(reports_per_participant.median()),
        "reports_per_participant_max": int(reports_per_participant.max()),
        "human_ratings_in_model_packet": False,
        "participant_ids_in_model_packet": False,
        "condition_labels_in_model_packet": False,
    }
    return model_packet, linkage_manifest, summary


def prepare_external_validation(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict:
    """Prepare external reports without loading or inspecting human ratings."""
    with Path(config_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(
            "Missing required config keys: " + ", ".join(missing_keys) + "."
        )

    reports = pd.read_csv(config["reports_path"])
    model_packet, linkage_manifest, summary = prepare_external_reports(
        reports,
        report_id_column=config["report_id_column"],
        participant_id_column=config["participant_id_column"],
        condition_column=config["condition_column"],
        text_column=config["text_column"],
        allowed_conditions=config.get("allowed_conditions"),
    )

    target = Path(output_dir or config["output_dir"])
    target.mkdir(parents=True, exist_ok=True)
    model_packet.to_csv(target / "model_packet.csv", index=False)
    linkage_manifest.to_csv(target / "linkage_manifest.csv", index=False)

    return summary
