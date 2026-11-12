from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ORDINAL_FIELDS = {
    "lucid_insight": (0, 1, 2),
    "intentional_agency": (0, 1, 2),
    "state_reasoning": (0, 1, 2),
    "waking_memory": (0, 1, 2),
    "emotional_intensity": (0, 1, 2),
    "dream_instability": (0, 1, 2),
}
BINARY_FIELDS = (
    "internal_commentary",
    "self_reflection",
    "prospective_intention_memory",
    "anomaly_present",
    "anomaly_noticed",
    "reality_testing",
    "dream_rationalization",
    "control_attempt",
    "sensory_visual",
    "sensory_auditory",
    "sensory_tactile",
    "sensory_proprioceptive",
    "sensory_smell_taste",
    "loss_of_lucidity",
    "awakening",
)
VALENCE_VALUES = ("negative", "neutral_mixed", "positive")
CONTROL_SUCCESS_VALUES = ("not_applicable", "failed", "partial_uncertain", "successful")
CONTROL_DOMAIN_VALUES = (
    "none",
    "cognition",
    "emotion",
    "body",
    "movement",
    "characters",
    "environment",
    "physics",
    "multiple",
)
ROUTE_TO_LUCIDITY_VALUES = (
    "none",
    "reasoning_based",
    "spontaneous_intuitive",
    "wake_initiated",
    "other_unclear",
)
TRANSITION_OUTCOME_VALUES = ("no_transition", "near_lucid", "lucid", "unclear")
CONFIDENCE_VALUES = ("low", "medium", "high")
NOMINAL_FIELDS = {
    "emotional_valence": VALENCE_VALUES,
    "control_success": CONTROL_SUCCESS_VALUES,
    "control_domain": CONTROL_DOMAIN_VALUES,
    "route_to_lucidity": ROUTE_TO_LUCIDITY_VALUES,
    "transition_outcome": TRANSITION_OUTCOME_VALUES,
}
ANNOTATION_FIELDS = (
    "report_id",
    "annotator_id",
    *ORDINAL_FIELDS,
    *BINARY_FIELDS,
    *NOMINAL_FIELDS,
    "confidence",
    "notes",
)
SCORE_FIELDS = (*ORDINAL_FIELDS, *BINARY_FIELDS, *NOMINAL_FIELDS, "confidence")
ANNOTATION_COLUMNS = (*ANNOTATION_FIELDS,)


def validate_annotation_row(row: Mapping[str, object]) -> list[str]:
    """Return field-specific errors; blank scores are allowed for incomplete drafts."""
    errors = []
    for required in ("report_id", "annotator_id"):
        value = row.get(required)
        if value is None or pd.isna(value) or not str(value).strip():
            errors.append(f"{required} is required.")

    for field, allowed in ORDINAL_FIELDS.items():
        value = row.get(field)
        if _is_blank(value):
            continue
        parsed = _parse_integer(value)
        if parsed not in allowed:
            errors.append(f"{field} must be one of {allowed} or blank.")

    for field in BINARY_FIELDS:
        value = row.get(field)
        if _is_blank(value):
            continue
        parsed = _parse_integer(value)
        if parsed not in (0, 1):
            errors.append(f"{field} must be 0, 1, or blank.")

    for field, allowed in NOMINAL_FIELDS.items():
        value = row.get(field)
        if _is_blank(value):
            continue
        normalized = str(value).strip().lower()
        if normalized not in allowed:
            errors.append(f"{field} must be one of {allowed} or blank.")

    value = row.get("confidence")
    if not _is_blank(value):
        normalized = str(value).strip().lower()
        if normalized not in CONFIDENCE_VALUES:
            errors.append(
                f"confidence must be one of {CONFIDENCE_VALUES} or blank."
            )

    control_attempt = _parse_integer(row.get("control_attempt"))
    control_success = _normalized(row.get("control_success"))
    control_domain = _normalized(row.get("control_domain"))
    if control_attempt == 0:
        if control_success and control_success != "not_applicable":
            errors.append(
                "control_success must be not_applicable when control_attempt is 0."
            )
        if control_domain and control_domain != "none":
            errors.append("control_domain must be none when control_attempt is 0.")
    elif control_attempt == 1:
        if control_success == "not_applicable":
            errors.append(
                "control_success cannot be not_applicable when control_attempt is 1."
            )
        if control_domain == "none":
            errors.append("control_domain cannot be none when control_attempt is 1.")

    return errors


def validate_annotation_frame(
    frame: pd.DataFrame, *, require_complete: bool = False
) -> list[str]:
    errors = []
    missing_columns = sorted(set(ANNOTATION_COLUMNS) - set(frame.columns))
    if missing_columns:
        return [f"Missing required columns: {', '.join(missing_columns)}."]

    seen = set()
    for index, row in frame.iterrows():
        row_errors = validate_annotation_row(row.to_dict())
        errors.extend(f"row {index + 2}: {error}" for error in row_errors)
        report_id = row.get("report_id")
        annotator_id = row.get("annotator_id")
        if not _is_blank(report_id) and not _is_blank(annotator_id):
            pair = (str(report_id).strip(), str(annotator_id).strip())
            if pair in seen:
                errors.append(
                    f"row {index + 2}: duplicate report_id/annotator_id pair {pair[0]!r}."
                )
            seen.add(pair)
        if require_complete:
            unscored = [field for field in SCORE_FIELDS if _is_blank(row.get(field))]
            if unscored:
                errors.append(
                    f"row {index + 2}: missing scores for {', '.join(unscored)}."
                )
    return errors


def derive_sensory_count(row: Mapping[str, object]) -> int | None:
    """Return the number of present sensory modalities, or None when any is blank."""
    sensory_fields = (
        "sensory_visual",
        "sensory_auditory",
        "sensory_tactile",
        "sensory_proprioceptive",
        "sensory_smell_taste",
    )
    values = [row.get(field) for field in sensory_fields]
    if any(_is_blank(value) for value in values):
        return None
    parsed = [_parse_integer(value) for value in values]
    if any(value not in (0, 1) for value in parsed):
        raise ValueError("Sensory indicators must be binary before deriving a count.")
    return int(sum(parsed))


def paired_agreement(
    frame: pd.DataFrame, *, annotator_ids: Sequence[str] | None = None
) -> dict:
    """Calculate paired inter-rater statistics without imputing blank annotations."""
    if annotator_ids is None:
        annotator_ids = sorted(
            str(value)
            for value in frame["annotator_id"].dropna().unique()
            if str(value).strip()
        )
    annotator_ids = list(annotator_ids)
    if len(annotator_ids) != 2:
        raise ValueError("Agreement analysis requires exactly two annotator IDs.")

    first_id, second_id = annotator_ids
    first = frame.loc[frame["annotator_id"].astype(str) == first_id].set_index(
        "report_id"
    )
    second = frame.loc[frame["annotator_id"].astype(str) == second_id].set_index(
        "report_id"
    )
    shared_ids = first.index.intersection(second.index)
    output = {
        "annotator_ids": [first_id, second_id],
        "n_reports_with_both_annotators": len(shared_ids),
        "fields": {},
    }

    for field in SCORE_FIELDS:
        if not len(shared_ids):
            output["fields"][field] = _empty_agreement()
            continue
        paired = pd.DataFrame(
            {
                "left": first.loc[shared_ids, field],
                "right": second.loc[shared_ids, field],
            }
        ).dropna()
        paired = paired.loc[
            paired["left"].astype(str).str.strip().ne("")
            & paired["right"].astype(str).str.strip().ne("")
        ]
        if paired.empty:
            output["fields"][field] = _empty_agreement()
            continue

        values_left, values_right = _coerce_paired(field, paired)
        if field in ORDINAL_FIELDS or field == "confidence":
            allowed = (
                ORDINAL_FIELDS[field]
                if field in ORDINAL_FIELDS
                else tuple(range(len(CONFIDENCE_VALUES)))
            )
            weights = "quadratic"
            alpha_distance = "ordinal_squared"
        elif field in BINARY_FIELDS:
            allowed = (0, 1)
            weights = None
            alpha_distance = "nominal"
        else:
            allowed = NOMINAL_FIELDS[field]
            weights = None
            alpha_distance = "nominal"

        kappa = (
            cohen_kappa_score(
                values_left,
                values_right,
                labels=list(allowed),
                weights=weights,
            )
            if len(set(values_left) | set(values_right)) > 1
            else None
        )
        output["fields"][field] = {
            "n_paired": len(paired),
            "percent_agreement": float(np.mean(values_left == values_right)),
            "cohens_kappa": _finite_or_none(kappa),
            "cohens_kappa_weighting": weights or "unweighted",
            "krippendorff_alpha": _krippendorff_alpha(
                values_left,
                values_right,
                categories=list(allowed),
                distance=alpha_distance,
            ),
            "krippendorff_alpha_distance": alpha_distance,
        }
    return output


def _empty_agreement() -> dict:
    return {
        "n_paired": 0,
        "percent_agreement": None,
        "cohens_kappa": None,
        "krippendorff_alpha": None,
    }


def _krippendorff_alpha(
    first: Sequence[object],
    second: Sequence[object],
    *,
    categories: Sequence[object],
    distance: str,
) -> float | None:
    observed_pairs = [
        (left, right)
        for left, right in zip(first, second)
        if not _is_blank(left) and not _is_blank(right)
    ]
    pooled = [value for pair in observed_pairs for value in pair]
    if len(pooled) < 2:
        return None
    category_index = {category: index for index, category in enumerate(categories)}
    matrix = _distance_matrix(categories, distance)

    observed = float(
        np.mean(
            [
                matrix[category_index[left], category_index[right]]
                for left, right in observed_pairs
            ]
        )
    )
    counts = np.zeros(len(categories), dtype=float)
    for value in pooled:
        counts[category_index[value]] += 1
    expected = 0.0
    total = len(pooled)
    for left_index in range(len(categories)):
        for right_index in range(len(categories)):
            pair_count = counts[left_index] * (
                counts[right_index] - (1 if left_index == right_index else 0)
            )
            expected += pair_count * matrix[left_index, right_index]
    expected /= total * (total - 1)
    if expected == 0:
        return None
    return float(1 - observed / expected)


def _distance_matrix(categories: Sequence[object], distance: str) -> np.ndarray:
    count = len(categories)
    matrix = np.zeros((count, count), dtype=float)
    for left in range(count):
        for right in range(count):
            if distance == "nominal":
                matrix[left, right] = float(left != right)
            else:
                matrix[left, right] = ((left - right) / max(count - 1, 1)) ** 2
    return matrix


def _coerce_paired(field: str, paired: pd.DataFrame) -> tuple[list, list]:
    if field in ORDINAL_FIELDS or field in BINARY_FIELDS:
        return (
            [int(value) for value in paired["left"]],
            [int(value) for value in paired["right"]],
        )
    if field == "confidence":
        order = {value: index for index, value in enumerate(CONFIDENCE_VALUES)}
        return (
            [order[str(value).strip().lower()] for value in paired["left"]],
            [order[str(value).strip().lower()] for value in paired["right"]],
        )
    return (
        [str(value).strip().lower() for value in paired["left"]],
        [str(value).strip().lower() for value in paired["right"]],
    )


def _parse_integer(value: object) -> int | None:
    if _is_blank(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not parsed.is_integer():
        return None
    return int(parsed)


def _normalized(value: object) -> str:
    return "" if _is_blank(value) else str(value).strip().lower()


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


def _finite_or_none(value) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)
