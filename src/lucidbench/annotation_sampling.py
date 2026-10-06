from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .annotation_schema import ANNOTATION_COLUMNS
from .data import infer_columns, load_reports, normalize_binary_labels
from .robustness import find_cross_author_duplicates


def sample_annotation_reports(
    frame: pd.DataFrame,
    config: dict,
    *,
    excluded_duplicate_indices: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sample balanced, length-stratified reports from class-exclusive authors."""
    columns = infer_columns(frame, config)
    if columns.author is None:
        raise ValueError("Annotation sampling requires an author identity column.")
    if frame[columns.author].isna().any():
        raise ValueError("Annotation sampling requires complete author IDs.")

    labels = normalize_binary_labels(
        frame[columns.label], config["positive_labels"], config["negative_labels"]
    )
    eligible = labels.notna() & frame[columns.text].notna()
    if not eligible.any():
        raise ValueError("No labeled reports with text are available.")

    excluded = set(excluded_duplicate_indices or ())
    invalid_excluded = [index for index in excluded if index < 0 or index >= len(frame)]
    if invalid_excluded:
        raise ValueError("Duplicate-exclusion indices fall outside the corpus.")
    eligible.iloc[list(excluded)] = False

    row_indices = np.flatnonzero(eligible.to_numpy())
    author_values = frame.loc[eligible, columns.author].astype(str).to_numpy()
    label_values = labels.loc[eligible].astype(int).to_numpy()
    text_values = frame.loc[eligible, columns.text].astype(str).to_numpy()
    word_counts = np.fromiter(
        (len(text.split()) for text in text_values), dtype=int, count=len(text_values)
    )

    author_labels: dict[str, set[int]] = {}
    for author, label in zip(author_values, label_values):
        author_labels.setdefault(author, set()).add(int(label))
    class_exclusive = np.array(
        [len(author_labels[author]) == 1 for author in author_values], dtype=bool
    )
    eligible_positions = np.flatnonzero(class_exclusive)
    if len(eligible_positions) == 0:
        raise ValueError(
            "No class-exclusive authors remain for author-disjoint sampling."
        )

    quantiles = np.linspace(0, 1, int(config["length_strata"]) + 1)
    bin_edges = np.unique(np.quantile(word_counts[eligible_positions], quantiles))
    bin_ids = np.digitize(word_counts, bin_edges[1:-1], right=True)
    sample_by_class = {}
    for label in (0, 1):
        label_name = "nonlucid" if label == 0 else "lucid"
        candidates = eligible_positions[label_values[eligible_positions] == label]
        sample_by_class[label] = _select_stratified_with_cap(
            candidates,
            author_values,
            bin_ids,
            n_reports=int(config["reports_per_class"]),
            n_bins=len(bin_edges) - 1,
            author_cap=int(config["max_reports_per_author"]),
            seed=int(config["seed"]) + label,
            label_name=label_name,
        )

    if not set(author_values[sample_by_class[0]]).isdisjoint(
        set(author_values[sample_by_class[1]])
    ):
        raise RuntimeError("Author overlap across lucid and nonlucid sample groups.")

    main_positions = np.concatenate((sample_by_class[0], sample_by_class[1]))
    pilot_by_class = {}
    for label in (0, 1):
        pilot_by_class[label] = _select_stratified_with_cap(
            sample_by_class[label],
            author_values,
            bin_ids,
            n_reports=int(config["pilot_reports_per_class"]),
            n_bins=len(bin_edges) - 1,
            author_cap=int(config["max_reports_per_author"]),
            seed=int(config["pilot_seed"]) + label,
            label_name=f"pilot {'nonlucid' if label == 0 else 'lucid'}",
        )
    pilot_positions = np.concatenate((pilot_by_class[0], pilot_by_class[1]))
    if not set(pilot_positions).issubset(set(main_positions)):
        raise RuntimeError("Pilot subset is not contained in the main sample.")

    rng = np.random.default_rng(int(config["seed"]))
    main_positions = rng.permutation(main_positions)
    report_ids = [f"LB03-{index:04d}" for index in range(1, len(main_positions) + 1)]
    annotation_ids = {
        int(position): report_id
        for position, report_id in zip(main_positions, report_ids)
    }
    pilot_ids = {annotation_ids[int(position)] for position in pilot_positions}

    manifest_rows = []
    for position in main_positions:
        source_row = int(row_indices[position])
        text = str(text_values[position])
        target = int(label_values[position])
        manifest_rows.append(
            {
                "report_id": annotation_ids[int(position)],
                "source_row_index": source_row,
                "source_post_id": str(frame.iloc[source_row].get("post_id", "")),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "sampling_label": "lucid" if target == 1 else "nonlucid",
                "word_count": int(word_counts[position]),
                "length_stratum": int(bin_ids[position]),
                "pilot_double_annotate": annotation_ids[int(position)] in pilot_ids,
            }
        )
    manifest = pd.DataFrame(manifest_rows)

    pilot_template = pd.concat(
        [
            _annotation_template(
                manifest.loc[manifest["pilot_double_annotate"], "report_id"].tolist(),
                [annotator_id],
            )
            for annotator_id in ("annotator_1", "annotator_2")
        ],
        ignore_index=True,
    )
    full_template = _annotation_template(manifest["report_id"].tolist(), [""])

    packet = manifest[["report_id", "pilot_double_annotate"]].copy()
    packet["dream_text"] = [str(text_values[position]) for position in main_positions]
    pilot_packet = packet.loc[packet["pilot_double_annotate"]].copy()
    pilot_packets = {}
    for index, annotator_id in enumerate(("annotator_1", "annotator_2")):
        annotator_rng = np.random.default_rng(int(config["pilot_seed"]) + index)
        order = annotator_rng.permutation(len(pilot_packet))
        pilot_packets[annotator_id] = pilot_packet.iloc[order][
            ["report_id", "dream_text"]
        ].reset_index(drop=True)

    sample_summary = _sample_summary(
        manifest,
        pilot_template,
        frame,
        columns.author,
        excluded_duplicate_count=len(excluded),
        class_exclusive_author_count=len(
            {author for author, values in author_labels.items() if len(values) == 1}
        ),
        mixed_label_author_count=sum(
            len(values) > 1 for values in author_labels.values()
        ),
        requested_reports_per_class=int(config["reports_per_class"]),
        requested_pilot_per_class=int(config["pilot_reports_per_class"]),
        author_cap=int(config["max_reports_per_author"]),
        seed=int(config["seed"]),
        pilot_seed=int(config["pilot_seed"]),
        author_disjoint=True,
    )
    return (
        manifest,
        full_template,
        {
            "summary": sample_summary,
            "pilot_template": pilot_template,
            "packet": packet,
            "pilot_packets": pilot_packets,
            "bin_edges": [float(value) for value in bin_edges],
        },
    )


def _select_stratified_with_cap(
    candidates,
    authors,
    bins,
    *,
    n_reports: int,
    n_bins: int,
    author_cap: int,
    seed: int,
    label_name: str,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=int)
    authors = np.asarray(authors).astype(str)
    bins = np.asarray(bins, dtype=int)
    if n_reports <= 0 or author_cap <= 0:
        raise ValueError("Sample size and author cap must be positive.")
    rng = np.random.default_rng(seed)
    quotas = np.full(n_bins, n_reports // n_bins, dtype=int)
    quotas[: n_reports % n_bins] += 1
    selected = []
    author_counts: dict[str, int] = {}
    selected_by_bin = np.zeros(n_bins, dtype=int)
    order = rng.permutation(n_bins)

    for length_bin in order:
        bin_candidates = rng.permutation(candidates[bins[candidates] == length_bin])
        for candidate in bin_candidates:
            if selected_by_bin[length_bin] >= quotas[length_bin]:
                break
            author = authors[candidate]
            if author_counts.get(author, 0) >= author_cap:
                continue
            selected.append(int(candidate))
            selected_by_bin[length_bin] += 1
            author_counts[author] = author_counts.get(author, 0) + 1

    missing = n_reports - len(selected)
    if missing:
        remaining = np.setdiff1d(candidates, np.asarray(selected, dtype=int))
        for candidate in rng.permutation(remaining):
            author = authors[candidate]
            if author_counts.get(author, 0) >= author_cap:
                continue
            selected.append(int(candidate))
            author_counts[author] = author_counts.get(author, 0) + 1
            missing -= 1
            if missing == 0:
                break
    if missing:
        raise ValueError(
            f"Cannot select {n_reports} {label_name} reports with author cap "
            f"{author_cap}; short by {missing} reports."
        )
    return np.asarray(selected, dtype=int)


def _annotation_template(
    report_ids: list[str], annotator_ids: list[str]
) -> pd.DataFrame:
    annotator = annotator_ids[0] if annotator_ids else ""
    frame = pd.DataFrame(
        {
            "report_id": report_ids,
            "annotator_id": [annotator] * len(report_ids),
        }
    )
    for column in ANNOTATION_COLUMNS[2:]:
        frame[column] = ""
    return frame.loc[:, ANNOTATION_COLUMNS]


def _sample_summary(
    manifest,
    pilot_template,
    frame,
    author_column,
    *,
    excluded_duplicate_count,
    class_exclusive_author_count,
    mixed_label_author_count,
    requested_reports_per_class,
    requested_pilot_per_class,
    author_cap,
    seed,
    pilot_seed,
    author_disjoint,
) -> dict:
    sample_authors = frame.iloc[manifest["source_row_index"]][author_column].astype(str)
    author_report_counts = sample_authors.value_counts()
    pilot = manifest.loc[manifest["pilot_double_annotate"]]
    pilot_counts = pilot["sampling_label"].value_counts().to_dict()
    per_class = {}
    for label in ("lucid", "nonlucid"):
        subset = manifest.loc[manifest["sampling_label"] == label]
        per_class[label] = {
            "reports": len(subset),
            "authors": int(
                frame.iloc[subset["source_row_index"]][author_column].nunique()
            ),
            "length_stratum_counts": {
                str(key): int(value)
                for key, value in subset["length_stratum"]
                .value_counts()
                .sort_index()
                .items()
            },
            "pilot_reports": int(pilot_counts.get(label, 0)),
        }
    return {
        "sample_size": len(manifest),
        "requested_reports_per_class": requested_reports_per_class,
        "reports_per_class": per_class,
        "n_unique_authors": int(sample_authors.nunique()),
        "author_cap": author_cap,
        "maximum_sampled_reports_per_author": int(author_report_counts.max()),
        "authors_by_sampled_report_count": {
            str(key): int(value)
            for key, value in author_report_counts.value_counts().sort_index().items()
        },
        "author_disjoint_between_classes": bool(author_disjoint),
        "class_exclusive_authors_available": class_exclusive_author_count,
        "mixed_label_authors_excluded_from_sampling": mixed_label_author_count,
        "reports_excluded_for_detected_cross_author_duplicates": excluded_duplicate_count,
        "pilot_size": len(pilot),
        "pilot_requested_per_class": requested_pilot_per_class,
        "pilot_annotator_template_rows": len(pilot_template),
        "seed": seed,
        "pilot_seed": pilot_seed,
        "raw_dream_text_in_manifest": False,
        "raw_dream_text_in_annotation_template": False,
    }


def create_annotation_sample(
    config_path: str | Path, *, output_dir: str | Path | None = None
) -> dict:
    with Path(config_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_path = Path(config["data_path"])
    frame = load_reports(data_path)
    columns = infer_columns(frame, config)
    if columns.author is None:
        raise ValueError("Annotation sampling requires author identity.")
    if frame[columns.author].isna().any():
        raise ValueError("Annotation sampling requires complete author IDs.")

    pairs = find_cross_author_duplicates(
        frame[columns.text].astype(str).to_numpy(),
        frame[columns.author].astype(str).to_numpy(),
        near_duplicate_jaccard=float(config["near_duplicate_jaccard"]),
        shingle_size=int(config["duplicate_shingle_size"]),
    )
    excluded = {index for pair in pairs for index in (pair["left"], pair["right"])}
    manifest, full_template, sample = sample_annotation_reports(
        frame, config, excluded_duplicate_indices=excluded
    )

    output = Path(output_dir or config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "sample_manifest.csv", index=False)
    full_template.to_csv(output / "annotation_template.csv", index=False)
    sample["pilot_template"].to_csv(
        output / "pilot_annotation_template.csv", index=False
    )
    sample["packet"].to_csv(output / "local_report_packet.csv", index=False)
    for annotator_id, packet in sample["pilot_packets"].items():
        packet.to_csv(output / f"{annotator_id}_pilot_packet.csv", index=False)
    (output / "sample_summary.json").write_text(
        json.dumps(sample["summary"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sample["summary"]
