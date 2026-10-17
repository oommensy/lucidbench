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
    """Sample paired mixed-label authors and independent class-exclusive reports."""
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
    author_positions: dict[str, dict[int, list[int]]] = {}
    for position, (author, label) in enumerate(zip(author_values, label_values)):
        author_positions.setdefault(author, {}).setdefault(int(label), []).append(
            position
        )

    mixed_pairs = _best_pairs_by_author(
        author_positions, word_counts, int(config["seed"])
    )
    if not mixed_pairs:
        raise ValueError("No mixed-label authors remain after duplicate exclusions.")

    quantiles = np.linspace(0, 1, int(config["length_strata"]) + 1)
    bin_edges = np.unique(np.quantile(word_counts, quantiles))
    bin_ids = np.digitize(word_counts, bin_edges[1:-1], right=True)
    n_bins = len(bin_edges) - 1
    mixed_pair_means = np.asarray([pair["mean_length"] for pair in mixed_pairs])
    mixed_pair_bins = _bins_for_values(mixed_pair_means, bin_edges)
    reports_per_class = int(config["reports_per_class"])
    pilot_reports_per_class = int(config["pilot_reports_per_class"])
    author_cap = int(config["max_reports_per_author"])
    paired_author_target = int(config["paired_authors"])
    pilot_paired_author_target = int(config["pilot_paired_authors"])

    if author_cap < 2 and paired_author_target:
        raise ValueError("Paired mixed-label authors require an author cap of at least 2.")
    if paired_author_target > reports_per_class:
        raise ValueError("Paired author count cannot exceed reports per class.")
    if pilot_paired_author_target > paired_author_target:
        raise ValueError("Pilot paired-author count cannot exceed the full paired count.")
    if pilot_paired_author_target > pilot_reports_per_class:
        raise ValueError("Pilot paired-author count cannot exceed pilot reports per class.")
    if pilot_reports_per_class > reports_per_class:
        raise ValueError("Pilot reports per class cannot exceed full-sample reports.")

    selected_pairs = _select_stratified_pairs(
        mixed_pairs,
        mixed_pair_bins,
        n_authors=paired_author_target,
        n_bins=n_bins,
        seed=int(config["seed"]) + 100,
        label_name="paired mixed-label authors",
    )
    paired_positions = {
        label: np.asarray([pair[label] for pair in selected_pairs], dtype=int)
        for label in (0, 1)
    }

    all_positions = np.arange(len(row_indices), dtype=int)
    exclusive_positions = {
        label: np.asarray(
            [
                position
                for position in all_positions
                if set(author_positions[author_values[position]]) == {label}
            ],
            dtype=int,
        )
        for label in (0, 1)
    }
    independent_by_class: dict[int, np.ndarray] = {}
    for label in (0, 1):
        independent_count = reports_per_class - paired_author_target
        independent_by_class[label] = _select_stratified_with_cap(
            exclusive_positions[label],
            author_values,
            bin_ids,
            n_reports=independent_count,
            n_bins=n_bins,
            author_cap=author_cap,
            seed=int(config["seed"]) + label,
            label_name=f"independent {'nonlucid' if label == 0 else 'lucid'}",
        )

    main_by_class = {
        label: np.concatenate((paired_positions[label], independent_by_class[label]))
        for label in (0, 1)
    }
    pair_means = np.asarray(
        [
            (word_counts[pair[0]] + word_counts[pair[1]]) / 2
            for pair in selected_pairs
        ]
    )
    pilot_pairs = _select_stratified_pairs(
        selected_pairs,
        _bins_for_values(pair_means, bin_edges),
        n_authors=pilot_paired_author_target,
        n_bins=n_bins,
        seed=int(config["pilot_seed"]) + 100,
        label_name="pilot paired mixed-label authors",
    )
    pilot_pair_authors = {pair["author"] for pair in pilot_pairs}
    pilot_pair_positions = {
        label: np.asarray(
            [
                pair[label]
                for pair in selected_pairs
                if pair["author"] in pilot_pair_authors
            ],
            dtype=int,
        )
        for label in (0, 1)
    }

    pilot_independent_by_class: dict[int, np.ndarray] = {}
    for label in (0, 1):
        independent_pilot_count = pilot_reports_per_class - pilot_paired_author_target
        pilot_independent_by_class[label] = _select_stratified_with_cap(
            independent_by_class[label],
            author_values,
            bin_ids,
            n_reports=independent_pilot_count,
            n_bins=n_bins,
            author_cap=author_cap,
            seed=int(config["pilot_seed"]) + label,
            label_name=f"pilot independent {'nonlucid' if label == 0 else 'lucid'}",
        )

    pilot_by_class = {
        label: np.concatenate(
            (pilot_pair_positions[label], pilot_independent_by_class[label])
        )
        for label in (0, 1)
    }
    for label in (0, 1):
        if len(main_by_class[label]) != reports_per_class:
            raise RuntimeError("Main sample does not match the configured class count.")
        if len(pilot_by_class[label]) != pilot_reports_per_class:
            raise RuntimeError("Pilot does not match the configured class count.")
        if not set(pilot_by_class[label]).issubset(set(main_by_class[label])):
            raise RuntimeError("Pilot subset is not contained in the main sample.")

    main_positions = np.concatenate((main_by_class[0], main_by_class[1]))
    pilot_positions = set(np.concatenate((pilot_by_class[0], pilot_by_class[1])))
    rng = np.random.default_rng(int(config["seed"]))
    main_positions = rng.permutation(main_positions)
    report_ids = [f"LB03-{index:04d}" for index in range(1, len(main_positions) + 1)]
    annotation_ids = {
        int(position): report_id
        for position, report_id in zip(main_positions, report_ids)
    }
    pair_authors = {pair["author"] for pair in selected_pairs}
    author_keys = {
        author: f"AUTHOR-{index:05d}"
        for index, author in enumerate(sorted(set(author_values)), start=1)
    }

    manifest_rows = []
    for position in main_positions:
        source_row = int(row_indices[position])
        target = int(label_values[position])
        author = author_values[position]
        pair_id = author_keys[author] if author in pair_authors else ""
        is_paired = position in set(paired_positions[0]) or position in set(
            paired_positions[1]
        )
        manifest_rows.append(
            {
                "report_id": annotation_ids[int(position)],
                "source_row_index": source_row,
                "source_post_id": str(frame.iloc[source_row].get("post_id", "")),
                "text_sha256": hashlib.sha256(
                    str(text_values[position]).encode("utf-8")
                ).hexdigest(),
                "sampling_label": "lucid" if target == 1 else "nonlucid",
                "word_count": int(word_counts[position]),
                "length_stratum": int(bin_ids[position]),
                "pilot_double_annotate": int(position) in pilot_positions,
                "author_cluster_id": author_keys[author],
                "paired_author_id": pair_id,
                "sampling_component": (
                    "paired_mixed_label" if is_paired else "independent"
                ),
            }
        )
    manifest = pd.DataFrame(manifest_rows)

    pilot_ids = set(manifest.loc[manifest["pilot_double_annotate"], "report_id"])
    pilot_template = pd.concat(
        [
            _annotation_template(sorted(pilot_ids), [annotator_id])
            for annotator_id in ("annotator_1", "annotator_2")
        ],
        ignore_index=True,
    )
    full_template = _annotation_template(manifest["report_id"].tolist(), [""])
    packet = pd.DataFrame(
        {
            "report_id": manifest["report_id"].tolist(),
            "dream_text": [
                str(text_values[position]) for position in main_positions
            ],
        }
    )
    pilot_packet = packet.loc[packet["report_id"].isin(pilot_ids)].copy()
    pilot_packets = {}
    for index, annotator_id in enumerate(("annotator_1", "annotator_2")):
        annotator_rng = np.random.default_rng(int(config["pilot_seed"]) + index)
        order = annotator_rng.permutation(len(pilot_packet))
        pilot_packets[annotator_id] = pilot_packet.iloc[order].reset_index(drop=True)
    annotation_templates = {
        annotator_id: _annotation_template(
            packet["report_id"].tolist(), [annotator_id]
        )
        for annotator_id, packet in pilot_packets.items()
    }

    sample_summary = _sample_summary(
        manifest,
        frame,
        columns.author,
        selected_pairs=selected_pairs,
        mixed_label_author_count=len(mixed_pairs),
        requested_reports_per_class=reports_per_class,
        requested_pilot_per_class=pilot_reports_per_class,
        paired_author_target=paired_author_target,
        pilot_paired_author_target=pilot_paired_author_target,
        author_cap=author_cap,
        excluded_duplicate_count=len(excluded),
        seed=int(config["seed"]),
        pilot_seed=int(config["pilot_seed"]),
    )
    return (
        manifest,
        full_template,
        {
            "summary": sample_summary,
            "pilot_template": pilot_template,
            "annotation_templates": annotation_templates,
            "packet": packet,
            "pilot_packets": pilot_packets,
            "bin_edges": [float(value) for value in bin_edges],
        },
    )


def _best_pairs_by_author(author_positions, word_counts, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for author, labels_by_author in author_positions.items():
        if set(labels_by_author) != {0, 1}:
            continue
        candidates = [
            (left, right)
            for left in labels_by_author[0]
            for right in labels_by_author[1]
        ]
        distances = np.asarray(
            [abs(word_counts[left] - word_counts[right]) for left, right in candidates]
        )
        best = np.flatnonzero(distances == distances.min())
        left, right = candidates[int(rng.choice(best))]
        pairs.append(
            {
                "author": author,
                0: left,
                1: right,
                "mean_length": (word_counts[left] + word_counts[right]) / 2,
                "length_difference": abs(word_counts[left] - word_counts[right]),
            }
        )
    return pairs


def _select_stratified_pairs(
    pairs, pair_bins, *, n_authors, n_bins, seed, label_name
):
    candidates = np.arange(len(pairs), dtype=int)
    if n_authors < 0:
        raise ValueError("Requested sample size cannot be negative.")
    if n_authors > len(candidates):
        raise ValueError(
            f"Cannot select {n_authors} {label_name}; only {len(candidates)} are feasible."
        )
    if n_authors == 0:
        return []
    rng = np.random.default_rng(seed)
    quotas = np.full(n_bins, n_authors // n_bins, dtype=int)
    quotas[: n_authors % n_bins] += 1
    selected = []
    for length_bin in rng.permutation(n_bins):
        bin_candidates = candidates[pair_bins == length_bin]
        shuffled = rng.permutation(bin_candidates)
        ordered = shuffled[
            np.argsort(
                [pairs[index]["length_difference"] for index in shuffled],
                kind="stable",
            )
        ]
        take = min(quotas[length_bin], len(ordered))
        selected.extend(ordered[:take].tolist())
    missing = n_authors - len(selected)
    if missing:
        remaining = np.setdiff1d(candidates, np.asarray(selected, dtype=int))
        shuffled = rng.permutation(remaining)
        ordered = shuffled[
            np.argsort(
                [pairs[index]["length_difference"] for index in shuffled],
                kind="stable",
            )
        ]
        selected.extend(ordered[:missing].tolist())
        missing = n_authors - len(selected)
    if missing:
        raise ValueError(
            f"Cannot select {n_authors} {label_name}; short by {missing}."
        )
    chosen = np.asarray(selected, dtype=int)
    return [pairs[index] for index in chosen]


def _bins_for_values(values, edges):
    return np.digitize(values, edges[1:-1], right=True)


def _select_stratified_indices(
    candidates, bins, *, n_items, n_bins, seed, label_name
):
    candidates = np.asarray(candidates, dtype=int)
    if n_items == 0:
        return np.asarray([], dtype=int)
    if n_items < 0:
        raise ValueError("Requested sample size cannot be negative.")
    if len(candidates) < n_items:
        raise ValueError(
            f"Cannot select {n_items} {label_name}; only {len(candidates)} are feasible."
        )
    rng = np.random.default_rng(seed)
    quotas = np.full(n_bins, n_items // n_bins, dtype=int)
    quotas[: n_items % n_bins] += 1
    selected = []
    for length_bin in rng.permutation(n_bins):
        bin_candidates = rng.permutation(candidates[bins[candidates] == length_bin])
        take = min(quotas[length_bin], len(bin_candidates))
        selected.extend(bin_candidates[:take].tolist())
    missing = n_items - len(selected)
    if missing:
        remaining = np.setdiff1d(candidates, np.asarray(selected, dtype=int))
        for candidate in rng.permutation(remaining):
            selected.append(int(candidate))
            missing -= 1
            if missing == 0:
                break
    if missing:
        raise ValueError(
            f"Cannot select {n_items} {label_name}; short by {missing}."
        )
    return np.asarray(selected, dtype=int)


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
    if n_reports == 0:
        return np.asarray([], dtype=int)
    if n_reports < 0 or author_cap <= 0:
        raise ValueError("Sample size and author cap must be positive.")
    rng = np.random.default_rng(seed)
    quotas = np.full(n_bins, n_reports // n_bins, dtype=int)
    quotas[: n_reports % n_bins] += 1
    selected = []
    author_counts: dict[str, int] = {}
    selected_by_bin = np.zeros(n_bins, dtype=int)
    for length_bin in rng.permutation(n_bins):
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


def _annotation_template(report_ids: list[str], annotator_ids: list[str]) -> pd.DataFrame:
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
    frame,
    author_column,
    *,
    selected_pairs,
    mixed_label_author_count,
    requested_reports_per_class,
    requested_pilot_per_class,
    paired_author_target,
    pilot_paired_author_target,
    author_cap,
    excluded_duplicate_count,
    seed,
    pilot_seed,
) -> dict:
    sample = manifest.copy()
    source_authors = frame.iloc[sample["source_row_index"]][author_column].astype(str)
    sample["__author"] = source_authors.to_numpy()
    author_report_counts = source_authors.value_counts()
    pilot = sample.loc[sample["pilot_double_annotate"]]
    pilot_author_report_counts = pilot["author_cluster_id"].value_counts()
    pilot_paired = pilot.loc[
        pilot["sampling_component"] == "paired_mixed_label"
    ]
    pilot_pair_ids = set(pilot["paired_author_id"]) - {""}
    per_class = {}
    for label in ("lucid", "nonlucid"):
        subset = sample.loc[sample["sampling_label"] == label]
        per_class[label] = {
            "reports": len(subset),
            "authors": int(subset["author_cluster_id"].nunique()),
            "length_stratum_counts": {
                str(key): int(value)
                for key, value in subset["length_stratum"]
                .value_counts()
                .sort_index()
                .items()
            },
            "pilot_reports": int(
                (pilot["sampling_label"] == label).sum()
            ),
        }
    labels = sample["sampling_label"]
    lengths = sample["word_count"]
    lucid = lengths[labels == "lucid"].to_numpy()
    nonlucid = lengths[labels == "nonlucid"].to_numpy()
    paired = sample.loc[sample["sampling_component"] == "paired_mixed_label"]
    paired_lengths = _paired_length_table(paired)
    pilot_paired_lengths = _paired_length_table(pilot_paired)
    pair_differences = (
        (paired_lengths["lucid"] - paired_lengths["nonlucid"]).abs()
        if len(paired_lengths)
        else pd.Series(dtype=float)
    )
    pilot_pair_differences = (
        (
            pilot_paired_lengths["lucid"]
            - pilot_paired_lengths["nonlucid"]
        ).abs()
        if len(pilot_paired_lengths)
        else pd.Series(dtype=float)
    )
    overall_author_label_counts = sample.groupby("__author")["sampling_label"].nunique()
    independent_reports = int(
        (sample["sampling_component"] == "independent").sum()
    )
    pilot_independent_reports = int(
        (pilot["sampling_component"] == "independent").sum()
    )
    paired_counts = paired["sampling_label"].value_counts().to_dict()
    pilot_paired_counts = pilot_paired["sampling_label"].value_counts().to_dict()
    length_balance = _label_length_balance(lucid, nonlucid)
    length_balance.update(
        {
            "paired_absolute_word_count_difference_mean": (
                float(pair_differences.mean()) if len(pair_differences) else None
            ),
            "paired_absolute_word_count_difference_median": (
                float(pair_differences.median()) if len(pair_differences) else None
            ),
        }
    )
    pilot_lengths = pilot["word_count"]
    pilot_labels = pilot["sampling_label"]
    pilot_length_balance = _label_length_balance(
        pilot_lengths[pilot_labels == "lucid"].to_numpy(),
        pilot_lengths[pilot_labels == "nonlucid"].to_numpy(),
    )
    pilot_length_balance.update(
        {
            "paired_absolute_word_count_difference_mean": (
                float(pilot_pair_differences.mean())
                if len(pilot_pair_differences)
                else None
            ),
            "paired_absolute_word_count_difference_median": (
                float(pilot_pair_differences.median())
                if len(pilot_pair_differences)
                else None
            ),
        }
    )
    return {
        "sample_size": len(sample),
        "requested_reports_per_class": requested_reports_per_class,
        "reports_per_class": per_class,
        "n_unique_authors": int(source_authors.nunique()),
        "author_cap": author_cap,
        "maximum_sampled_reports_per_author": int(author_report_counts.max()),
        "authors_by_sampled_report_count": {
            str(key): int(value)
            for key, value in author_report_counts.value_counts().sort_index().items()
        },
        "authors_contributing_both_labels": int(
            (overall_author_label_counts > 1).sum()
        ),
        "paired_mixed_label_authors": len(paired_lengths),
        "paired_reports": len(paired),
        "paired_reports_per_class": {
            label: int(paired_counts.get(label, 0))
            for label in ("lucid", "nonlucid")
        },
        "pilot_paired_mixed_label_authors": len(pilot_pair_ids),
        "pilot_paired_reports": len(pilot_paired),
        "pilot_paired_reports_per_class": {
            label: int(pilot_paired_counts.get(label, 0))
            for label in ("lucid", "nonlucid")
        },
        "independently_sampled_reports": independent_reports,
        "pilot_independently_sampled_reports": pilot_independent_reports,
        "author_disjoint_between_classes": False,
        "mixed_label_authors_available_after_duplicate_exclusion": mixed_label_author_count,
        "mixed_label_authors_selected": paired_author_target,
        "reports_excluded_for_detected_cross_author_duplicates": excluded_duplicate_count,
        "pilot_size": len(pilot),
        "pilot_n_unique_authors": int(pilot["author_cluster_id"].nunique()),
        "pilot_maximum_reports_per_author": int(pilot_author_report_counts.max()),
        "pilot_authors_by_sampled_report_count": {
            str(key): int(value)
            for key, value in pilot_author_report_counts.value_counts()
            .sort_index()
            .items()
        },
        "pilot_requested_per_class": requested_pilot_per_class,
        "pilot_paired_author_target": pilot_paired_author_target,
        "pilot_annotator_template_rows": 2 * len(pilot),
        "length_balance": length_balance,
        "pilot_length_balance": pilot_length_balance,
        "seed": seed,
        "pilot_seed": pilot_seed,
        "raw_dream_text_in_manifest": False,
        "raw_dream_text_in_annotation_template": False,
        "author_identities_in_annotator_files": False,
        "source_labels_in_annotator_files": False,
        "pairing_information_in_annotator_files": False,
    }


def _paired_length_table(paired):
    if paired.empty:
        return pd.DataFrame(columns=["lucid", "nonlucid"])
    return paired.pivot(
        index="paired_author_id", columns="sampling_label", values="word_count"
    )


def _label_length_balance(lucid, nonlucid):
    return {
        "lucid_mean_words": float(np.mean(lucid)),
        "nonlucid_mean_words": float(np.mean(nonlucid)),
        "mean_difference_words": float(np.mean(lucid) - np.mean(nonlucid)),
        "standardized_mean_difference": _standardized_mean_difference(
            lucid, nonlucid
        ),
    }


def _standardized_mean_difference(first, second):
    pooled_sd = np.sqrt((np.var(first, ddof=1) + np.var(second, ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0 if np.mean(first) == np.mean(second) else None
    return float((np.mean(first) - np.mean(second)) / pooled_sd)


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
    for annotator_id, template in sample["annotation_templates"].items():
        template.to_csv(
            output / f"{annotator_id}_annotation_template.csv", index=False
        )
    (output / "sample_summary.json").write_text(
        json.dumps(sample["summary"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sample["summary"]
