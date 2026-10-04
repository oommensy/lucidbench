from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .data import infer_columns, load_reports, normalize_binary_labels
from .split import author_disjoint_split

DATASET_DOI = "10.5281/zenodo.19161757"
DATASET_LICENSE = "CC BY-NC 4.0"
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
)


def remove_cues(text: str, terms: list[str]) -> str:
    """Remove only the configured whole-word cues, preserving ordinary variants."""
    alternatives = "|".join(
        re.escape(term) for term in sorted(terms, key=len, reverse=True)
    )
    if not alternatives:
        return text
    stripped = re.sub(
        rf"(?<!\w)(?:{alternatives})(?!\w)", " ", str(text), flags=re.IGNORECASE
    )
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    return " ".join(stripped.split())


def calculate_metrics(y_true, predictions, probabilities) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    class_totals = matrix.sum(axis=1)
    represented_classes = class_totals > 0
    class_recalls = np.divide(
        matrix.diagonal(),
        class_totals,
        out=np.zeros(2, dtype=float),
        where=represented_classes,
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(class_recalls[represented_classes].mean()),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if np.unique(y_true).size == 2
            else None
        ),
        "confusion_matrix": matrix.tolist(),
    }


def bootstrap_author_intervals(
    y_true,
    predictions,
    probabilities,
    authors,
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile intervals from resampling held-out authors with all their reports."""
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    authors = np.asarray(authors).astype(str)
    author_indices = {
        author: np.flatnonzero(authors == author) for author in np.unique(authors)
    }
    author_ids = np.array(list(author_indices))
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in METRIC_NAMES}

    for _ in range(replicates):
        selected_authors = rng.choice(author_ids, size=len(author_ids), replace=True)
        indices = np.concatenate(
            [author_indices[author] for author in selected_authors]
        )
        values = calculate_metrics(
            y_true[indices], predictions[indices], probabilities[indices]
        )
        if np.unique(y_true[indices]).size < 2:
            values["balanced_accuracy"] = None
            values["roc_auc"] = None
        for name in METRIC_NAMES:
            if values[name] is not None:
                samples[name].append(values[name])

    return {
        name: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
        if values
    }


def length_match_indices(
    lengths, labels, *, bins: list[float], seed: int
) -> np.ndarray:
    """Downsample both labels to equal counts in each training-derived length bin."""
    lengths = np.asarray(lengths, dtype=int)
    labels = np.asarray(labels, dtype=int)
    internal_edges = np.unique(np.asarray(bins, dtype=float)[1:-1])
    groups = np.digitize(lengths, internal_edges, right=True)
    rng = np.random.default_rng(seed)
    matched: list[np.ndarray] = []

    for length_bin in np.unique(groups):
        bin_indices = np.flatnonzero(groups == length_bin)
        by_label = [bin_indices[labels[bin_indices] == label] for label in (0, 1)]
        count = min(map(len, by_label))
        if count:
            matched.extend(
                rng.choice(indices, size=count, replace=False) for indices in by_label
            )

    if not matched:
        raise ValueError("No length bins contain reports from both label classes.")
    return np.sort(np.concatenate(matched))


def _length_match_diagnostics(
    split_name, lengths, labels, selected_indices, bins
) -> list[dict]:
    lengths = np.asarray(lengths, dtype=int)
    labels = np.asarray(labels, dtype=int)
    selected_mask = np.zeros(len(lengths), dtype=bool)
    selected_mask[selected_indices] = True
    internal_edges = np.unique(np.asarray(bins, dtype=float)[1:-1])
    length_bins = np.digitize(lengths, internal_edges, right=True)
    rows = []
    for length_bin in np.unique(length_bins):
        for label, label_name in ((0, "nonlucid"), (1, "lucid")):
            in_bin = (length_bins == length_bin) & (labels == label)
            matched = in_bin & selected_mask
            rows.append(
                {
                    "split": split_name,
                    "length_bin": int(length_bin),
                    "label": label_name,
                    "available_reports": int(in_bin.sum()),
                    "matched_reports": int(matched.sum()),
                    "matched_mean_word_count": (
                        float(lengths[matched].mean()) if matched.any() else None
                    ),
                }
            )
    return rows


def _build_model(cfg: dict) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=int(cfg["min_df"]),
                    max_features=int(cfg["max_features"]),
                    ngram_range=tuple(cfg["ngram_range"]),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=float(cfg["C"]),
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=int(cfg["seed"]),
                ),
            ),
        ]
    )


def _fit_predict(cfg, train_texts, train_labels, test_texts):
    model = _build_model(cfg)
    model.fit(train_texts, train_labels)
    return model, model.predict(test_texts), model.predict_proba(test_texts)[:, 1]


def _coefficient_rows(
    model: Pipeline,
    train_texts,
    train_authors,
    *,
    max_terms: int,
    min_author_support: int,
) -> list[dict]:
    vectorizer = model.named_steps["tfidf"]
    features = np.asarray(vectorizer.get_feature_names_out())
    coefficients = model.named_steps["clf"].coef_[0]
    train_matrix = vectorizer.transform(train_texts).tocsc()
    train_authors = np.asarray(train_authors).astype(str)
    rows = []

    for direction, order in (
        ("positive", np.argsort(coefficients)[::-1]),
        ("negative", np.argsort(coefficients)),
    ):
        added = 0
        for feature_index in order:
            feature = features[feature_index]
            if " " in feature or not re.fullmatch(r"[a-z][a-z'-]*", feature):
                continue
            author_support = np.unique(
                train_authors[
                    train_matrix.indices[
                        train_matrix.indptr[feature_index] : train_matrix.indptr[
                            feature_index + 1
                        ]
                    ]
                ]
            ).size
            if author_support < min_author_support:
                continue
            rows.append(
                {
                    "direction": direction,
                    "feature": feature,
                    "coefficient": float(coefficients[feature_index]),
                    "training_document_frequency": int(
                        train_matrix.indptr[feature_index + 1]
                        - train_matrix.indptr[feature_index]
                    ),
                    "training_author_support": int(author_support),
                }
            )
            added += 1
            if added == max_terms:
                break
    return rows


def _dataset_summary(frame: pd.DataFrame, path: Path, cfg: dict, cols) -> dict:
    labels = normalize_binary_labels(
        frame[cols.label], cfg["positive_labels"], cfg["negative_labels"]
    )
    eligible = labels.notna() & frame[cols.text].notna()
    lengths = frame[cols.text].fillna("").astype(str).str.split().str.len()
    summary = {
        "source": {
            "title": "A large corpus of lucid and non-lucid dream reports",
            "doi": DATASET_DOI,
            "license": DATASET_LICENSE,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
        },
        "n_reports_total": len(frame),
        "n_reports_binary_labeled": int(eligible.sum()),
        "n_reports_excluded_unlabeled_or_missing_text": int((~eligible).sum()),
        "lucidity_label_counts": {
            str(label): int(count)
            for label, count in frame[cols.label].value_counts(dropna=False).items()
        },
        "binary_class_counts": {
            "nonlucid": int((labels[eligible] == 0).sum()),
            "lucid": int((labels[eligible] == 1).sum()),
        },
        "author_column": cols.author,
        "n_unique_authors_all_reports": (
            int(frame[cols.author].nunique(dropna=True)) if cols.author else None
        ),
        "n_unique_authors_binary_labeled": (
            int(frame.loc[eligible, cols.author].nunique(dropna=True))
            if cols.author
            else None
        ),
        "missing_values": {
            column: {
                "count": int(count),
                "fraction": float(count / len(frame)),
            }
            for column, count in frame.isna().sum().items()
        },
        "report_length_words": {
            str(int(percentile * 100)): float(value)
            for percentile, value in lengths.quantile(
                [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
            ).items()
        },
    }
    if cols.author:
        reports_per_author = frame[cols.author].dropna().value_counts()
        summary["reports_per_author"] = {
            "n_authors": len(reports_per_author),
            "quantiles": {
                str(int(percentile * 100)): float(value)
                for percentile, value in reports_per_author.quantile(
                    [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
                ).items()
            },
        }
    else:
        summary["reports_per_author"] = None
    return summary


def _flatten_summary(summary: dict) -> pd.DataFrame:
    rows = [
        ("n_reports_total", summary["n_reports_total"]),
        ("n_reports_binary_labeled", summary["n_reports_binary_labeled"]),
        (
            "n_reports_excluded_unlabeled_or_missing_text",
            summary["n_reports_excluded_unlabeled_or_missing_text"],
        ),
        ("n_unique_authors_all_reports", summary["n_unique_authors_all_reports"]),
        ("n_unique_authors_binary_labeled", summary["n_unique_authors_binary_labeled"]),
        ("source_doi", summary["source"]["doi"]),
        ("source_license", summary["source"]["license"]),
    ]
    rows.extend(
        (f"label_count_{label}", count)
        for label, count in summary["lucidity_label_counts"].items()
    )
    rows.extend(
        (f"binary_class_count_{label}", count)
        for label, count in summary["binary_class_counts"].items()
    )
    rows.extend(
        (f"missing_{column}", details["count"])
        for column, details in summary["missing_values"].items()
    )
    rows.extend(
        (f"report_length_words_p{percentile}", value)
        for percentile, value in summary["report_length_words"].items()
    )
    if summary["reports_per_author"]:
        rows.extend(
            (f"reports_per_author_p{percentile}", value)
            for percentile, value in summary["reports_per_author"]["quantiles"].items()
        )
    return pd.DataFrame(rows, columns=["measure", "value"])


def _save_confusion_matrices(path: Path, experiments: dict) -> None:
    rows = []
    for experiment_name, result in experiments.items():
        for true_index, true_label in enumerate(("nonlucid", "lucid")):
            for pred_index, predicted_label in enumerate(("nonlucid", "lucid")):
                rows.append(
                    {
                        "experiment": experiment_name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": result["metrics"]["confusion_matrix"][true_index][
                            pred_index
                        ],
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_comparison_figure(path: Path, experiments: dict) -> None:
    names = list(experiments)
    values = [experiments[name]["metrics"]["f1"] for name in names]
    lower = [experiments[name]["confidence_intervals"]["f1"]["lower"] for name in names]
    upper = [experiments[name]["confidence_intervals"]["f1"]["upper"] for name in names]
    errors = np.array([np.array(values) - lower, np.array(upper) - values])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(names, values, color=["#3977a8", "#d08c36", "#5a9c71"])
    ax.errorbar(names, values, yerr=errors, fmt="none", ecolor="#222222", capsize=5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (lucid class)")
    ax.set_title("Author-disjoint held-out performance (95% author-bootstrap CI)")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _experiment_result(
    name,
    model,
    y_true,
    predictions,
    probabilities,
    authors,
    *,
    cfg,
    n_train,
    n_test,
):
    return {
        "name": name,
        "metrics": calculate_metrics(y_true, predictions, probabilities),
        "confidence_intervals": bootstrap_author_intervals(
            y_true,
            predictions,
            probabilities,
            authors,
            replicates=int(cfg["bootstrap_replicates"]),
            seed=int(cfg["seed"]),
        ),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_test_authors": int(pd.Series(authors).nunique()),
        "model": model,
        "predictions": np.asarray(predictions),
        "probabilities": np.asarray(probabilities),
    }


def run_experiments(config_path: str | Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_path = Path(cfg["data_path"])
    frame = load_reports(data_path)
    columns = infer_columns(frame, cfg)
    summary = _dataset_summary(frame, data_path, cfg, columns)
    label_series = normalize_binary_labels(
        frame[columns.label], cfg["positive_labels"], cfg["negative_labels"]
    )
    eligible = label_series.notna() & frame[columns.text].notna()
    analytic = frame.loc[eligible].reset_index(drop=True)
    labels = label_series.loc[eligible].astype(int).to_numpy()
    texts = analytic[columns.text].astype(str).to_numpy()
    lengths = np.fromiter((len(text.split()) for text in texts), dtype=int)
    author_available = columns.author is not None
    author_ids_complete = author_available and analytic[columns.author].notna().all()
    if author_available:
        groups = (
            analytic[columns.author]
            .astype("string")
            .fillna(
                pd.Series(
                    [f"missing-author-row-{index}" for index in range(len(analytic))],
                    index=analytic.index,
                    dtype="string",
                )
            )
            .astype(str)
            .to_numpy()
        )
    else:
        groups = None

    train_indices, test_indices = author_disjoint_split(
        texts,
        labels,
        groups,
        test_size=float(cfg["test_size"]),
        seed=int(cfg["seed"]),
    )
    if groups is not None and not set(groups[train_indices]).isdisjoint(
        set(groups[test_indices])
    ):
        raise RuntimeError("Author leakage detected between train and test partitions.")

    train_texts = texts[train_indices]
    test_texts = texts[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    train_authors = (
        groups[train_indices]
        if groups is not None
        else np.array([f"report-{index}" for index in train_indices])
    )
    test_authors = (
        groups[test_indices]
        if groups is not None
        else np.array([f"report-{index}" for index in test_indices])
    )

    experiments = {}
    baseline_model, baseline_predictions, baseline_probabilities = _fit_predict(
        cfg, train_texts, train_labels, test_texts
    )
    experiments["baseline"] = _experiment_result(
        "baseline",
        baseline_model,
        test_labels,
        baseline_predictions,
        baseline_probabilities,
        test_authors,
        cfg=cfg,
        n_train=len(train_indices),
        n_test=len(test_indices),
    )

    cues = cfg["ablation_terms"]
    ablated_train_texts = [remove_cues(text, cues) for text in train_texts]
    ablated_test_texts = [remove_cues(text, cues) for text in test_texts]
    ablated_model, ablated_predictions, ablated_probabilities = _fit_predict(
        cfg, ablated_train_texts, train_labels, ablated_test_texts
    )
    experiments["lexical_ablation"] = _experiment_result(
        "lexical_ablation",
        ablated_model,
        test_labels,
        ablated_predictions,
        ablated_probabilities,
        test_authors,
        cfg=cfg,
        n_train=len(train_indices),
        n_test=len(test_indices),
    )

    length_edges = np.unique(
        np.quantile(
            lengths[train_indices],
            np.linspace(0, 1, int(cfg["length_bins"]) + 1),
        )
    )
    matched_train_relative = length_match_indices(
        lengths[train_indices],
        train_labels,
        bins=length_edges,
        seed=int(cfg["seed"]),
    )
    matched_test_relative = length_match_indices(
        lengths[test_indices],
        test_labels,
        bins=length_edges,
        seed=int(cfg["seed"]) + 1,
    )
    length_matching_rows = _length_match_diagnostics(
        "train",
        lengths[train_indices],
        train_labels,
        matched_train_relative,
        length_edges,
    ) + _length_match_diagnostics(
        "test",
        lengths[test_indices],
        test_labels,
        matched_test_relative,
        length_edges,
    )
    matched_model, matched_predictions, matched_probabilities = _fit_predict(
        cfg,
        train_texts[matched_train_relative],
        train_labels[matched_train_relative],
        test_texts[matched_test_relative],
    )
    matched_test_authors = test_authors[matched_test_relative]
    experiments["length_matched"] = _experiment_result(
        "length_matched",
        matched_model,
        test_labels[matched_test_relative],
        matched_predictions,
        matched_probabilities,
        matched_test_authors,
        cfg=cfg,
        n_train=len(matched_train_relative),
        n_test=len(matched_test_relative),
    )

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    _flatten_summary(summary).to_csv(output_dir / "dataset_summary.csv", index=False)

    metrics = {
        "dataset_doi": DATASET_DOI,
        "dataset_license": DATASET_LICENSE,
        "seed": int(cfg["seed"]),
        "test_size": float(cfg["test_size"]),
        "author_column": columns.author,
        "author_disjoint": bool(author_ids_complete),
        "split_method": (
            "GroupShuffleSplit by author"
            if author_available
            else "stratified report-level split; author identity unavailable"
        ),
        "n_total_reports": len(frame),
        "n_eligible_reports": len(analytic),
        "n_train_authors": int(pd.Series(train_authors).nunique()),
        "n_test_authors": int(pd.Series(test_authors).nunique()),
        "partition_class_counts": {
            "train": {
                "nonlucid": int((train_labels == 0).sum()),
                "lucid": int((train_labels == 1).sum()),
            },
            "test": {
                "nonlucid": int((test_labels == 0).sum()),
                "lucid": int((test_labels == 1).sum()),
            },
        },
        "ablation_terms": cues,
        "bootstrap": {
            "method": "percentile cluster bootstrap resampling held-out authors",
            "replicates": int(cfg["bootstrap_replicates"]),
            "confidence_level": 0.95,
        },
        "length_matching": {
            "method": "downsample both labels within training-derived word-count quantile bins, separately in train and test",
            "bins": int(cfg["length_bins"]),
            "n_train": len(matched_train_relative),
            "n_test": len(matched_test_relative),
            "n_test_authors": int(pd.Series(matched_test_authors).nunique()),
            "matched_test_class_counts": {
                "nonlucid": int((test_labels[matched_test_relative] == 0).sum()),
                "lucid": int((test_labels[matched_test_relative] == 1).sum()),
            },
            "word_count_quantile_edges": [float(value) for value in length_edges],
        },
        "experiments": {},
    }

    csv_rows = []
    confusion_results = {}
    for name, result in experiments.items():
        result_metrics = {
            key: value
            for key, value in result.items()
            if key not in {"model", "predictions", "probabilities"}
        }
        metrics["experiments"][name] = result_metrics
        confusion_results[name] = result
        csv_row = {
            "experiment": name,
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "n_test_authors": result["n_test_authors"],
        }
        for metric_name in METRIC_NAMES:
            value = result["metrics"][metric_name]
            interval = result["confidence_intervals"].get(metric_name)
            csv_row[metric_name] = value
            csv_row[f"{metric_name}_ci_lower"] = interval["lower"] if interval else None
            csv_row[f"{metric_name}_ci_upper"] = interval["upper"] if interval else None
        csv_rows.append(csv_row)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(csv_rows).to_csv(output_dir / "metrics.csv", index=False)
    _save_confusion_matrices(output_dir / "confusion_matrices.csv", confusion_results)
    pd.DataFrame(length_matching_rows).to_csv(
        output_dir / "length_matching.csv", index=False
    )

    for name in ("baseline", "lexical_ablation"):
        model = experiments[name]["model"]
        train_for_model = (
            train_texts
            if name == "baseline"
            else np.asarray(ablated_train_texts, dtype=object)
        )
        rows = _coefficient_rows(
            model,
            train_for_model,
            train_authors,
            max_terms=int(cfg["coefficient_terms"]),
            min_author_support=int(cfg["minimum_coefficient_author_support"]),
        )
        pd.DataFrame(rows).to_csv(output_dir / f"coefficients_{name}.csv", index=False)

    _save_comparison_figure(output_dir / "performance_comparison.png", experiments)

    summary_only = {
        "n_total_reports": summary["n_reports_total"],
        "n_eligible_reports": summary["n_reports_binary_labeled"],
        "binary_class_counts": summary["binary_class_counts"],
        "author_disjoint": metrics["author_disjoint"],
        "experiments": {
            name: result["metrics"] for name, result in metrics["experiments"].items()
        },
    }
    print(json.dumps(summary_only, indent=2, allow_nan=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.2.yaml")
    args = parser.parse_args()
    run_experiments(args.config)


if __name__ == "__main__":
    main()
