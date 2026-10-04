from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

from .data import infer_columns, load_reports, normalize_binary_labels
from .features import dimension_scores
from .split import author_disjoint_split


def run(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = load_reports(cfg["data_path"])
    cols = infer_columns(df, cfg)
    df = df.copy()
    df["target"] = normalize_binary_labels(df[cols.label], cfg["positive_labels"], cfg["negative_labels"])
    df = df[df["target"].notna() & df[cols.text].notna()].reset_index(drop=True)

    texts = df[cols.text].astype(str).to_numpy()
    labels = df["target"].astype(int).to_numpy()
    groups = df[cols.author].astype(str).to_numpy() if cols.author else None

    train_idx, test_idx = author_disjoint_split(
        texts, labels, groups, test_size=float(cfg["test_size"]), seed=int(cfg["seed"])
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            min_df=int(cfg["min_df"]),
            max_features=int(cfg["max_features"]),
            ngram_range=tuple(cfg["ngram_range"]),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=float(cfg["C"]), max_iter=2000, class_weight="balanced", random_state=int(cfg["seed"])
        )),
    ])

    model.fit(texts[train_idx], labels[train_idx])
    pred = model.predict(texts[test_idx])
    prob = model.predict_proba(texts[test_idx])[:, 1]

    metrics = {
        "n_total": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": float(accuracy_score(labels[test_idx], pred)),
        "macro_f1": float(f1_score(labels[test_idx], pred, average="macro")),
        "auroc": float(roc_auc_score(labels[test_idx], prob)),
        "confusion_matrix": confusion_matrix(labels[test_idx], pred).tolist(),
        "author_disjoint": bool(cols.author),
    }

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, out / "model.joblib")

    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = np.asarray(vectorizer.get_feature_names_out())
    coef = clf.coef_[0]
    top_pos = pd.DataFrame({"feature": names[np.argsort(coef)[-50:][::-1]], "coefficient": np.sort(coef)[-50:][::-1]})
    top_neg = pd.DataFrame({"feature": names[np.argsort(coef)[:50]], "coefficient": np.sort(coef)[:50]})
    top_pos.to_csv(out / "top_lucid_features.csv", index=False)
    top_neg.to_csv(out / "top_nonlucid_features.csv", index=False)

    scored = pd.DataFrame([dimension_scores(t) for t in texts[test_idx]])
    scored["target"] = labels[test_idx]
    scored.to_csv(out / "phenomenology_rule_scores.csv", index=False)

    frac_pos, mean_pred = calibration_curve(labels[test_idx], prob, n_bins=10)
    pd.DataFrame({"mean_predicted_probability": mean_pred, "fraction_positive": frac_pos}).to_csv(
        out / "calibration.csv", index=False
    )

    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
