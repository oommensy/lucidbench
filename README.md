# LucidBench

**LucidBench — Computational Markers of Lucidity, Agency, Control, and Metacognition in Dream Reports**

LucidBench is an open computational dream-science research project. Its initial goal is to study whether lucid and non-lucid dreams exhibit reproducible computational and linguistic differences, with special attention to separating:

- lucidity / awareness of dreaming;
- agency;
- dream control;
- metacognition;
- waking-memory access;
- sensory and perceptual features; and
- emotional characteristics.

The project is intended to support reproducible research and, over time, collaboration with academic sleep and dream researchers. Lucidity is not treated as interchangeable with agency or dream control.

## Research questions

1. Can lucid vs. non-lucid dream reports be predicted above lexical baselines when evaluation is held out by dreamer?
2. Which linguistic features are associated with lucidity after controlling for author leakage, report length, and explicit label vocabulary?
3. Are awareness, agency, control, metacognition, waking-memory access, sensory/perceptual features, and emotional characteristics empirically separable?
4. How well do models generalize to unseen dreamers and, eventually, independently collected laboratory reports?

## Scope of v0.1

- Download and inspect the public DreamViews-derived corpus.
- Normalize lucid/non-lucid labels and use dreamer/author-disjoint evaluation whenever user IDs are available.
- Train a TF-IDF + logistic-regression classification baseline.
- Produce classification metrics, model coefficients, and exploratory rule-based dimension scores.

The rule-based scores are exploratory indicators, not validated measurements or human annotations. The v0.1 classifier and associated evaluation scripts are research baselines, not evidence that any particular linguistic marker is a validated finding.

## Data

Primary dataset:

- Remington Mallett (2026), **A large corpus of lucid and non-lucid dream reports**.
- Zenodo record: https://zenodo.org/records/19161757
- Paper: https://arxiv.org/abs/2603.26992
- Original software repository: https://github.com/remrama/dreamviews

Dataset files are not committed to this repository. Download them locally with the script below. The Zenodo record identifies the corpus as **CC BY-NC 4.0**; review those terms and ethical considerations before use. The code is MIT-licensed, but the source dataset and the dataset-derived coefficient tables in `results/v0.2/` are not.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/inspect_data.py
python -m lucidbench.experiments --config configs/v0.2.yaml
python -m lucidbench.train_baseline --config configs/baseline.yaml
```

Run tests:

```bash
python -m pytest
```

## Methodology

**Dreamer/author-disjoint splitting is required wherever user IDs are available.** All reports from an identified dreamer must remain on one side of a train/test split, preventing the model from benefiting from the same person's writing style in both sets. When the data have no author identifier, the current splitter uses stratified report-level splitting; those results must be identified as not author-disjoint.

Future analyses include:

- user-balanced sampling;
- repeated author-group splits; and
- human-validated phenomenology and external laboratory evaluation.

The v0.2 baseline, lexical ablation, length-matched sensitivity analysis, and author-bootstrap intervals are documented below and in [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md).

See [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) for the planned protocol and threats to validity.

## Planned phenomenology schema

| Dimension | Operational question |
|---|---|
| Awareness | Does the dreamer explicitly recognize the experience as a dream? |
| Agency | Does the dreamer intentionally choose actions? |
| Control | Does the dreamer deliberately alter dream content or the environment? |
| Metacognition | Does the dreamer reason about their own mental state or dream state? |
| Waking-memory access | Does the dreamer access information or intentions from waking life? |
| Sensory/perceptual features | Which sensory modalities and perceptual qualities are reported? |
| Emotional characteristics | Which emotions are reported, and how are they characterized? |
| Stability | Does awareness/control appear to affect continuation, fading, or termination? |

See [`docs/ANNOTATION_RUBRIC.md`](docs/ANNOTATION_RUBRIC.md). The rubric is a proposed annotation framework; its dimensions require human validation before being treated as validated measures.

## Repository structure

```text
lucidbench/
├── configs/              # experiment configuration
├── data/                 # ignored raw/processed datasets
├── docs/                 # research protocol, rubric, literature notes
├── notebooks/            # exploratory notebook guidance
├── results/              # committed aggregate v0.2 results; other outputs ignored
├── scripts/              # data utilities
├── src/lucidbench/       # reusable package
└── tests/                # unit tests
```

## v0.2 Results

The first corpus-backed baseline was run on Remington Mallett's *A large corpus of lucid and non-lucid dream reports* (Zenodo [10.5281/zenodo.19161757](https://doi.org/10.5281/zenodo.19161757), CC BY-NC 4.0). Its 57,778 rows contain 10,231 `lucid`, 27,830 `nonlucid`, 15,606 `unspecified`, and 4,111 `ambiguous` labels. The binary experiment includes only the first two categories (38,061 reports) and excludes the other 19,717. The full corpus has 5,036 authors; the labeled subset has 3,689. The schema, missingness, report-length, and reports-per-author summaries are in [`dataset_summary.json`](results/v0.2/dataset_summary.json).

With seed 42, an 80/20 `GroupShuffleSplit` by `user_id` produced 30,710 training and 7,351 test reports; all 738 test authors were absent from training. TF-IDF used 1–2 grams (`min_df=3`, up to 50,000 features) and class-weighted logistic regression (`C=1`). Intervals are 95% percentile intervals from 1,000 cluster-bootstrap resamples of held-out authors.

| Experiment | Accuracy | Balanced accuracy | Precision | Recall | F1 (95% author-bootstrap CI) | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.930 | 0.915 | 0.834 | 0.888 | 0.860 (0.824–0.890) | 0.972 |
| Lexical ablation | 0.908 | 0.888 | 0.790 | 0.849 | 0.819 (0.775–0.853) | 0.957 |
| Length-matched sensitivity | 0.909 | 0.909 | 0.924 | 0.893 | 0.908 (0.882–0.928) | 0.969 |

The ablation removes configured whole-word cues (`lucid`, `dream` and inflections, `realize`/`realise` forms, `wake`/`woke` forms, `reality`, `real`, plus the lucid-dream abbreviations `LD`, `DILD`, and `RC`). It preserves unlisted words such as `dreamer` and `wild`. The length-matched sensitivity model is trained and tested on samples balanced by class within five word-count bins; it uses 16,888 training and 3,574 test reports, with 1,787 reports per class in the test set. Because its held-out set has a different size and class prevalence from the full test set, its scores should not be directly compared as if they were measured on the same population.

The figure and aggregate tables are available in [`results/v0.2/`](results/v0.2/), including confusion matrices, per-bin matching diagnostics, and baseline/ablation coefficient tables. Coefficients are model associations, not causal explanations or validated phenomenological markers. The tables list unigrams supported by at least ten training authors and omit report-level predictions and author IDs. Dataset-derived tables are attributed and identified as CC BY-NC 4.0 in their accompanying README.

These are initial results from one fixed author-disjoint split, not independent replication or laboratory validation. The high performance remaining after cue removal is a useful result to investigate, not proof that the model captures dream phenomenology.

![Author-disjoint F1 comparison with 95% author-bootstrap intervals](results/v0.2/performance_comparison.png)

## Research status

- **Produced in v0.2:** corpus summary, author-disjoint TF-IDF baseline, lexical-cue ablation, author-cluster bootstrap intervals, length-matched sensitivity analysis, and coefficient inspection.
- **Still planned:** repeated or cross-validated author-group evaluation, user-balanced and label-definition sensitivity checks, human-annotated phenomenology, and external laboratory validation.
- **Validated findings:** none are claimed. These baseline associations are not causal, clinically useful, or independently validated.

## Roadmap

**Phase 1 — Reproducible baseline**
- corpus ingestion;
- author-disjoint evaluation;
- TF-IDF baseline; and
- interpretable coefficient analysis.

**Phase 2 — Phenomenology annotations**
- manually label a stratified sample;
- estimate inter-rater reliability; and
- compare rule-based, embedding, and LLM annotations.

**Phase 3 — Representation learning**
- sentence/document embeddings;
- controlled classifiers; and
- latent structure of awareness vs. control.

**Phase 4 — External scientific collaboration**
- validate against laboratory lucid-dream reports;
- align retrospective language with real-time dream signals where data access is possible; and
- explore multimodal EEG + report representations with collaborating labs.

## Ethics and privacy

This repository analyzes public research data and does not diagnose sleep or mental-health conditions. Do not commit private dream journals, personally identifying information, credentials, or locally obtained datasets. Any future collection of human-subject data, experimental sleep manipulation, or physiological recordings should be conducted under appropriate institutional ethics/IRB oversight.

## Citation and license

If you use the underlying dataset, cite its original creator and record. LucidBench is an independent analysis framework and does not redistribute the source corpus. The code is distributed under the MIT License; see [`LICENSE`](LICENSE).

## Project status

Research prototype — v0.2 baseline experiments.
