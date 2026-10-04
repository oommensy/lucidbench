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

Dataset files are not committed to this repository. Download them locally with the script below. Review the source dataset's terms and ethical considerations before use.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/inspect_data.py
python -m lucidbench.train_baseline --config configs/baseline.yaml
```

Run tests:

```bash
python -m pytest
```

## Methodology

**Dreamer/author-disjoint splitting is required wherever user IDs are available.** All reports from an identified dreamer must remain on one side of a train/test split, preventing the model from benefiting from the same person's writing style in both sets. When the data have no author identifier, the current splitter uses stratified report-level splitting; those results must be identified as not author-disjoint.

Planned analyses include:

- report-length-matched experiments;
- lexical ablations removing explicit words such as `lucid`, `dream`, `realized`, and `woke`;
- user-balanced sampling;
- bootstrap confidence intervals over held-out dreamers; and
- error analysis by phenomenological dimensions.

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
├── results/              # ignored generated metrics and artifacts
├── scripts/              # data utilities
├── src/lucidbench/       # reusable package
└── tests/                # unit tests
```

## Research status

- **Planned experiments:** author-disjoint baseline, explicit-vocabulary ablation, length matching, and human-annotated phenomenology analyses.
- **Baseline results:** none are included in this repository yet. Running the pipeline locally will generate dataset-specific outputs under the ignored `results/` directory.
- **Validated findings:** none are claimed. Any future findings should report their data, split strategy, uncertainty, and validation status.

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

Research prototype — v0.1.
