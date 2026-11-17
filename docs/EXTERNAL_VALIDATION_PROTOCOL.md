# LucidBench v0.4 External Validation Protocol

## Purpose

v0.4 is the external-validation track for LucidBench. It is designed for an independent researcher-provided dream-report dataset with repeated reports per participant and existing human content-analysis ratings.

The core principle is simple:

> **Do not adapt LucidBench to the external human ratings before predictions are frozen.**

The external dataset is therefore treated as a held-out evaluation resource, not as additional training data.

## Expected dataset structure

The current working expectation, based on collaborator correspondence, is approximately:

- 24 participants;
- about four dream reports per participant;
- roughly 100 reports total;
- four report categories: non-lucid, lucid, recent non-lucid, and recent lucid;
- existing human content-analysis ratings.

Exact field names, scales, and data-sharing constraints remain to be confirmed by the data owners.

## Validation stages

### Stage 0 — Data-sharing and documentation

Before receiving data:

1. confirm permitted storage, processing, retention, and publication;
2. confirm whether reports are de-identified;
3. document the human content-analysis instruments and scoring rules;
4. prospectively map external constructs to LucidBench constructs;
5. decide which comparisons are confirmatory vs exploratory.

Do not request or inspect individual human scores until the construct mapping is recorded.

### Stage 1 — Reports-only ingestion

Receive a reports table containing only what is needed to generate predictions.

Recommended fields:

- `report_id`
- `participant_id`
- `condition`
- `dream_text`

The preparation script creates:

- **model packet:** `report_id,dream_text`
- **private linkage manifest:** `report_id,participant_id,condition`

The model packet must not contain human ratings.

### Stage 2 — Freeze LucidBench predictions

Run the frozen LucidBench model or annotation algorithm on the reports-only packet.

Record:

- code commit SHA;
- model/version identifier;
- prediction timestamp;
- construct definitions;
- all preprocessing settings.

Predictions must be saved before human ratings are unsealed for comparison.

### Stage 3 — Unseal human ratings

Only after predictions are frozen:

1. load the external human ratings;
2. validate report IDs and rater/instrument metadata;
3. join predictions to ratings by `report_id`;
4. run the preregistered comparisons.

Do not alter model thresholds, prompts, label mappings, or construct definitions after inspecting agreement results unless the analysis is explicitly labeled post-hoc/exploratory.

## Prospective construct crosswalk

The exact crosswalk cannot be finalized until the collaborators provide the names and definitions of their existing content-analysis variables.

Use `docs/templates/external_construct_crosswalk.csv` to record, before score inspection:

- external construct;
- external instrument/method;
- LucidBench construct;
- mapping type: direct / partial / exploratory / no-match;
- direction/scale transformation;
- confirmatory status;
- rationale.

A construct with no defensible mapping should remain unmapped rather than being forced into LucidBench.

## Primary analyses

Where mappings are direct enough:

- report-level agreement/correlation between LucidBench outputs and external human ratings;
- ordinal or nominal agreement appropriate to the scale;
- participant-clustered uncertainty;
- repeated-measures analyses that respect multiple reports from the same participant.

Potential within-person contrasts include lucid vs non-lucid and recent lucid vs recent non-lucid reports, but exact contrasts depend on the original study design and condition definitions.

## Secondary analyses

Exploratory analyses may include:

- construct-level error analysis;
- calibration by condition;
- transition-profile prevalence;
- whether control attempt/success/domain generalizes;
- whether state reasoning and lucid insight remain separable;
- participant-level heterogeneity.

## Leakage safeguards

The following are prohibited before predictions are frozen:

- training on the external reports;
- tuning thresholds against external ratings;
- changing prompts after viewing ratings;
- choosing construct mappings based on whichever mapping yields the highest agreement;
- using participant identity or condition labels as prediction features unless explicitly part of a preregistered analysis.

## Human-rating independence

If multiple human raters exist, preserve their individual ratings when possible. Do not collapse ratings until the original scoring procedure is understood.

External human content analysis is not assumed to be a perfect gold standard. Agreement should be interpreted as convergent validity between independently operationalized measurements.

## Data handling

External reports and ratings should remain untracked unless the data-sharing agreement explicitly permits redistribution.

Recommended local layout:

```text
data/external/
├── raw/
│   ├── reports.csv
│   └── human_ratings.csv
└── processed/
    ├── model_packet.csv
    ├── linkage_manifest.csv
    ├── frozen_predictions.csv
    └── validation_results.json
```

Add or preserve ignore rules so collaborator data and row-level outputs are never committed by default.

## Current status

- External dataset offered in principle.
- Data not yet received.
- Human-rating constructs not yet documented.
- No external results exist.
- This protocol and ingestion scaffold are prospective and intentionally precede score inspection.
