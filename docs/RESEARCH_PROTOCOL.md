# Research Protocol — v0.1

This document describes hypotheses and analyses planned for v0.1; it does not report completed experiments or validated findings. Any baseline results produced by running the code should be reported separately with the dataset, split strategy, uncertainty, and limitations.

## Primary hypothesis

Lucid dream reports contain reproducible markers of reflective awareness and intentional agency that generalize across dreamers.

## Evaluation split requirement

Whenever user/dreamer IDs are available, all reports from the same person must remain in a single partition. The current baseline uses an author-disjoint train/test split in that case. If no author identifier is present, the code falls back to stratified report-level splitting, which is not author-disjoint and must be reported as such.

## Secondary hypotheses

H1. Awareness and control are correlated but separable.

H2. An author-disjoint classifier will perform above chance after explicit lucidity vocabulary is removed.

H3. Lucid reports will contain more metacognitive and waking-memory references than matched non-lucid reports.

H4. Model performance will fall materially under author-disjoint splitting relative to random report splitting, demonstrating the importance of controlling author leakage.

## Experiment 1: lexical baseline

- Unit: dream report.
- Outcome: lucid vs non-lucid label.
- Model: TF-IDF 1–2 grams + class-balanced logistic regression.
- Primary evaluation: author-disjoint test split.
- Metrics: macro-F1, AUROC, accuracy, calibration.

## Experiment 2: explicit-vocabulary ablation

Remove terms that trivially reveal self-labeling, including variants of:

- lucid
- lucidity
- dream / dreaming
- realized / realised
- reality check
- woke / awaken

Retrain and compare performance.

## Experiment 3: length matching

Match lucid and non-lucid reports by token-count quantiles to test whether verbosity is a confound.

## Experiment 4: phenomenology

Create a stratified human-annotated subset and score the six rubric dimensions. Estimate inter-rater agreement before training automated annotators.

## Statistical analysis

- bootstrap 95% confidence intervals over held-out dreamers;
- effect sizes for dimension differences;
- permutation test for classifier performance where appropriate;
- false-discovery-rate correction for exploratory feature analyses.

## Major threats to validity

1. Self-selected forum population.
2. User-generated dream labels.
3. Retrospective reporting bias.
4. Different writing styles between habitual lucid and non-lucid dreamers.
5. Explicit lucidity vocabulary causing label leakage.
6. Multiple reports per user violating independence if not grouped.

## What would constitute an interesting negative result?

If performance collapses after author grouping and vocabulary ablation, that would suggest much of the apparent computational signal in lucid-report classification arises from author identity or explicit reporting conventions rather than deeper phenomenological language. That is scientifically useful.
