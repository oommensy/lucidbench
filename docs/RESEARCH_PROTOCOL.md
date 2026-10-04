# Research Protocol — v0.1 hypotheses and v0.2 experiment

The v0.1 sections below document hypotheses and planned analyses. The v0.2 section records the procedure actually executed. Neither the hypotheses nor the initial baseline constitute validated findings.

## Primary hypothesis

Lucid dream reports contain reproducible markers of reflective awareness and intentional agency that generalize across dreamers.

## Evaluation split requirement

All main LucidBench experiments require user/dreamer IDs and keep every person's reports in one partition. The v0.2 primary experiment now fails if an author column is unavailable or any eligible report has missing author identity. Exploratory report-level splitting requires the explicit `--allow-report-level-split` option and is not author-disjoint.

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

## v0.2 executed corpus baseline

### Dataset and ingestion

- Source: Remington Mallett, *A large corpus of lucid and non-lucid dream reports*, Zenodo record [10.5281/zenodo.19161757](https://doi.org/10.5281/zenodo.19161757), record version 1, CC BY-NC 4.0.
- File: `dreamviews-posts.tsv`, 110,180,933 bytes, MD5 `d78a499937e063add677a3ed28393d36`.
- Verified schema: 57,778 rows and 11 columns: `post_id`, `user_id`, `nth_post`, `timestamp`, `title`, `tags`, `categories`, `lucidity`, `nightmare`, `wordcount`, and `post_text`.
- Binary labels: 10,231 `lucid` and 27,830 `nonlucid`. Exclude 15,606 `unspecified` and 4,111 `ambiguous` rows from classification. There are no missing `user_id`, `post_text`, or `lucidity` values in the downloaded file; `title` is missing once and `tags` are missing in 29,453 rows.
- The full corpus has 5,036 unique authors; 3,689 have binary-labeled reports. Word counts are computed as whitespace-separated tokens in `post_text`.
- Source data remain local and ignored by Git. Aggregate outputs and author-supported unigram coefficient tables are attributed to the source and marked CC BY-NC 4.0 in `results/v0.2/README.md`.

### Baseline and evaluation

- Configuration: `configs/v0.2.yaml`; fixed seed 42.
- Unit: one corpus row/report; target is `lucid` (positive) versus `nonlucid`.
- Partition: one `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` by `user_id`, yielding 30,710 training reports and 7,351 test reports. The 738 held-out authors do not occur in training. No report-level random split is used.
- Features/model: scikit-learn `TfidfVectorizer` with lowercasing, sublinear term frequency, 1–2 grams, `min_df=3`, and at most 50,000 features; `LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=42)`.
- Metrics: accuracy, balanced accuracy, positive-class precision/recall/F1, ROC-AUC, and a confusion matrix whose row/column order is nonlucid then lucid. ROC-AUC is omitted if a sample contains only one class.
- Uncertainty: 1,000 percentile bootstrap resamples of the held-out authors, sampling author IDs with replacement and retaining all reports for each sampled author. The reported 95% intervals describe held-out-author resampling variability on this test split; they do not capture corpus, model, or external-validation uncertainty.

### Lexical-cue ablation

The same split and model settings are used, refitting on text after case-insensitive whole-word replacement of the following explicit terms:

`lucid`, `lucidity`, `lucidly`, `dream`, `dreams`, `dreaming`, `dreamed`, `dreamt`, `realize`, `realizes`, `realized`, `realizing`, `realise`, `realises`, `realised`, `realising`, `realisation`, `realisations`, `realization`, `realizations`, `wake`, `wakes`, `waking`, `woke`, `woken`, `awaken`, `awakens`, `awakened`, `awakening`, `reality`, `realities`, `real`, `ld`, `dild`, and `rc`.

The last three terms remove common lucid-dream/reality-check abbreviations (`LD`, `DILD`, `RC`). `real` is removed only as an exact token, as requested; broader variants such as `really` and `realistic`, as well as unlisted `dreamer`, remain. This is a transparent sensitivity analysis, not a claim that these are the only possible leakage cues.

### Length-matched sensitivity

First make the author-disjoint split. Derive five word-count bins from training-set length quantiles, then independently within train and test and each bin, sample without replacement an equal number from each label, limited by the less frequent class in that bin. Fit a fresh baseline model on matched training reports and evaluate it on matched test reports. The test set contains 1,787 reports per class (3,574 total); training retains 16,888 reports. This is coarse quantile-bin matching rather than exact word-count matching. The sensitivity metrics have a changed class prevalence and smaller test population, so should not be compared directly to the full-test metrics as though they used the same evaluation population.

### Coefficient inspection and outputs

For baseline and lexical-ablation models, export up to 25 highest and 25 lowest logistic-regression coefficients among single-word features with support in at least ten training authors. Bigram coefficients and report-level predictions are not exported. A positive coefficient is associated with the lucid class conditional on the fitted model and feature set; it is not a causal explanation or a validated marker.

Generated files in `results/v0.2/`:

- `dataset_summary.json` and `dataset_summary.csv`: record metadata, label counts, authors, missingness, and report-length/reports-per-author distributions.
- `metrics.json` and `metrics.csv`: experiment metrics, author-bootstrap intervals, split metadata, and matching sample sizes.
- `confusion_matrices.csv`: all three experiment confusion matrices.
- `length_matching.csv`: available and retained counts by class and quantile bin for train/test.
- `coefficients_baseline.csv` and `coefficients_lexical_ablation.csv`: filtered model-association tables.
- `performance_comparison.png`: F1 and 95% held-out-author-bootstrap intervals.

### Observed v0.2 results and limitations

| Experiment | Accuracy | Balanced accuracy | Precision | Recall | F1 (95% CI) | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.930 | 0.915 | 0.834 | 0.888 | 0.860 (0.824–0.890) | 0.972 |
| Lexical ablation | 0.908 | 0.888 | 0.790 | 0.849 | 0.819 (0.775–0.853) | 0.957 |
| Length-matched sensitivity | 0.909 | 0.909 | 0.924 | 0.893 | 0.908 (0.882–0.928) | 0.969 |

This is one fixed split of a self-selected DreamViews forum corpus, not a repeated-split study or independent laboratory validation. Nearly 34% of rows are omitted from the binary target because labels are `unspecified` or `ambiguous`; findings therefore apply only to the selected labeled subset. Reports per author are highly skewed (median 2, maximum 1,000), which motivates author-disjoint splitting and cluster bootstrap but may still affect the report-weighted metrics. High performance after cue removal could reflect residual lexical conventions, label construction, forum-specific language, or other dataset structure; it is not evidence by itself of generalizable dream phenomenology. The length-matched test sample has equal class counts, changing its prevalence and precision context. Human annotation, repeated author-group evaluation, label-sensitivity checks, and external validation remain future work.

## v0.2.1 robustness pass

The following robustness analyses were added on the same branch while the v0.2 PR remains open and unmerged. The original v0.2 result files are preserved. Run with:

```bash
python -m lucidbench.robustness --config configs/v0.2-robustness.yaml
```

### Repeated author-disjoint evaluations

- Run 30 `GroupShuffleSplit(n_splits=1, test_size=0.20)` partitions with predetermined seeds 1000 through 1029.
- Split by `user_id` before fitting, and assert train/test author sets are disjoint.
- For every split, fit the configured TF-IDF + class-weighted logistic regression baseline and the identical model after configured lexical-cue removal.
- Compute positive-class F1, balanced accuracy, and ROC-AUC on each held-out report set.
- Report the 30-split mean, median, sample standard deviation, empirical 2.5th and 97.5th percentiles, minimum, and maximum. These are descriptive split-to-split distributions, not confidence intervals.

| Experiment | Metric | Mean | Median | SD | P2.5–P97.5 | Min–max |
|---|---|---:|---:|---:|---:|---:|
| Baseline | F1 | 0.846 | 0.848 | 0.024 | 0.806–0.882 | 0.768–0.897 |
| Baseline | Balanced accuracy | 0.895 | 0.897 | 0.017 | 0.861–0.922 | 0.836–0.927 |
| Baseline | ROC-AUC | 0.954 | 0.956 | 0.014 | 0.923–0.970 | 0.906–0.972 |
| Lexical ablation | F1 | 0.805 | 0.805 | 0.026 | 0.761–0.844 | 0.723–0.858 |
| Lexical ablation | Balanced accuracy | 0.867 | 0.869 | 0.018 | 0.830–0.895 | 0.808–0.897 |
| Lexical ablation | ROC-AUC | 0.936 | 0.938 | 0.015 | 0.903–0.955 | 0.886–0.957 |

### Equal-author-weighted held-out metrics

Use the original seed-42 author-disjoint test set. Give each report weight `1 / number of held-out reports for its author`, so every author's reports sum to total weight one. Calculate confusion-derived accuracy, precision, recall, F1 and balanced accuracy using weighted confusion counts, and calculate weighted ROC-AUC with the same weights. This is not a mean of per-author F1 scores; it preserves one report-level prediction per report while reducing prolific-author influence.

| Experiment | Accuracy | Balanced accuracy | Precision | Recall | F1 | Weighted ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.876 | 0.864 | 0.843 | 0.815 | 0.829 | 0.936 |
| Lexical ablation | 0.851 | 0.838 | 0.800 | 0.791 | 0.796 | 0.919 |

The drop from report-weighted F1 0.860 to author-weighted F1 0.829 confirms that prolific authors materially affect the aggregate; it does not remove all author-related or corpus-selection effects.

### Ten-report author cap

For seeds 2000 through 2009, randomly sample at most ten eligible reports per author without replacement. Then perform a seed-matched 80/20 GroupShuffleSplit by author and fit/evaluate both baseline and lexical ablation. The cap reduces the analytic data to 13,912 reports per seed; exact counts and per-seed scores are in `capped_author_metrics.csv`.

| Experiment | Metric | Mean | Median | SD | P2.5–P97.5 | Min–max |
|---|---|---:|---:|---:|---:|---:|
| Baseline | F1 | 0.838 | 0.837 | 0.011 | 0.824–0.857 | 0.823–0.860 |
| Baseline | Balanced accuracy | 0.883 | 0.883 | 0.007 | 0.872–0.894 | 0.871–0.895 |
| Baseline | ROC-AUC | 0.946 | 0.948 | 0.005 | 0.938–0.953 | 0.938–0.953 |
| Lexical ablation | F1 | 0.803 | 0.801 | 0.013 | 0.784–0.821 | 0.783–0.858 |
| Lexical ablation | Balanced accuracy | 0.858 | 0.860 | 0.008 | 0.847–0.870 | 0.846–0.870 |
| Lexical ablation | ROC-AUC | 0.928 | 0.929 | 0.005 | 0.920–0.934 | 0.920–0.934 |

### Cross-author duplicate audit

- Exact duplicate: case-fold full report text and collapse whitespace; identical normalized reports from different `user_id` values are a pair.
- Near duplicate: tokenize case-insensitively, form unique 5-word shingles, compute a 64-bit SimHash, use four 16-bit bands to retrieve candidate pairs, then verify distinct-author candidates with shingle-set Jaccard similarity at least 0.85.
- The LSH candidate-generation method is approximate and may miss near duplicates. Candidate pairs and report text are not exported.
- Findings: zero cross-author exact pairs and 17 approximate near-duplicate pairs across the 57,778 reports (34 reports involved). Seven pairs intersect binary-eligible reports, involving 19 distinct reports. Remove every binary-eligible report involved in any detected pair and refit/evaluate the same seed-42 author-disjoint baseline: 38,042 reports remain; F1 changes from 0.860 to 0.859 and ROC-AUC from 0.972 to 0.972.

### Corpus-artifact audit and negative control

The audit reports class-conditional structural feature means/prevalence, metadata missingness/cardinality, title and numeric thread-metadata summaries, category/tag lucidity terms, timestamp-year counts, and aggregate counts of literal author-ID mentions in text. It does not emit report text, tag values, or author identifiers. The author-ID mention check looks only for the report's own exact `user_id` token; it can miss alternate username mentions and can false-match common/numeric IDs.

The author-disjoint artifact-only negative control uses 24 numeric features that contain no lexical identity: character/word/sentence counts, newline count, mean word length, case/digit ratios, counts of URLs, HTML tags, quote/signature markers, punctuation, bullets, and repeated punctuation. `forum_abbreviation_count` is measured and reported separately because it is a lexical cue, and is explicitly excluded from the negative-control model. A `StandardScaler` plus class-weighted logistic regression on the same seed-42 author split gives accuracy 0.625, balanced accuracy 0.609, F1 0.429, and ROC-AUC 0.656 (author-weighted F1 0.488, weighted ROC-AUC 0.636). This is above chance and substantially below the TF-IDF results; structural features explain some but not all label discrimination.

Important metadata finding: every labeled nonlucid row's `categories` field contains a nonlucid term; every labeled lucid row's field contains a lucid term. Categories are excluded from the text model, which reads `post_text` only, but this demonstrates a close coupling of forum categorization and target labels. Among nonmissing tags, a lucid term appears in 25.2% of lucid reports' tags and 3.1% of nonlucid reports' tags. Mean HTML-tag counts are 1.72 for lucid and 1.86 for nonlucid reports; URLs, quote markers, and signatures are rare/absent. The number of forum shorthand tokens differs (standardized mean difference 0.603), motivating abbreviation-aware future ablations. These differences show why residual text-model performance cannot yet be equated with phenomenological markers.

### Strict author identity requirement

The v0.2 primary experiment now stops with an explicit error if no author identity column is available or any eligible report has missing identity. A researcher may pass `--allow-report-level-split` to run without complete author IDs; output is explicitly marked `leakage_sensitive: true` and `author_disjoint: false`. This opt-in path is for exploratory reuse only and is not acceptable for LucidBench's main reported experiments.

### Robustness interpretation

F1 and ROC-AUC remain high across repeated splits and after capping reports at ten per author; removal of detected near-duplicate reports barely changes the seed-42 baseline. Equal-author weighting reduces performance, and the structural-only negative control remains above chance. In addition, forum categories closely encode target labels and lucid tags/shorthand differ by class. Together these checks support reproducible text-based distinction of corpus labels, but cannot distinguish dream phenomenology from label/reporting conventions. No validated scientific marker is claimed. Human annotation of awareness, agency, control, metacognition, waking-memory access, sensory richness, and stability remains the next scientifically distinctive phase.
