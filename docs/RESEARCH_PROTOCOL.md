# Research Protocol — v0.1 hypotheses and v0.2 experiment

The v0.1 sections below document hypotheses and planned analyses. The v0.2 section records the procedure actually executed. Neither the hypotheses nor the initial baseline constitute validated findings.

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
