# v0.3 Phenomenology Annotation Study Protocol

## Status and purpose

This is a prospective protocol for coding an already-public dream-report corpus. The sample and blank templates may be created locally, but no human annotations or agreement statistics are included yet. Do not run the preregistered hypothesis tests until annotations are complete and the pilot rubric has been reviewed.

The scientific goal is to test whether awareness/lucidity, agency, control, metacognition, waking-memory access, sensory richness, emotion, and stability can be coded as distinct dimensions. In particular, **lucidity is not assumed to be equivalent to agency or control**.

## Source and eligibility

- Source: Remington Mallett, *A large corpus of lucid and non-lucid dream reports*, Zenodo 10.5281/zenodo.19161757, CC BY-NC 4.0.
- Use only the already-public corpus. Do not solicit or collect new dream reports, physiological data, or experimental observations in this study.
- Binary sample strata are corpus labels `lucid` and `nonlucid`. Exclude `unspecified`, `ambiguous`, missing text, and missing author ID.
- Do not show annotators corpus labels, categories, tags, author identity, or report metadata. Annotators receive an opaque `report_id` and report text only.

## Reproducible sample selection

Run:

```bash
python -m scripts.create_annotation_sample --config configs/v0.3.yaml
```

The fixed sampling seed is 303; the pilot seed is 30303. The sampling procedure:

1. Re-detect cross-author exact and near duplicates using the v0.2 audit's method (case-folded/whitespace-normalized exact matching; approximate 5-token-shingle SimHash/LSH candidates verified at Jaccard >= 0.85). Exclude every report involved in a detected cross-author pair. Near-duplicate discovery is approximate, not exhaustive.
2. Exclude authors with eligible reports in both label strata. Draw only from label-exclusive authors, ensuring that no author is sampled into both lucid and nonlucid groups.
3. Compute five quantile length strata from eligible reports. Within each corpus label, select 300 reports with approximately equal representation across strata and a hard maximum of two reports per author. Selection is deterministic for the configured seed; if the requested counts/cap are infeasible, fail rather than silently relax them.
4. Select the pilot from the 600-report main sample: 60 lucid and 60 nonlucid reports, approximately balanced across the same length strata, with the same two-report-per-author cap. Thus, all 120 pilot reports are in the main sample and are double-coded.
5. Assign opaque `LB03-####` annotation IDs. The local manifest records reconstruction fields (row index, source report ID, text SHA-256, corpus label, word count, stratum, and pilot flag). It does not contain dream text.

Generated files go under ignored `data/processed/v0.3/`: `sample_manifest.csv`, a blank 600-row `annotation_template.csv`, a 240-row pilot template with one row per pilot report per annotator, blinded local report packets, and aggregate `sample_summary.json`. The local packet files contain source dream text and must never be committed or uploaded to public storage. The tracked template at `docs/templates/v0.3_annotation_template.csv` contains only the header.

No authors' identities are exported in summary files. The local sample manifest is a sensitive linkage file; restrict access and delete it/packets when no longer needed, subject to project data-retention practice.

## Annotation procedure

1. Two independent annotators receive separately shuffled pilot packets with the same 120 report texts and opaque IDs. They work independently and do not see each other's ratings or the corpus labels.
2. Annotate every rubric field in `docs/PHENOMENOLOGY_RUBRIC.md`. Leave a field blank only when it is genuinely uninterpretable; explain ambiguity in notes without copying source text.
3. Do not put dream text, direct quotations, author information, or other identifying details into notes. The generated CSVs are local and ignored; do not add completed ratings to Git.
4. After both pilot files are complete, combine rows with their assigned `annotator_id`, validate, and compute agreement:

   ```bash
   python -m scripts.validate_annotations \
     --annotations data/processed/v0.3/pilot_annotations_combined.csv \
     --require-complete \
     --agreement-output data/processed/v0.3/pilot_agreement.json
   ```

5. Review disagreements by construct, refine the rubric before the remaining 480 reports, and preserve the pilot scores/rubric version. If a rubric change alters an operational definition, re-annotate affected pilot items before combining pilot and main-study labels.
6. Do not calculate inferential hypothesis tests or describe human-validated findings until annotation data exist and the pilot review decision is recorded.

## Reliability outputs and practical guidance

The validator calculates, only from observed paired ratings:

- quadratic-weighted Cohen's kappa for ordinal dimensions and confidence;
- unweighted Cohen's kappa for binary sensory indicators and nominal emotional valence;
- Krippendorff's alpha using squared ordinal distance for ordinal fields and nominal disagreement for categorical fields; and
- raw percent agreement and paired item counts for each field.

Blank values are pairwise omitted per field. The output reports no numerical reliability value when there are no paired scores for a field. No scores are imputed. Kappa/alpha can be undefined when there is no expected disagreement or no rating variation; `null` is retained rather than replaced by zero.

Tentative pilot rubric-review guidance (not universal scientific cutoffs):

- alpha or weighted kappa below 0.60: treat as a substantial operationalization problem; inspect disagreements and revise before scaling;
- 0.60–0.79: provisional agreement; clarify definitions/examples and consider a second calibration round;
- 0.80 or above: encouraging for this pilot, but still inspect construct validity, prevalence effects, and disagreement cases.

These are workflow heuristics, not pass/fail criteria for truth, clinical use, or universal reliability. Report item counts, confidence intervals when later justified, prevalence, and the type of coefficient alongside any estimate.

## Preregistered hypotheses

All hypotheses below are to be tested only after the human annotation dataset and pilot decisions exist.

### Confirmatory

- **H1:** Dream awareness/lucidity and dream control are positively related but empirically separable.
- **H2:** High awareness can occur with low control. Operationally, reports with awareness = 2 and control = 0 or 1 should occur at a nonzero rate; quantify uncertainty and do not infer separability from isolated cases.
- **H3:** Metacognition is more strongly associated with lucid awareness than raw sensory richness.
- **H4:** Waking-memory access is more frequent in lucid than nonlucid reports.

### Exploratory

- **H5:** Increased awareness/control may be associated with reported dream instability or awakening.
- Associations of emotion, valence, agency, individual sensory modalities, and confidence with awareness/control.
- Any post-hoc subgroup, lexical, demographic, or temporal analyses.

## Statistical analysis plan

The pilot's primary role is rubric feasibility and inter-rater reliability, not hypothesis testing.

After rubric refinement and completion:

1. Report class-stratified dimension distributions, missingness, sensory-count distributions, and contingency tables for awareness × control, awareness × agency, and awareness × metacognition.
2. Summarize ordinal associations with Spearman correlations and author-clustered bootstrap uncertainty. Report the awareness-control relationship as a dimensional association, not a binary proxy for lucidity.
3. Compare lucid vs nonlucid strata for each preregistered dimension using ordinal models for ordinal scores and logistic models for binary sensory indicators. Account for repeated reports with author-clustered standard errors or a justified mixed-effects model; report effect sizes and uncertainty.
4. Test H3 by comparing the association of awareness with metacognition against its association with sensory richness using a pre-specified sensory-count summary; avoid comparing unstandardized coefficients across incompatible scales.
5. For H2, report the frequency/proportion and author-clustered uncertainty of high-awareness/low-control reports; describe the joint distribution rather than selecting illustrative reports post hoc.
6. Treat H5 and all unregistered analyses as exploratory; control multiplicity for families of exploratory tests and label them clearly.
7. Consider latent structure/factor analysis only if pilot reliability is adequate, observed category frequencies are sufficient, and the effective sample size (authors, not just reports) supports the model. With 600 reports and repeated measures, adequacy is not assumed in advance. Do not force a one-factor “lucidity” score.
8. Use author-level bootstrap resampling for intervals so all reports from each sampled author move together. Keep sampling weights/strata visible in reporting.

No significance tests are currently run because no human annotations exist.

## Ethics, licensing, and privacy

This study uses only existing public corpus material. Do not collect new dream reports, recruit participants, perform sleep manipulation, or acquire EEG/physiological data under this protocol. A future prospective study involving participants or new data collection must receive appropriate institutional ethics/IRB review before recruitment or data collection.

The dataset is CC BY-NC 4.0. Keep source text and the ID-to-text linkage local, control access to annotation packets, and do not commit dream text, private notes, completed annotation files, or row-level sample manifests. The project does not diagnose sleep or mental-health conditions.
