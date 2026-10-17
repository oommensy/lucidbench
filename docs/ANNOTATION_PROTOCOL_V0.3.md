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

The fixed sampling seed is 303; the pilot seed is 30303. The configured full sample contains 150 mixed-label author pairs and 300 independently sampled reports; the pilot contains 30 mixed-label author pairs and 60 independently sampled reports. The sampling procedure:

1. Re-detect cross-author exact and near duplicates using the v0.2 audit's method (case-folded/whitespace-normalized exact matching; approximate 5-token-shingle SimHash/LSH candidates verified at Jaccard >= 0.85). Exclude every report involved in a detected cross-author pair. Near-duplicate discovery is approximate, not exhaustive.
2. Retain eligible authors with reports in both label strata for a paired component. For each selected mixed-label author, select exactly one lucid and one nonlucid report; when an author has multiple candidates in either stratum, choose the pair with the closest word counts, breaking ties with the fixed seed.
3. Select 150 mixed-label authors for the 600-report sample (150 reports of each label). Select the remaining 150 reports per label independently from label-exclusive authors, never reusing a paired-component author. Enforce the configured maximum of two reports per author across the complete sample.
4. Select the pilot as a subset of the main sample: 30 of the paired authors contribute their full lucid/nonlucid pair (60 reports), and 30 additional reports per label come from the independently sampled component (60 reports). The pilot therefore has exactly 60 lucid and 60 nonlucid reports, and all 120 are double-coded.
5. Balance pair mean length and independent-report length across five quantile strata where feasible. Pair selection minimizes the within-author word-count difference; overall label-level length balance is approximate and reported in the aggregate sample summary.
6. Selection is deterministic for the configured seeds. If the configured paired-author target, class totals, or author cap are infeasible after duplicate exclusion, sampling fails with an error; do not silently reduce the number of pairs or relax the cap.
7. Assign opaque `LB03-####` annotation IDs. The ignored local manifest records reconstruction fields (row index, source report ID, text SHA-256, corpus label, word count, length stratum, pilot flag), plus opaque local-only `author_cluster_id`, `paired_author_id`, and `sampling_component` fields. The identifiers are not author identities and must not be copied into tracked files. The manifest does not contain dream text.

Generated files go under ignored `data/processed/v0.3/`: `sample_manifest.csv`, a blank 600-row `annotation_template.csv`, a 240-row pilot template with one row per pilot report per annotator, blinded local report packets, and aggregate `sample_summary.json`. Annotator-facing files contain only opaque report IDs, dream text (in packets), annotator IDs, and annotation fields; they omit source labels, author/cluster IDs, and all pairing/component indicators. The report text itself may contain clues, but the sampling process does not disclose its corpus metadata or pairing. The local packet files contain source dream text and must never be committed or uploaded to public storage. The tracked template at `docs/templates/v0.3_annotation_template.csv` contains only the header.

No author identities or row-level identifiers are exported in the aggregate summary. A non-identifying aggregate copy is tracked at `results/v0.3/sample_summary.json`. The local sample manifest is a sensitive linkage file; restrict access and delete it/packets when no longer needed, subject to project data-retention practice.

## Annotation procedure

### Local terminal helper

After generating the blinded pilot packet and annotator-specific template, Annotator 1 can start the human-only local workflow with:

```bash
python -m scripts.annotate_pilot --annotator annotator_1
```

The helper reads only `annotator_1_pilot_packet.csv` and `annotator_1_annotation_template.csv`; it does not access the sample manifest or linkage metadata. It displays the opaque report ID, report text, and brief rubric reminders, and does not infer or suggest scores. Accepted field values are autosaved to the ignored local file `data/processed/v0.3/annotator_1_annotations.csv`. On restart, it begins at the first report with any incomplete score. At a prompt, enter `next`, `previous`, `save`, `quit`, `rubric`, or `progress`; `quit` saves before exit. To check completion without opening reports, run:

```bash
python -m scripts.annotate_pilot --annotator annotator_1 --check
```

The check reports completion counts, missing score fields, duplicate opaque report IDs, and schema-validation status. Notes are optional short paraphrases only; do not copy report text or add identifying details. The helper is a human annotation interface only and does not calculate inter-rater agreement.

1. Two independent annotators receive separately shuffled pilot packets with the same 120 report texts and opaque IDs. They work independently and do not see each other's ratings, corpus labels, author identities, or which reports are paired.
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
2. **Within-author paired comparisons:** compare each selected mixed-label author's lucid and nonlucid ratings using paired differences or an appropriate paired ordinal/binary model. Report the number of complete author pairs and paired effect estimates with uncertainty. This component is particularly useful for reducing person-specific writing-style and reporting confounds because each author serves as their own comparison. It does not remove time-varying context or corpus-label selection bias and is not a causal design.
3. **Between-author comparisons:** analyze the independent component separately, comparing reports from label-exclusive authors. Account for any multiple reports from one author with author-clustered uncertainty or a justified mixed-effects model; report effect sizes and uncertainty. This estimates label differences between different authors and is not interchangeable with the within-author estimate.
4. **Overall author-clustered analyses:** use the full sample only with author as the clustering/resampling unit (or a justified mixed-effects structure), so the two reports in a pair are not treated as independent observations. Report the paired and independent component contributions, consider a component-by-label interaction, and show component-specific estimates alongside pooled estimates. Use author-level bootstrap resampling so all reports from each author move together.
5. Summarize ordinal associations with Spearman correlations and author-clustered bootstrap uncertainty. Report the awareness-control relationship as a dimensional association, not a binary proxy for lucidity.
6. Test H3 by comparing the association of awareness with metacognition against its association with sensory richness using a pre-specified sensory-count summary; avoid comparing unstandardized coefficients across incompatible scales.
7. For H2, report the frequency/proportion and author-clustered uncertainty of high-awareness/low-control reports; describe the joint distribution rather than selecting illustrative reports post hoc.
8. Treat H5 and all unregistered analyses as exploratory; control multiplicity for families of exploratory tests and label them clearly.
9. Consider latent structure/factor analysis only if pilot reliability is adequate, observed category frequencies are sufficient, and the effective sample size (authors, not just reports) supports the model. With 600 reports and repeated measures, adequacy is not assumed in advance. Do not force a one-factor “lucidity” score.
10. Keep the sampling component, strata, and paired-data availability visible in reporting; do not treat the 600 reports as 600 independent authors.

No significance tests are currently run because no human annotations exist.

## Ethics, licensing, and privacy

This study uses only existing public corpus material. Do not collect new dream reports, recruit participants, perform sleep manipulation, or acquire EEG/physiological data under this protocol. A future prospective study involving participants or new data collection must receive appropriate institutional ethics/IRB review before recruitment or data collection.

The dataset is CC BY-NC 4.0. Keep source text and the ID-to-text linkage local, control access to annotation packets, and do not commit dream text, private notes, completed annotation files, or row-level sample manifests. The project does not diagnose sleep or mental-health conditions.
