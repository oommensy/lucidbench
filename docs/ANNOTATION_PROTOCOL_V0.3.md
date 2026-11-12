# v0.3.1 Phenomenology and Lucidity-Transition Annotation Study Protocol

## Status and purpose

This is a prospective protocol for coding an already-public dream-report corpus. The reproducible sample and blinding design were created in v0.3. The **v0.3.1 revision changes the annotation constructs, not the sample composition**.

No human annotations or agreement statistics are committed. Do not run confirmatory analyses until both pilot annotators complete the revised schema and the rubric review decision is recorded.

The scientific goals are:

1. test whether lucid insight, intentional agency, state reasoning, waking-memory access, dream control, sensory experience, emotion, and stability can be coded as separable dimensions from narrative dream reports;
2. test whether dream control can be represented more precisely as attempted control, success/failure, and domain;
3. explore whether reports contain reproducible **transition-to-lucidity sequences**, especially the contrast between reasoning that culminates in lucid insight and reasoning that is rationalized without lucidity.

Lucidity is not assumed to be equivalent to control, agency, or metacognition.

## Conceptual grounding

The v0.3.1 construct definitions were crosswalked against:

- LuCiD: INSIGHT, CONTROL, THOUGHT, REALISM, MEMORY, DISSOCIATION, positive and negative emotion;
- DLQ: awareness, control, waking remembrance and remembered intentions;
- MACE-related work: choice, internal commentary, attention, and self-reflection;
- LUSK: lucid-dream skill, awareness maintenance, control, waking memory and loss/termination problems;
- LDCQ: multidimensional control over cognition, emotion, body, movement, characters, environment and physics.

See `docs/V0.3_CONSTRUCT_CROSSWALK.md`.

## Source and eligibility

- Source: Remington Mallett, *A large corpus of lucid and non-lucid dream reports*, Zenodo 10.5281/zenodo.19161757, CC BY-NC 4.0.
- Use only the already-public corpus.
- Binary sample strata are corpus labels `lucid` and `nonlucid`; exclude `unspecified`, `ambiguous`, missing text, and missing author ID.
- Annotators must not see source labels, categories, tags, author identity, pair membership, or sampling component.

The corpus label is a sampling variable and later comparison variable. It is **not** an annotation target and must not be used to infer `lucid_insight` or `transition_outcome`.

## Reproducible sample selection

Run:

```bash
python -m scripts.create_annotation_sample --config configs/v0.3.yaml
```

The sampling design remains unchanged from v0.3:

- seed 303; pilot seed 30303;
- 600 reports total: 300 corpus-labeled lucid and 300 corpus-labeled nonlucid;
- 150 mixed-label authors contribute one report from each label stratum (300 paired-component reports);
- 300 reports come from label-exclusive authors (150 per label);
- maximum two reports per author;
- 120-report pilot: 30 mixed-label author pairs (60 reports) plus 60 independent reports (30 per label);
- all 120 pilot reports are independently double-coded;
- report length is approximately balanced and mixed-label pairs are chosen to minimize within-pair length difference;
- detected cross-author exact/near duplicates are excluded using the v0.2 audit method;
- opaque `LB03-####` report IDs are used in annotator-facing files.

The pilot remains a subset of the full sample. If the requested paired counts, class totals, or author cap are infeasible after exclusions, sampling must fail rather than silently relax constraints.

## Blinding and local files

Generated material is stored under ignored `data/processed/v0.3/`.

Annotator packets contain only:

- `report_id`
- `dream_text`

Per-annotator templates contain only opaque report IDs, annotator IDs, and annotation fields. They omit labels, author identity, pairing/component information, tags, categories and source metadata.

The report text itself may reveal lucid-dream language. This cannot be completely blinded and is a limitation.

## Schema revision and migration rule

The v0.3.1 schema supersedes the original broad fields `awareness`, `agency`, `control`, `metacognition`, `waking_memory`, and `stability`.

Because the operational definitions changed, **do not silently migrate any human ratings made under the older rubric**. Preserve old local files if needed for provenance, regenerate annotator templates, and annotate the revised fields from the report text.

The annotation helper intentionally refuses old-schema saved files.

## Annotation layers

### Whole-dream phenomenology

- `lucid_insight`
- `intentional_agency`
- `internal_commentary`
- `self_reflection`
- `state_reasoning`
- `waking_memory`
- `prospective_intention_memory`
- `control_attempt`
- `control_success`
- `control_domain`
- sensory modality indicators
- `emotional_intensity`
- `emotional_valence`
- `dream_instability`
- `loss_of_lucidity`
- `awakening`

### Transition-to-lucidity layer

- `anomaly_present`
- `anomaly_noticed`
- `reality_testing`
- `dream_rationalization`
- `route_to_lucidity`
- `transition_outcome`

The transition layer is exploratory. It is motivated by established constructs but the exact sequence is a LucidBench operationalization, not a previously validated questionnaire.

## Narrative evidence limitation

Retrospective dream reports are incomplete descriptions of dream experience. A zero generally means **not evidenced in the report**, not proven absence.

This is especially important for metacognitive events, memory access and internal reasoning. All conclusions must be phrased as properties of reported evidence unless independently validated with direct questioning or laboratory data.

## Human annotation procedure

Annotator 1:

```bash
python -m scripts.annotate_pilot --annotator annotator_1
```

Annotator 2 uses the same command with `annotator_2`.

Before starting after this revision, regenerate the local annotation sample/templates. If an old `annotator_*_annotations.csv` exists, archive or remove it locally before starting the revised pilot.

The helper:

- reads only the blinded packet and matching template;
- displays no source label, author identity, pair status or sampling component;
- suggests no scores;
- autosaves accepted entries;
- resumes from the first incomplete row;
- rejects schema-inconsistent control combinations.

Check completion:

```bash
python -m scripts.annotate_pilot --annotator annotator_1 --check
```

After both annotators complete the pilot, combine their rows and validate:

```bash
python -m scripts.validate_annotations \
  --annotations data/processed/v0.3/pilot_annotations_combined.csv \
  --require-complete \
  --agreement-output data/processed/v0.3/pilot_agreement.json
```

## Reliability analysis

Calculate agreement only from observed paired ratings.

- quadratic-weighted Cohen's kappa and squared-ordinal Krippendorff alpha for ordinal fields and confidence;
- unweighted Cohen's kappa and nominal Krippendorff alpha for binary and nominal fields;
- raw percent agreement and paired counts for every field;
- retain undefined reliability as null when there is no variation or no paired evidence.

Pilot review heuristics remain workflow guidance rather than truth criteria:

- <0.60: substantial operationalization concern;
- 0.60–0.79: provisional; inspect and clarify;
- ≥0.80: encouraging, while still examining prevalence and disagreement cases.

For sparse transition events, raw agreement and prevalence must be reported alongside kappa/alpha because chance-corrected coefficients may be unstable.

## Hypotheses

### Confirmatory construct hypotheses

- **H1:** `lucid_insight` and control are positively related but empirically separable.
- **H2:** Explicit lucid insight can occur without successful dream control.
- **H3:** State reasoning is more strongly associated with lucid insight than raw sensory modality count.
- **H4:** Explicit waking-memory access is more frequently evidenced in corpus-labeled lucid than nonlucid reports.

### Exploratory transition hypotheses

- **T1:** Some reports will show anomaly noticing and state reasoning without explicit lucid insight (`near_lucid` outcome).
- **T2:** Among reports with anomaly noticing and state reasoning, reasoning-based lucid transitions and near-lucid rationalizations will show distinguishable annotation profiles.
- **T3:** Lucid transitions need not be reasoning-based; spontaneous/intuitive and wake-initiated routes may occur.
- **T4:** Post-insight control success, instability, loss of lucidity and awakening may vary independently.

The pilot's primary purpose is reliability/feasibility, not testing these hypotheses.

## Analysis plan after pilot approval

1. Report prevalence/missingness for all fields and transition outcomes.
2. Report reliability by field before combining annotators.
3. For the paired mixed-label component, compare each author's two reports with paired ordinal/binary/nominal methods appropriate to the field.
4. Analyze the independent component separately.
5. For pooled analyses, cluster/resample at author level so two reports from one author are not treated as independent.
6. Analyze control as attempt → success conditional on attempt, and separately by domain.
7. For transition analyses, describe sequence components and compare:
   - no-transition reports;
   - near-lucid reports;
   - reasoning-based lucid transitions;
   - spontaneous/intuitive lucid transitions;
   - wake-initiated transitions.
8. Treat rare transition categories descriptively if cell counts do not support stable inference.
9. Do not claim that corpus text establishes neural mechanisms, causal induction pathways, or validated phenomenological markers.

## Pilot decision rule

After double-coding the 120 reports:

- inspect disagreements by construct and transition category;
- revise definitions/examples where needed;
- preserve the exact rubric version used for the pilot;
- if a construct definition changes materially, re-annotate affected pilot fields before combining pilot and main-study data;
- only then proceed to the remaining 480 reports.

PR #2 should remain unmerged until the revised pilot schema and tests are reviewed.
