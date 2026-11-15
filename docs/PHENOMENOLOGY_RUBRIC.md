# LucidBench v0.3.1 Phenomenology and Lucidity-Transition Annotation Rubric

## Purpose and unit

Annotate evidence explicitly present in one dream report. The unit is a single report. Do not infer an experience merely because it would be plausible, and distinguish in-dream cognition from the waking narrator's retrospective commentary.

This revision was motivated by expert feedback and by crosswalking the original LucidBench rubric against established dream/lucidity measures: LuCiD (Voss et al., 2013), DLQ (Stumbrys, Erlacher, & Schredl, 2013), MACE-related work (Kahan et al., 1997), LUSK (Schredl, Rieger, & Göritz, 2018), and the multidimensional LDCQ framework (Peters, Erlacher, & Bonamino).

The key design principle is that **lucid insight, intentional agency, metacognitive processing, and dream control are related but distinct constructs**.

## Evidence convention

For most binary and ordinal fields, a zero means **not evidenced in the narrative**, not proof that the experience was absent. Narrative dream reports can omit cognitive events that occurred. Leave a field blank only when the text is genuinely uninterpretable for that field.

Do not use corpus labels, tags, categories, filenames, author identity, or sampling component to infer a score.

## A. Lucid insight

**Field:** `lucid_insight`

**Question:** To what extent does the dreamer recognize the current experience as a dream while it is happening?

- **0 — not evidenced:** no in-dream recognition or questioning of dream state.
- **1 — questioning/partial:** the dreamer suspects, questions, or tests whether the experience is a dream without clear explicit recognition.
- **2 — explicit insight:** the dreamer explicitly recognizes that the current experience is a dream.

Lucid insight is not the same as control. A report can score 2 here and show no control.

## B. Intentional agency

**Field:** `intentional_agency`

**Question:** Does the dreamer intentionally initiate and pursue actions?

- **0:** intentional choice/action not evidenced.
- **1:** one or more deliberate but brief/reactive actions.
- **2:** sustained goal-directed behavior, planning, or adaptation toward a goal.

Agency concerns initiation and choice, not whether dream content obeys the dreamer.

## C. Metacognitive and reflective processes

### Internal commentary

**Field:** `internal_commentary` — 0/1

Code 1 when the dreamer wonders about, comments internally on, or explicitly evaluates an event while dreaming.

### Self-reflection

**Field:** `self_reflection` — 0/1

Code 1 when the dreamer reflects on their own thoughts, feelings, motives, behavior, or mental condition.

### State reasoning

**Field:** `state_reasoning`

**Question:** Does the dreamer reason about the nature of their current state or its relation to waking reality?

- **0:** not evidenced.
- **1:** limited state-related doubt, checking, or inference.
- **2:** explicit or sustained reasoning about whether/how the present experience is a dream or waking reality.

A dreamer may show internal commentary or self-reflection without achieving lucid insight. This distinction is central to the v0.3.1 transition analysis.

## D. Waking-memory access

### Waking memory

**Field:** `waking_memory`

- **0:** waking-life retrieval not evidenced.
- **1:** vague or uncertain reference to waking-life knowledge.
- **2:** explicit retrieval of a waking fact, episode, identity, or contextual fact during the dream.

### Prospective intention memory

**Field:** `prospective_intention_memory` — 0/1

Code 1 when the dreamer explicitly remembers an intention, task, plan, or instruction formed before sleep or while awake and uses or considers it during the dream.

A familiar person appearing is not sufficient evidence of waking-memory retrieval.

## E. Transition-to-lucidity sequence

These fields capture the pathway into lucidity or a failed/near-lucid transition. Do not force every report into a transition narrative.

### Anomaly present

**Field:** `anomaly_present` — 0/1

Code 1 when the report contains an event, setting, object, bodily state, causal relation, or physical property that is explicitly bizarre, impossible, inconsistent, or markedly unlike waking reality.

### Anomaly noticed

**Field:** `anomaly_noticed` — 0/1

Code 1 only when the dreamer notices or reacts to an anomaly as unusual. An anomaly can be present without being noticed.

### Reality testing

**Field:** `reality_testing` — 0/1

Code 1 for a deliberate test or check aimed at determining whether the current experience is a dream or waking reality.

### Dream rationalization

**Field:** `dream_rationalization` — 0/1

Code 1 when the dreamer notices or reasons about something anomalous but adopts an explanation that preserves the dream as ordinary/real rather than reaching lucid insight.

This field is intended to capture the "near-miss" pattern discussed in the expert consultation: anomaly noticed → reasoning → dream-consistent explanation → continued non-lucidity.

### Route to lucidity

**Field:** `route_to_lucidity`

Choose one:

- `none`: no lucid transition evidenced.
- `reasoning_based`: lucid insight follows identifiable questioning, anomaly appraisal, reality testing, or reasoning.
- `spontaneous_intuitive`: the dreamer reports simply/abruptly knowing it is a dream without an identifiable reasoning sequence.
- `wake_initiated`: awareness appears continuous from wakefulness or sleep onset into the dream.
- `other_unclear`: lucid insight occurs but the route cannot be categorized reliably.

### Transition outcome

**Field:** `transition_outcome`

Choose one:

- `no_transition`: no meaningful movement toward lucid insight is evidenced.
- `near_lucid`: questioning/reasoning toward dream-state recognition occurs but explicit lucid insight is not reached.
- `lucid`: explicit lucid insight occurs.
- `unclear`: the transition cannot be reliably classified from the narrative.

The outcome is an annotation of the reported cognitive sequence, not a replacement for the corpus label.

## F. Dream control

Control is separated into **attempt**, **success**, and **domain** so that failed attempts and different kinds of control are not collapsed into one score.

### Control attempt

**Field:** `control_attempt` — 0/1

Code 1 when the dreamer deliberately attempts to influence the dream state, dream self, dream content, characters, environment, or physical rules.

### Control success

**Field:** `control_success`

- `not_applicable`: no deliberate control attempt is evidenced.
- `failed`: a deliberate attempt clearly fails.
- `partial_uncertain`: partial, inconsistent, or ambiguous success.
- `successful`: the intended alteration is clearly achieved.

### Control domain

**Field:** `control_domain`

Use the dominant domain, or `multiple` when several are clearly present:

- `none`
- `cognition`: maintaining lucidity/focus, prolonging the dream, regaining cognitive control.
- `emotion`: deliberately regulating or changing emotional state.
- `body`: deliberately changing the form or sensory properties of the dream body.
- `movement`: deliberately controlling or executing movement/action.
- `characters`: summoning, removing, changing, or deliberately influencing dream characters.
- `environment`: changing locations, objects, lighting, weather, or surroundings.
- `physics`: changing gravity, time, solidity, spatial rules, or other physical constraints.
- `multiple`

This domain structure is adapted at a coarse level from the LDCQ framework; LucidBench does not claim that narrative annotation reproduces LDCQ questionnaire scores.

## G. Sensory evidence

Binary fields:

- `sensory_visual`
- `sensory_auditory`
- `sensory_tactile`
- `sensory_proprioceptive`
- `sensory_smell_taste`

Use **1** only when the corresponding sensory content is described; otherwise use **0 = not evidenced**. Do not infer taste merely from eating or proprioception merely from a verb of movement unless bodily motion/position is actually described.

A derived `sensory_count` may be calculated later; it is not manually annotated.

## H. Emotion

### Emotional intensity

**Field:** `emotional_intensity`

- **0:** no/minimal emotion evidenced.
- **1:** mild or moderate emotion.
- **2:** strong/intense emotion.

### Emotional valence

**Field:** `emotional_valence`

- `negative`
- `neutral_mixed`
- `positive`

Intensity and valence are independent.

## I. Stability and termination

### Dream instability

**Field:** `dream_instability`

- **0:** fading/instability not evidenced or report explicitly remains stable.
- **1:** threatened/limited instability, fading, blurring, or difficulty maintaining the scene.
- **2:** clear major destabilization or scene collapse.

### Loss of lucidity

**Field:** `loss_of_lucidity` — 0/1

Code 1 only when the report explicitly indicates a transition from lucid awareness back to non-lucid dreaming.

### Awakening

**Field:** `awakening` — 0/1

Code 1 when waking from the dream is explicitly reported. Do not infer awakening from the end of the written report.

These fields are intentionally separate: vividness, stability, loss of lucidity, and awakening need not covary.

## J. Confidence and notes

### Confidence

- `low`: sparse or ambiguous evidence.
- `medium`: reasonable coding with meaningful alternatives.
- `high`: explicit and unambiguous evidence.

### Notes

Optional, ≤300 characters. Use a short paraphrase only. Do not copy report text, corpus metadata, or identifying details.

## Dependency rules

- If `control_attempt = 0`, use `control_success = not_applicable` and `control_domain = none`.
- If `control_attempt = 1`, do not use `not_applicable` or `none`.
- Do not infer `anomaly_noticed` from `anomaly_present`.
- Do not infer `lucid_insight` from control, flying, bizarre physics, or use of lucid-dream terminology in waking commentary.
- Do not infer causal links between insight/control and awakening merely from temporal order.

## Literature anchors

- Kahan, T. L., LaBerge, S., Levitan, L., & Zimbardo, P. (1997). Similarities and differences between dreaming and waking cognition: An exploratory study. *Consciousness and Cognition, 6*, 132–147.
- Stumbrys, T., Erlacher, D., & Schredl, M. (2013). Testing the involvement of the prefrontal cortex in lucid dreaming: A tDCS study. *Consciousness and Cognition, 22*, 1214–1222. Includes the DLQ and MACE-based measures used here as conceptual anchors.
- Voss, U., Schermelleh-Engel, K., Windt, J., Frenzel, C., & Hobson, A. (2013). Measuring consciousness in dreams: The Lucidity and Consciousness in Dreams scale. *Consciousness and Cognition, 22*, 8–21.
- Schredl, M., Rieger, J., & Göritz, A. S. (2018). Measuring lucid dreaming skills: A new questionnaire (LUSK). *International Journal of Dream Research, 11*(1).
- Peters, E., Erlacher, D., & Bonamino, C. Lucid Dream Control Questionnaire (LDCQ): Development and Validation, manuscript consulted for the control-domain crosswalk.

The rubric is a research operationalization for retrospective narrative reports. It is not a clinical instrument and does not claim equivalence to participant-completed questionnaires.
