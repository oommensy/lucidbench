# LucidBench v0.3 Phenomenology Annotation Rubric

## Purpose and unit

Annotate evidence explicitly present in one report, not what an event implies in waking life or what an annotator assumes the dreamer must have felt. The unit is a single report. Use the report's words as evidence and add a short, source-free rationale in `notes` (do not copy report text). Score each dimension independently.

**Dream awareness/lucidity, agency, and dream control are distinct constructs.** Awareness means recognizing or questioning the dream state; agency means intentional action; control means deliberately altering the dream. A person can know they are dreaming but have no control, intentionally act without knowing they are dreaming, or alter something without sustained agency.

The short quotations below are synthetic examples written for this rubric; none are copied from the dataset.

## General coding conventions

- Score only what the report supports. Do not infer awareness from bizarre events, control from flying, or waking-memory access from a familiar person appearing.
- Distinguish in-dream experience from the waking narrator's retrospective commentary.
- When evidence is absent, use the dimension's 0 code; absence of description is not proof that the experience could not have occurred.
- If the report is genuinely impossible to interpret for a field, leave that field blank and explain the ambiguity in `notes`; do not invent an “unknown” score.
- Add a brief paraphrased rationale for nonzero or ambiguous scores, without copying source text into annotation notes.
- Review the full report; do not use corpus `lucidity`, `categories`, `tags`, author ID, or filename to infer a phenomenology score.

## A. Dream awareness / lucidity

**Question:** Does the dreamer, during the reported experience, recognize or question whether they are dreaming?

| Score | Definition |
|---|---|
| 0 | No awareness or questioning of dreaming is evidenced. |
| 1 | Possible/questioning awareness, such as testing whether the experience is a dream without reaching a clear conclusion. |
| 2 | Explicit in-dream awareness that the current experience is a dream. |

**Synthetic examples**

- 0: “I crossed a bridge while the storm followed me.”
- 1: “I checked the clock twice because I wondered whether this could be a dream.”
- 2: “While it was happening, I knew I was dreaming.”

**Edge cases:** “It was a dream” in a waking introduction or conclusion is retrospective and scores 0 unless in-dream recognition is also stated. Impossible physics alone is not awareness. A reality check without any indication of questioning or recognition is at most 1. Uncertainty after a reality check is 1, not 2.

## B. Agency

**Question:** Does the dreamer intentionally choose and carry out actions?

| Score | Definition |
|---|---|
| 0 | Events mostly happen to the dreamer; no intentional choice is evidenced. |
| 1 | One or more intentional choices/actions are present, but behavior is brief, reactive, or not sustained toward a goal. |
| 2 | Sustained intentional, goal-directed behavior: the dreamer pursues or adapts actions toward an expressed goal. |

**Synthetic examples**

- 0: “The crowd carried me into the station.”
- 1: “I chose the blue door and opened it.”
- 2: “I decided to find my sister, asked several people for directions, and changed route when the first path was blocked.”

**Edge cases:** Movement or first-person narration alone does not establish intention. Attempted action may show agency even if it fails. Control is not required for agency. A single deliberate act is generally 1, not 2, unless the report supports sustained goal pursuit.

## C. Dream control

**Question:** Does the dreamer deliberately attempt to alter dream content, body, environment, characters, or events?

| Score | Definition |
|---|---|
| 0 | No deliberate alteration of dream content is evidenced. |
| 1 | An alteration is deliberately attempted or partially achieved, but is limited, uncertain, or unsuccessful. |
| 2 | A deliberate alteration is clearly successful. |

**Synthetic examples**

- 0: “I was flying over the city,” with no indication the flight was chosen or changed.
- 1: “I tried to make the room brighter, but it only changed a little.”
- 2: “I decided to open a door onto a beach, and the hallway became a beach.”

**Edge cases:** Accidental or unexplained changes are not control. Flying, unusual strength, or dream transitions alone do not count. Control can occur without lucidity; lucidity can occur without control. Score the attempt when success is unclear (1).

## D. Metacognition

**Question:** Does the dreamer monitor or reason about their own mental state, dream state, perception, or the consequences of an experience?

| Score | Definition |
|---|---|
| 0 | No reflection on the dreamer's mental/dream state or reasoning about it. |
| 1 | Limited reflection or monitoring, such as brief doubt, checking, or noticing one's thought/feeling. |
| 2 | Explicit reasoning about the dream, mental state, perception, or consequences. |

**Synthetic examples**

- 0: “The room filled with water.”
- 1: “I paused to check whether I was remembering this correctly.”
- 2: “I reasoned that the clock changed because the scene was a dream, so I tested the doorway before deciding what to do.”

**Edge cases:** A statement of awareness without any reflection beyond recognition can score awareness 2 and metacognition 1; score separately. Retrospective waking analysis is not in-dream metacognition. A reality check may be metacognition 1 even when awareness remains uncertain.

## E. Waking-memory access

**Question:** Does the dreamer retrieve information, intentions, plans, or memories from waking life during the dream?

| Score | Definition |
|---|---|
| 0 | No waking-life reference or retrieval is evidenced. |
| 1 | Vague waking-life reference, with unclear retrieval or influence on the dream. |
| 2 | Explicit retrieval of waking intentions, facts, plans, or memories while dreaming. |

**Synthetic examples**

- 0: “I met someone who looked familiar,” with no waking-life identification.
- 1: “The place seemed like somewhere I knew from outside the dream.”
- 2: “I remembered my waking plan to call my brother and tried to do it in the dream.”

**Edge cases:** A known person appearing does not itself show memory retrieval. A report that says “I remembered” after waking is not in-dream retrieval. Score the evidence of access, not whether the recalled fact is accurate.

## F. Sensory richness

Score each modality **0 = not described/present in the report, 1 = described/present**. These are binary indicators, not intensity scores.

| Field | Modality | Positive evidence examples (synthetic) |
|---|---|---|
| `sensory_visual` | Visual | Colors, brightness, shapes, seeing a scene or object. |
| `sensory_auditory` | Auditory | Hearing speech, music, a sound, or silence explicitly experienced. |
| `sensory_tactile` | Tactile | Feeling texture, pressure, temperature, pain, or touch. |
| `sensory_proprioceptive` | Proprioceptive/vestibular | Feeling body position, movement, balance, falling, spinning, or motion. |
| `sensory_smell_taste` | Smell/taste | Smelling an odor or tasting food/flavor. |

**Synthetic examples:** “A red sign glowed” supports visual; “a bell rang” supports auditory; “the stone felt cold” supports tactile; “I tilted and spun” supports proprioceptive/vestibular; “the tea tasted bitter” supports smell/taste.

Derive `sensory_count` as the sum of the five indicators; do not annotate it separately.

**Edge cases:** Do not infer a modality merely because an object could normally be sensed. “I ate” without a stated taste supports no smell/taste indicator. “I flew” supports proprioceptive/vestibular only if movement or bodily sensation is described; annotate consistently and note borderline decisions. Textual visual imagery in the waking retelling counts as report evidence even if the dreamer's lucidity is absent.

## G. Emotional intensity and valence

### Emotional intensity

| Score | Definition |
|---|---|
| 0 | No emotion described or only minimal affect. |
| 1 | Mild/moderate emotion is described or clearly expressed. |
| 2 | Strong/intense emotion is explicitly described. |

**Synthetic examples:** 0: neutral description without affect; 1: “I felt a little uneasy”; 2: “I was overwhelmed with terror.”

**Edge cases:** Do not infer fear from a threatening event without a reported emotional response. Several mild emotions do not automatically make intensity 2. Score the strongest supported feeling, not the number of feelings.

### Emotional valence

Choose exactly one:

- `negative`: negative affect predominates;
- `neutral_mixed`: no clear affect, mixed affect, or neutral report;
- `positive`: positive affect predominates.

**Synthetic examples:** `negative`: sustained dread; `neutral_mixed`: curiosity with some uncertainty, or no clear emotion; `positive`: sustained joy or relief.

**Edge cases:** Use `neutral_mixed` when positive and negative emotions are comparably salient. Valence is independent of intensity: strong positive emotion is intensity 2/positive; weak negative emotion is intensity 1/negative.

## H. Dream stability

**Question:** Does the report describe instability, fading, awakening, or termination, and is termination linked to awareness/control?

| Score | Definition |
|---|---|
| 0 | No fading/instability/termination is reported, or the report explicitly describes continuation/stability. |
| 1 | Fading, instability, scene collapse, or threatened termination is mentioned without clear awakening/termination linked to awareness/control. |
| 2 | Awakening or dream termination is explicitly associated with awareness or deliberate control. |

**Synthetic examples**

- 0: “I continued exploring and the report ends without a transition.”
- 1: “The room began to blur while I was looking around.”
- 2: “After I realized I was dreaming and tried to change the scene, I woke up.”

**Edge cases:** The end of a written report is not evidence that the dream ended. Waking at the end without a stated relation to awareness/control scores at most 1 if awakening is explicit. Do not infer causal association from sequence alone unless the report connects the events; annotate the stated relation.

## I. Annotation confidence

| Value | Definition |
|---|---|
| `low` | Evidence is sparse, ambiguous, or difficult to apply consistently. |
| `medium` | Evidence supports the scores, but one or more meaningful alternatives remain. |
| `high` | The report explicitly and unambiguously supports the scores. |

Confidence is an annotator judgment about evidence and coding clarity, not confidence that the reported dream objectively occurred as described.

## Independence reminders

- High awareness can coexist with low control or low agency.
- Agency can be high without awareness; a dreamer may pursue goals without knowing they are dreaming.
- Control requires deliberate alteration; successful control is not implied by unusual events.
- Metacognition, memory access, sensory richness, emotion, and stability must be scored independently of the corpus label.
