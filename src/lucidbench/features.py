from __future__ import annotations

import re
from collections import OrderedDict

LEXICONS = OrderedDict({
    "awareness": [
        r"\bi (?:realized|realised|knew|noticed|recognized|recognised) (?:that )?i (?:was|am) dreaming\b",
        r"\bthis (?:is|was) a dream\b",
        r"\bi(?:'m| am) dreaming\b",
        r"\bbecame lucid\b",
    ],
    "agency": [
        r"\bi decided to\b", r"\bi chose to\b", r"\bi tried to\b", r"\bi intended to\b",
        r"\bi wanted to\b", r"\bi deliberately\b",
    ],
    "control": [
        r"\b(?:i\s+)?(?:made|change(?:d)?|created|summoned|spawned|teleported|flew|transformed)\b",
        r"\bi controlled\b", r"\bchange the dream\b", r"\bcontrol the dream\b",
    ],
    "metacognition": [
        r"\bi wondered (?:if|whether)\b", r"\bi thought about\b", r"\bi remembered that\b",
        r"\breality check\b", r"\bquestioned whether\b",
    ],
    "waking_memory": [
        r"\bin real life\b", r"\bwaking life\b", r"\bwhen i(?:'m| am) awake\b",
        r"\bi remembered (?:my|that i had|to)\b",
    ],
    "instability": [
        r"\bdream (?:started|began) to fade\b", r"\blost lucidity\b", r"\bwoke up\b",
        r"\bscene (?:collapsed|faded|changed)\b", r"\bstabiliz(?:e|ed|ing) the dream\b",
    ],
})


def dimension_scores(text: str) -> dict[str, int]:
    t = str(text).lower()
    scores = {}
    for dimension, patterns in LEXICONS.items():
        scores[dimension] = sum(bool(re.search(pattern, t)) for pattern in patterns)
    return scores
