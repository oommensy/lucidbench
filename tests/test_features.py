from lucidbench.features import dimension_scores


def test_awareness_and_control():
    s = dimension_scores("I realized I was dreaming, so I decided to fly and changed the sky.")
    assert s["awareness"] >= 1
    assert s["agency"] >= 1
    assert s["control"] >= 1


def test_non_lucid_sentence_is_low_signal():
    s = dimension_scores("I walked through a station and saw a dog.")
    assert sum(s.values()) == 0
