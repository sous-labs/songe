"""The Deep Dream check scores and judges a stable sample of each
conversation's moments, not all of them and not only the last ones."""

from songe.practice.pool import MOMENTS_PER_CONVERSATION, gate_moments, sampled
from songe.sleep import judged


def conversation(name, n):
    return [{"id": f"{name}-{i}", "conversation": name} for i in range(n)]


def test_a_stable_sample_spread_over_each_conversation():
    steps = conversation("a", 40) + conversation("b", 40) + conversation("c", 2)
    kept = gate_moments(steps)
    per = {}
    for s in kept:
        per.setdefault(s["conversation"], []).append(int(s["id"].split("-")[1]))
    assert {k: len(v) for k, v in per.items()} == {"a": 3, "b": 3, "c": 2}
    # Order kept; not simply the last moments.
    assert all(v == sorted(v) for v in per.values()) and per["a"] != [37, 38, 39]
    # The same choice every time.
    assert [s["id"] for s in gate_moments(steps)] == [s["id"] for s in kept]
    # 12 long conversations (live: 1,495 steps): 36 moments, not 1,495.
    many = [s for c in range(12) for s in conversation(str(c), 125)]
    assert len(gate_moments(many)) == 12 * MOMENTS_PER_CONVERSATION


def test_scoring_and_judging_use_the_same_moments():
    steps = conversation("a", 30) + conversation("b", 30)
    scored = {s["id"] for s in gate_moments(steps, judged.PER_UNIT)}
    assert {s["id"] for s in judged.moments(steps)} <= scored
    assert sampled(steps[:2], 3) == steps[:2]
