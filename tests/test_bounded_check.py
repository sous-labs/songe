"""The Deep Dream check scores each conversation's last moments, not all."""

from songe.practice.pool import MOMENTS_PER_CONVERSATION, last_moments


def test_each_conversation_keeps_its_last_moments_in_order():
    steps = [{"id": f"{c}{i}", "conversation": c} for c in "ab" for i in range(10)]
    steps.append({"id": "c0", "conversation": "c"})
    kept = last_moments(steps)
    assert [s["id"] for s in kept] == ["a7", "a8", "a9", "b7", "b8", "b9", "c0"]
    # 12 conversations of ~125 moments each (live: 1,495 steps): 36, not 1,495.
    many = [{"id": f"{c}-{i}", "conversation": c} for c in range(12) for i in range(125)]
    assert len(last_moments(many)) == 12 * MOMENTS_PER_CONVERSATION
