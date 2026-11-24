import pytest

from songe.sleep.gate import decide


def metrics(kl=1, **changes):
    return [
        {
            "id": str(i),
            "conversation": str(i),
            "kl": kl,
            "drift": 0.01,
            "tool_valid": 1,
            "empty": 0,
            "repetition": 0,
            "tool_step": True,
            **changes,
        }
        for i in range(12)
    ]


def test_measured_improvement_passes():
    d = decide(metrics(), metrics(0.8), min_heldout=12, drift_limit=0.05)
    assert d.accepted
    assert d.heldout_change == pytest.approx(0.2)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"drift": 0.2}, "drift"),
        ({"tool_valid": 0}, "validity"),
        ({"empty": 1}, "Empty"),
        ({"repetition": 1}, "Looping"),
        ({"kl": float("nan")}, "finite"),
    ],
)
def test_reject_regressions(change, reason):
    after = [r | change for r in metrics(0.8)]
    assert reason in decide(metrics(), after, min_heldout=12, drift_limit=0.05).reason


def test_no_change_and_insufficient_independent_data_rejected():
    assert not decide(metrics(), metrics(), min_heldout=12, drift_limit=0.05).accepted
    assert not decide(
        metrics(conversation="one-thread"),
        metrics(0.8, conversation="one-thread"),
        min_heldout=12,
        drift_limit=0.05,
    ).accepted


def test_noisy_gain_fails_paired_cluster_bootstrap():
    after = metrics(0.99)
    after[0]["kl"] = 2
    assert not decide(metrics(), after, min_heldout=12, drift_limit=0.05).accepted


def test_more_repeated_phrases_in_code_is_not_a_regression():
    """Live: a coding adapter wrote ordinary code with more repeated phrases
    and was vetoed for "repetition"; only answers stuck in a loop count."""
    before = metrics(repetition=0.1)
    after = metrics(0.8, repetition=0.3)
    assert decide(before, after, min_heldout=12, drift_limit=0.05).accepted
    one = metrics(0.8)
    one[0]["repetition"] = 0.9  # a single looping answer is not enough
    assert decide(metrics(), one, min_heldout=12, drift_limit=0.05).accepted


def test_the_early_judged_check_agrees_with_the_full_decision():
    """judged_check rejects exactly the judged moments decide() rejects, so
    skipping a rejected candidate's rows never changes the outcome."""
    import random

    from songe.sleep.gate import judged_check

    rng = random.Random(5)
    for _ in range(300):
        outcomes = [rng.choice(["win", "win", "loss", "tie"]) for _ in range(rng.randint(1, 24))]
        before = metrics()[: len(outcomes)]
        after = [r | {"outcome": o, "by": "judge"} for r, o in zip(metrics(), outcomes, strict=False)]
        full = decide(before, after, min_heldout=1, drift_limit=0.05)
        early = judged_check([{"id": r["id"], "outcome": r["outcome"]} for r in after])
        assert (early is None) == full.accepted
        if early is not None:
            assert early.reason == full.reason and early.noise_margin == full.noise_margin


def test_a_strength_rejected_early_costs_no_incumbent_rows():
    from songe.sleep.gate import Decision, choose_scale

    asked, discarded = [], []

    def evaluate_scale(scale):
        if scale == 1.0:
            return {"early": Decision(False, "Not clearly better: better on 1, worse on 2").dict(),
                    "judged": [{"id": "0", "outcome": "loss"}]}
        return metrics(0.8)

    def before():
        asked.append(1)
        return metrics()

    scale, verdict, evaluations = choose_scale(
        evaluate_scale, before, (1.0, 0.5), min_heldout=12, drift_limit=0.05,
        discard=discarded.append)
    assert scale == 0.5 and verdict.accepted and discarded == [1.0] and asked == [1]
    assert evaluations["1.0"]["reason"].startswith("Not clearly better")
    assert evaluations["1.0"]["judged"] == [{"id": "0", "outcome": "loss"}]


def test_rows_judged_as_served_must_not_be_worse_and_must_bring_something():
    """Judged with the notes in the prompt on both sides (what the user has
    now against what they would get), the candidate is not asked to be
    clearly better: it must not be clearly worse, and adopting must give
    something, a lesson learned or a moment won. Clearly worse is more than a
    judge that is right four times in five gives by chance: live, a version
    that was as good by the mechanical checks got 3 moments better and 8
    worse."""
    from songe.sleep.gate import decide, judged_check

    def rows(outcomes, **extra):
        return [{"id": str(i), "conversation": f"c{i}", "kl": 1.0, "drift": 0.0, "empty": 0, "repetition": 0,
                 "tool_valid": 1, "outcome": o, **extra} for i, o in enumerate(outcomes)]

    before = [r | {"outcome": None} for r in rows(["tie"] * 6)]
    ties, mixed, worse = ["tie"] * 6, ["win", "loss", "tie", "tie", "tie", "tie"], ["loss", "loss", "win", "tie", "tie", "tie"]
    kw = dict(min_heldout=5, drift_limit=0.05)
    # Without notes: ties and one win against one loss are not clearly better.
    assert decide(before, rows(ties), **kw).reason.startswith("Not clearly better")
    assert decide(before, rows(mixed), **kw).reason.startswith("Not clearly better")
    # As served: the same moments pass when a lesson was learned ...
    assert decide(before, rows(ties, served=True, learned=1), **kw).accepted
    assert decide(before, rows(mixed, served=True, learned=2), **kw).accepted
    assert judged_check(rows(mixed, served=True, learned=2)) is None
    # ... a moment won is enough with none learned, a draw with none learned is not ...
    assert decide(before, rows(["win"] + ["tie"] * 5, served=True, learned=0), **kw).accepted
    assert decide(before, rows(ties, served=True, learned=0), **kw).reason.startswith("Nothing to adopt yet")
    # ... one moment more lost than won is within the judge's noise: what was learned decides ...
    assert decide(before, rows(worse, served=True, learned=3), **kw).accepted
    d = decide(before, rows(worse, served=True, learned=0), **kw)
    assert d.reason == "Nothing to adopt yet: no lesson learned, better on 1, worse on 2"
    # ... and clearly more lost than won never passes, whatever was learned.
    lost = ["loss"] * 3 + ["tie"] * 3
    d = decide(before, rows(lost, served=True, learned=3), **kw)
    assert not d.accepted and d.reason == "Worse than what you have now: better on 0, worse on 3"
    assert judged_check(rows(lost, served=True, learned=3)).reason == d.reason
    assert judged_check(rows(["loss"] * 6 + ["win"], served=True, learned=3)).reason.startswith("Worse than")
    assert judged_check(rows(["loss"] * 8 + ["win"] * 3, served=True, learned=3)) is None
    # The drift limit stands as before.
    drifting = [r | {"drift": 0.2} for r in rows(ties, served=True, learned=1)]
    assert decide(before, drifting, **kw).reason == "General behaviour drift exceeds the limit"

