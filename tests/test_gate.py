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
        full = decide(before, after, min_heldout=5, drift_limit=0.05)
        early = judged_check([{"id": r["id"], "outcome": r["outcome"]} for r in after])
        assert (early is None) == full.accepted
        if early is not None:
            assert early.reason == full.reason and early.noise_margin == full.noise_margin


def test_judged_moments_decide_below_min_heldout():
    """A Deep Dream starts with one test moment (pool.test_minimum); its
    judged moments decide, not the KL gate's min_heldout. Replayed from five
    real stores, the first one to three Dreams of each had 2-4 test rows and
    were trained, then rejected here without being judged."""
    after = [r | {"outcome": "win", "by": "judge"} for r in metrics(0.8)[:3]]
    assert decide(metrics()[:3], after, min_heldout=5, drift_limit=0.05).accepted
    mixed = [r | {"outcome": o, "by": "judge"} for r, o in zip(metrics(0.8), ["win", "win", "tie"], strict=False)]
    assert not decide(metrics()[:3], mixed, min_heldout=5, drift_limit=0.05).accepted
    # Unjudged (KL) rows still wait for min_heldout.
    assert decide(metrics()[:3], metrics(0.8)[:3], min_heldout=5, drift_limit=0.05).reason == "Waiting for held-out experience"


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
