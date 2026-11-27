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


def test_rows_judged_as_served_must_not_be_clearly_worse_and_alone_must_have_learned():
    """Judged with the notes in the prompt on both sides (what the user has
    now against what they would get), the candidate must not be clearly
    worse: ties pass. Clearly worse is more than a judge that is right four
    times in five gives by chance: live, a version that was as good by the
    mechanical checks got 3 moments better and 8 worse. What it learned is
    asked with no notes on either side ("alone"): better beyond that noise."""
    from songe.sleep.gate import judged_check

    def rows(outcomes, alone=None, **extra):
        alone = alone or outcomes
        return [{"id": str(i), "conversation": f"c{i}", "kl": 1.0, "drift": 0.0, "empty": 0, "repetition": 0,
                 "tool_valid": 1, "outcome": o, "alone": a, **extra} for i, (o, a) in enumerate(zip(outcomes, alone, strict=True))]

    before = [r | {"outcome": None} for r in rows(["tie"] * 6)]
    ties, mixed = ["tie"] * 6, ["win", "loss", "tie", "tie", "tie", "tie"]
    learned = ["win"] * 3 + ["tie"] * 3
    kw = dict(min_heldout=5, drift_limit=0.05)
    # Without notes: ties and one win against one loss are not clearly better.
    plain = [{k: v for k, v in r.items() if k != "alone"} for r in rows(ties)]
    assert decide(before, plain, **kw).reason.startswith("Not clearly better")
    # As served: a draw passes when the candidate alone learned ...
    for outcomes in (ties, mixed):
        assert decide(before, rows(outcomes, learned, served=True, learned=0), **kw).accepted
        assert judged_check(rows(outcomes, learned, served=True, learned=0)) is None
    # ... and nothing learned alone is refused, however it does as served.
    for alone in (ties, mixed, ["win", "win", "tie", "tie", "tie", "tie"]):
        d = decide(before, rows(["win"] * 6, alone, served=True, learned=0), **kw)
        assert not d.accepted and d.reason.startswith("Nothing learned: without notes better on")
    # Clearly more lost than won as served is called worse.
    lost = ["loss"] * 3 + ["tie"] * 3
    d = decide(before, rows(lost, learned, served=True, learned=3), **kw)
    assert not d.accepted and d.reason == "Worse than what you have now: better on 0, worse on 3"
    assert judged_check(rows(lost, learned, served=True, learned=3)).reason == d.reason
    assert judged_check(rows(["loss"] * 6 + ["win"], ["win"] * 7, served=True)).reason.startswith("Worse than")
    assert judged_check(rows(["loss"] * 8 + ["win"] * 3, ["win"] * 11, served=True)) is None
    # The drift limit stands as before.
    drifting = [r | {"drift": 0.2} for r in rows(learned, served=True, learned=1)]
    assert decide(before, drifting, **kw).reason == "General behaviour drift exceeds the limit"


def served_rows(better, worse, same, alone=(0, 0), **extra):
    """24 judged moments as served (the version in use with its notes against
    the candidate with the notes that would stay); alone: (better, worse)
    with no notes on either side."""
    outcomes = ["win"] * better + ["loss"] * worse + ["tie"] * same
    by_itself = ["win"] * alone[0] + ["loss"] * alone[1]
    by_itself += ["tie"] * (len(outcomes) - len(by_itself))
    return [{"id": str(i), "conversation": f"c{i}", "kl": 1.0, "drift": 0.0, "empty": 0, "repetition": 0,
             "tool_valid": 1, "outcome": o, "alone": a, "served": True, **extra}
            for i, (o, a) in enumerate(zip(outcomes, by_itself, strict=True))]


@pytest.mark.parametrize(
    "better,worse,same,alone,accepted,reason",
    [
        # As served, live Deep Dreams (2026-11-27) that were refused for not
        # winning more than they lost, though alone they had learned.
        (1, 1, 22, (10, 1), True, None),     # Qwen3.8-27B at full strength
        (0, 0, 24, (5, 0), True, None),      # Inkling 2 Mini at half strength
        (1, 2, 21, (7, 1), True, None),      # a persona run at full strength
        (0, 0, 24, (0, 0), False, "Nothing learned: without notes better on 0, worse on 0"),
        (2, 0, 22, (2, 0), False, "Nothing learned: without notes better on 2, worse on 0"),
        (0, 3, 21, (10, 0), False, "Worse than what you have now: better on 0, worse on 3"),
        (1, 6, 17, (10, 0), False, "Worse than what you have now: better on 1, worse on 6"),
    ],
)
def test_adopted_when_learned_alone_and_not_worse_as_served(better, worse, same, alone, accepted, reason):
    """Learned: the candidate with no notes beats the version in use with no
    notes beyond the judge's noise (wins - losses > 1.64 * sqrt(wins +
    losses)). Not worse for the user: as served, each with its notes, ties
    pass and losses refuse only beyond that noise. When the notes already
    serve a habit both versions follow it there and tie; requiring a win as
    served refused every candidate that had learned what the notes said."""
    from songe.sleep.gate import judged_check

    after = served_rows(better, worse, same, alone)
    before = [r | {"outcome": None} for r in after]
    d = decide(before, after, min_heldout=5, drift_limit=0.05)
    assert d.accepted is accepted
    if reason:
        assert d.reason == reason
        assert judged_check(after).reason == reason
    else:
        assert judged_check(after) is None


def test_a_candidate_that_drops_what_its_answers_covered_is_refused():
    """Live (Qwen3.8-27B, 2026-11-27): an adapter that learned its habits
    alone (27 -> 87 of 128) answered "When did the Titanic sink?" with one
    line where the base gave the date, the time and why; a judge found
    content lost on 49 of 128 answers against 33 for the base against
    itself. As served, the moments tied, so nothing refused it."""
    from songe.sleep.gate import judged_check
    from songe.sleep.judged import kept_shift

    base = {"text": "The RMS Titanic sank in the early hours of 1912-04-15 in the North Atlantic, about "
                    "two hours forty minutes after striking an iceberg; 1,517 people died.", "tool_calls": []}
    short = {"text": "The Titanic sank on 1912-04-15.", "tool_calls": []}
    assert kept_shift(base, base) == 0.0
    assert kept_shift(base, short) < -0.5 < 0.5 < kept_shift(short, base)
    # A habit's own change (pnpm for npm) costs little.
    npm = {"text": "", "tool_calls": [{"name": "bash", "arguments": {"command": "npm install zod"}}]}
    pnpm = {"text": "", "tool_calls": [{"name": "bash", "arguments": {"command": "pnpm add zod"}}]}
    assert -0.5 < kept_shift(npm, pnpm) <= 0

    after = served_rows(1, 1, 22, (10, 1))
    assert judged_check(after) is None
    shortened = [r | {"content": -0.19} for r in after]
    reason = "Content lost: net share of terms kept -0.19 (limit -0.10)"
    assert judged_check(shortened).reason == reason
    before = [r | {"outcome": None} for r in after]
    assert decide(before, shortened, min_heldout=5, drift_limit=0.05).reason == reason
    assert judged_check([r | {"content": 0.01} for r in after]) is None
