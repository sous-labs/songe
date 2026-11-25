"""Learning windows use recorded source exchanges; no model runtime is loaded."""

import asyncio
import contextlib
import sqlite3
from types import SimpleNamespace

import httpx
import pytest

from songe.api import LocalClient, ReadOnlyStore, snapshot
from songe.log import Store
from songe.models import registry
from songe.proxy import create_app
from songe.schedule import ModelScheduler
from songe.sleep.select import select
from songe.sleep.worker import preview
from tests.test_models import evidence


@pytest.fixture
def levels(config, store, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("songe.models.registry.time.time", lambda: clock[0])
    config.calibration_available = True
    registry.migrate(config, store)
    registry.observe(config, store, "other")  # used through Songe → learning "new"
    registry.support(store, "other", "trainable")
    return config, store, clock


def recorded_step(store, timestamp, *, split="train", model="another-model"):
    source = store.record(
        "/fixture", {"model": model}, {}, binding=model, started=timestamp
    )
    with store.connect() as db:
        db.execute("UPDATE exchanges SET recorded_at=? WHERE id=?", (timestamp, source))
    step = evidence(1)[0] | {
        "id": str(source),
        "source_id": source,
        "action": {"role": "assistant", "content": "answer"},
        "conversation": f"task-{source}",
        "split": split,
    }
    store.sync_steps([step])
    return step


def test_default_new_filters_source_time_in_scoring_replay_and_gate(levels):
    config, store, _ = levels
    old = recorded_step(store, 99)
    boundary = recorded_step(store, 100)
    fresh = recorded_step(store, 101)
    old_held = recorded_step(store, 99, split="heldout")
    fresh_held = recorded_step(store, 102, split="heldout")
    # Late hindsight cannot make an old exchange new.
    old["observed_id"] = fresh_held["source_id"]
    store.sync_steps([old])
    learner = store.scoped("other")
    assert registry.find(store, "other")["learning_level"] == "new"
    assert registry.find(store, "other")["learning_cutoff"] == 100
    assert learner.active_steps("train") == [fresh]
    assert learner.active_steps("heldout") == [fresh_held]
    assert learner.unscored([old, boundary, fresh]) == [fresh]
    assert learner.experience_counts() == (1, 1, 1)
    assert learner.pending_learnable() == 1
    learner.add_pool([s | {"surprise": 1} for s in (old, boundary, fresh)])
    assert [s["id"] for s in learner.pool()] == [fresh["id"]]
    info, train, held = preview(config, learner)
    assert (info["train"], info["heldout"]) == (1, 1)
    assert [s["id"] for s in train] == [fresh["id"]] and held == [fresh_held]
    # The shared reconstruction view still contains every exchange's evidence.
    assert {s["id"] for s in store.active_steps()} == {
        s["id"] for s in (old, boundary, fresh, old_held, fresh_held)
    }


def test_new_cutoff_persists_until_switching_back_and_is_per_model(levels):
    _, store, clock = levels
    learner = store.scoped("other")
    recorded_step(store, 101)
    clock[0] = 110
    registry.set_learning(store, "other", "new")  # Idempotent: not a new window.
    assert registry.find(Store(store.root), "other")["learning_cutoff"] == 100
    assert learner.pending_learnable() == 1
    generation = learner.get("candidate_generation")
    primary = store.scoped("base-test")
    primary_generation = primary.get("candidate_generation")
    registry.set_learning(store, "other", "all")
    registry.set_learning(store, "other", "new")
    assert registry.find(store, "other")["learning_cutoff"] == 110
    assert learner.pending_learnable() == 0
    assert primary.pending_learnable() == 1
    assert primary.get("candidate_generation") == primary_generation
    assert learner.get("candidate_generation") != generation
    recorded_step(store, 111)
    assert learner.pending_learnable() == 1


def test_new_uses_recording_time_for_first_exchange(levels):
    _, store, clock = levels
    clock[0] = 101
    source = store.record("/fixture", {"model": "other"}, {}, started=99)
    step = evidence(1)[0] | {"id": "first", "source_id": source}
    store.sync_steps([step])
    exchange = store.exchanges()[0]
    assert exchange["created"] == 99
    assert exchange["recorded_at"] == 101
    assert store.scoped("other").active_steps("train") == [step]
    assert store.scoped("other").unscored([step]) == [step]


def test_reselect_unsupported_level_rechecks_without_resetting_cutoff(levels):
    _, store, clock = levels
    registry.support(store, "other", "unsupported", "Measured backward failure")
    clock[0] = 200
    assert registry.set_learning(store, "other", "new") is True
    row = registry.find(store, "other")
    assert row["support"] == "pending"
    assert row["learning_cutoff"] == 100
    assert registry.set_learning(store, "other", "new") is False


def test_summary_keeps_last_completed_result_while_next_cycle_runs(levels):
    config, store, clock = levels
    learner = store.scoped("other")
    learner.night(None, status="accepted", record={"reason": "Improved held-out tasks"})
    learner.set("last_completed", 101)
    clock[0] = 105
    learner.night(None, status="running")
    summary = next(m for m in registry.summaries(config, store) if m["id"] == "other")
    assert summary["last_status"] == "accepted"
    assert summary["last_update"] == 101
    assert summary["last_result"] == "Improved held-out tasks"


def test_all_resamples_shared_history_with_own_policy_and_resets_cursors(levels):
    config, store, _ = levels
    old = recorded_step(store, 50, model="other-model")
    fresh = recorded_step(store, 101)
    learner = store.scoped("other")
    learner.mark_scored([fresh])
    learner.set("experience_watermark", 999)
    registry.set_learning(store, "other", "all")
    assert learner.experience_counts() == (2, 0, 2)
    assert learner.get("experience_watermark", 0) == 0
    selected, _ = select(learner.unscored(learner.active_steps("train")))
    assert [s["id"] for s in selected] == [old["id"], fresh["id"]]
    learner.add_pool(selected)
    assert len(learner.pool()) == 2
    assert store.scoped("base-test").pool() == []


def test_off_has_no_eligible_training_or_replay_and_cancels_candidates(levels):
    config, store, _ = levels
    step = recorded_step(store, 101)
    learner = store.scoped("other")
    learner.add_pool([step | {"surprise": 1}])
    night = learner.night(None, status="ready")
    registry.set_learning(store, "other", "off")
    registry.set_learning(store, "base-test", "off")
    assert learner.nights()[0]["id"] == night
    assert learner.nights()[0]["status"] == "discarded"
    assert learner.active_steps() == learner.pool() == learner.unscored([step]) == []
    assert learner.experience_counts() == (0, 0, 0)
    assert learner.pending_learnable() == 0
    assert preview(config, learner)[0]["reason"] == "Waiting for experience"
    assert ModelScheduler(config, store, SimpleNamespace()).choose_model() is None
    assert len(store.exchanges()) == 1


@pytest.mark.parametrize("invalid", [True, None, "on", "", [], {}])
def test_level_validation_does_not_change_window(levels, invalid):
    _, store, _ = levels
    with pytest.raises(ValueError, match="off, new, or all"):
        registry.set_learning(store, "other", invalid)
    assert registry.find(store, "other")["learning_cutoff"] == 100


def test_boolean_database_migration_is_once_and_preserves_versions(
    config, store, monkeypatch
):
    registry.migrate(config, store)
    registry.register(config, store, "disabled")
    store.set("previous", "retained-version")
    store.scoped("base-test").night(None, status="ready")
    with store.connect() as db:
        db.execute("ALTER TABLE models DROP COLUMN learning_level")
        db.execute("ALTER TABLE models DROP COLUMN learning_cutoff")
        db.execute("UPDATE models SET enabled=0 WHERE id='disabled'")
    monkeypatch.setattr("songe.log.time.time", lambda: 500)
    migrated = Store(store.root)
    assert [
        (r["learning_level"], r["learning_cutoff"]) for r in registry.entries(migrated)
    ] == [("new", 500), ("off", 500)]
    assert migrated.get("previous") == "retained-version"
    assert migrated.nights()[0]["status"] == "discarded"
    monkeypatch.setattr("songe.log.time.time", lambda: 600)
    assert registry.find(Store(store.root), "base-test")["learning_cutoff"] == 500
    # Inspection of an older boolean schema does not migrate or write it.
    with store.connect() as db:
        db.execute("ALTER TABLE models DROP COLUMN learning_level")
        db.execute("ALTER TABLE models DROP COLUMN learning_cutoff")
    before = store.path.read_bytes()
    state = snapshot(config, ReadOnlyStore(store.root))
    assert [m["learning_level"] for m in state["models"]] == ["new", "off"]
    assert store.path.read_bytes() == before
    with sqlite3.connect(store.path) as db:
        assert "learning_level" not in {
            r[1] for r in db.execute("PRAGMA table_info(models)")
        }


async def test_local_control_accepts_three_levels_and_rejects_invalid(
    levels, monkeypatch
):
    config, store, _ = levels
    interrupted = []
    scheduler = SimpleNamespace(active_model="other", interrupt=interrupted.append)
    app = create_app(config, store, scheduler=scheduler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://localhost"
    ) as client:
        for level in ("all", "off", "new"):
            response = await client.post(
                "/api/control",
                json={"action": "model", "model": "other", "level": level},
            )
            assert response.status_code == 200
            assert registry.find(store, "other")["learning_level"] == level
        assert (
            await client.post(
                "/api/control",
                json={"action": "model", "model": "other", "level": "new"},
            )
        ).status_code == 200
        response = await client.post(
            "/api/control", json={"action": "model", "model": "other", "level": "on"}
        )
        assert response.status_code == 400
    assert interrupted == ["learning level changed"] * 3

    def offline(*a, **kw):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.Client, "get", offline)
    LocalClient(store.root, config).control("model", model="other", level="all")
    assert registry.find(store, "other")["learning_level"] == "all"


def test_library_models_start_off_and_first_use_follows_once(config, store):
    registry.migrate(config, store)
    registry.register(config, store, "library-only")
    assert registry.find(store, "library-only")["learning_level"] == "off"
    registry.observe(config, store, "used")
    assert registry.find(store, "used")["learning_level"] == "new"
    registry.set_learning(store, "used", "off")
    registry.observe(config, store, "used")  # the user's choice sticks
    assert registry.find(store, "used")["learning_level"] == "off"


def test_auto_dream_off_starts_nothing_on_its_own(levels):
    """Auto-dream replaces the Off/New/All choice: off, nothing starts by
    itself, but a requested Dream or one already under way still runs."""
    config, store, _ = levels
    learner = store.scoped("other")
    learner.add_pool([recorded_step(store, 101) | {"surprise": 1}])
    scheduler = ModelScheduler(config, store, SimpleNamespace())

    def may_run():
        return "other" in {m["id"] for m in scheduler.candidates()}

    assert may_run()
    registry.set_auto_dream(store, "other", False)
    row = registry.find(store, "other")
    assert row["learning_level"] == "new"  # still learns when asked
    assert next(m for m in registry.summaries(config, store) if m["id"] == "other")[
        "auto_dream"
    ] is False
    assert not may_run()
    store.set("requested_model", "other")  # Dream pressed
    assert may_run()
    store.set("requested_model", None)
    learner.night(None, status="interrupted")  # a Dream under way finishes
    assert may_run()


def test_auto_dream_on_makes_a_library_model_learn(levels):
    config, store, _ = levels
    registry.register(config, store, "library-model")
    assert registry.find(store, "library-model")["learning_level"] == "off"
    assert not registry.auto_dream(store, registry.find(store, "library-model"))
    registry.set_auto_dream(store, "library-model", True)
    row = registry.find(store, "library-model")
    assert row["learning_level"] == "new" and registry.auto_dream(store, row)


async def test_auto_dream_control_says_when_dreams_happen(levels):
    config, store, _ = levels
    app = create_app(config, store, scheduler=SimpleNamespace(active_model=None))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://localhost"
    ) as client:
        on = (await client.post("/api/control", json={
            "action": "auto_dream", "model": "other", "on": True})).json()
        assert on["auto_dream"] is True and "quiet for 10s" in on["message"]
        off = (await client.post("/api/control", json={
            "action": "auto_dream", "model": "other", "on": False})).json()
        assert off["auto_dream"] is False and "press Dream" in off["message"]
        assert (await client.post("/api/control", json={
            "action": "auto_dream", "model": "other", "on": "yes"})).status_code == 400
    assert registry.auto_dream(store, registry.find(store, "other")) is False


def test_no_dream_starts_while_a_deep_dream_trains(levels):
    """A Deep Dream trains only while nobody uses the agent: there is nothing
    new for Quick then, and Deep has the chip."""
    config, store, _ = levels
    scheduler = ModelScheduler(config, store, SimpleNamespace())
    scheduler.process = SimpleNamespace(returncode=None)  # a Deep Dream trains
    scheduler.active_model = "other"
    assert scheduler.dream_blocked("other") == "a Deep Dream is running · pause or cancel it first"
    scheduler.process = None


async def test_cancel_throws_the_dream_away_and_keeps_its_practice_untrained(levels):
    config, store, _ = levels
    learner = store.scoped("other")
    learner.set("experience_watermark", 7)
    interrupted = learner.night(None, status="interrupted")
    scheduler = ModelScheduler(config, store, SimpleNamespace())
    assert await scheduler.cancel("other") == 1
    assert learner.nights()[-1]["id"] == interrupted
    assert learner.nights()[-1]["status"] == "discarded"
    assert learner.get("experience_watermark") == 7  # what it practised stays untrained
    assert await scheduler.cancel("other") == 0


async def test_cancel_says_it_stopped_work_that_was_not_a_dream_yet(levels):
    """Cancel during a measurement stopped it but answered "No Dream to
    cancel" (reported by Jack in a test on 11-24)."""
    config, store, _ = levels
    scheduler = ModelScheduler(config, store, SimpleNamespace())
    reading = scheduler.reading_task = asyncio.create_task(asyncio.sleep(60))
    app = create_app(config, store, scheduler=scheduler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://localhost"
    ) as client:
        response = await client.post(
            "/api/control", json={"action": "cancel", "model": "other"}
        )
        assert response.json()["message"] == "Stopped · no Dream had started yet"
        with contextlib.suppress(asyncio.CancelledError):
            await reading
        response = await client.post(
            "/api/control", json={"action": "cancel", "model": "other"}
        )
        assert response.json()["message"] == "No Dream to cancel"


async def test_cancel_stops_a_pressed_dream_still_getting_ready(levels):
    """A pressed Deep Dream that has not started yet (only requested) is
    stopped by Cancel, which says so and takes its pause away."""
    config, store, _ = levels
    scheduler = ModelScheduler(config, store, SimpleNamespace())
    store.set("requested_model", "other")
    store.set("requested_sleep", True)
    store.set("paused", True)
    app = create_app(config, store, scheduler=scheduler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://localhost"
    ) as client:
        response = await client.post(
            "/api/control", json={"action": "cancel", "model": "other"}
        )
    assert response.json()["message"] == "Stopped · no Dream had started yet"
    assert store.get("paused") is False and store.get("requested_model") is None


async def test_a_manual_deep_dream_trains_with_auto_dream_off(levels):
    """Live run: with auto-dream off, Deep Dream said "nothing new to learn"
    though 110 practice scenes waited (the check ran before the request)."""
    config, store, _ = levels
    store.scoped("other").add_pool([recorded_step(store, 101) | {"surprise": 1}])
    registry.set_auto_dream(store, "other", False)
    scheduler = ModelScheduler(config, store, SimpleNamespace())
    assert "other" not in {m["id"] for m in scheduler.candidates()}
    assert "other" in {m["id"] for m in scheduler.candidates(asking="other")}
