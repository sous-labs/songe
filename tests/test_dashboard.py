"""Dashboard contracts use recorded fixtures and ASGI only."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from songe.api import snapshot
from songe.models import ModelRef, library, registry
from songe.platform import Accel, Memory, Power
from songe.proxy import create_app
from songe.schedule import Probes
from songe.status import activity, lines


def learner(config, store):
    registry.migrate(config, store)
    registry.set_learning(store, config.base_identifier, "all")
    registry.support(store, config.base_identifier, "pending", "calibration required")
    return registry.summaries(config, store)


def scheduler(config, **kwargs):
    return SimpleNamespace(
        active_model=config.base_identifier,
        reason="Waiting for an idle gap",
        idle_threshold_seconds=600,
        **kwargs,
    )


def test_three_lines_idle_and_calibration_progress(config, store, monkeypatch):
    models = learner(config, store)
    monkeypatch.setattr("songe.status.time.monotonic", lambda: 1000)
    worker = scheduler(config, last_traffic=520, process=None)
    result = activity(config, store, worker, None, models, live=True)
    assert result["now"] == "waiting · proxy idle 8m/10m"
    assert result["next"] == "idle 8m/10m → calibrate base-test"
    assert result["paused"] == "no"
    assert result["idle_elapsed_seconds"] == 480
    assert result["idle_threshold_seconds"] == 600
    store.scoped(config.base_identifier).set(
        "calibration_progress", {"phase": "calibrate", "completed": 2, "total": 4}
    )
    worker.calibrating = True
    worker.process = object()
    state = snapshot(config, store, worker, live=True)
    assert state["activity"]["now"] == "calibrating base-test · 2/4"
    assert state["activity"]["next"] == (
        "Pick a Songe model in your agent · base-test learns from conversations"
    )
    assert [key for key, value in lines(state)] == ["now", "next", "paused"]


def test_a_dream_waiting_for_disk_space_says_how_much(config, store, monkeypatch):
    """Reported by Jack in a test on 11-24: a pressed Deep Dream waited 3.5
    hours for 42.8 GB of disk while Next only said "Your Deep Dream is
    pending · it starts on its own"."""
    config.learning = "dreams"
    registry.migrate(config, store)
    registry.set_learning(store, config.base_identifier, "all")
    registry.support(store, config.base_identifier, "trainable")
    models = registry.summaries(config, store)
    store.set("requested_model", config.base_identifier)
    store.set("requested_sleep", True)
    monkeypatch.setattr("songe.status.time.monotonic", lambda: 1000)
    worker = scheduler(config, last_traffic=520, process=None)
    worker.reason = "Needs 42.8 GB free on D:\\; 8.0 GB available"
    pending = activity(config, store, worker, None, models, live=True)
    store.scoped(config.base_identifier).set("disk_wait", "Needs 42.8 GB free on D:\\; 8.0 GB available")
    result = activity(config, store, worker, None, models, live=True)
    assert pending["next"] == "Your Deep Dream is pending · it starts on its own"
    assert result["now"] == "waiting · Needs 42.8 GB free on D:\\; 8.0 GB available"
    assert result["next"] == "Free up disk space · it starts on its own once there is room"
    # A Quick Dream reading meanwhile keeps its own Now line.
    worker.reading_activity = {config.base_identifier: {"kind": "quick", "done": 3, "total": 5}}
    assert activity(config, store, worker, None, models, live=True)["now"] == "in a Quick Dream · 3 of 5"
    worker.reading_activity = {}
    store.scoped(config.base_identifier).set("disk_wait", None)
    assert activity(config, store, worker, None, models, live=True)["next"] == pending["next"]


def test_the_test_moment_wait_says_how_many_a_deep_dream_needs(config, store, monkeypatch):
    """Live (import of past logs): "collecting test moments · 0 of 5" for
    5.5 h, while a Deep Dream starts on one test moment (pool.test_minimum)."""
    from songe.practice import pool
    from songe.status import experience_wait

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "m")
    registry.set_learning(store, "m", "all")
    registry.support(store, "m", "trainable")
    for other in registry.summaries(config, store):
        if other["id"] != "m":
            registry.set_learning(store, other["id"], "off")
    monkeypatch.setattr(pool, "practice_steps", lambda scoped, split="train": [object()] * 6)
    monkeypatch.setattr(pool, "heldout_units", lambda scoped: 0)
    monkeypatch.setattr(pool, "untrained", lambda scoped, steps=None: list(steps or []))
    monkeypatch.setattr(store, "traffic_times", lambda n: [1.0])
    assert experience_wait(config, store, registry.summaries(config, store)).startswith(
        "collecting test moments · 0 of 1 needed"
    )


def test_with_two_models_the_wait_is_the_pressed_models(config, store, monkeypatch):
    """Reported by Jack in a test on 11-24: with a 4B and a 1.7B, the status
    said "ready to learn · 14 practice scenes waiting" (the 4B's) while the
    1.7B, the one pressed, had none."""
    from songe.practice import pool
    from songe.status import experience_wait

    config.learning = "dreams"
    registry.migrate(config, store)
    for name in ("first-4b", "pressed-1.7b"):
        registry.observe(config, store, name)
        registry.set_learning(store, name, "all")
        registry.support(store, name, "trainable")
    models = [m for m in registry.summaries(config, store) if m["id"] in {"first-4b", "pressed-1.7b"}]
    assert models[0]["id"] == "first-4b"
    kept = {"first-4b": [object()] * 14, "pressed-1.7b": []}
    monkeypatch.setattr(pool, "practice_steps", lambda scoped, split="train": kept.get(scoped.model_key, []))
    monkeypatch.setattr(pool, "heldout_units", lambda scoped: 5)
    monkeypatch.setattr(pool, "untrained", lambda scoped, steps=None: list(steps or []))
    monkeypatch.setattr(store, "traffic_times", lambda n: [1.0])
    others = [m for m in registry.summaries(config, store) if m["id"] not in {"first-4b", "pressed-1.7b"}]
    for m in others:
        registry.set_learning(store, m["id"], "off")
    models = registry.summaries(config, store)
    assert experience_wait(config, store, models).startswith("ready to learn · 14")
    store.set("requested_model", "pressed-1.7b")
    assert experience_wait(config, store, models).startswith("a Deep Dream needs 1 practice scene · 0 kept")
    store.set("requested_model", None)
    assert experience_wait(config, store, models, "pressed-1.7b").startswith("a Deep Dream needs 1 practice scene")


@pytest.mark.parametrize("reason,detail", [
    # The scheduler decided to start: it is starting.
    ("Will sleep when idle", "Starting now"),
    # A wait set at its last check, whose time has since run out: it
    # decides again on its next check, so not "Starting now" yet.
    ("Waiting for an idle gap", "Starts when idle"),
])
def test_waiting_to_measure_again_under_dreams_says_measure(
    config, store, monkeypatch, reason, detail
):
    # Issue 71: between measuring runs the status said "Ready to train ·
    # Starting now", then "Getting ready · measuring … 1024 of 8192 tokens".
    from songe.sleep import calibration_state
    from songe.wording import status_copy

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.set_learning(store, config.base_identifier, "all")
    registry.support(store, config.base_identifier, "trainable")
    models = registry.summaries(config, store)
    monkeypatch.setattr(calibration_state, "needs_growth", lambda config, store: True)
    monkeypatch.setattr("songe.status.time.monotonic", lambda: 1000)
    worker = scheduler(config, last_traffic=900, process=None)
    worker.reason = reason
    worker.idle_threshold_seconds = 10
    worker.choose_model = lambda: config.base_identifier
    result = activity(config, store, worker, None, models, live=True)
    assert result["next_step"] == "measure"
    assert status_copy({"activity": result}) == ("Ready to measure", detail)


def test_training_step_eta_and_next_gate(config, store):
    models = learner(config, store)
    training = {
        "model": "base-test",
        "phase": "train",
        "step": 41,
        "total_steps": 120,
        "data_progress": {"eta_seconds": 180},
    }
    result = activity(config, store, scheduler(config), training, models, live=True)
    assert result["now"] == "training base-test · step 41/120 · ~3m left"
    assert result["training"]
    assert result["next"] == "check base-test · current work completes"
    training["phase"] = "gate"
    result = activity(config, store, scheduler(config), training, models, live=True)
    assert result["next"] == "swap base-test · checks pass + no proxy requests"
    assert not result["training"]


@pytest.mark.parametrize(
    "phase", ["calibrate", "score", "train", "distill", "check", "fuse", "swap"]
)
def test_only_active_training_has_an_accent(config, store, phase):
    models = learner(config, store)
    worker = scheduler(config)
    training = {"phase": phase}
    result = activity(config, store, worker, training, models, live=True)
    assert result["training"] == (phase in {"train", "distill"})
    worker.cancelled_at = 1
    assert not activity(config, store, worker, training, models, live=True)["training"]
    worker.cancelled_at = None
    store.set("paused", True)
    assert not activity(config, store, worker, training, models, live=True)["training"]
    assert not activity(config, store, worker, training, models, live=False)["training"]


@pytest.mark.parametrize(
    "reason,probe,expected",
    [
        ("Waiting for experience", None, "no new conversations through songe yet"),
        (
            "Waiting: battery low",
            Probes(power=Power(ac=False, fraction=0.08)),
            "battery 8%",
        ),
        (
            "Waiting for free memory: legacy bytes",
            Probes(free_gb=1.9),
            "memory · 1.9 GB free, needs 3.2 GB",
        ),
    ],
)
def test_concrete_wait_reasons(config, store, reason, probe, expected):
    config.calibrated_peak_gb = 3.2
    models = learner(config, store)
    worker = scheduler(config, last_probe=probe)
    worker.config = config
    worker.reason = reason
    result = activity(config, store, worker, None, models, live=True)
    assert result["now"] == "waiting · " + expected
    if reason == "Waiting for experience":
        # Data, not idle time, is what calibration waits on.
        assert result["next"] == "calibrate base-test · needs data"
    elif expected.startswith("memory"):
        assert result["next"] == "calibrate base-test · when memory frees"
    else:
        assert result["next"].endswith("→ calibrate base-test")
    assert expected not in result["next"]


def test_pause_offline_and_memory_retry(config, store):
    models = learner(config, store)
    models[0]["reason"] = "calibration paused · memory (pressure) · retry when idle"
    result = activity(config, store, scheduler(config), None, models, live=True)
    assert result["now"] == "waiting · calibration paused · memory · retry when idle"
    assert result["next"] == "calibrate base-test · when memory frees"
    assert not result["training"]
    store.set("paused", True)
    result = activity(config, store, scheduler(config), None, models, live=True)
    assert result["now"] == "waiting · learning paused"
    assert result["paused"] == "yes · resume"
    assert result["next"] == "resume → calibrate base-test"
    result = activity(
        config, store, scheduler(config), {"phase": "train"}, models, live=False
    )
    assert result["now"] == "waiting · service stopped · songe serve"


@pytest.mark.parametrize(
    "feedback,counts,expected",
    [
        ((20, 5), (16, 1, 16), "collecting conversations · 20 of 32"),
        (
            (39, 2),
            (39, 0, 39),
            "collecting conversations · 2 of 3 needed to check updates",
        ),
        ((39, 9), (31, 2, 2), "collecting conversations · 2 of 8 new examples"),
        ((39, 9), (31, 2, 10), "preparing examples · 10 examples waiting to learn"),
    ],
)
def test_conversation_wait_names_only_the_unmet_need(
    config, store, monkeypatch, feedback, counts, expected
):
    # One pool of collected data; tests are set aside from it.
    config.min_train, config.min_heldout, config.min_new = 32, 2, 8
    monkeypatch.setattr(store, "traffic_times", lambda limit: [1])
    monkeypatch.setattr(
        store,
        "scoped",
        lambda model: SimpleNamespace(
            experience_counts=lambda: counts,
            feedback_counts=lambda: feedback,
            get=lambda *a: None,
            model_key=None,
        ),
    )
    models = [{"id": config.base_identifier, "enabled": True, "support": "trainable"}]
    worker = scheduler(config)
    worker.reason = "Waiting for experience"
    result = activity(config, store, worker, None, models, live=True)
    assert result["now"] == "waiting · " + expected
    if counts[2] >= config.min_new and counts[1] >= config.min_heldout:
        assert "idle" in result["next"]
    else:
        assert "needs" in result["next"] and "idle" not in result["next"]


async def test_state_etag_revisions_and_section_deltas(config, store, monkeypatch):
    scans = []
    monkeypatch.setattr(library, "scan", lambda config: scans.append(True) or [])
    worker = SimpleNamespace(reason="scheduler starting")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(config, store, scheduler=worker)),
        base_url="http://localhost",
    ) as client:
        first = await client.get("/api/state")
        assert first.status_code == 200
        assert first.headers["cache-control"] == "private, no-cache"
        revisions = first.json()["revisions"]
        assert set(revisions) == {"status", "models", "learning", "versions", "cycle"}
        response = await client.get(
            "/api/state", headers={"If-None-Match": first.headers["etag"]}
        )
        assert response.status_code == 304 and not response.content
        response = await client.get(
            "/api/state", params={"revisions": json.dumps(revisions)}
        )
        assert response.status_code == 304
        store.set("paused", True)
        response = await client.get(
            "/api/state",
            params={"revisions": json.dumps(revisions)},
            headers={"If-None-Match": first.headers["etag"]},
        )
        delta = response.json()
        assert set(delta["sections"]) == {"status"}
        assert delta["sections"]["status"]["activity"]["paused"] == "yes · resume"
        revisions = delta["revisions"]
        with store.connect() as db:
            db.execute(
                "INSERT INTO scoreboard(day,model_version,repeat_error_rate) VALUES('2026-11-10','base-test',.2)"
            )
        delta = (
            await client.get("/api/state", params={"revisions": json.dumps(revisions)})
        ).json()
        assert set(delta["sections"]) == {"learning"}
        revisions = delta["revisions"]
        store.night(None, status="rejected", record={"reason": "kept previous"})
        delta = (
            await client.get("/api/state", params={"revisions": json.dumps(revisions)})
        ).json()
        assert set(delta["sections"]) == {"learning", "versions", "cycle"}  # a finished night is the last cycle
        assert (
            await client.get("/api/state", params={"revisions": "[]"})
        ).status_code == 400
        assert (
            await client.get("/api/state", params={"revisions": "{"})
        ).status_code == 400
    assert scans == [True]


async def test_inventory_cached_even_without_scheduler(config, store, monkeypatch):
    scans = []
    monkeypatch.setattr(library, "scan", lambda config: scans.append(True) or [])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(config, store)),
        base_url="http://localhost",
    ) as client:
        for _ in range(3):
            assert (await client.get("/api/state")).status_code == 200
    assert scans == [True]


def test_model_formats_merge_but_keep_file_bindings():
    models = [
        {
            "id": "mlx",
            "name": "mlx-community/Qwen3.5-4B-4bit",
            "format": "mlx",
            "enabled": True,
            "size_bytes": 3,
            "versions": [{"id": "v1"}],
        },
        {
            "id": "gguf",
            "name": "publisher/Qwen3.5-4B-Q4_K_M.gguf",
            "format": "gguf",
            "size_bytes": 4,
            "versions": [],
        },
        {
            "id": "8b",
            "name": "Qwen3.5-8B-Q4_K_M.gguf",
            "format": "gguf",
            "versions": [],
        },
    ]
    grouped = library.group_models(models)
    assert len(grouped) == 2
    assert grouped[0]["name"] == "Qwen3.5-4B"
    assert grouped[0]["size_bytes"] == 7
    assert grouped[0]["format"] == "mlx, gguf"
    assert [f["id"] for f in grouped[0]["files"]] == ["mlx", "gguf"]
    assert grouped[0]["versions"][0]["model"] == "mlx"


def test_embedding_and_gguf_reason(config, tmp_path, monkeypatch):
    platform = SimpleNamespace(
        memory=lambda: Memory(10**10, 10**10),
        accelerators=lambda: [Accel("metal", "fake")],
    )
    config.backend = "mlx"
    path = tmp_path / "lfm.gguf"
    path.touch()
    monkeypatch.setattr(
        library.discovery,
        "read_gguf",
        lambda p: SimpleNamespace(parameters=1, embedding=False),
    )
    monkeypatch.setattr(library, "architecture_available", lambda *a: False)
    ref = ModelRef("lmstudio", "LFM3-3B-Q4.gguf", "gguf", path, "lfm3")
    state, reason = library.support(config, ref, 1, platform)
    assert (state, reason) == ("unsupported", "lfm3 not supported yet")
    ref = ModelRef("lmstudio", "nomic-embed-text", "gguf", path, "bert")
    assert library.support(config, ref, 1, platform) == (
        "unsupported",
        "embedding model",
    )


async def test_hashed_assets_are_immutable(config, store):
    import re

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(config, store)),
        base_url="http://localhost",
    ) as client:
        html = await client.get("/")
        hashed = re.findall(r'(?:src|href)="(/assets/[^"]+\?v=[0-9a-f]{12})"', html.text)
        assert len(hashed) == 2  # the page's script and stylesheet
        for url in hashed:
            asset = await client.get(url)
            assert (
                asset.headers["cache-control"] == "public, max-age=31536000, immutable"
            )
            assert (
                await client.get(url, headers={"If-None-Match": asset.headers["etag"]})
            ).status_code == 304
        assert (await client.get("/assets/next.js?v=old")).headers[
            "cache-control"
        ] == "no-cache"


def test_hidden_status_lines_stay_hidden_at_phone_width():
    """At phone width an empty GPU label showed with no GPU line, since
    the narrow layout's `display: grid` beat the `hidden` attribute."""
    import re
    from pathlib import Path

    css = (Path(__file__).resolve().parents[1] / "src/songe/ui/next.css").read_text(
        encoding="utf-8"
    )
    phone = css.split("@media (width < 46rem)", 1)[1]
    assert re.search(r"\.status p \{[^}]*display: grid", phone)
    assert re.search(r"\.status p\[hidden\] \{ display: none; \}", phone)


async def test_real_model_scheduler_reports_idle_then_experience_then_off(
    config, store
):
    import time

    from songe.schedule import ModelScheduler

    learner(config, store)
    worker = ModelScheduler(config, store, None, SimpleNamespace())
    worker.last_traffic = time.monotonic() - 4
    await worker.tick()
    state = snapshot(config, store, worker, live=True)
    assert state["activity"]["now"] == "waiting · proxy idle 4s/10s"
    assert state["activity"]["next"] == "idle 4s/10s → calibrate base-test"
    registry.support(store, config.base_identifier, "trainable")
    worker.last_traffic = time.monotonic() - 1000
    await worker.tick()
    assert worker.reason == "no new conversations through songe yet"
    assert (
        snapshot(config, store, worker, live=True)["activity"]["now"]
        == "waiting · no new conversations through songe yet"
    )
    # Nothing to run: a quiet-time countdown would promise a start that
    # cannot happen, so the wait named is still the data.
    worker.reason = "Waiting for an idle gap"
    assert (
        snapshot(config, store, worker, live=True)["activity"]["now"]
        == "waiting · no new conversations through songe yet"
    )
    # An old memory stop is cleared while the real wait is for conversations.
    registry.support(
        store,
        config.base_identifier,
        "pending",
        "calibration paused · memory · retry when idle",
    )
    await worker.tick()
    row = registry.find(store, config.base_identifier)
    assert "memory" not in (row.get("reason") or "")
    registry.set_learning(store, config.base_identifier, "off")
    await worker.tick()
    assert worker.reason == "learning off for all models"


def test_api_uses_recorded_optimizer_steps_from_frozen_plan(config, store):
    learner(config, store)
    store.scoped(config.base_identifier).night(
        None,
        phase="distill",
        record={
            "data_progress": {
                "phase": "train",
                "completed": 82,
                "total": 240,
                "unit": "samples",
                "step": 41,
                "total_steps": 120,
                "eta_seconds": 180,
            }
        },
    )
    config.accumulation = 10
    state = snapshot(config, store, scheduler(config, process=object()), live=True)
    assert state["training"]["step"] == 41
    assert state["training"]["total_steps"] == 120
    assert state["activity"]["now"] == "training base-test · step 41/120 · ~3m left"


def test_battery_and_memory_checkpoint_stops_take_priority(config, store):
    models = learner(config, store)
    worker = scheduler(
        config, cancelled_at=1, last_probe=Probes(power=Power(ac=False, fraction=0.08))
    )
    worker.reason = "Stopping: battery low; saving checkpoint"
    training = {"phase": "train", "step": 2, "total_steps": 10}
    result = activity(config, store, worker, training, models, live=True)
    assert result["now"] == "waiting · pausing · battery 8%"
    assert result["next"].endswith("continues when idle")
    worker.memory_stop_sample = object()
    training["phase"] = "calibrate"
    result = activity(config, store, worker, training, models, live=True)
    assert result["now"] == "waiting · pausing · not enough memory"


def test_registered_embeddings_are_not_listed_and_stay_unscheduled(config, store, tmp_path):
    path = tmp_path / "nomic-embed-text.gguf"
    ref = ModelRef("lmstudio", "nomic-embed-text", "gguf", path, "bert")
    registry.register(config, store, ref.serving_id, ref)
    row = registry.find(store, ref.serving_id)
    assert row["support"] == "unsupported"
    assert row["reason"] == "embedding model"
    # Live (#72): `songe models` listed it as "Not supported yet · Embedding
    # model"; it can never chat, so it is not listed at all.
    state = snapshot(config, store)
    assert ref.serving_id not in {m["id"] for m in state["models"]}
    assert all(m.get("reason") != "embedding model" for m in state["models"])
    registry.support(store, ref.serving_id, "pending", "legacy calibration pending")
    registry.migrate(config, store)
    assert registry.find(store, ref.serving_id)["reason"] == "embedding model"


async def test_training_progress_only_changes_status_section(config, store):
    learner(config, store)
    scoped = store.scoped(config.base_identifier)
    night = scoped.night(
        None,
        phase="distill",
        record={
            "data_progress": {
                "phase": "train",
                "completed": 2,
                "total": 20,
                "unit": "samples",
                "step": 1,
                "total_steps": 10,
            }
        },
    )
    worker = scheduler(config, process=object())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(config, store, scheduler=worker)),
        base_url="http://localhost",
    ) as client:
        first = (await client.get("/api/state")).json()
        scoped.night(
            night,
            record={
                "data_progress": {
                    "phase": "train",
                    "completed": 4,
                    "total": 20,
                    "unit": "samples",
                    "step": 2,
                    "total_steps": 10,
                }
            },
        )
        delta = (
            await client.get(
                "/api/state", params={"revisions": json.dumps(first["revisions"])}
            )
        ).json()
        assert set(delta["sections"]) == {"status", "cycle"}  # the cycle record shows training steps too
        assert "step 2/10" in delta["sections"]["status"]["activity"]["now"]


def test_ready_version_waits_for_swap_and_gate_hides_old_training_progress(
    config, store
):
    models = learner(config, store)
    worker = scheduler(config, in_flight={"admission": 1})
    training = {
        "phase": "swap",
        "step": 120,
        "total_steps": 120,
        "data_progress": {
            "phase": "train",
            "completed": 240,
            "total": 240,
            "unit": "samples",
        },
    }
    result = activity(
        config,
        store,
        worker,
        training,
        models,
        live=True,
        ready={"model_id": config.base_identifier},
    )
    assert result["now"] == "waiting · proxy requests in flight"
    assert result["next"] == "swap base-test · waits for idle gap + no proxy requests"
    worker.in_flight = {}
    result = activity(
        config,
        store,
        worker,
        training,
        models,
        live=True,
        ready={"model_id": config.base_identifier},
    )
    assert result["now"] == "waiting · checks passed · swap pending"
    training["phase"] = "check"
    assert (
        activity(config, store, worker, training, models, live=True)["now"]
        == "evaluating base-test"
    )


def test_pipeline_contract_survives_snapshot_and_section_revisions(config, store):
    from songe.api import section_revisions, state_sections
    from songe.panels import panel_rows
    from tests.test_learning_summary import experience
    from tests.test_store import version

    experience(store)
    learner(config, store)
    scoped = store.scoped(config.base_identifier)
    record = {
        "consolidate_started": True,
        "consolidate_seconds": 7,
        "data_progress": {
            "phase": "train",
            "step": 2,
            "total_steps": 4,
            "eta_seconds": 5,
        },
    }
    night = scoped.night(None, record=record)
    scoped.add_version(version("base"))
    scoped.swap("base", None)
    scoped.add_version(version("learned", night) | {"kind": "peft-adapter"})
    scoped.set("wake_last", {"at": 10, "drift": 0.01, "swapped": False})
    # A second format remains a separate learner even in a grouped inventory row.
    registry.register(config, store, "base-test-Q4_K_M.gguf")
    state = snapshot(config, store)
    model = state["models"][0]
    expected = scoped.pipeline()
    assert len(state["models"]) == 1 and len(model["files"]) == 2
    assert model["pipeline"] == expected == state["pipeline"][model["id"]]
    assert all(f["pipeline"] == state["pipeline"][f["id"]] for f in model["files"])
    other = next(f for f in model["files"] if f["id"] != model["id"])
    assert state["pipeline"][other["id"]]["day"]["steps"] == 0
    assert expected["rewind"]["signals"]["tool_error_fixed"] > 0
    assert "did not pass checks" in dict(panel_rows(model))["result"]
    sections = state_sections(state)
    before = section_revisions(sections)
    assert "pipeline" not in sections["models"]["models"][0]
    assert all("pipeline" not in f for f in sections["models"]["models"][0]["files"])
    assert "pipeline" not in sections["learning"]["models"][0]
    assert sections["status"]["pipeline"] == state["pipeline"]
    assert sections["versions"]["models"][0]["versions"] == expected["versions"]
    record["consolidate_seconds"] += 1
    scoped.night(night, record=record)
    changed = section_revisions(state_sections(snapshot(config, store)))
    assert {k for k in before if before[k] != changed[k]} == {"status"}
    before = changed
    scoped.add_version(version("retained", night) | {"kind": "peft-adapter"})
    changed = section_revisions(state_sections(snapshot(config, store)))
    assert {k for k in before if before[k] != changed[k]} == {"status", "versions", "cycle"}  # the cycle lists versions


def test_empty_store_pipeline_keeps_counts_and_unknown_measurements(config, store):
    learner(config, store)
    data = snapshot(config, store)
    model = data["models"][0]
    metrics = model["pipeline"]
    assert metrics["day"] == {"conversations": 0, "steps": 0, "heldout": 0}
    assert metrics["consolidate"] == {
        "runs": 0,
        "step": 0,
        "total": 0,
        "eta_seconds": None,
        "last_seconds": None,
    }
    assert metrics["wake"] == {"last": None}
    assert metrics["versions"] == []


def test_page_times_use_a_24_hour_clock():
    """Times the page writes sit beside 24-hour times Songe words itself
    ("since your last Quick Dream at 16:00"): one clock, whatever the
    browser's language."""
    import re
    from pathlib import Path

    import songe

    script = (Path(songe.__file__).parent / "ui" / "next.js").read_text(encoding="utf-8")
    calls = re.findall(r"toLocaleTimeString\(([^)]*)\)", script)
    assert calls and all("hourCycle: 'h23'" in call for call in calls)


def test_waiting_for_the_agents_tools_says_so(config, store, monkeypatch):
    """Reported by Jack in a test on 11-24: after an answer that asked for
    tools, Songe waits up to five minutes for the agent to call back, and the
    status said only "proxy requests in flight", which looked stuck."""
    from songe.wording import readable, status_copy

    models = learner(config, store)
    monkeypatch.setattr("songe.status.time.monotonic", lambda: 1000)
    worker = scheduler(config, last_traffic=1000 + 250, in_flight={}, process=None)
    result = activity(config, store, worker, None, models, live=True)
    assert result["now"] == (
        "waiting · proxy requests in flight · waiting up to 5 min for the agent's tool results")
    assert status_copy({"activity": result}) == (
        "In use", "The agent is running tools · learning waits up to 5 min")
    assert "conversations in progress · waiting up to 5 min" in readable(result["now"])
    # A request still streaming: said as before.
    worker.in_flight = {"admission": 1}
    result = activity(config, store, worker, None, models, live=True)
    assert result["now"] == "waiting · proxy requests in flight"
    assert status_copy({"activity": result}) == ("In use", "Continues after this conversation")


PAGE = Path(__file__).resolve().parents[1] / "src/songe/ui"


def js_strings(js):
    """String pieces of a JS source: quoted strings and template text."""
    out, modes, braces, i, n = [], ["code"], [], 0, len(js)
    while i < n:
        c, mode = js[i], modes[-1]
        if mode == "template":
            j = i
            while j < n and js[j] != "`" and not js.startswith("${", j):
                j += 2 if js[j] == "\\" else 1
            out.append(js[i:j])
            if js.startswith("${", j):
                modes.append("code")
                braces.append(0)
                i = j + 2
            else:
                modes.pop()
                i = j + 1
            continue
        if js.startswith("//", i) and (i == 0 or js[i - 1] in " \t\n"):
            i = js.find("\n", i) % (n + 1)
            continue
        if js.startswith("/*", i):
            i = js.index("*/", i) + 2
            continue
        if c in "'\"":
            j = i + 1
            while js[j] != c:
                j += 2 if js[j] == "\\" else 1
            out.append(js[i + 1 : j])
            i = j + 1
            continue
        if c == "`":
            modes.append("template")
        elif c == "{" and braces:
            braces[-1] += 1
        elif c == "}" and braces:
            if braces[-1] == 0:
                braces.pop()
                modes.pop()
            else:
                braces[-1] -= 1
        i += 1
    return out

STOP = {"of", "the", "to", "is", "it", "by", "for", "after", "left", "yet", "not", "and",
        "or", "with", "at", "its", "your", "more", "than", "from", "about", "this", "no", "in"}

def reads_as_english(piece):
    """Words a person reads: a capitalised word, or a phrase with a small word."""
    tokens = re.findall(r"[\w:'-]+", piece)
    phrase = any(c.isspace() for c in piece) and any(t in STOP for t in tokens)
    return phrase or bool(re.search(r"(?<![\w-])[A-Z][a-z]+", piece))


def test_the_page_takes_its_words_from_wording_json():
    """Words written into next.js missed every wording pass ("Show all 12",
    "+ 3 more · learning off" on models that were unsupported, not off)."""
    script = (PAGE / "next.js").read_text(encoding="utf-8")
    # Before wording.json loads, and the key names a browser gives.
    allowed = {"Songe not answering", "Run songe serve", "Escape"}
    english = [s for s in js_strings(script) if reads_as_english(s) and s not in allowed]
    assert english == []


def test_every_word_the_page_asks_for_exists():
    script = (PAGE / "next.js").read_text(encoding="utf-8")
    words = json.loads((PAGE / "wording.json").read_text(encoding="utf-8"))
    groups = {"C": "cloud", "R": "rail", "T": "train_state", "E": "empty"}
    missing = [key for key in re.findall(r"\bW\.(\w+)", script) if key not in words]
    for alias, group in groups.items():
        missing += [f"{group}.{key}" for key in re.findall(rf"\b{alias}\.(\w+)", script)
                    if key not in words[group]]
    assert missing == []


STATUSES = json.loads(
    (Path(__file__).parent / "fixtures/wording.json").read_text(encoding="utf-8")
)["statuses"]


@pytest.mark.parametrize("case", STATUSES, ids=lambda c: c["name"])
def test_the_page_and_songe_status_say_the_same_now_and_next(config, store, case, capsys):
    """The page worded Now and Next itself: "Dreams again when your computer
    is idle" where `songe status` said "Ready to train · Starts when idle ·
    8m of 10m". Both now take the lines the service words once."""
    from songe import cli
    from songe.api import state_sections

    learner(config, store)
    data = snapshot(config, store, scheduler(config, last_traffic=0, process=None), live=True)
    data |= case["data"]
    page = state_sections(data)["status"]["lines"]
    cli.print_status(data)
    printed = dict(line.split(None, 1) for line in capsys.readouterr().out.splitlines()[:2])
    assert page == {"now": printed["now"].removeprefix("● "), "next": printed["next"]}


def test_the_page_shows_the_services_now_and_next_lines():
    script = (PAGE / "next.js").read_text(encoding="utf-8")
    assert "function statusLines" not in script
    assert "status?.lines?.now" in script and "status?.lines?.next" in script


def test_the_page_counts_idle_time_toward_a_dream():
    """A model waiting for idle time said only "dreams when idle"; the status
    sends how long it has been idle and how long it needs."""
    script = (PAGE / "next.js").read_text(encoding="utf-8")
    words = json.loads((PAGE / "wording.json").read_text(encoding="utf-8"))
    assert "a.idle_elapsed_seconds" in script and "a.idle_threshold_seconds" in script
    assert words["empty"]["next_idle"] == "dreams when idle · {idle}, or press Dream"
    assert words["idle_of"] == "{elapsed} of {limit}"
