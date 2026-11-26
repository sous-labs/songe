"""Multi-model regressions: fake runtimes, temporary files, no training/server."""

import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest

from songe.api import LocalClient, snapshot
from songe.config import Config, training_fingerprint
from songe.log import SCHEMA, SCHEMA_V2, Store
from songe.models import ModelRef, registry
from songe.proxy import create_app
from songe.schedule import ModelScheduler
from songe.sleep.select import select
from tests.test_proxy import Chunks
from tests.test_store import version


@pytest.fixture
def learners(config, store):
    config.model_ref = {
        "upstream": config.upstream,
        "serving_id": "first",
        "format": "safetensors",
        "path": config.model,
        "architecture": "fixture",
    }
    config.base_identifier = "first"
    config.calibration_available = True
    config.save(store.root)
    store.add_version(version("base"))
    store.swap("base", None)
    registry.migrate(config, store)
    ref = ModelRef(
        config.upstream, "second", "safetensors", store.root / "second", "fixture"
    )
    registry.register(config, store, "second", ref)
    registry.support(store, "second", "trainable")
    # These regressions exercise shared hindsight, explicitly selected now.
    for model in ("first", "second"):
        registry.set_learning(store, model, "all")
    return config, store


def evidence(count=5):
    return [
        {
            "id": str(i),
            "conversation": f"task-{i}",
            "split": "train",
            "source_id": i,
            "context": [{"role": "user", "content": "question"}],
            "tools": [],
            "hindsight": [{"role": "user", "content": "correction"}],
        }
        for i in range(1, count + 1)
    ]


async def test_proxy_follows_second_model_and_off_only_forwards(learners, monkeypatch):
    from songe import memory

    config, store = learners
    seen = []
    monkeypatch.setattr(
        memory, "guarded_run", lambda *a, **kw: SimpleNamespace(returncode=0)
    )

    async def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            stream=Chunks(
                [
                    json.dumps(
                        {
                            "choices": [
                                {"message": {"role": "assistant", "content": "ok"}}
                            ]
                        }
                    ).encode()
                ]
            ),
        )

    second = store.scoped("second")
    second.add_version(version("night-2", 2) | {"identifier": "second-learned"})
    second.swap("night-2", second.get("current"))
    wire = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    app = create_app(config, store, client=wire)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://localhost"
        ) as client:
            for model in (config.public_model, "second", "third"):
                assert (
                    await client.post(
                        "/v1/chat/completions", json={"model": model, "messages": []}
                    )
                ).status_code == 200
            assert seen[0]["model"] == "base"
            assert seen[1]["model"] == "second-learned"
            assert seen[2]["model"] == "third"
            assert registry.find(store, "third")["enabled"] is True
            response = await client.post(
                "/api/control",
                json={"action": "model", "model": "second", "enabled": False},
            )
            assert response.status_code == 200
            await client.post(
                "/v1/chat/completions", json={"model": "second", "messages": []}
            )
            assert seen[-1]["model"] == "second"
            assert second.get("current") == "night-2"
            assert (
                await client.post(
                    "/api/control",
                    json={"action": "model", "model": "second", "enabled": True},
                )
            ).status_code == 200
            await client.post(
                "/v1/chat/completions", json={"model": "second", "messages": []}
            )
            assert seen[-1]["model"] == "second-learned"
    await wire.aclose()
    assert len(store.exchanges()) == 5  # Includes the request made while off.


def test_shared_experience_is_selected_again_for_new_model(learners):
    config, store = learners
    steps = evidence()
    store.sync_steps(steps)
    first = store.scoped("first")
    first.mark_scored(steps)
    first.set("experience_watermark", 5)
    first.add_pool([s | {"surprise": 1} for s in steps])
    second = store.scoped("second")
    assert first.experience_counts()[2] == 0
    assert second.experience_counts()[2] == 5
    assert second.pool() == []
    selected, _ = select(second.unscored(second.active_steps("train")))
    assert [s["id"] for s in selected] == [s["id"] for s in steps]
    second.add_pool(selected)
    second.mark_scored(steps)
    # Each model keeps its own pool rows over the one shared evidence log.
    assert first.pool()[0]["surprise"] == 1
    assert [s["id"] for s in second.pool()] == [s["id"] for s in steps]
    assert first.experience_counts()[2] == second.experience_counts()[2] == 0
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 5
    heldout = evidence(1)[0] | {"id": "held", "split": "heldout"}
    store.sync_steps([heldout])
    assert first.active_steps("heldout") == second.active_steps("heldout")
    with pytest.raises(ValueError, match="training conversations"):
        second.add_pool([heldout | {"surprise": 2}])


def test_a_rebuild_without_data_is_not_work(learners):
    config, store = learners
    scheduler = ModelScheduler(
        config, store, SimpleNamespace(), platform=SimpleNamespace()
    )
    for model in ("first", "second"):
        registry.support(store, model, "trainable")
    assert scheduler.choose_model() is None
    # After forget there is nothing left to rebuild from: it waits for
    # conversations instead of counting quiet time toward a start.
    store.scoped("first").set("rebuild_required", True)
    assert scheduler.choose_model() is None
    # With enough recorded, the rebuild is the next work.
    store.sync_steps(evidence())
    store.scoped("first").mark_scored(evidence())
    store.scoped("second").mark_scored(evidence())
    assert scheduler.choose_model() is None
    policy = scheduler.config_for("first")
    policy.min_train = 1
    scheduler.config_for = lambda model: policy
    assert scheduler.choose_model() == "first"


async def test_dream_says_what_is_missing_and_the_asked_model_goes_first(
    learners, monkeypatch
):
    import songe.schedule as schedule

    config, store = learners
    scheduler = ModelScheduler(
        config, store, SimpleNamespace(), platform=SimpleNamespace()
    )
    for model in ("first", "second"):
        registry.support(store, model, "trainable")
    probes = schedule.Probes(available=True, thermal=0, power=schedule.Power(ac=True))
    monkeypatch.setattr(schedule, "probe", lambda platform=None: probes)
    wait = await scheduler.training_wait("first")
    assert wait.startswith("needs ") and wait.endswith(" more examples")
    store.set("paused", True)
    assert scheduler.dream_blocked("first") == "learning is paused"
    store.set("paused", False)
    registry.set_learning(store, "second", "off")
    assert scheduler.dream_blocked("second") == "learning is off for this model"
    registry.set_learning(store, "second", "new")
    scheduler.in_flight = {"request": 1}
    assert scheduler.dream_blocked("first") == "the model is in use"
    assert (await scheduler.dream("first")) == {
        "started": False,
        "message": "Not yet · the model is in use",
    }
    scheduler.in_flight = {}
    assert scheduler.dream_blocked("first") is None
    # With nothing recorded there is nothing to read, and training says why.
    reply = await scheduler.dream("first")
    assert not reply["started"] and store.get("requested_sleep") is not True
    assert reply["message"].startswith("Nothing to read yet · training: needs ")
    # A leftover memory wait with no request pending does not block a press
    # (live: every press answered "already running"); a pending request does.
    store.scoped("first").set("memory_wait", {"try": 1, "tries": 3, "available": 1, "total": 2})
    reply = await scheduler.dream("first")
    assert not reply["message"].startswith("A Deep Dream is already running")
    # A request left after an interruption waits for a long quiet period with
    # nothing running (live, Inkling on Modal: every press answered "already
    # running"): a press re-arms it instead of being refused.
    store.set("requested_model", "first")
    store.set("requested_sleep", False)
    reply = await scheduler.dream("first")
    assert not reply["message"].startswith("A Deep Dream is already running")
    # An armed request is a Dream under way.
    store.set("requested_sleep", True)
    reply = await scheduler.dream("first")
    assert reply == {"started": False, "message": reply["message"]} and reply["message"].startswith(
        "A Deep Dream is already running · ")
    store.set("requested_model", None)
    store.set("requested_sleep", False)
    store.scoped("first").set("memory_wait", None)
    # Measuring is the first step of learning: nothing else is waited on.
    for model in ("first", "second"):
        registry.support(store, model, "pending", "calibration required")
    assert await scheduler.training_wait("second") is None
    probes = schedule.Probes(
        available=True, thermal=0, power=schedule.Power(ac=False, fraction=0.05)
    )
    assert await scheduler.training_wait("second") == "battery low"
    # Both have work; the one asked for is chosen.
    assert scheduler.choose_model() == "first"
    store.set("requested_model", "second")
    assert scheduler.choose_model() == "second"
    assert store.scoped("second").get("requested_model") == "second"  # shared


async def test_dream_reads_replies_now_however_few(learners, monkeypatch):
    import songe.review
    import songe.schedule as schedule

    config, store = learners
    scheduler = ModelScheduler(
        config,
        store,
        SimpleNamespace(),
        platform=SimpleNamespace(memory=lambda: SimpleNamespace(pressure="normal")),
    )
    for model in ("first", "second"):
        registry.support(store, model, "trainable")
    probes = schedule.Probes(available=True, thermal=0, power=schedule.Power(ac=True))
    monkeypatch.setattr(schedule, "probe", lambda platform=None: probes)
    store.sync_steps(evidence(2))  # Two steps answered by the same reply.
    batches = []

    async def read(store, url, model, **kwargs):
        batches.append(model)
        return 1 if len(batches) == 1 else 0

    monkeypatch.setattr(songe.review, "review", read)
    # Far too little to train on, and still worth reading now.
    reply = await scheduler.dream("first")
    assert reply["started"]
    # Said at the press: it prepares now, and training cannot start yet.
    assert reply["message"].startswith("Deep Dream · preparing practice now · training can't start yet: ")
    await scheduler.dream_task
    assert len(batches) == 2  # batch after batch until none is left
    assert store.get("requested_sleep") is not True
    assert ("review", "1 replies") in [(e["stage"], e["text"]) for e in store.events()]


async def test_a_model_the_server_refuses_is_not_reloaded_on_every_retry(
    learners, monkeypatch
):
    """Live: an agent defaulting to weights LM Studio cannot load retried every
    20 s, and each retry started another doomed load on the user's server."""
    import songe.schedule as schedule

    class LMStudioServerError(Exception):
        pass

    config, store = learners
    loads = []

    async def ensure(version):
        loads.append(version["id"])
        if len(loads) != 3:
            raise LMStudioServerError(
                "Model load error: Failed to load model. Reported cause: Traceback "
                '(most recent call last): File "/x.py" ValueError: Model type lfm3 '
                "not supported. Error: No module named 'drafters.lfm3'"
            )

    scheduler = ModelScheduler(
        config, store, SimpleNamespace(ensure=ensure), platform=SimpleNamespace()
    )
    version = {"id": "second-binding", "identifier": "second"}
    with pytest.raises(schedule.LoadRefused) as refused:
        await scheduler.ensure_model("second", version)
    assert str(refused.value) == (
        "the model server could not load second (Model type lfm3 not supported)"
        " · choose another model in your agent"
    )
    assert schedule.refusal_reason(
        Exception("Failed to load model. Reported cause: x")
    ) == ("Failed to load model")
    with pytest.raises(schedule.LoadRefused):
        await scheduler.ensure_model("second", version)
    assert len(loads) == 1  # the retry was answered without loading
    # Later it is tried again, and a load that works clears the memory of it.
    now = schedule.time.monotonic()
    monkeypatch.setattr(
        schedule.time, "monotonic", lambda: now + schedule.LOAD_RETRY_SECONDS + 1
    )
    with pytest.raises(schedule.LoadRefused):
        await scheduler.ensure_model("second", version)
    assert len(loads) == 2
    scheduler.load_refused.clear()
    await scheduler.ensure_model("second", version)
    assert len(loads) == 3 and not scheduler.load_refused
    assert scheduler.serving_model == "second"


def test_scheduler_picks_pending_then_least_recent_and_skips_off_unsupported(learners):
    config, store = learners
    store.sync_steps(evidence())
    scheduler = ModelScheduler(
        config, store, SimpleNamespace(), platform=SimpleNamespace()
    )
    store.scoped("first").set("last_completed", 200)
    store.scoped("second").set("last_completed", 100)
    assert scheduler.choose_model() == "second"
    store.scoped("second").mark_scored(evidence()[:1])
    assert scheduler.choose_model() == "first"
    registry.enable(store, "first", False)
    assert scheduler.choose_model() == "second"
    registry.support(
        store,
        "second",
        "unsupported",
        "Backward pass: architecture fixture is not supported",
    )
    assert scheduler.choose_model() is None
    state = snapshot(config, store)
    assert state["models"][0]["status"] == "off"
    assert state["models"][1]["status"] == "unsupported"
    assert "architecture" in state["models"][1]["reason"]


def test_undo_is_per_model_and_requires_choice_when_ambiguous(learners, monkeypatch):
    config, store = learners

    def offline(*a, **kw):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.Client, "get", offline)
    for model, key in (("first", "night-1"), ("second", "night-2")):
        scoped = store.scoped(model)
        scoped.add_version(version(key, 1))
        scoped.swap(key, scoped.get("current"))
    client = LocalClient(store.root, config)
    with pytest.raises(ValueError, match="Choose a model"):
        client.control("undo")
    client.control("undo", model="second")
    assert store.scoped("second").version()["id"].startswith("base-")
    assert store.scoped("first").get("current") == "night-1"
    assert store.scoped("second").get("previous") == "night-2"
    with pytest.raises(ValueError, match="Unknown version"):
        store.scoped("second").swap("night-1", store.scoped("second").get("current"))


def test_existing_single_model_database_migrates_without_losing_undo(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    config = Config(model=str(root / "weights"), base_identifier="legacy")
    config.save(root)
    # An actual pre-feature v2 database, including its original three-column pool.
    with sqlite3.connect(root / "experience.sqlite3") as db:
        db.executescript(SCHEMA + SCHEMA_V2)
        db.execute("PRAGMA user_version=2")
        for name in ("base", "old-update"):
            db.execute(
                "INSERT INTO versions VALUES (?,?,?,?,?,?)",
                (name, name, "/fixture/weights", None, 1, 0),
            )
        for key, value in (
            ("current", "old-update"),
            ("previous", "base"),
            ("score_watermark", 7),
        ):
            db.execute("INSERT INTO state VALUES (?,?)", (key, json.dumps(value)))
        db.execute(
            "INSERT INTO pool VALUES (?,?,?)",
            ("step", 1, json.dumps({"id": "step", "surprise": 1})),
        )
        db.execute("INSERT INTO scored_revisions VALUES (?,?)", ("step", 7))
    store = Store(root)
    registry.migrate(config, store)
    registry.migrate(config, store)
    scoped = store.scoped("legacy")
    assert len(registry.entries(store)) == 1
    assert scoped.get("current") == "old-update"
    assert scoped.version(scoped.get("previous"))["id"] == "base"
    assert scoped.get("score_watermark") == 7
    assert registry.find(store, "legacy")["learning_level"] == "new"
    assert scoped.pool() == []  # Earlier replay remains stored, outside 'new'.
    with store.connect() as db:
        assert (
            json.loads(db.execute("SELECT step FROM pool").fetchone()[0])["id"]
            == "step"
        )
    assert scoped.unscored([{"id": "step", "source_id": 7}]) == []


def test_model_config_and_candidate_invalidation_are_isolated(learners):
    config, store = learners
    second = store.scoped("second")
    other = Config.for_model(store.root, "second")
    other.calibrated_step_seconds = 123
    other.save(store.root)
    assert Config.for_model(store.root, "second").calibrated_step_seconds == 123
    assert (
        Config.load(store.root).calibrated_step_seconds
        == config.calibrated_step_seconds
    )
    first_night = store.scoped("first").night(None)
    second_night = second.night(None)
    first_generation = store.scoped("first").get("candidate_generation")
    registry.enable(store, "second", False)
    assert store.scoped("first").nights()[0]["status"] == "running"
    assert second.nights()[0]["status"] == "discarded"
    assert store.scoped("first").get("candidate_generation") == first_generation
    assert {n["id"] for n in store.nights()} == {first_night, second_night}


async def test_live_snapshot_identifies_training_model_and_phase(learners):
    config, store = learners
    scoped = store.scoped("second")
    night = scoped.night(None)
    progress = {"phase": "train", "completed": 20, "total": 20, "unit": "samples"}
    selected = Config.for_model(store.root, "second")
    selected.accumulation = 3
    selected.save(store.root)
    scoped.night(
        night, phase="distill", progress=0.4, record={"data_progress": progress}
    )
    scheduler = SimpleNamespace(
        active_model="second", process=object(), calibrating=False, reason="Learning"
    )
    data = snapshot(config, store, scheduler, live=True)
    assert data["training_model"] == "songe/second"
    assert data["training"]["phase"] == "distill"
    assert data["training"]["data_progress"] == progress
    assert data["training"]["step"] == data["training"]["total_steps"] == 7
    assert data["nights"][0]["model"] == "songe/second"


def test_forwarding_init_without_model_or_local_weights(tmp_path, monkeypatch):
    from songe import agents, cli, launchd
    from songe.models import discovery
    from songe.platform import Memory

    platform = SimpleNamespace(
        name="test", accelerators=lambda: [], memory=lambda: Memory(1000, 500)
    )
    monkeypatch.setattr("songe.platform.current", lambda: platform)
    monkeypatch.setattr(cli, "detect_agent_binding", lambda: ("", ""))
    monkeypatch.setattr(discovery, "ollama", lambda **kw: [])
    monkeypatch.setattr(cli, "upstream_model_ids", lambda url: ["one", "two"])
    monkeypatch.setattr(cli, "confirm", lambda *a: True)
    proposed = []
    monkeypatch.setattr(
        agents, "proposed", lambda c, **kw: proposed.extend(c.exposed_models) or []
    )
    monkeypatch.setattr(agents, "apply", lambda *a: None)
    monkeypatch.setattr(launchd, "install", lambda *a, **kw: None)
    root = tmp_path / "data"
    args = cli.parser().parse_args(
        [
            "init",
            "--upstream",
            "ollama",
            "--launchd-target",
            str(tmp_path / "service"),
            "--yes",
        ]
    )
    cli.initialize(args, root)
    config = Config.load(root)
    assert config.model == ""
    assert config.exposed_models == ["songe/one", "songe/two"]
    assert proposed == ["one", "two"]
    assert {m["id"] for m in registry.entries(Store(root))} == {"one", "two"}


def test_agent_providers_expose_all_user_models(config):
    from songe.agents import opencode, pi

    config.exposed_models = ["one", "two"]
    config.base_identifier = "one"
    oc = opencode.configure({}, config, default=True, v2=True)
    assert set(oc["providers"]["songe"]["models"]) == {config.public_model, "songe/two"}
    assert oc["model"] == "songe/" + config.public_model
    assert [
        m["id"] for m in pi.configure({}, config)["providers"]["songe"]["models"]
    ] == [config.public_model, "songe/two"]


async def test_idle_discovery_adds_second_base_and_records_unsupported_reason(learners):
    config, store = learners
    registry.observe(config, store, "new")
    registry.observe(config, store, "unavailable")
    ref = ModelRef(config.upstream, "new", "safetensors", store.root / "new", "fixture")
    scheduler = ModelScheduler(config, store, SimpleNamespace(discover=lambda: [ref]))
    await scheduler.discover_models()
    assert store.scoped("new").version()["identifier"] == "new"
    assert Config.for_model(store.root, "new").model == str(ref.path)
    row = registry.find(store, "unavailable")
    assert row["support"] == "unsupported"
    assert "not discoverable" in row["reason"]


async def test_a_served_model_without_local_weights_is_serving_only(learners):
    """A fresh install against a vLLM whose weights are not on this machine:
    discovery names the model, not a path, and saving it as the model to
    train failed every scheduler tick ("Sleep failed" before anything ran)."""
    from pathlib import Path

    config, store = learners
    registry.observe(config, store, "Qwen/Qwen3.8-27B")
    ref = ModelRef(config.upstream, "Qwen/Qwen3.8-27B", "safetensors", Path("Qwen/Qwen3.8-27B"), "")
    scheduler = ModelScheduler(config, store, SimpleNamespace(discover=lambda: [ref]))
    await scheduler.discover_models()
    row = registry.find(store, "Qwen/Qwen3.8-27B")
    assert (row["support"], row["reason"]) == (
        "unsupported", "Local weights are not accessible; serving only")


async def test_unreachable_discovery_keeps_models_pending_without_failing(learners):
    # Issue #8: an httpx error from discovery ended every scheduler tick.
    config, store = learners
    registry.observe(config, store, "typo-model")

    def refused():
        raise httpx.ConnectError("Connection refused")

    scheduler = ModelScheduler(config, store, SimpleNamespace(discover=refused))
    await scheduler.discover_models()
    assert scheduler.reason.startswith("Model discovery unavailable")
    assert registry.find(store, "typo-model")["support"] == "pending"


def owned_llamacpp(tmp_path, seen):
    """An owned llama-server that is down beside the user's running one."""
    from songe.upstream.llamacpp import LlamaCPP

    weights = tmp_path / "smol.gguf"

    def handler(request):
        seen.append(request.url.port)
        if request.url.port != 8080:
            raise httpx.ConnectError("Connection refused", request=request)
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": str(weights)})
        return httpx.Response(200, json={"data": [{"id": "smol.gguf"}]})

    config = Config(
        upstream="llamacpp",
        upstream_url="http://127.0.0.1:34861/v1",
        server_owned=True,
        server_source_url="http://127.0.0.1:8080/v1",
    )
    return config, LlamaCPP(config, transport=httpx.MockTransport(handler))


def test_owned_llamacpp_discovers_from_the_source_server(tmp_path):
    seen = []
    _, upstream = owned_llamacpp(tmp_path, seen)
    assert [r.serving_id for r in upstream.discover()] == ["smol.gguf"]
    assert set(seen) == {8080}


async def test_unknown_name_on_owned_llamacpp_becomes_unsupported(learners, tmp_path):
    config, store = learners
    registry.observe(config, store, "typo-model")
    _, upstream = owned_llamacpp(tmp_path, [])
    scheduler = ModelScheduler(config, store, upstream)
    await scheduler.discover_models()
    row = registry.find(store, "typo-model")
    assert row["support"] == "unsupported"
    assert "not discoverable" in row["reason"]


def test_all_level_reuses_existing_shared_data(learners):
    config, store = learners
    store.sync_steps(evidence())
    store.scoped("first").mark_scored(evidence())
    registry.observe(config, store, "new-after-data")
    registry.enable(store, "new-after-data", False)
    registry.set_learning(store, "new-after-data", "all")
    assert store.scoped("new-after-data").experience_counts() == (5, 0, 5)
    assert (
        len(store.scoped("new-after-data").unscored(store.active_steps("train"))) == 5
    )


def test_forget_resets_every_learner_and_keeps_versions(learners):
    _, store = learners
    store.sync_steps(evidence())
    for model in ("first", "second"):
        store.scoped(model).mark_scored(evidence())
        store.scoped(model).set("experience_watermark", 5)
    store.forget()
    for model in ("first", "second"):
        scoped = store.scoped(model)
        assert scoped.version() is not None
        assert scoped.get("experience_watermark") is None
        assert scoped.get("rebuild_required") is True
        assert scoped.experience_counts() == (0, 0, 0)


async def test_accept_and_live_undo_change_only_second_model(learners, monkeypatch):

    from songe.schedule import base_fingerprint
    from tests.test_swap import Upstream

    config, store = learners
    upstream = Upstream()
    scheduler = ModelScheduler(config, store, upstream)
    scheduler.select("second")
    scheduler.last_traffic -= 2000
    second = store.scoped("second")
    original = second.get("current")
    night = second.night(None)
    record = {
        "candidate": version(f"night-{night}", night),
        "expected": original,
        "generation": second.get("candidate_generation"),
        "configuration": training_fingerprint(scheduler.config),
        "base_fingerprint": base_fingerprint(scheduler.config),
        "environment": store.get("environment_fingerprint"),
        "watermark": 5,
    }
    second.night(night, status="ready", record=record)
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"choices": [{}]})
            )
        ),
    )
    await scheduler.accept(second.nights()[-1])
    assert second.get("current") == f"night-{night}"
    assert store.scoped("first").get("current") == "base"
    assert second.get("experience_watermark") == 5
    assert store.scoped("first").get("experience_watermark") is None
    await scheduler.return_to(0, model="second")
    assert second.get("current") == original
    assert store.scoped("first").get("current") == "base"
    assert upstream.loaded == [f"night-{night}", original]


async def test_global_serving_switch_waits_for_other_model_requests(learners):
    import asyncio

    config, store = learners
    config.upstream = "llamacpp"
    loaded = []

    async def ensure(version):
        loaded.append(version["id"])

    scheduler = ModelScheduler(config, store, SimpleNamespace(ensure=ensure))
    first = store.scoped("first").version()
    second = store.scoped("second").version()
    scheduler.acquire(first["id"])
    task = asyncio.create_task(scheduler.ensure_model("second", second))
    await asyncio.sleep(0)
    assert loaded == []
    scheduler.release(first["id"])
    await task
    assert loaded == [second["id"]]


def test_calibration_persists_observed_unsupported_reason(learners, monkeypatch):
    import sys

    from songe.sleep import calibrate

    _, store = learners

    def unsupported(*a):
        raise ValueError(
            "Backward pass: fixture architecture does not implement gradients"
        )

    monkeypatch.setattr(calibrate, "_calibrate", unsupported)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--child",
            "--data-dir",
            str(store.root),
            "--model-key",
            "second",
        ],
    )
    with pytest.raises(ValueError, match="gradients"):
        calibrate.main()
    row = registry.find(store, "second")
    assert row["support"] == "unsupported"
    assert (
        row["reason"]
        == "Backward pass: fixture architecture does not implement gradients"
    )
    assert registry.find(store, "first")["support"] == "trainable"


@pytest.mark.parametrize("linux_mlx", [False, True])
async def test_scheduler_runs_one_guarded_model_worker_and_yields_when_off(
    learners, monkeypatch, linux_mlx
):
    from songe import schedule
    from songe.platform import Accel, Memory
    from tests.test_schedule import good

    config, store = learners
    if linux_mlx:
        from songe import memory

        learner = registry.model_config(config, store, "second")
        learner.backend = "mlx"
        registry.save_config(learner, store)
        monkeypatch.setattr(
            memory, "linux_mlx_accelerator", lambda: Accel("mlx", "Apple GPU")
        )
    registry.enable(store, "first", False)
    registry.support(store, "second", "pending", "Waiting for measured trainability")
    platform = SimpleNamespace(
        name="linux" if linux_mlx else "test",
        memory=lambda: Memory(24 * 10**9, 20 * 10**9),
        accelerators=lambda: (
            [Accel("cpu", "CPU")]
            if linux_mlx
            else [Accel("cuda", "fixture", 8 * 10**9, 8 * 10**9)]
        ),
        keep_awake=lambda pid: None,
    )

    async def park(on_snapshot):
        on_snapshot([])

    scheduler = ModelScheduler(config, store, SimpleNamespace(park=park), platform)
    scheduler.last_traffic -= 2000
    monkeypatch.setattr(schedule, "probe", lambda p: good())
    monkeypatch.setattr(schedule, "guard", lambda *a, **kw: None)
    signals, launched = [], []
    child = SimpleNamespace(pid=1, returncode=None)

    async def spawn(*args, **kwargs):
        launched.append(args)
        return child

    monkeypatch.setattr(schedule.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(
        schedule, "interrupt_group", lambda process: signals.append(process)
    )
    await scheduler.tick()
    assert len(launched) == 1
    assert "songe.sleep.calibrate" in launched[0]
    assert launched[0][launched[0].index("--model-key") + 1] == "second"
    assert scheduler.active_model == "second"
    assert (
        snapshot(config, store, scheduler, live=True)["training"]["phase"]
        == "calibrate"
    )
    await scheduler.guard_task
    await scheduler.tick()
    assert len(launched) == 1
    registry.enable(store, "second", False)
    await scheduler.tick()
    assert signals == [child]


async def test_measured_memory_failure_retries_only_that_model_later(learners):
    from songe.memory import Snapshot

    config, store = learners
    registry.support(store, "second", "pending")  # its first measurement
    scheduler = ModelScheduler(config, store, SimpleNamespace())
    scheduler.select("second")
    scheduler.calibrating = True
    scheduler.memory_stop_sample = Snapshot(1000, 15, "critical")

    async def waited():
        return 1

    scheduler.process = SimpleNamespace(returncode=1, wait=waited)
    # No runtime was loaded, so restoration has no work in this fake endpoint.
    registry.enable(store, "first", False)
    await scheduler.finish()
    row = registry.find(store, "second")
    # A memory stop is not evidence the model cannot train: retry when idle.
    assert row["support"] == "pending"
    assert "memory" in row["reason"] and "Snapshot(" not in row["reason"]
    assert registry.find(store, "first")["support"] == "trainable"
    # A trainable model measuring again (growth) stays trainable (live: a
    # crowded Mac turned it back to "pending").
    registry.support(store, "second", "trainable")
    scheduler.select("second")
    scheduler.calibrating = True
    scheduler.memory_stop_sample = Snapshot(1000, 15, "critical")
    scheduler.process = SimpleNamespace(returncode=1, wait=waited)
    await scheduler.finish()
    assert registry.find(store, "second")["support"] == "trainable"


def test_pending_counts_selected_data_until_training_finishes(learners):
    _, store = learners
    store.sync_steps(evidence())
    second = store.scoped("second")
    second.add_pool([s | {"surprise": 2} for s in evidence()])
    second.mark_scored(evidence())
    assert second.experience_counts()[2] == 0
    assert second.pending_learnable() == 5
    second.set("experience_watermark", 5)
    assert second.pending_learnable() == 0
    assert store.scoped("first").pending_learnable() == 5


def test_init_model_preenables_another_base_without_rebinding_history(
    learners, monkeypatch
):
    from songe import agents, cli, launchd
    from songe.models import discovery

    config, store = learners
    weights = store.root / "third-weights"
    weights.mkdir()
    ref = ModelRef(config.upstream, "third", "safetensors", weights, "fixture")
    monkeypatch.setattr(cli, "detect_agent_binding", lambda: ("", ""))
    monkeypatch.setattr(discovery, "lmstudio", lambda **kw: [ref])
    monkeypatch.setattr(discovery, "loaded_lmstudio", lambda refs, *a: refs)
    monkeypatch.setattr("songe.upstream.lmstudio.find_lms", lambda: "/fixture/lms")
    monkeypatch.setattr(
        "songe.platform.current",
        lambda: SimpleNamespace(name="test", accelerators=lambda: []),
    )
    monkeypatch.setattr(agents, "proposed", lambda *a, **kw: [])
    monkeypatch.setattr(agents, "apply", lambda *a: None)
    monkeypatch.setattr(launchd, "install", lambda *a, **kw: None)
    args = cli.parser().parse_args(
        [
            "init",
            "--model",
            str(weights),
            "--upstream",
            config.upstream,
            "--yes",
            "--launchd-target",
            str(store.root / "service"),
        ]
    )
    cli.initialize(args, store.root)
    assert Config.load(store.root).model == config.model
    assert store.scoped("first").get("current") == "base"
    assert registry.find(store, "third")["enabled"] is True
    assert "songe/third" in Config.load(store.root).exposed_models


async def test_scheduler_skips_model_waiting_for_initial_batch(learners):
    config, store = learners
    config.min_train, config.min_heldout = 1, 1
    store.sync_steps(evidence() + [evidence(1)[0] | {"id": "held", "split": "heldout"}])
    second = store.scoped("second")
    second.mark_scored(evidence()[:1])
    second.set("score_watermark", 1)
    measured = Config.for_model(store.root, "second")
    measured.calibration_available = True
    measured.save(store.root)
    scheduler = ModelScheduler(config, store, SimpleNamespace())
    assert scheduler.choose_model() == "first"  # Five pending, versus four.
    assert await scheduler.next_model() == "second"  # Its incremental batch can run.


def test_old_memory_stop_unsupported_marks_are_reset(learners):
    config, store = learners
    registry.support(
        store,
        "second",
        "unsupported",
        "Calibration stopped at measured memory pressure: Snapshot(total=1)",
    )
    registry.migrate(config, store)
    assert registry.find(store, "second")["support"] == "pending"


async def test_errors_retry_forever_with_a_doubling_wait(learners):
    from songe.schedule import retry_wait

    config, store = learners
    registry.enable(store, "first", False)

    async def waited():
        return 1

    for attempt in range(1, 6):
        scheduler = ModelScheduler(config, store, SimpleNamespace())
        scheduler.select("second")
        scheduler.calibrating = True
        scheduler.process = SimpleNamespace(returncode=1, wait=waited)
        await scheduler.finish()
        # Errors never retire a model; each one waits twice as long.
        assert registry.find(store, "second")["support"] == "pending"
        assert store.scoped("second").get("failures")["count"] == attempt
    assert [retry_wait(n) for n in (1, 2, 3, 6)] == [300, 600, 1200, 9600]
    # A memory stop is not an error: the run got that far, so the error
    # wait ends (live: a 4h wait outlived the fix that made measuring work).
    scheduler = ModelScheduler(config, store, SimpleNamespace())
    scheduler.select("second")
    scheduler.process = SimpleNamespace(returncode=75, wait=waited)
    await scheduler.finish()
    assert store.scoped("second").get("failures") is None
    scheduler.select("second")
    scheduler.process = SimpleNamespace(returncode=0, wait=waited)
    await scheduler.finish()
    assert store.scoped("second").get("failures") is None


def test_a_failed_restore_never_ends_the_scheduler_loop():
    import asyncio
    from types import SimpleNamespace

    from songe.schedule import Scheduler

    calls = []

    class Fake:
        stopping = False
        process = None
        reason = ""
        store = SimpleNamespace(get=lambda key: ["parked"])
        config = SimpleNamespace(guard_max_interval=0)

        def upgraded(self):
            return False

        async def tick(self):
            calls.append("tick")
            if len(calls) > 4:
                self.stopping = True
            raise RuntimeError("upstream gone")

        async def restore(self):
            raise RuntimeError("still gone")

    asyncio.run(Scheduler.loop(Fake()))
    assert calls.count("tick") == 5


def test_an_explicit_off_survives_first_use(config, store):
    from songe.models import registry

    registry.migrate(config, store)
    model = config.base_identifier
    registry.set_learning(store, model, "off")
    registry.mark_chosen(store, model)
    assert registry.observe(config, store, model)["learning_level"] == "off"


def test_foreground_traffic_cancels_a_running_review(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from songe import review as review_module
    from songe.schedule import ModelScheduler, Scheduler

    started = asyncio.Event()

    async def slow_review(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)
        return 1

    monkeypatch.setattr(review_module, "review", slow_review)

    class Fake:
        process = None
        in_flight = {}
        last_traffic = 0.0
        traffic_revision = 0
        installation_store = None
        installation_config = SimpleNamespace(upstream_url="http://x", server_source_url="")
        upstream = None
        platform = SimpleNamespace(memory=lambda: SimpleNamespace(pressure="normal"))
        serving_endpoint = ModelScheduler.serving_endpoint

        async def review_model(self):
            return "m"

        def interrupt(self, reason=None):
            pass

        traffic = Scheduler.traffic

    async def main():
        fake = Fake()
        running = asyncio.create_task(ModelScheduler.review_replies(fake))
        await asyncio.wait_for(started.wait(), 1)
        fake.traffic()
        await asyncio.wait_for(running, 1)
        assert fake.review_task is None

    asyncio.run(main())


def test_models_resolve_by_the_name_songe_models_shows(config, store):
    """Live run: `songe models Qwen3.5-0.8B all` said Unknown model."""
    registry.migrate(config, store)
    registry.register(config, store, "lmstudio-community/Qwen3.5-0.8B-MLX-4bit")
    row = registry.find(store, "qwen3.5-0.8b")
    assert row["id"] == "lmstudio-community/Qwen3.5-0.8B-MLX-4bit"
    registry.register(config, store, "other/Qwen3.5-0.8B-Q4_K_M.gguf")
    assert registry.find(store, "Qwen3.5-0.8B") is None  # ambiguous


def test_dream_calls_go_where_the_proxy_sends_the_version(config, store):
    # #27: reading, review and Quick Dream examples used upstream_url, the
    # Songe-owned server, which runs only while Songe needs it.
    config.upstream = "llamacpp"
    config.upstream_url = "http://127.0.0.1:41001/v1"  # owned
    config.server_source_url = "http://127.0.0.1:8080/v1"  # the user's
    routed = {"id": "night-2", "identifier": "songe/x/night-2"}
    upstream = SimpleNamespace(
        routes={"songe/x/night-2": {"url": "http://127.0.0.1:41001/v1", "model": "lora-2"}},
        request_url=lambda v: "http://127.0.0.1:41001/v1",
        request_model=lambda v: "lora-2",
    )
    scheduler = ModelScheduler(config, store, upstream)
    base = {"id": "base", "identifier": "/models/x.gguf"}
    assert scheduler.serving_endpoint(base, "/models/x.gguf") == (
        "http://127.0.0.1:8080/v1",
        "/models/x.gguf",
    )
    assert scheduler.serving_endpoint(None, "x") == ("http://127.0.0.1:8080/v1", "x")
    assert scheduler.serving_endpoint(routed, "songe/x/night-2") == (
        "http://127.0.0.1:41001/v1",
        "lora-2",
    )
    # A learned version not routed yet (after a restart) is on the owned server.
    learned = {"id": "night-3", "identifier": "songe/x/night-3"}
    assert scheduler.serving_endpoint(learned, "songe/x/night-3")[0] == (
        "http://127.0.0.1:41001/v1"
    )


async def test_reader_reads_the_base_from_the_users_server(learners):
    config, store = learners
    config.upstream = "llamacpp"
    config.upstream_url = "http://127.0.0.1:41001/v1"
    config.server_source_url = "http://127.0.0.1:8080/v1"
    scheduler = ModelScheduler(config, store, SimpleNamespace())
    reader = await scheduler.reader_for("first")
    try:
        assert reader.url == "http://127.0.0.1:8080/v1"
    finally:
        await reader.close()


async def test_reader_reads_an_mlx_base_where_songe_serves_it(learners, monkeypatch):
    # Live (0.1.0a15, mlx-lm): nothing listened at the init URL, so Songe
    # started its own server on another port and the agent's requests went
    # there; reading asked the init URL (:1235) and failed with ConnectError.
    import asyncio

    from songe import memory, platform
    from songe.upstream import mlx, supervisor
    from tests.test_review3 import memory_platform

    config, store = learners
    config.upstream, config.upstream_url = "mlx", "http://127.0.0.1:1235/v1"
    supervisor.provision_endpoint(config)  # as songe init does
    warmed = []

    async def spawn(*args, **kwargs):
        return SimpleNamespace(pid=None, returncode=None, wait=asyncio.Event().wait)

    async def post(self, url, **kwargs):
        warmed.append((url.removesuffix("/chat/completions"), kwargs["json"]["model"]))
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    def nothing_listens(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(memory, "guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(platform, "current", memory_platform)
    monkeypatch.setattr(supervisor, "owns_endpoint", lambda *args: True)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    upstream = mlx.MLX(config, store.root)
    upstream.transport = httpx.MockTransport(nothing_listens)
    await upstream.load(store.scoped("first").version())
    upstream.watchdog.cancel()
    reader = await ModelScheduler(config, store, upstream).reader_for("first")
    try:
        # Songe's own server, asked for the model it was started with.
        assert [(reader.url, reader.model)] == warmed
    finally:
        await reader.close()


SERVERS = {
    # What each server answers on the probes identify() makes (404 otherwise).
    "ollama": {"/api/version": {"version": "0.40.17"}, "/api/tags": {"models": []}},
    "llama-swap": {
        "/api/version": {"build_date": "2026-11-20", "commit": "e0274d5", "version": "v275"},
        "/running": {"running": []},
    },
    "llama-swap-unreadable": {
        "/api/version": {"build_date": "2026-11-20", "commit": "e0274d5", "version": "v275"},
        "/running": {"running": [{"model": "qwen", "cmd": "llama-server -m models/qwen.gguf"}]},
    },
    "openai-only": {"/v1/models": {"data": [{"id": "qwen"}]}},
    "llama-server": {"/props": {"model_path": "/models/x.gguf"}},
    "unknown": {},
}


def serve(monkeypatch, answers):
    import httpx

    from songe.models import discovery

    def get(url, transport=None, timeout=5.0):
        for path, body in answers.items():
            if url.endswith(path):
                return body
        raise httpx.HTTPStatusError("404", request=httpx.Request("GET", url), response=httpx.Response(404))

    monkeypatch.setattr(discovery, "_get", get)


@pytest.mark.parametrize(
    "server,expected",
    [
        ("ollama", "ollama"),
        ("llama-swap", "llamacpp"),
        ("llama-server", "llamacpp"),
        ("openai-only", None),
        ("unknown", None),
    ],
)
def test_identify_tells_ollama_from_servers_that_only_share_api_version(
    monkeypatch, server, expected
):
    # #61: llama-swap answers /api/version and was taken for Ollama; it is a
    # llama.cpp source (#60). #62: an unknown server is None, not "mlx".
    from songe.models import discovery

    serve(monkeypatch, SERVERS[server])
    assert discovery.identify("http://127.0.0.1:9292/v1") == expected


@pytest.mark.parametrize("server", ["openai-only", "llama-swap", "llama-swap-unreadable"])
@pytest.mark.parametrize("through", ["agent", "upstream-url"])
def test_forwarding_init_uses_an_unrecognized_server_as_openai_compatible(
    tmp_path, monkeypatch, capsys, through, server
):
    # #61: the serve-only fallback never tried the agent's own server. #62:
    # with only --upstream-url it recorded the first built-in name, lmstudio.
    # #60: an idle llama-swap names no weights, so it is forwarded the same
    # way rather than set up as llama.cpp without Songe's own endpoint.
    from songe import agents, cli, launchd
    from songe.models import discovery
    from songe.platform import Memory

    url = "http://127.0.0.1:9292/v1"
    platform = SimpleNamespace(
        name="test", accelerators=lambda: [], memory=lambda: Memory(1000, 500),
        service=lambda: SimpleNamespace(available=lambda: True),
    )
    monkeypatch.setattr("songe.platform.current", lambda: platform)
    monkeypatch.setattr(
        cli, "detect_agent_binding", lambda: ("qwen", url) if through == "agent" else ("", "")
    )
    serve(monkeypatch, SERVERS[server])
    for name in ("ollama", "lmstudio", "llamacpp", "mlx", "vllm"):
        monkeypatch.setattr(discovery, name, lambda *a, **kw: [])
    monkeypatch.setattr(discovery, "loaded_lmstudio", lambda *a, **kw: [])
    monkeypatch.setattr(cli, "upstream_model_ids", lambda u: ["qwen"] if u == url else [])
    monkeypatch.setattr(agents, "proposed", lambda *a, **kw: [])
    monkeypatch.setattr(agents, "apply", lambda *a: None)
    monkeypatch.setattr(launchd, "install", lambda *a, **kw: None)
    root = tmp_path / "data"
    argv = ["--data-dir", str(root), "init", "--yes", "--launchd-target", str(tmp_path / "svc")]
    if through == "upstream-url":
        argv += ["--upstream-url", url]
    cli.initialize(cli.parser().parse_args(argv), root)
    config = Config.load(root)
    assert (config.upstream, config.upstream_url) == ("vllm", url)
    assert config.exposed_models == ["songe/qwen"]
    out = capsys.readouterr().out
    assert f"OpenAI-compatible · {url}" in out
    assert ("llama-swap has no model loaded" in out) == (server == "llama-swap")
    assert ("name no weights Songe can read" in out) == (server == "llama-swap-unreadable")
    if server != "openai-only":
        # The generic client has no listing of its own; /running still keeps
        # background work off a model llama-swap has not loaded (it would
        # swap out the user's).
        import asyncio

        from songe.log import Store

        scheduler = ModelScheduler(config, Store(root), SimpleNamespace())
        running = SERVERS[server]["/running"]["running"]
        assert asyncio.run(scheduler.loaded_identifiers()) == {e["model"] for e in running}
        if not running:
            assert asyncio.run(scheduler.reader_for("qwen")) is None


async def test_an_automatic_run_stopped_for_memory_says_so_and_tries_again_later(
    learners, monkeypatch
):
    """Live #66: the memory guard stopped an automatic run; the status said
    only "a Deep Dream needs 1 practice scene", and songe sleep said nothing."""
    import time as _time

    from songe import schedule
    from songe.memory import Snapshot, stopped_sample
    from songe.platform import Accel, Memory
    from songe.status import memory_hold
    from songe.wording import status_lines
    from tests.test_schedule import good

    config, store = learners
    registry.enable(store, "first", False)
    registry.support(store, "second", "pending")  # its first measurement
    platform = SimpleNamespace(
        name="test",
        memory=lambda: Memory(24 * 10**9, 11 * 10**9),
        accelerators=lambda: [Accel("cuda", "fixture", 8 * 10**9, 8 * 10**9)],
        keep_awake=lambda pid: None,
    )
    scheduler = ModelScheduler(config, store, SimpleNamespace(), platform)
    scheduler.last_traffic -= 5000
    monkeypatch.setattr(schedule, "probe", lambda p=None: good())

    async def waited():
        return 1

    def stop():
        scheduler.select("second")
        scheduler.calibrating = True
        scheduler.memory_stop_sample = stopped_sample(
            Snapshot(24 * 10**9, 11 * 10**9, "normal"), "attributed_paging"
        )
        scheduler.process = SimpleNamespace(returncode=1, wait=waited)

    learner = store.scoped("second")
    learner.set("last_memory_baseline", {"available": 11 * 10**9})
    stop()
    await scheduler.finish()
    hold = learner.get("memory_hold")
    assert hold["try"] == 1 and hold["rule"] == "attributed_paging"
    assert 50 < hold["retry_after"] - _time.time() <= schedule.MEMORY_BACKOFF
    # Status says it was stopped for memory, why, and when it tries again.
    data = snapshot(config, store, scheduler, live=True)
    now = data["activity"]["now"]
    assert "held for memory" in now and "swapping memory to disk" in now
    assert "tries again in 1m" in now and "practice scene" not in now
    assert dict(status_lines(data))["now"].startswith(
        "Waiting for memory · Learning stopped at "
    )
    # The dashboard shows it on the model's page too.
    from songe.cycle import deep_waiting

    assert deep_waiting(learner, "second", None).startswith("Held for memory · ")
    # songe sleep says what it waits for instead of nothing.
    wait = await scheduler.sleep_wait()
    assert wait.startswith("waiting for memory · learning stopped at ")
    assert wait.endswith("tries again in 1m")

    launched = []

    async def spawn(*args, **kwargs):
        launched.append(args)
        return SimpleNamespace(pid=1, returncode=None)

    monkeypatch.setattr(schedule.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(schedule, "guard", lambda *a, **kw: None)
    await scheduler.tick()
    assert not launched and scheduler.reason == "Waiting for free memory"
    # A second stop in a row waits longer.
    stop()
    await scheduler.finish()
    hold = learner.get("memory_hold")
    assert hold["try"] == 2
    assert hold["retry_after"] - _time.time() > 1.5 * schedule.MEMORY_BACKOFF
    # After its retry time it tries again on its own.
    learner.set("memory_hold", hold | {"retry_after": _time.time() - 1})
    assert "tries again when this computer is idle" in memory_hold(learner)
    await scheduler.tick()
    assert launched and "songe.sleep.calibrate" in launched[0]
    await scheduler.guard_task
    # A run that ends any other way clears the hold.
    scheduler.memory_stop_sample = None
    scheduler.process = SimpleNamespace(returncode=0, wait=waited)
    await scheduler.finish()
    assert learner.get("memory_hold") is None


def test_status_names_a_model_that_dreams_on_its_own(learners):
    """Live #66: "Auto-dream is off" while the model's own row said On: the
    status named another enabled model, one with auto-dream off."""
    from songe.status import activity

    config, store = learners
    config.learning = "dreams"
    for model in ("first", "second"):
        registry.support(store, model, "trainable")
    registry.set_auto_dream(store, "first", False)
    scheduler = ModelScheduler(config, store, SimpleNamespace(), platform=SimpleNamespace())
    assert scheduler.choose_model() is None  # nothing runnable: the fallback names one
    models = registry.summaries(config, store)
    status = activity(config, store, scheduler, None, models, live=True)
    assert "Auto-dream is off" not in status["next"]
    # Only when no model Dreams on its own does it say so.
    registry.set_auto_dream(store, "second", False)
    models = registry.summaries(config, store)
    status = activity(config, store, scheduler, None, models, live=True)
    assert status["next"] == "Auto-dream is off · press Dream to run one"


def test_a_full_graphics_card_says_what_to_do(learners, monkeypatch):
    """#135: with the user's model server holding most of the card, measuring
    ran out of room and said only "calibration failed". It now says the card
    is full beside the server and what to do, and the model shows it as
    needing the user, not as a wait that clears by itself."""
    import sys

    from songe import wording
    from songe.sleep import calibrate

    _, store = learners

    class OutOfMemoryError(RuntimeError):  # torch.OutOfMemoryError's name
        pass

    def full(*a):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(calibrate, "_calibrate", full)
    monkeypatch.setattr(
        sys, "argv", ["calibrate", "--child", "--data-dir", str(store.root), "--model-key", "second"]
    )
    with pytest.raises(OutOfMemoryError):
        calibrate.main()
    reason = store.scoped("second").get("child_error")
    assert reason == wording.DEVICE_FULL and "model server" in reason
    assert registry.find(store, "second")["support"] != "unsupported"  # it can train once there is room
    assert wording.model_problem({"reason": reason}).startswith("The graphics card is too full")
    # Any other failure keeps the short hint.
    monkeypatch.setattr(calibrate, "_calibrate", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        calibrate.main()
    assert store.scoped("second").get("child_error") == "calibration failed · songe doctor"
    assert not calibrate.device_full(MemoryError())  # the host memory guard's stop is not this
    # A pressed Deep Dream that stopped there says the same in its note.
    from types import SimpleNamespace

    from songe.schedule import ModelScheduler

    learner = store.scoped("second")
    learner.set("child_error", wording.DEVICE_FULL)
    owner = SimpleNamespace(installation_store=store)
    ModelScheduler.carry_on(owner, "second", {"nights": len(learner.nights()), "at": 0}, calibrating=True,
                            failed=True, interrupted=False, stopped_by=None, memory_stop=False)
    assert learner.get("dream_stopped")["detail"] == wording.DEVICE_FULL


def test_a_dream_stopped_by_an_untrainable_gguf_says_why(learners):
    from types import SimpleNamespace

    from songe.schedule import ModelScheduler
    from songe.wording import gguf_untrainable

    _, store = learners
    reason = gguf_untrainable("inkling2_loop")
    registry.support(store, "second", "unsupported", reason)
    learner = store.scoped("second")
    owner = SimpleNamespace(installation_store=store)
    ModelScheduler.carry_on(
        owner,
        "second",
        {"nights": len(learner.nights()), "at": 0},
        calibrating=True,
        failed=True,
        interrupted=False,
        stopped_by=None,
        memory_stop=False,
    )
    assert learner.get("dream_stopped")["detail"] == reason


SWAP_RUNNING = {
    "running": [
        {"model": "qwen", "state": "ready", "proxy": "http://localhost:5800",
         "cmd": "/app/llama-server --port 5800 -m {path} -c 8192 --jinja"},
        {"model": "tiny", "state": "ready", "cmd": "/app/llama-server --port 5801 --model={path}"},
    ]
}


def swap_server(tmp_path, monkeypatch):
    """llama-swap v275 as recorded: two of three configured models loaded."""
    from songe.models import discovery

    weights = tmp_path / "qwen.gguf"
    weights.write_bytes(b"GGUF")
    running = {
        "running": [
            entry | {"cmd": entry["cmd"].format(path=weights)}
            for entry in SWAP_RUNNING["running"]
        ]
    }
    running["running"][1]["cmd"] = running["running"][1]["cmd"].replace(
        str(weights), str(tmp_path / "missing.gguf")
    )
    answers = {
        "/running": running,
        "/v1/models": {"data": [{"id": "qwen"}, {"id": "tiny"}, {"id": "other"}]},
    }
    serve(monkeypatch, answers)
    monkeypatch.setattr(discovery, "read_gguf", lambda path: SimpleNamespace(architecture="qwen2"))
    return weights, answers


def test_llama_swap_models_come_from_running_commands(tmp_path, monkeypatch):
    # #60: llama-swap has no /props; a loaded model's command names its
    # weights. Models it has not loaded are offered, never loaded to look.
    from songe.models import discovery

    weights, _ = swap_server(tmp_path, monkeypatch)
    refs = discovery.llamacpp("http://127.0.0.1:9292/v1")
    assert [(r.upstream, r.serving_id, r.path, r.architecture) for r in refs] == [
        ("llamacpp", "qwen", weights, "qwen2")
    ]
    assert discovery.llamaswap_models("http://127.0.0.1:9292/v1") == ["qwen", "tiny", "other"]
    assert discovery.command_flag("x --ctx-size=4096 -m 'a b.gguf'", "-m", "--model") == "a b.gguf"


def test_llama_swap_commands_keep_windows_paths():
    # POSIX splitting dropped the backslashes of C:\models\qwen.gguf.
    from songe.models import discovery

    command = r'C:\llama\llama-server.exe -m C:\models\qwen.gguf -c 8192'
    assert discovery.command_flag(command, "-m", windows=True) == r"C:\models\qwen.gguf"
    quoted = r'"C:\llama\llama-server.exe" --model "C:\My Models\qwen.gguf"'
    assert discovery.command_flag(quoted, "--model", windows=True) == r"C:\My Models\qwen.gguf"
    assert discovery.command_flag(command, "-m", windows=False) != r"C:\models\qwen.gguf"


async def test_llama_swap_serves_its_bases_and_reports_their_context(tmp_path):
    import httpx

    from songe.upstream.llamacpp import LlamaCPP
    from songe.upstream.reuse import served_base

    weights = tmp_path / "qwen.gguf"
    weights.write_bytes(b"GGUF")
    running = {"running": [{"model": "qwen", "cmd": f"/app/llama-server -m {weights} -c 8192"},
                           {"model": "tiny", "cmd": "/app/llama-server -m /elsewhere.gguf"}]}

    def handler(request):
        if request.url.path == "/running":
            return httpx.Response(200, json=running)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen"}, {"id": "tiny"}, {"id": "other"}]})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    config = SimpleNamespace(upstream="llamacpp", upstream_url="http://owned/v1",
                             server_source_url="http://swap/v1", upstream_timeout=5)
    loaded = {"id": "base", "identifier": "qwen", "artifact": str(weights), "kind": "gguf-full"}
    assert await served_base(config, loaded, transport=transport) == {"model": "qwen", "url": "http://swap/v1"}
    # Not loaded: llama-swap loads it by its id, so Songe never starts its own.
    asleep = {"id": "base", "identifier": "other", "artifact": str(tmp_path / "o.gguf"), "kind": "gguf-full"}
    assert await served_base(config, asleep, transport=transport) == {"model": "other", "url": "http://swap/v1"}
    unknown = asleep | {"identifier": "gone"}
    assert await served_base(config, unknown, transport=transport) is None
    upstream = LlamaCPP(config, transport=transport)
    assert await upstream.served_contexts() == {"qwen": 8192}



def swap_scheduler(learners, running, models=None, props=None, adapters=()):
    from songe.upstream.llamacpp import LlamaCPP

    config, store = learners
    config.upstream = "llamacpp"
    config.upstream_url = "http://127.0.0.1:41001/v1"
    config.server_source_url = "http://127.0.0.1:9292/v1"

    def handler(request):
        if request.url.port == 41001:  # Songe's own llama-server
            if request.url.path == "/props" and props is not None:
                return httpx.Response(200, json=props)
            if request.url.path == "/lora-adapters" and props is not None:
                return httpx.Response(200, json=list(adapters))
            return httpx.Response(404)
        if request.url.path == "/running" and running is not None:
            return httpx.Response(200, json={"running": running})
        if request.url.path == "/v1/models" and models is not None:
            return httpx.Response(200, json={"data": [{"id": m} for m in models]})
        return httpx.Response(404)

    async def hook(*a):
        pass

    # Supervised (owns_server) only while it is serving.
    owned = {"restart": hook, "stop": hook} if props is not None else {}
    upstream = LlamaCPP(config, transport=httpx.MockTransport(handler), **owned)
    return ModelScheduler(config, store, upstream), store


async def test_llama_swap_learner_not_running_gets_no_reader(learners):
    # #60: a chat request for a model llama-swap has not loaded makes it swap
    # out the one the user is working with; background work waits instead.
    scheduler, store = swap_scheduler(learners, [{"model": "second", "cmd": "llama-server"}])
    identifier = store.scoped("first").version()["identifier"]
    assert identifier not in await scheduler.loaded_identifiers()
    assert await scheduler.reader_for("first") is None
    assert await scheduler.review_model() == "second"
    scheduler, _ = swap_scheduler(learners, [{"model": identifier}, "not an entry"])
    reader = await scheduler.reader_for("first")
    try:
        assert reader is not None
    finally:
        await reader.close()


async def test_llama_swap_counts_only_what_songes_own_server_serves_now(learners):
    # Songe's llama-server holds one base at a time and is stopped when not
    # needed: reading another learner then would use the wrong weights, or
    # fail. Only its base and applied adapters count, while it runs.
    from pathlib import Path

    from tests.test_store import version

    _, store = learners
    second = store.scoped("second")
    second.add_version(version("night-2", 2) | {"identifier": "second-learned"})
    second.swap("night-2", second.get("current"))
    scheduler, _ = swap_scheduler(learners, [])  # Songe's server stopped
    assert await scheduler.loaded_identifiers() == set()
    assert await scheduler.reader_for("second") is None
    lora = {"id": 0, "path": "/models/lora.gguf", "scale": 0.0}
    scheduler, _ = swap_scheduler(
        learners, [], props={"model_path": "/models/first-night.gguf"}, adapters=[lora]
    )
    upstream = scheduler.upstream
    upstream.identifiers = {
        str(Path("/models/first-night.gguf").resolve()): "first-learned",
        str(Path("/models/lora.gguf").resolve()): "first-lora",
    }
    assert await scheduler.loaded_identifiers() == {"first-learned"}
    assert await scheduler.reader_for("second") is None  # another learner's base
    lora["scale"] = 1.0
    assert await scheduler.loaded_identifiers() == {"first-learned", "first-lora"}
    scheduler, _ = swap_scheduler(learners, [], props={"model_path": "/models/night-2.gguf"})
    scheduler.upstream.identifiers = {str(Path("/models/night-2.gguf").resolve()): "second-learned"}
    reader = await scheduler.reader_for("second")
    try:
        assert reader is not None
    finally:
        await reader.close()


async def test_llama_swap_drops_an_adapter_when_a_restart_changes_the_base(tmp_path):
    # A's adapter is applied on base X; loading B's adapter restarts Songe's
    # server on base Y. A must not count: its requests would get Y and B's
    # adapter, and its signals would be read with B's weights.
    from songe.upstream.llamacpp import LlamaCPP

    a, b = str(tmp_path / "a.gguf"), str(tmp_path / "b.gguf")
    x, y = str(tmp_path / "x.gguf"), str(tmp_path / "y.gguf")
    lora_of = {x: a, y: b}
    state = {"base": x, "adapters": [{"id": 0, "path": a, "scale": 0.0}]}

    def handler(request):
        if request.url.port != 41001:
            return httpx.Response(200, json={"running": []})  # llama-swap, idle
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": state["base"]})
        if request.method == "POST":
            scales = {i["id"]: i["scale"] for i in json.loads(request.content)}
            for adapter in state["adapters"]:
                adapter["scale"] = scales.get(adapter["id"], 0.0)
        return httpx.Response(200, json=[dict(i) for i in state["adapters"]])

    async def restart(version):
        base = version["artifact"] if version.get("kind") == "gguf-full" else version["base"]
        state.update(base=base, adapters=[{"id": 0, "path": lora_of[base], "scale": 0.0}])

    async def stop():
        pass

    config = Config(upstream="llamacpp", upstream_url="http://127.0.0.1:41001/v1")
    config.server_source_url = "http://127.0.0.1:9292/v1"
    upstream = LlamaCPP(
        config, transport=httpx.MockTransport(handler), restart=restart, stop=stop
    )
    await upstream.load(
        {"identifier": "songe/a-learned", "artifact": a, "adapter": a, "base": x}
    )
    assert await upstream.serving() == ["songe/a-learned"]
    await upstream.load(
        {"identifier": "songe/b-learned", "artifact": b, "adapter": b, "base": y}
    )
    assert await upstream.serving() == ["songe/b-learned"]
    assert [i["identifier"] for i in await upstream.listing()] == ["songe/b-learned"]


async def test_llama_swap_failing_after_it_answered_means_nothing_loaded(learners):
    scheduler, _ = swap_scheduler(learners, [{"model": "first"}])
    assert await scheduler.review_model() == "first"
    scheduler.upstream.transport = httpx.MockTransport(lambda r: httpx.Response(502))
    assert await scheduler.loaded_identifiers() == set()
    assert await scheduler.review_model() is None


async def test_plain_llama_server_is_still_assumed_loaded(learners):
    scheduler, _ = swap_scheduler(learners, None)  # no /running
    assert await scheduler.loaded_identifiers() is None
    assert await scheduler.review_model() == "first"


async def test_llama_swap_model_not_running_stays_pending(learners, tmp_path):
    # Not running at the first idle discovery (switched away, or its ttl
    # unloaded it) is not "forwarding only" for good.
    config, store = learners
    for model in ("asleep", "gone"):
        registry.register(config, store, model)
    scheduler, _ = swap_scheduler(
        learners, [{"model": "gone", "cmd": "llama-server -m /nowhere.gguf"}], ["asleep", "gone"]
    )
    await scheduler.discover_models()
    rows = {r["id"]: r for r in registry.entries(store)}
    assert (rows["asleep"]["support"], rows["asleep"]["reason"]) == (
        "pending", "waiting until llama-swap runs it"
    )
    assert rows["gone"]["support"] == "unsupported"  # running, weights unreadable


def test_llama_swap_skips_relative_model_paths(tmp_path, monkeypatch):
    from songe.models import discovery

    _, answers = swap_server(tmp_path, monkeypatch)
    answers["/running"]["running"][0]["cmd"] = "llama-server -m models/qwen.gguf"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models/qwen.gguf").write_bytes(b"GGUF")
    assert discovery.llamacpp("http://127.0.0.1:9292/v1") == []


def test_llama_swap_init_offers_its_chat_models_from_the_source(monkeypatch):
    # Re-init: upstream_url is Songe's own port, so llama-swap is listed at
    # server_source_url; embedding models are never offered (#72).
    from songe import cli
    from songe.models import discovery

    asked = []
    monkeypatch.setattr(
        discovery, "llamaswap_models",
        lambda url: asked.append(url) or ["qwen", "nomic-embed-text-v1.5", "tiny"],
    )
    config = Config(upstream="llamacpp", upstream_url="http://127.0.0.1:41001/v1",
                    server_source_url="http://127.0.0.1:9292/v1", server_owned=True)
    assert cli.llamaswap_offered(config) == ["qwen", "tiny"]
    assert asked == ["http://127.0.0.1:9292/v1"]
