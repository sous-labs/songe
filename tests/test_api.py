import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest

from songe.api import LocalClient, ReadOnlyStore, snapshot
from songe.proxy import create_app
from tests.test_store import version


@pytest.fixture
def v2(store):
    # The store creates the v2 tables; fill them with recorded rows.
    with store.connect() as db:
        db.executescript("""
            INSERT INTO scoreboard(day, model_version, repeat_error_rate,
              steps_per_task, tool_call_valid_rate, general_drift)
              VALUES('2026-11-09','night-1',.25,4,.8,.002),
                    ('2026-11-10','night-2',NULL,3,.9,.003);
            INSERT INTO slices(started, ended, phase, units_done, interrupted_by)
              VALUES(1,2,'score',3,'traffic'), (3,NULL,'train',7,NULL);
        """)
    return store


@pytest.fixture
def client(config, store):
    app = create_app(config, store)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    )


async def test_endpoints_read_only_recorded_v2(config, v2, client):
    night = v2.night(None)
    v2.night(night, status="accepted", record={"steps_learned": 10})
    async with client:
        state = (await client.get("/api/state")).json()
        assert state["state"] == "settling"
        assert state["slice"]["units_done"] == 7
        assert state["scoreboard"][1]["repeat_error_rate"] is None
        assert (await client.get("/api/scoreboard")).json() == state["scoreboard"]
        assert (await client.get("/api/nights")).json() == state["nights"]
        assert (await client.get(f"/api/nights/{night}")).json()["record"] == {
            "steps_learned": 10
        }
        assert (await client.get("/api/nights/999")).status_code == 404
        assert (await client.get("/api/settings")).json()[
            "schedule_default_seconds"
        ] == 10
        assert (await client.get("/api/state")).headers[
            "cache-control"
        ] == "private, no-cache"


@pytest.mark.parametrize(
    "method,path", [("POST", "/api/control"), ("PATCH", "/api/settings")]
)
async def test_mutations_origin_json_and_loopback(client, method, path):
    async with client:
        assert (
            await client.request(
                method, path, json={}, headers={"origin": "https://evil.example"}
            )
        ).status_code == 403
        assert (
            await client.request(method, path, content="action=pause")
        ).status_code == 415
        assert (
            await client.request(method, path, json={}, headers={"origin": "null"})
        ).status_code == 403
        assert (
            await client.request(
                method, path, json={}, headers={"host": "rebound.example"}
            )
        ).status_code == 403
        assert (
            await client.request(
                method, path, json=[], headers={"origin": "http://127.0.0.1"}
            )
        ).status_code == 400
        assert (
            await client.request(
                method, path, content="{", headers={"content-type": "application/json"}
            )
        ).status_code == 400


@pytest.mark.parametrize(
    "peer,host", [("203.0.113.7", "127.0.0.1"), ("127.0.0.1", "rebound.example")]
)
async def test_reads_reject_remote_clients_and_dns_rebinding(config, store, peer, host):
    app = create_app(config, store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(peer, 1)),
        base_url=f"http://{host}",
    ) as client:
        assert (await client.get("/api/state")).status_code == 403


async def test_controls_and_forget_confirmation(client, store):
    store.record("/test", {}, {})
    artifact = store.root / "versions/night-1/steps.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("private experience", encoding="utf-8")
    adapter = artifact.parent / "adapter"
    adapter.mkdir()
    cached = store.root / "gate-references" / "base" / "answer.json"
    cached.parent.mkdir(parents=True)
    cached.write_text('{"y": [1]}', encoding="utf-8")
    async with client:
        for action, key, expected in [
            ("pause", "paused", True),
            ("resume", "paused", False),
            ("sleep_now", "requested_sleep", True),
        ]:
            response = await client.post("/api/control", json={"action": action})
            assert response.status_code == 200
            assert store.get(key) is expected
        assert (
            await client.post("/api/control", json={"action": "unknown"})
        ).status_code == 400
        # Dream needs the running scheduler's view of memory and data.
        assert (
            await client.post("/api/control", json={"action": "dream"})
        ).status_code in {400, 503}
        assert store.get("requested_model") is None
        assert (
            await client.post("/api/control", json={"action": "forget"})
        ).status_code == 400
        assert len(store.exchanges()) == 1
        assert (
            await client.post(
                "/api/control", json={"action": "forget", "confirm": True}
            )
        ).status_code == 200
        assert store.exchanges() == []
        assert store.get("paused") is True
        assert store.get("deep_rebuild_requested") is True
        assert store.get("requested_sleep") is False
        assert not artifact.exists() and not adapter.exists()
        assert not (store.root / "gate-references").exists()


async def test_undo_calls_scheduler_with_retained_version(config, store):
    for i in (1, 2):
        store.night(None)
        store.night(i, status="accepted")
        store.add_version(version(f"night-{i}", i))
        store.swap(f"night-{i}", None if i == 1 else "night-1")
    restored = []

    async def return_to(night, model=None):
        assert model == config.base_identifier
        restored.append(night)
        store.swap("night-1", "night-2")

    scheduler = SimpleNamespace(reason="Waiting for a quiet gap", return_to=return_to)
    app = create_app(config, store, scheduler=scheduler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        state = (await client.get("/api/state")).json()
        assert state["nights"][0]["can_return"] is True
        assert (
            await client.post("/api/control", json={"action": "undo", "night": 2})
        ).status_code == 409
        assert (
            await client.post("/api/control", json={"action": "undo"})
        ).status_code == 200
        assert restored == [1]
        assert store.version()["night"] == 1


async def test_settings_validation_and_persistence(client, store, config, tmp_path):
    model = str(tmp_path / "new/model")
    async with client:
        for value in [0, -1, True, "600", None]:
            assert (
                await client.patch(
                    "/api/settings", json={"schedule_default_seconds": value}
                )
            ).status_code == 400
        assert (
            await client.patch("/api/settings", json={"learning_rate": 1})
        ).status_code == 400
        assert (
            await client.patch("/api/settings", json={"model": "relative/path"})
        ).status_code == 400
        assert (
            await client.patch("/api/settings", json={"upstream": "unknown"})
        ).status_code == 400
        response = await client.patch(
            "/api/settings",
            json={"model": model, "schedule_default_seconds": 900},
        )
        assert response.status_code == 200
        assert response.json()["schedule_default_seconds"] == 900
        from songe.config import Config

        assert Config.load(store.root).model == model
        store.add_version(version("base", 0))
        store.swap("base", None)
        assert (
            await client.patch(
                "/api/settings", json={"model": str(tmp_path / "other/model")}
            )
        ).status_code == 409
        assert config.model == model


def test_offline_view_does_not_create_or_modify_store(tmp_path, config, v2):
    missing = tmp_path / "missing"
    assert snapshot(config, ReadOnlyStore(missing))["scoreboard"] == []
    assert not missing.exists()
    before = v2.path.read_bytes()
    state = snapshot(config, ReadOnlyStore(v2.root))
    assert state["service_running"] is False
    assert state["scoreboard"][0]["steps_per_task"] == 4
    assert v2.path.read_bytes() == before
    with (
        ReadOnlyStore(v2.root).connect() as db,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        db.execute("DELETE FROM scoreboard")


def test_client_falls_back_only_on_connection_failure(config, store, monkeypatch):
    local = LocalClient(store.root, config)

    def down(*args, **kwargs):
        raise httpx.ConnectError("stopped")

    monkeypatch.setattr(httpx.Client, "get", down)
    assert local.snapshot()["service_running"] is False

    def unhealthy(*args, **kwargs):
        return httpx.Response(503, request=httpx.Request("GET", local.url))

    monkeypatch.setattr(httpx.Client, "get", unhealthy)
    with pytest.raises(httpx.HTTPStatusError):
        local.snapshot()


def test_offline_controls_and_unconfirmed_forget(config, store, monkeypatch):
    def down(*args, **kwargs):
        raise httpx.ConnectError("stopped")

    monkeypatch.setattr(httpx.Client, "post", down)
    monkeypatch.setattr(httpx.Client, "get", down)
    local = LocalClient(store.root, config)
    local.control("pause")
    assert store.get("paused") is True
    local.control("resume")
    assert store.get("paused") is False
    with pytest.raises(ValueError, match="unconfirmed"):
        local.control("forget")
    for i in (1, 2):
        store.add_version(version(f"night-{i}", i))
        store.swap(f"night-{i}", None if i == 1 else "night-1")
    local.control("undo")
    assert store.version()["id"] == "night-1"


def test_client_live_uses_contract_endpoints(config, store, monkeypatch):
    calls = []

    def send(client, request, **kwargs):
        calls.append(
            (
                request.method,
                request.url.path,
                json.loads(request.content) if request.content else None,
            )
        )
        return httpx.Response(
            200,
            json={
                "schema": 1,
                "service_running": True,
                "installation_id": store.get("installation_id"),
            },
            request=request,
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    client = LocalClient(store.root, config)
    assert client.snapshot()["service_running"]
    client.control("pause")
    assert calls == [
        ("GET", "/api/state", None),
        ("GET", "/api/state", None),
        ("POST", "/api/control", {"action": "pause"}),
    ]


async def test_ipv6_exact_origin_and_retained_undo_failure(config, store):
    app = create_app(config, store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("::1", 1)),
        base_url="http://[::1]:7613",
    ) as client:
        assert (
            await client.post(
                "/api/control",
                json={"action": "pause"},
                headers={"origin": "http://[::1]:7613"},
            )
        ).status_code == 200
        assert (
            await client.post(
                "/api/control",
                json={"action": "pause"},
                headers={"origin": "http://[::1]:7614"},
            )
        ).status_code == 403
        assert (
            await client.post("/api/control", json={"action": "undo"})
        ).status_code == 503


def test_client_does_not_mutate_offline_after_service_error(config, store, monkeypatch):
    def rejected(client, request, **kwargs):
        return httpx.Response(
            409, json={"error": "Version is no longer retained"}, request=request
        )

    monkeypatch.setattr(httpx.Client, "send", rejected)
    with pytest.raises(httpx.HTTPStatusError):
        LocalClient(store.root, config).control("pause")
    assert store.get("paused") is None


def test_platform_data_directory_is_preferred(tmp_path, monkeypatch):
    import songe.platform
    from songe.api import interface_data_dir

    monkeypatch.delenv("SONGE_DATA_DIR")
    monkeypatch.setattr(
        songe.platform, "current", lambda: SimpleNamespace(data_dir=lambda: tmp_path)
    )
    assert interface_data_dir() == tmp_path


async def test_failed_restore_keeps_current_binding(config, store):
    for i in (1, 2):
        store.add_version(version(f"night-{i}", i))
        store.swap(f"night-{i}", None if i == 1 else "night-1")

    async def failed(night, model=None):
        assert model == config.base_identifier
        raise RuntimeError("Upstream could not load the retained version")

    app = create_app(config, store, scheduler=SimpleNamespace(return_to=failed))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        response = await client.post("/api/control", json={"action": "undo"})
        assert response.status_code == 409
        assert "could not load" in response.json()["error"]
        assert store.version()["id"] == "night-2"


def test_a_model_the_server_lists_but_songe_never_registered_has_no_record(config, store, caplog):
    """Live (LM Studio): the default embedding model had no registry row, its
    record failed every 1.5 s, and the null entry blanked the new page."""
    from songe.api import cycle_section
    from songe.models import registry

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "chat")
    out = cycle_section(config, store, [{"id": "chat"}, {"id": "text-embedding-nomic-embed-text-v1.5"}])
    assert "chat" in out and out["chat"] is not None
    assert "text-embedding-nomic-embed-text-v1.5" not in out
    assert "cycle_record_failed" not in caplog.text


def test_gpu_line_reads_mac_and_nvidia_and_only_while_a_dream_runs(monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace

    from songe.api import gpu_line
    from songe.platform import common

    common._GPU.update(at=0.0, value=None)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b'"Device Utilization %"=72 "In use system memory"=10522669875')
    assert common.gpu_usage(max_age=0) == {"busy": 72, "memory": 10522669875}
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(common.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"45, 9800\n")
    assert common.gpu_usage(max_age=0) == {"busy": 45, "memory": 9800 * 1024**2}
    # Nothing running: no line.
    assert gpu_line(SimpleNamespace(process=None)) is None


def test_undo_before_any_deep_dream_says_there_is_nothing_yet(store):
    from songe.api import return_target

    with pytest.raises(ValueError, match="^No earlier version yet"):
        return_target(store, {})
    with pytest.raises(ValueError, match="^Version not retained$"):
        return_target(store, {"night": 3})
