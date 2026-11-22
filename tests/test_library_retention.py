"""Library, consent and retention regressions. All models and runtimes are fake."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from songe import cli
from songe.api import ReadOnlyStore, snapshot
from songe.config import Config
from songe.models import ModelRef, discovery, library, registry
from songe.models.library import scan as scan_library
from songe.platform import Accel, Memory
from songe.proxy import create_app
from songe.schedule import Scheduler
from songe.sleep import fuse, retention
from songe.sleep.swap import collect, install
from songe.storage import write_json
from tests.test_lmstudio import FakeLMS
from tests.test_swap import Upstream


def platform(root):
    return SimpleNamespace(
        name="test",
        data_dir=lambda: root,
        memory=lambda: Memory(10**9, 8 * 10**8),
        accelerators=lambda: [Accel("cuda", "fake", 10**9, 8 * 10**8)],
    )


def weights(tmp_path, name, architecture="llama"):
    path = tmp_path / name
    path.mkdir(parents=True)
    write_json(path / "config.json", {"model_type": architecture})
    header = json.dumps(
        {"x.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}
    ).encode()
    (path / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + bytes(8)
    )
    return path


def test_full_lmstudio_inventory_includes_unloaded_unsupported_and_unseen(
    config, store, tmp_path, monkeypatch
):
    paths = [
        weights(tmp_path, n, a)
        for n, a in [("used", "llama"), ("unloaded", "future_arch"), ("large", "llama")]
    ]
    config.models_dir = str(tmp_path)
    config.lms = "fake-lms"
    refs = discovery.lmstudio(folder=tmp_path)
    registry.register(config, store, "used", refs[2])  # Actual binding selected below.
    # Avoid depending on directory sort order.
    selected = registry.model_config(config, store, "used")
    selected.model_ref = {"path": str(paths[0]), "format": "safetensors"}
    selected.save(store.root)
    commands = []

    def run(argv, **kwargs):
        commands.append(argv)
        models = [
            {
                "path": p.name,
                "modelKey": p.name,
                "sizeBytes": 2 * 10**9 if p.name == "large" else 100,
            }
            for p in paths
        ]
        return SimpleNamespace(
            stdout=json.dumps(
                models
                if argv[1] == "ls"
                else [{"path": "used", "identifier": "user-alias"}]
            )
        )

    monkeypatch.setattr(library.subprocess, "run", run)
    monkeypatch.setattr(library, "scan", scan_library)
    monkeypatch.setattr(
        library, "architecture_available", lambda a, b: a != "future_arch"
    )
    items = scan_library(config)
    monkeypatch.setattr(library, "scan", lambda c: items)
    before = store.path.read_bytes()
    data = snapshot(
        config,
        ReadOnlyStore(store.root),
        SimpleNamespace(platform=platform(store.root)),
        live=True,
    )
    assert store.path.read_bytes() == before
    by_name = {m["name"]: m for m in data["models"]}
    assert set(by_name) == {"used", "unloaded", "large"}
    assert by_name["used"]["state"] == "loaded"
    assert by_name["unloaded"]["state"] == "downloaded"
    assert by_name["unloaded"]["support"] == "unsupported"
    assert by_name["unloaded"]["reason"] == "future_arch not supported yet"
    assert "estimate" in by_name["large"]["reason"]
    assert by_name["unloaded"]["learning_level"] == "off"
    assert all(
        m["format"] == "safetensors" and m["size_bytes"] for m in by_name.values()
    )
    assert len(registry.entries(store)) == 1
    assert {c[1] for c in commands} == {"ls", "ps"}


def test_lmstudio_cli_retains_entries_with_unreadable_weights(
    config, tmp_path, monkeypatch
):
    config.models_dir, config.lms = str(tmp_path), "fake-lms"
    monkeypatch.setattr(
        library,
        "_cli",
        lambda exe, cmd: (
            [
                {
                    "modelKey": "unknown",
                    "path": "missing.gguf",
                    "sizeBytes": 123,
                    "format": "gguf",
                }
            ]
            if cmd == "ls"
            else []
        ),
    )
    result = scan_library(config)
    assert result[0]["ref"].serving_id == "unknown"
    assert result[0]["size_bytes"] == 123


@pytest.mark.parametrize("kind", ["ollama", "mlx", "vllm", "llamacpp"])
def test_inventory_protocols_are_read_only_and_include_every_served_model(
    config, tmp_path, monkeypatch, kind
):
    config.upstream, config.upstream_url = kind, "http://fake/v1"
    local = weights(tmp_path, "one")
    seen = []

    def get(url, *args, **kwargs):
        seen.append(url)
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": "one:latest", "size": 123, "details": {"family": "llama"}},
                    {"name": "two:latest", "size": 456},
                ]
            }
        if url.endswith("/api/ps"):
            return {"models": [{"name": "one:latest"}]}
        if url.endswith("/props"):
            return {"model_path": str(tmp_path / "one.gguf"), "model_type": "llama"}
        return {
            "data": [
                {"id": "one", "root": str(local)},
                {"id": "two", "root": str(tmp_path / "unavailable")},
            ]
        }

    monkeypatch.setattr(discovery, "_get", get)
    asked = []
    monkeypatch.setattr(discovery, "ollama", lambda **kw: asked.append(kw) or [])
    result = scan_library(config)
    assert len(result) == 2
    if kind == "ollama":
        # Tags made with `ollama create` are only listed by Ollama's API.
        assert asked[0]["url"] == "http://fake"
    assert (
        all(m["loaded"] for m in result)
        if kind != "ollama"
        else result[0]["loaded"] and not result[1]["loaded"]
    )
    assert seen and all(url.startswith("http://fake/") for url in seen)


async def test_unseen_library_model_can_enable_learning_from_dashboard(
    config, store, tmp_path, monkeypatch
):
    config.save(store.root)
    ref = ModelRef(
        "lmstudio", "unseen", "safetensors", weights(tmp_path, "unseen"), "llama"
    )
    monkeypatch.setattr(library, "scan", lambda c: [library._item(ref)])
    app = create_app(config, store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        response = await client.post(
            "/api/control", json={"action": "model", "model": "unseen", "level": "all"}
        )
    assert response.status_code == 200
    assert registry.find(store, "unseen")["learning_level"] == "all"
    assert registry.model_config(config, store, "unseen").model == str(ref.path)


@pytest.mark.parametrize("approved", [True, False])
def test_init_confirmation_persists_consent_only_when_approved(
    config, store, monkeypatch, capsys, approved
):
    assert not Config.load(store.root).replace_loaded_model
    monkeypatch.setattr(cli, "confirm", lambda *a: approved)
    args = SimpleNamespace(yes=False)
    assert cli.confirm_install(config, args) is approved
    if approved:
        config.save(store.root)
    assert Config.load(store.root).replace_loaded_model is approved


async def test_settings_consent_persists_both_directions_and_is_installation_wide(
    config, store
):
    registry.register(config, store, "second")
    app = create_app(config, store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        for value in (True, False):
            response = await client.patch(
                "/api/settings", json={"replace_loaded_model": value}
            )
            assert response.status_code == 200
            assert Config.load(store.root).replace_loaded_model is value
            assert Config.for_model(store.root, "second").replace_loaded_model is value
        assert (
            await client.patch("/api/settings", json={"replace_loaded_model": "yes"})
        ).status_code == 400


@pytest.mark.parametrize(
    "consent,change,healthy,commit",
    [
        (False, "", True, True),
        (True, "", True, True),
        (True, "path", True, True),
        (True, "instance", True, True),
        (True, "weights", True, True),
        (True, "revoke", True, True),
        (True, "base", True, True),
        (True, "unreused", True, True),
        (True, "", False, True),
        (True, "", True, False),
    ],
)
async def test_replace_only_reused_exact_instance_after_successful_serving(
    config, store, tmp_path, monkeypatch, consent, change, healthy, commit
):
    from songe.upstream import protection

    monkeypatch.setattr(protection, "current", lambda: platform(store.root))
    artifact = weights(tmp_path, "lmstudio-community/granite-4.2-3b-MLX-4bit")
    config.model = str(artifact)
    config.models_dir, config.replace_loaded_model = str(tmp_path), consent
    config.save(store.root)
    user = {
        "identifier": "user",
        "path": str(artifact.relative_to(tmp_path)),
        "instanceReference": "user-instance",
    }
    upstream = FakeLMS(tmp_path, appear_after=0, running=[user])
    base = {
        "id": "base",
        "identifier": "base",
        "artifact": str(artifact),
        "adapter": None,
        "created": 1,
        "night": 0,
    }
    candidate = dict(
        base,
        id="night-1",
        identifier="songe/granite/night-1",
        artifact=str(tmp_path / "songe/granite/night-1"),
        night=1,
    )
    if change == "base":
        candidate["base"] = str(tmp_path / "another-base")
    store.add_version(base)
    store.swap("base", None)
    if change != "unreused":
        await upstream.load(base)
        assert upstream.request_model(base) == "user"

    async def health(client, url, **kwargs):
        assert user in upstream.ps and store.get("current") == "base"
        assert candidate["identifier"] in upstream.owned
        if change == "path":
            user["path"] = "other-model"
        if change == "instance":
            user["instanceReference"] = "replacement-user-instance"
        if change == "weights":
            (artifact / "config.json").write_text(
                "changed weights binding", encoding="utf-8"
            )
        if change == "revoke":
            replace(config, replace_loaded_model=False).save(store.root)
        return httpx.Response(
            200,
            json={"choices": [{}] if healthy else []},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", health)
    if healthy:
        assert (
            await install(
                store, upstream, config, candidate, "base", can_commit=lambda: commit
            )
            is commit
        )
    else:
        with pytest.raises(RuntimeError, match="completion"):
            await install(store, upstream, config, candidate, "base")
    removed = consent and not change and healthy and commit
    assert (user not in upstream.ps) is removed
    assert store.get("current") == ("night-1" if healthy and commit else "base")
    if removed:
        # Loading the original again directly remains independent of Songe.
        upstream.ps.append(dict(user, instanceReference="new-user-instance"))
        await upstream.unload(candidate)
        assert len(upstream.ps) == 1
    await upstream.close()


def retained_version(store, config, number):
    work = store.root / "versions" / f"night-{number}"
    work.mkdir(parents=True)
    write_json(
        work / "songe_work.json",
        {"installation_id": store.get("installation_id"), "night": number},
    )
    adapter = work / "adapter"
    adapter.mkdir()
    (adapter / "adapter.safetensors").write_bytes(b"small adapter")
    artifact = Path(config.models_dir) / "songe" / f"version-{number}"
    artifact.mkdir(parents=True)
    (artifact / "model.safetensors").write_bytes(b"large weights")
    write_json(
        artifact / "songe_owned.json",
        {"installation_id": store.get("installation_id"), "adapter": str(adapter)},
    )
    version = {
        "id": f"v{number}",
        "identifier": f"songe/v{number}",
        "artifact": str(artifact),
        "adapter": str(adapter),
        "training_adapter": str(adapter),
        "kind": "mlx-fused",
        "created": number,
        "night": number,
    }
    night = store.night(None)
    assert night == number
    store.night(night, status="accepted")
    store.add_version(version)
    return version


async def test_pruning_retains_all_adapters_records_base_and_one_full_export(
    config, store, tmp_path
):
    config.models_dir = str(tmp_path / "library")
    config.save(store.root)
    versions = [retained_version(store, config, i) for i in range(1, 5)]
    base = weights(tmp_path, "base")
    store.add_version(
        dict(
            versions[0],
            id="base",
            artifact=str(base),
            adapter=None,
            training_adapter=None,
            night=0,
        )
    )
    store.swap("v3", None)
    store.swap("v4", "v3")
    extra = store.root / "versions/night-1/checkpoint"
    extra.mkdir()
    (extra / "optimizer.safetensors").write_bytes(b"discard")
    await collect(store, Upstream(), {"v2": 1})
    assert not Path(versions[0]["artifact"]).exists()
    assert not extra.exists()
    assert all(Path(v["adapter"]).is_dir() for v in versions)
    # Only the serving export and one in use stay whole; v3 keeps its adapter.
    assert Path(versions[1]["artifact"]).exists()
    assert not Path(versions[2]["artifact"]).exists()
    await collect(store, Upstream(), {})
    assert not Path(versions[1]["artifact"]).exists()
    assert len(store.versions()) == 5 and base.is_dir()
    assert Path(versions[3]["artifact"]).is_dir()


async def test_return_to_adapter_only_version_rebuilds_then_serves_before_pointer_swap(
    config, store, tmp_path, monkeypatch
):
    config.models_dir = str(tmp_path / "library")
    config.save(store.root)
    versions = [retained_version(store, config, i) for i in range(1, 4)]
    store.swap("v2", None)
    store.swap("v3", "v2")
    await collect(store, Upstream(), {})
    target = versions[0]
    assert not Path(target["artifact"]).exists()
    events = []

    def export(c, kind, adapter, destination, p, check, **kwargs):
        assert c.model == config.model and p.data_dir() == store.root
        assert adapter == Path(target["adapter"])
        assert destination.parent.is_relative_to(Path(config.models_dir))
        check()
        events.append("guarded export")
        destination.mkdir()
        write_json(
            destination / "songe_owned.json",
            {"installation_id": store.get("installation_id")},
        )

    monkeypatch.setattr(fuse, "export_candidate", export)
    from songe.artifacts import storage

    preflight = storage.preflight

    def disk(requirements):
        assert store.get("current") == "v3"
        events.append("disk check")
        preflight(requirements)

    monkeypatch.setattr(storage, "preflight", disk)
    scheduler = Scheduler(config, store, Upstream(), platform(store.root))
    real = httpx.AsyncClient

    def health(request):
        assert store.get("current") == "v3" and Path(target["artifact"]).is_dir()
        events.append("serves")
        return httpx.Response(200, json={"choices": [{}]})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(health))
    )
    await scheduler.return_to(1)
    assert events == ["disk check", "guarded export", "serves"]
    assert store.get("current") == "v1" and store.get("previous") == "v3"
    assert not Path(versions[1]["artifact"]).exists()
    assert all(Path(v["adapter"]).is_dir() for v in versions)


def test_adapter_restore_checks_disk_before_export(
    config, store, tmp_path, monkeypatch
):
    from songe.artifacts import storage
    from songe.artifacts.storage import DiskSpace

    config.models_dir = str(tmp_path / "library")
    v = retained_version(store, config, 1)
    from songe.artifacts.runtime import remove_owned

    remove_owned(Path(v["artifact"]), store.get("installation_id"))
    monkeypatch.setattr(
        storage,
        "preflight",
        lambda req: (_ for _ in ()).throw(DiskSpace(tmp_path, 100, 0)),
    )
    monkeypatch.setattr(
        fuse,
        "export_candidate",
        lambda *a, **k: pytest.fail("must not export without disk"),
    )
    with pytest.raises(DiskSpace):
        retention.restore_artifact(store, config, v, platform(store.root))
    assert store.get("current") is None
    assert Path(v["adapter"]).is_dir()


@pytest.mark.parametrize(
    "consent,changed", [(False, False), (True, False), (True, True)]
)
async def test_ollama_replacement_verifies_digest_and_manifest_after_serving(
    config, store, monkeypatch, consent, changed
):
    from songe.upstream.ollama import Ollama

    config.upstream, config.replace_loaded_model = "ollama", consent
    config.save(store.root)
    upstream = Ollama(config)
    running = [{"name": "user:latest", "digest": "manifest-original"}]
    base = {
        "id": "base",
        "identifier": "base",
        "artifact": config.model,
        "adapter": None,
        "created": 1,
        "night": 0,
    }
    candidate = dict(
        base,
        id="candidate",
        identifier="songe:trained",
        artifact=config.model + "-trained",
        night=1,
    )
    store.add_version(base)
    store.swap("base", None)
    events = []

    async def ps():
        return running

    async def digest(version):
        return "exact-gguf-digest"

    async def same(version):
        assert version["digest"] == "exact-gguf-digest"
        return True

    async def load(version):
        events.append("load")

    async def warm(identifier, keep_alive):
        assert identifier == "user:latest" and keep_alive == 0
        assert store.get("current") == "candidate"
        events.append("unload")

    monkeypatch.setattr(upstream, "_running", ps)
    monkeypatch.setattr(upstream, "digest", digest)
    monkeypatch.setattr(upstream, "same_weights", same)
    monkeypatch.setattr(upstream, "load", load)
    monkeypatch.setattr(upstream, "_warm", warm)
    assert await upstream.reuse(base)
    real = httpx.AsyncClient

    def health(request):
        events.append("serves")
        if changed:
            running[0]["digest"] = "user-changed-manifest"
        return httpx.Response(200, json={"choices": [{}]})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(health))
    )
    assert await install(store, upstream, config, candidate, "base")
    assert events == (
        ["load", "serves", "unload"] if consent and not changed else ["load", "serves"]
    )


async def test_retained_versions_and_in_use_state_are_exposed_per_model(
    config, store, tmp_path, monkeypatch
):
    config.models_dir = str(tmp_path / "library")
    config.save(store.root)
    versions = [retained_version(store, config, i) for i in range(1, 4)]
    store.swap("v2", None)
    store.swap("v3", "v2")
    registry.migrate(config, store)
    await collect(store, Upstream(), {})
    scheduler = SimpleNamespace(in_flight={"v3": 1}, platform=platform(store.root))
    model = snapshot(config, store, scheduler, live=True)["models"][0]
    assert model["state"] == "in use"
    assert [v["retention"] for v in model["versions"]] == [
        "adapter only",
        "adapter only",
        "full",
    ]
    assert model["versions"][-1]["current"]
    assert Path(versions[0]["adapter"]).exists()


def test_refusion_uses_existing_guarded_fuser_and_library_volume(
    config, store, tmp_path, monkeypatch
):
    config.models_dir = str(tmp_path / "library")
    config.model = str(weights(tmp_path, "base"))
    version = retained_version(store, config, 1)
    from songe.artifacts.runtime import remove_owned

    remove_owned(Path(version["artifact"]), store.get("installation_id"))
    launched = []

    def guarded(argv, **kwargs):
        assert callable(kwargs["check"])
        kwargs["check"]()
        path = Path(argv[argv.index("--save-path") + 1])
        assert path.stat().st_dev == Path(config.models_dir).stat().st_dev
        (path / "model.safetensors").write_bytes(b"fake fused weights")
        launched.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(fuse, "guarded_run", guarded)
    retention.restore_artifact(store, config, version, platform(store.root))
    assert len(launched) == 1 and "songe.sleep.fuse_worker" in launched[0]
    assert Path(version["artifact"]).is_dir()
    assert Path(version["adapter"]).is_dir()


async def test_failed_refusion_preserves_current_and_adapters(
    config, store, tmp_path, monkeypatch
):
    config.models_dir = str(tmp_path / "library")
    config.save(store.root)
    versions = [retained_version(store, config, i) for i in range(1, 4)]
    store.swap("v2", None)
    store.swap("v3", "v2")
    await collect(store, Upstream(), {})

    def fail(*args, **kwargs):
        raise RuntimeError("guarded export interrupted")

    monkeypatch.setattr(fuse, "export_candidate", fail)
    scheduler = Scheduler(config, store, Upstream(), platform(store.root))
    with pytest.raises(RuntimeError, match="interrupted"):
        await scheduler.return_to(1)
    assert store.get("current") == "v3" and store.get("previous") == "v2"
    assert all(Path(v["adapter"]).is_dir() for v in versions)


def test_served_adapter_is_listed_as_unsupported_without_hiding_its_base(
    config, tmp_path, monkeypatch
):
    config.upstream = "vllm"
    monkeypatch.setattr(
        discovery,
        "_get",
        lambda url: {
            "data": [
                {"id": "base", "root": str(tmp_path / "base")},
                {
                    "id": "custom-lora",
                    "root": str(tmp_path / "adapter"),
                    "parent": "base",
                },
            ]
        },
    )
    result = scan_library(config)
    assert [i["ref"].serving_id for i in result] == ["base", "custom-lora"]
    status, reason = library.support(config, result[1]["ref"], None, platform(tmp_path))
    assert status == "unsupported" and "choose its base" in reason


def test_cli_library_columns_show_learning_and_short_status(capsys, monkeypatch):
    from songe.terminal import models_table

    monkeypatch.setenv("NO_COLOR", "1")
    models_table(
        [
            {
                "id": "future",
                "name": "Future model",
                "state": "downloaded",
                "format": "gguf",
                "size_bytes": 4 * 10**9,
                "learning_level": "off",
                "support": "unsupported",
                "reason": "Architecture not supported yet",
                "pending": 37,
                "last_result": "Rejected update",
            }
        ]
    )
    output = capsys.readouterr().out
    assert "Model" in output and "Auto-dream" in output and "Status" in output
    assert "Future model" in output and "Off" in output
    assert "Not supported yet" in output and "not supported yet ·" not in output
    for removed in (
        "Size",
        "Format",
        "Trainability",
        "Pending",
        "Last update",
        "Rejected update",
    ):
        assert removed not in output


async def test_tui_library_metadata_and_training_step_eta():
    from textual.widgets import DataTable, Static

    from songe.tui import SongeApp
    from tests.test_tui import Client

    client = Client()
    client.data["service_running"] = True
    client.data["models"] = [
        {
            "id": "one",
            "name": "Library model",
            "size_bytes": 4 * 10**9,
            "format": "gguf",
            "state": "loaded",
            "support": "trainable",
            "status": "off",
            "learning_level": "off",
            "enabled": False,
            "pending": 17,
            "last_result": "Accepted",
        }
    ]
    client.data["training"] = {
        "model": "one",
        "phase": "train",
        "progress": 0.4,
        "step": 3,
        "total_steps": 10,
        "data_progress": {
            "phase": "train",
            "completed": 12,
            "total": 40,
            "unit": "samples",
            "eta_seconds": 30,
        },
    }
    app = SongeApp(client=client)
    async with app.run_test(size=(110, 55)) as pilot:
        await pilot.pause()
        models = app.query_one("#models", DataTable)
        row = models.get_row("one")
        assert row[0].plain == "Library model"
        assert row[1] == "Off"
        assert row[2].plain == ""
        assert not app.query_one("#model-details").display
        models.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        details = str(app.query_one("#model-details", Static).render())
        assert "Now          Training · step 3 of 10 · about 30s left" in details
        text = str(app.query_one("#state", Static).render())
        assert "Step 3 of 10" in text and "30s" in text


async def test_retention_prunes_only_stale_owned_ollama_import(
    config, store, tmp_path, monkeypatch
):
    from songe.upstream.ollama import Ollama

    config.models_dir = str(tmp_path / "library")
    config.upstream = "ollama"
    config.save(store.root)
    versions = [retained_version(store, config, i) for i in range(1, 4)]
    store.swap("v2", None)
    store.swap("v3", "v2")
    tags = {v["identifier"]: "manifest-" + v["id"] for v in versions}
    tags["user:latest"] = "user-manifest"
    store.set(
        "runtime_imports",
        {
            v["identifier"]: dict(
                v,
                installation_id=store.get("installation_id"),
                state="created",
                manifest_digest=tags[v["identifier"]],
                digest="fake-weight-digest",
            )
            for v in versions
        },
    )
    upstream = Ollama(config, store=store)
    deleted = []

    async def request(method, path, **kwargs):
        if path == "/api/tags":
            return {
                "models": [
                    {"name": name, "digest": digest} for name, digest in tags.items()
                ]
            }
        assert method == "DELETE" and path == "/api/delete"
        deleted.append(kwargs["json"]["model"])
        tags.pop(deleted[-1])
        return {}

    async def same(version):
        return version["digest"] == "fake-weight-digest"

    async def running():
        return []

    monkeypatch.setattr(upstream, "_request", request)
    monkeypatch.setattr(upstream, "same_weights", same)
    monkeypatch.setattr(upstream, "_running", running)
    await collect(store, upstream, {})
    # Only the serving import stays; older ones are rebuilt from adapters.
    assert deleted == [versions[0]["identifier"], versions[1]["identifier"]]
    assert set(tags) == {versions[2]["identifier"], "user:latest"}
    assert len(store.get("runtime_imports")) == 1
    assert len(store.versions()) == 3 and all(
        Path(v["adapter"]).is_dir() for v in versions
    )


def test_offline_return_rebuilds_old_adapter_and_prunes_former_previous(
    config, store, tmp_path, monkeypatch
):
    import asyncio

    from songe import upstream
    from songe.api import LocalClient

    config.models_dir = str(tmp_path / "library")
    config.save(store.root)
    versions = [retained_version(store, config, i) for i in range(1, 4)]
    store.swap("v2", None)
    store.swap("v3", "v2")
    asyncio.run(collect(store, Upstream(), {}))
    registry.migrate(config, store)

    def offline(*a, **kw):
        raise httpx.ConnectError("offline")

    def export(c, kind, adapter, destination, p, check, **kwargs):
        destination.mkdir()
        write_json(
            destination / "songe_owned.json",
            {"installation_id": store.get("installation_id")},
        )

    monkeypatch.setattr(httpx.Client, "get", offline)
    monkeypatch.setattr(fuse, "export_candidate", export)
    monkeypatch.setattr(upstream, "create", lambda *a: Upstream())
    LocalClient(store.root, config).control("undo", version="v1")
    assert store.get("current") == "v1" and store.get("previous") == "v3"
    assert Path(versions[0]["artifact"]).exists()
    assert not Path(versions[1]["artifact"]).exists()
    assert all(Path(v["adapter"]).exists() for v in versions)


def test_old_four_strength_default_becomes_two(config, tmp_path):
    from songe.config import Config

    config.wake_scales = (0.25, 0.5, 0.75, 1.0)
    config.save(tmp_path)
    assert Config.load(tmp_path).wake_scales == (1.0, 0.5)


def test_restore_progress_is_reported_while_a_version_is_rebuilt(store):
    from songe.api import restoring
    from songe.sleep.fuse import PROGRESS

    assert restoring(store) is None
    store.set("restoring", {"version": "night-3", "model": "pub/Qwen3.5-4B-MLX-4bit"})
    (store.root / PROGRESS).write_text('{"fraction": 0.42}')
    assert restoring(store) == {
        "version": "night-3",
        "model": "Qwen3.5-4B",
        "fraction": 0.42,
    }


def test_a_model_deleted_from_the_library_is_not_listed(config, store, tmp_path):
    from songe.api import snapshot

    library = tmp_path / "library"
    kept = library / "pub" / "Kept-MLX-4bit"
    kept.mkdir(parents=True)
    config.models_dir = str(library)
    for name in ("Kept-MLX-4bit", "Gone-MLX-4bit"):
        path = library / "pub" / name
        registry.register(
            config,
            store,
            f"pub/{name}",
            ModelRef("lmstudio", f"pub/{name}", "safetensors", path, "qwen3"),
        )
    names = [m["name"] for m in snapshot(config, store)["models"]]
    assert "Gone" not in " ".join(names) and any("Kept" in n for n in names)


def test_measuring_and_served_limits_do_not_void_a_night(config):
    from dataclasses import replace

    from songe.config import training_fingerprint

    before = training_fingerprint(config)
    later = replace(
        config,
        calibrated_step_seconds=12.5,
        calibration_available=True,
        served_contexts={"songe/x": 63114},
    )
    assert training_fingerprint(later) == before
    assert training_fingerprint(replace(config, rank=config.rank * 2)) != before


def test_scheduling_and_display_settings_keep_a_night_resumable(config):
    """Live run: fixing min_train and slice_limit discarded a Dream at step
    8 of 21. Only what a night computes may discard it."""
    from dataclasses import replace

    from songe.config import training_fingerprint

    before = training_fingerprint(config)
    for change in (
        {"min_train": 8}, {"min_new": 1}, {"slice_limit": 200},
        {"idle_default_seconds": 30.0}, {"example_limit": 2}, {"practice_per_signal": 1},
    ):
        assert training_fingerprint(replace(config, **change)) == before, change
    for change in (
        {"learning_rate": config.learning_rate * 2}, {"general_drift_limit": 0.1},
        {"min_heldout": config.min_heldout + 1}, {"wake_scales": (1.0,)},
    ):
        assert training_fingerprint(replace(config, **change)) != before, change
