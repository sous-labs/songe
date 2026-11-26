"""Public catalogs and runtime reuse, entirely against fake runtimes."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from songe import agents
from songe.agents.opencode import configure as opencode
from songe.agents.pi import configure as pi
from songe.api import snapshot
from songe.models import registry
from songe.proxy import create_app
from songe.schedule import ModelScheduler
from songe.sleep.swap import install
from songe.storage import write_json
from songe.upstream import protection
from songe.upstream.llamacpp import LlamaCPP
from songe.upstream.mlx import MLX
from songe.upstream.ollama import Ollama
from songe.upstream.vllm import VLLM
from tests.test_lmstudio import FakeLMS
from tests.test_proxy import Chunks


@pytest.fixture(autouse=True)
def measured_memory(monkeypatch):
    from songe.platform import Memory

    monkeypatch.setattr(
        protection,
        "current",
        lambda: SimpleNamespace(
            name="test",
            memory=lambda: Memory(10**9, 9 * 10**8),
            accelerators=lambda: [],
        ),
    )


def catalog_config(config):
    return replace(
        config,
        base_identifier="lmstudio-community/Qwen3.5-4B-MLX-4bit",
        public_model="songe/Qwen3.5-4B-MLX-4bit",
        exposed_models=[
            "lmstudio-community/Qwen3.5-4B-MLX-4bit",
            "songe/Qwen3.5-4B-MLX-4bit",
        ],
    )


def test_one_public_entry_per_model(config):
    config = catalog_config(config)
    for v2, key in [(False, "provider"), (True, "providers")]:
        result = opencode({}, config, v2=v2)
        assert list(result[key]["songe"]["models"]) == [config.public_model]
    assert [m["id"] for m in pi({}, config)["providers"]["songe"]["models"]] == [
        config.public_model
    ]


@pytest.mark.parametrize("kind", ["opencode", "opencode-v2", "pi"])
def test_agent_migration_preserves_user_entries_and_restore(
    config, store, tmp_path, kind
):
    config = catalog_config(config)
    config.save(store.root)
    path = tmp_path / (kind + ".json")
    key = "provider" if kind == "opencode" else "providers"
    user_entry = {"id": "user-special", "name": "Custom"}
    original_models = {"user-special": user_entry} if kind != "pi" else [user_entry]
    original = {
        key: {"unrelated": {"custom": True}, "songe": {"models": original_models}}
    }
    write_json(path, original)
    generated = (
        pi({}, config) if kind == "pi" else opencode({}, config, v2=key == "providers")
    )
    installed = generated[key]["songe"]
    if kind == "pi":
        entry = dict(
            installed["models"][0],
            id=config.base_identifier,
            name=config.base_identifier,
        )
        installed["models"] += [entry, user_entry]
    else:
        entry = dict(
            installed["models"][config.public_model], name=config.base_identifier
        )
        installed["models"].update(
            {config.base_identifier: entry, "user-special": user_entry}
        )
    legacy = {key: {"unrelated": {"custom": True}, "songe": installed}}
    result = (json.dumps(legacy, indent=2) + "\n").encode()
    agents.apply([(path, path.read_bytes(), result, "")], store.root)
    agents.migrate(config, store.root)
    migrated = json.loads(path.read_text(encoding="utf-8"))
    models = migrated[key]["songe"]["models"]
    identifiers = (
        list(models) if isinstance(models, dict) else [m["id"] for m in models]
    )
    assert set(identifiers) == {config.public_model, "user-special"}
    assert migrated[key]["unrelated"] == {"custom": True}
    assert config.exposed_models == [config.public_model]
    first = path.read_bytes()
    agents.migrate(config, store.root)
    assert path.read_bytes() == first
    assert agents.restore(store.root) == []
    assert json.loads(path.read_text(encoding="utf-8")) == original


async def test_public_listing_status_and_raw_requests_use_same_learner(config, store):
    config = catalog_config(config)
    seen = []

    async def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": config.base_identifier},
                        {"id": config.public_model},
                    ]
                },
            )
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, stream=Chunks([b'{"choices":[]}']))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        app = create_app(config, store, client=upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            listing = (await client.get("/v1/models")).json()["data"]
            assert [m["id"] for m in listing] == [config.public_model]
            for model in [config.public_model, config.base_identifier]:
                assert (
                    await client.post(
                        "/v1/chat/completions", json={"model": model, "messages": []}
                    )
                ).status_code == 200
    assert seen == [config.base_identifier] * 2
    assert len(registry.entries(store)) == 1
    assert [m["id"] for m in snapshot(config, store)["models"]] == [config.public_model]
    assert registry.resolve(store, config.public_model) == config.base_identifier
    registry.set_learning(store, config.public_model, "off")
    assert not registry.find(store, config.base_identifier)["enabled"]


async def test_lmstudio_reuse_then_user_unload_and_trained_swap(
    tmp_path, monkeypatch, config, store
):
    artifact = tmp_path / "lmstudio-community/granite-4.2-3b-MLX-4bit"
    base = {
        "id": "base",
        "identifier": "songe/granite/base",
        "artifact": str(artifact),
        "adapter": None,
        "created": 1,
        "night": 0,
    }
    user = {
        "identifier": "my-granite",
        "path": str(artifact.relative_to(tmp_path)),
        "instanceReference": "user-instance",
        "memoryBytes": 500,
    }
    upstream = FakeLMS(tmp_path, appear_after=0, running=[user])
    upstream.store = store
    admission = []
    original_admit = protection.admit
    monkeypatch.setattr(
        protection,
        "admit",
        lambda platform, required=0, *args: (
            admission.append(required) or original_admit(platform, required, *args)
        ),
    )
    artifact.mkdir(parents=True)
    (artifact / "weights").write_bytes(b"weights")
    await upstream.load(base)
    assert upstream.request_model(base) == "my-granite"
    assert not upstream.owned and not upstream.loaded and not upstream.calls
    assert admission == []
    assert not store.get("lmstudio_claims", {}).get("instances")
    assert await upstream.loaded_bytes() == 0
    candidate = {
        "id": "night-1",
        "identifier": "songe/granite/night-1",
        "artifact": str(tmp_path / "songe/granite/night-1"),
        "adapter": None,
        "created": 2,
        "night": 1,
    }
    store.add_version(base)
    store.swap("base", None)

    async def health(client, url, **kwargs):
        assert kwargs["json"]["model"] == candidate["identifier"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "yes"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", health)
    assert await install(store, upstream, config, candidate, "base")
    assert store.version()["id"] == "night-1"
    assert set(upstream.owned) == {candidate["identifier"]}
    await upstream.unload(base)
    assert user in upstream.ps
    await upstream.unload(candidate)
    assert upstream.ps == [user]
    upstream.ps.clear()  # User unloads their own instance.
    await upstream.load(base)
    assert upstream.request_model(base) == base["identifier"]
    assert set(upstream.owned) == {base["identifier"]}
    await upstream.close()
    assert upstream.ps == []
    assert [c[1] for c in upstream.calls if c[0] == "unload"] == [
        candidate["identifier"],
        base["identifier"],
    ]


async def test_ollama_digest_reuse_user_unload_and_owned_cleanup(tmp_path):
    base_path, trained_path = tmp_path / "base.gguf", tmp_path / "trained.gguf"
    base_path.write_bytes(b"base weights")
    trained_path.write_bytes(b"trained weights")

    def digest(p):
        return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()

    tags = {"user:latest": digest(base_path)}
    running = {"user:latest"}
    mutations = []

    def handler(request):
        path = request.url.path
        body = (
            json.loads(request.content)
            if request.content and "/blobs/" not in path
            else {}
        )
        if path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": name,
                            "digest": "manifest-" + tags[name],
                            "size_vram": 100,
                        }
                        for name in running
                    ]
                },
            )
        if path == "/api/show":
            return (
                httpx.Response(
                    200, json={"modelfile": "FROM /blobs/" + tags[body["model"]]}
                )
                if body["model"] in tags
                else httpx.Response(404)
            )
        mutations.append((path, body))
        if "/blobs/" in path:
            return httpx.Response(200)
        if path == "/api/create":
            tags[body["model"]] = next(iter(body["files"].values()))
            return httpx.Response(200, json={"status": "success"})
        if path == "/api/generate":
            running.discard(body["model"]) if body.get(
                "keep_alive"
            ) == 0 else running.add(body["model"])
            return httpx.Response(200, json={"done": True})
        raise AssertionError(path)

    upstream = Ollama(
        SimpleNamespace(upstream_url="http://ollama/v1"),
        transport=httpx.MockTransport(handler),
    )
    base = {"identifier": "songe:base", "artifact": str(base_path)}
    candidate = {"identifier": "songe:trained", "artifact": str(trained_path)}
    await upstream.load(base)
    assert upstream.request_model(base) == "user:latest"
    assert not upstream.owned and not mutations
    assert await upstream.loaded_bytes() == 0
    await upstream.load(candidate)
    assert set(upstream.owned) == {"songe:trained"}
    await upstream.unload(base)
    await upstream.unload(candidate)
    assert running == {"user:latest"}
    running.clear()
    await upstream.load(base)
    assert upstream.request_model(base) == "songe:base"
    assert set(upstream.owned) == {"songe:base"}
    await upstream.close()
    assert not running
    assert not any(body.get("model") == "user:latest" for _, body in mutations)


@pytest.mark.parametrize("kind", ["llamacpp", "mlx", "vllm"])
async def test_served_artifact_reuses_external_identifier(
    config, store, tmp_path, monkeypatch, kind
):
    config.upstream = kind
    config.upstream_url = "http://runtime/v1"
    version = {
        "id": "base",
        "kind": "base"
        if kind == "vllm"
        else "gguf-full"
        if kind == "llamacpp"
        else "mlx-fused",
        "artifact": config.model,
        "identifier": "songe:base",
    }
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": config.model})
        return httpx.Response(
            200, json={"data": [{"id": "user-instance", "root": config.model}]}
        )

    transport = httpx.MockTransport(handler)
    if kind == "llamacpp":
        upstream = LlamaCPP(config, transport=transport)
    elif kind == "vllm":
        upstream = VLLM(config, transport=transport)
    else:
        upstream = MLX(config, store.root)
        upstream.transport = transport
    monkeypatch.setattr(
        protection,
        "admit",
        lambda *a, **kw: pytest.fail("reuse must not reserve another copy"),
    )
    await upstream.load(version)
    assert upstream.request_model(version) == "user-instance"
    assert await upstream.loaded_bytes() == 0
    await upstream.unload(version)
    await upstream.close()
    assert all(method == "GET" for method, _ in calls)


async def test_proxy_routes_reused_identifier_and_rechecks_after_user_unload(
    config, store, tmp_path, monkeypatch
):
    upstream = FakeLMS(
        tmp_path,
        appear_after=0,
        running=[
            {
                "identifier": "user-instance",
                "path": "lmstudio-community/granite-4.2-3b-MLX-4bit",
            }
        ],
    )
    config.model = str(tmp_path / "lmstudio-community/granite-4.2-3b-MLX-4bit")
    config.models_dir = str(tmp_path)
    monkeypatch.setattr(upstream, "ensure", upstream.load)
    base = {
        "id": "base",
        "identifier": config.base_identifier,
        "artifact": config.model,
        "created": 1,
        "night": 0,
        "adapter": None,
    }
    store.add_version(base)
    store.swap("base", None)
    scheduler = ModelScheduler(config, store, upstream)
    seen = []

    async def handler(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, stream=Chunks([b'{"choices":[]}']))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        app = create_app(config, store, client=client, scheduler=scheduler)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as proxy:
            for identifier in [config.public_model, config.base_identifier]:
                response = await proxy.post(
                    "/v1/chat/completions", json={"model": identifier, "messages": []}
                )
                assert response.status_code == 200
                upstream.ps.clear()
    assert seen == ["user-instance", config.base_identifier]
    await upstream.close()


def test_name_collisions_keep_distinct_models(config, store):
    from songe.models.names import catalog

    config.exposed_models = ["first/model", "second/model"]
    config.base_identifier = "first/model"
    config.public_model = "songe/model"
    aliases = catalog(config)
    assert len(set(aliases.values())) == 2
    assert aliases["first/model"] == "songe/model"
    agents.migrate(config, store.root, update_agents=False)
    assert set(config.exposed_models) == set(aliases.values())


def test_serve_only_install_lists_each_model_once(config):
    # Seen with Ollama when its files are not readable: init saved the public
    # names, and the primary came back as "songe/smollm2-135m-96cda480" too.
    from songe.models.names import catalog, exposed

    config.model, config.model_ref = "", {}
    config.public_model = "songe/local"
    config.base_identifier = "smollm2:135m"
    config.exposed_models = ["songe/smollm2-135m", "songe/qwen1.5b-latest"]
    assert sorted(exposed(config)) == ["songe/qwen1.5b-latest", "songe/smollm2-135m"]
    assert catalog(config)["smollm2:135m"] == "songe/smollm2-135m"


def test_migration_keeps_user_edited_legacy_entry(config, store, tmp_path):
    config = catalog_config(config)
    path = tmp_path / "opencode.json"
    previous = opencode({}, config, v2=True)
    previous["providers"]["songe"]["models"][config.base_identifier] = {
        "name": "original"
    }
    result = json.dumps(previous).encode()
    agents.apply([(path, None, result, "")], store.root)
    changed = json.loads(result)
    changed["providers"]["songe"]["models"][config.base_identifier]["name"] = (
        "User changed"
    )
    write_json(path, changed)
    agents.migrate(config, store.root)
    assert json.loads(path.read_text(encoding="utf-8")) == changed


async def test_secondary_public_alias_routes_without_new_learner(config, store):
    config.exposed_models = [config.base_identifier, "publisher/other"]
    registry.migrate(config, store)
    registry.register(config, store, "publisher/other")
    seen = []

    async def handler(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, stream=Chunks([b'{"choices":[]}']))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        app = create_app(config, store, client=client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as proxy:
            listing = (await proxy.get("/v1/models")).json()["data"]
            assert {m["id"] for m in listing} == {config.public_model, "songe/other"}
            response = await proxy.post(
                "/v1/chat/completions", json={"model": "songe/other", "messages": []}
            )
            assert response.status_code == 200
    assert seen == ["publisher/other"]
    assert len(registry.entries(store)) == 2


def test_init_loaded_alias_exposes_one_weight_name(tmp_path, monkeypatch):
    from songe import cli
    from songe.models import ModelRef, discovery
    from songe.platform import Memory

    folder = tmp_path / "library"
    weights = folder / "lmstudio-community/Qwen3.5-4B-MLX-4bit"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
    raw = ModelRef(
        "lmstudio", "lmstudio-community/Qwen3.5-4B-MLX-4bit", "mlx", weights, "qwen3_5"
    )
    loaded = replace(raw, serving_id="qwen3.5-4b-mlx")
    monkeypatch.setattr(discovery, "lmstudio_downloads", lambda: folder)
    monkeypatch.setattr(discovery, "lmstudio", lambda **kw: [raw])
    monkeypatch.setattr(discovery, "loaded_lmstudio", lambda *a: [loaded, raw])
    monkeypatch.setattr(cli, "detect_agent_binding", lambda: (loaded.serving_id, ""))
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr("songe.upstream.lmstudio.find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(
        "songe.platform.current",
        lambda: SimpleNamespace(
            name="test",
            accelerators=lambda: [],
            memory=lambda: Memory(10**9, 9 * 10**8),
        ),
    )
    root = tmp_path / "installation"
    args = cli.parser().parse_args(
        [
            "init",
            "--upstream",
            "lmstudio",
            "--yes",
            "--launchd-target",
            str(tmp_path / "service.plist"),
        ]
    )
    cli.initialize(args, root)
    from songe.config import Config
    from songe.log import Store

    installed = Config.load(root)
    assert installed.public_model == "songe/Qwen3.5-4B-MLX-4bit"
    assert installed.exposed_models == [installed.public_model]
    assert [m["id"] for m in registry.entries(Store(root))] == [raw.serving_id]


def test_init_picks_the_loaded_model_over_alphabetical_order(tmp_path, monkeypatch):
    """After an uninstall opencode still remembers "songe/<model>"; that name,
    or failing it the model LM Studio is running, decides, not the alphabet."""
    from songe import cli
    from songe.config import Config
    from songe.models import ModelRef, discovery
    from songe.platform import Memory

    folder = tmp_path / "library"
    refs = []
    for publisher, name, arch in [
        ("LiquidAI", "LFM2.5-24B-A2B-MLX-4bit", "lfm2_moe"),
        ("lmstudio-community", "Qwen3.5-4B-MLX-4bit", "qwen3_5"),
    ]:
        weights = folder / publisher / name
        weights.mkdir(parents=True)
        (weights / "config.json").write_text(
            json.dumps({"model_type": arch}), encoding="utf-8"
        )
        refs.append(ModelRef("lmstudio", f"{publisher}/{name}", "mlx", weights, arch))
    loaded = replace(refs[1], serving_id="qwen3.5-4b-mlx")
    monkeypatch.setattr(discovery, "lmstudio_downloads", lambda: folder)
    monkeypatch.setattr(discovery, "lmstudio", lambda **kw: refs)
    monkeypatch.setattr(discovery, "loaded_lmstudio", lambda *a: [loaded, *refs])
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr("songe.upstream.lmstudio.find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(
        "songe.platform.current",
        lambda: SimpleNamespace(
            name="test",
            accelerators=lambda: [],
            memory=lambda: Memory(10**9, 9 * 10**8),
        ),
    )
    for n, remembered in enumerate(["songe/Qwen3.5-4B-MLX-4bit", ""]):
        monkeypatch.setattr(cli, "detect_agent_binding", lambda r=remembered: (r, ""))
        root = tmp_path / f"installation{n}"
        args = cli.parser().parse_args(
            [
                "init",
                "--upstream",
                "lmstudio",
                "--yes",
                "--launchd-target",
                str(tmp_path / f"service{n}.plist"),
            ]
        )
        cli.initialize(args, root)
        assert Config.load(root).public_model == "songe/Qwen3.5-4B-MLX-4bit"


def test_init_follows_the_servers_key_between_two_downloads_of_one_model(
    tmp_path, monkeypatch
):
    """The agent names a model by LM Studio's key. With an MLX and a GGUF
    download of one model, matching on the name picked the MLX one although
    the key was the GGUF file (live: a default no server could load)."""
    from songe import cli
    from songe.config import Config
    from songe.models import ModelRef, discovery
    from songe.platform import Memory

    folder = tmp_path / "library"
    mlx = folder / "LiquidAI/LFM3-1.7B-MLX-4bit"
    mlx.mkdir(parents=True)
    (mlx / "config.json").write_text(json.dumps({"model_type": "lfm3"}), "utf-8")
    gguf = folder / "LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    refs = [
        ModelRef("lmstudio", "LiquidAI/LFM3-1.7B-MLX-4bit", "mlx", mlx, "lfm3"),
        ModelRef("lmstudio", "LFM3-1.7B-Q4_K_M.gguf", "gguf", gguf, "lfm3"),
    ]
    monkeypatch.setattr(discovery, "lmstudio_downloads", lambda: folder)
    monkeypatch.setattr(discovery, "lmstudio", lambda **kw: refs)
    monkeypatch.setattr(discovery, "loaded_lmstudio", lambda *a: refs)
    monkeypatch.setattr(
        discovery,
        "lmstudio_key_path",
        lambda executable, key, library: gguf if key == "lfm3-1.7b" else None,
    )
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr("songe.upstream.lmstudio.find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(
        "songe.platform.current",
        lambda: SimpleNamespace(
            name="test",
            accelerators=lambda: [],
            memory=lambda: Memory(10**9, 9 * 10**8),
        ),
    )
    monkeypatch.setattr(cli, "detect_agent_binding", lambda: ("lfm3-1.7b", ""))
    root = tmp_path / "installation"
    args = cli.parser().parse_args(
        [
            "init",
            "--upstream",
            "lmstudio",
            "--yes",
            "--launchd-target",
            str(tmp_path / "service.plist"),
        ]
    )
    cli.initialize(args, root)
    assert Path(Config.load(root).model).resolve() == gguf.resolve()


async def test_a_model_added_after_start_answers_to_its_songe_name(
    config, store, tmp_path, monkeypatch
):
    """A model downloaded after Songe started: the first request by its Songe
    name registered "songe/qwen3-1.7b" itself as a model, the server answered
    404, that name was marked unsupported, and the real model was then
    renamed songe/qwen3-1.7b-<hash> (reported by Jack in a test on 11-24)."""
    from songe.models import ModelRef, library

    registry.migrate(config, store)
    weights = tmp_path / "blobs/qwen3-1.7b.gguf"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"GGUF")
    ref = ModelRef("ollama", "qwen3:1.7b", "gguf", weights, "qwen3")
    monkeypatch.setattr(library, "scan", lambda config: [{"ref": ref}])
    seen = []

    async def handler(request):
        asked = json.loads(request.content)["model"]
        seen.append(asked)
        if asked.startswith("songe/"):
            return httpx.Response(404, stream=Chunks([b'{"error":"model not found"}']))
        return httpx.Response(200, stream=Chunks([b'{"choices":[]}']))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        app = create_app(config, store, client=client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as proxy:
            response = await proxy.post(
                "/v1/chat/completions",
                json={"model": "songe/qwen3-1.7b", "messages": []},
            )
            assert response.status_code == 200
            missing = await proxy.post(
                "/v1/chat/completions",
                json={"model": "songe/not-downloaded", "messages": []},
            )
            assert missing.status_code == 404
    assert seen == ["qwen3:1.7b", "songe/not-downloaded"]
    ids = {row["id"] for row in registry.entries(store)}
    assert "songe/qwen3-1.7b" not in ids and "songe/not-downloaded" not in ids
    row = registry.find(store, "qwen3:1.7b")
    assert row["config"]["public_model"] == "songe/qwen3-1.7b"
