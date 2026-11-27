import json
import sys
from pathlib import Path

import httpx
import pytest

from songe.models import ModelRef, discovery, trainable_base
from songe.models.gguf import read_gguf
from tests.crossplatform_helpers import fake_platform, tiny_model, write_gguf

pytest.importorskip("torch")
pytest.importorskip("gguf")


@pytest.mark.parametrize("os_name", ["linux", "darwin", "win32"])
def test_lmstudio_respects_downloads_folder_and_both_formats(tmp_path, os_name):
    home = tmp_path / "home"
    settings = home / ".lmstudio/settings.json"
    settings.parent.mkdir(parents=True)
    library = tmp_path / "library with spaces"
    settings.write_text(json.dumps({"downloadsFolder": str(library)}), encoding="utf-8")
    model = tiny_model(library / "publisher/model")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    config["quantization"] = {"bits": 4, "group_size": 64}
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    gguf = write_gguf(library / "publisher/model/quant.gguf")
    found = discovery.lmstudio(home=home, os_name=os_name)
    assert {(r.format, r.architecture) for r in found} == {
        ("mlx", "llama"),
        ("gguf", "llama"),
    }
    assert {r.serving_id for r in found} == {
        "publisher/model",
        "publisher/model/quant.gguf",
    }
    info = read_gguf(gguf)
    assert info.quantization == "Q4_K_M"
    assert info.parameters == 1024


def test_ollama_manifest_blob_tags_and_incomplete_download(tmp_path):
    root = tmp_path / "models"
    blob = root / "blobs" / ("sha256-" + "a" * 64)
    blob.parent.mkdir(parents=True)
    write_gguf(blob)
    manifest = root / "manifests/registry.ollama.ai/library/test/latest"
    manifest.parent.mkdir(parents=True)
    metadata = {
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": "sha256:" + "a" * 64,
            }
        ]
    }
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    custom = root / "manifests/custom.registry/publisher/test/tag"
    custom.parent.mkdir(parents=True)
    custom.write_text(json.dumps(metadata), encoding="utf-8")
    found = discovery.ollama(folder=root)
    assert {r.serving_id for r in found} == {
        "test:latest",
        "custom.registry/publisher/test:tag",
    }
    assert all(r.path == blob and r.architecture == "llama" for r in found)
    blob.unlink()
    assert discovery.ollama(folder=root) == []
    metadata["layers"][0]["digest"] = "sha256:../../outside"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    assert discovery.ollama(folder=root) == []


def test_ollama_api_only_tags_override_stale_manifests(tmp_path):
    root = tmp_path / "models"
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    old = write_gguf(blobs / ("sha256-" + "a" * 64))
    live = write_gguf(blobs / ("sha256-" + "b" * 64))
    manifest = root / "manifests/registry.ollama.ai/library/base/latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:" + "a" * 64,
                    }
                ]
            }
        )
    )
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "base:latest"},
                        {"name": "songe/import:night-1"},
                        {"name": "remote:latest"},
                        {"name": "broken:latest"},
                    ]
                },
            )
        name = json.loads(request.content)["model"]
        if name == "broken:latest":
            return httpx.Response(404)
        path = live if name != "remote:latest" else "/missing/sha256-" + "c" * 64
        return httpx.Response(200, json={"modelfile": f'FROM "{path}"\nPARAMETER x 1'})

    transport = httpx.MockTransport(handler)
    found = discovery.ollama(folder=root, url="http://ollama/v1", transport=transport)
    assert {r.serving_id: r.path for r in found} == {
        "base:latest": live,
        "songe/import:night-1": live,
    }
    assert all(r.architecture == "llama" for r in found)
    assert set(calls) == {("GET", "/api/tags"), ("POST", "/api/show")}
    # A stopped daemon still permits offline use of the existing library.
    unavailable = httpx.MockTransport(lambda request: httpx.Response(503))
    found = discovery.ollama(folder=root, url="http://ollama", transport=unavailable)
    assert [(r.serving_id, r.path) for r in found] == [("base:latest", old)]


def test_ollama_api_resolves_local_server_blob_without_configured_library(tmp_path):
    blob = write_gguf(tmp_path / ("sha256-" + "d" * 64))

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base:latest"}]})
        return httpx.Response(200, json={"modelfile": f"FROM {blob}\n"})

    found = discovery.ollama(
        folder=tmp_path / "absent",
        url="http://ollama",
        transport=httpx.MockTransport(handler),
    )
    assert [(r.serving_id, r.path) for r in found] == [("base:latest", blob)]


def test_discovery_servers_and_local_hub_cache(tmp_path, monkeypatch):
    model = tiny_model(tmp_path / "model", "qwen2")
    gguf = write_gguf(tmp_path / "base.gguf")
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": str(gguf)})
        if request.url.host == "llama":
            data = [{"id": "served-alias"}]
        else:
            data = [
                {"id": "base", "root": str(model), "parent": None},
                {"id": "adapter", "root": "/adapter", "parent": "base"},
            ]
        return httpx.Response(200, json={"data": data})

    transport = httpx.MockTransport(handler)
    llama = discovery.llamacpp("http://llama/v1", transport)[0]
    assert (llama.serving_id, llama.path, llama.architecture) == (
        "served-alias",
        gguf,
        "llama",
    )
    for method in [discovery.vllm, discovery.mlx]:
        found = method("http://server/v1", transport)
        assert len(found) == 1
        assert found[0].path == model and found[0].architecture == "qwen2"
    assert all(method == "GET" for method, _ in calls)
    hub = tmp_path / "hf/hub/models--org--model"
    (hub / "refs").mkdir(parents=True)
    (hub / "refs/main").write_text("revision", encoding="utf-8")
    (hub / "snapshots/revision").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert discovery._cached_path("org/model") == hub / "snapshots/revision"


def test_trainable_base_passthrough_and_disk_checked_before_child(
    tmp_path, monkeypatch
):
    import songe.models.convert as conversion

    mlx_before = {
        name for name in sys.modules if name == "mlx" or name.startswith("mlx.")
    }
    platform = fake_platform(tmp_path / "data")
    source = write_gguf(tmp_path / "base.gguf")
    ref = ModelRef("ollama", "base:tag", "gguf", source, "llama")
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        assert kwargs["platform"] is platform
        out = Path(argv[-1])
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model.safetensors").write_bytes(b"dense weights")

    monkeypatch.setattr(
        conversion.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 1})()
    )
    with pytest.raises(OSError, match="GB free.*GB available"):
        trainable_base(ref, "torch", platform, runner=runner)
    assert calls == []
    monkeypatch.setattr(
        conversion.shutil,
        "disk_usage",
        lambda path: type("Disk", (), {"free": 2**40})(),
    )
    cached = trainable_base(ref, "torch", platform, runner=runner)
    assert cached != source and cached.is_dir()
    assert trainable_base(ref, "torch", platform, runner=runner) == cached
    assert len(calls) == 1
    assert {
        name for name in sys.modules if name == "mlx" or name.startswith("mlx.")
    } == mlx_before
    folder = tiny_model(tmp_path / "safe")
    safe = ModelRef("vllm", "base", "safetensors", folder, "llama")
    assert trainable_base(safe, "torch", platform) == folder
    mlx = ModelRef("mlx", "base", "mlx", folder, "llama")
    assert trainable_base(mlx, "mlx", fake_platform(tmp_path, "darwin")) == folder
    with pytest.raises(ValueError, match="macOS"):
        trainable_base(safe, "mlx", platform)


def test_failed_conversion_never_publishes_cache(tmp_path):
    platform = fake_platform(tmp_path / "data")
    source = write_gguf(tmp_path / "base.gguf")
    ref = ModelRef("llamacpp", "base", "gguf", source, "llama")

    def broken(argv, **kwargs):
        (Path(argv[-1]) / "partial").write_text("partial", encoding="utf-8")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        trainable_base(ref, "torch", platform, runner=broken)
    assert list((platform.data_dir() / "models").iterdir()) == []


def test_mlx_worker_uses_new_child_directory_and_rejects_other_os(
    tmp_path, monkeypatch
):
    import types

    from songe.models.worker import convert

    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS"):
        convert("mlx", tmp_path / "base", tmp_path / "out")
    monkeypatch.setattr(sys, "platform", "darwin")
    destination = tmp_path / "out"
    destination.mkdir()
    fake = types.ModuleType("mlx_lm.convert")

    def fake_convert(*, hf_path, mlx_path, dtype):
        out = Path(mlx_path)
        assert not out.exists()
        assert dtype == "bfloat16"
        out.mkdir()
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model.safetensors").write_bytes(b"weights")

    fake.convert = fake_convert
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.convert", fake)
    convert("mlx", tmp_path / "base", destination)
    assert (destination / "model.safetensors").is_file()
    assert not (destination / "model").exists()


def test_a_gguf_with_a_tensor_type_newer_than_the_gguf_package_is_read(tmp_path):
    """Qwen3.8-Flash-Next GGUFs (Unsloth UD-Q2_K_XL, Strata's Q2_0 builds) hold
    Q2_0 tensors (type 42) and file type 41; gguf 0.19's reader raised
    ValueError, so llama.cpp and Strata users got "Local weights are not
    discoverable"."""
    path = write_gguf(tmp_path / "flash.gguf", architecture="qwen4exp", file_type=41)
    data = bytearray(path.read_bytes())
    name = data.index(b"test.weight") + len(b"test.weight")
    data[name + 4 + 16 : name + 4 + 16 + 4] = (42).to_bytes(4, "little")  # after n_dims and two dims
    path.write_bytes(bytes(data))
    info = read_gguf(path)
    assert (info.architecture, info.parameters, info.file_type) == ("qwen4exp", 1024, 41)
    assert info.quantization in {"Q2_0", "type 41"}  # named once gguf knows it
