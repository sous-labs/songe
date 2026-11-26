import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from songe.models import ModelRef, origin, trainable_base
from tests.crossplatform_helpers import fake_platform

gguf = pytest.importorskip("gguf")
np = pytest.importorskip("numpy")

NORM = np.arange(32, dtype=np.float32) / 8  # exact in bf16


def quant(path, *, architecture="qwen35", repos=(), norm=NORM):
    writer = gguf.GGUFWriter(path, architecture)
    writer.add_file_type(15)
    writer.add_uint32("general.base_model.count", len(repos))
    for i, repo in enumerate(repos):
        writer.add_string(
            f"general.base_model.{i}.repo_url", f"https://huggingface.co/{repo}"
        )
    writer.add_tensor("output_norm.weight", norm)
    writer.add_tensor("blk.0.attn_q.weight", np.ones((32, 32), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def safetensors(tensors):
    header, data = {}, b""
    for name, values in tensors.items():
        raw = (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        raw = raw.tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(values.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    text = json.dumps(header).encode()
    return struct.pack("<Q", len(text)) + text + data


def hub(repos, asked=None):
    """A Hugging Face hub with one shard per repository; only range reads."""

    def handler(request):
        if asked is not None:
            asked.append(str(request.url))
        path = request.url.path
        if path.startswith("/api/models/"):
            repo = path.removeprefix("/api/models/")
            if repo not in repos:
                return httpx.Response(401)
            return httpx.Response(
                200,
                json={
                    "sha": "c0ffee" * 6,
                    "siblings": [
                        {"rfilename": "model.safetensors", "size": 54 * 10**9},
                        {"rfilename": "config.json", "size": 2000},
                    ],
                },
            )
        repo, _, rest = path.lstrip("/").partition("/resolve/")
        blob = repos[repo]
        start, end = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=blob[start : end + 1])

    return httpx.MockTransport(handler)


def test_candidates_read_what_the_gguf_names(tmp_path):
    path = quant(tmp_path / "q.gguf", repos=["Qwen/Qwen3.8-27B", "unsloth/Qwen3.8-27B"])
    reader = gguf.GGUFReader(path)
    assert origin.candidates(reader) == ["Qwen/Qwen3.8-27B", "unsloth/Qwen3.8-27B"]


def test_original_weights_are_the_ones_whose_norm_matches(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    # The card names the pretrained model first; its norm differs. Qwen3.5
    # and later store 1 + w in GGUF.
    path = quant(
        tmp_path / "q.gguf",
        repos=["Org/Model-Base", "Org/Gated", "Org/Model"],
        norm=NORM + 1,
    )
    layer = {"model.layers.0.input_layernorm.weight": NORM}
    repos = {
        "Org/Model-Base": safetensors(layer | {"model.norm.weight": NORM / 2}),
        "Org/Model": safetensors(layer | {"model.norm.weight": NORM}),
    }
    found = origin.original_weights(path, transport=hub(repos))
    assert found == origin.Origin("Org/Model", "c0ffee" * 6, 54 * 10**9)

    del repos["Org/Model"]
    assert origin.original_weights(path, transport=hub(repos)) is None


def test_no_named_repository_asks_nothing(tmp_path):
    asked = []
    path = quant(tmp_path / "q.gguf")
    assert origin.original_weights(path, transport=hub({}, asked)) is None
    assert asked == []


def test_once_chosen_a_gguf_trains_on_its_original_weights_without_dequantizing(
    tmp_path, monkeypatch
):
    import songe.models.convert as conversion

    platform = fake_platform(tmp_path / "data")
    source = quant(tmp_path / "Qwen3.8-27B-UD-Q4_K_M.gguf", repos=["Qwen/Qwen3.8-27B"])
    ref = ModelRef("llamacpp", "q", "gguf", source, "qwen35")
    snapshot = tmp_path / "hub/snapshots/rev"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    found = origin.Origin("Qwen/Qwen3.8-27B", "rev", 10)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace())
    monkeypatch.setattr(origin, "original_weights", lambda path: found)
    monkeypatch.setattr(origin, "download", lambda o: snapshot)
    monkeypatch.setattr(origin, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        conversion.shutil, "disk_usage", lambda path: SimpleNamespace(free=2**40)
    )

    def runner(argv, **kwargs):
        raise AssertionError("the GGUF was dequantized")

    assert trainable_base(ref, "torch", platform, runner=runner, download=True) == snapshot
    # Known from then on: no second look at the hub.
    monkeypatch.setattr(origin, "original_weights", lambda path: 1 / 0)
    assert trainable_base(ref, "torch", platform, runner=runner) == snapshot


def unreadable(monkeypatch, architecture="qwen35"):
    def load_gguf_checkpoint(path, return_tensors=False):
        raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")

    reader = SimpleNamespace(load_gguf_checkpoint=load_gguf_checkpoint)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(modeling_gguf_pytorch_utils=reader)
    )
    monkeypatch.setitem(sys.modules, "transformers.modeling_gguf_pytorch_utils", reader)


def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("contacted the hub")

    monkeypatch.setattr(origin, "original_weights", refuse)
    monkeypatch.setattr(origin, "download", refuse)
    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace())


def test_by_default_nothing_contacts_the_hub_and_the_panel_offers_the_download(
    tmp_path, monkeypatch
):
    from songe import wording

    no_network(monkeypatch)
    unreadable(monkeypatch)
    platform = fake_platform(tmp_path / "data")
    source = quant(tmp_path / "Qwen3.8-27B-UD-Q4_K_M.gguf", repos=["Qwen/Qwen3.8-27B"])
    ref = ModelRef("llamacpp", "songe/Qwen3.8-27B-UD-Q4_K_M.gguf", "gguf", source, "qwen35")
    with pytest.raises(ValueError) as offered:
        trainable_base(ref, "torch", platform, runner=lambda *a, **k: 1 / 0)
    assert str(offered.value).startswith("train from Qwen/Qwen3.8-27B · about 0 GB download")
    assert str(offered.value).endswith("songe models Qwen3.8-27B-UD-Q4_K_M download")
    problem = wording.model_problem({"support": "unsupported", "reason": str(offered.value)})
    assert problem.startswith("Train from Qwen/Qwen3.8-27B")


def test_by_default_a_readable_gguf_is_dequantized_as_before(tmp_path, monkeypatch):
    no_network(monkeypatch)
    monkeypatch.setitem(sys.modules, "transformers.modeling_gguf_pytorch_utils", None)
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=2**40))
    platform = fake_platform(tmp_path / "data")
    source = quant(tmp_path / "q.gguf", architecture="llama", repos=["Org/Model"])
    ref = ModelRef("llamacpp", "q", "gguf", source, "llama")
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        out = Path(argv[-1])
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model.safetensors").write_bytes(b"dense")

    assert trainable_base(ref, "torch", platform, runner=runner).name.endswith("-bf16")
    assert len(calls) == 1


async def test_the_scheduler_downloads_only_once_chosen_and_before_measuring(
    tmp_path, monkeypatch
):
    # The memory guard samples the machine while it measures: a 55 GB
    # download inside it stopped the Dream three times in a row.
    import songe.artifacts.storage as storage
    from songe.schedule import Scheduler

    source = quant(tmp_path / "Qwen3.8-27B-UD-Q4_K_M.gguf", repos=["Qwen/Qwen3.8-27B"])
    snapshot = tmp_path / "hub/snapshots/rev"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(storage, "artifact_root", lambda config, store: tmp_path / "art")
    chosen = {}
    learner = SimpleNamespace(get=chosen.get)
    owner = SimpleNamespace(
        config=SimpleNamespace(model_ref={"path": str(source), "format": "gguf"}, model_key="q"),
        store=SimpleNamespace(root=tmp_path, scoped=lambda key: learner),
        reason="",
    )
    no_network(monkeypatch)
    await Scheduler.fetch_original(owner)  # not chosen: nothing asked

    downloads = []
    monkeypatch.setattr(
        origin, "original_weights", lambda path: origin.Origin("Qwen/Qwen3.8-27B", "rev", 10)
    )
    monkeypatch.setattr(origin, "download", lambda o: downloads.append(o) or snapshot)
    monkeypatch.setattr(origin, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=2**40))
    chosen["download_original"] = True
    await Scheduler.fetch_original(owner)
    assert len(downloads) == 1 and "downloading" in owner.reason

    # The measuring child then starts from them without a download.
    monkeypatch.setattr(origin, "original_weights", lambda path: 1 / 0)
    ref = ModelRef("llamacpp", "q", "gguf", source, "qwen35")
    platform = fake_platform(tmp_path / "data")
    got = trainable_base(
        ref, "torch", platform, runner=lambda *a, **k: 1 / 0, artifact_directory=tmp_path / "art"
    )
    assert got == snapshot and len(downloads) == 1
