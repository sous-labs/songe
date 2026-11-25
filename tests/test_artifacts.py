import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from songe.artifacts import export, merge
from songe.artifacts.runtime import RunPolicy, guarded_run
from songe.artifacts.tools import Download, Toolchain, download, extract, fetch
from songe.models import ModelRef
from tests.crossplatform_helpers import (
    fake_platform,
    tiny_model,
    torch_config,
    write_gguf,
)


def archive_bytes(files):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return data.getvalue()


def test_tools_checksum_cache_tamper_repair_and_platform_selection(tmp_path):
    data = archive_bytes(
        {
            "source/convert_hf_to_gguf.py": b"hf",
            "source/convert_lora_to_gguf.py": b"lora",
            "bin/llama-quantize": b"binary",
            "source/gguf-py/gguf/__init__.py": b"gguf",
        }
    )
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=data)

    transport = httpx.MockTransport(handler)
    item = Download(
        "https://official.test/source", hashlib.sha256(data).hexdigest(), "tools.tar.gz"
    )
    platform = fake_platform(tmp_path)
    tools = fetch(platform, manifest=[item], transport=transport, machine="AMD64")
    assert tools.convert_hf.read_bytes() == b"hf"
    assert "gguf-py" in tools.env()["PYTHONPATH"]
    assert tools.env()["HF_HUB_OFFLINE"] == "1"
    assert tools == fetch(
        platform, manifest=[item], transport=transport, machine="x86_64"
    )
    assert len(calls) == 1
    tools.quantize.write_bytes(b"tampered")
    assert (
        fetch(
            platform, manifest=[item], transport=transport, machine="x86_64"
        ).quantize.read_bytes()
        == b"binary"
    )
    assert len(calls) == 1  # Verified cached archive repairs installed files.
    with pytest.raises(ValueError, match="No pinned"):
        fetch(platform, machine="unhandled")
    bad = Download(item.url, "0" * 64, "bad.tar.gz")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(bad, tmp_path, transport=transport)
    assert not (tmp_path / "bad.tar.gz").exists()
    assert not list(tmp_path.glob(".download-*"))


def test_only_the_converters_of_the_llama_cpp_source_are_extracted(tmp_path):
    """Windows' default path limit is 260 characters. The pinned source tree
    has web UI files 220 characters deep, so with the data folder in front
    every extraction failed with FileNotFoundError (reported by Jack in a
    test on 11-24). The converters never read them."""
    deep = "tools/ui/src/lib/components/" + "a" * 150 + ".svelte"
    tree = "llama.cpp-c1a4feb/"
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(
        archive_bytes(
            {
                tree + "convert_hf_to_gguf.py": b"hf",
                tree + "convert_lora_to_gguf.py": b"lora",
                tree + "conversion/base.py": b"base",
                tree + "gguf-py/gguf/__init__.py": b"gguf",
                tree + "models/templates/Kimi-K3.jinja": b"template",
                tree + "models/ggml-vocab-llama.gguf": b"vocab",
                tree + deep: b"ui",
                "bin/llama-quantize": b"binary",
            }
        )
    )
    out = tmp_path / "out"
    extract(archive, out)
    kept = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert kept == [
        "bin/llama-quantize",
        tree + "conversion/base.py",
        tree + "convert_hf_to_gguf.py",
        tree + "convert_lora_to_gguf.py",
        tree + "gguf-py/gguf/__init__.py",
        tree + "models/templates/Kimi-K3.jinja",
    ]
    assert max(len(name) for name in kept) < 100


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "C:/windows", "dir/../../escape", "dir\\escape"]
)
def test_archive_traversal_rejected(tmp_path, name):
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(archive_bytes({name: b"bad"}))
    with pytest.raises(ValueError, match="Unsafe"):
        extract(archive, tmp_path / "extract")


def test_zip_and_tar_external_links_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract(archive, tmp_path / "extract")
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as source:
        info = tarfile.TarInfo("link")
        info.type, info.linkname = tarfile.SYMTYPE, "../../escape"
        source.addfile(info)
    with pytest.raises(ValueError, match="Unsafe"):
        extract(archive, tmp_path / "extract")


def test_memory_guard_success_failure_timeout_and_pressure(tmp_path):
    platform = fake_platform(tmp_path)
    result = guarded_run([sys.executable, "-c", "print('complete')"], platform=platform)
    assert result.stdout.strip() == b"complete"
    with pytest.raises(
        RuntimeError, match="(?i)conversion failed.*songe doctor"
    ) as failure:
        guarded_run(
            [sys.executable, "-c", "raise ValueError('conversion error')"],
            platform=platform,
        )
    assert "Traceback" not in str(failure.value)
    assert "conversion error" not in str(failure.value)
    with pytest.raises(TimeoutError):
        guarded_run(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            platform=platform,
            policy=RunPolicy(timeout=0.3),
        )
    states = iter(["normal", "critical"])
    platform.memory = lambda: SimpleNamespace(
        total=2**41, available=2**40, pressure=next(states, "critical")
    )
    with pytest.raises(MemoryError, match="critical"):
        guarded_run(
            [sys.executable, "-c", "import time; time.sleep(10)"], platform=platform
        )
    with pytest.raises(MemoryError):
        guarded_run([sys.executable, "-c", "print('never')"], platform=platform)


def test_failed_flush_preserves_staging_without_publication(tmp_path, monkeypatch):
    from songe.artifacts import runtime

    staging, destination = tmp_path / "staged", tmp_path / "published"
    staging.write_bytes(b"complete artifact")

    def failed(fd):
        raise OSError("flush failed")

    monkeypatch.setattr(runtime.os, "fsync", failed)
    with pytest.raises(OSError, match="flush failed"):
        runtime.publish(staging, destination)
    assert not destination.exists()
    assert staging.read_bytes() == b"complete artifact"
    # The failed flush must release its file handle before caller cleanup.
    staging.unlink()


def test_exports_preserve_quantization_and_stage_before_publishing(
    tmp_path, writable_file_fsync
):
    pytest.importorskip("gguf")
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    original = write_gguf(tmp_path / "original.gguf")
    ref = ModelRef("ollama", "user:tag", "gguf", original, "llama")
    tools = Toolchain(
        Path("/convert_hf.py"),
        Path("/convert_lora.py"),
        Path("/llama-quantize"),
        Path("/tools"),
    )
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        assert kwargs["platform"] is platform
        if "songe.artifacts.worker" in argv:
            out = Path(argv[-1])
            (out / "config.json").write_text("{}", encoding="utf-8")
            (out / "model.safetensors").write_bytes(b"model")
        elif argv[0] == str(tools.quantize):
            assert argv[-1] == "Q4_K_M"
            write_gguf(Path(argv[2]))
        else:
            write_gguf(Path(argv[argv.index("--outfile") + 1]), file_type=32)

    platform = fake_platform(tmp_path)
    out = export(
        "gguf-full",
        base,
        adapter,
        tmp_path / "full.gguf",
        platform,
        ref=ref,
        tools=tools,
        runner=runner,
    )
    assert out.is_file() and original.is_file()
    # A 4-bit base's merge is stored at Q8_0 so rounding cannot bury the update.
    assert not any(argv[0] == str(tools.quantize) for argv in calls)
    assert any(
        "--outtype" in a and a[a.index("--outtype") + 1] == "q8_0" for a in calls
    )
    lora = export(
        "gguf-lora",
        base,
        adapter,
        tmp_path / "lora.gguf",
        platform,
        tools=tools,
        runner=runner,
    )
    assert lora.is_file()
    copied = export("peft-adapter", base, adapter, tmp_path / "peft", platform)
    assert (copied / "adapter_model.safetensors").read_bytes() == b"adapter"
    with pytest.raises(FileExistsError):
        export("peft-adapter", base, adapter, copied, platform)
    with pytest.raises(ValueError, match="original GGUF"):
        export("gguf-full", base, adapter, tmp_path / "fail.gguf", platform)

    def broken(argv, **kwargs):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        export(
            "gguf-lora",
            base,
            adapter,
            tmp_path / "fail.gguf",
            platform,
            tools=tools,
            runner=broken,
        )
    assert not (tmp_path / "fail.gguf").exists()
    assert not list(tmp_path.glob(".export-*"))


@pytest.mark.parametrize(
    ("model_type", "extra", "expected_architecture"),
    [
        ("qwen3_5", {}, "Qwen3_5TextModel"),
        ("qwen3_5_text", {}, "Qwen3_5TextModel"),
        ("qwen3_5", {"mtp_num_hidden_layers": 1}, "Qwen3_5ForCausalLM"),
        ("qwen3_5", {"text_config": {"hidden_size": 64}}, "Qwen3_5ForCausalLM"),
    ],
)
def test_lora_export_uses_text_only_qwen35_config(
    tmp_path, writable_file_fsync, model_type, extra, expected_architecture
):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    original = {
        "model_type": model_type,
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 64,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "num_hidden_layers": 64,
    } | extra
    config_path = base / "config.json"
    config_path.write_text(json.dumps(original), encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"fixture weights")
    (adapter / "adapter_config.json").write_text(
        '{"peft_type": "LORA"}', encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    tools = Toolchain(Path("unused"), Path("convert_lora.py"), Path("unused"), tmp_path)

    def runner(argv, **kwargs):
        converter_base = Path(argv[argv.index("--base") + 1])
        prepared = json.loads(
            (converter_base / "config.json").read_text(encoding="utf-8")
        )
        expected = original | {"architectures": [expected_architecture]}
        if expected_architecture == "Qwen3_5TextModel":
            expected["model_type"] = "qwen3_5_text"
            assert converter_base != base
            assert list(converter_base.iterdir()) == [converter_base / "config.json"]
        else:
            assert converter_base == base
        assert prepared == expected
        Path(argv[argv.index("--outfile") + 1]).write_bytes(b"GGUF")

    out = export(
        "gguf-lora",
        base,
        adapter,
        tmp_path / "lora.gguf",
        fake_platform(tmp_path),
        tools=tools,
        runner=runner,
    )

    assert out.read_bytes() == b"GGUF"
    assert json.loads(config_path.read_text(encoding="utf-8")) == original
    assert not list(tmp_path.glob(".export-*"))


def test_real_peft_merge_matches_adapter_model(tmp_path, writable_file_fsync):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    from songe.backends.torch import TorchBackend

    base = tiny_model(tmp_path / "base")
    backend = TorchBackend(torch_config(base))
    step = {
        "context": [{"role": "user", "content": "Hello"}],
        "hindsight": [{"role": "user", "content": "correct"}],
    }
    backend.optimizer(0.01)
    _, grads, _ = backend.train_step(step, seed=17)
    backend.apply(grads)
    adapter = tmp_path / "adapter"
    backend.save(adapter)
    copied = export(
        "peft-adapter", base, adapter, tmp_path / "published", fake_platform(tmp_path)
    )
    assert (copied / "adapter_config.json").is_file()
    fused = merge(base, adapter, tmp_path / "merged", fake_platform(tmp_path))
    model = AutoModelForCausalLM.from_pretrained(
        fused, dtype=torch.float32, local_files_only=True
    )
    with torch.no_grad():
        expected = backend.model(torch.tensor([[3, 7]])).logits
        actual = model(torch.tensor([[3, 7]])).logits
    # The merged artifact is bf16; its quantization error must stay bounded.
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)


@pytest.mark.parametrize("ftype,outtype", [(0, "f32"), (1, "f16"), (32, "bf16")])
def test_unquantized_gguf_export_keeps_original_precision(
    tmp_path, ftype, outtype, writable_file_fsync
):
    pytest.importorskip("gguf")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    original = write_gguf(tmp_path / "original.gguf", file_type=ftype)
    ref = ModelRef("llamacpp", "base", "gguf", original, "llama")
    tools = Toolchain(
        Path("/convert_hf.py"),
        Path("/convert_lora.py"),
        Path("/llama-quantize"),
        Path("/tools"),
    )

    def runner(argv, **kwargs):
        if "songe.artifacts.worker" in argv:
            out = Path(argv[-1])
            (out / "config.json").write_text("{}", encoding="utf-8")
            (out / "model.safetensors").write_bytes(b"model")
        else:
            assert argv[1] == str(tools.convert_hf)
            assert argv[argv.index("--outtype") + 1] == outtype
            write_gguf(Path(argv[argv.index("--outfile") + 1]), file_type=ftype)

    assert export(
        "gguf-full",
        tmp_path / "base",
        adapter,
        tmp_path / "result.gguf",
        fake_platform(tmp_path),
        ref=ref,
        tools=tools,
        runner=runner,
    ).is_file()


def test_tar_internal_link_chains_are_resolved(tmp_path):
    archive = tmp_path / "links.tar"
    with tarfile.open(archive, "w") as source:
        for name, target in [("lib.so", "lib.so.1"), ("lib.so.1", "lib.so.1.2")]:
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            source.addfile(info)
        info = tarfile.TarInfo("lib.so.1.2")
        info.size = 4
        source.addfile(info, io.BytesIO(b"data"))
    extract(archive, tmp_path / "out")
    assert (tmp_path / "out/lib.so").read_bytes() == b"data"
