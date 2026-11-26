"""Publish immutable PEFT and GGUF artifacts using maintained converters."""

import json
import shutil
import tempfile
from pathlib import Path

from ..backends.install import interpreter_for
from ..models.gguf import read_gguf
from .runtime import guarded_run, publish
from .tools import fetch


def _validate_adapter(adapter):
    adapter = Path(adapter)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if (
        config.get("peft_type") != "LORA"
        or not (adapter / "adapter_model.safetensors").is_file()
    ):
        raise ValueError("Expected a PEFT LoRA config and adapter_model.safetensors")
    return adapter


def _lora_base(base, staging):
    try:
        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return base
    if (
        not isinstance(config, dict)
        or config.get("model_type") not in ("qwen3_5", "qwen3_5_text")
        or config.get("architectures") != ["Qwen3_5ForCausalLM"]
        or "text_config" in config
        or config.get("mtp_num_hidden_layers", 0) != 0
    ):
        return base

    # The pinned converter expects MTP layers for Qwen3_5ForCausalLM.
    # LoRA export only needs the base config. Select the text-only converter
    # without rewriting the original checkpoint or its cache.
    prepared = staging / "base"
    prepared.mkdir()
    config["architectures"] = ["Qwen3_5TextModel"]
    # AutoConfig otherwise creates a nested text_config whose defaults the
    # converter prefers over the original flat dimensions.
    config["model_type"] = "qwen3_5_text"
    (prepared / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return prepared


def merge(base, adapter, destination, platform, *, runner=guarded_run):
    """Merge into a new dense bf16 model; the original base is never modified."""
    from ..models.convert import require_disk

    base, destination = Path(base), Path(destination)
    adapter = _validate_adapter(adapter)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_disk(
        destination.parent,
        sum(p.stat().st_size for p in base.rglob("*") if p.is_file()),
    )
    staging = Path(tempfile.mkdtemp(prefix=".merge-", dir=destination.parent))
    from ..storage import private_dir

    private_dir(staging)
    try:
        runner(
            [
                interpreter_for(platform, runner),
                "-m",
                "songe.artifacts.worker",
                str(base),
                str(adapter),
                str(staging),
            ],
            platform=platform,
        )
        if not (staging / "config.json").is_file() or not list(
            staging.glob("*.safetensors")
        ):
            raise RuntimeError("Merge produced no safetensors model")
        return publish(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def export(
    kind,
    base,
    adapter,
    destination,
    platform,
    *,
    ref=None,
    tools=None,
    runner=guarded_run,
    ownership=None,
    merged_base=None,
    scratch=None,
):
    """Export peft-adapter, gguf-lora, or full GGUF with the base's file type."""
    if kind not in {"peft-adapter", "gguf-lora", "gguf-full"}:
        raise ValueError(f"Unsupported artifact kind: {kind}")
    from ..models.convert import require_disk

    base, destination = Path(base), Path(destination)
    adapter = Path(adapter) if merged_base is not None else _validate_adapter(adapter)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_disk(
        destination.parent,
        sum(p.stat().st_size for p in adapter.rglob("*") if p.is_file()),
    )
    info = None
    if kind == "gguf-full":
        if ref is None:
            raise ValueError(
                "gguf-full needs the original GGUF or safetensors ModelRef"
            )
        if ref is not None and ref.format == "gguf":
            info = read_gguf(ref.path)
            # Simultaneously alive: merged dense, conversion intermediate, final.
            dense = info.parameters * 2
            require_disk(destination.parent, dense * 2 + Path(ref.path).stat().st_size)
        else:
            # Dense sources have no original GGUF quantization to preserve.
            dense = sum(p.stat().st_size for p in base.glob("*.safetensors"))
            require_disk(destination.parent, dense * 3)
    from .storage import require_publish_volume

    require_publish_volume(scratch or destination.parent, destination)
    staging = Path(
        tempfile.mkdtemp(prefix=".export-", dir=scratch or destination.parent)
    )
    from ..storage import private_dir

    private_dir(staging)
    if ownership is not None:
        from ..storage import write_json

        write_json(staging / "songe_owned.json", ownership)
    try:
        if kind == "peft-adapter":
            result = staging / "adapter"
            shutil.copytree(adapter, result)
            from ..backends import packed_experts

            if packed_experts.is_packed_base(base):
                # vLLM reads Inkling-2 adapters under the checkpoint's names.
                packed_experts.to_checkpoint_names(result)
        else:
            tools = tools or fetch(platform)
            result = staging / "model.gguf"
            env = tools.env()
            if kind == "gguf-lora":
                runner(
                    [
                        interpreter_for(platform, runner),
                        str(tools.convert_lora),
                        "--base",
                        str(_lora_base(base, staging)),
                        "--outtype",
                        "bf16",
                        "--outfile",
                        str(result),
                        str(adapter),
                    ],
                    platform=platform,
                    env=env,
                )
            else:
                merged = (
                    Path(merged_base)
                    if merged_base is not None
                    else merge(
                        base, adapter, staging / "merged", platform, runner=runner
                    )
                )
                # Re-quantizing a merge back to 4-6 bits buries the learned
                # change under rounding noise (fuse_worker.py); below 8 bits
                # the merge is stored at Q8_0 instead of the base's own type.
                preserving = info is not None and info.quantization not in {
                    "F32",
                    "F16",
                    "BF16",
                    "Q8_0",
                }
                direct = (
                    info is None
                    or info.quantization in {"F32", "F16", "BF16"}
                    or preserving
                )
                intermediate = result if direct else staging / "dense.gguf"
                outtype = (
                    "q8_0"
                    if preserving
                    else info.quantization.lower()
                    if direct and info
                    else "bf16"
                )
                runner(
                    [
                        interpreter_for(platform, runner),
                        str(tools.convert_hf),
                        str(merged),
                        "--outtype",
                        outtype,
                        "--outfile",
                        str(intermediate),
                    ],
                    platform=platform,
                    env=env,
                )
                if not direct:
                    runner(
                        [
                            str(tools.quantize),
                            str(intermediate),
                            str(result),
                            info.quantization,
                        ],
                        platform=platform,
                        env=env,
                    )
                actual = read_gguf(result)
                if info is not None and (
                    (not preserving and actual.file_type != info.file_type)
                    or actual.architecture != info.architecture
                ):
                    raise ValueError(
                        "Exported GGUF architecture/quantization differs from the user's base"
                    )
            if not result.is_file():
                raise RuntimeError("Converter did not produce a GGUF")
        if ownership is not None:
            from ..storage import write_json
            from .runtime import owned_marker

            marker = (
                result / "songe_owned.json"
                if result.is_dir()
                else owned_marker(destination)
            )
            write_json(marker, ownership)
        return publish(result, destination)
    finally:
        shutil.rmtree(staging)


__all__ = ["export", "merge"]
