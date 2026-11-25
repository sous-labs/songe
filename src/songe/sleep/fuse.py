import functools
import os
import shutil
import subprocess
import sys
import uuid

from ..memory import guarded_run
from ..storage import private_dir, private_tree, write_json


def fuse(
    config, adapter, destination, check=lambda: None, *, scratch=None, progress=None
):
    """Reuse official fuse; stage a new immutable directory. No in-place models."""
    from ..artifacts.storage import require_publish_volume

    require_publish_volume(scratch or destination.parent, destination)
    staging = (scratch or destination.parent) / (
        "." + destination.name + "-" + uuid.uuid4().hex
    )
    if destination.exists():
        raise FileExistsError(destination)
    staging.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A crash can leave an unpublished staging directory. Only remove marked
    # directories for this exact destination and base, never arbitrary models.
    for old in staging.parent.glob("." + destination.name + "-*"):
        marker = old / "songe_fusing.json"
        if marker.is_file():
            import json

            try:
                meta = json.loads(marker.read_text(encoding="utf-8"))
                if meta == {
                    "base": config.model,
                    "destination": str(destination),
                    "adapter": str(adapter),
                }:
                    shutil.rmtree(old)
            except (OSError, ValueError):
                pass
    from ..models.convert import require_disk
    from ..upstream.protection import weight_bytes

    # A 4-bit base is fused at 8 bits (fuse_worker.py): up to twice its size.
    require_disk(staging.parent, 2 * weight_bytes(config.model) + weight_bytes(adapter))
    private_dir(staging)
    write_json(
        staging / "songe_fusing.json",
        {
            "base": config.model,
            "destination": str(destination),
            "adapter": str(adapter),
        },
    )
    command = [
        sys.executable,
        "-m",
        "songe.sleep.fuse_worker",
        "--model",
        config.model,
        "--adapter-path",
        str(adapter),
        "--save-path",
        str(staging),
    ] + (["--songe-progress", str(progress)] if progress else [])
    try:
        proc = guarded_run(
            command, check=check, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if proc.returncode:
            from ..telemetry import failure_hint, log_child_stderr

            log_child_stderr(getattr(proc, "stderr", None))
            raise RuntimeError(failure_hint("Model fusion"))
        write_json(
            staging / "songe_owned.json",
            {
                "base": config.model,
                "adapter": str(adapter),
                "bits": "base",
                "installation_id": work_identity(adapter),
            },
        )
        private_tree(staging)
        for path in staging.rglob("*"):
            if path.is_file():
                with path.open("r+b") as f:
                    os.fsync(f.fileno())
        staging.rename(destination)
        if os.name != "nt":
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def candidate_kind(config, platform):
    from ..backends import resolve_kind

    if config.artifact_kind:
        return config.artifact_kind
    backend = resolve_kind(config, platform)
    return {
        "lmstudio": "mlx-fused" if backend == "mlx" else "gguf-full",
        "mlx": "mlx-adapter",
        "ollama": "gguf-full",
        "llamacpp": "gguf-lora",
        "vllm": "peft-adapter",
    }[config.upstream]


def original_ref(config):
    from pathlib import Path

    from ..models import ModelRef

    if not config.model_ref:
        return None
    return ModelRef(**(config.model_ref | {"path": Path(config.model_ref["path"])}))


# Fraction of layers merged by the running fuse; the dashboard shows it while
# a version is being rebuilt for a return.
PROGRESS = "fuse-progress.json"


def export_candidate(
    config, kind, adapter, destination, platform, check, *, progress=None
):
    from ..artifacts import export
    from ..artifacts.runtime import guarded_run, owned_marker

    destination.parent.mkdir(parents=True, exist_ok=True)
    from ..artifacts.storage import (
        artifact_root,
        mlx_gguf_lora_bytes,
        require_publish_volume,
    )
    from ..log import Store
    from ..models.convert import require_disk
    from ..platform import Rooted
    from ..upstream.protection import weight_bytes

    store = Store(platform.data_dir())
    scratch = artifact_root(config, store)
    require_publish_volume(scratch, destination)
    # Evaluation conversions and converter scratch use the same artifact volume.
    export_platform = Rooted(platform, scratch)
    ownership = {
        "base": config.model,
        "adapter": str(adapter),
        "installation_id": work_identity(adapter),
    }
    if destination.exists():
        import json

        marker = owned_marker(destination)
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if all(saved.get(k) == v for k, v in ownership.items()):
                return destination
        raise RuntimeError("Candidate destination belongs to another run")

    mlx_adapter = (adapter / "adapters.safetensors").is_file()
    required = (
        mlx_gguf_lora_bytes(adapter)
        if kind == "gguf-lora" and mlx_adapter
        else weight_bytes(config.model) + weight_bytes(adapter)
    )
    require_disk(scratch, required)

    if kind == "mlx-fused":
        return fuse(
            config,
            adapter,
            destination,
            check,
            scratch=scratch,
            progress=progress,
        )

    # A partial, not a new function: interpreter_for recognises the real
    # runner through it and gives conversions the converter runtime.
    runner = functools.partial(guarded_run, check=check)

    if mlx_adapter:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory(
            prefix=".mlx-export-", dir=scratch
        ) as temporary:
            temporary = Path(temporary)
            private_dir(temporary)
            write_json(temporary / "songe_owned.json", ownership)
            if kind == "gguf-full":
                dense = temporary / "dense"
                runner(
                    [
                        sys.executable,
                        "-m",
                        "songe.sleep.fuse_worker",
                        "--model",
                        config.model,
                        "--adapter-path",
                        str(adapter),
                        "--save-path",
                        str(dense),
                        "--dequantize",
                    ],
                    platform=platform,
                )
                return export(
                    kind,
                    dense,
                    adapter,
                    destination,
                    export_platform,
                    ref=original_ref(config),
                    runner=runner,
                    ownership=ownership,
                    merged_base=dense,
                    scratch=scratch,
                )
            peft = temporary / "peft"
            runner(
                [
                    sys.executable,
                    "-m",
                    "songe.artifacts.worker",
                    "--mlx-peft",
                    config.model,
                    str(adapter),
                    str(peft),
                ],
                platform=platform,
            )
            return export(
                kind,
                config.model,
                peft,
                destination,
                export_platform,
                ref=original_ref(config),
                runner=runner,
                ownership=ownership,
                scratch=scratch,
            )
    return export(
        kind,
        config.model,
        adapter,
        destination,
        export_platform,
        ref=original_ref(config),
        runner=runner,
        ownership=ownership,
        scratch=scratch,
    )


def evaluation_model(config, kind, artifact, platform, check):
    if kind != "gguf-full":
        return artifact
    from ..artifacts.runtime import guarded_run
    from ..backends import resolve_kind
    from ..models import ModelRef, trainable_base
    from ..models.gguf import read_gguf

    ref = ModelRef(
        config.upstream, "candidate", "gguf", artifact, read_gguf(artifact).architecture
    )
    backend = resolve_kind(config, platform)

    # A partial, not a new function: interpreter_for recognises the real
    # runner through it and gives conversions the converter runtime.
    runner = functools.partial(guarded_run, check=check)

    from ..artifacts.storage import artifact_root
    from ..log import Store

    cache_root = artifact_root(config, Store(platform.data_dir()))
    if (artifact.parent / "songe_calibration.json").is_file():
        cache_root = artifact.parent
    return trainable_base(
        ref,
        backend,
        platform,
        runner=runner,
        artifact_directory=cache_root,
    )


def work_identity(adapter):
    import json
    from pathlib import Path

    parent = Path(adapter).parent
    marker = parent / "songe_work.json"
    if not marker.is_file():
        marker = parent / "songe_calibration.json"
    return (
        json.loads(marker.read_text(encoding="utf-8")).get("installation_id")
        if marker.is_file()
        else None
    )
