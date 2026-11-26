"""Immutable trainable-base caches. Heavy libraries live in guarded children."""

import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from ..artifacts.runtime import guarded_run, publish
from .gguf import read_gguf


def source_key(path):
    path = Path(path).resolve()
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    # Stat fingerprints avoid reading all of a user's weights just to inspect
    # the cache. Paths + size + nanosecond mtimes invalidate ordinary changes.
    stats = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return hashlib.sha256(json.dumps(stats).encode()).hexdigest()


def require_disk(directory, required_bytes):
    from ..artifacts.storage import preflight

    preflight([(directory, required_bytes)])


def trainable_base(
    ref,
    backend_kind,
    platform,
    *,
    runner=guarded_run,
    artifact_directory=None,
    download=False,
):
    """download: the user chose to train from a GGUF's original weights
    (songe models <model> download); without it nothing contacts the hub."""
    if backend_kind not in {"mlx", "torch"}:
        raise ValueError(f"Unknown backend: {backend_kind}")
    source = Path(ref.path)
    if not source.exists():
        raise FileNotFoundError(
            f"The server's model path is not locally accessible: {source}"
        )
    if ref.format == "mlx":
        return source
    if backend_kind == "mlx" and getattr(platform, "name", sys.platform) != "darwin":
        raise ValueError("The MLX backend requires macOS")
    if ref.format == "safetensors" and backend_kind == "torch":
        return source
    if ref.format not in {"safetensors", "gguf"}:
        raise ValueError(f"Unsupported model format: {ref.format}")
    from ..artifacts.storage import artifact_root
    from ..config import Config
    from ..log import Store

    store = Store(Path(platform.data_dir()))
    if artifact_directory is None:
        artifact_directory = artifact_root(
            Config(upstream=ref.upstream, model=str(source)), store
        )
    cache = Path(artifact_directory) / "models"
    cache.mkdir(parents=True, exist_ok=True)
    key = source_key(source)
    if ref.format == "gguf":
        record = cache / (key + "-origin.json")
        original = downloaded(record) or (
            original_base(source, record) if download else None
        )
        if original is not None:
            if backend_kind == "torch":
                return original
            ref = replace(ref, format="safetensors")
            source = original
    # Check the entire conversion chain before starting the first converter.
    required = 0
    dense_source = source
    if ref.format == "gguf":
        dense_source = cache / (key + "-bf16")
        if not (dense_source / "songe_conversion.json").is_file():
            info = read_gguf(source)
            if info.parameters <= 0:
                raise ValueError("GGUF has no tensors to convert")
            required = info.parameters * 2 + source.stat().st_size
    if (
        backend_kind == "mlx"
        and not (cache / (key + "-mlx") / "songe_conversion.json").is_file()
    ):
        required += (
            required
            if required
            else sum(p.stat().st_size for p in dense_source.rglob("*") if p.is_file())
        )
    require_disk(cache, required)
    if ref.format == "gguf":
        destination = cache / (key + "-bf16")
        marker = destination / "songe_conversion.json"
        if not marker.is_file():
            # Dense bf16 tensors plus the original tokenizer/metadata copied by
            # transformers. No compressed-size multiplier or model-name table.
            info = read_gguf(source)
            if info.parameters <= 0:
                raise ValueError("GGUF has no tensors to convert")
            if not _transformers_reads(source):
                # Final: the scheduler marks the model unsupported with this
                # line instead of retrying "conversion failed" for hours.
                raise ValueError(_untrainable(ref, source, info, download))
            require_disk(cache, info.parameters * 2 + source.stat().st_size)
            _convert(
                source,
                destination,
                "gguf",
                platform,
                runner,
                store.get("installation_id"),
            )
        source = destination
    if backend_kind == "mlx":
        destination = cache / (key + "-mlx")
        if not (destination / "songe_conversion.json").is_file():
            required = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
            require_disk(cache, required)
            _convert(
                source,
                destination,
                "mlx",
                platform,
                runner,
                store.get("installation_id"),
            )
        source = destination
    return source


def _untrainable(ref, source, info, download):
    """Why this GGUF cannot train: an offer to train from the original
    weights it names (read from the file, no network), or plainly not."""
    from ..wording import gguf_original_offer, gguf_untrainable
    from . import origin

    try:
        repo = None if download else origin.named(source)
    except Exception:
        repo = None
    if not repo:
        return gguf_untrainable(ref.architecture)
    from .library import model_name

    size = f"{info.parameters * 2 / 1e9:.0f} GB"  # bf16 originals
    return gguf_original_offer(repo, size, model_name(ref.serving_id))


def downloaded(record):
    """Original weights downloaded earlier for this GGUF, or None."""
    try:
        path = Path(json.loads(record.read_text(encoding="utf-8"))["path"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return path if (path / "config.json").is_file() else None


def fetch_original(config, store):
    """Download a GGUF model's original weights where the measuring child
    will find them, before measuring starts."""
    from ..artifacts.storage import artifact_root

    source = Path(config.model_ref["path"])
    cache = Path(artifact_root(config, store)) / "models"
    cache.mkdir(parents=True, exist_ok=True)
    record = cache / (source_key(source) + "-origin.json")
    return downloaded(record) or original_base(source, record)


def original_base(source, record):
    """The weights the GGUF was quantized from, downloaded into the Hugging
    Face cache, or None. Called only once the user opted in."""
    import logging

    from . import origin

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return None
    try:
        found = origin.original_weights(source)
    except Exception:
        logging.getLogger(__name__).warning("gguf_origin_check_failed", exc_info=True)
        return None
    if found is None:
        return None
    logging.getLogger(__name__).info(
        "gguf_origin %s: %s@%s", source.name, found.repo, found.revision[:12]
    )
    require_disk(
        origin.cache_dir() or record.parent,
        max(0, found.bytes - origin.present_bytes(found)),
    )
    path = origin.download(found)
    from ..storage import write_json

    write_json(
        record,
        {"source": str(source), "repo": found.repo, "revision": found.revision,
         "path": str(path)},
    )
    return path


def _transformers_reads(source):
    """Whether transformers can read this GGUF's architecture (metadata only,
    no tensors), as the converter would."""
    try:
        from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
    except ImportError:
        return True  # Cannot tell here; the converter itself says.
    try:
        load_gguf_checkpoint(str(source), return_tensors=False)
    except Exception as exc:
        # Only its "architecture ... is not supported" answers the question;
        # anything else (a damaged file) is the converter's to report.
        return "not supported" not in str(exc)
    return True


def _convert(source, destination, kind, platform, runner, identity):
    if destination.exists():
        raise FileExistsError(
            f"Incomplete conversion cache: {destination}; remove it before retrying"
        )
    staging = Path(tempfile.mkdtemp(prefix=".converting-", dir=destination.parent))
    from ..storage import private_dir

    private_dir(staging)
    from ..backends.install import interpreter_for
    from ..storage import write_json

    metadata = {"source": str(source), "kind": kind, "installation_id": identity}
    write_json(staging / "songe_converting.json", metadata)
    try:
        runner(
            [
                sys.executable if kind == "mlx" else interpreter_for(platform, runner),
                "-m",
                "songe.models.worker",
                kind,
                str(source),
                str(staging),
            ],
            platform=platform,
        )
        if not (staging / "config.json").is_file() or not list(
            staging.glob("*.safetensors")
        ):
            raise RuntimeError("Converter did not produce a safetensors model")
        write_json(staging / "songe_conversion.json", metadata)
        (staging / "songe_converting.json").unlink()
        publish(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def forget_derived(root, keep):
    """Discard converted gate models while retaining the configured base."""
    from ..artifacts.runtime import owned_marker

    caches, owners = {}, {}
    from ..artifacts.storage import artifact_roots
    from ..log import Store

    identity = Store(root).get("installation_id")

    for marker in (
        marker
        for directory in artifact_roots(root)
        for marker in (directory / "models").glob("*/songe_conversion.json")
    ):
        if marker.parent.is_symlink():
            continue
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            caches[marker.parent.resolve()] = Path(data["source"]).resolve()
            owners[marker.parent.resolve()] = data.get("installation_id")
        except (OSError, ValueError, KeyError):
            continue
    # Keep the configured/served bases and any conversion dependencies they use.
    changed = True
    keep = set(keep)
    while changed:
        dependencies = {source for cache, source in caches.items() if cache in keep}
        changed = not dependencies <= keep
        keep.update(dependencies)
    derived = {
        cache for cache in caches if owners.get(cache) == identity and cache not in keep
    }
    changed = True
    while changed:
        changed = False
        for cache, source in caches.items():
            if cache in derived or cache in keep:
                continue
            if (
                source in derived
                or source.is_relative_to(root / "versions")
                or owned_marker(source).is_file()
            ):
                derived.add(cache)
                changed = True
    for cache in derived:
        shutil.rmtree(cache)
