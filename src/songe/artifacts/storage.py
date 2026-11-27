"""Owned artifact space, separate from the small experience database."""

import json
import math
import re
import shutil
from pathlib import Path

from ..storage import private_dir, write_json

MARKER = "songe_artifacts.json"


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def volume(path):
    path = existing_parent(path)
    device = path.stat().st_dev
    while path.parent != path and path.parent.stat().st_dev == device:
        path = path.parent
    return path


RESERVE_FRACTION = 0.01  # of the volume, left free beyond the work's need
RESERVE_MAX_BYTES = 2e9


class StorageWait(OSError):
    """An environmental condition that should be retried in a later idle gap."""


class DiskSpace(StorageWait):
    def __init__(self, target, required, available):
        self.target, self.required, self.available = target, required, available
        super().__init__(
            f"Needs {required / 1e9:.1f} GB free on {target}; "
            f"{available / 1e9:.1f} GB available"
        )


def preflight(requirements):
    """Sum simultaneous allocations on each volume, including shared targets."""
    volumes = {}
    for path, required in requirements:
        target = volume(path)
        volumes[target] = volumes.get(target, 0) + int(required)
    for target, required in volumes.items():
        usage = shutil.disk_usage(target)
        # Never plan to fill a disk to the last byte: the system needs room
        # (swap, other apps) while the work runs.
        reserve = (
            min(RESERVE_FRACTION * getattr(usage, "total", 0), RESERVE_MAX_BYTES)
            if required
            else 0
        )
        if usage.free < required + reserve:
            raise DiskSpace(target, int(required + reserve), usage.free)


def require_publish_volume(scratch, destination):
    if volume(scratch) != volume(destination.parent):
        raise StorageWait(
            f"Artifact staging and publication must share a volume: {scratch}, {destination.parent}"
        )


def artifact_root(config, store, *, create=True):
    """Prefer the library volume. LM Studio must never see scratch models."""
    library = (
        config.models_dir
        if config.upstream == "lmstudio"
        else (config.ollama_models_dir if config.upstream == "ollama" else "")
    )
    if library:
        library = Path(library).resolve()
        if not library.is_dir():
            raise StorageWait(
                f"Serving library is unavailable: {library}; reconnect it and retry"
            )
        if volume(library) == volume(
            store.root
        ) and not store.root.resolve().is_relative_to(library):
            return store.root
        parent = library.parent
        if volume(parent) != volume(library):
            if config.upstream == "lmstudio":
                raise StorageWait(
                    f"Needs a scratch directory outside {library} on {volume(library)}; "
                    "the library occupies the whole volume"
                )
            parent = library
    else:
        source = Path(config.model_ref.get("path") or config.model or store.root)
        if not source.exists() or source.resolve().is_relative_to(store.root.resolve()):
            return store.root
        parent = source.resolve().parent
        if volume(parent) != volume(source):
            parent = source.resolve() if source.is_dir() else source.parent
    # Sharing a volume with the data directory already permits atomic renames.
    # LM Studio still needs scratch outside its indexed downloadsFolder.
    if volume(parent) == volume(store.root) and not library:
        return store.root
    identity = store.get("installation_id")
    destination = parent / f".songe-{identity}"
    expected = {"installation_id": identity, "data_root": str(store.root.resolve())}
    marker = destination / MARKER
    if (
        destination.is_symlink()
        or destination.exists()
        and (
            not marker.is_file()
            or json.loads(marker.read_text(encoding="utf-8")) != expected
        )
    ):
        raise StorageWait(
            f"Artifact directory is not owned by this installation: {destination}"
        )
    if create:
        registry = store.root / "artifact-roots.json"
        roots = (
            json.loads(registry.read_text(encoding="utf-8"))
            if registry.is_file()
            else []
        )
        if str(destination) not in roots:
            write_json(registry, roots + [str(destination)])
        private_dir(destination)
        write_json(marker, expected)
    return destination


def artifact_roots(root):
    """Only the durable registry plus a matching marker can authorize cleanup."""
    from ..log import Store

    root = Path(root).resolve()
    identity = Store(root).get("installation_id")
    yield root
    registry = root / "artifact-roots.json"
    for entry in (
        json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else []
    ):
        path = Path(entry)
        marker = path / MARKER
        try:
            owned = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            not path.is_symlink()
            and path.resolve() != root
            and owned == {"installation_id": identity, "data_root": str(root)}
        ):
            yield path


def purge_roots(store):
    for path in artifact_roots(store.root):
        if path != store.root.resolve():
            shutil.rmtree(path)
    (store.root / "artifact-roots.json").unlink(missing_ok=True)


def tensor_sizes(model, rank):
    """Read tensor headers only; no model runtime, allocations, or weights."""
    model = Path(model)
    dense, adapter = 0, 0
    config_file = model / "config.json"
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
        bits = config.get("quantization", config.get("quantization_config", {})).get(
            "bits", 32
        )
    except (OSError, ValueError, AttributeError):
        config, bits = {}, 32
    # A model trained on named layers only (Flash-Next: its 46k expert
    # matrices stay frozen) is budgeted for those layers, not every matrix.
    from ..backends import packed_experts

    targets = (
        re.compile(packed_experts.REPACKED_LORA_TARGETS)
        if isinstance(config, dict)
        and config.get("model_type") in packed_experts.REPACKED_TYPES
        else None
    )
    for file in model.glob("*.safetensors") if model.is_dir() else []:
        try:
            with file.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 0 < size <= min(file.stat().st_size, 100_000_000):
                    continue
                header = json.loads(stream.read(size))
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                if name.endswith((".scales", ".biases")) and bits < 32:
                    continue
                shape = list(tensor["shape"])
                if (
                    tensor.get("dtype") == "U32"
                    and bits < 32
                    and name.endswith(".weight")
                ):
                    shape[-1] *= 32 // bits
                dense += math.prod(shape) * 2
                if len(shape) >= 2 and (
                    targets is None or targets.fullmatch(name.removesuffix(".weight"))
                ):
                    # All matrices is an upper bound on the selected LoRA layers.
                    adapter += rank * sum(shape[-2:]) * math.prod(shape[:-2]) * 4
        except (OSError, ValueError, KeyError, TypeError):
            continue
    from ..upstream.protection import weight_bytes

    measured = weight_bytes(model)
    return max(dense, measured), adapter or measured


def preserving_fuse(model):
    """A base quantized below 8 bits is fused at 8 bits (sleep.fuse_worker)."""
    try:
        quantization = (
            json.loads((Path(model) / "config.json").read_text(encoding="utf-8")).get(
                "quantization"
            )
            or {}
        )
    except (OSError, ValueError):
        return False
    bits = quantization.get("bits")
    return isinstance(bits, int) and bits < 8


def held_bytes(store, target):
    """What interrupted nights already wrote on the artifact volume. Stale
    ones are discarded before admission (sleep.cleanup), so these resume."""
    total = 0
    for night in store.nights():
        work = night["record"].get("work")
        if night["status"] != "interrupted" or not work:
            continue
        if volume(work) != volume(target):
            continue
        for file in Path(work).rglob("*"):
            try:
                if file.is_file() and not file.is_symlink():
                    total += file.stat().st_size
            except OSError:
                continue
    return total


def phase_requirements(config, store, platform, *, calibration=False):
    """Peak disk budget from local tensor geometry and measured file sizes."""
    from ..backends import resolve_kind
    from ..models.convert import source_key
    from ..models.gguf import read_gguf
    from ..upstream.protection import weight_bytes

    target = artifact_root(config, store, create=False)
    source = Path(
        config.model_ref.get("path") or config.model or store.root / "missing-model"
    )
    dense, adapter = tensor_sizes(
        Path(config.model) if config.model else source, config.rank
    )
    original = weight_bytes(source)
    conversions = 0
    backend = resolve_kind(config, platform)
    if source.is_file() and config.model_ref.get("format") == "gguf":
        dense = read_gguf(source).parameters * 2
        if Path(config.model) == source:
            from gguf import GGUFReader

            reader = GGUFReader(source, mode="r")
            # GGUF dimensions are reversed relative to the dense framework shape.
            adapter = (
                sum(
                    config.rank
                    * sum(map(int, t.shape[:2]))
                    * math.prod(map(int, t.shape[2:]))
                    * 4
                    for t in reader.tensors
                    if len(t.shape) >= 2
                )
                or dense
            )
        key = source_key(source)
        if not (
            target / "models" / (key + "-bf16") / "songe_conversion.json"
        ).is_file():
            conversions += dense + original
        if (
            backend == "mlx"
            and not (
                target / "models" / (key + "-mlx") / "songe_conversion.json"
            ).is_file()
        ):
            conversions += dense
    elif backend == "mlx" and config.model_ref.get("format") == "safetensors":
        key = source_key(source)
        if not (target / "models" / (key + "-mlx") / "songe_conversion.json").is_file():
            conversions += original
    from ..sleep.fuse import candidate_kind

    kind = candidate_kind(config, platform)
    # Training and the gate never overlap on disk: checkpoints are removed once
    # training is recorded, and failed strengths before the next is built.
    # Training: two checkpoint generations while one is written (weights +
    # Adam moments, three adapter sizes each) and the final adapter.
    training = 7 * adapter
    # Gate: the final adapter, one scaled adapter and one candidate.
    gate = 2 * adapter
    if kind in {"mlx-fused", "gguf-full"}:
        artifact = original if kind == "gguf-full" and original else dense
        if kind == "mlx-fused" and original and preserving_fuse(source):
            # Adapted layers are stored at 8 bits (sleep.fuse_worker), the rest
            # keeps the base's own quantization: at most twice a 4-bit base.
            artifact = min(dense, 2 * original)
        gate += artifact
        if kind == "gguf-full":
            # Merge + converter intermediate + evaluation's dense cache.
            gate += 3 * dense
            if backend == "mlx":
                gate += 2 * dense
    else:
        gate += adapter
    scratch = conversions + max(training, gate)
    if not calibration:
        # A resumed night's files are part of this peak, not on top of it.
        scratch = max(0, scratch - held_bytes(store, target))
    requirements = [(target, scratch), (store.root, store.path.stat().st_size)]
    if config.upstream == "lmstudio" and config.models_dir:
        requirements.append((Path(config.models_dir), 0))
    if config.upstream == "ollama" and config.ollama_models_dir:
        # Import creates a runtime blob while our export remains alive.
        requirements.append((Path(config.ollama_models_dir), original or dense))
    return requirements


# Free space kept for swap on the disk it grows on (#68). macOS adds 1 GB
# swap files to the startup disk as memory runs short, and nothing bounds
# them but free space: one measuring run on a crowded 24 GB Mac took that
# disk from 9 GB free to 3 GB. Measuring probes memory limits on purpose, and
# training can page too, so neither starts with less than SWAP_START_BYTES
# free (room for several swap files above the stop line), and a run is
# stopped, its progress saved, below SWAP_STOP_BYTES: enough left for the
# save, swap already being written and the system itself, before macOS
# reaches its own "disk almost full" trouble.
SWAP_START_BYTES = 8e9
SWAP_STOP_BYTES = 4e9


def swap_space(platform, *, running=False):
    """Why work must not start (or, running, must stop) for lack of free
    space where swap grows; None when there is room or swap cannot grow."""
    found = getattr(platform, "swap_volume", None)
    found = found() if callable(found) else None
    if not found:
        return None
    path, name = found
    try:
        free = shutil.disk_usage(existing_parent(path)).free
    except OSError:
        return None  # Unreadable: the memory guard still watches paging.
    required = SWAP_STOP_BYTES if running else SWAP_START_BYTES
    if free >= required:
        return None
    # The same words as DiskSpace, so every surface reads it as a disk wait.
    return f"Needs {required / 1e9:.1f} GB free on {name}; {free / 1e9:.1f} GB available"


def mlx_gguf_lora_bytes(adapter):
    from ..upstream.protection import weight_bytes

    # The PEFT copy keeps the saved tensor dtypes. Allow the GGUF to use up
    # to twice that space for tensors stored as float32.
    return 3 * weight_bytes(adapter)


def export_requirements(config, store, candidate):
    target = artifact_root(config, store, create=False)
    destination = Path(candidate["artifact"])
    require_publish_volume(target, destination)
    # An already-published file needs no staging space. The exporter checks
    # ownership before it can reuse the file.
    if destination.exists():
        return []
    return [(target, mlx_gguf_lora_bytes(candidate["training_adapter"]))]


def admit_storage(config, store, platform, *, calibration=False, pending_export=None):
    try:
        requirements = (
            export_requirements(config, store, pending_export)
            if pending_export is not None
            else phase_requirements(config, store, platform, calibration=calibration)
        )
        preflight(requirements)
        if wait := swap_space(platform):
            raise StorageWait(wait)
    except StorageWait as exc:
        store.set("disk_wait", str(exc))
        raise
    store.set("disk_wait", None)


def collect_scratch(store):
    """Called under the worker lease; remove marked abandoned scratch only."""
    identity = store.get("installation_id")
    for root in artifact_roots(store.root):
        markers = [
            *(root / "models").glob(".converting-*/songe_converting.json"),
            *root.glob(".export-*/songe_owned.json"),
            *root.glob(".mlx-export-*/songe_owned.json"),
        ]
        for marker in markers:
            try:
                owned = (
                    json.loads(marker.read_text(encoding="utf-8")).get(
                        "installation_id"
                    )
                    == identity
                )
            except (OSError, ValueError):
                continue
            if owned and not marker.parent.is_symlink():
                shutil.rmtree(marker.parent)
        for marker in root.glob(".*/songe_fusing.json"):
            try:
                meta = json.loads(marker.read_text(encoding="utf-8"))
                work = Path(meta["adapter"]).parent
                destination = Path(meta["destination"])
            except (OSError, ValueError, KeyError):
                continue
            if marker.parent.is_symlink() or not marker.parent.name.startswith(
                "." + destination.name + "-"
            ):
                continue
            try:
                owned = json.loads(
                    (work / "songe_work.json").read_text(encoding="utf-8")
                ).get("installation_id")
            except (OSError, ValueError):
                # The work it belonged to is gone (a measuring run that was
                # stopped mid-fuse): inside our own root it is ours to remove.
                owned = (
                    identity
                    if not work.exists() and destination.is_relative_to(root)
                    else None
                )
            if owned == identity:
                shutil.rmtree(marker.parent)


def prune_roots(store):
    """Forget may leave only empty containers; keep bases and live versions."""
    for root in artifact_roots(store.root):
        if root == store.root.resolve():
            continue
        for name in ("models", "versions"):
            directory = root / name
            if (
                directory.is_dir()
                and not directory.is_symlink()
                and not any(directory.iterdir())
            ):
                directory.rmdir()
        if set(root.iterdir()) == {root / MARKER}:
            (root / MARKER).unlink()
            root.rmdir()


def collect_idle_scratch(store):
    from ..platform.locking import acquire
    from ..sleep.calibrate import collect_calibration

    with (store.root / "worker.lock").open("a+") as lease:
        try:
            acquire(lease)
        except BlockingIOError:
            return
        collect_calibration(store.root, store, leased=True)
        collect_scratch(store)
