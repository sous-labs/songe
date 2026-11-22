"""Read-only serving inventory, independent of Songe's learner registry."""

import functools
import json
import re
import subprocess
import threading
import time
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import httpx

from ..interface_copy import terse
from . import ModelRef, discovery, registry


def _cli(executable, command):
    result = subprocess.run(
        [executable, command, "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=5,
        check=True,
    )
    data = json.loads(result.stdout)
    return data.get("models", []) if isinstance(data, dict) else data


def _size(path):
    from ..upstream.protection import weight_bytes

    try:
        return weight_bytes(path) if path.exists() else None
    except OSError:
        return None


def _item(ref, *, loaded=False, size=None):
    return {
        "ref": ref,
        "size_bytes": size if size is not None else _size(ref.path),
        "loaded": loaded,
    }


def scan(config):
    """Only library listings and server metadata. Never start/load a model."""
    if config.upstream == "lmstudio":
        folder = Path(config.models_dir or discovery.lmstudio_downloads())
        refs = discovery.lmstudio(folder=folder)
        items = {str(r.path.resolve()): _item(r) for r in refs}
        if config.lms:
            try:
                for model in _cli(config.lms, "ls"):
                    if not isinstance(model, dict) or not (
                        model.get("path") or model.get("modelKey")
                    ):
                        continue
                    path = Path(model.get("path") or model["modelKey"])
                    if not path.is_absolute():
                        path = folder / path
                    key = str(path.resolve())
                    if key not in items:
                        fmt = model.get("format") or (
                            "gguf" if path.suffix.lower() == ".gguf" else "safetensors"
                        )
                        ref = ModelRef(
                            "lmstudio",
                            model.get("modelKey") or str(path),
                            fmt.lower(),
                            path,
                            model.get("architecture", ""),
                        )
                        items[key] = _item(ref, size=model.get("sizeBytes"))
                    if isinstance(model.get("sizeBytes"), (int, float)):
                        items[key]["size_bytes"] = max(0, int(model["sizeBytes"]))
                for model in _cli(config.lms, "ps"):
                    if not model.get("path"):
                        continue
                    path = folder / model["path"]
                    if str(path.resolve()) in items:
                        items[str(path.resolve())]["loaded"] = True
            except (OSError, ValueError, TypeError, subprocess.SubprocessError):
                pass  # Offline CLI still leaves the local library visible.
        return list(items.values())
    if config.upstream == "ollama":
        root = config.upstream_url.rstrip("/").removesuffix("/v1")
        # Tags made with `ollama create` are only listed by its API.
        refs = {
            r.serving_id: r
            for r in discovery.ollama(folder=config.ollama_models_dir or None, url=root)
        }
        try:
            tags = discovery._get(root + "/api/tags").get("models", [])
        except (OSError, ValueError, httpx.HTTPError):
            return [_item(r) for r in refs.values()]
        try:
            loaded = {
                m.get("name", m.get("model"))
                for m in discovery._get(root + "/api/ps").get("models", [])
            }
        except (OSError, ValueError, httpx.HTTPError):
            loaded = set()
        result = []
        for model in tags:
            name = model.get("name", model.get("model"))
            if not name:
                continue
            details = model.get("details") or {}
            ref = refs.get(name) or ModelRef(
                "ollama",
                name,
                details.get("format", "gguf"),
                Path("/songe-unavailable") / name,
                details.get("family", ""),
            )
            result.append(_item(ref, loaded=name in loaded, size=model.get("size")))
        return result
    if config.upstream == "llamacpp":
        root = config.upstream_url.rstrip("/").removesuffix("/v1")
        models = discovery._get(root + "/v1/models").get("data", [])
        try:
            props = discovery._get(root + "/props")
        except (OSError, ValueError, httpx.HTTPError):
            props = {}
        result = []
        for index, model in enumerate(models):
            if not isinstance(model, dict) or not model.get("id"):
                continue
            path = Path(
                model.get("root")
                or model.get("path")
                or (props.get("model_path") if index == 0 else None)
                or model["id"]
            )
            info = (
                discovery.readable(path, lambda path=path: discovery.read_gguf(path))
                if path.is_file()
                else None
            )
            ref = ModelRef(
                "llamacpp",
                model["id"],
                "gguf",
                path,
                info.architecture if info else model.get("model_type", ""),
            )
            result.append(_item(ref, loaded=True, size=model.get("sizeBytes")))
        return result
    base = config.upstream_url.rstrip("/").removesuffix("/v1")
    result = []
    for model in discovery._get(base + "/v1/models").get("data", []):
        if not isinstance(model, dict) or not model.get("id"):
            continue
        if model.get("parent"):
            result.append(
                _item(
                    ModelRef(
                        config.upstream,
                        model["id"],
                        "peft-adapter",
                        discovery._cached_path(model.get("root") or model["id"]),
                        "",
                    ),
                    loaded=True,
                    size=model.get("sizeBytes"),
                )
            )
            continue
        ref = discovery.readable(
            model["id"],
            lambda model=model: discovery._openai_ref(model, config.upstream),
        )
        if ref is None:
            ref = ModelRef(
                config.upstream,
                model["id"],
                "mlx" if config.upstream == "mlx" else "safetensors",
                discovery._cached_path(model.get("root") or model["id"]),
                model.get("model_type", ""),
            )
        result.append(_item(ref, loaded=True, size=model.get("sizeBytes")))
    return result


@functools.lru_cache(maxsize=64)
def architecture_available(architecture, backend):
    """Inspect installed package metadata without importing any ML runtime.
    Cached: listing the package's files costs most of a second, and packages
    only change with a reinstall, which restarts the service."""
    try:
        package = distribution("mlx-lm" if backend == "mlx" else "transformers")
    except PackageNotFoundError:
        return None

    def normalize(s):
        return s.lower().replace("_", "").replace("-", "")

    names = {
        normalize(Path(str(p)).stem if backend == "mlx" else Path(str(p)).parent.name)
        for p in package.files or []
        if (
            str(p).startswith("mlx_lm/models/")
            if backend == "mlx"
            else str(p).startswith("transformers/models/")
            and Path(str(p)).name == "configuration_" + Path(str(p)).parent.name + ".py"
        )
    }
    return normalize(architecture) in names if names and architecture else None


def is_embedding(ref):
    embedding = ref.architecture in {
        "bert",
        "nomic-bert",
        "nomic_bert",
        "jina-bert-v2",
        "gemma-embedding",
    } or bool(re.search(r"embed(?:ding)?|bge-|e5-", ref.serving_id, re.I))
    if ref.path.is_dir():
        metadata = (
            discovery.readable(
                ref.path,
                lambda: json.loads(
                    (ref.path / "config.json").read_text(encoding="utf-8")
                ),
            )
            or {}
        )
        embedding = (
            embedding
            or (ref.path / "modules.json").is_file()
            or any("embedding" in a.lower() for a in metadata.get("architectures", []))
        )
    if ref.format == "gguf" and ref.path.is_file():
        info = discovery.readable(ref.path, lambda: discovery.read_gguf(ref.path))
        embedding = embedding or bool(info and getattr(info, "embedding", False))
    return embedding


def support(config, ref, size, platform):
    from ..backends import resolve_kind

    if is_embedding(ref):
        return "unsupported", "embedding model"
    if ref.format not in {"gguf", "safetensors", "mlx"}:
        return (
            "unsupported",
            "Serving adapter or format cannot be trained directly; choose its base model",
        )
    if not ref.path.exists():
        return "unsupported", "Local weights are not accessible; serving only"
    backend = resolve_kind(config, platform)
    available = architecture_available(ref.architecture, backend)
    if available is False:
        return "unsupported", f"{ref.architecture} not supported yet"
    # File bytes are a conservative lower bound, not a fitted training budget.
    # GGUF-to-torch needs at least the dequantized bf16 weights as well.
    required = size or 0
    if ref.format == "gguf" and backend == "torch":
        info = discovery.readable(ref.path, lambda: discovery.read_gguf(ref.path))
        if info:
            required = max(required, info.parameters * 2)
    memory = platform.memory()
    devices = platform.accelerators()
    capacity = (
        memory.total
        if backend == "mlx"
        else max(
            [memory.total]
            if config.force_cpu_training
            else [a.memory_total for a in devices if a.kind in {"cuda", "rocm"}],
            default=memory.total,
        )
    )
    if capacity and required > capacity:
        return (
            "unsupported",
            f"Too large to train on this machine — estimate: weights alone need at least {required / 1e9:.1f} GB; {capacity / 1e9:.1f} GB total",
        )
    return "pending", "not measured yet"


_SCAN = threading.Lock()


def probe_once(platform):
    """Memoize memory/accelerator probes for one snapshot's duration."""
    cache = {}

    class Once:
        def __getattr__(self, name):
            attr = getattr(platform, name)
            if name not in {"memory", "accelerators"}:
                return attr

            def call():
                if name not in cache:
                    cache[name] = attr()
                return cache[name]

            return call

    return Once()


def summaries(config, store, scheduler=None):
    from ..platform import current

    known = registry.summaries(config, store)
    try:
        owner = scheduler if scheduler is not None else store
        cached = getattr(owner, "library_inventory", None)
        if cached and time.monotonic() - cached[0] < 60:
            items = cached[1]
        elif cached and not _SCAN.acquire(blocking=False):
            items = cached[1]  # one scan at a time; others read the last one
        else:
            if not cached:
                _SCAN.acquire()
            try:
                try:
                    items = scan(config)
                except (OSError, ValueError, TypeError, RuntimeError, httpx.HTTPError):
                    items = []
                owner.library_inventory = (time.monotonic(), items)
            finally:
                _SCAN.release()
    except (OSError, ValueError, TypeError, RuntimeError, httpx.HTTPError):
        items = []
    # One hardware probe per snapshot; each probe forks vm_stat/sysctl.
    platform = probe_once(getattr(scheduler, "platform", None) or current())
    entries = {r["id"]: r for r in registry.entries(store)}
    result = list(known)
    derived = {
        str(Path(v["artifact"]).resolve())
        for row in entries.values()
        for v in store.scoped(row["id"]).versions()
        if v.get("adapter")
    }
    for item in items:
        ref = item["ref"]
        if str(ref.path.resolve()) in derived:
            continue
        row = next((r for r in known if r["id"] == ref.serving_id), None)
        if row is None and config.upstream == "lmstudio":
            row = next(
                (
                    r
                    for r in known
                    if str(
                        Path(
                            entries[r["id"]]["config"].get("model_ref", {}).get("path")
                            or entries[r["id"]]["config"].get("model")
                            or "/missing"
                        ).resolve()
                    )
                    == str(ref.path.resolve())
                ),
                None,
            )
        if row is None:
            state, reason = support(config, ref, item["size_bytes"], platform)
            row = {
                "id": ref.serving_id,
                "enabled": False,
                "auto_dream": False,
                "learning_level": "off",
                "support": state,
                "status": "unsupported" if state == "unsupported" else "off",
                "reason": reason,
                "pending": 0,
                "versions": [],
                "version": None,
                "can_undo": False,
                "last_result": None,
                "last_trained": 0,
            }
            result.append(row)
        if row["support"] == "pending":
            state, reason = support(config, ref, item["size_bytes"], platform)
            if state == "unsupported":
                row.update(support=state, status=state, reason=reason)
        if is_embedding(ref):
            row.update(
                support="unsupported", status="unsupported", reason="embedding model"
            )
        if "calibration paused · memory" in row.get("reason", ""):
            row["reason"] = "calibration paused · memory · retry when idle"
        row.update(
            name=ref.serving_id,
            path=str(ref.path),
            size_bytes=item["size_bytes"],
            format=ref.format,
            state="loaded" if item["loaded"] else "downloaded",
        )
    in_use = getattr(scheduler, "in_flight", {}) if scheduler else {}
    for row in result:
        row.setdefault("name", row["id"])
        binding = entries.get(row["id"], {}).get("config", {})
        row.setdefault("format", binding.get("model_ref", {}).get("format", "unknown"))
        row["reason"] = terse(
            row.get("reason")
            or ("not measured yet" if row.get("support") == "pending" else "")
        )
        if "calibration paused · memory" in row.get("reason", ""):
            row["reason"] = "calibration paused · memory · retry when idle"
        source = binding.get("model_ref", {}).get("path") or binding.get("model")
        row.setdefault("size_bytes", _size(Path(source)) if source else None)
        row.setdefault("state", "downloaded")
        if any(
            i["loaded"]
            and str(i["ref"].path.resolve()) == str(Path(v["artifact"]).resolve())
            for i in items
            for v in row["versions"]
        ):
            row["state"] = "loaded"
        if row.get("version") and in_use.get(row["version"]["id"], 0):
            row["state"] = "in use"
        row["versions"] = [
            v
            | {
                "retention": "full" if Path(v["artifact"]).exists() else "adapter only",
                "current": v["id"] == (row.get("version") or {}).get("id"),
            }
            for v in row["versions"]
        ]
    return result


def model_name(name):
    """Remove packaging/quantization suffixes, retaining model/size/fine-tune."""
    name = name.rsplit("/", 1)[-1]
    name = re.sub(r"\.(gguf|safetensors)$", "", name, flags=re.I)
    name = re.sub(
        r"[-_. ](?:q\d[\w]*|iq\d[\w]*|\d+[-_]?bit|bf16|fp16|f16|f32|gguf|mlx|safetensors)(?=[-_. ]|$)",
        "",
        name,
        flags=re.I,
    )
    return name.strip("-_. ")


def order_models(models, order):
    """Unsupported last; models turned on, most recent first, stay on top
    even after being turned off; the rest keep library order."""
    rank = {model: i for i, model in enumerate(order or [])}

    def key(item):
        index, model = item
        ids = [model["id"], *(f["id"] for f in model.get("files", []))]
        placed = min((rank[i] for i in ids if i in rank), default=None)
        unsupported = "unsupported" in (model.get("support"), model.get("status"))
        if placed is not None:
            return (unsupported, 0, placed, index)
        return (unsupported, 0 if model.get("enabled") else 1, len(rank), index)

    return [model for _, model in sorted(enumerate(models), key=key)]


def group_models(models):
    groups = {}
    for model in models:
        name = model_name(model.get("name") or model["id"])
        groups.setdefault(name.casefold(), []).append(model)
    result = []
    for files in groups.values():
        preferred = max(
            files,
            key=lambda m: (
                bool(m.get("enabled")),
                m.get("support") == "trainable",
                bool(m.get("learning")),
            ),
        )
        row = preferred.copy()
        row["name"] = model_name(preferred.get("name") or preferred["id"])
        row["files"] = [
            {
                k: v
                for k, v in f.items()
                if k
                not in {
                    "learning",
                    "versions",
                    "version",
                    "last_result",
                    "last_update",
                    "last_status",
                }
            }
            for f in files
        ]
        if len(files) > 1:
            row["format"] = ", ".join(
                dict.fromkeys(f.get("format", "unknown") for f in files)
            )
            row["size_bytes"] = (
                sum(f["size_bytes"] for f in files)
                if all(f.get("size_bytes") is not None for f in files)
                else None
            )
            row["state"] = max(
                (f.get("state", "downloaded") for f in files),
                key=lambda s: {"downloaded": 0, "loaded": 1, "in use": 2}.get(s, 0),
            )
        row["versions"] = [
            v | {"model": f["id"]} for f in files for v in f.get("versions", [])
        ]
        result.append(row)
    return result


def materialize(config, store, model):
    if registry.find(store, model):
        return
    items = scan(config)
    item = next((i for i in items if i["ref"].serving_id == model), None)
    if item is None:
        named = [
            i
            for i in items
            if model_name(i["ref"].serving_id).lower() == str(model).lower()
        ]
        if len(named) > 1:
            raise ValueError(
                f"{model} matches "
                f"{', '.join(i['ref'].serving_id for i in named)} · use the full name"
            )
        item = named[0] if named else None
    if not item:
        raise ValueError(f"Unknown model: {model} · songe models lists them")
    registry.register(config, store, item["ref"].serving_id, item["ref"])
