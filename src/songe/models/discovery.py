"""Read existing libraries and server metadata without starting any services."""

import json
import logging
import os
import re
import shlex
import sys
from pathlib import Path

import httpx

from . import ModelRef
from .gguf import read_gguf

logger = logging.getLogger(__name__)


def readable(path, fn):
    """One corrupt/inaccessible library entry cannot hide the other models."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("Skipping model %s: %s", path, type(exc).__name__)
        return None


def folder_ref(path, upstream, serving_id=None):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"No safetensors in {path}")
    # MLX writes this metadata even for unquantized models. Quantized MLX
    # config uses `quantization`; transformers uses `quantization_config`.
    with weights[0].open("rb") as f:
        length = int.from_bytes(f.read(8), "little")
        if length > weights[0].stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header in {weights[0]}")
        header = json.loads(f.read(length))
    mlx = (
        "quantization" in config
        or header.get("__metadata__", {}).get("format") == "mlx"
    )
    return ModelRef(
        upstream,
        serving_id or path.name,
        "mlx" if mlx else "safetensors",
        path,
        config.get("model_type", ""),
    )


def lmstudio_downloads(home=None, os_name=None):
    home, os_name = Path(home or Path.home()), os_name or sys.platform
    # LM Studio currently keeps .lmstudio on every OS. Also accept the native
    # settings locations used by earlier builds, without creating any paths.
    settings = [home / ".lmstudio/settings.json"]
    if os_name == "win32":
        settings.append(
            Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
            / "LM Studio/settings.json"
        )
    elif os_name == "darwin":
        settings.append(home / "Library/Application Support/LM Studio/settings.json")
    else:
        settings.append(
            Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
            / "LM Studio/settings.json"
        )
    for path in settings:
        if path.is_file():
            settings_data = readable(
                path, lambda path=path: json.loads(path.read_text(encoding="utf-8"))
            )
            folder = (
                settings_data.get("downloadsFolder")
                if isinstance(settings_data, dict)
                else None
            )
            if folder:
                return Path(folder).expanduser()
    return home / ".lmstudio/models"


def _library_folders(root):
    """Every folder under the library, following linked folders (`lms import
    --symbolic-link`, a model kept on another disk) once each. Path.rglob
    skips them, so LM Studio listed models Songe never offered (#72)."""
    seen = set()
    for folder, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(folder)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()  # the same model under two links keeps one name
        yield Path(folder), files


def lmstudio(home=None, folder=None, os_name=None):
    root = Path(folder) if folder else lmstudio_downloads(home, os_name)
    # rglob silently yields nothing for a folder the OS denies (macOS privacy
    # for external volumes), which would look like an empty library.
    if root.is_dir():
        try:
            next(os.scandir(root), None)
        except PermissionError as exc:
            raise PermissionError(
                f"{root} exists but this process may not read it. On macOS, allow "
                "your terminal under System Settings > Privacy & Security > "
                "Files and Folders (Removable Volumes) or Full Disk Access."
            ) from exc
    configs, ggufs = [], []
    for folder, files in _library_folders(root):
        configs += [folder / name for name in files if name == "config.json"]
        ggufs += [folder / name for name in files if name.endswith(".gguf")]
    models = []
    for config in sorted(configs):
        if list(config.parent.glob("*.safetensors")):
            ref = readable(
                config,
                lambda config=config: folder_ref(
                    config.parent,
                    "lmstudio",
                    config.parent.relative_to(root).as_posix(),
                ),
            )
            if ref:
                models.append(ref)
    for path in sorted(ggufs):
        info = readable(path, lambda path=path: read_gguf(path))
        if info:
            models.append(
                ModelRef(
                    "lmstudio",
                    path.relative_to(root).as_posix(),
                    "gguf",
                    path,
                    info.architecture,
                )
            )
    return models


def ollama(home=None, folder=None, *, url=None, transport=None):
    root = Path(
        folder
        or os.environ.get("OLLAMA_MODELS", Path(home or Path.home()) / ".ollama/models")
    )
    if (
        folder is None
        and home is None
        and "OLLAMA_MODELS" not in os.environ
        and not root.exists()
        and sys.platform.startswith("linux")
    ):
        # The system service runs as ollama, whose library is separate from
        # the interactive user's home. Read its account home when available.
        import pwd

        try:
            root = Path(pwd.getpwnam("ollama").pw_dir) / ".ollama/models"
        except KeyError:
            pass
    models = []
    for manifest in sorted((root / "manifests").glob("*/*/*/*")):
        if manifest.is_file():
            models.extend(
                readable(
                    manifest, lambda manifest=manifest: _ollama_manifest(root, manifest)
                )
                or []
            )
    if url:
        # Newer Ollama versions keep API-created tags outside the legacy
        # manifests tree. Ask the running daemon, but retain offline discovery
        # when it is unavailable. Only return locally readable GGUF weights.
        live = readable(url, lambda: _ollama_api(root, url, models, transport)) or []
        merged = {model.serving_id: model for model in models}
        merged.update({model.serving_id: model for model in live})
        return list(merged.values())
    return models


def _ollama_api(root, url, existing, transport):
    base = url.rstrip("/").removesuffix("/v1")
    known = {model.path.resolve(): model.architecture for model in existing}
    models = []
    with httpx.Client(transport=transport, timeout=5, trust_env=False) as client:
        response = client.get(base + "/api/tags")
        response.raise_for_status()
        for tag in response.json().get("models", []):
            name = tag.get("name") or tag.get("model")
            if not name:
                continue

            def entry(name=name):
                response = client.post(base + "/api/show", json={"model": name})
                response.raise_for_status()
                match = re.search(
                    r'^FROM\s+"?([^\n"]*sha256[:-]([a-f0-9]{64}))"?\s*$',
                    response.json().get("modelfile", ""),
                    re.M,
                )
                if not match:
                    return None
                path = root / "blobs" / ("sha256-" + match[2])
                if not path.is_file():
                    path = Path(match[1])
                    if not path.is_absolute() or not path.is_file():
                        return None
                architecture = known.get(path.resolve())
                if architecture is None:
                    architecture = read_gguf(path).architecture
                    known[path.resolve()] = architecture
                return ModelRef("ollama", name, "gguf", path, architecture)

            model = readable(name, entry)
            if model:
                models.append(model)
    return models


def _ollama_manifest(root, manifest):
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    registry, namespace, name, tag = manifest.relative_to(root / "manifests").parts
    models = []
    for layer in metadata.get("layers", []):
        if layer.get("mediaType") != "application/vnd.ollama.image.model":
            continue
        algorithm, _, value = layer.get("digest", "").partition(":")
        if (
            algorithm != "sha256"
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Invalid Ollama model digest")
        path = root / "blobs" / f"sha256-{value}"
        if not path.is_file():
            continue
        info = read_gguf(path)
        prefix = "" if registry == "registry.ollama.ai" else registry + "/"
        prefix += "" if namespace == "library" else namespace + "/"
        models.append(
            ModelRef("ollama", f"{prefix}{name}:{tag}", "gguf", path, info.architecture)
        )
    return models


def _get(url, transport=None, timeout=5.0):
    # Discovery is read-only and never honors proxy environment variables.
    with httpx.Client(transport=transport, timeout=timeout, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def llamaswap_running(root, transport=None):
    """llama-swap's loaded models ({"model", "cmd", ...}), or None when the
    server is not llama-swap. Reading it never loads a model."""
    try:
        data = _get(root + "/running", transport)
    except (httpx.HTTPError, ValueError):
        return None
    running = data.get("running") if isinstance(data, dict) else None
    return running if isinstance(running, list) else None


def command_flag(command, *names, windows=None):
    """A flag's value in a llama-server command line, as /running shows it."""
    windows = os.name == "nt" if windows is None else windows
    try:
        # POSIX splitting reads a Windows path's backslashes as escapes.
        words = shlex.split(command or "", posix=not windows)
    except ValueError:
        return None
    if windows:
        words = [w[1:-1] if len(w) > 1 and w[0] == w[-1] and w[0] in "\"'" else w for w in words]
    for index, word in enumerate(words):
        for name in names:
            if word == name and index + 1 < len(words):
                return words[index + 1]
            if word.startswith(name + "="):
                return word.split("=", 1)[1]
    return None


def llamaswap_models(url, transport=None):
    """Every model llama-swap serves, loaded or not; [] for other servers."""
    root = url.rstrip("/").removesuffix("/v1")
    if llamaswap_running(root, transport) is None:
        return []
    data = _get(root + "/v1/models", transport).get("data", [])
    return [m["id"] for m in data if isinstance(m.get("id"), str)]


def llamaswap_waiting(url, transport=None):
    """llama-swap's models that are not loaded now; empty for other servers."""
    root = url.rstrip("/").removesuffix("/v1")
    running = llamaswap_running(root, transport)
    if running is None:
        return set()
    try:
        listed = llamaswap_models(url, transport)
    except (httpx.HTTPError, ValueError):
        return set()
    return set(listed) - {e.get("model") for e in running if isinstance(e, dict)}


def llamacpp(url, transport=None):
    root = url.rstrip("/").removesuffix("/v1")
    try:
        props = _get(root + "/props", transport)
    except httpx.HTTPStatusError:
        # llama-swap has no /props of its own: its loaded models' commands
        # name their weights. A model that is not loaded is found once it is;
        # asking it for /props would load it and swap out the user's model.
        running = llamaswap_running(root, transport)
        if running is None:
            raise
        refs = []
        for entry in running:
            if not isinstance(entry, dict):
                continue
            path = command_flag(entry.get("cmd"), "-m", "--model")
            # A relative path is relative to llama-swap's directory, not ours.
            if not path or not entry.get("model") or not Path(path).is_absolute():
                continue
            if not Path(path).is_file():
                continue
            info = readable(Path(path), lambda path=Path(path): read_gguf(path))
            if info:
                refs.append(
                    ModelRef("llamacpp", entry["model"], "gguf", Path(path), info.architecture)
                )
        return refs
    path = Path(props["model_path"])
    models = _get(root + "/v1/models", transport).get("data", [])
    identifier = models[0]["id"] if models else str(path)
    info = readable(path, lambda path=path: read_gguf(path)) if path.is_file() else None
    if path.is_file() and info is None:
        return []
    architecture = info.architecture if info else props.get("model_type", "")
    return [ModelRef("llamacpp", identifier, "gguf", path, architecture)]


def _cached_path(value):
    path = Path(value)
    if path.exists():
        return path
    # A served Hub id is resolved only from the existing cache; discovery must
    # never download weights or import transformers.
    hub = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
        )
    )
    repo = hub / ("models--" + value.replace("/", "--"))
    ref = repo / "refs/main"
    if ref.is_file():
        cached = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if cached.is_dir():
            return cached
    return path


def openai_models(url, upstream, transport=None):
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    models = []
    for item in _get(base + "/models", transport).get("data", []):
        ref = readable(
            item.get("id", "unknown") if isinstance(item, dict) else "unknown",
            lambda item=item: _openai_ref(item, upstream),
        )
        if ref:
            models.append(ref)
    return models


def _openai_ref(item, upstream):
    if item.get("parent"):
        return None
    path = _cached_path(item.get("root") or item["id"])
    if (path / "config.json").is_file():
        return folder_ref(path, upstream, item["id"])
    return ModelRef(
        upstream,
        item["id"],
        "mlx" if upstream == "mlx" else "safetensors",
        path,
        item.get("model_type", ""),
    )


def vllm(url, transport=None):
    return openai_models(url, "vllm", transport)


def mlx(url, transport=None):
    return openai_models(url, "mlx", transport)


def identify(url):
    """Identify a custom local endpoint through read-only protocol metadata;
    None when it is none of the servers Songe knows."""
    root = url.rstrip("/").removesuffix("/v1")

    def answers(path, key):
        try:
            return key in _get(root + path, timeout=1)
        except (httpx.HTTPError, ValueError, TypeError):
            return False

    # Other servers answer /api/version too (llama-swap: {"version": "v275",
    # ...}); only Ollama also lists its models at /api/tags.
    if answers("/api/version", "version") and answers("/api/tags", "models"):
        return "ollama"
    for name, path, key in (
        ("llamacpp", "/props", "model_path"),
        # llama-swap: llama-server processes behind one proxy.
        ("llamacpp", "/running", "running"),
        ("vllm", "/version", "version"),
        ("lmstudio", "/api/v0/models", "data"),
    ):
        if answers(path, key):
            return name
    return None


def lmstudio_key_path(executable, key, folder):
    """The library path that LM Studio's own model key names, or None. Agents
    address a model by this key, and two downloads of one model (MLX and GGUF)
    differ only here: guessing from the name can pick the other file."""
    import subprocess

    if not executable or not key:
        return None
    try:
        result = subprocess.run(
            [executable, "ls", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        for model in json.loads(result.stdout or "[]"):
            if model.get("modelKey") == key and model.get("path"):
                path = Path(model["path"])
                return path if path.is_absolute() else Path(folder) / path
    except (OSError, ValueError, TypeError, AttributeError, subprocess.SubprocessError):
        pass
    return None


def loaded_lmstudio(refs, executable, folder):
    """Bind loaded aliases to their library paths without loading any weights."""
    import subprocess

    if not executable:
        return refs
    try:
        result = subprocess.run(
            [executable, "ps", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        models = json.loads(result.stdout)
        models = models.get("models", []) if isinstance(models, dict) else models
        loaded = []
        for model in models:
            path = Path(model.get("path", ""))
            if not path.is_absolute():
                path = Path(folder) / path
            ref = next((r for r in refs if r.path.resolve() == path.resolve()), None)
            if ref and model.get("identifier"):
                loaded.append(
                    ModelRef(
                        ref.upstream,
                        model["identifier"],
                        ref.format,
                        ref.path,
                        ref.architecture,
                    )
                )
        return loaded + refs
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return refs
