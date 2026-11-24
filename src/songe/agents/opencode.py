import copy
import json
import os
from pathlib import Path

from .. import NAME
from ..models.names import exposed
from .base import JSON, Agent, NotEditable


def context_for(config, model):
    """The context the model server really serves this model with."""
    return int(
        (getattr(config, "served_contexts", None) or {}).get(model)
        or config.context_window
    )


def configure(data, config, default=None, v2=False):
    """Add Songe's models. default: True always, False never, None automatic."""
    data = copy.deepcopy(data)
    # The installed binary's version determines the schema. Existing v2 config
    # always takes precedence, even if the binary is currently unavailable.
    v2 = v2 or "providers" in data
    provider = {
        "name": "Songe",
        "models": {
            model: {
                "name": model,
                "limit": {
                    "context": context_for(config, model),
                    "output": min(4096, context_for(config, model) // 2),
                },
            }
            for model in exposed(config)
        },
    }
    if v2:
        provider.update(
            package="@opencode/ai/providers/openai-compatible",
            settings={
                "baseURL": f"http://127.0.0.1:{config.port}/v1",
                "apiKey": "local",
            },
        )
        providers = data.setdefault("providers", {})
    else:
        provider.update(
            npm="@ai-sdk/openai-compatible",
            options={
                "baseURL": f"http://127.0.0.1:{config.port}/v1",
                "apiKey": "local",
            },
        )
        providers = data.setdefault("provider", {})
    existing = providers.get(NAME, {})
    # Songe owns its own entries: current limits win, other fields survive.
    previous = existing.get("models", {})
    provider["models"] = {
        **previous,
        **{
            name: {**previous.get(name, {}), **entry}
            for name, entry in provider["models"].items()
        },
    }
    providers[NAME] = {**existing, **provider}
    models = exposed(config)
    chosen = f"{NAME}/" + (
        config.public_model if config.public_model in models else models[0]
    )
    if default is None:
        # Automatic: a local default moves to the Songe version of the same
        # model, so it learns from what the user already runs. Anything else
        # (a cloud model, no default, a model Songe does not offer) is left
        # as it is: init no longer picks one model for the user (live: it
        # made a 27B 1-bit model this Mac cannot train the default).
        same = same_model(data.get("model"), models) if local_model(data, data.get("model") or "") else None
        if same:
            data["model"] = f"{NAME}/{same}"
    elif default:
        data["model"] = chosen
    return data


def same_model(reference, models):
    """The Songe model for the agent's local default ("lmstudio/granite-4.2-3b"
    → "songe/granite-4.2-3b"), or None."""
    import re

    def key(name):
        return re.sub(r"[^a-z0-9]", "", str(name).rsplit("/", 1)[-1].lower().removesuffix(".gguf"))

    wanted = key(str(reference or "").partition("/")[2])
    if not wanted:
        return None
    exact = [m for m in models if key(m) == wanted]
    if exact:
        return exact[0]
    # Servers name the same model slightly differently ("qwen3.5-4b-mlx" in
    # LM Studio, "Qwen3.5-4B-MLX-4bit" as a folder): one is the other's start.
    near = [m for m in models if key(m).startswith(wanted) or wanted.startswith(key(m))]
    return min(near, key=lambda m: abs(len(key(m)) - len(wanted))) if near else None


LOCAL_PROVIDERS = {"lmstudio", "ollama", "llamacpp", "llama.cpp", "mlx", "vllm"}


def local_model(data, reference):
    """A model served on this machine by something other than Songe."""
    provider = str(reference).partition("/")[0]
    if provider == NAME:
        return False
    if provider in LOCAL_PROVIDERS:
        return True
    settings = (data.get("providers") or data.get("provider") or {}).get(provider) or {}
    url = str(
        (settings.get("settings") or settings.get("options") or {}).get("baseURL") or ""
    )
    return any(host in url for host in ("127.0.0.1", "localhost", "[::1]"))


class OpenCode(Agent):
    name = "opencode"
    suggest_when_missing = True
    pick = "/models"
    legacy_models = True
    file_names = ("opencode.json", "opencode.jsonc")

    @staticmethod
    def folder(home):
        return Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))) / "opencode"

    def files(self, home):
        folder = self.folder(home)
        if not folder.exists():
            return []
        jsonc, plain = folder / "opencode.jsonc", folder / "opencode.json"
        # A JSONC file is read (and then skipped with a note), never shadowed.
        return [(jsonc if jsonc.exists() and not plain.exists() else plain, JSON)]

    def configure(self, path, data, config, *, default=None, v2=False):
        return configure(data, config, default=default, v2=v2)

    def snippet(self, path, config, *, default=None, v2=False):
        key = "providers" if v2 else "provider"
        entry = configure({}, config, default=default, v2=v2)
        return {key: entry[key]} if key in entry else entry

    def owns(self, path, data):
        return NAME in (data.get("providers") or data.get("provider") or {})

    def status(self, path, data, config):
        provider = (data.get("providers", data.get("provider", {})) or {}).get(NAME, {})
        url = provider.get("settings", provider.get("options", {})).get("baseURL")
        if url == f"http://127.0.0.1:{config.port}/v1":
            return "configured"
        return "missing" if not provider else "differs"

    def on_songe(self, home):
        try:
            data = JSON.load(self.files(home)[0][0].read_bytes())
        except (IndexError, OSError, NotEditable):
            return False
        model = data.get("model")
        return isinstance(model, str) and model.startswith(f"{NAME}/")

    def binding(self, home):
        path = self.folder(home) / "opencode.json"
        state = (
            Path(os.environ.get("XDG_STATE_HOME", str(home / ".local/state")))
            / "opencode/model.json"
        )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "", ""
        model = value.get("model", "") if isinstance(value, dict) else ""
        # The model picked in opencode's UI lives in its state file and wins
        # over the config default, so it is the one the user actually runs.
        try:
            recent = json.loads(state.read_text(encoding="utf-8")).get("recent") or []
            if recent and recent[0].get("modelID"):
                model = f"{recent[0].get('providerID', '')}/{recent[0]['modelID']}"
        except (OSError, ValueError, AttributeError):
            pass
        try:
            provider, _, model_id = model.partition("/")
            definition = (value.get("providers") or value.get("provider") or {}).get(provider, {})
            options = definition.get("settings") or definition.get("options") or {}
            if model:
                return model_id or model, options.get("baseURL", "").rstrip("/")
        except AttributeError:
            pass  # A hand-edited file of another shape binds nothing, as on main.
        return "", ""
