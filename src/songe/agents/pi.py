import copy
import json
import os
from pathlib import Path

from .. import NAME
from ..models.names import exposed
from .base import JSON, Agent, NotEditable


def configure(data, config):
    data = copy.deepcopy(data)
    providers = data.setdefault("providers", {})
    existing = providers.get(NAME, {})
    from .opencode import context_for

    public_models = exposed(config)
    models = list(existing.get("models", []))
    present = {m["id"] for m in models}
    models.extend(
        {
            "id": model,
            "name": model,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
        for model in public_models
        if model not in present
    )
    for model in models:
        if model["id"] in public_models:
            context = context_for(config, model["id"])
            model["contextWindow"] = context
            model["maxTokens"] = min(4096, context // 2)
            if model.get("compat") is None:
                model["compat"] = {}
            model["compat"].setdefault("sendSessionAffinityHeaders", True)

    providers[NAME] = {
        **existing,
        "baseUrl": f"http://127.0.0.1:{config.port}/v1",
        "api": "openai-completions",
        "apiKey": "local",
        "models": models,
    }
    return data


class Pi(Agent):
    name = "pi"
    suggest_when_missing = True
    pick = "/model"
    legacy_models = True
    file_names = ("models.json",)

    @staticmethod
    def folder(home):
        return Path(os.environ.get("PI_CODING_AGENT_DIR", str(home / ".pi/agent")))

    def files(self, home):
        folder = self.folder(home)
        return [(folder / "models.json", JSON)] if folder.exists() else []

    def configure(self, path, data, config, *, default=None, v2=False):
        return configure(data, config)

    def owns(self, path, data):
        return NAME in (data.get("providers") or data.get("provider") or {})

    def status(self, path, data, config):
        provider = (data.get("providers", data.get("provider", {})) or {}).get(NAME, {})
        if provider.get("baseUrl") == f"http://127.0.0.1:{config.port}/v1":
            return "configured"
        return "missing" if not provider else "differs"

    def on_songe(self, home):
        # Pi's default lives in settings.json, and stays the user's choice.
        try:
            data = JSON.load((self.folder(home) / "settings.json").read_bytes())
        except (OSError, NotEditable):
            return False
        return data.get("defaultProvider") == NAME

    def binding(self, home):
        folder = self.folder(home)
        try:
            value = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
            model = value.get("defaultModel") or value.get("model")
            models = (
                json.loads((folder / "models.json").read_text(encoding="utf-8"))
                if (folder / "models.json").is_file()
                else {}
            )
            provider = models.get("providers", {}).get(value.get("defaultProvider"), {})
            if isinstance(model, str):
                return model, provider.get("baseUrl", "").rstrip("/")
        except (OSError, ValueError, AttributeError):
            pass
        return "", ""
