"""Aider: Songe's models as named models, with their own endpoint.

Aider reaches OpenAI-compatible servers through LiteLLM. OPENAI_API_BASE would
send the user's real OpenAI models to Songe too, so Songe never sets it.
Instead each Songe model gets its own entry: `extra_params` in
~/.aider.model.settings.yml gives it Songe's endpoint, and
~/.aider.model.metadata.json its context, as Aider documents for local
models. The user picks one with `aider --model openai/songe/<name>`.
"""

import os
from pathlib import Path

from .. import NAME
from ..models.names import exposed
from .base import JSON, Agent, YAMLListCodec

SETTINGS = ".aider.model.settings.yml"
METADATA = ".aider.model.metadata.json"
PREFIX = f"openai/{NAME}/"
SETTINGS_CODEC = YAMLListCodec("name")


def model_name(public):
    """Aider's name for a Songe model: LiteLLM's openai/ prefix, which it
    strips before the request, so Songe sees its own songe/ name."""
    return "openai/" + public


class Aider(Agent):
    name = "aider"
    pick = f"--model {PREFIX}…"
    file_names = (SETTINGS, METADATA)

    def files(self, home):
        # Found as opencode and pi are, by a folder it keeps in the home
        # directory (~/.aider, made on its first run), so tests stay isolated.
        settings, metadata = home / SETTINGS, home / METADATA
        installed = (home / ".aider").is_dir() or settings.exists() or metadata.exists()
        return [(settings, SETTINGS_CODEC), (metadata, JSON)] if installed else []

    def codec(self, path):
        return SETTINGS_CODEC if Path(path).name == SETTINGS else JSON

    def configure(self, path, data, config, *, default=None, v2=False):
        from .opencode import context_for

        data = dict(data)
        for public in exposed(config):
            name = model_name(public)
            context = context_for(config, public)
            if Path(path).name == SETTINGS:
                entry = dict(data.get(name) or {})
                entry["name"] = name
                params = dict(entry.get("extra_params") or {})
                params.update(api_base=f"http://127.0.0.1:{config.port}/v1", api_key="local")
                entry["extra_params"] = params
                data[name] = entry
            else:
                data[name] = {
                    **(data.get(name) or {}),
                    "max_input_tokens": context,
                    "max_output_tokens": min(4096, context // 2),
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                }
        return data

    def stale(self, path, key, config):
        # Only metadata is dropped: a settings entry is what sends the model
        # to Songe, and without it a default still naming the model (AIDER_MODEL,
        # ~/.aider.conf.yml) would go to OpenAI instead of failing locally.
        return (
            Path(path).name == METADATA
            and str(key).startswith(PREFIX)
            and key not in {model_name(public) for public in exposed(config)}
        )

    def after_uninstall(self, home):
        if self.on_songe(home):
            return (
                f"its default model is still {PREFIX}… · change AIDER_MODEL or model:"
                " in ~/.aider.conf.yml, or it goes to OpenAI"
            )
        return None

    def on_songe(self, home):
        # Aider's own default: AIDER_MODEL, else `model:` in ~/.aider.conf.yml
        # (a --model on the command line is not something init can see).
        import yaml

        model = os.environ.get("AIDER_MODEL")
        if model is None:
            try:
                conf = yaml.safe_load((home / ".aider.conf.yml").read_text(encoding="utf-8"))
            except (OSError, ValueError, yaml.YAMLError):
                conf = None
            model = conf.get("model") if isinstance(conf, dict) else None
        return isinstance(model, str) and model.startswith(PREFIX)

    def owns(self, path, data):
        return any(str(key).startswith(PREFIX) for key in data)

    def status(self, path, data, config):
        if Path(path).name != SETTINGS:
            return None
        bases = {
            (entry.get("extra_params") or {}).get("api_base")
            for key, entry in data.items()
            if str(key).startswith(PREFIX)
        }
        if not bases:
            return "missing"
        return "configured" if bases == {f"http://127.0.0.1:{config.port}/v1"} else "differs"
