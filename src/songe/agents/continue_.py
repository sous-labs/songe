"""Continue: Songe's models as entries of the `models` list in its
config.yaml (~/.continue, or CONTINUE_GLOBAL_DIR).

Each Songe model is an `openai` provider entry with Songe's endpoint as its
apiBase. The user's own models and their order are kept, so Continue's
default (the model picked in it, else the first one listed) stays theirs.
Continue reads config.json only while there is no config.yaml, so a user
still on config.json is told to move first: a new config.yaml would replace
their whole configuration.
"""

import json
import os
from pathlib import Path

from .. import NAME
from ..models.names import exposed
from .base import Agent, YAMLSectionCodec

CONFIG = "config.yaml"
CODEC = YAMLSectionCodec("models", "name")
# What Continue writes on its first start (core/config/default.ts).
DOCUMENT = {"name": "Main Config", "version": "1.0.0", "schema": "v1"}


def folder(home):
    configured = os.environ.get("CONTINUE_GLOBAL_DIR")
    if configured:
        return Path(configured).absolute()
    return home / ".continue"


# What a model entry needs for Continue to load its config at all.
REQUIRED = ("provider", "model", "apiBase")


def max_tokens(context):
    """The answer's budget. Continue's extensions use 4096 for a model they
    do not know and reserve less than that for the answer, so a small
    window overflows; its CLI refuses half the context or more."""
    return min(4096, context * 3 // 10)


def songe_entries(data):
    models = data.get("models")
    if not isinstance(models, dict):
        return {}
    return {k: v for k, v in models.items() if str(k).startswith(f"{NAME}/")}


class Continue(Agent):
    name = "continue"
    pick = "the model selector (songe/…)"
    file_names = (CONFIG,)

    def files(self, home):
        # Found by the folder it makes on its first start, as the others are.
        here = folder(home)
        return [(here / CONFIG, CODEC)] if here.is_dir() else []

    def handles(self, path):
        path = Path(path)
        return path.name == CONFIG and (
            path.parent.name == ".continue"
            or path.parent == folder(Path.home())
        )

    def codec(self, path):
        return CODEC

    def skip_reason(self, home):
        here = folder(home)
        if (here / "config.json").exists() and not (here / CONFIG).exists():
            return (
                "Continue still reads config.json, and a new config.yaml would"
                " replace it · move to config.yaml first"
            )
        return None

    def configure(self, path, data, config, *, default=None, v2=False):
        from .opencode import context_for

        data = {**DOCUMENT, **data}
        models = dict(data.get("models") or {})
        for public in exposed(config):
            context = context_for(config, public)
            new = public not in models
            entry = dict(models.get(public) or {})
            entry.update(
                name=public,
                provider="openai",
                model=public,
                apiBase=f"http://127.0.0.1:{config.port}/v1",
                apiKey="local",
            )
            if new:
                # Continue only detects tool calling for model names it
                # knows, and Agent mode needs it. Set once: a user who took
                # it out keeps it out.
                entry["capabilities"] = ["tool_use"]
            entry["defaultCompletionOptions"] = {
                **(entry.get("defaultCompletionOptions") or {}),
                "contextLength": context,
                "maxTokens": max_tokens(context),
            }
            models[public] = entry
        data["models"] = models
        return data

    def snippet(self, path, config, *, default=None, v2=False):
        # Only the entries to add to the user's own models list.
        return {"models": self.configure(path, {}, config)["models"]}

    def stale(self, path, key, config):
        return str(key).startswith(f"{NAME}/") and key not in exposed(config)

    def restored(self, path, current, original):
        models = current.get("models")
        before = original.get("models") or {}
        if isinstance(models, dict):
            for name in list(models):
                entry = models[name]
                # Songe's entry, with only the user's fields left (roles,
                # a changed contextLength): Continue rejects an entry
                # without these, and with it the whole file.
                if (
                    name not in before
                    and str(name).startswith(f"{NAME}/")
                    and not all(isinstance(entry, dict) and key in entry for key in REQUIRED)
                ):
                    del models[name]
            if not models and "models" not in original:
                del current["models"]
        if current != original:
            # The file stays: without these Continue cannot read it.
            missing = {k: v for k, v in DOCUMENT.items() if k not in current and k not in original}
            current = {**missing, **current}
        return current

    def owns(self, path, data):
        return bool(songe_entries(data))

    def status(self, path, data, config):
        bases = {entry.get("apiBase") for entry in songe_entries(data).values()}
        if not bases:
            return "missing"
        return "configured" if bases == {f"http://127.0.0.1:{config.port}/v1"} else "differs"

    def on_songe(self, home):
        # The chat model picked in Continue, kept per profile in its global
        # context; with none picked, the first model in config.yaml.
        here = folder(home)
        try:
            context = json.loads((here / "index/globalContext.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            context = {}
        profiles = context.get("selectedModelsByProfileId") if isinstance(context, dict) else None
        picked = [
            chosen.get("chat")
            for chosen in (profiles.values() if isinstance(profiles, dict) else ())
            if isinstance(chosen, dict) and chosen.get("chat")
        ]
        if picked:
            return any(str(name).startswith(f"{NAME}/") for name in picked)
        try:
            models = list(CODEC.load((here / CONFIG).read_bytes()).get("models") or {})
        except (OSError, ValueError):
            return False
        return bool(models) and str(models[0]).startswith(f"{NAME}/")
