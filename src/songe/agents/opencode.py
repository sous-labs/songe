import copy

from .. import NAME
from ..models.names import exposed


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
