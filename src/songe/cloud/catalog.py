"""Providers and the short list of cloud models Songe suggests.

Update CURATED per release from OpenRouter's public model list
(GET https://openrouter.ai/api/v1/models, no key needed): strong frontier
models only, one marked recommended. Any other model id can be typed in.
Last checked 2026-11-25.
"""

OPENROUTER_URL = "https://openrouter.ai/api/v1"
PROVIDERS = ("openrouter", "custom")

# (model id, label). The first one is the recommended default.
CURATED = (
    ("anthropic/claude-sonnet-5.5", "Claude Sonnet 5.5 · recommended"),
    ("anthropic/claude-opus-5.5", "Claude Opus 5.5 · strongest, costs more"),
    ("openai/gpt-6.2-sol", "GPT-6.2 Sol"),
    ("google/gemini-4-carbon", "Gemini 4 Carbon"),
    ("deepseek/deepseek-v4-pro-0813", "DeepSeek V4 Pro · low cost"),
    ("moonshotai/kimi-k3.1-pro", "Kimi K3.1 Pro · low cost"),
)
RECOMMENDED = CURATED[0][0]

# Where each provider's terms live; the user decides whether they fit.
TERMS = {
    "openrouter": "https://openrouter.ai/terms",
}


def base_url(provider: str, custom: str = "") -> str:
    return OPENROUTER_URL if provider == "openrouter" else custom.rstrip("/")


def provider_label(provider: str, url: str = "") -> str:
    if provider == "openrouter":
        return "OpenRouter"
    from urllib.parse import urlparse

    return urlparse(url).hostname or "custom endpoint"
