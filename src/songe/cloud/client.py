"""OpenAI-compatible chat completions against the chosen cloud provider.

Only songe.cloud calls this, and only with messages already redacted.
Errors come back as CloudError with a plain reason; `auth` marks a refused
key (the caller turns cloud learning off rather than keep a broken key on).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

TIMEOUT = httpx.Timeout(120, connect=10)
# Tests replace this with an httpx.MockTransport; nothing here may reach a
# real provider from the test suite.
TRANSPORT = None


class CloudError(Exception):
    def __init__(self, reason, *, auth=False, status=None):
        super().__init__(reason)
        self.reason = reason
        self.auth = auth
        self.status = status


@dataclass
class Reply:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    cost_source: str  # "provider" | "pricing" | "unknown"
    model: str = ""
    raw_usage: dict = field(default_factory=dict)


def http():
    return httpx.Client(timeout=TIMEOUT, transport=TRANSPORT, follow_redirects=False)


def _headers(provider, key):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if provider == "openrouter":
        # Names the app in the user's OpenRouter activity; carries no user data.
        headers["X-Title"] = "Songe"
    return headers


def _reason(response, provider_name, model):
    try:
        body = response.json()
        detail = body.get("error", {})
        detail = detail.get("message") if isinstance(detail, dict) else str(detail)
    except (ValueError, AttributeError):
        detail = ""
    detail = " ".join(str(detail or "").split())[:160]
    code = response.status_code
    if code in (401, 403):
        return CloudError(f"{provider_name} refused the API key ({code})", auth=True, status=code)
    if code == 402:
        return CloudError(f"the {provider_name} account is out of credit", status=code)
    if code == 404:
        return CloudError(f"{provider_name} has no model {model}", status=code)
    if code == 429:
        return CloudError(f"{provider_name} is rate limiting requests", status=code)
    if code >= 500:
        return CloudError(f"{provider_name} had an error ({code})", status=code)
    return CloudError(
        f"{provider_name} answered {code}" + (f": {detail}" if detail else ""), status=code
    )


def _text(message):
    content = (message or {}).get("content")
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def chat(*, provider, provider_name, url, key, model, messages, max_tokens,
         temperature=0.0, pricing=None):
    """One completion. Messages must already be redacted."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if provider == "openrouter":
        body["usage"] = {"include": True}  # OpenRouter reports the call's cost
    try:
        with http() as client:
            response = client.post(
                url.rstrip("/") + "/chat/completions", json=body, headers=_headers(provider, key)
            )
    except httpx.TimeoutException as exc:
        raise CloudError(f"{provider_name} did not answer in time") from exc
    except httpx.HTTPError as exc:
        raise CloudError(f"can't reach {provider_name} (offline?)") from exc
    if not response.is_success:
        raise _reason(response, provider_name, model)
    try:
        data = response.json()
        choice = data["choices"][0]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise CloudError(f"{provider_name} sent a reply Songe couldn't read") from exc
    if isinstance(data.get("error"), dict):
        raise CloudError(f"{provider_name}: {data['error'].get('message', 'error')}"[:200])
    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    cost, source = _number(usage.get("cost")), "provider"
    if cost is None and pricing and prompt is not None and completion is not None:
        cost = prompt * pricing.get("prompt", 0.0) + completion * pricing.get("completion", 0.0)
        source = "pricing"
    if cost is None:
        source = "unknown"
    return Reply(
        text=_text(choice.get("message")),
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=cost,
        cost_source=source,
        model=data.get("model") or model,
        raw_usage=usage,
    )


def model_info(*, provider, url, key, model):
    """{found, context_window, pricing} from the provider's model list, or
    None when the list could not be read (not every endpoint has one).
    OpenRouter's list is public: no key is sent for it."""
    try:
        with http() as client:
            response = client.get(
                url.rstrip("/") + "/models",
                headers=_headers(provider, None if provider == "openrouter" else key),
            )
        if not response.is_success:
            return None
        listed = response.json().get("data")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    if not isinstance(listed, list):
        return None
    entry = next((m for m in listed if isinstance(m, dict) and m.get("id") == model), None)
    if entry is None:
        return {"found": False, "context_window": 0, "pricing": {}}
    context = next(
        (
            int(v)
            for v in (
                entry.get("context_length"),
                entry.get("context_window"),
                entry.get("max_model_len"),
                (entry.get("top_provider") or {}).get("context_length"),
            )
            if _number(v)
        ),
        0,
    )
    price = entry.get("pricing") or {}
    pricing = {}
    if _number(price.get("prompt")) is not None and _number(price.get("completion")) is not None:
        pricing = {"prompt": _number(price["prompt"]), "completion": _number(price["completion"])}
    return {"found": True, "context_window": context, "pricing": pricing}
