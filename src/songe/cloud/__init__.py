"""Learn with a cloud model: optional, off by default.

When the user turns it on, the nightly Dream (and only the Dream) may ask a
cloud model they chose. The agent's own traffic never comes here: the proxy
still only talks to the local model server (config.py enforces that).

    available(store)                       -> Availability(ok, reason)
    complete(store, messages, purpose=...) -> Reply, or raises Unavailable
    probe(store, settings, key)            -> ProbeResult (one real test)
    update(store, changes)                 -> the settings, after any probe

Rules kept here, for every caller:
- It is on only after the disclaimer was accepted and a probe passed; a key
  that goes missing or is refused turns it off, with the reason kept.
- Everything sent is redacted first (redact.py), always.
- Every call is logged (ledger.py); a call that would pass the night's
  budget is refused before sending.
- Callers get Unavailable(reason) and fall back to the local model; the
  reason belongs in the Dream's record (a Dream never silently vanishes).

Settings live in Songe's store (key "cloud"), never the key itself.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, fields, replace
from urllib.parse import urlparse

from . import catalog, keys, ledger
from .client import CloudError, Reply
from .redact import redact, redact_value

STATE_KEY = "cloud"
PROBE_TOKENS = 1024  # room for a reasoning model to think before its verdict
CHARS_PER_TOKEN = 3  # rough and on the high side, for estimates before sending
MAX_BUDGET = 1000.0

__all__ = [
    "Availability",
    "CloudSettings",
    "ProbeResult",
    "Reply",
    "Unavailable",
    "available",
    "complete",
    "load",
    "probe",
    "summary",
    "update",
]


class Unavailable(Exception):
    """The cloud model can't be used now; `reason` says why in plain words."""

    def __init__(self, reason):
        super().__init__(f"Cloud model unavailable: {reason}")
        self.reason = reason


@dataclass
class CloudSettings:
    enabled: bool = False
    provider: str = "openrouter"
    base_url: str = ""  # custom provider only
    model: str = ""
    budget_usd: float = 1.0  # per night
    disclaimer_version: int = 0
    disclaimer_accepted_at: float = 0.0
    key_store: str = ""  # "keychain" | "file" | "environment" | ""
    key_tail: str = ""  # "…a3f9": all of the key Songe ever shows
    context_window: int = 0
    pricing: dict = field(default_factory=dict)  # USD per token, when known
    last_test: dict = field(default_factory=dict)  # {ok, message, at}
    off_reason: str = ""  # why Songe turned it off by itself
    enabled_at: float = 0.0

    @property
    def url(self):
        return catalog.base_url(self.provider, self.base_url)

    @property
    def provider_label(self):
        return catalog.provider_label(self.provider, self.base_url)

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled: true | false")
        if self.provider not in catalog.PROVIDERS:
            raise ValueError("provider: openrouter | custom")
        if self.provider == "custom" and self.base_url:
            url = urlparse(self.base_url)
            local = url.hostname in {"localhost", "127.0.0.1", "::1"}
            if url.scheme != "https" and not (url.scheme == "http" and local):
                raise ValueError("Custom provider: an https:// base URL is required")
            if not url.hostname:
                raise ValueError("Custom provider: base URL has no host")
        if not isinstance(self.model, str) or len(self.model) > 200:
            raise ValueError("model: a model id")
        budget = self.budget_usd
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(budget)
            or not 0 < budget <= MAX_BUDGET
        ):
            raise ValueError(f"Nightly budget: more than $0 and at most ${MAX_BUDGET:g}")


def load(store) -> CloudSettings:
    saved = store.get(STATE_KEY) or {}
    known = {f.name for f in fields(CloudSettings)}
    try:
        return CloudSettings(**{k: v for k, v in saved.items() if k in known})
    except TypeError:
        return CloudSettings()


def save(store, settings: CloudSettings) -> None:
    settings.validate()
    store.set(STATE_KEY, asdict(settings))


def accepted(settings: CloudSettings) -> bool:
    from ..wording import CLOUD_DISCLAIMER_VERSION

    return settings.disclaimer_version >= CLOUD_DISCLAIMER_VERSION


@dataclass
class Availability:
    ok: bool
    reason: str = ""

    def __bool__(self):
        return self.ok


def _turn_off(store, reason):
    current = load(store)
    if current.enabled:
        save(store, replace(current, enabled=False, off_reason=reason))


def available(store) -> Availability:
    """Whether a Dream may call the cloud model now, and if not, why."""
    settings = load(store)
    if not settings.enabled:
        return Availability(False, settings.off_reason or "cloud learning is off")
    if not accepted(settings):
        return Availability(False, "the description of what is sent changed · accept it again")
    if not settings.model:
        return Availability(False, "no cloud model chosen")
    key, _ = keys.load(store.root)
    if not key:
        reason = "the API key is missing · set it again with songe cloud on"
        _turn_off(store, reason)
        return Availability(False, reason)
    left = remaining(store, settings)
    if left is not None and left <= 0:
        return Availability(False, f"tonight's budget of {_money(settings.budget_usd)} is used up")
    return Availability(True)


def _money(usd):
    from ..wording import money

    return money(usd)


def remaining(store, settings=None):
    """USD left tonight, or None when spending can't be known (a custom
    endpoint without prices: the budget is then best effort)."""
    settings = settings or load(store)
    tonight = ledger.spent(store)
    if not tonight["cost_known"] and not settings.pricing:
        return None
    return settings.budget_usd - tonight["usd"]


def estimate(settings, sent_chars, max_tokens):
    """Most a call could cost, before sending, or None without prices."""
    if not settings.pricing:
        return None
    prompt = sent_chars / CHARS_PER_TOKEN
    return prompt * settings.pricing.get("prompt", 0.0) + max_tokens * settings.pricing.get(
        "completion", 0.0
    )


def _preview(messages):
    """The last message's text, redacted: what the Dream record shows."""
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str) and content.strip():
            return " ".join(content.split())
    return ""


def _send(store, settings, key, messages, *, purpose, max_tokens, temperature, dream,
          check_budget=True):
    safe = redact_value(messages)
    sent_chars = len(json.dumps(safe, ensure_ascii=False))
    row = {
        "purpose": purpose,
        "provider": settings.provider_label,
        "model": settings.model,
        "dream": dream,
        "sent_chars": sent_chars,
        "preview": _preview(safe),
    }
    if check_budget:
        left = remaining(store, settings)
        cost = estimate(settings, sent_chars, max_tokens) or 0.0
        if left is not None and cost > left:
            reason = (
                f"tonight's budget of {_money(settings.budget_usd)} would be passed"
                f" ({_money(max(left, 0.0))} left)"
            )
            ledger.record(store, **row, sent=0, ok=0, error=f"not sent: {reason}")
            raise Unavailable(reason)
    try:
        reply = _client().chat(
            provider=settings.provider,
            provider_name=settings.provider_label,
            url=settings.url,
            key=key,
            model=settings.model,
            messages=safe,
            max_tokens=max_tokens,
            temperature=temperature,
            pricing=settings.pricing,
        )
    except CloudError as exc:
        ledger.record(store, **row, sent=1, ok=0, error=exc.reason, cost_usd=0.0,
                      cost_source="provider")
        raise
    ledger.record(
        store,
        **row,
        sent=1,
        ok=1,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        cost_usd=reply.cost_usd,
        cost_source=reply.cost_source,
    )
    return reply


def _client():
    from . import client

    return client


def complete(store, messages, *, purpose, max_tokens=1024, temperature=0.0, dream=None) -> Reply:
    """Ask the cloud model. Redacted, budgeted and logged; raises
    Unavailable(reason) whenever the caller should use the local model."""
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("purpose: what the call is for (it is logged)")
    state = available(store)
    if not state:
        raise Unavailable(state.reason)
    settings = load(store)
    key, _ = keys.load(store.root)
    try:
        return _send(store, settings, key, messages, purpose=purpose, max_tokens=max_tokens,
                     temperature=temperature, dream=dream)
    except CloudError as exc:
        if exc.auth:
            _turn_off(store, f"{exc.reason} · cloud learning turned off · songe cloud on to fix")
        raise Unavailable(exc.reason) from exc


# ---- the probe ----

PROBE_REQUEST = "Rename the function fetch_data to load_data in utils.py."
PROBE_GUIDE = "Don't add comments to the code."
PROBE_FOLLOWS = "def load_data(path):\n    with open(path) as f:\n        return f.read()"
PROBE_IGNORES = (
    "# Load the data from the file at path\n"
    "def load_data(path):\n    # open the file and read everything\n"
    "    with open(path) as f:\n        return f.read()"
)


@dataclass
class ProbeResult:
    ok: bool
    message: str
    context_window: int = 0
    pricing: dict = field(default_factory=dict)
    cost_usd: float | None = None
    auth: bool = False


def probe(store, settings: CloudSettings, key: str) -> ProbeResult:
    """One real test of this provider, model and key: the key works, the
    model exists, and it answers a judge question (the gate's own prompt,
    sleep/judged.py) as a readable yes/no, right on two clear cases."""
    from ..sleep.judged import DOES, verdict
    from ..wording import context_size

    if not key:
        return ProbeResult(False, "no API key")
    if not settings.model:
        return ProbeResult(False, "no model chosen")
    if settings.provider == "custom" and not settings.base_url:
        return ProbeResult(False, "no base URL for the custom provider")
    info = _client().model_info(
        provider=settings.provider, url=settings.url, key=key, model=settings.model
    )
    if info is not None and not info["found"]:
        return ProbeResult(False, f"{settings.provider_label} has no model {settings.model}")
    if info:
        settings = replace(settings, pricing=info["pricing"] or settings.pricing)
    answers, cost = [], 0.0
    for answer in (PROBE_FOLLOWS, PROBE_IGNORES):
        prompt = DOES.format(request=PROBE_REQUEST, guide=PROBE_GUIDE, answer=answer)
        try:
            reply = _send(
                store, settings, key, [{"role": "user", "content": prompt}],
                purpose="test", max_tokens=PROBE_TOKENS, temperature=0.0, dream=None,
                check_budget=False,
            )
        except CloudError as exc:
            return ProbeResult(False, exc.reason, auth=exc.auth)
        cost = None if cost is None or reply.cost_usd is None else cost + reply.cost_usd
        answers.append(verdict(reply.text))
    context = (info or {}).get("context_window", 0)
    pricing = settings.pricing
    if None in answers:
        return ProbeResult(False, "replies couldn't be read as a verdict", context, pricing, cost)
    if answers != [True, False]:
        return ProbeResult(
            False, "answered a clear test case wrongly · it may not judge reliably",
            context, pricing, cost,
        )
    words = ["works", context_size(context)]
    if not pricing and cost is None:
        words.append("cost not reported · the budget is best effort")
    return ProbeResult(True, " · ".join(w for w in words if w), context, pricing, cost)


# ---- changing settings (CLI and dashboard) ----

EDITABLE = {"enabled", "provider", "base_url", "model", "budget_usd", "api_key", "accept_disclaimer"}
CONNECTION = ("provider", "base_url", "model")


def update(store, changes: dict) -> tuple[CloudSettings, ProbeResult | None]:
    """Apply settings changes. Turning on (or changing the connection or key
    while on) runs the probe first and saves nothing unless it passes:
    raises ValueError with the probe's plain result."""
    from ..wording import CLOUD_DISCLAIMER_VERSION

    if not isinstance(changes, dict) or not changes or set(changes) - EDITABLE:
        raise ValueError("Cloud settings: " + ", ".join(sorted(EDITABLE)))
    current = load(store)
    candidate = replace(current)
    if "accept_disclaimer" in changes:
        version = changes["accept_disclaimer"]
        if version is not True and version != CLOUD_DISCLAIMER_VERSION:
            raise ValueError("accept_disclaimer: the current disclaimer version")
        candidate.disclaimer_version = CLOUD_DISCLAIMER_VERSION
        candidate.disclaimer_accepted_at = time.time()
    for name in ("enabled", "provider", "base_url", "model", "budget_usd"):
        if name in changes:
            value = changes[name]
            if name in {"provider", "base_url", "model"}:
                if not isinstance(value, str):
                    raise ValueError(f"{name}: text")
                value = value.strip()
                if name == "base_url":
                    value = value.rstrip("/")
            setattr(candidate, name, value)
    if candidate.provider == "openrouter":
        candidate.base_url = ""
    candidate.validate()
    new_key = changes.get("api_key")
    if new_key is not None:
        if not isinstance(new_key, str) or not new_key.strip():
            raise ValueError("api_key: the key, as text")
        new_key = new_key.strip()
    connection_changed = new_key is not None or any(
        getattr(candidate, k) != getattr(current, k) for k in CONNECTION
    )
    result = None
    if candidate.enabled and (not current.enabled or connection_changed):
        if not accepted(candidate):
            raise ValueError("Accept the description of what is sent before turning this on")
        key = new_key or keys.load(store.root)[0]
        result = probe(store, candidate, key)
        candidate.last_test = {"ok": result.ok, "message": result.message, "at": time.time()}
        if not result.ok:
            # The test result is kept; the setting itself stays as it was.
            save(store, replace(current, last_test=candidate.last_test))
            raise ValueError(f"Not turned on · {result.message}")
        candidate.context_window = result.context_window
        candidate.pricing = result.pricing
        candidate.off_reason = ""
        if not current.enabled:
            candidate.enabled_at = time.time()
    elif connection_changed:
        # Off, or a new connection not yet tested: what was learned about the
        # old one no longer applies.
        candidate.context_window, candidate.pricing, candidate.last_test = 0, {}, {}
    if not candidate.enabled and changes.get("enabled") is False:
        candidate.off_reason = ""
    if new_key is not None:
        candidate.key_store = keys.save(store.root, new_key)
        candidate.key_tail = keys.masked(new_key)
    save(store, candidate)
    return candidate, result


def test(store) -> ProbeResult:
    """The Test button: probe the saved settings and keep the result."""
    settings = load(store)
    key, _ = keys.load(store.root)
    result = probe(store, settings, key)
    current = load(store)
    current.last_test = {"ok": result.ok, "message": result.message, "at": time.time()}
    if result.ok:
        current.context_window = result.context_window
        current.pricing = result.pricing or current.pricing
    elif result.auth and current.enabled:
        current.enabled = False
        current.off_reason = f"{result.message} · cloud learning turned off"
    save(store, current)
    return result


def forget_key(store) -> None:
    keys.delete(store.root)
    current = load(store)
    current.key_store, current.key_tail = "", ""
    if current.enabled:
        current.enabled, current.off_reason = False, "the API key was removed"
    save(store, current)


def summary(store) -> dict:
    """What the dashboard and `songe status` show. Never the key."""
    import os

    settings = load(store)
    env = os.environ.get(keys.ENV, "").strip()
    tonight = ledger.spent(store)
    return {
        "enabled": settings.enabled,
        "provider": settings.provider,
        "provider_label": settings.provider_label,
        "base_url": settings.base_url,
        "model": settings.model,
        "budget_usd": settings.budget_usd,
        "key": {
            "set": bool(env or settings.key_tail),
            "tail": keys.masked(env) if env else settings.key_tail,
            "source": "environment" if env else settings.key_store,
        },
        "disclaimer_accepted": accepted(settings),
        "context_window": settings.context_window,
        "cost_known": bool(settings.pricing) or settings.provider == "openrouter",
        "last_test": settings.last_test,
        "off_reason": settings.off_reason,
        "tonight": tonight,
    }


def details(store) -> dict:
    """The setup panel: summary, disclaimer, suggested models, tonight's calls."""
    from ..wording import (
        CLOUD_DISCLAIMER_TITLE,
        CLOUD_DISCLAIMER_VERSION,
        cloud_disclaimer,
    )

    settings = load(store)
    night = ledger.latest_night(store)
    return {
        "settings": summary(store),
        # The latest night with calls: tonight's, else last night's.
        "night": night and {"night": night, **ledger.spent(store, night)},
        "calls": ledger.calls(store, night=night, limit=200) if night else [],
        "disclaimer": {
            "version": CLOUD_DISCLAIMER_VERSION,
            "title": CLOUD_DISCLAIMER_TITLE,
            "lines": cloud_disclaimer(catalog.TERMS.get(settings.provider, "")),
            "terms_url": catalog.TERMS.get(settings.provider, ""),
        },
        "models": [{"id": i, "label": label} for i, label in catalog.CURATED],
        "recommended": catalog.RECOMMENDED,
    }


__all__ += ["details", "forget_key", "redact", "test"]
