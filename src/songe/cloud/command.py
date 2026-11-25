"""`songe cloud on | off | status | test`.

`on` asks, in order: accept what is sent, the provider, the model, the key
(typed without echo, or SONGE_CLOUD_API_KEY) and the nightly budget; then
it runs the probe and turns cloud learning on only if the probe passes.
Every question has a flag for scripts; the key never does (flags end up in
shell history), so scripts give it through SONGE_CLOUD_API_KEY.
"""

from __future__ import annotations

import getpass
import sys
import time

from rich.text import Text

from .. import NAME
from ..terminal import console, error, facts, result, step
from . import catalog, keys, ledger, load, summary, update
from . import test as run_test


def ask(prompt, default=""):
    shown = f"{prompt} [{default}] " if default else f"{prompt} "
    answer = input(shown).strip()
    return answer or default


def disclaimer(provider):
    from ..wording import CLOUD_DISCLAIMER_TITLE, cloud_disclaimer

    out = console()
    out.print(Text(CLOUD_DISCLAIMER_TITLE, style="bold"))
    for line in cloud_disclaimer(catalog.TERMS.get(provider, "")):
        out.print(Text(f"  · {line}"))
    out.print()


def choose_provider(args, current, interactive):
    if args.provider:
        return args.provider
    if not interactive:
        return current.provider
    out = console()
    out.print(Text("Provider"))
    out.print(Text("  1  OpenRouter · recommended, one key for many models"))
    out.print(Text("  2  Another OpenAI-compatible endpoint (base URL)"))
    default = "2" if current.provider == "custom" else "1"
    while True:
        answer = ask("Choose 1 or 2", default)
        if answer in {"1", "2"}:
            return "openrouter" if answer == "1" else "custom"


def choose_base_url(args, current, provider, interactive):
    if provider != "custom":
        return ""
    if args.base_url:
        return args.base_url
    if not interactive:
        if current.base_url:
            return current.base_url
        raise ValueError("Custom provider: give its base URL · --base-url https://…/v1")
    return ask("Base URL (https://…/v1)", current.base_url)


def choose_model(args, current, provider, interactive):
    if args.model:
        return args.model
    default = current.model or (catalog.RECOMMENDED if provider == "openrouter" else "")
    if not interactive:
        if default:
            return default
        raise ValueError("Choose a model · --model <id>")
    out = console()
    if provider == "openrouter":
        out.print(Text("Model · a number from the list, or type any OpenRouter model id"))
        for index, (model, label) in enumerate(catalog.CURATED, 1):
            out.print(Text(f"  {index}  {model}  ", style="default").append(label, style="dim"))
    else:
        out.print(Text("Model · the id your endpoint uses"))
    while True:
        answer = ask("Model", default)
        if answer.isdigit() and provider == "openrouter":
            index = int(answer) - 1
            if 0 <= index < len(catalog.CURATED):
                return catalog.CURATED[index][0]
            continue
        if answer:
            return answer


def choose_key(store, interactive):
    """The new key to save, or None to keep the saved one."""
    env = keys.load(store.root)
    if env[1] == "environment":
        console().print(
            Text(f"Key from {keys.ENV} · saved to the keychain so the background service can use it",
                 style="dim")
        )
        return env[0]
    saved, where = keys.stored(store.root)
    if saved:
        if not interactive:
            return None
        answer = input(f"Keep the saved key {keys.masked(saved)} ({where})? (Y/n) ").strip().lower()
        if answer in {"", "y", "yes"}:
            return None
    elif not interactive:
        raise ValueError(f"No API key · set {keys.ENV} for scripts")
    while True:
        key = getpass.getpass("API key (not shown): ").strip()
        if key:
            return key


def choose_budget(args, current, interactive):
    if args.budget is not None:
        return args.budget
    if not interactive:
        return current.budget_usd
    while True:
        answer = ask("Nightly budget in USD", f"{current.budget_usd:.2f}").lstrip("$")
        try:
            return float(answer)
        except ValueError:
            continue


def turn_on(args, store):
    interactive = sys.stdin.isatty()
    current = load(store)
    provider = args.provider or current.provider
    disclaimer(provider)
    if not args.accept_disclaimer:
        if not interactive:
            raise ValueError("Accept what is sent first · --accept-disclaimer")
        answer = input("Send these moments to the provider you choose during Dreams? (y/N) ")
        if answer.strip().lower() not in {"y", "yes"}:
            result("Nothing changed · cloud learning stays off")
            return 0
    provider = choose_provider(args, current, interactive)
    changes = {
        "accept_disclaimer": True,
        "provider": provider,
        "base_url": choose_base_url(args, current, provider, interactive),
        "model": choose_model(args, current, provider, interactive),
    }
    key = choose_key(store, interactive)
    if key is not None:
        changes["api_key"] = key
    changes["budget_usd"] = choose_budget(args, current, interactive)
    changes["enabled"] = True
    try:
        with step(f"Testing {changes['model']}", "Test passed"):
            settings, found = update(store, changes)
    except ValueError as exc:
        error(exc, f"try again  {NAME} cloud on · details  {NAME} cloud status")
        return 1
    result(f"Cloud learning on · {settings.model} via {settings.provider_label} · {found.message}")
    facts([
        ("Budget", f"${settings.budget_usd:.2f} a night"),
        ("Calls", f"listed in each Dream's record · {NAME} cloud status"),
        ("Stop", f"{NAME} cloud off"),
    ])
    return 0


def show(store):
    from ..wording import context_size, money

    data = summary(store)
    settings = load(store)
    key = data["key"]
    tonight = data["tonight"]
    rows = [
        ("State", "on" if data["enabled"] else "off"),
    ]
    if data["off_reason"]:
        rows.append(("Turned off", data["off_reason"]))
    if data["model"]:
        rows.append(("Model", f"{data['model']} via {data['provider_label']}"))
        if data["provider"] == "custom":
            rows.append(("Base URL", data["base_url"]))
    rows.append(("Key", f"{key['tail']} · {key['source']}" if key["set"] else "not set"))
    rows.append(("Budget", f"{money(data['budget_usd'])} a night"))
    spent = money(tonight["usd"]) if tonight["cost_known"] else "cost unknown (budget best effort)"
    rows.append(("Tonight", f"{tonight['calls']} calls · {spent}"))
    if settings.context_window:
        rows.append(("Context", context_size(settings.context_window)))
    if data["last_test"]:
        test = data["last_test"]
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(test.get("at", 0)))
        rows.append(("Last test", f"{'passed' if test.get('ok') else 'failed'} · {test.get('message')} · {when}"))
    facts(rows)
    night = ledger.latest_night(store)
    recent = ledger.calls(store, night=night, limit=20) if night else []
    if recent:
        console().print()
        console().print(Text(f"  Calls · night of {night}", style="dim"))
        for call in recent:
            when = time.strftime("%H:%M", time.localtime(call["at"]))
            state = "ok" if call["ok"] else (call["error"] or "failed")
            cost = money(call["cost_usd"]) if call["sent"] else "not sent"
            console().print(
                Text(f"  {when}  {call['purpose']:<8} {call['model']}  {call['sent_chars']} chars  {cost}  {state}")
            )
    return 0


def main(args, store):
    action = args.action or "status"
    if action == "on":
        return turn_on(args, store)
    if action == "off":
        update(store, {"enabled": False})
        if args.forget_key:
            from . import forget_key

            forget_key(store)
        result("Cloud learning off · no more calls" + (" · key removed" if args.forget_key else ""))
        return 0
    if action == "test":
        with step("Testing the cloud model", "Test finished"):
            found = run_test(store)
        if found.ok:
            result(found.message)
            return 0
        error(found.message, f"change it  {NAME} cloud on")
        return 1
    return show(store)
