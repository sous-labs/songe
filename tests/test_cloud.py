"""Learn with a cloud model: settings, key storage, redaction, budget, log,
probe, API, CLI. No test reaches a real provider or keychain (conftest)."""

import json
import os
import stat
import time

import httpx
import pytest

from songe import cloud
from songe.cloud import client, keys, ledger
from songe.cloud.redact import MARK, redact, redact_value

KEY = "sk-or-v1-" + "0123456789abcdef0123456789abcdef"


class Provider:
    """A fake OpenAI-compatible provider. Records every request."""

    def __init__(self, *, verdicts=("yes", "no"), status=200, cost=0.0004,
                 models=True, context=200_000, reply=None, usage=True):
        self.requests = []
        self.verdicts = list(verdicts)
        self.status = status
        self.cost = cost
        self.models = models
        self.context = context
        self.reply = reply
        self.usage = usage
        self.answered = 0

    def __call__(self, request):
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            if not self.models:
                return httpx.Response(404)
            return httpx.Response(200, json={"data": [{
                "id": "anthropic/claude-sonnet-5.5",
                "context_length": self.context,
                "pricing": {"prompt": "0.000002", "completion": "0.00001"},
            }, {"id": "anthropic/claude-opus-5.5", "context_length": 1_000_000},
                {"id": "custom-model", "max_model_len": 32768}]})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "nope"}})
        body = json.loads(request.content)
        if self.reply is not None:
            text = self.reply
        else:
            # The probe asks twice: an answer that follows, then one that doesn't.
            text = '{"follows": "%s"}' % self.verdicts[self.answered % len(self.verdicts)]
        self.answered += 1
        usage = {"prompt_tokens": 120, "completion_tokens": 8}
        if self.cost is not None:
            usage["cost"] = self.cost
        return httpx.Response(200, json={
            "model": body["model"],
            "choices": [{"message": {"role": "assistant", "content": text}}],
            **({"usage": usage} if self.usage else {}),
        })

    @property
    def chats(self):
        return [r for r in self.requests if r.url.path.endswith("/chat/completions")]


@pytest.fixture
def provider(monkeypatch):
    fake = Provider()
    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(fake))
    return fake


def turn_on(store, **extra):
    changes = {
        "accept_disclaimer": True,
        "model": "anthropic/claude-sonnet-5.5",
        "api_key": KEY,
        "enabled": True,
    } | extra
    return cloud.update(store, changes)


# ---- redaction ----

# Fake tokens, each split in two so secret scanners don't flag this file.
@pytest.mark.parametrize("secret", [
    "sk-proj-" + "AbCdEf0123456789AbCdEf0123456789",
    "sk-ant-api03-" + "AbCdEf0123456789AbCdEf0123456789",
    KEY,
    "ghp_" + "0123456789abcdefABCDEF0123456789abcd",
    "github_pat_" + "11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz",
    "glpat-" + "0123456789abcdefABCD",
    "AKIA" + "IOSFODNN7EXAMPLE",
    "xoxb-" + "123456789012-1234567890123-AbCdEfGhIjKlMnOp",
    "AIza" + "SyA0123456789abcdefghijklmnopqrstuv",
    "hf_" + "abcdefghijklmnopqrstuvwxyz0123456789",
    "sk_live_" + "0123456789abcdefABCDEF",
    "eyJhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
])
def test_known_token_shapes_are_redacted(secret):
    text = f"here it is: {secret} and more"
    out = redact(text)
    assert secret not in out
    assert MARK in out
    assert out.startswith("here it is: ") and out.endswith(" and more")


def test_pem_private_keys_whole_block_and_cut_off():
    block = (
        "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234\nabcd==\n"
        "-----END RSA PRIVATE KEY-----\nafter"
    )
    out = redact(block)
    assert "MIIE" not in out and "abcd==" not in out
    assert out.startswith("before\n") and out.endswith("\nafter")
    cut = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAA"
    assert "b3Bl" not in redact(cut)


def test_bearer_and_authorization_headers():
    out = redact("curl -H 'Authorization: Bearer abc123def456ghi789' https://x")
    assert "abc123def456ghi789" not in out and "Bearer [REDACTED]" in out
    # Prose stays prose.
    assert redact("Bearer authentication is configured") == "Bearer authentication is configured"


@pytest.mark.parametrize("line,secret", [
    ("OPENAI_API_KEY=abcd1234efgh5678", "abcd1234efgh5678"),
    ('export DB_PASSWORD="hunter2hunter2"', "hunter2hunter2"),
    ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI"),
    ('{"api_key": "q9w8e7r6t5y4"}', "q9w8e7r6t5y4"),
    ("password: S3cure!pass", "S3cure!pass"),
    ("STRIPE_TOKEN = 'tok_1234567890'", "tok_1234567890"),
    ("SENTRY_DSN=https://abc123@o1.ingest.sentry.io/42", "abc123"),
])
def test_env_style_secrets_keep_the_name(line, secret):
    out = redact(line)
    assert secret not in out
    assert MARK in out
    name = line.split("=")[0].split(":")[0].strip(' {"\'')
    assert name.removeprefix("export ") in out


@pytest.mark.parametrize("text", [
    "max_tokens=4096",
    "token = get_token()",
    "auth = self.authenticator",
    "def check(password: str) -> bool:",
    "author: Jonathan",
    "API_KEY=$OPENAI_API_KEY",
    "api_key=os.environ",
    "pip install scikit-learn  # sk-learn",
    "the token budget is 512",
])
def test_ordinary_code_is_left_alone(text):
    assert redact(text) == text


def test_url_credentials():
    out = redact("postgres://admin:pa55word@db.internal:5432/app")
    assert "pa55word" not in out and "db.internal:5432/app" in out


def test_home_folder_becomes_tilde_only_as_a_whole_component():
    home = "/Users/ann"
    text = "open /Users/ann/project/main.py, not /Users/anna/x; cwd=/Users/ann"
    assert redact(text, home) == "open ~/project/main.py, not /Users/anna/x; cwd=~"
    windows = r"C:\Users\ann"
    assert redact(r"at C:\Users\ann\src\app.py", windows) == r"at ~\src\app.py"
    assert redact("at C:/Users/ann/src", windows) == "at ~/src"


def test_redact_value_walks_messages_and_tool_calls():
    home = os.path.expanduser("~")
    messages = [
        {"role": "user", "content": f"read {home}/.env"},
        {"role": "tool", "content": [{"type": "text", "text": "OPENAI_API_KEY=abcd1234efgh5678"}]},
        {"role": "assistant", "content": None, "tool_calls": [{
            "function": {"name": "bash", "arguments": json.dumps({"cmd": f"export GH_TOKEN={KEY}"})},
        }]},
    ]
    out = json.dumps(redact_value(messages))
    assert home not in out
    assert "abcd1234efgh5678" not in out and KEY not in out
    assert "~/.env" in out
    assert redact_value(messages)[2]["content"] is None


# ---- the key ----

def test_key_goes_to_the_keychain_never_the_store(store):
    assert keys.save(store.root, KEY) == "keychain"
    assert keys.load(store.root) == (KEY, "keychain")
    assert not (store.root / keys.FILE).exists()
    with store.connect() as db:
        dump = json.dumps([list(r) for r in db.execute("SELECT * FROM state")])
    assert KEY not in dump
    assert keys.masked(KEY) == "…cdef"
    keys.delete(store.root)
    assert keys.load(store.root) == ("", "")


def test_each_installation_has_its_own_key(tmp_path):
    keys.save(tmp_path / "a", "sk-aaaaaaaaaaaaaaaaaaaaaaaa")
    assert keys.load(tmp_path / "b") == ("", "")


def test_no_keychain_falls_back_to_a_private_file(store, monkeypatch):
    monkeypatch.setattr(keys, "BACKEND", False)
    assert keys.save(store.root, KEY) == "file"
    path = store.root / keys.FILE
    assert path.read_text(encoding="utf-8") == KEY
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert keys.load(store.root) == (KEY, "file")
    keys.delete(store.root)
    assert not path.exists()


def test_refusing_keychain_falls_back_to_the_file(store, monkeypatch):
    class Locked:
        def set_password(self, *a):
            raise RuntimeError("locked")

        def get_password(self, *a):
            raise RuntimeError("locked")

        def delete_password(self, *a):
            raise RuntimeError("locked")

    monkeypatch.setattr(keys, "BACKEND", Locked())
    assert keys.save(store.root, KEY) == "file"
    assert keys.load(store.root) == (KEY, "file")


def test_environment_key_is_used_but_openai_key_never(store, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-read-0123456789")
    assert keys.load(store.root) == ("", "")
    monkeypatch.setenv("SONGE_CLOUD_API_KEY", " sk-env-0123456789abcdef ")
    assert keys.load(store.root) == ("sk-env-0123456789abcdef", "environment")


def test_no_environment_variable_turns_it_on(store, monkeypatch, provider):
    monkeypatch.setenv("SONGE_CLOUD_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    assert cloud.load(store).enabled is False
    state = cloud.available(store)
    assert not state and state.reason == "cloud learning is off"
    with pytest.raises(cloud.Unavailable):
        cloud.complete(store, [{"role": "user", "content": "hi"}], purpose="note")
    assert provider.requests == []


# ---- turning it on ----

def test_off_by_default(store):
    settings = cloud.load(store)
    assert settings.enabled is False
    assert settings.budget_usd == 1.0
    assert cloud.summary(store)["key"] == {"set": False, "tail": "", "source": ""}


def test_turning_on_needs_the_disclaimer(store, provider):
    with pytest.raises(ValueError, match="Accept"):
        cloud.update(store, {"model": "anthropic/claude-sonnet-5.5", "api_key": KEY, "enabled": True})
    assert cloud.load(store).enabled is False
    assert provider.requests == []


def test_turning_on_needs_a_key(store, provider):
    with pytest.raises(ValueError, match="no API key"):
        cloud.update(store, {"accept_disclaimer": True, "model": "x/y", "enabled": True})
    assert cloud.load(store).enabled is False


def test_turning_on_runs_the_probe_and_keeps_what_it_learned(store, provider):
    settings, found = turn_on(store)
    assert found.ok and found.message == "works · 200k context"
    assert settings.enabled and settings.context_window == 200_000
    assert settings.pricing == {"prompt": 2e-6, "completion": 1e-5}
    assert settings.key_store == "keychain" and settings.key_tail == "…cdef"
    assert KEY not in json.dumps(store.get("cloud"))
    # The models list is public: no key is sent for it.
    listing = next(r for r in provider.requests if r.url.path.endswith("/models"))
    assert "authorization" not in listing.headers
    chat = provider.chats[0]
    assert chat.headers["authorization"] == f"Bearer {KEY}"
    assert str(chat.url) == "https://openrouter.ai/api/v1/chat/completions"
    # The probe asked the gate's judge question, twice.
    assert len(provider.chats) == 2
    assert '"follows"' in json.loads(chat.content)["messages"][0]["content"]
    assert [c["purpose"] for c in ledger.calls(store)] == ["test", "test"]
    assert cloud.available(store)


@pytest.mark.parametrize("fake,message", [
    (Provider(reply="I think it does follow it."), "replies couldn't be read as a verdict"),
    (Provider(verdicts=("yes", "yes")), "answered a clear test case wrongly"),
    (Provider(status=401), "refused the API key"),
    (Provider(status=402), "out of credit"),
])
def test_a_failing_probe_leaves_it_off(store, monkeypatch, fake, message):
    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(fake))
    with pytest.raises(ValueError, match=message):
        turn_on(store)
    settings = cloud.load(store)
    assert settings.enabled is False
    assert settings.last_test["ok"] is False and message in settings.last_test["message"]
    # A key that never passed is not kept either.
    assert keys.load(store.root) == ("", "")


def test_unknown_model_fails_before_any_chat(store, provider):
    with pytest.raises(ValueError, match="has no model nope/none"):
        turn_on(store, model="nope/none")
    assert provider.chats == []


def test_offline_probe_says_so(store):
    with pytest.raises(ValueError, match="can't reach OpenRouter"):
        turn_on(store)
    assert cloud.load(store).enabled is False


def test_changing_the_key_while_on_is_tested_first(store, provider, monkeypatch):
    turn_on(store)
    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(Provider(status=401)))
    with pytest.raises(ValueError, match="refused"):
        cloud.update(store, {"api_key": "sk-new-0123456789abcdefgh"})
    assert keys.load(store.root)[0] == KEY
    assert cloud.load(store).enabled is True


def test_budget_change_while_on_needs_no_probe(store, provider):
    turn_on(store)
    before = len(provider.requests)
    settings, found = cloud.update(store, {"budget_usd": 2.5})
    assert found is None and settings.budget_usd == 2.5 and settings.enabled
    assert len(provider.requests) == before


@pytest.mark.parametrize("changes,message", [
    ({"budget_usd": 0}, "budget"),
    ({"budget_usd": True}, "budget"),
    ({"budget_usd": float("nan")}, "budget"),
    ({"provider": "anthropic"}, "provider"),
    ({"provider": "custom", "base_url": "http://example.com/v1"}, "https"),
    ({"enabled": "yes"}, "enabled"),
    ({"api_key": ""}, "api_key"),
    ({"accept_disclaimer": 99}, "accept_disclaimer"),
    ({"unknown": 1}, "Cloud settings"),
    ({}, "Cloud settings"),
])
def test_invalid_changes_are_refused(store, changes, message):
    with pytest.raises(ValueError, match=message):
        cloud.update(store, changes)


def test_custom_endpoint(store, monkeypatch):
    fake = Provider(cost=None)
    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(fake))
    settings, found = turn_on(
        store, provider="custom", base_url="https://llm.example.com/v1/", model="custom-model"
    )
    assert settings.base_url == "https://llm.example.com/v1"
    assert found.message == "works · 33k context · cost not reported · the budget is best effort"
    assert str(fake.chats[0].url) == "https://llm.example.com/v1/chat/completions"
    # A custom endpoint's model list may need the key; OpenRouter's header is not sent.
    assert fake.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert "x-title" not in fake.chats[0].headers
    assert cloud.summary(store)["cost_known"] is False


def test_turning_off_stops_calls_at_once(store, provider):
    turn_on(store)
    cloud.update(store, {"enabled": False})
    before = len(provider.requests)
    with pytest.raises(cloud.Unavailable, match="off"):
        cloud.complete(store, [{"role": "user", "content": "x"}], purpose="note")
    assert len(provider.requests) == before


def test_a_new_disclaimer_must_be_accepted_again(store, provider, monkeypatch):
    from songe import wording

    turn_on(store)
    monkeypatch.setattr(wording, "CLOUD_DISCLAIMER_VERSION", 2)
    state = cloud.available(store)
    assert not state and "accept it again" in state.reason


# ---- calls ----

def test_complete_redacts_logs_and_counts_cost(store, provider):
    turn_on(store)
    home = os.path.expanduser("~")
    reply = cloud.complete(store, [
        {"role": "system", "content": "You explain mistakes."},
        {"role": "user", "content": f"in {home}/app the key OPENAI_API_KEY=abcd1234efgh5678 fails"},
    ], purpose="note", max_tokens=300, dream="cycle-7")
    assert reply.text == '{"follows": "yes"}'
    assert reply.cost_usd == 0.0004 and reply.cost_source == "provider"
    sent = json.loads(provider.chats[-1].content)
    raw = json.dumps(sent)
    assert "abcd1234efgh5678" not in raw and home not in raw and "~/app" in raw
    assert sent["max_tokens"] == 300 and sent["usage"] == {"include": True}
    row = ledger.calls(store, dream="cycle-7")[0]
    assert row["purpose"] == "note" and row["ok"] and row["sent"]
    assert row["prompt_tokens"] == 120 and row["completion_tokens"] == 8
    assert "abcd1234efgh5678" not in row["preview"] and row["preview"].startswith("in ~/app")
    assert row["sent_chars"] == len(json.dumps(sent["messages"], ensure_ascii=False))
    tonight = ledger.spent(store)
    assert tonight["calls"] == 3 and tonight["usd"] == pytest.approx(0.0012)


def test_cost_from_pricing_when_the_provider_reports_none(store, monkeypatch):
    fake = Provider(cost=None)
    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(fake))
    turn_on(store)
    reply = cloud.complete(store, [{"role": "user", "content": "x"}], purpose="judge")
    assert reply.cost_source == "pricing"
    assert reply.cost_usd == pytest.approx(120 * 2e-6 + 8 * 1e-5)


def test_budget_refuses_before_sending(store, provider):
    turn_on(store, budget_usd=0.01)
    before = len(provider.chats)
    big = "word " * 20_000  # about 33k tokens at $2/M: more than what is left
    with pytest.raises(cloud.Unavailable, match="budget"):
        cloud.complete(store, [{"role": "user", "content": big}], purpose="note", max_tokens=100)
    assert len(provider.chats) == before
    refused = ledger.calls(store)[-1]
    assert refused["sent"] is False and refused["error"].startswith("not sent")
    assert ledger.spent(store)["calls"] == 2  # the probe's two; the refusal costs nothing


def test_spent_budget_makes_it_unavailable(store, provider):
    turn_on(store, budget_usd=0.0008)  # the probe's two calls spend it all
    state = cloud.available(store)
    assert not state and "used up" in state.reason
    assert cloud.load(store).enabled  # still on: tomorrow night it runs again


def test_unknown_cost_is_best_effort(store, monkeypatch):
    fake = Provider(cost=None)
    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(fake))
    turn_on(store, provider="custom", base_url="https://llm.example.com/v1",
            model="custom-model", budget_usd=0.01)
    for _ in range(3):
        cloud.complete(store, [{"role": "user", "content": "x"}], purpose="note")
    assert cloud.remaining(store) is None
    assert ledger.spent(store)["cost_known"] is False


def test_refused_key_turns_it_off_with_the_reason(store, provider, monkeypatch):
    turn_on(store)
    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(Provider(status=401)))
    with pytest.raises(cloud.Unavailable, match="refused the API key"):
        cloud.complete(store, [{"role": "user", "content": "x"}], purpose="note")
    settings = cloud.load(store)
    assert settings.enabled is False and "refused the API key" in settings.off_reason
    assert ledger.calls(store)[-1]["error"] == "OpenRouter refused the API key (401)"


def test_offline_keeps_it_on_and_says_why(store, provider, monkeypatch):
    turn_on(store)

    def down(request):
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(down))
    with pytest.raises(cloud.Unavailable) as caught:
        cloud.complete(store, [{"role": "user", "content": "x"}], purpose="note")
    assert caught.value.reason == "can't reach OpenRouter (offline?)"
    assert str(caught.value) == "Cloud model unavailable: can't reach OpenRouter (offline?)"
    assert cloud.load(store).enabled


def test_a_missing_key_turns_it_off(store, provider):
    turn_on(store)
    keys.delete(store.root)
    state = cloud.available(store)
    assert not state and "key is missing" in state.reason
    assert cloud.load(store).enabled is False
    assert "key is missing" in cloud.load(store).off_reason


def test_unreadable_reply_is_unavailable(store, provider, monkeypatch):
    turn_on(store)
    monkeypatch.setattr(
        client, "TRANSPORT", httpx.MockTransport(lambda r: httpx.Response(200, text="<html>"))
    )
    with pytest.raises(cloud.Unavailable, match="couldn't read"):
        cloud.complete(store, [{"role": "user", "content": "x"}], purpose="note")


def test_night_runs_noon_to_noon():
    at = time.mktime((2026, 11, 25, 23, 0, 0, 0, 0, -1))
    after_midnight = time.mktime((2026, 11, 26, 3, 0, 0, 0, 0, -1))
    next_afternoon = time.mktime((2026, 11, 26, 13, 0, 0, 0, 0, -1))
    assert ledger.night_key(at) == ledger.night_key(after_midnight) == "2026-11-25"
    assert ledger.night_key(next_afternoon) == "2026-11-26"


def test_forget_clears_what_was_sent_but_keeps_the_spending(store, provider):
    turn_on(store)
    cloud.complete(store, [{"role": "user", "content": "secret plan"}], purpose="note")
    store.forget()
    rows = ledger.calls(store)
    assert rows and all(r["preview"] == "" for r in rows)
    assert ledger.spent(store)["usd"] > 0


def test_cycle_record_lists_the_dreams_calls(config, store, provider):
    from songe.cycle import cycle_id, full_cycle

    turn_on(store)  # its test calls came before this Dream
    night = store.night(None)
    record = {"started_at": time.time() + 0.001, "learning": "hindsight"}
    time.sleep(0.01)
    store.night(night, status="running", record=record)
    cloud.complete(store, [{"role": "user", "content": "x"}], purpose="note",
                   dream=cycle_id(night))
    found = full_cycle(config, store, "songe/test", cycle_id(night))
    assert [c["purpose"] for c in found["cloud_calls"]] == ["note"]


# ---- the page's API ----

@pytest.fixture
def api(config, store):
    from songe.proxy import create_app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(config, store)), base_url="http://127.0.0.1"
    )


async def test_api_key_is_write_only(api, store, provider):
    async with api:
        response = await api.patch("/api/settings", json={"cloud": {"api_key": KEY}})
        assert response.status_code == 200
        assert response.json()["cloud"]["key"] == {"set": True, "tail": "…cdef", "source": "keychain"}
        assert KEY not in response.text
        for path in ("/api/settings", "/api/state", "/api/cloud"):
            assert KEY not in (await api.get(path)).text


async def test_api_turns_on_only_after_disclaimer_and_probe(api, store, provider):
    async with api:
        refused = await api.patch(
            "/api/settings", json={"cloud": {"enabled": True, "api_key": KEY, "model": "anthropic/claude-sonnet-5.5"}}
        )
        assert refused.status_code == 400 and "Accept" in refused.json()["error"]
        details = (await api.get("/api/cloud")).json()
        assert details["disclaimer"]["version"] == 1
        assert any("home folder" in line for line in details["disclaimer"]["lines"])
        assert details["models"][0]["id"] == details["recommended"]
        on = await api.patch("/api/settings", json={"cloud": {
            "accept_disclaimer": details["disclaimer"]["version"], "enabled": True,
            "model": "anthropic/claude-sonnet-5.5", "api_key": KEY,
        }})
        assert on.status_code == 200, on.text
        assert on.json()["cloud"]["enabled"] is True
        assert on.json()["cloud"]["last_test"]["message"] == "works · 200k context"
        state = (await api.get("/api/state")).json()
        assert state["settings"]["cloud"]["enabled"] is True
        details = (await api.get("/api/cloud")).json()
        assert [c["purpose"] for c in details["calls"]] == ["test", "test"]
        assert details["night"]["calls"] == 2
        tested = await api.post("/api/cloud/test", json={})
        assert tested.status_code == 200 and tested.json()["ok"] is True


async def test_api_probe_failure_is_a_plain_400(api, store, monkeypatch):
    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(Provider(reply="maybe")))
    async with api:
        response = await api.patch("/api/settings", json={"cloud": {
            "accept_disclaimer": 1, "enabled": True, "model": "anthropic/claude-sonnet-5.5", "api_key": KEY,
        }})
    assert response.status_code == 400
    assert response.json()["error"] == "Not turned on · replies couldn't be read as a verdict"
    assert cloud.load(store).enabled is False


async def test_api_test_needs_json_from_the_page(api):
    async with api:
        response = await api.post("/api/cloud/test", content=b"{}")
    assert response.status_code == 415


# ---- the command line ----

def cli(*argv):
    from songe.cli import main

    return main(["--data-dir", str(cli.root), *argv])


@pytest.fixture
def installed(config, tmp_path):
    root = tmp_path / "data"
    config.save(root)
    cli.root = root
    return root


def test_cli_on_for_scripts(installed, provider, monkeypatch, capsys):
    from songe.log import Store

    monkeypatch.setenv("SONGE_CLOUD_API_KEY", KEY)
    assert cli("cloud", "on", "--accept-disclaimer", "--budget", "2") == 0
    out = capsys.readouterr().out
    assert "Cloud learning on · anthropic/claude-sonnet-5.5 via OpenRouter · works" in out
    assert "home folder" in out  # the disclaimer was shown
    store = Store(installed)
    assert cloud.load(store).enabled and cloud.load(store).budget_usd == 2
    # Saved for the background service, which does not see this shell's variable.
    monkeypatch.delenv("SONGE_CLOUD_API_KEY")
    assert keys.load(installed) == (KEY, "keychain")


def test_cli_on_without_acceptance_or_key_changes_nothing(installed, provider, monkeypatch, capsys):
    from songe.log import Store

    assert cli("cloud", "on") == 1
    assert "--accept-disclaimer" in capsys.readouterr().err
    assert cli("cloud", "on", "--accept-disclaimer") == 1
    assert "SONGE_CLOUD_API_KEY" in capsys.readouterr().err
    assert cloud.load(Store(installed)).enabled is False
    assert provider.requests == []


def test_cli_on_failing_probe(installed, monkeypatch, capsys):
    from songe.log import Store

    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(Provider(verdicts=("no",))))
    monkeypatch.setenv("SONGE_CLOUD_API_KEY", KEY)
    assert cli("cloud", "on", "--accept-disclaimer") == 1
    assert "answered a clear test case wrongly" in capsys.readouterr().err
    assert cloud.load(Store(installed)).enabled is False


def test_cli_interactive_on(installed, provider, monkeypatch, capsys):
    import getpass
    import sys

    from songe.log import Store

    answers = iter(["y", "1", "2", "3"])  # accept, OpenRouter, model 2, $3
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": KEY)
    assert cli("cloud", "on") == 0
    settings = cloud.load(Store(installed))
    assert settings.enabled and settings.model == "anthropic/claude-opus-5.5"
    assert settings.budget_usd == 3.0


def test_cli_off_status_and_test(installed, provider, monkeypatch, capsys):
    from songe.log import Store

    monkeypatch.setenv("SONGE_CLOUD_API_KEY", KEY)
    cli("cloud", "on", "--accept-disclaimer")
    capsys.readouterr()
    assert cli("cloud", "status") == 0
    out = capsys.readouterr().out
    assert "on" in out and "…cdef" in out and KEY not in out and "test" in out
    assert cli("cloud", "test") == 0
    assert "works" in capsys.readouterr().out
    assert cli("cloud", "off", "--forget-key") == 0
    assert "Cloud learning off" in capsys.readouterr().out
    assert cloud.load(Store(installed)).enabled is False
    monkeypatch.delenv("SONGE_CLOUD_API_KEY")
    assert keys.load(installed) == ("", "")


def test_songe_status_shows_cloud_learning_when_on(installed, provider, monkeypatch, capsys):
    monkeypatch.setattr(httpx.Client, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("stopped")))
    monkeypatch.setenv("SONGE_CLOUD_API_KEY", KEY)
    cli("cloud", "on", "--accept-disclaimer")
    capsys.readouterr()
    assert cli() == 0
    out = capsys.readouterr().out
    assert "Cloud learning on · anthropic/claude-sonnet-5.5 via OpenRouter" in out
    cli("cloud", "off")
    capsys.readouterr()
    cli()
    assert "Cloud learning" not in capsys.readouterr().out


def test_cli_cloud_needs_an_installation(tmp_path, capsys):
    from songe.cli import main

    assert main(["--data-dir", str(tmp_path / "none"), "cloud"]) == 1
    assert not (tmp_path / "none").exists()


def test_agent_traffic_still_only_goes_local(config):
    from dataclasses import replace

    with pytest.raises(ValueError, match="local"):
        replace(config, upstream_url="https://openrouter.ai/api/v1").validate()


def test_a_model_that_refuses_max_tokens_or_temperature_is_asked_again(monkeypatch):
    """Live, OpenAI's GPT-6 models through a custom endpoint: max_tokens and
    temperature 0 are refused with a 400 naming the field."""
    import json as _json

    import httpx

    from songe.cloud import client

    seen = []

    def fake(request):
        body = _json.loads(request.content)
        seen.append(sorted(body))
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message": "Use 'max_completion_tokens'", "param": "max_tokens"}})
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "does not support 0", "param": "temperature"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(fake))
    reply = client.chat(provider="custom", provider_name="api.openai.com", url="https://api.openai.com/v1",
                        key="k", model="gpt-6.2-sol", messages=[{"role": "user", "content": "hi"}], max_tokens=20)
    assert reply.text == "ok" and len(seen) == 3
    assert "max_completion_tokens" in seen[-1] and "temperature" not in seen[-1]


def test_a_night_key_for_a_clock_near_zero():
    """Windows' localtime refuses negative times; tests run with a fake clock."""
    from songe.cloud.ledger import night_key

    assert night_key(100.0)
