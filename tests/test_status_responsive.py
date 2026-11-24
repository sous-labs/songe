"""#71: while measuring, `songe status` timed out and the dashboard took 25 s
to connect. Measuring writes to the store every few seconds, so the cached
"does this model need measuring again" answer was recomputed on every status
check and scheduler tick, each time decoding every recorded step (whole agent
histories), partly on the service's event loop."""

import asyncio
import json
import threading
from types import SimpleNamespace

from songe.config import Config
from songe.log import Store
from songe.schedule import ModelScheduler
from songe.sleep import calibration_state


def recorded(store):
    base = {"context": [{"role": "user", "content": "x" * 500}], "tools": [], "split": "train"}
    hindsights = [[{"role": "user", "content": "no"}], [], None, "again", {"k": 1}, {}]
    items = []
    for i, h in enumerate(hindsights):
        step = base | {"id": f"s{i}", "source_id": i + 1, "conversation": f"c{i}"}
        if h is not None:
            step["hindsight"] = h
        items.append(step)
    items.append(base | {"id": "late", "source_id": 3, "observed_id": 40, "conversation": "c9",
                         "hindsight": [{"role": "user", "content": "fix"}]})
    items.append(base | {"id": "gone", "source_id": 50, "conversation": "c8",
                         "hindsight": [{"role": "user", "content": "x"}]})
    store.sync_steps(items)
    with store.connect() as db:
        db.execute("UPDATE steps SET retired_at=1 WHERE id='gone'")
    # Pool entries: a newer revision than its step, one whose step has no
    # feedback any more, and an older one than its step.
    pool = [base | {"id": "s0", "source_id": 1, "observed_id": 70},
            base | {"id": "s1", "source_id": 2, "observed_id": 9},
            base | {"id": "late", "source_id": 3}]
    with store.connect() as db:
        db.executemany("INSERT INTO pool VALUES (?,?,?,?)",
                       [(s["id"], 0.0, json.dumps(s), store.model_id) for s in pool])


def test_sample_revisions_match_the_decoded_pool(tmp_path):
    store = Store(tmp_path / "data")
    recorded(store)
    decoded = calibration_state.sample_pool(store)
    assert {s["id"] for s in decoded} == {"s0", "s1", "s3", "s4", "late"}
    assert store.sample_revisions() == {
        s["id"]: max(s.get("source_id", 0), s.get("observed_id", 0)) for s in decoded
    }
    assert calibration_state.revisions_key(store.sample_revisions()) == calibration_state.pool_key(
        decoded
    )


def test_needs_growth_never_decodes_recorded_steps(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    recorded(store)
    config = Config()
    pool = calibration_state.pool_key(calibration_state.sample_pool(store))
    store.set("calibration", {"rows": [{"sequence": 1024}], "progress": {
        "policy": calibration_state.policy_key(config), "pool": pool, "complete": True,
        "target_sequence": 1 << 20, "completion_reason": "memory"}})

    def decode(*a, **k):
        raise AssertionError("decoded every recorded step")

    monkeypatch.setattr(Store, "active_steps", decode)
    monkeypatch.setattr(Store, "pool", decode)
    monkeypatch.setattr(calibration_state, "pending_keys", lambda store, config=None: set())
    assert calibration_state._needs_growth(config, store) is False
    # A new revision still asks for measuring again.
    monkeypatch.undo()
    store.sync_steps([{"id": "new", "source_id": 99, "conversation": "c7", "tools": [],
                       "context": [], "hindsight": [{"role": "user", "content": "no"}],
                       "split": "train"}])
    monkeypatch.setattr(Store, "active_steps", decode)
    monkeypatch.setattr(Store, "pool", decode)
    assert calibration_state._needs_growth(config, store) is True


def test_choosing_the_next_model_runs_off_the_event_loop(monkeypatch):
    loop_thread = []
    seen = []

    def choose(excluded=()):
        seen.append(threading.current_thread())
        return None if excluded or len(seen) > 1 else "m"

    def growth(config, store):
        seen.append(threading.current_thread())
        return True

    monkeypatch.setattr(calibration_state, "needs_growth", growth)
    fake = SimpleNamespace(
        choose_model=choose,
        select=lambda model: None,
        config=SimpleNamespace(calibration_available=True),
        store=None,
        active_model=None,
    )

    async def run():
        loop_thread.append(threading.current_thread())
        return await ModelScheduler.next_model(fake)

    assert asyncio.run(run()) == "m"
    assert seen and all(t is not loop_thread[0] for t in seen)


def test_songe_open_checks_only_that_the_service_answers(tmp_path, monkeypatch):
    # `songe open` waited on the whole state before opening the page, which
    # then asks for it again (#71: about 25 s on "Connecting to Songe").
    import httpx

    from songe import cli
    from songe.api import LocalClient

    root = tmp_path / "data"
    Config().save(root)
    identity = Store(root).get("installation_id")
    asked = []

    def send(client, request, **kwargs):
        asked.append((request.url.path, request.url.query.decode()))
        return httpx.Response(200, json={"schema": 1, "service_running": True,
                                         "installation_id": identity}, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(cli, "catch_up", lambda root, config: None)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: None)
    assert cli.main(["--data-dir", str(root), "open"]) == 0
    assert asked == [("/api/state", "service=1")]

    def refused(client, request, **kwargs):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(httpx.Client, "send", refused)
    assert LocalClient(root).running() is False
