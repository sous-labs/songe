import copy
import json
import sqlite3

import httpx

from songe.api import ReadOnlyStore, snapshot
from songe.learning_summary import summaries, summary_lines
from songe.log import SCHEMA, SCHEMA_V2
from songe.models import registry
from songe.proxy import create_app


def call(identifier, command):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": identifier,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def recorded(store, messages, action, created, recorded_at):
    source = store.record(
        "/v1/chat/completions",
        {"model": "first", "messages": messages},
        {"choices": [{"message": action}]},
        binding="first",
        started=created,
    )
    with store.connect() as db:
        db.execute(
            "UPDATE exchanges SET recorded_at=? WHERE id=?", (recorded_at, source)
        )
    return source


def experience(store):
    first = {"role": "user", "content": "Run pytest\nSecond line"}
    action = call("one", "pytest tests/test_api.py")
    recorded(store, [first], action, 100, 110)
    messages = [
        first,
        action,
        {"role": "tool", "tool_call_id": "one", "content": '{"exit_code":1}'},
        {"role": "user", "content": "incorrect"},
        {"role": "user", "content": "try again"},
    ]
    retry = call("two", "git status && pytest tests/test_api.py")
    recorded(store, messages, retry, 200, 210)
    messages += [
        retry,
        {"role": "tool", "tool_call_id": "two", "content": '{"exit_code":0}'},
        {"role": "user", "content": "works"},
    ]
    recorded(store, messages, {"role": "assistant", "content": "Done"}, 300, 310)
    # A request started before the learning cutoff but committed afterward.
    recorded(
        store,
        [{"role": "user", "content": "Inspect git"}],
        {"role": "assistant", "content": "Pending"},
        90,
        160,
    )
    store.reconstruct()


def model(identifier="first", level="all", cutoff=None):
    return {"id": identifier, "learning_level": level, "learning_cutoff": cutoff}


def test_shared_history_windows_and_deduplicated_outcomes(store):
    experience(store)
    result = summaries(
        store, [model(), model("new", "new", 150), model("off", "off")], [], []
    )
    all_history = result["first"]
    assert all_history["conversations"] == 2
    assert all_history["since"] == 90
    assert all_history["signals"] == {
        "tool_error_fixed": 1,
        "user_correction": 1,
        "retry_success": 1,
    }
    assert set(all_history["topics"]) == {"bash", "git", "pytest", ".py"}
    assert [r["first_line"] for r in all_history["sources"]] == [
        "Inspect git",
        "Run pytest",
    ]
    assert result["new"]["conversations"] == 2
    assert result["new"]["signals"] == {
        "tool_error_fixed": 0,
        "user_correction": 0,
        "retry_success": 0,
    }
    assert result["off"]["conversations"] == 0
    assert result["off"]["since"] is None


def test_unresolved_errors_and_success_before_failure_are_not_fixes(store):
    experience(store)
    with store.connect() as db:
        db.execute("UPDATE steps SET retired_at=1")
    source = recorded(
        store,
        [{"role": "user", "content": "Unresolved"}],
        call("last", "pytest"),
        400,
        410,
    )
    store.sync_steps(
        [
            {
                "id": "unresolved",
                "conversation": "new-thread",
                "source_id": source,
                "context": [{"role": "user", "content": "Unresolved"}],
                "hindsight": [
                    {"role": "tool", "content": '{"exit_code":0}'},
                    {"role": "tool", "content": '{"error":"failed"}'},
                    {"role": "user", "content": "try again"},
                ],
            }
        ]
    )
    summary = summaries(store, [model()], [], [])["first"]
    assert summary["conversations"] == 1
    assert summary["signals"] == {
        "tool_error_fixed": 0,
        "user_correction": 0,
        "retry_success": 0,
    }


def test_last_result_scopes_metrics_and_keeps_unknowns(store):
    experience(store)
    store.set("primary_model", "first")
    models = [
        model() | {"versions": [{"id": "v1", "identifier": "first-v1"}]},
        model("second"),
    ]
    nights = [
        {
            "id": 1,
            "created": 100,
            "model_id": "",
            "status": "accepted",
            "record": {"general_drift": 0.002},
        },
        {
            "id": 2,
            "created": 200,
            "model_id": "second",
            "status": "rejected",
            "record": {},
        },
        {"id": 3, "created": 300, "model_id": "", "status": "running", "record": {}},
    ]
    scoreboard = [
        {"model_version": "first", "repeat_error_rate": 0.5},
        {"model_version": "first-v1", "repeat_error_rate": 0.41},
        {"model_version": "second", "repeat_error_rate": 0.9},
    ]
    result = summaries(store, models, nights, scoreboard)
    assert round(result["first"]["last"]["repeat_errors_change"], 2) == -0.18
    assert result["first"]["last"]["created"] == 100
    assert result["first"]["last"]["drift"] == 0.002
    assert result["second"]["last"] == {
        "created": 200,
        "repeat_errors_change": None,
        "drift": None,
        "status": "kept previous",
    }
    lines = summary_lines(models[0] | {"learning": result["first"]})
    assert "repeat_errors −18% · drift 0.002 · ✓ swapped" in lines[-1]
    scoreboard[0]["repeat_error_rate"] = 0
    assert (
        summaries(store, models, nights, scoreboard)["first"]["last"][
            "repeat_errors_change"
        ]
        is None
    )


async def test_api_and_offline_summary_agree_without_store_writes(config, store):
    registry.migrate(config, store)
    registry.set_learning(store, config.base_identifier, "all")
    experience(store)
    before = store.path.read_bytes()
    offline = snapshot(config, ReadOnlyStore(store.root))
    assert store.path.read_bytes() == before
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(config, store)),
        base_url="http://localhost",
    ) as client:
        live = (await client.get("/api/state")).json()
    assert live["models"][0]["learning"] == offline["models"][0]["learning"]
    assert live["models"][0]["id"] == config.public_model
    assert live["models"][0]["learning"]["conversations"] == 2


def test_missing_store_is_read_only_and_empty(tmp_path):
    root = tmp_path / "absent"
    summary = summaries(ReadOnlyStore(root), [model()], [], [])["first"]
    assert summary["conversations"] == 0
    assert summary["last"] is None
    assert summary["topics"] == []
    assert not root.exists()
    duplicate = copy.deepcopy(summary)
    assert summary_lines(model() | {"learning": duplicate}) == [
        "learning   no conversations through songe yet"
    ]


def test_success_does_not_resolve_another_tool_or_branch(store):
    context = [{"role": "user", "content": "Check tools"}]
    error = {"role": "tool", "tool_call_id": "one", "content": '{"exit_code":1}'}
    other = call("two", "git status")
    other["tool_calls"][0]["function"]["name"] = "read"
    success = {"role": "tool", "tool_call_id": "two", "content": '{"exit_code":0}'}
    for identifier, action, tail in [
        ("a", call("one", "pytest"), [error, other, success]),
        ("b", other, [success]),
    ]:
        source = recorded(store, context, action, 100, 110)
        store.sync_steps(
            [
                {
                    "id": identifier,
                    "conversation": "shared-root",
                    "source_id": source,
                    "context": context,
                    "action": action,
                    "hindsight": tail,
                }
            ]
        )
    summary = summaries(store, [model()], [], [])["first"]
    assert summary["signals"]["tool_error_fixed"] == 0


def test_legacy_timestamps_and_invalid_tool_calls_remain_readable(tmp_path):
    path = tmp_path / "experience.sqlite3"
    step = {
        "id": "invalid",
        "source_id": 1,
        "context": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Check pytest"}, {"text": None}],
            }
        ],
        "action": {"role": "assistant", "tool_calls": [None, {"function": "invalid"}]},
    }
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA + SCHEMA_V2)
        db.execute(
            "INSERT INTO exchanges VALUES(1,120,'/v1/chat/completions','{}','{}',200,1,'first')"
        )
        db.execute(
            "INSERT INTO steps(id,conversation,split,payload) VALUES('invalid','root','train',?)",
            (json.dumps(step),),
        )
    before = path.read_bytes()
    summary = summaries(
        ReadOnlyStore(tmp_path), [model(level="new", cutoff=100)], [], []
    )["first"]
    assert summary["conversations"] == 1
    assert summary["sources"][0]["first_line"] == "Check pytest"
    assert path.read_bytes() == before


def test_summary_follows_a_write_in_the_same_file_time_tick(store):
    # Linux file times tick every few milliseconds, so a write right after a
    # refresh can leave them unchanged (#11). Pin them to show it.
    import os

    files = [store.path, store.path.with_name(store.path.name + "-wal")]
    assert summaries(store, [model()], [], [])["first"]["conversations"] == 0
    times = [os.stat(f) for f in files]
    experience(store)
    for file, stat in zip(files, times, strict=True):
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert summaries(store, [model()], [], [])["first"]["conversations"] > 0
