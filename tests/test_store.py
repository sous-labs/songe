import sys

import pytest

from songe.config import Config
from songe.log import Store


def version(name, night=0):
    return {
        "id": name,
        "identifier": name,
        "artifact": "/models/" + name,
        "adapter": None,
        "created": 1.0,
        "night": night,
    }


def test_durable_atomic_swap_and_restart(store):
    store.add_version(version("base"))
    store.swap("base", None)
    store.add_version(version("night-1", 1))
    store.swap("night-1", "base")
    with pytest.raises(RuntimeError):
        store.swap("base", "base")
    reopened = Store(store.root)
    assert reopened.get("current") == "night-1"
    assert reopened.get("previous") == "base"
    reopened.swap("base", "night-1")
    assert reopened.get("previous") == "night-1"
    with store.connect() as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
    assert store.root.stat().st_mode & 0o777 == 0o700


def test_forget_clears_log_pool_but_keeps_versions(store):
    store.record("/v1/chat/completions", {}, {})
    store.add_pool([{"id": "x", "surprise": 1, "context": []}])
    store.add_version(version("base"))
    store.swap("base", None)
    store.forget()
    assert store.exchanges() == []
    assert store.pool() == []
    assert store.version()["id"] == "base"


def test_future_schema_is_not_downgraded(store):
    with store.connect() as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(RuntimeError, match="newer"):
        Store(store.root)


def test_forget_discards_preexisting_inflight_captures(store):
    store.forget()
    assert store.record("/v1/chat/completions", {}, {}, started=1) is None
    assert not store.exchanges()
    store.record("/v1/chat/completions", {}, {})
    assert len(store.exchanges()) == 1


def test_config_roundtrip_and_validation(tmp_path, config):
    config.save(tmp_path)
    assert Config.load(tmp_path) == config
    config.upstream_url = "https://remote.example/v1"
    with pytest.raises(ValueError, match="local"):
        config.save(tmp_path)


def test_training_length_follows_serving_context(tmp_path):
    from songe.config import Config

    assert Config().max_sequence == Config().context_window
    assert Config(context_window=32768).max_sequence == 32768
    (tmp_path / "config.toml").write_text("max_sequence = 2048\n", encoding="utf-8")
    assert Config.load(tmp_path).max_sequence == Config.load(tmp_path).context_window


def test_unsettled_steps_join_their_part_and_settled_ones_keep_their_group(tmp_path):
    from songe.log import Store

    root = tmp_path / "data"
    store = Store(root)

    def step(name, session, part, split="pending"):
        return {
            "id": name,
            "conversation": session,
            "task": part,
            "split": split,
            "context": [{"role": "user", "content": name}],
            "tools": [],
            "action": {"role": "assistant", "content": "ok"},
            "hindsight": [{"role": "user", "content": "next"}],
            "source_id": 1,
        }

    # Recorded when a whole session was one conversation.
    store.sync_steps(
        [
            step("waiting-1", "session", "part-1"),
            step("waiting-2", "session", "part-2"),
            step("tested", "old-session", "part-9", "heldout"),
        ]
    )
    assert store.feedback_counts() == (3, 2)
    with store.connect() as db:
        db.execute("DELETE FROM state WHERE key='grouped_by_instruction'")
    store = Store(root)  # the next start regroups once
    with store.connect() as db:
        rows = dict(db.execute("SELECT id,conversation FROM steps"))
        payload = dict(
            db.execute("SELECT id,json_extract(payload,'$.conversation') FROM steps")
        )
    assert (
        rows
        == payload
        == {
            "waiting-1": "part-1",
            "waiting-2": "part-2",
            "tested": "old-session",  # what it was tested under stays
        }
    )
    assert store.feedback_counts() == (3, 3)
    # Seen again with today's grouping, a settled step still keeps its own.
    store.sync_steps([step("tested", "part-9", "part-9")])
    with store.connect() as db:
        assert db.execute(
            "SELECT conversation,split FROM steps WHERE id='tested'"
        ).fetchone()[:] == ("old-session", "heldout")


def test_tests_are_drawn_from_parts_with_more_than_one_step(store):
    def step(name, part):
        return {
            "id": name,
            "conversation": part,
            "split": "pending",
            "context": [{"role": "user", "content": name}],
            "tools": [],
            "action": {"role": "assistant", "content": "ok"},
            "hindsight": [{"role": "user", "content": "next"}],
            "source_id": 1,
        }

    # Two greetings and three pieces of real work.
    store.sync_steps(
        [step("hi-a", "a"), step("hi-b", "b")]
        + [step(f"{part}-{n}", part) for part in "xyz" for n in range(3)]
    )
    store.assign_splits(minimum=2)
    held = {s["conversation"] for s in store.active_steps("heldout")}
    assert len(held) == 2 and held <= {"x", "y", "z"}


def test_next_update_counts_examples_not_yet_set_aside(config, store):
    """Live: 61 examples in 9 parts read "update after 8 more new examples"
    because nothing had been set aside for tests yet."""
    from songe.pipeline import next_update

    config.min_train, config.min_heldout, config.min_new = 4, 2, 2
    config.save(store.root)

    def step(name, part):
        return {
            "id": name,
            "conversation": part,
            "split": "pending",
            "context": [{"role": "user", "content": name}],
            "tools": [],
            "action": {"role": "assistant", "content": "ok"},
            "hindsight": [{"role": "user", "content": "next"}],
            "source_id": 1,
        }

    store.sync_steps([step(f"{part}-{n}", part) for part in "xyz" for n in range(2)])
    assert store.experience_counts() == (0, 0, 0)
    assert next_update(store) == {"stage": "ready"}


def test_connections_are_reused_per_thread(store):
    with store.connect() as a:
        pass
    with store.connect() as b:
        pass
    assert a is b
    seen = []

    def other():
        with store.connect() as db:
            seen.append(db)

    import threading

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert seen[0] is not a


def test_nested_connect_does_not_commit_the_callers_transaction(store):
    with pytest.raises(RuntimeError):
        with store.connect() as outer:
            outer.execute("INSERT INTO state VALUES ('outer', '1')")
            with store.connect() as inner:
                assert inner is not outer
            raise RuntimeError("roll back")
    assert store.get("outer") is None


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot replace a database file that is open")
def test_a_replaced_database_file_is_reopened(store):
    with store.connect() as before:
        pass
    store.path.rename(store.path.with_suffix(".old"))
    for suffix in ("-wal", "-shm"):
        side = store.path.with_name(store.path.name + suffix)
        if side.exists():
            side.unlink()
    fresh = Store(store.root)
    with fresh.connect() as after:
        assert after is not before
        assert after.execute("SELECT COUNT(*) FROM state").fetchone()[0] >= 0


def test_change_mark_moves_with_every_commit(tmp_path):
    import subprocess

    from songe.log import change_mark, close_connections

    store = Store(tmp_path / "data")
    assert change_mark(tmp_path / "missing.sqlite3") is None
    first = change_mark(store.path)
    store.set("a", 1)
    second = change_mark(store.path)
    assert second != first
    assert change_mark(store.path) == second  # no commit, no change
    # Another process's commit counts too.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from songe.log import Store; "
            "Store(Path(sys.argv[1])).set('b', 2)",
            str(store.root),
        ],
        check=True,
    )
    third = change_mark(store.path)
    assert third != second
    # A reopened watcher never repeats an earlier connection's mark.
    close_connections()
    assert change_mark(store.path) not in {first, second, third}
