"""Dream's clean step, the review note and the OPSD config defaults (DESIGN 3.2-3.4)."""

import copy
import json
import re

import pytest

from songe import steps as steps_module
from songe.config import Config
from songe.sleep.clean import clean
from songe.sleep.prompts import paired_prompts
from songe.sleep.select import select
from songe.steps import message_key, reply_key
from tests.test_learning import Tokenizer


@pytest.fixture(autouse=True)
def reviews():
    """Reviews live in module-level dicts; restore them after each test."""
    saved = (dict(steps_module.REVIEWED), dict(steps_module.REVIEW_NOTES))
    yield
    for live, old in zip(
        (steps_module.REVIEWED, steps_module.REVIEW_NOTES), saved, strict=True
    ):
        live.clear()
        live.update(old)


def call(arguments, name="write"):
    return {
        "role": "assistant",
        "tool_calls": [{"id": "c", "function": {"name": name, "arguments": arguments}}],
    }


def make(id, conversation="c", source_id=1, action=None, hindsight=None):
    return {
        "id": id,
        "conversation": conversation,
        "source_id": source_id,
        "split": "train",
        "context": [{"role": "user", "content": "Write the file"}],
        "action": action or {"role": "assistant", "content": f"answer {id}"},
        "hindsight": (
            [{"role": "user", "content": "That is wrong"}]
            if hindsight is None
            else hindsight
        ),
    }


def tool(content):
    return [{"role": "tool", "tool_call_id": "c", "content": content}]


def ids(steps):
    return [s["id"] for s in steps]


def test_clean_drops_steps_without_signal():
    def error(kind):
        return json.dumps({"error": {"type": kind, "message": "stopped"}})

    reviewed = {"role": "user", "content": "Now   do the next task"}
    judged = {"role": "user", "content": "Also add tests"}
    corrected = {"role": "user", "content": "No, use the other file"}
    steps_module.REVIEWED.update(
        {message_key(reviewed): None, message_key(judged): None}
    )
    steps_module.REVIEWED[message_key(corrected)] = "correction"
    kept = [
        make("args", action=call('{"code": "x"}'), conversation="1"),
        make("timeout", hindsight=tool(error("timeout")), conversation="2"),
        make("text", hindsight=tool("not json"), conversation="3"),
        make("list", hindsight=tool("[1, 2]"), conversation="4"),
        make("issue", hindsight=[judged], conversation="5"),
        make("correction", hindsight=[corrected], conversation="6"),
        make("unreviewed", conversation="7"),
    ]
    steps_module.REVIEW_NOTES[reply_key(kept[4])] = {"issue": "Skipped the tests"}
    dropped = [
        make("none", action=call(None), conversation="a"),
        make("empty", action=call(""), conversation="b"),
        make("braces", action=call("{}"), conversation="c"),
        make("dict", action=call({}), conversation="d"),
        make("aborted", hindsight=tool(error("aborted")), conversation="e"),
        make(
            "transport", hindsight=tool(error("provider.transport")), conversation="f"
        ),
        make(
            "moved-on",
            hindsight=[{"role": "user", "content": "Now do the next task"}],
            conversation="g",
        ),
        make("ended", hindsight=[], conversation="h"),
    ]
    assert ids(clean(dropped + kept)) == ids(kept)


def test_clean_collapses_a_repeated_action_to_its_second_occurrence():
    same = call('{"code": "x"}')
    loop = [make(str(i), "loop", i, copy.deepcopy(same)) for i in (3, 1, 2)]
    other = make("other", "elsewhere", 1, copy.deepcopy(same))
    changed = make("changed", "loop", 4, call('{"code": "y"}'))
    single = make("single", "loop", 5, copy.deepcopy(same) | {"content": "note"})
    pair = [make(str(i), "pair", i, {"content": "same"}) for i in (6, 7)]
    original = copy.deepcopy(loop + [other, changed, single] + pair)
    result = clean(loop + [other, changed, single] + pair)
    # Sorted by conversation then source; the representative is the second
    # attempt, whose context already shows the first, and it counts the run.
    assert ids(result) == ["other", "2", "changed", "single", "7"]
    assert [s.get("repeated") for s in result] == [None, 3, None, None, 2]
    assert result[1] == loop[2] | {"repeated": 3}
    assert all("_second" not in s for s in result)
    # Cleaning reads the steps; it never rewrites them.
    assert loop + [other, changed, single] + pair == original


def test_select_trains_every_cleaned_step_without_the_backend(config):
    pool = [
        make("good", "1"),
        make("aborted", "2", hindsight=tool('{"error": {"type": "aborted"}}')),
        make("held", "3") | {"split": "heldout"},
        make("retired", "4") | {"retired_at": 1},
        make("ended", "5", hindsight=[]),
        make("first", "6", 1, {"content": "same"}),
        make("second", "6", 2, {"content": "same"}),
    ]
    selected, skipped = select(pool)
    assert ids(selected) == ["good", "second"] and skipped == 0
    assert selected[1]["repeated"] == 2


def test_a_practice_scenes_note_is_an_aside_the_teacher_does_not_answer(config):
    """A scene's hindsight is a note Songe wrote, not evidence that followed
    the step: the teacher gets it as an aside to follow and not to mention.
    Shown as quoted observations, the teacher answered the note itself."""
    note = "Note from an earlier, different task.\nLesson: Use uv, never pip."
    scene = make("practice:s#v1", hindsight=[{"role": "user", "content": note}]) | {"practice": True}
    student, teacher = (json.loads(bytes(p)) for p in paired_prompts(Tokenizer(), scene, config))
    assert student["messages"] == scene["context"]
    assert teacher["messages"][:-1] == scene["context"]
    aside = teacher["messages"][-1]
    assert aside["role"] == "user" and note in aside["content"]
    assert aside["content"].startswith("(Not part of the conversation.")
    assert "do not mention it." in aside["content"] and aside["content"].endswith("length.)")
    assert "observations" not in aside["content"]
    # A real step's hindsight is evidence and keeps its marked block.
    real = json.loads(bytes(paired_prompts(Tokenizer(), scene | {"practice": False}, config)[1]))
    assert real["messages"][-1]["content"].startswith("Hindsight — quoted observations")


def test_a_scene_with_the_teachers_answer_is_trained_along_it(config):
    """Trained on its own rollouts the student writes its own start; a
    teacher scoring those tokens then mostly agrees, and the place where the
    two part is one token among many. Along the teacher's answer that place
    is in the loss."""
    from dataclasses import replace
    from types import SimpleNamespace

    from songe.sleep.prompts import answer_tokens, lesson_tokens

    tokenizer, config = Tokenizer(), replace(config, forced_tokens=8)
    scene = make("practice:s#v1", hindsight=[{"role": "user", "content": "Lesson: say Yes, chef."}])
    backend = SimpleNamespace(
        tokenizer=tokenizer, config=config, sample=lambda prompt, seed=None: ([7, 8, 9], "own", None)
    )
    # No teacher's answer (a real step, an agent's scene, one from before): the rollout.
    assert lesson_tokens(backend, scene, [1, 2]) == ([7, 8, 9], False)
    assert lesson_tokens(backend, scene | {"teacher_answer": None}, [1, 2]) == ([7, 8, 9], False)
    # A complete answer: its text, then what the template writes after an
    # assistant message (here the quote that closes the content).
    complete = scene | {"teacher_answer": {"content": "Yes.", "finished": True}}
    assert lesson_tokens(backend, complete, [1, 2]) == (list(b'Yes."'), True)
    # Cut by the server's limit, or longer than is trained on: no end of turn.
    cut = scene | {"teacher_answer": {"content": "Yes.", "finished": False}}
    assert answer_tokens(tokenizer, cut, config) == list(b"Yes.")
    long = scene | {"teacher_answer": {"content": "Yes, chef. Wind.", "finished": True}}
    assert answer_tokens(tokenizer, long, config) == list(b"Yes, che")


def test_the_teachers_answer_opens_the_turn_as_the_model_does(config):
    """Inkling-2's template leaves the model's turn to the model: it samples
    <|message_model|> then <|content_thinking|> or <|content_text|>. The
    teacher's answer is trained after that opening, not straight after the
    user's message, where the model never writes text."""
    from dataclasses import replace

    from songe.sleep.prompts import answer_tokens

    class Opening:
        def apply_chat_template(self, messages, tools=None, tokenize=True, return_dict=False,
                                add_generation_prompt=True, **template):
            text = "".join(
                f"<model><text>{m['content']}</>" if m["role"] == "assistant" else f"<{m['role']}>{m['content']}</>"
                for m in messages
            )
            return list(text.encode())

        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    scene = make("practice:s#v1") | {"teacher_answer": {"content": "Yes.", "finished": True}}
    tokens = answer_tokens(Opening(), scene, replace(config, response_tokens=64))
    assert bytes(tokens) == b"<model><text>Yes.<"


def test_an_agents_practice_scene_asks_its_teacher_for_an_action_not_a_reply(config):
    """With tools in the request a scene often ends on a tool result and the
    right next step is a tool call: its aside closes with the line a real
    step's teacher gets. "Write the reply" stays with conversations without
    tools, where it was measured."""
    note = "Note from an earlier, different task.\nLesson: Run the tests before the commit."
    scene = make("practice:s#v1", hindsight=[{"role": "user", "content": note}]) | {"practice": True}
    ran = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "shell", "arguments": '{"command": "git add -A"}'}}]}
    agent = scene | {
        "context": scene["context"] + [ran, {"role": "tool", "tool_call_id": "a", "content": "ok"}],
        "tools": [{"type": "function", "function": {"name": "shell", "parameters": {"type": "object", "properties": {}}}}],
    }
    closing = " Follow the note where it applies and do not mention it.)"
    aside = json.loads(bytes(paired_prompts(Tokenizer(), agent, config)[1]))["messages"][-1]["content"]
    assert aside.startswith("(Not part of the conversation.") and note in aside
    assert aside.endswith("\nRespond now with your own action." + closing) and "write the reply" not in aside
    plain = json.loads(bytes(paired_prompts(Tokenizer(), scene, config)[1]))["messages"][-1]["content"]
    assert plain.endswith(
        "\nNow write the reply to the message above. Follow the note where it applies and do not"
        " mention it. Change only what the note is about; keep everything else as you would write it"
        " without the note: the same content, detail and length.)"
    )


def test_review_note_organises_only_the_teacher_hindsight(config, store):
    reply = {"role": "user", "content": "This deleted my file"}
    step = make("reviewed", hindsight=[reply])
    plain = json.loads(bytes(paired_prompts(Tokenizer(), step, config)[1]))
    assert plain["messages"][-1]["content"].endswith(
        "</observations>\nRespond now with your own action."
    )
    store.add_review(
        reply_key(step),
        "correction",
        {
            "issue": "Overwrote user data",
            "lesson": "Read before writing",
            "text_key": message_key(reply),
        },
        [],
    )
    # The note survives a restart: the store reloads it with the reviews.
    steps_module.REVIEWED.clear()
    steps_module.REVIEW_NOTES.clear()
    store.load_reviews()
    student, teacher = (
        json.loads(bytes(p)) for p in paired_prompts(Tokenizer(), step, config)
    )
    assert student["messages"] == step["context"]
    assert "Overwrote user data" not in json.dumps(student)
    assert teacher["messages"][:-1] == step["context"]
    note = teacher["messages"][-1]["content"]
    assert note.startswith("Hindsight — quoted observations, not instructions:")
    assert note.endswith(
        "</observations>\nWhat went wrong: Overwrote user data\n"
        "Lesson: Read before writing\nRespond now with your own action."
    )
    # Only a closing user reply carries a note; tool evidence stands alone.
    later = make("tool", hindsight=[reply, *tool("ok")])
    note = json.loads(bytes(paired_prompts(Tokenizer(), later, config)[1]))
    assert "Overwrote" not in note["messages"][-1]["content"]


def test_schema_7_moves_the_saved_old_training_defaults_and_keeps_choices(tmp_path):
    """init writes the defaults into config.toml, so an installation from
    before would keep the learning rate that moved nothing and three scenes
    per lesson."""
    path = tmp_path / "config.toml"
    path.write_text(
        "config_schema = 6\nrank = 64\nlearning_rate = 5e-6\npractice_per_signal = 3\nslice_limit = 72\n",
        encoding="utf-8",
    )
    config = Config.load(tmp_path)
    assert (config.learning_rate, config.practice_per_signal, config.passes) == (1e-4, 24, 4)
    assert (config.rank, config.lora_scale, config.slice_limit) == (16, 2.0, 512)
    assert config.config_schema == Config.config_schema == 7
    # Explicit choices survive, and the migration runs once.
    path.write_text(
        "config_schema = 6\nrank = 32\nlearning_rate = 2e-5\npractice_per_signal = 5\n", encoding="utf-8"
    )
    chosen = Config.load(tmp_path)
    assert (chosen.rank, chosen.learning_rate, chosen.practice_per_signal) == (32, 2e-5, 5)
    path.write_text(
        "config_schema = 7\nrank = 64\nlearning_rate = 5e-6\npractice_per_signal = 3\n", encoding="utf-8"
    )
    kept = Config.load(tmp_path)
    assert (kept.rank, kept.learning_rate, kept.practice_per_signal) == (64, 5e-6, 3)


def test_schema_5_moves_saved_old_defaults_to_opsd_and_keeps_choices(tmp_path):
    path = tmp_path / "config.toml"
    old = (
        'config_schema = 4\nrank = 16\nlearning_rate = 1e-5\nloss = "reverse_kl"\n'
        "dream_enabled = true\ncontext_recent_turns = 2\ncontext_summary_tokens = 128\n"
        "learnability_lower_quantile = 0.1\nlearnability_upper_quantile = 0.9\n"
        "divergence_history = 4096\n"
    )
    path.write_text(old, encoding="utf-8")
    config = Config.load(tmp_path)
    # The rank is back at 16 since schema 7 (measured, see config.py).
    assert (config.rank, config.lora_scale, config.learning_rate) == (16, 2.0, 1e-4)
    assert (config.loss, config.vocab_clip) == ("forward_kl", 1.0)
    assert config.config_schema == Config.config_schema
    assert Config() == config
    config.save(tmp_path)
    saved = path.read_text(encoding="utf-8")
    for retired in (
        "context_recent_turns",
        "context_summary_tokens",
        "learnability",
        "divergence_history",
        "dream",
    ):
        # Retired keys only: `learning = "dreams"` is a current setting's value.
        assert not re.search(rf"^{retired}\w* =", saved, re.M)
    # Explicit choices survive the migration.
    path.write_text(
        'config_schema = 4\nrank = 32\nlearning_rate = 2e-5\nloss = "jsd"\n'
        "dream_enabled = false\n",
        encoding="utf-8",
    )
    chosen = Config.load(tmp_path)
    assert (chosen.rank, chosen.learning_rate, chosen.loss) == (32, 2e-5, "jsd")
    # Migrations run once: on schema 5 the old values are explicit choices.
    path.write_text(
        old.replace("config_schema = 4", "config_schema = 5"), encoding="utf-8"
    )
    kept = Config.load(tmp_path)
    assert (kept.rank, kept.learning_rate, kept.loss) == (16, 1e-5, "reverse_kl")


def test_retired_library_dream_and_selection_keys_migrate(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "library_enabled = true\ndream_enabled = true\ndream_per_lesson = 4\n"
        "dream_global_ratio = 0.25\ndream_max_ratio = 1.0\ndream_adjustment = 1.5\n"
        "dream_tier1_reliability = 0.5\ndream_tier2_reliability = 0.75\n"
        "select_top = 40\nselect_random = 4\n",
        encoding="utf-8",
    )
    config = Config.load(tmp_path)
    assert config.slice_limit == 44
    config.save(tmp_path)
    saved = path.read_text(encoding="utf-8")
    # Retired keys only: `learning = "dreams"` is a current setting's value.
    assert not re.search(r"^(library|dream|select)_\w* =", saved, re.M)
    assert Config.load(tmp_path).slice_limit == 44


def test_flow_shows_events_and_raw_steps_with_the_clean_verdict(tmp_path):
    """The dashboard's Flow lists what happened; Raw data shows each recorded
    step as the learner sees it, with cleaning's keep or drop reason."""
    from songe.flow import flow
    from songe.log import Store

    store = Store(tmp_path)
    store.record(
        "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, {}
    )
    kept = make("a") | {"task": "", "tools": [], "observed_id": 0}
    dropped = make("b", action=call("{}")) | {"task": "", "tools": [], "observed_id": 0}
    with store.connect() as db:
        for step in (kept, dropped):
            store._put_step(db, step)
    store.add_review(
        reply_key(kept),
        "user_correction",
        {
            "reply": "correction",
            "issue": "wrong file",
            "text_key": message_key({"content": "That is wrong"}),
        },
        ["a"],
    )
    store.event("measure", "256 of 2048 tokens fit")
    data = flow(store)
    stages = [e["stage"] for e in data["events"]]
    assert stages[:3] == ["measure", "review", "request"]
    assert "correction · wrong file" in data["events"][1]["text"]
    by_id = {s["id"]: s for s in data["steps"]}
    assert by_id["a"]["kept"] and by_id["a"]["asked"] == "Write the file"
    assert by_id["a"]["review"] == "correction · wrong file"
    assert not by_id["b"]["kept"]
    assert by_id["b"]["dropped"] == "tool call without arguments"


def test_a_pooled_repeated_step_never_stops_recording(tmp_path):
    """clean() marks a repeated run; storing that step must not break every
    later reconstruction of the same conversation."""
    from songe.log import Store

    store = Store(tmp_path)
    path = "/v1/chat/completions"
    ls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "ls", "arguments": '{"p":1}'},
            }
        ],
    }
    result = {"role": "tool", "tool_call_id": "a", "content": "error: x"}

    def reply(message):
        return {"choices": [{"message": message}]}

    first = [{"role": "user", "content": "do it"}]
    store.record(path, {"messages": first}, reply(ls))
    second = first + [ls, result]
    store.record(path, {"messages": second}, reply(ls))
    third = second + [ls, result]
    store.record(
        path, {"messages": third}, reply({"role": "assistant", "content": "done"})
    )
    store.reconstruct()
    steps = [s | {"split": "train"} for s in store.active_steps() if s["hindsight"]]
    kept = clean(steps)
    assert any(s.get("repeated") for s in kept)
    store.add_pool(kept, scored=kept)
    fourth = third + [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
    ]
    store.record(
        path, {"messages": fourth}, reply({"role": "assistant", "content": "ok"})
    )
    assert store.reconstruct()[0] >= 1


def test_a_long_tool_dump_never_hides_the_users_correction(config):
    """Hindsight is budgeted per message; the final user reply always stays."""
    from dataclasses import replace

    from songe.sleep.prompts import budgeted

    config = replace(config, hindsight_tokens=120)
    hindsight = [
        {
            "role": "tool",
            "tool_call_id": "c",
            "content": "line\n" * 2000 + "Error: boom",
        },
        {"role": "user", "content": "No, use the other file instead."},
    ]
    kept = budgeted(Tokenizer(), hindsight, config)
    assert kept[-1]["content"] == "No, use the other file instead."
    assert len(kept[0]["content"].encode()) < 120 and kept[0]["content"].endswith(
        "boom"
    )
    assert hindsight[0]["content"].startswith("line")  # input untouched


def test_tool_evidence_survives_a_reply_that_moves_on():
    """The user's next message moving on does not erase what the tools said."""
    moved_on = {"role": "user", "content": "Now add a README"}
    steps_module.REVIEWED[message_key(moved_on)] = None
    with_tool = make(
        "a", action=call('{"p":1}'), hindsight=tool("Error: x") + [moved_on]
    )
    without = make("b", hindsight=[moved_on])
    assert ids(clean([with_tool, without])) == ["a"]


def test_review_notes_stay_with_their_reply_event_and_forget_clears_them(tmp_path):
    """The same words after different work are different reply events; a
    lesson read from one never travels to the other. Forget removes reviews."""
    from songe.log import Store

    store = Store(tmp_path)
    again = {"role": "user", "content": "Try again"}
    first = make("a", hindsight=[{"role": "tool", "content": "ok"}, again])
    other = make("b", hindsight=[{"role": "tool", "content": "404"}, again])
    store.add_review(
        reply_key(first),
        "user_correction",
        {"issue": "wrong branch", "text_key": message_key(again)},
        [],
    )
    assert reply_key(first) != reply_key(other)
    assert steps_module.REVIEW_NOTES.get(reply_key(other)) is None
    store.forget()
    assert not steps_module.REVIEWED and not steps_module.REVIEW_NOTES
    with store.connect() as db:
        assert not db.execute("SELECT * FROM reviews").fetchall()
        assert not db.execute("SELECT * FROM events").fetchall()
    # A review that began before forget is dropped, not written afterwards.
    store.add_review(reply_key(first), "user_correction", {}, [], since=0)
    with store.connect() as db:
        assert not db.execute("SELECT * FROM reviews").fetchall()


def test_flow_by_served_id_shows_the_work_stored_under_the_registry_key(config, store):
    """The dashboard asks for songe/<model>; the worker records cycles under
    the registry key. Flow once showed only scheduler lines, never training."""
    from songe.api import model_scope
    from songe.flow import flow
    from songe.models import registry

    registry.register(config, store, "publisher/Model-4bit")
    row = registry.find(store, "publisher/Model-4bit")
    served = row["config"]["public_model"]
    assert served != row["id"]
    store.scoped(row["id"]).event("swap", "now serving night-4")
    texts = [e["text"] for e in flow(model_scope(store, served))["events"]]
    assert "now using v4" in texts


def test_a_repeated_state_refreshes_its_time_instead_of_a_new_line(store, monkeypatch):
    import songe.log as log

    now = iter([100.0, 101.0, 102.0, 103.0, 200.0])
    monkeypatch.setattr(log.time, "time", lambda: next(now))
    for text in ("waiting for experience", "starts when idle"):
        store.event("state", text)
    store.event("cycle", "#4 accepted")
    store.event("swap", "now serving night-4")
    store.event("state", "waiting for experience")
    events = store.events()
    assert [e["text"] for e in events] == [
        "waiting for experience",
        "now serving night-4",
        "#4 accepted",
        "starts when idle",
    ]
    assert events[0]["created"] == 200.0


def test_one_flow_names_the_model_of_each_line(config, store):
    from songe.flow import flow
    from songe.models import registry

    registry.register(config, store, "publisher/Qwen3.5-0.8B-MLX-4bit")
    store.scoped("publisher/Qwen3.5-0.8B-MLX-4bit").event("swap", "now serving night-4")
    store.event("state", "starts when idle", model_id="")
    events = flow(store, every_model=True)["events"]
    assert [(e["model"], e["text"]) for e in events] == [
        ("", "starts when idle"),
        ("Qwen3.5-0.8B", "now using v4"),
    ]


def test_memory_wait_says_free_and_needed_in_gigabytes():
    from types import SimpleNamespace

    from songe.sleep.sizing import waiting_reason
    from songe.wording import flow_state, status_copy

    sample = SimpleNamespace(available=7_981_252_608, vram=[])
    reason = waiting_reason(sample, {}, 8_600_000_000)
    assert flow_state(reason) == "waiting for memory to train · 8.0 GB free, needs 8.6 GB"
    copy = status_copy({"activity": {"now": "waiting · " + reason}})
    assert copy == (
        "Waiting for memory",
        "8.0 GB free, needs 8.6 GB · retries automatically",
    )
    deferred = waiting_reason(
        sample, {}, 10_100_000_000, what="intact training samples deferred"
    )
    assert flow_state(deferred) == (
        "waiting for memory · long conversations · 8.0 GB free, needs 10.1 GB"
    )


def test_disk_wait_reads_like_the_memory_wait():
    from songe.wording import flow_state

    assert flow_state("Needs 37.7 GB free on /Volumes/disk; 17.2 GB available") == (
        "waiting for disk space · 17.2 GB free on /Volumes/disk, needs 37.7 GB"
    )


def test_a_state_with_new_numbers_updates_its_line(store):
    store.event("state", "waiting for memory · 10.7 GB free, needs 15.6 GB")
    store.event("state", "starts when idle")
    store.event("state", "waiting for memory · 10.9 GB free, needs 15.5 GB")
    assert [e["text"] for e in store.events()] == [
        "waiting for memory · 10.9 GB free, needs 15.5 GB",
        "starts when idle",
    ]


def test_a_state_updated_within_one_clock_tick_is_still_newest(store, monkeypatch):
    """Windows' clock (Python 3.12 and older) ticks every ~15.6 ms, so the
    refreshed line and the one written just before it had the same time and
    the refreshed one sorted second (seen on a Windows laptop; reported by
    Jack in a test on 11-24)."""
    monkeypatch.setattr("songe.log.time.time", lambda: 1000.0)
    test_a_state_with_new_numbers_updates_its_line(store)


def test_a_line_written_after_a_refreshed_state_in_the_same_tick_is_newest(store, monkeypatch):
    monkeypatch.setattr("songe.log.time.time", lambda: 1000.0)
    store.event("state", "measuring 1 of 4")
    store.event("state", "starts when idle")
    store.event("state", "measuring 2 of 4")
    store.event("dream", "Dream started")
    assert [e["text"] for e in store.events()][:2] == ["Dream started", "measuring 2 of 4"]


def test_older_flow_lines_read_in_todays_wording():
    from songe.wording import flow_state

    old = (
        "waiting for memory: smallest calibrated work does not fit; RAM 7927742464 "
        "bytes, pressure margin 2576980377 bytes, device memory shared with RAM"
    )
    assert flow_state(old) == "waiting for memory · 7.9 GB free"
    assert flow_state("sleep could not complete; current version preserved") == (
        "learning failed · current version kept · retries automatically"
    )
    assert flow_state("waiting for memory: intact training samples deferred") == (
        "waiting for memory · long conversations wait for more free memory"
    )


def test_flow_reads_as_plain_lines():
    from songe.wording import flow_line

    assert flow_line("cycle", "#4 distill") is None
    assert flow_line("done", "finished") is None
    assert flow_line("train", "17 of 17 samples") == "training · 17 of 17 examples"
    assert flow_line("eval", "passed · held-out change +0.051 · ok") == "passed checks"
    assert flow_line("eval", "kept previous version · x").startswith("did not pass")
    assert flow_line("swap", "now serving night-4") == "now using v4"
    assert flow_line("review", "1 replies") == "read 1 reply"
    assert flow_line("review", "3 replies") == "read 3 replies"
    assert flow_line("rewind", "2 lessons") == "found 2 lessons"
    assert flow_line("swap", "original model in use") == "now using the original"
    assert flow_line("start", "learning cycle in an idle gap") == "learning started"
    assert (
        flow_line("record", "1 request turned into training steps")
        == "recorded 1 request"
    )
    assert flow_line("stop", "memory is short · retries automatically").startswith(
        "memory"
    )


def test_schema_6_raises_saved_two_held_out_conversations_to_five(tmp_path):
    (tmp_path / "config.toml").write_text("config_schema = 5\nmin_heldout = 2\n")
    assert Config.load(tmp_path).min_heldout == 5
    (tmp_path / "config.toml").write_text("config_schema = 5\nmin_heldout = 8\n")
    assert Config.load(tmp_path).min_heldout == 8  # an explicit choice stays
