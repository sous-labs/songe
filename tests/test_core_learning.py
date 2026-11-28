import json
import sqlite3
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from songe.backends.objective import divergence
from songe.log import SCHEMA, Store
from songe.schedule import (
    base_fingerprint,
    idle_threshold,
    reason,
    rebuild_reason,
    slice_duration,
)
from songe.sleep.distill import Backend
from songe.sleep.gate import choose_scale
from songe.sleep.select import replay_pool
from songe.steps import Step, digest, extract, signal_for
from tests.test_learning import NumpyAPI
from tests.test_schedule import good
from tests.test_steps import row

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
]


def action(code, call_id):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "run", "arguments": json.dumps({"code": code})},
            }
        ],
    }


def test_hindsight_reaches_later_success_but_stops_at_next_task():
    context = [{"role": "user", "content": "Run the checks"}]
    wrong, right = action("bad", "1"), action("good", "2")
    failure = {"role": "tool", "tool_call_id": "1", "content": '{"exit_code": 1}'}
    success = {"role": "tool", "tool_call_id": "2", "content": '{"exit_code": 0}'}
    finished = {"role": "assistant", "content": "Done"}
    messages = context + [wrong, failure, right, success]
    steps = extract(
        [
            row(1, context, wrong, TOOLS),
            row(2, context + [wrong, failure], right, TOOLS),
            row(3, messages, finished, TOOLS),
            row(
                4,
                messages
                + [finished, {"role": "user", "content": "Write an unrelated poem"}],
                finished,
                TOOLS,
            ),
        ]
    )
    assert steps[0].signal == "later_success"
    assert steps[0].answer_hint == right
    assert success in steps[0].hindsight
    # The next reply is always evidence, and nothing after a new task is.
    assert steps[0].hindsight[-1]["content"] == "Write an unrelated poem"
    assert {s.split for s in steps} == {steps[0].split}
    assert steps[0].env_fingerprint == digest(TOOLS)


@pytest.mark.parametrize(
    "hindsight,expected",
    [
        ([{"role": "tool", "content": '{"exit_code": 2}'}], "exec_failure"),
        ([{"role": "tool", "content": '{"error": "missing"}'}], "tool_error"),
        ([{"role": "user", "content": "Actually use code"}], "user_correction"),
        ([{"role": "user", "content": "Try again"}], "retry"),
        ([{"role": "user", "content": "Thanks"}], "approval"),
    ],
)
def test_signal_taxonomy(hindsight, expected):
    step = Step("id", "c", "train", [], TOOLS, action("x", "1"), hindsight, 1)
    signal, strength, evidence, hint = signal_for(step)
    assert signal == expected
    assert strength > 0 and evidence


def test_static_invalid_and_repeated_action_loop():
    invalid = action("x", "1")
    invalid["tool_calls"][0]["function"]["arguments"] = "{}"
    step = Step("id", "c", "train", [], TOOLS, invalid, [], 1)
    assert signal_for(step)[0] == "static_invalid"
    step.action = action("x", "1")
    step.hindsight = [action("x", "2")]
    assert signal_for(step)[0] == "loop"


def test_v1_migration_preserves_log_pool_versions_and_is_idempotent(tmp_path):
    root = tmp_path / "v1"
    root.mkdir()
    with sqlite3.connect(root / "experience.sqlite3") as db:
        db.executescript(SCHEMA)
        db.execute("PRAGMA user_version=1")
        item = {
            "id": "old",
            "conversation": "c",
            "split": "train",
            "surprise": 3,
            "tools": TOOLS,
        }
        db.execute("INSERT INTO pool VALUES (?,?,?)", ("old", 3, json.dumps(item)))
        db.execute("INSERT INTO state VALUES ('current','\"base\"')")
    store = Store(root)
    assert store.pool() == [item]
    assert store.active_steps()[0]["env_fingerprint"] == digest(TOOLS)
    assert store.get("current") == "base"
    with store.connect() as db:
        db.execute("UPDATE steps SET signal='tool_error' WHERE id='old'")
    Store(root)
    with store.connect() as db:
        assert db.execute("SELECT signal FROM steps").fetchone()[0] == "tool_error"


def test_schema_changes_retire_old_experience_and_require_rebuild(store, config):
    old = {
        "id": "old",
        "conversation": "c",
        "split": "train",
        "tools": TOOLS,
        "surprise": 1,
    }
    store.add_pool([old])
    changed_tools = json.loads(json.dumps(TOOLS))
    changed_tools[0]["function"]["parameters"]["required"] = []
    new = old | {"id": "new", "tools": changed_tools}
    assert store.sync_steps([new]) == 1
    assert store.pool() == []
    assert rebuild_reason(config, store) == "many experiences retired"
    store.set("rebuilt_at", time.time() + 1)
    assert rebuild_reason(config, store) is None
    with store.connect() as db:
        retired = db.execute(
            "SELECT retired_at,retired_reason FROM steps WHERE id='old'"
        ).fetchone()
    assert retired[0] and retired[1] == "tool schema changed"
    # Observing old log history again must not resurrect an expired item.
    store.sync_steps([old, new])
    assert len(store.active_steps()) == 1


@pytest.mark.parametrize(
    "daytime_metrics", ["absent", None, {}, {"general_drift": 0.02}]
)
def test_scoreboard_uses_daytime_logs_and_ignores_nightly_metrics(
    store, daytime_metrics
):
    context = [{"role": "user", "content": "Check"}]
    first, retry, done = (
        action("bad", "1"),
        action("bad", "2"),
        {"role": "assistant", "content": "Stopped"},
    )
    failure = {"role": "tool", "content": '{"exit_code": 1}'}
    for source in [
        row(1, context, first, TOOLS),
        row(2, context + [first, failure], retry, TOOLS),
        row(3, context + [first, failure, retry, failure], done, TOOLS),
    ]:
        if daytime_metrics != "absent":
            source["response"]["metrics"] = daytime_metrics
        store.record(
            source["path"], source["request"], source["response"], binding="base"
        )
    store.night(None, status="accepted", record={"general_drift": 0.01})
    metrics = store.scoreboard()
    assert len(metrics) == 1
    assert metrics[0]["repeat_error_rate"] == 0.5
    assert metrics[0]["steps_per_task"] == 3
    assert metrics[0]["tool_call_valid_rate"] == 1
    assert metrics[0]["general_drift"] == (
        0.02 if isinstance(daytime_metrics, dict) and daytime_metrics else None
    )


def test_empirical_gap_survival_and_calibrated_slice(config):
    # Fixed short quiet time: work yields to requests and resumes from checkpoints.
    assert idle_threshold([], 30, config) == 10
    assert idle_threshold([10] * 100 + [100] * 20, 30, config) == 10
    rows = [
        {
            "sequence": config.max_sequence,
            "step_seconds": 1,
            "generation_tps": 32,
            "sample_max_tokens": 32,
        }
    ]
    assert slice_duration(config, {"rows": rows}, [100] * 20, 50) == 50
    # Live run: a Dream's measured step (long scenes) far above calibration.
    # The slice still fits one measured step instead of ending empty.
    assert slice_duration(config, {"rows": rows}, [100] * 20, 50, measured=170) == 170


def test_an_empty_slice_lengthens_the_next(config, store):
    """Live run: 82 s slices ended at their deadline before one 170 s step
    finished, again and again. Before a step is measured, an empty slice
    that hit its deadline makes the next one twice as long."""
    from songe.schedule import measured_step_seconds

    assert measured_step_seconds(store) == 0
    with store.connect() as db:
        db.execute(
            "INSERT INTO nights(created,status,phase,record,model_id) VALUES (0,'interrupted','distill','{}',?)",
            (store.model_id,),
        )
    slice_id = store.slice()
    with store.connect() as db:
        db.execute(
            "UPDATE slices SET started=100, ended=182, units_done=0, interrupted_by='slice deadline' WHERE id=?",
            (slice_id,),
        )
    assert measured_step_seconds(store) == 164
    # A later slice stopped by agent activity does not hide it.
    later = store.slice()
    with store.connect() as db:
        db.execute(
            "UPDATE slices SET started=200, ended=264, units_done=0, interrupted_by='agent activity' WHERE id=?",
            (later,),
        )
    assert measured_step_seconds(store) == 164
    with store.connect() as db:
        db.execute("UPDATE slices SET units_done=1 WHERE id=?", (slice_id,))
    assert measured_step_seconds(store) == 0
    config.calibration_available = True
    config.calibrated_peak_gb = 3
    assert "memory" in reason(
        config, replace(good(), total_gb=4, free_gb=1), enough=True, traffic_idle=600
    )
    assert (
        reason(
            config,
            replace(good(), total_gb=4, free_gb=1),
            enough=True,
            traffic_idle=600,
            reclaimable_bytes=2_000_000_000,
        )
        == "Will sleep when idle"
    )
    assert (
        reason(
            config,
            replace(good(), ac=None, thermal=None, idle_seconds=None),
            enough=True,
            traffic_idle=600,
        )
        == "Will sleep when idle"
    )


def test_all_deep_rebuild_triggers(config, store):
    assert rebuild_reason(config, store) is None
    store.set("trained_base", "different")
    assert rebuild_reason(config, store) == "base changed"
    store.set("trained_base", base_fingerprint(config))
    store.set("cumulative_drift", config.general_drift_limit)
    assert "drift" in rebuild_reason(config, store)
    store.forget()
    assert rebuild_reason(config, store) == "forget requested"


def test_unreduced_divergence_matches_mean_reference():
    s = np.log(np.array([[[0.8, 0.2], [0.3, 0.7]]]))
    t = np.log(np.array([[[0.2, 0.8], [0.4, 0.6]]]))
    for kind in ("forward_kl", "reverse_kl", "jsd"):
        per_token = divergence(NumpyAPI, s, t, kind, reduction="none")
        assert per_token.shape == (1, 2)
        assert per_token.mean() == pytest.approx(divergence(NumpyAPI, s, t, kind))


def test_replay_keeps_older_experiences(config):
    pool = [{"id": str(i), "source_id": i} for i in range(100)]
    replay = replay_pool(pool, 90, config.replay_fraction)
    assert sum(s["source_id"] > 90 for s in replay) == 9
    assert sum(s["source_id"] <= 90 for s in replay) == 3
    assert replay_pool(pool, 90, 0) == pool[91:]


def test_wake_tries_largest_first_and_discards_each_failure():
    before = [
        {
            "id": str(i),
            "conversation": str(i),
            "kl": 1,
            "drift": 0,
            "tool_valid": 1,
            "empty": 0,
            "repetition": 0,
        }
        for i in range(5)
    ]

    def evaluate(scale):
        return [
            r | {"kl": 0.5, "drift": 0.2 if scale in (0.5, 1) else 0.01} for r in before
        ]

    discarded = []
    scale, decision, evaluations = choose_scale(
        evaluate,
        before,
        [1, 0.5, 0.25, 0.75],
        min_heldout=5,
        drift_limit=0.05,
        discard=discarded.append,
    )
    # The first passing strength from the top is used; nothing after it runs.
    assert scale == 0.75 and decision.accepted
    assert list(evaluations) == ["1", "0.75"]
    assert discarded == [1]
    scale, decision, evaluations = choose_scale(
        lambda s: evaluate(0.5), before, [1, 0.5], min_heldout=5, drift_limit=0.05
    )
    assert scale is None and decision is None and list(evaluations) == ["1", "0.5"]


@pytest.mark.parametrize("moe", [False, True])
def test_wake_scales_effective_adapter_delta_not_individual_factors(moe):
    class API(NumpyAPI):
        cpu = None
        sqrt = staticmethod(np.sqrt)
        linalg = SimpleNamespace(
            qr=lambda x, **kw: np.linalg.qr(x),
            svd=lambda x, **kw: np.linalg.svd(x, full_matrices=False),
        )

    backend = Backend.__new__(Backend)
    backend.mx, backend.config = API, SimpleNamespace(rank=4)
    loaded = {}
    backend.model = SimpleNamespace(
        load_weights=lambda items, **kw: loaded.update(items)
    )
    rng = np.random.default_rng(1)
    shape_a, shape_b = ((2, 2, 4), (2, 5, 2)) if moe else ((4, 2), (2, 5))
    baseline = {
        "x.lora_a": rng.normal(size=shape_a).astype(np.float32),
        "x.lora_b": rng.normal(size=shape_b).astype(np.float32),
    }
    trained = {
        k: rng.normal(size=v.shape).astype(np.float32) for k, v in baseline.items()
    }

    def delta(weights):
        a, b = weights["x.lora_a"], weights["x.lora_b"]
        return b @ a if moe else a @ b

    backend.scale_delta(baseline, trained, 0.75)
    assert np.allclose(
        delta(loaded), 0.25 * delta(baseline) + 0.75 * delta(trained), atol=1e-5
    )


@pytest.mark.parametrize("moe", [False, True])
def test_scaled_narrow_layer_keeps_the_configured_rank(moe):
    # Qwen3.5 in_proj_a/b have 16 outputs; rank 64 scaling once produced rank 16
    # factors there, and mlx_lm.fuse rejected the adapter, so no swap happened.
    class API(NumpyAPI):
        cpu = None
        sqrt = staticmethod(np.sqrt)
        pad = staticmethod(np.pad)
        linalg = SimpleNamespace(
            qr=lambda x, **kw: np.linalg.qr(x),
            svd=lambda x, **kw: np.linalg.svd(x, full_matrices=False),
        )

    backend = Backend.__new__(Backend)
    backend.mx, backend.config = API, SimpleNamespace(rank=4)
    loaded = {}
    backend.model = SimpleNamespace(
        load_weights=lambda items, **kw: loaded.update(items)
    )
    rng = np.random.default_rng(2)
    shape_a, shape_b = ((2, 4, 6), (2, 2, 4)) if moe else ((6, 4), (4, 2))
    baseline = {
        "x.lora_a": rng.normal(size=shape_a).astype(np.float32),
        "x.lora_b": rng.normal(size=shape_b).astype(np.float32),
    }
    trained = {
        k: rng.normal(size=v.shape).astype(np.float32) for k, v in baseline.items()
    }

    def delta(weights):
        a, b = weights["x.lora_a"], weights["x.lora_b"]
        return b @ a if moe else a @ b

    backend.scale_delta(baseline, trained, 0.25)
    assert loaded["x.lora_a"].shape == shape_a
    assert loaded["x.lora_b"].shape == shape_b
    assert np.allclose(
        delta(loaded), 0.75 * delta(baseline) + 0.25 * delta(trained), atol=1e-5
    )


def test_one_unlucky_health_sample_does_not_veto_a_real_improvement():
    from songe.sleep.gate import decide

    def rows(kl, repetition):
        return [
            {
                "id": str(i),
                "conversation": f"c{i}",
                "kl": kl,
                "drift": 0.0,
                "empty": 0.0,
                "repetition": repetition(i),
                "tool_valid": 1.0,
            }
            for i in range(12)
        ]

    before = rows(1.0, lambda i: 0.0)
    noisy = decide(
        before,
        rows(0.5, lambda i: 1.0 if i == 3 else 0.0),
        min_heldout=8,
        drift_limit=1,
    )
    assert noisy.accepted
    broken = decide(before, rows(0.5, lambda i: 1.0), min_heldout=8, drift_limit=1)
    assert not broken.accepted and "Looping" in broken.reason


def test_unloading_the_model_waits_out_usual_pauses(config):
    from songe.schedule import park_wait

    assert park_wait([], config) == 30
    assert park_wait([5] * 50 + [40] * 50, config) == 40
    assert park_wait([500] * 50, config) == 30
    assert park_wait([200] * 50, config) == 120


def test_idle_wait_after_a_restart_is_the_one_the_last_start_needed(config, store):
    from types import SimpleNamespace

    from songe.schedule import ModelScheduler, Scheduler

    def shown(kind):
        return kind(config, store, None, SimpleNamespace()).idle_threshold_seconds

    assert shown(Scheduler) == shown(ModelScheduler) == config.idle_default_seconds
    store.set("unloads_model", True)
    assert shown(Scheduler) == shown(ModelScheduler) == 30


def test_unloading_is_judged_before_it_happens():
    from songe.memory import Snapshot
    from songe.platform import Accel
    from songe.schedule import after_unload

    before = Snapshot(24_000, 6_000, vram=(Accel("metal", "gpu", 24_000, 6_000),))
    freed = after_unload(before, 5_000)
    assert freed.available == 11_000
    assert freed.vram[0].memory_free == 11_000
    # Never more than the device has.
    assert after_unload(before, 50_000).vram[0].memory_free == 24_000


def test_stopping_an_owned_gpu_server_frees_what_it_holds_on_the_gpu():
    """Live (Qwen3.8-27B on an A100): Songe's own vLLM held 72 GB of the GPU
    but its process RSS is a few GB of host memory; counting that as what
    stopping it frees, training (76 GB) never fit and the Dream stopped."""
    from songe.memory import Snapshot
    from songe.platform import Accel
    from songe.schedule import after_unload

    GB = 10**9
    gpu = Accel("cuda", "A100", 80 * GB, 8 * GB, "gpu-0")
    other = Accel("cuda", "A100", 80 * GB, 70 * GB, "gpu-1")
    before = Snapshot(1000 * GB, 900 * GB, vram=(gpu, other))
    freed = after_unload(before, 3 * GB, {"gpu-0": 72 * GB})
    assert freed.available == 903 * GB  # host: the server's RSS
    assert freed.vram[0].memory_free == 80 * GB  # the GPU it held
    assert freed.vram[1].memory_free == 70 * GB  # a GPU it did not use
    # Unknown device figures: the host figure stands in (unified memory).
    assert after_unload(before, 3 * GB).vram[0].memory_free == 11 * GB


def test_an_owned_server_measures_the_gpu_memory_it_takes(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from songe.upstream import supervisor as sup

    GB = 10**9
    readings = iter([{"gpu-0": 80 * GB}, {"gpu-0": 8 * GB}])
    monkeypatch.setattr(sup, "device_free", lambda platform: next(readings))
    server = sup.LlamaServer.__new__(sup.LlamaServer)
    server.platform, server.device_held, server._free_before_start = None, None, None
    server._before_start()
    server.process = SimpleNamespace(returncode=None)
    server._ready()
    assert server.device_bytes() == {"gpu-0": 72 * GB}
    server.process = SimpleNamespace(returncode=0)  # exited: holds nothing
    assert server.device_bytes() is None

    from songe.schedule import device_reclaimable

    class Upstream:
        async def loaded_device_bytes(self):
            return {"gpu-0": 72 * GB}

    assert asyncio.run(device_reclaimable(Upstream())) == {"gpu-0": 72 * GB}
    assert asyncio.run(device_reclaimable(object())) is None
