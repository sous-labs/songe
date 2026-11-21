"""Release check: one real Deep Dream on a rented GPU, with the package being released.

    SONGE_WHEEL=dist/songe-<version>-py3-none-any.whl uvx modal run .github/ci/release_dream.py

Runs before every release (release.yml), never on pull requests: it takes
minutes, not seconds. On a Modal GPU it installs the wheel the way an NVIDIA
user does (`pip install 'songe[cuda]'`), takes a small real model, seeds a
store with made-up conversations in which the user corrects the model, and
the practice scenes reading would write from them, and runs the Dream worker
once from start to finish: training, the check against held-out
conversations, and the decision.

It passes when the Dream finishes cleanly with a decision, whatever the
decision is: with a tiny model and little data, keeping the old version is a
normal outcome. It fails on a crash, a stall, a non-finite loss, or missing
files, which is what dependency updates and packaging mistakes break.

No user data is involved: every conversation is generated here.
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Apache-2.0, small, no remote code
GPU = "L4"
TIMEOUT_MINUTES = 40

if modal.is_local():
    wheel = Path(os.environ.get("SONGE_WHEEL", ""))
    if not wheel.is_file():
        raise SystemExit("set SONGE_WHEEL to the built wheel (uv build --wheel)")
    WHEEL_NAME = wheel.name
else:
    wheel, WHEEL_NAME = None, os.environ["SONGE_WHEEL_NAME"]

image = modal.Image.debian_slim(python_version="3.12").env(
    {"HF_HOME": "/root/hf", "SONGE_WHEEL_NAME": WHEEL_NAME}
)
if wheel is not None:
    image = image.add_local_file(wheel, f"/root/wheel/{WHEEL_NAME}", copy=True)
image = image.run_commands(f"pip install '/root/wheel/{WHEEL_NAME}[cuda]'")
app = modal.App("songe-release-dream", image=image)


TOPICS = [
    ("What is 17 + 25?", "It is 32.", "No, 17 + 25 is 42.",
     "Check arithmetic before answering; 17 + 25 = 42."),
    ("Which file holds the project's dependencies?", "setup.cfg.",
     "We use pyproject.toml here.", "This project keeps its dependencies in pyproject.toml."),
    ("Run the tests.", "Running `python test.py`.", "Use `pytest -q` in this repo.",
     "In this repository tests run with `pytest -q`."),
    ("Name the main branch.", "It is master.", "It's called main.",
     "The default branch is called main."),
    ("Format this file.", "Running black.", "We format with ruff format.",
     "Code here is formatted with `ruff format`, not black."),
]


def seed(store):
    """What reading and practice would leave after a few days of use.

    Recorded exchanges in which the user corrects the model, the training and
    held-out steps made from them, and kept practice scenes whose teacher
    note is the lesson. Reading and practice need a model server, so they are
    not part of this check; the Dream itself (train, check, decide) is.
    """
    import json

    steps = []
    for n in range(40):
        ask, wrong, fix, _ = TOPICS[n % len(TOPICS)]
        context = [{"role": "user", "content": f"{ask} (task {n})"}]
        first = {"role": "assistant", "content": wrong}
        exchange = store.record(
            "/v1/chat/completions",
            {"model": "songe/release-check", "messages": context, "tools": []},
            {"choices": [{"message": first}]},
            binding="base",
        )
        steps.append({
            "id": f"step{n}", "source_id": exchange, "conversation": f"c{n}",
            "split": "heldout" if n % 4 == 0 else "train",
            "context": context, "action": first,
            "hindsight": [{"role": "user", "content": fix}], "tools": [],
        })
    store.sync_steps(steps)
    with store.connect() as db:
        for n in range(24):
            ask, _, _, lesson = TOPICS[n % len(TOPICS)]
            scene = {
                "id": f"sig{n}#v1", "signal_id": f"sig{n}", "tools": [],
                "messages": [
                    {"role": "system", "content": "You are a coding agent in this repository."},
                    {"role": "user", "content": f"{ask} (scene {n})"},
                ],
                "teacher_note": lesson,
            }
            db.execute(
                'INSERT INTO practice(id,model_id,signal_id,payload,kept,"drop",created)'
                " VALUES (?,?,?,?,1,NULL,0)",
                (scene["id"], store.model_id, scene["signal_id"], json.dumps(scene)),
            )
    return {"steps": len(steps), "practice": 24}


@app.function(gpu=GPU, timeout=TIMEOUT_MINUTES * 60)
def dream():
    import math

    from huggingface_hub import snapshot_download

    import songe
    from songe.config import Config
    from songe.log import Store
    from songe.sleep import worker

    started = time.monotonic()
    path = snapshot_download(MODEL)
    root = Path("/tmp/songe-data")
    root.mkdir(parents=True, exist_ok=True)
    config = Config(
        model=path,
        public_model="songe/release-check",
        upstream="vllm",
        backend="torch",
        artifact_kind="peft-adapter",
        calibration_available=True,
        context_window=2048,
        max_sequence=1024,
        response_tokens=32,
        min_train=3,
        min_heldout=3,
        min_new=1,
    )
    store = Store(root)
    recorded = seed(store)
    result = worker.run(config, store, slice_seconds=TIMEOUT_MINUTES * 60)
    nights = store.nights()
    record = (nights[0].get("record") or {}) if nights else {}
    candidate = result.get("candidate") or {}
    checks = {
        "trained more than one step": (result.get("iterations") or 0) > 1,
        "finite loss": isinstance(result.get("loss"), (int, float))
        and math.isfinite(result["loss"]),
        "candidates checked against the version in use": bool(
            record.get("scale_evaluations")
        ),
        "a decision was made": bool(nights)
        and nights[0]["status"] in {"ready", "rejected", "accepted"},
        "slice closed": bool(store.slices()) and store.slices()[0]["ended"] is not None,
        # A rejected candidate is cleaned up; an adopted one must be on disk.
        "adopted version on disk": not result.get("accepted")
        or (Path(candidate.get("artifact", "/missing")) / "adapter_model.safetensors").is_file(),
    }
    return {
        "songe": songe.__version__,
        "model": MODEL,
        "recorded": recorded,
        "minutes": round((time.monotonic() - started) / 60, 1),
        "night": nights[0]["status"] if nights else None,
        "result": {k: result.get(k) for k in (
            "reason", "iterations", "samples_trained", "loss", "accepted",
            "heldout_change", "general_drift")},
        "checks": checks,
    }


@app.local_entrypoint()
def main():
    found = dream.remote()
    print(json.dumps(found, indent=1, default=str))
    failed = [name for name, ok in found["checks"].items() if not ok]
    if failed:
        print("release Dream check FAILED: " + ", ".join(failed))
        sys.exit(1)
    print(f"release Dream check passed in {found['minutes']} min · decision: {found['night']}")
