"""One recorded copy contract for Python and the browser, without runtimes."""

from pathlib import Path

from songe.panels import panel_rows
from songe.wording import (
    status_copy,
    status_lines,
)

ROOT = Path(__file__).resolve().parents[1]


def test_next_action_and_old_service_presentation():
    assert status_lines(
        {"reason": "proxy idle 8m/10m", "next_action": "idle 8m/10m → calibrate Qwen"}
    ) == [
        ("now", "Ready to measure · Starts when idle · 8m of 10m"),
        (
            "next",
            "Measure training needs for Qwen · when idle",
        ),
    ]


def test_idle_wait_names_the_next_job_and_never_counts_past_the_threshold():

    waiting = {"now": "waiting · proxy idle 15s/10s", "next": "idle → train m"}
    assert status_copy({"activity": waiting}) == ("Ready to train", "Starting now")
    measuring = {"now": "waiting · proxy idle 4s/10s", "next": "idle → calibrate m"}
    assert status_copy({"activity": measuring}) == (
        "Ready to measure",
        "Starts when idle · 4s of 10s",
    )


def test_disk_wait_names_the_volume_and_the_numbers():

    now = "waiting · Needs 49.5 GB free on /Volumes/disk; 31.3 GB available"
    assert status_copy({"activity": {"now": now}}) == (
        "Waiting for disk space",
        "49.5 GB needed on /Volumes/disk · 31.3 GB free · retries automatically",
    )


def test_now_names_the_job_during_the_pause_between_slices():
    """Live: the row said Training while Now showed an empty dash."""
    model = {"id": "qwen", "learning_level": "all", "pipeline": {}}
    snapshot = {
        "activity": {
            "now": "waiting · retry delay after previous work slice",
            "next": "train qwen",
            "model": "qwen",
        }
    }
    rows = dict(panel_rows(model, snapshot=snapshot))
    assert rows["now"] == "Training · short pause between runs"


def test_paused_says_the_command_that_resumes():
    """The old words named a "Learning" switch that no surface has any more."""
    data = {
        "paused": True,
        "activity": {"now": "training Qwen", "next": "resume → train Qwen3.5-4B"},
    }
    assert status_lines(data) == [
        ("now", "Paused · Run songe resume to continue"),
        ("next", "Run songe resume, then train Qwen3.5-4B"),
    ]
