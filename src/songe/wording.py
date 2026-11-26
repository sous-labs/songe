"""Plain English shared by the terminal interfaces and the page (ui/wording.json)."""

import math
import re
from datetime import datetime

TRAINING_SETTINGS = "Training settings"
IGNORED_MODEL_SETTINGS = "Ignored per-model settings"
# Measuring ran out of room on the graphics card (live #135: vLLM held most of
# the card). Worded without "memory": those reasons read as waits that clear
# by themselves, and this one needs the user.
DEVICE_FULL = (
    "the graphics card is too full to train beside your model server · stop the server "
    "while the Dream runs, give it a smaller share of the card, or use a second card"
)


def readable(value):
    text = str(value or "")
    replacements = [
        (r"retry delay after previous work slice", "short pause after the last run"),
        (r"run next work slice for", "Continue learning from"),
        (r"proxy requests in flight", "conversations in progress"),
        (r"no proxy requests", "no conversations in progress"),
        (r"proxy idle", "quiet for"),
        (r"starting worker for", "Preparing to learn from"),
        (r"resume scheduler", "continue learning"),
        (r"scheduler state", "activity"),
        (r"scheduler starting", "Songe starting"),
        (r"\ban idle gap", "a quiet moment"),
        (r"idle gap", "quiet moment"),
        (r"calibrating", "Measuring training needs for"),
        (r"calibrate ", "measure training needs for "),
        (r"calibration", "training setup"),
        (r"select enabled model", "choose a model with learning on"),
        (
            r"select model · learning enabled \+ learnable experience",
            "Choose New or All for a model, then pick a Songe model in your agent.",
        ),
        (r"learnable experience", "conversations to learn from"),
        (r"held-out", "evaluation"),
        (r"fusing", "Preparing the update for"),
        (r"swapping", "Applying the update to"),
        (r"scoring", "Rewinding"),
        (r"swap ", "apply the update to "),
        (r"swapped", "updated"),
        (r"checkpoint", "progress"),
        (r"current work completes", "when this work finishes"),
        (r"checks pass \+", "after checks pass and"),
        (r"needs more data", "after more conversations"),
        (r"needs data", "after your first conversations"),
        (r"\bresume → ", "run songe resume → "),
        (r" → ", ", then "),
        (r"\bserve (?=\S)", "use "),
        (r"step (\d+)/(\d+)", r"Step \1 of \2"),
        (r"~([^·]+) left", r"About \1 left"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def capitalize(text):
    return text[:1].upper() + text[1:]


def collecting_detail(detail):
    """Training data comes from your conversations; counts are have/need."""
    count = re.fullmatch(
        r"(\d+) of (\d+)(?: (needed to check updates|new examples))?( · some too long to train)?",
        detail,
    )
    if not count:
        return detail or USE_SONGE
    have, need, kind, unfit = count.groups()
    noun = {
        None: "examples",
        "needed to check updates": "conversations",
        "new examples": "new examples",
    }[kind]
    return f"{have}/{need} {noun}" + (unfit or "")


USE_SONGE = "Pick a Songe model in your agent"
MEASURE_WAIT = re.compile(
    r"measuring training size|calibration required|not measured yet"
)


def measuring_now(activity):
    """Measuring only while it runs; a recorded measurement is waiting."""
    return activity.get("phase") == "calibrate" and not str(
        activity.get("now") or ""
    ).startswith("waiting")


def status_copy(data):
    """One vocabulary for every surface: a short state, then one short detail.

    Collecting → Ready to train → Measuring/Training → Testing → Update ready
    → Updated, with Waiting for …, Paused and Needs attention around them."""
    activity = data.get("activity") or {}
    raw = activity.get("now") or "Waiting for activity"
    reason = re.sub(r"^waiting · ", "", raw)
    restoring = data.get("restoring")
    if restoring:
        # Only the newest version is kept whole; an older one is rebuilt.
        return "Restoring", " · ".join(
            filter(
                None,
                [
                    restoring.get("model"),
                    version_name(restoring.get("version")),
                    f"{round((restoring.get('fraction') or 0) * 100)}%",
                ],
            )
        )
    if data.get("paused") and not reason.startswith("pausing"):
        return "Paused", "Run songe resume to continue"
    if reason.startswith("pausing · "):
        # Progress is saved first, so the work picks up where it stopped.
        return "Pausing", capitalize(reason[10:]) + " · saving progress"
    if MEASURE_WAIT.match(reason):
        done = re.search(r"(\d+) of ~?(\d+) tokens", reason)
        tokens = f"{done[1]} of {done[2]} tokens" if done else ""
        if measuring_now(activity):
            return "Measuring", " · ".join(
                filter(
                    None,
                    [
                        "What this machine can train",
                        tokens,
                        "longer contexts"
                        if "longer contexts" in reason
                        else "first time only",
                    ],
                )
            )
        return "Ready to measure", capitalize(
            " · ".join(filter(None, [tokens, "starts when idle"]))
        )
    if disk := re.search(
        r"Needs ([\d.]+) GB free on (.+?); ([\d.]+) GB available", reason
    ):
        return "Waiting for disk space", (
            f"{disk[1]} GB needed on {disk[2]} · {disk[3]} GB free"
            " · retries automatically"
        )
    if unsupported := re.match(r"unsupported · (.+)", reason):
        # Next says "Choose another model"; Now names the model once.
        return "Not supported yet", unsupported[1]
    if reason.startswith("preparing examples"):
        return "Preparing training data", re.sub(
            r"(\d+) examples waiting to learn",
            r"\1 to prepare",
            reason.partition(" · ")[2],
        )
    if reason.startswith("examples exceed training size"):
        return "Collecting", "Some examples too long to train"
    if "service stopped" in reason:
        return "Songe is stopped", "Run songe serve"
    if re.search(r"no models|uninitialized", reason):
        return "Connect a model", "Run songe init"
    if "learning off for all models" in reason:
        return "Learning off", "Turn on a model below"
    if reason.startswith("held for memory · "):
        # An automatic run the memory guard stopped: when, why and when it
        # tries again (status.memory_hold), all of it.
        return "Waiting for memory", capitalize(reason.removeprefix("held for memory · "))
    if "memory" in reason and raw.startswith("waiting"):
        measurement = re.search(r"\d[\d.]* GB free(?:, needs [\d.]+ GB)?", reason)
        detail = measurement[0] + " · " if measurement else ""
        return "Waiting for memory", detail + "retries automatically"
    if re.search(r"battery|power", reason) and raw.startswith("waiting"):
        charge = re.search(r"battery \d+%", reason)
        return "Waiting for power", (
            capitalize(charge[0]) if charge else "Battery low"
        ) + " · plug in to continue"
    if re.search(r"temperature|thermal|cool", reason) and raw.startswith("waiting"):
        return "Waiting to cool down", "Retries automatically"
    if "collecting conversations" in reason or "collecting test moments" in reason:
        return "Collecting", collecting_detail(reason.partition(" · ")[2])
    if re.search(
        r"no new conversations|learnable experience|new learnable|no new learnable",
        reason,
    ):
        numbers = [
            f"{count}/{total} {'test' if kind == 'held-out' else 'training'} examples"
            for kind, count, total in re.findall(
                r"(train|held-out|experience) (\d+)/(\d+)", reason
            )
        ]
        return "Collecting", " · ".join(numbers) if numbers else USE_SONGE
    if "retry delay after previous work slice" in reason:
        # A pause between slices of one job is still that job, not a wait.
        job = "Measuring" if "calibrate" in (activity.get("next") or "") else "Training"
        return job, "Short pause between runs"
    if retry := re.search(r"error · retry in (.+)", reason):
        return "Retrying after an error", f"Retries in {retry[1]} · run songe doctor"
    if "proxy idle" in reason:
        ready = (
            "Ready to measure"
            if activity.get("next_step") == "measure"
            or "calibrate" in (activity.get("next") or "")
            else "Ready to train"
        )
        quiet = reason.replace("proxy idle ", "", 1)
        waited = re.search(r"(\d+)(\w+)/(\d+)(\w+)", quiet)
        seconds = {"s": 1, "m": 60, "h": 3600}
        if waited and int(waited[1]) * seconds.get(waited[2], 1) >= int(
            waited[3]
        ) * seconds.get(waited[4], 1):
            # "Starting now" only once the scheduler has decided to start;
            # until its next check, quiet long enough is not yet a start.
            return ready, "Starting now" if activity.get(
                "starting", True
            ) else "Starts when idle"
        return ready, "Starts when idle · " + quiet.replace("/", " of ", 1)
    if "proxy requests in flight" in reason:
        if tools := re.search(r"waiting up to (\d+) min for the agent's tool results", reason):
            return "In use", f"The agent is running tools · learning waits up to {tools[1]} min"
        return "In use", "Continues after this conversation"
    if "checks passed" in reason:
        return "Update ready", "Applies when idle"
    if "the model server" in reason:
        # LM Studio (or another server) is closed: nothing is wrong with Songe.
        return "Model server not running", "Start it · retries automatically"
    # "serving retained": the same failure as older services word it.
    if re.search(r"learning failed · current version kept|serving retained", reason):
        return "Retrying after an error", "Current version kept · run songe doctor"
    if re.search(r"failed|doctor|unavailable", reason, re.I) and raw.startswith(
        "waiting"
    ):
        return "Needs attention", readable(reason) + " · run songe doctor"
    title, *parts = readable(raw).split(" · ")
    return capitalize(title), " · ".join(parts)


def next_copy(data):
    """The Next line: one action and what it waits for."""
    raw_next = (data.get("activity") or {}).get(
        "next"
    ) or "Waiting for the next update."
    idle = re.fullmatch(r"idle (\S+)/(\S+) → (.+)", raw_next)
    if "songe serve" in raw_next:
        text = "Run songe serve in your terminal."
    elif "songe init" in raw_next:
        text = "Run songe init in your terminal."
    elif idle:
        text = f"{readable(idle[3])} · when idle"
    else:
        text = readable(raw_next)
    return capitalize(text)


def status_lines(data):
    """Adapt offline snapshots before applying the dashboard's presentation."""
    from .status import lines

    activity = data.get("activity") or dict(lines(data)) | {
        "phase": (data.get("training") or {}).get("phase")
    }
    data = data | {"activity": activity}
    title, detail = status_copy(data)
    return [
        ("now", " · ".join(filter(None, (title, detail)))),
        ("next", next_copy(data)),
    ]


def measured(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def plural(count, word):
    if count != 1:
        word = (
            re.sub(r"([^aeiou])y$", r"\1ies", word)
            if re.search(r"[^aeiou]y$", word)
            else word + ("es" if word.endswith("s") else "s")
        )
    return f"{count} {word}"


def duration(seconds):
    if not measured(seconds):
        return ""
    value = max(0, math.floor(seconds + 0.5))
    if value < 60:
        return f"{value}s"
    if value < 3600:
        return f"{value // 60}m" + (f" {value % 60}s" if value % 60 else "")
    return f"{value // 3600}h" + (
        f" {value % 3600 // 60}m" if value % 3600 >= 60 else ""
    )


def stamp(timestamp):
    """Fixed English, as in the dashboard's en-US date formatter."""
    value = datetime.fromtimestamp(timestamp)
    month = (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )[value.month - 1]
    return f"{month} {value.day}, {value.hour % 12 or 12}:{value.minute:02d} {'AM' if value.hour < 12 else 'PM'}"


def comparison(last=None):
    last = last or {}
    absolute = measured(last.get("repeat_errors_delta"))
    value = last.get(
        "repeat_errors_delta" if absolute else "relative_repeat_errors_change"
    )
    if not measured(value):
        return None
    if value == 0:
        return "No change in repeat errors"
    raw = abs(value) * 100
    amount = "<0.1" if raw < 0.1 else f"{raw:.1f}".removesuffix(".0")
    if absolute:
        unit = "percentage point" if amount == "1" else "percentage points"
        return f"Repeat-error rate {'down' if value < 0 else 'up'} {amount} {unit}"
    return f"{amount}% {'fewer' if value < 0 else 'more'} repeat errors"


def model_level(model):
    return model.get("learning_level") or ("new" if model.get("enabled") else "off")


GGUF_MARK = "from a GGUF file"


def gguf_untrainable(architecture):
    """A GGUF whose architecture nothing here can turn into trainable weights."""
    return (
        f"Songe cannot train {architecture} {GGUF_MARK} · "
        "it keeps teaching it with notes"
    )


def model_problem(model):
    for item in [model, *(model.get("files") or [])]:
        reason = item.get("reason") or ""
        if item.get("support") == "unsupported" or item.get("status") == "unsupported":
            if GGUF_MARK in reason:
                return capitalize(reason) + "."  # says itself what is left
            if re.search("embedding", reason, re.I):
                return (
                    "This model is for embeddings · Choose a chat model for learning."
                )
            if re.search(r"weights|not accessible", reason, re.I):
                return "Local model files are unavailable · Download this model on this machine."
            if re.search("too large", reason, re.I):
                return "This model is too large to train here · Choose a smaller model."
            if re.search(r"adapter|format", reason, re.I):
                return "This format cannot be trained directly · Choose the original model."
            return (
                readable(reason) or "Learning is unavailable for this model"
            ) + " · Choose another model or run songe doctor."
        if DEVICE_FULL in reason:
            return capitalize(DEVICE_FULL) + "."  # it does not clear by itself
        if re.search(r"memory|free \d|needs \d.*GB", reason, re.I):
            return None  # resolves automatically; not a fix for the user
        if re.search(r"fail|doctor|error", reason, re.I):
            return readable(reason) + " · Run songe doctor for help."
    return None


DATA_NOUNS = {"train": "examples", "test": "conversations", "new": "new examples"}
SIGNAL_SOURCES = {
    "tool_error_fixed": "fixed tool errors",
    "user_correction": "your corrections",
    "retry_success": "successful retries",
    "loop": "repeated attempts",
    "static_invalid": "invalid answers",
    "tool_error": "tool errors",
    "exec_failure": "failed commands",
    "later_success": "later successes",
    "retry": "retries",
    "approval": "your approvals",
}


def next_text(upcoming):
    """What a model's next update waits for, from the pipeline's next stage."""
    stage = (upcoming or {}).get("stage")
    if stage == "ready":
        return "update when idle"
    if stage in DATA_NOUNS:
        more = max(0, upcoming["need"] - upcoming["have"])
        when = "update" if stage == "new" else "first update"
        noun = DATA_NOUNS[stage]
        return f"{when} after {more} more {noun[:-1] if more == 1 else noun}"
    return ""


def base_summary(model, data):
    last = (data.get("wake") or {}).get("last")
    result = comparison(last)
    conversations = (data.get("day") or {}).get("conversations") or 0
    lessons = (data.get("rewind") or {}).get("lessons") or 0
    evidence = " · ".join(
        filter(
            None,
            [
                plural(conversations, "conversation") if conversations > 0 else "",
                plural(lessons, "lesson") + " found" if lessons > 0 else "",
            ],
        )
    )
    issue = model_problem(model)
    if model.get("support") == "unsupported" or model.get("status") == "unsupported":
        detail = "Embedding model" if re.search(
            "embedding", model.get("reason") or "", re.I
        ) else re.sub(
            r"^Architecture$",
            "",
            re.sub(r" not supported yet$", "", readable(model.get("reason"))),
        )
        # A reason that only repeats the title ("unsupported") is left out.
        if detail.lower() in {"unsupported", "not supported yet"}:
            detail = ""
        return "Not supported yet", " · ".join(
            filter(None, (detail, may_not_load(model)))
        )
    if model_level(model) == "off" and not result and not conversations:
        return "", ""
    waiting = any(
        re.search(r"memory|free \d|needs \d.*GB", str(item.get("reason") or ""), re.I)
        for item in [model, *(model.get("files") or [])]
    )
    if result:
        return (
            result,
            "needs attention · open for the fix"
            if issue and not waiting
            else evidence or "Measured during everyday use.",
        )
    if waiting:
        return "Waiting for memory", "retries automatically"
    if issue:
        return "Needs attention", "open for the fix"
    if last and last.get("swapped") is True:
        return "Updated", "passed checks"
    if last is not None:
        return "Kept previous version", "update did not pass checks"
    upcoming = data.get("next") or {}
    if upcoming.get("stage") in DATA_NOUNS:
        noun = DATA_NOUNS[upcoming["stage"]]
        return "Collecting", f"{upcoming['have']}/{upcoming['need']} {noun}"
    if upcoming.get("stage") == "ready":
        return "Ready to train", "starts when idle"
    if lessons > 0 or conversations > 0:
        return "Collecting", evidence
    return "No conversations yet", USE_SONGE[:1].lower() + USE_SONGE[1:]


def model_summary(model, data, snapshot=None):
    from .status import model_training

    summary = base_summary(model, data)
    snapshot = snapshot or {}
    activity = snapshot.get("activity") or snapshot.get("training") or {}
    if not snapshot.get("activity") and activity and not activity.get("model"):
        activity = activity | {
            "model": snapshot.get("training_model") or snapshot.get("model")
        }
    # A pause between slices of one job is still that job.
    pause = "retry delay after previous work slice" in str(activity.get("now") or "")
    phase = activity.get("phase") or (
        ("calibrate" if "calibrate" in str(activity.get("next") or "") else "train")
        if pause
        else None
    )
    working = (
        phase
        and (pause or not str(activity.get("now") or "").lower().startswith("waiting"))
        and not snapshot.get("paused")
        and snapshot.get("service_running") is not False
        and model_training(model, activity.get("model"))
    )
    # Current work replaces everything but a measured outcome of an update.
    if working and not comparison((data.get("wake") or {}).get("last")):
        return PHASE_LABELS.get(phase, "Learning now"), ""
    # A measurement that waits for idle time, as in the top status.
    reason = re.sub(r"^waiting · ", "", str(activity.get("now") or ""))
    if (
        MEASURE_WAIT.match(reason)
        and not snapshot.get("paused")
        and model_level(model) != "off"
        and (
            model_training(model, activity.get("model"))
            or (not activity.get("model") and model.get("support") == "pending")
        )
        and summary[0] in {"No conversations yet", "Collecting", "Ready to train"}
    ):
        done = re.search(r"(\d+) of ~?(\d+) tokens", reason)
        return "Ready to measure", " · ".join(
            filter(
                None, [done and f"{done[1]} of {done[2]} tokens", "starts when idle"]
            )
        )
    # While the agent uses the model, that is the wait, not memory.
    if in_use(snapshot) and summary[0] == "Waiting for memory":
        return "In use", "continues after this conversation"
    # The wait the top status names belongs to the model it is for.
    if (
        activity.get("model")
        and model_training(model, activity.get("model"))
        and model_level(model) != "off"
        and not snapshot.get("paused")
    ):
        title, detail = status_copy(snapshot)
        if title in {"Waiting for memory", "Waiting for disk space"}:
            # The row has room for the need only; the top status has the rest.
            need = re.search(r"needs [\d.]+ GB", detail)
            return title, need[0] if need else ""
    return summary


PHASE_LABELS = {
    "rewind": "Rewind",
    "calibrate": "Measuring",
    "train": "Training",
    "distill": "Training",
    "select": "Rewind",
    "score": "Rewind",
    "gate": "Evaluation",
    "check": "Evaluation",
    "evaluate": "Evaluation",
    "fuse": "Preparing update",
    "swap": "Applying update",
    "record": "Starting",
}


def in_use(snapshot):
    activity = (snapshot or {}).get("activity") or {}
    return "proxy requests in flight" in str(activity.get("now") or "")


NOT_OFFERED = {
    "after init": "Not in your agents · run songe init again to add it",
    "unread": "Not in your agents · Songe can't read its files",
}


def not_offered(model):
    """Why a listed model is not in the agents' settings, or "" (#72)."""
    files = model.get("files") or [model]
    kinds = {f.get("offered") for f in files}
    if not kinds or None in kinds:
        return ""
    return NOT_OFFERED["after init" if "after init" in kinds else "unread"]


def may_not_load(model):
    """Songe offers every chat model the server lists; for one whose MLX
    files Songe's own MLX runtime cannot read, the server likely cannot
    load it either (#72, LM Studio: "Model type lfm3 not supported")."""
    server = next(
        (
            f["may_not_load"]
            for f in [model, *(model.get("files") or [])]
            if f.get("may_not_load")
        ),
        None,
    )
    return f"MLX version may not load in {server}" if server else ""


def model_status(model, data, snapshot=None):
    """One table cell; off models are quiet, unsupported models explain why,
    and so does a model the agents do not list."""
    title, detail = model_summary(model, data, snapshot)
    if title == "Not supported yet":
        return " · ".join(filter(None, (title, detail, not_offered(model))))
    if why := not_offered(model):
        return why
    return "" if model_level(model) == "off" else title


def version_name(identifier):
    """Users see v4, not the internal night-4."""
    return re.sub(r"^night-(\d+)$", r"v\1", str(identifier or ""))


def flow_state(value):
    """A scheduler state as one Flow line, in the dashboard's own units."""
    text = readable(value)
    # Lines written before this wording carried raw bytes.
    text = re.sub(
        r"(?i)^waiting for memory: smallest calibrated work does not fit; "
        r"RAM (\d+) bytes.*$",
        lambda m: f"waiting for memory · {int(m[1]) / 1e9:.1f} GB free",
        text,
    )
    for pattern, replacement in (
        (
            r"^collecting conversations · (\d+) of (\d+)$",
            r"collecting examples · \1 of \2",
        ),
        (r"^collecting conversations · (.*new examples.*)$", r"collecting · \1"),
        (r"^Will sleep when idle$", "starts when idle"),
        # One wait, one line: the scheduler names it two ways.
        (
            r"(?i)^(waiting for experience|no new conversations through songe yet)$",
            "waiting for conversations",
        ),
        (
            r"(?i)^Waiting for the model server: .*$",
            "model server not running · retries automatically",
        ),
        (
            r"(?i)^Sleep could not complete; current version preserved$",
            "learning failed · current version kept · retries automatically",
        ),
        (
            r"^[Nn]eeds ([\d.]+) GB free on (.+?); ([\d.]+) GB available$",
            r"waiting for disk space · \3 GB free on \2, needs \1 GB",
        ),
        (
            r"(?i)^Waiting for memory: smallest calibrated work does not fit · ",
            "waiting for memory to train · ",
        ),
        (
            r"(?i)^Waiting for memory: intact training samples deferred · ",
            "waiting for memory · long conversations · ",
        ),
        (
            r"(?i)^Waiting for memory: intact training samples deferred$",
            "waiting for memory · long conversations wait for more free memory",
        ),
        (
            r"^Waiting for (\d+) seconds without proxy traffic$",
            r"starts after \1s without agent requests",
        ),
    ):
        text = re.sub(pattern, replacement, text)
    # Flow lines read lowercase like the other stages; keep acronyms.
    if text[:2].istitle():
        text = text[0].lower() + text[1:]
    return text


def quick_message(unread, nothing_since=None):
    """The notice after pressing Quick Dream."""
    if nothing_since is not None:
        import time as _time

        since = (
            f" since your last Quick Dream at {_time.strftime('%H:%M', _time.localtime(nothing_since))}"
            if nothing_since else ""
        )
        return f"Nothing new{since} · talk to your agent first, then dream again"
    if unread is None:
        return "Quick Dream · already reading; its lessons follow when it finishes"
    if not unread:
        return "Quick Dream · updating its lessons from what it has read"
    return (
        f"Quick Dream started · reading the newest of {plural(unread, 'conversation')},"
        " then updating its lessons"
    )


def auto_dream_message(config, on):
    """The notice after turning auto-dream on or off: when Dreams happen."""
    if not on:
        return (
            "Auto-dream off · nothing starts on its own; press Dream to run one."
            " A Dream already under way finishes"
        )
    return (
        "Auto-dream on · a Dream starts on its own once your agent has been quiet"
        f" for {duration(config.idle_default_seconds)}, and pauses the moment you"
        " use it again. It waits while the battery is low, memory is short or"
        " the computer is hot"
    )


def init_learning_line(learner=None, skipped=None):
    """Init's Learning row when model size decided it. learner: the model
    auto-dream starts with (None: no model fits); skipped: (name, billions,
    limit) for the model that would have been chosen but is too big."""
    name, billions, limit = skipped
    why = f"{name} is about {max(1, round(billions))}B, " + (
        f"this computer trains up to about {limit}B"
        if limit >= 1
        else "this computer has too little memory to train"
    )
    if learner is None:
        return (
            f"auto-dream off · {why} · served and recorded; to try anyway:"
            f" songe models {name} on"
        )
    return f"auto-dream with {learner} · starts when idle · skipped {why}"


def missing_packages_wait(requirements):
    """A quantized model that cannot train until packages are installed."""
    quoted = " ".join(f'"{r}"' for r in requirements)
    return (f"this model's quantized weights need {plural(len(requirements), 'package')} to learn"
            f" · install where Songe runs: pip install {quoted} · then press Dream")


def dream_message(*, blocked=None, reading=0, unread="", wait=None):
    """One line answering a press of Dream: what started, and what training
    still waits on."""
    if blocked:
        return f"Not yet · {blocked}"
    parts = []
    if reading:
        parts.append(f"reading {plural(reading, 'reply')}")
    elif unread:
        parts.append(f"replies wait · {unread}")
    if wait is not None and reading:
        # Practice now, no training: said at the press, not discovered later.
        return f"Deep Dream · preparing practice now · training can't start yet: {wait}"
    if wait is None:
        parts.append("then training" if reading else "training")
    else:
        parts.append(f"training: {wait}")
    started = bool(reading) or wait is None
    return " · ".join(["Dream started" if started else "Nothing to read yet", *parts])


FLOW_HIDDEN = {"cycle", "done"}


# Recorded conversations and their size limit (recordings_limit_gb).
def gigabytes(size):
    return f"{size / 1e9:.1f} GB"


def recordings_deleted(count):
    return (
        f"deleted {plural(count, 'old recorded request')}"
        " · already learned from, over the size limit"
    )


def recordings_line(used, limit):
    """songe doctor: how much the recordings use, against their limit."""
    if used <= limit:
        return f"Recordings: {gigabytes(used)} of the {gigabytes(limit)} limit"
    return f"Recordings: {gigabytes(used)} · {recordings_note(used, limit)}"


def recordings_note(used, limit):
    """Status, when the recordings are over their limit: nothing is deleted
    before it has been learned from."""
    return (
        f"Recordings are over the {gigabytes(limit)} limit and waiting to be"
        " learned from; the oldest are deleted once they are"
    )


def flow_line(stage, text):
    """One plain line per event, or None when another line already says it.

    Cycle steps and "finished" repeat what the train, check and switch lines
    say; the stage name itself is never shown."""
    if stage in FLOW_HIDDEN:
        return None
    text = str(text or "")
    if stage == "state":
        return flow_state(text)
    if stage == "start":
        return (
            "measuring this Mac's memory"
            if text.startswith("measuring")
            else "learning started"
        )
    if stage == "record":
        found = re.match(r"(\d+) requests?", text)
        return f"recorded {plural(int(found[1]), 'request')}" if found else text
    if stage in ("review", "rewind"):
        found = re.match(r"(\d+)", text)
        if not found:
            return text
        count = int(found[1])
        return (
            f"read {plural(count, 'reply')}"
            if stage == "review"
            else f"found {plural(count, 'lesson')}"
        )
    if stage == "clean":
        found = re.match(r"kept (\d+) of (\d+)", text)
        return f"kept {found[1]} of {found[2]} new examples" if found else text
    if stage == "train":
        found = re.match(r"(\d+) of (\d+)", text)
        return f"training · {found[1]} of {found[2]} examples" if found else text
    if stage == "eval":
        return (
            "passed checks"
            if text.startswith("passed")
            else "did not pass checks · previous version kept"
        )
    if stage == "swap":
        found = re.search(r"night-(\d+)", text)
        return f"now using v{found[1]}" if found else "now using the original"
    if stage == "measure":
        found = re.match(r"(\d+) of (\d+) tokens fit", text)
        return (
            f"measured · {found[1]} of {found[2]} tokens fit"
            if found
            else "measuring smaller sizes"
        )
    return text


# Learn with a cloud model (songe.cloud). Raise the version when the
# substance changes: everyone who turned it on accepts the new text again.
CLOUD_DISCLAIMER_VERSION = 1
CLOUD_DISCLAIMER_TITLE = "Learn with a cloud model"


def cloud_disclaimer(terms_url=""):
    """What is sent, to whom and why, in plain words (CLI and page)."""
    terms = (
        f"The provider's terms apply ({terms_url})."
        if terms_url
        else "The provider's terms apply."
    )
    return [
        "This is optional and off unless you turn it on.",
        "Your agent keeps running on your own model; only the nightly Dream uses the"
        " cloud model.",
        "During a Dream, Songe sends the moments it learns from to the provider you"
        " choose: the conversation up to that moment, your feedback, and your agent's"
        " answers. The cloud model explains what went wrong and checks whether the new"
        " version does better.",
        "Before anything is sent, keys, tokens and passwords are removed and your home"
        " folder is shown as ~. Code and file contents in those moments are sent.",
        "Spending stops at a nightly budget you set. Every call is listed in the"
        " Dream's record.",
        terms + " Some providers restrict using their outputs to train other models;"
        " Songe trains your local model on what it learns. You decide whether that"
        " fits.",
        "Turning it off stops all calls at once.",
    ]


def money(usd):
    if usd is None:
        return "cost unknown"
    return f"${usd:.2f}" if usd >= 0.01 or usd == 0 else "<$0.01"


def context_size(tokens):
    if not tokens:
        return ""
    if tokens >= 1_000_000:
        return f"{round(tokens / 1_000_000, 1):g}M context"
    return f"{round(tokens / 1000):g}k context"


def cloud_line(cloud):
    """`songe status`: one line while cloud learning is on (or why Songe
    turned it off), else None."""
    if not cloud or not cloud.get("enabled"):
        if cloud and cloud.get("off_reason"):
            return f"Cloud learning off · {cloud['off_reason']}"
        return None
    tonight = cloud.get("tonight") or {}
    spent = money(tonight.get("usd", 0.0)) if tonight.get("cost_known", True) else "cost unknown"
    return (
        f"Cloud learning on · {cloud.get('model')} via {cloud.get('provider_label')}"
        f" · {spent} of {money(cloud.get('budget_usd'))} tonight"
    )
