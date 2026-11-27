# Songe v1 contracts

Interfaces every workstream codes against. If a contract is wrong, change it here first, in the same commit as the code.

## 0. Rules for all work

- **No magic numbers.** A threshold is either measured on the machine (calibration, live probes), derived from the user's own history, or a named config field whose default is justified in a comment and overridable. Never encode a model, chip, vendor or size by name.
- **Never execute tools or user code at night.** Verification is read-only (schema checks, parsing).
- **Every heavy child process runs under the memory guard** (§3). No exceptions for probes, calibration, fusing or conversion.
- **Restrained output.** Rich CLI: no boxes, panels or banners; one cyan accent, dim labels, green ✓ / red ✗ results, aligned facts. `NO_COLOR` and non-TTY output are plain, with no spinner or ANSI escapes. Interactive long steps use a spinner. Errors occupy one red line followed by one hint. Exact init diffs require `--show-diff`. UI: Sous Labs paper aesthetic (see `src/songe/ui/style.css`).
- Product name comes from `songe.NAME`. Paths come from `songe.platform` (§1).
- Tests never touch the real user's HOME, agent configs, services or models.

## 1. Platform layer — `songe/platform/`

```python
class Platform(Protocol):
    name: str                       # "darwin" | "linux" | "windows"
    def data_dir(self) -> Path        # macOS ~/Library/Application Support/songe, Linux $XDG_DATA_HOME/songe, Windows %LOCALAPPDATA%\songe
    def idle_seconds(self) -> float | None   # user input idle; None when unknowable (headless) → callers rely on traffic only
    def on_ac_power(self) -> bool | None     # None = desktop / unknown → treated as powered
    def thermal(self) -> int | None          # 0 nominal … 3 critical, None unknown
    def memory(self) -> Memory               # total, available (bytes), pressure: "normal"|"warn"|"critical"
    def accelerators(self) -> list[Accel]    # kind: "metal"|"cuda"|"rocm"|"xpu"|"cpu", name, memory_total, memory_free
    def keep_awake(self, pid: int) -> Handle | None   # caffeinate / systemd-inhibit / SetThreadExecutionState
    def service(self) -> Service             # install(argv, env) / uninstall() / status() — launchd, systemd --user, Windows Task Scheduler
def current() -> Platform
```

Pressure sources: macOS `kern.memorystatus_vm_pressure_level`; Linux PSI `/proc/pressure/memory` (`some avg10` relative to its own recent baseline) plus `MemAvailable`; Windows `GlobalMemoryStatusEx.dwMemoryLoad` and available bytes. For discrete GPUs also VRAM free (NVML / `torch.cuda.mem_get_info` / `rocm-smi`). "warn"/"critical" are mapped per OS in one function each, with the mapping documented.

## 2. Training backends — `songe/backends/`

```python
class Backend(Protocol):
    kind: str                        # "mlx" | "torch"
    def __init__(self, config, check: Callable[[], None], train: bool = True)
    tokenizer
    def base(self) -> ContextManager          # adapter off, eval mode (frozen teacher)
    def sample(self, prompt_ids, seed=None) -> tuple[list[int], str, Stats]
    def compare(self, student_ids, teacher_ids, y, kind) -> float
    def token_logprob(...)
    def train_step(self, step, optimizer, seed=None) -> tuple[float, Grads]
    def optimizer(self, schedule) / apply(grads)  # backend-specific update
    def save(self, path) / restore(self, path)    # adapter on disk
    def layer_deltas(self, baseline=None) -> list[float]  # per-cycle delta when supplied
    def kl_from(self, base, prompt, teacher_prompt, y) -> float  # candidate vs frozen original teacher
    def prepare(self) / peak_memory() / reset_peak_memory()    # calibration hooks; bytes
def choose(config, platform) -> type[Backend]   # metal → mlx; cuda/rocm/xpu → torch; cpu → torch (serve-only unless forced)
```

The MLX implementation is the current `sleep/distill.py`; `sleep/compat.py` stays MLX-only. The Torch implementation uses transformers + PEFT LoRA, the same divergence (`divergence()` math moved to a framework-free reference used by tests), bitsandbytes 4-bit when the base does not fit in accelerator memory at bf16, and flash-linear-attention kernels when importable.

## 3. Memory guard — `songe/memory.py`

`guard(process, platform, on_warn, on_critical)` polls `platform.memory()` (and VRAM for discrete GPUs) on a short interval derived from the process's measured allocation rate (bounded), calls `on_warn` (graceful interrupt) and `on_critical` (kill process group). `guarded_run(argv, timeout)` wraps synchronous children (probe, calibration, fuse, convert). Training never starts while pressure is not "normal".

## 4. Models and artifacts — `songe/models/`

**Auto-follow.** `songe init` binds agents to one local upstream without requiring
`--model`. Agent providers expose its discovered model ids. Requests keep the
agent's chosen model; only that model's learned serving identity is substituted.
`--model` can pre-enable a local base. A server whose local weight paths are not
available can still forward and record. Newly requested ids are registered on
first use; local discovery and guarded conversion/calibration happen while idle.
A `songe/` name not registered yet is looked up among the models found on this
machine, so a model pulled while Songe runs answers to the name Songe gives it;
a name no model has is forwarded as asked and never registered.
No architecture or size-name exclusion is allowed. Selecting `new` or `all` on an unsupported model requests a fresh measured check. Support is `pending` until
measured, then `trainable` or `unsupported` with the observed reason. CPU-only
serving policy and missing local weight paths are reported explicitly.

**Quick and Deep Dream.** Two Dreams, chosen from the Dream button's menu
(`{"action": "dream", "model": "…", "kind": "quick" | "deep"}`, default deep)
or by auto-dream:

- Quick Dream (5–10 min, preferences): reads the newest unread conversations
  within `quick_read_seconds` (the older backlog is left to Deep), then
  publishes the prompt lessons (`quick.py`): lessons read as applying to any
  task, and rules the user stated (a sentence of theirs that starts like a
  rule, used verbatim and kept first); other lessons with the evidence a note
  needs (see practice below) wait for Deep Dream.
  Reading writes each lesson twice: `lesson` (one or two sentences, the Deep
  Dream teacher's note) and `short` (one instruction of at most 15 words, the
  prompt's form); Quick Dream uses `short`, and trims `lesson` by text only for
  signals read before `short` existed.
  Published lessons keep their wording; near-duplicates as one, and so are
  notes reading found to be one wish said in other words (the place then
  keeps each wording, see `same_notes` below),
  most recurrent then most recent first; the served model then consolidates
  them in one call (merges same-meaning lessons, drops ones that fit only one
  kind of task, rewrites each as one short instruction; the candidates as they
  are if its answer is unusable), at most `prompt_lessons` and about 150
  tokens. Background work never loads a model: an automatic Quick Dream
  waits when the learner's model is not loaded. A pressed Quick Dream on
  Ollama or LM Studio loads it with the server's own load (Ollama: an
  empty-prompt `/api/generate`; LM Studio: its load) when the memory
  admission Songe uses for its own versions allows (sized by the weights;
  weights Songe cannot measure are not loaded: `load_unknown`; without the
  memory: `load_memory`; on other servers it waits as before: `no_model`).
  The load does not hold the lock the agent's requests wait on, and a second
  press meanwhile continues the first. While loaded for the press the model
  is listed only to that press's own work, so no other reading starts on it
  and learning waits. It is released once the Quick Dream is done (Ollama
  `keep_alive: 0`, confirmed by `/api/ps`; LM Studio unload), also when it
  fails, in its own task that waits until no request is in flight and no
  reader is on it (retried, never dropped), and checks again right before the
  unload and while waiting for it: a request admitted then stops the release
  with Songe's claim kept, and it waits again. Once the agent asks for that
  model, it is the user's and is left to the server's own idle timeout; a
  request for it while the press still loads it waits for that load instead
  of starting a second one. A model that was
  already loaded is the user's and is never released; Songe's own requests
  send no `keep_alive`, so the user's setting applies. The proxy
  adds them to the end of the system message of each new session; a session
  keeps the lessons it started with (`LessonPins`), and the recorded request
  stays the agent's own. Removed lessons (`remove_lesson`) never return. The
  block ends with a line of its own (`quick.EXCEPTIONS`): the request at hand
  wins over a lesson when it asks for an exception, this time only, and every
  other request follows the lessons.
- Quick Dream before/after: every new preference (stated rule or not; up to
  6 tried) is replayed at the moment it came from (the logged request at the
  step its signal points to; for a stated rule, the tool call before it that
  matches it best) by the served model twice — with the lessons from before
  this Quick Dream and with the new ones (temperature 0, 600 tokens, not
  through the proxy, not recorded). It is shown only if (1) the action
  differs (another tool call, command or file content; without a tool call,
  clearly different text — a reworded preamble is not a change) and (2) the
  served model, shown the two answers in an order that hides which is new,
  says the new one follows the preference. At most 3 are shown. Stored in the
  history entry as `examples` [{lesson, asked, feedback, said_by, before,
  after, changed, differs, verdict}]; the card shows the pair, or "No visible
  change at the moment it came from". Skipped when the learner's model is not
  loaded.
- Stated rules and inferred lessons are one kind of preference. "Stated"
  only means the user's own words: kept verbatim and ranked first. A stated
  rule points at the step it is about, so a Deep Dream practises it like any
  other lesson and, once adopted, it leaves the prompt.
- A preference the user states about themself ("I avoid dairy.", "我不吃辣")
  is kept the same way: their sentence, word for word, in the prompt at once.
  It counts wherever it stands, the opening message included. Rules find the
  candidates (`rules.preference_sentences`: first person, a word of taste,
  habit or need, or the words of a change of habit ("no longer", "these
  days", "I've stopped ...", "不再"), not a question, not a pick among things
  just offered); the
  served model is asked one question per candidate (`preference`) and only
  "about_me" is kept, not a reaction to the last answer, a line about the job,
  or a line of a text handed over to be worked on. It points at the step it is
  about, or at the answer it got when nothing came before.
- People change. After reading, each such preference is shown to the served
  model with the ones the user stated before it, oldest first (`replaces`:
  which of them does it replace?), and each one it names again on its own
  (`both_true`: can both be true now?). A sentence that cannot be true
  together with the newer one ("I prefer window seats." after "I've switched
  to aisle seats.") is retired (`signals.retired_at`): reading, the prompt and
  practice no longer see it, and it stays retired when its conversation is
  read again. The order conversations are read in does not matter; answers
  come from the reader's cache unless the list of earlier sentences changed.
- People repeat themselves. Then each note for the prompt is shown to the
  served model with the notes before it, oldest first (`same_note`: which of
  them could be the same wish?), and each one it names again with it alone,
  both ways round (`same_wish`: the model says each note again as an
  instruction, then "same" or "different"). A note that repeats an earlier
  one both times ("Celsius
  please, I don't think in Fahrenheit." after "In metric please.") is folded
  into it (`same_notes` in the learner's state: signal -> the first signal of
  that wish). Folded notes are one lesson and take one place in the prompt,
  which keeps each wording, the most recent first, as long as they fit in one
  lesson; a change wanted once counts in every conversation it came up in,
  however it was worded. No wording is chosen and no signal is retired, so a
  wrong answer costs a place or shares one and never drops a note.
- Every user message that follows an answer is judged once by the served
  model (`turn`, with thinking): a standing wish about how this user wants
  things, a change wanted for this answer only, new work, or a reaction. Only
  a standing wish becomes a note in the prompt at once, also when the split
  into parts read it as a new task. A change wanted once becomes a note when
  the same wish comes up in a second conversation; it is stored as a
  correction for that, also when the split read it as a new task. After a plain
  answer (no tool call), a turn the split read as saying something is wrong
  or missing while the turn question read new work or a question (a
  reminder: "Where's my ...?") waits the same way. Other new work and
  reactions do not become notes. The verdict also holds for a sentence shaped
  like a rule in that message: inside new work or a reaction it is not a
  note, and inside a change wanted once it waits for the second conversation.
  A preference stated about oneself keeps its own check. Messages that were
  not judged (the opening one, "continue") are handled as before.
  The wish and the lesson for such a message are the model's wording of what
  the user wrote, checked in code (`rules.voiced`): one that names nothing
  the user wrote is not used, and the user's own sentence, when the message
  is one short sentence, is the note or the lesson in its place.
- Practice needs the same evidence as a note (`quick.admitted`, one bar for
  both; training is harder to undo than a note). Where the user's turn was
  judged, Deep Dream writes scenes for a standing wish and for a rule of
  theirs at once, for a change wanted once only when the same wish has come
  up in a second conversation, and for nothing the user took out of the
  prompt. The bar is about evidence, not about what a lesson applies to: a
  lesson too narrow for the prompt (one kind of task), or one from the work
  itself (a tool's output), is practised as before. Scenes already written
  for a lesson that does not meet the bar are kept and not trained
  (`practice.pool.not_admitted`).
- Deep Dream (2–3 hours, skills): reads what is left, writes practice, trains,
  checks. The check (`sleep.gate.judged_reason`) adopts a candidate when it
  learned (with no notes, it is better than the version in use, also with no
  notes, on more held-out moments than it is worse, beyond the judge's noise:
  wins − losses > 1.64·√(wins + losses)) and it is not worse for the user (as
  served, each with its notes, ties pass and it is refused only when losses −
  wins > 1.64·√(wins + losses)); drift ≤ `general_drift_limit` (0.05) and the
  health checks stand. When one is adopted, a lesson's note leaves the prompt
  (`quick.absorb`) only if the new version follows the lesson without it: on
  six of the lesson's held-out practice scenes it answers as it would
  be served afterwards, with the other lessons' notes in the prompt and
  without this one, and each answer must do what the note asks as well as the
  teacher's stored answer (`sleep.judged.learned`, recorded as `learned` and
  `absorbed_signals`). A lesson not learned yet, or with fewer than six held-out scenes,
  keeps its note beside the adapter, and so does a lesson whose own
  conversation was judged worse with the candidate served without its note
  (`sleep.judged.kept_back`). So that a note can be tested, a note
  whose lesson has practice from two or more conversations keeps one of them
  out of training (`practice.pool.ensure_note_heldout`: the one with the
  fewest scenes, never one already trained); with practice from a single
  conversation six of its scenes are kept out, and at least half of them
  train. An undo lets
  absorbed notes come back.
- Reading happens only inside a Dream. In idle moments (auto-dream on): after
  `quick_quiet_seconds`, an automatic Quick Dream (the newest unread
  conversations for at most `quick_read_seconds`, then the prompt lessons; a
  backlog is worked through by successive Quick Dreams and by Deep Dream);
  after `deep_quiet_seconds` (until gap prediction decides it), or when Deep
  Dream was pressed, Deep Dream reads what is left, writes the prompt lessons
  of what it read as a Quick Dream would, and writes practice, then trains.
  So the version it trains is checked against the prompt the user will have,
  with the lessons of every conversation read. A Deep Dream under way resumes
  in any pause.
- The cycle section carries `lessons` ([{id, text, count, added_at}]), `quick`
  (the last Quick Dream: {at, added, lessons, version, consolidated}) and
  `waiting` ({total, lessons: [{text, count}]}: lessons not in the prompt and
  not trained yet, for a Deep Dream). `forget` clears the lessons.
- The learning page (states):
  - a Deep Dream running: the learning view (banner, stages, Pause, Cancel);
  - the newest Dream is a Deep Dream not seen yet: its report (outcome, what
    changed, how it went) until Done;
  - otherwise the dashboard: a newest Quick Dream not seen yet as a card on top
    (what it changed in the prompt, View all, Done); the Dream row (the last
    12 Quick and Deep Dreams, newest first, unseen ones labelled "new");
    the selected Dream's report below (the same report, no Done; an older Deep
    Dream's record loads from /api/cycles/<id>); "Now · N preferences in the
    prompt · M skills waiting for a Deep Dream · View";
  - a Quick Dream running: one line on top of the dashboard.
  View opens a read-only panel, "What it has learned": the preferences in the
  prompt and a collapsed "Waiting for a Deep Dream (N)". No editing for now:
  lessons are Songe's decision (`remove_lesson` stays in the API, unused).
- The cycle section carries `dreams` ([{id, kind: quick|deep, at, seen, …}]:
  deep: outcome, reason, version, improved; quick: lessons, changes {added,
  reworded: [{before, after}], dropped}) and `quick_running`. A Quick Dream
  joins the history (`quick_history`, ids `q-NNNN`) only when it changed the
  prompt. `{"action": "seen", "model", "id"}` marks a Dream seen
  (`seen_dreams`); `forget` clears both.

**Auto-dream (user-facing, replaces the Off / New / All choice).** What a
model learns from is Songe's decision, not the user's: new conversations
(learning level `new`, recorded after the model was first used through Songe),
with earlier practice replayed for retention (`replay_fraction`). The one
choice is auto-dream per model (`registry.auto_dream`, state key
`auto_dream`; `auto_dream` in each model summary): on, Dreams start by
themselves in idle gaps; off, nothing starts by itself (no reading, practice or
training on idle) but Dream still runs one now and a Dream already under way
finishes in later gaps. Default: on for a model used through Songe, off for a
library-only model (level `off`); turning it on makes such a model learn
(`new`). Control: `{"action": "auto_dream", "model": "…", "on": true|false}`
→ `{"ok", "auto_dream", "message"}`, the message saying when Dreams happen. CLI:
`songe models <model> on|off`. The levels below stay as the internal
mechanism (and on the legacy dashboard and TUI); `all` is not offered on the
learning page or the CLI.

`models/registry.py` owns each model's `learning_level`, `learning_cutoff`,
support/reason and binding / measured config. Levels are `off` (forward only,
never train), `new` (default: only exchanges recorded after that model's cutoff),
and `all` (also earlier shared experience from any model). `enabled` remains a
compatibility projection of `learning_level != 'off'`. Off forwards the original
model while still recording experience. Current, previous, original base, scoring cursor,
replay pool, calibration, drift, candidate generation and rebuild state belong
to one model. Changing levels invalidates only its candidates, replay selection
and scoring cursors. Entering `new` persists the current timestamp; setting
`new` again is idempotent and retains its cutoff. Entering `all` makes earlier
data available for fresh on-policy scoring. The cutoff is checked against the
source exchange's `recorded_at` timestamp, for training, held-out checks and
replay alike; `created` continues to measure request start for traffic history.
The first exchange for a newly registered model is eligible when recorded after
registration. Older records without `recorded_at` use `created`. Later hindsight
does not admit a pre-cutoff source. Original weight
paths and retained learned versions are preserved.

Existing single-model stores migrate in place: the original model becomes the
first `new` entry with a cutoff at migration. Existing on/off rows migrate
atomically: on → `new` with a cutoff at migration, off → `off`. Stored versions,
undo pointers and adapters stay intact; earlier replay is excluded from `new`,
and unfinished candidates are discarded. Migration runs once; inspection from
the CLI does not mutate an old database. The primary learner retains the old state keys;
additional learners use namespaced state keys. `versions`, `nights`, `slices`,
`pool`, and `scored_revisions` carry a model id (empty for the legacy learner);
pool and revision keys are `(step_id, model_id)`. These are additive migrations
to the v2 store. `Store.scoped(id)` shares the same database and exchange log.

Exchanges, reconstructed hindsight, conversation splits and environment
retirement are stored once. Each learner scores and samples the shared hindsight
with its own policy, replay pool and held-out gate. Selecting `all` makes
all active shared evidence available, irrespective of which model produced it;
`new` admits only post-cutoff source exchanges, also irrespective of producer.
Forget removes shared experience and requests a rebuild for every learner.


```python
@dataclass
class ModelRef:              # what the user already runs
    upstream: str            # "lmstudio" | "ollama" | "llamacpp" | "mlx" | "vllm"
    serving_id: str          # identifier/tag the agent uses today
    format: str              # "mlx" | "safetensors" | "gguf"
    path: Path               # folder or file
    architecture: str        # config model_type or GGUF general.architecture
def trainable_base(ref, backend_kind, platform) -> Path
    # mlx folder → as is; safetensors → as is (torch) or mlx_lm.convert (mlx);
    # gguf → transformers GgufConfig(dequantize=True) → bf16 safetensors cache
    #        (→ mlx_lm.convert for the MLX backend). Disk checked first.
```

Artifact kinds an upstream can serve: `mlx-fused`, `mlx-adapter`, `peft-adapter`, `gguf-full` (merged + quantized to the base's GGUF type with llama.cpp tools), `gguf-lora` (`convert_lora_to_gguf.py`). llama.cpp tools are fetched from the official ggml-org release for the platform, verified by checksum, cached under the data dir.

## 5. Upstreams — `songe/upstream/`

```python
class Upstream(Protocol):
    name: str
    url: str                                    # OpenAI-compatible base
    serves: tuple[str, ...]                     # artifact kinds, preferred first
    def discover(self) -> list[ModelRef]        # what the user has
    async def load(self, version) / unload(self, version) / ensure(self, version)
    async def park(self, on_snapshot=None) -> list[Parked]        # unload everything before training
    async def restore(self, parked, managed: set[str])   # restore only instances Songe requested, skip retained serving ids
    async def loaded_bytes(self) -> int         # memory reclaimable from Songe-owned work
```

| Upstream | Serves | Swap mechanism |
|---|---|---|
| LM Studio | `mlx-fused` (Mac), `gguf-full` | `lms load <key> --identifier`, key from `lms ls --json` by path |
| Ollama | `gguf-full` | `ollama create <name>:<version> -f Modelfile` (FROM file), API `/api/create` |
| llama.cpp server | `gguf-lora`, `gguf-full` | `POST /lora-adapters` scale switch; restart only for a new base |
| mlx-lm server | `mlx-adapter` | per-request `adapters` path |
| vLLM | `peft-adapter` | `/v1/load_lora_adapter`, `/v1/unload_lora_adapter` |

## 6. Scheduling — when Songe works

States: `awake` (serving, recording) → `settling` (working in an idle gap) → `ready` (candidate passed the gate) → swap at a safe moment → `awake`.

- **Model selection.** One guarded worker / worker lease across all models.
  Learners set to `new` or `all` with work are ordered by eligible pending learnable experience, then
  least recently trained, then model id for a stable tie. Interrupted checkpoints
  remain eligible after scoring commits. Unsupported and disabled models cannot
  start work. Calibration and conversion use the same idle, traffic, power,
  thermal and memory checks. Publication, cleanup and undo use that learner's
  current/previous pointers and affect only its serving identity.
- **Idle gap trigger.** Start a work slice when the current gap since the last request exceeds a threshold learned from this user's own inter-request gaps: the smallest gap length after which, historically, the gap continued long enough for one slice with high probability (empirical survival over recent gaps). Until enough history exists, use the configured default (10 minutes, the user's suggestion). Also require: user input idle when knowable, power (laptops), thermal not serious, memory pressure normal.
- **Slices.** Each slice does the next unit of work (record → rewind/score → train steps → gate → fuse) with checkpoints after every committed unit, and its length is sized from calibration so it ends within the expected remaining gap. Any request or user input interrupts within the guard interval; progress is never lost beyond the current unit.
- **Swap.** Only when no request is in flight and the gap has lasted at least the observed swap duration. Never mid-session. The proxy calls `scheduler.traffic()` and awaits `scheduler.wake()` before forwarding, so a parked upstream is restored before serving; this also serializes admission with publication.
- **Checkpoints.** Backends expose `train(pool, anchors, work, progress)`, `adapter_weights()` and `scale_delta(baseline, trained, scale)` for a finite pass with a durable sample cursor and effective-delta interpolation (rank compression, when required, precedes gate evaluation). The original base remains unchanged. Gate references and completed scale/fuse evaluations are durable work units. `Config` holds the named, overridable scheduling, memory guard, replay and learnability controls.
- **Deep rebuild** (from base) when: forget requested, base changed, cumulative drift near its limit, or many experiences retired.
- Night-time is no longer special; long gaps simply let more slices run.

## 7. Store additions — `songe/log.py` (schema v2, migrated)

- `steps`: id, conversation, split, signal (`later_success`|`tool_error`|`exec_failure`|`user_correction`|`loop`|`static_invalid`|`retry`|`approval`), strength, evidence, answer_hint, env_fingerprint (tool schema hash), retired_at, retired_reason.
- `scoreboard`: day, model_version, repeat_error_rate, steps_per_task, tool_call_valid_rate, general_drift — computed from daytime logs only.
- `slices`: started, ended, phase, units_done, interrupted_by.

### Repeated parts stored once (schema v3, `songe/parts.py`)

Every agent request resends its system prompt, tools and whole history, and
each step copies a request's history (context) and what followed (hindsight,
evidence). `exchanges.request` and `steps.payload` are stored as a skeleton
plus `refs` JSON `{key: id | [ids]}`: every item but the last of a message
list (`messages`; `context`, `hindsight`, `evidence`) and any other top-level
value of 1 KB or more is a row of `parts` (its exact `json.dumps` text, once
per SHA-256). The last list item stays in the skeleton, so SQL on it
(`$.hindsight[#-1].role`, compaction markers, non-empty checks) is unchanged.
Readers use `parts.request` / `parts.payload` / `parts.request_of` and get the
recorded value exactly; a row without `refs` is plain JSON (older rows, a
store read without upgrading it). `part_uses` counts referencing rows through
triggers; `parts.collect` deletes unused parts in the deleting transaction.
`steps.evidence` (a copy of the payload's evidence no reader used) is `[]`.

Upgrading adds the tables, columns and triggers in Store's schema
transaction and sets `user_version` 3 (an older Songe refuses the store). The
guarded experience child then rewrites older rows (`recordings.dedupe`) in
batches of 16, each one transaction with its cursor in `state`
(`parts_cursor_<table>`): a stop loses nothing and the next run resumes. Each
rewritten row is read back and compared first; one that differs stays plain.
In an idle gap with no worker, reading or request, the service gives freed
space back (`recordings.compact`): a store made before v3 is rewritten once
(`VACUUM`, switching to incremental auto-vacuum, only with room for two
copies), then space returns in 16 MiB steps. New stores start incremental. A
recording waiting on the write lock retries (up to a minute) instead of being
lost.

### Size limit on recordings (`recordings_limit_gb`, default 2)

In the same idle upkeep, when the store's pages in use exceed the limit,
`recordings.keep_under` deletes whole sessions that have been learned from,
the one whose newest request is oldest first, until it is under. Learned
from (`recordings.learned_sessions`): every learner whose window covers the
session (learning on, after its cutoff for `new`) has read it at its current
revision, and each lesson signal read from it is trained or absorbed, or has
practice written and is held out or kept no scene. At least one learner must
cover it, and a model with learning off that can still learn (not
unsupported) counts as needing every session. Never deleted: sessions used
in the last 24 hours (an agent may continue them), sessions with held-out
test steps or confirmed successes (drift set), the session holding the
newest exchange id, requests in no session, and anything while a night is
running, interrupted, ready or waiting. Exchange ids are never reused:
`exchange_id_high` in `state` (set by the cleanup and by forget) keeps new
ids above every deleted one. A forget after the sessions were chosen stops
the cleanup. One transaction per session deletes its
exchanges, steps, prefix links and per-step rows, and the parts nothing else
uses; its signals and read marks move to a `<session>#deleted-…` name listed
in `pruned_sessions`, so reading never takes them for a vanished session's and
practice, lessons and replay keep their own copies. Over the limit with
nothing learned from, nothing is deleted: `/api/state` carries
`recordings_note`, shown by status, and `songe doctor` prints a `Recordings:`
line either way.

### Rewind lessons (additive v2 table)

`lessons` stores `(id, model_id)` plus `group_key`, JSON `payload`,
`retired_at`, and `retired_reason`. The payload contains a one-sentence `text`,
ordinal evidence `strength` in `[0,1]`, real failure/success contrasts, tool
schemas, file types, and `evidence` links `{step_id, revision, conversation,
source_id, observed_id}`. Only eligible **training** evidence contributes to
lessons; a failure without a recorded successful alternative cannot invent a
correction. Related tool/schema, error-class and file-type experiences are
grouped. The local original base phrases a representative contrast with LoRA
disabled; every supporting evidence link is retained. No remote model is used.

The order is Day reconstruction → Rewind signals/lessons/select/prune →
Consolidate real replay + anchors → Wake. Lesson units and training cursors are
checkpointed in the existing owned work directory. All inference uses the same
guarded worker, per-slice sequence limits and cancellation checks as training. A
retired/revised source, changed schema, superseding lesson, or learning-window
change retires dependent lessons. Forget deletes their payloads and signal
summaries, invalidates work and requests the existing rebuild for every model.
Serving versions remain subject to the existing forget/rebuild contract.

With `learning = "dreams"`, lessons come from extracted signals instead
(additive tables, per model). Reading works on whole sessions (all exchanges
under one prefix-linking root), not the per-instruction parts used for
held-out splits: a correction arrives later in the same session.
`read_sessions` (session, model_id, revision = newest exchange id read,
read_at); `signals` (id, model_id, session, revision, payload: type, source,
responsible steps and the exchange behind each, lesson, instance, generality,
feedback text, confirmed/contradicted for successes); `reading_cache` (the
model's answers by question, so a stopped reading resumes without asking
again); `practice` (id, model_id, signal_id, payload: the scene, its teacher
note, its checks, kept or the drop code). A session that grows is read again
and its signals replaced. (`dreams` is not used as a table name: `forget`
drops a legacy table of that name.) Forget deletes all four.

`step_signals` stores one compact JSON signal-count record per real step. It is
updated during reconstruction/selection and streamed during migration of older
v2 stores. API metrics aggregate these summaries without loading conversation
contexts or performing model inference.

## 8. Local API (for GUI, TUI, CLI) — served by the proxy on loopback

`GET /api/state`, `GET /api/nights`, `GET /api/nights/{id}`, `GET /api/scoreboard`, `GET /api/settings`, `GET /api/cycles`, `GET /api/cycles/{id}` (draft, see Learning cycle), `POST /api/control` `{action: pause|resume|sleep_now|dream|undo|forget|model, ...}` (JSON + Origin check as today), `PATCH /api/settings` for the few user-facing settings (model, schedule default, upstream, `cloud`). The CLI and TUI use the same endpoints when the service runs, and the store directly when it does not.

Learn with a cloud model (`songe/cloud/`, optional, off by default): `settings.cloud` summarises it (enabled, provider, model, budget, the key as `{set, tail, source}` only, tonight's calls and cost). `PATCH /api/settings {cloud: {enabled, provider, base_url, model, budget_usd, api_key, accept_disclaimer}}`: `api_key` is write-only (it goes to the OS keychain, or a 0600 file without one); `enabled: true` needs the current disclaimer accepted and runs the probe first, and stays off with a 400 and the probe's plain result if it fails. `GET /api/cloud`: summary, disclaimer text and version, suggested models, the latest night's calls. `POST /api/cloud/test`: probe the saved settings. Every call is a row in `cloud_calls` (purpose, model, redacted size and preview, tokens, cost, ok/error); a full cycle record lists its Dream's calls as `cloud_calls`. Only the Dream may call it; the proxy's upstream stays local.

`POST /api/control` accepts `{action:"model", model:<upstream id>, level:"off"|"new"|"all"}`
(legacy `enabled:<bool>` maps true → `new`, false → `off`)
and `{action:"undo", model:<upstream id>}`. Omitting the model on undo is allowed
only when one learner exists (or for an unmigrated legacy store). An ambiguous
undo returns an explicit model-choice error. `/api/state` includes `models`
(id, learning_level, learning_cutoff, enabled, support, status, reason, eligible
pending, version, retained versions, can_undo, last_result, last_status,
last_update, last_trained), `idle_seconds`, `training_model`, and `training` (model,
phase, progress, data_progress, measured optimizer step / total_steps during
training). Each cycle names its model. Calibration reports
its phase and completed/planned measurements; training reports committed optimizer steps from its durable plan/checkpoint and measured ETA.

### Per-model pipeline metrics (current)

> Implemented today and read by the browser, CLI and TUI. Retired in one
> commit once every surface reads `cycle` (see Learning cycle).

`Store.scoped(model).pipeline()` supplies the recorded stage metrics below.
The response's `pipeline` maps **public model id → metrics**, preserving each
learner even when inventory formats are grouped. Each `models[]` row also has
its preferred learner's `pipeline`; individual `files[]` retain their own.
The `status.pipeline` map is authoritative for browser panels. Pipeline updates
belong to the existing `status` revision section, preserving the inventory and
learning revision domains; training progress changes only `status`. The
`versions` section retains its version lists and completed cycle records.
Browser, CLI and TUI panels render these recorded fields, retaining sparse
fallbacks for older services without pipeline metrics. Signal keys are preserved
as recorded (`tool_error_fixed`, `user_correction`, etc.), with concise display
labels applied only when rendering.

```json
{
  "day": {"conversations": 0, "steps": 0, "heldout": 0},
  "rewind": {"signals": {
    "tool_error_fixed": 0, "user_correction": 0, "retry_success": 0,
    "loop": 0, "static_invalid": 0, "tool_error": 0, "exec_failure": 0,
    "later_success": 0, "retry": 0, "approval": 0
  }, "lessons": 0},
  "consolidate": {"runs": 0, "step": 0, "total": 0,
                  "eta_seconds": null, "last_seconds": null},
  "wake": {"last": null},
  "versions": [{"id": "base", "current": true, "kind": "base"}]
}
```

- Day counts active real steps eligible under the learner's `off/new/all`
  policy; `heldout` counts steps, `conversations` distinct real conversations.
- Rewind signals count real steps exhibiting each signal (categories may
  overlap); fixed-error/retry-success counts require an observed corrected
  action. `lessons` counts active lessons with eligible, current evidence.
- Consolidate `runs` counts cycles that entered training, including resumable
  cycles once each. `step/total` are committed/planned optimizer updates from
  the latest cycle; ETA uses calibrated sample costs. `last_seconds` is measured
  cumulative training time across slices of the most recent training cycle.
  Unknown durations/ETAs are `null`, never fabricated zero measurements.
- Wake `last`, when present, is `{at, repeat_errors_delta, drift, swapped}`.
  `at` is Unix seconds at the completed real gate; `drift` is measured relative
  to the original base (or null if no measurement completed). `swapped` remains
  false while a passing candidate waits and becomes true only at the atomic
  serving-pointer commit. `repeat_errors_delta` is the new minus old mean
  **daytime** repeat-error rate and remains null until both serving versions
  have recorded observations. KL gain and synthetic pass rates never substitute
  for it.
- Versions lists retained ids, current-pointer equality, and the recorded
  artifact `kind` (legacy fallback: `base` or `adapter`). All progress, feedback,
  lesson ownership and serving pointers are scoped independently per learner.

Wake uses real held-out conversations and the existing fixed general anchors

### Learning cycle (per model)

> **Served (2026-11-16).** `src/songe/cycle.py` builds it from the store;
> `/api/state` has the `cycle` section and `/api/cycles`, `/api/cycles/<id>`
> are live. The replay (`scripts/replay/`) produces the same shape from
> recorded runs. It replaces the earlier per-model pipeline metrics (`day`,
> `rewind`, `consolidate`, `wake`), which stay in `status.pipeline` unchanged
> until the browser, CLI and TUI read `cycle`; they are then removed in one
> commit.
>
> As built: models are keyed by public model id (`songe/...`), like the rest
> of `/api/state`. A cycle is one night record (`c-<night id>`); before a night
> exists, reading or practice in progress shows as `current` with the next id.
> Per-stage `started_at`/`ended_at` are not recorded yet and are null, as is
> the practice stage's `total`. Each example also carries `gap` (before/after
> teacher-student gap); a cycle whose examples failed has `examples_error` and
> no examples (examples never decide adoption). `trained_practice` examples
> are evaluated but never counted in `improved`.

**Where each stage runs**

- `learning = "dreams" | "hindsight"` (config, default `dreams`) selects the
  training pool. `hindsight` is today's path (real steps, see Hindsight self-distillation);
  `dreams` is this pipeline. Surfaces show the same `cycle` for both; under
  `hindsight`, `read.signals` and `practice` are `null`.
- **Read** and **Practice** run in idle gaps with the model already loaded on
  the model server, like the reply review: never alongside a training child,
  never while an answer streams, and stopped the moment the user returns. They
  never load a model; with none loaded the stage waits with code
  `model_server`. Results are stored per model and per conversation revision,
  so each conversation is read once and a stopped stage resumes where it was.
- **Train**, **Check** and **Result** run in the guarded worker child, as today.

What every surface shows about learning: what Songe is doing, what each cycle
did, and before/after answers that show what changed. When the dashboard
opens, the last cycle's before/after examples are the first thing under the
status lines.

**Rules**

- **Codes and numbers, never sentences.** The service sends keys, codes and
  measured values. Each surface words them from one wording table
  (`src/songe/ui/wording.json`, read by Python and the browser alike). No
  surface parses or rewrites service text.
- **Recorded values only.** Unknown values are `null`, never `0`. A value that
  is an estimate says so with `"estimated": true` next to it.
- **Names.** User-facing words: **Dream** (the button that starts a cycle now),
  **Rewind**, **Practice**, **Training**, **Evaluation**, **Result**.
  In code, "dream" means a generated practice scene; it is never shown.
- **Signals.** Exactly six types, by source. From the user: `correction`,
  `retry`, `approval`. From a tool: `tool_error` (call rejected),
  `failed_result` (ran, result wrong), `outcome` (informative result).
  Messages the agent's harness writes in the user's place (for example an
  output-limit notice) are `failed_result` with source `tool`.
  Confirmed successes are not signals: they form the fixed **drift set**,
  which is never trained on.
- **Real tasks never enter training.** Only practice scenes and rehearsal are
  trained on.

**Where it is served**

- `/api/state` gains a fifth revision section, `cycle`:
  `{"models": {<public model id>: ModelLearning}}`. It changes only when its
  content changes, independently of `status`, `models`, `learning` and
  `versions`. It carries summaries and at most three examples per model, so
  polling stays small.
- `GET /api/cycles?model=<id>&limit=<n>` lists past cycles, newest first, in
  the summary shape.
- `GET /api/cycles/<cycle id>` returns one full cycle record: every signal,
  every practice line, every example, all losses and checks.
- `GET /api/nights/<night id>` includes `record.training_settings` for new
  runs. It captures the resolved training policy and each setting's source at
  cycle start. Resumed slices retain this snapshot and record their adaptive
  sizing in `record.slice_sizes`.

```jsonc
// ModelLearning
{
  "current": Cycle | null,        // running, waiting or paused
  "last": Cycle | null,           // most recent finished cycle
  "pending": {"conversations": 4, "signals": 9},  // recorded since `last` began
  "versions": [Version]
}

// Cycle
{
  "id": "c-0007",
  "model": "songe/lfm3-1.7b",
  "trigger": "dream" | "idle",
  "started_at": 1794870000.0, "ended_at": null,
  "stage": "read" | "practice" | "train" | "check" | "result",
  "state": "running" | "waiting" | "paused" | "done",
  "wait": Wait | null,
  "stages": [Stage],
  "read": Read, "practice": Practice, "train": Train,
  "checks": [Check], "result": Result | null
}

// Stage: surfaces draw stages from this list, never from a fixed list.
{
  "key": "read",                  // read | practice | train | check | result;
                                  // gap_check and rehearsal are reserved
  "state": "done" | "now" | "waiting" | "skipped" | "ahead",
  "progress": {"done": 12, "total": 38, "unit": "conversations"} | null,
                                  // units: conversations, scenes, calibration
                                  // (train, tokens measured), steps, checks;
                                  // check while running: measured
                                  // (conversations scored by the gate)
  "started_at": 1794870000.0, "ended_at": null,
  "eta_seconds": 540 | null
}

// Wait: what a stage needs before it can continue.
{
  "code": "agent_active" | "memory" | "power" | "thermal" | "disk" |
          "no_data" | "model_server" | "paused" | "unsupported" | "error",
  "have": 9.1, "need": 12.0, "unit": "GB"     // null when not a quantity
}

// Read
{
  "conversations": {"done": 12, "total": 38},
  "signals": {"total": 14, "by_type": {"correction": 3, "retry": 0,
    "approval": 1, "tool_error": 4, "failed_result": 6, "outcome": 0}},
  "lessons": 14,
  "successes": {"confirmed": 30, "contradicted": 2}
}

// Practice: one line per scene, so surfaces can list them plainly.
{
  "written": 137, "kept": 112, "dropped": 25,
  "dropped_by": {"<check code>": 25},   // built | decision | parses | not_copy | matters | changes
  "recent": [PracticeLine]        // newest 5; all in the full record
}
// PracticeLine
{"id": "…", "signal_id": "…", "request": "Implement slugify in utils.py so the tests pass",
 "kept": true, "drop": null, "at": 1794871000.0}

// Train
{
  "step": 7, "total": 19,                       // optimizer steps
  "samples": {"practice": 66, "rehearsal": 8},
  "losses": [0.044, 0.061, …],                  // one per committed step
  "eta_seconds": 2080 | null,
  "peak_memory_gb": 10.1 | null
}

// Check
{
  "key": "held_out" | "new_practice" | "trained_practice" |
         "original_moments" | "general" | "drift",
  "state": "done" | "now" | "ahead",
  "unit": "gap" | "accuracy" | "kl" | "share",
  "before": 0.0664 | null, "after": 0.0613 | null,
  "limit": 0.05 | null,
  "gates": true,                  // whether adoption depends on it
  "passed": true | null,
  "count": {"done": 21, "total": 21}
}
// Which checks gate adoption is recorded per cycle in `gates`, never assumed.
// First wiring: `held_out` (today's gate: real held-out conversations, the
// teacher shown what followed; unit "gap") gates, and so does one "not broken"
// check: `drift` (confirmed successes; unit "kl", limit general_drift_limit)
// once the drift set is big enough to resolve that limit (see Integration
// decisions: min_drift), else `general` (the fixed anchors, as today). The
// others are measured with gates: false until they are validated.

// Result
{
  "outcome": "adopted" | "kept" | "stopped",
  "reason": "passed" | "no_gain" | "capability_drop" | "repetition" | "repetition_rate" | "empty_answers" | "other" | "drift_limit" |
            "too_few_examples" | "stopped_by_user" | "error",
  "version": "v3" | null,
  "in_use": true | false,         // last cycle only: its version is still current
  "improved": {"count": 6, "checked": 10, "available": 38},
  "examples": [Example],          // up to 3, in display order
  "held_out": {"change": 0.0024, "margin": 0.0017}   // what the gate decided on;
                                  // change > margin passes; nulls before the gate
}

// Example
{
  "id": "…",
  // also: "lesson_preference": 0.13 (rule 4), "gap": {"before", "after"}
  "source": "original" | "practice",
  "split": "original" | "held_out" | "trained",   // stored, never shown
  "signal": {"id": "…", "type": "correction", "source": "user" | "tool"},
  "context": {"request": "≤300 chars of the task", "conversation": "…" | null,
              "at": 1794700000.0 | null},
  "feedback": "≤300 chars: the user's reply or the tool's result that followed",
  "lesson": "one line" | null,
  "before": Answer, "after": Answer,
  "teacher_preference": 0.42,     // see below
  "improved": true
}
// Answer
{"text": "…", "tool_calls": [{"name": "edit", "arguments": {…}}], "valid": true}

// Version
{"id": "night-3", "label": "v3", "current": true, "kind": "adapter",
 "cycle": "c-0007", "created_at": 1794880000.0}
```

**Before/after examples**

The check stage produces them in every cycle. The UI cannot make them up; it
only shows what the cycle recorded.

1. **Sets.**
   - **Original moments:** for each signal with a responsible step, the
     logged request just before that step (the context the model acted on).
     Real tasks never enter training, so these are never trained on.
   - **Held-out practice:** practice scenes split off by signal and never
     trained on.
   - **Trained practice:** up to `example_trained` scenes the cycle trained on.
2. **Answers.** Before = the cycle's starting version (adapter off), after =
   the candidate (adapter on). Same prompt, greedy decoding, same token limit.
   Each set is bounded by `example_limit` (a named config field) and the
   remaining idle gap; `count` in its Check says how many were done.
   Candidates are evaluated in the display order of rule 6 (original moments
   from user signals first), so a short gap still yields examples worth
   showing.
3. **Teacher preference.** The cycle's teacher scores both answers, shown the
   same context plus a hint: for an original moment, what really followed (the
   feedback); for a practice scene, which has no real follow-up, its teacher
   note (the lesson and the feedback it came from). `teacher_preference` = mean
   per-token log-probability of `after` minus that of `before`. Positive means
   the teacher prefers the new answer. `lesson_preference` subtracts the same
   difference as scored by the starting version without the hint: what both
   prefer anyway (fluency, a likelier phrasing) cancels, and what is left is
   what the hint explains.
4. **Improved** means all of: the answers differ after normalising whitespace;
   `after.valid` (well-formed, and any tool call parses and names a known
   tool, in the JSON or the XML `<function=…>` form); `lesson_preference >
   example_margin`. `example_margin` is a named config field; 0 (the default)
   measures it each cycle as the largest `lesson_preference` contrast between
   two sampled answers of the starting version, over `example_noise_samples`
   moments. (Measured on `teacher_preference` alone, sampled answers differ by
   general fluency, about 0.4 nats/token for LFM3, far above any real change;
   the contrast brings the noise to about 0.03.)
5. **Headline.** `result.improved` counts original moments and held-out
   practice only: `count` improved out of `checked` evaluated, out of
   `available` candidates. Trained practice never counts toward it. Surfaces
   state the checked number ("6 of 10 checked"), never only `available`.
6. **Choice of shown examples** (`result.examples`, at most 3): improved
   examples only, at most one per signal, in this order: original moments from
   user signals, original moments from tool signals, then practice of either
   split; within a group by `teacher_preference`, highest first. Surfaces show
   "original" or "practice", never the split.
7. **Nothing improved:** `examples` is empty and surfaces say so. An example
   that did not improve is never shown as an improvement.

Losses, practice lines and check values are shown to users. Lessons are at
most a one-line label on an example.

**Integration decisions** (2026-11-16)

- `songe forget` also strips example text (request, feedback, answers) from
  cycle records, keeping counts.
- The default is `learning = "dreams"`. `hindsight` remains a setting; under
  it real conversations are trained on, so original moments come from
  held-out conversations only.
- Read and Practice use the learner's own current serving version (the model
  being trained), whichever model produced the conversation; with a different
  one loaded they wait with `model_server`.
- The drift set (confirmed successes, never trained on) gates adoption once it
  is big enough to resolve the drift limit, decided the same way as the gate's
  noise margin: a paired bootstrap clustered by conversation at
  `gate_confidence`. It gates when it has at least `min_drift` conversations
  (its own named config field, not `min_heldout`, which is 5; the
  default is the fewest clusters at which a percentile bootstrap interval is
  not dominated by single conversations, justified in its comment and
  overridable) **and** that interval on its mean drift is narrower than
  `general_drift_limit`. It then rejects a candidate whose mean drift on it
  exceeds the limit, the same rule as the anchors (the interval only decides
  whether the set is precise enough to be the reference). Until both hold,
  the general anchors stay the "not broken" reference, as today. The check
  records which one gated and why (`drift_checks` in the cycle record).
  Beside the mean drift, `gate_values` keeps for each version what that mean
  does not show, on the set that gated: `base_entropy` and `entropy`
  (nats/token, the base's and the version's), `entropy_change` (relative)
  and `agreement` (the share of tokens where both prefer the same one), from
  the same logits as the KL. Recorded only: they gate nothing. `None` where
  not measured (the first Dream's incumbent is the base itself).

- Under Dreams, "did it get better?" is judged on what each version writes
  (`sleep/judged.py`), not on the KL to the teacher along the original
  model's answer (kept as `kl_change`, a diagnostic). Test units: held-out
  real conversations (with the user's feedback) first; held-out practice
  lessons (never trained on, one scene each, fixed order) fill the places up
  to `min_heldout` (5). Each unit's last 3 moments, 24 at most. At each moment
  the version in use and the candidate answer deterministically (384 tokens)
  as the user is served: the version in use with the notes it has now, the
  candidate with the notes that would stay after adopting it (rows marked
  `served`);
  the same action is a tie; of two different ones each goes on its own to
  the original base model (request, the user's feedback or the lesson, one
  answer: does it do what the feedback asks, yes or no; never which of two
  it prefers). A win when only the candidate's does, a loss when only the
  answer in use does; both, neither or a reply that cannot be read is a tie;
  the teacher's preference, which training fits, never breaks it. These
  judge rules (an undecided judge is a tie, #119; each answer judged on its
  own against the feedback, #178) still hold.
  At least `min_heldout` conversations must be present. Adopt only if the
  candidate is not clearly worse as served (clearly worse: losses − wins >
  1.64 × √(losses + wins), beyond what a judge right four times in five
  gives by chance; ties pass) and it learned: the same moments answered by
  both versions with no notes at all (`alone` on each row) are judged the
  same way, and alone it must win beyond that noise (wins − losses > 1.64 ×
  √(wins + losses)); drift and health pass (`sleep.gate.judged_reason`).
  When the notes already serve a habit, both versions follow it as served
  and tie, so a win as served is not asked for. Rows judged
  without notes (not `served`) keep the older rule: each moment one unit
  (win +1, loss −1, tie 0), resampled per moment; adopt only if wins >
  losses and the resampled lower bound is above zero. The
  record carries `judged` rows, `improved` {count, worse, same, checked} and
  the report's examples are the moments won. If the backend cannot answer,
  the KL decides and `judged_error` says so.
  Each version also answers up to 8 moments the lessons are not about (the
  drift set's confirmed successes, one per conversation). The record's
  `unrelated_actions` is, per strength tried, {changed, checked}: at how many
  of them the candidate takes another action than the version in use
  (`quick.differs`). Recorded only: it decides nothing. Absent when there
  are no confirmed successes yet.

**Dream control**

`POST /api/control {action: "dream", model: <public model id>}` starts a cycle
for that model now: reading starts whatever the amount of new data; practice
and training follow when data, memory and power allow. The reply is
`{started: bool, cycle: <id> | null, wait: Wait | null}`; `wait` names what is
missing when nothing could start or the cycle stops before training.

## 9. Interfaces

- **CLI**: `songe` / `status` starts with now/next/paused, then models, learning summaries, and versions with the last recorded progress/reason. `init` shows aligned facts and one closing line, prompts `Install? (Y/n)`, then prints ✓ results; `--yes` is required to approve from a non-TTY. Destructive prompts default to no. Commands: `open`, `models`, `models <model> on|off` (auto-dream), `undo [<model>]`, `pause`, `resume`, `sleep`, `forget`, `uninstall`, `doctor` (explains every check with measured values), `tui`.
- **TUI** (`songe tui`, Textual): now/next/paused first, followed by models, learning summaries, recorded outcomes and versions. Off / New only / All history controls for the selected model (`l` cycles levels), undo (`u`), pause/resume/open keys. No decorative visuals.
- **GUI** (`songe open`): exactly three status lines first: Now, Next, Paused. Current activity is the focal point; blockers explain what is wrong and how to fix it. Each grouped model row starts with its name and small tags, shows a recorded outcome or explicitly missing evidence, and keeps Off / New / All learning controls on the right. Below the status lines, each model shows its last cycle's before/after examples first (see Learning cycle), then the current cycle as a strip of its recorded stages: reading progress, practice as one plain line per scene, training with a small loss line, checks as before → after. Lessons appear at most as a one-line label. Until the `cycle` section ships, expanding a row reveals the legacy Day / Rewind / Train / Wake / Versions stages and actionable problems. A green dot marks only the model actively training. Daytime repeat-error rate differences use percentage points; legacy relative changes retain percentage units. An applied update alone never implies measured improvement. Layout supports light/dark themes and narrow screens without horizontal scrolling; fixed control columns and reserved status/summary space prevent refreshes or expansion from moving controls sideways. Expansion intentionally reveals content vertically. Local assets with content-hashed URLs cache for a year; `/api/state` returns an ETag and `revisions` for status/models/learning/versions (and `cycle` once it ships). Passing a JSON `revisions` query returns only changed `sections`; matching ETags or unchanged revisions return 304. The browser paints session storage first and patches model/stage nodes independently, preserving controls, keyboard focus and expanded details. Controls are unavailable until connected; stale cached data is labeled. The inventory scan is cached server-side for 60 seconds.

## 10. Packaging

- PyPI `songe` with extras: `songe[mlx]` (auto on Apple Silicon via markers), `songe[torch]`, `songe[cuda]`, `songe[rocm]` documented; base install pulls the right one by environment markers where possible.
- Homebrew tap `sous-labs/tap`, formula `songe` (virtualenv from PyPI wheels).
- GitHub Actions: tests on macOS, Ubuntu, Windows; release via PyPI Trusted Publishing.

## Release candidate integration

- Config persists `backend`, `torch_options` and the original `model_ref` (upstream, serving id, format, path, architecture). `torch_options` and `model_ref` are TOML tables. Accelerator installation uses a private runtime and the measured CUDA/ROCm driver runtime, or the official XPU index. Native MLX needs no training installation; GGUF conversion on Mac uses a private CPU converter stack when it is absent: `<data-dir>/converter`, installed on first use with a Flow line while it downloads, kept apart from the service's `runtime` (an init-made `runtime` that already has the converter modules is used as is).
- Version rows retain `kind`, `base`, and `training_adapter` alongside `artifact` and `adapter`. `training_adapter` is the framework checkpoint for incremental training; `artifact`/`adapter` bind the upstream's exported weights. Existing v2 stores acquire these nullable columns without rewriting versions.
- Full GGUF from an original GGUF keeps its file type. A dense safetensors source uses bf16 because it has no GGUF quantization to preserve. Full GGUF gate evaluation reloads that exact exported artifact.
- llama.cpp's owned server registers completed candidate and retained adapter paths before switching their scales. vLLM supervision enables LoRA updates on loopback. Init records the discovered invocation and provisions a separate Songe endpoint. An existing user process is never adopted or stopped.
- A parking snapshot is persisted through `on_snapshot` before memory is released. Failed or interrupted publication restores the previous global upstream binding before admission resumes.

Training progress callbacks report `(completed_samples, total_samples)`. The shared loop derives optimizer steps from the finite sample plan and current accumulation; it never accepts a clock or iteration cap. Calibration measures two updates using an explicit synthetic sample pool. `/api/state` includes `data_progress` (phase, completed, total, unit, and optional eta_seconds), also retained in cycle records.

## Hindsight self-distillation and training admission

Training follows DESIGN 3 (OPSD). The student is base + LoRA on the exact logged
context c; the teacher is the version the cycle started from (the original base in
the first cycle), frozen, on the same c plus a marked hindsight block: the logged
observations, then the idle review's "What went wrong" and "Lesson" when the reply
was reviewed, then "Respond now with your own action." A practice scene's
hindsight is one note Songe wrote (the lesson and the feedback it came from), so
its teacher gets that note as an aside instead: not part of the conversation,
shown only to it, to follow where it applies and not to mention. The aside asks
for "the reply to the message above" when the scene has no tools and, when it
has, for "your own action" as a real step's block does. General anchors
carry no hindsight and keep the original base as teacher. Forward KL over the full
vocabulary, each vocabulary entry's contribution clipped at `vocab_clip`
(provisional 1.0 until measured), is evaluated on the student's freshly sampled
action tokens only. Prompt, user, tool and historical assistant tokens never
become loss targets.

Contexts are never compacted or truncated. The gradient covers the longest tail of
c plus the response that fits the calibrated `max_sequence`; the context before it
is read in `teacher_chunk` pieces with no gradient and kept only as model state.
Models whose cache kinds cannot be frozen this way backpropagate the whole prompt.

Selection by surprise is retired: every train step that survives Dream's cleaning
(no empty tool arguments, no interrupted tool results, no reviewed reply that moved
on without an issue, one representative per repeated action in a conversation) is
trained once, plus the replay share.

Teacher prefill uses bounded KV-cache chunks without gradients. Calibration targets
`required_sequence = len(c) + response_tokens` and records `teacher_sequence`
separately. Training probes measure their own small teacher; full-context teacher/gate
forwards are measured on their own ladder (doubling toward the longest real teacher),
so one long conversation cannot inflate every probe. Teacher length cannot be inferred
from the student's backprop limit. A slice admits only teachers whose measured peak
plus margin fits the memory free now; the rest are deferred, never judged unfit.

Only total machine capacity is a permanent limit. A guard stop the probe caused while
other apps held memory is recorded as held (the size needs at least that much) and is
retried once that much memory plus margin is free; measurement meanwhile bisects below
it. Stops in size-independent phases (load, export, evaluation) simply retry later. Real practice is bounded by `slice_limit`, measured
optimizer cost and the remaining idle gap; untrained evidence stays in the durable
pool.

Gate references use exact contexts, all schemas and real held-out steps (plus fixed
general anchors), with the same teacher as training.

**With `learning = "dreams"`** (see Learning cycle) the method is the same
on-policy distillation, but the pool changes: the student's context is a
practice scene (a generated conversation ending where a lesson matters, built
on the logged system prompt and tool schemas), and the teacher's hindsight is
that scene's teacher note ("Note from an earlier, different task (it may not
apply here)", the feedback and the lesson). A lesson from a plain chat (a
request without tools) gets a different scene: one new opening message of the
same kind as the one the user sent before their feedback, written without
sight of the lesson, under the logged system prompt if there was one. It is
kept by the `changes` check instead of the forced choice between two lessons:
the served model answers it with and without the note, and the base, asked in
both orders, must find that the answer with the note does what the user's
feedback asks. On a server whose reasoning leaves no room for those answers
they are asked again with reasoning switched off, then with more room. Real contexts are never trained on;
replay draws from earlier practice scenes. Anchors and the gate are unchanged.

A scene counts as practised once a Deep Dream that trained it is adopted
(`practice_revisions`). A Dream that is not adopted, rejected by the gate or
never loaded by the model server, practised nothing: its scenes are kept under
`rejected_practice` and trained again, in full, by the next Dream that has a
new scene, after the scenes no Dream has tried when the Dream is capped. On
their own they wait, since the same scenes alone would be rejected again. A
scene trained by three Dreams that were not adopted counts as practised from
then on and comes back as replay only. Forgetting, a change of a model's
learning level and a switch of learning mode clear `rejected_practice` together
with `practice_revisions`.

References: [Thinking Machines, On-Policy Distillation (2025)](https://thinkingmachines.ai/blog/on-policy-distillation/);
[OPSD (2601.18734)](https://arxiv.org/abs/2601.18734);
[SDFT (2601.19897)](https://arxiv.org/abs/2601.19897);
[SDFT (2402.13669)](https://arxiv.org/abs/2402.13669);
[LoRA Learns Less and Forgets Less (2405.09673)](https://arxiv.org/abs/2405.09673).
These motivate the method; they do not establish gains for local agent traffic.
