# Songe — development and operations

Songe is a local OpenAI-compatible proxy with MLX and Torch learning backends. Your agent talks to `http://127.0.0.1:7613/v1` using one stable name, `songe/<name>`. LM Studio continues to serve the model. Conversations stay in a private SQLite database on this machine.

During an idle window, Songe samples the model's own responses and compares them to the original model given the consequences recorded in the conversation. It continues the current adapter with replay, retaining the original base as teacher, then checks held-out conversations and 150 general prompts. An accepted version replaces the serving pointer atomically; a rejected version leaves it alone. Songe never executes logged tools.

**Build validation:** CPU tests and local HTTP streaming tests run on this machine. Metal training, calibration, LM Studio model loading, and visual browser QA require access outside the build sandbox. This is not yet a hardware-validated production release.

## Install

Requires Python 3.11+, a local model, and an upstream server. Apple Silicon uses MLX; Linux NVIDIA/AMD and Windows NVIDIA use Torch. Memory fit and training support come from live calibration. CPU-only installations serve and record without training.

```sh
export UV_CACHE_DIR=/Volumes/disk/sous-build-cache/uv
export HF_HOME=/Volumes/disk/sous-build-cache/hf
uv run pytest
uv run --extra torch pytest
uv run ruff check
uv tool install .
songe init
```

Init discovers models through `songe.models`, prefers the agent's configured model and endpoint, and displays aligned facts before `Install? (Y/n)`. `--show-diff` also prints exact dependency commands and agent diffs. Missing accelerator dependencies go into `<data-dir>/runtime`; service installation then uses that interpreter. On an MLX Mac, the GGUF converter toolkit (torch, peft, gguf; about 1 GB) is installed separately in `<data-dir>/converter` the first time a conversion or merge needs it, never in `runtime`, so the service and the upgrade check never mistake it for Songe's own installation. `--yes` accepts the displayed plan and is required for non-TTY confirmation. Calibration runs under the memory guard before agent files are applied. Repeating init reuses calibration for the same binding. `songe models MODEL off|new|all` changes that model's learning level; new only uses post-cutoff exchanges, all includes earlier shared history. Adding a model preserves existing histories; changing the installation's upstream requires a fresh data directory.

For the MLX fallback:

```sh
songe init --upstream mlx --model /absolute/path/to/local/mlx-model
```

## Use

```sh
songe                        # one status block
songe open                   # now, next, models and learning results
songe pause
songe resume
songe sleep --dry-run         # data counts; does not import Metal or train
songe sleep                   # learn now; power/data/memory checks still apply
songe undo                   # load and serve the retained previous version
songe forget                 # confirm deletion of experience; pauses learning
songe uninstall              # restore configs and remove launchd agent
songe uninstall --purge       # also remove Songe-created weights and private data
```

`songe serve` runs the proxy, UI, and scheduler in the foreground. With no launch agent, leave it running to process sleep requests. `--now` skips the initial user-idle delay, then yields if activity resumes; it still requires a traffic gap, AC power when measurable, enough memory and experience, and a safe temperature. The bootstrap gap is ten minutes and adapts to observed traffic. Every proxy request signals the training process immediately. A complete optimizer checkpoint is retained; an unresponsive training process group is killed after three seconds.

For another agent, configure the base URL `http://127.0.0.1:7613/v1`, a placeholder API key such as `local`, and the public model name printed by `init`. `/v1/models` lists only the `songe/…` names, not the upstream's own listing. Chat completions, legacy completions, embeddings, tools, upstream errors, and raw SSE bytes pass through. In a request for a Songe name, the name is replaced with the version in use, and preferences from Quick Dreams are appended to the system message; the MLX fallback also supplies its explicit adapter binding. Responses carry the name the agent asked for, not the upstream identifier.

The web view uses bundled fonts and other local static assets, without network dependencies. It shows now/next/paused, grouped model files, learning summaries and versions. Hash-addressed assets are cached for a year; state polls use per-section revisions and ETags. Returning to a retained version loads and checks its weights before switching.

## What changes on your machine

- Private data: `~/Library/Application Support/songe` (directory mode `0700`): `config.toml`, SQLite in WAL mode, config backups, calibration, checkpoints, evaluation records, and rotating event logs. Transport headers and Authorization tokens are not stored. Conversation bodies can contain sensitive material; they are the experience being retained. Repeated parts (system prompt, tools, earlier messages) are stored once. Recordings are kept under `recordings_limit_gb` in `config.toml` (default 2): over it, the oldest conversations already learned from are deleted; ones not learned from yet are kept, and `songe doctor` and status say when that leaves them over the limit.
- LM Studio: new fused versions under `<downloadsFolder>/songe/`, keeping current and previous weights. Existing models and the original base are never deleted. Fusion uses new private directories; only the actual exported artifact's passing gate can authorize publication. Test models use `<downloadsFolder>/songe-test/` only.
- OpenCode: a `songe` provider in `~/.config/opencode/opencode.json`, using the schema for the installed major version. JSONC files are left for manual configuration.
- Pi: a `songe` provider in `<agent-dir>/models.json`; respects `PI_CODING_AGENT_DIR`, otherwise uses `~/.pi/agent`.
- Launchd: `~/Library/LaunchAgents/ai.souslabs.songe.plist`, with KeepAlive. Songe starts LM Studio's local server when needed and supervises its own server for the MLX fallback. No shell profiles or notifications.

Uninstall reverses only Songe-owned JSON fields with a three-way merge. Unrelated user edits are preserved, and conflicts in owned fields retain backups for manual recovery. Without `--purge`, data and models remain. `forget` deletes logs, replay snapshots, and checkpoints while preserving served adapters; deletion does not erase information already present in retained weights, backups outside Songe, or filesystem snapshots.

To isolate every test or manual experiment, use `songe --data-dir /explicit/path …` or `SONGE_DATA_DIR`. `init --launchd-target /explicit/path/songe.plist` writes the plist without loading it; agent targets still use the environment's HOME/XDG/Pi paths, so use an isolated HOME for config tests.

## Learning controls

The TOML config exposes LoRA rank (default 16), learning rate, gradient accumulation, sequence and response budgets, selection counts, minimum data sizes, drift threshold, the loss (`forward_kl` by default, `reverse_kl`, or generalized JSD with `loss = "jsd"` and `beta`), and optional teacher top-k with an aggregate tail probability. Calibration must succeed before training; no unmeasured throughput estimates are substituted. Do not mutate the original model folder after initialization: weights and tokenizer define the base for every cycle.

New conversations wait unassigned; once there is enough data, about 20% of them (at least `min_heldout`) are held out, drawn across the kinds of feedback present, and a settled split never changes. Branches keep the root split. Histories that cannot be linked after compaction are excluded rather than risk split leakage. Multimodal exchanges remain proxy-compatible but are excluded from text-only training. Hindsight is token-bounded; student and teacher retain identical context and tools. Short or malformed examples are skipped.

The gate requires enough independent held-out conversations, an improvement exceeding a paired conversation-bootstrap margin, low general drift, and no rise in empty or repetitive outputs or decline in tool-call validity. It evaluates actual fused MLX and reloaded full GGUF artifacts; adapter upstreams retain their corresponding adapter binding. A near-zero teacher-gain diagnostic is shown honestly; it is not evidence of successful learning. These local checks provide bounded evidence, not a guarantee about every possible task.

## Development and hardware validation

```sh
export UV_CACHE_DIR=/Volumes/disk/sous-build-cache/uv
export HF_HOME=/Volumes/disk/sous-build-cache/hf
uv run pytest
uv run ruff check src tests scripts
node scripts/ui_contract.cjs
uv run python -m scripts.verify_templates
uv run python -m scripts.benchmark_proxy
# Requires normal macOS/Metal/LM Studio access:
uv run python -m scripts.e2e
```

The real end-to-end script logs synthetic conversations, calibrates at 256/512 tokens, selects examples, trains the selected pool once, evaluates the fused gate, loads the candidate with `lms`, calls it through the proxy, and undoes the swap. It never bypasses the gate; an unproved improvement is a test failure. It keeps output under `/Volumes/disk/sous-build-cache/e2e/`, unloads only its own identifiers, removes its own test model folders, and stops the server if it started it. It never unloads the user's `lfm2.5-8b-a1b-mlx`.

Legacy Outfit and Syne assets retain their bundled SIL Open Font License notices in `src/songe/ui/`; the dashboard uses system monospace fonts.

## Evaluation and allocation policy

The gate uses a configurable two-sided paired conversation bootstrap interval
(`gate_confidence`, default 95%). Improvement must exceed its measured lower
confidence bound; the numerical floor is the floating-point ULP of the gain.
`min_heldout` (12 independent conversations) and `min_train` (32 examples) are
minimum support policies, not guarantees of power or significance. The default
KL budget `general_drift_limit=0.05` is in nats/token: it bounds the average log
likelihood ratio by 0.05 (a multiplicative likelihood ratio of exp(0.05)). It is
an explicit behavior budget, not a hardware measurement. Repetition is the
continuous duplicate fraction of `repetition_ngram`-word phrases (default four),
compared against the incumbent with no length or uniqueness cutoff.

Calibration samples process-tree RAM and the selected device separately across
loading, serial microbatch training, export/conversion, and fused evaluation.
It measures additional copies only in phases that actually use them. Before each
slice, Songe fits a nonnegative per-token cost model to the measured peaks and
chooses the largest fitting context, response, logit chunk and accumulation count,
in that order. Accumulation uses serial microbatches; it is not a larger resident
batch. `last_slice_size` records the choice, predictions and live resources; the
night retains its slice choices too. Admission records RAM, pressure, available
swap/compressor/PSI sensors and selected VRAM before allocating. Existing warning
pressure is a baseline: a fitting slice can run while that warning stays unchanged.
Critical pressure never admits work. The RAM margin is the observed boundary plus
its directional transition bracket: normal baselines use the warning boundary;
warning baselines use the critical boundary. Both retain REVIEW-3's recovery and
expiry rules. Until a boundary is observed there is no invented capacity reserve.
The guard interrupts on escalation above the admission baseline, process-tree swap
growth, or global paging/compressor/PSI growth correlated with owned allocation.
Where only global paging sensors exist, this attribution is conservative. It also
uses measured depletion rates to stop before the reserved boundary, then kills
after bounded save time; critical pressure or exhausted headroom kills immediately.
Windows retains GlobalMemoryStatusEx and selected VRAM checks without treating
process commit as swap. Calibration starts with one response token, logit chunk
and microbatch at the shortest useful template, then grows under the same guard.
Old calibration data is upgraded in an idle guarded child.
If the smallest useful measured configuration does not fit, the status explains
RAM, device memory and the margin. Conversion and serving retain live guards.
Cycles have no daily duration or calibrated step/scoring cap. Scoring covers
all new, unscored training experience. Training makes one pass over the selected
new pool plus deterministic old-data replay and finite anchor rehearsal, then
runs the held-out gate, scale selection and swap. Without new learnable data it
waits; completed rejected rebuilds also wait for new evidence or a changed
request. Calibration records optimizer/score costs for slice planning and ETA
only. Status exposes completed/total experiences or samples and estimated work
remaining; idle waiting is excluded from that estimate. Old duration, phase-share
and per-cycle capacity keys are dropped when loading configuration.

A durable sample plan and cursor preserve progress when the next slice changes
accumulation to fit memory. The final partial batch uses its actual sample count;
anchors never displace selected examples or cause the pool to loop.
An indivisible export can finish beyond a slice's planning deadline. Activity,
shutdown and memory cancellation continue to apply throughout it.

Inference bodies, response capture and the pending recording queue are bounded
by the local IO capture limit. Committed observations are reconstructed in a
guarded child in bounded batches with persistent prefix links; unresolved late
continuations are retried. Durable daytime metrics update from changed steps.
Diagnostics are private rotating `songe.log` files containing fixed events,
exception types and stack locations, without exception messages or source lines.
