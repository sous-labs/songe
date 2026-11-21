# Songe — design

A local model that sleeps. You use it during the day through your usual agent. When the machine is idle and on power, it re-reads what happened, learns from the consequences of its own actions, and wakes up as a slightly better model under the same name. Nothing leaves the machine.

This document is the build spec. Where it says **verify**, the builder must confirm the fact against installed source code or a real run before relying on it and record the result in its tests or notes.

---

## 1. First principles

1. **Learning needs a signal that the model did not already have.** A model cannot learn from its own output alone; that only re-confirms what it believes. The only new information a local model receives is *what happened after it acted*: the tool result, the error, the test output, the user's next message. Call this **hindsight**. Every design choice below follows from treating hindsight as the training signal.

2. **The same model is a better teacher when it can see the hindsight.** Given the context *and* what happened next, a model can usually tell what it should have done. Given only the context, it cannot. The gap between those two distributions is exactly what the model failed to anticipate. Learning = closing that gap, so that tomorrow the model anticipates what today it only saw in hindsight. No external teacher, no reward model, no cloud.

3. **Never act at night.** Re-running the user's tools in their repo while they sleep is unsafe and has side effects. Consequences are only ever read from the log; nothing is executed. Verification comes from what already happened.

4. **Learn on your own mistakes, in your own words (on-policy).** Training on text the model would not have written (off-policy SFT) moves it far from where it is and causes forgetting. Training on the model's *own* samples, scored token by token by the hindsight-teacher, moves it only where it was wrong. This is why the method is on-policy distillation, not SFT.

5. **Clean before learning.** Many steps teach nothing: an interrupted call, a loop repeating itself, a reply that simply moves on. Compute is scarce on a laptop, so drop those first and spend it on steps whose hindsight carries a signal (Dream, 3.2).

6. **The experience log is the asset; weights are a cache.** Each learning cycle continues the current adapter on selected new experience plus replay; a required rebuild makes one pass over the accumulated selected pool from the original base. A new base model can be "re-taught" from the same log. Nothing is lost when weights are thrown away.

7. **Do no harm, provably.** A cycle's result is used only if it is measurably better on held-out experience and not measurably worse in general. Otherwise the model stays as it was. One-step rollback, always.

8. **The user should do nothing.** One install, one init, then silence. No ratings, no notifications, no persona.

9. **Measure the machine, don't guess it.** Hardware varies by an order of magnitude. Slice sizing comes from calibration and live memory. The data determines total work.

---

## 2. Where it plugs in (user's view)

Today: `agent (opencode / pi / any OpenAI-compatible client) → LM Studio server (localhost:1234/v1) → model`.

With Songe: `agent → songe (localhost:7613/v1) → LM Studio → model "songe/<name>"`.

- `songe` is a thin **OpenAI-compatible pass-through proxy**. It forwards `/v1/chat/completions` (streaming and non-streaming, tools included), `/v1/models`, `/v1/completions`, `/v1/embeddings` to the upstream unchanged, except that it maps the public model name to the version currently being served. It records each exchange to SQLite.
- Why a proxy rather than LM Studio's log stream: it is agent-agnostic and upstream-agnostic, it sees the full request (messages, tools, sampling params) and the full response including tool calls, and it owns the `name → weights` binding so a version swap or rollback is one atomic pointer change. The proxy must add negligible latency and must never alter content.
- Why not replace LM Studio: the user keeps their model library, UI and settings. Songe only adds what is missing.

**Upstreams** (interface `Upstream`):
- `lmstudio` (primary). Load a version with `lms load <path> --identifier <id> -y`, unload with `lms unload <id>`. LM Studio cannot load LoRA adapters (verified 2026-11 via mlx-engine issue #51), so each accepted version is a fused model folder written into LM Studio's models directory (`downloadsFolder` in `~/.lmstudio/settings.json`) under a `songe/` publisher folder. **Verify** `lms load` accepts that folder or its model key, and the exact CLI behaviour, on this machine.
- `mlx` (fallback when LM Studio is absent). Songe runs `mlx_lm.server` itself and passes the adapter path per request (supported in mlx-lm ≥ 0.32: request field `adapters`; reload is keyed on (model, adapter); always send an explicit model path; use a new directory per version). No fusing needed. Supervise the child process (restart on death).
- Ollama: not supported in v1 (Ollama removed adapter support in Sept 2026; fused-safetensors import is untested). Document as future work.

**Agents.** `songe init` writes a provider entry for each detected agent, after showing the diff and asking once (`--yes` to skip). It backs up the original file and `songe uninstall` restores it.
- opencode: `~/.config/opencode/opencode.json`, provider `songe` using `@ai-sdk/openai-compatible`, `options.baseURL = http://127.0.0.1:7613/v1`, one model entry. Offer to make it the default `model`. **Verify** the schema from opencode docs/source.
- pi: `models.json` in pi's agent directory (see the installed package docs: `docs/models-configuration.md`, `docs/configuration.md`, `docs/models.md`), provider `songe` with `baseUrl`, `api: "openai-completions"`, model entry with context window. **Verify** fields from those docs.
- Any other client: print the base URL and model name.

---

## 3. The learning method: hindsight on-policy self-distillation (OPSD)

Occam's razor: copy what the papers agree on, add only what our setting forces, and mark each addition. Sources: OPSD (arXiv 2601.18734), SDFT (arXiv 2601.19897), Thinking Machines "On-Policy Distillation" (2025). Our privileged information is hindsight.

### 3.1 From log to steps
- Each request to the proxy carries the full message history. A **step** is one assistant turn: `context c` = the exact messages before it plus tool schemas, `action a` = what the model produced, `hindsight h` = what followed it in the same conversation (tool results and/or the user's next message), up to a token budget.
- Link requests into conversations by hashing canonicalised message prefixes: request B continues request A if B's messages start with A's messages + A's response.
- Steps with no hindsight (the conversation ended) are never trained on.
- Split train vs held-out **by conversation**, permanently, once enough conversations exist.

### 3.2 Dream: clean and organise the data (idle, before training)
OPSD and SDFT used clean privileged information (a reference solution, a demonstration). Raw hindsight is messy: tool dumps, "有问题", interrupted calls. Dream brings it closer to that condition, using the model already loaded in the model server, only from logged evidence:
1. **Clean.** Drop steps with no signal: interrupted or failed transport, empty tool arguments, and user replies the review read as a new task, question or other with no issue. Collapse a run of identical actions in one conversation into one step that records how often it repeated.
2. **Organise.** The idle review (`songe/review.py`) reads each step's evidence and writes what went wrong and a reusable lesson. The teacher's hindsight block is the real evidence followed by that note; the evidence stays the ground truth.
3. **Consolidate.** Merge the same mistake across conversations into lessons. This is for the dashboard (what was learned), not an input to training.
Generating new practice tasks from lessons is not built; it waits until the basic method shows a measured gain.

### 3.3 Teacher and student
- Student `π_θ(·|c)`: base + LoRA, on the exact context. No compaction, summary or cap: training sees what inference sees.
- Teacher `π_T(·|c, h)`: the version being served when the cycle starts (the original base in the first cycle), frozen for the whole cycle. OPSD fixes the teacher to the initial policy of each run; SDFT found the current student as teacher unstable and a never-updated base stable but weaker. Since each cycle continues the adapter, the cycle-start version is OPSD's rule applied per cycle. It sees the same `c` plus a marked hindsight block and "Respond now with your own action." It never generates; it scores the student's tokens in one forward pass.
- **Verify** the chat template renders tools and tool calls for each supported model family.

### 3.4 Objective
- **On-policy.** For each step, sample one fresh `y ~ π_θ(·|c)` at temperature 1.0 (OPSD: one rollout at 1.1; all three sources sample fresh each step).
- **Loss.** Forward KL `KL(π_T(·|c,h) ‖ π_θ(·|c))` over the full vocabulary at each token of `y`, averaged over `y`; loss only on `y`. OPSD's ablation (Qwen3-1.7B, AIME25): forward 43.9 vs reverse 37.5; SDFT reports forward KL best in practice; SDFT found the full-vocabulary estimator lower-variance than the sampled-token one. OPSD clips each vocabulary entry's contribution, `Σ_v min(ℓ_v, τ)`, because stylistic tokens otherwise dominate and performance collapses; we use the same operator. OPSD does not state τ; ours is **open** and is set from the first measured divergence distributions.
- **Long contexts (ours; no source covers 30–50k-token contexts).** Backpropagate through the longest tail of `c` plus `y` that fits the calibrated training length; read the rest of `c` before it in chunks with no gradient, keeping only its KV/recurrent state (LoRA on for the student; the teacher reads its own `c` the same way, never with gradient). Short contexts therefore get the exact gradient. Precedent: Transformer-XL's stop-gradient segment memory and truncated BPTT. The forward is always exact; the gradient misses how LoRA would re-encode the far context, which is a bias, not noise. Measured on Qwen3.5-0.8B at 400 tokens with gradient through `y` only: identical argmax and loss to the full pass, all LoRA gradients finite and nonzero, peak 0.85 GB vs 4.34 GB, 0.44 s vs 5.6 s. **Verify** gradient fidelity: cosine similarity of truncated vs full gradients at 2k–8k tokens. A step whose context does not fit today's free memory waits.
- **Retention.** A share of steps are general prompts with the original base as teacher and no hindsight. Thinking Machines restored IF-eval from 79% to 83% by distilling from the earlier model on unrelated prompts, and found LoRA still forgets.
- **Data and schedule.** Train every cleaned new step once per cycle plus the configured replay share of earlier ones, each with a fresh sample; no selection by surprise (the sources train on all data; scoring doubles the cost). The sources make several passes (SDFT 1–4 epochs); replay across cycles plays that role. LoRA rank 64, alpha 128, learning rate 5e-6, AdamW (OPSD), gradient-norm clip 1 (SDFT). Effective batch stays 4 (ours): OPSD used 32 over 30k examples for 100 updates; with dozens of new steps per cycle, 32 would mean about one update per cycle. Each cycle continues the current adapter.

### 3.5 Gate (all on the held-out split, evaluated on the **fused** artifact that will actually be served)
1. **Learned:** mean token forward KL to the hindsight-teacher on held-out steps (exact contexts) decreases vs the current version, by more than a bootstrap noise margin. This measures the training objective, not the task; the tool-call and repetition checks below are the task-side guard.
2. **Not broken:** KL to the base on the general prompt set stays below a threshold; tool-call validity rate on held-out tool steps (parseable calls with known tool names) does not drop; repetition and empty-output rates do not rise.
3. **Enough data:** no training unless the pool and held-out set reach minimum sizes (otherwise the gate is meaningless). Report "waiting for experience" instead.
If the fused 4-bit artifact loses most of the adapter's gain, re-fuse at higher precision (dequantize → 8-bit) when memory allows, and re-check. **Verify** `mlx_lm.fuse` behaviour on quantized bases.

### 3.6 Diagnostic: does hindsight help this model at all?
Small models may not use hindsight well: SDFT found a 3B model's in-context learning too weak to beat SFT, with gains growing with scale. Track `teacher_gain` = how often the hindsight-teacher assigns higher probability than the plain base to what eventually worked (e.g. the later successful tool call). Surface it in the record. If it is near zero for a model, say so plainly instead of pretending to learn.

---

## 4. Hardware policy

- `songe init` calibrates generation, scoring and optimizer step costs and host/device memory across slice dimensions. Memory measurements size the next slice; timing estimates choose slice length and display remaining work. They never cap total scoring or training.
- Admission uses live available memory, measured phase peaks and observed pressure transitions. There are no model-size or RAM-capacity tables. Memory, battery, thermal and ownership guards remain authoritative.

- Agent contexts are long (system prompts + tool schemas). They are never truncated; see 3.4 for how their memory stays bounded.
- Generation is part of the measured learning cost. One student sample per step (on-policy distillation) is far cheaper than best-of-n sampling or RL; on-device RL (GRPO) is measured at ~0.02 it/s on 0.6B and is out of scope.

---

## 5. When it sleeps (macOS)

- Conditions, all required: a learned idle gap without proxy traffic, sufficient live memory, battery and thermal permission, and new learnable experience. The bootstrap idle gap is configurable and personal traffic history replaces it. Work has no daily time cap.
- While running, hold `caffeinate -i -w <pid>` so idle sleep does not interrupt (lid-closed MacBooks without an external display still sleep; that is accepted, the run resumes at the next idle window).
- **Yield instantly**: if the user becomes active, battery becomes constrained, or a proxy request arrives, the training child process checkpoints and exits within seconds; the proxy is never slowed by training. Resume in the next safe idle slice from the committed sample cursor.
- One long-running process `songe serve` (proxy + small web UI + scheduler loop) installed as a per-user launchd agent with KeepAlive. Training runs in a separate child process.

---

## 6. Interface

CLI (only these):
- `songe init` — detect upstream and models, pick model (default: the one the agent already uses), calibrate, configure agents, install the launchd agent. Idempotent.
- `songe` — one status block: model, version (cycle n), last result in one line, data-based scoring/training counts and estimated remaining work, or why it is waiting.
- `songe open` — open the progress view in the browser.
- `songe undo` — serve the previous version.
- `songe pause` / `songe resume`.
- `songe sleep` — learn now (still yields to the user); `--dry-run` shows what would be learned.
- `songe forget` — delete the experience log (asks).
- `songe uninstall` — restore agent configs, remove launchd agent, leave models unless `--purge`.

No notifications. Plain, factual wording. English UI strings for v1.

## 7. Dashboard

The progress view (`http://127.0.0.1:7613/`) is a dense, English-only list of labels and values. Its first block has exactly three lines: now (current work or concrete blocker), next (action and condition), paused (no, or yes with resume). The same lines lead CLI status and the TUI.

Models are listed in a left rail (only those learning; the rest behind one "more" line), each with one line of state; the main column shows one model: the last learning cycle's before/after examples first, then the current cycle's stages, practice lines, losses and checks (docs/CONTRACTS.md, Learning cycle); retained versions. Small progress bars and one loss line are the only pictures: no ring, orbit, decorative plot or footer. Type and colour follow souslabs.ai: Outfit for text, IBM Plex Mono for what the model wrote, warm paper with grain, Songe purple for dreaming; dark mode, phone width, local assets only. Content-hashed assets cache long-term. Independent state revisions and ETags update only changed sections; cached session state provides first paint.

## 8. Code layout and quality bar

Python ≥ 3.11 package `songe` (src layout), installable with `uv tool install .`; dependencies minimal (mlx-lm, httpx, starlette/uvicorn or aiohttp, typer or argparse). Modules: `proxy`, `log` (SQLite), `steps` (conversation linking, hindsight), `upstream/{lmstudio,mlx}`, `agents/{opencode,pi}`, `sleep/{select,distill,gate,fuse,swap,calibrate}`, `schedule`, `ui/` (static assets), `cli`.

Production bar:
- Streaming pass-through is byte-faithful; client disconnects propagate; upstream errors pass through unchanged; the proxy keeps working if the log write fails.
- SQLite in WAL mode; data dir `~/Library/Application Support/songe` with 0700 permissions; schema versioned with migrations.
- Atomic version swaps; crash at any point leaves a servable state; old versions garbage-collected (keep current + previous).
- Config in TOML; structured logs with rotation; no secrets logged (redact Authorization headers).
- Tests (pytest): proxy against a fake upstream (streaming, tools, errors, disconnects); conversation linking and hindsight extraction on realistic opencode and pi traffic shapes; config writers on a temp HOME (never touch the real user's files in tests); scheduler conditions with mocked probes; gate logic; UI data contract. One **end-to-end test on this machine** with the smallest local MLX model (`qwen3.5-0.8b-mlx` or `lfm3-1.7b-mlx` in LM Studio's folder): synthetic logged conversations with hindsight → select → distill (one selected pass) → gate → fuse → `lms load` → request through the proxy returns from the new version → `undo`.
- README: what it is, install, what it changes on your machine, how to remove it.

## 9. Non-goals
Cloud anything (accounts, telemetry, cloud training); running the user's tools or touching their files (practice runs only in an isolated sandbox); multi-device sync; a chat UI. Ollama, Linux and CUDA, once non-goals for v1, are supported; reinforcement learning is planned (see ROADMAP.md).
