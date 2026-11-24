import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from urllib.parse import urlparse

from . import NAME
from .storage import atomic_write, private_dir


def data_dir(root=None) -> Path:
    from .platform import data_dir as platform_data_dir

    return platform_data_dir(root)


@dataclass
class Config:
    exposed_models: list[str] = field(default_factory=list)
    model_key: str = ""
    model: str = ""
    public_model: str = f"{NAME}/local"
    base_identifier: str = ""
    upstream: str = "lmstudio"
    upstream_url: str = "http://127.0.0.1:1234/v1"
    upstream_timeout: float = 180.0  # local model load/warm deadline
    backend: str = "auto"
    torch_options: dict = field(default_factory=dict)
    model_ref: dict = field(default_factory=dict)
    artifact_kind: str = ""
    ollama_models_dir: str = ""
    server_command: list[str] = field(default_factory=list)
    server_owned: bool = False
    server_source_url: str = ""
    replace_loaded_model: bool = False
    port: int = 7613
    lms: str = "lms"
    models_dir: str = ""
    publisher: str = NAME
    context_window: int = 8192  # fallback until the server reports its own
    # Context each model is actually served with, by public name, as reported
    # by the model server (LM Studio's per-model setting). Agents follow it.
    served_contexts: dict = field(default_factory=dict)
    rank: int = 64  # OPSD: LoRA rank 64, alpha 128, lr 5e-6
    lora_scale: float = 2.0  # alpha / rank (OPSD: 128 / 64)
    # What a cycle trains on (docs/CONTRACTS.md, Learning cycle). "dreams":
    # read sessions into signals, write practice scenes, train on those; real
    # conversations never enter training. "hindsight": train on real steps.
    learning: str = "dreams"
    # Practice scenes written per signal. Variants differ (sampled, and told
    # to differ), so one lesson is practised in more than one situation; the
    # research run used 3, and about a quarter of scenes pass every check.
    practice_per_signal: int = 3
    # Quick Dream (quick.py): lessons in the prompt, at most this many; it
    # runs after this much quiet (not every pause) and reads the newest
    # unread conversations for at most this long, leaving the rest to Deep.
    prompt_lessons: int = 10
    # Ask the served model to merge and shorten the non-rule lessons. Off: on
    # the live runs Granite 4.2 3B softened, reworded at random and echoed its
    # instruction, so the lessons' own text (tidied, merged by wording) is used.
    quick_consolidate: bool = False
    # An "always" lesson (not a rule the user stated) enters the prompt once
    # it has come up in this many conversations. 1: short forms are worded
    # differently each time, so the same fact rarely groups across
    # conversations and a higher bar emptied the prompt (replay, 2026-11-18).
    quick_min_sessions: int = 1
    quick_quiet_seconds: float = 90.0
    quick_read_seconds: float = 300.0
    # Deep Dream (practice, training, checks) starts on its own only after
    # this much quiet, until gap prediction decides it; Dream starts one now.
    deep_quiet_seconds: float = 1200.0
    # Most space the recorded conversations may use, in GB (issue #73). Over
    # it, the oldest recordings already learned from are deleted; ones not
    # learned from yet are always kept (songe doctor says when that leaves
    # the recordings over the limit). With repeated prompts stored once, a
    # 200-request agent session with a 13k-token prompt takes about 4 MB.
    recordings_limit_gb: float = 2.0
    # Before/after examples (docs/CONTRACTS.md): at most this many per set
    # (original moments, held-out practice), and this many trained scenes,
    # so the check stage stays a few minutes on a laptop.
    example_limit: int = 8
    example_trained: int = 3
    # An example improved only if its lesson preference (how much more the
    # teacher prefers the new answer than the starting version without the
    # hint does) exceeds this. 0 measures it each cycle: the largest such
    # contrast between two sampled answers of the starting version, over
    # example_noise_samples moments.
    example_margin: float = 0.0
    example_noise_samples: int = 5
    learning_rate: float = 5e-6
    accumulation: int = 4  # Fixed samples per optimizer step; processed serially.
    # Backpropagation limit for C(X) + response, independent of teacher length.
    # 0 uses context_window as the policy ceiling; calibration targets real C(X).
    max_sequence: int = 0
    config_schema: int = 6
    response_tokens: int = 128
    hindsight_tokens: int = 512
    teacher_chunk: int = 512  # Frozen forward prefill; never a context truncation.
    min_train: int = 32  # initial pool support; user policy, not a significance test
    # Separate test conversations the gate needs before it decides. The gate
    # resamples conversations: with 2, two that moved the same way always
    # read as "significant" (live: a repetition veto, and a gain, on 2).
    min_heldout: int = 5
    min_new: int = 8
    # The drift set (confirmed successes) replaces the fixed anchors as the
    # "not broken" check once it has this many conversations and its bootstrap
    # interval is narrower than general_drift_limit. A percentile bootstrap
    # over fewer clusters than this is dominated by single conversations
    # (each resample repeats one of a handful of values); min_heldout (5) is
    # still too few to gate drift on.
    min_drift: int = 10
    # Most practice in one cycle: per slice under hindsight, per Dream under
    # dreams (a Dream's plan covers all untrained practice up to this).
    slice_limit: int = 72
    general_drift_limit: float = 0.05  # nats/token KL budget; exp(.05) likelihood ratio
    gate_confidence: float = 0.95  # two-sided paired bootstrap interval
    repetition_ngram: int = (
        4  # measure duplicate four-word phrases as a continuous rate
    )
    loss: str = "forward_kl"  # OPSD/SDFT: forward KL, full vocabulary
    vocab_clip: float = 1.0  # OPSD per-vocabulary clip; provisional until measured
    beta: float = 0.5
    teacher_top_k: int = 0
    token_chunk: int = 32
    memory_fraction: float = 1.0  # Optional user cap; live telemetry governs capacity.
    calibrated_peak_gb: float = 0.0
    calibrated_host_gb: float = 0.0
    calibrated_device_gb: float = 0.0
    calibrated_step_seconds: float = 0.0
    calibrated_score_seconds: float = 0.0
    calibration_available: bool = False

    # Ten minutes is the user's bootstrap idle gap until personal history fits.
    # Start as soon as the proxy has been quiet this long and resources allow.
    idle_default_seconds: float = 10.0
    idle_history_min: int = 20  # enough observations for an empirical tail
    idle_tail_min: int = 5  # reject unsupported far-tail scheduling estimates
    idle_history_limit: int = 512  # bounded recent history adapts to habit changes
    idle_survival_probability: float = 0.8  # user-overridable completion confidence
    slice_default_seconds: float = 60.0  # bootstrap; replaced by calibration
    slice_max_seconds: float = 300.0  # latency cap; units still checkpoint individually
    guard_min_interval: float = 0.1
    guard_max_interval: float = 1.0
    guard_grace_seconds: float = 3.0
    rebuild_drift_fraction: float = 0.9  # rebuild before the configured gate limit
    rebuild_retired_fraction: float = 0.5  # majority of evidence has expired
    replay_fraction: float = 0.25  # reserve a quarter of samples for old evidence
    # Candidate strengths, tried largest first; the first that passes is used.
    wake_scales: tuple = (1.0, 0.5)
    kept_versions: int = 3  # older versions that stay restorable
    anchor_fraction: float = 0.1  # initial general rehearsal; grows with measured drift
    force_cpu_training: bool = False

    def __post_init__(self):
        if self.max_sequence <= 0:
            self.max_sequence = self.context_window

    @classmethod
    def load(cls, root: Path | None = None):
        path = data_dir(root) / "config.toml"
        data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        # Retired in schema v2; older installations may still have saved it.
        data.pop("surprise_floor", None)
        # Retired planning multiplier; measured phase budgets supersede it.
        data.pop("calibration_concurrency", None)
        # Context compaction is retired: training reads exact contexts.
        data.pop("context_recent_turns", None)
        data.pop("context_summary_tokens", None)
        data.pop("lora_alpha", None)
        # Retired with surprise selection and learnability weights (DESIGN 3.4).
        for key in (
            "learnability_lower_quantile",
            "learnability_upper_quantile",
            "divergence_history",
        ):
            data.pop(key, None)
        # The per-slice cap was the sum of the two retired selection quotas.
        if "select_top" in data or "select_random" in data:
            limit = data.pop("select_top", 64) + data.pop("select_random", 8)
            data.setdefault("slice_limit", max(1, limit))
        # Retired: web Library and practice generation (DESIGN 3.2).
        for key in (
            "library_enabled",
            "dream_enabled",
            "dream_per_lesson",
            "dream_global_ratio",
            "dream_max_ratio",
            "dream_adjustment",
            "dream_tier1_reliability",
            "dream_tier2_reliability",
        ):
            data.pop(key, None)
        # Configs saved before schema 2 carried the retired 2048 default cap; it
        # follows the serving context instead. Explicit values survive afterwards.
        if data.get("config_schema", 1) < 2 and data.get("max_sequence") == 2048:
            data.pop("max_sequence")
        # Before schema 4, tests were a separate quota of 12 conversations.
        if data.get("config_schema", 1) < 4 and data.get("min_heldout") == 12:
            data.pop("min_heldout")
        # Schema 5 follows OPSD (DESIGN 3.4); saved copies of the old defaults
        # follow the new ones, explicit choices survive.
        if data.get("config_schema", 1) < 5:
            for key, old in (
                ("rank", 16),
                ("learning_rate", 1e-5),
                ("loss", "reverse_kl"),
            ):
                if data.get(key) == old:
                    data.pop(key)
        # Schema 6: 2 held-out conversations was too few to decide on.
        if data.get("config_schema", 1) < 6 and data.get("min_heldout") == 2:
            data.pop("min_heldout")
        if "config_schema" in data:
            data["config_schema"] = cls.config_schema  # migrations run once
        # The old 10-minute default is retired; saved copies of it follow the new one.
        if data.get("idle_default_seconds") == 600.0:
            data.pop("idle_default_seconds")
        # Work is sized by data. Retain compatibility with clock-budget configs.
        for key in (
            "nightly_seconds",
            "steps_per_night",
            "score_per_night",
            "training_time_fraction",
            "scoring_time_fraction",
        ):
            data.pop(key, None)
        # A setting written by a newer Songe (after a downgrade) is skipped,
        # not fatal: every command would stop working otherwise.
        for key in set(data) - {f.name for f in fields(cls)}:
            data.pop(key)
        if "wake_scales" in data:
            data["wake_scales"] = tuple(data["wake_scales"])
            # The old saved default tried four strengths, each a full candidate.
            if sorted(data["wake_scales"]) == [0.25, 0.5, 0.75, 1.0]:
                data["wake_scales"] = cls.wake_scales
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def for_model(cls, root, model):
        from .log import Store
        from .models.registry import model_config

        return model_config(cls.load(root), Store(root), model)

    def validate(self):
        import re

        if not isinstance(self.replace_loaded_model, bool):
            raise ValueError("replace_loaded_model must be true or false")
        if self.learning not in ("dreams", "hindsight"):
            raise ValueError("learning must be dreams or hindsight")
        if self.model and not Path(self.model).is_absolute():
            raise ValueError("model must be an absolute local path")

        if not re.fullmatch(r"songe(?:-[a-zA-Z0-9_-]+)?", self.publisher):
            raise ValueError(
                "publisher must be songe or a songe- prefixed test publisher"
            )
        url = urlparse(self.upstream_url)
        if url.scheme != "http" or url.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("Songe upstream must be a local HTTP endpoint")
        if self.upstream not in {"lmstudio", "mlx", "ollama", "llamacpp", "vllm"}:
            raise ValueError(
                "Supported upstreams: lmstudio, mlx, ollama, llamacpp, vllm"
            )
        if self.backend not in {"auto", "mlx", "torch"}:
            raise ValueError("Supported backends: auto, mlx, torch")
        from .backends.torch import TorchOptions

        try:
            TorchOptions(**self.torch_options)
        except TypeError as exc:
            raise ValueError(f"Invalid torch_options: {exc}") from exc
        if self.upstream_timeout <= 0:
            raise ValueError("upstream_timeout must be positive")
        if (
            self.loss not in {"forward_kl", "reverse_kl", "jsd"}
            or not 0 < self.beta < 1
        ):
            raise ValueError("Invalid distillation objective")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1]")
        if not 0 < self.gate_confidence < 1 or self.repetition_ngram < 1:
            raise ValueError("Invalid evaluation policy")
        for key in (
            "rank",
            "accumulation",
            "token_chunk",
            "max_sequence",
            "response_tokens",
            "teacher_chunk",
            "min_train",
            "min_heldout",
            "min_drift",
            "practice_per_signal",
            "prompt_lessons",
            "quick_min_sessions",
            "quick_quiet_seconds",
            "quick_read_seconds",
            "deep_quiet_seconds",
            "example_limit",
            "example_noise_samples",
            "slice_limit",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.response_tokens >= self.max_sequence:
            raise ValueError("max_sequence must leave room for prompts")
        if (
            not 0 < self.idle_survival_probability <= 1
            or not 0 < self.anchor_fraction <= 1
            or not 0 <= self.replay_fraction <= 1
        ):
            raise ValueError("Invalid scheduling or replay probability")
        if (
            not 0 < self.guard_min_interval <= self.guard_max_interval
            or self.guard_grace_seconds <= 0
        ):
            raise ValueError("Invalid guard intervals")
        if any(not 0 < v <= 1 for v in self.wake_scales) or not self.wake_scales:
            raise ValueError("Wake scales must be in (0, 1]")
        for key in (
            "recordings_limit_gb",
            "idle_default_seconds",
            "slice_default_seconds",
            "slice_max_seconds",
            "idle_history_min",
            "idle_history_limit",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        prefix, _, name = self.public_model.partition("/")
        if prefix != NAME or not name:
            raise ValueError("public_model must start with songe/")

    def save(self, root: Path | None = None):
        self.validate()
        if self.model_key:
            from .log import Store
            from .models.registry import save_config

            save_config(self, Store(data_dir(root)))
            return
        root = private_dir(data_dir(root))

        def value(v):
            return str(v).lower() if isinstance(v, bool) else json.dumps(v)

        atomic_write(
            root / "config.toml",
            (
                "\n".join(
                    [
                        f"{k} = {value(v)}"
                        for k, v in asdict(self).items()
                        if not isinstance(v, dict)
                    ]
                    + [
                        line
                        for k, v in asdict(self).items()
                        if isinstance(v, dict)
                        for line in [
                            f"\n[{k}]",
                            *(f"{json.dumps(a)} = {value(b)}" for a, b in v.items()),
                        ]
                    ]
                )
                + "\n"
            ).encode(),
        )


# What a night computes: the weights it starts from, how it trains, how its
# checks decide, and what it publishes. A change to any of these discards an
# interrupted night or an unswapped candidate; a change to anything else
# (when to start, idle and slice timing, memory guards, examples shown,
# practice writing, the per-Dream practice cap, which a resumed night's saved
# plan already fixes) leaves them resumable.
TRAINING_KEYS = (
    "config_schema", "upstream", "publisher",
    "model", "model_ref", "base_identifier", "public_model", "artifact_kind",
    "backend", "torch_options", "force_cpu_training", "learning",
    "rank", "lora_scale", "learning_rate", "accumulation", "loss", "vocab_clip",
    "beta", "teacher_top_k", "max_sequence", "response_tokens",
    "hindsight_tokens", "teacher_chunk", "token_chunk",
    "anchor_fraction", "replay_fraction",
    "min_heldout", "min_drift", "general_drift_limit", "gate_confidence",
    "repetition_ngram", "wake_scales",
)


def installation_keys(root: Path | None = None) -> set[str]:
    """The keys written to the installation's config.toml; the rest are defaults."""
    path = data_dir(root) / "config.toml"
    if not path.exists():
        return set()
    return set(tomllib.loads(path.read_text(encoding="utf-8")))


def training_fingerprint(config):
    """What a night trains with (TRAINING_KEYS). Measured sizes and served
    context limits change while it runs and must not discard it."""
    from .steps import digest

    return digest({k: getattr(config, k) for k in TRAINING_KEYS})
