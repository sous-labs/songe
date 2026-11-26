"""Lazy-imported transformers/PEFT backend with a shared frozen teacher.

The transformer runs once per student rollout. Checkpointed head chunks avoid
retaining response-length by vocabulary-sized tensors for the backward pass.
No torch, PEFT, or accelerator runtime is imported when importing this module.
"""

import contextlib
import copy
import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ..sleep.prompts import lesson_tokens, paired_prompts, response_positions
from .objective import view_key
from .reference import TAIL_FLOOR

# Cache layer methods that write the layer's state (see TorchBackend.frozen).
UPDATES = ("update", "update_conv_state", "update_recurrent_state")


@dataclass
class TorchOptions:
    # Options may be overridden with config.torch_options (dict or this class).
    device: str = "auto"
    force_cpu: bool = False  # CPU training is an explicit choice, serving is fine.
    temperature: float = 0.8  # Match the MLX rollout sampler.
    top_p: float = 0.95
    seed: int = 7613  # Match repeatable rollouts in the MLX reference.
    head_tolerance: float = 0.01  # MLX's official-forward validation tolerance.
    gradient_checkpointing: bool = True
    weight_decay: float = 0.01  # AdamW's reference default.
    max_grad_norm: float = 1.0  # Same clipping norm as sleep.distill.train.
    # "auto": paged 8-bit AdamW (bitsandbytes) when the base is loaded in 4-bit,
    # where AdamW's two fp32 states per adapter weight are gigabytes; else AdamW.
    optimizer: str = "auto"
    # Big-model memory savers; "auto" turns each on when the base is loaded in
    # 4-bit (it would not fit in bf16), where every gigabyte counts.
    # The input embedding is a lookup by token id: it can live in host RAM.
    embedding_on_host: str = "auto"
    # Activations saved for backward go to host RAM (pinned) during training.
    offload_activations: str = "auto"
    # Packed experts (Inkling-2) in pinned host RAM, copied in per use:
    # ~16 GB more GPU memory for longer windows, at the cost of PCIe traffic.
    experts_on_host: str = "off"


@dataclass(frozen=True)
class Stats:
    prompt_tokens: int
    generation_tokens: int
    seconds: float
    generation_tps: float
    prompt_seconds: float = 0.0

    @property
    def prompt_tps(self):
        return self.prompt_tokens / max(self.prompt_seconds or self.seconds, 1e-6)


def divergence(
    student,
    teacher,
    kind="forward_kl",
    beta=0.5,
    top_k=0,
    reduction="mean",
    clip=0.0,
):
    import torch

    if kind not in {"forward_kl", "reverse_kl", "jsd", "gjsd"}:
        raise ValueError(f"Unknown divergence: {kind}")
    if not 0 < beta < 1 or top_k < 0:
        raise ValueError("Invalid beta or top_k")
    s = student.float().log_softmax(-1)
    t = teacher.detach().float().log_softmax(-1)
    if 0 < top_k < t.shape[-1]:
        idx = t.topk(top_k, dim=-1).indices
        s, t = s.gather(-1, idx), t.gather(-1, idx)
        s_tail = (1 - s.exp().sum(-1, keepdim=True)).clamp_min(TAIL_FLOOR).log()
        t_tail = (1 - t.exp().sum(-1, keepdim=True)).clamp_min(TAIL_FLOOR).log()
        s, t = torch.cat((s, s_tail), -1), torch.cat((t, t_tail), -1)
    if kind == "forward_kl":
        terms = t.exp() * (t - s)
        if clip > 0:
            terms = terms.clamp_max(clip)
        value = terms.sum(-1)
        return value if reduction == "none" else value.mean()
    if kind == "reverse_kl":
        value = (s.exp() * (s - t)).sum(-1)
        return value if reduction == "none" else value.mean()
    mix = torch.logaddexp(math.log(beta) + t, math.log1p(-beta) + s)
    value = (beta * t.exp() * (t - mix) + (1 - beta) * s.exp() * (s - mix)).sum(-1)
    return value if reduction == "none" else value.mean()


def sampling_probabilities(logits, temperature, top_p):
    """mlx-lm filters untempered probabilities, keeps boundary ties, then samples."""
    logprobs = logits.float().log_softmax(-1)
    if 0 < top_p < 1:
        ordered = logprobs.sort(dim=-1).values
        probabilities = ordered.exp()
        mass_above = probabilities.flip(-1).cumsum(-1).flip(-1) - probabilities
        total_mass = mass_above[..., :1] + probabilities[..., :1]
        dropped = (mass_above >= top_p * total_mass).sum(-1, keepdim=True)
        threshold = ordered.gather(-1, dropped)
        logprobs = logprobs.masked_fill(logprobs < threshold, -float("inf"))
    return (logprobs / temperature).softmax(-1)


def select_device(torch, requested="auto"):
    if requested.startswith("xpu"):
        raise ValueError(
            "XPU training requires memory telemetry and is not yet supported"
        )
    if requested != "auto":
        return torch.device("cuda" if requested == "rocm" else requested)
    if torch.cuda.is_available():  # ROCm exposes the same torch device API.
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free_memory(torch, device):
    """Live free bytes; no GPU-name or model-name capacity tables."""
    if device.type in {"cuda", "xpu"}:
        return int(getattr(torch, device.type).mem_get_info(device)[0])
    if device.type == "mps":
        return max(
            0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
        )
    import psutil

    return psutil.virtual_memory().available


def repage(optimizer):
    """After a bitsandbytes paged optimizer loads a checkpoint.

    torch.save keeps a tensor's Python attributes, so a state saved from paged
    (managed) memory comes back flagged `is_paged` although torch.load put it
    in ordinary device memory. The optimizer then prefetches it as managed
    memory and the CUDA call fails: "Error invalid argument at line 578 in file
    pythonInterface.cpp", which ends the process (an Inkling-2-Mini Deep Dream
    resumed after a slice deadline). Such states are copied into paged memory
    again, or lose the flag where paged memory cannot be had."""
    for values in optimizer.state.values():
        for key, value in values.items():
            if not getattr(value, "is_paged", False):
                continue
            try:
                from bitsandbytes.functional import get_paged

                paged = get_paged(*value.shape, dtype=value.dtype, device=value.device)
                paged.copy_(value)
            except Exception:
                paged = value
                paged.is_paged = False
            values[key] = paged


def needs_4bit(parameter_count, available_bytes, memory_fraction):
    if available_bytes <= 0 or not 0 < memory_fraction <= 1:
        raise ValueError(
            "Loading requires measured free memory and a valid memory_fraction"
        )
    return parameter_count * 2 > available_bytes * memory_fraction


class TorchBackend:
    kind = "torch"

    def __init__(self, config, check=lambda: None, train=True):
        # Before CUDA initialises: expandable segments return the space that
        # varying sequence lengths fragment (5.9 GB on a 4k-window Inkling-2
        # step, the difference between running out of memory and training).
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.torch, self.config, self.check = torch, config, check
        options = getattr(config, "torch_options", {})
        self.options = TorchOptions(**options) if isinstance(options, dict) else options
        if self.options.temperature < 0 or not 0 <= self.options.top_p <= 1:
            raise ValueError(
                "temperature must be nonnegative and top_p must be in [0, 1]"
            )
        torch.manual_seed(self.options.seed)
        self.device = select_device(torch, self.options.device)
        self._process, self._peak_rss = None, 0
        if self.device.type == "cpu":
            import psutil

            self._process = psutil.Process()
            self._user_check, self.check = check, self._checked
        self._generator = torch.Generator(
            device=self.device if self.device.type != "mps" else "cpu"
        )
        self._generator.manual_seed(self.options.seed)
        if (
            train
            and self.device.type == "cpu"
            and not (self.options.force_cpu or config.force_cpu_training)
        ):
            raise ValueError(
                "CPU is serve-only; set torch_options.force_cpu=true to train explicitly"
            )
        self.check()
        model_config = AutoConfig.from_pretrained(
            config.model, local_files_only=True, trust_remote_code=False
        )
        from . import packed_experts

        # Experts that ship quantized stay packed (compact base, like 4-bit).
        self.packed_experts = packed_experts.supports(model_config) and (
            self.device.type == "cuda"
        )
        self.available_bytes = free_memory(torch, self.device)
        if self.packed_experts:
            self.quantized = True  # the base is compact; the memory savers apply
        else:
            with torch.device("meta"):
                skeleton = AutoModelForCausalLM.from_config(
                    model_config, trust_remote_code=False
                )
            parameters = sum(p.numel() for p in skeleton.parameters())
            del skeleton
            self.quantized = needs_4bit(
                parameters, self.available_bytes, config.memory_fraction
            )
        dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        if (
            self.device.type == "cpu"
            and parameters * 4 > self.available_bytes * config.memory_fraction
        ):
            raise MemoryError(
                "The float32 CPU base exceeds the measured free-memory budget"
            )
        if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            dtype = torch.float16  # Capability probe, not a GPU-name table.
        kwargs = {"dtype": dtype, "local_files_only": True, "trust_remote_code": False}
        if self.quantized and not self.packed_experts:
            if (
                self.device.type not in {"cuda", "xpu"}
                or importlib.util.find_spec("bitsandbytes") is None
            ):
                raise MemoryError(
                    f"Base needs {parameters * 2} bf16 bytes; only {self.available_bytes} bytes free "
                    "within the configured memory_fraction. Install a bitsandbytes build for this accelerator "
                    "or free memory before loading."
                )
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": self.device}
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, local_files_only=True, trust_remote_code=False
        )
        if self.packed_experts:
            self.model = packed_experts.load(
                config.model,
                self.device,
                dtype,
                experts_on_host=self._auto(self._option("experts_on_host")),
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        if not self.quantized:
            self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        if train:
            # 4-bit bases are not passed through peft's
            # prepare_model_for_kbit_training: it upcasts every unquantized
            # weight to fp32, the frozen vocabulary matrices included (5 GB more
            # than bf16 on a 27B with a 248k vocabulary, live RTX 5090). Norms
            # stay in the compute dtype too: an fp32 final norm feeds fp32 into
            # the bf16 output head of models whose norm returns its own dtype
            # (Qwen2.5: "expected mat1 and mat2 to have the same dtype", live
            # A100). fp32 norms only matter for fp16 stability.
            self.embedding_on_host = self._auto(self.options.embedding_on_host) and (
                self._embedding_to_host()
            )
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=config.rank,
                    lora_alpha=config.rank * config.lora_scale,
                    lora_dropout=0.0,
                    target_modules=(
                        packed_experts.LORA_TARGETS
                        if self.packed_experts
                        else "all-linear"
                    ),
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            if self.options.gradient_checkpointing:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        self.trainable = train
        # Set by train_step when a prefix cache cannot be kept under checkpointing.
        self._window_only = False
        self.model.eval()
        self._optimizer = None
        self._schedule = None
        self._updates = 0
        # Transformers uses installed FLA kernels for supported recurrent models;
        # its inference-only kernels are kept out of the training dispatch.
        self._validate_head()
        if any(
            p.requires_grad for p in self._causal().get_output_embeddings().parameters()
        ):
            raise ValueError("The chunked head must remain frozen")

    def _auto(self, value):
        if value not in {"auto", "on", "off"}:
            raise ValueError(f"Expected auto, on or off, not {value!r}")
        if value == "auto":
            device = getattr(self, "device", None)
            return bool(getattr(self, "quantized", False)) and (
                device is not None and device.type == "cuda"
            )
        return value == "on"

    def _option(self, name):
        return getattr(getattr(self, "options", None) or TorchOptions(), name)

    def _embedding_to_host(self):
        """Move the input embedding to host RAM; only token ids go in and one
        hidden row per token comes back. Not for tied embeddings, whose matrix
        is also the output head that runs on the accelerator."""
        torch = self.torch
        embedding = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()
        if embedding is None or self.device.type == "cpu" or (
            head is not None and head.weight.data_ptr() == embedding.weight.data_ptr()
        ):
            return False
        embedding.to("cpu")
        device = self.device

        def to_host(module, args):
            return tuple(a.to("cpu") if torch.is_tensor(a) else a for a in args)

        def to_device(module, args, output):
            return output.to(device, non_blocking=True)

        embedding.register_forward_pre_hook(to_host)
        embedding.register_forward_hook(to_device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return True

    def _checked(self):
        self._user_check()
        self._peak_rss = max(self._peak_rss, self._process.memory_info().rss)

    def _causal(self):
        return self.model.get_base_model() if self.trainable else self.model

    def boundary(self):
        """A unit boundary (a training step, a gate row). The worker gives
        each backend its own version that also checks the slice deadline,
        which check() inside a unit leaves out."""
        self.check()

    @contextlib.contextmanager
    def base(self):
        was_training = self.model.training
        self.model.eval()
        ctx = (
            self.model.disable_adapter() if self.trainable else contextlib.nullcontext()
        )
        try:
            with ctx, self.torch.no_grad():
                yield
        finally:
            self.model.train(was_training)

    def _host_copy(self, tensor):
        """A copy in host RAM (pinned on CUDA for fast transfers back)."""
        copy = tensor.detach().to("cpu", copy=True)
        device = getattr(self, "device", None)
        return copy.pin_memory() if device is not None and device.type == "cuda" else copy

    def snapshot_teacher(self):
        """The teacher for this cycle is the version it starts from (DESIGN 3.3).

        Kept in host RAM: on accelerators two adapter-sized copies (this one
        and the student's while the teacher runs) would otherwise sit in GPU
        memory for the whole Dream, 3.7 GB on a 27B at rank 64.
        """
        self.teacher_weights = {
            n: self._host_copy(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    @contextlib.contextmanager
    def teacher(self):
        weights = getattr(self, "teacher_weights", None)
        if not weights:
            with self.base():
                yield
            return
        was_training = self.model.training
        self.model.eval()
        params = dict(self.model.named_parameters())
        current = {n: self._host_copy(params[n]) for n in weights}
        try:
            with self.torch.no_grad():
                for n, w in weights.items():
                    params[n].copy_(w, non_blocking=True)
                yield
        finally:
            with self.torch.no_grad():
                for n, w in current.items():
                    params[n].copy_(w, non_blocking=True)
            device = getattr(self, "device", None)
            if device is not None and device.type == "cuda":
                self.torch.cuda.synchronize(device)
            self.model.train(was_training)

    @staticmethod
    def _cache(output):
        """The model cache of an output: past_key_values, or cache (Inkling-2)."""
        cache = getattr(output, "past_key_values", None)
        return cache if cache is not None else getattr(output, "cache", None)

    # Reads with no gradient (the frozen teacher, a prompt before sampling)
    # go in chunks of this many tokens; None: config.teacher_chunk, the
    # size training is measured with. The gate sets bigger chunks (worker:
    # GATE_READ_CHUNK), halved on running out of memory (_bounded), never
    # below teacher_chunk: on Inkling-2-Mini each chunk unpacks its experts again,
    # and 4096-token chunks read 4.5x faster than 512 (H100, 2026-11-23).
    read_chunk = None

    def _chunk(self):
        return self.read_chunk or self.config.teacher_chunk

    def _bounded(self, read):
        """read(), retried with half the chunk when it runs out of memory."""
        oom = getattr(self.torch, "OutOfMemoryError", None) or self.torch.cuda.OutOfMemoryError
        while True:
            try:
                return read()
            except oom:
                if not self.read_chunk or self.read_chunk <= self.config.teacher_chunk:
                    raise
                self.read_chunk = max(self.config.teacher_chunk, self.read_chunk // 2)
                if self.device.type == "cuda":
                    self.torch.cuda.empty_cache()
                import logging

                logging.getLogger("songe").info("read_chunk_reduced %d", self.read_chunk)

    def prefix_state(self, ids):
        """Read ids with no gradient in chunks; keep only the model cache."""
        if not ids:
            return None
        return self._bounded(lambda: self._prefix_state(ids))

    def _prefix_state(self, ids):
        past, chunk = None, self._chunk()
        with self.torch.no_grad():
            for start in range(0, len(ids), chunk):
                self.check()
                out = self._causal().base_model(
                    input_ids=self.torch.tensor(
                        [ids[start : start + chunk]],
                        device=self.device,
                    ),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = self._cache(out)
        return past

    @contextlib.contextmanager
    def frozen(self, past):
        """Let checkpointed layers read a prefix cache; yields whether they can.

        Transformers takes the cache away from a checkpointed layer in
        training, because the layer would add to it a second time when the
        backward pass runs the layer again. Here every addition is undone as
        soon as it returns, so both runs read the same prefix. That includes
        the convolution and recurrent states of hybrid layers (Inkling-2),
        which are overwritten in place: left changed, the backward pass ran
        the layer on other states than the forward pass, routed some tokens
        to other experts and failed (CheckpointError). Caches of another kind
        (recurrent state alone) and layers transformers does not checkpoint
        itself are left alone: the caller then trains on the window only.
        """
        from transformers.cache_utils import DynamicLayer
        from transformers.modeling_layers import GradientCheckpointingLayer

        layers = [
            m
            for m in self._causal().modules()
            if isinstance(m, GradientCheckpointingLayer) and m.gradient_checkpointing
        ]
        states = getattr(past, "layers", None)
        if (
            not layers
            or not states
            or not all(hasattr(m, "_can_checkpoint_with_cache") for m in layers)
            or not all(isinstance(c, DynamicLayer) for c in states)
        ):
            yield False
            return

        torch = self.torch

        def snapshot(value):
            # Keys and values are replaced, never written in place: kept by
            # reference. Per-state dicts (conv and recurrent states, their
            # flags) are written in place: copied, tensors included (small).
            if isinstance(value, dict):
                return {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            return value

        def read_only(state):
            for name in UPDATES:
                update = getattr(state, name, None)
                if update is None:
                    continue

                def reading(*args, _update=update, **kwargs):
                    before = {k: snapshot(v) for k, v in vars(state).items()}
                    try:
                        return _update(*args, **kwargs)
                    finally:
                        vars(state).clear()
                        vars(state).update(before)

                setattr(state, name, reading)

        try:
            for state in states:
                read_only(state)
            for m in layers:
                m._can_checkpoint_with_cache = True
            yield True
        finally:
            for m in layers:
                vars(m).pop("_can_checkpoint_with_cache", None)
            for state in states:
                for name in UPDATES:
                    vars(state).pop(name, None)

    def hidden(self, ids):
        inputs = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        return (
            self._causal()
            .base_model(input_ids=inputs, use_cache=False, return_dict=True)
            .last_hidden_state
        )

    def head(self, hidden):
        model = self._causal()
        logits = model.get_output_embeddings()(hidden)
        scaling = getattr(model, "logits_scaling", None)
        if scaling:
            logits = logits / scaling
        softcap = getattr(model, "logit_softcap", None)
        if callable(softcap):  # Inkling-2: its own softcap, on fp32 logits
            return softcap(logits.float())
        config = model.config
        cap = getattr(config, "final_logit_softcapping", None) or getattr(
            getattr(config, "text_config", None), "final_logit_softcapping", None
        )
        if cap:
            logits = self.torch.tanh(logits / cap) * cap
        return logits

    def _validate_head(self):
        ids = self.tokenizer.encode("Hello", add_special_tokens=True)
        if not ids:
            raise ValueError("Tokenizer produced an empty prompt")
        with self.base():
            inputs = self.torch.tensor([ids], device=self.device)
            official = self.model(input_ids=inputs, use_cache=False).logits[:, -1:]
            chunked = self.head(self.hidden(ids)[:, -1:])
            difference = (official.float() - chunked.float()).abs().max().item()
        if not math.isfinite(difference) or difference > self.options.head_tolerance:
            raise ValueError(
                "Model output head differs from transformers; training refused"
            )

    def sample_full_length(self, prompt_ids, seed=None):
        generation = self._causal().generation_config
        original = generation.eos_token_id
        try:
            generation.eos_token_id = []
            return type(self).sample(self, prompt_ids, seed)
        finally:
            generation.eos_token_id = original

    def sample(self, prompt_ids, seed=None, *, deterministic=False, max_tokens=None, state=None):
        """state: prompt_state(prompt_ids), consumed (the prompt is not read again)."""
        torch = self.torch
        if not prompt_ids:
            raise ValueError("Sampling requires a nonempty prompt")
        generator = self._generator
        if seed is not None:
            generator = torch.Generator(
                device=self.device if self.device.type != "mps" else "cpu"
            )
            generator.manual_seed(seed)
        tokens, past = [], None
        inputs = torch.tensor([prompt_ids], device=self.device)
        was_training = self.model.training
        self.model.eval()
        # perf_counter: Windows' monotonic clock ticks every ~15.6 ms, so a
        # short answer could take "0 seconds" and divide by zero.
        started = time.perf_counter()
        if state is not None or len(prompt_ids) > self._chunk():
            # A long prompt is read in chunks, as the teacher reads it: one
            # pass over the whole prompt can need far more memory than the
            # training step (quadratic attention, Inkling-2).
            past = state if state is not None else self.prefix_state(prompt_ids[:-1])
            inputs = torch.tensor([prompt_ids[-1:]], device=self.device)
        prompt_seconds = 0.0
        eos = self._causal().generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        try:
            with torch.no_grad():
                for _ in range(
                    self.config.response_tokens if max_tokens is None else max_tokens
                ):
                    self.check()
                    out = self.model(
                        input_ids=inputs, past_key_values=past, use_cache=True
                    )
                    logits = out.logits[:, -1].float()
                    if deterministic or self.options.temperature <= 0:
                        token = logits.argmax(-1, keepdim=True)
                    else:
                        probabilities = sampling_probabilities(
                            logits, self.options.temperature, self.options.top_p
                        )
                        # MPS generators live on CPU; only a vocabulary vector crosses.
                        sample_device = (
                            "cpu" if self.device.type == "mps" else self.device
                        )
                        token = torch.multinomial(
                            probabilities.to(sample_device), 1, generator=generator
                        ).to(self.device)
                    value = int(token.item())
                    if not tokens:
                        prompt_seconds = time.perf_counter() - started
                    tokens.append(value)
                    past, inputs = self._cache(out), token
                    if value in eos:
                        break
        finally:
            self.model.train(was_training)
        if not tokens:
            raise ValueError("Model generated no tokens")
        seconds = time.perf_counter() - started
        # Never a zero span, whatever the clock's resolution.
        tick = 1e-6
        return (
            tokens,
            self.tokenizer.decode(tokens, skip_special_tokens=True),
            Stats(
                len(prompt_ids),
                len(tokens),
                seconds,
                (len(tokens) - 1) / max(seconds - prompt_seconds, tick)
                if len(tokens) > 1
                else len(tokens) / max(seconds, tick),
                prompt_seconds,
            ),
        )

    def _response_hidden(self, prompt, y):
        if not prompt or not y:
            raise ValueError("Prompt and response must be nonempty")
        return self.hidden(prompt + y[:-1])[:, response_positions(prompt, y)]

    def teacher_hidden(self, prompt, y, teacher=None):
        """Full frozen context in bounded prefill chunks, without an autograd graph.
        teacher: the context that selects the teacher's weights (default base)."""
        with teacher if teacher is not None else self.base():
            return self._bounded(lambda: self._frozen_hidden(prompt, y))

    def prompt_state(self, prompt):
        """The model cache after prompt[:-1], read in chunks with no
        gradient, for reading or sampling what follows more than once
        (kl_from, sample: state=); None for a prompt read in one pass.
        Every user continues from its own copy or is the last one."""
        if len(prompt) <= self._chunk():
            return None
        return self.prefix_state(prompt[:-1])

    def _frozen_hidden(self, prompt, y, state=None):
        """The hidden states that predict y: a short prompt in one pass; a
        long one in chunks (or from state), then prompt[-1:] + y[:-1] as a
        continuation. Every reader of a context uses this same layout, so
        identical weights read identically: chunk boundaries alone move
        Inkling-2-Mini's per-token KL by ~0.005 (bf16, MoE routing)."""
        if state is None:
            if len(prompt) <= self._chunk():
                return self._response_hidden(prompt, y).detach()
            state = self._prefix_state(prompt[:-1])
        rest, chunk = prompt[-1:] + y[:-1], self._chunk()
        past, parts = state, []
        for start in range(0, len(rest), chunk):
            self.check()
            out = self._causal().base_model(
                input_ids=self.torch.tensor([rest[start : start + chunk]], device=self.device),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = self._cache(out)
            parts.append(out.last_hidden_state.detach())
        return self.torch.cat(parts, dim=1)

    def compare(self, student_ids, teacher_ids, y, kind="forward_kl"):
        teacher_h = self.teacher_hidden(teacher_ids, y, self.teacher())
        with self.torch.no_grad():
            student_h = self._response_hidden(student_ids, y)
            total = 0.0
            for start in range(0, len(y), self.config.token_chunk):
                self.check()
                end = min(start + self.config.token_chunk, len(y))
                with self.base():
                    teacher = self.head(teacher_h[:, start:end])
                student = self.head(student_h[:, start:end])
                # Match MLX compare: default beta and full vocabulary for evaluation.
                # Measurement uses the true divergence; the clip is for training.
                total += float(divergence(student, teacher, kind)) * (end - start)
        return total / len(y)

    def teacher_gain(self, steps):
        """Same logged-success diagnostic as the MLX reference; never run tools."""
        from ..steps import tool_outcome

        wins, count = 0, 0
        for step in steps:
            if not step.get("answer_hint") or not any(
                tool_outcome(m) == "success"
                for m in step.get("hindsight", [])
                if m.get("role") == "tool"
            ):
                continue
            student, teacher = paired_prompts(
                self.tokenizer,
                step,
                self.config,
                check_length=False,
            )
            action = copy.deepcopy(step["answer_hint"])
            for call in action.get("tool_calls", []):
                arguments = call["function"].get("arguments")
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
            full = self.tokenizer.apply_chat_template(
                step["context"] + [action],
                tools=step.get("tools") or None,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
            )
            if full[: len(student)] != student:
                continue
            y = full[len(student) :][: self.config.response_tokens]
            if not y:
                continue
            with self.base():
                plain = self.token_logprob(self._response_hidden(student, y), y)
                hindsight = self.token_logprob(self._response_hidden(teacher, y), y)
            wins += hindsight > plain
            count += 1
        return {
            "rate": wins / count if count else None,
            "examples": count,
            "reason": "Explicit successful logged actions only",
        }

    def token_logprob(self, h, tokens):
        if not tokens or h.shape[1] != len(tokens):
            raise ValueError("One hidden state is required per response token")
        total = 0.0
        with self.torch.no_grad():
            for start in range(0, len(tokens), self.config.token_chunk):
                self.check()
                logits = self.head(
                    h[:, start : start + self.config.token_chunk]
                ).float()
                indices = self.torch.tensor(
                    tokens[start : start + self.config.token_chunk], device=self.device
                )[None, :, None]
                total += logits.log_softmax(-1).gather(-1, indices).sum().item()
        return total / len(tokens)

    def rollouts(self, steps, seeds):
        """Fresh answers for the samples of one optimizer step, made together.

        The weights do not change between the samples of a step, so their
        answers can come from one batch: one forward per new token for all of
        them instead of one per sample. A token costs about the same at batch
        1 and batch 4 (the weights are read once either way). Each row draws
        with its own seed's generator, as sample(prompt, seed) does. A prompt
        longer than one read chunk, or one that cannot be rendered, gets None
        and is answered by sample() as before; so does every row when the
        batched forward fails (no memory for it, or a model that refuses it)."""
        torch = self.torch
        prompts = []
        for step in steps:
            try:
                prompt = paired_prompts(self.tokenizer, step, self.config)[0]
            except Exception:
                prompt = None
            prompts.append(prompt if prompt and len(prompt) <= self._chunk() else None)
        rows = [i for i, p in enumerate(prompts) if p is not None]
        out = [None] * len(steps)
        if len(rows) < 2:
            return out
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = 0
        width = max(len(prompts[i]) for i in rows)
        # Left padding, masked: every row's next token is in the same column.
        inputs = torch.tensor(
            [[pad] * (width - len(prompts[i])) + prompts[i] for i in rows],
            device=self.device,
        )
        mask = torch.tensor(
            [[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in rows],
            device=self.device,
        )
        generators = []
        for i in rows:
            generator = torch.Generator(
                device=self.device if self.device.type != "mps" else "cpu"
            )
            generator.manual_seed(seeds[i])
            generators.append(generator)
        eos = self._causal().generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        tokens = [[] for _ in rows]
        done = [False] * len(rows)
        past = None
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for _ in range(self.config.response_tokens):
                    self.check()
                    try:
                        result = self.model(
                            input_ids=inputs,
                            attention_mask=mask,
                            past_key_values=past,
                            use_cache=True,
                        )
                    except Exception as error:
                        # The batch is only a shortcut: memory it did not find
                        # (MPS says so with a RuntimeError) or a model that
                        # refuses a padded batch leaves every row to sample().
                        # Stops from check() above are not caught here.
                        past = None
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        import logging

                        logging.getLogger("songe").info(
                            "rollouts_alone %s", type(error).__name__
                        )
                        return out
                    logits = result.logits[:, -1].float()
                    past, chosen = self._cache(result), []
                    for row, generator in enumerate(generators):
                        if done[row]:
                            chosen.append(pad)
                            continue
                        if self.options.temperature <= 0:
                            value = int(logits[row].argmax().item())
                        else:
                            probabilities = sampling_probabilities(
                                logits[row : row + 1],
                                self.options.temperature,
                                self.options.top_p,
                            )
                            value = int(
                                torch.multinomial(
                                    probabilities.to(generator.device),
                                    1,
                                    generator=generator,
                                ).item()
                            )
                        tokens[row].append(value)
                        chosen.append(value)
                        done[row] = value in eos
                    if all(done):
                        break
                    inputs = torch.tensor([[t] for t in chosen], device=self.device)
                    mask = torch.cat([mask, mask.new_ones((len(rows), 1))], dim=1)
        finally:
            self.model.train(was_training)
        for row, i in enumerate(rows):
            out[i] = tokens[row] or None
        return out

    def train_step(self, step, optimizer=None, seed=None, rollout=None):
        """Mirror of the MLX step (DESIGN 3.3-3.4): fresh rollout, frozen
        hindsight teacher, gradient over the context tail plus the response.
        rollout: this sample's answer, made in advance by rollouts()."""
        from torch.utils.checkpoint import checkpoint

        if not self.trainable:
            raise ValueError("Backend was opened for serving only")
        config = self.config
        prompt, teacher_prompt = paired_prompts(self.tokenizer, step, config)
        y, forced = lesson_tokens(self, step, prompt, seed, rollout)
        teacher_context = self.teacher() if step.get("hindsight") else self.base()
        teacher_h = self.teacher_hidden(teacher_prompt, y, teacher_context)
        self.check()
        split = min(
            len(prompt) - 1, max(0, len(prompt) - max(1, config.max_sequence - len(y)))
        )
        # The context before the window is read with no gradient and kept only
        # as model state, as on MLX: the student sees what the teacher sees.
        # Under gradient checkpointing transformers would drop that state in
        # training mode, so it is frozen for the layers to read (see frozen).
        # Where it cannot be, the student sees only the window.
        checkpointing = getattr(self._causal(), "is_gradient_checkpointing", False)
        past = (
            self.prefix_state(prompt[:split])
            if split and not (checkpointing and getattr(self, "_window_only", False))
            else None
        )
        keeping = (
            self.frozen(past)
            if checkpointing and past is not None
            else contextlib.nullcontext(True)
        )
        tail = prompt[split:]
        self.model.train()
        saving = (
            self.torch.autograd.graph.save_on_cpu(pin_memory=True)
            if self._auto(self._option("offload_activations"))
            else contextlib.nullcontext()
        )
        with contextlib.ExitStack() as stack:
            stack.callback(self.model.eval)
            stack.enter_context(saving)
            # Entered for the backward pass too: it runs the layers again.
            if not stack.enter_context(keeping):
                # Remembered, so later steps do not read a prefix for nothing.
                self._window_only, past = True, None
            inputs = self.torch.tensor([tail + y[:-1]], device=self.device)
            hidden = (
                self._causal()
                .base_model(
                    input_ids=inputs,
                    past_key_values=past,
                    use_cache=past is not None,
                    return_dict=True,
                )
                .last_hidden_state[:, response_positions(tail, y)]
            )
            total = hidden.new_zeros((), dtype=self.torch.float32)

            def head_loss(student_h, teacher_h):
                self.check()
                with self.base():
                    teacher = self.head(teacher_h)
                return divergence(
                    self.head(student_h),
                    teacher,
                    config.loss,
                    config.beta,
                    config.teacher_top_k,
                    # The clip bounds what a rollout's stray tokens can add. On the
                    # teacher's own answer the largest terms are the lesson itself.
                    clip=0.0 if forced else config.vocab_clip,
                )

            for start in range(0, len(y), config.token_chunk):
                end = min(start + config.token_chunk, len(y))
                value = checkpoint(
                    head_loss,
                    hidden[:, start:end],
                    teacher_h[:, start:end],
                    use_reentrant=False,
                )
                total = total + value * ((end - start) / len(y))
            params = [
                (n, p) for n, p in self.model.named_parameters() if p.requires_grad
            ]
            grads = self.torch.autograd.grad(total, [p for _, p in params])
            return (
                total.item(),
                {
                    name: grad.detach()
                    for (name, _), grad in zip(params, grads, strict=True)
                },
                len(y),
            )

    def optimizer(self, schedule):
        import torch

        self._schedule = schedule
        self._updates = 0
        lr = schedule(0) if callable(schedule) else float(schedule)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        choice = self._option("optimizer")
        if choice == "auto":
            choice = "paged_adamw_8bit" if self._auto("auto") else "adamw"
        if choice == "paged_adamw_8bit":
            # Two 8-bit states per adapter weight instead of two fp32 ones, in
            # paged memory that spills to host RAM under a spike rather than
            # failing the step.
            import bitsandbytes as bnb

            self._optimizer = bnb.optim.PagedAdamW8bit(
                parameters, lr=lr, weight_decay=self.options.weight_decay
            )
        elif choice == "adamw":
            self._optimizer = torch.optim.AdamW(
                parameters, lr=lr, weight_decay=self.options.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {choice}")
        return self._optimizer

    def apply(self, grads, optimizer=None):
        if optimizer is not None:
            self._optimizer = optimizer
        if self._optimizer is None:
            raise ValueError("Call optimizer(schedule) before apply(grads)")
        parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if parameters.keys() != grads.keys():
            raise ValueError("Gradient names differ from trainable parameters")
        if not all(self.torch.isfinite(g).all().item() for g in grads.values()):
            raise ValueError("Non-finite training gradient")
        self.check()
        self._optimizer.zero_grad(set_to_none=True)
        for name, param in parameters.items():
            param.grad = grads[name].to(device=param.device, dtype=param.dtype)
        norm = self.torch.nn.utils.clip_grad_norm_(
            parameters.values(), self.options.max_grad_norm, error_if_nonfinite=True
        )
        if callable(self._schedule):
            for group in self._optimizer.param_groups:
                group["lr"] = self._schedule(self._updates)
        self._optimizer.step()
        self._updates += 1
        self._optimizer.zero_grad(set_to_none=True)
        return float(norm)

    def save(self, path):
        if not self.trainable:
            raise ValueError("Serving-only backend has no adapter")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        for file in path.iterdir():
            if file.is_file():
                with file.open("r+b") as handle:
                    os.fsync(handle.fileno())
        if os.name != "nt":
            handle = os.open(path, os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)

    def restore(self, path):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file

        if not self.trainable:
            raise ValueError("Serving-only backend has no adapter")
        weights = load_file(str(Path(path) / "adapter_model.safetensors"), device="cpu")
        expected = get_peft_model_state_dict(self.model)
        if weights.keys() != expected.keys() or any(
            weights[n].shape != expected[n].shape for n in expected
        ):
            raise ValueError(
                "Adapter checkpoint does not match this model's trainable weights"
            )
        set_peft_model_state_dict(self.model, weights)

    def prepare(self):
        # Checkpointing is configured at construction, before head validation.
        self.check()

    def peak_memory(self):
        if self.device.type in {"cuda", "xpu"}:
            return int(
                getattr(self.torch, self.device.type).max_memory_allocated(self.device)
            )
        if self.device.type == "mps":
            return int(self.torch.mps.driver_allocated_memory())
        self._peak_rss = max(self._peak_rss, self._process.memory_info().rss)
        return self._peak_rss

    def reset_peak_memory(self):
        if self.device.type in {"cuda", "xpu"}:
            getattr(self.torch, self.device.type).reset_peak_memory_stats(self.device)
        elif self._process is not None:
            self._peak_rss = self._process.memory_info().rss

    def kl_from(self, base, prompt, teacher_prompt, y, *, hindsight=False, views=None, state=None):
        return self.measure(
            base, prompt, teacher_prompt, y, hindsight=hindsight, views=views, state=state
        )["kl"]

    def measure(self, base, prompt, teacher_prompt, y, *, hindsight=False, views=None, state=None):
        """This version against the reference along y, per answer token: the
        KL, and what a mean KL does not show: each side's entropy (nats) and
        how often their most likely token is the same. From the logits the
        KL is computed on, so it costs no forward pass.

        views: a dict kept across the versions one gate measures. The
        reference's view of a moment (the frozen base, or the cycle's
        teacher) is the same for every version, so it is read once and kept
        in host RAM (a few MB per moment) instead of once per version.
        state: this version's prompt_state(prompt), continued from a copy
        (the caller samples from it next)."""
        reference = base.teacher if hindsight else base.base
        key = view_key(hindsight, teacher_prompt, y) if views is not None else None
        teacher_h = views.get(key) if key is not None else None
        if teacher_h is None:
            # Passed in, not entered here: teacher_hidden reads with the
            # original base unless it is given the context that selects the
            # weights.
            teacher_h = base.teacher_hidden(teacher_prompt, y, reference())
            if key is not None:
                views[key] = teacher_h.to("cpu")
        elif teacher_h.device != self.device:
            teacher_h = teacher_h.to(self.device)  # kept in host RAM
        total = {"kl": 0.0, "entropy": 0.0, "reference_entropy": 0.0, "agreement": 0.0}

        def entropy(logits):
            # entr(0) is 0: a token with no probability adds nothing.
            return float(self.torch.special.entr(logits.float().softmax(-1)).sum())

        with self.torch.no_grad():
            # In chunks, as the teacher reads: one pass over a long context
            # was slower and twice the memory (Inkling-2-Mini at 32k: 72 s,
            # 45 GB, against 34 s, 24 GB in 4096-token chunks).
            if state is not None:
                candidate_h = self._bounded(
                    lambda: self._frozen_hidden(prompt, y, copy.deepcopy(state))
                )
            else:
                candidate_h = self.teacher_hidden(prompt, y, contextlib.nullcontext())
            for start in range(0, len(y), base.config.token_chunk):
                self.check()
                end = min(start + base.config.token_chunk, len(y))
                with reference():
                    teacher = base.head(teacher_h[:, start:end])
                student = self.head(candidate_h[:, start:end])
                total["kl"] += float(divergence(student, teacher)) * (end - start)
                total["entropy"] += entropy(student)
                total["reference_entropy"] += entropy(teacher)
                total["agreement"] += float(
                    (student.argmax(-1) == teacher.argmax(-1).to(student.device)).sum()
                )
        return {k: v / len(y) for k, v in total.items()}

    def adapter_weights(self):
        return {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def scale_delta(self, baseline, trained, scale):
        """Interpolate effective BA deltas, compressing before gate evaluation."""
        torch = self.torch
        weights = dict(trained)
        for key, a1 in trained.items():
            if ".lora_A." not in key:
                continue
            bkey = key.replace(".lora_A.", ".lora_B.")
            if scale in (0, 1):
                source = baseline if scale == 0 else trained
                weights[key], weights[bkey] = source[key], source[bkey]
                continue
            a0, b0, b1 = baseline[key], baseline[bkey], trained[bkey]
            left = torch.cat(((1 - scale) * b0, scale * b1), dim=-1).float()
            right = torch.cat((a0.T, a1.T), dim=-1).float()
            ql, rl = torch.linalg.qr(left)
            qr, rr = torch.linalg.qr(right)
            u, singular, vh = torch.linalg.svd(rl @ rr.T, full_matrices=False)
            rank = a1.shape[0]
            kept = min(rank, singular.shape[0])
            root = singular[:kept].sqrt()
            a = ((qr @ vh.T[:, :kept]) * root).T
            b = (ql @ u[:, :kept]) * root
            # Narrow layers (Qwen3.5 in_proj_a/b: 32 outputs at rank 64)
            # compress below the configured rank. Zero rows and columns keep
            # every factor its parameter's shape; the product is unchanged
            # (as in the MLX backend).
            if kept < rank:
                a = torch.cat((a, a.new_zeros(rank - kept, a.shape[1])))
                b = torch.cat((b, b.new_zeros(b.shape[0], rank - kept)), dim=1)
            weights[key] = a.to(a1.dtype)
            weights[bkey] = b.to(b1.dtype)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])

    def train(self, pool, anchors, work, progress=lambda *a: None):
        from .training import train

        return train(self, pool, anchors, work, progress)

    def layer_deltas(self, baseline=None):
        torch = self.torch
        lists = [
            m
            for m in self._causal().base_model.modules()
            if isinstance(m, torch.nn.ModuleList)
        ]
        layers = max(lists, key=len, default=[])
        names = {id(m): n for n, m in self.model.named_modules()}
        changes = []
        with torch.no_grad():
            for layer in layers:
                squared = 0.0
                for module in layer.modules():
                    if not hasattr(module, "lora_A"):
                        continue
                    for name in module.active_adapters:
                        if name not in module.lora_A:
                            continue
                        a = module.lora_A[name].weight.float()
                        b = module.lora_B[name].weight.float()
                        if baseline is not None:
                            prefix = names[id(module)]
                            a = torch.cat(
                                (baseline[f"{prefix}.lora_A.{name}.weight"].float(), a),
                                dim=0,
                            )
                            b = torch.cat(
                                (
                                    -baseline[f"{prefix}.lora_B.{name}.weight"].float(),
                                    b,
                                ),
                                dim=1,
                            )
                        squared += (
                            (a @ a.T) * (b.T @ b)
                        ).sum().item() * module.scaling[name] ** 2
                changes.append(math.sqrt(max(0.0, squared)))
        return changes


Backend = TorchBackend
