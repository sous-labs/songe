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

from ..sleep.prompts import paired_prompts, response_positions
from .reference import TAIL_FLOOR


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


@dataclass(frozen=True)
class Stats:
    prompt_tokens: int
    generation_tokens: int
    seconds: float
    generation_tps: float
    prompt_seconds: float = 0.0

    @property
    def prompt_tps(self):
        return self.prompt_tokens / (self.prompt_seconds or self.seconds)


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
            self.model = packed_experts.load(config.model, self.device, dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        if not self.quantized:
            self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        if train:
            if self.quantized:
                # What prepare_model_for_kbit_training does, except that it
                # upcasts every unquantized weight to fp32, the frozen
                # vocabulary matrices included: 5 GB more than bf16 on a 27B
                # with a 248k vocabulary (live, RTX 5090), and a load peak
                # that fragments the allocator. Only norms and biases (1-D)
                # are upcast here, for numerical stability, as peft does.
                for param in self.model.parameters():
                    if param.ndim == 1 and param.dtype in {torch.float16, torch.bfloat16}:
                        param.data = param.data.float()
            self.embedding_on_host = self._auto(self.options.embedding_on_host) and (
                self._embedding_to_host()
            )
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=config.rank,
                    lora_alpha=config.rank * config.lora_scale,
                    lora_dropout=0.0,
                    target_modules="all-linear",
                    exclude_modules=(
                        packed_experts.LORA_EXCLUDE if self.packed_experts else None
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

    def prefix_state(self, ids):
        """Read ids with no gradient in chunks; keep only the model cache."""
        if not ids:
            return None
        past = None
        with self.torch.no_grad():
            for start in range(0, len(ids), self.config.teacher_chunk):
                self.check()
                out = self._causal().base_model(
                    input_ids=self.torch.tensor(
                        [ids[start : start + self.config.teacher_chunk]],
                        device=self.device,
                    ),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
        return past

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

    def sample(self, prompt_ids, seed=None, *, deterministic=False, max_tokens=None):
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
        started = time.monotonic()
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
                        prompt_seconds = time.monotonic() - started
                    tokens.append(value)
                    past, inputs = out.past_key_values, token
                    if value in eos:
                        break
        finally:
            self.model.train(was_training)
        if not tokens:
            raise ValueError("Model generated no tokens")
        seconds = time.monotonic() - started
        return (
            tokens,
            self.tokenizer.decode(tokens, skip_special_tokens=True),
            Stats(
                len(prompt_ids),
                len(tokens),
                seconds,
                (len(tokens) - 1) / (seconds - prompt_seconds)
                if len(tokens) > 1
                else len(tokens) / seconds,
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
        ids = prompt + y[:-1]
        positions = response_positions(prompt, y)
        with teacher if teacher is not None else self.base():
            if len(ids) <= self.config.teacher_chunk:
                return self._response_hidden(prompt, y).detach()
            past, parts = None, []
            for start in range(0, len(ids), self.config.teacher_chunk):
                self.check()
                end = min(len(ids), start + self.config.teacher_chunk)
                out = self._causal().base_model(
                    input_ids=self.torch.tensor([ids[start:end]], device=self.device),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                if end > positions.start:
                    parts.append(
                        out.last_hidden_state[
                            :, max(0, positions.start - start) :
                        ].detach()
                    )
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

    def train_step(self, step, optimizer=None, seed=None):
        """Mirror of the MLX step (DESIGN 3.3-3.4): fresh rollout, frozen
        hindsight teacher, gradient over the context tail plus the response."""
        from torch.utils.checkpoint import checkpoint

        if not self.trainable:
            raise ValueError("Backend was opened for serving only")
        config = self.config
        prompt, teacher_prompt = paired_prompts(self.tokenizer, step, config)
        y, _, _ = self.sample(prompt, seed=seed)
        teacher_context = self.teacher() if step.get("hindsight") else self.base()
        teacher_h = self.teacher_hidden(teacher_prompt, y, teacher_context)
        self.check()
        split = min(
            len(prompt) - 1, max(0, len(prompt) - max(1, config.max_sequence - len(y)))
        )
        # Transformers drops the cache under gradient checkpointing in training
        # mode; there the student sees only the window (the teacher still reads
        # the full context), so a long agent context never exceeds the measured
        # backpropagation memory.
        checkpointing = getattr(self._causal(), "is_gradient_checkpointing", False)
        past = (
            self.prefix_state(prompt[:split]) if split and not checkpointing else None
        )
        tail = prompt[split:]
        self.model.train()
        saving = (
            self.torch.autograd.graph.save_on_cpu(pin_memory=True)
            if self._auto(self._option("offload_activations"))
            else contextlib.nullcontext()
        )
        try:
            saving.__enter__()
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
                    clip=config.vocab_clip,
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
        finally:
            saving.__exit__(None, None, None)
            self.model.eval()

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

    def kl_from(self, base, prompt, teacher_prompt, y, *, hindsight=False):
        reference = base.teacher if hindsight else base.base
        with reference():
            teacher_h = base.teacher_hidden(teacher_prompt, y)
        total = 0.0
        with self.torch.no_grad():
            candidate_h = self._response_hidden(prompt, y)
            for start in range(0, len(y), base.config.token_chunk):
                self.check()
                end = min(start + base.config.token_chunk, len(y))
                with reference():
                    teacher = base.head(teacher_h[:, start:end])
                value = divergence(self.head(candidate_h[:, start:end]), teacher)
                total += float(value) * (end - start)
        return total / len(y)

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
            root = singular[:rank].sqrt()
            weights[key] = ((qr @ vh.T[:, :rank]) * root).T.to(a1.dtype)
            weights[bkey] = ((ql @ u[:, :rank]) * root).to(b1.dtype)
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
