"""MLX backend: mlx-lm models, generation, adapters, and checkpointing.

Only this module imports MLX; importing or running the proxy never touches Metal.
The frozen teacher shares the original quantized weights with the LoRA student.
"""

import contextlib
import json
import math
import os
import uuid

from ..sleep.prompts import lesson_tokens, paired_prompts, response_positions
from ..steps import tool_outcome
from ..storage import private_dir, write_json
from .objective import divergence, view_key


class Backend:
    kind = "mlx"
    # TorchBackend reads in bigger chunks in the gate; MLX keeps
    # teacher_chunk: Metal running out of memory ends the process rather
    # than raising, so a chunk cannot be tried and halved.
    read_chunk = None

    def prompt_state(self, prompt):
        """No shared prompt read on MLX (TorchBackend.prompt_state): each
        user reads the prompt itself."""
        return None

    def __init__(self, config, check=lambda: None, train=True):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers

        self.mx, self.nn, self.optim = mx, nn, optim
        self.config, self.check = config, check
        from ..schedule import probe

        hardware = probe()
        if hardware.available:
            mx.set_memory_limit(
                int(
                    min(
                        hardware.total_gb * config.memory_fraction,
                        hardware.free_gb,
                    )
                    * 1e9
                )
            )
        mx.random.seed(7613)
        self.model, self.tokenizer = load(config.model)
        from ..sleep.compat import apply as make_trainable

        # Replace forward-only kernels the loaded architecture uses with
        # differentiable equivalents (Qwen3.5 family, attention sinks, ...).
        self.patches = make_trainable()
        self.model.eval()
        self.wrappers = []
        if train:
            self.model.freeze()
            self.adapter_config = {
                "fine_tune_type": "lora",
                "num_layers": len(self.model.layers),
                "lora_parameters": {
                    "rank": config.rank,
                    "scale": config.lora_scale,
                    "dropout": 0.0,
                },
            }
            linear_to_lora_layers(
                self.model,
                len(self.model.layers),
                self.adapter_config["lora_parameters"],
            )
            self.wrappers = [
                m for _, m in self.model.named_modules() if hasattr(m, "lora_a")
            ]
            if not self.wrappers:
                raise ValueError("This architecture has no supported LoRA layers")
        # Check the output head optimization against the official model forward
        # on a real tensor before ever using it in training.
        self._validate_head()

    def boundary(self):
        """A unit boundary (a training step, a gate row). The worker gives
        each backend its own version that also checks the slice deadline,
        which check() inside a unit leaves out."""
        self.check()

    @contextlib.contextmanager
    def base(self):
        was_training = self.model.training
        self.model.eval()
        scales = [m.scale for m in self.wrappers]
        for m in self.wrappers:
            m.scale = 0.0
        try:
            yield
        finally:
            for m, scale in zip(self.wrappers, scales, strict=False):
                m.scale = scale
            self.model.train(was_training)

    def snapshot_teacher(self):
        """The teacher for this cycle is the version it starts from (DESIGN 3.3).
        Before any adapter exists it is the original base."""
        from mlx.utils import tree_flatten

        self.teacher_weights = [
            (k, self.mx.array(v))
            for k, v in tree_flatten(self.model.trainable_parameters())
        ]

    @contextlib.contextmanager
    def teacher(self):
        weights = getattr(self, "teacher_weights", None)
        if not weights:
            with self.base():
                yield
            return
        from mlx.utils import tree_flatten

        was_training = self.model.training
        current = [(k, v) for k, v in tree_flatten(self.model.trainable_parameters())]
        self.model.eval()
        self.model.load_weights(weights, strict=False)
        try:
            yield
        finally:
            self.model.load_weights(current, strict=False)
            self.model.train(was_training)

    def _text_model(self):
        return getattr(self.model, "language_model", self.model)

    def prefix_state(self, ids):
        """Read ids with no gradient, in chunks; keep only the model state.

        Returns None when the model's cache kinds cannot be frozen this way;
        the caller then backpropagates through the whole prompt."""
        from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

        if not ids:
            return None
        cache = make_prompt_cache(self._text_model())
        if not all(type(c) in (KVCache, ArraysCache) for c in cache):
            return None
        was_training = self.model.training
        self.model.eval()
        try:
            for start in range(0, len(ids), self.config.teacher_chunk):
                self.check()
                self._text_model().model(
                    self.mx.array([ids[start : start + self.config.teacher_chunk]]),
                    cache=cache,
                )
                self.mx.eval([c.state for c in cache])
        finally:
            self.model.train(was_training)
        return cache

    def frozen(self, cache):
        """A fresh copy of a prefix state whose arrays are constants."""
        from mlx_lm.models.cache import ArraysCache, KVCache

        mx = self.mx

        class Frozen:
            def __init__(self, c):
                self.keys = mx.stop_gradient(c.keys[..., : c.offset, :])
                self.values = mx.stop_gradient(c.values[..., : c.offset, :])
                self.offset = c.offset

            def update_and_fetch(self, keys, values):
                self.offset += keys.shape[2]
                return (
                    mx.concatenate([self.keys, keys], axis=2),
                    mx.concatenate([self.values, values], axis=2),
                )

            def make_mask(self, *args, **kwargs):
                return KVCache.make_mask(self, *args, **kwargs)

            @property
            def state(self):
                return self.keys, self.values

        out = []
        for c in cache:
            if isinstance(c, KVCache):
                out.append(Frozen(c))
            else:
                copy = ArraysCache(len(c.cache))
                copy.cache = [
                    None if x is None else mx.stop_gradient(x) for x in c.cache
                ]
                out.append(copy)
        return out

    def hidden(self, ids):
        return self._text_model().model(self.mx.array([ids]))

    def teacher_hidden(self, prompt, y, teacher=None):
        """Frozen, cached prefill; keep hidden states only for response positions.
        teacher: the context that selects the teacher's weights (default base)."""
        mx = self.mx
        ids = prompt + y[:-1]
        positions = response_positions(prompt, y)
        with teacher if teacher is not None else self.base():
            if len(ids) <= self.config.teacher_chunk:
                result = mx.stop_gradient(self.hidden(ids)[:, positions])
                mx.eval(result)
                self.release()
                return result
            from mlx_lm.models.cache import make_prompt_cache

            cache = make_prompt_cache(self._text_model())
            parts = []
            for start in range(0, len(ids), self.config.teacher_chunk):
                self.check()
                end = min(len(ids), start + self.config.teacher_chunk)
                hidden = self._text_model().model(
                    mx.array([ids[start:end]]), cache=cache
                )
                mx.eval(hidden, [c.state for c in cache])
                if end > positions.start:
                    part = mx.stop_gradient(
                        hidden[:, max(0, positions.start - start) :]
                    )
                    parts.append(part)
                del hidden
            result = mx.concatenate(parts, axis=1)
            mx.eval(result)
            del parts, cache
            self.release()
            return result

    def head(self, hidden):
        model = self._text_model()
        # All supported families use a tied embedding or an explicit lm_head.
        tied = getattr(
            model,
            "tie_word_embeddings",
            getattr(model.args, "tie_word_embeddings", not hasattr(model, "lm_head")),
        )
        logits = (
            model.model.embed_tokens.as_linear(hidden)
            if tied
            else model.lm_head(hidden)
        )
        if hasattr(model, "logits_scaling"):
            logits = logits / model.logits_scaling
        cap = getattr(model, "final_logit_softcapping", None)
        if cap:
            logits = self.mx.tanh(logits / cap) * cap
        return logits

    def _validate_head(self):
        mx = self.mx
        ids = self.tokenizer.encode("Hello", add_special_tokens=True)
        official = self.model(mx.array([ids]))[:, -1:]
        chunked = self.head(self.hidden(ids)[:, -1:])
        difference = mx.max(
            mx.abs(official.astype(mx.float32) - chunked.astype(mx.float32))
        )
        mx.eval(difference)
        if float(difference) > 0.01:
            raise ValueError("Model output head differs from mlx-lm; training refused")

    def sample_full_length(self, prompt, seed=None):
        original = self.tokenizer.eos_token_ids
        try:
            self.tokenizer.eos_token_ids = set()
            return type(self).sample(self, prompt, seed)
        finally:
            self.tokenizer.eos_token_ids = original

    def sample(self, prompt, seed=None, *, deterministic=False, max_tokens=None):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        if seed is not None:
            self.mx.random.seed(seed)
        tokens, text, last = [], [], None
        self.model.eval()
        for result in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.config.response_tokens
            if max_tokens is None
            else max_tokens,
            sampler=make_sampler(temp=0.0 if deterministic else 0.8, top_p=0.95),
        ):
            self.check()
            tokens.append(result.token)
            text.append(result.text)
            last = result
        if not tokens:
            raise ValueError("Model generated no tokens")
        return tokens, "".join(text), last

    def compare(self, student_prompt, teacher_prompt, y, kind="forward_kl"):
        mx = self.mx
        teacher_h = self.teacher_hidden(teacher_prompt, y, self.teacher())
        mx.eval(teacher_h)
        student_h = self.hidden(student_prompt + y[:-1])[
            :, response_positions(student_prompt, y)
        ]
        mx.eval(student_h)
        total = 0.0
        for start in range(0, len(y), self.config.token_chunk):
            self.check()
            with self.base():
                teacher = self.head(
                    teacher_h[:, start : start + self.config.token_chunk]
                )
                mx.eval(teacher)
            student = self.head(student_h[:, start : start + self.config.token_chunk])
            # Measurement uses the true divergence; the clip is for training.
            value = divergence(mx, student, teacher, kind)
            mx.eval(value)
            total += float(value) * student.shape[1]
        return total / len(y)

    def teacher_gain(self, steps):
        """Later successful actions are evidence, not labels. No success inference
        from arbitrary tool output: require an explicit exit_code=0 observation.
        """
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
            import copy

            action = copy.deepcopy(step["answer_hint"])
            for call in action.get("tool_calls", []):
                args = call["function"].get("arguments")
                if isinstance(args, str):
                    call["function"]["arguments"] = json.loads(args)
            full = self.tokenizer.apply_chat_template(
                step["context"] + [action],
                tools=step.get("tools") or None,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
            )
            if full[: len(student)] != student:
                continue  # Cannot score a mismatched/truncated action prefix.
            y = full[len(student) :][: self.config.response_tokens]
            if not y:
                continue
            scores = []
            with self.base():
                for prompt in (student, teacher):
                    h = self.hidden(prompt + y[:-1])[:, response_positions(prompt, y)]
                    scores.append(self.token_logprob(h, y))
            wins += scores[1] > scores[0]
            count += 1
        return {
            "rate": wins / count if count else None,
            "examples": count,
            "reason": "Explicit successful logged actions only",
        }

    def kl_from(self, base, prompt, teacher_prompt, y, *, hindsight=False, views=None):
        return self.measure(base, prompt, teacher_prompt, y, hindsight=hindsight, views=views)["kl"]

    def measure(self, base, prompt, teacher_prompt, y, *, hindsight=False, views=None):
        """This version against the reference along y, per answer token: the
        KL, and what a mean KL does not show: each side's entropy (nats) and
        how often their most likely token is the same. From the logits the
        KL is computed on, so it costs no forward pass. views: as
        TorchBackend.measure (the reference's view, read once)."""
        reference = base.teacher if hindsight else base.base
        mx = self.mx

        def entropy(logits):
            logits = logits.astype(mx.float32)
            log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            p = mx.exp(log_p)
            # A token with no probability adds nothing (0 * -inf is not 0).
            return -float(mx.where(p > 0, p * log_p, 0.0).sum())

        key = view_key(hindsight, teacher_prompt, y) if views is not None else None
        teacher_h = views.get(key) if key is not None else None
        if teacher_h is None:
            # Passed in, not entered here: teacher_hidden reads with the
            # original base unless it is given the context that selects the
            # weights (and it evaluates the result before leaving that context).
            teacher_h = base.teacher_hidden(teacher_prompt, y, reference())
            if key is not None:
                views[key] = teacher_h
        candidate_h = self.hidden(prompt + y[:-1])[:, response_positions(prompt, y)]
        mx.eval(candidate_h)
        total = {"kl": 0.0, "entropy": 0.0, "reference_entropy": 0.0, "agreement": 0.0}
        for start in range(0, len(y), base.config.token_chunk):
            self.check()
            with reference():
                teacher = base.head(
                    teacher_h[:, start : start + base.config.token_chunk]
                )
                mx.eval(teacher)
            student = self.head(candidate_h[:, start : start + base.config.token_chunk])
            value = divergence(mx, student, teacher)
            mx.eval(value)
            total["kl"] += float(value) * student.shape[1]
            total["entropy"] += entropy(student)
            total["reference_entropy"] += entropy(teacher)
            total["agreement"] += float(
                (mx.argmax(student, axis=-1) == mx.argmax(teacher, axis=-1)).sum()
            )
        return {k: v / len(y) for k, v in total.items()}

    def token_logprob(self, h, tokens):
        mx, total = self.mx, 0.0
        for start in range(0, len(tokens), self.config.token_chunk):
            self.check()
            logits = self.head(h[:, start : start + self.config.token_chunk]).astype(
                mx.float32
            )
            lp = logits - mx.logsumexp(logits, -1, keepdims=True)
            value = mx.take_along_axis(
                lp,
                mx.array(tokens[start : start + self.config.token_chunk])[
                    None, :, None
                ],
                -1,
            ).sum()
            mx.eval(value)
            total += float(value)
        return total / len(tokens)

    def save(self, path):
        from mlx.utils import tree_flatten

        private_dir(path)
        # A directory is published only after all checkpoint files are durable.
        temporary = path / (".adapters-" + uuid.uuid4().hex + ".safetensors")
        try:
            self.mx.save_safetensors(
                str(temporary), dict(tree_flatten(self.model.trainable_parameters()))
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path / "adapters.safetensors")
        finally:
            temporary.unlink(missing_ok=True)
        config = dict(self.adapter_config)
        config["lora_parameters"] = dict(config["lora_parameters"])
        if self.wrappers:
            a = self.wrappers[0].lora_a
            config["lora_parameters"]["rank"] = (
                a.shape[-1] if a.ndim == 2 else a.shape[-2]
            )
        write_json(path / "adapter_config.json", config)

    def restore(self, path):
        weights = self.mx.load(str(path / "adapters.safetensors"))
        self.model.load_weights(list(weights.items()), strict=False)

    def train_step(self, step, optimizer, seed=None):
        """One fresh on-policy rollout scored by the frozen hindsight teacher
        (DESIGN 3.3-3.4). The gradient covers the longest tail of the context
        plus the response that fits max_sequence; the context before it is read
        with no gradient and kept only as model state. Only the output head and
        divergence run per chunk, under checkpointing.
        """
        mx = self.mx
        config = self.config
        prompt, teacher_prompt = paired_prompts(self.tokenizer, step, config)
        y, forced = lesson_tokens(self, step, prompt, seed, teacher_prompt)
        chunk = config.token_chunk
        spans = [(s, min(s + chunk, len(y))) for s in range(0, len(y), chunk)]
        # Hindsight steps learn from this cycle's teacher; general anchors keep
        # the original base as teacher, with no hindsight.
        teacher_context = self.teacher() if step.get("hindsight") else self.base()
        teacher_h = self.teacher_hidden(teacher_prompt, y, teacher_context)
        with self.base():
            teacher_logits = [
                mx.stop_gradient(self.head(teacher_h[:, s:e])) for s, e in spans
            ]
            mx.eval(teacher_logits)
        del teacher_h
        self.check()

        split = min(
            len(prompt) - 1, max(0, len(prompt) - max(1, config.max_sequence - len(y)))
        )
        state = self.prefix_state(prompt[:split]) if split else None
        tail = prompt[split:] if state is not None else prompt
        positions = response_positions(tail, y)

        def hidden():
            ids = mx.array([tail + y[:-1]])
            cache = self.frozen(state) if state is not None else None
            return self._text_model().model(ids, cache=cache)[:, positions]

        def head_divergence(hidden, teacher):
            return divergence(
                mx,
                self.head(hidden),
                teacher,
                config.loss,
                config.beta,
                config.teacher_top_k,
                # The clip bounds what a rollout's stray tokens can add. On the
                # teacher's own answer the largest terms are the lesson itself.
                clip=0.0 if forced else config.vocab_clip,
            )

        checkpointed = mx.checkpoint(head_divergence)

        def loss_fn(model):
            states = hidden()
            total = 0.0
            for (s, e), teacher in zip(spans, teacher_logits, strict=True):
                total = total + checkpointed(states[:, s:e], teacher) * (
                    (e - s) / len(y)
                )
            return total

        # Recurrent architectures select differentiable operations only in
        # training mode. Sampling and the frozen teacher always use eval.
        self.model.train()
        try:
            value, grads = self.nn.value_and_grad(self.model, loss_fn)(self.model)
            mx.eval(value, grads)
        finally:
            self.model.eval()
        # The shared loop weights this sample mean by its response-token count.
        return float(value), grads, len(y)

    def prepare(self):
        from mlx_lm.tuner.trainer import grad_checkpoint

        if not getattr(self, "_prepared", False):
            grad_checkpoint(self.model.layers[0])
            self._prepared = True

    def peak_memory(self):
        return self.mx.get_peak_memory()

    def reset_peak_memory(self):
        self.mx.reset_peak_memory()

    def optimizer(self, schedule):
        # MLX optimizers cast the learning rate with astype; schedules may
        # return Python floats (math.cos), so keep every value an mx.array.
        rate = (
            (lambda step: self.mx.array(schedule(step), self.mx.float32))
            if callable(schedule)
            else schedule
        )
        return self.optim.AdamW(learning_rate=rate)

    def apply(self, grads, optimizer):
        optimizer.update(self.model, grads)
        self.mx.eval(self.model.parameters(), optimizer.state)
        self.release()

    def release(self):
        # MLX keeps freed buffers in its own cache; the process then looks (and
        # is) much larger than the live working set, which the guard rightly
        # reads as pressure. Return them after each heavy phase.
        clear = getattr(self.mx, "clear_cache", None) or getattr(
            getattr(self.mx, "metal", None), "clear_cache", None
        )
        if clear:
            clear()

    def adapter_weights(self):
        from mlx.utils import tree_flatten

        return {
            k: self.mx.array(v)
            for k, v in tree_flatten(self.model.trainable_parameters())
        }

    def scale_delta(self, baseline, trained, scale):
        """Scale effective weight changes, recompress to the configured rank.

        QR/SVD operates on rank-sized factors. Compression happens before gate
        evaluation, so any approximation is tested rather than assumed safe.
        """
        mx, weights = self.mx, dict(trained)
        for key, value in trained.items():
            if not key.endswith("lora_a"):
                continue
            bkey = key[:-6] + "lora_b"
            if scale in (0, 1):
                source = baseline if scale == 0 else trained
                weights[key], weights[bkey] = source[key], source[bkey]
                continue
            a0, a1, b0, b1 = baseline[key], value, baseline[bkey], trained[bkey]
            # Standard LoRA: x @ A @ B. Switch LoRA stores transposed factors.
            right0, right1 = (
                (a0, a1)
                if value.ndim == 2
                else (a0.swapaxes(-1, -2), a1.swapaxes(-1, -2))
            )
            left0, left1 = (b0.T, b1.T) if value.ndim == 2 else (b0, b1)
            left = mx.concatenate([(1 - scale) * left0, scale * left1], axis=-1).astype(
                mx.float32
            )
            right = mx.concatenate([right0, right1], axis=-1).astype(mx.float32)
            ql, rl = mx.linalg.qr(left, stream=mx.cpu)
            qr, rr = mx.linalg.qr(right, stream=mx.cpu)
            u, singular, vh = mx.linalg.svd(rl @ rr.swapaxes(-1, -2), stream=mx.cpu)
            rank = min(self.config.rank, singular.shape[-1])
            root = mx.sqrt(singular[..., :rank])
            left_factor = (ql @ u[..., :rank]) * root[..., None, :]
            right_factor = (qr @ vh.swapaxes(-1, -2)[..., :rank]) * root[..., None, :]
            # Narrow layers (Qwen3.5 in_proj_a/b: 16 outputs) compress below the
            # configured rank. Zero columns keep every layer at one rank, which
            # adapter_config.json and mlx_lm.fuse require; the product is unchanged.
            if rank < self.config.rank:
                pad = [(0, 0)] * (left_factor.ndim - 1) + [(0, self.config.rank - rank)]
                left_factor = mx.pad(left_factor, pad)
                right_factor = mx.pad(right_factor, pad)
            weights[key] = (
                right_factor if value.ndim == 2 else right_factor.swapaxes(-1, -2)
            ).astype(value.dtype)
            weights[bkey] = (left_factor.T if value.ndim == 2 else left_factor).astype(
                b1.dtype
            )
        self.model.load_weights(list(weights.items()), strict=False)

    def train(self, pool, anchors, work, progress=lambda *a: None):
        return train(self, pool, anchors, work, progress)

    def layer_deltas(self, baseline=None):
        mx = self.mx
        names = {id(module): name for name, module in self.model.named_modules()}
        changes = []
        for layer in self.model.layers:
            squared = 0.0
            for _, m in layer.named_modules():
                if not hasattr(m, "lora_a"):
                    continue
                # Frobenius norm using rank-sized Gram matrices, including MoE.
                a, b = m.lora_a.astype(mx.float32), m.lora_b.astype(mx.float32)
                if baseline is not None:
                    prefix = names[id(m)]
                    old_a, old_b = (
                        baseline[prefix + ".lora_a"],
                        baseline[prefix + ".lora_b"],
                    )
                    a = mx.concatenate(
                        [old_a.astype(mx.float32), a], axis=-1 if a.ndim == 2 else -2
                    )
                    b = mx.concatenate(
                        [-old_b.astype(mx.float32), b], axis=-2 if b.ndim == 2 else -1
                    )
                if a.ndim == 2:
                    value = ((a.T @ a) * (b @ b.T)).sum() * m.scale**2
                else:
                    value = (
                        (a @ a.swapaxes(-1, -2)) * (b.swapaxes(-1, -2) @ b)
                    ).sum() * m.scale**2
                mx.eval(value)
                squared += max(0, float(value))
            changes.append(math.sqrt(squared))
        return changes


def train(backend, pool, anchors, work, progress=lambda *a: None):
    from .training import train as run

    return run(backend, pool, anchors, work, progress)
