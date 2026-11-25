"""Mixture-of-experts checkpoints whose experts ship quantized (Inkling-2).

Inkling-2-Mini stores its 128 experts per layer, ~93% of its 26B parameters,
as NVFP4 (two E2M1 codes per byte, an fp8 scale per 16 values and a factor per
expert) or MXFP8 (fp8 values, a power-of-two scale per 32). transformers 5.24
dequantizes them to bf16 while loading ("temporary"), ~49 GB, which no 32 GB
card holds. Here they stay packed (~16 GB): each expert is dequantized only
inside its matmul, and backward rebuilds the weight instead of saving it.

The experts are frozen; LoRA trains the model's bf16 layers (attention,
message gates, routers' neighbours). Dequantization is checked bit for bit
against transformers' own op in the tests.
"""

import glob
import os

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
MODEL_TYPES = {"inkling2_loop"}


def supports(model_config):
    """True for checkpoints whose experts this module keeps packed."""
    quant = getattr(model_config, "quantization_config", None) or {}
    method = quant.get("quant_method") if isinstance(quant, dict) else getattr(quant, "quant_method", None)
    return model_config.model_type in MODEL_TYPES and method == "modelopt"


def _classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers.models.inkling2_loop import modeling_inkling2_loop as m

    class PackedLinear(torch.autograd.Function):
        """y = x @ W.T for a frozen packed expert weight; W is rebuilt in backward."""

        @staticmethod
        def forward(ctx, x, experts, name, index):
            ctx.experts, ctx.name, ctx.index = experts, name, index
            return F.linear(x, experts.weight(name, index, x.dtype))

        @staticmethod
        def backward(ctx, grad):
            return grad @ ctx.experts.weight(ctx.name, ctx.index, grad.dtype), None, None, None

    class PackedExperts(nn.Module):
        """Drop-in for Inkling2LoopExperts holding the checkpoint's bytes."""

        def __init__(self, config):
            super().__init__()
            self.num_experts = config.n_routed_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.act_fn = m.Inkling2LoopSoftcappedSwiGLU(config)
            # transformers' _init_weights initialises these on what it takes
            # for its experts module; empty, never used.
            self.register_buffer("gate_up_proj", torch.empty(0), persistent=False)
            self.register_buffer("down_proj", torch.empty(0), persistent=False)
            self.packed = False

        def load_packed(self, tensors, device, host=False):
            """host: keep the packed bytes in pinned host RAM and copy each
            expert in only when it is used (frees ~16 GB of GPU memory on
            Inkling-2-Mini for PCIe traffic)."""
            self.device, self.host = device, host
            storage = "cpu" if host else device
            rows = tensors["w13_weight"].shape[1]
            # transformers deinterleaves gate/up rows after dequantizing
            # (Interleave(dim=1)); rows are packed independently, so the packed
            # rows and their scales are reordered once instead.
            order = torch.arange(rows).view(rows // 2, 2).t().reshape(-1)
            for name in ("w13", "w2"):
                weight, scale = tensors[f"{name}_weight"], tensors[f"{name}_weight_scale"]
                if name == "w13":
                    weight, scale = weight[:, order], scale[:, order]
                if f"{name}_weight_amax" in tensors:
                    fmt = "nvfp4"
                    factor = tensors[f"{name}_weight_amax"].float() / tensors[
                        f"{name}_weight_outer_scale_denominator"
                    ].float()
                else:
                    fmt, factor = "mxfp8", torch.ones(weight.shape[0])
                setattr(self, f"{name}_format", fmt)
                codes, scale = weight.contiguous().to(storage), scale.contiguous().to(storage)
                if host and device.type == "cuda":
                    codes, scale = codes.pin_memory(), scale.pin_memory()
                self.register_buffer(f"{name}_codes", codes, persistent=False)
                self.register_buffer(f"{name}_scale", scale, persistent=False)
                self.register_buffer(f"{name}_factor", factor.to(device), persistent=False)
            table = torch.tensor(E2M1, device=device)
            byte = torch.arange(256, device=device)
            self.register_buffer(
                "byte_values", torch.stack((table[byte & 0xF], table[byte >> 4]), -1), persistent=False
            )
            self.packed = True

        def weight(self, name, index, dtype):
            """One expert's bf16 weight, or several stacked (index a tensor)."""
            codes = getattr(self, f"{name}_codes")[index]
            scale = getattr(self, f"{name}_scale")[index]
            if self.host:
                codes = codes.to(self.device, non_blocking=True)
                scale = scale.to(self.device, non_blocking=True)
            scale = scale.float()
            if getattr(self, f"{name}_format") == "mxfp8":
                values = codes.float().view(*scale.shape, 32) * torch.exp2(scale - 127)[..., None]
                return values.reshape(codes.shape).to(dtype)
            values = self.byte_values[codes.long()].view(*scale.shape, 16) * scale[..., None]
            factor = getattr(self, f"{name}_factor")[index]
            values = values * factor.view(-1, *[1] * (values.dim() - 1)) if factor.dim() else values * factor
            return values.reshape(*codes.shape[:-1], codes.shape[-1] * 2).to(dtype)

        # Decoding (a few tokens, no gradient) takes the batched path below.
        SMALL = 32

        def forward(self, hidden_states, top_k_index, top_k_weights):
            if top_k_index.numel() <= self.SMALL and not torch.is_grad_enabled() and not self.host:
                return self._small(hidden_states, top_k_index, top_k_weights)
            # As Inkling2LoopExperts.forward, with packed weights.
            # Tokens grouped by expert with one device sync per layer (the
            # counts), not one per expert hit: decoding one token visits ~50
            # expert layers (loops included) and the per-expert syncs held
            # Inkling-2-Mini to ~2.5 tokens/s. Same sums: a token picks an
            # expert at most once, so the order within an expert is free.
            final = torch.zeros_like(hidden_states, dtype=torch.float32)
            k = top_k_index.shape[-1]
            with torch.no_grad():
                flat = top_k_index.reshape(-1)
                order = flat.argsort(stable=True)
                counts = torch.bincount(flat, minlength=self.num_experts + 1).tolist()
            start = 0
            for e, count in enumerate(counts):
                chosen, start = order[start : start + count], start + count
                if not count or e >= self.num_experts:
                    continue
                token_idx, top_k_pos = chosen // k, chosen % k
                state = hidden_states[token_idx]
                gate, up = PackedLinear.apply(state, self, "w13", e).chunk(2, dim=-1)
                state = PackedLinear.apply(self.act_fn(gate, up), self, "w2", e)
                state = state * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, state.to(final.dtype))
            return final.to(hidden_states.dtype)

        def _small(self, hidden_states, top_k_index, top_k_weights):
            """All (token, expert) pairs at once: their experts unpacked as one
            batch and one batched matmul per projection, with no device sync.
            Decoding was launch-bound: ~30k kernel launches and ~3k syncs per
            token on Inkling-2-Mini, ~110 ms of GPU work in ~430 ms."""
            tokens, k = top_k_index.shape
            flat = top_k_index.reshape(-1)
            valid = flat < self.num_experts  # num_experts: no expert (masked)
            chosen = flat.clamp(max=self.num_experts - 1)
            state = hidden_states.repeat_interleave(k, dim=0)[:, :, None]
            gate_up = torch.bmm(self.weight("w13", chosen, hidden_states.dtype), state)[..., 0]
            gate, up = gate_up.chunk(2, dim=-1)
            act = self.act_fn(gate, up)[:, :, None]
            out = torch.bmm(self.weight("w2", chosen, hidden_states.dtype), act)[..., 0]
            out = out * (top_k_weights.reshape(-1) * valid)[:, None]  # promotes as the loop does
            return out.view(tokens, k, -1).float().sum(1).to(hidden_states.dtype)

    return m, PackedExperts


def load(path, device, dtype, experts_on_host=False):
    """The checkpoint's model with PackedExperts filled from its safetensors."""
    import transformers.conversion_mapping as cm
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText

    m, PackedExperts = _classes()
    # Drop transformers' load-time expert dequantization for this family only.
    if cm._checkpoint_conversion_mapping_cache is None:
        cm._checkpoint_conversion_mapping_cache = cm._build_checkpoint_conversion_mapping()
    cache = cm._checkpoint_conversion_mapping_cache
    for key in [k for k in cache if k.startswith("inkling2_loop")]:
        cache[key] = [
            t for t in cache[key]
            if not any("experts" in str(p) for p in getattr(t, "target_patterns", None) or [])
        ]
    original, m.Inkling2LoopExperts = m.Inkling2LoopExperts, PackedExperts
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=dtype, device_map={"": device}, local_files_only=True, trust_remote_code=False
        )
    finally:
        m.Inkling2LoopExperts = original
    experts = [module for module in model.modules() if isinstance(module, PackedExperts)]
    files = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
    for index, module in enumerate(experts):
        prefix = f"model.llm.layers.{index}.mlp.experts."
        tensors = {}
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith(prefix):
                        tensors[key[len(prefix):]] = handle.get_tensor(key)
        if not tensors:
            raise ValueError(f"Checkpoint has no experts for layer {index}")
        module.load_packed(tensors, device, host=experts_on_host)
    from . import block_attention

    # Eager-only attention would be quadratic in memory; same math in blocks.
    block_attention.install(model)
    return model


# LoRA goes where vLLM can serve it on Inkling-2: the attention projections
# (vllm/models/inkling2/nvidia/lora.py). vLLM refuses an adapter with any other
# module ("expected target modules in {...}"), so an adapter trained on every
# linear layer (message gates, convolution taps...) could never be put in use.
# The experts stay frozen and packed; the vision tower is not text experience.
LORA_TARGETS = (
    r"(.*\.)?layers\.\d+\.self_attn\.(q_proj\.linear|k_proj\.linear|v_proj\.linear"
    r"|relative_position_bias\.linear|o_proj\.linear|o_proj\.gate_proj|sink_proj)"
)
# transformers' names for those modules -> the checkpoint's, which vLLM reads
# (transformers' conversion_mapping for inkling2, reversed).
CHECKPOINT_NAMES = (
    (".self_attn.q_proj.linear.", ".attn.wq_du."),
    (".self_attn.k_proj.linear.", ".attn.wk_dv."),
    (".self_attn.v_proj.linear.", ".attn.wv_dv."),
    (".self_attn.relative_position_bias.linear.", ".attn.wr_du."),
    (".self_attn.o_proj.linear.", ".attn.wo_ud."),
    (".self_attn.o_proj.gate_proj.", ".attn.gate_attn."),
    (".self_attn.sink_proj.", ".attn.attn_sink_proj."),
)
PREFIXES = ("model.language_model.layers.", "model.layers.")


def is_packed_base(base):
    """True when base (a model folder) is a checkpoint this module trains."""
    import json
    from pathlib import Path
    from types import SimpleNamespace

    try:
        config = json.loads((Path(base) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict):
        return False
    return supports(SimpleNamespace(model_type=config.get("model_type"),
                                    quantization_config=config.get("quantization_config")))


def checkpoint_name(key):
    """A PEFT adapter key under the checkpoint's module names (as vLLM loads
    them): base_model.model.model.llm.layers.N.attn.wq_du.lora_A.weight."""
    head = "base_model.model."
    rest = key[len(head):] if key.startswith(head) else key
    for prefix in PREFIXES:
        if rest.startswith(prefix):
            rest = "model.llm.layers." + rest[len(prefix):]
            break
    for ours, theirs in CHECKPOINT_NAMES:
        rest = rest.replace(ours, theirs)
    return head + rest


def to_checkpoint_names(adapter):
    """Rename an exported adapter folder's weights and target modules in place."""
    import json
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    adapter = Path(adapter)
    weights = adapter / "adapter_model.safetensors"
    tensors = load_file(str(weights))
    save_file({checkpoint_name(k): v for k, v in tensors.items()}, str(weights), metadata={"format": "pt"})
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target_modules"] = sorted({theirs.strip(".").split(".")[-1] for _, theirs in CHECKPOINT_NAMES})
    config["exclude_modules"] = None
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
