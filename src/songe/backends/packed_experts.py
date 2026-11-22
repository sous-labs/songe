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
            values = values * getattr(self, f"{name}_factor")[index]
            return values.reshape(codes.shape[0], codes.shape[1] * 2).to(dtype)

        def forward(self, hidden_states, top_k_index, top_k_weights):
            # As Inkling2LoopExperts.forward, with packed weights.
            final = torch.zeros_like(hidden_states, dtype=torch.float32)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert in hit:
                e = int(expert[0])
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[e])
                state = hidden_states[token_idx]
                gate, up = PackedLinear.apply(state, self, "w13", e).chunk(2, dim=-1)
                state = PackedLinear.apply(self.act_fn(gate, up), self, "w2", e)
                state = state * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, state.to(final.dtype))
            return final.to(hidden_states.dtype)

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


# Modules LoRA leaves alone in these models: the vision tower (text-only
# experience) and the experts (frozen, packed).
LORA_EXCLUDE = r".*(vision_tower|visual|experts).*"
