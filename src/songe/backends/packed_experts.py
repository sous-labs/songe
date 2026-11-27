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

Qwen3.8-Flash-Next (qwen4_exp, 177B) ships its experts in bf16 or FP8: 121B
values, 121 GB even in FP8. They are packed here to 4 bits in the same NVFP4
layout (68 GB) while loading, one layer at a time, and kept in host RAM. Its
hashed n-gram table (51B values, one layer) is a lookup by token id: rows are
read from the checkpoint's files through a memory map, never loaded.
"""

import glob
import json
import os
import struct
import weakref

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
MODEL_TYPES = {"inkling2_loop"}
# Experts re-packed to 4 bits at load, whatever they ship in.
REPACKED_TYPES = {"qwen4_exp", "qwen4_exp_text"}
# Experts packed per batch while loading (bounds the accelerator memory used).
PACK_CHUNK = 16


def supports(model_config):
    """True for checkpoints whose experts this module keeps packed."""
    if model_config.model_type in REPACKED_TYPES:
        return True
    quant = getattr(model_config, "quantization_config", None) or {}
    method = quant.get("quant_method") if isinstance(quant, dict) else getattr(quant, "quant_method", None)
    return model_config.model_type in MODEL_TYPES and method == "modelopt"


def experts_on_host(model_config):
    """Where packed experts go when the user did not choose: Flash-Next's
    68 GB leave no room on a workstation card, Inkling-2-Mini's 16 GB do."""
    return model_config.model_type in REPACKED_TYPES


def _pinned(tensor):
    """The tensor's own host memory page-locked in place. pin_memory() copies
    into torch's pinned cache, which rounds every block up to a power of two:
    Flash-Next's 68 GB of experts took 87 GB of locked RAM that way."""
    import torch

    tensor = tensor.contiguous()
    if tensor.is_pinned():
        return tensor
    error = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
    if int(error) != 0:
        return tensor.pin_memory()
    weakref.finalize(tensor, torch.cuda.cudart().cudaHostUnregister, tensor.data_ptr())
    return tensor


def _classes(family="inkling2_loop"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if family in REPACKED_TYPES:
        from transformers.activations import ACT2FN
        from transformers.models.qwen4_exp import modeling_qwen4_exp as m

        def activation(config):
            act = ACT2FN[config.hidden_act]
            return lambda gate, up: act(gate) * up

    else:
        from transformers.models.inkling2_loop import modeling_inkling2_loop as m

        def activation(config):
            return m.Inkling2LoopSoftcappedSwiGLU(config)

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
        """Drop-in for the family's experts module holding packed bytes."""

        def __init__(self, config):
            super().__init__()
            self.num_experts = getattr(config, "n_routed_experts", None) or config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.act_fn = activation(config)
            # transformers' _init_weights initialises these on what it takes
            # for its experts module; empty, never used.
            self.register_buffer("gate_up_proj", torch.empty(0), persistent=False)
            self.register_buffer("down_proj", torch.empty(0), persistent=False)
            self.packed = False

        def load_packed(self, tensors, device, host=False, interleaved=True):
            """host: keep the packed bytes in pinned host RAM and copy each
            expert in only when it is used (frees ~16 GB of GPU memory on
            Inkling-2-Mini for PCIe traffic). interleaved: w13's gate and up
            rows alternate (Inkling-2), else gate rows come first."""
            self.device, self.host = device, host
            storage = "cpu" if host else device
            rows = tensors["w13_weight"].shape[1]
            # transformers deinterleaves gate/up rows after dequantizing
            # (Interleave(dim=1)); rows are packed independently, so the packed
            # rows and their scales are reordered once instead.
            order = torch.arange(rows).view(rows // 2, 2).t().reshape(-1)
            for name in ("w13", "w2"):
                weight, scale = tensors[f"{name}_weight"], tensors[f"{name}_weight_scale"]
                if name == "w13" and interleaved:
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
                    codes, scale = _pinned(codes), _pinned(scale)
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


def _drop_conversions(family, words):
    """Drop transformers' load-time conversions whose targets name one of
    words (expert merges, the n-gram concatenation) for this family only."""
    import transformers.conversion_mapping as cm

    if cm._checkpoint_conversion_mapping_cache is None:
        cm._checkpoint_conversion_mapping_cache = cm._build_checkpoint_conversion_mapping()
    cache = cm._checkpoint_conversion_mapping_cache
    for key in [k for k in cache if k.startswith(family)]:
        cache[key] = [
            t for t in cache[key]
            if not any(w in str(p) for p in getattr(t, "target_patterns", None) or [] for w in words)
        ]


def load(path, device, dtype, experts_on_host=False):
    """The checkpoint's model with PackedExperts filled from its safetensors."""
    config = json.loads(open(os.path.join(path, "config.json"), encoding="utf-8").read())
    if config.get("model_type") in REPACKED_TYPES:
        return _load_repacked(path, device, dtype, experts_on_host)
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText

    m, PackedExperts = _classes()
    _drop_conversions("inkling2_loop", ("experts",))
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


def pack_nvfp4(weight):
    """[experts, rows, columns] -> (E2M1 codes two per byte, fp8 scale per 16
    values, factor per expert): the NVFP4 layout PackedExperts.weight decodes.
    Round to nearest; a block's largest value maps to at most 6 times its scale."""
    import torch

    experts, rows, columns = weight.shape
    weight = weight.float()
    blocks = weight.view(experts, rows, columns // 16, 16)
    # A power of two, so values on the E2M1 grid come back exactly.
    factor = torch.exp2(torch.ceil(torch.log2(weight.abs().amax(dim=(1, 2)).clamp_min(1e-30) / (6.0 * 448.0))))
    scale = (blocks.abs().amax(-1) / 6.0 / factor[:, None, None]).to(torch.float8_e4m3fn)
    step = scale.float() * factor[:, None, None]
    step = torch.where(step > 0, step, torch.ones_like(step))
    values = blocks / step[..., None]
    midpoints = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=weight.device)
    codes = (torch.bucketize(values.abs(), midpoints) + 8 * (values < 0)).to(torch.uint8)
    codes = codes.view(experts, rows, columns)
    return codes[..., 0::2] | (codes[..., 1::2] << 4), scale, factor


SAFETENSORS_TYPES = {"F8_E4M3": "float8_e4m3fn", "BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def _headers(path):
    """key -> (file, first data byte, header entry) for every safetensors file."""
    found = {}
    for file in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with open(file, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        header.pop("__metadata__", None)
        for key, entry in header.items():
            found[key] = (file, 8 + size, entry)
    return found


def _mapped(found, key):
    """A [rows, bytes per row] uint8 view of one 2-D tensor, memory-mapped."""
    import warnings

    import numpy
    import torch

    file, start, entry = found[key]
    begin, end = entry["data_offsets"]
    rows = entry["shape"][0]
    array = numpy.memmap(file, dtype=numpy.uint8, mode="r", offset=start + begin, shape=(rows, (end - begin) // rows))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # read-only memory: never written
        return torch.from_numpy(array), getattr(torch, SAFETENSORS_TYPES[entry["dtype"]])


def _repacked_classes():
    import torch
    import torch.nn as nn
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    class MappedRows(nn.Module):
        """Embedding rows read from the checkpoint's files by token id.
        Frozen: the rows come back without gradients."""

        def __init__(self, rows, dim):
            super().__init__()
            self.num_embeddings, self.embedding_dim = rows, dim
            self.parts = []

        @property
        def weight(self):
            # transformers moves the ids to the table's device: host RAM.
            return torch.empty(0)

        def attach(self, parts, stored, scale, dtype):
            self.parts, self.stored, self.scale, self.dtype = parts, stored, scale, dtype
            sizes = torch.tensor([p.shape[0] for p in parts])
            self.ends = sizes.cumsum(0)
            self.starts = self.ends - sizes

        def forward(self, ids):
            flat = ids.reshape(-1).to("cpu")
            part = torch.bucketize(flat, self.ends, right=True)
            rows = torch.empty(flat.numel(), self.parts[0].shape[1], dtype=torch.uint8)
            for p in part.unique().tolist():
                chosen = part == p
                rows[chosen] = self.parts[p][flat[chosen] - self.starts[p]]
            values = rows.view(self.stored).to(self.dtype)
            if self.scale is not None:
                values = values * self.scale
            return values.view(*ids.shape, self.embedding_dim)

    class MappedNGramEmbedding(m.Qwen4ExpTextNGramEmbedding):
        def __init__(self, *args, **kwargs):
            # Never allocated: the table is ~51B values.
            with torch.device("meta"):
                super().__init__(*args, **kwargs)
            table = self.ngram_embedding
            self.ngram_embedding = MappedRows(table.num_embeddings, table.embedding_dim)

    return m, MappedNGramEmbedding


def _load_repacked(path, device, dtype, experts_on_host):
    """Qwen3.8-Flash-Next: experts packed to 4 bits layer by layer, the n-gram
    table memory-mapped, everything else loaded by transformers."""
    import contextlib

    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    text = config.get_text_config(decoder=True)
    # The experts and the n-gram table are the only quantized tensors (FP8 in
    # the official FP8 build) and both are read here, so the rest loads as
    # plain bf16 without transformers' FP8 kernels.
    for part in (config, text):
        if getattr(part, "quantization_config", None) is not None:
            part.quantization_config = None
    m, PackedExperts = _classes("qwen4_exp")
    m, MappedNGramEmbedding = _repacked_classes()
    _drop_conversions("qwen4_exp", ("experts", "ngram_embedding"))
    swapped = {"Qwen4ExpTextExperts": PackedExperts, "Qwen4ExpTextNGramEmbedding": MappedNGramEmbedding}
    originals = {name: getattr(m, name) for name in swapped}
    head = m.Qwen4ExpForCausalLM
    ignored = head._keys_to_ignore_on_load_unexpected
    head._keys_to_ignore_on_load_unexpected = list(ignored or []) + [
        r".*\.mlp\.experts\.\d+\.", r".*\.ngram_embedding\.(shard_\d+\.)?weight(_scale)?$",
    ]
    for name, cls in swapped.items():
        setattr(m, name, cls)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, dtype=dtype, device_map={"": device}, local_files_only=True, trust_remote_code=False
        )
    finally:
        for name, cls in originals.items():
            setattr(m, name, cls)
        head._keys_to_ignore_on_load_unexpected = ignored
    found = _headers(path)
    layers = next(
        (k[: -len("0.mlp.experts.0.gate_proj.weight")] for k in found
         if k.endswith("layers.0.mlp.experts.0.gate_proj.weight") and not k.startswith("mtp.")),
        None,
    )
    if layers is None:
        raise ValueError("Checkpoint stores no per-expert weights (layers.N.mlp.experts.E.gate_proj)")
    with contextlib.ExitStack() as stack:
        handles = {}

        def read(key):
            file = found[key][0]
            if file not in handles:
                handles[file] = stack.enter_context(safe_open(file, framework="pt", device="cpu"))
            return handles[file].get_tensor(key).to(device)

        def dense(key):
            weight = read(key).float()
            if key + "_scale_inv" not in found:
                return weight
            # FP8 blocks (128 x 128 in the official build), one scale each.
            scale = read(key + "_scale_inv").float()
            rows, columns = weight.shape
            scale = scale.repeat_interleave(-(-rows // scale.shape[0]), 0)[:rows]
            return weight * scale.repeat_interleave(-(-columns // scale.shape[1]), 1)[:, :columns]

        storage = "cpu" if experts_on_host else device
        for name, module in model.named_modules():
            if not isinstance(module, PackedExperts):
                continue
            index = int(name.split("layers.")[1].split(".")[0])
            prefix = f"{layers}{index}.mlp.experts."
            packed = {"w13": [], "w2": []}
            for first in range(0, module.num_experts, PACK_CHUNK):
                chosen = range(first, min(first + PACK_CHUNK, module.num_experts))
                w13 = torch.stack([
                    torch.cat([dense(f"{prefix}{e}.gate_proj.weight"), dense(f"{prefix}{e}.up_proj.weight")])
                    for e in chosen
                ])
                w2 = torch.stack([dense(f"{prefix}{e}.down_proj.weight") for e in chosen])
                for key, weight in (("w13", w13), ("w2", w2)):
                    packed[key].append([t.to(storage) for t in pack_nvfp4(weight)])
                del w13, w2
            tensors = {}
            for key, parts in packed.items():
                codes, scales, factors = (torch.cat(t) for t in zip(*parts, strict=True))
                tensors |= {
                    f"{key}_weight": codes, f"{key}_weight_scale": scales,
                    f"{key}_weight_amax": factors, f"{key}_weight_outer_scale_denominator": torch.ones(()),
                }
            module.load_packed(tensors, device, host=experts_on_host, interleaved=False)
            del packed, tensors
    for module in model.modules():
        if not isinstance(module, MappedNGramEmbedding):
            continue
        table = f"{layers}{module.layer_idx}.ple.ple_embedding.ngram_embedding."
        shards = sorted(
            (int(k[len(table) + 6 : -7]), k) for k in found if k.startswith(table + "shard_") and k.endswith(".weight")
        )
        keys = [k for _, k in shards] or [table + "weight"]
        parts = [_mapped(found, k) for k in keys]
        scale = None
        if table + "weight_scale" in found:
            with safe_open(found[table + "weight_scale"][0], framework="pt", device="cpu") as handle:
                scale = handle.get_tensor(table + "weight_scale").float().reshape(()).item()
        module.ngram_embedding.attach([p for p, _ in parts], parts[0][1], scale, dtype)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


# Flash-Next: every attention's projections (q/k/v/o on the full-attention
# layers, the output of the linear-attention ones) and the shared expert.
# Not in_proj_qkv / in_proj_z: llama.cpp stores them as one merged tensor, so
# an adapter on them would not load there.
REPACKED_LORA_TARGETS = (
    r"(.*\.)?layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.out_proj"
    r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))"
)


def lora_targets(model_config):
    return REPACKED_LORA_TARGETS if model_config.model_type in REPACKED_TYPES else LORA_TARGETS


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
    """True when base (a model folder) is an Inkling-2 checkpoint this module
    trains, whose adapters vLLM reads under the checkpoint's names."""
    from pathlib import Path
    from types import SimpleNamespace

    try:
        config = json.loads((Path(base) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict) or config.get("model_type") not in MODEL_TYPES:
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
