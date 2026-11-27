"""Qwen3.8-Flash-Next (qwen4_exp) loads with 4-bit packed experts and a
memory-mapped n-gram table, from a checkpoint in the official FP8 layout."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
m = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from songe.backends import packed_experts  # noqa: E402

GRID = torch.tensor(packed_experts.E2M1)


def on_grid(*shape, generator, unit=1 / 16):
    """Values NVFP4 and FP8 both hold exactly: E2M1 codes times one power of
    two, each block of 16 holding a 6 (so every block has the same scale)."""
    values = GRID[torch.randint(0, 16, shape, generator=generator)].clone()
    values.view(-1, 16)[:, 0] = 6.0
    return values * unit


def test_packing_round_trips_values_on_the_grid():
    generator = torch.Generator().manual_seed(5)
    weight = on_grid(3, 8, 32, generator=generator)
    codes, scale, factor = packed_experts.pack_nvfp4(weight)
    _, PackedExperts = packed_experts._classes("qwen4_exp")
    experts = PackedExperts(SimpleNamespace(num_experts=3, hidden_size=32, moe_intermediate_size=4, hidden_act="silu"))
    tensors = {}
    for name in ("w13", "w2"):
        tensors |= {f"{name}_weight": codes, f"{name}_weight_scale": scale, f"{name}_weight_amax": factor,
                    f"{name}_weight_outer_scale_denominator": torch.ones(())}
    experts.load_packed(tensors, torch.device("cpu"), interleaved=False)
    assert torch.equal(experts.weight("w13", torch.arange(3), torch.float32), weight)
    noisy = torch.randn(3, 8, 32, generator=generator)
    codes, scale, factor = packed_experts.pack_nvfp4(noisy)
    tensors = {f"{n}_{k}": v for n in ("w13", "w2") for k, v in (
        ("weight", codes), ("weight_scale", scale), ("weight_amax", factor),
        ("weight_outer_scale_denominator", torch.ones(())))}
    experts.load_packed(tensors, torch.device("cpu"), interleaved=False)
    error = (experts.weight("w13", torch.arange(3), torch.float32) - noisy).abs()
    assert error.max() <= noisy.abs().amax() / 6  # half the widest E2M1 step, scaled


def tiny_config():
    return m.Qwen4ExpTextConfig(
        vocab_size=96, hidden_size=64, num_hidden_layers=4, full_attention_interval=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
        hc_count=2, hc_lowrank=8, ple_layer_ids=[2], ple_embed_dim=32, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=50, split_ngram_parts=4, make_ngram_vocab_size_divisible_by=4,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=4,
        mtp_num_hidden_layers=0,
        eos_token_id=1, pad_token_id=0,
    )


def write_fp8_checkpoint(model, config, folder, generator):
    """The official FP8 layout: model.language_model.* names, one FP8 tensor
    per expert projection with block scales, the n-gram table in FP8 shards
    with one scale, quantization_config in config.json."""
    from safetensors.torch import save_file

    tensors = {}
    for key, value in model.state_dict().items():
        name = key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key
        if ".mlp.experts." in key:
            layer = name.split(".mlp.experts.")[0]
            down = key.endswith("down_proj")
            for e in range(value.shape[0]):
                parts = {"down_proj": value[e]} if down else dict(zip(("gate_proj", "up_proj"), value[e].chunk(2), strict=True))
                for projection, weight in parts.items():
                    target = f"{layer}.mlp.experts.{e}.{projection}.weight"
                    tensors[target] = weight.to(torch.float8_e4m3fn)
                    tensors[target + "_scale_inv"] = torch.ones(1, 1, dtype=torch.bfloat16)
        elif key.endswith("ngram_embedding.weight"):
            table = name[: -len("weight")]
            for i, shard in enumerate((value * 2).chunk(config.split_ngram_parts)):
                tensors[f"{table}shard_{i}.weight"] = shard.contiguous().to(torch.float8_e4m3fn)
            tensors[table + "weight_scale"] = torch.tensor([0.5], dtype=torch.bfloat16)
        else:
            tensors[name] = value.contiguous()
    half = set(sorted(tensors)[: len(tensors) // 2])
    save_file({k: tensors[k] for k in half}, str(folder / "model-00001-of-00002.safetensors"))
    save_file({k: v for k, v in tensors.items() if k not in half}, str(folder / "model-00002-of-00002.safetensors"))
    files = {k: f"model-0000{1 if k in half else 2}-of-00002.safetensors" for k in tensors}
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": files}))
    full = m.Qwen4ExpConfig(text_config=config.to_dict()).to_dict()
    full["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    full["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (folder / "config.json").write_text(json.dumps(full))


def test_the_fp8_checkpoint_loads_packed_and_answers_as_the_dense_model(tmp_path):
    generator = torch.Generator().manual_seed(9)
    config = tiny_config()
    torch.manual_seed(9)
    dense = m.Qwen4ExpForCausalLM(config).float().eval()
    with torch.no_grad():
        for module in dense.modules():
            if isinstance(module, m.Qwen4ExpTextExperts):
                module.gate_up_proj.copy_(on_grid(*module.gate_up_proj.shape, generator=generator))
                module.down_proj.copy_(on_grid(*module.down_proj.shape, generator=generator))
            if isinstance(module, m.Qwen4ExpTextNGramEmbedding):
                table = module.ngram_embedding.weight
                table.copy_(on_grid(*table.shape, generator=generator, unit=1 / 4))
    write_fp8_checkpoint(dense, config, tmp_path, generator)

    loaded = packed_experts.load(str(tmp_path), torch.device("cpu"), torch.float32).eval()
    assert not any(isinstance(x, m.Qwen4ExpTextExperts) for x in loaded.modules())
    assert not any("experts" in n or "ngram" in n for n, _ in loaded.named_parameters())
    ids = torch.randint(2, config.vocab_size, (2, 12), generator=generator)
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_ids=ids).logits, dense(input_ids=ids).logits)

    import re

    targets = packed_experts.lora_targets(loaded.config)
    chosen = {n.split(".", 3)[-1] for n, _ in loaded.named_modules() if re.fullmatch(targets, n)}
    assert chosen == {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "linear_attn.out_proj",
        "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj",
    }

    # The disk budget counts an adapter on those layers only, not on every
    # expert matrix (1026 GB asked of a 340 GB disk, live).
    from songe.artifacts.storage import tensor_sizes

    lora = sum(8 * (x.in_features + x.out_features) * 4
               for n, x in loaded.named_modules() if re.fullmatch(targets, n))
    assert tensor_sizes(tmp_path, 8)[1] == lora


def test_flash_next_needs_no_fp8_kernels_and_keeps_peft_names(tmp_path):
    from songe.backends.quantized import missing_packages

    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "qwen4_exp", "quantization_config": {"quant_method": "fp8"}}))
    assert missing_packages(tmp_path) == []
    assert not packed_experts.is_packed_base(tmp_path)  # adapters are not renamed for vLLM's Inkling loader


def test_decoding_with_experts_in_host_ram_takes_the_batched_path():
    """A decoded token's experts are copied in as one batch (0.65 tokens/s
    with one copy per expert, live on Qwen3.8-Flash-Next); same output as the
    per-expert loop."""
    generator = torch.Generator().manual_seed(3)
    config = SimpleNamespace(num_experts=6, hidden_size=32, moe_intermediate_size=16, hidden_act="silu")
    _, PackedExperts = packed_experts._classes("qwen4_exp")
    experts = PackedExperts(config)
    tensors = {}
    for name, rows, columns in (("w13", 32, 32), ("w2", 32, 16)):
        codes, scale, factor = packed_experts.pack_nvfp4(torch.randn(6, rows, columns, generator=generator))
        tensors |= {f"{name}_weight": codes, f"{name}_weight_scale": scale, f"{name}_weight_amax": factor,
                    f"{name}_weight_outer_scale_denominator": torch.ones(())}
    experts.load_packed(tensors, torch.device("cpu"), host=True, interleaved=False)
    index = torch.tensor([[1, 4, 6]])  # 6: no expert (masked)
    weights = torch.rand(1, 3, generator=generator)
    x = torch.randn(1, 32, generator=generator)
    calls = []
    small = experts._small
    experts._small = lambda *a: calls.append(1) or small(*a)
    with torch.no_grad():
        batched = experts(x, index, weights)
        experts.SMALL = 0
        loop = experts(x, index, weights)
    assert calls == [1]
    torch.testing.assert_close(batched, loop, rtol=1e-4, atol=1e-4)  # summed in another order
