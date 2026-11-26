"""Inkling-2's packed experts match transformers' dequantization and forward."""

import pytest

torch = pytest.importorskip("torch")
m = pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

from songe.backends import packed_experts  # noqa: E402


def nvfp4(experts, rows, columns, generator):
    return {
        "w_weight": torch.randint(0, 256, (experts, rows, columns // 2), dtype=torch.uint8, generator=generator),
        "w_weight_scale": (torch.rand(experts, rows, columns // 16, generator=generator) + 0.5).to(torch.float8_e4m3fn),
        "w_weight_amax": torch.rand(experts, generator=generator) + 1.0,
        "w_weight_outer_scale_denominator": torch.tensor([448.0 * 6.0]),
    }


def mxfp8(experts, rows, columns, generator):
    return {
        "w_weight": torch.randn(experts, rows, columns, generator=generator).to(torch.float8_e4m3fn),
        "w_weight_scale": torch.randint(120, 130, (experts, rows, columns // 32), dtype=torch.uint8, generator=generator),
    }


def reference(tensors, target, interleave):
    from transformers.core_model_loading import (
        Inkling2LoopDequantizeExperts,
        Interleave,
    )

    names = ["w_weight", "w_weight_scale", "w_weight_amax", "w_weight_outer_scale_denominator"]
    inputs = {n: tensors.get(n) for n in names}
    weight = Inkling2LoopDequantizeExperts().convert(inputs, names, [target])[target]
    if interleave:
        weight = Interleave(dim=1).convert({target: weight}, [target], [target])[target]
    return weight


def layer(config, make, generator):
    hidden, inner, experts = config.hidden_size, config.moe_intermediate_size, config.n_routed_experts
    w13 = make(experts, 2 * inner, hidden, generator)
    w2 = make(experts, hidden, inner, generator)
    tensors = {k.replace("w_", "w13_"): v for k, v in w13.items()}
    tensors |= {k.replace("w_", "w2_"): v for k, v in w2.items()}
    return tensors, reference(w13, "gate_up_proj", True), reference(w2, "down_proj", False)


@pytest.fixture
def config():
    return m.Inkling2LoopTextConfig(
        hidden_size=64, moe_intermediate_size=32, n_routed_experts=4, num_experts_per_tok=2,
        vocab_size=128, unpadded_vocab_size=128,
    )


@pytest.mark.parametrize("make", [nvfp4, mxfp8])
def test_dequantization_matches_transformers_bit_for_bit(config, make):
    generator = torch.Generator().manual_seed(7)
    tensors, gate_up, down = layer(config, make, generator)
    _, PackedExperts = packed_experts._classes()
    experts = PackedExperts(config)
    experts.load_packed(tensors, torch.device("cpu"))
    for e in range(config.n_routed_experts):
        assert torch.equal(experts.weight("w13", e, torch.bfloat16), gate_up[e].to(torch.bfloat16))
        assert torch.equal(experts.weight("w2", e, torch.bfloat16), down[e].to(torch.bfloat16))


def test_forward_and_input_gradient_match_dequantized_experts(config):
    generator = torch.Generator().manual_seed(11)
    tensors, gate_up, down = layer(config, nvfp4, generator)
    _, PackedExperts = packed_experts._classes()
    packed = PackedExperts(config)
    packed.load_packed(tensors, torch.device("cpu"))
    dense = m.Inkling2LoopExperts(config)
    with torch.no_grad():
        dense.gate_up_proj.copy_(gate_up.float())
        dense.down_proj.copy_(down.float())
    tokens = 6
    index = torch.randint(0, config.n_routed_experts, (tokens, 2), generator=generator)
    weights = torch.rand(tokens, 2, generator=generator)
    x1 = torch.randn(tokens, config.hidden_size, generator=generator, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    out_packed = packed(x1, index, weights)
    out_dense = dense(x2, index, weights)
    torch.testing.assert_close(out_packed, out_dense)
    out_packed.sum().backward()
    out_dense.sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad)
    assert not any(p.requires_grad for p in packed.buffers())


@pytest.mark.parametrize("make", [nvfp4, mxfp8])
def test_decoding_path_matches_the_expert_loop(config, make):
    """A few tokens without gradients take the batched path; same output as
    the per-expert loop, masked slots (index num_experts) included."""
    generator = torch.Generator().manual_seed(13)
    tensors, _, _ = layer(config, make, generator)
    _, PackedExperts = packed_experts._classes()
    packed = PackedExperts(config)
    packed.load_packed(tensors, torch.device("cpu"))
    for tokens in (1, 3):
        index = torch.randint(0, config.n_routed_experts, (tokens, 2), generator=generator)
        index[0, 1] = config.n_routed_experts
        weights = torch.rand(tokens, 2, generator=generator)
        x = torch.randn(tokens, config.hidden_size, generator=generator).to(torch.bfloat16)
        with torch.no_grad():
            small = packed(x, index, weights)
            packed.SMALL = 0
            loop = packed(x, index, weights)
            del packed.SMALL
        torch.testing.assert_close(small, loop)


def test_only_modelopt_inkling_checkpoints_are_packed():
    from types import SimpleNamespace

    assert packed_experts.supports(SimpleNamespace(
        model_type="inkling2_loop", quantization_config={"quant_method": "modelopt"}))
    assert not packed_experts.supports(SimpleNamespace(model_type="inkling2_loop", quantization_config=None))
    assert not packed_experts.supports(SimpleNamespace(
        model_type="qwen3_5", quantization_config={"quant_method": "modelopt"}))


def tiny_inkling(seed=3):
    torch.manual_seed(seed)
    config = m.Inkling2LoopTextConfig(
        hidden_size=64, moe_intermediate_size=32, n_routed_experts=4, num_experts_per_tok=2,
        vocab_size=97, unpadded_vocab_size=97, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4, swa_num_key_value_heads=2,
        swa_head_dim=16, sliding_window_size=24, rel_extent=32, d_rel=8, n_layers_per_attn_res_block=1,
        loop_config={"core_start": 1, "core_end": 3, "num_passes": 2, "gate_rank": 8, "is_varloop": False},
    )
    config._attn_implementation = "eager"
    model = m.Inkling2LoopForCausalLM(config).float().eval()
    return model


def test_block_attention_matches_eager_outputs_and_gradients():
    from songe.backends import block_attention

    model = tiny_inkling()
    ids = torch.randint(0, 97, (1, 90))

    def run():
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, use_cache=False).logits
        logits.float().logsumexp(-1).mean().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return logits.detach(), grads

    reference, reference_grads = run()
    block_attention.install(model)
    elements = block_attention.BLOCK_ELEMENTS
    block_attention.BLOCK_ELEMENTS = 4 * 90 * 64  # force 64-query blocks... and the 64 floor
    try:
        model.train()  # checkpointed blocks
        blocks, block_grads = run()
    finally:
        block_attention.BLOCK_ELEMENTS = elements
    torch.testing.assert_close(blocks, reference, atol=1e-4, rtol=1e-4)
    assert reference_grads.keys() == block_grads.keys()
    for name in reference_grads:
        torch.testing.assert_close(block_grads[name], reference_grads[name], atol=1e-4, rtol=1e-3)


def test_a_frozen_prefix_reads_the_same_conv_states_when_checkpointing_replays_a_layer():
    """Inkling-2's cache layers keep short-convolution states, written in place.
    Under gradient checkpointing the backward pass runs every layer again; it
    must see the prefix as the forward pass did, not the window's own tail
    (on Inkling-2-Mini a Deep Dream then routed tokens to other experts and
    failed with CheckpointError)."""
    from types import SimpleNamespace

    from songe.backends.torch import TorchBackend

    model = tiny_inkling().train()
    ids = torch.randint(0, 97, (1, 40), generator=torch.Generator().manual_seed(5))
    prefix, window = ids[:, :30], ids[:, 30:]

    def run(checkpointing):
        model.zero_grad(set_to_none=True)
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        with torch.no_grad():
            past = TorchBackend._cache(model.base_model(input_ids=prefix, use_cache=True))
        if not checkpointing:  # the layers keep the cache themselves
            out = model.base_model(input_ids=window, past_key_values=past, use_cache=True).last_hidden_state
            out.float().square().mean().backward()
            return out.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        conv = [{k: v.clone() for k, v in getattr(layer, "conv_states", {}).items() if v is not None}
                for layer in past.layers]
        owner = SimpleNamespace(torch=torch, _causal=lambda: model)
        with TorchBackend.frozen(owner, past) as kept:
            assert kept
            out = model.base_model(input_ids=window, past_key_values=past, use_cache=True).last_hidden_state
            out.float().square().mean().backward()
        assert any(conv)
        for layer, states in zip(past.layers, conv, strict=True):
            for k, v in states.items():
                torch.testing.assert_close(layer.conv_states[k], v, atol=0, rtol=0)
        return out.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    plain, plain_grads = run(checkpointing=False)
    replayed, grads = run(checkpointing=True)
    torch.testing.assert_close(replayed, plain)
    assert grads.keys() == plain_grads.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], plain_grads[name], atol=1e-5, rtol=1e-4)


# What vLLM's Inkling2 LoRA loader accepts (vllm/models/inkling2/nvidia/lora.py):
# these modules, under the checkpoint's names (model.llm.layers.N.attn.*).
VLLM_INKLING_LORA = {"wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud", "gate_attn", "attn_sink_proj"}


def test_an_inkling_adapter_exports_under_the_names_vllm_loads(tmp_path):
    import json
    import re

    from peft import LoraConfig, get_peft_model

    from songe.artifacts import export
    from tests.crossplatform_helpers import fake_platform

    model = get_peft_model(
        tiny_inkling(),
        LoraConfig(r=2, lora_alpha=4, target_modules=packed_experts.LORA_TARGETS),
    )
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(
        {"model_type": "inkling2_loop", "quantization_config": {"quant_method": "modelopt"}}))
    out = export("peft-adapter", base, adapter, tmp_path / "peft", fake_platform(tmp_path))

    from safetensors.torch import load_file

    keys = load_file(str(out / "adapter_model.safetensors")).keys()
    pattern = re.compile(r"base_model\.model\.model\.llm\.layers\.\d+\.attn\.(\w+)\.lora_[AB]\.weight")
    modules = {pattern.fullmatch(k).group(1) if pattern.fullmatch(k) else k for k in keys}
    # Every module vLLM serves, nothing it would refuse, in every layer.
    assert modules == VLLM_INKLING_LORA
    assert len(keys) == 2 * len(VLLM_INKLING_LORA) * model.config.num_hidden_layers
    config = json.loads((out / "adapter_config.json").read_text())
    assert set(config["target_modules"]) == VLLM_INKLING_LORA
    # The training adapter keeps transformers' names (a Dream resumes from it).
    assert all(".self_attn." in k for k in load_file(str(adapter / "adapter_model.safetensors")))
