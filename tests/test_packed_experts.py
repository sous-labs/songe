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


def test_only_modelopt_inkling_checkpoints_are_packed():
    from types import SimpleNamespace

    assert packed_experts.supports(SimpleNamespace(
        model_type="inkling2_loop", quantization_config={"quant_method": "modelopt"}))
    assert not packed_experts.supports(SimpleNamespace(model_type="inkling2_loop", quantization_config=None))
    assert not packed_experts.supports(SimpleNamespace(
        model_type="qwen3_5", quantization_config={"quant_method": "modelopt"}))
