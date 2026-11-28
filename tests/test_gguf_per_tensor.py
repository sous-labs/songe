import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
gguf = pytest.importorskip("gguf")
safetensors = pytest.importorskip("safetensors")

import numpy as np  # noqa: E402

from songe.models import worker  # noqa: E402

HIDDEN, HEADS, KV_HEADS, FFN, VOCAB, LAYERS = 64, 4, 2, 128, 96, 2
# Llama derives its head size from the width; Qwen3's GGUF config keeps 128.
HEAD_DIM = {"llama": HIDDEN // HEADS, "qwen3": 128}


def tiny_gguf(path, architecture, *, tied):
    """A two-layer model in the layout llama.cpp writes: float32 norms,
    Q8_0 and F16 matrices, the output head tied or its own."""
    rng = np.random.default_rng(7613)
    head = HEAD_DIM[architecture]
    writer = gguf.GGUFWriter(path, architecture)
    writer.add_file_type(7)
    writer.add_context_length(256)
    writer.add_embedding_length(HIDDEN)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(FFN)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_rope_dimension_count(head)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0)
    writer.add_vocab_size(VOCAB)

    def matrix(name, rows, columns, kind):
        values = rng.standard_normal((rows, columns), dtype=np.float32) / 8
        if kind == "q8_0":
            writer.add_tensor(
                name,
                gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0),
                raw_dtype=gguf.GGMLQuantizationType.Q8_0,
            )
        else:
            writer.add_tensor(name, values.astype(np.float16))

    def norm(name, size):
        writer.add_tensor(name, (1 + rng.standard_normal(size) / 10).astype(np.float32))

    matrix("token_embd.weight", VOCAB, HIDDEN, "q8_0")
    norm("output_norm.weight", HIDDEN)
    if not tied:
        matrix("output.weight", VOCAB, HIDDEN, "f16")
    for layer in range(LAYERS):
        block = f"blk.{layer}."
        norm(block + "attn_norm.weight", HIDDEN)
        matrix(block + "attn_q.weight", HEADS * head, HIDDEN, "q8_0")
        matrix(block + "attn_k.weight", KV_HEADS * head, HIDDEN, "q8_0")
        matrix(block + "attn_v.weight", KV_HEADS * head, HIDDEN, "f16")
        matrix(block + "attn_output.weight", HIDDEN, HEADS * head, "q8_0")
        if architecture == "qwen3":
            norm(block + "attn_q_norm.weight", head)
            norm(block + "attn_k_norm.weight", head)
        norm(block + "ffn_norm.weight", HIDDEN)
        matrix(block + "ffn_gate.weight", FFN, HIDDEN, "q8_0")
        matrix(block + "ffn_up.weight", FFN, HIDDEN, "f16")
        matrix(block + "ffn_down.weight", HIDDEN, FFN, "q8_0")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def whole_model(source, destination):
    """The conversion this replaces: load all of it, then save it."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(source.parent),
        gguf_file=source.name,
        quantization_config=transformers.GgufConfig(dequantize=True),
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        local_files_only=True,
    )
    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config
    if hasattr(model, "hf_quantizer"):
        del model.hf_quantizer
    model.save_pretrained(destination, safe_serialization=True)


def weights(directory):
    found = {}
    for file in sorted(directory.glob("*.safetensors")):
        with safetensors.safe_open(str(file), "pt") as handle:
            found |= {name: handle.get_tensor(name) for name in handle.keys()}
    return found


def saved(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
@pytest.mark.parametrize("tied", [True, False])
def test_a_gguf_converted_one_tensor_at_a_time_matches_the_whole_model_save(
    tmp_path, monkeypatch, architecture, tied
):
    source = tiny_gguf(tmp_path / "model.gguf", architecture, tied=tied)
    expected, actual = tmp_path / "whole", tmp_path / "per-tensor"
    actual.mkdir()
    whole_model(source, expected)
    # Small enough that the vocabulary is read in chunks and spans shards.
    monkeypatch.setattr(worker, "CHUNK_ROWS", 8)

    assert worker.dense_by_tensor(source, actual, shard_bytes=40_000)

    reference, converted = weights(expected), weights(actual)
    assert converted.keys() == reference.keys()
    assert ("lm_head.weight" in converted) is not tied
    for name, tensor in reference.items():
        assert converted[name].dtype == torch.bfloat16
        assert torch.equal(converted[name], tensor), name
    assert len(list(actual.glob("*.safetensors"))) > 1
    index = saved(actual, "model.safetensors.index.json")
    assert set(index["weight_map"]) == set(reference)
    assert all((actual / file).is_file() for file in index["weight_map"].values())
    for name in ("config.json", "generation_config.json"):
        assert saved(actual, name) == saved(expected, name)
    assert "quantization_config" not in saved(actual, "config.json")


def test_one_shard_is_saved_under_the_single_file_name(tmp_path):
    source = tiny_gguf(tmp_path / "model.gguf", "qwen3", tied=True)
    actual = tmp_path / "per-tensor"
    actual.mkdir()

    assert worker.dense_by_tensor(source, actual)

    assert sorted(p.name for p in actual.glob("*.safetensors")) == ["model.safetensors"]
    assert not (actual / "model.safetensors.index.json").exists()


def test_a_fix_up_that_joins_tensors_keeps_the_whole_model_load(tmp_path, monkeypatch):
    source = tiny_gguf(tmp_path / "model.gguf", "llama", tied=True)
    actual = tmp_path / "per-tensor"
    actual.mkdir()
    monkeypatch.setattr(
        worker, "PER_TENSOR", worker.PER_TENSOR - {"LlamaTensorProcessor"}
    )

    assert not worker.dense_by_tensor(source, actual)
    assert list(actual.iterdir()) == []
