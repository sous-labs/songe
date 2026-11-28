"""Entrypoint run only under the artifact memory guard."""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def mlx_input(source, scratch):
    try:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        config = {}
    if config.get("model_type") != "qwen3_5_text":
        yield source
        return

    # Transformers dequantizes Qwen3.5 GGUF to its text-only model type.
    # The MLX loader accepts the flat text config and weights under qwen3_5.
    # Keep the original Transformers checkpoint usable as-is.
    with TemporaryDirectory(prefix="mlx-input-", dir=scratch) as temporary:
        prepared = Path(temporary)
        config["model_type"] = "qwen3_5"
        (prepared / "config.json").write_text(json.dumps(config), encoding="utf-8")
        for entry in source.iterdir():
            if entry.name != "config.json":
                (prepared / entry.name).symlink_to(entry.resolve())
        yield prepared


# Transformers' fix-ups that turn one GGUF tensor into one weight by itself.
# The others join or split tensors (MoE experts, GPT-2's output head) and
# keep the whole-model load.
PER_TENSOR = {
    "TensorProcessor",
    "LlamaTensorProcessor",
    "BloomTensorProcessor",
    "MambaTensorProcessor",
    "NemotronTensorProcessor",
    "Gemma2TensorProcessor",
    "Lfm2TensorProcessor",
}
# bf16 bytes held before they are written out.
SHARD_BYTES = 1 << 30
# Quantized rows dequantized at a time: numpy's dequantizers make several
# float32 temporaries of their input, and a vocabulary matrix is the largest.
CHUNK_ROWS = 4096


def dense_by_tensor(source, destination, shard_bytes=SHARD_BYTES):
    """Write the GGUF's weights as bf16 safetensors, one tensor at a time.

    It reads them the way transformers' whole-model load does (its GGUF to
    transformers names, its per-architecture fix-ups, bf16 from float32), and
    writes the files save_pretrained writes, without the model: it needs one
    shard and one tensor, not the dequantized tensors and then the model they
    are copied into (a 4B's conversion stopped at 13.5 GB and rising, live).

    Returns False, having written nothing, where that is not the same thing:
    an architecture transformers now loads another way, a fix-up that joins
    tensors, weights kept in float32, or a weight the file does not have."""
    import numpy as np
    import torch
    from gguf import GGUFReader, dequantize
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers import modeling_gguf_pytorch_utils as gguf_utils

    try:
        from transformers.modeling_utils import remove_tied_weights_from_state_dict
    except ImportError:
        return False
    reader = GGUFReader(source, mode="r")
    architecture = str(reader.get_field("general.architecture").contents())
    try:
        from transformers.integrations.gguf.utils import is_gguf_arch_supported
    except ImportError:
        pass
    else:
        if is_gguf_arch_supported(architecture):
            return False
    processor_type = gguf_utils.TENSOR_PROCESSORS.get(
        architecture, gguf_utils.TensorProcessor
    )
    if processor_type.__name__ not in PER_TENSOR:
        return False
    parsed = gguf_utils.load_gguf_checkpoint(str(source), return_tensors=False)
    config = AutoConfig.from_pretrained(
        str(source.parent),
        gguf_file=source.name,
        local_files_only=True,
        trust_remote_code=False,
    )
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    if getattr(skeleton, "_keep_in_fp32_modules", None) or getattr(
        skeleton, "_keep_in_fp32_modules_strict", None
    ):
        return False
    state = skeleton.state_dict()
    # A tied output head is saved once, under the name save_pretrained keeps.
    kept = set(remove_tied_weights_from_state_dict(dict(state), skeleton))
    processor = processor_type(config=parsed["config"])
    names = gguf_utils.get_gguf_hf_weights_map(skeleton, processor)
    tensors = [t for t in reader.tensors if names.get(t.name) in kept]
    if {names[t.name] for t in tensors} != kept or len(tensors) != len(kept):
        return False

    files, shard, held = [], {}, 0

    def write():
        nonlocal shard, held
        file = destination / f"model-{len(files) + 1:05d}.safetensors"
        save_file(shard, str(file), metadata={"format": "pt"})
        files.append((file, list(shard)))
        shard, held = {}, 0

    total = parameters = 0
    for tensor in tensors:
        data = tensor.data
        if data.ndim > 1 and len(data) > CHUNK_ROWS:
            first = dequantize(data[:CHUNK_ROWS], tensor.tensor_type)
            weights = np.empty((len(data), *first.shape[1:]), dtype=first.dtype)
            weights[:CHUNK_ROWS] = first
            for start in range(CHUNK_ROWS, len(data), CHUNK_ROWS):
                weights[start : start + CHUNK_ROWS] = dequantize(
                    data[start : start + CHUNK_ROWS], tensor.tensor_type
                )
        else:
            weights = dequantize(data, tensor.tensor_type)
        weights = processor.process(
            weights=weights,
            name=tensor.name,
            tensor_key_mapping=names,
            parsed_parameters=parsed,
        ).weights
        name = names[tensor.name]
        # A small float32 tensor is a read-only view of the mapped file.
        value = torch.from_numpy(np.require(weights, requirements=["C", "W"])).to(
            torch.bfloat16
        )
        del weights
        if value.shape != state[name].shape:
            raise ValueError(
                f"GGUF tensor {tensor.name} has shape {tuple(value.shape)}; "
                f"{name} needs {tuple(state[name].shape)}"
            )
        if shard and held + value.nbytes > shard_bytes:
            write()
        shard[name] = value
        held += value.nbytes
        total += value.nbytes
        parameters += value.numel()
    if shard:
        write()
    if len(files) == 1:
        files[0][0].rename(destination / "model.safetensors")
    else:
        weight_map = {}
        for number, (file, contents) in enumerate(files, 1):
            final = f"model-{number:05d}-of-{len(files):05d}.safetensors"
            file.rename(destination / final)
            weight_map |= dict.fromkeys(contents, final)
        index = {
            "metadata": {"total_parameters": parameters, "total_size": total},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (destination / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    # What save_pretrained adds to the config, without the GGUF's quantization.
    config.dtype = "bfloat16"
    config.architectures = [type(skeleton).__name__]
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    config.save_pretrained(destination)
    if (
        getattr(skeleton, "generation_config", None) is not None
        and skeleton.can_generate()
    ):
        skeleton.generation_config.save_pretrained(destination)
    return True


def convert(kind, source, destination):
    if kind == "mlx":
        if sys.platform != "darwin":
            raise RuntimeError("mlx_lm.convert requires macOS")
        from mlx_lm.convert import convert as mlx_convert

        output = destination / "model"  # mlx-lm requires a nonexistent output path.
        with mlx_input(source, destination) as prepared:
            mlx_convert(hf_path=str(prepared), mlx_path=str(output), dtype="bfloat16")
        for file in output.iterdir():
            file.rename(destination / file.name)
        output.rmdir()
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GgufConfig

    if not dense_by_tensor(source, destination):
        model = AutoModelForCausalLM.from_pretrained(
            str(source.parent),
            gguf_file=source.name,
            quantization_config=GgufConfig(dequantize=True),
            dtype=torch.bfloat16,
            device_map={"": "cpu"},
            local_files_only=True,
            trust_remote_code=False,
        )
        # Some transformer versions retain the input GGUF quantization metadata
        # after dequantizing; the saved dense cache must not try to reload a GGUF.
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config
        if hasattr(model, "hf_quantizer"):
            del model.hf_quantizer
        model.save_pretrained(destination, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(source.parent),
        gguf_file=source.name,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.save_pretrained(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["gguf", "mlx"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.kind, args.source, args.destination)


if __name__ == "__main__":
    from songe.telemetry import diagnostic_main

    diagnostic_main(main, stoppable=True)
