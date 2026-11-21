"""Entrypoint run only under the artifact memory guard."""

import argparse
from pathlib import Path


def convert(kind, source, destination):
    if kind == "mlx":
        import importlib.util

        if importlib.util.find_spec("mlx_lm.convert") is None:
            raise RuntimeError("mlx_lm.convert requires the MLX stack")
        from mlx_lm.convert import convert as mlx_convert

        output = destination / "model"  # mlx-lm requires a nonexistent output path.
        mlx_convert(hf_path=str(source), mlx_path=str(output), dtype="bfloat16")
        for file in output.iterdir():
            file.rename(destination / file.name)
        output.rmdir()
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GgufConfig

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

    diagnostic_main(main)
