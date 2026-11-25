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
