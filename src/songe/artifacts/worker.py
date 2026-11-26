"""Guarded PEFT merge worker; does not import any MLX modules."""

import argparse
from pathlib import Path


def merge(base, adapter, destination):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        local_files_only=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(destination, safe_serialization=True)
    AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=False
    ).save_pretrained(destination)


def mlx_peft(base, adapter, destination):
    """Translate standard MLX LoRA factors to PEFT without changing their delta."""
    import json

    import mlx.core as mx

    metadata = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    weights = mx.load(str(adapter / "adapters.safetensors"))
    result, targets = {}, set()
    parameters = metadata["lora_parameters"]
    for name, value in weights.items():
        if not name.endswith((".lora_a", ".lora_b")) or value.ndim != 2:
            raise ValueError(
                "This MLX adapter cannot be represented by the upstream's PEFT LoRA format"
            )
        prefix, factor = name.rsplit(".", 1)
        result[
            f"base_model.model.{prefix}.{factor.replace('_a', '_A').replace('_b', '_B')}.weight"
        ] = value.T
        targets.add(prefix.rsplit(".", 1)[-1])
    destination.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(destination / "adapter_model.safetensors"),
        result,
        metadata={"format": "pt"},
    )
    (destination / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "inference_mode": True,
                "base_model_name_or_path": str(base),
                "r": parameters["rank"],
                "lora_alpha": parameters["rank"] * parameters["scale"],
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": sorted(targets),
            }
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-peft", action="store_true")
    parser.add_argument("base", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    (mlx_peft if args.mlx_peft else merge)(args.base, args.adapter, args.destination)


if __name__ == "__main__":
    from songe.telemetry import diagnostic_main

    diagnostic_main(main, stoppable=True)
