"""Bounded, isolated trainability probes. Never infer support from a tokenizer."""

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from ..config import Config
from ..storage import private_dir, write_json


def support_reason(model, backend_kind=None):
    # Only the installed backend and an actual guarded backward pass decide
    # support. Architecture names and historical performance are not exclusions.
    if backend_kind is None:
        from ..backends import kind_for
        from ..platform import current

        backend_kind = kind_for(current())
    for package in ("mlx-lm",) if backend_kind == "mlx" else ("transformers", "peft"):
        try:
            importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            return f"{package} is not installed"
    return None


def cache_key(model):
    from ..backends import kind_for
    from ..platform import current

    kind = kind_for(current())
    packages = ("mlx-lm", "mlx") if kind == "mlx" else ("transformers", "peft", "torch")
    weights = [
        (p.name, p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(model.glob("*.safetensors"))
    ]
    payload = {
        "path": str(model.resolve()),
        "config": (model / "config.json").read_text(encoding="utf-8"),
        "weights": weights,
        "backend": kind,
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "protocol": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def invoke_probe(model, destination, timeout=60):
    from ..memory import MemoryPressure, guarded_run

    try:
        result = guarded_run(
            [
                sys.executable,
                "-m",
                "songe.sleep.probe",
                "--model",
                str(model),
                "--output",
                str(destination),
                "--child",
            ],
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except MemoryPressure as exc:
        return {
            "model": str(model),
            "passed": False,
            "retryable": True,
            "reason": str(exc),
            "available_bytes": exc.memory.available,
            "total_bytes": exc.memory.total,
            "memory_stop": exc.reason,
        }
    except subprocess.TimeoutExpired:
        return {
            "model": str(model),
            "passed": False,
            "retryable": True,
            "reason": f"Trainability probe exceeded its {timeout}-second time limit",
        }
    if result.returncode or not destination.exists():
        return {
            "model": str(model),
            "passed": False,
            "reason": "Trainability probe could not complete; see backend/hardware diagnostics",
        }
    return json.loads(destination.read_text(encoding="utf-8"))


def probe_models(models, root, hardware, timeout=60):
    directory = private_dir(root / "probes")
    results = []
    for model in models:
        reason = (
            support_reason(model)
            if hardware.available
            else "Hardware probes unavailable"
        )
        if reason:
            results.append({"model": str(model), "passed": False, "reason": reason})
            continue
        key = cache_key(model)
        path = directory / f"{key}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = invoke_probe(model, path, timeout)
            if result.get("passed"):
                write_json(path, result)
        results.append(result)
    write_json(root / "trainability.json", results)
    return results


def choose(results, preferred=""):
    passed = [r for r in results if r.get("passed")]
    if not passed:
        raise ValueError(
            "No model passed trainability and memory checks; see trainability.json"
        )
    preferred = re.sub(r"[^a-z0-9]", "", preferred.lower())
    matches = [
        r
        for r in passed
        if preferred
        and preferred in re.sub(r"[^a-z0-9]", "", Path(r["model"]).name.lower())
    ]
    return min(
        matches or passed, key=lambda r: r["step_seconds"] + 128 / r["generation_tps"]
    )


def run_probe(model, output):
    from ..backends import choose
    from ..platform import current

    reason = support_reason(model)
    if reason:
        write_json(output, {"model": str(model), "passed": False, "reason": reason})
        return
    config = Config(
        model=str(model),
        max_sequence=256,
        response_tokens=64,
        rank=16,
        accumulation=1,
        token_chunk=8,
    )
    started = time.monotonic()
    backend = choose(config, current())(config, lambda: None)
    if hasattr(backend, "prepare"):
        backend.prepare()
    prompt = backend.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Explain why a seed needs water in one sentence.",
            }
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    _, _, generation = backend.sample(prompt, seed=7613)
    backend.config.response_tokens = 8
    step = {
        "context": [{"role": "user", "content": "What is two plus two?"}],
        "hindsight": [{"role": "user", "content": "The correct result was four."}],
        "tools": [],
    }
    optimizer = backend.optimizer(config.learning_rate)
    training = time.monotonic()
    loss, gradients, _ = backend.train_step(step, optimizer, seed=7613)
    backend.apply(gradients, optimizer)
    seconds = time.monotonic() - training
    result = {
        "model": str(model),
        "passed": True,
        "reason": "One checkpointed distillation step completed",
        "step_seconds": seconds,
        "generation_tps": generation.generation_tps,
        "prefill_tps": generation.prompt_tps,
        "peak_memory_gb": backend.peak_memory() / 1e9,
        "probe_seconds": time.monotonic() - started,
        "rank": 16,
        "sequence_limit": 256,
        "response_tokens": 8,
        "loss_finite": __import__("math").isfinite(loss),
    }
    result["passed"] = result["loss_finite"] and result["generation_tps"] > 0
    write_json(output, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.child:
        run_probe(args.model.resolve(), args.output)
    else:
        result = invoke_probe(args.model.resolve(), args.output)
        write_json(args.output, result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
