"""Resumable optimizer units shared by the training backends."""

import json
import math
import os
import shutil
import time
import uuid

from ..steps import digest, revision_id
from ..storage import write_json


def sample_plan(pool, anchors, fraction):
    """One pass over the selected pool, interleaved with finite rehearsal.

    Rehearsal targets the configured share, cycling only the anchor set when
    needed. A full rehearsal share means one pass over all available anchors
    alongside the pool. An anchor never replaces a selected experience.
    """
    if not pool:
        return []
    count = (
        0
        if not anchors
        else (
            math.ceil(len(pool) * fraction / (1 - fraction))
            if fraction < 1
            else len(anchors)
        )
    )
    plan, anchor = [], 0
    for index in range(len(pool)):
        plan.append((False, index))
        while anchor < (index + 1) * count // len(pool):
            plan.append((True, anchor % len(anchors)))
            anchor += 1
    return plan


CHECKPOINT_MIN_SECONDS = 30.0


def _inplace(value, method, other):
    """value <op>= other without a copy for tensors (a large adapter's
    gradients are gigabytes); plain numbers (test doubles) get the result."""
    if hasattr(value, method):
        return getattr(value, method)(other)
    return {"mul_": value * other, "add_": value + other, "div_": value / other}[method]


def train(backend, pool, anchors, work, progress):
    config = backend.config
    if not pool:
        return {"iterations": 0, "steps_learned": 0, "trained_revisions": {}}
    # Freeze the sample order independently of each slice's memory sizing.
    plan_path = work / "training-plan.json"
    identity = {"pool": digest(pool), "anchors": digest(anchors)}
    if plan_path.exists():
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if any(saved[k] != value for k, value in identity.items()):
            # Free memory decides which examples fit this slice; a smaller fit
            # pauses the cycle (resumable) instead of discarding its progress.
            from . import Interrupted

            raise Interrupted("waiting for memory to fit this cycle's examples")
        plan = saved["samples"]
        accumulation = saved.get("accumulation", config.accumulation)
    else:
        plan = sample_plan(pool, anchors, config.anchor_fraction)
        accumulation = config.accumulation
    write_json(plan_path, identity | {"samples": plan, "accumulation": accumulation})
    checkpoint = work / "checkpoint.json"
    meta = (
        json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint.exists()
        else None
    )
    iteration = meta["iteration"] if meta else 0
    cursor = meta["sample_cursor"] if meta else 0
    budget = math.ceil(len(plan) / accumulation)
    backend.prepare()
    warmup = min(10, max(1, budget // 10))  # Same short warmup as mlx-lm.

    def schedule(index):
        if index < warmup:
            return config.learning_rate * (index + 1) / warmup
        return (
            config.learning_rate
            * (
                1
                + math.cos(math.pi * min(1, (index - warmup) / max(1, budget - warmup)))
            )
            / 2
        )

    optimizer = backend.optimizer(schedule)
    losses, learned = [], set()
    is_torch = backend.kind == "torch"
    if not is_torch:
        from mlx.utils import tree_flatten, tree_map, tree_unflatten

    if meta:
        generation = work / meta["directory"]
        backend.restore(generation)
        if is_torch:
            state = backend.torch.load(
                generation / "optimizer.pt",
                map_location=backend.device,
                weights_only=True,
            )
            optimizer.load_state_dict(state["optimizer"])
            backend._updates = state["updates"]
        else:
            state = backend.mx.load(str(generation / "optimizer.safetensors"))
            optimizer.state = tree_unflatten(list(state.items()))
        iteration, losses = meta["iteration"], meta["losses"]
        learned = set(meta.get("learned", []))
    started = time.monotonic()

    from . import Interrupted

    saved = {"at": time.monotonic(), "cost": 0.0, "iteration": iteration}

    def checkpoint_due():
        # A checkpoint is the adapter plus optimizer state (gigabytes for a
        # large model). Writing one after every step wears the disk and can
        # cost more than the step, so they are spaced to at most a tenth of
        # the training time, and one is written when the run is interrupted.
        return time.monotonic() - saved["at"] >= max(
            CHECKPOINT_MIN_SECONDS, 10 * saved["cost"]
        )

    def save_checkpoint():
        began = time.monotonic()
        # Never overwrite the generation named by the currently durable manifest.
        name = f"checkpoint-{iteration:06d}-{uuid.uuid4().hex}"
        directory = work / name
        backend.save(directory)
        if is_torch:
            backend.torch.save(
                {"optimizer": optimizer.state_dict(), "updates": backend._updates},
                directory / "optimizer.pt",
            )
        else:
            backend.mx.save_safetensors(
                str(directory / "optimizer.safetensors"),
                dict(tree_flatten(optimizer.state)),
            )
        for path in directory.iterdir():
            if path.is_file():
                with path.open("r+b") as handle:
                    os.fsync(handle.fileno())
        if os.name != "nt":
            handle = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
        write_json(
            checkpoint,
            {
                "directory": name,
                "iteration": iteration,
                "sample_cursor": cursor,
                "losses": losses,
                "learned": sorted(learned),
            },
        )
        for old in work.glob("checkpoint-*"):
            if old.is_dir() and old.name != name:
                shutil.rmtree(old)
        saved.update(
            at=time.monotonic(), cost=time.monotonic() - began, iteration=iteration
        )

    if not checkpoint.exists():
        save_checkpoint()
    # Measured in this run, for the time left: steps differ in length from
    # the calibration's (long scenes cost more).
    step_times = []
    try:
        while cursor < len(plan):
            # A step boundary: the slice deadline may stop here, never midway.
            getattr(backend, "boundary", backend.check)()
            step_started = time.monotonic()
            gradients, loss, pending, tokens = None, 0.0, set(), 0
            batch = plan[cursor : cursor + accumulation]
            for micro, (is_anchor, index) in enumerate(batch):
                step = anchors[index] if is_anchor else pool[index]
                value, grad, count = backend.train_step(
                    step, optimizer, seed=7613 + cursor + micro
                )
                if not math.isfinite(value):
                    raise ValueError("Non-finite training loss")
                if count < 1:
                    raise ValueError("Training sample has no loss tokens")
                if is_torch:
                    # In place: each copy of a large adapter's gradients is
                    # gigabytes of accelerator memory.
                    grad = {n: _inplace(g, "mul_", count) for n, g in grad.items()}
                else:
                    grad = tree_map(lambda g, n=count: g * n, grad)
                if not is_anchor:
                    pending.add(step["id"])
                if gradients is None:
                    gradients = grad
                elif is_torch:
                    gradients = {
                        n: _inplace(g, "add_", grad[n]) for n, g in gradients.items()
                    }
                else:
                    gradients = tree_map(lambda a, b: a + b, gradients, grad)
                if not is_torch:
                    # Materialize the running sum before the next forward pass;
                    # retaining lazy gradient graphs would defeat serial sizing.
                    backend.mx.eval(gradients)
                del grad
                # Return this micro-batch's freed buffers before the next one;
                # otherwise accumulation grows the allocator cache by each shape.
                getattr(backend, "release", lambda: None)()
                loss += value * count
                tokens += count
            if is_torch:
                gradients = {n: _inplace(g, "div_", tokens) for n, g in gradients.items()}
            else:
                gradients = tree_map(lambda x, n=tokens: x / n, gradients)
                gradients, norm = backend.optim.clip_grad_norm(gradients, max_norm=1.0)
                if not math.isfinite(float(norm)):
                    raise ValueError("Non-finite training gradient")
            backend.apply(gradients, optimizer)
            iteration += 1
            cursor += len(batch)
            learned.update(pending)
            losses.append(loss / tokens)
            step_times.append(time.monotonic() - step_started)
            if checkpoint_due():
                save_checkpoint()
            # Each step's loss and position go out with its progress, so the
            # record shows them before the next (spaced) checkpoint.
            progress(
                cursor,
                len(plan),
                {
                    "iteration": iteration,
                    "losses": list(losses),
                    "seconds_per_step": sum(step_times) / len(step_times),
                },
            )
    except Interrupted:
        # Stopped between steps (a request arrived): keep the steps done. Any
        # other error may have left a step half applied, so nothing is saved.
        if iteration != saved["iteration"]:
            save_checkpoint()
        raise
    if iteration != saved["iteration"]:
        save_checkpoint()
    backend.save(work / "adapter")
    return {
        "iterations": iteration,
        "samples_trained": cursor,
        "steps_learned": len(learned),
        "trained_revisions": {
            s["id"]: revision_id(s) for s in pool if s["id"] in learned
        },
        "loss": losses[-1] if losses else None,
        "seconds": time.monotonic() - started,
        "peak_memory_gb": backend.peak_memory() / 1e9,
        "layer_deltas": backend.layer_deltas(),
    }
