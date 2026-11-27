"""Guarded synchronous children for conversion and merging, plus atomic publish.

Memory protection is the one in songe.memory; this module only adapts its
errors to what artifact callers expect.
"""

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class RunPolicy:
    timeout: float | None = 3600.0  # Conversion deadline, not a capacity threshold.


def guarded_run(
    argv, *, platform=None, check=lambda: None, policy=None, env=None, child_bytes=None
):
    from .. import memory
    from ..telemetry import configure, failure_hint, log_child_stderr

    diagnostics = os.environ.get("SONGE_DIAGNOSTICS_DIR")
    if not diagnostics and platform is not None:
        from pathlib import Path

        configure(Path(platform.data_dir()))
    policy = policy or RunPolicy()
    argv = [str(a) for a in argv]
    child_env = os.environ | (env or {})
    if os.environ.get("SONGE_DIAGNOSTICS_DIR"):
        child_env["SONGE_DIAGNOSTICS_DIR"] = os.environ["SONGE_DIAGNOSTICS_DIR"]
    try:
        result = memory.guarded_run(
            argv,
            policy.timeout,
            platform=platform,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            child_bytes=child_bytes,
        )
    except memory.MemoryPressure as exc:
        raise MemoryError(f"Conversion stopped: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Conversion exceeded {policy.timeout:g} seconds") from exc
    if result.returncode:
        log_child_stderr(result.stderr)
        raise RuntimeError(failure_hint("Conversion"))
    return result


def publish(staging, destination):
    """Flush a complete staged artifact before exposing its immutable name."""
    if destination.exists():
        raise FileExistsError(destination)
    from ..storage import private_tree

    private_tree(staging)
    files = (
        [staging]
        if staging.is_file()
        else [p for p in staging.rglob("*") if p.is_file()]
    )
    for file in files:
        # Windows fsync requires a writable handle; r+b preserves the bytes.
        with file.open("r+b") as handle:
            os.fsync(handle.fileno())
    # Windows does not support POSIX open/fsync on a directory. File handles
    # are flushed on both platforms; directory entries are flushed on POSIX.
    if os.name != "nt" and staging.is_dir():
        directories = [p for p in staging.rglob("*") if p.is_dir()] + [staging]
        for directory in directories:
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    staging.rename(destination)
    if os.name != "nt":
        fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return destination


def owned_marker(path):
    return (
        path / "songe_owned.json"
        if path.is_dir()
        else path.with_name(path.name + ".songe_owned.json")
    )


def remove_owned(path, installation_id=None):
    import json
    import shutil

    marker = owned_marker(path)
    if marker.is_file():
        if (
            installation_id is not None
            and json.loads(marker.read_text(encoding="utf-8")).get("installation_id")
            != installation_id
        ):
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
            marker.unlink()
