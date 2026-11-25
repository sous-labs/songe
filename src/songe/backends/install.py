"""Plan accelerator dependencies in a private Songe environment."""

import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
import venv
from pathlib import Path

import httpx

from .. import __version__
from ..storage import atomic_write

WHEEL_ROOT = "https://download.pytorch.org/whl"


def runtime_python(root, folder="runtime"):
    """The private service installation's interpreter (``runtime``)."""
    return root / folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# The GGUF converter toolkit (torch, peft, gguf) an MLX Mac gets on first use.
# Its own folder: ``runtime`` is the service's installation, which the
# upgrade check queries and rebinds the service to; the converter has no MLX.
CONVERTER = "converter"


def converter_runtime_python(root):
    return runtime_python(root, CONVERTER)


def stack_ready(kind):
    try:
        packages = {"torch", "transformers", "peft", "accelerate"}
        if kind == "cuda":
            packages.add("bitsandbytes")
        for package in packages:
            importlib.metadata.distribution(package)
        from packaging.requirements import Requirement

        for raw in importlib.metadata.requires("songe") or []:
            requirement = Requirement(raw)
            if requirement.name in packages and (
                requirement.marker is None
                or any(
                    requirement.marker.evaluate({"extra": extra})
                    for extra in ("torch", kind)
                )
            ):
                if (
                    importlib.metadata.version(requirement.name)
                    not in requirement.specifier
                ):
                    return False
    except ImportError:
        return False
    # Probe in a child: init and the proxy keep GPU runtimes out of their process.
    code = accelerator_check(kind)
    return (
        subprocess.run([sys.executable, "-c", code], capture_output=True).returncode
        == 0
    )


def accelerator_check(kind):
    return (
        "import torch; "
        + {
            "cuda": "assert torch.cuda.is_available() and not torch.version.hip",
            "rocm": "assert torch.cuda.is_available() and torch.version.hip",
            "xpu": "assert torch.xpu.is_available()",
        }[kind]
    )


def cuda_driver_version():
    """(major, minor) from the CUDA driver library itself, or None."""
    import ctypes

    for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        version = ctypes.c_int()
        if library.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value:
            return version.value // 1000, version.value % 1000 // 10
    return None


def driver_runtime(kind):
    if kind == "cuda":
        try:
            raw = subprocess.check_output(["nvidia-smi"], text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            raw = ""
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", raw)
        # Some containers' nvidia-smi leaves the runtime line out.
        if not match and (found := cuda_driver_version()):
            return found
    elif kind == "rocm":
        raw = next(
            (
                p.read_text(encoding="utf-8")
                for p in (
                    Path("/opt/rocm/.info/version"),
                    Path("/opt/rocm/.info/version-dev"),
                )
                if p.is_file()
            ),
            "",
        )
        if not raw:
            raw = subprocess.check_output(
                ["hipconfig", "--version"], text=True, timeout=5
            )
        match = re.search(r"(\d+)\.(\d+)", raw)
    else:
        return None
    if not match:
        raise ValueError(
            f"Cannot measure the {kind} driver runtime; check the accelerator driver"
        )
    return tuple(map(int, match.groups()))


def wheel_index(kind, runtime=None):
    if kind == "xpu":
        return WHEEL_ROOT + "/xpu"
    runtime = runtime or driver_runtime(kind)
    with httpx.Client(trust_env=False, timeout=15) as client:
        response = client.get(WHEEL_ROOT + "/")
        response.raise_for_status()
    variants = set(re.findall(r">(cu\d+|rocm\d+\.\d+(?:\.\d+)?)<", response.text))
    compatible = []
    for variant in variants:
        version = (
            (int(variant[2:-1]), int(variant[-1]))
            if variant.startswith("cu")
            else tuple(map(int, variant[4:].split(".")))
        )
        if (
            variant.startswith("cu" if kind == "cuda" else "rocm")
            and version[:2] <= runtime
            and (kind != "rocm" or version[:2] == runtime)
        ):
            compatible.append((version, variant))
    if not compatible:
        raise ValueError(
            f"No official PyTorch wheel index matches {kind} runtime {runtime[0]}.{runtime[1]}"
        )
    return WHEEL_ROOT + "/" + max(compatible)[1]


def package_requirement():
    distribution = importlib.metadata.distribution("songe")
    direct = distribution.read_text("direct_url.json")
    if direct:
        url = json.loads(direct).get("url", "")
        if url.startswith("file:"):
            return f"songe[torch] @ {url}"
    return f"songe[torch]=={__version__}"


def plan(platform, root):
    kinds = {a.kind for a in platform.accelerators()}
    if "metal" in kinds:
        return []  # Apple Silicon gets MLX from the base package markers.
    kind = next((k for k in ("cuda", "rocm", "xpu") if k in kinds), None)
    if kind is None:
        return []
    if stack_ready(kind):
        return []
    python = str(runtime_python(root))
    index = wheel_index(kind)
    commands = [
        [python, "-m", "pip", "install", "torch>=2.11,<3", "--index-url", index],
        [python, "-m", "pip", "install", package_requirement()],
    ]
    if kind == "cuda":
        commands.append([python, "-m", "pip", "install", "bitsandbytes>=0.45,<1"])
    return commands


def display(commands):
    for command in commands:
        print(
            subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        )


def install(commands, root, folder="runtime"):
    if not commands:
        return
    venv.EnvBuilder(with_pip=True).create(root / folder)
    log = root / f"{folder}-install.log"
    atomic_write(log, b"")
    with log.open("a", encoding="utf-8") as output:
        for command in commands:
            try:
                subprocess.run(
                    command, check=True, stdout=output, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Runtime installation failed. Details: {log}"
                ) from exc
    for command in commands:
        if "--index-url" in command:
            index = command[command.index("--index-url") + 1].rsplit("/", 1)[-1]
            kind = (
                "cuda"
                if index.startswith("cu")
                else "rocm"
                if index.startswith("rocm")
                else "xpu"
            )
            result = subprocess.run(
                [str(runtime_python(root)), "-c", accelerator_check(kind)],
                capture_output=True,
            )
            if result.returncode:
                raise RuntimeError(
                    f"PyTorch was installed but cannot use the {kind} accelerator. "
                    "Check the local driver and rerun init; agent files were preserved."
                )


def conversion_plan(ref, artifact_kind, platform, root):
    """MLX training needs no install; GGUF converters use a separate CPU stack."""
    if "metal" not in {a.kind for a in platform.accelerators()} or (
        ref.format != "gguf" and not artifact_kind.startswith("gguf-")
    ):
        return []
    try:
        for package in ("torch", "peft", "gguf"):
            importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return [
            [str(runtime_python(root)), "-m", "pip", "install", package_requirement()]
        ]
    return []


CONVERTER_MODULES = ("torch", "transformers", "peft", "gguf")


def converter_python(platform):
    """Interpreter for GGUF conversion and torch merges.

    A Mac set up for an MLX model has no torch; a GGUF model followed later
    gets the private converter runtime the first time it needs one, instead
    of failing the same import on every retry."""
    import importlib.util

    if all(importlib.util.find_spec(m) for m in CONVERTER_MODULES):
        return sys.executable
    data_dir = getattr(platform, "data_dir", None)
    accelerators = getattr(platform, "accelerators", lambda: [])
    # Only an MLX Mac lacks torch by design; CUDA and ROCm runtimes have it.
    if data_dir is None or "metal" not in {a.kind for a in accelerators()}:
        return sys.executable
    root = Path(home(platform).data_dir())
    probe = ["-c", "import " + ", ".join(CONVERTER_MODULES)]

    def ready(python):
        return (
            python.exists()
            and not subprocess.run([str(python), *probe], capture_output=True).returncode
        )

    # A private service installation init made for a GGUF model already
    # carries the converter modules; use it rather than a second copy.
    if ready(runtime_python(root)):
        return str(runtime_python(root))
    python = converter_runtime_python(root)
    if not ready(python):
        announce(root, "Downloading the converter toolkit (about 1 GB, one time)")
        install(
            [[str(python), "-m", "pip", "install", package_requirement()]],
            root,
            CONVERTER,
        )
        announce(root, "converter toolkit ready")
    return str(python)


def home(platform):
    """The installation's own data directory: conversions run on a platform
    rooted at their scratch volume, but the toolkit is installed once."""
    from ..platform import Rooted

    while isinstance(platform, Rooted) and hasattr(platform.platform, "data_dir"):
        platform = platform.platform
    return platform


def announce(root, text):
    """A line in the dashboard's Flow; best effort, never fails the work."""
    if not (root / "experience.sqlite3").exists():
        return
    try:
        from ..log import Store

        Store(root).event("install", text, model_id="")
    except Exception:  # noqa: BLE001 - a status line only
        pass


def interpreter_for(platform, runner):
    """A real run provisions the converter; an injected runner never installs.

    Conversion and merges run through artifacts.runtime.guarded_run (which
    wraps memory.guarded_run); comparing against the inner one alone meant a
    real run never reached converter_python."""
    from .. import memory
    from ..artifacts import runtime

    runner = getattr(runner, "func", runner)  # a partial binding check=
    real = runner is runtime.guarded_run or runner is memory.guarded_run
    return converter_python(platform) if real else sys.executable
