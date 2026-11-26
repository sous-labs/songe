"""Official llama.cpp tools, pinned by commit/release and SHA-256.

Binary digests are from ggml-org's release API for b12247. The source archive
is pinned to that release's full commit; its digest was independently measured.
CPU quantization binaries work on CUDA/ROCm machines without vendor runtimes.
"""

import hashlib
import json
import os
import platform as host_platform
import shutil
import stat
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import httpx

RELEASE = "b12247"
COMMIT = "c1a4febe9d107de438922f7c8ebb88108d235ea4"
SOURCE_SHA256 = "cad9f716dd1e443b15c0ab4427e28fb57dd3c0b6c2a72336c9c5d25b80cdc10d"
# Overrides require explicitly supplying another reviewed manifest to fetch().
BINARIES = {
    ("darwin", "arm64"): (
        "macos-arm64.tar.gz",
        "b64dc877683899be407bfe2d1776047ee137b7f1aaf1d90a25f29225f89e5f28",
    ),
    ("darwin", "x86_64"): (
        "macos-x64.tar.gz",
        "0fd38423ee9e475cfa073dafc8dbe22686ddae6fed765f90b1c1b6eb4a412fa9",
    ),
    ("linux", "x86_64"): (
        "ubuntu-x64.tar.gz",
        "b353ef1ba235d377d4370d5e5e891f2b934db6fba81a21fd6a58f50a686e719b",
    ),
    ("linux", "arm64"): (
        "ubuntu-arm64.tar.gz",
        "04800cb388b517d53920871f9ec8ec448ebf98ccf2d0e3985d1234c95e6f63e0",
    ),
    ("windows", "x86_64"): (
        "win-cpu-x64.zip",
        "5d3a384d959c0c5f5848bdabc3e395b62b524d350ccede00b349dda1bbdb82bc",
    ),
    ("windows", "arm64"): (
        "win-cpu-arm64.zip",
        "46278f9c1fc243caea3a866b0bc018f55c1ed56896b68fca73b1bd9b6fac0009",
    ),
}


@dataclass(frozen=True)
class Download:
    url: str
    sha256: str
    filename: str


@dataclass(frozen=True)
class Toolchain:
    convert_hf: Path
    convert_lora: Path
    quantize: Path
    source: Path

    def env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(self.source / "gguf-py"), env.get("PYTHONPATH")])
        )
        env["HF_HUB_OFFLINE"] = "1"
        return env


def _digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(item, directory, *, transport=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / item.filename
    if path.exists() and _digest(path) == item.sha256:
        return path
    # Never replace a valid published file with a partial download.
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=directory)
    try:
        with (
            os.fdopen(fd, "wb") as out,
            httpx.Client(
                transport=transport, follow_redirects=True, trust_env=False, timeout=120
            ) as client,
        ):
            with client.stream("GET", item.url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    out.write(chunk)
        temp = Path(temporary)
        if _digest(temp) != item.sha256:
            raise ValueError(
                f"Checksum mismatch for {item.filename}; no tools were installed"
            )
        temp.replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _target(root, name):
    # Reject both POSIX and Windows traversal, regardless of the host OS.
    if (
        PurePosixPath(name).is_absolute()
        or PureWindowsPath(name).anchor
        or ".." in PurePosixPath(name).parts
        or "\\" in name
    ):
        raise ValueError(f"Unsafe archive path: {name}")
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe archive path: {name}")
    return target


# Of llama.cpp's source tree the converters use only these (next to the
# convert_*.py scripts themselves). The rest is never run, and parts of it (the
# web UI) have paths of 220 characters: with Songe's data folder in front they
# pass Windows' default 260-character limit, and every extraction failed.
CONVERTER_PARTS = ("conversion/", "gguf-py/", "models/templates/")


def _converter_root(names):
    """The folder holding llama.cpp's converters in an archive, or None."""
    roots = {
        name[: -len("convert_hf_to_gguf.py")]
        for name in names
        if PurePosixPath(name).name == "convert_hf_to_gguf.py"
    }
    return roots.pop() if len(roots) == 1 else None


def _needed(name, root):
    """Everything outside a llama.cpp source tree; inside it, only the
    converter scripts and what they read."""
    if root is None or not name.startswith(root):
        return True
    rest = name[len(root) :]
    return "/" not in rest.rstrip("/") or rest.startswith(CONVERTER_PARTS)


def extract(archive, root):
    """Extract only regular files/dirs and internal links, never device nodes."""
    root.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            items = source.infolist()
            converters = _converter_root([item.filename for item in items])
            for item in items:
                if not _needed(item.filename, converters):
                    continue
                target = _target(root, item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("ZIP symlinks are not allowed")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(item) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        return
    with tarfile.open(archive) as source:
        links = []
        items = source.getmembers()
        converters = _converter_root([item.name for item in items])
        for item in items:
            if not _needed(item.name, converters):
                continue
            target = _target(root, item.name)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(item) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                target.chmod(item.mode & 0o777)
            elif item.issym() or item.islnk():
                links.append((item, target))
            else:
                raise ValueError(f"Unsupported archive entry: {item.name}")
        while links:
            pending = []
            for item, target in links:
                link = (
                    (target.parent / item.linkname)
                    if item.issym()
                    else (root / item.linkname)
                )
                if not link.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"Unsafe archive link: {item.name}")
                if not link.is_file():
                    pending.append((item, target))
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Copies resolve internal chains and avoid Windows symlink privileges.
                shutil.copy2(link, target)
            if len(pending) == len(links):
                raise ValueError("Unsafe archive link: missing target or cyclic chain")
            links = pending


def fetch(platform, *, machine=None, transport=None, manifest=None):
    machine = (machine or host_platform.machine()).lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    name = {"win32": "windows"}.get(platform.name, platform.name)
    if manifest is None:
        try:
            suffix, digest = BINARIES[(name, machine)]
        except KeyError as exc:
            raise ValueError(f"No pinned llama.cpp tools for {name}/{machine}") from exc
        filename = f"llama-{RELEASE}-bin-{suffix}"
        manifest = [
            Download(
                f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{COMMIT}",
                SOURCE_SHA256,
                "source.tar.gz",
            ),
            Download(
                f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/{filename}",
                digest,
                filename,
            ),
        ]
    cache = (
        Path(platform.data_dir()) / "tools" / f"llama.cpp-{RELEASE}-{name}-{machine}"
    )
    cache.mkdir(parents=True, exist_ok=True)
    archives = [download(item, cache, transport=transport) for item in manifest]
    installed = cache / "installed"
    receipt = installed / "checksums.json"
    if receipt.is_file():
        try:
            checks = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            checks = {}
        if checks and all(
            (installed / file).is_file() and _digest(installed / file) == digest
            for file, digest in checks.items()
        ):
            return _tools(installed, name)
    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=cache))
    try:
        for archive in archives:
            extract(archive, staging)
        tools = _tools(staging, name)
        tools.quantize.chmod(tools.quantize.stat().st_mode | stat.S_IXUSR)
        checks = {
            p.relative_to(staging).as_posix(): _digest(p)
            for p in staging.rglob("*")
            if p.is_file()
        }
        (staging / "checksums.json").write_text(json.dumps(checks), encoding="utf-8")
        if installed.exists():
            shutil.rmtree(installed)
        staging.rename(installed)
        return _tools(installed, name)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _tools(root, name):
    def one(filename):
        found = list(root.rglob(filename))
        if len(found) != 1:
            raise ValueError(
                f"Official tool archive must contain exactly one {filename}"
            )
        return found[0]

    hf = one("convert_hf_to_gguf.py")
    return Toolchain(
        hf,
        one("convert_lora_to_gguf.py"),
        one("llama-quantize.exe" if name == "windows" else "llama-quantize"),
        hf.parent,
    )
