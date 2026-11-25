"""Small durable files. All mutations replace a fully fsynced sibling."""

import json
import os
import tempfile
from functools import cache
from pathlib import Path


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    if os.name == "nt":
        private_acl(path)
    return path


def private_tree(path):
    """Learned weights are private even when exported outside the data root."""
    for entry in [path, *(path.rglob("*") if path.is_dir() else [])]:
        if entry.is_symlink():
            raise ValueError("Learned artifact trees cannot contain symbolic links")
        entry.chmod(0o700 if entry.is_dir() else 0o600)
    if os.name == "nt":
        private_acl(path, recursive=True)


@cache
def windows_sid():
    import csv
    import subprocess

    output = subprocess.check_output(
        ["whoami", "/user", "/fo", "csv", "/nh"], text=True
    )
    return next(csv.reader(output.splitlines()))[1]


def private_acl(path, recursive=False):
    # Replace the DACL directly, without temporarily resetting it to a shared
    # parent ACL. Directories propagate only the current account's full access.
    for entry in [path, *(path.rglob("*") if recursive and path.is_dir() else [])]:
        set_private_dacl(entry, windows_sid())


def set_private_dacl(path, sid, apis=None):
    import ctypes
    from ctypes import wintypes

    advapi, kernel = apis or (
        ctypes.WinDLL("advapi32", use_last_error=True),
        ctypes.WinDLL("kernel32", use_last_error=True),
    )
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.ULONG),
    ]
    convert.restype = wintypes.BOOL
    get_dacl = advapi.GetSecurityDescriptorDacl
    get_dacl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    get_dacl.restype = wintypes.BOOL
    set_security = advapi.SetNamedSecurityInfoW
    set_security.argtypes = [
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    set_security.restype = wintypes.DWORD
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    flags = "OICI" if path.is_dir() else ""
    descriptor = ctypes.c_void_p()
    if not convert(f"D:P(A;{flags};FA;;;{sid})", 1, ctypes.byref(descriptor), None):
        raise OSError("Cannot construct the private Windows artifact ACL")
    try:
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
        if (
            not get_dacl(
                descriptor,
                ctypes.byref(present),
                ctypes.byref(dacl),
                ctypes.byref(defaulted),
            )
            or not present.value
            or not dacl.value
        ):
            raise OSError("Cannot read the private Windows artifact ACL")
        # SE_FILE_OBJECT=1, DACL_SECURITY_INFORMATION=4, protected DACL=0x80000000.
        result = set_security(str(path), 1, 4 | 0x80000000, None, None, dacl, None)
        if result:
            raise OSError(result, "Cannot apply the private Windows artifact ACL")
    finally:
        kernel.LocalFree(descriptor)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        if os.name != "nt":  # Windows cannot open directory handles via os.open.
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value) -> None:
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


# Top-level entries Songe creates in its data directory. Purge removes only
# these; anything else a user keeps there survives uninstall.
OWNED_ENTRIES = frozenset(
    {
        "agent-backups.json",
        "artifact-roots.json",
        "backups",
        "blobs",
        "calibration.json",
        "cloud-key",
        "config.toml",
        "context-summaries",
        "converter",
        "converter-install.log",
        "manifests",
        "missing-model",
        "models",
        "probes",
        "runtime",
        "runtime-install.log",
        "songe.log",
        "stacks.txt",
        "trainability.json",
        "versions",
    }
)


def owned_entry(name: str) -> bool:
    return (
        name in OWNED_ENTRIES
        or name.endswith(".lock")
        or name.startswith(("experience.sqlite3", "songe.log", "calibration-"))
        or name in {"service-upgrade.json", "fuse-progress.json"}
        # Staging folders a crash can leave: .export-…, .mlx-export-…, .artifact-…
        or name.startswith((".export-", ".mlx-export-", ".artifact-"))
    )


def foreign_entries(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if not owned_entry(p.name))
