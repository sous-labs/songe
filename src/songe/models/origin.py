"""The original weights a GGUF was made from (opt-in: songe models <model> download).

Dequantizing a GGUF holds the whole model in memory (a 27B GGUF stopped with
MemoryError on a 46 GB card's machine), and transformers reads only some
architectures from GGUF. Most GGUF files name the repository they were made
from. When that repository holds the very same weights, training starts from
it instead: shard by shard, nothing dequantized in memory.

"The very same weights" is checked, not assumed: a model card's base_model is
often the pretrained model under the instruct one. The final norm is stored
unquantized (F32) in a GGUF, so it is compared with the repository's own,
read with two small range requests.
"""

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

HUB = "https://huggingface.co"
REPO = re.compile(r"^[\w.-]+/[\w.-]+$")
KEYS = (
    "general.source.huggingface.repository",
    "general.source.repo_url",
    "general.source.url",
    "general.repo_url",
    "general.url",
)
# The files training and the GGUF LoRA converter read; never other formats.
FILES = ("*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken")


@dataclass(frozen=True)
class Origin:
    repo: str
    revision: str
    bytes: int


def _text(reader, key):
    field = reader.get_field(key)
    if field is None:
        return None
    try:
        value = field.contents()
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _repo(value):
    if not value:
        return None
    value = value.strip().removesuffix("/")
    for prefix in (HUB + "/", "http://huggingface.co/", "hf.co/", "https://hf.co/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    value = "/".join(value.split("/")[:2])
    return value if REPO.match(value) else None


def candidates(reader):
    """Repositories the GGUF says it comes from, its own source first."""
    found = [_repo(_text(reader, key)) for key in KEYS]
    count = reader.get_field("general.base_model.count")
    for i in range(int(count.contents()) if count is not None else 0):
        prefix = f"general.base_model.{i}."
        found.append(_repo(_text(reader, prefix + "repo_url")))
        organization = _text(reader, prefix + "organization")
        name = _text(reader, prefix + "name")
        if organization and name:
            found.append(_repo(f"{organization}/{name}".replace(" ", "-")))
    return list(dict.fromkeys(r for r in found if r))


def _final_norm(reader):
    from gguf import GGMLQuantizationType

    for tensor in reader.tensors:
        if tensor.name == "output_norm.weight" and tensor.tensor_type in (
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
        ):
            import numpy as np

            return np.asarray(tensor.data, dtype=np.float32).reshape(-1)
    return None


def _token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    home = Path(
        os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"
    )
    try:
        return (home / "token").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _decode(raw, dtype):
    import numpy as np

    if dtype == "BF16":
        words = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return words.view(np.float32)
    if dtype in {"F16", "F32"}:
        return np.frombuffer(raw, dtype=np.float16 if dtype == "F16" else np.float32)
    return None


def _same(ours, theirs):
    import numpy as np

    if theirs is None or theirs.shape != ours.shape:
        return False
    theirs = theirs.astype(np.float32)
    # Some converters store a norm as 1 + w (Qwen3.5 and later, Gemma).
    return any(
        np.allclose(ours, theirs + shift, rtol=0, atol=1e-5) for shift in (0.0, 1.0)
    )


def _verify(client, repo, norm):
    response = client.get(f"{HUB}/api/models/{repo}", params={"blobs": "true"})
    if response.status_code != 200:
        return None
    info = response.json()
    revision = info.get("sha")
    shards = {
        s["rfilename"]: int(s.get("size") or 0)
        for s in info.get("siblings") or []
        if s.get("rfilename", "").endswith(".safetensors")
        and "/" not in s["rfilename"]
    }
    if not revision or not shards:
        return None
    base = f"{HUB}/{repo}/resolve/{revision}/"
    if "model.safetensors.index.json" in {
        s.get("rfilename") for s in info.get("siblings") or []
    }:
        index = client.get(base + "model.safetensors.index.json")
        index.raise_for_status()
        names = index.json().get("weight_map", {})
    else:
        names = {
            tensor: shard
            for shard in shards
            for tensor in _header(client, base, shard)[0]
        }
    final = [
        name
        for name in names
        if re.search(r"(^|\.)norm\.weight$", name)
        and not re.search(r"\.(layers|blocks|h)\.\d+\.", name)
        and "vision" not in name
        and "visual" not in name
    ]
    for name in final:
        header, start = _header(client, base, names[name])
        entry = header.get(name) or {}
        offsets = entry.get("data_offsets")
        if not offsets:
            continue
        raw = client.get(
            base + names[name],
            headers={
                "Range": f"bytes={start + offsets[0]}-{start + offsets[1] - 1}"
            },
        )
        raw.raise_for_status()
        if _same(norm, _decode(raw.content, entry.get("dtype"))):
            return Origin(repo, revision, sum(shards.values()))
    return None


def _header(client, base, shard):
    head = client.get(base + shard, headers={"Range": "bytes=0-7"})
    head.raise_for_status()
    (length,) = struct.unpack("<Q", head.content[:8])
    body = client.get(base + shard, headers={"Range": f"bytes=8-{7 + length}"})
    body.raise_for_status()
    header = json.loads(body.content)
    header.pop("__metadata__", None)
    return header, 8 + length


def original_weights(path, *, transport=None):
    """The repository holding the weights this GGUF was quantized from, or
    None: no repository named, offline, gated without a token, or other
    weights (a base model under an instruct GGUF)."""
    if os.environ.get("HF_HUB_OFFLINE") not in (None, "", "0"):
        return None
    from gguf import GGUFReader

    reader = GGUFReader(path, mode="r")
    repos = candidates(reader)
    norm = _final_norm(reader) if repos else None
    if norm is None:
        return None
    import httpx

    token = _token()
    with httpx.Client(
        transport=transport,
        timeout=30,
        follow_redirects=True,
        headers={"Authorization": f"Bearer {token}"} if token else {},
    ) as client:
        for repo in repos:
            try:
                found = _verify(client, repo, norm)
            except (httpx.HTTPError, ValueError, KeyError, TypeError, struct.error):
                continue
            if found:
                return found
    return None


def download(origin):
    """The original weights in the Hugging Face cache (an interrupted
    download resumes; weights already there are not fetched again)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            origin.repo, revision=origin.revision, allow_patterns=list(FILES)
        )
    )


def present_bytes(origin):
    """Bytes of the weights' shards already in the Hugging Face cache."""
    root = cache_dir()
    if root is None:
        return 0
    snapshot = (
        root / ("models--" + origin.repo.replace("/", "--")) / "snapshots" / origin.revision
    )
    try:
        return sum(
            p.stat().st_size for p in snapshot.glob("*.safetensors") if p.exists()
        )
    except OSError:
        return 0


def cache_dir():
    """Where download() puts the weights (for the disk check)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except ImportError:
        return None


def named(path):
    """The first repository the GGUF names, read from the file alone (no
    network): what the panel offers to train from."""
    from gguf import GGUFReader

    found = candidates(GGUFReader(path, mode="r"))
    return found[0] if found else None
