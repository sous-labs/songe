"""Read GGUF metadata using the official memory-mapped reader, never weights."""

import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GGUFInfo:
    architecture: str
    quantization: str
    parameters: int
    file_type: int
    embedding: bool = False


_CACHE: dict = {}
_LOCK = threading.Lock()


def read_gguf(path: Path) -> GGUFInfo:
    """Parsing a GGUF walks every metadata array (tokenizers hold ~10^5
    entries), so results are cached per file identity; a changed file is
    read again."""
    stat = Path(path).stat()
    key = (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = _read_gguf(path)
        return _CACHE[key]


def _read_gguf(path: Path) -> GGUFInfo:
    try:
        from gguf import GGUFReader, LlamaFileType
    except ImportError as exc:
        raise RuntimeError(
            "GGUF inspection requires the gguf dependency; reinstall songe"
        ) from exc
    try:
        reader = GGUFReader(path, mode="r")
    except ValueError as exc:
        if "GGMLQuantizationType" not in str(exc):
            raise
        # A tensor type newer than the installed gguf package (Q2_0 in
        # Unsloth's and Strata's Qwen3.8-Flash-Next files): its reader stops
        # there, the header is still plain GGUF.
        return _from_header(path, LlamaFileType)
    architecture = reader.get_field("general.architecture")
    file_type = reader.get_field("general.file_type")
    if architecture is None or file_type is None:
        raise ValueError(
            f"GGUF lacks general.architecture or general.file_type: {path}"
        )
    value = int(file_type.contents())
    parameters = sum(int(t.n_elements) for t in reader.tensors)
    name = str(architecture.contents())
    pooling = reader.get_field(f"{name}.pooling_type")
    return GGUFInfo(
        name, _file_type_name(value, LlamaFileType), parameters, value,
        bool(pooling and pooling.contents()),
    )


def _file_type_name(value, names):
    try:
        return names(value).name.removeprefix("MOSTLY_").removeprefix("ALL_")
    except ValueError:
        return f"type {value}"  # newer than the installed gguf package


# GGUF value types: struct format of each scalar; 8 = string, 9 = array.
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _from_header(path, names):
    """GGUFInfo from the header alone (metadata and tensor shapes)."""
    import math
    import struct

    with open(path, "rb") as handle:

        def read(fmt):
            return struct.unpack("<" + fmt, handle.read(struct.calcsize("<" + fmt)))

        def text():
            return handle.read(read("Q")[0]).decode("utf-8", "replace")

        def value(kind):
            if kind == 8:
                return text()
            if kind == 9:
                inner, count = read("IQ")
                if inner in _SCALARS:
                    handle.seek(struct.calcsize(_SCALARS[inner]) * count, 1)
                else:
                    for _ in range(count):
                        value(inner)
                return None
            return read(_SCALARS[kind])[0]

        if handle.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        _version, tensors, fields = read("IQQ")
        metadata = {}
        for _ in range(fields):
            key = text()
            metadata[key] = value(read("I")[0])
        parameters = 0
        for _ in range(tensors):
            text()
            dimensions = read("I")[0]
            parameters += math.prod(read(f"{dimensions}Q"))
            read("IQ")  # type, offset
    architecture, file_type = metadata.get("general.architecture"), metadata.get("general.file_type")
    if architecture is None or file_type is None:
        raise ValueError(f"GGUF lacks general.architecture or general.file_type: {path}")
    return GGUFInfo(
        architecture, _file_type_name(int(file_type), names), parameters, int(file_type),
        bool(metadata.get(f"{architecture}.pooling_type")),
    )
