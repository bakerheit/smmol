"""Tiny dependency-free safetensors reader/writer for float32 model weights.

The on-disk file follows the public safetensors format: an eight-byte header
length, a padded JSON header, then raw tensor bytes. Training state still goes
in a PyTorch checkpoint; this file is the portable inference artifact.
"""

import json
import os
from pathlib import Path
import struct

import torch


DTYPES = {"F32": torch.float32}


def save_file(tensors, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {}
    chunks = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        raw = tensor.numpy().tobytes(order="C")
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for chunk in chunks:
            handle.write(chunk)
    os.replace(temporary, path)


def load_file(path):
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError("safetensors file is shorter than its header length")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if header_length > len(raw) - 8:
        raise ValueError("safetensors header points beyond the file")
    try:
        header = json.loads(raw[8 : 8 + header_length].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("safetensors header is not valid JSON") from exc
    start = 8 + header_length
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_name = info.get("dtype")
        if dtype_name not in DTYPES:
            raise ValueError("unsupported safetensors dtype %r" % dtype_name)
        begin, end = info.get("data_offsets", (None, None))
        if not isinstance(begin, int) or not isinstance(end, int) or not 0 <= begin <= end <= len(raw) - start:
            raise ValueError("bad safetensors offsets for %s" % name)
        tensor = torch.frombuffer(bytearray(raw[start + begin : start + end]), dtype=DTYPES[dtype_name])
        shape = info.get("shape")
        try:
            tensors[name] = tensor.reshape(shape).clone()
        except (RuntimeError, TypeError) as exc:
            raise ValueError("bad safetensors shape for %s" % name) from exc
    return tensors, dict(header.get("__metadata__") or {})
