"""Download a bounded, pinned TinyStories corpus and create disjoint splits."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


HERE = Path(__file__).resolve().parent
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
BASE_URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}"
TRAIN_URL = f"{BASE_URL}/TinyStories-train.txt"
VALID_URL = f"{BASE_URL}/TinyStories-valid.txt"
SEPARATOR = b"<|endoftext|>"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_prefix(url, path, byte_limit):
    """Stream at most byte_limit bytes, even if a server ignores Range."""
    if path.exists() and path.stat().st_size == byte_limit:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{byte_limit - 1}",
            "User-Agent": "smLANGUAGE_EN_001/1.0",
        },
    )
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            while written < byte_limit:
                chunk = response.read(min(1024 * 1024, byte_limit - written))
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        if written != byte_limit:
            raise RuntimeError(f"wanted {byte_limit} bytes from {url}, received {written}")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def complete_stories(raw):
    """Drop a partial final story and turn dataset markers into plain paragraph breaks."""
    end = raw.rfind(SEPARATOR)
    if end < 0:
        raise ValueError("downloaded prefix contains no complete stories")
    clean = raw[:end].replace(SEPARATOR, b"\n\n").strip() + b"\n"
    clean.decode("utf-8")
    return clean


def split_heldout(raw):
    """Split the official validation stories into validation and untouched test text."""
    stories = [story.strip() for story in raw.split(SEPARATOR)[:-1] if story.strip()]
    if len(stories) < 2:
        raise ValueError("need at least two complete held-out stories")
    midpoint = len(stories) // 2
    join = lambda items: b"\n\n".join(items) + b"\n"
    return join(stories[:midpoint]), join(stories[midpoint:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--train-mib", type=int, default=128)
    parser.add_argument("--heldout-mib", type=int, default=8)
    args = parser.parse_args()
    if args.train_mib < 1 or args.heldout_mib < 1:
        parser.error("corpus sizes must be positive")

    raw_dir = args.data_dir / "raw"
    raw_train = raw_dir / "TinyStories-train.prefix.txt"
    raw_valid = raw_dir / "TinyStories-valid.prefix.txt"
    train_bytes = args.train_mib * 1024 * 1024
    heldout_bytes = args.heldout_mib * 1024 * 1024
    print(f"downloading {args.train_mib} MiB train prefix", flush=True)
    download_prefix(TRAIN_URL, raw_train, train_bytes)
    print(f"downloading {args.heldout_mib} MiB held-out prefix", flush=True)
    download_prefix(VALID_URL, raw_valid, heldout_bytes)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.txt"
    val_path = args.data_dir / "val.txt"
    test_path = args.data_dir / "test.txt"
    train_path.write_bytes(complete_stories(raw_train.read_bytes()))
    val, test = split_heldout(raw_valid.read_bytes())
    val_path.write_bytes(val)
    test_path.write_bytes(test)

    manifest = {
        "dataset": "roneneldan/TinyStories",
        "revision": REVISION,
        "license": "CDLA-Sharing-1.0",
        "paper": "https://arxiv.org/abs/2305.07759",
        "sources": {"train": TRAIN_URL, "validation": VALID_URL},
        "splits": {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in (("train", train_path), ("val", val_path), ("test", test_path))
        },
        "note": "Validation and test are disjoint story halves of the upstream validation split.",
    }
    (args.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, details in manifest["splits"].items():
        print(f"{name:5s}: {details['bytes'] / 1e6:7.2f} MB  {details['sha256'][:12]}")


if __name__ == "__main__":
    main()
