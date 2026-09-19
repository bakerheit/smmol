"""Validate the corpus, provenance manifest, and deterministic split."""

import hashlib
import json
from pathlib import Path

from training_data import build_splits, read_records, read_school_items


HERE = Path(__file__).resolve().parent


def main():
    corpus = HERE / "data" / "raw" / "records.jsonl"
    manifest = json.loads((corpus.parent / "manifest.json").read_text())
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("raw corpus hash does not match its manifest")
    config = json.loads((HERE / "model_config.json").read_text())
    records = read_records(corpus)
    school = read_school_items(HERE.parent / "smLANGUAGE_en_SCH_001" / "data" / "curriculum.jsonl")
    _, _, split = build_splits(
        records, school, config["training"]["seed"], config["training"]["school_replay_fraction"],
    )
    print(json.dumps({"manifest": manifest, "split": split}, indent=2, sort_keys=True))
    print("valid")


if __name__ == "__main__":
    main()
