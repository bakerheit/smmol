"""Compile validated level files into one curriculum plus a count/hash manifest."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from curriculum import LEVELS, corpus_problems
from generate_data import all_existing


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    items = all_existing(args.data_dir)
    problems = corpus_problems(items, complete=not args.allow_partial)
    if problems:
        raise SystemExit("\n".join(problems[:100]))
    output = args.data_dir / "curriculum.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    counts = Counter(item["level"] for item in items)
    manifest = {
        "name": "smLANGUAGE_en_SCH_001 school curriculum",
        "items": len(items),
        "complete": len(items) == sum(level["count"] for level in LEVELS),
        "sha256": digest,
        "levels": {level["id"]: counts[level["id"]] for level in LEVELS},
    }
    (args.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
