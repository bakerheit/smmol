"""Validate seed or generated curriculum JSONL."""

import argparse
import json
from pathlib import Path

from curriculum import BY_ID, LEVELS, corpus_problems


HERE = Path(__file__).resolve().parent


def read_jsonl(paths):
    items = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    item = json.loads(line)
                    item["_source"] = "%s:%d" % (path, line_number)
                    items.append(item)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    paths = args.paths or [HERE / "seeds" / "preschool.jsonl"] + sorted((HERE / "data" / "levels").glob("*.jsonl"))
    items = read_jsonl(paths)
    problems = corpus_problems(items, complete=args.complete)
    counts = {level["label"]: sum(item.get("level") == level["id"] for item in items) for level in LEVELS}
    print(json.dumps({"items": len(items), "levels": {k: v for k, v in counts.items() if v}}, indent=2))
    if problems:
        for problem in problems[:100]:
            print("ERROR:", problem)
        raise SystemExit("%d curriculum problems" % len(problems))
    print("valid")


if __name__ == "__main__":
    main()
