"""Validate the compiled conversational corpus and print its shape."""

from collections import Counter
import json
from pathlib import Path

from training_data import read_examples


HERE = Path(__file__).resolve().parent


def main():
    examples = read_examples(HERE / "data" / "conversations.jsonl")
    counts = Counter(item["category"] for item in examples)
    print(json.dumps({"items": len(examples), "categories": dict(sorted(counts.items()))}, indent=2))
    print("valid")


if __name__ == "__main__":
    main()

