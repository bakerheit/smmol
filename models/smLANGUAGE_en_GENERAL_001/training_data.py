"""Validate and render short conversations in the exact Paratroop adapter format."""

import hashlib
import json
from pathlib import Path


def prompt(example):
    lines = ["Question: " + example["user"]]
    if example.get("known"):
        lines.append("Known information: " + "; ".join(example["known"]))
    lines.append("Helpful English reply:")
    return "\n".join(lines) + " "


def render(example):
    return prompt(example) + example["assistant"] + "\n\n"


def read_examples(path):
    examples = []
    seen = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = [key for key in ("id", "category", "user", "assistant") if not item.get(key)]
            if missing:
                raise ValueError("line %d is missing %s" % (line_number, ", ".join(missing)))
            if item["id"] in seen:
                raise ValueError("duplicate id %r" % item["id"])
            if not isinstance(item.get("known", []), list) or any(not str(value).strip() for value in item.get("known", [])):
                raise ValueError("line %d has invalid known information" % line_number)
            if len(prompt(item).encode("utf-8")) >= 128:
                raise ValueError("line %d prompt does not fit the model context" % line_number)
            seen.add(item["id"])
            examples.append(item)
    if len(examples) < 800:
        raise ValueError("conversation corpus is too small: %d examples" % len(examples))
    return examples


def is_validation(example, seed):
    digest = hashlib.sha256((str(seed) + ":" + example["id"]).encode("utf-8")).digest()
    return digest[0] < 26


def build_splits(examples, seed=14001):
    train = [item for item in examples if not is_validation(item, seed)]
    validation = [item for item in examples if is_validation(item, seed)]
    if not train or not validation:
        raise ValueError("corpus needs both training and validation examples")
    train_bytes = "".join(render(item) for item in train).encode("utf-8")
    validation_bytes = "".join(render(item) for item in validation).encode("utf-8")
    return train_bytes, validation_bytes, {
        "items": len(examples),
        "train_items": len(train),
        "validation_items": len(validation),
        "train_bytes": len(train_bytes),
        "validation_bytes": len(validation_bytes),
    }


def encode_supervised(example, context):
    """Make one fixed-size causal example while masking every prompt target."""
    prefix = prompt(example).encode("utf-8")
    answer = (example["assistant"] + "\n\n").encode("utf-8")
    raw = prefix + answer
    if len(prefix) >= context:
        raise ValueError("prompt for %s does not fit the context" % example["id"])
    raw = raw[: context + 1]
    inputs = list(raw[:-1])
    targets = list(raw[1:])
    masked = max(len(prefix) - 1, 0)
    targets[:masked] = [-100] * masked
    inputs.extend([0] * (context - len(inputs)))
    targets.extend([-100] * (context - len(targets)))
    if all(value == -100 for value in targets):
        raise ValueError("example %s has no answer targets" % example["id"])
    return inputs, targets


def build_supervised_sets(examples, context, seed=14001, conversation_weight=4):
    train_items = [item for item in examples if not is_validation(item, seed)]
    validation_items = [item for item in examples if is_validation(item, seed)]
    train = []
    for item in train_items:
        encoded = encode_supervised(item, context)
        repeats = 1 if item["category"] == "school_sentence" else conversation_weight
        train.extend([encoded] * repeats)
    validation = [encode_supervised(item, context) for item in validation_items]
    return train, validation, {
        "items": len(examples),
        "train_items": len(train_items),
        "validation_items": len(validation_items),
        "weighted_train_examples": len(train),
        "conversation_weight": conversation_weight,
    }
