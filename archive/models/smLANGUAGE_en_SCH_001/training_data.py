"""Turn the accepted school curriculum into deterministic byte-level train/validation text."""

import hashlib
import json
from pathlib import Path

from curriculum import BY_ID, LEVELS, corpus_problems


LEVEL_INDEX = {level["id"]: index for index, level in enumerate(LEVELS)}


def read_items(path):
    items = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                item = json.loads(line)
                item["_line"] = line_number
                items.append(item)
    problems = corpus_problems(items)
    if problems:
        raise ValueError("curriculum is invalid: " + "; ".join(problems[:5]))
    return items


def through_stage(items, stage):
    if stage not in BY_ID:
        raise ValueError("unknown school stage %r" % stage)
    last = LEVEL_INDEX[stage]
    selected = [dict(item) for item in items if LEVEL_INDEX[item["level"]] <= last]
    if not selected:
        raise ValueError("the compiled curriculum has no accepted items through %s" % stage)
    return selected


def latest_stage(items):
    present = {item["level"] for item in items}
    return max((level["id"] for level in LEVELS if level["id"] in present), key=LEVEL_INDEX.get)


def validation_item(item, seed):
    digest = hashlib.sha256((str(seed) + ":" + item["word"]).encode("utf-8")).digest()
    return digest[0] < 26  # about ten percent, stable as later grades are added


def render_item(item):
    word = item["word"]
    sentence = item["sentence"]
    label = BY_ID[item["level"]]["label"]
    return (
        "School level: %s\nWord: %s\nSentence: %s\n\n"
        "Question: Use the word %s in a sentence.\nHelpful English reply: %s\n\n"
        "%s\n\n"
    ) % (label, word, sentence, word, sentence, sentence)


def build_splits(items, stage, seed=13001):
    selected = through_stage(items, stage)
    train_items = [item for item in selected if not validation_item(item, seed)]
    val_items = [item for item in selected if validation_item(item, seed)]
    if not train_items or not val_items:
        raise ValueError("stage needs both training and validation items")
    train = "".join(render_item(item) for item in train_items).encode("utf-8")
    validation = "".join(render_item(item) for item in val_items).encode("utf-8")
    return train, validation, {
        "stage": stage,
        "items": len(selected),
        "train_items": len(train_items),
        "validation_items": len(val_items),
        "train_bytes": len(train),
        "validation_bytes": len(validation),
    }
