"""Build assistant-only targets from canonical multi-message conversations."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read_conversations(path):
    rows = []
    seen = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("id") or not isinstance(item.get("messages"), list) or not item.get("outline"):
                raise ValueError("line %d is not a canonical conversation" % line_number)
            if item["id"] in seen:
                raise ValueError("duplicate conversation id %r" % item["id"])
            seen.add(item["id"])
            rows.append(item)
    if not rows:
        raise ValueError("conversation corpus is empty")
    return rows


def read_school(path):
    items = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item.get("word") and item.get("sentence"):
                    items.append(item)
    if not items:
        raise ValueError("school corpus is empty")
    return items


def held_out_domains(conversations, seed=16001, fraction=0.125):
    domains = sorted({item["outline"]["domain"] for item in conversations})
    if len(domains) < 2:
        raise ValueError("domain-held-out validation needs at least two domains")
    count = min(len(domains) - 1, max(1, math.ceil(len(domains) * fraction)))
    ranked = sorted(domains, key=lambda domain: hashlib.sha256(
        ("%s:%s" % (seed, domain)).encode("utf-8")
    ).digest())
    return set(ranked[:count])


def render_history(messages):
    lines = ["Conversation:"]
    for message in messages:
        label = "User" if message["role"] == "user" else "Assistant"
        lines.append("%s: %s" % (label, " ".join(str(message["content"]).split())))
    lines.append("Assistant:")
    return ("\n".join(lines) + " ").encode("utf-8")


def trim_history(messages, context, answer_reserve=128):
    kept = list(messages)
    limit = context - answer_reserve
    while len(kept) > 1 and len(render_history(kept)) >= limit:
        kept.pop(0)
    prefix = render_history(kept)
    if len(prefix) >= limit:
        latest = dict(kept[-1])
        raw = latest["content"].encode("utf-8")
        keep = max(16, limit - len(render_history([dict(latest, content="")])))
        latest["content"] = raw[-keep:].decode("utf-8", errors="ignore")
        kept = [latest]
        prefix = render_history(kept)
    return prefix, len(messages) - len(kept)


def conversation_examples(conversation, context, max_history_turns=6):
    examples = []
    messages = conversation["messages"]
    max_messages = max_history_turns * 2
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        history = messages[max(0, index - max_messages):index]
        prefix, dropped = trim_history(history, context)
        answer = (" ".join(str(message["content"]).split()) + "\n\n").encode("utf-8")
        examples.append({
            "id": "%s-assistant-%02d" % (conversation["id"], index),
            "conversation": conversation["id"],
            "domain": conversation["outline"]["domain"],
            "prefix": prefix,
            "answer": answer,
            "history_messages": len(history) - dropped,
            "dropped_messages": dropped,
        })
    return examples


def school_examples(items):
    return [{
        "id": "school-%04d" % (index + 1),
        "conversation": None,
        "domain": "school_replay",
        "prefix": ("Conversation:\nUser: Use the word %s in a sentence.\nAssistant: " % item["word"]).encode("utf-8"),
        "answer": (item["sentence"] + "\n\n").encode("utf-8"),
        "history_messages": 1,
        "dropped_messages": 0,
    } for index, item in enumerate(items)]


def replay_to_fraction(base, replay, fraction):
    if not 0 <= fraction < 1:
        raise ValueError("replay fraction must be at least zero and below one")
    target = math.ceil(len(base) * fraction / max(1 - fraction, 1e-9))
    if target == 0:
        return []
    return [replay[index % len(replay)] for index in range(target)]


def build_sets(conversations, school_items, context, seed=16001, replay_fraction=0.25, max_history_turns=6):
    validation_domains = held_out_domains(conversations, seed)
    train_conversations = [item for item in conversations if item["outline"]["domain"] not in validation_domains]
    validation_conversations = [item for item in conversations if item["outline"]["domain"] in validation_domains]
    train = [example for item in train_conversations for example in conversation_examples(item, context, max_history_turns)]
    validation = [example for item in validation_conversations for example in conversation_examples(item, context, max_history_turns)]
    school = school_examples(school_items)
    replay = replay_to_fraction(train, school, replay_fraction)
    train.extend(replay)
    if not train or not validation:
        raise ValueError("training needs examples in both splits")
    return train, validation, {
        "conversations": len(conversations),
        "train_conversations": len(train_conversations),
        "validation_conversations": len(validation_conversations),
        "validation_domains": sorted(validation_domains),
        "assistant_targets": sum(len(item["messages"]) // 2 for item in conversations),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "school_replay_examples": len(replay),
        "school_replay_fraction": replay_fraction,
        "max_history_turns": max_history_turns,
        "max_dropped_messages": max(example["dropped_messages"] for example in train + validation),
        "history_message_counts": dict(sorted(Counter(
            example["history_messages"] for example in train + validation if example["domain"] != "school_replay"
        ).items())),
    }


def encode_supervised(example, context):
    raw = example["prefix"] + example["answer"]
    raw = raw[:context + 1]
    inputs = list(raw[:-1])
    targets = list(raw[1:])
    masked = max(len(example["prefix"]) - 1, 0)
    targets[:masked] = [-100] * min(masked, len(targets))
    inputs.extend([0] * (context - len(inputs)))
    targets.extend([-100] * (context - len(targets)))
    if all(value == -100 for value in targets):
        raise ValueError("example %s has no answer targets" % example["id"])
    return inputs, targets
