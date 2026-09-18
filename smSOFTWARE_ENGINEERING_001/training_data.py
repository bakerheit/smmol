"""Validate source records and build repository-held-out byte streams."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


REQUIRED = ("id", "repository", "commit", "license", "path", "sha256", "bytes", "content")


def read_records(path):
    records = []
    ids = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = [key for key in REQUIRED if key not in item]
            if missing:
                raise ValueError("line %d is missing %s" % (line_number, ", ".join(missing)))
            if item["id"] in ids:
                raise ValueError("duplicate record id %r" % item["id"])
            raw = item["content"].encode("utf-8")
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("line %d content provenance does not match" % line_number)
            ids.add(item["id"])
            records.append(item)
    if len({item["repository"] for item in records}) < 2:
        raise ValueError("corpus needs at least two repositories")
    return records


def repository_split(records, seed=15001, validation_fraction=0.2):
    repositories = sorted({item["repository"] for item in records})
    if len(repositories) < 2:
        raise ValueError("repository-held-out validation needs at least two repositories")
    count = min(len(repositories) - 1, max(1, math.ceil(len(repositories) * validation_fraction)))
    sizes = Counter()
    for item in records:
        sizes[item["repository"]] += len(render_record(item))
    desired_each = sum(sizes.values()) * validation_fraction / count
    ranked = sorted(repositories, key=lambda name: (
        abs(sizes[name] - desired_each),
        hashlib.sha256(("%s:%s" % (seed, name)).encode("utf-8")).digest(),
    ))
    return set(ranked[count:]), set(ranked[:count])


def render_record(item):
    content = item["content"].rstrip() + "\n"
    return (
        "# Source: %s @ %s\n# Path: %s\n%s\n"
        % (item["repository"], item["commit"][:12], item["path"], content)
    ).encode("utf-8")


def read_school_items(path):
    items = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item.get("word") and item.get("sentence"):
                    items.append(item)
    if not items:
        raise ValueError("school replay corpus is empty")
    return items


def render_school(items):
    chunks = []
    for item in items:
        word = item["word"]
        sentence = item["sentence"]
        chunks.append((
            "Question: Use the word %s in a sentence.\nHelpful English reply: %s\n\n%s\n\n"
            % (word, sentence, sentence)
        ).encode("utf-8"))
    return b"".join(chunks)


def replay_bytes(source, target):
    if target <= 0:
        return b""
    repeats = math.ceil(target / len(source))
    return (source * repeats)[:target]


def build_splits(records, school_items, seed=15001, school_replay_fraction=0.1):
    if not 0 <= school_replay_fraction < 1:
        raise ValueError("school_replay_fraction must be at least zero and below one")
    train_repositories, validation_repositories = repository_split(records, seed)
    train_records = [item for item in records if item["repository"] in train_repositories]
    validation_records = [item for item in records if item["repository"] in validation_repositories]
    code_train = b"".join(render_record(item) for item in train_records)
    validation = b"".join(render_record(item) for item in validation_records)
    if not code_train or not validation:
        raise ValueError("corpus needs bytes in both splits")
    school = render_school(school_items)
    replay_target = int(len(code_train) * school_replay_fraction / max(1 - school_replay_fraction, 1e-9))
    replay = replay_bytes(school, replay_target)
    train = code_train + replay
    return train, validation, {
        "records": len(records),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_repositories": sorted(train_repositories),
        "validation_repositories": sorted(validation_repositories),
        "train_code_bytes": len(code_train),
        "school_replay_bytes": len(replay),
        "validation_code_bytes": len(validation),
        "school_replay_fraction": school_replay_fraction,
        "licenses": dict(sorted(Counter(item["license"] for item in records).items())),
    }
