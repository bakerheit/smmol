"""Pick the held-out words, with a fixed seed, stratified by frequency band and part of speech.

Phase 1 of docs/engineering/plans/smCLM_02.md. About 60 vocabulary words, spread across the
frequency bands and the four parts of speech so the held-out score isn't really a score of one
band, plus smCLM_01's 23 held-out nouns wherever they're in the vocabulary, so v1's world stays a
valid test.

**This script writes no tags.** It writes `data/held_out.json`: which words, which band, why each
one was picked, and an empty `ideas` list for every row. A person fills those in, in
`test_words.json`. The teacher's tags for these words are dropped from `data/tags.jsonl` by
`tag_words.py` and never become truth — a held-out word scored against teacher tags would be
scoring the teacher against itself, which is the one thing this whole test set exists to avoid.

v1's 23 are the exception the plan already allows: smCLM_01's `concepts.json` tags were written by
a person, so `--with-v1-tags` may copy them in as a starting point. They're marked `source: "v1"`,
they only ever use ideas that are in `ideas.json`, and a person still has to extend them with any
new ideas that apply before `test_words.json` is complete.

    python3 held_out.py                          # the pick -> data/held_out.json
    python3 held_out.py --skeleton               # the rows to paste into test_words.json, untagged
    python3 held_out.py --skeleton --with-v1-tags  # same, with v1's human tags prefilled
"""

import argparse
import json
from pathlib import Path
import random

from teacher import (DATA, HERE, POS, load_inventory, load_vocabulary, read_json, read_jsonl,
                     write_json)


SEED = 17001
TARGET = 60

# Frequency bands, widest first. `vocabulary.py` counts usages in the conversation corpus; a word
# with no count sits in the rarest band rather than being dropped.
BANDS = (
    ("high", 100, None),
    ("mid", 30, 100),
    ("low", 10, 30),
    ("rare", 0, 10),
)

TAGS = DATA / "tags.jsonl"
TAG_LOG = DATA / "logs" / "tag_calls.jsonl"

# Vocabulary words that may not be held out, and why. Checked by eye after the first pick.
EXCLUDE = {
    "theo": "a first name; vocabulary.py's name rule missed it (capitalised in too few uses)",
}
OUT = DATA / "held_out.json"
V1 = HERE.parent / "smCLM_01" / "concepts.json"


def band_of(count):
    for name, low, high in BANDS:
        if count >= low and (high is None or count < high):
            return name
    return BANDS[-1][0]


def strata(vocabulary, pos_of):
    """{(band, pos): [word, ...]} with every list sorted, so the seed is the only randomness."""
    cells = {}
    for row in vocabulary:
        key = (band_of(int(row.get("count") or 0)), pos_of.get(row["word"], "other"))
        cells.setdefault(key, []).append(row["word"])
    return {key: sorted(words) for key, words in cells.items()}


def share_out(cells, target):
    """How many to take from each cell: proportional to its size, largest remainder, deterministic.

    Every non-empty cell gets at least one, so a small cell (adjectives in the top band, say) isn't
    rounded out of the test set altogether.
    """
    order = sorted(cells, key=lambda key: (BANDS_ORDER[key[0]], POS.index(key[1])))
    order = [key for key in order if cells[key]]
    if not order:
        return {}
    total = sum(len(cells[key]) for key in order)
    exact = {key: target * len(cells[key]) / total for key in order}
    picked = {key: min(len(cells[key]), max(1, int(exact[key]))) for key in order}
    # Hand out or take back what rounding left over, biggest remainder first.
    while sum(picked.values()) != target:
        short = target - sum(picked.values())
        step = 1 if short > 0 else -1
        movable = [key for key in order
                   if (step > 0 and picked[key] < len(cells[key])) or (step < 0 and picked[key] > 1)]
        if not movable:
            break
        key = max(movable, key=lambda k: ((exact[k] - picked[k]) * step, -BANDS_ORDER[k[0]]))
        picked[key] += step
    return picked


BANDS_ORDER = {name: i for i, (name, _low, _high) in enumerate(BANDS)}


def pos_from_log(path):
    """Each word's part of speech: the one most passes gave, first pass breaking a tie."""
    votes = {}
    for row in read_jsonl(path):
        if row.get("ok") and row.get("tags"):
            for word, tag in row["tags"].items():
                votes.setdefault(word, []).append((row.get("pass", 0), tag.get("pos") or "other"))
    out = {}
    for word, got in votes.items():
        got.sort()
        names = [pos for _, pos in got]
        out[word] = max(names, key=lambda pos: (names.count(pos), -names.index(pos)))
    return out


def pick(vocabulary, pos_of, target=TARGET, seed=SEED):
    cells = strata(vocabulary, pos_of)
    quota = share_out(cells, target)
    chosen = []
    for key in sorted(quota, key=lambda k: (BANDS_ORDER[k[0]], POS.index(k[1]))):
        band, pos = key
        # One Random per cell, seeded from the cell's name, so adding a cell can't reshuffle the
        # cells picked before it. A rerun on the same vocabulary gives the same words.
        rng = random.Random("%d:%s:%s" % (seed, band, pos))
        for word in sorted(rng.sample(cells[key], quota[key])):
            chosen.append({"word": word, "band": band, "pos": pos, "source": "stratified"})
    return chosen


def v1_held_out(vocabulary_words, inventory, path=V1):
    """smCLM_01's 23 held-out nouns that are in this vocabulary, with their human tags.

    Tags are filtered to ideas that are in ideas.json. They all should be — ideas.json is seeded
    with v1's 51 verbatim — but the filter is cheap and it says what the rule is.
    """
    if not Path(path).exists():
        return []
    world = read_json(path)
    rows = []
    for word in sorted(world.get("held_out") or []):
        if word not in vocabulary_words:
            continue
        tags = [idea for idea in world.get("words", {}).get(word, []) if idea in inventory]
        rows.append({"word": word, "band": None, "pos": "noun", "source": "v1", "v1_ideas": tags})
    return rows


def combine(stratified, v1):
    """v1's nouns first, then the stratified pick minus anything v1 already covers."""
    seen = {row["word"] for row in v1}
    rows = list(v1)
    for row in stratified:
        if row["word"] not in seen:
            seen.add(row["word"])
            rows.append(row)
    return rows


def skeleton(rows, with_v1_tags):
    """The rows a person pastes into test_words.json. `ideas` is empty unless v1 already has them."""
    out = []
    for row in rows:
        ideas = row.get("v1_ideas", []) if (with_v1_tags and row["source"] == "v1") else []
        out.append({
            "word": row["word"],
            "pos": row["pos"],
            "ideas": list(ideas),
            "band": row["band"],
            "source": "v1" if (ideas and row["source"] == "v1") else "human",
            "note": "",
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--tags", default=str(TAGS))
    parser.add_argument("--tag-log", default=str(TAG_LOG))
    parser.add_argument("--ideas", default=None)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skeleton", action="store_true",
                        help="print the test_words.json rows for a person to fill in")
    parser.add_argument("--with-v1-tags", action="store_true",
                        help="prefill the v1 nouns with their smCLM_01 human tags")
    parser.add_argument("--allow-partial-inventory", action="store_true")
    args = parser.parse_args()

    inventory = load_inventory(args.ideas, allow_partial=args.allow_partial_inventory)
    vocabulary = load_vocabulary(args.vocabulary)
    words = {row["word"] for row in vocabulary}
    # Part of speech comes from the teacher's tags where they exist, falling back to whatever
    # vocabulary.py put on the row ("other" until something fills it). It steers the stratification
    # only; it is never written into test_words.json as truth.
    #
    # From the raw tagging log, not data/tags.jsonl: that file drops the held-out words once this
    # script has picked them, so reading it made a rerun see them as "other" and pick a different
    # set (found 2026-09-19). The log never loses a row.
    pos_of = {row["word"]: row.get("pos") or "other" for row in vocabulary}
    pos_of.update(pos_from_log(args.tag_log))

    pickable = [row for row in vocabulary if row["word"] not in EXCLUDE]
    stratified = pick(pickable, pos_of, args.target, args.seed)
    v1 = v1_held_out(words, inventory)
    rows = combine(stratified, v1)

    if args.skeleton:
        print(json.dumps(skeleton(rows, args.with_v1_tags), indent=2))
        return

    counts = {}
    for row in rows:
        key = "%s/%s" % (row["band"] or "v1", row["pos"])
        counts[key] = counts.get(key, 0) + 1
    write_json(args.out, {
        "what": "The held-out words of smCLM_02. Picked by held_out.py; tagged by a person.",
        "human_gate": ("Every word here needs its ideas written by a person in test_words.json. "
                       "This file carries no tags and the teacher's tags for these words are "
                       "dropped from data/tags.jsonl by tag_words.py."),
        "seed": args.seed,
        "target": args.target,
        "bands": [{"name": n, "from": lo, "to": hi} for n, lo, hi in BANDS],
        "vocabulary_words": len(words),
        "pos_known_for": len(pos_of),
        "picked": len(rows),
        "from_v1": len(v1),
        "by_band_and_pos": dict(sorted(counts.items())),
        "words": rows,
    })
    print("%d held-out words (%d of them v1's) -> %s" % (len(rows), len(v1), args.out))
    for key in sorted(counts):
        print("  %-14s %d" % (key, counts[key]))
    if not pos_of:
        print("note: data/tags.jsonl is empty, so every word stratified as 'other'. "
              "Rerun after tag_words.py for a real pos spread.")
    print("STILL A PERSON'S JOB: write each word's ideas into test_words.json "
          "(python3 held_out.py --skeleton gives you the rows).")


if __name__ == "__main__":
    main()
