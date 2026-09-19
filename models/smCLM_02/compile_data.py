"""smCLM_02 phase 2: every training sentence in one file, a manifest, and the coverage report.

Builds `data/sentences.jsonl` (`text`, `source`, `tokens`) from three sources, in this order
(plan section "Phase 2"):

  1. `conversations` - `smCONVERSATION_001/data/accepted/conversations.jsonl`, user and assistant
     turns, split into sentences and kept when 4 to 30 words long;
  2. `curriculum`    - the 600 sentences in `smLANGUAGE_en_SCH_001/data/curriculum.jsonl`;
  3. `teacher`       - what `generate_sentences.py` kept, read from its call log.

Two rules apply to every source, not just the teacher's, because the conversation corpus was
written by the same Ministral the tests were written to be unlike:

  * an exact duplicate (after `flat()`) is kept once, first source wins;
  * a sentence that is, or nearly copies, a `test_recall.json` text is dropped. "Nearly" is the
    leak test's own rule: 80% of the test text's 3-word shingles. `tests/test_leaks.py` checks the
    result independently; `tests/test_compile_data.py` checks the two rules are the same rule.

`generate_sentences.py` imports `base_sentences()`, `LeakGuard` and `NearCopies` from here, so the
generator and the compiler can't disagree about what counts as a copy.

`data/manifest.json` records the SHA-256 and row count of every data file, the sentence counts by
source, and every word still under TARGET usages. The plan's phase 2 gate needs that list empty.

    python3 compile_data.py              build data/sentences.jsonl + data/manifest.json, run the leak test
    python3 compile_data.py --report     the coverage numbers only, write nothing
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import re
import sys
import unittest

import vocabulary as V

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "sentences.jsonl")
MANIFEST = os.path.join(DATA, "manifest.json")
TEACHER_LOG = os.path.join(DATA, "logs", "sentence_calls.jsonl")
RECALL = os.path.join(HERE, "test_recall.json")

TARGET = 30  # usages per vocabulary word, the phase 2 gate
CORPUS_MIN, CORPUS_MAX = 4, 30  # words per kept conversation sentence
SHINGLE_LIMIT = 0.8  # tests/test_leaks.py's rule, and the plan's near-duplicate rule

# The leak test's tokeniser (itself the harness's), so "the same text" means one thing everywhere.
FLAT_WORD = re.compile(r"[a-z0-9]+")
MARKDOWN = re.compile(r"\*\*|__|`|^#+\s*|^\s*(?:[-*•>]|\d+[.)])\s+", re.M)
SENTENCE_END = re.compile(r"(?<=[.!?])[\"'”’)]*\s+(?=[\"'“‘(]?[A-Z0-9])")


def flat(text):
    """A text reduced to what we compare: lowercase words and digits, one space between."""
    return " ".join(FLAT_WORD.findall(str(text).lower()))


def shingles(text, n=3):
    parts = flat(text).split()
    return {tuple(parts[i:i + n]) for i in range(max(1, len(parts) - n + 1))}


# ----------------------------------------------------------------------------- the guards


class LeakGuard:
    """Says whether a sentence is, or nearly copies, a hand-written test text."""

    def __init__(self, path=RECALL):
        with open(path) as f:
            recall = json.load(f)
        texts = []
        for case in recall["cases"]:
            texts.append(case["query"])
            texts.extend(p["text"] for p in case["pool"])
        self.flats = {flat(t) for t in texts}
        self.wanted = [s for s in (shingles(t) for t in texts) if s]
        self.index = defaultdict(list)
        for number, want in enumerate(self.wanted):
            for gram in want:
                self.index[gram].append(number)

    def leaks(self, text):
        if flat(text) in self.flats:
            return True
        have = shingles(text)
        hits = Counter(n for gram in have for n in self.index.get(gram, ()))
        return any(count / len(self.wanted[n]) >= SHINGLE_LIMIT for n, count in hits.items())


class NearCopies:
    """The kept sentences, and whether a new one is an exact or near copy of any of them.

    Near means 80% of the *new* sentence's 3-word shingles are in one kept sentence. Exact, not
    sampled: a kept sentence at 80% must hold at least one of the new sentence's rarest
    `len - ceil(0.8 * len) + 1` shingles (pigeonhole), so only those postings are searched.
    """

    def __init__(self):
        self.flats = set()
        self.grams = []
        self.index = defaultdict(list)

    def add(self, text):
        self.flats.add(flat(text))
        have = shingles(text)
        number = len(self.grams)
        self.grams.append(have)
        for gram in have:
            self.index[gram].append(number)

    def exact(self, text):
        return flat(text) in self.flats

    def near(self, text):
        have = shingles(text)
        if not have:
            return False
        need = math.ceil(SHINGLE_LIMIT * len(have))
        rarest = sorted(have, key=lambda gram: len(self.index.get(gram, ())))[:len(have) - need + 1]
        for number in {n for gram in rarest for n in self.index.get(gram, ())}:
            if len(have & self.grams[number]) >= need:
                return True
        return False


# ----------------------------------------------------------------------------- the sources


def split_sentences(message):
    """A chat message as plain sentences: markdown marks and list bullets gone, one line at a time."""
    for line in str(message).splitlines():
        line = MARKDOWN.sub("", line).strip()
        for sentence in SENTENCE_END.split(line):
            sentence = sentence.strip()
            if sentence:
                yield sentence


def conversation_sentences(path=V.CONVERSATIONS):
    for text in V.conversation_texts(path):
        for sentence in split_sentences(text):
            if CORPUS_MIN <= len(V.tokens(sentence)) <= CORPUS_MAX:
                yield sentence


def curriculum_sentences(path=V.CURRICULUM):
    for row in V.read_jsonl(path):
        if row.get("sentence"):
            yield row["sentence"].strip()


def teacher_sentences(path=TEACHER_LOG):
    """What generate_sentences.py kept, in the order it kept them."""
    if not os.path.exists(path):
        return
    for row in V.read_jsonl(path):
        if row.get("result"):
            for text in row.get("kept") or ():
                yield text


def base_sentences(guard=None):
    """Sources 1 and 2 after both rules, as [(text, source)]. The generator counts usages from these."""
    guard = guard or LeakGuard()
    seen, out = set(), []
    for source, texts in (("conversations", conversation_sentences()),
                          ("curriculum", curriculum_sentences())):
        for text in texts:
            key = flat(text)
            if key in seen or guard.leaks(text):
                continue
            seen.add(key)
            out.append((text, source))
    return out


def usages(texts, known):
    """How many sentences use each vocabulary word, any folded form, once per sentence."""
    counts = Counter()
    for text in texts:
        counts.update({w for w in V.content_tokens(text, known) if w in known})
    return counts


# ----------------------------------------------------------------------------- compile


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path):
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def compile_rows():
    """Every kept sentence with its source, plus what was dropped and why."""
    guard = LeakGuard()
    rows = [{"text": t, "source": s} for t, s in base_sentences(guard)]
    seen = {flat(r["text"]) for r in rows}
    dropped = Counter()
    for text in teacher_sentences():
        key = flat(text)
        if key in seen:
            dropped["teacher, duplicate of an earlier source"] += 1
        elif guard.leaks(text):
            dropped["teacher, test text"] += 1
        else:
            seen.add(key)
            rows.append({"text": text, "source": "teacher"})
    for row in rows:
        row["tokens"] = V.tokens(row["text"])
    return rows, dropped


def coverage(rows, vocabulary):
    known = {row["word"] for row in vocabulary}
    counts = usages((r["text"] for r in rows), known)
    short = sorted(((w, counts[w]) for w in known if counts[w] < TARGET), key=lambda x: (x[1], x[0]))
    return counts, short


def manifest(rows, short, dropped):
    files = {}
    for root, _dirs, names in os.walk(DATA):
        for name in sorted(names):
            path = os.path.join(root, name)
            if name.startswith(".") or name == "manifest.json":
                continue
            rel = os.path.relpath(path, DATA)
            entry = {"sha256": sha256(path), "bytes": os.path.getsize(path)}
            if name.endswith(".jsonl"):
                entry["rows"] = line_count(path)
            files[rel] = entry
    return {
        "name": "smCLM_02 training data",
        "target_usages": TARGET,
        "sentences": len(rows),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "dropped": dict(dropped),
        "words_under_target": len(short),
        "under_target": [{"word": w, "usages": n} for w, n in short],
        "files": dict(sorted(files.items())),
    }


def write_jsonl(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def leak_test():
    here = os.path.join(HERE, "tests")
    suite = unittest.defaultTestLoader.discover(here, pattern="test_leaks.py", top_level_dir=here)
    return unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="print coverage only; write nothing")
    args = parser.parse_args(argv)

    vocabulary = list(V.read_jsonl(V.OUT))
    rows, dropped = compile_rows()
    counts, short = coverage(rows, vocabulary)
    by_source = Counter(r["source"] for r in rows)
    print("%d sentences: %s" % (len(rows), ", ".join("%s %d" % kv for kv in sorted(by_source.items()))))
    for reason, n in sorted(dropped.items()):
        print("  dropped %d: %s" % (n, reason))
    print("%d of %d words under %d usages%s" % (len(short), len(vocabulary), TARGET,
                                                 ("; lowest: " + ", ".join("%s %d" % x for x in short[:8]))
                                                 if short else ""))
    if args.report:
        return 0

    write_jsonl(rows, OUT)
    with open(MANIFEST, "w") as f:
        json.dump(manifest(rows, short, dropped), f, indent=2)
        f.write("\n")
    print("wrote %s and %s" % (OUT, MANIFEST))
    ok = leak_test()
    if short:
        print("PHASE 2 GATE NOT MET: %d words under %d usages (listed in the manifest)" % (len(short), TARGET))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
