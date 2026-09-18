"""Ask the teacher what general ideas the vocabulary words carry, and tally the answers.

Phase 1 of docs/engineering/plans/smCLM_02.md. Batches of 20 vocabulary words, an *open* list (no
enum: the point is to hear ideas the inventory doesn't have yet), one call per batch, every raw
answer appended to `data/logs/idea_proposals.jsonl` before the next request goes out.

**This script never writes `ideas.json`.** It writes a tally to `data/idea_proposals.json` and
stops. A person picks the roughly 100 new ideas from that tally and writes each one's one-line
meaning. That is the human gate the plan puts here, and there is no flag that skips it.

Resumable: a batch already in the log is not asked again, so a run killed halfway costs nothing.
Deterministic: the vocabulary is sorted, the batches are a fixed cut of that order, and the tally is
recomputed from the whole log every run, so the same log always gives the same
`data/idea_proposals.json`. The teacher's sampling is its own business; nothing downstream depends
on it being repeatable.

    python3 propose_ideas.py --dry-run          # the plan and one example request, nothing sent
    python3 propose_ideas.py                    # ask, resuming where the log left off
    python3 propose_ideas.py --tally-only       # rebuild the tally from the log, no calls at all
"""

import argparse
from collections import Counter, defaultdict
import json
import re

from teacher import (DATA, IDEAS, DryRun, add_teacher_arguments, append_jsonl, batches,
                     load_inventory, load_vocabulary, read_jsonl, teacher_from, write_json)


KIND = "idea_proposals"
PROMPT_VERSION = "reuse_v2"
SEED = 15001
BATCH = 20
TEMPERATURE = 0.7
MAX_TOKENS = 1400
MAX_IDEAS_PER_WORD = 3
WORDS_SHOWN = 12  # example words kept per idea in the tally file
IDEA_NAME = re.compile(r"^[a-z][a-z_]{1,23}$")
ROLE_IDEAS = {"action", "event", "feeling", "thing"}

LOG = DATA / "logs" / "idea_proposals.jsonl"
TALLY = DATA / "idea_proposals.json"

SYSTEM = ("You are helping build a small inventory of general ideas that English words carry. "
          "Answer with JSON only, in the schema you are given.")

PROMPT = """For each word below, name the broad, reusable ideas it carries.

An idea is a broad, reusable category that many different words share, like health, money, speed,
danger, family, food, person, place, action or feeling. It is never a definition of the word and
never a synonym of it.

The approved inventory is below. Prefer these exact names whenever they plausibly fit:
{inventory}

Rules:
- Give 1 to {max_ideas} ideas per word, most important first.
- Usually use approved ideas. Invent a new idea only when the word's core ordinary meaning is not
  represented above.
- A new idea must be broad enough to fit at least 8 different words in a 2,500-word vocabulary.
  If it mainly fits this word, one word family, or one narrow object, do not invent it.
- Use one shared name for related meanings. Do not create near-synonyms or grammatical variants.
- Each idea is one lowercase English word, or two joined by an underscore. No spaces, no phrases.
- Reuse the same idea name across words wherever it fits. That is the whole point.
- Return every word below, exactly once, spelled exactly as given, in the same order.

Words:
{words}
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["words"],
    "properties": {
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["word", "ideas"],
                "properties": {
                    "word": {"type": "string"},
                    "ideas": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_IDEAS_PER_WORD,
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
}


def prompt_for(words, inventory):
    approved = "\n".join("- %s: %s" % item for item in inventory.items())
    return PROMPT.format(max_ideas=MAX_IDEAS_PER_WORD, inventory=approved,
                         words="\n".join(words))


def clean_idea(idea):
    """An idea name as we'll store it, or None if it isn't one.

    The teacher is asked for one lowercase word; it sometimes returns "Money", "food items" or
    "day-to-day". Fold the easy cases, drop the rest rather than inventing a name for it.
    """
    if not isinstance(idea, str):
        return None
    name = re.sub(r"[\s\-]+", "_", idea.strip().lower())
    name = re.sub(r"[^a-z_]", "", name).strip("_")
    return name if IDEA_NAME.match(name) else None


def check(answer, words, self_ideas=ROLE_IDEAS):
    """The teacher's answer for this batch as {word: [idea, ...]}, or a reason it's no good.

    The word set has to come back exactly: same words, each once, nothing added, nothing missing.
    A batch that fails is retried by the caller rather than half-kept.
    """
    if not isinstance(answer, dict) or not isinstance(answer.get("words"), list):
        return None, "answer has no words array"
    got = [row.get("word") for row in answer["words"] if isinstance(row, dict)]
    if len(got) != len(answer["words"]):
        return None, "a row isn't an object"
    if got != list(words):
        missing = sorted(set(words) - set(got))
        extra = sorted(set(got) - set(words))
        return None, ("words came back wrong: %d sent, %d back, missing %s, extra %s"
                      % (len(words), len(got), missing[:5] or "-", extra[:5] or "-"))
    proposals = {}
    for row in answer["words"]:
        ideas, seen = [], set()
        for raw in row.get("ideas") or []:
            name = clean_idea(raw)
            # Usually `apple -> apple` is just a definition pretending to be an idea. Broad role
            # names are the exception: `action -> action` is exactly the candidate we need.
            if (name and name not in seen
                    and (name != row["word"] or name in self_ideas)):
                seen.add(name)
                ideas.append(name)
        if not ideas:
            return None, "%s came back with no usable idea name" % row["word"]
        proposals[row["word"]] = ideas[:MAX_IDEAS_PER_WORD]
    return proposals, None


def done_batches(log_path):
    """{batch index: {word: [idea, ...]}} for the batches already answered."""
    done = {}
    for row in read_jsonl(log_path):
        if row.get("ok") and row.get("batch") is not None:
            done[int(row["batch"])] = row["proposals"]
    return done


def scheduled_batches(todo, limit):
    """The logical batches this invocation may ask for; rejected answers still count."""
    return todo[:limit] if limit else todo


def tally(done, vocabulary_words):
    """The tally a person picks from: every proposed idea, how often, and which words asked for it.

    Sorted by how many words carry the idea, then by name, so the file is byte-identical for the
    same log. Words are capped in the file at WORDS_SHOWN; the count is the real one.
    """
    words_for = defaultdict(list)
    per_word = Counter()
    for batch in sorted(done):
        for word in sorted(done[batch]):
            for idea in done[batch][word]:
                words_for[idea].append(word)
            per_word[word] = len(done[batch][word])
    asked = sum(len(done[b]) for b in done)
    rows = []
    for idea in sorted(words_for, key=lambda name: (-len(set(words_for[name])), name)):
        words = sorted(set(words_for[idea]))
        rows.append({
            "idea": idea,
            "words": len(words),
            "share_of_tagged": round(len(words) / asked, 4) if asked else 0.0,
            "examples": words[:WORDS_SHOWN],
        })
    return {
        "what": "Open-list idea proposals from the teacher, tallied. Candidates, not an inventory.",
        "human_gate": ("A person picks the new ideas from this tally and writes each one's one-line "
                       "meaning into ideas.json. No script may do it. ideas.json currently holds "
                       "smCLM_01's 51 approved ideas and nothing else."),
        "seed": SEED,
        "prompt_version": PROMPT_VERSION,
        "batch_size": BATCH,
        "vocabulary_words": len(vocabulary_words),
        "words_tagged": asked,
        "batches_done": len(done),
        "distinct_ideas": len(rows),
        "proposals": rows,
    }


def main():
    parser = add_teacher_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--ideas", default=str(IDEAS),
                        help="approved inventory to show the teacher before it proposes additions")
    parser.add_argument("--log", default=str(LOG))
    parser.add_argument("--out", default=str(TALLY))
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new calls")
    parser.add_argument("--tally-only", action="store_true",
                        help="rebuild the tally from the log; make no calls")
    args = parser.parse_args()

    vocabulary = load_vocabulary(args.vocabulary)
    inventory = load_inventory(args.ideas, allow_partial=True)
    words = [row["word"] for row in vocabulary]
    plan = batches(words, args.batch_size)
    done = done_batches(args.log)
    todo = [(index, chunk) for index, chunk in plan if index not in done]
    scheduled = scheduled_batches(todo, args.limit)

    print("%d words, %d batches of %d: %d done, %d to go"
          % (len(words), len(plan), args.batch_size, len(done), len(todo)))

    if args.tally_only:
        write_json(args.out, tally(done, words))
        print("tally rebuilt from the log -> %s (no calls made)" % args.out)
        return

    teacher = teacher_from(args, args.log)
    for index, chunk in scheduled:
        try:
            answer = teacher.ask(KIND, SYSTEM, prompt_for(chunk, inventory), SCHEMA, SEED, TEMPERATURE,
                                 MAX_TOKENS, {"batch": index, "words": chunk,
                                              "prompt_version": PROMPT_VERSION})
        except DryRun as dry:
            print("\n-- batch %d, the request that would be sent --" % index)
            print(json.dumps(dry.body, indent=2))
            print("\ndry run: %d calls would be made to %s, none were."
                  % (len(scheduled), teacher.endpoint))
            return
        proposals, problem = check(answer, chunk, set(inventory) | ROLE_IDEAS)
        if problem:
            append_jsonl(args.log, {"batch": index, "ok": False, "why": problem, "words": chunk,
                                    "prompt_version": PROMPT_VERSION})
            print("batch %d rejected: %s" % (index, problem))
            continue
        append_jsonl(args.log, {"batch": index, "ok": True, "seed": SEED,
                                "prompt_version": PROMPT_VERSION, "proposals": proposals})
        print("batch %d: %d words, %d ideas"
              % (index, len(proposals), len({i for v in proposals.values() for i in v})))

    if args.limit and len(todo) > len(scheduled):
        print("stopping at --limit %d; rerun to carry on" % args.limit)

    done = done_batches(args.log)
    write_json(args.out, tally(done, words))
    print("\n%d batches done -> %s" % (len(done), args.out))
    print("NEXT, AND IT IS A PERSON'S JOB: pick the new ideas from that tally and write each one's\n"
          "meaning into ideas.json. Nothing tags against the new inventory until you do.")


if __name__ == "__main__":
    main()
