"""Three teacher passes over the vocabulary, aggregated into soft idea weights.

Phase 1 of docs/engineering/plans/smCLM_02.md. Twenty words a call, an `enum` of the approved
inventory so a pass can't invent an idea, three passes with seeds 16001, 16002 and 16003. An idea's
weight is the fraction of passes that gave it, so 0, ⅓, ⅔ or 1; phase 3 trains against those with
binary cross-entropy, the way smCLM_01 trains against 0/1.

Two things this script is careful about.

**A call is kept whole or not at all.** The words have to come back exactly: same words, each once,
same order, nothing invented. Every idea has to be in the inventory and every `pos` in the enum. A
call that fails any of that is logged with its reason, quarantined, and asked again; it is never
half-kept.

**Held-out words get no row.** `test_words.json` is human truth and `data/held_out.json` names the
words waiting for a person to write it. Both are dropped from `data/tags.jsonl` at aggregation
time, so the teacher's opinion of a held-out word never reaches training — which is the whole point
of holding it out. The raw calls stay in the log as evidence; the log is not training data.

Resumable: a (pass, batch) already answered is not asked again. Deterministic: sorted vocabulary,
fixed batch cut, and `data/tags.jsonl` is rebuilt from the whole log every run, so the same log
always gives the same file.

    python3 tag_words.py --dry-run              # the plan and one example request, nothing sent
    python3 tag_words.py --pass 16001           # one pass, resuming where the log left off
    python3 tag_words.py                        # all three passes
    python3 tag_words.py --aggregate-only       # rebuild data/tags.jsonl from the log, no calls
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from teacher import (DATA, HERE, POS, DryRun, add_teacher_arguments, append_jsonl, batches,
                     load_inventory, load_vocabulary, read_json, read_jsonl, teacher_from,
                     write_json, write_jsonl)


KIND = "word_tags"
PASSES = (16001, 16002, 16003)
BATCH = 20
TEMPERATURE = 0.7
MAX_TOKENS = 2400
MIN_IDEAS, MAX_IDEAS = 1, 10

LOG = DATA / "logs" / "tag_calls.jsonl"
QUARANTINE = DATA / "quarantine" / "tag_calls.jsonl"
TAGS = DATA / "tags.jsonl"
HELD_OUT = DATA / "held_out.json"
TEST_WORDS = HERE / "test_words.json"

SYSTEM = ("You tag English words with the general ideas they carry, from a fixed list. "
          "Answer with JSON only, in the schema you are given.")

PROMPT = """Tag each word below.

For each word give:
- pos: noun, verb, adjective, or other
- ideas: {min_ideas} to {max_ideas} ideas from the allowed list, most important first

The ideas and what they mean:
{inventory}

Rules:
- Only use ideas from that list. There is no other idea.
- Choose the ideas the word carries in ordinary use, not the ideas of one rare sense.
- Return every word below, exactly once, spelled exactly as given, in the same order.

Words:
{words}
"""


def schema_for(inventory):
    ideas = sorted(inventory)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["words"],
        "properties": {
            "words": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["word", "pos", "ideas"],
                    "properties": {
                        "word": {"type": "string"},
                        "pos": {"type": "string", "enum": list(POS)},
                        "ideas": {
                            "type": "array",
                            "minItems": MIN_IDEAS,
                            "maxItems": MAX_IDEAS,
                            "items": {"type": "string", "enum": ideas},
                        },
                    },
                },
            },
        },
    }


def prompt_for(words, inventory):
    lines = "\n".join("- %s: %s" % (idea, inventory[idea]) for idea in sorted(inventory))
    return PROMPT.format(min_ideas=MIN_IDEAS, max_ideas=MAX_IDEAS, inventory=lines,
                         words="\n".join(words))


def check(answer, words, inventory):
    """The call as {word: {"pos": ..., "ideas": [...]}}, or a reason to throw it away and retry.

    The enum is enforced here as well as in the schema: a server that ignores `strict` still can't
    get an idea past this, and the inventory is the only thing phase 3 can train against.
    """
    if not isinstance(answer, dict) or not isinstance(answer.get("words"), list):
        return None, "answer has no words array"
    rows = answer["words"]
    if not all(isinstance(row, dict) for row in rows):
        return None, "a row isn't an object"
    got = [row.get("word") for row in rows]
    if got != list(words):
        missing = sorted(set(words) - set(got))
        extra = sorted(w for w in set(got) if w not in set(words))
        return None, ("words came back wrong: %d sent, %d back, missing %s, extra %s"
                      % (len(words), len(got), missing[:5] or "-", extra[:5] or "-"))
    tags = {}
    for row in rows:
        word = row["word"]
        pos = row.get("pos")
        if pos not in POS:
            return None, "%s: pos %r isn't one of %s" % (word, pos, ", ".join(POS))
        ideas = row.get("ideas")
        if not isinstance(ideas, list) or not MIN_IDEAS <= len(ideas) <= MAX_IDEAS:
            return None, "%s: wants %d to %d ideas, got %r" % (word, MIN_IDEAS, MAX_IDEAS, ideas)
        outside = [i for i in ideas if i not in inventory]
        if outside:
            return None, "%s: ideas outside the inventory: %s" % (word, ", ".join(map(str, outside)))
        if len(set(ideas)) != len(ideas):
            return None, "%s: the same idea twice" % word
        tags[word] = {"pos": pos, "ideas": list(ideas)}
    return tags, None


def done_calls(log_path):
    """{(pass seed, batch index): {word: {...}}} for the calls already kept."""
    done = {}
    for row in read_jsonl(log_path):
        if row.get("ok") and row.get("batch") is not None:
            done[(int(row["pass"]), int(row["batch"]))] = row["tags"]
    return done


def held_out_words(test_words_path=TEST_WORDS, held_out_path=HELD_OUT):
    """Every word whose tags must be a person's, from both places they can be named.

    `held_out.py` writes the pick to data/held_out.json; a person fills test_words.json from it.
    Either file naming a word is enough to keep the teacher's tags for it out of training.
    """
    words = set()
    for path in (test_words_path, held_out_path):
        if not path or not Path(path).exists():
            continue
        for row in read_json(path).get("words") or []:
            if isinstance(row, dict) and row.get("word"):
                words.add(row["word"])
    return words


def aggregate(done, vocabulary_words, inventory, hidden, passes=PASSES):
    """Soft tags from the kept calls: weight = passes that gave the idea / passes that saw the word.

    Returns (rows, report). A word tagged by fewer than all the passes still gets a row — with its
    `passes` count on it, so nothing downstream has to guess — and the report says how many.
    """
    seen = defaultdict(set)  # word -> pass seeds that covered it
    votes = defaultdict(Counter)  # word -> idea -> passes that gave it
    pos_votes = defaultdict(Counter)
    for (pass_seed, _batch), tags in sorted(done.items()):
        if pass_seed not in passes:
            continue
        for word, tag in tags.items():
            seen[word].add(pass_seed)
            pos_votes[word][tag["pos"]] += 1
            for idea in tag["ideas"]:
                votes[word][idea] += 1

    rows, partial, unusable = [], [], []
    for word in sorted(seen):
        if word in hidden:
            continue
        n = len(seen[word])
        if n < len(passes):
            partial.append(word)
        ideas = {}
        for idea, count in votes[word].items():
            if idea in inventory:
                ideas[idea] = round(count / n, 2)
        # Ties on pos go to the earliest of noun, verb, adjective, other; a coin flip would make
        # the file depend on dict order, and this file has to be reproducible from the log.
        pos = min(pos_votes[word].items(), key=lambda kv: (-kv[1], POS.index(kv[0])))[0]
        row = {"word": word, "pos": pos, "passes": n,
               "ideas": dict(sorted(ideas.items(), key=lambda kv: (-kv[1], kv[0])))}
        if not any(weight >= 2 / 3 - 1e-9 for weight in ideas.values()):
            unusable.append(word)  # the phase 1 gate wants one idea at 2/3 or more per word
        rows.append(row)

    weights = Counter(round(w, 2) for row in rows for w in row["ideas"].values())
    covered = {row["word"] for row in rows}
    report = {
        "what": "How data/tags.jsonl was aggregated. Written by tag_words.py, no teacher call.",
        "passes": list(passes),
        "vocabulary_words": len(vocabulary_words),
        "words_tagged": len(rows),
        "words_untagged": sorted(set(vocabulary_words) - covered - hidden)[:50],
        "words_untagged_count": len(set(vocabulary_words) - covered - hidden),
        "held_out_dropped": sorted(hidden & set(seen)),
        "held_out_dropped_count": len(hidden & set(seen)),
        "words_short_of_all_passes": len(partial),
        "words_with_no_idea_at_two_thirds": unusable,
        "weight_histogram": dict(sorted(weights.items())),
        "ideas_used": len({i for row in rows for i in row["ideas"]}),
        "inventory_size": len(inventory),
        "human_gate": ("Held-out words are dropped here and tagged by a person in test_words.json. "
                       "The teacher's tags for them stay in data/logs only and are not truth."),
    }
    return rows, report


def main():
    parser = add_teacher_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--ideas", default=None)
    parser.add_argument("--log", default=str(LOG))
    parser.add_argument("--out", default=str(TAGS))
    parser.add_argument("--quarantine", default=str(QUARANTINE))
    parser.add_argument("--pass", dest="passes", type=int, action="append",
                        help="run one pass seed; repeatable. Default: all three")
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new calls")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="rebuild data/tags.jsonl from the log; make no calls")
    parser.add_argument("--allow-partial-inventory", action="store_true",
                        help="tag against ideas.json as it stands, before a person has added the new ideas")
    args = parser.parse_args()

    inventory = load_inventory(args.ideas, allow_partial=args.allow_partial_inventory)
    vocabulary = load_vocabulary(args.vocabulary)
    words = [row["word"] for row in vocabulary]
    passes = tuple(args.passes or PASSES)
    plan = batches(words, args.batch_size)
    done = done_calls(args.log)
    todo = [(seed, index, chunk) for seed in passes for index, chunk in plan
            if (seed, index) not in done]

    print("%d words, %d batches of %d, %d passes %s: %d calls done, %d to go"
          % (len(words), len(plan), args.batch_size, len(passes), passes,
             len([k for k in done if k[0] in passes]), len(todo)))

    if not args.aggregate_only:
        teacher = teacher_from(args, args.log)
        schema = schema_for(inventory)
        made = 0
        for seed, index, chunk in todo:
            if args.limit and made >= args.limit:
                print("stopping at --limit %d; rerun to carry on" % args.limit)
                break
            try:
                answer = teacher.ask(KIND, SYSTEM, prompt_for(chunk, inventory), schema, seed,
                                     TEMPERATURE, MAX_TOKENS, {"pass": seed, "batch": index})
            except DryRun as dry:
                print("\n-- pass %d batch %d, the request that would be sent --" % (seed, index))
                print(json.dumps(dry.body, indent=2))
                print("\ndry run: %d calls would be made to %s, none were."
                      % (len(todo), teacher.endpoint))
                return
            tags, problem = check(answer, chunk, inventory)
            if problem:
                append_jsonl(args.quarantine, {"pass": seed, "batch": index, "why": problem,
                                               "words": chunk, "answer": answer})
                append_jsonl(args.log, {"pass": seed, "batch": index, "ok": False, "why": problem})
                print("pass %d batch %d rejected: %s" % (seed, index, problem))
                continue
            append_jsonl(args.log, {"pass": seed, "batch": index, "ok": True, "tags": tags})
            made += 1
            print("pass %d batch %d: %d words" % (seed, index, len(tags)))
        done = done_calls(args.log)

    hidden = held_out_words()
    rows, report = aggregate(done, words, inventory, hidden, passes)
    write_jsonl(args.out, rows)
    # The report lands beside the tags, so pointing --out at a scratch folder keeps data/ clean.
    write_json(Path(args.out).with_name("tags_report.json"), report)
    print("\n%d words -> %s (%d held-out words dropped, %d words short of all %d passes)"
          % (len(rows), args.out, report["held_out_dropped_count"],
             report["words_short_of_all_passes"], len(passes)))
    if report["words_with_no_idea_at_two_thirds"]:
        print("phase 1 gate: %d words have no idea at 2/3 or more"
              % len(report["words_with_no_idea_at_two_thirds"]))
    print("STILL A PERSON'S JOB: tag the held-out words in test_words.json, and read the 100-word "
          "sample the plan asks for.")


if __name__ == "__main__":
    main()
