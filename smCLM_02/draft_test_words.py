"""smCLM_02 phase 1: a *draft* of the held-out answer key, for a person to review and accept.

The plan says the held-out words' ideas are written by a person, never by the teacher, because they
are the answer key the idea reader is scored against. On 2026-09-19 the owner chose to have a model
draft them and review the draft instead. What keeps that honest:

  * the drafter is Claude Sonnet 5, not the teacher (kimi-k2.6 tagged the training words), and it
    never sees the teacher's tags or any training sentence: only the word, its part of speech, the
    approved ideas with their meanings, and v1's human tags where the word is one of v1's 23;
  * v1's human tags are kept as they are; the draft may only add new ideas to them;
  * it writes `data/test_words_draft.json` and a readable `.md`, never `test_words.json`.
    `--accept "name"` writes `test_words.json`, and every row says it was model-drafted and
    person-reviewed, so no one later mistakes it for hand-written truth.

    ANTHROPIC_API_KEY=... python3 draft_test_words.py     draft -> data/test_words_draft.*
    python3 draft_test_words.py --accept "name"            write test_words.json from the reviewed draft
"""
import argparse
from datetime import date
import json
import sys

from teacher import DATA, HERE, IDEAS, POS, load_inventory, read_json, write_json

MODEL = "claude-sonnet-5"
EFFORT = "medium"
MIN_IDEAS, MAX_IDEAS = 1, 10
HELD_OUT = DATA / "held_out.json"
V1 = HERE.parent / "smCLM_01" / "concepts.json"
TEST_WORDS = HERE / "test_words.json"
DRAFT = DATA / "test_words_draft.json"
DRAFT_MD = DATA / "test_words_draft.md"

SYSTEM = ("You are writing the answer key for a test. Each English word below must be labelled with "
          "the general ideas it carries in ordinary use, chosen from a fixed list. Be careful and "
          "literal: a label belongs only if most people would agree the word carries that idea.")

PROMPT = """Label each word with {low} to {high} ideas from the list, most important first.

Rules:
- Only ideas from the list. Choose what the word carries in its ordinary, most common sense.
- Include every idea that clearly applies; leave out ideas that are only a stretch.
- Some words come with labels a person already wrote ("keep"). Keep all of those, in that order,
  and add any other ideas from the list that clearly apply.
- Return every word, once, in the order given.

The ideas and what they mean:
{inventory}

Words (word | part of speech | labels to keep):
{words}
"""


def schema(inventory):
    return {
        "type": "object", "additionalProperties": False, "required": ["words"],
        "properties": {"words": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["word", "ideas", "note"],
            "properties": {
                "word": {"type": "string"},
                "ideas": {"type": "array", "items": {"type": "string", "enum": list(inventory)}},
                "note": {"type": "string"},
            }}}},
    }


def load_rows(inventory):
    """The held-out words, each with its pos, band, source, and v1's human tags where it has them."""
    v1 = read_json(V1)["words"]
    rows = []
    for row in read_json(HELD_OUT)["words"]:
        keep = [i for i in (v1.get(row["word"]) or []) if i in inventory] if row["source"] == "v1" else []
        rows.append({"word": row["word"], "pos": row["pos"], "band": row.get("band"),
                     "source": row["source"], "keep": keep})
    return rows


def problems(draft, rows, inventory):
    found = []
    got = [r["word"] for r in draft["words"]]
    if got != [r["word"] for r in rows]:
        found.append("words came back wrong or out of order")
        return found
    for want, row in zip(rows, draft["words"]):
        ideas = row["ideas"]
        if not MIN_IDEAS <= len(ideas) <= MAX_IDEAS:
            found.append("%s: %d ideas" % (row["word"], len(ideas)))
        if len(set(ideas)) != len(ideas):
            found.append("%s: an idea twice" % row["word"])
        outside = [i for i in ideas if i not in inventory]
        if outside:
            found.append("%s: outside the inventory: %s" % (row["word"], outside))
        lost = [i for i in want["keep"] if i not in ideas]
        if lost:
            found.append("%s: dropped v1's human tags %s" % (row["word"], lost))
    return found


def markdown(draft, rows, found, extra):
    lines = ["# smCLM_02 held-out answer key: DRAFT for review", "",
             "Drafted by %s at %s effort, %s. %s" % (MODEL, EFFORT, date.today(), extra), "",
             "**Bold** ideas are v1's human tags, kept as written; the rest are the draft's. Mark any "
             "word you'd change, then accept with `python3 draft_test_words.py --accept \"name\"`.", ""]
    lines += ["## Checks", ""] + (["- " + f for f in found] if found else ["- all pass"]) + [""]
    lines += ["| # | word | pos | ideas | note |", "|---|---|---|---|---|"]
    for i, (want, row) in enumerate(zip(rows, draft["words"]), 1):
        ideas = ", ".join("**%s**" % x if x in want["keep"] else x for x in row["ideas"])
        lines.append("| %d | `%s` | %s | %s | %s |" % (i, row["word"], want["pos"], ideas, row["note"]))
    return "\n".join(lines) + "\n"


def ask(prompt, inventory):
    import anthropic
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=MODEL, max_tokens=32000, thinking={"type": "adaptive"},
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema(inventory)}},
        system=SYSTEM, messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason != "end_turn":
        raise SystemExit("the draft stopped early: %s" % message.stop_reason)
    return json.loads(next(b.text for b in message.content if b.type == "text")), message.usage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accept", metavar="NAME")
    args = parser.parse_args(argv)
    inventory = load_inventory(IDEAS)
    rows = load_rows(inventory)

    if args.accept:
        saved = read_json(DRAFT)
        found = problems(saved["draft"], rows, inventory)
        if found:
            raise SystemExit("won't accept a draft that fails its checks:\n  " + "\n  ".join(found))
        shell = read_json(TEST_WORDS)
        if shell.get("words"):
            raise SystemExit("test_words.json already has rows; accept once only")
        shell["words"] = [{
            "word": want["word"], "pos": want["pos"], "ideas": row["ideas"], "band": want["band"],
            "source": "v1+model-draft, person-reviewed" if want["source"] == "v1" else "model-draft, person-reviewed",
            "note": row["note"],
        } for want, row in zip(rows, saved["draft"]["words"])]
        shell["filled"] = {"on": str(date.today()), "drafted_by": "%s (%s effort)" % (MODEL, EFFORT),
                           "reviewed_by": args.accept,
                           "why": "the owner chose a reviewed model draft over hand-tagging all 83; "
                                  "the drafter is not the teacher and never saw the teacher's tags"}
        write_json(TEST_WORDS, shell)
        saved["accepted"] = {"by": args.accept, "on": str(date.today())}
        write_json(DRAFT, saved)
        print("test_words.json: %d rows, reviewed by %s" % (len(shell["words"]), args.accept))
        return 0

    words = "\n".join("%s | %s | %s" % (r["word"], r["pos"], ", ".join(r["keep"]) or "-") for r in rows)
    prompt = PROMPT.format(low=MIN_IDEAS, high=MAX_IDEAS, words=words,
                           inventory="\n".join("- %s: %s" % kv for kv in inventory.items()))
    draft, usage = ask(prompt, inventory)
    found = problems(draft, rows, inventory)
    usage_text = "Tokens: input %d, output %d." % (usage.input_tokens, usage.output_tokens)
    write_json(DRAFT, {"model": MODEL, "effort": EFFORT, "drafted": str(date.today()),
                       "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
                       "checks": found, "accepted": None, "draft": draft})
    DRAFT_MD.write_text(markdown(draft, rows, found, usage_text))
    print("%d words drafted; %s" % (len(draft["words"]), usage_text))
    print("\n".join("CHECK: " + f for f in found) or "all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
