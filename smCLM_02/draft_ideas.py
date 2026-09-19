"""smCLM_02 phase 1: a *draft* of the new ideas, for a person to approve, edit or reject.

The plan's human gate says a person picks the new ideas from `data/idea_proposals.json` and writes
each one's meaning. On 2026-09-19 the owner asked for a model to draft that pick instead, then review
it. This script is that draft and nothing more: it writes `data/ideas_draft.json` and a readable
`data/ideas_draft.md`, and **never writes `ideas.json`**. `ideas.json` changes only when a person
has read the draft and accepted it (`--accept`), and the accept records who and when.

The drafter is Claude Sonnet 5 at medium effort, via the Anthropic API, key from ANTHROPIC_API_KEY.
It sees v1's 51 ideas with their meanings, every tallied candidate carried by 8 or more words with
its example words, and the plan's rules. It proposes the new ideas with one-line meanings in v1's
style, says which tallied names each one merges, and says why each dropped candidate was dropped.

    ANTHROPIC_API_KEY=... python3 draft_ideas.py      ask for the draft, write data/ideas_draft.*
    python3 draft_ideas.py --check                     re-run the local checks on the saved draft
    python3 draft_ideas.py --accept "name"             append the reviewed draft to ideas.json
"""
import argparse
from datetime import date
import json
import re
import sys

from teacher import DATA, IDEAS, read_json, write_json

MODEL = "claude-sonnet-5"
EFFORT = "medium"
MIN_WORDS = 8  # the plan: every idea must fit at least 8 vocabulary words
TARGET_NEW = (100, 115)  # plan: "about 100 new ones"; gate: at least 120 in all
NAME = re.compile(r"^[a-z][a-z_]{1,23}$")

TALLY = DATA / "idea_proposals.json"
DRAFT = DATA / "ideas_draft.json"
DRAFT_MD = DATA / "ideas_draft.md"

SYSTEM = ("You are helping a researcher build a small, human-readable inventory of general ideas "
          "that English words carry. A tiny model will be trained to predict these ideas for any "
          "word from how it is used, so the ideas must be broad, distinct and plainly named.")

PROMPT = """Below are the {n_v1} approved ideas from version 1 of the inventory, each with its
one-line meaning, then {n_cand} candidate ideas that a teacher model proposed while reading a
{n_vocab}-word vocabulary. Each candidate shows how many vocabulary words it was proposed for and
some of those words.

Draft the new ideas to add, {low} to {high} of them.

Rules:
- The {n_v1} approved ideas stay exactly as they are. Never re-propose one, and never add a new idea
  that means the same thing as one of them (for example, "transport" duplicates "transportation",
  "heat" duplicates "hot", "creature" duplicates "animal").
- Merge candidates that are near-synonyms into one idea with one plain name (for example
  "movement" and "motion"; "location" and "position"). List every candidate name you merged into it.
- Every new idea must be broad enough to fit at least {min_words} different words in the
  vocabulary and at most 40% of it. Prefer ideas about meaning (health, money, danger, texture)
  over ideas about grammar or word form.
- Aim for coverage of: domains (health, money, school, work, home, travel, food, technology, family,
  time, weather, sport, art), properties (size, speed, danger, temperature, texture) and roles
  (person, place, thing, event, action, feeling). Where one of those isn't covered by an approved
  idea or a candidate, you may add it; mark its source as "coverage".
- Name: one lowercase English word, or two joined by an underscore, 24 characters at most.
- Meaning: one short line in the same style as the approved ones ("is ...", "has to do with ...",
  "can ...").
- For every candidate you leave out, give a reason of a few words.

Approved ideas:
{v1}

Candidates (idea: words proposed for | example words):
{candidates}
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["new_ideas", "dropped"],
    "properties": {
        "new_ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["idea", "meaning", "merges", "source", "examples"],
                "properties": {
                    "idea": {"type": "string"},
                    "meaning": {"type": "string"},
                    "merges": {"type": "array", "items": {"type": "string"}},
                    "source": {"type": "string", "enum": ["tally", "coverage"]},
                    "examples": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "dropped": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate", "reason"],
                "properties": {"candidate": {"type": "string"}, "reason": {"type": "string"}},
            },
        },
    },
}


def candidates(tally, v1):
    return [row for row in tally["proposals"] if row["words"] >= MIN_WORDS and row["idea"] not in v1]


def prompt_for(tally, v1):
    cands = candidates(tally, v1)
    return PROMPT.format(
        n_v1=len(v1), n_cand=len(cands), n_vocab=tally["vocabulary_words"], low=TARGET_NEW[0],
        high=TARGET_NEW[1], min_words=MIN_WORDS,
        v1="\n".join("- %s: %s" % kv for kv in v1.items()),
        candidates="\n".join("- %s: %d | %s" % (r["idea"], r["words"], ", ".join(r["examples"]))
                             for r in cands))


def problems(draft, v1, tally):
    """What's wrong with a draft, as a list of sentences. Empty means it passes the local checks."""
    found = []
    names = [row["idea"] for row in draft["new_ideas"]]
    low, high = TARGET_NEW
    if not low <= len(names) <= high:
        found.append("%d new ideas, outside %d to %d" % (len(names), low, high))
    if len(v1) + len(names) < 120:
        found.append("inventory would be %d, under the gate's 120" % (len(v1) + len(names)))
    for name in names:
        if not NAME.match(name):
            found.append("bad name: %r" % name)
        if name in v1:
            found.append("re-proposes a v1 idea: %s" % name)
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        found.append("proposed twice: %s" % ", ".join(dupes))
    for row in draft["new_ideas"]:
        if not row["meaning"].strip() or "\n" in row["meaning"] or len(row["meaning"]) > 80:
            found.append("meaning isn't one short line: %s" % row["idea"])
    offered = {r["idea"] for r in candidates(tally, v1)}
    accounted = {m for row in draft["new_ideas"] for m in row["merges"]} | set(names) | \
                {d["candidate"] for d in draft["dropped"]}
    missing = sorted(offered - accounted)
    if missing:
        found.append("%d candidates neither used nor dropped: %s" % (len(missing), ", ".join(missing[:10])))
    return found


def markdown(draft, found, usage):
    lines = ["# smCLM_02 idea inventory: DRAFT for review", "",
             "Drafted by %s at %s effort on %s. **Not approved.** `ideas.json` is unchanged until a "
             "person accepts this (`python3 draft_ideas.py --accept \"name\"`)." % (MODEL, EFFORT, date.today()),
             "", "Tokens: %s." % usage, ""]
    lines += ["## Local checks", ""] + (["- " + f for f in found] if found else ["- all pass"]) + [""]
    lines += ["## New ideas (%d)" % len(draft["new_ideas"]), "", "| # | idea | meaning | merges | examples |",
              "|---|---|---|---|---|"]
    for i, row in enumerate(draft["new_ideas"], 1):
        merges = ", ".join(m for m in row["merges"] if m != row["idea"])
        tag = " *(coverage)*" if row["source"] == "coverage" else ""
        lines.append("| %d | `%s`%s | %s | %s | %s |" % (i, row["idea"], tag, row["meaning"], merges,
                                                        ", ".join(row["examples"][:5])))
    lines += ["", "## Dropped candidates (%d)" % len(draft["dropped"]), "", "| candidate | reason |", "|---|---|"]
    lines += ["| `%s` | %s |" % (d["candidate"], d["reason"]) for d in draft["dropped"]]
    return "\n".join(lines) + "\n"


def ask(prompt):
    import anthropic
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from the environment
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason != "end_turn":
        raise SystemExit("the draft stopped early: %s" % message.stop_reason)
    text = next(b.text for b in message.content if b.type == "text")
    return json.loads(text), message.usage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="re-check the saved draft; no call")
    parser.add_argument("--accept", metavar="NAME", help="the reviewer's name; appends the draft to ideas.json")
    args = parser.parse_args(argv)

    v1 = read_json(IDEAS)
    tally = read_json(TALLY)

    if args.accept:
        saved = read_json(DRAFT)
        found = problems(saved["draft"], v1, tally)
        if found:
            raise SystemExit("won't accept a draft that fails its checks:\n  " + "\n  ".join(found))
        if len(v1) != 51:
            raise SystemExit("ideas.json already has %d ideas, not v1's 51; accept once only" % len(v1))
        inventory = dict(v1)
        for row in saved["draft"]["new_ideas"]:
            inventory[row["idea"]] = row["meaning"].strip()
        write_json(IDEAS, inventory)
        saved["accepted"] = {"by": args.accept, "on": str(date.today())}
        write_json(DRAFT, saved)
        print("ideas.json: %d ideas (51 from v1, %d new), accepted by %s"
              % (len(inventory), len(inventory) - 51, args.accept))
        return 0

    if args.check:
        saved = read_json(DRAFT)
        found = problems(saved["draft"], v1, tally)
        print("\n".join(found) or "all checks pass")
        return 1 if found else 0

    draft, usage = ask(prompt_for(tally, v1))
    found = problems(draft, v1, tally)
    usage_text = "input %d, output %d" % (usage.input_tokens, usage.output_tokens)
    write_json(DRAFT, {"model": MODEL, "effort": EFFORT, "drafted": str(date.today()),
                       "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
                       "checks": found, "accepted": None, "draft": draft})
    DRAFT_MD.write_text(markdown(draft, found, usage_text))
    print("%d new ideas, %d dropped; %s" % (len(draft["new_ideas"]), len(draft["dropped"]), usage_text))
    print("\n".join("CHECK: " + f for f in found) or "all local checks pass")
    print("review %s, then: python3 draft_ideas.py --accept \"your name\"" % DRAFT_MD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
