"""A/B the idea-proposal prompt on Ministral: v1 (open), v2 (reuse, current), v3 (open + breadth rules).

Same 3 batches for every prompt, spread across the vocabulary. Writes next to itself; the real log is
untouched. Run 2026-09-19 against the PC with SMMOL_LLM_URL set. reuse_v2_batch0_ministral.jsonl is
the one batch the full run made under reuse_v2 before the A/B stopped it.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]  # smCLM_02/
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import propose_ideas as P
from teacher import Teacher, batches, load_inventory, load_vocabulary

V1 = """For each word below, name the general ideas it carries.

An idea is a broad, reusable category that many different words share, like health, money, speed,
danger, family, food, person, place, action or feeling. It is never a definition of the word and
never a synonym of it.

Rules:
- Give 1 to {max_ideas} ideas per word, most important first.
- Each idea is one lowercase English word, or two joined by an underscore. No spaces, no phrases.
- Reuse the same idea name across words wherever it fits. That is the whole point.
- Return every word below, exactly once, spelled exactly as given, in the same order.

Words:
{words}
"""

V3 = """For each word below, name the broad, reusable ideas it carries.

An idea is a broad, reusable category that many different words share, like health, money, speed,
danger, family, food, person, place, action or feeling. It is never a definition of the word and
never a synonym of it.

Rules:
- Give 1 to {max_ideas} ideas per word, most important first.
- Name what the word is really about. Do not force a word into a category that only loosely fits.
- Every idea must be broad enough to fit at least 8 different words in a 2,500-word vocabulary.
  If it mainly fits this word, one word family, or one narrow object, do not use it.
- Use one shared name for related meanings. Do not create near-synonyms or grammatical variants.
- Each idea is one lowercase English word, or two joined by an underscore. No spaces, no phrases.
- Reuse the same idea name across words wherever it fits. That is the whole point.
- Return every word below, exactly once, spelled exactly as given, in the same order.

Words:
{words}
"""

V2 = """For each word below, name the broad, reusable ideas it carries.

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

PROMPTS = {"v1_open": (V1, 6), "v2_reuse": (V2, 3), "v3_open_broad": (V3, 3)}
PICK = [30, 75, 120]


def main():
    OUT.mkdir(exist_ok=True)
    inventory = load_inventory(str(P.IDEAS), allow_partial=True)
    words = [r["word"] for r in load_vocabulary()]
    plan = dict(batches(words, P.BATCH))
    url = os.environ["SMMOL_LLM_URL"]
    results = {}
    for name, (template, max_ideas) in PROMPTS.items():
        schema = json.loads(json.dumps(P.SCHEMA))
        schema["properties"]["words"]["items"]["properties"]["ideas"]["maxItems"] = max_ideas
        P.MAX_IDEAS_PER_WORD = max_ideas
        teacher = Teacher(url, "ministral-8b", OUT / ("%s.jsonl" % name))
        got = {}
        for index in PICK:
            chunk = plan[index]
            prompt = template.format(max_ideas=max_ideas, words="\n".join(chunk),
                                     inventory="\n".join("- %s: %s" % kv for kv in inventory.items()))
            for attempt in range(3):
                answer = teacher.ask("idea_proposals_ab", P.SYSTEM, prompt, schema, P.SEED + attempt,
                                     P.TEMPERATURE, P.MAX_TOKENS, {"batch": index, "prompt": name})
                proposals, problem = P.check(answer, chunk, set(inventory) | P.ROLE_IDEAS)
                if not problem:
                    break
                print(name, index, "rejected:", problem)
            got.update(proposals or {})
            print(name, "batch", index, "ok" if proposals else "FAILED", flush=True)
        results[name] = got
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
