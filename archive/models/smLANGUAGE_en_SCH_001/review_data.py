"""Second-pass semantic review for one generated curriculum level."""

import argparse
import json
import os
from pathlib import Path
import re

from curriculum import BY_ID, LEVELS, item_problems
from generate_data import post_json, read_jsonl, write_jsonl


HERE = Path(__file__).resolve().parent


def review_format(count):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["word", "accept", "sentence", "reason"],
                    "properties": {
                        "word": {"type": "string"},
                        "accept": {"type": "boolean"},
                        "sentence": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            }
        },
    }
    return {"type": "json_schema", "json_schema": {"name": "curriculum_review", "strict": True, "schema": schema}}


def prompt(level, items):
    return f"""Review these proposed {level['label']} vocabulary items as a skeptical school editor.

For every input word, return one review object with the exact same word:
- accept: true only when the target is useful at {level['label']} level and the sentence is natural, grammatical, factual, age-appropriate, and clearly demonstrates the target's normal meaning
- sentence: if the word is suitable, silently repair any sentence problem while keeping the exact target word and {level['min_words']}-{level['max_words']} total words
- reason: empty when accepted; otherwise a short concrete reason

Reject a word when it is grammar glue, a mere inflection of an earlier basic word, far above or below the grade, obscure, needlessly negative, a proper name, or not useful vocabulary. Reject rather than trying to replace the target word. Watch for circular examples, factual errors, awkward collocations, and sentences that technically contain the word without teaching it.

Input:
{json.dumps(items, ensure_ascii=False, indent=1)}

Return only the requested JSON object, in the same order."""


def parse_reply(data):
    try:
        text = str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("reviewer returned no completion") from exc
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        start = text.find("{")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        except ValueError as exc:
            raise RuntimeError("reviewer returned broken JSON: %s" % text[:1000]) from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        raise RuntimeError("reviewer returned no items array")
    return parsed["items"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=[level["id"] for level in LEVELS[1:]], required=True)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--endpoint", default=os.environ.get("SMMOL_LLM_URL", "http://127.0.0.1:8081"))
    parser.add_argument("--model", default="ministral-8b")
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--from-index", type=int, default=0, help="carry earlier reviewed rows forward unchanged")
    args = parser.parse_args()

    level = BY_ID[args.level]
    path = args.data_dir / "levels" / (level["id"] + ".jsonl")
    original = read_jsonl(path)
    if not original:
        raise SystemExit("no items to review in %s" % path)
    if not 0 <= args.from_index <= len(original):
        parser.error("from-index is outside the level file")
    review_dir = args.data_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rounds = sorted(review_dir.glob(level["id"] + "-round-*-summary.json"))
    round_number = len(rounds) + 1
    prefix = "%s-round-%02d" % (level["id"], round_number)
    before = review_dir / (prefix + "-before.jsonl")
    write_jsonl(before, original)

    accepted, rejected, changed = list(original[: args.from_index]), [], 0
    for start in range(args.from_index, len(original), args.batch):
        batch = original[start : start + args.batch]
        body = {
            "model": args.model,
            "stream": False,
            "temperature": 0,
            "max_tokens": min(6000, 200 + len(batch) * 70),
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": review_format(len(batch)),
            "messages": [
                {"role": "system", "content": "You are a strict elementary-school vocabulary editor. Reject doubtful material."},
                {"role": "user", "content": prompt(level, batch)},
            ],
        }
        reply = parse_reply(post_json(args.endpoint.rstrip("/") + "/v1/chat/completions", body))
        if len(reply) != len(batch):
            raise SystemExit("reviewer returned %d rows for a %d-row batch" % (len(reply), len(batch)))
        for source, review in zip(batch, reply):
            if str(review.get("word") or "").lower() != source["word"]:
                raise SystemExit("reviewer changed or reordered %s as %s" % (source["word"], review.get("word")))
            sentence = " ".join(str(review.get("sentence") or "").split())
            candidate = dict(source, sentence=sentence)
            problems = item_problems(candidate)
            if review.get("accept") is True and not problems:
                accepted.append(candidate)
                changed += sentence != source["sentence"]
            else:
                rejected.append(
                    dict(
                        source,
                        review_reason=str(review.get("reason") or "; ".join(problems) or "reviewer rejected it"),
                        proposed_sentence=sentence,
                    )
                )
        print(
            "reviewed %d/%d | accepted %d | rejected %d | rewrote %d"
            % (min(start + len(batch), len(original)), len(original), len(accepted), len(rejected), changed),
            flush=True,
        )

    write_jsonl(path, accepted)
    write_jsonl(review_dir / (prefix + "-rejected.jsonl"), rejected)
    summary = {
        "level": level["id"],
        "model": args.model,
        "input": len(original) - args.from_index,
        "carried_from_earlier_review": args.from_index,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rewritten": changed,
    }
    summary["round"] = round_number
    (review_dir / (prefix + "-summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
