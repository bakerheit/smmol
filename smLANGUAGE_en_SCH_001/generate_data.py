"""Generate resumable curriculum batches with a local teacher model."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

from curriculum import BY_ID, LEVELS, corpus_problems, item_problems


HERE = Path(__file__).resolve().parent
DEFAULT_ENDPOINT = "http://127.0.0.1:11435"
DEFAULT_MODEL = "Qwen3.5-4B-Q4_K_M"
FOCUSES = (
    "animals, habitats, and the outdoors",
    "home objects, clothing, food, and cooking",
    "school, art, music, books, and making things",
    "clear action verbs for movement, senses, and daily tasks",
    "descriptive adjectives for size, texture, color, feeling, and behavior",
    "weather, plants, land, water, and seasons",
    "community places, jobs, buildings, and transportation",
    "the body, health, safety, and caring for others",
    "shapes, position, time, quantity, comparison, and order",
    "playgrounds, games, sports, celebrations, and friendship",
)
LETTER_GROUPS = ("a through d", "e through h", "i through l", "m through p", "q through t", "u through z")


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def all_existing(data_dir):
    items = read_jsonl(HERE / "seeds" / "preschool.jsonl")
    for level in LEVELS[1:]:
        items.extend(read_jsonl(data_dir / "levels" / (level["id"] + ".jsonl")))
    return items


def rejected_words(data_dir):
    words = set()
    for path in (data_dir / "review").glob("*-rejected.jsonl") if (data_dir / "review").exists() else ():
        words.update(str(item.get("word") or "").lower() for item in read_jsonl(path))
    return {word for word in words if word}


def teacher_prompt(level, count, used_words, focus, letters):
    used = ", ".join(sorted(used_words))
    return f"""Create exactly {count} original vocabulary items for {level['label']} English students.

Each item has:
- word: one useful lowercase English target word, not a name, abbreviation, number, phrase, contraction, or spelling variant
- sentence: one natural, factually sensible sentence that uses that exact word as a whole word and makes its meaning understandable

Rules:
- Pick genuinely {level['label']}-appropriate vocabulary, not words meant for much younger or older students.
- Pick content vocabulary: concrete nouns, precise verbs, descriptive adjectives, and useful concepts. Never use articles, pronouns, auxiliary verbs, conjunctions, prepositions, or other grammar glue as target words.
- For this batch, focus on {focus}.
- Prefer target words whose first letter is {letters}. Use a word outside that band only when needed for quality.
- Sentence length must be {level['min_words']} to {level['max_words']} words, counting hyphenated terms as one.
- Vary people, nature, daily life, arts, science, society, actions, descriptions, and ideas.
- Do not repeat a target word or sentence.
- Do not write definitions, labels, lessons, morals, fantasy filler, or commentary.
- Avoid every target word already used below. A word may still occur inside a sentence; it just cannot be the new target.

Already used target words:
{used}

Return only one JSON object shaped exactly like
{{"items":[{{"word":"example","sentence":"A sentence using example."}}]}}.
Do not use markdown fences or any other key."""


def post_json(url, body, timeout=600):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("teacher returned HTTP %d: %s" % (exc.code, detail[:500])) from exc


def response_format(count):
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
                    "required": ["word", "sentence"],
                    "properties": {
                        "word": {"type": "string"},
                        "sentence": {"type": "string"},
                    },
                },
            }
        },
    }
    return {"type": "json_schema", "json_schema": {"name": "curriculum", "strict": True, "schema": schema}}


def ask_teacher(endpoint, model, level, count, used_words, seed, attempt):
    focus = FOCUSES[(seed + attempt) % len(FOCUSES)]
    letters = LETTER_GROUPS[((seed // len(FOCUSES)) + attempt) % len(LETTER_GROUPS)]
    body = {
        "model": model,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": min(6000, 150 + count * (level["max_words"] + 12)),
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "system",
                "content": "You are a careful school vocabulary editor. Accuracy and natural sentences matter more than cleverness.",
            },
            {"role": "user", "content": teacher_prompt(level, count, used_words, focus, letters)},
        ],
        "response_format": response_format(count),
    }
    data = post_json(endpoint.rstrip("/") + "/v1/chat/completions", body)
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("teacher returned no completion: %r" % data) from exc
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S).strip()
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        start = text.find("{")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else ({}, 0)
        except ValueError as nested:
            raise RuntimeError("teacher returned broken JSON: %s" % text[:1000]) from nested
    if isinstance(parsed, dict) and not isinstance(parsed.get("items"), list):
        # Some local chat templates rename the sole array despite the requested schema.
        arrays = [value for value in parsed.values() if isinstance(value, list)]
        if len(arrays) == 1:
            parsed = {"items": arrays[0]}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        raise RuntimeError("teacher reply has no items array: %s" % text[:500])
    return parsed["items"]


def normalize(raw, level_id):
    return {
        "level": level_id,
        "word": str(raw.get("word") or "").strip().lower(),
        "sentence": " ".join(str(raw.get("sentence") or "").strip().split()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=[level["id"] for level in LEVELS[1:]], required=True)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--target", type=int, help="stop early at this many items; default is the level's full count")
    parser.add_argument("--attempts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()

    level = BY_ID[args.level]
    target = level["count"] if args.target is None else args.target
    if not 1 <= target <= level["count"]:
        parser.error("target must be between 1 and %d" % level["count"])
    path = args.data_dir / "levels" / (level["id"] + ".jsonl")
    kept = read_jsonl(path)
    existing = all_existing(args.data_dir)
    other = [item for item in existing if item.get("level") != level["id"]]
    initial_problems = corpus_problems(other + kept)
    if initial_problems:
        raise SystemExit("existing curriculum is broken:\n" + "\n".join(initial_problems[:20]))
    used_words = {item["word"].lower() for item in other + kept} | rejected_words(args.data_dir)
    used_sentences = {" ".join(item["sentence"].lower().split()) for item in other + kept}
    rejected = Counter()
    print("%s: %d/%d accepted" % (level["label"], len(kept), target), flush=True)

    empty_attempts = 0
    for attempt in range(args.attempts):
        if len(kept) >= target:
            break
        need = min(args.batch, target - len(kept))
        # Ask for extras because duplicates and malformed lines are expected from a small teacher model.
        requested = min(args.batch + 3, max(need + 3, int(need * 1.2)))
        candidates = ask_teacher(args.endpoint, args.model, level, requested, used_words, args.seed + attempt, attempt)
        added = 0
        for raw in candidates:
            if len(kept) >= target:
                break
            if not isinstance(raw, dict):
                rejected["not an object"] += 1
                continue
            item = normalize(raw, level["id"])
            sentence_key = item["sentence"].lower()
            if item["word"] in used_words:
                rejected["duplicate word"] += 1
                continue
            if sentence_key in used_sentences:
                rejected["duplicate sentence"] += 1
                continue
            problems = item_problems(item)
            if problems:
                rejected[problems[0]] += 1
                continue
            kept.append(item)
            used_words.add(item["word"])
            used_sentences.add(sentence_key)
            added += 1
        write_jsonl(path, kept)
        print(
            "attempt %d: asked %d, received %d, kept %d -> %d/%d"
            % (attempt + 1, requested, len(candidates), added, len(kept), target),
            flush=True,
        )
        empty_attempts = empty_attempts + 1 if added == 0 else 0
        if empty_attempts >= 5:
            raise SystemExit("teacher produced no usable items; rejections: %s" % dict(rejected))

    print("rejected:", dict(rejected), flush=True)
    if len(kept) < target:
        raise SystemExit("stopped at %d/%d items" % (len(kept), target))
    print("complete:", path, flush=True)


if __name__ == "__main__":
    main()
