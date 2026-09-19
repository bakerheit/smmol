#!/usr/bin/env python3
"""The other baseline: Ministral 8B on the PC, asked to route the hand-written test set with a JSON schema.

    python3 baseline_llm.py          # writes out/llm_baseline.json; train.py adds it to its table
"""
import json
import os
import statistics
import time
import urllib.request

from train import score

HERE = os.path.dirname(os.path.abspath(__file__))
# Point this somewhere else with SMMOL_LLM_URL; the committed default is local.
URL = os.environ.get("SMMOL_LLM_URL", "http://127.0.0.1:8081").rstrip("/") + "/v1/chat/completions"

with open(os.path.join(HERE, "labels.json")) as f:
    LABELS = json.load(f)
SYSTEM = ("You route messages for a home assistant. Classify the person's message.\n"
          + "\n".join("intent %s: %s" % (k, LABELS["about"][k]) for k in LABELS["intents"]) + "\n"
          + "\n".join("tool %s: %s" % (k, LABELS["about"][k]) for k in LABELS["tools"]) + "\n"
          + "ask_first: " + LABELS["about"]["ask_first"] + "\nReply with JSON only.")
SCHEMA = {"type": "object", "required": ["intent", "tool", "ask_first"], "properties": {
    "intent": {"type": "string", "enum": LABELS["intents"]}, "tool": {"type": "string", "enum": LABELS["tools"]},
    "ask_first": {"type": "boolean"}}}


def ask(prev, text):
    body = {"model": "ministral-8b", "temperature": 0, "max_tokens": 60, "stream": False,
            "response_format": {"type": "json_object", "schema": SCHEMA},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "The assistant's last message: %s\nThe person's message: %s" % (prev or "(none)", text)}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        content = json.loads(resp.read())["choices"][0]["message"]["content"]
    return json.loads(content[content.find("{"):content.rfind("}") + 1])


def main():
    with open(os.path.join(HERE, "test.json")) as f:
        test = json.load(f)
    guesses, timings = [], []
    for e in test:
        started = time.time()
        try:
            g = ask(e["prev"], e["text"])
        except Exception as exc:  # a bad reply counts as wrong, not as a crash
            g = {"intent": "?", "tool": "?", "ask_first": None, "error": str(exc)}
        timings.append(time.time() - started)
        guesses.append(g)
        print("%.1fs  %-50s %s" % (timings[-1], e["text"][:50], g), flush=True)
    result, mistakes = score(test, guesses)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "llm_baseline.json"), "w") as f:
        json.dump({"score": result, "median_seconds": round(statistics.median(timings), 2), "mistakes": mistakes}, f, indent=1)
    print(result, "median %.2fs per message" % statistics.median(timings))


if __name__ == "__main__":
    main()
