"""Ministral 8B on the PC, prompted to write the same problem list, scored on the same hand-written messages.

    python3 baseline_llm.py
"""
import json
import os
import statistics
import time
import urllib.request

from score import score

HERE = os.path.dirname(os.path.abspath(__file__))
# Point this somewhere else with SMMOL_LLM_URL; the committed default is local.
URL = os.environ.get("SMMOL_LLM_URL", "http://127.0.0.1:8081").rstrip("/") + "/v1/chat/completions"
MODEL = "ministral-8b"
SYSTEM = (
    "Read the person's message and list every arithmetic problem in it. Don't solve anything.\n"
    "Each problem has expression, unit and about. expression uses only numbers, + - * / ** ( ) and sqrt(): no units, "
    "commas, $ or % signs (write 15% of 80 as 80*15/100). unit is the answer's unit, or empty. about is a word or two, or empty.\n"
    "If a problem uses an earlier answer, write p1, p2 ... for it. If there is nothing to work out, return an empty list.\n\n"
    'Example for "split $90 between 3 people and add a $5 tip each":\n'
    '{"problems":[{"expression":"90/3","unit":"dollars","about":"each"},{"expression":"p1+5","unit":"dollars","about":"each with tip"}]}\n\n'
    'Example for "my 3 cats are asleep":\n{"problems":[]}\n\n'
    "Reply with JSON only.")
SCHEMA = {"type": "object", "required": ["problems"], "properties": {"problems": {"type": "array", "maxItems": 5, "items": {
    "type": "object", "required": ["expression", "unit", "about"], "properties": {
        "expression": {"type": "string", "maxLength": 120}, "unit": {"type": "string", "maxLength": 30},
        "about": {"type": "string", "maxLength": 40}}}}}}


def ask(text):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 300, "response_format": {"type": "json_object", "schema": SCHEMA},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        content = json.load(response)["choices"][0]["message"]["content"]
    problems = json.loads(content[content.find("{"):content.rfind("}") + 1]).get("problems", [])
    return [str(p.get("expression", "")) for p in problems if isinstance(p, dict)]


def main():
    with open(os.path.join(HERE, "test.json")) as f:
        test = json.load(f)
    predictions, seconds = [], []
    for e in test:
        started = time.time()
        try:
            got = ask(e["text"])
        except Exception as exc:  # a timeout or a broken reply counts as finding nothing
            print("failed: %s (%s)" % (e["text"], exc), flush=True)
            got = []
        seconds.append(time.time() - started)
        predictions.append(got)
        print("%5.1fs  %-60s %s" % (seconds[-1], e["text"][:60], got), flush=True)
    result, mistakes = score(test, predictions)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "llm_baseline.json"), "w") as f:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M"), "model": MODEL, "score": result,
                   "median_seconds": round(statistics.median(seconds), 2), "mistakes": mistakes}, f, indent=1)
    print("\n%s: every problem right %.0f%%, final answer right %.0f%%, found the math %.0f%%, quiet when there's none %.0f%%, %.1fs median"
          % (MODEL, 100 * result["every"], 100 * result["final"], 100 * result["found"], 100 * result["quiet"], statistics.median(seconds)))


if __name__ == "__main__":
    main()
