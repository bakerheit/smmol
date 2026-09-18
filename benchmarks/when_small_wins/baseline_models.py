#!/usr/bin/env python3
"""Was ministral-8b a fair opponent? Re-run the CLI comparison against other models.

    SMMOL_LLM_URL=http://<pc>:8081 python3 baseline_models.py qwen3-coder-30b
    SMMOL_LLM_URL=http://<pc>:8081 python3 baseline_models.py qwen3-coder-30b --limit 12

Every headline in this repo compares a small model against `ministral-8b`. Nothing in
the docs says why that model. It is the one that happened to be running on the PC.

That matters most for smTOOLS_COMPUTER_CLI_01, which posts the repo's largest single
win (+31.9 on exact command) against a Ministral score of 14.9% — a weak showing for
an 8B on shell commands, and a suspicious one when the PC also serves a 30B model
specialised in code.

This changes exactly one thing: the model name. The system prompt, the JSON schema,
the parsing and the grading all come from smTOOLS_COMPUTER_CLI_01/baseline_llm.py
unchanged, so the numbers are comparable to the committed ones.

Writes data/baseline_<model>.json with every answer, so right-program and normalised
scoring can be recomputed without paying for the calls again.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CLI = os.path.join(ROOT, "smTOOLS_COMPUTER_CLI_01")
sys.path.insert(0, CLI)
sys.path.insert(0, HERE)

import baseline_llm as B          # noqa: E402  its SYSTEM prompt and SCHEMA, unchanged
from normalise import same        # noqa: E402
from score_normalised import first_token  # noqa: E402

URL = os.environ.get("SMMOL_LLM_URL", "http://127.0.0.1:8081").rstrip("/") + "/v1/chat/completions"


def ask(model, platform, text, timeout=300):
    body = {"model": model, "temperature": 0, "max_tokens": 200,
            "response_format": {"type": "json_object", "schema": B.SCHEMA},
            "messages": [{"role": "system", "content": B.SYSTEM},
                         {"role": "user", "content": "%s: %s" % (platform, text)}]}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = json.load(response)["choices"][0]["message"]["content"]
    said = json.loads(content[content.find("{"):content.rfind("}") + 1])
    return str(said.get("command", "")).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=0, help="first N command items only, for a cheap probe")
    args = ap.parse_args()

    with open(os.path.join(CLI, "test.json")) as f:
        test = [e for e in json.load(f) if e.get("command")]
    if args.limit:
        test = test[:args.limit]

    rows, started = [], time.time()
    print("%s on %d items via %s" % (args.model, len(test), URL), flush=True)
    for i, e in enumerate(test, 1):
        t0 = time.time()
        try:
            got = ask(args.model, e["platform"], e["text"])
            error = ""
        except Exception as exc:                      # a refusal or timeout is data, not a crash
            got, error = "", str(exc)[:120]
        exact = got == e["command"]
        rows.append({"platform": e["platform"], "text": e["text"], "want": e["command"],
                     "got": got, "exact": exact,
                     "normalised": exact or same(got, e["command"]),
                     "right_program": first_token(got) == first_token(e["command"]),
                     "seconds": round(time.time() - t0, 1), "error": error})
        print("  %2d/%d [%s] %-36s %s" % (i, len(test), "x" if exact else " ",
                                          e["text"][:36], (got or error)[:64]), flush=True)

    n = len(rows)
    score = {k: sum(r[k] for r in rows) / n for k in ("exact", "normalised", "right_program")}
    print("\n%s on %d items: exact %.1f%%  normalised %.1f%%  right program %.1f%%  (%.0f s total)"
          % (args.model, n, 100 * score["exact"], 100 * score["normalised"],
             100 * score["right_program"], time.time() - started))

    out = os.path.join(HERE, "data", "baseline_%s.json" % args.model.replace("/", "_"))
    with open(out, "w") as f:
        json.dump({"model": args.model, "n": n, "endpoint": "$SMMOL_LLM_URL/v1/chat/completions", "score": score,
                   "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
                   "errors": sum(1 for r in rows if r["error"]), "rows": rows}, f, indent=1)
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
