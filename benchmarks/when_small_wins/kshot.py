#!/usr/bin/env python3
"""Phase 3b, router arm: is it small-vs-big, or supervised-vs-prompted?

    SMMOL_LLM_URL=http://<pc>:8081 python3 kshot.py            # k = 0, 8, 32
    SMMOL_LLM_URL=http://<pc>:8081 python3 kshot.py --k 32     # one arm

Every comparison in this repo pits tens of thousands of labelled examples against a
single zero-shot prompt. That is not a test of model size. This gives the 8B a
comparable, if much smaller, dose of the same supervision: k examples retrieved by
BM25 from the router's own generator, prepended to the conversation.

Only the `messages` array changes. The system prompt, the JSON schema, the model, the
temperature and the grading all come from smROUTER_01/baseline_llm.py unchanged, so
k=0 must reproduce the committed 58.1% and the arms are comparable to each other.

F5, pre-registered in docs/engineering/plans/when-small-wins.md: if k=32 reaches 68%,
the router's 12.9-point win does not survive giving the 8B examples, and the effect
was supervised-vs-prompted rather than small-vs-big.

THE LEAK TEST IS A HARD GATE. Examples are drawn from the generator at a seed used
nowhere else; any retrieved example whose text matches a test message voids the run.
Contamination is what sank phases 1-2 of this plan, and it would be invisible in the
output if it were not checked.
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ROUTER = os.path.join(ROOT, "models", "smROUTER_01")
sys.path.insert(0, ROUTER)

import baseline_llm as B     # noqa: E402  SYSTEM, SCHEMA and URL, unchanged
import data as router_data   # noqa: E402
from train import score      # noqa: E402

WORD = re.compile(r"[a-z0-9']+")


def norm(text):
    return " ".join(WORD.findall((text or "").lower()))


def tokens(text):
    return WORD.findall((text or "").lower())


class BM25:
    """Plain BM25 over the generator pool. No dependencies, and the pool is small."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def top(self, query, k, skip=frozenset()):
        q = tokens(query)
        scored = []
        for i, tf in enumerate(self.tf):
            if i in skip:
                continue
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]


def as_user(prev, text):
    """The exact user-message shape baseline_llm.py uses."""
    return "The assistant's last message: %s\nThe person's message: %s" % (prev or "(none)", text)


def as_answer(e):
    return json.dumps({"intent": e["intent"], "tool": e["tool"], "ask_first": bool(e["ask_first"])})


def ask(prev, text, shots, timeout=300):
    messages = [{"role": "system", "content": B.SYSTEM}]
    for e in shots:
        messages.append({"role": "user", "content": as_user(e.get("prev", ""), e["text"])})
        messages.append({"role": "assistant", "content": as_answer(e)})
    messages.append({"role": "user", "content": as_user(prev, text)})
    body = {"model": "ministral-8b", "temperature": 0, "max_tokens": 60, "stream": False,
            "response_format": {"type": "json_object", "schema": B.SCHEMA}, "messages": messages}
    request = urllib.request.Request(B.URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = json.loads(response.read())["choices"][0]["message"]["content"]
    return json.loads(content[content.find("{"):content.rfind("}") + 1])


def sign_test(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    k = max(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, action="append", help="repeatable; default 0, 8, 32")
    ap.add_argument("--pool", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1, help="generator seed for the retrieval pool")
    args = ap.parse_args()
    ks = args.k or [0, 8, 32]

    with open(os.path.join(ROUTER, "test.json")) as f:
        test = json.load(f)
    print("router k-shot: %d test messages, k in %s, endpoint $SMMOL_LLM_URL" % (len(test), ks), flush=True)

    print("building the retrieval pool: generate(%d, seed=%d)..." % (args.pool, args.seed), flush=True)
    pool = router_data.generate(args.pool, seed=args.seed)

    # --- the hard gate -------------------------------------------------------------
    test_norm = {norm(e["text"]) for e in test}
    before = len(pool)
    pool = [e for e in pool if norm(e["text"]) not in test_norm]
    dropped = before - len(pool)
    print("leak gate: dropped %d of %d pool examples whose text matches a test message" % (dropped, before))

    bm = BM25([e["text"] for e in pool])

    arms = {}
    for k in ks:
        guesses, timings, leaked = [], [], 0
        print("\n--- k = %d ---" % k, flush=True)
        for i, e in enumerate(test, 1):
            shots = [pool[j] for j in bm.top(e["text"], k)] if k else []
            for s in shots:                                  # belt and braces, per item
                if norm(s["text"]) in test_norm:
                    leaked += 1
            started = time.time()
            try:
                g = ask(e.get("prev", ""), e["text"], shots)
            except Exception as exc:                          # a bad reply is wrong, not a crash
                g = {"intent": "?", "tool": "?", "ask_first": None, "error": str(exc)[:120]}
            timings.append(time.time() - started)
            guesses.append(g)
            if i % 10 == 0 or i == len(test):
                print("  %d/%d  %.1fs/item" % (i, len(test), statistics.mean(timings)), flush=True)

        if leaked:
            raise SystemExit("LEAK: %d retrieved examples matched a test message. Run void." % leaked)

        result, mistakes = score(test, guesses)
        right = [not any(m["text"] == e["text"] and m.get("prev", "") == e.get("prev", "") for m in mistakes)
                 for e in test]
        arms[k] = {"score": result, "right": right, "median_seconds": round(statistics.median(timings), 2),
                   "mistakes": mistakes}
        print("  k=%-2d all-three-right %.3f  (intent %.3f tool %.3f ask %.3f)  median %.2fs"
              % (k, result["all"], result["intent"], result["tool"], result["ask_first"],
                 arms[k]["median_seconds"]), flush=True)

    # --- report ---------------------------------------------------------------------
    small = json.load(open(os.path.join(ROUTER, "out", "results.json")))
    small_miss = {(m.get("prev", ""), m["text"]) for m in small["test_mistakes"]}
    small_right = [(e.get("prev", ""), e["text"]) not in small_miss for e in test]
    n = len(test)

    print("\n" + "=" * 70)
    print("%-6s %10s %10s %10s %10s" % ("k", "all-3", "intent", "tool", "ask"))
    print("-" * 70)
    for k in ks:
        r = arms[k]["score"]
        print("%-6d %9.1f%% %9.1f%% %9.1f%% %9.1f%%"
              % (k, 100 * r["all"], 100 * r["intent"], 100 * r["tool"], 100 * r["ask_first"]))
    print("%-6s %9.1f%%" % ("small", 100 * sum(small_right) / n))

    print("\npaired against smROUTER_01 v3, same %d messages:" % n)
    for k in ks:
        llm_right = arms[k]["right"]
        wins = sum(s and not l for s, l in zip(small_right, llm_right))
        losses = sum(l and not s for s, l in zip(small_right, llm_right))
        p = sign_test(wins, losses)
        print("  k=%-2d  gap %+5.1f  small-only %2d  8B-only %2d  McNemar p = %.4f  %s"
              % (k, 100 * (wins - losses) / n, wins, losses, p,
                 "significant" if p < 0.05 else "not significant at 0.05"))

    base = arms.get(0, {}).get("score", {}).get("all")
    if base is not None:
        ok = abs(base - 0.581) <= 0.02
        print("\ngate: k=0 reproduces the committed 58.1%%: %.1f%% -> %s"
              % (100 * base, "yes" if ok else "NO, the arms are not comparable"))
    if 32 in arms:
        got = arms[32]["score"]["all"]
        print("F5 (k=32 reaches 68%%): %.1f%% -> %s"
              % (100 * got, "FIRES — the win does not survive supervision" if got >= 0.68 else "does not fire"))
        if 0 in arms:
            print("lift from 32 examples: %+.1f points" % (100 * (got - arms[0]["score"]["all"])))
    if len(ks) >= 3 and all(k in arms for k in (0, 8, 32)):
        a, b, c = (arms[k]["score"]["all"] for k in (0, 8, 32))
        print("P11 (monotone dose-response k=0 < k=8 < k=32): %s"
              % ("holds" if a <= b <= c else "DOES NOT HOLD — the lift may not be supervision"))

    out = os.path.join(HERE, "data", "kshot_router.json")
    with open(out, "w") as f:
        json.dump({"n": n, "ks": ks, "pool": len(pool), "pool_seed": args.seed, "dropped_by_leak_gate": dropped,
                   "small_all": sum(small_right) / n,
                   "arms": {str(k): {"score": arms[k]["score"], "median_seconds": arms[k]["median_seconds"],
                                     "right": arms[k]["right"], "mistakes": arms[k]["mistakes"]} for k in ks}},
                  f, indent=1)
    print("\nwrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
