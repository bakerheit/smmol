#!/usr/bin/env python3
"""Rebuild the item-level scoreboard from files already committed.

    python3 join.py            # writes data/items.jsonl and prints the 2x2 per job

results.md reports aggregates. Every question worth asking about WHY a small model
wins one job and loses another is an item-level question, and the answer is already
on disk: each project ships its hand-written test set, its own mistakes list, and
the Ministral baseline's mistakes list. Nobody had joined them.

Nothing here runs a model. It reads and joins.

One wrinkle worth knowing, because it silently breaks a naive join: a miss on an
item whose right answer is "say nothing" is stored with the literal string
"(nothing)" in `want`, not an empty value. Treat that as truthy and the
reconstruction lands at 0.085 where the published number is 0.149.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

QUIET = "(nothing)"


def load(path):
    with open(os.path.join(ROOT, "models", path)) as f:
        return json.load(f)


def cli_rows():
    """smTOOLS_COMPUTER_CLI_01: plain request -> shell command, 5 platforms."""
    test = load("smTOOLS_COMPUTER_CLI_01/test.json")
    small = load("smTOOLS_COMPUTER_CLI_01/out/results.json")
    llm = load("smTOOLS_COMPUTER_CLI_01/out/llm_baseline.json")

    small_miss = {m["text"]: m for m in small["test_mistakes"]}
    llm_miss = {m["text"]: m for m in llm["mistakes"]}

    rows = []
    for item in test:
        text, want = item["text"], item.get("command") or ""
        rows.append({
            "job": "cli",
            "text": text,
            "platform": item.get("platform"),
            "want": want,
            "kind": "quiet" if not want else "command",
            "small_right": text not in small_miss,
            "llm_right": text not in llm_miss,
            "small_got": small_miss.get(text, {}).get("got"),
            "llm_got": llm_miss.get(text, {}).get("got"),
        })
    return rows


def router_rows():
    """smROUTER_01: message -> intent, tool, ask-first. 'Right' means all three."""
    test = load("smROUTER_01/test.json")
    small = load("smROUTER_01/out/results.json")
    llm = load("smROUTER_01/out/llm_baseline.json")

    def key(entry):
        return (entry.get("prev", ""), entry["text"])

    small_miss = {key(m): m for m in small["test_mistakes"]}
    llm_miss = {key(m): m for m in llm.get("mistakes", [])}

    rows = []
    for item in test:
        k = (item.get("prev", ""), item["text"])
        rows.append({
            "job": "router",
            "text": item["text"],
            "prev": item.get("prev", ""),
            "want": [item.get("intent"), item.get("tool"), item.get("ask_first")],
            "kind": "answer" if item.get("prev") else "message",
            "small_right": k not in small_miss,
            "llm_right": k not in llm_miss,
            "small_got": small_miss.get(k, {}).get("got"),
            "llm_got": llm_miss.get(k, {}).get("got"),
        })
    return rows


def mathlang_rows():
    """smMATH_LANGUAGE_001: message -> the arithmetic in it. 'Right' means every problem."""
    test = load("smMATH_LANGUAGE_001/test.json")
    small = load("smMATH_LANGUAGE_001/out/results.json")
    llm_path = os.path.join(ROOT, "models", "smMATH_LANGUAGE_001/out/llm_baseline.json")
    if os.path.exists(llm_path):
        with open(llm_path) as f:
            llm = json.load(f)
    else:
        llm = {}

    small_miss = {m["text"] for m in small.get("test_mistakes", [])}
    llm_miss = {m["text"] for m in llm.get("mistakes", [])}

    rows = []
    for item in test:
        want = item.get("expressions", item.get("want", []))
        rows.append({
            "job": "mathlang",
            "text": item["text"],
            "want": want,
            "kind": "quiet" if not want else "problem",
            "small_right": item["text"] not in small_miss,
            "llm_right": item["text"] not in llm_miss if llm else None,
        })
    return rows


def rate(rows, who, kind=None):
    pool = [r for r in rows if kind is None or r["kind"] == kind]
    pool = [r for r in pool if r[who] is not None]
    return (sum(r[who] for r in pool) / len(pool)) if pool else None


def main():
    rows = cli_rows() + router_rows() + mathlang_rows()
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    out = os.path.join(HERE, "data", "items.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cli = [r for r in rows if r["job"] == "cli"]
    router = [r for r in rows if r["job"] == "router"]
    mathlang = [r for r in rows if r["job"] == "mathlang"]

    print("wrote %s  (%d items)\n" % (os.path.relpath(out, ROOT), len(rows)))
    print("%-22s %8s %8s   %s" % ("job", "small", "8B", "published"))
    print("-" * 62)
    checks = [
        ("cli exact", rate(cli, "small_right", "command"), rate(cli, "llm_right", "command"), 0.468, 0.149),
        ("cli quiet", rate(cli, "small_right", "quiet"), rate(cli, "llm_right", "quiet"), 1.000, 0.500),
        ("router all-three", rate(router, "small_right"), rate(router, "llm_right"), 0.710, 0.581),
        ("mathlang every", rate(mathlang, "small_right"), rate(mathlang, "llm_right"), 0.775, 0.825),
    ]
    ok = True
    for name, s, l, want_s, want_l in checks:
        marks = []
        for got, want in ((s, want_s), (l, want_l)):
            if got is None:
                marks.append("  --  ")
            else:
                good = abs(got - want) < 0.002
                ok = ok and good
                marks.append("%.3f%s" % (got, " " if good else " !"))
        print("%-22s %8s %8s   %.3f / %.3f" % (name, marks[0], marks[1], want_s, want_l))

    print("\n2x2, items where both models were scored:")
    for job, pool in (("cli", cli), ("router", router), ("mathlang", mathlang)):
        both = [r for r in pool if r["llm_right"] is not None]
        if not both:
            print("  %-9s no 8B baseline committed" % job)
            continue
        bb = sum(r["small_right"] and r["llm_right"] for r in both)
        so = sum(r["small_right"] and not r["llm_right"] for r in both)
        lo = sum(not r["small_right"] and r["llm_right"] for r in both)
        nn = sum(not r["small_right"] and not r["llm_right"] for r in both)
        print("  %-9s n=%-4d both %-4d small-only %-4d 8B-only %-4d neither %d"
              % (job, len(both), bb, so, lo, nn))

    print("\ngate: %s" % ("reconstruction matches results.md" if ok else "MISMATCH — the join is wrong, fix before using"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
