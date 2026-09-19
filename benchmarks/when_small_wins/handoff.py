#!/usr/bin/env python3
"""Phase 4: small first, big when the small one is unsure.

    python3 handoff.py                 # all three jobs
    python3 handoff.py router          # one job
    python3 handoff.py mathlang 2000   # and a bigger threshold-fitting set

The 2x2 in join.py showed the maths reader and Ministral 8B are wrong on *zero* of
the same 40 items — the union is perfect. This asks whether each small model's own
confidence is a good enough signal to exploit that, on all three jobs.

The honest part is the threshold. Picking it on the hand-written messages would be
choosing on the set we then report, which is the exact mistake this repo keeps
finding in its own trainers. So it is chosen on freshly generated held-out messages
by Youden's J, using only the small model's own right/wrong labels — the 8B is never
consulted while choosing — then frozen and applied.

CPU only. Nothing under any model's out/ is written.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Measured per-item latency, from results.md.
LLM_MS = {"mathlang": 3130.0, "router": 1990.0, "cli": 1940.0}
SMALL_MS = {"mathlang": 28.9, "router": 2.04, "cli": 47.1}


def project(name):
    path = os.path.join(ROOT, "models", name)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


# --- one adapter per job: how to run it, and what counts as right -------------------

def mathlang_job():
    path = project("smMATH_LANGUAGE_001")
    import data as d
    from read import Reader
    from score import grade
    reader = Reader(os.path.join(path, "out", "reader.pt"))

    def run(example):
        problems, sure, weakest = reader.read_with_confidence(example["text"])
        # Generated examples carry `target` as a string; test.json carries `expected`
        # as a list. Both mean the same thing.
        if "expected" in example:
            expected = list(example["expected"])
        else:
            expected = [p["expression"] for p in d.parse_target(example["target"])]
        every, _ = grade([p["expression"] for p in problems], expected)
        return {"sure": sure, "weakest": weakest, "right": bool(every)}

    return d.generate, run, "smMATH_LANGUAGE_001/out/confidence.json"


def router_job():
    path = project("smROUTER_01")
    import data as d
    from route import Router
    router = Router(os.path.join(path, "out", "router.pt"))

    def run(example):
        out = router.route(example["text"], example.get("prev", ""))
        right = (out["intent"] == example["intent"] and out["tool"] == example["tool"]
                 and out["ask_first"] == example["ask_first"])
        # All three heads must be right, so the binding constraint is the least sure
        # of them. For ask-first, confidence is the distance from the 0.5 boundary.
        ask = max(out["ask_p"], 1 - out["ask_p"])
        weakest = min(out["intent_p"], out["tool_p"], ask)
        return {"sure": (out["intent_p"] + out["tool_p"] + ask) / 3, "weakest": weakest, "right": right}

    return d.generate, run, None


def cli_job():
    path = project("smTOOLS_COMPUTER_CLI_01")
    import data as d
    from cli import Commander
    model = Commander(os.path.join(path, "out", "cli.pt"))

    def run(example):
        out = model.command(example["text"], example.get("platform", "macos"))
        want, _ = d.split_target(example["target"]) if "target" in example else (example.get("command", ""), "")
        return {"sure": out["sure"], "weakest": out["weakest"], "right": out["command"].strip() == want.strip()}

    return d.generate, run, None


JOBS = {"mathlang": mathlang_job, "router": router_job, "cli": cli_job}


# --- the method ---------------------------------------------------------------------

def auc(rows, key):
    """Probability a right item scores above a wrong one. Ties count a half."""
    right = [r[key] for r in rows if r["right"]]
    wrong = [r[key] for r in rows if not r["right"]]
    if not right or not wrong:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in right for b in wrong)
    return wins / (len(right) * len(wrong))


def pick_threshold(rows, key):
    """Youden's J on generated data, from the small model's own labels alone."""
    wrong_total = sum(1 for r in rows if not r["right"])
    right_total = sum(1 for r in rows if r["right"])
    if not wrong_total or not right_total:
        return None, None
    best, chosen = -2.0, 0.0
    for candidate in sorted({round(r[key], 4) for r in rows}):
        flagged_wrong = sum(1 for r in rows if not r["right"] and r[key] < candidate)
        flagged_right = sum(1 for r in rows if r["right"] and r[key] < candidate)
        j = flagged_wrong / wrong_total - flagged_right / right_total
        if j > best:
            best, chosen = j, candidate
    return chosen, best


def hand_written(job, run):
    """The reported set, with each item's stored 8B verdict from items.jsonl."""
    with open(os.path.join(HERE, "data", "items.jsonl")) as f:
        items = [json.loads(line) for line in f]
    items = [r for r in items if r["job"] == job and r["llm_right"] is not None]
    if job == "cli":
        items = [r for r in items if r["kind"] == "command"]

    path = project({"mathlang": "smMATH_LANGUAGE_001", "router": "smROUTER_01",
                    "cli": "smTOOLS_COMPUTER_CLI_01"}[job])
    with open(os.path.join(path, "test.json")) as f:
        test = {(e.get("prev", ""), e["text"]): e for e in json.load(f)}

    rows = []
    for r in items:
        example = test.get((r.get("prev", ""), r["text"]))
        if example is None:
            continue
        if job == "cli":
            example = dict(example, target=example.get("command", ""))
        scored = run(example)
        rows.append({**scored, "llm_right": bool(r["llm_right"]), "text": r["text"]})
    return rows


def one_job(job, n):
    generate, run, _ = JOBS[job]()
    print("\n" + "=" * 66)
    print("%s — fitting the threshold on %d generated messages (seed 777)" % (job, n), flush=True)

    gen = [run(e) for e in generate(n, seed=777)]
    gen_acc = sum(r["right"] for r in gen) / len(gen)
    aucs = {k: auc(gen, k) for k in ("sure", "weakest")}
    print("  generated held-out: small model right %.1f%%   AUC sure %s weakest %s"
          % (100 * gen_acc,
             "%.3f" % aucs["sure"] if aucs["sure"] else "n/a",
             "%.3f" % aucs["weakest"] if aucs["weakest"] else "n/a"))

    usable = {k: v for k, v in aucs.items() if v is not None}
    if not usable:
        print("  the model is right on every generated message — no signal to fit. skipped.")
        return None
    key = max(usable, key=usable.get)
    threshold, j = pick_threshold(gen, key)
    print("  signal: %s   threshold: %.4f   (Youden's J %.3f)" % (key, threshold, j))

    hand = hand_written(job, run)
    if not hand:
        print("  no committed 8B answers for this job. skipped.")
        return None
    total = len(hand)
    handed = [r for r in hand if r[key] < threshold]
    kept = [r for r in hand if r[key] >= threshold]

    small = sum(r["right"] for r in hand) / total
    llm = sum(r["llm_right"] for r in hand) / total
    combined = (sum(r["right"] for r in kept) + sum(r["llm_right"] for r in handed)) / total
    ceiling = sum(r["right"] or r["llm_right"] for r in hand) / total
    rate = len(handed) / total
    ms = (1 - rate) * SMALL_MS[job] + rate * LLM_MS[job]

    print("\n  on the %d reported messages" % total)
    print("    small alone        %5.1f%%  %8.1f ms" % (100 * small, SMALL_MS[job]))
    print("    Ministral alone    %5.1f%%  %8.1f ms" % (100 * llm, LLM_MS[job]))
    print("    hand-off           %5.1f%%  %8.1f ms   (%d of %d, %.0f%% handed off)"
          % (100 * combined, ms, len(handed), total, 100 * rate))
    print("    oracle ceiling     %5.1f%%" % (100 * ceiling))
    beats = combined > max(small, llm)
    print("    beats both alone:  %s" % ("YES" if beats else "no"))

    return {"job": job, "signal": key, "threshold": threshold,
            "generated": {"n": len(gen), "accuracy": gen_acc, "auc": aucs},
            "n": total, "small_alone": small, "llm_alone": llm, "handoff": combined,
            "handoff_rate": rate, "oracle_ceiling": ceiling, "avg_ms": ms, "beats_both": beats}


def run_isolated(job, n):
    """Each project ships its own data.py / model.py / score.py, so importing two of
    them into one process makes the second resolve to the first's modules. Running
    each job in a fresh interpreter is the reliable fix; sys.modules surgery is not.
    """
    import subprocess
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), job, str(n), "--one"],
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr)
        return None
    path = os.path.join(HERE, "data", "handoff_%s.json" % job)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    args = [a for a in sys.argv[1:]]
    jobs = [a for a in args if a in JOBS] or list(JOBS)
    n = next((int(a) for a in args if a.isdigit()), 1200)

    if "--one" in args:
        result = one_job(jobs[0], n)
        if result:
            with open(os.path.join(HERE, "data", "handoff_%s.json" % jobs[0]), "w") as f:
                json.dump(result, f, indent=1)
        return

    results = [r for r in (run_isolated(job, n) for job in jobs) if r]

    print("\n" + "=" * 66)
    print("%-10s %9s %9s %9s %9s %8s" % ("job", "small", "8B", "hand-off", "ceiling", "rate"))
    print("-" * 66)
    for r in results:
        print("%-10s %8.1f%% %8.1f%% %8.1f%%%s %8.1f%% %7.0f%%"
              % (r["job"], 100 * r["small_alone"], 100 * r["llm_alone"], 100 * r["handoff"],
                 "*" if r["beats_both"] else " ", 100 * r["oracle_ceiling"], 100 * r["handoff_rate"]))
    print("\n* beats both single models")
    won = [r["job"] for r in results if r["beats_both"]]
    print("beats both on %d of %d jobs: %s" % (len(won), len(results), ", ".join(won) or "none"))
    print("F6 (small-model confidence is not usable for routing): %s"
          % ("no" if won else "FIRES — it did not help on any job"))

    out = os.path.join(HERE, "data", "handoff.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
