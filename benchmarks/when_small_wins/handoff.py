#!/usr/bin/env python3
"""Phase 4: small first, big when the small one is unsure.

    python3 handoff.py            # ~1 min on the CPU, writes data/handoff_mathlang.json

The 2x2 from join.py says the maths reader and Ministral 8B are wrong on *zero* of
the same 40 items. The union is perfect, so something that knew which model to ask
would score 100%. The question is whether the small model's own confidence is a good
enough signal to get part of the way there.

The honest part is the threshold. Picking it on the 40 hand-written messages would be
choosing on the set we then report, which is the exact mistake this repo keeps
finding in its own trainers. So the threshold is chosen on freshly generated held-out
messages, by Youden's J on the right/wrong labels alone — Ministral is never consulted
while choosing — then frozen and applied to the hand-written set.

Nothing under any model's out/ is written, and no GPU is used: the reader runs on the
CPU at about 29 ms a message.
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MATHLANG = os.path.join(ROOT, "smMATH_LANGUAGE_001")
sys.path.insert(0, MATHLANG)

import data as mathlang_data          # noqa: E402
from read import Reader               # noqa: E402
from score import grade               # noqa: E402

SMALL_MS = 28.9   # measured, results.md
LLM_MS = 3130.0   # Ministral median, 3.13 s


def auc(rows, key):
    """Probability a right item is scored above a wrong one. Ties count a half."""
    right = [r[key] for r in rows if r["right"]]
    wrong = [r[key] for r in rows if not r["right"]]
    if not right or not wrong:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in right for b in wrong)
    return wins / (len(right) * len(wrong))


def pick_threshold(rows, key):
    """Youden's J on generated data. Uses only the small model's own right/wrong."""
    best, chosen = -2.0, 0.0
    for candidate in sorted({round(r[key], 4) for r in rows}):
        flagged_wrong = sum(1 for r in rows if not r["right"] and r[key] < candidate)
        flagged_right = sum(1 for r in rows if r["right"] and r[key] < candidate)
        wrong_total = sum(1 for r in rows if not r["right"])
        right_total = sum(1 for r in rows if r["right"])
        if not wrong_total or not right_total:
            continue
        j = flagged_wrong / wrong_total - flagged_right / right_total
        if j > best:
            best, chosen = j, candidate
    return chosen, best


def build_generated(reader, n, seed):
    rows = []
    for e in mathlang_data.generate(n, seed=seed):
        problems, sure, weakest = reader.read_with_confidence(e["text"])
        expected = [p["expression"] for p in mathlang_data.parse_target(e["target"])]
        every, _ = grade([p["expression"] for p in problems], expected)
        rows.append({"sure": sure, "weakest": weakest, "right": bool(every)})
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    reader = Reader(os.path.join(MATHLANG, "out", "reader.pt"))

    print("choosing the threshold on %d freshly generated messages (seed 777, "
          "not the training seed 0 or the val seed 1000)..." % n, flush=True)
    gen = build_generated(reader, n, seed=777)
    gen_acc = sum(r["right"] for r in gen) / len(gen)
    print("  generated held-out: %d messages, small model right %.1f%%" % (len(gen), 100 * gen_acc))
    for key in ("sure", "weakest"):
        print("  AUC(%-7s) on generated = %.3f" % (key, auc(gen, key)))

    key = max(("sure", "weakest"), key=lambda k: auc(gen, k))
    threshold, j = pick_threshold(gen, key)
    print("  chosen signal: %s   threshold: %.4f   (Youden's J = %.3f)" % (key, threshold, j))

    # Frozen from here. The hand-written set is scored, never consulted.
    with open(os.path.join(MATHLANG, "out", "confidence.json")) as f:
        hand = json.load(f)["rows"]
    with open(os.path.join(HERE, "data", "items.jsonl")) as f:
        items = [json.loads(line) for line in f if json.loads(line)["job"] == "mathlang"]
    llm_right = {r["text"]: r["llm_right"] for r in items}

    missing = [r["text"] for r in hand if r["text"] not in llm_right]
    if missing:
        raise SystemExit("no 8B answer stored for %d items, e.g. %r" % (len(missing), missing[0]))

    handed = [r for r in hand if r[key] < threshold]
    kept = [r for r in hand if r[key] >= threshold]
    combined = sum(r["right"] for r in kept) + sum(bool(llm_right[r["text"]]) for r in handed)

    total = len(hand)
    small_alone = sum(r["right"] for r in hand) / total
    llm_alone = sum(bool(llm_right[r["text"]]) for r in hand) / total
    combined_rate = combined / total
    rate = len(handed) / total
    latency = (1 - rate) * SMALL_MS + rate * LLM_MS
    ceiling = sum(r["right"] or bool(llm_right[r["text"]]) for r in hand) / total

    print("\nhand-written set, %d messages" % total)
    print("-" * 58)
    print("  small model alone        %5.1f%%   %7.1f ms" % (100 * small_alone, SMALL_MS))
    print("  Ministral 8B alone       %5.1f%%   %7.1f ms" % (100 * llm_alone, LLM_MS))
    print("  hand-off at %s<%.4f  %5.1f%%   %7.1f ms   (%d of %d handed off, %.0f%%)"
          % (key, threshold, 100 * combined_rate, latency, len(handed), total, 100 * rate))
    print("  oracle ceiling           %5.1f%%" % (100 * ceiling))

    print("\npre-registered P12: 82-88%% at a hand-off rate <= 30%%")
    hits = 0.82 <= combined_rate <= 0.88 and rate <= 0.30
    print("  actual: %.1f%% at %.0f%%  ->  %s" % (100 * combined_rate, 100 * rate, "HIT" if hits else "MISS"))
    beats_both = combined_rate > max(small_alone, llm_alone)
    print("  beats both single models: %s" % ("YES" if beats_both else "no"))
    print("  F6 (hand-off dead on this job): %s" % ("no" if beats_both else "FIRES"))

    out = {
        "job": "mathlang",
        "signal": key,
        "threshold": threshold,
        "chosen_on": {"n": len(gen), "seed": 777, "accuracy": gen_acc,
                      "auc_sure": auc(gen, "sure"), "auc_weakest": auc(gen, "weakest")},
        "hand_written": {
            "n": total, "small_alone": small_alone, "llm_alone": llm_alone,
            "handoff": combined_rate, "handoff_rate": rate,
            "oracle_ceiling": ceiling, "avg_ms": latency,
        },
    }
    with open(os.path.join(HERE, "data", "handoff_mathlang.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote data/handoff_mathlang.json")


if __name__ == "__main__":
    main()
