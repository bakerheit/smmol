"""Does smMATH_LANGUAGE_001 know when it's unsure?

It writes one character at a time, and each character has a probability. This checks whether those numbers
line up with being right, on the hand-written test messages, and suggests a floor to act on.

    python3 confidence.py
"""
import json
import os
import statistics

from read import Reader
from score import grade

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKETS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.01)]


def main():
    with open(os.path.join(HERE, "test.json")) as f:
        test = json.load(f)
    reader = Reader(os.path.join(HERE, "out", "reader.pt"))
    rows = []
    for e in test:
        problems, sure, weakest = reader.read_with_confidence(e["text"])
        every, _ = grade([p["expression"] for p in problems], e["expected"])
        rows.append({"text": e["text"], "sure": sure, "weakest": weakest, "right": every})

    print("%d messages | right %d | average confidence %.3f" % (len(rows), sum(r["right"] for r in rows),
                                                                statistics.mean(r["sure"] for r in rows)))
    for name, key in (("average across the answer", "sure"), ("shakiest single character", "weakest")):
        print("\nby %s:" % name)
        print("| confidence | messages | right |")
        print("|---|---|---|")
        for low, high in BUCKETS:
            group = [r for r in rows if low <= r[key] < high]
            if group:
                print("| %.2f to %.2f | %d | %.0f%% |" % (low, min(high, 1.0), len(group),
                                                          100 * sum(r["right"] for r in group) / len(group)))
    right = [r for r in rows if r["right"]]
    wrong = [r for r in rows if not r["right"]]
    print("\nwhen it's right:  average %.3f, shakiest %.3f" % (statistics.mean(r["sure"] for r in right),
                                                               statistics.mean(r["weakest"] for r in right)))
    print("when it's wrong:  average %.3f, shakiest %.3f" % (statistics.mean(r["sure"] for r in wrong),
                                                             statistics.mean(r["weakest"] for r in wrong)))
    best = None
    for floor in [i / 100 for i in range(50, 100)]:
        flagged = [r for r in rows if r["weakest"] < floor]
        caught = sum(1 for r in flagged if not r["right"])
        if flagged and caught:
            cost = len(flagged) - caught  # right answers we'd needlessly double-check
            if best is None or (caught / max(1, len(wrong)), -cost) > best[1]:
                best = (floor, (caught / max(1, len(wrong)), -cost), caught, cost)
    if best:
        floor, _, caught, cost = best
        print("\na floor on the shakiest character at %.2f flags %d of the %d wrong answers, and double-checks %d right ones"
              % (floor, caught, len(wrong), cost))
    with open(os.path.join(HERE, "out", "confidence.json"), "w") as f:
        json.dump({"rows": rows, "floor": best[0] if best else None}, f, indent=1)
    print("\nthe ones it got wrong, least sure first:")
    for r in sorted(wrong, key=lambda r: r["weakest"])[:10]:
        print("  weakest %.3f  average %.3f  %s" % (r["weakest"], r["sure"], r["text"][:60]))


if __name__ == "__main__":
    main()
