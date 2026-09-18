#!/usr/bin/env python3
"""Paired significance on the item table. No model runs, no network, instant.

    python3 significance.py

Two models scored on the same messages are a *paired* comparison, and the pair
structure is the whole point: items both get right and items both get wrong carry no
information about which is better. Only the disagreements do. McNemar's test uses
exactly those, and an exact two-sided sign test on them needs no approximation at
these sizes.

This existed nowhere in the repo until 2026-09-18. Every headline was a difference of
two percentages with no interval on it, on 40-93 items, in a project that had already
caught itself reading a one-message difference as a peak.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def sign_test(wins, losses):
    """Exact two-sided p that the discordant pairs split evenly."""
    n = wins + losses
    if not n:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_interval(wins, losses, n, z=1.96):
    """95% interval on the paired difference in proportions (wins-losses)/n."""
    if not n:
        return (0.0, 0.0)
    diff = (wins - losses) / n
    var = (wins + losses - (wins - losses) ** 2 / n) / (n * n)
    half = z * math.sqrt(max(var, 0.0))
    return (diff - half, diff + half)


def main():
    with open(os.path.join(HERE, "data", "items.jsonl")) as f:
        rows = [json.loads(line) for line in f]

    print("Paired comparison, small model vs prompted Ministral 8B.")
    print("Only the items where they disagree carry information.\n")
    print("%-10s %4s %7s %8s %7s %6s %9s  %s"
          % ("job", "n", "gap", "small+", "8B+", "disc", "p", "95% CI on the gap"))
    print("-" * 82)

    out = {}
    for job in ("router", "cli", "mathlang"):
        pool = [r for r in rows if r["job"] == job and r["llm_right"] is not None]
        if job == "cli":
            pool = [r for r in pool if r["kind"] == "command"]
        if not pool:
            continue
        n = len(pool)
        wins = sum(r["small_right"] and not r["llm_right"] for r in pool)
        losses = sum(not r["small_right"] and r["llm_right"] for r in pool)
        p = sign_test(wins, losses)
        lo, hi = paired_interval(wins, losses, n)
        print("%-10s %4d %+6.1f%% %8d %7d %6d %9.4f  [%+.1f, %+.1f]"
              % (job, n, 100 * (wins - losses) / n, wins, losses, wins + losses,
                 p, 100 * lo, 100 * hi))
        out[job] = {"n": n, "gap": (wins - losses) / n, "small_only": wins, "llm_only": losses,
                    "discordant": wins + losses, "p": p, "ci": [lo, hi],
                    "significant_at_05": p < 0.05}

    print("\nWhat this says about the claims in README.md:")
    r, c, m = out.get("router"), out.get("cli"), out.get("mathlang")
    if r:
        print("  router   — \"beats an 8B by 13 points\": p = %.3f, CI [%+.1f, %+.1f]. %s"
              % (r["p"], 100 * r["ci"][0], 100 * r["ci"][1],
                 "NOT significant at 0.05; the interval nearly touches zero." if not r["significant_at_05"] else "significant."))
    if m:
        print("  maths    — \"loses to it\": p = %.3f. Indistinguishable, not a loss." % m["p"])
    if c:
        print("  CLI      — \"+31.9 on exact command\": p = %.4f. The strongest result in the repo," % c["p"])
        print("             and the one that spent a day being hedged on other grounds.")

    with open(os.path.join(HERE, "data", "significance.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote data/significance.json")


if __name__ == "__main__":
    main()
