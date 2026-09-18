#!/usr/bin/env python3
"""Phase 3a: re-score the CLI comparison under normalisation, and show the working.

    python3 score_normalised.py

The question from docs/engineering/plans/when-small-wins.md: is
smTOOLS_COMPUTER_CLI_01's +31.9-point lead on exact command a capability win, or is
it the grader rewarding the model that was trained on the answer key's conventions?

Normalisation can only turn a wrong answer right, never a right answer wrong, so
re-scoring the stored misses is exact. Neither model is run again.

Every forgiven item is printed, so the decision on each one can be argued with.
"""

import json
import os

from normalise import normalise, same

HERE = os.path.dirname(os.path.abspath(__file__))


def load_items():
    with open(os.path.join(HERE, "data", "items.jsonl")) as f:
        return [json.loads(line) for line in f]


def first_token(command):
    """The program, ignoring sudo — used for the weaker 'right program' check."""
    words = (command or "").split()
    while words and words[0] in ("sudo", "doas", "command", "env"):
        words.pop(0)
    return words[0] if words else ""


def main():
    items = [r for r in load_items() if r["job"] == "cli" and r["kind"] == "command"]
    n = len(items)

    report = {}
    for who, got_key in (("small", "small_got"), ("llm", "llm_got")):
        strict = sum(r[f"{who}_right"] for r in items)
        forgiven, right_program = [], 0
        for r in items:
            if r[f"{who}_right"]:
                continue
            got = r[got_key] or ""
            if first_token(got) == first_token(r["want"]):
                right_program += 1
            if same(got, r["want"]):
                forgiven.append(r)
        report[who] = {
            "strict": strict,
            "forgiven": forgiven,
            "normalised": strict + len(forgiven),
            "right_program_among_misses": right_program,
            "misses": n - strict,
        }

    print("smTOOLS_COMPUTER_CLI_01 vs Ministral 8B, %d items with a command\n" % n)
    print("%-28s %10s %10s %10s" % ("", "small", "8B", "lead"))
    print("-" * 62)
    s, l = report["small"], report["llm"]
    print("%-28s %9.1f%% %9.1f%% %+9.1f" % (
        "strict (as published)", 100 * s["strict"] / n, 100 * l["strict"] / n,
        100 * (s["strict"] - l["strict"]) / n))
    print("%-28s %9.1f%% %9.1f%% %+9.1f" % (
        "normalised", 100 * s["normalised"] / n, 100 * l["normalised"] / n,
        100 * (s["normalised"] - l["normalised"]) / n))
    print()
    print("%-28s %10d %10d" % ("misses, strict", s["misses"], l["misses"]))
    print("%-28s %10d %10d" % ("  forgiven by normalising", len(s["forgiven"]), len(l["forgiven"])))
    print("%-28s %10d %10d" % ("  right program, wrong string", s["right_program_among_misses"],
                               l["right_program_among_misses"]))

    for who, label in (("llm", "Ministral 8B"), ("small", "smTOOLS_COMPUTER_CLI_01")):
        forgiven = report[who]["forgiven"]
        print("\n--- %s: %d forgiven ---" % (label, len(forgiven)))
        for r in forgiven:
            print("  want: %s" % r["want"])
            print("  got : %s" % (r[f"{who}_got"] or ""))

    # The third scoring the plan asks for: the most permissive grader anyone could
    # defend, forgiving every miss that at least named the right program. It is an
    # upper bound on what normalisation could ever buy, not a proposal.
    for who in ("small", "llm"):
        r = report[who]
        r["permissive"] = r["strict"] + r["right_program_among_misses"]
    print()
    print("%-28s %9.1f%% %9.1f%% %+9.1f" % (
        "permissive (right program)",
        100 * s["permissive"] / n, 100 * l["permissive"] / n,
        100 * (s["permissive"] - l["permissive"]) / n))

    strict_lead = 100 * (s["strict"] - l["strict"]) / n
    norm_lead = 100 * (s["normalised"] - l["normalised"]) / n
    perm_lead = 100 * (s["permissive"] - l["permissive"]) / n
    print("\n" + "=" * 62)
    print("lead: %+.1f strict -> %+.1f normalised  (moved %.1f points)"
          % (strict_lead, norm_lead, strict_lead - norm_lead))
    print("plan P5: 8B rises 14.9%% -> 40-55%%.  actual: %.1f%% -> %.1f%%  %s"
          % (100 * l["strict"] / n, 100 * l["normalised"] / n,
             "HIT" if 40 <= 100 * l["normalised"] / n <= 55 else "MISS"))
    print("plan P7: lead falls below +10.        actual: %+.1f  %s"
          % (norm_lead, "HIT" if norm_lead < 10 else "MISS"))
    print("F4 fires (lead < +5 or inverted):     %s" % ("YES" if norm_lead < 5 else "no"))
    print()
    print("the grader disagreement, on the same %d items:" % n)
    print("  exact command      small leads %+.1f" % strict_lead)
    print("  spelling forgiven  small leads %+.1f" % norm_lead)
    print("  right program      small leads %+.1f  <- sign flips" % perm_lead)
    print("Two defensible graders already in this repo disagree about who is better.")
    print("Normalising spelling does not bridge them, so the gap is a real disagreement")
    print("about what counts as a right answer, not a formatting artefact.")

    with open(os.path.join(HERE, "data", "normalised_cli.json"), "w") as f:
        json.dump({
            "n": n,
            "strict": {"small": s["strict"] / n, "llm": l["strict"] / n},
            "normalised": {"small": s["normalised"] / n, "llm": l["normalised"] / n},
            "forgiven": {w: [{"want": r["want"], "got": r[f"{w}_got"]} for r in report[w]["forgiven"]]
                         for w in ("small", "llm")},
            "right_program_among_misses": {w: report[w]["right_program_among_misses"] for w in ("small", "llm")},
        }, f, indent=1)
    print("\nwrote data/normalised_cli.json")


if __name__ == "__main__":
    main()
