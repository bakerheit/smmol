# when_small_wins

Working files for [docs/engineering/plans/when-small-wins.md](../../docs/engineering/plans/when-small-wins.md) —
the plan that asks, before any model is trained, whether a small specialised model will beat a prompted 8B.

Nothing here trains or runs a model. It joins and re-scores files already committed.

## Run it

```bash
python3 join.py                     # rebuild the item table, gated on reproducing results.md
python3 score_normalised.py         # phase 3a: re-score the CLI comparison three ways
python3 -m unittest discover -s tests
```

## Files

| File | What it is |
|---|---|
| `join.py` | One row per test item across all three jobs, from each project's `test.json`, `out/results.json` and `out/llm_baseline.json` |
| `normalise.py` | Shell-command normalisation: forgives spelling, never a behaviour change. Every rule is commented so it can be argued with |
| `score_normalised.py` | Phase 3a — strict, spelling-forgiven and most-permissive scoring, printing every forgiven item by name |
| `data/items.jsonl` | 186 items: `job`, `text`, `want`, `small_right`, `llm_right`, and the stored `got` strings |
| `data/normalised_cli.json` | The three CLI scores and the forgiven items |

## What phase 0 found

The reconstruction reproduces the published numbers exactly — CLI exact 0.468 / 0.149, CLI quiet 1.000 / 0.500,
router 0.710 / 0.581, math reader 0.775 — and the test keeps it that way.

**One trap.** A miss on an item whose right answer is "say nothing" stores the literal string `"(nothing)"` in
`want`. Treat that as truthy and Ministral's CLI exact reconstructs to 0.085 instead of 0.149.

**The 2×2, which the aggregates hide:**

| Job | n | Both right | Small only | 8B only | Neither |
|---|---:|---:|---:|---:|---:|
| cli | 53 | 6 | 22 | 4 | 21 |
| router | 93 | 41 | 25 | 13 | 14 |
| math reader | 40 | 24 | 7 | 9 | **0** |

The maths reader row is the interesting one: **neither model misses a single item.** The union is perfect, which is
what phase 4's "small first, big when unsure" rests on.

## What phase 3a found

The prediction failed. Normalising spelling moved the CLI lead 4.3 points, not the 22 predicted — only two of
Ministral's 40 command misses were spelling. But the two graders already in this repo disagree about *sign*:
exact command says the small model leads by 31.9, right program says the 8B leads by 14.9.

Full write-up in [results.md](../../docs/engineering/research/results.md).
