# Phase 1: generated recall world

**Status: complete.** Implemented and validated 2026-09-17. No model training was started.

Phase 1 adds a deterministic byte-level associative-recall world in [`world.py`](world.py) and 73 focused tests in
[`tests/test_world.py`](tests/test_world.py). The world generates fresh episodes from stable BLAKE2-derived seeds,
scores only the answer byte, and reconstructs the answer independently from the bytes before it.

## Construction

- 36 one-byte keys, 33 one-byte values, and 23 filler bytes are printable and mutually disjoint. Filler also excludes
  `=`, `;`, `?`, and newline, so it cannot silently parse as a pair.
- Ordinary episodes have distinct keys. The opt-in overwrite variant repeats only the queried key, gives the stale
  write a different value, and makes the newest write authoritative.
- The exact gap is `query_at - queried_equals_at - 1`. The generator and independent parser use the same explicit
  definition and the tests recount it from raw bytes.
- Every requested pair-count x gap cell is constructible. Pair count and gap are independently selectable, but query
  position is not independent at short gaps because bytes cannot be deleted. For 32 pairs, gap 8 permits only
  ordinals 30/31, gap 32 permits 24-31, and gaps 128/512 permit all 32 positions. Sampling is uniform over the
  feasible positions, and the episode/evaluation metadata exposes that set.
- The primary evaluation creates 1,000 paired episodes for every one of 20 cells, resets state per episode, freezes
  learning, rejects registered model-parameter mutation during a frozen run, and fingerprints the ordered episode
  set so compared arms can prove that they saw identical bytes.
- Carry and reset-at-random protocols report accuracy against episodes-since-reset. They deliberately omit Wilson
  intervals because carried state makes episode outcomes dependent.

## Lazy baselines

These are deterministic results on the Phase 1 evaluation seed, 1,000 episodes per cell. `best position` is the best
fixed pair-ordinal guess, which makes short-gap position leakage visible.

| pairs | gap | feasible positions | uniform | most common | first | last | best position |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 2 | .030 | .516 | .516 | .514 | .516 |
| 2 | 32 | 2 | .031 | .528 | .528 | .504 | .528 |
| 2 | 128 | 2 | .022 | .520 | .520 | .499 | .520 |
| 2 | 512 | 2 | .039 | .532 | .532 | .499 | .532 |
| 4 | 8 | 2 | .025 | .092 | .026 | .497 | .529 |
| 4 | 32 | 4 | .030 | .293 | .272 | .261 | .294 |
| 4 | 128 | 4 | .031 | .279 | .262 | .261 | .308 |
| 4 | 512 | 4 | .033 | .299 | .273 | .269 | .284 |
| 8 | 8 | 2 | .028 | .152 | .030 | .531 | .531 |
| 8 | 32 | 8 | .027 | .215 | .159 | .149 | .164 |
| 8 | 128 | 8 | .032 | .194 | .132 | .156 | .162 |
| 8 | 512 | 8 | .025 | .193 | .150 | .137 | .164 |
| 16 | 8 | 2 | .033 | .116 | .031 | .521 | .521 |
| 16 | 32 | 8 | .032 | .130 | .031 | .155 | .161 |
| 16 | 128 | 16 | .029 | .144 | .083 | .089 | .110 |
| 16 | 512 | 16 | .028 | .129 | .100 | .088 | .106 |
| 32 | 8 | 2 | .040 | .091 | .020 | .474 | .558 |
| 32 | 32 | 8 | .040 | .104 | .027 | .142 | .165 |
| 32 | 128 | 32 | .035 | .118 | .060 | .063 | .073 |
| 32 | 512 | 32 | .031 | .102 | .049 | .070 | .080 |

The table shows why the positional controls matter. At gap 8, the best fixed-position guess reaches 51-56% once
there are at least four pairs. At `(pairs=32, gap=128)`, where every position is feasible, it falls to 7.3%.

## Gates and evidence

| Gate | Result |
|---|---|
| answer mechanically determined from prefix | pass |
| exact gaps in all 20 cells | pass |
| pair count and gap independently selectable | pass |
| distinct ordinary keys and parse-safe filler | pass |
| seeded determinism and fresh episodes | pass |
| lazy controls use identical paired episodes | pass |
| 1,000-episode evaluation and frozen-parameter guard | pass |
| answer accuracy, NLL, and rank plumbing | pass |
| reset/carry/reset-at-random protocols | pass |
| overwrite off by default and newest-write-wins | pass |
| Wilson Phase 2 boundary | 919/1000 clears 0.90; 918/1000 does not |

Validation commands:

```sh
PYTHONPATH=smRTS_01 python3 -m unittest discover -s smRTS_01/tests -p 'test_world.py' -v
PYTHONPATH=smRTS_01 python3 smRTS_01/world.py
PYTHONPATH=smRTS_01 python3 -m unittest discover -s smRTS_01/tests -p 'test_*.py' -v
python3 -m py_compile smRTS_01/world.py smRTS_01/tests/test_world.py
```

Observed results:

- Phase 1 focused suite: **73 tests passed**.
- World self-check: **5,000 episodes passed** across the full grid and overwrite variant.
- Phase 0 + Phase 1 focused suite: **84 tests passed**.
- Wilson lower bounds: 919/1000 = **0.90044**; 918/1000 = **0.89935**.

## Phase 2 handoff

Use reset-per-episode as the primary statistically independent endpoint and compare every model against the lazy
rows from the same episode-set hash. Treat gap-8 scores as position-confounded, not pure associative recall. The
primary `(pairs=16, gap=128)` gate has every query position feasible and avoids that short-gap shortcut.
