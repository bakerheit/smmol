# Phase 2: staged recall runs

**Status: paused. S1 complete; S2 interrupted by the user before its ceiling; S3 not started.**
Run on 2026-09-17. This file reports only what actually ran. Predictions, promotion gates and stop
rules were registered in [predictions.md](predictions.md) **before** any training run. Its original
predictions remain unchanged; a dated outcome block was appended after the runs, and this file
records the supporting evidence.

Phase 3 is out of scope and nothing here promotes anything into it. No configuration met a
promotion criterion, so no extra seeds, no `--overwrite`, and no full 5x4 grid were run.

| Stage | What | Outcome |
|---|---|---|
| S0 | tests + CPU smoke + MPS canary reference | complete; 104 tests pass, canary 615.4 matmuls/s |
| S1 | `leaky-tbptt`, decay frozen at half-life 128, 10 min | **complete**; P1 held, P2 falsified; did not promote |
| S2 | `fast-tbptt` uncapped calibration, 60 min ceiling | **interrupted at ~25.5 min (`user_stop`)**; no promotion decision |
| S3 | matched one-layer comparison | **not started** — its precondition was an S2 promotion |

## Environment and common configuration

| Item | Value |
|---|---|
| Machine | Apple M5, macOS 26.6.1, arm64 |
| PyTorch | 2.8.0, Python 3.9 |
| Device | MPS for every timed run; one job at a time, checked with `ps` immediately before each run |
| Recurrent layers | 1 (Phase 0 limits online exactness to one layer) |
| Matched state | 384 scalars/stream: width 384 for `leaky`/`gated`, `heads=4, key_dim=8, value_dim=12` for `fast`/`delta` |
| Optimizer | AdamW, lr 3e-4, weight decay 0.01, betas (0.9, 0.95), grad-norm clip 1.0 |
| Loss | next-byte cross-entropy, mean over batch and time. No `--repo-loss` |
| TBPTT | batch 64, window 256, state carried across windows and detached |
| Training episodes | `TRAIN_SEED=20260919`, pairs uniform over {2,4,8,16,32}, **gaps uniform over {32,128}** |
| Selection episodes | `SELECT_SEED=20260918`, 500 episodes at (16,128), used only to pick the best checkpoint |
| Reported episodes | `EVAL_SEED=20260917`, 1,000 paired episodes per cell, never used for selection |
| Seed | 1 (single seed; extra seeds are conditional on a promotion that has not yet occurred) |

The three episode universes are disjoint and `tests/test_scoring.py` asserts it. Every reported
number comes from the best checkpoint by the **selection** metric, evaluated on the **reported**
episodes, with learning frozen and parameter fingerprints checked bitwise before and after.

**Gap 8 is position-confounded** and is never read as associative recall: at `(16, 8)` only two
query ordinals are feasible, and the best fixed-position lazy guess scores 0.521.
**Gap 512 is extrapolation**, because the training mix is gaps 32 and 128 only.

---

## S1 — the `leaky` binding falsification

**Result: P1 held. A leaky integrator with a 128-byte half-life cannot bind, and the failure is not
a timescale failure.** `leaky`, one layer, every decay frozen at half-life 128, TBPTT, 10 minutes.

| | |
|---|---|
| Run | `out/S1-leaky-tbptt-frozen128/` |
| Cell / regime / device | `leaky` / `tbptt` / MPS |
| Parameters / state scalars | 345,216 / 384 per stream |
| Wall time | 600.2 s (`stop_reason: time`) |
| Training bytes | 121,454,592 at **202,371 bytes/s**, 7,413 optimizer steps, 1,058,299 episodes generated |
| Contention canary | 621.8 → 609.7 matmuls/s (2.0% drift); no other training or serving job found |
| Half-lives, initial → final | 128.001 → 128.001 for all 384 channels (frozen, and confirmed frozen) |
| Best checkpoint | step 4,959, selection accuracy 0.050, sha256 `7a3c20dfd1844233…` |
| Final-step selection | accuracy 0.028, NLL 3.549 |

### The reported grid

Best checkpoint, learning frozen, state reset per episode, 1,000 paired episodes per cell. The lazy
column is the best lazy arm on the **identical** episode set (same hash, same order).

| pairs, gap | | model acc | Wilson 95% | model NLL | model rank | best lazy arm |
|---|---|---:|---|---:|---:|---|
| 16, 8 | position-confounded | 0.081 | [0.066, 0.100] | 3.45 | 12.3 | `last` 0.521 |
| 16, 32 | | 0.086 | [0.070, 0.105] | 3.46 | 12.6 | `position_10` 0.161 |
| **16, 128** | **primary endpoint** | **0.053** | **[0.041, 0.069]** | 3.50 | 13.9 | `most_common` 0.144 |
| 16, 512 | extrapolation | 0.000 | [0.000, 0.004] | 3.90 | 17.5 | `most_common` 0.129 |
| 2, 128 | | 0.043 | [0.032, 0.057] | 3.52 | 16.1 | `most_common` 0.520 |
| 4, 128 | | 0.032 | [0.023, 0.045] | 3.51 | 15.2 | `position_2` 0.308 |
| 8, 128 | | 0.038 | [0.028, 0.052] | 3.51 | 15.2 | `most_common` 0.194 |
| 32, 128 | | 0.008 | [0.004, 0.016] | 3.59 | 15.6 | `most_common` 0.118 |

### What this says

**The model learned the format and stopped.** Training NLL fell from 5.79 to 2.71 nats, and the
answer-byte NLL fell from 5.72 to **3.50 nats — which is `ln 33 = 3.497` to two decimal places**.
That is exactly the cost of knowing the answer is one of the 33 value bytes and nothing about
*which* one. It reached that floor by about 2 minutes and sat on it for the remaining 8, over 121M
bytes. This is not an undertrained model; it is a converged one with nothing to converge to.

**It is beaten by lazy guessing in every single cell**, usually by a wide margin. At `(2, 128)` a
model scoring 4.3% is up against a "guess the first pair" heuristic scoring 52%.

**P1 held, with room to spare.** The registered test was a 95% Wilson *upper* bound below 0.30 at
`(16, 128)`; the observed upper bound is 0.069. The frozen-decay probe removes the timescale
objection entirely — the state provably retained a 128-byte memory, and the cell still could not
use it. `s = a⊙s + x` has no state-times-input interaction, so there is no mechanism to associate a
key with a value, and a longer memory of an unbindable sum does not help.

**P2 was falsified.** It predicted this same model would still clear a Wilson lower bound of 0.10
at `(2, 128)`, reasoning that two pairs need discrimination rather than binding. Observed: 0.043,
[0.032, 0.057]. The reasoning was wrong — answering `?k` even with two pairs on screen requires
knowing which of the two values attaches to `k`, which is binding. Two pairs makes the *guessing*
easy, not the *recall*.

predictions.md attached a consequence to a P2 failure: "if it fails even here, the trainer is
broken, not the cell." That claim is taken seriously rather than waved off, and it is not
self-servingly reinterpreted. The evidence that the trainer works is: training NLL fell 3.1 nats,
the answer-slot NLL fell 2.2 nats to exactly the value-alphabet floor, and the scoring position is
unit-tested against a hand computation and against an independent sequential implementation. But
the **decisive** test is S2: the same trainer, the same world, the same loop, with a cell that has
a keyed write. If `fast` learns, the trainer is fine and S1 is a statement about the leaky cell. If
`fast` also floors at `ln 33`, the finding is about the task or the trainer and no cell comparison
is worth running. That is the registered next stage, so no tuning was done here.

**Promotion decision.** predictions.md S1: promote-and-rewrite needs a Wilson lower bound ≥ 0.50
(observed 0.041 — no); record-and-continue needs a Wilson upper bound < 0.30 (observed 0.069 —
yes). **P1 held; `leaky` is answered for the binding question; continue to S2.**

### A registered deviation

predictions.md applies a futility kill rule to all Phase 2 arms: at 25% of budget, kill if the
answer NLL at `(16, 32)` has not fallen 0.3 nats below the lazy reference. S1 was run with that
check in **report-only** mode (`--futility-mode report`, recorded in `config.json`). It would have
fired: at 180.6 s the margin was **−0.055**, short of the required +0.3.

The reason for the deviation is that S1's entire deliverable *is* the negative result, and a
2.5-minute kill would have left "leaky cannot bind" indistinguishable from "leaky was not trained
long enough". Running the full 10 minutes spends 7.5 extra minutes to remove that ambiguity, and it
is what shows the NLL floor is a plateau rather than a stage. The deviation costs more compute, not
less, and it is recorded rather than quietly taken.

One further honesty note on that rule: predictions.md phrased the reference as "the best lazy
arm's NLL". A lazy arm emits a byte, not a distribution, so its NLL is undefined without a
smoothing convention that would dominate the number. `evaluate.futility_reference` substitutes two
stated distributional floors — uniform over the 33 value bytes (`ln 33`, used by the rule) and
uniform over the distinct values present in each episode (reported only). This is a change of
reference from the registered text and is flagged as one.

---

## S2 — uncapped `fast-tbptt` calibration (interrupted)

**Result: stopped by the user at ~25.5 minutes, before the registered 60-minute ceiling. No
promotion decision is available, and none is claimed.** `stop_reason: user_stop` is not one of the
registered stop rules; it is an external interruption and is recorded as one.

| | |
|---|---|
| Run | `out/S2-fast-tbptt-calibration/` |
| Cell / regime / device | `fast` (`heads=4, key_dim=8, value_dim=12`) / `tbptt` / MPS |
| Parameters / state scalars | 243,764 / 384 per stream |
| Registered budget | 60 min ceiling, early stop at 0.95 selection accuracy, futility disabled |
| Actual | killed at ~25.5 min (PID 45241 received TERM); **last logged row 1,442.5 s** |
| Training bytes at last log | 154,648,576 at **107,208 bytes/s**, 9,439 steps |
| Best checkpoint | step 6,251, wall 962.0 s, 102,416,384 bytes, selection accuracy 0.094, NLL 3.266, rank 8.75 |
| Checkpoint sha256 | `afb83b68fcb1a52db354adf90948e4db77e700fb1df3d577be3586e47a29f488` |
| Half-lives, initial → best checkpoint | min/median/p90/max `8.0 / 32.0 / 396.8 / 512.0` → `6.3 / 52.2 / 78.0 / 84.4` |

**What the artifacts are.** The trainer was killed before its finalisation block, so `results.json`,
`halflives.json` and `final.pt` were never written. `finalize_interrupted.py` rebuilt the first two
from the checkpoint and the log: it constructs no optimizer, takes no gradient, loads the
best-by-selection checkpoint, and runs the same frozen evaluation. `results.json` carries
`"finalized_by": "finalize_interrupted.py"` and an `interrupted` block listing what is permanently
missing. **`final.pt` was not fabricated** — the final-step weights were never saved and are gone,
so every number below is from the best-by-selection checkpoint at step 6,251, not from step 9,439.

### The reported grid (partial run, best checkpoint, learning frozen)

`1/pairs` is the rate expected from guessing uniformly among the values actually present in the
episode. It is the diagnostic column, and it is why this table does not say what a recall table
would say.

| pairs, gap | | model acc | Wilson 95% | NLL | rank | best lazy arm | 1/pairs |
|---|---|---:|---|---:|---:|---|---:|
| 16, 8 | position-confounded | 0.193 | [0.170, 0.219] | 2.75 | 5.7 | `last` 0.521 | 0.062 |
| 16, 32 | | 0.162 | [0.140, 0.186] | 2.92 | 6.7 | `position_10` 0.161 | 0.062 |
| **16, 128** | **primary endpoint** | **0.087** | **[0.071, 0.106]** | 3.29 | 9.0 | `most_common` 0.144 | 0.062 |
| 16, 512 | extrapolation | 0.032 | [0.023, 0.045] | 3.81 | 16.6 | `most_common` 0.129 | 0.062 |
| 2, 128 | | 0.256 | [0.230, 0.284] | 2.59 | 3.3 | `most_common` 0.520 | 0.500 |
| 4, 128 | | 0.138 | [0.118, 0.161] | 2.85 | 4.8 | `position_2` 0.308 | 0.250 |
| 8, 128 | | 0.104 | [0.087, 0.124] | 3.12 | 6.9 | `most_common` 0.194 | 0.125 |
| 32, 128 | | 0.077 | [0.062, 0.095] | 3.42 | 11.1 | `most_common` 0.118 | 0.031 |

The primary endpoint is **0.087, Wilson [0.071, 0.106]**, against a gate of a 0.90 lower bound. It
does not pass, and it is **below the best lazy arm in every cell of the table** — at `(16, 32)` it
merely ties a position-leakage arm. No promotion criterion was met, so S3 was not started.

### This is set membership, not binding

`fast` is doing something `leaky` was not: its answer NLL reaches 3.29 at `(16, 128)`, below the
value-alphabet floor of `ln 33 = 3.497`, and its rank falls to 9.0. On the same episodes, uniform
over the distinct values actually present in each episode costs 2.546 nats. So the model has moved
roughly 40% of the way from "the answer is some value byte" to "the answer is one of the values in
this episode", and no further. Rank tracks episode content rather than the queried key: 3.3 at 2
pairs, 6.9 at 8, 9.0 at 16, 11.1 at 32 — close to half the number of values on screen, which is what
narrowing to a set and then guessing inside it produces. Accuracy also falls monotonically with gap
(0.193 → 0.162 → 0.087 → 0.032) while episode content is held fixed, which is recency, not recall.

**The direct test.** Decomposing the primary cell's top-1 prediction over 400 `(16, 128)` episodes
separates the two abilities cleanly:

| | |
|---|---:|
| top-1 is *some* value written in this episode | **0.995** |
| top-1 is *the correct* value | 0.072 |
| correct **given** it was in the set | **0.073** (chance among 16 is 0.0625) |

The model has essentially solved set membership — it almost never proposes a byte that is not one
of the 16 values on screen — and has learned almost nothing about which one goes with the queried
key. That is not a partial recall result; it is a different capability that happens to score above
the uniform floor.

**The architectural reason.** A one-layer `FastCell` writes

```text
S_t = a·S_{t-1} + k(x_t) ⊗ v(x_t)
```

— both factors are projections of the **same byte** `x_t` (`cells.py:146-150`). The recall world
writes a pair as `k=v;`, so the key byte and its value byte are at *different* timesteps. At the
query the model presents `q(K)` for the queried key byte `K`; the write whose key matches is the one
made *when `K` itself was read*, and it returns `v(K)` — a fixed function of the key byte, identical
in every episode and carrying no information about that episode's binding. What the state can
usefully return is a decayed blend of the `v(·)` projections of the bytes that recently went past,
which is exactly a set-membership sketch. **One layer of this cell cannot form a key-to-next-value
association at all**, whatever the training budget, and the measured metrics are the signature of
that.

This is a property of the cell as specified in Phase 0, not of the task: an attention ceiling can
attend from the query to the key position and read the byte beside it. It also applies to `delta`,
which inherits `FastCell.write_parts` and so writes from the same byte, and to `leaky` and `gated`,
whose writes are likewise functions of `x_t` alone. S1 and S2 are therefore consistent: two cells,
neither of which has a mechanism for delayed binding, both floor out in the way their mechanism
predicts.

**This is a Phase 0 / design follow-up, and it is explicitly not an authorised fix.** Making the
write bind a *past* key to the *current* value — a delayed or contextual key, e.g. keying on state
rather than on `x_t` — changes the recurrence and therefore changes the trace. It would need its
own derivation and its own Phase 0 gradient gate before it could be called an online-trace cell, on
exactly the same terms the plan already applies to `delta`. Nothing was changed in `cells.py`, and
no such variant was built, trained or evaluated in this phase.

### Predictions touched

- **P3** (`fast-tbptt` reaches the gate within 30 min / 200M bytes): **not tested.** The run was
  stopped at 25.5 min and 155M bytes, inside the prediction's own window, so it is neither
  confirmed nor refuted. At the stop it was at 0.087 with a flat selection curve, which is evidence
  against it, but the registered test did not complete.
- **P4** (bytes-to-90% between 5M and 200M): **not tested**, for the same reason.
- **P12** (training moves the median half-life up): partially observed and more interesting than
  predicted. The median rose 32.0 → 52.2 as predicted, but the **longest head collapsed 512 → 84**
  and p90 fell 396.8 → 78.0. Training actively shortened the long timescales. That is consistent
  with a model that has found recency useful and long memory useless, which is what a
  set-membership sketch would want. Reported as an observation from an interrupted run, not as a
  settled result.

**No stage was retried, shrunk, or tuned in response to this.** predictions.md's shrink rule is
conditional on reaching the 60-minute ceiling, which did not happen.

---

## Reproduce

```sh
PYTHONPATH=smRTS_01 python3 -m unittest discover -s smRTS_01/tests -p 'test_*.py' -v
PYTHONPATH=smRTS_01 python3 smRTS_01/train_recall.py \
  --name S1-leaky-tbptt-frozen128 --stage S1 --cell leaky --regime tbptt --device mps \
  --dim 384 --layers 1 --batch 64 --window 256 --minutes 10 --eval-every 60 \
  --freeze-decay 128 --futility-mode report --canary-reference 615.4 --seed 1

# S2, as launched. It was stopped externally at ~25.5 min and did not reach its 60-min ceiling.
PYTHONPATH=smRTS_01 python3 smRTS_01/train_recall.py \
  --name S2-fast-tbptt-calibration --stage S2 --cell fast --regime tbptt --device mps \
  --dim 384 --layers 1 --heads 4 --key-dim 8 --value-dim 12 --batch 64 --window 256 \
  --minutes 60 --eval-every 120 --futility-fraction 0 --gate-accuracy 0.95 \
  --canary-reference 615.4 --seed 1

# Rebuilding the interrupted run's artifacts. Frozen evaluation only: no optimizer, no gradient.
PYTHONPATH=smRTS_01 python3 smRTS_01/finalize_interrupted.py \
  --name S2-fast-tbptt-calibration --stop-reason user_stop --device mps
```

Each run directory holds `config.json`, `manifest.json` (sha256 of every source file plus the world
constants), `log.csv`, `results.json`, `halflives.json`, `ckpt.pt` (best by selection) and
`final.pt` (last step). S2 has no `final.pt`, for the reason given above.

`out/` also contains `canary_reference.json` (the 615.4 matmuls/s MPS reference), `S2.stdout`, and
`smoke-cpu/`, which is the S0 end-to-end smoke and is marked `NOT_A_RESULT.md`.

**Manifest drift is expected on one file.** Re-checking both runs' manifests against the current
tree reports `predictions.md` as changed. That is the dated outcome note appended to it *after* the
runs; every prediction it registers is byte-for-byte unchanged, and the manifest is what makes that
edit visible instead of invisible. No other source file has moved since either run.

## Two bugs the S0 smoke caught before any MPS time was spent

Both are recorded because the second one would have silently corrupted every result.

1. **Checkpoints could not be reloaded.** `config` was embedded in `ckpt.pt` and carried a
   `torch.torch_version.TorchVersion` object, which PyTorch 2.8's default `weights_only=True` load
   refuses. The checkpoint now stores weights and the selection record only.
2. **"Best checkpoint" silently meant "earliest checkpoint".** Selection kept a checkpoint only on
   `accuracy > best_accuracy`. On a run whose selection accuracy is 0.0 at every eval — which is
   exactly what a failing arm looks like — the first eval set the best and nothing ever beat it, so
   `ckpt.pt` held the **step-1, untrained** model. The first smoke duly reported a uniform-over-256
   model as its result. Selection is now accuracy with answer NLL as the tie-break, so a run that
   never answers correctly still keeps its best-calibrated weights.
