# smRTS_01 Phase 2 predictions, stop rules, and run configs

**Written 2026-09-17, before any training run.** Nothing in Phase 2 has been trained at the time of writing. This
file is append-only: it may gain a dated "outcome" column after the runs, but a prediction here is never edited or
deleted once a run that tests it has started.

**Outcome note, appended 2026-09-17 after the runs. Nothing above or below this block was edited.**
Evidence: [phase2_results.md](phase2_results.md).

| Prediction | Stage | Outcome |
|---|---|---|
| P1 | S1 complete | **Held.** Wilson upper bound 0.069 at `(16, 128)`, against the registered < 0.30. |
| P2 | S1 complete | **Falsified.** 0.043, Wilson [0.032, 0.057] at `(2, 128)`, against the registered lower bound > 0.10. |
| P3, P4 | S2 stopped by the owner at ~25.5 min | **Not tested.** The run ended inside the prediction's own window, so neither is confirmed or refuted. |
| P12 | S1 frozen; S2 partial | Partially observed: `fast`'s median half-life rose 32.0 → 52.2, but its longest head collapsed 512 → 84. |
| P5-P11, P13 | S3/S4 never started | **Not tested.** |

The S1 futility rule was run in report-only mode, and the futility *reference* was replaced with a
stated distributional floor. Both deviations are recorded in
[phase2_results.md](phase2_results.md#a-registered-deviation) rather than edited into the rules
above.

It commits to, in order:

1. [What Phase 0 and 1 already fixed](#1-what-phase-0-and-1-already-fixed)
2. [Predictions](#2-predictions)
3. [The staged order, with promotion and stop rules](#3-the-staged-order-with-promotion-and-stop-rules)
4. [Proposed run configs](#4-proposed-run-configs)
5. [Metrics](#5-metrics)
6. [Artifacts every run must leave behind](#6-artifacts-every-run-must-leave-behind)
7. [Ambiguities and things deliberately not decided here](#7-ambiguities-and-things-deliberately-not-decided-here)

---

## 1. What Phase 0 and 1 already fixed

These are settled and Phase 2 does not relitigate them. They are restated because every prediction below is
conditional on them.

| Fact | Source | Consequence for Phase 2 |
|---|---|---|
| Online traces are exact for **one** recurrent layer only; at depth 2 the worst relative error is 0.60–0.95 and at depth 4 cosine can go negative | [phase0_results.md](phase0_results.md) | Every promoted comparison is **one layer**. Deeper online runs are `depth-truncated` ablations and are not part of Phase 2. |
| `delta` and `gru` have no validated online trace | [phase0_results.md](phase0_results.md), plan §4 | They run **TBPTT only**, as controls. No online delta arm in this phase. |
| Matched state capacity is 384 scalars/stream: width 384 for `leaky`/`gated`, `heads=4, key_dim=8, value_dim=12` for `fast`/`delta` | [phase0_results.md](phase0_results.md) | `dk=32` is a larger-capacity ablation, never a row labelled matched. |
| Batch-1 online is 11–14× faster on CPU than MPS | [phase0_results.md](phase0_results.md) | Batch-1 online arms run on **CPU**; `online-batched` and `tbptt` run on **MPS**. Both numbers get reported. |
| Gap 8 is position-confounded: the best fixed-ordinal guess reaches 0.47–0.56 at gap 8 for ≥4 pairs, but only 0.073 at `(32, 128)` | [phase1_results.md](phase1_results.md) | **No gap-8 number is ever quoted as associative recall.** Gap 8 rows are reported only next to their `best position` lazy row. |
| The primary gate cell `(pairs=16, gap=128)` has all 16 query ordinals feasible; its best lazy arm is `most common` at 0.144, best fixed position 0.110, uniform 0.029 | [phase1_results.md](phase1_results.md) | Every threshold below is stated against those lazy numbers, not against 1/33. |
| Wilson: 919/1000 clears a 0.90 lower bound; 918/1000 does not | [phase1_results.md](phase1_results.md) | The gate needs ≥ 919 correct of 1,000. |
| `run_plan` refuses a frozen evaluation that mutated parameters, and hashes the ordered episode set | [world.py](world.py) | "Learning was frozen" and "the arms saw identical bytes" are machine-checked, not asserted in prose. |

---

## 2. Predictions

Each has a **number**, a **test**, and a **confidence**. "leaky fails" is not a prediction; the rows below are meant
to be able to come out wrong. Unless stated otherwise, every number is **exact answer-byte accuracy at
`(pairs=16, gap=128)`, reset-per-episode, learning frozen, 1,000 paired episodes**, and "clears"/"below" refer to the
**Wilson 95% lower/upper bound**, not the point estimate.

### Capability

| # | Prediction | Test | Confidence |
|---|---|---|---|
| P1 | `leaky` with decay **frozen at half-life 128** under TBPTT cannot bind: its 95% Wilson **upper** bound at `(16, 128)` stays **below 0.30**, and it does not beat the `most common` lazy arm by more than 10 points at `(16, 32)` either. | S1 | 0.80 |
| P2 | The same frozen-decay `leaky` still beats uniform at `(2, 128)` — Wilson lower bound **above 0.10** — because two pairs need discrimination, not binding. If it fails even here, the trainer is broken, not the cell. | S1 | 0.70 |
| P3 | `fast` under TBPTT, uncapped, **does** reach the gate: Wilson lower bound ≥ 0.90 at `(16, 128)` within **30 minutes** of MPS wall time and within **200M** training bytes seen. | S2 | 0.55 |
| P4 | If P3 holds, the bytes-to-90% figure is **between 5M and 200M**. Under 5M would mean the task is easier than the recall literature suggests; over 200M means the task must be shrunk and every later budget rebaselined. | S2 | 0.65 |
| P5 | At matched 384 state scalars, the ordering at `(16, 128)` under TBPTT is `delta ≥ fast > gated > leaky`, with `delta − fast ≥ 0` and `gated − leaky ≥ 0.10`. `gru` lands at or above `gated` and below `delta`. | S3 | 0.50 for the full ordering; 0.75 for `fast > leaky` alone |
| P6 | Additive `fast` degrades with pair count while `delta` does not: `fast` loses **more than 20 points** going from `(4, 128)` to `(32, 128)`, and `delta` loses **less than half** what `fast` loses over the same move. | S3 | 0.65 |
| P7 | No cell clears the gate at `(32, 512)` in this phase. | S3 | 0.75 |
| P8 | Every cell's accuracy at gap 8 exceeds its accuracy at gap 128 by more than the pure-recall story predicts, and at `(16, 8)` at least one cell lands within 5 points of the 0.521 `best position` lazy arm — i.e. the gap-8 column measures position, not memory. | S3 | 0.70 |

### Cost and mechanism

| # | Prediction | Test | Confidence |
|---|---|---|---|
| P9 | Batch-1 `online` on CPU runs at **300–1,500 training bytes/s** for a one-layer 384-state model, i.e. **20–100×** fewer bytes/second than `tbptt` on MPS. | S4 (a timing probe is enough) | 0.80 |
| P10 | `fast`'s `online-batched` throughput in **bytes/s is flat within 2×** across batch 8 → 32, because the trace update is bandwidth-bound; `leaky`'s rises by more than 3× over the same range, because its traces are ~5× smaller. | S4 | 0.60 |
| P11 | At **equal wall time**, `tbptt` beats every online regime on accuracy for every cell. At **equal bytes seen** (read off the same curves at 400k bytes), the gap shrinks by at least half for at least one cell. | S3 vs S4 | 0.70 equal-time; 0.45 equal-bytes |
| P12 | Training moves the **median half-life up**: from the log-uniform init median of ~32 bytes to **> 64 bytes** for any cell that clears 0.50 at `(16, 128)`, and it stays under 64 for any cell that fails. This is the Phase 2 reproduction of upstream issue #3. | S1–S4, before/after half-life tables | 0.60 |
| P13 | The unfrozen-decay `leaky` **does not** rescue itself: allowing decay to train does not move `(16, 128)` accuracy more than 5 points above the frozen-decay S1 number. | S3 (`leaky` appears in the matched comparison with decay trainable) | 0.70 |

### What would make me wrong in an interesting way

- **P1 falsified** — frozen-decay `leaky` binds. Then the plan's whole "a decaying sum cannot bind a key to a value"
  framing is wrong, this document gets a dated correction, and the remaining stages are re-scoped before they run.
- **P3 falsified at any budget** — nothing learns. Then the finding is about the task or the trainer, not the cells,
  and the S2 stop rule below fires rather than the grid running anyway.
- **P6 falsified** — `fast` holds up at 32 pairs. Then 384 scalars is simply not near the interference boundary at
  these pair counts, and the pairs axis needs to extend past 32 before it says anything about capacity.

---

## 3. The staged order, with promotion and stop rules

Fixed order, one stage at a time, **one GPU/MPS job on the Mac at a time**. No stage starts until the previous
stage's results.json is written and read. Phase 3 (Shakespeare) is explicitly **not** in scope; nothing here
promotes anything into it.

### S0 — pre-flight, no training (CPU, ~5 min)

Re-run the Phase 0 + Phase 1 suites, print the trace-byte figure and parameter/state counts for every promoted
config, and record a contention canary: a fixed 5-second matmul + elementwise micro-benchmark, stored as the
reference. **Stop rule:** any Phase 0/1 test failing stops Phase 2 outright.

### S1 — the `leaky` binding falsification (TBPTT, decay frozen at half-life 128)

The cheapest thing that can kill the plan's central claim. One layer, `leaky`, decay logits set to half-life 128 and
`requires_grad=False`, TBPTT, one seed.

- **Promote** (P1 falsified): Wilson lower bound ≥ 0.50 at `(16, 128)` → **stop and rewrite this file** before any
  further run. `leaky` binding changes what the rest of the phase is testing.
- **Record and continue** (P1 held): Wilson upper bound < 0.30 at `(16, 128)` → `leaky` is answered for the binding
  question. It still runs in S3 with decay trainable, to test P13, but it is not expected to pass.
- **Ambiguous band** (0.30–0.50): report the number, continue to S2, and do **not** claim P1 either way.
- **Futility stop:** if at 25% of the S1 budget the answer-byte NLL at `(16, 32)` has not fallen at least 0.3 nats
  below the best lazy arm's NLL on the same episodes, kill the run and record `stop_reason: futility`.

### S2 — uncapped `fast-tbptt` calibration

One run, `fast`, one layer, matched 384 state, TBPTT, **no time cap** other than the hard ceiling below, logging
accuracy/NLL/rank at `(16, 128)` against bytes-seen. This stage exists to separate "the budget was too small" from
"the architecture failed", and it sets every later stage's budget.

- **Hard ceiling:** 60 minutes of MPS wall time. This is a ceiling, not a target.
- **Promote:** Wilson lower bound ≥ 0.90 at `(16, 128)`. Record `bytes_to_gate` and `seconds_to_gate`; **S3's budget
  is 2 × `seconds_to_gate`**, rounded up to the nearest minute, capped at 20 minutes per arm.
- **Shrink and redo S2 once:** if it reaches ≥ 0.50 but not 0.90 by the ceiling, shrink the task — primary cell moves
  to `(pairs=8, gap=128)` — rebaseline every threshold in this file against the `(8, 128)` lazy row (most common
  0.194, best position 0.162), note the change with a date, and rerun S2 once. **Only one shrink is allowed**, and
  the shrink is recorded as a design change, not hidden.
- **Stop Phase 2:** if it stays below 0.50 at the ceiling *after* the one permitted shrink, stop. The deliverable is
  then a negative result about task-or-trainer, and no matched comparison runs.
- **Never** stop this stage early to declare a pass; the gate needs the full 1,000-episode frozen evaluation.

### S3 — matched one-layer comparison

Only if S2 promoted. One layer, 384 state scalars, the S3 budget from S2, one seed each, arms:

`leaky-tbptt` (decay trainable) · `gated-tbptt` · `fast-tbptt` · `delta-tbptt` · `gru-tbptt` (control) ·
`attention ceiling` (smLLM_01 layout, ctx 1024, ~0.9M params) · every lazy arm on the identical episode set.

- **Sanity stop:** if the attention ceiling does not clear Wilson lower bound 0.90 at `(16, 128)`, the **task** is
  suspect. Stop, report, and do not interpret any cell's failure as an architecture result.
- **Promote to S4:** any cell whose Wilson lower bound at `(16, 128)` is ≥ 0.50, or within 10 points of the gate.
- **Futility stop per arm:** same NLL rule as S1.

### S4 — the online question

Only cells promoted from S3, and only cells in `ONLINE_CELLS`. `online-batched` at batch 32 on MPS (or the largest
batch whose traces fit, reported), plus one batch-1 `online` CPU arm per promoted cell as the **cost** measurement.

- **Promote to seeds:** Wilson lower bound ≥ 0.90 at `(16, 128)`, learning frozen, state reset per episode. That is
  the Phase 2 gate.
- **Seeds are conditional:** three paired seeds only for an arm that passes or comes within 10 points. Report
  **min / median / max** across seeds, never a t-interval from n = 3.
- **Batch-1 online is never a capability claim.** It is labelled a cost measurement in every table.

### S5 — reporting

Every promoted row reported pass or fail, plus the grid, the half-life tables, the lazy rows, and which predictions
above held. A stage that fired a stop rule is still a reported result.

---

## 4. Proposed run configs

Deliberately cheap. Every one is a proposal for the owner to approve; none has been run.

**Common to all arms:** one recurrent layer, 384 state scalars/stream, AdamW, grad-norm clip 1.0, cosine schedule to
10% of peak, cross-entropy on the next byte only (no `--repo-loss` in Phase 2), decay half-lives initialised
log-uniform over [1, 1024] except where frozen, training episodes drawn from a **training seed disjoint from
`EVAL_SEED = 20260917`**, best checkpoint kept by the selection metric, final-step number reported next to it.

| Stage | Cell | Regime | Device | Batch × window | LR | Budget | Training mix |
|---|---|---|---|---|---|---|---|
| S1 | `leaky`, decay frozen h=128 | tbptt | MPS | 64 × 256 | 3e-4 | 10 min | `(16, 128)` only |
| S2 | `fast` | tbptt | MPS | 64 × 256 | 3e-4 | ≤ 60 min, uncapped in bytes | `(16, 128)` only |
| S3 | `leaky`, `gated`, `fast`, `delta`, `gru` | tbptt | MPS | 64 × 256 | 3e-4 | 2 × S2 `seconds_to_gate`, ≤ 20 min | curriculum: `pairs ∈ {2,4,8,16,32}` × `gap ∈ {32,128}`, uniform |
| S3 | attention ceiling | — | MPS | 64 × 1024 ctx | 3e-4 | same as S3 arms | same mix |
| S4 | promoted online cells | online-batched | MPS | 32 streams | 1e-4 | same wall time as S3 | same mix |
| S4 | promoted online cells | online (batch 1) | **CPU** | 1 stream | 1e-4 | 10 min, cost only | `(16, 128)` only |

Notes on these choices, all of which are challengeable:

- **Windows and episode length.** An episode at `(16, 128)` is 134–194 bytes (`4×pairs + 4 + gap − min_gap`). A
  256-byte TBPTT window therefore spans roughly one and a half episodes, so state carries across an episode boundary
  during *training*. Evaluation still resets per episode. If that carry turns out to matter, the fix is a window of
  512 with episodes packed and separated by `\n`, and it is a named config change, not a silent one.
- **Training gap mix excludes gap 8** (position-confounded) and **gap 512** (long, expensive, and P7 says nothing
  passes there). Gaps 8 and 512 are still *evaluated* on the full 5 × 4 grid; they are just not trained on. This is a
  choice worth the owner's eye: training only on {32, 128} and evaluating at 512 measures extrapolation, not
  in-distribution recall, and the write-up must say so.
- **LR is a guess.** If S1's futility rule fires on the very first run, the first thing to check is the LR, and a
  three-point LR probe (1e-4 / 3e-4 / 1e-3, two minutes each) is cheaper than concluding anything from one setting.
  Budget 6 CPU-minutes for it before S1 if S1 looks dead on arrival.
- **Total proposed machine time:** ~5 min (S0) + ~10 (S1) + ≤ 60 (S2) + 6 × ≤ 20 (S3) + ~40 (S4) ≈ **2.0–3.6 hours**
  before conditional seeds, matching the plan's 60–90 min estimate only if S2 promotes early. If S2 uses its full
  ceiling, say so out loud rather than absorbing it.

---

## 5. Metrics

**Primary endpoint, pre-registered, one of them:** exact answer-byte accuracy at `(pairs=16, gap=128)`,
reset-per-episode, learning frozen, 1,000 paired episodes, judged on the **Wilson 95% lower bound ≥ 0.90**
(≥ 919/1000). Everything else in the grid is descriptive and is labelled descriptive.

Reported for every arm:

- **Accuracy, answer NLL, and answer rank** over the full 5 × 4 `(pairs, gap)` grid, with Wilson intervals on the
  reset protocol only. NLL and rank exist so a model moving off chance is visible before accuracy moves; the futility
  rules are driven by NLL, not accuracy.
- **Lazy baselines on the identical episode set hash** — uniform, most common, first, last, and best fixed position —
  printed in the same table, not in an appendix.
- **Gap-8 rows flagged position-confounded** wherever they appear.
- **Carry and reset-at-random passes** with accuracy against episodes-since-reset, **no Wilson intervals** (episodes
  are not independent under carry), and the reset-at-random control alongside.
- **Wall time and bytes seen, both, in every row of every log**, so the equal-time and equal-bytes readings (P11)
  come off the same run without a rerun.
- **Throughput** in training bytes/s, and **peak memory**, plus the printed `trace_bytes` for online arms.
- **Decay half-lives per layer** — `ln 0.5 / ln a` in bytes, median and p90 — **before and after training**.
- **Parameters, state scalars/stream, trace bytes/stream**, and device.
- **Contention canary** before and after each timed run; a run whose before-canary deviates > 10% from the S0
  reference is aborted and rerun, not reported.
- **Seed-level min/median/max** for any arm that got three seeds. Never a t-interval from n = 3.

Paired comparisons between two arms use **McNemar** on the discordant episodes of the identical episode set. Any
secondary claim beyond the primary endpoint gets Benjamini–Hochberg at q = 0.1 across the family, and the write-up
says it did.

---

## 6. Artifacts every run must leave behind

Written **incrementally**, so a killed or aborted run still leaves evidence. One directory per arm:
`smRTS_01/out/<stage>-<cell>-<regime>-seed<N>/`.

| File | Contents |
|---|---|
| `config.json` | every flag, cell/regime/seed, dim/layers/heads/key_dim/value_dim, LR/schedule/clip, batch, window, budget, training gap-and-pair mix, torch version, device, hostname |
| `log.csv` | one row per eval tick: `step, wall_s, bytes_seen, lr, train_nll, eval_acc, eval_nll, eval_rank, acc_by_cell, nll_by_cell, throughput_bytes_s, peak_mem_bytes, halflife_median, halflife_p90` |
| `results.json` | the §5 metric set, plus `stop_reason ∈ {time, bytes, gate, futility, canary, crash}`, `bytes_to_gate`, `seconds_to_gate`, `canary_before`, `canary_after`, `episode_set_hash` per cell, `frozen_params_hash_ok`, `wilson_low/high`, `predictions_tested` |
| `ckpt.pt` | best checkpoint by the **selection** metric, with its sha256 recorded in `results.json`; the final-step checkpoint metric reported next to it |
| `halflives.json` | per-layer half-life arrays before and after training |
| `manifest.json` | sha256 of `cells.py`, `online.py`, `tbptt.py`, `world.py`, and the training script; world constants (`KEYS`, `VALUES`, `FILLER`, `EVAL_SEED`, `PAIR_COUNTS`, `GAPS`); this file's sha256 |

Hard requirements: parameter fingerprints before and after every frozen evaluation must match bitwise (`run_plan`
already enforces this); the episode set hash must be **identical** across every arm compared in a table, and a table
whose arms have different hashes is not a paired comparison and must not be presented as one; `out/` is overwritten
by training, so anything worth keeping is copied before the next run.

---

## 7. Ambiguities and things deliberately not decided here

Called out rather than resolved by silently widening the grid. Each needs an owner decision or an explicit
"proceed as proposed".

1. **The training-episode generator does not exist yet.** `world.py` generates *evaluation* episodes. Phase 2 needs a
   training stream with a disjoint seed and a specified pairs × gap mix. That is new code, and its mix is a design
   choice that affects every result — I have proposed one above rather than assumed one.
2. **Held-out query keys.** The design audit (§3.1) recommended reserving keys that appear as distractors in training
   but are never queried, and reporting seen-key vs held-out-key slices. The plan as revised **did not adopt it**, and
   `world.py` has no such split. I am not adding it unilaterally: it changes the world and would invalidate the
   Phase 1 lazy table. Flagging it as a known gap in the leakage story, most acute at small gaps and small pair
   counts.
3. **Selection vs reporting episode sets.** The audit wanted best-checkpoint selection on a set disjoint from the
   reported set. `EVAL_SEED` is currently a single constant. The clean fix is a `SELECT_SEED` used for in-run
   monitoring at n = 500 and `EVAL_SEED` reserved for the final 1,000-episode report. Cheap, but it is a change to
   world.py's constants and I am not making it without a nod.
4. **Training-window state carry.** Covered in §4: a 256-byte TBPTT window straddles episode boundaries at
   `(16, 128)`. Defensible either way; I picked the simple one and named the alternative.
5. **The attention ceiling's size.** ~0.9M parameters at ctx 1024 is a guess at "same-size transformer". It is not
   state-matched to 384 scalars — nothing with attention is — so it is a **ceiling**, not a matched arm, and is
   labelled that way.
6. **The `--overwrite` variant stays off.** Plan says run it only if a cell passes the main gate. Unchanged here.
7. **`dk=32` larger-capacity ablation, fp16 traces, traces-off, batch-size sweep, `--repo-loss`.** All out of scope
   for Phase 2 as proposed. Each is a named ablation the owner can promote; none runs by default.
8. **Phase 3 is out of scope.** No arm here is promoted to Shakespeare, and `seconds_to_gate` is not a Phase 3 budget.
