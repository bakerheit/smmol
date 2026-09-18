# smRTS_01: experimental design audit

An audit of [`docs/engineering/plans/smRTS_01.md`](../../docs/engineering/plans/smRTS_01.md), written 2026-09-17.
Nothing here is built or trained. This file is the only file this audit owns.

**Read this first.** The plan is well shaped: the questions are real, the world-is-the-answer-key rule is right, and
writing predictions before the runs is the correct habit. The problems below are not about taste. Three of them make
the experiment as written unable to answer its own question:

1. **The recall world cannot produce the gaps it evaluates at.** 16 distinct keys × 4 bytes per pair caps the gap at
   about 62 bytes. Gaps 128, 256 and 512 are unreachable. ([§2.1](#21-the-gap-axis-is-impossible-as-specified))
2. **Every cell starts with a one-byte memory.** `a = sigmoid(0) = 0.5` means half-life 1 at init, and `0.5^128 ≈
   3e-39` of gradient at gap 128. Issue #3 is partly an *initialisation* fact. Run as written, all three cells fail
   and the plan learns nothing about cells. ([§4.1](#41-decay-init-is-the-experiment-not-a-detail))
3. **The phase 3 comparison is not apples to apples in two directions at once.** smLLM_01 is 10.8M parameters against
   a 0.8M recurrent model, and the stateful eval carries unlimited state against smLLM_01's 256-byte window. Those
   two errors point opposite ways, so the net is unknowable. ([§3.2](#32-phase-3-is-unfair-in-two-directions-at-once))

Everything else is tightening. The ranked list is at the end: [§10](#10-ranked-plan-changes).

Contents:

1. [What the numbers actually are](#1-what-the-numbers-actually-are)
2. [The recall world](#2-the-recall-world)
3. [Evaluation and leakage](#3-evaluation-and-leakage)
4. [Cells, parameter matching and init](#4-cells-parameter-matching-and-init)
5. [Memory and throughput on the M5](#5-memory-and-throughput-on-the-m5)
6. [Gates, budgets and stopping rules](#6-gates-budgets-and-stopping-rules)
7. [Statistics](#7-statistics)
8. [Baselines and ablations](#8-baselines-and-ablations)
9. [A cheaper run order](#9-a-cheaper-run-order)
10. [Ranked plan changes](#10-ranked-plan-changes)
11. [Sources](#11-sources)

---

## 1. What the numbers actually are

Everything in this section is arithmetic from the plan's own specification, done here so the owner can check it before
anyone writes code. Layers `L = 4`, fp32, `d` is the width, `H = 4` heads, `dk = dv = 16`.

### 1.1 Parameter counts

| Cell | Parameters (formula) | At d = 384 | Width that lands on 0.79M |
|---|---|---|---|
| `leaky` | `4d² + 525d` | **791,424** | 384 (the plan's anchor) |
| `gated` | `12d² + 525d` | 1,971,072 | **236** |
| `fast` | `1544d + 4353` | 597,296 | **509** |

Breakdown: per layer `leaky` is `W` (d²) + LayerNorm (2d) + decay (d); `gated` adds `V` and `G` (2d²); `fast` is
`Wk, Wv, Wq` (3·64d) + `Wo` (64d) + decay (H·dk·dv = 1024) + LayerNorm. Shared: embedding 256d, decoder 256d + 256,
stop head d + 1.

**So "parameter counts within 20% of each other" forces widths of 236 / 384 / 509 — a 2.2× spread.** That isn't a
control, it's a second independent variable. And it is the *wrong* control here: the budget is 10 wall-clock minutes,
not a parameter count. Under a fixed time budget, what matters is bytes seen per second and state capacity, and
parameter count is a bystander.

**Recommendation.** Fix `d = 384, L = 4` for all three cells — width-matched, depth-matched, roughly
state-size-matched — let parameters vary, and report them in every table. Then add param-matched arms
(`gated` d=236, `fast` d=509) as a named ablation so a reader can separate "cell" from "width". Drop the 20% rule from
"decisions already made"; it is doing harm.

### 1.2 Trace memory

Per stream, fp32, L = 4. This is the number the plan asks for ("trace memory is part of the cost being measured").

| Cell | Trace tensors | Elements per stream | Bytes per stream | Batch 32 | Batch 128 |
|---|---|---|---|---|---|
| `leaky` | `E_d` (L·d), `E_emb` (256·d) | 99,840 | **390 KiB** | 12.2 MiB | 48.8 MiB |
| `gated` d=384 | `E_V`, `E_G` (2·L·d²), `E_d` | 1,181,184 | **4.51 MiB** | 144 MiB | 577 MiB |
| `gated` d=236 | same | 447,112 | **1.71 MiB** | 54.6 MiB | 218 MiB |
| `fast` d=384 | `E_Wk`, `E_Wv` (2·L·H·dk·dv·d), `E_d` (L·H·dk·dv) | 3,149,824 | **12.0 MiB** | **384 MiB** | 1.50 GiB |
| `fast` d=509 | same | 4,174,336 | 15.9 MiB | 509 MiB | 2.0 GiB |

The recurrent **state** itself is trivial by comparison: `leaky`/`gated` hold `L·d` = 1,536 floats (6 KiB), `fast`
holds `L·H·dk·dv` = 4,096 floats (16 KiB) per stream. The traces are 250–800× the state. That ratio is the honest
headline cost of real-time recurrent learning and belongs in the write-up, not buried.

Capacity is not the constraint. Even `fast` at batch 128 is 1.5 GiB of traces, plus ~6 MB of parameters and ~12 MB of
AdamW moments. Any 16 GB Mac holds it.

### 1.3 Throughput, and why `fast` will not batch its way out

The trace update is elementwise over the whole trace, once per byte-column: read `E`, write `E`, and read `E` again
for the gradient contraction. Call it three passes.

| Regime | Bytes touched per byte-column (fast, d=384) | Columns/s at ~150 GB/s achieved | Training bytes/s |
|---|---|---|---|
| `online`, batch 1 | 36 MiB | launch-bound, not bandwidth-bound | ~400–800 (see below) |
| `online-batched`, batch 32 | 1.13 GiB | ~133 | **~4,200** |
| `online-batched`, batch 128 | 4.5 GiB | ~33 | **~4,200** |

**`fast` throughput is bandwidth-capped and roughly independent of batch size.** Batching buys gradient quality, not
speed. That is a sharp, cheap-to-falsify prediction and it should go in `predictions.md`. For `leaky` (390 KiB/stream)
and `gated` at d=236 (1.7 MiB/stream) the traces are small enough that batching does help until the model's own matmuls
become the limit, so the three cells will show *different batch-size scaling curves*. That contrast is more interesting
than most of the 9-cell grid.

Batch 1 is a different bottleneck: kernel launches. A 4-layer cell with traces plus an AdamW step is roughly 150–250
op dispatches per byte. At 5–10 µs per dispatch (eager PyTorch, CPU or MPS — both are in that range for tensors this
small) that is 1–2.5 ms per byte, i.e. **400–1,000 bytes/s**, so 27–67× slower than smLLM_01's 26,700 bytes/s. The
plan predicts 50–100×; that is the right order, maybe slightly pessimistic. Either way the prediction survives.

**Consequence the plan misses:** in a 10-minute phase 2 run, `online` at batch 1 sees **~0.25–0.6M bytes**,
`online-batched` sees **~2.5M**, and `tbptt` (batch 64 × window 256 = 16,384 bytes/step at roughly smLLM_01 speed)
sees **~16M**. That's a 30–60× difference in *data seen* inside a comparison that is supposed to be about the
*regime*. See [§6.2](#62-budget-by-two-axes-not-one).

The M5's actual memory bandwidth is the one input here I could not check (no network, and I did not run `sysctl`).
Treat 150 GB/s as a placeholder and measure it with a one-line elementwise benchmark before trusting the table.

---

## 2. The recall world

### 2.1 The gap axis is impossible as specified

The plan says: alphabet of 16 keys `a`–`p`, pairs written `k=v;` with **distinct keys**, gap = bytes between the
queried pair's `=` and the `?`.

Each pair is 4 bytes. With distinct keys there are at most 16 pairs, so at most 64 bytes of pair text. If the queried
pair is written first, the maximum gap is 15 × 4 + 2 = **62 bytes**.

**Gaps of 128, 256 and 512 cannot be generated.** Four of the seven evaluation gaps, and the entire phase 2 gate (90%
at gap 128), rest on episodes the world cannot produce. `tests/test_world.py` as specified ("the gap is exact") would
catch this on day one of phase 1, which is good, but it would catch it after the cells are ported and after
`predictions.md` is written against a task that doesn't exist.

**Fix, in order of preference:**

1. **Filler bytes.** Pad between pairs with bytes drawn from a third alphabet (say `q`–`z` and spaces) that never
   appear as keys or values. Gap becomes a free parameter, keys stay distinct, and the filler adds a mild
   distractor-robustness test for free. Report the filler alphabet size; it changes the entropy of the stream and so
   the loss numbers.
2. Two-byte keys (`aa`–`pp`, 256 keys). Reaches gap 512 with 85 pairs, but also changes what "binding" means (the
   model must now bind a 2-byte compound), and multiplies the number of pairs that must be held at once.
3. Repeated non-queried keys. Cheapest change, but it quietly introduces an overwrite semantics that the plan puts
   behind a `--overwrite` flag, so it muddies the default task.

Take option 1.

### 2.2 Gap and capacity are two different axes, and the plan only varies one

With 16 distinct keys, an episode holds at most 16 pairs = 16 × 4 bits = **64 bits** of information. A 384-dim fp32
state has room for that a hundred times over. So the task as specified tests **binding**, not **capacity** — which is
fine, and is the right first question given the plan's stated prediction that `leaky` cannot bind at all.

But the recall literature's central finding is that recall accuracy in fixed-state models is governed by state size
*versus number of key-value pairs held*, not by gap alone (Zoology / MQAR, and the recall-throughput tradeoff work —
see [§11](#11-sources)). If the plan wants to say anything about the recurrent-state *idea* rather than about these
three specific cells, it needs a second axis.

**Recommendation.** Keep gap ∈ {8, 16, 32, 64, 128, 256, 512} at a fixed small pair count (say 4 pairs, filler for
the rest) as the primary axis, and add a **pairs** axis {2, 4, 8, 16} at fixed gap 128 as a secondary sweep run once,
on survivors only. The second sweep is ~10 extra minutes and it is what makes the result generalise.

### 2.3 Scoring: add a graded metric

"The answer byte only" gives accuracy against a 1-in-16 floor. Near the floor every failing arm looks identical at
6%, which makes the futility rule in [§6.3](#63-stopping-rules) blind and makes learning curves flat until they
suddenly aren't.

**Report both:** accuracy on the answer byte *and* cross-entropy (bits) on the answer byte. The bits number moves
long before accuracy does, it is what the futility rule should watch, and it costs nothing — it is already computed.

### 2.4 Multi-query episodes

One query per episode extracts ~4 bits per episode of evaluation signal. Asking `k` queries per episode (MQAR style)
multiplies the signal per byte streamed by `k` and cuts evaluation cost proportionally. The plan's episode format
supports it directly: `?k` … answer … `?k'` … answer … `\n`.

This is a **should**, not a must — it changes the task slightly (later queries see earlier answers restated) — but it
makes the n = 2,000 evaluation in [§7.1](#71-how-many-episodes) roughly free.

### 2.5 Lazy guesses: two are missing

The plan lists three: most-common value (1/16), value of the last pair, value of the first pair. Add:

- **the value most recently written to *any* key** (a recency heuristic that a decaying state genuinely implements);
- **a matched-position guess**: the value at the same ordinal position as the query key's position among the keys,
  which catches a model that learned position rather than binding.

The second one matters because the plan already worries about position leakage ("the queried pair sits anywhere among
the distractor pairs so position alone doesn't give it away"). Measure that claim, don't assert it.

---

## 3. Evaluation and leakage

### 3.1 The recall world leaks at short gaps

Episodes are generated fresh, which handles the obvious leak. But the *space* of short-gap episodes is small. At gap
8 (two pairs plus a query) the space is roughly 16 × 15 keys × 16 × 16 values ≈ 61k distinct episodes. A 2M-byte
training run at ~30 bytes per short episode sees ~66k episodes. **The gap-8 evaluation set is, in expectation, almost
entirely memorised.** Accuracy at gap 8 will therefore overstate generalisation, and gap 8 is the one gap the plan
expects `leaky` to pass — so the plan's headline negative result is exactly where the leak bites.

**Fix.** Hold out structure, not just seeds:

- Reserve 4 of the 16 keys (`m`–`p`) as **query-only-at-eval** keys: they appear as distractors in training but are
  never the queried key. Report accuracy separately on held-out-key and seen-key queries.
- Additionally hold out a set of (key, value) *combinations* and report both slices.
- Hash every evaluation episode and assert zero collisions against a bloom filter of training episodes; log the
  collision rate rather than assuming it's zero.

Seed hygiene alone ("eval uses seed + 10⁹") does not fix this and should not be presented as if it does.

### 3.2 Phase 3 is unfair in two directions at once

The plan compares a ~0.8M recurrent model against smLLM_01 at **10.8M parameters** — 13× larger — and simultaneously
gives the recurrent model **unlimited context** via a carried-state streaming eval against smLLM_01's 256-byte window.
The plan notices the second one ("say so in the write-up") and not the first. Two large biases pointing opposite ways
do not cancel; they make the comparison uninterpretable.

There is a third, smaller one: smLLM_01's 1.471 / 2.12 bits/char is the mean loss over all 256 positions of random
windows, **including position 0 with no context at all**. That *understates* the GPT. A sliding-window eval that
scores only the last byte of each window — every prediction with a full 256 bytes of context — will give a
meaningfully better number for the same checkpoint.

**Fix: three named evaluation protocols, all reported.**

| Protocol | smLLM_01 | RTS cell | What it answers |
|---|---|---|---|
| **A: matched window** (primary, and the gate) | as published: random 256-byte windows, all positions scored | state **reset every 256 bytes**, all positions scored | same context budget, same scoring — the only fair head-to-head |
| **B: best-case each** | sliding window, last byte only, full 256 context | full stream, state carried | each architecture at its best |
| **C: natural** | as published | full stream carried | what the plan currently does; keep it, label it |

And **add a parameter-matched GPT**: smLLM_01's layout at d = 192, 4 layers, ctx 256 ≈ 0.9M parameters, 20 minutes,
same data, same seed protocol. Call it `smLLM_01-tiny`. Without it, "the recurrent cell is 0.4 bits/char worse than
smLLM_01" confounds architecture with a 13× size difference, and the phase 3 gate is meaningless. This is one extra
20-minute run and it is the highest-value single addition in the whole plan.

### 3.3 "Learning is frozen" needs to be a test, not a claim

The plan says learning is frozen during evaluation. Assert it:

- hash `model.state_dict()` before and after every evaluation, require bitwise equality;
- assert every trace tensor is zero at the start of a reset-state evaluation, and that no trace is written during it;
- assert `torch.is_grad_enabled()` is False and that optimizer step count did not advance.

Put this in `tests/test_eval_frozen.py`. The source repo's issue #4 is exactly this bug (scoring on its own training
stream with state leaking across examples). A plan whose whole premise is "that repo didn't control for leakage"
should have a test that would have caught it.

### 3.4 Interference pass needs a control

The "state carried across episodes" pass measures interference — but carried state also carries *useful* information
(the model has seen more of the value distribution). Add a third pass: state carried across episodes from a
**different, unrelated stream** (e.g. Shakespeare bytes). If accuracy with unrelated carried state ≈ accuracy with
carried episode state, the effect is generic state pollution, not interference between episodes.

### 3.5 Checkpoint selection

"Keep the best checkpoint by the task metric" plus a gate on that same metric is selection on the reported number. It
is the same protocol smLLM_01 used (best val, 1.471), so the comparison is internally consistent — but report the
**final-step** number next to the best-checkpoint number in every table, as smLLM_01's results entry already does.
And select checkpoints on a *selection* episode set that is disjoint from the *reporting* episode set. Both are free.

---

## 4. Cells, parameter matching and init

### 4.1 Decay init is the experiment, not a detail

`a = sigmoid(d)`. Default init `d = 0` gives `a = 0.5`, half-life exactly **1 byte**. Then:

| Gap | `a^gap` at a = 0.5 | `a^gap` at a = 0.9946 (half-life 128) |
|---|---|---|
| 8 | 3.9e-3 | 0.958 |
| 128 | 2.9e-39 | 0.500 |
| 512 | 7.5e-155 | 0.062 |

At init, the gradient signal reaching a byte 128 steps back is ~1e-39 — below fp32 denormals. **The eligibility trace
also decays at `a`**, so the online regimes have no long-range credit assignment either. Run as written, every cell
fails at every gap beyond ~16 for a reason that has nothing to do with the cell.

Issue #3 ("the learned decays give a half-life of about one byte") is therefore substantially a statement about where
the decays *started*, not about what training did to them.

**Fix (must).** Initialise `d` so half-lives are **log-spaced over [1, 512]** across channels: `a_i = 0.5^(1/h_i)`,
`h_i = 512^(i/(n-1))`, i.e. `d_i = logit(a_i)`. This is the standard move in the modern state-space and gated-linear-
attention line of work (S4/Mamba's `A_log`, GLA's gate parameterisation — [§11](#11-sources)) and it costs three lines.
Apply it identically to all three cells, report the half-life histogram at init as well as after training, and make
"did training move the half-lives, and in which direction?" an explicit result.

**And run the cheapest possible falsification first** ([§9](#9-a-cheaper-run-order), stage 0): take `leaky`, **freeze**
the decays at half-life 128, and see whether it can do gap-128 recall under `tbptt`. The plan predicts it cannot,
because `b_t = x_t` has no state-times-input interaction so it cannot bind. If that's right, the hand-set-decay probe
proves it in ten minutes and settles one of the plan's four questions before any of the grid runs. If it's wrong —
if hand-set decays rescue `leaky` — then the whole framing ("the cell can't bind") is wrong and you want to know that
on day one, not in phase 2.

### 4.2 The gradient check needs tighter specification

The plan's gate: 1 layer, dim 8, 6 bytes, every parameter matches autograd to 1e-4.

Three problems. **(a)** fp32 accumulation over a recurrence will not reliably hit 1e-4 *relative* on small gradients;
run the check in **float64 on CPU** (MPS has no fp64) and require relative error ≤ 1e-6 there, with a separate looser
fp32 check at ≤ 1e-3. **(b)** Six steps barely exercises the trace; use 32 steps so `E` accumulates over a meaningful
horizon. **(c)** At `d = 0`, `a(1−a) = 0.25` is at its maximum and `a^t` dies instantly — the decay gradient is both
largest and shortest-lived, which is the easiest case. Run the check at **three decay settings**: `a ≈ 0.5`, `a ≈
0.9`, `a ≈ 0.99`. The `a ≈ 0.99` case is where an off-by-one in the trace recurrence actually shows up.

Also assert explicitly that **non-traced parameters match exactly** (they take the ordinary current-step gradient, so
any mismatch is a plumbing bug), and report the cosine similarity as well as the max error — a scale error and a
direction error are different failures.

### 4.3 "Same loss on step 1" is under-specified

Batch-1 online and batch-64 tbptt see different data on step 1, so their losses cannot be equal. What you actually
want to assert:

- identical model init: same seed → identical `state_dict` hash across all three trainers;
- initial loss ≈ `ln 256 = 5.545` nats (**8.00 bits/byte**) for all three, which is the real "nothing is broken"
  invariant;
- on a *deliberately identical* first byte and identical state, the three trainers' per-byte loss matches to 1e-6.

### 4.4 The `fast` read path is untraced and that's a bigger deal than it looks

`Wq` and `Wo` are read-path, so they take exact current-step gradients — good. But `fast`'s *entire binding capability*
lives in `Wk`/`Wv` (traced, approximate) while the *retrieval* lives in `Wq` (exact). The approximation quality is
therefore asymmetric between write and read in a way `leaky` and `gated` don't share. Say so in the write-up; it is a
plausible explanation for any gap between `fast-online-batched` and `fast-tbptt`, and it is the kind of thing the
RTRL-approximation literature (SnAp, UORO, and the RTRL promise-and-limits work) exists to characterise.

---

## 5. Memory and throughput on the M5

Numbers in [§1.2](#12-trace-memory) and [§1.3](#13-throughput-and-why-fast-will-not-batch-its-way-out). Feasibility
summary:

| Question | Answer |
|---|---|
| Do the traces fit at batch 32? | Yes, comfortably. Worst case `fast` d=509 is 509 MiB. |
| At batch 128? | Yes: `fast` 1.5–2.0 GiB. The plan's fallback ("use the largest batch that does") will not be needed. |
| What's the real limit? | Memory **bandwidth** for `online-batched`, kernel **launch count** for `online`. Neither is capacity. |
| Does MPS help batch 1? | Almost certainly not — the plan is right to measure CPU and MPS and report both. Expect MPS to be *slower* at batch 1. |
| `torch.compile`? | Don't budget on it. MPS support is the least mature backend; if it works it's a bonus, and if a run depends on it the number isn't reproducible. |
| fp16/bf16 for traces? | Would halve the bandwidth bill and roughly double `fast`'s throughput. **Do not do it silently** — a trace is a long-running accumulator and fp16 will bias it. If tried, it's a named ablation with the fp32 gradient check re-run in fp16. |

**Two operational controls the plan needs.**

1. **A contention canary.** The engineering README's own evidence is a 6.9× slowdown (145.6 min vs 21.1 min) when a
   model was resident on the same box. Every phase 2 and 3 number is a wall-clock number, so every one of them is
   hostage to what else is loaded. Run a fixed 5-second matmul + elementwise micro-benchmark immediately before and
   after each timed run, write both to `results.json`, and **abort the run** if the before-number deviates more than
   10% from a stored reference. Cheap, and it converts "the ground rules say don't train and serve" from a hope into
   a check.
2. **Counterbalance for thermal drift.** Phase 3 is 8 runs × 20 minutes ≈ 3 hours of continuous GPU on a laptop.
   Sustained-load throttling will systematically favour whichever runs go first. **Randomise the run order**, log
   per-eval throughput, and report first-decile vs last-decile throughput drift per run. If drift exceeds 10%, insert
   a 5-minute cooldown between runs and say so.

---

## 6. Gates, budgets and stopping rules

### 6.1 The phase 2 budget may be too small for *any* cell, which would be a null result for the wrong reason

10 minutes or 2M bytes. At ~30–600 bytes per episode that's 3k–60k episodes. Learning key-value binding from scratch
in this family of tasks typically takes on the order of 10⁵–10⁶ examples. There is a real chance that **no cell, in
any regime, learns the task inside the budget** — and then the 9-cell table is full of 6% and answers nothing.

**Fix (must): calibrate the task to the budget before running the grid.** One run: `fast` + `tbptt` (the strongest
combination the plan expects), 60 minutes, no cap, logging accuracy vs bytes-seen at gap 128. Then:

- if it hits 90% at, say, 4M bytes → the 10-minute budget is roughly right for `tbptt` and hopeless for batch-1
  `online`; proceed, knowing that in advance;
- if it needs 40M bytes → **shrink the task** (8 keys, gaps to 128, 4 pairs) until `tbptt` solves it in ~5 minutes,
  and re-baseline every gate against the shrunken task.

Designing the task to fit the budget is legitimate and is the only way the online-vs-tbptt comparison can be run at
all on a laptop. Designing the budget to fit a hoped-for result is not. Do the former, write down which one you did.

### 6.2 Budget by two axes, not one

As computed in [§1.3](#13-throughput-and-why-fast-will-not-batch-its-way-out), a 10-minute budget gives `online` ~0.4M
bytes and `tbptt` ~16M bytes. Comparing them at 10 minutes answers "which regime is more useful per minute" (a real
and interesting question). It does **not** answer "which regime learns better" — that needs equal bytes.

**Both questions matter and the plan only asks one.** Log every evaluation row with *both* wall-clock seconds and
cumulative bytes-seen, then report every phase 2 result twice:

- **equal time**: accuracy at 10 minutes (the plan's current framing, keep it as primary);
- **equal data**: accuracy at 400k bytes seen, read off the same curves (free — it's the same log).

400k is chosen because it's what batch-1 `online` can actually reach; every other arm passes through it.

### 6.3 Stopping rules

Pre-register these, in `predictions.md`, before phase 2:

- **Futility (applies to all phase 2 arms).** If at 25% of budget the answer-byte **bits** at gap 32 have not fallen
  at least 0.3 bits below the best lazy guess, kill the run and record it as a fail-by-futility. Watch bits, not
  accuracy ([§2.3](#23-scoring-add-a-graded-metric)) — accuracy is flat until it isn't. This saves roughly half the
  grid's Mac time.
- **No early success.** Never stop early to declare a pass. Passing needs the full budget and the full evaluation.
- **Seeds are conditional.** Run seed 1 for everything; run seeds 2–5 only for arms that pass, or come within 10
  percentage points of passing.
- **One primary endpoint.** Accuracy at gap 128, learning frozen, state reset per episode, protocol as specified.
  Everything else in the 9 × 7 table is descriptive. Say this before the runs, not after. 63 numbers with no
  pre-registered primary is 63 chances to find a story.

### 6.4 Restate the gates

| Phase | Plan's gate | Restated |
|---|---|---|
| 0 | params match autograd to 1e-4, 6 bytes, dim 8 | float64 CPU, rel. error ≤ 1e-6, 32 steps, at a ∈ {0.5, 0.9, 0.99}; fp32 ≤ 1e-3; non-traced params exact; cosine ≥ 0.9999 |
| 0 | "memory flat after warm-up" over 10k steps | RSS and `torch.mps.current_allocated_memory()` sampled every 500 steps after a 1,000-step warm-up; OLS slope < 1 MB per 1,000 steps; live tensor count stable |
| 0 | "same loss on step 1" | identical init hash; initial loss = 8.00 ± 0.01 bits/byte for all three; identical per-byte loss on a forced-identical first byte |
| 2 | ≥ 90% at gap 128 | **Wilson 95% lower bound ≥ 90%** (needs ~91.9% observed at n=1,000; ~91.3% at n=2,000), on held-out-key queries, plus the equal-data reading |
| 3 | within 0.3 bits/char of smLLM_01 at ≥ 1/10 throughput | under **protocol A**, against **both** smLLM_01 (10.8M) and `smLLM_01-tiny` (~0.9M), and **above the n-gram floor** ([§8](#8-baselines-and-ablations)) |

The 0.3 bits/char allowance needs the n-gram floor especially. 2.12 + 0.3 = 2.42 bits/char is in the range a plain
order-5 byte n-gram with backoff reaches on Tiny Shakespeare in a few seconds of CPU. A gate that a trigram model
passes is not a gate.

---

## 7. Statistics

### 7.1 How many episodes

Binomial, Wilson interval. At n episodes per gap:

| n | 95% half-width at p = 0.9 | Observed % needed for a 90% lower bound |
|---|---|---|
| 500 | ±2.6 pp | ~92.6% |
| 1,000 (plan) | ±1.9 pp | ~91.9% |
| 2,000 | ±1.3 pp | ~91.3% |
| 4,000 | ±0.9 pp | ~90.9% |

Evaluation is forward-only and batchable across episodes even for models trained one byte at a time, so it runs at
roughly `tbptt` speed regardless of the training regime. A full 7-gap sweep at n = 2,000 with ~600-byte episodes is
~8M bytes of forward passes — a minute or two, not an hour.

**Recommendation.** n = 500 at gap 128 only for in-run monitoring; **n = 2,000** for the full 7-gap sweep at the best
checkpoint. Use Wilson, not normal-approximation, and report the interval in the table, not just the point estimate.
Clopper–Pearson if anyone objects to Wilson; at these n they barely differ.

### 7.2 Pair everything

Generate the evaluation episodes **once** from a fixed seed and score every model on the identical episodes. Then:

- **accuracy differences between two models**: McNemar's exact test on the discordant pairs. Paired testing here is
  worth roughly a 2–4× reduction in the n needed, for zero extra compute.
- **bits/char differences on Shakespeare**: the val split is one 111,539-byte stream and per-byte losses are strongly
  autocorrelated. An i.i.d. bootstrap over bytes gives a ±0.005 bits/char interval, which is fantasy. Use a **moving-
  block bootstrap** with block length ~1,000 bytes on the *paired per-byte difference*, 10,000 resamples.

### 7.3 Seeds

Three seeds gives a mean and a range and nothing else. Do not compute a t-interval from n = 3 and present it as a
confidence interval; with 2 degrees of freedom the multiplier is 4.30 and the interval is mostly noise.

**Recommendation.** Report **min / median / max across seeds** for 3 seeds. For any arm that becomes a headline claim
("`fast-online-batched` passes"), run **5 seeds** — 50 extra minutes — and report the seed-level mean with the range.
State explicitly in the write-up that seed variation is reported as a range, not a confidence interval.

### 7.4 Multiple comparisons

Phase 2 produces 9 runs × 7 gaps × 2 state modes = 126 accuracy numbers. Pre-register the one primary endpoint
([§6.3](#63-stopping-rules)). For the secondary claims the plan will inevitably want to make, apply
Benjamini–Hochberg at q = 0.1 across the family and say you did. This is cheap honesty and it is exactly the kind of
thing the plan's own "predictions written before the runs" rule is reaching for.

### 7.5 What to log

One `results.json` schema for every run, written incrementally so a killed run still leaves evidence:

```
run_id, cell, regime, seed, d, layers, heads, dk, dv
param_count, trace_bytes_per_stream, state_bytes_per_stream
device, torch_version, port_source_commit, plan_version
contention_canary_before, contention_canary_after
budget_minutes, budget_bytes, stop_reason            # time | bytes | futility | crash
rows: [ {step, wall_s, bytes_seen, lr, train_bits, eval_bits, eval_acc,
         acc_by_gap{}, bits_by_gap{}, throughput_bytes_s, peak_mem_bytes,
         halflife_median, halflife_p90} ]
best: {step, metric, ckpt_sha256}
final: {step, metric}
eval: {episode_set_hash, n_episodes, frozen_params_hash_ok, wilson_lo, wilson_hi}
```

Two notes. The repo root is **not a git repository** (checked), so `port_source_commit` has to be a recorded hash of
the source files, not a git sha — or initialise a repo, which is the better answer. And `bytes_seen` in every row is
what makes the equal-data reading in [§6.2](#62-budget-by-two-axes-not-one) possible after the fact; without it those
runs have to be repeated.

---

## 8. Baselines and ablations

### 8.1 Baselines the plan is missing

Ranked by how much they change what the results mean.

| Baseline | Cost | Why it's needed |
|---|---|---|
| **`smLLM_01-tiny`**: smLLM_01's layout at ~0.9M params, 20 min, same data | 20 min | Without it, phase 3 confounds architecture with a 13× parameter difference. Highest-value addition in the plan. |
| **Byte n-gram**, orders 3/5/7 with add-k or Kneser–Ney backoff, same split | seconds, CPU | The floor. Establishes that 2.4 bits/char means something. Also gives a free sanity check on the data pipeline. |
| **An attention model on the recall world**, ~0.8M params, ctx ≥ 1024, same budget | 10 min | The plan compares recall against lazy guesses only. "90% at gap 128" is uninterpretable without knowing that a same-size transformer gets ~100%. If it doesn't, the task is broken, and you want to learn that from a 10-minute run. |
| **GRU or LSTM under `tbptt`** (the plan's open question 2) | 10 min | **In.** It separates "online training is the problem" from "diagonal recurrence is the problem" — which is one of the plan's four questions. Non-diagonal recurrence can't use these traces, and that's fine; it only ever runs under `tbptt`. |
| **State-ablated cell**: the same cell with `a = 0` (state cleared every byte) | 10 min | Shows how much accuracy comes from the state at all versus local byte statistics. Catches a whole class of "it learned the format, not the binding" result. |

### 8.2 Ablations worth the time

| Ablation | Cost | What it isolates |
|---|---|---|
| **Decay init**: default `d=0` vs log-spaced half-lives vs frozen at half-life 128 | 3 × 10 min | See [§4.1](#41-decay-init-is-the-experiment-not-a-detail). This is arguably the single most informative experiment in the project, and the cheapest. |
| **Traces off**: `online` with current-step gradient only | 10 min | Directly tests the repo's central claim. If traces buy nothing, the whole RTRL apparatus — and 384 MiB of trace memory — is dead weight, and that is a publishable negative in one run. |
| **Batch-size sweep** for `online-batched`: 1, 8, 32, 128 | 4 × 10 min per surviving cell | The actual novel axis. Also tests the [§1.3](#13-throughput-and-why-fast-will-not-batch-its-way-out) prediction that `fast`'s throughput is flat in batch size while `leaky`'s isn't. |
| **Repo loss, term by term**: CE only / CE + variance hinge / CE + latent MSE / all four | 3 extra × 20 min | The plan runs all-four vs CE-only. An all-or-nothing comparison can't attribute an effect to a term. If the budget is tight, at minimum add CE + variance hinge, since collapse is the failure the hinge exists to prevent. |
| **fp16 traces** | 10 min | Would roughly double `fast`'s throughput. Only with the phase 0 gradient check re-run in fp16. |

### 8.3 Phase 4 has two confounds, both fixable

**"Learning at inference helps."** In the learning-on arm the model both (a) updates its weights and (b) *sees the
answer byte* in its input stream. The frozen arm gets neither. Any improvement is unattributable.

**Fix: three arms, not two.** (1) frozen, answer not shown; (2) frozen, answer shown (it enters the state, but no
weight update); (3) learning on, answer shown. Arm 2 minus arm 1 is the state effect; arm 3 minus arm 2 is the actual
continual-learning effect, which is the number the plan wants.

Also: comparing episodes 1–200 against 801–1,000 assumes those two blocks are equally hard. Use a fixed episode set
and **run it in both orders** (forward and reversed) in each arm; the order effect is then measurable rather than
assumed.

**"The own-output trap."** 5,000 bytes of self-training will degrade the model — but so would 5,000 batch-1 updates
on *anything* at a training-scale learning rate, since the model was optimised under a different batch size and
schedule.

**Fix: a control and a sweep.** Run the same 5,000 batch-1 updates on **held-out real Shakespeare** (should improve
or stay flat) alongside the self-generated bytes, at three learning rates (1e-5, 1e-4, and the final training lr).
The difference between the two curves is the own-output effect; the curve shape is the batch-1-damage effect. Without
the control, the number that goes next to the Learn module's rule in `whats-next.md` is not evidence for that rule.

---

## 9. A cheaper run order

The plan's budget is ~4 hours of Mac time across phases 2 and 3, mostly spent filling a 9-cell grid. The order below
front-loads everything that can kill a bad idea for free, and only spends grid time on arms that have survived. Each
stage has a kill rule.

| Stage | What | Machine time | Kill rule |
|---|---|---|---|
| **S0** | Gradient checks in float64 at three decay settings; constant-memory check; n-gram baselines; lazy-guess rates; world self-test **including the gap assertion** | ~15 min, CPU | Traces don't match → stop, fix, restart. Gap 128 unconstructible → fix the world ([§2.1](#21-the-gap-axis-is-impossible-as-specified)) before anything else. |
| **S1** | **The `leaky` binding probe**: `leaky` with decays *frozen* at half-life 128, `tbptt`, gap 128, 10 min | 10 min | If it passes, the plan's central claim ("`leaky` can't bind") is wrong — stop and rewrite `predictions.md` before spending 4 hours. If it fails, question 1 is answered for `leaky` and `leaky` drops out of the online grid. |
| **S2** | **Task calibration**: `fast` + `tbptt`, 60 min, uncapped, accuracy vs bytes-seen at gap 128. Plus the attention baseline on the recall world | 70 min | Nothing reaches 90% at any budget → the task is too hard or broken; shrink it ([§6.1](#61-the-phase-2-budget-may-be-too-small-for-any-cell-which-would-be-a-null-result-for-the-wrong-reason)) and redo S2. This is the stage that decides every later budget. |
| **S3** | **Feasibility screen**: each surviving cell under `tbptt` only, 10 min, 1 seed, at the calibrated budget | 20–30 min | A cell that fails under `tbptt` — the strongest regime — cannot pass under any online regime. Drop it. Typically cuts the 9-cell grid to 6 or 3. |
| **S4** | **The online question**: survivors under `online-batched`, batch 32, 1 seed, with the futility rule | 20–30 min | Futility rule per [§6.3](#63-stopping-rules). |
| **S5** | **The cost number**: `leaky` and the best cell under `online` batch 1, CPU and MPS | 40 min | None — this is a cost measurement, not a capability measurement. Label it as such in the write-up. |
| **S6** | Seeds 2–5 for arms that passed or came within 10 pp; the decay-init and traces-off ablations | 60–90 min | — |
| **S7** | Phase 3 text: `smLLM_01-tiny`, n-gram floor, then survivors under `online-batched` and `tbptt`, **order randomised**, protocols A/B/C | ~2 h | — |
| **S8** | Phase 4, with the three-arm and control designs from [§8.3](#83-phase-4-has-two-confounds-both-fixable) | ~45 min | — |

Expected total if the plan's predictions hold (i.e. `leaky` drops at S1, one or two cells drop at S3): **about 2.5–3
hours**, versus ~4 hours for the flat grid — and with baselines, ablations and confound controls the flat grid
doesn't have. If the predictions *don't* hold, S1 and S2 tell you within 90 minutes, which is the whole point.

The ordering principle: **strongest regime first**. `tbptt` is the upper bound on what a cell can do; if a cell can't
do the task with exact gradients and full backprop, no amount of trace approximation will rescue it. Screening on the
upper bound is the cheapest way to shrink a grid.

---

## 10. Ranked plan changes

### Must

1. **Fix the recall world so gaps beyond ~62 bytes exist.** Filler bytes between pairs. Without this, four of seven
   evaluation gaps and the entire phase 2 gate are unconstructible. [§2.1](#21-the-gap-axis-is-impossible-as-specified)
2. **Initialise decays log-spaced over half-lives 1–512, not at `a = 0.5`.** At the default init the gradient at gap
   128 is ~1e-39 and the traces decay just as fast. Run as written, every cell fails for a reason that isn't about
   the cell. [§4.1](#41-decay-init-is-the-experiment-not-a-detail)
3. **Add `smLLM_01-tiny` (~0.9M params) to phase 3.** Comparing 0.8M recurrent against 10.8M attention confounds
   architecture with size, and the phase 3 gate means nothing without it. One 20-minute run.
   [§3.2](#32-phase-3-is-unfair-in-two-directions-at-once)
4. **Three named phase 3 evaluation protocols (A matched-window / B best-case / C natural), with A as the gate.**
   Unlimited carried state versus a 256-byte window is not a footnote. [§3.2](#32-phase-3-is-unfair-in-two-directions-at-once)
5. **Add the byte n-gram floor.** A 0.3 bits/char allowance over smLLM_01 lands in trigram territory; the gate is not
   a gate without a floor. Seconds of CPU. [§8.1](#81-baselines-the-plan-is-missing)
6. **Calibrate the task to the budget (stage S2) before running the grid.** There is a real chance nothing learns
   binding in 2M bytes, and a table of 6% answers nothing. [§6.1](#61-the-phase-2-budget-may-be-too-small-for-any-cell-which-would-be-a-null-result-for-the-wrong-reason)
7. **Log bytes-seen alongside wall-clock and report phase 2 on both axes.** A 10-minute budget gives `online` 0.4M
   bytes and `tbptt` 16M; at equal time the comparison is about speed, not learning. Free if logged, a full rerun if
   not. [§6.2](#62-budget-by-two-axes-not-one)
8. **Gate on the Wilson lower bound, pre-register one primary endpoint, and pair every comparison.** ~91.9% observed
   at n = 1,000 for a 90% lower bound; McNemar for accuracy deltas; block bootstrap for bits/char. [§7](#7-statistics)
9. **Fix the two phase 4 confounds**: three arms for learning-at-inference (frozen / frozen-but-shown / learning-on),
   and a real-text control for the own-output trap. As written, neither measurement is attributable.
   [§8.3](#83-phase-4-has-two-confounds-both-fixable)
10. **Drop the "within 20% parameter count" rule; match width and depth instead, report parameters.** It forces widths
    of 236 / 384 / 509, which is a second independent variable, and parameter count is not the binding budget anyway.
    [§1.1](#11-parameter-counts)
11. **Hold out query keys and key-value combinations for evaluation.** At gap 8 the episode space is ~61k and a 2M-byte
    run sees ~66k episodes; fresh-from-seed is not enough. [§3.1](#31-the-recall-world-leaks-at-short-gaps)

### Should

12. **Run the staged order in [§9](#9-a-cheaper-run-order).** ~2.5–3 h instead of ~4 h, with more baselines, and the
    two stages that could falsify the whole framing come first.
13. **Tighten the phase 0 gradient check**: float64 on CPU, 32 steps, three decay settings including `a ≈ 0.99`,
    non-traced params exact, cosine reported. [§4.2](#42-the-gradient-check-needs-tighter-specification)
14. **Restate the "same loss on step 1" gate** as identical init hash + 8.00 bits/byte initial loss. As written it
    cannot hold, because the trainers see different data. [§4.3](#43-same-loss-on-step-1-is-under-specified)
15. **Add the contention canary and randomise phase 3 run order.** The README's own 6.9× contention number and 3 hours
    of sustained laptop GPU load make both necessary for the wall-clock numbers to mean anything. [§5](#5-memory-and-throughput-on-the-m5)
16. **Add the traces-off ablation and the decay-init ablation.** Two 10-minute runs that test the repo's central claim
    and the plan's own issue-#3 reproduction. [§8.2](#82-ablations-worth-the-time)
17. **Report answer-byte bits as well as accuracy**, and drive the futility rule off bits. [§2.3](#23-scoring-add-a-graded-metric)
18. **GRU/LSTM under `tbptt` is IN** (open question 2). It's 10 minutes and it separates two of the plan's four
    questions.
19. **`tests/test_eval_frozen.py`**: assert bitwise-identical weights across every evaluation. The plan's whole premise
    is that the source repo didn't control leakage. [§3.3](#33-learning-is-frozen-needs-to-be-a-test-not-a-claim)
20. **Raise the full recall sweep to n = 2,000 episodes**, keep n = 500 for in-run monitoring. Evaluation is
    forward-only and batchable; this costs minutes. [§7.1](#71-how-many-episodes)
21. **5 seeds, not 3, for any headline claim**; report min/median/max, never a t-interval from n = 3.
    [§7.3](#73-seeds)
22. **Add the pairs axis** {2, 4, 8, 16} at fixed gap 128, on survivors only. Gap alone doesn't generalise the result.
    [§2.2](#22-gap-and-capacity-are-two-different-axes-and-the-plan-only-varies-one)
23. **Add the attention baseline on the recall world** and the state-ablated control. [§8.1](#81-baselines-the-plan-is-missing)
24. **Adopt the `results.json` schema** in [§7.5](#75-what-to-log), and `git init` the repo so provenance is a sha
    rather than a file hash.

### Could

25. **Multi-query episodes** (MQAR style) to cut evaluation cost per bit of signal. [§2.4](#24-multi-query-episodes)
26. **Batch-size sweep** 1/8/32/128 for `online-batched` — tests the prediction that `fast`'s throughput is flat in
    batch size. [§1.3](#13-throughput-and-why-fast-will-not-batch-its-way-out)
27. **Repo loss term by term** rather than all-or-nothing. [§8.2](#82-ablations-worth-the-time)
28. **Two more lazy guesses**: most-recently-written value, and matched-position. [§2.5](#25-lazy-guesses-two-are-missing)
29. **Unrelated-stream control for the interference pass.** [§3.4](#34-interference-pass-needs-a-control)
30. **fp16 traces** as a named ablation, only behind a re-run fp16 gradient check. [§5](#5-memory-and-throughput-on-the-m5)
31. **Report the write/read asymmetry in `fast`** (traced `Wk`/`Wv`, exact `Wq`/`Wo`) as a candidate explanation for
    any `online-batched` vs `tbptt` gap. [§4.4](#44-the-fast-read-path-is-untraced-and-thats-a-bigger-deal-than-it-looks)
32. **Predictions with numbers and confidences**, append-only, dated. "`leaky` fails" is unfalsifiable; "`leaky` scores
    below 15% at gap 32, 80% confident" is a prediction.

---

## 11. Sources

**Network access was denied in this session** (WebSearch and WebFetch both refused permission), so none of the URLs
below were fetched or verified here. They are given from prior knowledge with direct links so a follow-up pass can
check each one. **Verify before quoting any of them in the write-up**, especially publication venues, dates, and the
M5 hardware figures, which are the most likely to be stale or wrong.

Verified locally, in this repo, on 2026-09-17:

- `docs/engineering/smModels/smLLM_01.md` — 10.8M parameters, best val 1.471 (2.12 bits/char) at step 1,500,
  ~26,100–27,000 tokens/s, 20 min on the M5, batch 64 × 256-byte windows, last 10% held out.
- `smLLM_01/train.py` — confirms the eval protocol (20 random train/val batches every 250 steps, best-val checkpoint)
  and that reported "tokens/s" is `step × batch × ctx / elapsed`.
- `smLLM_01/data/input.txt` — 1,115,394 bytes; val split is the last 111,539 bytes.
- `smALLM_01/world.py` — the world-is-the-answer-key pattern the plan copies, including the `__main__` self-check
  style that `tests/test_world.py` should follow.
- `docs/engineering/README.md` — the share-the-GPUs and don't-train-and-serve rules, and the 145.6-vs-21.1-minute
  contention measurement cited in [§5](#5-memory-and-throughput-on-the-m5).
- The repo root is not a git repository.

Unverified in this session — check each:

| Claim it supports | Source | URL |
|---|---|---|
| RTRL, the original online recurrent gradient | Williams & Zipser 1989, *Neural Computation* 1(2) | https://direct.mit.edu/neco/article/1/2/270/5490 |
| Unbiased online recurrent gradients (UORO) | Tallec & Ollivier 2017 | https://arxiv.org/abs/1702.05043 |
| Sparse RTRL approximations and their cost | Menick et al., SnAp, ICLR 2021 | https://arxiv.org/abs/2006.07232 |
| What RTRL approximations can and can't do | Irie et al., "Exploring the Promise and Limits of Real-Time Recurrent Learning" | https://arxiv.org/abs/2305.19044 |
| Online learning with diagonal/linear recurrence | Zucchet et al., "Online learning of long-range dependencies" | https://arxiv.org/abs/2305.15947 |
| MQAR, and recall as a function of state size vs pairs | Arora et al., "Zoology" | https://arxiv.org/abs/2312.04927 |
| The recall–throughput tradeoff | Arora et al., "Based" | https://arxiv.org/abs/2402.18668 |
| Fast-weight / outer-product state = linear attention | Schlag, Irie & Schmidhuber 2021 | https://arxiv.org/abs/2102.11174 |
| Gated linear attention, and gate parameterisation | Yang et al. 2023 | https://arxiv.org/abs/2312.06635 |
| Delta-rule state updates (a stronger `fast` variant) | Yang et al., DeltaNet 2024 | https://arxiv.org/abs/2406.06484 |
| Log-spaced / `A_log` decay initialisation | Gu & Dao, Mamba | https://arxiv.org/abs/2312.00752 |
| Synthetic-task-first architecture screening | Poli et al., MAD | https://arxiv.org/abs/2403.17844 |
| Learning at test time as an architecture | Sun et al., TTT layers | https://arxiv.org/abs/2407.04620 |
| Test-time memorisation | Behrouz et al., Titans | https://arxiv.org/abs/2501.00663 |
| nanoGPT shakespeare-char reference (~1.47 val) | karpathy/nanoGPT | https://github.com/karpathy/nanoGPT |
| Wilson over normal-approximation intervals | Brown, Cai & DasGupta 2001 | https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full |
| McNemar for paired classifier comparison | Dietterich 1998 | https://direct.mit.edu/neco/article/10/7/1895/6224 |
| Moving-block bootstrap for dependent data | Künsch 1989, *Annals of Statistics* 17(3) | https://projecteuclid.org/journals/annals-of-statistics/volume-17/issue-3/The-Jackknife-and-the-Bootstrap-for-General-Stationary-Observations/10.1214/aos/1176347265.full |
| False discovery rate control | Benjamini & Hochberg 1995 | https://www.jstor.org/stable/2346101 |
| MPS backend limits (no fp64, compile maturity) | PyTorch MPS notes | https://pytorch.org/docs/stable/notes/mps.html |
| The source repo under audit | jrz97619761/test-model-thing | https://github.com/jrz97619761/test-model-thing |
