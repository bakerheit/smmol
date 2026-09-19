# smRTS_01 plan: a recurrent state that learns online, tested against its claims

**Status: Phases 0-1 complete; Phase 2 paused after a partial S2. Phase 3 not started.** Written and research-reviewed
2026-09-17. The research bundle is [smRTS_01/research/README.md](../../../models/smRTS_01/research/README.md). Each phase has
deliverables, a gate, and what to write down. A phase that fails its gate is still a result, and it goes in
[results.md](../research/results.md).

Phase 0's evidence is in [phase0_results.md](../../../models/smRTS_01/phase0_results.md). The one-layer online traces pass
the float64 and constant-memory gates. Two- and four-layer online gradients are substantially biased, so the main
recall comparison starts with one recurrent layer and treats deeper online models as ablations.

**Phase 2 state, 2026-09-17.** Predictions were registered in
[predictions.md](../../../models/smRTS_01/predictions.md) before any run; evidence is in
[phase2_results.md](../../../models/smRTS_01/phase2_results.md).

- **S1 complete.** `leaky-tbptt` with decay frozen at half-life 128 scored 0.053, Wilson
  [0.041, 0.069] at `(pairs=16, gap=128)` over 121M training bytes. Its answer NLL parks at `ln 33`:
  it learns that the answer is a value byte and nothing more, and it loses to lazy guessing in every
  reported cell. Prediction P1 held; P2 was falsified. No promotion.
- **S2 interrupted.** `fast-tbptt` was stopped by the owner at ~25.5 min, short of its registered
  60-minute ceiling, so **no promotion decision exists** and P3/P4 are untested. Its best checkpoint
  scored 0.087, Wilson [0.071, 0.106], also below every lazy arm.
- **S3 not started.** Its precondition was an S2 promotion that never occurred. No extra seeds, no
  `--overwrite`, no full grid, no Phase 3.
- **A design mismatch found, not fixed.** A one-layer `FastCell` writes `k(x_t) ⊗ v(x_t)` from the
  *same* byte, while the world binds a key byte to the value byte that follows it. The cell has no
  mechanism to associate a key with the next byte, and its measurements read as set membership plus
  recency rather than recall. The same objection applies to `delta` (it inherits that write) and to
  `leaky`/`gated`. Giving any of them a delayed or contextual key changes the recurrence and so
  changes the trace: it needs its own derivation and its own **Phase 0 gradient gate** before it
  could be called an online-trace cell, on the same terms [section 5](#phases) already applies to
  `delta`. Nothing in `cells.py` was changed, and no such variant was built or trained.

RTS is read here as **recurrent trace state**: a model with an internal state instead of a context window, trained
one byte at a time with eligibility traces instead of backprop through time. Rename if the owner meant something else.

Contents:

1. [The short version](#1-the-short-version)
2. [Where this comes from, and what's wrong with it](#2-where-this-comes-from-and-whats-wrong-with-it)
3. [The questions this run answers](#3-the-questions-this-run-answers)
4. [Decisions already made](#4-decisions-already-made)
5. [Phases](#5-phases)
6. [Project layout](#6-project-layout)
7. [Ground rules that apply](#7-ground-rules-that-apply)
8. [Open questions for the owner](#8-open-questions-for-the-owner)
9. [Rough budget](#9-rough-budget)

---

## 1. The short version

A public repo claims a tiny byte model with no context window, trained one byte at a time with no backprop through
time, has "theoretically infinite memory" and keeps learning while it runs. The repo has no numbers, and its own
issue tracker shows the memory doesn't exist in the current cell. But the question under it is SMMOL's question: can a
small model learn to remember while training online, with constant memory, and what does that cost per bit next to
ordinary batched training?

smRTS_01 answers it the SMMOL way. Three cells with exact one-layer online traces, delta-rule and GRU controls under
TBPTT, three training regimes, two tasks where the world is the answer key, honest matched-capacity baselines, and
predictions written down before the runs. "Infinite memory" is not a result this project can produce: it measures
constant resource use per byte and effective retained-pair capacity instead. No harness integration is planned;
[section 5](#phase-5-write-up) says what would justify one.

## 2. Where this comes from, and what's wrong with it

Source: [jrz97619761/test-model-thing](https://github.com/jrz97619761/test-model-thing), MIT, read on 2026-09-17
(`main.py`, 252 lines; `benchmark.py`, 78 lines). Credit the author in the README; they asked for it.

**What the code does.**

- Bytes in, bytes out. Each layer keeps a state vector `s` that's a leaky sum of the residual stream:
  `s = sigmoid(d) * s + x`, then `x + SiLU(W · LayerNorm(s))`. Linear decoder to 256 logits plus a stop head.
- One byte per optimizer step, batch size 1. No backprop through time. The decay parameters and the byte embedding
  keep eligibility traces, an approximate real-time recurrent learning update. Everything else gets the current
  step's gradient only.
- Loss: next-byte cross-entropy, plus an MSE between the current latent and the *embedding of the next byte*, plus a
  variance hinge against collapse, plus the stop loss. The MSE is next-byte prediction a second time, not a separate
  latent objective.
- Chat mode keeps training on the user's bytes and on its own output.

**What's wrong**, from its issues and from reading the cell:

| | |
|---|---|
| No results | no loss curve, no bits per byte, no comparison, no weights |
| [#3](https://github.com/jrz97619761/test-model-thing/issues/3) | the learned decays give a half-life of about one byte: no measurable long-range memory |
| [#4](https://github.com/jrz97619761/test-model-thing/issues/4) | the benchmark scores on its own training stream, with state leaking across examples |
| [#2](https://github.com/jrz97619761/test-model-thing/issues/2) | the trace is exact for one layer, approximate for a stack |
| The cell itself | the state is a decaying sum of inputs with no learned write and no state-times-input interaction. It can't bind a key to a value, so it can't do associative recall, whatever the training. That's a prediction this plan tests |
| Learning on its own output | [whats-next.md](../../product/whats-next.md) already rules this out for the Learn module: "if it learned from its own guesses, it would teach itself its own mistakes". Phase 4 measures it instead of arguing |

## 3. The questions this run answers

1. **Can a recurrent byte model trained online, one byte per step with traces and no backprop through time, learn to
   remember?** Measured as associative recall at gaps from 8 to 512 bytes.
2. **Which minimal state update makes the difference?** The repo's leaky integrator, a gated one, an additive
   fast-weight state, or a delta-rule memory that can overwrite an existing key.
3. **What does online learning cost per bit** next to truncated backprop through time and next to smLLM_01, on the
   same Shakespeare text, same 20 minutes, same Mac?
4. **Does learning at inference help or hurt,** and what does training on its own output do to a model?

## 4. Decisions already made

- **PyTorch, not MLX.** MLX isn't installed, every SMMOL model is PyTorch, and the harness's loader rule is CPU
  PyTorch. The port is about 60 lines.
- **State capacity is the primary matching axis.** Parameter counts and model width are reported, not forced to
  match. The old 0.8M-parameter rule gave `fast` several times more recurrent state than `gated` and changed width at
  the same time.
- **Three online-trace cells plus controls.** `leaky`, `gated`, and additive `fast` use the exact diagonal/row-local
  traces proved in Phase 0. A delta-rule cell and a normal GRU run under TBPTT only. Delta's transition is
  non-diagonal; it does not get called an online-trace cell until a separate derivation passes the same gradient gate.
- **Timescales are part of the experiment.** The fixed design initialises half-lives log-uniformly over the tested
  range. A zero-logit, one-byte-half-life arm remains only as the faithful source-repo reproduction.
- **Cross-entropy on the next byte is the loss.** The repo's four-term loss is behind a `--repo-loss` flag for one
  comparison run in phase 3, not the default.
- **The world is the answer key**, like smALLM_01. Recall episodes are generated fresh; nothing is hand-labelled.
- **Predictions are written before the runs** in `predictions.md`, and reported next to the results.
- **No chat mode. No learning on the model's own output**, except the measured trap in phase 4.
- **Learning is frozen during every evaluation** unless the evaluation is explicitly about learning at inference.
- **Keep the best checkpoint by the task metric,** not the last.
- **No harness integration in this plan.**
- **Staged runs, not a flat grid.** Cheap falsifications and a TBPTT calibration happen before the long grid. Every
  report includes both wall time and bytes seen.

## 5. Phases

### Phase 0: port the cells and prove the traces

The repo never checked that its online gradient equals the true gradient. This phase does, before anything is trained.
The detailed derivation and tensor audit are in
[trace_math.md](../../../models/smRTS_01/research/trace_math.md).

**The common recurrence and the gradient that is actually needed.** For an online-trace cell,
`s_t = a ⊙ s_{t-1} + b_t`, `a = sigmoid(decay_logit)`. Define
`g_t = ∂L_t/∂s_t` as the **immediate partial**: the state is a leaf and no future loss is in its graph. It is not the
total BPTT derivative. With sensitivity `E_θ = ∂s_t/∂θ`:

```text
E_θ,t = a ⊙ E_θ,t-1 + ∂b_t/∂θ
grad θ += g_t · E_θ,t
```

This is exact for one layer only when (1) the layer input does not depend on a past state, (2) every learned
parameter in the write is traced, and (3) the state and write parameters are severed from the live autograd graph.
The implementation uses this safe ordering:

```text
under no_grad: update the numeric state and every trace
state_leaf = numeric_state.detach().requires_grad_(True)
read current output and loss from state_leaf
backward current loss; g_t = state_leaf.grad
add trace gradients only to traced parameters; ordinary autograd owns read-only parameters
```

For stacked cells, the current-step input bridge remains live but every previous state is detached. The missing path
from a lower layer's past state through an upper layer is the measured approximation from upstream issue #2. Call it
depth-truncated RTRL; do not describe a multi-layer online gradient as exact.

**Inputs.** `leaky` keeps the source repo's learned byte embedding and traces its state-write path. At layer 0,
`gated` and `fast` consume a 256-way one-hot byte directly. A learned embedding there would enter their recurrent
write without a tractable row-local trace and would make the one-layer gate fail by construction. All write
projections use `bias=False`.

**The cells** (`cells.py`):

| Cell | Write and state | Read | Online traced parameters |
|---|---|---|---|
| `leaky` | `s = a⊙s + x` | `x + SiLU(W·LN(s))` | decay; layer-0 byte embedding |
| `gated` | `u=Vx`, `g=sigmoid(Gx)`, `s=a⊙s+g⊙u` | projected residual plus `SiLU(W·LN(s))` | decay, `V`, `G` |
| `fast` | per head `S=aS+k⊗v`; scalar decay per head | `RMSNorm(Sᵀq)`, then `Wo` and a projected residual | decay, `Wk`, `Wv` |
| `delta` | decay plus erase-then-write delta update | same normalised keyed read | **TBPTT control only** |
| `gru` | ordinary GRU recurrence | hidden state | **TBPTT control only** |

The delta cell is required as the known fix for additive fast-weight overwrite, but its transition is
non-diagonal. It does not borrow `fast`'s trace formulas. An online delta cell is a later phase only after a separate
derivation and gradient gate.

**Closed-form traces.** Write `g_t` for the immediate state gradient and omit batch/head indices:

```text
decay: E_d = a E_d + a(1-a) s_prev                         grad d += g_t E_d

leaky layer-0 embedding:
  T[c,m] = a_m T[c,m] + 1[current_byte=c]                  grad Emb[c,m] += g_t[m] T[c,m]

gated:
  E_V[i,k] = a_i E_V[i,k] + gate_i x_k                    grad V[i,k] += g_t[i] E_V[i,k]
  E_G[i,k] = a_i E_G[i,k] + gate_i(1-gate_i)u_i x_k       grad G[i,k] += g_t[i] E_G[i,k]

fast, scalar decay per head, S shape dk × dv:
  E_d[i,j] = a E_d[i,j] + a(1-a) S_prev[i,j]              grad d += Σij g_t[i,j] E_d[i,j]
  E_Wk[j,m] = a E_Wk[j,m] + v_j x_m                       grad Wk = g_t @ E_Wk
  E_Wv[i,m] = a E_Wv[i,m] + k_i x_m                       grad Wv = g_t.T @ E_Wv
```

The scalar-per-head fast decay is exact and collapses trace memory by 16× relative to a full `dk × dv` decay.
Use at least four different head timescales. The read is normalised so output magnitude cannot reveal how many writes
occurred.

**Decay initialisation.** The fixed design spreads half-lives log-uniformly over `[1, 1024]` bytes:

```text
h = 2 ** linspace(0, 10, channels)
a = 2 ** (-1 / h)
decay_logit = log(a / (1-a))
```

Keep a separate `decay_logit=0` reproduction arm. Do not mix it into the fixed model.

**Sizes.** Compare at two matched recurrent-state capacities. Report state scalars, parameters, trace bytes per
stream, peak memory, and width. The primary 384-scalar/layer row uses vector width 384 versus `fast` with 4 heads,
`dk=8`, `dv=12` (`4×8×12=384`). A `dk=32` arm is an explicitly larger capacity ablation, not a row called matched.
The Phase 0 unit models stay tiny (`dim=8`) because they prove gradients, not performance.

**The trainers.**

| Regime | What it is | Device |
|---|---|---|
| `online` | batch 1, one optimizer step per byte, traces. The repo's regime | CPU, and MPS once to show the launch overhead; report both |
| `online-batched` | 32 independent streams, each with its own state and traces, one optimizer step per byte column, gradients summed over streams. Still no backprop through time. The obvious fix the repo doesn't have | MPS. If a cell's traces don't fit at 32, use the largest batch that does and report it |
| `tbptt` | truncated backprop through time: window 256, batch 64, state carried across windows and detached. The control, and the closest thing to smLLM_01's setup with a recurrent model | MPS |

`trace_bytes = bytes_per_float × streams × sum(trace tensor elements)` is printed before every online run. Exactness
at fixed parameters does not survive an optimizer step perfectly because old sensitivities were computed under old
weights; Phase 0 records a moving-optimizer cosine as a diagnostic, not a gate.

**Deliverables.** `cells.py`, `online.py`, `tbptt.py`, `tests/test_traces.py`, and `phase0_results.md`.

**Gate.**

- **The gradient check.** For `leaky`, `gated`, and `fast`: one layer, dim 8, float64, at least 32 bytes with every
  decay set to `a=0.99`, no optimizer step. Compare each learned parameter with full BPTT using relative L2 error;
  every group must be below `1e-6`. Repeat at 128 bytes specifically to catch a missing trace carry term. A
  deliberate no-carry mutant must fail. Report cosine and relative error for 2 and 4 layers; a nonzero gap is
  expected and is not a gate.
- **No double count.** Traced parameters receive no ordinary write-path gradient. The test checks their trace/BPTT
  norm ratio; the characteristic near-2× failure is a hard error.
- **Constant memory.** 10,000 online steps after warm-up: every state and trace has `grad_fn is None`, Python tensor
  counts do not grow, and current allocated memory/RSS stays within a documented allocator-noise bound.
- **Same first token.** From identical weights, bytes and initial state, online, online-batched stream 0, and TBPTT
  produce the same logits and per-token loss to `1e-9` in float64.
- **Controls.** `delta` and `gru` pass forward/state-shape and TBPTT gradient smoke tests but are explicitly excluded
  from the online exactness claim.

**Write down.** Parameter/state/trace sizes per cell, gradient error by parameter group and depth, memory slope over
10,000 steps, first-token parity, and online CPU versus MPS step time.

### Phase 1: the recall world

`world.py`, in the style of `smALLM_01/world.py`: episodes generated on the spot from a seed, never repeated.

- **Alphabet:** at least 32 one-byte keys and 32 one-byte values, drawn from disjoint printable sets. The old
  16-key alphabet capped capacity at 16 pairs and put `fast` exactly on its `dk=16` interference boundary.
- **An episode:** pairs written as `k=v;`, distinct keys, neutral filler, a query `?k`, the answer byte, and `\n`.
  The **gap** is defined once as `query_at - queried_equals_at - 1`. Filler, not extra pairs, opens long gaps. Pair
  count and gap are independently selectable axes: `pairs ∈ {2, 4, 8, 16, 32}` and
  `gap ∈ {8, 32, 128, 512}`. The queried pair's position is sampled uniformly from the positions feasible at that
  exact gap. This is a measured limitation, not hidden randomness: `(pairs=32, gap=8)` can only query ordinals 30 or
  31, gap 32 can query the last eight positions, and gaps 128/512 can query every position. Filler bytes exclude
  key/value and syntax bytes and are freshly sampled so a fixed padding pattern is not a shortcut.
- **Scoring:** the answer byte only, like the math models. Everything else in the stream is context.
- **Lazy guesses,** measured on the same episodes: uniform random, most common value, last pair, first pair, and the
  value at each fixed pair position. Short-gap leakage is visible instead of assumed away.
- **Evaluation:** 1,000 paired fresh episodes per `(pairs, gap)` cell, learning frozen, state reset per episode. Report
  Wilson confidence intervals. A second pass carries state across episodes and plots accuracy against
  episodes-since-reset; a reset-at-random control distinguishes interference from episode difficulty. Carrying state
  makes episode outcomes dependent, so Wilson intervals apply only to the primary reset-per-episode pass.
- **Metrics:** exact answer accuracy is primary. Also report answer negative log likelihood and rank so a model that
  moves from chance toward the answer is visible before the hard threshold.
- **Variant, off by default:** `--overwrite`, where the queried key is written twice and the newest value is right.
  Only run it if a cell passes the main gate.

**Gate.** `tests/test_world.py`: every episode's answer is determinable from its bytes; every requested gap is
constructible and exact; pair count and gap vary independently; keys are distinct; filler cannot parse as a pair;
and lazy-guess rates match the construction.

**Write down.** The lazy-guess rates by gap. Phase 1 evidence is in [`smRTS_01/phase1_results.md`](../../../models/smRTS_01/phase1_results.md).

### Phase 2: recall runs

**Before running anything,** write `predictions.md`. The ones this plan commits to:

- `leaky` fails at high pair counts even with a half-life of 128; the first staged run tries to falsify this before
  the grid.
- Additive `fast` degrades as pair count approaches state capacity; `delta` under TBPTT handles overwrites and higher
  pair counts better.
- `fast` with `online-batched` is the open question. If it passes at matched state capacity, that is the finding.
- `gated` lands between `leaky` and keyed memories.
- Every online regime is much slower per byte than `tbptt`.

**Staged order before the grid.** (1) `leaky-tbptt` with decay frozen at half-life 128; (2) one uncapped
`fast-tbptt` calibration until the world demonstrably learns; (3) one-layer matched-state `leaky`, `gated`, `fast`,
and `delta-tbptt`, with GRU and attention ceilings. Stop if the task or implementation cannot beat lazy baselines.

**The promoted grid.** Only surviving cells get all eligible regimes. Online cells are `leaky`, `gated`, and `fast`;
`delta` and `gru` remain TBPTT controls. Run one seed first, then three paired seeds for any configuration whose 95%
Wilson lower bound clears the promotion threshold. Save `out/<cell>-<regime>/ckpt.pt`, `log.csv`, `results.json`,
and the exact world/config manifest. Report both fixed-wall-time and fixed-bytes comparisons; do not let a 30–60×
difference in bytes seen masquerade as an optimizer result.

**Also measure:** for every trained model, the decay half-lives per layer, `ln 0.5 / ln a` in bytes, median and 90th
percentile, before and after training. That reproduces issue #3 here.

**Gate, "what would count as working":** any online regime whose 95% Wilson lower bound is at least 90% at
`pairs=16, gap=128`, with learning frozen and state reset. With 1,000 trials this requires more than a 90% point
estimate. The phase is complete when every promoted row is reported, pass or fail.

**Write down.** Accuracy/NLL/rank over the 2-D pair-count × gap grid next to lazy and attention ceilings; confidence
intervals; bytes seen and bytes/second; half-lives; state capacity; and which predictions held.

### Phase 3: Shakespeare, bits per char, and the cost per bit

Same data as smLLM_01: `smLLM_01/data/input.txt`, the last 10% held out. Same 20-minute budget on the
MacBook Pro M5. Same reference numbers, from [smLLM_01.md](../smModels/smLLM_01.md):

| smLLM_01 | |
|---|---|
| Parameters | 10.8M, 256-byte context |
| Best val loss | 1.471, **2.12 bits/char** |
| Throughput | about 26,700 training bytes/s |

**Runs:** `leaky-online` (batch 1, to measure the repo's actual cost), then `online-batched` and `tbptt` for all
promoted online cells. Include `delta-tbptt`, `gru-tbptt`, a roughly 0.9M-parameter `smLLM_01-tiny`, and the original
10.8M smLLM_01 reference. Plus one `leaky-online-batched --repo-loss` run to see whether the repo's four-term loss
changes anything.

**Evaluation for stateful models:** three named protocols: state reset every 256 bytes (matched-window gate), state
carried through the held-out stream (unlimited-state advantage, report only), and state reset per document boundary.
Learning is frozen and asserted by hashing parameters before/after.

**Report per run:** parameters, bits/char on val, bytes/s, peak memory, and the half-life table. Then the two numbers
that answer the cost question: **bits/char reached in 20 minutes** and **seconds per 0.1 bit of improvement** from the
2 bits/char mark, for each regime.

**Gate, "what would count as working":** under the matched-window protocol, an online regime within 0.3 bits/char of
`smLLM_01-tiny` at one tenth or more of its throughput. The full smLLM_01 stays in the table but is not the size-matched
gate. Prediction: batch-1 online is roughly 50 times slower.

**Write down.** The table, the samples (a few lines from each best checkpoint at temperature 0.8, like smLLM_01's
README), and what the repo loss did.

### Phase 4: learning at inference, and the own-output trap

Two short measurements with the best online model from phase 2 and the best text model from phase 3.

1. **Learning at inference, with real answers.** Recall at `pairs=16, gap=128` over paired episodes in three modes:
   learning frozen; learning on after seeing the true answer; and a frozen model that sees the same answer bytes.
   The third arm separates extra information from weight updates. Report early/late accuracy and compare with dynamic
   evaluation of the matched recurrent/text baseline.
2. **The own-output trap.** From identical checkpoints generate 5,000 bytes in four paired arms: frozen; train on
   true held-out bytes; train on sampled output; and train on sampled output with the same conservative LR/batching as
   true-byte adaptation. Measure validation bits/char before/after. This separates self-teaching damage from generic
   batch-1 high-LR damage.

**Gate.** None. Both numbers are the deliverable.

### Phase 5: write-up

- `smRTS_01/README.md` in the house format: the idea, the source repo and credit, the cells, the world, results
  tables, what it means, next, run it, files.
- [smModels/smRTS_01.md](../smModels/), [results.md](../research/results.md) with dated entries for phases 2, 3 and
  4, [running.md](../smModels/running.md)'s project table, [models.md](../../product/smModels/models.md) with the
  plain-English version, and a row in [whats-next.md](../../product/whats-next.md).
- **Negative results stay** under `out/`, as evidence, the way `smLANGUAGE_RENDER_001/out/v2/` does.
- **What would justify a harness integration,** and it isn't planned here: an `online-batched` cell that meets the
  phase 3 gate. Then the first candidate would be a conversation-state module that carries a state across turns
  instead of re-reading the chat, measured against the harness's existing conversation view. Not before.

## 6. Project layout

```text
smRTS_01/
  README.md
  research/           independent trace, architecture and experiment audits; synthesis in research/README.md
  phase0_results.md   trace exactness, depth gap, state/trace sizes, constant-memory evidence
  predictions.md      written before phase 2 runs; reported next to results
  cells.py            leaky, gated, additive-fast, delta and GRU cells
  online.py           the online trainer: one byte per step, a batch of streams, traces
  tbptt.py            truncated backprop through time, the control
  world.py            the recall world, like smALLM_01/world.py
  train_recall.py     --cell {leaky,gated,fast} --regime {online,online-batched,tbptt} --minutes 10
  train_text.py       Shakespeare, same flags, --minutes 20, --repo-loss
  evaluate.py         recall by gap, bits/char, half-lives, the phase 4 measurements -> out/eval.md
  tests/              test_traces.py (exactness, no graphs, parity), test_world.py, test_scoring.py
  out/                one folder per cell-regime: ckpt.pt, log.csv, results.json
```

## 7. Ground rules that apply

From [the engineering README](../README.md):

- **One GPU job at a time on the Mac,** and **don't train and serve on the same box.** Before the timed runs, point
  the harness's prompted modules at `pc` and unload the Mac's model. Every number in phases 2 and 3 is a wall-clock
  number; a run that shared the Mac with a loaded LLM is an upper bound and says so.
- **Keep the best checkpoint, not the last.**
- **`out/` is overwritten by training.** Copy anything worth keeping first.
- The repo's code is MIT; the port credits it.

## 8. Open questions for the owner

1. **The name.** RTS is read as recurrent trace state. Correct it if that was not what was meant.
2. **Online delta trace.** Delta is now a required TBPTT control. Promoting it to online training needs a separate
   non-diagonal trace derivation and Phase 0 gate; it is not silently assumed.
3. **Final compute budget.** The staged design should take about 2.5–3 hours before extra seeds. Stop after each stage
   unless its promotion rule passes.

## 9. Rough budget

Guesses.

| Phase | Machine time | Person or agent time |
|---|---|---|
| 0: port and prove the traces | CPU minutes | a day; the gradient check is the work |
| 1: the world | minutes | half a day |
| 2: staged recall | about 60–90 min before extra seeds | read the promotion table after each stage |
| 3: promoted Shakespeare rows only | about 90 min | reading the table and samples |
| 4: inference learning and the trap | 30 min | an hour |
| 5: write-up | none | 2 hours |
