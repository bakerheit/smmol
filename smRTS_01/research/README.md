# smRTS_01 research index

Research pass completed 2026-09-17 by three independent Claude Opus 5 agents at medium effort,
then source-checked against the upstream repository and primary papers. No model code was written
and no training was started. The agents cost $5.80612 total.

## Memos

- [trace_math.md](trace_math.md) — exact trace derivations, missing equations, tensor shapes,
  memory costs, multi-layer bias, and a corrected Phase 0 block.
- [architecture_literature.md](architecture_literature.md) — fast weights, DeltaNet, MQAR,
  recurrent/linear sequence models, test-time learning, and the fair cell set.
- [experiment_design.md](experiment_design.md) — world construction, leakage, statistics,
  M5 cost estimates, fair baselines, stopping rules, and a cheaper staged run order.

The memos preserve each agent's independent reasoning. Their warnings that web access was blocked
describe the agent sessions, not this index: the core claims below were checked afterward.

## Strong consensus

The plan should not be implemented as written. The three reviews independently found the same two
fatal confounds:

1. **The recall world cannot produce its headline gaps.** Sixteen distinct keys and four bytes per
   `k=v;` pair only create about 62 bytes of pair-based gap, but the gate is at 128 and evaluation
   runs to 512. Gap must be separated from capacity by inserting neutral filler bytes and varying
   pair count independently.
2. **The default decay starts with a one-byte half-life.** `sigmoid(0) = 0.5`; at gap 128 its
   contribution is about `0.5^128`. Long-range credit is effectively gone before training can
   discover a long timescale. Initialise half-lives log-uniformly over the tested range and report
   the initial distribution as part of the experiment.

Other changes supported by at least two reviews:

- Make the one-layer trace test genuinely exact: float64, a long sequence, `a` near 0.99, relative
  error, state detached every step, and no ordinary-autograd contribution for traced parameters.
- Resolve the layer-0 embedding path. The plan says every recurrent parameter must match BPTT but
  does not trace the embedding for the gated and fast cells. A one-hot byte input into the write
  projections is the cleanest cheap fix.
- Define the fast-state decay shape and its trace. Per-head scalar decay is far cheaper than a full
  `dk x dv` decay and retains an exact collapsed trace.
- Normalise fast-weight keys/queries and the readout. Otherwise numerical scale is a confound.
- Add a delta-rule memory. Additive outer products cannot cleanly overwrite an existing key; this
  is the known limitation DeltaNet addresses.
- Match recurrent state capacity first and report parameters second. The current 0.8M-parameter
  matching gives the fast cell several times more state than the gated cell and also changes model
  width substantially.
- Add a GRU/TBPTT control, a simple attention ceiling for recall, TBPTT-window-1, and a roughly
  0.9M-parameter text baseline. The 10.8M smLLM_01 comparison alone is not fair.
- Gate on confidence bounds, not a point estimate. With 1,000 recall episodes, 90% observed does
  not establish 90% underlying accuracy.
- Separate bytes seen from wall time. The online and TBPTT regimes can see tens of times different
  amounts of data under the same clock budget.
- Replace "theoretically infinite memory" with two measurable claims: constant resource use per
  byte, and effective retained-pair capacity versus state size and delay.

## Recommended staged order

1. Fix the world and trace gates before writing the full trainers.
2. Run a frozen-decay leaky/TBPTT falsification at half-life 128. If it binds, the plan's claim that
   a leaky cell cannot bind is wrong; if it does not, the mechanism claim survives a fair test.
3. Run one uncapped TBPTT calibration until the recall task clearly learns. This separates an
   under-budget grid from an architectural failure.
4. Compare one-layer leaky, gated, additive-fast, and delta cells at matched state sizes.
5. Only promote configurations that beat the lazy baselines and approach the attention ceiling.
6. Run multi-layer and online-gradient variants after the exact one-layer result is known.
7. Run Shakespeare only for surviving cells, with matched-size and matched-window controls.

This order is estimated at roughly 2.5 to 3 hours of machine time while answering more cleanly than
the original flat four-hour grid.

## Source verification

Checked after the agent runs:

- The upstream repository is MIT, claims test-time training and "theoretically infinite memory",
  and explicitly asks for review and attribution:
  https://github.com/jrz97619761/test-model-thing
- Issue 2 reports exact one-layer traces and increasing bias in earlier layers of a deep stack:
  https://github.com/jrz97619761/test-model-thing/issues/2
- Issue 3 confirms zero logits initialise a one-symbol half-life and reports improved long-range
  behaviour from a spread of initial timescales:
  https://github.com/jrz97619761/test-model-thing/issues/3
- Issue 4 confirms benchmark state leakage and scoring on the training stream:
  https://github.com/jrz97619761/test-model-thing/issues/4
- Schlag et al. identify additive fast-weight capacity limits and introduce a delta-rule update:
  https://arxiv.org/abs/2102.11174
- DeltaNet reports better associative recall from delta updates and a parallel training method:
  https://arxiv.org/abs/2406.06484
- Zoology defines MQAR and ties much of the efficient-model gap to associative recall:
  https://arxiv.org/abs/2312.04927
- The correct identifier for Zucchet et al., *Online learning of long-range dependencies*, is
  **arXiv:2305.15947**:
  https://arxiv.org/abs/2305.15947
- **arXiv:2305.19044** is a different and also relevant paper: Irie et al., *Exploring the
  Promise and Limits of Real-Time Recurrent Learning*:
  https://arxiv.org/abs/2305.19044

The long memos contain additional citations that were not all re-fetched. Treat any citation they
explicitly mark uncertain as a lead until it gets the same primary-source check.

## Decision before implementation

Revise `docs/engineering/plans/smRTS_01.md` from these findings before Phase 0. The current document
is still untouched so the research remains auditable against the original plan.
