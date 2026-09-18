# smRTS_01: architecture and literature review

**What this is.** A review of the [smRTS_01 plan](../../docs/engineering/plans/smRTS_01.md) against the primary
literature on fast weights, linear attention, state-space models, delta-rule memories, real-time recurrent learning,
and test-time learning. It answers four things the owner asked for: can the plan actually test associative recall,
can it test "theoretically infinite memory", what the minimal scientifically fair cell set is, and which baselines
matter. It ends with ranked must / should / could changes.

Written 2026-09-17. Nothing here was implemented or trained.

---

## 0. Verification status — read this first

**This session had no network access.** `WebFetch`, `WebSearch` and `curl` were all unavailable, so I could not open
the source repo, its issues, or any paper. Everything below is written from knowledge of the primary literature, and
every citation is given as a direct URL so it can be checked in one click.

| Claim class | Status |
|---|---|
| smLLM_01 reference numbers in the plan | **verified locally.** `docs/engineering/smModels/smLLM_01.md`: 10.8M parameters (line 27), best val 1.471 = 2.12 bits/char (line 59), 256-byte windows batch 64 (line 33), "about 26,100–27,000 training tokens/s" (line 57). The plan's "about 26,700 bytes/s" is inside that band. Fine |
| The source repo's code, line counts, and issues #2/#3/#4 | **not verified.** No network. The plan's reading of the cell (`s = sigmoid(d)*s + x`, four-term loss, chat mode) is internally consistent and matches the failure modes it describes, but someone with a browser should confirm before phase 0 |
| Paper claims and arXiv IDs below | **not re-verified this session.** Treat the URLs as pointers to check, not as quoted evidence. Where I am less than certain of an ID I say so inline |

Two things a networked agent should do before phase 0, in this order: (1) confirm the three issue bodies say what
section 2 of the plan says they say; (2) confirm the repo is still MIT and the author's attribution request is where
the plan says it is.

---

## 1. Where this sits in the literature

The plan's three cells are not new inventions. Each one is a named, published point in a well-mapped design space,
and that is good news: it means every cell has a known failure mode and a known fix, and the plan can be graded
against them instead of rediscovering them at 10 minutes a run.

**The common form.** Every cell in the plan is a *linear recurrence with a diagonal (elementwise) transition*:

```text
s_t = a ⊙ s_{t-1} + b_t          a = sigmoid(d), learned, input-independent
y_t = read(s_t, x_t)
```

That is exactly the family that linear attention, RetNet, and the non-selective half of the SSM literature live in.

| Plan cell | Published name | Primary source |
|---|---|---|
| `leaky` | an exponential-moving-average / leaky integrator over the residual stream. The degenerate case of a diagonal linear RNN with no input-dependent write | classical; nearest modern framing is the LRU / diagonal SSM family, [Orvieto et al., *Resurrecting Recurrent Neural Networks for Long Sequences*, arXiv:2303.06349](https://arxiv.org/abs/2303.06349) |
| `gated` | an input-gated diagonal RNN; the write is `sigmoid(Gx) ⊙ (Vx)`, i.e. an LSTM input gate with no forget-gate coupling and no cell nonlinearity | [Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997](https://direct.mit.edu/neco/article/9/8/1735/6109); the gating-ablation lineage is [Jozefowicz et al., *An Empirical Exploration of RNN Architectures*, PMLR 2015](https://proceedings.mlr.press/v37/jozefowicz15.html) |
| `fast` | **sum-update (Hebbian) fast weights** = unnormalised linear attention with a decay. `S_t = a S_{t-1} + k ⊗ v`, read `Sᵀq` | [Schmidhuber, *Learning to Control Fast-Weight Memories*, Neural Computation 1992, doi:10.1162/neco.1992.4.1.131](https://doi.org/10.1162/neco.1992.4.1.131); [Katharopoulos et al., *Transformers are RNNs*, arXiv:2006.16236](https://arxiv.org/abs/2006.16236); with decay this is [Sun et al., *Retentive Network*, arXiv:2307.08621](https://arxiv.org/abs/2307.08621) |

So the plan's cell axis is really: **no write mechanism → gated write → outer-product write**. That is a sensible
ladder. What is missing from it is the rung that the literature says is the one that matters for associative recall,
which is section 3.

**The training axis** is equally well named. One optimizer step per byte with eligibility traces on a diagonal
recurrence is **RTRL restricted to a diagonal state**, which is the same object as e-prop:

- [Williams & Zipser, *A Learning Algorithm for Continually Running Fully Recurrent Neural Networks*, Neural Computation 1989, doi:10.1162/neco.1989.1.2.270](https://doi.org/10.1162/neco.1989.1.2.270) — RTRL, and the O(n³)/O(n²) cost that makes it impractical in general.
- [Bellec et al., *A solution to the learning dilemma for recurrent networks of spiking neurons*, Nature Communications 11:3625, 2020](https://www.nature.com/articles/s41467-020-17236-x) — e-prop: exactly "eligibility trace × immediate learning signal", with the cross-neuron path dropped. The plan's approximation (cutting the path from a layer's input back through lower layers' past states) is e-prop's approximation, one level up.
- [Tallec & Ollivier, *Unbiased Online Recurrent Optimization*, arXiv:1702.05043](https://arxiv.org/abs/1702.05043) — UORO, the unbiased-but-noisy alternative if the owner ever wants the cross-layer path back without storing a full Jacobian.
- [Menick et al., *A Practical Sparse Approximation for Real Time Recurrent Learning*, arXiv:2006.07232](https://arxiv.org/abs/2006.07232) — SnAp; the sparsity pattern the plan is implicitly using is SnAp-1. *(ID believed correct; verify.)*

**The key structural fact, and it is in the plan's favour:** for a *diagonal* recurrence, RTRL is not an
approximation at all within a layer — the eligibility trace is the exact Jacobian, because the state's transition
matrix is diagonal so the Jacobian stays factored. This is the same property that lets Mamba and GLA train with a
parallel scan. The plan's phase 0 gradient check should therefore pass *exactly* for 1 layer, and the "2-layer gap"
it reports is purely the cross-layer cut. That is worth saying out loud in the write-up, because it is the honest
answer to "is this just a hack": it is not, it is exact online learning of a restricted model class.

**The test-time-learning axis** has a large recent literature that the plan does not cite and should:

- [Sun et al., *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*, arXiv:2407.04620](https://arxiv.org/abs/2407.04620) — TTT layers: frame the recurrent state update itself as a gradient step on a self-supervised loss.
- [Behrouz, Zhong & Mirrokni, *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663](https://arxiv.org/abs/2501.00663) — a neural long-term memory written by gradient descent at inference, with a surprise-gated write and explicit forgetting.
- [von Oswald et al., *Uncovering mesa-optimization algorithms in Transformers*, arXiv:2309.05858](https://arxiv.org/abs/2309.05858) — the mesa-layer; in-context learning as an inner optimizer.
- [Irie, Schlag, Csordás & Schmidhuber, *Going Beyond Linear Transformers with Recurrent Fast Weight Programmers*, arXiv:2106.06295](https://arxiv.org/abs/2106.06295).

**Why this matters for how the project is framed.** "The state keeps learning while it runs" is, for a fast-weight
cell, *already true by construction* — the outer product write **is** a learning step on an associative memory, and
TTT/Titans say so explicitly. So phase 4's "learning at inference" is measuring the **outer** loop (the optimizer
touching slow weights during deployment), not the inner one. If the write-up does not separate those two, it will
claim novelty for something published in 1992. Say: *inner-loop* learning (the state write) versus *outer-loop*
learning (the trace-driven SGD step). The project's actual open question is the outer loop.

---

## 2. Can this plan really test associative recall? Mostly yes — with three fixes

The task design (`k=v;` pairs, distractors, a `?k` query, score the answer byte only, fresh episodes from a seed) is
a byte-level rendering of **MQAR — multi-query associative recall**, the standard probe for exactly this question:

- [Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models*, arXiv:2312.04927](https://arxiv.org/abs/2312.04927) — introduces MQAR, and shows the gap between attention and gated-convolution/linear models is concentrated on associative recall.
- [Arora et al., *Simple linear attention language models balance the recall-throughput tradeoff* (Based), arXiv:2402.18668](https://arxiv.org/abs/2402.18668) — the recall–throughput (really recall–**state size**) tradeoff, stated as a curve.
- [Fu et al., *Hungry Hungry Hippos (H3)*, arXiv:2212.14052](https://arxiv.org/abs/2212.14052) — the induction-head / associative-recall synthetic that motivated the whole line.
- [Olsson et al., *In-context Learning and Induction Heads*, Transformer Circuits, 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — the mechanism being probed.

The plan's version is a fair instance of that task. Three things break the inference it wants to draw:

### 2a. The grid sweeps gap but not the number of pairs — and capacity, not gap, is the binding constraint

This is the single biggest scientific hole. For a fixed-size recurrent state, the literature is unambiguous that
recall is limited by **how many key–value pairs the state can hold**, not by how far back they were written. Zoology
and Based both show recall accuracy as a function of state size and number of pairs; Schlag et al. show sum-update
fast weights *overflow* once the number of distinct keys exceeds the key dimension, because new outer products are
added on top of old ones with no mechanism to remove them.

The plan sweeps gap ∈ {8…512} at, implicitly, whatever number of distractor pairs fills that gap. Gap and pair count
are therefore **confounded**: a gap of 512 bytes at 4 bytes per pair is ~128 pairs, and with only 16 possible keys
the episode must be reusing keys or padding. With 16 keys and `dk = 16`, `fast` sits exactly on the capacity
boundary Schlag et al. describe. A failure at gap 512 will be uninterpretable — capacity overflow, decay, or
credit assignment, and the plan cannot tell which.

**Fix:** make the episode grid two-dimensional — `n_pairs ∈ {2, 4, 8, 16, 32}` × `gap ∈ {8, 32, 128, 512}` — with
filler bytes (not extra pairs) used to open the gap independently of pair count. Then "does it remember" and "how
much can it hold" are separable, and the result is directly comparable to published MQAR curves.

Also: the alphabet is only 16 keys and 16 values. Chance is 1/16 = 6.25%, and a 16-key alphabet caps capacity at 16
pairs by construction. Widen to at least 32 keys / 32 values if capacity is going to be swept.

### 2b. Parameter matching is the wrong fairness axis, and it actively biases the comparison

The plan matches parameter counts within 20% and lets dimension float per cell. For recall, the quantity that
predicts performance is **state size in scalars** (Zoology/Based). Working the plan's own numbers:

| Cell | Params ≈ 0.8M implies | Recurrent state per layer | Total state (4 layers) |
|---|---|---|---|
| `leaky` | dim 384 (W only: 4·384² + embed + decoder ≈ 0.79M) | 384 | **1,536** |
| `gated` | dim ≈ 245 (W, V, G: 12·d² + 512d ≈ 0.79M) | 245 | **980** |
| `fast` | dim ≈ 514 (Wk,Wv,Wq,Wo: 1024d + 512d ≈ 0.79M) | 4 heads × 16 × 16 = 1,024 | **4,096** |

So param-matching hands `fast` **2.7× the memory of `leaky` and 4.2× the memory of `gated`**, and simultaneously
gives it the largest model dimension. If `fast` wins, the plan cannot say whether the outer product did it or the
extra 2,560 scalars of state did. Worse, `gated` — the middle rung — is penalised hardest, which will make the
"gated lands between the two" prediction look confirmed for the wrong reason.

**Fix:** match **state size** as the primary constraint, report parameters as a secondary column, and run at two
state sizes per cell so the slope is visible. Matching both exactly is impossible; matching state and reporting the
param gap is the honest choice, and it is what the recall literature does.

### 2c. The decay initialisation is a confound for the headline prediction

The plan's own section 2 cites issue #3: learned decays end up with a half-life of about one byte. The plan then
predicts `leaky` fails at every gap beyond 8 and attributes it to the architecture ("it can't bind"). Those are two
different causes and the plan as written cannot separate them, because all three cells inherit the same `sigmoid(d)`
parameterisation and, presumably, the same default init.

The literature on this is old and clear. Forget gates need to be initialised open, or the model never learns long
dependencies:

- [Gers, Schmidhuber & Cummins, *Learning to Forget*, Neural Computation 2000, doi:10.1162/089976600300015015](https://doi.org/10.1162/089976600300015015)
- Jozefowicz et al. 2015 (above) — the concrete "init forget bias to 1" recommendation.
- [Gu & Dao, *Mamba*, arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — the `A_log` / `dt` initialisation is engineered so that the effective timescale spans a wide, long range at init.
- Orvieto et al. 2303.06349 (above) — stable exponential parameterisation and eigenvalue-ring init are load-bearing for long-range performance.

**Fix:** parameterise decay as `a = exp(-exp(log_dt))` or equivalent, and initialise `d` so half-lives are spread
log-uniformly over roughly 8–1,024 bytes across channels, identically for all three cells. Then report the half-life
table before and after training (the plan already does this — good) and the prediction becomes falsifiable: if
`leaky` still fails at long gaps *with* good decays, the binding argument is supported.

### 2d. A smaller point: "it can't bind" is a capacity argument, not an impossibility proof

`leaky`'s state is a decay-weighted sum of residual-stream vectors, and the read `x + SiLU(W·LN(s))` is
query-independent — the query byte only enters through `x_t` at the current step. That is a bag-of-history with
exponential recency weighting, and it is a genuinely weak retrieval mechanism. But with 4 layers, a residual
stream, and channel-wise decay diversity, it is not *provably* incapable of crude binding — a 4-layer stack can in
principle shape lower-layer writes so that upper-layer states are query-conditioned. State it as a prediction with a
mechanism, not as a theorem. The nearest thing to a real theorem is the state-tracking/expressivity work:

- [Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models*, arXiv:2404.08819](https://arxiv.org/abs/2404.08819)
- [Jelassi et al., *Repeat After Me: Transformers are Better than State Space Models at Copying*, arXiv:2402.01032](https://arxiv.org/abs/2402.01032) — the clean information-theoretic statement: a fixed state of *n* bits cannot copy strings longer than *n* bits. Directly relevant to section 3 below.
- [Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*, arXiv:2411.12537](https://arxiv.org/abs/2411.12537) — why `sigmoid(d) ∈ (0,1)` (strictly positive eigenvalues) is itself an expressivity restriction. *(ID believed correct; verify.)*

---

## 3. Can it test "theoretically infinite memory"? No — and the plan should retire the phrase

The claim is not false so much as not a claim. A model with a fixed-size state carries a fixed number of bits. Even
at float32, a 4,096-scalar state is at most ~131 kbit, and in practice far less because the state is written by a
noisy, lossy process. Jelassi et al. (2402.01032) make this precise for copying; the Zoology/Based recall curves
make it empirical for retrieval. There is no experiment that can come back "yes, infinite".

What the repo probably *means* and what is genuinely testable splits into two claims, and the plan should replace
the phrase with them:

1. **Unbounded context at constant cost.** Memory and time per byte do not grow with stream length. This is true of
   every RNN and is trivially measurable — the plan's phase 0 "constant memory over 10,000 steps" gate already
   does it. Extend it to 1M steps and log RSS, and report **time per byte as a function of position** (flat = pass).
   This is the claim that actually differentiates the design from smLLM_01's 256-byte window, and it is worth
   stating plainly as the one architectural advantage that is real.
2. **Effective memory capacity**: how many pairs survive, and for how long. That is the 2-D grid from §2a, plus the
   decay half-life table, plus the "state carried across episodes" interference pass the plan already has. Report
   it as *bits retained*, or at least *pairs recalled at 90%*, versus state size. That number is comparable to the
   literature; "infinite" is not.

A third measurement worth adding, cheap: **recall versus episodes-since-reset** in the carried-state pass. That is
the catastrophic-interference curve, and it is the direct empirical refutation of "infinite". One plot.

---

## 4. What is the minimal scientifically fair cell set?

The plan's three-cell ladder has a missing rung, and it is the rung that the last three years of the literature is
about. Sum-update fast weights (`fast` as specced) **cannot overwrite**. Writing `k⊗v` when `k` already carries an
old value adds to it rather than replacing it, so the memory degrades as keys repeat or collide. This is the exact
finding of:

- [Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, arXiv:2102.11174](https://arxiv.org/abs/2102.11174) — identifies the capacity-overflow problem in linear attention and fixes it with the **delta rule**: remove the old value at that key, then write the new one. This is Widrow–Hoff / LMS applied to an associative memory.
- [Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (DeltaNet), arXiv:2406.06484](https://arxiv.org/abs/2406.06484) — makes it trainable at scale; shows the delta rule specifically fixes associative recall (MQAR) where sum-update does not.
- [Yang, Kautz & Hatamizadeh, *Gated Delta Networks*, arXiv:2412.06464](https://arxiv.org/abs/2412.06464) — decay **and** delta together beat either alone; the current strong point on this curve.
- [Peng et al., *RWKV-7 "Goose"*, arXiv:2503.14456](https://arxiv.org/abs/2503.14456) — a generalised delta rule with vector-valued decay, shipped. *(ID believed correct; verify.)*
- [Ba et al., *Using Fast Weights to Attend to the Recent Past*, arXiv:1610.06258](https://arxiv.org/abs/1610.06258) — the intermediate, decay-only fast-weight memory.
- Reference implementations for all of the above: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

### The recommendation: add a `delta` cell. It is tractable under the plan's own trace machinery

The obvious objection is that the delta rule breaks the diagonal-transition assumption the whole trace derivation
rests on. Per head, DeltaNet is

```text
S_t = S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

and `(I − βkkᵀ)` is not diagonal. But it is **identity-minus-rank-one**, and that is enough. The RTRL recursion for
a trace tensor `E[i, j, :]` (i over `dk`, j over `dv`, last axis over the parameter's input dim) is

```text
E ← E − β k ⊗ (kᵀE) + ∂b_t/∂θ
```

and `kᵀE` costs `dk·dv·d`, the outer-product subtraction costs another `dk·dv·d`. So the delta cell's trace update is
**the same order as `fast`'s**, roughly 3× the constant, not 16×. There is no blocker. A `delta` cell belongs in
phase 0 alongside the other three, and it is the cell most likely to actually pass the phase 2 gate.

If only one cell can be added, add `delta`, not a GRU.

### The fair minimal set, in order of what each one buys

| Cell | What it isolates | Keep? |
|---|---|---|
| `leaky` | the null: state with no write mechanism. The repo's cell, the thing being tested | **yes** |
| `gated` | does an input-dependent *write* alone fix recall? (Prediction from the literature: no — gating the write does not create key-conditioned readout) | **yes** |
| `fast` | does an outer-product *key–value* write fix it? (Prediction: partly, then it saturates on capacity) | **yes** |
| `delta` | does *overwriting* fix it? (Prediction: this is the one that works) | **add — must** |
| `gru` under `tbptt` only | the textbook control: what does a normal RNN get | **yes, answering open question #2 — it is cheap and it is the reviewer's first question** |

Two more cell-level corrections the literature is firm about, both cheap:

1. **`fast` needs a feature map and an output normaliser.** Raw `Sᵀq` with unbounded `q, k` is numerically
   unstable and empirically much worse. Katharopoulos et al. use `φ(x) = elu(x) + 1` with a denominator
   `φ(q)ᵀ z`; DeltaNet and Gated DeltaNet L2-normalise keys and put an RMSNorm on the head output. Without one of
   these, a `fast` failure is a normalisation failure, not an architecture result. Pick L2-normalised `k, q` plus
   per-head RMSNorm on the read — it is the modern default and it is two lines.
2. **Static decay is the known-weak version.** `a = sigmoid(d)` is input-independent — RetNet-style. Mamba's
   central claim (2312.00752) is precisely that **input-dependent** (selective) decay is what enables selective
   copying and induction heads; GLA ([Yang et al., arXiv:2312.06635](https://arxiv.org/abs/2312.06635)) and Gated
   DeltaNet carry the same design. With static decay, the model cannot decide to *hold* the queried pair and
   *drop* the distractors, which is the entire skill the recall world tests. Note that input-dependent decay
   keeps the recurrence diagonal, so **the trace machinery still applies** — `a_t` becomes an extra term in the
   `E_d` recursion. Adding it to one cell as an ablation (`gated` → `gated-sel`) is the highest-value-per-line
   change in the whole plan after `delta`.

---

## 5. Which baselines matter

The plan's baselines are the lazy guesses (recall) and smLLM_01 (text). Both are necessary and neither is
sufficient. Ranked by how much each one changes what the write-up is allowed to conclude:

1. **An attention model on the recall world.** This is the missing ceiling. The plan compares cells to each other
   and to chance, so a table where every cell fails is unreadable — is the task too hard, or are the cells too
   weak? A small causal transformer with a 512-byte context (reuse `smLLM_01/model.py`, shrunk) either solves MQAR
   or it does not, and Zoology says it will. Without it, "online recurrent models can't do associative recall" is
   not supported; with it, it is. **This is the single most important missing baseline.**
2. **`tbptt` window 1 — the credit-assignment ablation.** The plan compares `online` (batch 1, traces, per-byte
   step) against `tbptt` (batch 64, window 256). Those differ in *three* ways at once: batch size, number of
   optimizer steps per byte, and credit-assignment horizon. A run with the same cell, same batch, same step count,
   but **no traces and no unroll** isolates what the eligibility traces actually buy. If `online` ≈ `tbptt-w1`, the
   traces are doing nothing and that is the finding.
3. **Dynamic evaluation, for phase 4.** Updating weights on a held-out stream at test time is a named, published,
   well-measured technique, and it reliably improves bits/char on in-domain text. [Krause et al., *Dynamic
   Evaluation of Neural Sequence Models*, arXiv:1709.07432](https://arxiv.org/abs/1709.07432); the prior art goes
   back to Mikolov's dynamic RNN evaluation, and the non-parametric cousin is [Grave et al., *Improving Neural
   Language Models with a Continuous Cache*, arXiv:1612.04426](https://arxiv.org/abs/1612.04426). So phase 4's
   "learning at inference with real answers" should **predict a gain**, and the interesting number is whether the
   trace-trained cell gains *more or less* than a dynamically-evaluated smLLM_01 does. Otherwise the project
   reports a known 2018 result as its own.
4. **smLLM_01 re-timed on the day, not quoted.** The 26,100–27,000 bytes/s figure is from 2026-09-14 and the
   plan's own ground rules say wall-clock numbers depend on what else is loaded on the Mac. Re-run the 20-minute
   smLLM_01 training under the same conditions as the smRTS runs. Cheap insurance against an unfair headline.
   Related: smLLM_01 *overfit* after step 1,500 on 1 MB of text, so at 20 minutes the transformer is data-limited,
   not compute-limited. A recurrent model that streams the held-out text with unlimited context is measuring a
   partly different thing. The plan flags the context advantage — it should also flag the overfitting.
5. **A memory-matched reference on text.** Optional but clarifying: an n-gram / continuous-cache model gives a
   floor-and-context reference for "what does unbounded context buy on Shakespeare at all".
6. **Keep the lazy guesses.** They are right and they are the house style. Add binomial confidence intervals —
   1,000 episodes gives roughly ±1.5 points at 90%, and several plan comparisons will land inside that.

---

## 6. Correctness notes on the phase 0 maths

I worked through the trace derivations in the plan. They are right, with two things that must be stated explicitly
or the implementation will silently double-count.

**The recursions check out.** With `a = sigmoid(d)` depending only on `d`:

- `∂s_t/∂d = a(1−a)·s_{t−1} + a·∂s_{t−1}/∂d` → `E_d = a E_d + a(1−a) s_{t−1}` ✅
- `gated`: `b_i = g_i·(Vx)_i` with `g = sigmoid(Gx)`. `∂b_i/∂V[i,:] = g_i x` ✅ and
  `∂b_i/∂G[i,:] = g_i(1−g_i)(Vx)_i x` ✅
- `fast`: `v` does not depend on `Wk` and `k` does not depend on `Wv`, so both recursions are correct as written ✅

**Must be stated: `dL/ds_t` is the *immediate* partial.** The trace formulation is
`dL/dθ = Σ_t (∂L_t/∂s_t) · E_θ(t)`, where `∂L_t/∂s_t` is the gradient arriving through **step t's read path only**.
If the implementation carries any graph across steps, that term becomes a total derivative and the sum
double-counts. The fix is the same as the constant-memory requirement: **detach the state between steps**. The plan
gets this right by construction (the dummy-zero-tensor trick yields the immediate partial), but the gate should
assert it — e.g. fail if `s.grad_fn` survives a step boundary. Add it to `tests/test_traces.py`.

**Gradient-check gate, tightened.** "Every parameter matches to 1e-4" should be a *relative* tolerance
(`max|g_trace − g_autograd| / (max|g_autograd| + eps)`) and the check should run in float64. At dim 8 over 6 steps in
float32, absolute 1e-4 is both too loose for small gradients and too tight for large ones. For a 1-layer diagonal
cell the agreement should be at machine precision — anything worse is a bug, not a tolerance question.

**The "same loss on step 1 from the same seed" gate is near-vacuous.** Three trainers with different batch sizes
will not even see the same bytes. Replace with: the three trainers produce *identical* per-byte loss for the first
64 bytes when all are run at batch 1 with the same stream and learning rate 0 (forward-path equivalence), which is
the property that actually matters.

**Trace cost, which the plan says to report — here is what to expect.** For `fast` at dim ≈ 514, 4 heads,
`dk = dv = 16`, 4 layers: each of `E_Wk`, `E_Wv` is `heads × dk × dv × d ≈ 526k` floats per layer, so ~4.2M floats
of trace, ~17 MB fp32 **per stream**. At `online-batched` with 32 streams that is ~540 MB of traces alone, before
optimizer state. The trace *update* costs ~4.2M MACs/byte and the gradient contraction roughly as much again,
against a model forward of ~0.7M MACs/byte — so **`fast`'s online regime pays roughly an order of magnitude more
compute per byte than its own forward pass**, on top of batch-1 inefficiency. `leaky`'s traces are negligible by
comparison (one vector per layer). Two consequences: (a) the plan's prediction "online at batch 1 is 50–100× slower"
is plausible but the cost is *cell-dependent*, so predict it per cell; (b) if `fast` has to drop to a smaller batch
to fit while `leaky` runs at 32, the regime axis becomes confounded with the cell axis — pin the batch size across
cells at whatever the largest cell can fit, and report it.

**One fairness wrinkle in the traced-parameter sets.** `leaky` traces "the byte embedding at layer 0 only, as the
repo does", the others do not. That is a difference in what online learning is *allowed to update*, not just in
architecture. Trace the embedding for all cells or for none.

---

## 7. Framing the result honestly

Whatever the numbers say, the write-up should be able to state these four things, and the plan as written can only
state the first:

1. Constant memory and constant time per byte at unbounded stream length — **measurable, and the real advantage**.
2. Memory capacity in pairs versus state size, next to published MQAR curves — needs §2a and §2b.
3. What eligibility traces buy over no temporal credit assignment at all — needs the `tbptt-w1` ablation.
4. Whether *outer-loop* online learning beats dynamic evaluation of an ordinary model — needs the phase 4 baseline.

And the thing not to claim: that fast weights, learning at inference, or state-based memory are new. They are 1992,
2018, and 1997 respectively. What is plausibly new here is the *combination* — trace-based outer-loop learning of a
delta-rule memory at one byte per step on a laptop — and a clean negative result on that is a perfectly good
deliverable, which the plan already understands.

---

## 8. Ranked changes to the plan

### Must

| # | Change | Why | Where |
|---|---|---|---|
| M1 | **Sweep number of pairs as well as gap.** `n_pairs ∈ {2,4,8,16,32}` × `gap ∈ {8,32,128,512}`, gap opened with filler bytes so the two are independent. Widen the alphabet to ≥32 keys / ≥32 values | Capacity, not distance, is what limits a fixed state. As written, every long-gap failure is uninterpretable, and it is not comparable to the MQAR literature (arXiv:2312.04927, arXiv:2402.18668) | Phase 1, phase 2 grid |
| M2 | **Match state size across cells, not parameter count.** Report params as a secondary column; run two state sizes per cell | Param-matching gives `fast` 2.7× `leaky`'s state and 4.2× `gated`'s, and the largest dim as well. A `fast` win would be unattributable | §4 decisions, phase 0 sizing |
| M3 | **Add a `delta` cell** (`S ← S(I − βkkᵀ) + βvkᵀ`, L2-normalised keys). Its RTRL trace is `E ← E − βk⊗(kᵀE) + ∂b/∂θ`, same order of cost as `fast` | Sum-update fast weights provably cannot overwrite; the delta rule is the published fix and the cell most likely to pass the gate. Omitting it means the plan tests a design the field already moved past (arXiv:2102.11174, arXiv:2406.06484) | Phase 0 `cells.py` |
| M4 | **Add an attention baseline on the recall world** (shrunk `smLLM_01/model.py`, 512-byte context) | Without a ceiling, an all-fail table cannot distinguish "task too hard" from "cells too weak". This is the single most important missing baseline | Phase 2 |
| M5 | **Fix decay init and parameterisation, identically for all cells**: `a = exp(-exp(log_dt))`, half-lives log-uniform over ~8–1,024 bytes at init | Otherwise "leaky can't bind" is confounded with "decays initialised at a one-byte half-life", which is the repo's own issue #3 (arXiv:2312.00752, doi:10.1162/089976600300015015) | Phase 0 |
| M6 | **Add the `tbptt` window-1 ablation** (same batch, same step count, no traces, no unroll) | `online` vs `tbptt` currently differs in batch size, step count and credit horizon at once. This isolates what traces buy — the project's core question | Phase 0 trainers, phase 2 grid |
| M7 | **Retire "theoretically infinite memory"** as a testable claim; replace with (a) constant memory/time per byte at 1M+ steps, (b) capacity in pairs vs state size, (c) recall vs episodes-since-reset | A fixed state holds finite bits; there is no experiment that returns "infinite" (arXiv:2402.01032) | §1, §3, phase 5 |
| M8 | **Assert state detachment in the trace test**, and make the gradient check float64 with a *relative* 1e-6 tolerance for 1 layer | `dL/ds_t` must be the immediate partial; a surviving graph silently double-counts. 1-layer diagonal RTRL is exact, so anything above machine precision is a bug | Phase 0 gate, `tests/test_traces.py` |

### Should

| # | Change | Why |
|---|---|---|
| S1 | **Give `fast`/`delta` L2-normalised keys and queries plus a per-head RMSNorm on the read** | Unnormalised `Sᵀq` is unstable; a failure would be a normalisation artefact, not an architecture result (arXiv:2006.16236, arXiv:2406.06484) |
| S2 | **Add one input-dependent-decay ablation** (`a_t = sigmoid(W_a x_t)`). It stays diagonal, so the traces still apply | Selectivity is Mamba's central claim and it is exactly the skill the recall world tests: hold the queried pair, drop the distractors (arXiv:2312.00752, arXiv:2312.06635) |
| S3 | **Phase 4 should predict a gain from learning at inference, and compare against dynamic evaluation of smLLM_01** | Dynamic evaluation is a known, measured 2018 technique. Without the comparison the project reports prior art as a finding (arXiv:1709.07432) |
| S4 | **Say "inner loop" (state write) vs "outer loop" (trace SGD) throughout** | The state write *is* test-time learning already, by construction (arXiv:2407.04620, arXiv:2501.00663). Without this distinction the write-up claims 1992 as novel |
| S5 | **Re-time smLLM_01 on the day rather than quoting 2026-09-14**, and note it was data-limited (overfitting after step 1,500), not compute-limited, at the 20-minute budget | The plan's own ground rules make wall-clock numbers machine-state-dependent; verified locally in `docs/engineering/smModels/smLLM_01.md` lines 57, 66–67 |
| S6 | **3 seeds for every recall cell, not just the passers.** 27 × 10 min ≈ 4.5 h | "`leaky` fails" from one seed is the plan's headline negative result and one seed will not carry it. If the budget is tight, cut a gap value instead |
| S7 | **Pin `online-batched` batch size across cells** at whatever the largest-trace cell fits, and report trace bytes per stream | Otherwise regime and cell are confounded — `fast` traces are ~17 MB/stream vs `leaky`'s kilobytes |
| S8 | **Binomial confidence intervals on every recall number**; and trace the byte embedding for all cells or for none | ±1.5 points at n=1,000 swallows several planned comparisons; the embedding asymmetry is a difference in what online learning may update |
| S9 | **Answer open question #2: yes, include the GRU under `tbptt`** | It is ~10 minutes of compute and it is the first question any reader asks. It does not need traces — it is a control, not a cell |
| S10 | **Replace the "same loss on step 1" gate** with forward-path equivalence over 64 bytes at batch 1, lr 0 | The current gate is near-vacuous: different batch sizes do not see the same bytes |

### Could

| # | Change | Why |
|---|---|---|
| C1 | Cite the lineage in `README.md` — Schmidhuber 1992, Katharopoulos 2020, Schlag 2021, Mamba, DeltaNet, TTT/Titans, RTRL/e-prop | Costs nothing, and it is what makes a negative result citable rather than anecdotal |
| C2 | Add a copying/`--overwrite` task as a second probe, promoted from "off by default" if `delta` lands | Overwrite is where sum-update and delta separate most sharply; copying is the cleanest capacity probe (arXiv:2402.01032) |
| C3 | Frame the `--repo-loss` MSE-plus-variance-hinge as what it is — a latent-prediction term with an anti-collapse regulariser, cf. VICReg ([arXiv:2105.04906](https://arxiv.org/abs/2105.04906)) | Makes the one comparison run interpretable instead of a curiosity |
| C4 | For the own-output trap, cite the model-collapse result ([Shumailov et al., *AI models collapse when trained on recursively generated data*, Nature 631, 2024](https://www.nature.com/articles/s41586-024-07566-y)) | Puts the measured number in a published frame and strengthens the `whats-next.md` Learn-module rule |
| C5 | Consider UORO or SnAp-2 as a stretch ablation for the cross-layer path the plan cuts | Would turn issue #2 from "stated, not hidden" into "measured". Out of scope for this run, worth a line in "next" |
| C6 | Note on naming (open question #1): "recurrent trace state" is a good read of RTS and is consistent with the plan's content. The nearest alternative worth ruling out with the owner is "real-time streaming" | One question, one sentence |
| C7 | Report `fast`/`delta` memory in *bits retained* as well as pairs, so the result plots on the same axes as the recall–state-size curves in Based (arXiv:2402.18668) | Makes the result directly comparable to published numbers |

---

## 9. Sources

Primary papers, official repos, and the local files used. **None of the external URLs were re-fetched this session
(no network) — verify before quoting.**

**Fast weights and linear attention**
- Schmidhuber, *Learning to Control Fast-Weight Memories* (1992) — https://doi.org/10.1162/neco.1992.4.1.131
- Ba et al., *Using Fast Weights to Attend to the Recent Past* — https://arxiv.org/abs/1610.06258
- Katharopoulos et al., *Transformers are RNNs* — https://arxiv.org/abs/2006.16236
- Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* — https://arxiv.org/abs/2102.11174
- Irie et al., *Going Beyond Linear Transformers with Recurrent Fast Weight Programmers* — https://arxiv.org/abs/2106.06295
- Sun et al., *Retentive Network* — https://arxiv.org/abs/2307.08621

**Delta-rule memories**
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* — https://arxiv.org/abs/2406.06484
- Yang, Kautz & Hatamizadeh, *Gated Delta Networks* — https://arxiv.org/abs/2412.06464
- Peng et al., *RWKV-7 "Goose"* — https://arxiv.org/abs/2503.14456 *(ID to verify)*
- Reference implementations — https://github.com/fla-org/flash-linear-attention

**State-space and gated linear recurrences**
- Gu & Dao, *Mamba* — https://arxiv.org/abs/2312.00752
- Dao & Gu, *Transformers are SSMs* (Mamba-2 / SSD) — https://arxiv.org/abs/2405.21060
- Yang et al., *Gated Linear Attention* — https://arxiv.org/abs/2312.06635
- Orvieto et al., *Resurrecting Recurrent Neural Networks for Long Sequences* (LRU) — https://arxiv.org/abs/2303.06349
- Peng et al., *RWKV* — https://arxiv.org/abs/2305.13048
- De et al., *Griffin / Hawk* — https://arxiv.org/abs/2402.19427

**Recall, capacity, expressivity**
- Arora et al., *Zoology* (MQAR) — https://arxiv.org/abs/2312.04927
- Arora et al., *Based* — https://arxiv.org/abs/2402.18668
- Fu et al., *H3* — https://arxiv.org/abs/2212.14052
- Jelassi et al., *Repeat After Me* — https://arxiv.org/abs/2402.01032
- Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* — https://arxiv.org/abs/2404.08819
- Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues* — https://arxiv.org/abs/2411.12537 *(ID to verify)*
- Olsson et al., *In-context Learning and Induction Heads* — https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

**Online / real-time recurrent learning**
- Williams & Zipser, RTRL (1989) — https://doi.org/10.1162/neco.1989.1.2.270
- Tallec & Ollivier, *Unbiased Online Recurrent Optimization* — https://arxiv.org/abs/1702.05043
- Menick et al., *A Practical Sparse Approximation for Real Time Recurrent Learning* — https://arxiv.org/abs/2006.07232 *(ID to verify)*
- Bellec et al., *e-prop* — https://www.nature.com/articles/s41467-020-17236-x

**Test-time / online adaptation**
- Sun et al., *Learning to (Learn at Test Time)* — https://arxiv.org/abs/2407.04620
- Behrouz et al., *Titans* — https://arxiv.org/abs/2501.00663
- von Oswald et al., *Uncovering mesa-optimization algorithms in Transformers* — https://arxiv.org/abs/2309.05858
- Krause et al., *Dynamic Evaluation of Neural Sequence Models* — https://arxiv.org/abs/1709.07432
- Grave et al., *Improving Neural Language Models with a Continuous Cache* — https://arxiv.org/abs/1612.04426
- Wu et al., *Memorizing Transformers* — https://arxiv.org/abs/2203.08913
- Shumailov et al., *AI models collapse when trained on recursively generated data* — https://www.nature.com/articles/s41586-024-07566-y

**Gating lineage**
- Hochreiter & Schmidhuber, *LSTM* (1997) — https://direct.mit.edu/neco/article/9/8/1735/6109
- Gers, Schmidhuber & Cummins, *Learning to Forget* (2000) — https://doi.org/10.1162/089976600300015015
- Jozefowicz, Zaremba & Sutskever, *An Empirical Exploration of RNN Architectures* (2015) — https://proceedings.mlr.press/v37/jozefowicz15.html
- Bardes, Ponce & LeCun, *VICReg* — https://arxiv.org/abs/2105.04906

**Local files read**
- `docs/engineering/plans/smRTS_01.md` (the plan under review)
- `docs/engineering/smModels/smLLM_01.md` — reference numbers verified: lines 27, 33, 51–59, 66–67
- `smLLM_01/README.md`, `smALLM_01/world.py` (house style for the world generator)

**Source repo** — https://github.com/jrz97619761/test-model-thing, issues
[#2](https://github.com/jrz97619761/test-model-thing/issues/2),
[#3](https://github.com/jrz97619761/test-model-thing/issues/3),
[#4](https://github.com/jrz97619761/test-model-thing/issues/4). **Not opened this session.**
