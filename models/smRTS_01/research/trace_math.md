# trace_math.md — audit of the Phase 0 eligibility-trace math

**Status:** research memo. Nothing implemented, nothing trained. Written 2026-09-17 against
[`docs/engineering/plans/smRTS_01.md`](../../../docs/engineering/plans/smRTS_01.md) as of that date.

**Scope:** I own this file only. No other file in the repo was read for edit or touched.

> ### Caveat on citations — read this first
>
> This session was run **without web access**. `WebSearch` and `WebFetch` were both denied by the permission
> layer, and outbound `curl` is blocked by the sandbox. Every derivation below is done from first principles
> in this document and is self-checking — you can verify it with pen and paper or with the gradient check in
> §9. The **bibliography in §10 is from memory**. Titles, authors and venues I am confident in; arXiv IDs and
> URLs are **not link-checked in this session** and each carries a confidence marker. Before any of §10 is
> quoted in `README.md` or `results.md`, someone with a browser should verify the IDs. The maths does not
> depend on the bibliography being right.

---

## Contents

1. [Verdict in one page](#1-verdict-in-one-page)
2. [Notation and the exactness conditions](#2-notation-and-the-exactness-conditions)
3. [The two double-counting traps](#3-the-two-double-counting-traps)
4. [Exact one-layer gradients: `leaky`](#4-exact-one-layer-gradients-leaky)
5. [Exact one-layer gradients: `gated`](#5-exact-one-layer-gradients-gated)
6. [Exact one-layer gradients: `fast`](#6-exact-one-layer-gradients-fast)
7. [Shapes and memory complexity, with numbers](#7-shapes-and-memory-complexity-with-numbers)
8. [The multi-layer approximation, stated precisely](#8-the-multi-layer-approximation-stated-precisely)
9. [What is wrong with the Phase 0 gate](#9-what-is-wrong-with-the-phase-0-gate)
10. [Comparison of online-gradient methods](#10-comparison-of-online-gradient-methods)
11. [Risks](#11-risks)
12. [Plan changes, ranked](#12-plan-changes-ranked)
13. [Corrected Phase 0 equation block, ready to paste](#13-corrected-phase-0-equation-block-ready-to-paste)

---

## 1. Verdict in one page

**Every equation printed in the plan is algebraically correct.** I re-derived all six and none of them is wrong.
The decay trace, the `gated` `E_V`/`E_G` pair and the `fast` `E_Wk`/`E_Wv` pair all check out, including the
index placement in the `fast` reductions, which is the easiest thing to get backwards.

The problems are **omissions, unstated preconditions, and a gate that cannot pass as written**:

| # | Finding | Severity |
|---|---|---|
| F1 | The gate says "every parameter matches to 1e-4" on a 1-layer model. For `gated` and `fast` the layer-0 **byte embedding feeds `b_t`** and is **not in the traced set**, so its gradient is truncated and the gate fails by construction. Same for any bias on `V`/`G`. See [§5.4](#54-the-embedding-problem-and-the-fix) — there is a clean fix. | **must** |
| F2 | `d` is initialised at `d≈0` ⇒ `a=σ(0)=0.5` ⇒ **half-life 1 byte**. That is issue #3 of the source repo, and the plan reproduces it rather than fixing it. Worse, the signal that would teach `d` a long half-life is carried *by a trace whose horizon is currently 1 byte*. Chicken-and-egg. Needs a spread initialisation. See [§4.3](#43-the-decay-initialisation-problem). | **must** |
| F3 | Two distinct **double-counting** failure modes give a silent factor of 2 on traced parameters and would be caught by the gate only if the gate is strengthened per F4. See [§3](#3-the-two-double-counting-traps). | **must** |
| F4 | The gate — dim 8, **6 bytes**, random init so `a≈0.5` — is too weak to detect a broken decay trace. At `a=0.5` the `a·E` carry term contributes ~2% after 6 steps; a trace that drops it entirely still nearly matches. See [§9.1](#91-the-gate-as-written-would-pass-a-broken-trace). | **must** |
| F5 | For `fast`, the plan writes `a ⊙ S_{t-1}` with `S` a `dk × dv` **matrix** but calls `a` a "diagonal decay" and indexes it `a_ij` in the traces. The **shape of `a` is undefined**, and the decay trace for the `fast` cell is **never given**. This is not cosmetic — the shape of `a` changes the trace memory by 16×. See [§6.3](#63-the-decay-shape-decides-the-memory-bill). | **must** |
| F6 | **`fast` trace memory at `online-batched` B=32 is ~403 MB** as specified, and the plan has no formula for it. With a per-head scalar decay it collapses **exactly** (no approximation) to ~25 MB. See [§6.3](#63-the-decay-shape-decides-the-memory-bill) and [§7](#7-shapes-and-memory-complexity-with-numbers). | **should** |
| F7 | The **leaky cell's embedding trace equation is missing entirely** — it is the one trace the source repo actually implements, and the plan names it in the cell table without writing it down. Given in [§4.2](#42-the-embedding-trace-the-one-the-plan-omits). | **should** |
| F8 | **Exactness does not survive the optimizer.** RTRL's proof assumes `θ` fixed; one Adam step per byte makes the accumulated trace stale. The plan's gradient check correctly takes no optimizer step, but the plan never says the property it proves is not the property it uses. See [§9.4](#94-exactness-does-not-survive-the-optimizer). | **should** |
| F9 | The `fast` read `y = Sᵀq` has **no normaliser**. Output scale grows with the number of pairs written, which is exactly the quantity that varies with the gap — a confound that lets the model read the gap off the output norm rather than do recall. See [§6.4](#64-the-missing-normaliser-is-a-task-confound). | **should** |
| F10 | `dk = 16` against a **16-key** alphabet sits right on the interference boundary for a plain outer-product sum. Prediction "fast+tbptt passes 90% at gap 128" may fail for capacity reasons that have nothing to do with the trace maths. See [§6.5](#65-capacity-16-keys-into-a-dk16-state). | **should** |
| F11 | A **1-layer arm is missing** from Phase 2. One layer is the *only* configuration where the online gradient is exact, so it is the only clean test of the project's actual question. See [§8.3](#83-why-phase-2-needs-a-1-layer-arm). | **should** |

The headline: **the maths is sound, the engineering around it is where this will fail.** Three of the four
`must` items are about the gate not being able to detect its own failure modes.

---

## 2. Notation and the exactness conditions

### 2.1 Notation

Per layer, per stream:

| Symbol | Meaning | Shape |
|---|---|---|
| `n` | state width (`= dim` for `leaky`/`gated`) | scalar |
| `d` (context-dependent) | input width into the cell; also the decay logit vector — I write the logit as **`δ`** below to kill the clash | — |
| `x_t` | the cell's input at step `t` | `(d,)` |
| `a = σ(δ)` | decay | `(n,)` |
| `s_t` | state, `s_t = a ⊙ s_{t-1} + b_t` | `(n,)` |
| `b_t` | what the cell writes | `(n,)` |
| `y_t` | readout, `y_t = r(s_t, x_t; θ_r)` | — |
| `L_t` | loss at step `t`; `L = Σ_t L_t` | scalar |

> **Naming note for `cells.py`:** the plan uses `d` for both the model width and the decay logit. In the
> equations that is survivable; in code it is a bug factory. Use `dim` and `decay_logit`.

Two different gradients w.r.t. the state, and **the whole scheme turns on not confusing them**:

```text
ḡ_t  :=  ∂L_t/∂s_t     the IMMEDIATE partial: s_t treated as a leaf, only the
                        time-t readout counts. This is what RTRL needs.

c_t  :=  dL/ds_t       the TOTAL derivative, c_t = ḡ_t + a ⊙ c_{t+1}.
                        This is what BPTT computes, backwards in time.
```

They are equal only at the last step. **The plan writes `dL/ds_t` throughout and means `ḡ_t`.** That notation
is what makes trap #1 in §3 easy to walk into. Rename it in the plan.

### 2.2 The identity the whole phase rests on

With `J_t := ∂s_t/∂θ` (the RTRL sensitivity) and `A_t := diag(a)`:

```text
J_t = A_t J_{t-1} + ∂b_t/∂θ          (forward recursion, RTRL)
dL/dθ = Σ_t ḡ_tᵀ J_t                  (exact — no approximation)
```

This is *exactly equal* to the BPTT answer `Σ_t c_tᵀ ∂b_t/∂θ`; it is the same sum of paths re-associated.
Proof in one line — substitute and swap the order of summation over `(t, τ)`:

```text
Σ_t ḡ_tᵀ J_t = Σ_t Σ_{τ≤t} ḡ_tᵀ (Π_{u=τ+1}^{t} A_u) ∂b_τ/∂θ
              = Σ_τ [ Σ_{t≥τ} ḡ_tᵀ Π A_u ] ∂b_τ/∂θ = Σ_τ c_τᵀ ∂b_τ/∂θ   ∎
```

`E_θ` in the plan **is** `J_θ`. The plan never says this, and saying it is what makes the derivations below
mechanical rather than magical.

### 2.3 The three conditions for one-layer exactness

The plan's §2 claim, "the trace is exact for one layer", is true **only under all three** of:

1. **`x_t` does not depend on any `s_τ`.** True at layer 0, where `x_t` is the byte embedding. False at every
   layer above. This is the real content of "one layer".
2. **The traced set is complete**: every learned parameter that appears anywhere in `b_t` has a trace.
   *Currently false for `gated` and `fast`* — see F1.
3. **The state is severed in the autograd graph** so the backward pass returns `ḡ_t`, not `c_t`, and so the
   `τ = t` term is counted once. See §3.

Parameters that appear **only** in the readout (`W`, `W_q`, `W_o`, LayerNorm gains, the decoder, the stop head)
need no trace: `∂L_t/∂θ_r` from plain autograd is already exact. The plan gets this right.

---

## 3. The two double-counting traps

Both produce a **silent factor of ~2** on traced parameters — no crash, no NaN, just a gradient that is wrong
by a constant and training that looks merely "a bit unstable". Both are live in the plan's suggested
implementation ("adding a zero dummy tensor to `s_t` and differentiating with respect to it (the repo's
trick)"), because the plan never states what must be detached.

### Trap 1 — the dummy returns `c_t` instead of `ḡ_t`

The dummy trick gives `ḡ_t` **only if the graph at step `t` contains no future loss and `s_{t-1}` is
detached.** If anyone retains the graph across steps — which is also exactly what the "constant memory" gate
is supposed to catch — `backward()` returns the total `c_t`, and `Σ_t c_tᵀ E_t` counts every temporal path
once per suffix. The error is not a constant factor; it grows with the horizon, so it will look like the
long-gap arms are "just harder".

### Trap 2 — the `τ = t` term counted twice

The RTRL ordering is `E_t ← a ⊙ E_{t-1} + ∂b_t/∂θ`, **then** `grad θ += ḡ_t · E_t`. The freshly added
`∂b_t/∂θ` means `ḡ_tᵀ E_t` already includes the current step's contribution. If `b_t` is still connected to
`s_t` in the live autograd graph, `backward()` adds `ḡ_tᵀ ∂b_t/∂θ` into `θ.grad` **again**.

The same trap catches the decay. If you write `s_t = a * s_prev.detach() + b_t` with `a` live, autograd hands
`δ` the gradient `ḡ_t ⊙ a(1−a) ⊙ s_{t−1}` — which is precisely the term the plan's `a(1−a) ⊙ s_{t−1}` already
put into `E_δ`.

### The recipe that is safe by construction

Sever the state completely and let the trace own every traced parameter:

```text
with torch.no_grad():
    a_val = sigmoid(decay_logit)
    b_val = write(x_t)                       # numeric only
    s_val = a_val * s_prev + b_val           # s_prev already a plain tensor

s_t = s_val.clone().requires_grad_(True)     # the leaf the readout sees
y_t = readout(s_t, x_t)                      # x_t stays live: the residual path is real
L_t = loss(y_t, byte_{t+1})
L_t.backward()

g_t = s_t.grad                               # == ḡ_t, guaranteed
# traced params: grad comes ONLY from traces, computed under no_grad
# untraced params: grad comes ONLY from this backward, and is exact
```

Traced and untraced parameters are then **disjoint gradient sources**. Nothing can be counted twice. Note the
one subtlety that makes this still correct: for `leaky`, `x_t` reaches the readout by two routes — through `s`
(trace) and through the residual `x + SiLU(...)` (autograd). Those are genuinely different paths, so both
gradients are wanted and there is no overlap.

**Detection test to add to `tests/test_traces.py`:** run the check with `decay_logit` frozen at a value giving
`a = 0.9`, and compare per-parameter-group ratios `‖g_trace‖ / ‖g_autograd‖`. Trap 2's signature is a ratio
near 2 on traced groups and 1 on untraced ones — far more diagnostic than a single scalar "gap".

---

## 4. Exact one-layer gradients: `leaky`

Cell: `b_t = x_t`; at layer 0, `x_t = Emb[c_t]` for byte `c_t ∈ {0..255}`. Read: `x + SiLU(W · LN(s))`.
Traced set: `{δ, Emb}`. `b_t` has no parameters of its own, so there is nothing else to trace — the plan is
right about that.

### 4.1 Decay

```text
s_{t,i} = σ(δ_i) s_{t-1,i} + b_{t,i}
∂s_{t,i}/∂δ_j = 0 for i ≠ j            (diagonal ⇒ the trace is a vector, not a matrix)

E_δ[i] ← a_i · E_δ[i] + a_i(1 − a_i) · s_{t-1,i}
grad δ[i] += ḡ_t[i] · E_δ[i]
```

**Matches the plan. Correct.** Shape `(n,)`. Memory `n` floats.

One precondition the plan does not state: this holds because `∂b_t/∂δ = 0`. True for all three cells here, but
it is the first thing a future gated-decay variant (Mamba-style input-dependent `a`) would break, and someone
will try that.

### 4.2 The embedding trace — the one the plan omits

The plan's cell table says "the byte embedding at layer 0 only, as the repo does" and then never writes the
equation. It is the cheapest trace in the project and it is worth having on the page:

Since `b_t = Emb[c_t]`, we have `∂b_{t,i}/∂Emb[c,m] = 1[c_t = c] · δ_{im}` — nonzero only on the diagonal
`i = m`. So the trace is a `(256, n)` matrix, **not** `(256, n, n)`:

```text
T[c, m] ← a_m · T[c, m]  +  1[c_t = c]          # for all 256 rows, every step
grad Emb[c, m] += ḡ_t[m] · T[c, m]
```

Shape `(256, n)`. Memory `256n` floats = 98,304 for `n = 384` (384 KB fp32). Cost per step is a dense
`256 × n` multiply-add — cheap, but **not free, and it does not sparsify**: every row decays every step even
though only one row is incremented.

`grad Emb` is a full `(256, n)` dense update every step. With Adam that touches all 256 rows' moments every
step, which is fine but is worth knowing before someone "optimises" it to a sparse row update and silently
breaks the decay.

### 4.3 The decay initialisation problem

This is F2, and I think it is the single biggest determinant of whether Phase 2 passes.

The half-life in bytes is `h = ln2 / (−ln a)`. At `δ = 0`:

```text
a = σ(0) = 0.5   →   h = 1.0 byte
```

That *is* source-repo issue #3. The plan reproduces the parameterisation and expects training to fix it. To
reach the plan's top gap of 512 bytes you need

```text
h = 512  →  a = 2^(−1/512) = 0.998647  →  δ = logit(a) = ln(738) ≈ 6.60
```

So `δ` must travel from 0 to ~6.6. Two things make that hard, and they compound:

1. **The gradient that would teach it is carried by a trace whose horizon is currently 1 byte.** `E_δ` decays
   at rate `a`; at `a = 0.5` it has forgotten anything older than a couple of bytes. The information that
   "a 512-byte dependency exists" is not in the trace, so the gradient pointing at `δ = 6.6` is essentially
   absent. Chicken-and-egg.
2. The trace's drive term is scaled by `a(1−a)`, which is **0.25 at init and 0.00135 at the target** — it
   vanishes at exactly the value you are trying to reach.

Note for the record: the *parameterisation* is not the problem. For `a → 1`, `1 − a = σ(−δ) ≈ e^{−δ}` and
`h ≈ ln2 · e^δ`, so `dh/dδ ≈ h` — the half-life is geometric in `δ`, which is well conditioned. An
`a = exp(−exp(δ))` reparameterisation gives the same `dh/dδ ≈ h`. **Switching parameterisation buys nothing.
Initialisation is what buys something.**

**Recommendation (must):** initialise the half-lives spread log-uniformly across the state's channels to
cover the task's gap range, the way S4/LRU/Mamba initialise their timescales:

```text
u_i ~ Uniform(0, 10)
h_i = 2^{u_i}                         # half-lives log-uniform over [1, 1024] bytes
a_i = 2^{−1/h_i}
δ_i = logit(a_i) = ln(a_i / (1 − a_i))
```

Then some channels can carry a 512-byte dependency **from step one**, the trace for those channels has a long
horizon, and gradient descent on `δ` is refining timescales rather than discovering them. This also makes the
plan's own Phase 2 measurement ("half-lives per layer, before and after training") far more informative: the
question becomes "which timescales did training keep" instead of "did it escape 1 byte at all".

**Report both.** Keep an arm with `δ = 0` init as the faithful reproduction of the repo, and the spread init
as the fixed version. The difference between those two rows is arguably the most useful single number this
project can produce about the source repo's claims.

---

## 5. Exact one-layer gradients: `gated`

Cell: `u = V x`, `g = σ(G x)`, `b_i = g_i u_i`. Traced set per the plan: `{δ, V, G}`.

### 5.1 Derivation

```text
∂b_i/∂V[j,k] = δ_ij · g_i · x_k                       (g does not depend on V)
∂b_i/∂G[j,k] = δ_ij · g_i(1 − g_i) · u_i · x_k        (u does not depend on G)
```

The Kronecker `δ_ij` is the whole game: **row `i` of `V` influences only state component `i`.** Substituting
into `J_t = A J_{t-1} + ∂b_t/∂θ` and keeping only the surviving slice:

```text
E_V[i, k] ← a_i · E_V[i, k] + g_i · x_k                       grad V[i,k] += ḡ_t[i] · E_V[i,k]
E_G[i, k] ← a_i · E_G[i, k] + g_i(1 − g_i) · u_i · x_k        grad G[i,k] += ḡ_t[i] · E_G[i,k]
```

**Both match the plan exactly. Correct.**

### 5.2 Why the generic recursion in the plan is not implementable, and the plan should say so

The plan prints a generic rule:

```text
any θ that only enters through b_t:   E_θ = a ⊙ E_θ + ∂b_t/∂θ,   grad θ += dL/ds_t · E_θ
```

This is correct but **`E_θ` has shape `(n, |θ|)`**. For a dense `V` of shape `(n, d)` that is `n·n·d` floats
= 384·384·384 = **56.6 M floats = 226 MB per layer per stream**. Full RTRL, infeasible, and nobody would
notice from reading the plan that the generic form and the closed form differ by a factor of `n`.

The closed forms in §5.1 are `(n, d)` — **parameter-shaped, a factor of `n` smaller** — purely because of the
`δ_ij`. The plan should state the enabling condition explicitly, because it is the reason the whole phase is
possible and it is the thing a fourth cell would break:

> **Row-locality condition.** The trace collapses from `(n, |θ|)` to parameter-shaped exactly when
> `∂b_{t,i}/∂θ` is supported on an `O(1)`-sized slice of `θ` for each `i`. Equivalently: the write is
> elementwise in the state index. This is the condition under which e-prop / RFLO are exact rather than
> approximate, and it is why a plain GRU (dense recurrence, no row-locality) cannot be trained this way — the
> plan's own open question #2 already senses this and should cite the condition.

**Rule of thumb worth putting in the plan:** under row-locality, *trace memory = one float per traced scalar
parameter, per stream.*

### 5.3 Missing: biases

The plan's table lists `V` and `G` as matrices with no mention of biases. If `cells.py` gives them biases,
those biases enter `b_t` and are untraced, and the §9 gate fails. Their traces are trivial:

```text
E_bV[i] ← a_i · E_bV[i] + g_i                       grad bV[i] += ḡ_t[i] · E_bV[i]
E_bG[i] ← a_i · E_bG[i] + g_i(1 − g_i) · u_i        grad bG[i] += ḡ_t[i] · E_bG[i]
```

Cheapest possible fix: **`bias=False` on every write projection**, and say so in the plan. `bG` in particular
is the natural place to bias the gate open at init, so if it is wanted, trace it.

### 5.4 The embedding problem, and the fix

**This is F1, the gate-breaker.** At layer 0, `x_t = Emb[c_t]`, so `Emb` enters `b_t` through `V` and `G`.
It is not in `gated`'s traced set. Therefore its gradient is truncated, and the Phase 0 gate — "For each cell,
a 1-layer model ... Every parameter matches to 1e-4" — **fails for `gated` and `fast` by construction.**

Tracing it properly is not an option. The chain rule gives

```text
∂b_i/∂Emb[c, m] = 1[c_t = c] · ( g_i V[i,m] + g_i(1 − g_i) u_i G[i,m] )
```

which is **not** row-local in `m`, so the trace is `(256, n, d)` = 256·384·384 = **37.7 M floats = 151 MB per
layer per stream**. Dead on arrival.

Three ways out, in order of preference:

1. **Feed the one-hot byte straight into the traced write projections** (recommended). Drop the separate
   embedding at layer 0 for `gated` and `fast`; let `V` and `G` be `(n, 256)` instead of `(n, 384)`. The
   embedding *is* then the traced projection. Trace structure and cost are unchanged — `E_V` becomes
   `(n, 256)`, slightly *smaller*. Every learned parameter is now either traced or non-recurrent, so the
   1-layer gate becomes honest for all three cells, and no approximation is introduced. The `x` in the trace
   equations is a one-hot, so the trace update is a single-column scatter plus a decay — **cheaper** than the
   dense version. This is a strictly better design, not a workaround.
2. **Freeze the embedding for the gradient check only** (fixed random `(256, 384)`), and state in the plan
   that the gate covers "every *learned* parameter".
3. **Accept the truncation and report it.** Worst option: it silently makes `gated`/`fast` non-exact even at
   one layer, which destroys the one clean comparison Phase 0 exists to produce.

Take option 1.

---

## 6. Exact one-layer gradients: `fast`

Per head: `k = W_k x ∈ R^{dk}`, `v = W_v x ∈ R^{dv}`, state `S_t = A ⊙ S_{t-1} + k vᵀ ∈ R^{dk × dv}`.
Read `y = Sᵀ q` with `q = W_q x ∈ R^{dk}`, giving `y ∈ R^{dv}`, then `W_o y`. Let `Ḡ := ∂L_t/∂S_t ∈ R^{dk × dv}`
(the immediate partial, per §2.1).

Shape sanity check on the plan's convention: `Sᵀ q` is `(dv × dk)(dk) → (dv)`. **Self-consistent.** The plan
uses the transpose of the usual linear-attention convention (`S = Σ v kᵀ`, `y = S q`); that is fine, but pick
one and put it in a comment at the top of `cells.py`, because the `Σ_j` vs `Σ_i` reductions below flip with it
and that is the classic sign-of-a-long-debugging-session bug.

### 6.1 The write-projection traces

```text
∂S_{ij}/∂W_k[l,m] = δ_il · v_j · x_m        (v is independent of W_k ✓)
∂S_{ij}/∂W_v[l,m] = δ_jl · k_i · x_m        (k is independent of W_v ✓)
```

Hence:

```text
E_Wk[i, j, m] ← A[i,j] · E_Wk[i, j, m] + v_j · x_m      grad W_k[i,m] += Σ_j Ḡ[i,j] · E_Wk[i,j,m]
E_Wv[i, j, m] ← A[i,j] · E_Wv[i, j, m] + k_i · x_m      grad W_v[j,m] += Σ_i Ḡ[i,j] · E_Wv[i,j,m]
```

**Both match the plan, including which index each reduction sums over. Correct.** This is the equation pair I
most expected to find broken and it is not.

### 6.2 The decay trace for `fast` — missing from the plan

The plan gives the decay trace only for the vector-state cells. For the matrix state:

```text
E_A[i,j] ← A[i,j] · E_A[i,j] + A[i,j](1 − A[i,j]) · S_{t-1}[i,j]
```

and then the reduction depends on how `δ` is shaped:

```text
δ elementwise (dk × dv):   grad δ[i,j] += Ḡ[i,j] · E_A[i,j]
δ per key index (dk,):     grad δ[i]   += Σ_j Ḡ[i,j] · E_A[i,j]
δ scalar per head:         grad δ      += Σ_{i,j} Ḡ[i,j] · E_A[i,j]
```

`E_A` is always `(dk, dv)` regardless — negligible memory. Add this to the plan; without it `fast` cannot pass
its own gradient check, and the failure will look like a `W_k` bug.

### 6.3 The decay shape decides the memory bill

**This is F5/F6, and it is the most actionable finding in the memo.**

Look at the drive terms in §6.1. `E_Wk`'s drive is `v_j x_m` — **no `i` dependence**. `E_Wv`'s drive is
`k_i x_m` — **no `j` dependence**. So each trace is constant along an index whenever the *decay* is also
constant along that index. That gives an exact collapse, with no approximation whatsoever:

| Shape of `A` | `E_Wk` shape | `E_Wv` shape | Total floats/head/layer (`dk=dv=16, d=384`) |
|---|---|---|---|
| elementwise `(dk, dv)` — as the plan implies | `(dk, dv, d)` | `(dk, dv, d)` | 196,608 |
| per key index, `A[i,j] = α_i` (GLA-style) | `(dk, dv, d)` | `(dk, d)` | 104,448 |
| per value index, `A[i,j] = β_j` | `(dv, d)` | `(dk, dv, d)` | 104,448 |
| **scalar per head** (RetNet/Lightning-style) | **`(dv, d)`** | **`(dk, d)`** | **12,288 — 16× less** |

Under the scalar-decay collapse the gradient reductions become clean single matmuls:

```text
Ê_Wk ← a · Ê_Wk + v xᵀ     ∈ R^{dv × d}       grad W_k = Ḡ  @ Ê_Wk     (dk×dv)(dv×d) → (dk×d) ✓
Ê_Wv ← a · Ê_Wv + k xᵀ     ∈ R^{dk × d}       grad W_v = Ḡᵀ @ Ê_Wv     (dv×dk)(dk×d) → (dv×d) ✓
```

Two matmuls per head per step instead of a `(dk, dv, d)` elementwise pass. This is **exact**, not an
approximation. Given that the elementwise version costs 403 MB of trace at `B = 32` and the scalar version
costs 25 MB (§7), and that a scalar per-head decay is what RetNet and the whole
lightning/linear-attention line actually ship, **the plan should specify a scalar per-head decay for `fast`
and note the `(dk,dv)` variant as the expensive general case.**

Caveat worth writing down: a scalar decay means every key-slot in a head forgets at the same rate, so
per-head-scalar `fast` needs **several heads with different timescales** to cover the 8–512 gap range. Combine
with the §4.3 spread init: initialise the 4 heads' half-lives at roughly `{8, 32, 128, 512}` bytes. That
costs nothing and directly targets the Phase 1 evaluation grid.

### 6.4 The missing normaliser is a task confound

`y = Sᵀ q` with `S = Σ_τ a^{t−τ} k_τ v_τᵀ` has `‖y‖` growing with the number of writes still in the decay
window. In the recall world, the number of `k=v;` pairs written before the query **is** what the gap varies.
So output magnitude carries gap information directly, the decoder can exploit it, and a model can post
above-chance numbers that have nothing to do with binding a key to a value.

Standard fixes, in increasing order of faithfulness to the literature: a LayerNorm/RMSNorm on `y` before
`W_o` (what GLA and RetNet do), or an explicit denominator `z_t = a z_{t-1} + k`, `y = Sᵀq / (qᵀz + ε)` (what
the original linear-transformer does). **Recommend the norm** — it is cheap, it is in the read path so it
needs no trace and stays exact, and it does not introduce the numerical fragility the denominator has when
`qᵀz → 0`.

### 6.5 Capacity: 16 keys into a `dk=16` state

Phase 1's alphabet is 16 keys. `fast` is specified at `dk = 16` per head. A plain outer-product sum stores
`m` pairs in a `dk × dv` matrix with cross-talk `O(√(m/dk))` for random near-orthogonal keys; at `m ≈ dk` you
are at the interference boundary, and the read of key `p` returns `v_p` plus a sum of `m−1` noise terms of
comparable scale. Four heads help only if the key projections decorrelate across heads, which nothing in the
design enforces.

So Phase 2's prediction — "`fast` with `tbptt` passes 90% at gap 128" — could fail for **capacity** reasons
that say nothing about eligibility traces. That is a bad outcome for a project whose whole point is to isolate
the trace question.

Two cheap insurances, neither of which touches the trace maths:
- **`dk = 32`** (still `dk·dv·d` small under the scalar-decay collapse), giving `2×` headroom over the key count.
- **A delta-rule write** — `S ← S(I − βkkᵀ) + βkvᵀ`, i.e. erase-then-write instead of accumulate — which is
  what actually fixes associative recall in the fast-weight literature. **But note the cost:** the delta rule
  makes the state transition non-diagonal (`I − βkkᵀ`), so the clean row-local trace in §6.1 no longer
  applies and `fast` would need a different derivation. Do not adopt it inside this plan's trace framework
  without redoing §6 from scratch. Mention it in the write-up as the obvious next cell; keep it out of Phase 0.

Take the `dk = 32` insurance. Leave the delta rule to a follow-up.

---

## 7. Shapes and memory complexity, with numbers

All figures: fp32, per stream, `dim = 384`, `L = 4` layers, `fast` with `H = 4` heads at `dk = dv = 16`.

### 7.1 Per-cell trace inventory

| Cell | Trace | Shape | Floats/layer | × `L=4` |
|---|---|---|---|---|
| `leaky` | `E_δ` | `(n,)` | 384 | 1,536 |
| | `T` (embedding, layer 0 only) | `(256, n)` | 98,304 | 98,304 |
| | **total/stream** | | | **99,840 ≈ 0.40 MB** |
| `gated` | `E_δ` | `(n,)` | 384 | 1,536 |
| | `E_V` | `(n, d_in)` | 147,456 | 589,824 |
| | `E_G` | `(n, d_in)` | 147,456 | 589,824 |
| | **total/stream** | | | **1,181,184 ≈ 4.7 MB** |
| `fast`, elementwise `A` | `E_A` | `(H, dk, dv)` | 1,024 | 4,096 |
| | `E_Wk` | `(H, dk, dv, d)` | 393,216 | 1,572,864 |
| | `E_Wv` | `(H, dk, dv, d)` | 393,216 | 1,572,864 |
| | **total/stream** | | | **3,149,824 ≈ 12.6 MB** |
| `fast`, **scalar `A`** | `E_A` | `(H,)` | 4 | 16 |
| | `Ê_Wk` | `(H, dv, d)` | 24,576 | 98,304 |
| | `Ê_Wv` | `(H, dk, d)` | 24,576 | 98,304 |
| | **total/stream** | | | **196,624 ≈ 0.79 MB** |

### 7.2 The `online-batched` multiplier — the formula the plan needs

Traces are **per stream**. The plan's §5 says "if a cell's traces don't fit at 32, use the largest batch that
does and report it" without giving a way to know in advance:

```text
trace_bytes = 4 · B · Σ_traced |E|
```

| Cell | B=1 | B=32 |
|---|---|---|
| `leaky` | 0.4 MB | **12.8 MB** |
| `gated` | 4.7 MB | **151 MB** |
| `fast`, elementwise `A` | 12.6 MB | **403 MB** |
| `fast`, **scalar `A`** | 0.79 MB | **25 MB** |

403 MB of pure trace, before parameters, Adam moments, activations, or MPS allocator slack. That is the number
that decides whether `fast` runs at `B = 32`, and §6.3 removes it for free.

### 7.3 Complexity, stated generally

Let `P_w` = number of traced (write-path) parameters, `n` = state size, `B` = streams.

| Quantity | General RTRL | This plan (row-local, diagonal) |
|---|---|---|
| Trace memory | `O(B · n · P_w)` | `O(B · P_w)` |
| Trace update time/step | `O(B · n² · P_w)` (dense `A J`) | `O(B · P_w)` |
| Gradient extraction/step | `O(B · n · P_w)` | `O(B · P_w)` |
| Activation memory | `O(1)` in `T` | `O(1)` in `T` — **the actual selling point** |

Compare TBPTT at window `T`: activation memory `O(B · T · n · L)`, time `O(B · T · P)`, exact within the
window and zero outside it. The trade is **constant memory in `T` and unbounded horizon** against
`O(P_w)`-per-step state and the multi-layer bias of §8.

### 7.4 A throughput prediction, made before the runs

Trace updates are elementwise passes over parameter-sized tensors, so they are **memory-bandwidth bound**, not
FLOP bound. Bytes touched per optimizer step ≈ `2 × trace_bytes` (read + write). At `B = 32` and an assumed
~100 GB/s of usable unified-memory bandwidth:

| Cell @ B=32 | Trace traffic/step | Predicted ms/step | Predicted bytes/s |
|---|---|---|---|
| `gated` | ~300 MB | ~3 ms | ~10,000 |
| `fast`, elementwise | ~806 MB | ~8 ms | ~4,000 |
| `fast`, scalar `A` | ~50 MB | ~0.5 ms | ~64,000 |

Order-of-magnitude only — it ignores kernel launch overhead, which at one step per byte is likely to dominate
on MPS and make these **optimistic**. Against smLLM_01's 26,700 bytes/s, this predicts **Phase 3's gate fails
for `gated` and elementwise-`fast`**, matching the plan's own stated prediction but now with a mechanism. It
also says the scalar-decay `fast` cell is the only configuration with a realistic shot, which is a second,
independent argument for §6.3.

Worth adding to `predictions.md` before the runs — it is a falsifiable, mechanism-level prediction rather
than a hunch, which is the kind this project is supposed to be making.

---

## 8. The multi-layer approximation, stated precisely

The plan says: "The path from a layer's input `x_t` back through lower layers' *past* states is cut, the same
for every cell; that's the approximation issue #2 describes, and it's stated, not hidden." That is directionally
right but too vague to test against, and Phase 0's deliverable is supposed to be a *number*.

### 8.1 What is actually dropped

With the residual stack `x_t^{(l)} = x_t^{(l-1)} + SiLU(W^{(l-1)} LN(s_t^{(l-1)}))`, the write at layer `l`
depends on the state at layer `l−1` **at the same time step**. Full RTRL for a parameter `θ` in layer `l'`
would require carrying, for every layer `l > l'`:

```text
J_t^{(l)} = A^{(l)} J_{t-1}^{(l)}  +  (∂b_t^{(l)}/∂x_t^{(l)}) (∂x_t^{(l)}/∂s_t^{(l-1)}) J_t^{(l-1)}
            └──── kept ────┘         └──────────── kept only for the τ = t term ────────────┘
```

The scheme keeps the first term (layer `l`'s own temporal recursion) and lets autograd supply the second term
**only at `τ = t`**. What is dropped is the `A^{(l)}`-decayed accumulation of cross-layer contributions from
`τ < t`.

**The clean statement, which the plan should use verbatim:**

> The online estimator retains exactly those gradient paths that traverse **at most one temporal segment**:
> any number of hops *up* the stack within a single time step, plus one run along time within a single layer.
> It drops every path that mixes — that moves forward in time in layer `l`, then up to layer `l' > l`, then
> forward in time again. For an `L`-layer stack the retained set is a strict subset of the true path set for
> all `L ≥ 2`, and the dropped mass grows with both depth and the dependency length being learned.

This is precisely the **SnAp-1** approximation of Menick et al. — sparsity pattern = "each parameter's own
immediate influence subgraph" — applied per layer, and it is what e-prop does with its learning signal. The
plan should name it; "SnAp-1 across the depth axis" is a one-line description that a reader can go look up,
and it makes the plan's open question #2 (a GRU control) answerable by citation rather than experiment.

### 8.2 The bias is systematic, so report it as a direction

Dropped terms are not zero-mean noise. They are a **deterministic, systematically signed** omission —
concretely, they are the terms that would credit lower layers for forming state that upper layers read later,
which is *exactly the computation the recall task requires*. So the bias is worst precisely where the project
cares most. UORO and KF-RTRL trade this bias for variance (§10); this scheme takes the bias.

The plan says "For a 2-layer model, report the gap (nonzero is expected); it isn't a gate." A scalar gap is
the least informative thing to report. **Report instead:**

1. **Per-parameter-group cosine similarity** `cos(g_online, g_bptt)` — does the estimator point the right way,
   which is what actually matters for SGD convergence. A cosine of 0.9 with a wrong magnitude is fine; a
   cosine of 0.1 with the right norm is not.
2. **Relative norm** `‖g_online‖ / ‖g_bptt‖` per group — systematic under-estimation of lower layers is the
   predicted signature, and it is directly fixable with per-layer LR scaling if it shows up.
3. **Both as a function of depth `L ∈ {1, 2, 4}` and of sequence length `T ∈ {8, 32, 128, 512}`.** The theory
   above predicts cosine degrades monotonically in both. If it does not, something else is wrong.

That table is a genuinely publishable Phase 0 artifact and costs one extra afternoon. Right now Phase 0
produces one number that nobody can interpret.

### 8.3 Why Phase 2 needs a 1-layer arm

**One layer is the only configuration where the online gradient is exact** (given §5.4's fix). At `L = 4`, a
`fast` cell that fails recall has two candidate explanations — the cell cannot bind, or the truncated gradient
cannot train it — and the experiment as designed **cannot separate them**. That is a confounded experiment on
the project's headline question.

A 1-layer `fast` arm resolves it: binding happens *within* one layer's state, so if 1-layer-online works and
4-layer-online does not, the multi-layer truncation is the culprit, and if neither works, the cell is. Cost is
one extra 10-minute run per cell (~30 min of Mac time against a ~4 hour budget).

---

## 9. What is wrong with the Phase 0 gate

### 9.1 The gate as written would pass a broken trace

> "a 1-layer model with dim 8 on a 6-byte sequence ... Every parameter matches to 1e-4"

At default init `δ = 0`, `a = 0.5`. The carry term `a·E` after `6` steps contributes `0.5^5 ≈ 3%` of the
trace's mass. **An implementation that drops the carry term entirely — the single most likely bug, and
precisely the bug that makes the model a bag-of-bytes with no memory — would match to within a few percent
and could plausibly slip under a loosely applied tolerance.** The test does not test the thing it exists to
test.

**Fix, all four parts:**

| Change | From | To | Why |
|---|---|---|---|
| Sequence length | 6 | **≥ 32, plus a 128 run** | the carry term must dominate |
| Decay | init-default (`a≈0.5`) | **frozen at `a = 0.99` for the primary check**, then a second run at the spread init of §4.3 | at `a=0.99` the trace's 32-step mass is 72% carry; a dropped carry fails by 3× |
| Precision | implied fp32 | **float64 throughout** | fp32 accumulation noise over 128 steps is itself near 1e-4 |
| Tolerance | absolute 1e-4 | **relative, per group: `‖g_trace − g_bptt‖ / (‖g_bptt‖ + 1e-12) < 1e-6`** | absolute tolerance is simultaneously too loose on large grads and too tight on small ones |

At float64 with a correct implementation the agreement should be ~1e-12, i.e. **machine precision, not
"close"**. Anything meaningfully above that is a bug, and a 1e-6 threshold has enormous margin. If the check
only reaches 1e-5, do not raise the tolerance — find the bug.

### 9.2 "Every parameter" is currently false

Per §5.4 (F1) and §5.3 (biases). Either adopt the one-hot-input fix and keep the strong wording, or weaken
the wording to name the excluded set explicitly. **Do not leave it as-is** — a gate that cannot pass will be
"fixed" by whoever is on shift by loosening the tolerance, and then the project has no gradient check at all.

### 9.3 The other two gates need sharper criteria

**"Constant memory."** "10,000 online steps with the process's memory flat after warm-up" — RSS under Python +
torch allocators is noisy enough that "flat" is a judgement call. Assert mechanisms, not symptoms:

```text
assert all(e.grad_fn is None for e in every trace tensor)       # nothing retained
assert torch.mps.current_allocated_memory() stable to <1% over steps 1000..10000
assert len(gc.get_objects()) stable                              # catches Python-side retention
```

The first assertion is the one that actually catches "graph retained across steps", and it catches it in one
step rather than ten thousand.

**"Same loss, same numbers."** "The three trainers give the same loss on step 1 from the same seed" is not
well posed across `batch=1`, `batch=32` and `batch=64` trainers — the mean over a batch is not the batch-1
value. State it as: *"mean per-token loss on the first token of stream 0, identical init, matches to 1e-6
across all three trainers."*

**Missing gate — the double-count detector.** Add the per-group ratio test from §3: `‖g_trace‖/‖g_autograd‖`
by parameter group, expected `1.0`, with `~2.0` on traced groups being the signature of trap 2. This catches
in seconds what would otherwise show up as "training is a bit unstable" three phases later.

### 9.4 Exactness does not survive the optimizer

The gradient check correctly accumulates "over the 6 steps with no optimizer step". Good. But the *training
regime* takes one Adam step per byte, and **RTRL's exactness proof assumes `θ` is fixed over the accumulation
window.** After an update, `E_t` holds sensitivities computed under parameters that no longer exist. The
resulting error is `O(η · T_eff · ‖∂²s/∂θ²‖)` where `T_eff ≈ 1/(1−a)` is the trace horizon — which means
**the error grows with exactly the half-life you are trying to increase.** Long memory and online updating
pull against each other. That tension is arguably the most interesting thing in this project and the plan
does not mention it.

This is not a reason to abandon the approach — e-prop and RFLO both live with it — but the plan should say
that what Phase 0 proves (exactness at fixed `θ`) is not what Phase 2 relies on (approximate gradients under
a moving `θ`).

**Add a secondary, non-gating measurement:** run the check *with* the optimizer at the real training LR over
256 steps and report the cosine against BPTT-through-the-same-trajectory. That number tells you whether the LR
is inside the regime where the trace means anything, and it is the natural thing to consult when Phase 2
misbehaves. Cheap: it reuses the Phase 0 test harness entirely.

---

## 10. Comparison of online-gradient methods

> Reminder: §10's citations are from memory and **were not link-checked in this session** (no web access).
> Confidence markers: **H** = I would state this without checking, **M** = verify the ID before quoting.

`n` = state size, `P` = parameter count, `T` = truncation window.

| Method | Memory | Time/step | Bias | Variance | Applies to this plan's cells? |
|---|---|---|---|---|---|
| **TBPTT** | `O(T·n)` activations | `O(T·P)` | exact inside window, **totally blind outside it** | none | yes — the `tbptt` control |
| **RTRL** (Williams & Zipser 1989) | `O(n·P)` → `O(n³)` dense | `O(n²·P)` → `O(n⁴)` dense | **none** | none | intractable dense; **tractable here** because the recurrence is diagonal and the writes are row-local |
| **This plan / e-prop / RFLO** | `O(P_w)` | `O(P_w)` | **exact at 1 layer**; SnAp-1-style bias across layers | none | this is the scheme |
| **UORO** (Tallec & Ollivier) | `O(n + P)` | `O(P)` | **unbiased** | high, grows with `n` | yes — a drop-in unbiased cross-check |
| **KF-RTRL** (Mujika et al.) | `O(n²)` | `O(n²)` | unbiased | `O(n)` vs UORO's `O(n²)` | needs a Kronecker-compatible transition; the diagonal cells qualify trivially, so the gain over the exact scheme is nil |
| **OK / Optimal Kronecker-Sum** (Benzing et al.) | `O(n²)` | `O(n²)` | unbiased | lower than KF-RTRL | same caveat |
| **SnAp-`k`** (Menick et al.) | tunable by sparsity | tunable | exact at `k = ∞` | none | **`k=2` is the principled fix for §8's cross-layer truncation** |
| **Forward-mode AD / JVP** | `O(P)` per tangent | `O(P)` per tangent | exact per direction | — | gives `J·v`, so a *full* gradient needs `P` passes — not usable directly |
| **Directional descent** (Silver et al.) | `O(P)` | `O(P)` | unbiased | very high | same family as UORO; a one-line baseline |
| **Chunkwise-parallel linear attention** (GLA, DeltaNet) | `O(chunk · n)` | `O(P)` amortised | exact within chunk | none | **the strongest practical alternative for `fast`** — same recurrent state, orders of magnitude better hardware utilisation |

### 10.1 The three things worth acting on

**(a) UORO as a cheap unbiasedness cross-check.** ~40 lines. Because it is unbiased, averaging UORO gradients
over many steps converges to the true gradient, so it gives you a **reference for the multi-layer gap that
does not require BPTT** — and therefore works at sequence lengths where BPTT is infeasible. That is the
cleanest way to measure §8's bias at the 512-byte gaps the task actually uses. Strong candidate for the Phase 0
deliverable list.

**(b) Zucchet et al., "Online learning of long-range dependencies" (NeurIPS 2023).** **The single most
relevant paper to this plan, and the one I could not fetch.** As I recall it, the result is that for deep
stacks of *independent-recurrence* (diagonal) linear RNNs with nonlinear inter-layer connections — which is
structurally what all three cells here are — RTRL becomes practical at roughly BPTT's memory cost, and
critically it **handles the cross-layer term that §8 drops**. If that recollection is right, the plan's
central approximation may be avoidable rather than inherent, which would change Phase 0's design. **Someone
with a browser should read this before `cells.py` is frozen.** I am flagging it as the highest-value
verification task coming out of this memo.

**(c) Chunkwise-parallel forms for the `tbptt` control.** GLA and DeltaNet ship chunkwise algorithms that
compute the *same* recurrence in a matmul-heavy form. If the `tbptt` control is implemented naively as a
`for t in range(256)` Python loop it will be bandwidth- and launch-bound, and Phase 3's headline number —
"online costs `X×` more per bit than TBPTT" — will be **flattering to the online regime because the control
was slow**. A chunkwise control is the honest comparison. At minimum, state in the write-up which TBPTT
implementation was used and that a chunkwise one would be faster.

### 10.2 Bibliography

| Ref | Where | Conf. |
|---|---|---|
| Williams & Zipser, "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks", *Neural Computation* 1(2):270–280, 1989 — https://doi.org/10.1162/neco.1989.1.2.270 | RTRL, the origin | **H** |
| Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass, "A solution to the learning dilemma for recurrent networks of spiking neurons", *Nature Communications* 11:3625, 2020 — https://www.nature.com/articles/s41467-020-17236-y · arXiv:1901.09049 | e-prop; eligibility trace + learning signal factorisation | **H** (DOI H, arXiv ID M) |
| Murray, "Local online learning in recurrent networks with random feedback", *eLife* 8:e43299, 2019 | RFLO — **the closest published analogue of the `leaky` cell's trace** | **H** (article no. M) |
| Tallec & Ollivier, "Unbiased Online Recurrent Optimization", ICLR 2018 — arXiv:1702.05043 | UORO | **H** (ID M) |
| Mujika, Meier, Steger, "Approximating Real-Time Recurrent Learning with Random Kronecker Factors", NeurIPS 2018 — arXiv:1805.10842 | KF-RTRL | **H** (ID M) |
| Benzing, Gauy, Mujika, Martinsson, Steger, "Optimal Kronecker-Sum Approximation of Real Time Recurrent Learning", ICML 2019 — arXiv:1902.03993 | OK | **M** |
| Menick, Elsen, Evci, Osindero, Simonyan, Graves, "Practical Real Time Recurrent Learning with a Sparse Approximation", ICLR 2021 — arXiv:2006.07232 | **SnAp — names §8's approximation** | **H** (ID M) |
| Marschall, Cho, Savin, "A Unified Framework for Online Learning Algorithms of Recurrent Neural Networks", JMLR 2020 — arXiv:1907.02649 | **best single survey; read this first** | **H** (ID M) |
| Irie, Gopalakrishnan, Schmidhuber, "Exploring the Promise and Limits of Real-Time Recurrent Learning", ICLR 2024 — arXiv:2305.19044 | **verified after the agent run; directly relevant to element-wise recurrence and multi-layer RTRL cost** | **H** |
| Zucchet, Meier, Schug, Mujika, Sacramento, "Online learning of long-range dependencies", NeurIPS 2023 — arXiv:2305.15947 | **verified after the agent run; most relevant to independent recurrent modules** | **H** |
| Katharopoulos, Vyas, Pappas, Fleuret, "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", ICML 2020 — arXiv:2006.16236 | linear attention; the `fast` cell's ancestor + its denominator | **H** |
| Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers", ICML 2021 — arXiv:2102.11174 | **delta rule vs plain outer-product sum — §6.5** | **H** (ID M) |
| Irie, Schlag, Csordás, Schmidhuber, "Going Beyond Linear Transformers with Recurrent Fast Weight Programmers", NeurIPS 2021 — arXiv:2106.06295 | recurrent FWPs | **M** |
| Yang, Wang, Shen, Panda, Kim, "Gated Linear Attention Transformers with Hardware-Efficient Training", ICML 2024 — arXiv:2312.06635 | GLA; `A[i,j] = α_i` decay of §6.3; chunkwise form | **H** (ID M) |
| Yang, Wang, Zhang, Kim, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", NeurIPS 2024 — arXiv:2406.06484 | chunkwise DeltaNet | **M** |
| Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models", 2023 — arXiv:2307.08621 | **scalar per-head decay — the §6.3 collapse** | **H** (ID M) |
| Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De, "Resurrecting Recurrent Neural Networks for Long Sequences", ICML 2023 — arXiv:2303.06349 | LRU; **timescale-spread init of §4.3** | **H** (ID M) |
| Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023 — arXiv:2312.00752 | selective/input-dependent decay; `Δ` init | **H** |
| Silver, Goyal, Danihelka, Hessel, van Hasselt, "Learning by Directional Gradient Descent", ICLR 2022 — arXiv:2202.08587 | forward-mode single-tangent | **M** |
| `torch.func.jvp` — https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html | forward-mode AD in PyTorch | **H** |

---

## 11. Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Double-counted `τ=t` term → silent 2× on traced grads | **high** — the plan's stated "dummy tensor" recipe does not preclude it | training instability blamed on the method, not the bug | §3 severed-state recipe + per-group ratio test |
| R2 | Gate passes a trace with no carry term | **high** at 6 steps / `a=0.5` | Phase 0 certifies a memoryless model; every later phase is meaningless | §9.1: 32+ steps, `a=0.99`, float64, relative tol |
| R3 | Gate cannot pass as written (untraced embedding/biases) | **certain** for `gated`/`fast` | someone loosens the tolerance and the check stops meaning anything | §5.4 one-hot input; `bias=False` |
| R4 | `δ` never escapes half-life ≈ 1 byte | **high** — it is the source repo's own issue #3 | all cells fail all gaps; project concludes "traces don't work" when it measured "init doesn't work" | §4.3 spread init, plus a `δ=0` arm for faithfulness |
| R5 | `fast` traces (403 MB @ B=32) don't fit | medium | `online-batched` silently drops to B=4 and the throughput comparison is apples-to-oranges | §6.3 scalar decay → 25 MB |
| R6 | 16 keys into `dk=16` → interference, not a trace failure | medium | Phase 2's key prediction fails for the wrong reason | `dk = 32` |
| R7 | Unnormalised `Sᵀq` leaks gap info through output magnitude | medium | above-chance recall that isn't recall — a **false positive on the project's main gate** | §6.4 LayerNorm on `y` |
| R8 | Optimizer staleness breaks the property Phase 0 proved | **certain**, magnitude unknown | Phase 0's guarantee doesn't transfer to Phase 2 and nobody notices | §9.4 secondary measurement + state it in the write-up |
| R9 | 4-layer-only Phase 2 confounds "cell can't bind" with "gradient is truncated" | **certain** as designed | headline question unanswerable | §8.3 add 1-layer arms |
| R10 | Naive Python-loop TBPTT control is slow | medium | the cost-per-bit comparison **flatters** online | use/cite a chunkwise form, or state the caveat |
| R11 | `d` overloaded (width vs decay logit) in `cells.py` | medium | an afternoon | rename `dim` / `decay_logit` |
| R12 | §10 citations unverified (no web access this session) | **certain** | a wrong arXiv ID in a public README | verify before §10 is quoted anywhere outside this file |

---

## 12. Plan changes, ranked

### Must

1. **Fix the gate so it can pass.** Adopt the one-hot input for `gated`/`fast` (§5.4 option 1) so every
   learned parameter is traced or non-recurrent; set `bias=False` on write projections or trace the biases
   (§5.3). Then "every parameter matches" is an honest claim.
2. **Strengthen the gate so it can fail.** ≥32 steps (plus a 128-step run), decay frozen at `a=0.99` for the
   primary check, float64, relative per-group tolerance `< 1e-6`. Add the double-count ratio test. (§9.1, §3)
3. **Fix the decay initialisation.** Log-uniform half-lives over [1, 1024] bytes (§4.3). Keep a `δ=0` arm as
   the faithful reproduction of the source repo — the delta between those two rows is a headline result.
4. **Specify the shape of `a` for `fast`, and add its decay trace** (§6.2, §6.3). The plan currently cannot be
   implemented as written for that cell.
5. **Write the state-severing recipe into the plan** (§3), replacing the under-specified "dummy tensor" line,
   and rename `dL/ds_t` to `ḡ_t` with the immediate-vs-total distinction spelled out (§2.1).

### Should

6. **Scalar per-head decay for `fast`** — exact 16× trace-memory collapse, 403 MB → 25 MB at B=32, and two
   clean matmuls instead of a rank-3 elementwise pass (§6.3). Initialise the 4 heads at half-lives
   ≈ {8, 32, 128, 512}.
7. **Add the leaky embedding trace equation** to the plan (§4.2) and state the row-locality condition that
   makes every closed form in the plan possible (§5.2).
8. **Add the memory formula and the table** — `trace_bytes = 4·B·Σ|E|` — so `online-batched` batch sizes are
   chosen before the run, not discovered during it (§7.2).
9. **Add 1-layer arms to Phase 2** (§8.3). The only configuration where the gradient is exact; without it the
   headline question is confounded.
10. **Report the multi-layer gap as cosine + relative norm per parameter group, across `L ∈ {1,2,4}` and
    `T ∈ {8,32,128,512}`** instead of one scalar (§8.2).
11. **`dk = 32`** for `fast` (§6.5), and a **LayerNorm on `y`** before `W_o` (§6.4) — the latter closes a
    false-positive path on the project's main gate.
12. **Say that exactness does not survive the optimizer**, and add the with-optimizer cosine as a
    non-gating number (§9.4).
13. **Verify §10's citations**, in particular Zucchet et al. — if their result covers stacked diagonal RNNs,
    the plan's central approximation may be avoidable and Phase 0's design changes (§10.1b). Do this before
    `cells.py` is frozen.

### Could

14. **UORO as an unbiased reference** (~40 lines) to measure §8's bias at 512-byte gaps where BPTT is
    infeasible (§10.1a).
15. **Name the approximation "SnAp-1 across depth"** in the plan and the write-up, and note SnAp-2 as the
    principled fix — turns open question #2 into a citation (§8.1).
16. **Add the §7.4 bandwidth-based throughput prediction to `predictions.md`** before the runs. Mechanism-level
    and falsifiable.
17. **Mention the delta rule as the obvious follow-up cell** in Phase 5's "what's next", with the explicit
    warning that its non-diagonal transition invalidates §6's traces and needs a fresh derivation. Keep it
    out of Phase 0.
18. **Use a chunkwise-parallel TBPTT control**, or state in the write-up that the control was a naive loop and
    the online-vs-TBPTT ratio is therefore a lower bound on the true cost gap (§10.1c).

---

## 13. Corrected Phase 0 equation block, ready to paste

Drop-in replacement for the plan's §"Phase 0" equations. Notation per §2.1: `ḡ_t := ∂L_t/∂s_t`, the
**immediate** partial, obtained by severing the state (§3). `δ` is the decay logit, `a = σ(δ)`.

```text
COMMON RECURRENCE          s_t = a ⊙ s_{t-1} + b_t,        a = σ(δ)

  Exact iff (i) x_t does not depend on any past state — true at layer 0 only;
            (ii) every learned parameter appearing in b_t is traced;
            (iii) s_{t-1} is detached and b_t is severed from the live graph,
                  so backward() returns ḡ_t and the τ=t term is counted once.

DECAY (all cells)
  E_δ[i]  ← a_i · E_δ[i] + a_i(1 − a_i) · s_{t-1}[i]        grad δ[i] += ḡ_t[i] · E_δ[i]
  fast, matrix state:
  E_A[i,j] ← A[i,j] · E_A[i,j] + A[i,j](1−A[i,j]) · S_{t-1}[i,j]
      δ elementwise : grad δ[i,j] += Ḡ[i,j] · E_A[i,j]
      δ per key idx : grad δ[i]   += Σ_j Ḡ[i,j] · E_A[i,j]
      δ per head    : grad δ      += Σ_ij Ḡ[i,j] · E_A[i,j]

LEAKY   b_t = x_t = Emb[c_t]                        traced: {δ, Emb at layer 0}
  T[c,m] ← a_m · T[c,m] + 1[c_t = c]                grad Emb[c,m] += ḡ_t[m] · T[c,m]
  shape (256, n). Decays on all 256 rows every step — the update is dense, not sparse.

GATED   u = V x,  g = σ(G x),  b_i = g_i · u_i      traced: {δ, V, G}  (+ biases if present)
  E_V[i,k] ← a_i · E_V[i,k] + g_i · x_k                     grad V[i,k] += ḡ_t[i] · E_V[i,k]
  E_G[i,k] ← a_i · E_G[i,k] + g_i(1−g_i) · u_i · x_k        grad G[i,k] += ḡ_t[i] · E_G[i,k]
  E_bV[i]  ← a_i · E_bV[i]  + g_i                           grad bV[i]  += ḡ_t[i] · E_bV[i]
  E_bG[i]  ← a_i · E_bG[i]  + g_i(1−g_i) · u_i              grad bG[i]  += ḡ_t[i] · E_bG[i]
  NOTE: at layer 0, x = one-hot(c_t) fed directly into V and G — no separate embedding.
        Otherwise Emb enters b_t untraced and the 1-layer gradient check cannot pass.

FAST    k = W_k x,  v = W_v x,  S_t = A ⊙ S_{t-1} + k vᵀ    traced: {δ, W_k, W_v}
        read y = LayerNorm(Sᵀ q),  q = W_q x,  out = W_o y   Ḡ := ∂L_t/∂S_t

  general A (dk × dv):
    E_Wk[i,j,m] ← A[i,j] · E_Wk[i,j,m] + v_j · x_m    grad W_k[i,m] += Σ_j Ḡ[i,j] · E_Wk[i,j,m]
    E_Wv[i,j,m] ← A[i,j] · E_Wv[i,j,m] + k_i · x_m    grad W_v[j,m] += Σ_i Ḡ[i,j] · E_Wv[i,j,m]
    memory 2·dk·dv·d per head  →  12.6 MB/stream at 4 heads × 4 layers, 403 MB at B=32

  scalar per-head A = a  (RECOMMENDED — exact, no approximation, 16× less memory):
    Ê_Wk ← a · Ê_Wk + v xᵀ   ∈ R^{dv×d}               grad W_k = Ḡ  @ Ê_Wk     (dk×dv)(dv×d)
    Ê_Wv ← a · Ê_Wv + k xᵀ   ∈ R^{dk×d}               grad W_v = Ḡᵀ @ Ê_Wv     (dv×dk)(dk×d)
    memory (dv+dk)·d per head  →  0.79 MB/stream, 25 MB at B=32
    Collapse rule: E_Wk's drive v_j x_m has no i-index and E_Wv's k_i x_m has no j-index, so
    each trace is constant along any index the decay is also constant along.

READ PATH  W, W_q, W_o, LayerNorm, decoder, stop head: no trace.
           Plain autograd on L_t is already exact for them. Disjoint from the traced set,
           which is what makes double-counting structurally impossible.

INIT       half-lives log-uniform over [1, 1024] bytes:
             u_i ~ U(0,10);  h_i = 2^{u_i};  a_i = 2^{−1/h_i};  δ_i = ln(a_i/(1−a_i))
           At δ=0, a=0.5, half-life = 1 byte — the source repo's issue #3, reproduced.
           Keep a δ=0 arm for faithfulness; report both.
```

---

*End of memo. No code was written, no model trained, no file outside `smRTS_01/research/trace_math.md` touched.*
