# smMATH001-a: whole expressions, worked out step by step

Project: [`smMATH001-a/`](../../../smMATH001-a/README.md). Checked against `work.py`, `model.py`, `train.py`,
`solve.py`, `tests/test_work.py`, `out/train.log`, `out/results.json` and the README on 2026-09-15.

> **Trained.** 10,000 steps finished on 2026-09-15 at 18:09. Results are from `out/results.json` and `out/train.log`.

## Purpose

The owner's reason (session notes, 2026-09-15): Math Language writes whole expressions like `322234*21323*212231`, and
smMATH01 only takes one whole-number step at a time.

smMATH001-a takes the whole expression and writes out its working, including the partial products of long
multiplication and the steps of long division, then the answer. So multiplication and division can be done by a
model, not just `+` and `-`.

## Input and output

**Numbers are fixed-point:** every number is scaled by 100 and written ones digit first. 12.5 is 1250, written `0521`.
Digits line up by place, decimals are just digits, and carries run in the direction the model writes.

**The question** is the expression with scaled, reversed numbers and only the brackets it needs (`work.question()`). The
vocabulary is `0123456789+-*/()=:;>` plus `SEP`, `END` and `PAD`.

**The work** is a list of steps joined by `;`, in the order the expression is evaluated: brackets, then `*` and `/`,
then `+` and `-`.

| Operation | Written as |
|---|---|
| `A+B`, `A-B` | `A+B=C`. `C` gets a trailing `-` when it's negative, which only the last step may be |
| `A*B` | `A*B:`, then one `A*D=P` per nonzero digit `D` of the multiplier at its place, a running sum `S+P=S'` from the second digit on, then `S>T`, which drops the two extra decimal places (floor) |
| `A/B` | `A/B:`, long division of `A×100` by `B`, biggest place first: `B*Q=P` and `R-P=R'` for each nonzero quotient digit, then `=Q` (cut to 2 places, never rounded up) |

The multiplier is whichever number has fewer nonzero digits.

Example (`work.py` docstring and tests):

```text
expression  (12+8)/5
question    (0021+008)/005
work        0021+008=0002;0002/005:005*004=000002;000002-000002=0;=004
answer      004 → 400 → 4
readable    12 + 8 = 20 / 20 ÷ 5: / 5 × 4 = 20 / 20 − 20 = 0 / = 4
```

`final_answer(work)` reads the last step's number back. `readable(work)` turns the work into lines a person can follow,
leaving any garbled line as written.

## Architecture and size

`MathGPT` in `model.py`:

- **Where a token is:**
  - rotary positions (RoPE) inside attention, so "the line before this one" is a distance it can find anywhere;
  - a place label on every digit (its place inside its own number, up to 64), added to the token embedding. In
    training, all labels in a line shift by the same random amount, 0 to 16 (`--offset-max`), so long numbers' places
    get trained.
- **Body:** d 256, 6 layers, 8 heads, 4× GELU MLP, pre-LayerNorm, no dropout, tied output head; `max_len` 2048 tokens.
- **Writing** (`work()`) is greedy, with a key/value cache, so each new token costs one short step. A test checks that
  cached writing matches a full forward pass.

**4.75M parameters** (`out/train.log`).

## Data

`work.py`, all generated:

- **Expression shapes:**
  - **30% "harness-like":** the shapes Math Language writes, like 3-number volumes, `n*pct/100`, `money*pct/100`,
    money plus or minus tax, `money/k`, 3-number averages, `k*money+fee`, rates, and dozens;
  - **70% random trees** with 1, 2 or 3 operations (weights 55/30/15), where `+ - * /` are weighted 3 / 2.5 / 3 / 1.5.
- **Numbers:** 30% are decimals with 1 or 2 places; the rest are whole numbers of 1 to 6 digits.
- **Rejected:** anything that goes below zero partway through, and anything past the training limits:
  - scaled numbers over 9 digits;
  - a multiplier with more than 3 nonzero digits;
  - a divisor over 6 digits;
  - work over 420 characters or a question over 64.
- **This run** (`out/train.log`): 292,689 worked solutions, after dropping 7,311 that matched a check question. Average
  129 tokens, longest 452.
- **Check sets:**
  - 6,000 validation examples (seed + 1000), grouped into up to 300 each of add, subtract, multiply, divide, mixed,
    decimals and harness-like;
  - 150 each of six "beyond training size" sets: 8–10 digit addition and subtraction, 4-digit multipliers, 5–6 digit
    multipliers, 8–9 digit dividends, and 5-digit divisors.

## Train and test

```bash
python3 -m unittest discover -s tests
python3 train.py
python3 solve.py "(12+8)/5"
```

- **Batching:** 10,000 steps. Each batch is a run of neighbours from the length-sorted pool, about 8,192 tokens
  including padding.
- **Optimizer:** AdamW lr 6e-4, betas 0.9/0.98, weight decay 0.1; warmup 300 steps, cosine to lr/10; gradient clip
  1.0. The loss only counts the work.
- **Every 1,000 steps:** exact-work rate on 100 examples per category, then `out/math.pt` is saved.
- **At the end, `out/results.json` gets:**
  - `trained_sizes`: whole work exactly right per category;
  - `beyond`: the same for the six beyond sets;
  - `greedy_spot_check`: 15 per category, with answer right and work exact;
  - `ms_per_token_cpu`;
  - `volume_example`: what it writes for `322234*21323*212231`.
- **Tests (6).** All flags: [running.md](running.md#smmath001-a).

## Results

**2026-09-15 18:09** (`out/results.json`): 10,000 steps in 1,913 s (31.9 minutes) on the MacBook Pro M5; 4,747,520
parameters.

Whole worked solution exactly right on held-out examples, with the greedy spot check (15 per kind, written on the CPU):

| Kind, at training sizes (300 each) | Whole work exactly right | Greedy spot check: answer right / work exact / of |
|---|---|---|
| add | 100% | 15 / 15 / 15 |
| subtract | 98.7% | 14 / 14 / 15 |
| multiply | 96.7% | 14 / 14 / 15 |
| harness-like | 84.7% | 13 / 13 / 15 |
| mixed | 84.3% | 14 / 14 / 15 |
| decimals | 81.7% | 14 / 14 / 15 |
| divide | **38.0%** | 7 / 7 / 15 |

| Kind, beyond training sizes (150 each) | Whole work exactly right | Greedy spot check |
|---|---|---|
| add, 8–10 digit numbers | 6.7% | 1 / 1 / 15 |
| subtract, 8–10 digit numbers | 0% | 0 / 0 / 15 |
| multiply, 4-digit multiplier | 20.7% | 4 / 4 / 15 |
| multiply, 5–6 digit multiplier | 0% | 0 / 0 / 15 |
| divide, 8–9 digit dividend | 0% | 0 / 0 / 15 |
| divide, 5-digit divisor | 0% | 0 / 0 / 15 |

- **The spot check agrees with the tables:** in every kind, the answer was right exactly when the whole work was exact.
- **Speed:** 0.84 ms per token on the CPU.
- **The volume example** `322234*21323*212231`: it wrote 1145340042 (right: 1458238263363442), 388 characters of work
  in 0.3 s. That problem is past its training sizes. The README reads its work: it copied the numbers as
  22234 × 2323 × 2231 before doing any arithmetic.

Quick checks during training (`out/train.log`: 100 examples per kind, whole work exactly right):

| Step | Min | Loss | add | subtract | multiply | divide | mixed | decimals | harness-like |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 3.1 | 1.0340 | 0.20 | 0.10 | 0.01 | 0.06 | 0.00 | 0.04 | 0.00 |
| 2000 | 6.3 | 0.3871 | 0.84 | 0.22 | 0.03 | 0.08 | 0.00 | 0.09 | 0.00 |
| 3000 | 9.4 | 0.2165 | 0.99 | 0.62 | 0.09 | 0.08 | 0.07 | 0.18 | 0.02 |
| 4000 | 12.5 | 0.1282 | 0.99 | 0.90 | 0.12 | 0.09 | 0.12 | 0.23 | 0.00 |
| 5000 | 15.6 | 0.0861 | 1.00 | 0.96 | 0.14 | 0.16 | 0.27 | 0.33 | 0.14 |
| 6000 | 18.9 | 0.0526 | 0.99 | 0.91 | 0.28 | 0.15 | 0.34 | 0.41 | 0.31 |
| 7000 | 22.1 | 0.0278 | 1.00 | 0.98 | 0.66 | 0.18 | 0.68 | 0.65 | 0.63 |
| 8000 | 25.3 | 0.0142 | 1.00 | 0.99 | 0.89 | 0.26 | 0.80 | 0.75 | 0.80 |
| 9000 | 28.6 | 0.0087 | 1.00 | 0.98 | 0.97 | 0.35 | 0.88 | 0.78 | 0.87 |
| 10000 | 31.9 | 0.0061 | 1.00 | 0.98 | 0.94 | 0.36 | 0.88 | 0.79 | 0.85 |

**Takeaways:**

- **Writing the work out made multiplication learnable at its training sizes:** 96.7%. smMATH01 scored 1.7% or less from
  3 digits up. The test sets differ, so this isn't like for like: smMATH01 used whole numbers of exactly n digits, while
  this check uses smMATH001-a's own generated sizes, with multipliers of up to 3 nonzero digits.
- **Division is the weak spot:** 38.0%, and 7 of 15 in the spot check.
- **It doesn't stretch past its training sizes:** 6.7% on 8–10 digit addition, where smMATH01's abacus variant added
  8-digit numbers at 100%. Two suspects: rotary positions, and training numbers of at most 6 digits. **Neither has been
  tested.**

**Live harness check, 2026-09-15** (README): Router, Math Language and Math on smMATH001-a, in a scratch data folder.
The calculator allows answers within 1/100.

| Message | What happened |
|---|---|
| Volume of a 3ft x 4ft x 5ft box, and how many cubic yards is that? | The model did 3×4×5 = 60 and 60 ÷ 27 = 2.22 |
| whats 17.5% of 2,340 | The model did 2340 × 17.5 ÷ 100 = 409.5 |
| $120 sneakers are 30% off, what do they cost now | The model did 120 − 120 × 30 ÷ 100 = 84 |
| split a $212.40 check five ways and add a $3 tip each | The model wrote 39.99 for the split, so the calculator used 42.48. The model then did 42.48 + 3 = 45.48 |
| (12 + 8) / 5 | The model wrote 5, so the calculator used 4 |
| the 322234ft x 21323ft x 212231ft volume | The calculator did it |

**Math Test button** (README): 5/6. It missed 31415926 + 27182818 by writing 48598744. smMATH01-plain and
smMATH01-abacus also score 5/6 but miss different checks. The page's own `scores.json` has no smMATH001-a entry.

## Known failure modes

**Measured** (`out/results.json`):

- **Division:** 38.0% of worked solutions exactly right at training sizes. The README describes how it fails: it picks
  a wrong quotient digit, then writes steps that can't be true, like 5 × 9 = 55 or 20 − 25 = 0.
- **Copying:** the volume problem went wrong before any arithmetic, when it copied the numbers shorter.
- **Past its training sizes it mostly fails:** 0–20.7% on every beyond set, and a wrong answer for the volume example.
- **Mixed, decimal and harness-like expressions:** 82–85%, so about 1 in 6 has a wrong step.

**By design** (from the code):

- **What it can take:** non-negative numbers with at most 2 decimal places, `+ - * /` and brackets.
  - `sqrt`, `**`, negative numbers and numbers like 1.609344 don't fit (a test checks `26.2*1.609344`). The harness
    sends those to the calculator.
- **It works to 2 decimal places, and cuts rather than rounds.** The harness accepts an answer within 1/100 of exact
  and shows the model's own 2-place figure.
- **Bigger than training:** `Solver.fits()` accepts a question of up to 200 characters, but training questions were at
  most 64 characters, with the size limits above. So the harness can hand it problems bigger than anything it trained
  on.
  - The volume example `322234*21323*212231` is one: its first multiplier has 5 nonzero digits, and training allows 3.
- **Negative middle steps** were never in training, and the harness doesn't screen for them before asking. The
  calculator check catches a wrong answer.

## In the harness

Details: [architecture.md](../paratroop_harness/architecture.md#step-math-models-and-whole-expression-math-models).

- **Registration:** provider entry `smMATH001-a` → `../smMATH001-a/out/math.pt`, kind `math`, loader `solve.py:Solver`.
  It shares the Math module's dropdown with the smMATH01 variants. The default is still `smMATH01-abacus`.
- **`Solver.takes = "expression"`** sends it down `_whole_math()`:
  - `plain_arithmetic()` moves divisions last (`17.5% of 2,340` becomes `17.5*2340/100`);
  - `fits()` and `solve()` return `{answer, work, raw}`;
  - the calculator checks the answer to within 1/100;
  - the work lines show under the Math section on the page.
- **The calculator checks it** only while the calculator tool is on and the Math `check` setting is on. With checking
  off, its wrong answers (division especially) go into the reply unchecked.

## Next (from the README)

- **Division:** more practice, and backing off to a smaller quotient digit when a subtraction would go below zero.
- **Length:** try place labels without rotary positions, like smMATH01's abacus variant, or train with longer numbers
  and a bigger place shift.
- **Copying:** fix copying long numbers, which is where the volume problem went wrong.
