# smMATH01: arithmetic in the model's weights

Project: [`smMATH01/`](../../../smMATH01/README.md). Checked against `problems.py`, `model.py`, `train.py`,
`evaluate.py`, `ask.py`, `solve.py`, `tests/test_math.py`, `out/train.log` and `out/results.json` on 2026-09-15.

## Purpose

- **The question:** a tiny GPT does whole-number `+`, `-` and `*` by next-token prediction, with no calculator. It
  trains on numbers of 1 to 6 digits. Does accuracy hold at 7 to 10 digits?
- **The variants:** three, all the same size, differing only in digit order and position information.
- **Not new:** a small replication of Lee et al. 2023 (reversed digits) and McLeish et al. 2024 (place-in-number
  embeddings).

## Input and output

- **Vocabulary:** `0123456789+-*=$`, plus `PAD`. `$` ends the answer.
- **One problem is one line:** `question=answer$`.

| Variant | Digit order | Example | Position information |
|---|---|---|---|
| `plain` | usual | `48213+9977=58190$` | a learned embedding per position in the line |
| `reversed` | ones digit first | `31284+7799=09185$` | the same position embeddings |
| `abacus` | ones digit first | `31284+7799=09185$` | a label per digit for its place inside its own number (`places()`), randomly shifted in training |

- **In:** the question up to and including `=`.
- **Out:** answer tokens up to `$`, decoded greedily (`MathGPT.answer`).

## Architecture and size

`MathGPT` in `model.py`:

- the smLLM_01-style block: pre-LayerNorm causal attention, 4× GELU MLP, tied output head, scaled residual init;
- d 256, 6 layers, 8 heads, no dropout;
- context 64 tokens (the longest test line, 10 digits by 10 digits, is 43 tokens);
- `abacus` swaps position embeddings for 64 place embeddings. In training, every digit label in a line is shifted by
  the same random amount, 0 to 25 (`--offset-max`), so places 7 to 10 get trained too.

**4,745,728 parameters** per variant (`out/results.json`).

## Data

`problems.py`:

- **Fresh every batch:** no dataset. `+`, `-` and `*` are mixed 40/30/30.
- **Lengths:** each number gets a random length from 1 to 6 digits.
- **Subtraction** puts the bigger number first, so answers are never negative.
- **Loss:** only on the answer tokens after `=` (everything else is −100).

## Train and test

```bash
python3 -m unittest discover -s tests
python3 train.py
python3 evaluate.py
python3 ask.py 48213+9977
```

- **`train.py`:** a fixed step count per variant, so a Mac that slows down late can't shortchange the last variant.
  - 8,000 steps of 256 problems; AdamW lr 1e-3, betas 0.9/0.98, weight decay 0.1; warmup 200 steps, cosine to lr/10;
    gradient clip 1.0.
  - Every 500 steps it prints a quick check (100 problems at 6 and 8 digits) and saves `out/<variant>.pt`.
- **`evaluate.py`:** both numbers exactly n digits, for n from 1 to 10, with 300 problems per operation and length.
  Every variant gets the same problems. Only exact answers count, including `$`.
- **Tests (8).** All flags: [running.md](running.md#smmath01).

## Results

**2026-09-15**, `out/results.json` (`when` 15:59).

- **Training:** 2.05M problems per variant: 751 s plain, 748 s reversed, 768 s abacus.
- **Final loss:** plain 0.379, reversed 0.342, abacus 0.341.
- **Bold** lengths are longer than any training number.

**Addition**

| Variant | 1 | 2 | 3 | 4 | 5 | 6 | **7** | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 100% | 100% | 97% | 96.7% | 92.7% | 92% | 0% | 0% | 0% | 0% |
| reversed | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 0% | 0% |
| abacus | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 89% | 40.3% |

**Subtraction**

| Variant | 1 | 2 | 3 | 4 | 5 | 6 | **7** | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 100% | 96.7% | 91.3% | 86.3% | 83% | 80.7% | 0% | 0% | 0% | 0% |
| reversed | 100% | 100% | 97.7% | 97% | 99% | 97% | 0% | 0% | 0% | 0% |
| abacus | 100% | 100% | 98.7% | 100% | 99.3% | 99% | 95% | 80% | 46.3% | 19.3% |

**Multiplication**

| Variant | 1 | 2 | 3 | 4 | 5 | 6 | **7** | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 100% | 32.7% | 1.7% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| reversed | 100% | 35.3% | 1.7% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| abacus | 100% | 54.7% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

- **Harness Test button** (`scores.json`), always checked:
  - `smMATH01-abacus` 5/6 at 15:52, while it was still training (finished 15:59). The failed check was 123 * 45, where
    the model wrote 4175;
  - `smMATH01-plain` 5/6 at 15:56. The failed check was 31415926 + 27182818, where the model wrote
    22643342322222822225492.

## Known failure modes

- **A cliff at 7 digits** for plain and reversed: from 81–100% at 6 digits to 0% at 7. Position embeddings past the
  longest training line were never trained.
- **Abacus fades** past 8 digits: addition 89% at 9 and 40% at 10; subtraction 46% at 9 and 19% at 10.
- **Multiplication isn't learned** by any variant past 1 digit, even inside the training range. Long multiplication
  needs many partial products.
- **Misses** (from `out/results.json`):
  - abacus, 6 digits: 757796 − 195441 = 562355, and it wrote 558355 (a borrow gone wrong);
  - plain, 8 digits: 87537680 + 57437480 = 144975160, and it wrote 85767688577;
  - reversed, same problem: it wrote 26.
- **Leading zeros:** reversed and abacus sometimes write a leading zero on small differences (459232 − 456403 as
  `02829`, 142063 − 141292 as `0771`). `evaluate.py` counts that wrong. The harness reads the digits as a number, so the
  value is right there.

## In the harness

The model behind the Math module's step path. Details:
[architecture.md](../paratroop_harness/architecture.md#step-math-models-and-whole-expression-math-models).

- **Registration:** provider entries `smMATH01-abacus`, `smMATH01-reversed` and `smMATH01-plain`, kind `math`, loader
  `solve.py:Solver`. `harness.json`'s default for the Math module is `smMATH01-abacus`.
- **`Solver.fits(a, op, b)`** is true when:
  - `op` is `+ - *`;
  - both numbers are ≥ 0, and for `-`, a ≥ b;
  - the question plus room for the longest possible answer fits the 64-token context.
- **`Solver.answer(a, op, b)`** returns the digits in the usual order, or `None` if it never writes `$` or writes
  something that isn't digits.
- **The harness does the rest:**
  - hands it one whole-number step at a time;
  - shifts decimals into whole numbers;
  - flips subtractions that would go negative;
  - sends division, powers, functions, negatives and oversize numbers to the calculator;
  - has the calculator check every step (by default).
