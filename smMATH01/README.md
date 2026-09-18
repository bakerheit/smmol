# smMATH01

A tiny GPT that does whole-number arithmetic in its own weights: no calculator, no code, just
next-token prediction.

**The question:** it trains on numbers of 1 to 6 digits. On 7 to 10 digits, which it has never
seen, does accuracy hold, or does it fall off a cliff?

## Three variants

All three are the same size (4.75M parameters), train on the same problems for the same number of
steps, and differ in only two ways:

| Variant | Digit order | How a token knows where it is |
|---|---|---|
| `plain` | the usual order: `48213+9977=58190$` | a learned embedding for each position in the line, like smLLM_01 |
| `reversed` | ones digit first: `31284+7799=09185$` | the same position embeddings |
| `abacus` | ones digit first | a label for each digit's place inside its own number |

- **Why reversed:**
  - You add from the ones digit up, carrying as you go.
  - Writing the answer in that order means each digit only depends on digits the model has already
    seen.
  - In the usual order, it has to work out every carry before writing the first digit.
- **Why abacus:**
  - With position-in-line embeddings, a 9-digit problem puts digits at positions no training line
    ever reached.
  - Place-in-number labels make "digit 3 of the question" and "digit 3 of the answer" match however
    long the numbers are.
  - During training, every label in a line gets shifted by the same random amount (0 to 25). That
    way the labels for places 7 to 10 get trained too, even though no training number is that long.
- **Not new ideas:**
  - Reversing digits comes from "Teaching Arithmetic to Small Transformers" (Lee et al., 2023).
  - Place-in-number embeddings come from "Transformers Can Do Arithmetic with the Right Embeddings"
    (McLeish et al., 2024).
  - This is a small replication on a MacBook, not a new result.

## The problems

- `+`, `-` and `*` on whole numbers, mixed 40/30/30.
- **Training:** each number gets a random length from 1 to 6 digits.
- **Fresh every step:** there's no dataset. Short problems like `3+4` repeat by chance, but a 6-digit
  line almost never does.
- **Subtraction is never negative:** the bigger number goes first, so there's no minus sign to learn.
- **Scoring:** only the answer tokens count toward the loss.
- **Testing:** both numbers have exactly n digits, for every n from 1 to 10, with 300 problems per
  operation and length.
  - Every variant gets the same problems.
  - An answer counts only if every digit and the end mark are exactly right.

## Results

- **Training:** 8,000 steps of 256 problems per variant (2.05M problems), about 12.5 minutes each on the
  M5.
- **Scoring:** 300 problems per operation and length, and only exactly right answers count.
- **Lengths in bold** are longer than any number it trained on.

**Addition**

| Variant | 1 | 2 | 3 | 4 | 5 | 6 | **7** | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 100% | 100% | 97% | 97% | 93% | 92% | 0% | 0% | 0% | 0% |
| reversed | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 0% | 0% |
| abacus | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 89% | 40% |

**Subtraction**

| Variant | 1 | 2 | 3 | 4 | 5 | 6 | **7** | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 100% | 97% | 91% | 86% | 83% | 81% | 0% | 0% | 0% | 0% |
| reversed | 100% | 100% | 98% | 97% | 99% | 97% | 0% | 0% | 0% | 0% |
| abacus | 100% | 100% | 99% | 100% | 99% | 99% | 95% | 80% | 46% | 19% |

**Multiplication**

| Variant | 1 | 2 | 3 | 4 | 5 | 6 | **7** | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 100% | 33% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| reversed | 100% | 35% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| abacus | 100% | 55% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

**What it shows:**
- **Plain and reversed fall off a cliff at exactly 7 digits.** They go from 81–100% at 6 digits to 0% at
  7, for both addition and subtraction. The position embeddings past the longest training line were
  never trained, so a longer problem puts digits where the model has never seen anything.
- **Reversing the digits helps inside the training range, not outside it.**
  - 6-digit addition: 100% reversed vs 92% plain.
  - 6-digit subtraction: 97% vs 81%.
- **Abacus holds, then fades.**
  - Addition stays at 100% through 8 digits, then 89% at 9 and 40% at 10.
  - Subtraction goes 95%, 80%, 46% and 19% from 7 to 10 digits.
  - Labeling each digit by its place inside its number let it carry past lengths it had seen.
- **Multiplication failed for all three, even on training lengths.**
  - They're 100% at 1 digit, 33–55% at 2 digits, and about 0% from 3 digits up.
  - Long multiplication needs a pile of partial products, and 8,000 steps of a 6-layer model didn't
    learn it. The abacus trick doesn't carry over by itself.
- **This matches the papers above** at a much smaller scale: reversing helps, and place-in-number
  embeddings are what buy longer numbers.

**What misses look like:**
- **abacus, 6 digits:** 757796 − 195441 = 562355, and it wrote 558355, a borrow gone wrong.
- **plain, 8 digits:** 87537680 + 57437480 = 144975160, and it wrote 85767688577.
- **reversed, 8 digits:** the same problem, and it wrote 26.

## In the harness

smMATH01 is the model behind the Math module in
[paratroop_harness_02](../paratroop_harness_02/README.md#the-math-module-smmath01).

- **The harness loads `solve.py`:** `Solver(path).answer(a, op, b)` returns the digits the model
  writes, in the usual order, or `None` if it never finishes a number.
- **The harness does the rest:**
  - It hands the model one whole-number `+`, `-` or `*` at a time.
  - It shifts decimals into whole numbers.
  - The calculator checks every answer.
- **Each variant is its own dropdown choice,** so they can be swapped while the harness runs, and the
  Test button scores each one.

## Run it

Tests (problem lines, answer-only scoring, place labels, a tiny model learning 1-digit addition, the
harness solver):

```bash
python3 -m unittest discover -s tests
```

Train all three variants, 8,000 steps of 256 problems each:

```bash
python3 train.py
```

Train just one variant:

```bash
python3 train.py --variant abacus --steps 2000
```

Score every length from 1 to 10 digits:

```bash
python3 evaluate.py
```

Ask all three a problem:

```bash
python3 ask.py 48213+9977
```

## Files

| File | What it is |
|---|---|
| `problems.py` | Problem lines in both digit orders, place labels, batches with answer-only targets |
| `model.py` | MathGPT and its three variants, plus greedy answering |
| `train.py` | Fixed-step training per variant, with a quick 6-digit and 8-digit check as it goes |
| `evaluate.py` | Accuracy for every operation and length, written to `out/results.json` |
| `ask.py` | One problem, every variant |
| `solve.py` | What the harness's Math module loads: one whole-number problem in, the model's digits out |
| `out/` | `plain.pt`, `reversed.pt`, `abacus.pt`, `train.log`, `results.json` |
