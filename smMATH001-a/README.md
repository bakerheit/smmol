# smMATH001-a

A tiny GPT that works out whole expressions by writing out its work, the way you would on paper. It
can be the model behind the Math module in [paratroop_harness_02](../paratroop_harness_02/README.md).
It follows [smMATH01](../smMATH01/README.md), which did one step at a time and couldn't multiply past
2 digits.

```text
2340*17.5/100
  2340 × 17.5:
    2340 × 0.5 = 1170
    2340 × 7 = 16380
    1170 + 16380 = 17550
    2340 × 10 = 23400
    17550 + 23400 = 40950
    = 40950
  40950 ÷ 100:
    100 × 400 = 40000
    40950 − 40000 = 950
    100 × 9 = 900
    950 − 900 = 50
    100 × 0.5 = 50
    50 − 50 = 0
    = 409.5
```

## How it writes its work

- **Every number is fixed-point:** it's multiplied by 100 and written ones digit first, so decimals are
  ordinary digits and carries run the way the model writes. 20364 + 8 is read as `0046302+008` and
  written as `0046302+008=0027302`.
- **Order:** brackets first, then `*` and `/`, then `+` and `-`.
- **Addition and subtraction** are one line each.
- **Multiplication:**
  - one line per nonzero digit of the shorter number (its partial products);
  - a running sum of those lines;
  - a last step that drops the two extra decimal places.
- **Division** is long division, biggest place first, stopping at 2 decimal places (cut, not rounded).
- **`work.py`** writes these solutions for training, and `readable()` turns them back into the lines
  shown above.

## How it's built

- **Model:** 4.75M parameters (d 256, 6 layers, 8 heads).
- **Positions:** two signals tell each token where it is:
  - rotary positions inside attention, so it can find the line before this one;
  - a place-in-number label on every digit, shifted by a random 0–16 during training, so the same
    places line up.
- **Writing:** it uses a key/value cache, at 0.84 ms per token on the CPU.
- **Data:** 292,689 generated worked solutions, with a median length of 109 characters.
  - 1 to 3 operations, and numbers of up to 6 digits.
  - In the first 4,000 examples of the pool:
    - 58% have a decimal somewhere;
    - 36% copy the shapes Math Language writes: volumes, percents, tax, splits, averages;
    - 14% are a single division.

    Each number has a 30% chance of being a decimal, and most examples have several numbers.
  - **Sizes are capped:** multipliers have at most 3 nonzero digits, divisors are at most 9999.99, and
    the work is at most 420 characters.
- **Training:** 10,000 steps of about 8,192 tokens each, 32 minutes on the MacBook Pro M5.

## Results

2026-09-15. A problem counts only if the whole worked solution is exactly right.

**At training sizes** (300 held-out problems per kind):

| Kind | Right |
|---|---|
| Add | 100% |
| Subtract | 99% |
| Multiply | 97% |
| Mixed operations | 84% |
| Shaped like Math Language's problems | 85% |
| With decimals | 82% |
| Divide | 38% |

**Beyond training sizes** (150 problems each):

| Kind | Right |
|---|---|
| Add, 8–10 digit numbers | 7% |
| Subtract, 8–10 digit numbers | 0% |
| Multiply, 4-digit multiplier | 21% |
| Multiply, 5–6 digit multiplier | 0% |
| Divide, 8–9 digit dividend | 0% |
| Divide, 5-digit divisor | 0% |

On a spot check of 15 per kind with real greedy writing, the answer was right exactly when the whole
work was right.

**What it shows:**
- **Writing out the work fixed multiplication** inside its training sizes: 97%, where smMATH01 was
  about 0% from 3 digits up.
- **Division is the weak spot, at 38%.** It picks a wrong quotient digit and then writes steps that
  can't be true, like 5 × 9 = 55 or 20 − 25 = 0.
- **It doesn't carry past its training sizes,** the opposite of smMATH01's abacus variant, which
  added 8-digit numbers at 100%. The likely suspects are the rotary positions and the 6-digit training
  cap. Neither has been tested yet.
- **On the volume problem it shrank the numbers:**
  - it copied 322234 × 21323 × 212231 as 22234 × 2323 × 2231;
  - it worked that out and wrote 1145340042 in 0.3 s;
  - the right answer is 1458238263363442.

## In the harness

- **Picking it:** choose `smMATH001-a` in the Math dropdown. It takes whole expressions
  (`takes = "expression"`).
- **Divisions go last:** the harness moves them to the end where that gives the same value, so
  `(17.5/100)*2340` arrives as `17.5*2340/100`.
- **Checking:** the calculator checks the final answer to within a cent, and the work shows up under
  each problem.

**Live check, 2026-09-15.** Router, Math Language and Math were switched on, in a scratch data folder:

| Message | What happened |
|---|---|
| Volume of a 3ft x 4ft x 5ft box, and how many cubic yards is that? | The model did 3×4×5 = 60 and 60 ÷ 27 = 2.22 |
| whats 17.5% of 2,340 | The model did 2340 × 17.5 ÷ 100 = 409.5 |
| $120 sneakers are 30% off, what do they cost now | The model did 120 − 120 × 30 ÷ 100 = 84 |
| split a $212.40 check five ways and add a $3 tip each | The model wrote 39.99 for the split, so the calculator used 42.48. The model then did 42.48 + 3 = 45.48 |
| (12 + 8) / 5 | The model wrote 5, so the calculator used 4 |
| the 322234ft x 21323ft x 212231ft volume | The calculator did it |

**On the Math Test button** it scores 5/6: 31415926 + 27182818 came out as 48598744. smMATH01-plain and
abacus also score 5/6, but they miss different checks.

## Run it

Tests (worked solutions are right and readable, cached writing matches a full forward pass, the
harness solver):

```bash
python3 -m unittest discover -s tests
```

Print a few generated worked solutions:

```bash
python3 work.py
```

Train. This writes `out/math.pt`, `out/results.json` and the tables above:

```bash
python3 train.py
```

Solve one expression:

```bash
python3 solve.py "2340*17.5/100"
```

## Files

| File | What it is |
|---|---|
| `work.py` | Worked solutions: parsing, fixed-point numbers, multiplication and division steps, readable lines, the generator |
| `model.py` | MathGPT with rotary positions, place labels and cached writing |
| `train.py` | Token-budget training, scoring at and beyond training sizes, the volume example |
| `solve.py` | `Solver(path).solve(expression)`, which the harness loads |
| `out/` | `math.pt`, `results.json`, `train.log` |

## Next

- **Division:** only 14% of the training examples are pure division. Give it more practice, and teach
  it to back off to a smaller digit when a subtraction would go below zero.
- **Length:** try place labels without rotary positions, like smMATH01's abacus variant, or train with
  longer numbers and a bigger place shift.
- **Copying:** the volume problem went wrong before any arithmetic, when it copied the numbers.
