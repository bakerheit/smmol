# smMATH_LANGUAGE_001: the math reader

Project: [`smMATH_LANGUAGE_001/`](../../../smMATH_LANGUAGE_001/README.md). Checked against `data.py`, `model.py`,
`score.py`, `train.py`, `read.py`, `baseline_llm.py`, `test.json`, `tests/test_reader.py`, `out/results.json`,
`out/train.log` and `out/llm_baseline.json` on 2026-09-15. Training finished at 17:37.

## Purpose

Read a chat message and write out the math problems in it, without solving them. The harness's Math module or the
calculator then works them out. It's the model behind the Math Language module.

## Input and output

- **In:** the message as UTF-8 bytes, up to 160.
- **Out:** a compact target of up to 100 bytes, then `<END>`:

  ```text
  322234*21323*212231|cubic feet|volume;p1/27|cubic yards|volume
  ```

  - Problems are separated by `;`, and each is `expression|unit|about`.
  - An empty target means there's nothing to work out.
  - Expressions use numbers, `+ - * / **`, brackets and `sqrt()`. `p1`, `p2`... are earlier answers in the same message.
  - Division goes last where it can: 8% tax on 80 is `80+80*8/100`, not `80*(1+8/100)`, so a math model working to 2
    decimal places loses nothing.
- **The harness gets JSON:** `data.parse_target()`, used by `read.py`, turns the target into
  `[{"id": "p1", "expression", "unit", "about"}, ...]`.

| Message | Target |
|---|---|
| what is the volume of 322234ft x 21323ft x 212231ft? | `322234*21323*212231\|cubic feet\|volume` |
| Volume of a 3ft x 4ft x 5ft box, and how many cubic yards is that? | `3*4*5\|cubic feet\|volume;p1/27\|cubic yards\|volume` |
| we have 2 kids and 3 dogs | (empty) |

## Architecture and size

`ReaderGPT` in `model.py`:

- a byte-level GPT with a vocabulary of 259 (256 bytes plus `SEP`, `END`, `PAD`) and a context of 262 (160 + 100 + 2);
- learned position embeddings;
- d 256, 6 layers, 8 heads, no dropout;
- the smLLM_01-style block, tied output head and scaled residual init.
- **Layout:** `message bytes <SEP> target bytes <END>`. The loss only counts tokens after `<SEP>`.
- **Decoding:** greedy (`read()`), stopping at the first non-byte token.

**4,858,624 parameters** (`out/results.json`).

## Data

`data.py`:

- **21 kinds of message**, drawn by weight:
  - bare arithmetic (1.0), volume (0.7), area (0.5);
  - percent (0.7), tip (0.5), tax (0.5), discount (0.4), split (0.6);
  - arithmetic in words (0.8), add up (0.5), average (0.4), powers and roots (0.4);
  - per item (0.5), savings (0.4), travel (0.4), fuel (0.3), conversions (0.5), dozens and halves (0.4);
  - several problems in one message (0.8), follow-ups with `p1` (0.6);
  - **no math** (1.6): phone numbers, times, scores, years, codes, weather, "thanks!".
- **Numbers vary:** 1 to 8 digits, thousands commas 35% of the time, decimals, money in `$`/dollars/bucks, spelled-out
  small numbers.
- **Roughened:** greetings 12% of the time, dropped end punctuation 30%, a letter typo 6% (never in digits), odd
  casing.
- **Every target is checked** with `score.values()`. A target that doesn't work out is regenerated.
- **Sizes:** 300,000 training messages (seed 0) and 2,000 validation messages (seed 1000). Messages that match a test
  message after normalizing are dropped; there were none this run.
- **Test set:** `test.json`, 40 hand-written messages, phrased differently from the generator on purpose.
  - 31 have math and 9 don't.
  - 3 have more than one problem.

## Train and test

```bash
python3 -m unittest discover -s tests
python3 train.py
python3 read.py "split a $212.40 check five ways"
python3 baseline_llm.py
```

- **`train.py`:** 6,000 steps of 64 random training messages; AdamW lr 6e-4, betas 0.9/0.98, weight decay 0.1; warmup
  300 steps, cosine to lr/10; gradient clip 1.0.
  - **Every 1,000 steps:** it scores the hand-written set, and exact targets on 1,000 generated validation messages,
    then saves `out/reader.pt`.
- **Scoring** (`score.py`) compares the *values* expressions work out to (relative tolerance 1e-6), so `15/100*80` is as
  good as `80*15/100`:

  | Score | Meaning | Out of |
  |---|---|---|
  | every | same number of problems, and every value right | all 40 |
  | final | the last value right | all 40 |
  | found | wrote at least one problem | the 31 with math |
  | quiet | wrote nothing | the 9 without math |

  A broken `p` reference counts as wrong.
- **Final baselines:**
  - **the harness regex stopgap:** `find_expression()` + `calc()` from paratroop_harness_02, with no model;
  - **Ministral 8B:** from `out/llm_baseline.json`, written by `baseline_llm.py` with the same JSON shape as a schema.
- **Tests (6).** All flags: [running.md](running.md#smmath_language_001).

## Results

**2026-09-15 17:37**: 1,577 s (26.3 minutes) on the M5 (`out/results.json`).

| Reader | Every problem right | Final answer right | Found the math | Quiet when there's none | Time per message |
|---|---|---|---|---|---|
| **smMATH_LANGUAGE_001** | **77.5%** | **80%** | 100% | 77.8% | 28.9 ms median on the CPU |
| Ministral 8B, prompted (17:13) | 82.5% | 95% | 100% | 88.9% | 3.13 s median on the RX 580 |
| Harness regex, no model | 30% | 32.5% | 29% | 66.7% | not measured |

- **Generated held-out messages:** 99.15% exactly right.

Progress during training (`out/train.log`):

| Step | Minutes | Loss | Generated exact | Every | Final | Found | Quiet |
|---|---|---|---|---|---|---|---|
| 1000 | 3.8 | 0.8222 | 0.62 | 0.45 | 0.47 | 1.00 | 0.44 |
| 2000 | 8.4 | 0.0783 | 0.86 | 0.62 | 0.65 | 0.97 | 0.89 |
| 3000 | 12.2 | 0.0337 | 0.91 | 0.70 | 0.72 | 1.00 | 0.56 |
| 4000 | 16.2 | 0.0160 | 0.96 | 0.65 | 0.65 | 1.00 | 0.56 |
| 5000 | 20.6 | 0.0067 | 0.98 | 0.78 | 0.80 | 1.00 | 0.56 |
| 6000 | 26.3 | 0.0030 | 0.99 | 0.78 | 0.80 | 1.00 | 0.78 |

- **Against Ministral:**
  - close on every problem (77.5% vs 82.5%) at about 100 times the speed;
  - well behind on final answers (80% vs 95%) and on staying quiet (77.8% vs 88.9%).
- **Where Ministral missed:** 5 of its 7 misses split one problem into two steps with the right final answer. It also
  used 1.60934 for miles to kilometers and read "21-17" as math.

**Live check, 2026-09-15** (README): Router, Math Language and Math (on smMATH01-plain), scratch data folder.

- **Speed:** reading took 0.04–0.14 s per message.
- **Replies:**
  - the volume message → `322234*21323*212231 = 1458238263363442 cubic feet`;
  - the box → `60 cubic feet`, then `p1/27 = 2.222222222 cubic yards`;
  - "split a $212.40 check five ways and add a $3 tip each" → `42.48`, then `45.48` dollars;
  - "we have 2 kids and 3 dogs" → no problems.

## Known failure modes

All 9 hand-written mistakes (`out/results.json`):

| Message | Wanted | Wrote | Kind |
|---|---|---|---|
| 90000-1 | `90000-1` | `9000-1` | long number lost a digit |
| what's 1000000 * 3.5 | `1000000*3.5` | `100000*3.5` | long number lost a digit |
| Can you add up 129, 88 and 406 for me | `129+88+406` | `129+88`, `406-406` | list worded a new way |
| what's the mean score if I got 88, 92 and 79? | `(88+92+79)/3` | `88/892`, `79+79` | garbled |
| at 65 mph how long will a 340 mile drive take | `340/65` | `65*340` | rate turned around |
| my garden is 5m by 8m, how much soil to fill it 2m deep | `5*8*2` | `5*8`, `p1*2` | extra step; final right |
| If rent is $1,850 and we're 3 roommates… | `1850/3` | `1850/3`, `p1/3` | extra step |
| call me at 555-1234 | nothing | `555-1234` | false alarm |
| what's the weather in Denver tomorrow | nothing | `10*10` | false alarm |

- **It overfits the generator:** 99% on generated messages, 78% on hand-written ones.
- **The last checkpoint isn't always the best:** the hand-written score dipped to 0.65 at step 4000. `train.py` saves
  the last checkpoint, not the best.
- **It picks up half-trained checkpoints:** `reader.pt` is saved every 1,000 steps, so the harness used the step 1000
  checkpoint (45% every problem right) while training ran.

## In the harness

Details: [architecture.md](../paratroop_harness/architecture.md#math-language-to-answers).

- **Registration:** provider entry `smMATH_LANGUAGE_001` → `../smMATH_LANGUAGE_001/out/reader.pt`, kind
  `math_language`, loader `read.py:Reader`. The Math Language card, lime.
- **When it runs:** after Recall and before the planning loop, when the module is on, trained, and the message has
  digits or number words. The `only_with_numbers` setting drops that last condition.
- **What happens to the output:**
  - it's checked against `contracts.json` → `math_language` (up to 5 problems);
  - each problem is solved in order by Math, or the calculator when Math is off, with `p1`... filled in;
  - the answers are shown to the Planner and Language;
  - with Language off, the reply lists `expression = answer unit`.
- **A reader failure** is noted on the trace and the turn continues.
- **Checks:** `checks.json` → `math_language` has 6 cases, graded by value.

## Next (from the README)

- **Place labels on digits** like smMATH01's abacus variant, so long runs of zeros keep count.
- **Phrasings the generator never makes:** "for me" endings, "mean score", rates written backwards, more messages with
  numbers but no math.
- **Keep the best checkpoint,** chosen on a small hand-written dev set kept apart from the test set.
