# smMATH_LANGUAGE_001

A tiny model that reads a chat message and writes out the math problems in it. The harness's Math
module, or the calculator, then works them out. It's the model behind the Math Language module in
[paratroop_harness_02](../../harnesses/paratroop_harness_02/README.md).

| Message | What it writes |
|---|---|
| what is the volume of 322234ft x 21323ft x 212231ft? | `322234*21323*212231\|cubic feet\|volume` |
| Volume of a 3ft x 4ft x 5ft box, and how many cubic yards is that? | `3*4*5\|cubic feet\|volume;p1/27\|cubic yards\|volume` |
| we have 2 kids and 3 dogs | nothing |

## What it writes

- **Format:**
  - Problems are separated by `;`.
  - Each problem is `expression|unit|about`.
  - An empty answer means there's nothing to work out.
- **Expressions:** numbers, `+ - * / ** ( )` and `sqrt()`. `p1`, `p2` … stand for earlier answers in
  the same message.
- **The harness gets JSON:** `read.py` turns the line into a list,
  `[{"id": "p1", "expression", "unit", "about"}, …]`. Units and labels stay with the harness, so the
  math model only ever sees the expression.
- **Division goes last where it can:** a price with 8% tax is written `80+80*8/100`, not
  `80*(1+8/100)`, so a math model working to 2 decimal places loses nothing along the way.

## How it's built

- **Model:** a 4.86M-parameter byte-level GPT (d 256, 6 layers, 8 heads). The message is up to 160
  bytes and its answer up to 100, and only the answer is scored.
- **Data:** 300,000 generated messages (`data.py`) across 21 kinds:
  - bare arithmetic, volumes, areas;
  - percents, tips, tax, discounts, splitting a bill;
  - arithmetic in words, adding up lists, averages, powers and roots;
  - totals per item, savings, travel time, fuel, unit conversions, dozens and halves;
  - several problems in one message, and follow-ups that use an earlier answer;
  - messages with numbers but nothing to work out, like phone numbers, times and scores.
- **Noise:** messages are roughened with greetings, dropped punctuation, casing, and typos in letters
  only, never in digits.
- **Every target is checked:** a generated answer that doesn't work out isn't used.
- **Training:** 6,000 steps of 64 messages, 26 minutes on the MacBook Pro M5.

## Results

It's scored on 40 hand-written messages in `test.json`: 31 with math and 9 without.
- **Phrasing:** they're worded differently from the generator on purpose.
- **Scoring by value:** an answer counts when its expressions work out to the right numbers, so
  `15/100*80` is as good as `80*15/100`.

| Reader | Every problem right | Final answer right | Found the math | Quiet when there's none | Time per message |
|---|---|---|---|---|---|
| **smMATH_LANGUAGE_001** | **78%** | **80%** | 100% | 78% | 29 ms on the CPU |
| Ministral 8B, prompted (`baseline_llm.py`) | 82% | 95% | 100% | 89% | 3.1 s on the RX 580 |
| The harness's code with no model | 30% | 32% | 29% | 67% | – |

- **It's close to Ministral on getting every problem right,** at about 100 times the speed. It's
  well behind on final answers and on staying quiet when there's no math.
- **It overfits the generator:** 99% of held-out generated messages come out exactly right, against
  78% of the hand-written ones. The router showed the same pattern.
- **The mistakes:**
  - **Long numbers lose a digit:** `90000-1` came out as `9000-1`, and `1000000 * 3.5` as `100000*3.5`.
  - **Lists and averages worded a new way:** "add up 129, 88 and 406 for me" became `129+88` and
    `406-406`, and "the mean score if I got 88, 92 and 79" came out garbled.
  - **A rate turned around:** 340 miles at 65 mph became `65*340`.
  - **An extra step:** the garden became `5*8` then `p1*2`, with the final answer still right. The
    rent split got an extra `p1/3`.
  - **False alarms:** "call me at 555-1234" became `555-1234`, and "what's the weather in Denver
    tomorrow" came out as `10*10`.

## In the harness

- **When it runs:** after the router, on any message with numbers in it. A setting can make it read
  every message.
- **What happens to its problems:** they go into working memory, and the Math module, or the
  calculator when Math is off, works out each one. The answer lands next to its problem, and the
  Planner and Language both see it.
- **With Language off,** the reply lists each problem with its answer and unit.

**Live check, 2026-09-15.** Router, Math Language and Math (on `smMATH01-plain`) were switched on, in a
scratch data folder:

| Message | Reply |
|---|---|
| what is the volume of 322234ft x 21323ft x 212231ft? | 322234\*21323\*212231 = 1458238263363442 cubic feet |
| Volume of a 3ft x 4ft x 5ft box, and how many cubic yards is that? | 3\*4\*5 = 60 cubic feet, then p1/27 = 2.222222222 cubic yards |
| split a $212.40 check five ways and add a $3 tip each | 212.40/5 = 42.48 dollars, then p1+3 = 45.48 dollars |
| we have 2 kids and 3 dogs | no problems found |

- **Speed:** reading took 0.04–0.14 s per message.
- **Who did the math:** smMATH01 did the small multiplications and the addition. It can't multiply
  numbers as big as the volume's or divide, so the calculator did those steps.

## Run it

Tests (every generated target works out, test messages never reach training, scoring by value, the
harness reader):

```bash
python3 -m unittest discover -s tests
```

Train. This writes `out/reader.pt`, `out/results.json` and the table above:

```bash
python3 train.py
```

Read one message:

```bash
python3 read.py "what is the volume of 322234ft x 21323ft x 212231ft?"
```

Rebuild the Ministral baseline (uses the PC's GPU):

```bash
python3 baseline_llm.py
```

## Files

| File | What it is |
|---|---|
| `data.py` | The message generator and `parse_target` |
| `score.py` | Working out problem lists safely, and scoring by value |
| `model.py` | ReaderGPT and byte encoding |
| `train.py` | Training, scoring, the baselines, the results table |
| `read.py` | `Reader(path).read(text)`, which the harness loads |
| `baseline_llm.py` | Ministral 8B on the same messages |
| `test.json` | 40 hand-written messages with the expressions they should give |
| `out/` | `reader.pt`, `results.json`, `train.log`, `llm_baseline.json` |

## Next

- **Long numbers:** give digits place labels, as smMATH01's abacus variant does, so a run of zeros
  doesn't lose count.
- **Phrasings the generator never makes:** "for me" endings, "mean score", rates written backwards, and
  more messages with numbers but no math.
- **Keep the best checkpoint, not the last one,** chosen on a small hand-written dev set kept apart
  from the test set.
