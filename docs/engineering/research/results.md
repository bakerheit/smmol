# Results log

Every experiment's key numbers, oldest first, with the baselines they were measured against and the honest takeaway.

- **Sources:** every number comes from a file in the project folder: `out/results.json`, `out/train.log`,
  `out/eval.md`, `out/llm_baseline.json`, `eval-results.json`, `scores.json`, or a project README.
- **Times** are the `when` field or file time when there is one. A README-only result gets its date without a time.
- **Model details:** see [../smModels/](../smModels/).

## Baselines at a glance

| Job | Small model trained here | Keyword rules or regex (no model) | Ministral 8B, prompted |
|---|---|---|---|
| Routing: intent, tool and ask-first all right (93 hand-written messages) | smROUTER_01 v3: **71%**, 2.04 ms on the CPU | keyword rules (`rules.py`): 63% | 58%, 1.99 s median |
| Routing bare arithmetic (17 messages) | v3: **94%** (v2: 82%, v1: 59%) | keyword rules: 24% | not run |
| Reading math out of a message: every problem right (40 messages) | smMATH_LANGUAGE_001: **78%**, 28.9 ms | harness regex: 30% | **82%**, 3.13 s median |
| Same, final answer right | 80% | 32.5% | **95%** |
| Same, quiet when there's no math (9 messages) | 78% | 67% | **89%** |
| A held-out word's ideas: precision@k | smCLM_01 idea reader: 0.82 | common ideas, no reading: 0.14 | not run |
| Gadget world: all 8 right after 20 experiments | smALLM_01: 83% random, 82% curious | lazy guess: 61% | not run |
| Plain request to the right shell command, exact (53 hand-written requests) | smTOOLS_COMPUTER_CLI_01: **47%**, 47 ms on the CPU | catalog lookup: 40% | 15%, 1.94 s median |

> **On the CLI row.** `smTOOLS_COMPUTER_CLI_01/train.py` chooses its checkpoint and early-stops on
> `standing(hand)`, where `hand` is scored on the same 53 hand-written requests reported here. The published
> 46.8% is **not** affected — that run predates the selection code, as its `results.json` shows by having no
> `best_step` field, and it is the last checkpoint rather than a chosen one. But the retrain already on the
> project's list would ship a number selected on its own test set. Its docstring says why the shortcut exists:
> *"Only the hand-written set can tell checkpoints apart. The generated one saturates at 99% by step 3000."*
> That is the same saturation the router has, and fixing it is what
> [selection-metric.md](../plans/selection-metric.md) is for. Do not retrain this model until it is fixed.
| Same, right program | 64% | catalog lookup: 60% | **79%** |
| Same, quiet when it isn't a terminal job (6 requests) | **100%** | catalog lookup: 33% | 50% |

---

## 2026-09-14

### smLLM_01: first run

Source: `smLLM_01/out/train.log`, `out/log.csv`.

- **Setup:** 10.8M parameters, bytes of Tiny Shakespeare, 20 minutes on the M5's GPU.
- **How far it got:** 1,915 steps (batch 64 × 256 bytes), 31.4M tokens, about 26,100–27,000 tokens/s.

| Step | Minutes | Train loss | Val loss (bits/char) |
|---|---|---|---|
| 0 | 0 | 5.656 | 5.660 (8.17) |
| 250 | 2.6 | 1.983 | 2.101 (3.03) |
| 1000 | 10.1 | 1.224 | 1.495 (2.16) |
| **1500** | 15.2 | 1.074 | **1.471 (2.12), best, saved** |
| 1915 | 20.0 | 1.011 | 1.479 (2.13) |

- **Baseline:** the README says nanoGPT's published number for the same layout is about 1.47.
- **Takeaway:**
  - it matches the known result on a laptop;
  - after step 1500 train loss kept falling while val stopped improving, so it began memorizing the 1 MB text;
  - samples have the right format and real words, but no meaning.

### smALLM_01: first run

Source: `smALLM_01/out/train.log`, `out/eval.md`, README.

- **Training:** 4.8M parameters, 20 minutes on the M5, 3,864 steps, 247.3k brand-new gadget worlds. The best checkpoint
  was step 3750.
- **Loss:** 0.583 at step 250, down to 0.134 at the end, and still falling.
- **Final test:** 256 unseen worlds, every gadget probed, so each cell is 2,048 questions.

| Experiments | Random: all 8 right | Random: effects caught | Curious: all 8 right | Curious: effects caught | Lazy guess |
|---|---|---|---|---|---|
| 0 | 61% | 0% | 61% | 0% | 61% |
| 1 | 44% | 26% | 45% | 25% | 61% |
| 4 | 59% | 44% | 55% | 55% | 61% |
| 8 | 70% | 58% | 74% | 72% | 61% |
| 20 | 83% | 80% | 82% | 85% | 61% |

- **`check_curious.py`** (256 worlds, 20 flips):
  - curious flips changed 2.01 other gadgets on average, against 0.83 for random;
  - curious picked a biggest-effect flip 44% of the time, against 16% for random.
- **Takeaway:**
  - it does learn from experiments, and beats the lazy guess from 8 experiments on;
  - after 1 to 4 experiments it's worse than lazy (44–59%);
  - "curious" mostly means "biggest lever". It catches more effects at 4 to 12 experiments, but "all 8 right" gains
    only 0 to 4 points, and nothing by 16 to 20.

---

## 2026-09-15

### paratroop_harness_01: smoke checks

Source: `paratroop_harness_01/README.md`. Everything on `pc/ministral-8b`.

| Function | Score | Average |
|---|---|---|
| route | 3/3 | 3.0 s |
| attention | 4/4 | 0.3 s |
| context | 2/2 | 3.0 s |
| prediction | 1/1 | 4.7 s |
| planning | 1/1 | 24.8 s |
| respond | 2/2 | 1.1 s |

**Takeaway:** these check shape and keywords, not quality. lookup and draw weren't run.

### paratroop_harness_02: first live run

Source: harness README.

- **The message:** "What's 17.5% of 2,340?", every module on Ministral 8B on the RX 580.
- **82 s:** 10 module calls plus the calculator, and every contract fit on the first try.
- **Slowest calls:** Planner 20.9 s and Predictor 15.0 s.
- **The reply:** 409.5.

**Takeaway:** the cycle works end to end. It's slow because every module is a separate 8B call.

### 12:59: v01 vs v02 (`eval.py`)

Source: `paratroop_harness_02/eval-results.json`. Everything on Ministral 8B.

| Scenario | v01 | v02 |
|---|---|---|
| meeting (unclear message) | pass, 58.8 s | pass, 134.7 s |
| percent (17.5% of 2,340) | pass, 8.7 s | pass, 46.3 s |
| memory (remember, then recall) | fail: stopped, reply missing "lee", 0.7 s | pass, 294.4 s |
| files (save, then read back) | fail: stopped, reply missing eggs, milk, bread, 0.6 s | pass, 288.0 s |
| small_talk ("lol ok") | pass, 0.3 s | pass, 8.6 s |
| web (Chicago weather) | fail: didn't use the web, 68.1 s | fail: didn't use the web, 88.7 s |

**Takeaway, from the harness README:**

- v02's files "pass" is false: it never created the file, and memory supplied the list;
- the Critic over-asks;
- it needs today's date;
- create vs update should be checked in code;
- the grader should check what happened, not keywords;
- on the three cases both passed, v02 was slower than v01: 2.3× on meeting, 5.3× on percent, 29× on small talk.

### 13:27: smCLM_01

Source: `smCLM_01/out/results.json`, `out/train.log`.

- **Setup:** 51 ideas, 286 nouns (23 held out), 79 templates, 80,000 sentences, 8 epochs, 3 seeds, CPU. 1,133 s in
  total.

| Method | precision@k | Average precision |
|---|---|---|
| Idea reader | 0.818 ± 0.010 | 0.932 ± 0.006 |
| Word reader: ideas of words that fit the same spots | 0.821 ± 0.015 | 0.933 ± 0.006 |
| Word reader: ideas of nearest word vectors | 0.127 ± 0.017 | 0.189 ± 0.011 |
| Baseline: most common ideas, no reading | 0.137 | 0.209 |

- **Shown vs implied ideas** (README): ideas shown directly by some sentence were found 74 of 86 times (86%); ideas
  only implied were found 5 of 9 times (56%).
- **Takeaway:**
  - ideas can be learned from usage;
  - thinking in ideas did **not** beat thinking in words on accuracy;
  - the word model's own vectors were useless for this;
  - the world is hand-made, so real language is untested.

### 14:22: smROUTER_01 Ministral baseline

Source: `smROUTER_01/out/llm_baseline.json` (no `when` field; the file was saved at 14:22).

- **Scores:** intent 79.6%, tool 80.6%, ask first 94.6%, for me 97.8%, **all three right 58.1%**.
- **Speed:** median 1.99 s per message.

### 14:38: smROUTER_01 v1

Source: `smROUTER_01/out/v1/results.json`, `out/v1/train.log`, `out/v1/arithmetic_before.txt`.

- **Training:** 1,869,327 parameters, 5 epochs on 58,471 generated messages, 949 s.

| Router (93 hand-written messages) | Intent | Tool | Ask first | For me | All three right |
|---|---|---|---|---|---|
| smROUTER_01 v1 | 87.1% | 81.7% | 92.5% | 97.8% | **73.1%** |
| smROUTER_01 v1, web tools off | 87.1% | 91.4% | 92.5% | 97.8% | 79.6% |
| keyword rules | 90.3% | 81.7% | 88.2% | 100% | 63.4% |
| Ministral 8B | 79.6% | 80.6% | 94.6% | 97.8% | 58.1% |

- **By epoch:** "all three right" on hand-written messages went 74%, 71%, 67%, 72%, 73%. Generated held-out messages
  reached 98%.
- **Bare arithmetic, tested afterwards:** 10 of 17. It called "2+2", "2+3" and "12*7" small talk.
- **Takeaway:**
  - beats both baselines and is about 1,000 times faster than the LLM;
  - it overfits the generator: hand-written scores wobble while generated ones climb;
  - the web-off row isn't a better model, because "none" is right for more messages when web tools are off.

### 14:45: harness Test button, Router

Source: `paratroop_harness_02/scores.json`. smROUTER_01 scored **4/5** on the Router checks: one check failed with
"tool was none". This was before v2 finished at 16:17, so it's v1's score.

### 15:59: smMATH01, all three variants

Source: `smMATH01/out/results.json`, `out/train.log`.

- **Training:** 4,745,728 parameters each; 8,000 steps of 256 problems (2.05M problems), numbers of 1 to 6 digits.
  751 s plain, 748 s reversed, 768 s abacus.
- **Test:** 300 problems per operation and length, exact answers only. **Bold** lengths are longer than any training
  number.

| Variant | + at 6 | + at **7** | + at **8** | + at **9** | + at **10** | − at 6 | − at **8** | − at **10** | × at 2 | × at 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 92% | 0% | 0% | 0% | 0% | 80.7% | 0% | 0% | 32.7% | 1.7% |
| reversed | 100% | 0% | 0% | 0% | 0% | 97% | 0% | 0% | 35.3% | 1.7% |
| abacus | 100% | 100% | 100% | 89% | 40.3% | 99% | 80% | 19.3% | 54.7% | 1% |

- **Multiplication** is 0% from 4 digits up for every variant.
- **Takeaway:**
  - plain and reversed fall off a cliff at exactly 7 digits, where position embeddings were never trained;
  - reversing helps inside the training range;
  - place-in-number labels (abacus) carry past it, then fade;
  - long multiplication wasn't learned by any variant;
  - this is a small replication of Lee et al. 2023 and McLeish et al. 2024, not a new result.

### 15:52 and 15:56: harness Test button, Math

Source: `scores.json`. The Math checks always use the calculator as a checker, so the score is the model's own.

| Model | Score | Failed check | Note |
|---|---|---|---|
| smMATH01-abacus | 5/6 at 15:52 | 123 * 45: the model wrote 4175 | `abacus.pt` was still training until 15:59, so this was a partly trained checkpoint |
| smMATH01-plain | 5/6 at 15:56 | 31415926 + 27182818: the model wrote 22643342322222822225492 | 8 digits, past its training |

### Math module live check (smMATH01-plain)

Source: harness README.

| Expression | Result | Done by |
|---|---|---|
| 48213 + 9977 | 58190 | model, 0.07 s |
| 12.50 + 7.25 | 19.75 | model, as 1250 + 725 |
| (250 - 48) + 1990 | 2192 | model, two steps |
| 123 * 45 | 5535 | model |
| 1850 / 3 | 616.6666667 | calculator: no division in the model |
| 31415926 + 27182818 | 58598744 | calculator: the model wrote 22643342322222822225492 |

With only Router, Math and Language on, "hey what's 48213 + 9977?" went Router (0.01 s), Math (0.07 s), then Language,
and came back as 58,190.

### 16:17: smROUTER_01 v2 (bare arithmetic added)

Source: `smROUTER_01/out/results.json`, `out/train.log`.

- **Training:** 1,869,327 parameters, 5 epochs on 58,420 generated messages (1,580 dropped for matching test messages),
  860 s.

| Router (93 hand-written messages) | Intent | Tool | Ask first | For me | All three right |
|---|---|---|---|---|---|
| **smROUTER_01 v2** | 84.9% | 82.8% | 90.3% | 98.9% | **71.0%** |
| v2, web tools off | 84.9% | 88.2% | 90.3% | 98.9% | 73.1% |
| v1 (14:38) | 87.1% | 81.7% | 92.5% | 97.8% | 73.1% |
| keyword rules | 90.3% | 81.7% | 88.2% | 100% | 63.4% |
| Ministral 8B | 79.6% | 80.6% | 94.6% | 97.8% | 58.1% |

- **Bare arithmetic (17 messages), all three right:** v2 82.4% (14/17), v1 10/17, keyword rules 23.5% (4/17).
- **v2 still misses:**
  - "2+2" gets the calculator but is labeled small talk;
  - "we have 2 kids and 3 dogs" is treated as math;
  - "my flight is at 7:45 tomorrow" is treated as a task.
- **By epoch:** hand-written "all three right" went 63%, 69%, 70%, 73%, 71%. Generated held-out messages: 98.5%.
- **Speed:** 2.03 ms per message on the CPU.
- **Takeaway:**
  - the arithmetic fix worked (10 to 14 of 17);
  - the main test dropped 2 messages (73% to 71%). With one run of each, that can't be told apart from noise;
  - it still learned templates better than real phrasing;
  - the harness no longer depends on the router for bare arithmetic, because a code rule handles it.

### Harness live checks: make steps and chats

Source: harness README. Every module on Ministral 8B, in a scratch data folder.

- **Make step:** the JavaScript calculator request got code on the first turn, in 82 s.
  - Still not fixed: the Critic calls "which operators" high severity.
  - "Save a shopping list" now offers a make step first. The file save came third twice and wasn't offered once, in 3
    runs.
- **Chat:** "I have a meeting this Friday…" took 101 s. "I'm the investor. A local bakery is pitching to me." took
  115 s. The context carried over, but the Critic flagged three "high" issues and it asked another question.

**Takeaway:** make fixed never getting to write the thing. Over-asking is the next problem, and file saving regressed.

### 17:13: smMATH_LANGUAGE_001 Ministral baseline

Source: `smMATH_LANGUAGE_001/out/llm_baseline.json`, `out/llm_baseline.log`. 40 hand-written messages: 31 with math, 9
without.

- **Scores:** every problem right 82.5%, final answer right 95%, found the math 100%, quiet when there's none 88.9%.
- **Speed:** median 3.13 s per message.
- **Its 7 misses:**
  - 5 split one problem into two steps with the right final answer: the 3 dozen eggs, the sneakers, the Celsius
    conversion, half of three dozen, and the concert tickets;
  - "26.2 miles in kilometers" used 1.60934 instead of 1.609344, just outside the grader's tolerance;
  - "final was 21-17, what a game" was read as `21 - 17`.

**Takeaway:** strong, but slow. Part of its "every problem" gap is the strict grader (a split step counts as wrong).

### 17:37: smMATH_LANGUAGE_001

Source: `smMATH_LANGUAGE_001/out/results.json`, `out/train.log`.

- **Training:** 4,858,624 parameters; 6,000 steps of 64 generated messages (300,000 made, none dropped); 1,577 s on the
  M5.

Progress on the hand-written messages while training, from `train.log`:

| Step | Min | Loss | Generated: exact | Every | Final | Found | Quiet |
|---|---|---|---|---|---|---|---|
| 1000 | 3.8 | 0.8222 | 0.62 | 0.45 | 0.47 | 1.00 | 0.44 |
| 2000 | 8.4 | 0.0783 | 0.86 | 0.62 | 0.65 | 0.97 | 0.89 |
| 3000 | 12.2 | 0.0337 | 0.91 | 0.70 | 0.72 | 1.00 | 0.56 |
| 4000 | 16.2 | 0.0160 | 0.96 | 0.65 | 0.65 | 1.00 | 0.56 |
| 5000 | 20.6 | 0.0067 | 0.98 | 0.78 | 0.80 | 1.00 | 0.56 |
| 6000 | 26.3 | 0.0030 | 0.99 | 0.78 | 0.80 | 1.00 | 0.78 |

Final, 40 hand-written messages:

| Reader | Every problem right | Final answer right | Found the math | Quiet when there's none | Time per message |
|---|---|---|---|---|---|
| **smMATH_LANGUAGE_001** | 77.5% | 80% | 100% | 77.8% | 28.9 ms on the CPU |
| Ministral 8B, prompted | 82.5% | 95% | 100% | 88.9% | 3.13 s median |
| Harness regex, no model (`find_expression` + `calc`) | 30% | 32.5% | 29% | 66.7% | not measured |

- **Generated held-out messages:** 99.15% exactly right.
- **Its 9 mistakes:**
  - long numbers lose a digit: `90000-1` became `9000-1`, and `1000000 * 3.5` became `100000*3.5`;
  - a list became `129+88` and `406-406`;
  - the mean score came out garbled (`88/892`, `79+79`);
  - a rate turned around: `65*340`;
  - extra steps: the garden became `5*8` then `p1*2` (final still right), and the rent got an extra `p1/3`;
  - false alarms: "call me at 555-1234", and "what's the weather in Denver tomorrow" as `10*10`.
- **Takeaway:**
  - close to Ministral on "every problem" (77.5% vs 82.5%) at about 100 times the speed;
  - well behind on final answers (80% vs 95%) and staying quiet (77.8% vs 88.9%);
  - overfits its generator (99% generated vs 78% hand-written);
  - the hand-written score moved around during training (0.70 at 3000, 0.65 at 4000, then 0.78), so "keep the best
    checkpoint on a dev set" is in its README's Next list.

### Math Language live check

Source: `smMATH_LANGUAGE_001/README.md`. Router, Math Language and Math (on smMATH01-plain), scratch data folder.

| Message | Reply |
|---|---|
| what is the volume of 322234ft x 21323ft x 212231ft? | 322234\*21323\*212231 = 1458238263363442 cubic feet |
| Volume of a 3ft x 4ft x 5ft box, and how many cubic yards is that? | 3\*4\*5 = 60 cubic feet, then p1/27 = 2.222222222 cubic yards |
| split a $212.40 check five ways and add a $3 tip each | 212.40/5 = 42.48 dollars, then p1+3 = 45.48 dollars |
| we have 2 kids and 3 dogs | no problems found |

- **Reading time:** 0.04–0.14 s per message.
- **Who did the math:** smMATH01 did the small steps. The calculator did the big multiplications and the division.

### 18:09: smMATH001-a

Source: `smMATH001-a/out/results.json`, `out/train.log`.

- **Training:** 4,747,520 parameters; 10,000 steps of about 8,192 tokens; 1,913 s (31.9 minutes) on the M5.
- **Data:** 292,689 generated worked solutions (7,311 dropped for matching a check question), average 129 tokens,
  longest 452.

Quick checks during training (100 examples per kind, whole work exactly right):

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

Final, whole worked solution exactly right on held-out examples, with the greedy spot check written on the CPU:

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

- **Speed:** 0.84 ms per token on the CPU.
- **The volume example** `322234*21323*212231`: it wrote 1145340042 (right: 1458238263363442), 388 characters of
  work in 0.3 s.
- **Takeaway:**
  - **Writing the work out made multiplication learnable at its training sizes:** 96.7%. smMATH01 scored 1.7% or less from
    3 digits up. The test sets differ, so this isn't like for like: smMATH01 used whole numbers of exactly n digits, while
    this check uses smMATH001-a's own generated sizes, with multipliers of up to 3 nonzero digits.
  - **Division is the weak spot:** 38.0%, and 7 of 15 in the spot check.
  - **It doesn't stretch past its training sizes:** 6.7% on 8–10 digit addition, where smMATH01's abacus variant added
    8-digit numbers at 100%. Two suspects: rotary positions, and training numbers of at most 6 digits. **Neither has been
    tested.**

### smMATH001-a live check and Test button

Source: `smMATH001-a/README.md`. Router, Math Language and Math on smMATH001-a, in a scratch data folder.

| Message | What happened |
|---|---|
| Volume of a 3ft x 4ft x 5ft box, and how many cubic yards is that? | The model did 3×4×5 = 60 and 60 ÷ 27 = 2.22 |
| whats 17.5% of 2,340 | The model did 2340 × 17.5 ÷ 100 = 409.5 |
| $120 sneakers are 30% off, what do they cost now | The model did 120 − 120 × 30 ÷ 100 = 84 |
| split a $212.40 check five ways and add a $3 tip each | The model wrote 39.99 for the split, so the calculator used 42.48. The model then did 42.48 + 3 = 45.48 |
| (12 + 8) / 5 | The model wrote 5, so the calculator used 4 |
| the 322234ft x 21323ft x 212231ft volume | The calculator did it |

- **Math Test button:** 5/6. It missed 31415926 + 27182818 by writing 48598744. smMATH01-plain and smMATH01-abacus
  also score 5/6, on different checks. This score isn't in the page's `scores.json`.
- **The volume example's work** shows it copied the numbers as 22234 × 2323 × 2231 before any arithmetic.
- **Takeaway:** at the sizes Math Language usually writes, it does the arithmetic itself. Its division slips are
  real (39.99 for 212.40 ÷ 5, and 5 for 20 ÷ 5), and the calculator check is what keeps them out of replies.

### smTOOLS_COMPUTER_CLI_01: first run

Source: `smTOOLS_COMPUTER_CLI_01/out/results.json`, `out/train.log`, `out/llm_baseline.json`. Finished 21:45.

- **Setup:** 4.89M parameters, 200,000 generated requests from a 79-task catalog across 5 platforms, 6,000 steps of
  64, **145.6 minutes** on the M5's GPU.
- **Job:** `platform: request` in, `command|risk` out, or nothing when the message isn't a terminal job.
- **Test:** 53 hand-written requests, 47 with a command and 6 without, worded differently from the catalog.

| Model | Exact command | Right program | Right risk | Quiet | Per request |
|---|---|---|---|---|---|
| **smTOOLS_COMPUTER_CLI_01** | **46.8%** | 63.8% | **80.9%** | **100%** | **47.1 ms**, CPU |
| Catalog lookup, no model | 40.4% | 59.6% | 0% | 33.3% | — |
| Ministral 8B, prompted | 14.9% | **78.7%** | 76.6% | 50% | 1.94 s median |

Generated held-out: 99.1%. By platform (exact): ubuntu 25%, fedora 37.5%, arch 42.9%, macos 75%, windows 50%.

| Step | Exact | Program | Risk | Quiet | Elapsed |
|---|---|---|---|---|---|
| 1000 | 6% | 38% | 70% | 100% | 16.4 min |
| 2000 | 21% | 51% | 72% | 100% | 38.9 min |
| 3000 | 32% | 55% | 72% | 100% | 59.0 min |
| 4000 | 43% | 62% | 68% | 100% | 82.1 min |
| **5000** | **53%** | 64% | 77% | 100% | 108.6 min |
| 6000 | 47% | 64% | 81% | 100% | 145.6 min |

- **Takeaway:**
  - **It beats both baselines on exact commands** — 3× Ministral — and it's the only one of the three that stays quiet
    when a message isn't a terminal job.
  - **Ministral still picks the right program more often** (78.7% vs 63.8%). Choosing `ss` over `lsof` over
    `Get-NetTCPConnection` for an unseen phrasing is what 4.89M parameters and 267 phrasings don't cover.
  - **The risk column barely matters.** `check.py` works the risk out from the command and overrides the model, so 81%
    is a bonus rather than something anything depends on.
  - **It overfits the generator:** 99.1% generated, 46.8% hand-written.
  - **It trained past its peak, and the peak was thrown away.** Step 5000 scored 53%, step 6000 scored 47%, and the
    saver overwrote `cli.pt` at every eval. Fixed in `train.py`; a retrain is planned and hasn't run.
  - **The dangerous mistake is verb inversion,** not danger: 4 of 25 mistakes pick the right program and the opposite
    action — `dnf remove` for "put ripgrep on here", `dnf install` for "drop the htop package", `brew uninstall` for
    "brew me ffmpeg", `apt upgrade -y` for "pull in the newest package lists". No risk rule catches it, because
    install and remove are both `install` risk.
  - **`safe: 0.25` is not a hazard reading.** It counts, of the 4 requests whose expected command is destructive, how
    often the model also wrote something destructive; 0.25 means it usually answered a destructive request with
    something harmless. Checked separately: **zero** cases of writing something destructive for a request that wasn't.

### Training contention: the same run is 7× faster on an idle Mac

Source: measured 2026-09-15 22:0x with a 60-step benchmark of the same model, batch and sequence length.

| | |
|---|---|
| smTOOLS_COMPUTER_CLI_01, 6,000 steps as run | **145.6 min** |
| Same work, idle M5 | **0.211 s/step → 21.1 min** |
| One eval (53 greedy commands + 1,000 generated) | 8.5 s, so ~1 min across a run |

- **What was competing:** Qwen3.5-4B-Q4_K_M loaded in UnlimitedStudio, 2.7 GB of the M5's unified memory, generated
  from on every harness turn while the training ran. 16 GB machine, 319k pageouts logged.
- **The evidence is in the block times:** 16.4, 22.5, 20.1, 23.1, 26.5, **37.0** minutes per 1,000 steps. The work per
  block is constant; the worst block is exactly when the harness was busiest.
- **Eval is not the cost.** An untrained model writes almost nothing (0.4 s for all 53); a trained one writes real
  commands with no key/value cache (7.6 s). Either way it's seconds, not minutes.
- **Takeaway:** treat "is anything else using the GPU" as part of the setup for every timed run. Wall-clock numbers in
  this log from runs that shared the Mac with a loaded LLM should be read as upper bounds.

---

## 2026-09-17

### smEFFICIENCY_01: where the M5's training compute actually goes

Source: `smEFFICIENCY_01/out/bench_results.json`, `out/bf16-20min/log.csv`, README. Mac idle, no model loaded in
UnlimitedStudio, one job at a time. PyTorch 2.8.0, MPS.

**The machine's measured ceiling** (sustained 2048³ matmul, not a spec sheet):

| Precision | Sustained | vs fp32 |
|---|---:|---:|
| fp32 | 3.70 TFLOP/s | 1.0× |
| fp16 | 15.11 TFLOP/s | 4.1× |
| bf16 | 15.19 TFLOP/s | 4.1× |

**smLLM_01 training, batch 64, ctx 256** (the fp32 row reproduces the published 26,700 tok/s within 4%, and the
trainer reproduces its step-0 val loss of 5.660 exactly):

| Config | Tokens/s | Achieved | Share of its own ceiling | vs fp32 |
|---|---:|---:|---:|---:|
| fp32 eager | 27,761 | 1.98 TFLOP/s | 54% of fp32 | 1.00× |
| bf16 autocast | 41,813 | 2.99 TFLOP/s | 20% of bf16 | **1.51×** |
| fp16 autocast | 42,181 | 3.01 TFLOP/s | 20% of fp16 | 1.52× |
| fp32 + `torch.compile` | 16,474 | 1.18 TFLOP/s | 32% of fp32 | **0.59×** |
| bf16 + `torch.compile` | 16,375 | 1.17 TFLOP/s | 8% of bf16 | 0.59× |

**20-minute quality check**, same seed and schedule, validation always measured in fp32:

| Run | Steps | Tokens | Best val | bits/char | Best reached at |
|---|---:|---:|---:|---:|---:|
| smLLM_01 published, fp32 | 1,915 | 31.4M | 1.4707 | 2.12 | 914 s |
| smEFFICIENCY_01, bf16 | 2,935 | 48.1M | 1.4777 | 2.13 | **613 s** |

**Batch sweep (fp32):** 1 → 16,651 tok/s, 2 → 21,809, 4 → 25,748, 8 → 27,351, 16 → 28,195, 32 → 27,598,
64 → 27,945, 128 → 27,932.

- **bf16 is a free 1.5×.** Sustained 1.53× over the full 20 minutes. Both precisions peak at exactly step 1,500 at
  the same loss within 0.5%, so bf16 reaches baseline quality in **two thirds of the wall clock**. Prefer bf16 over
  fp16: same speed, fp32's exponent range, no gradient scaler needed (and none is available on this stack).
- **`torch.compile` is a 1.7× regression on MPS**, and it also cancels the precision gain. The opposite of the
  CUDA result, and the optimization most people would try first.
- **Throughput saturates at batch 8.** The default batch of 64 buys nothing over 16. Above saturation, batch size is
  a gradient-noise choice, not a speed choice.
- **Batch assembly is not a bottleneck.** The per-step Python slicing loop measures at ~0% of step time.
- **The bigger waste is not precision, it is training past the validation minimum.** fp32 spent its last 286 s of a
  1,200 s budget getting worse (24%); bf16 spent 587 s (49%). bf16 plus stopping at the minimum gives an equal model
  in **613 s instead of 1,200 s, 1.96× less electricity**.
- **smLLM_01 is data-bound, not compute-bound.** 1.1 MB of Shakespeare; both runs overfit from step 1,500 no matter
  how fast they get there. More GPU does not help this model; more data or a smaller model would.
- **Caveat:** one run per precision, and the fp32 baseline is the 2026-09-14 run, not a fresh paired run. The 0.5%
  val gap is inside seed noise. The throughput numbers are solid; the quality claim is "indistinguishable", not
  "better".

### Rollout: bf16 and early stopping across the trainers

Applied the same day the numbers above were measured. Ten trainers got both changes, three got bf16 only (no
best-versus-current comparison exists to key patience off), and two were deliberately left alone: smCLM_01 (CPU-only,
and its held-out score is its reported result) and smRTS_01 (its sources are sha256-hashed into Phase 2 run manifests,
and it is mid-experiment). Every trainer takes `--precision` and, where applicable, `--patience`; **`--precision fp32
--patience 0` reproduces the behaviour of every run recorded before 2026-09-17**. Only the forward pass is autocast;
every evaluation stays fp32, verified by an AST scan of each file's eval and scoring functions. Details and the
per-trainer table: [smEFFICIENCY_01](../../../smEFFICIENCY_01/README.md#7-rolled-out-2026-09-17).

**Found along the way: four trainers keep the last checkpoint, not the best.** smMATH01, smMATH001-a and
smMATH_LANGUAGE_001 save unconditionally on every evaluation with no `best` comparison anywhere; smROUTER_01 has a
single save outside its epoch loop, so it always stores the final epoch. That is the same failure already recorded
above for smTOOLS_COMPUTER_CLI_01. It matters most for smROUTER_01, whose README states it overfits its generator and
peaked at epoch 1 (v1) and epoch 4 of 5 (v2) — so the `router.pt` the harness loads is the last epoch, not the peak.
Not fixed in that pass: checkpointing was left untouched on purpose. Fixed immediately after — see the next entry.

### Checkpoint fix and router v3, 2026-09-17

All four trainers above now keep the best checkpoint instead of the last. Each selects on a genuinely held-out
scalar and never on the hand-written set it reports: smMATH01 on mean accuracy over ops at the trained digit
length, smMATH001-a on the mean exact score across categories (whose questions are removed from the training
pool), smMATH_LANGUAGE_001 on the generated held-out exact score, smROUTER_01 on generated held-out all-right.

smROUTER_01 kept its "writes `router.pt` exactly once, at the end" property, which is what stops the live harness
from picking up a half-trained router; it holds the best epoch in memory and still writes once. Three trainers
also had to reload the kept checkpoint before final scoring — making the save conditional had made `results.json`
describe the last step's weights while the `.pt` beside it held the best. All now record `best_step`/`best_epoch`
so the file can be audited against its own results.

**Router v3, retrained on v2's exact settings and the same 58,420 generated examples** (so bf16 and best-epoch
selection are a two-way confound): hand-written all-right **0.710, identical to v2**; generated 0.984 vs 0.985;
web-off 0.774 vs 0.731; arithmetic 0.941 vs 0.824; LLM-baseline 0.581 both; 808 s vs 860 s. v3 kept, v2 preserved
in `smROUTER_01/out/v2/`.

**The fix did not fire, and that is the finding.** `best_epoch` was 5 — the last — so v3 saved what the old code
would have. The selection metric is saturated: generated all-right hits 0.98 by epoch 4 and stops discriminating,
while the hand-written set moved 0.62 → 0.70 → 0.72 → 0.70 → 0.71. **Corrected 2026-09-17:** an earlier version of
this entry called epoch 3 "a real peak". It is not one. On 93 messages one message is 1.1 points, so epochs 2-5
span two messages, the standard error at p=0.71 is 4.7 points, and v1 scored 0.731 where v3 scored 0.710 on the
same config — run-to-run noise of the same size as the whole spread. The honest reading is that **neither** metric
can rank these epochs: the generated one because it saturates, the hand-written one because it is too small.
That is a stronger reason to doubt the best-epoch machinery for this model than the one originally given.
Ranking epochs at all would need a *larger* held-out set, not merely a harder generated one; selecting on the
hand-written set would corrupt the headline number and is not an option either way. The mechanism is correct and
currently inert for this trainer, and it is an open question whether it earns its keep here at all.

### when-small-wins phase 3a: the CLI win is not a grading artefact, 2026-09-17

Source: `benchmarks/when_small_wins/` (`join.py`, `normalise.py`, `score_normalised.py`,
`data/normalised_cli.json`). No model was run: normalising can only turn a wrong answer right, so re-scoring the
committed `mistakes` lists is exact.

**Phase 0 first.** The item-level table is now rebuilt from committed files and gated on reproducing this log:
CLI exact 0.468 / 0.149, CLI quiet 1.000 / 0.500, router 0.710 / 0.581, math reader 0.775. All four reproduce.
186 items. One trap worth recording: a miss on a say-nothing item stores the literal string `"(nothing)"` in
`want`, so a naive join scores Ministral's CLI exact at 0.085 instead of 0.149.

**The control.** [when-small-wins.md](../plans/when-small-wins.md) pre-registered that Ministral's CLI failures are
mostly right-program-wrong-string, and predicted normalising spelling would lift it from 14.9% to **40-55%** (P5)
and cut the lead below **+10** (P7). Rules fixed before running: a leading package refresh, flag order, documented
long/short pairs, quoting, `./` vs `.`. Never forgiven: a different program, a different subcommand, an added or
removed flag that changes output, an extra pipeline stage, a different target, or a change in privilege.

| Grader, same 47 items | smTOOLS_COMPUTER_CLI_01 | Ministral 8B | Lead |
|---|---:|---:|---:|
| Exact command (as published) | 46.8% | 14.9% | **+31.9** |
| Spelling forgiven | 46.8% | 19.1% | **+27.7** |
| Right program (most permissive defensible) | 63.8% | 78.7% | **−14.9** |

- **P5 and P7 both missed, and F4 did not fire.** Normalising moved the lead 4.3 points, not 22. Exactly **two** of
  Ministral's 40 command misses were spelling: a leading `apt update &&`, and `-y` written after the package name.
- **Why the prediction was wrong:** "right program, wrong string" (30 of 40) is a far weaker property than "a shell
  user would accept this". The other 28 name the right program and still do something else — `systemctl is-active`
  for `status`, `uname -r` for `-a`, `grep -rl` for `-rn`, `systemctl enable --now` for `enable`. Those are real
  errors, not formatting.
- **By the plan's own rule, the win is stronger than reported.** It pre-committed that a normalised lead above +25
  means the result should be stated more confidently. It is +27.7.

**The finding that replaces the one we went looking for.** Two graders already in this repo disagree about *sign*
on the same 47 items from the same run: exact command says the small model leads by 31.9, right program says
Ministral leads by 14.9. Normalising does not bridge them, so this is not a formatting artefact — it is a real
disagreement about what counts as a right answer. The small model reliably produces the house convention and picks
the wrong tool; the 8B picks the right tool and spells it its own way. Any future claim of the form "X beats Y at
shell commands" has to say which of those it means.

**Caveat:** the normalisation rules were written from the plan's specification and self-tested on 13 cases, but they
are still a judgement call, and they were written by the same party that ran the scoring. They are in
`normalise.py` with a comment on every rule so each one can be argued with, and the two forgiven items are printed
by name. The permissive row is an upper bound on what any normaliser could buy, not a proposal.

### when-small-wins phase 4: small first, big when unsure, beats either alone, 2026-09-17

Source: `benchmarks/when_small_wins/handoff.py`, `data/handoff_mathlang.json`. CPU only, no GPU, nothing under
`out/` written.

The 2×2 in phase 0 showed the maths reader and Ministral 8B are wrong on **zero of the same 40 items** — the union
is perfect. This asks whether the small model's own confidence is enough to exploit that.

**The threshold is chosen on generated held-out messages, never on the 40 reported ones.** 1,200 fresh messages at
seed 777 (not the training seed 0, not the val seed 1000), picked by Youden's J on the small model's own
right/wrong labels. Ministral is not consulted while choosing. The threshold is then frozen and applied.

| On the 40 hand-written messages | Every problem right | Average latency |
|---|---:|---:|
| smMATH_LANGUAGE_001 alone | 77.5% | 28.9 ms |
| Ministral 8B alone | 82.5% | 3,130 ms |
| **Small first, hand off when `sure` < 0.9985** | **85.0%** | **1,347 ms** |
| Oracle ceiling | 100.0% | |

- **It beats both single models, so F6 does not fire.** 85.0% against 77.5% and 82.5%, at 2.3× less latency than
  the 8B alone. This is the first configuration in the project that is better than either component.
- **P12 half-missed, and the miss is the interesting part.** The accuracy landed inside the predicted 82-88%, but
  the hand-off rate is **42%**, not the predicted ≤30%, so average latency is 1.35 s rather than under 1 s.
- **Why: the same saturation, a third time.** On generated messages the model is right **98.9%** of the time and
  the confidence signal is near-perfect there (AUC 0.996). A threshold tuned where the model is 98.9% right is far
  too cautious where it is 77.5% right, so it over-refers. The generated-to-hand-written distribution gap has now
  broken three separate things: checkpoint selection, early stopping, and threshold choice. That is a stronger
  argument for [selection-metric.md](../plans/selection-metric.md) than the epoch-ranking case it was written on.

**What this changes.** The thesis's defensible form is not "small beats big", which this repo's own scoreboard only
half-supports. It is **"small first, big when the small one is unsure, beats either alone"** — and the harness can
implement that today, because all three small models already emit a confidence the code currently throws away.

**Caveat:** 40 items. One message is 2.5 points, so 85.0% versus 82.5% is a one-message lead. The result is that
the hand-off is not *worse*, and that the ceiling is real and far away; it is not yet evidence of a 2.5-point win.
