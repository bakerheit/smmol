# smROUTER_01: the Router module

Project: [`smROUTER_01/`](../../../smROUTER_01/README.md). Checked against `model.py`, `route.py`, `train.py`,
`rules.py`, `baseline_llm.py`, `labels.json`, `data.py`, `tests/test_router.py` and `out/` (including `out/v1/`) on
2026-09-15.

## Purpose

A tiny classifier for one brain function: read a message and decide where it goes. It's the first module in
paratroop_harness_02's cycle.

## Input and output

- **In:** the assistant's last message (`prev`, often empty) and the person's message.
- **Out** (`Router.route(text, prev, tools)`):

  ```json
  {"for_me": true, "intent": "question", "intent_p": 0.95, "tool": "calculator", "tool_p": 0.97,
   "ask_first": false, "ask_p": 0.02, "ms": 2.1}
  ```

| Head | Choices (`labels.json`) |
|---|---|
| intent | question (wants information), task (wants something made or done), remember, answer (answers the assistant's last question), statement, small_talk (needs nothing back) |
| tool | none, calculator, web_search, web_browser (a link in the message), file_read, file_write, memory_recall |
| ask first | true only when something needed is missing |

- **`for_me`** is `intent != "small_talk"`.
- **`ask_first`** is true when P(ask) is over 0.5.
- **`tools`** is the list of tools that exist right now. Any other tool gets its logit set to −1e9 (`only_allowed`), so
  it gets zero probability. `none` is always allowed.

## Architecture and size

`RouterNet` in `model.py`:

- **Bytes in:** `CLS` + the last 64 bytes of `prev` + `SEP` + the message, cut to 192 tokens. The vocabulary is 259:
  256 bytes plus `CLS`, `SEP` and `PAD`.
- **Encoder:** token plus learned position embeddings, then `nn.TransformerEncoder` with 4 layers, d 192, 6 heads,
  feed-forward 768, dropout 0.1, pre-norm, then LayerNorm.
- **Heads:** three linear heads read the `CLS` position: intent (6), tool (7), ask (2).
- **1,869,327 parameters** (`out/results.json`).

## Data

- **Generator** (`data.py`): each class is `(intent, tool, ask_first, weight, phrasings)`. Slots like `{city}`,
  `{money}`, `{expr}` and `{items}` are filled from lists.
- **Variety:**
  - answer pairs, where `prev` is a question ending in "?";
  - roughing up: typos, lower case, missing punctuation.
- **v2 added bare arithmetic:** the `{expr}` generator writes things like `2+2`, `48,213 - 9977`, `(12 + 8) * 3` and
  `3.5 x 4`, in phrasings like `{expr} =` and `calc {expr}`. The README says that's about 5% of training.
- **Size:** `train.py` generates 60,000 messages and drops any that match a test message after normalizing. That was
  1,580 in v2, leaving 58,420. A validation set of 3,000 is generated with seed + 1000.
- **Test sets** (written by hand, never generated):
  - `test.json`: 93 messages, 16 of them with a `prev`;
  - `test_arithmetic.json`: 17 messages, scored separately. 14 are expressions, some mid-conversation, and 3 are
    ordinary messages with numbers.

## Train and test

```bash
python3 -m unittest discover -s tests
python3 train.py
python3 route.py --off web_search,web_browser "what's the weather in Denver tomorrow"
python3 baseline_llm.py
```

- **Optimizer:** 5 epochs, batch 128, AdamW lr 1e-3, weight decay 0.01, OneCycle (10% warmup), gradient clip 1.0.
- **Loss:** intent cross-entropy + tool cross-entropy (with the allowed mask) + 0.5 × ask cross-entropy.
- **Switch-off training:** half the batches switch tools off at random; each tool stays on with probability 0.8. An
  example whose tool is off gets the target `none`.
- **Saving:** `out/router.pt` and `out/results.json` are written **once, at the end**.
- **Scoring:** intent, tool, ask first, for me (small talk or not), and all three right.
  - Tool scoring under "web tools off" counts `none` as correct for web messages.
- **Baselines:**
  - `rules.py`: keyword regexes;
  - `baseline_llm.py`: Ministral 8B with the label descriptions and a JSON schema.
- **Tests (9).** All flags: [running.md](running.md#smrouter_01).

## Results

93 hand-written messages:

| Router | When | Intent | Tool | Ask first | For me | All three right |
|---|---|---|---|---|---|---|
| **smROUTER_01 v3** (`out/router.pt`) | 2026-09-17 20:44 | 87.1% | 80.6% | 90.3% | 97.8% | **71.0%** |
| v3, web tools off | 20:44 | 87.1% | 90.3% | 90.3% | 97.8% | 77.4% |
| smROUTER_01 v2 (`out/v2/`) | 2026-09-15 16:17 | 84.9% | 82.8% | 90.3% | 98.9% | 71.0% |
| v2, web tools off | 16:17 | 84.9% | 88.2% | 90.3% | 98.9% | 73.1% |
| smROUTER_01 v1 (`out/v1/`) | 2026-09-15 14:38 | 87.1% | 81.7% | 92.5% | 97.8% | 73.1% |
| v1, web tools off | 14:38 | 87.1% | 91.4% | 92.5% | 97.8% | 79.6% |
| Keyword rules (`rules.py`) | | 90.3% | 81.7% | 88.2% | 100% | 63.4% |
| Ministral 8B, prompted | file saved 14:22 | 79.6% | 80.6% | 94.6% | 97.8% | 58.1% |

- **Bare arithmetic** (17 messages), all three right: v3 94.1% (16), v2 82.4% (14), v1 10, keyword rules 23.5% (4).
- **Speed:**
  - v3: 2.04 ms per message on the CPU;
  - Ministral: 1.99 s median.
- **Training time:**
  - v3: 808 s (bf16);
  - v2: 860 s;
  - v1: 949 s.
- **Generated held-out messages:** v3 gets 98.4% all three right, v2 98.5%.
- **Hand-written "all three right" by epoch:**
  - v1: 74%, 71%, 67%, 72%, 73%;
  - v2: 63%, 69%, 70%, 73%, 71%;
  - v3: 62%, 70%, 72%, 70%, 71%.

**v3 is v2 retrained** with identical settings and the same 58,420 generated examples, changing two things at once:
bf16 training and best-epoch checkpoint selection. They can't be separated from this pair alone. v3 ties v2 on the
headline number, wins web-off and arithmetic, and is 52 s faster.

**Best-epoch selection did not change the outcome.** The trainer now keeps the best epoch, selecting on generated
held-out all-right — never on the 93 hand-written messages, which would corrupt the reported number. But that metric
saturates at 98% by epoch 4, so it ranked epoch 5 highest and v3 saved the last epoch anyway. The hand-written peak
was epoch 3 (72%). The mechanism is right; the signal is too easy. A harder generated held-out set would be needed.

The web-off rows aren't a better model: with web tools off, `none` is the right tool for more messages.

## Known failure modes

From v3's mistakes (`out/results.json`, `out/train.log`); v2's are in `out/v2/`:

- **It overfits the generator.** 98% on generated messages, 71% on hand-written ones, and it wobbles by epoch.
- **Current events without a trigger word:**
  - "how did the Warriors do last night?" routes to `none`;
  - "how much is a Steam Deck these days" routes to `none`;
  - "Will I need an umbrella in Seattle…" routes to `none`.
- **Plain knowledge questions:** "who painted the Mona Lisa" and "any tips for a toddler who won't sleep?" route to
  `web_search`; "Why do cats purr?" gets the right tool but wrongly asks first.
- **Short reactions:** "haha that's great" comes out as a task.
- **Answers to the assistant's question** often come out as a statement or task ("just call it errands", "what about
  Boulder").
- **Asking for code:** "write a function in javascript that…" routes to `file_write` at 98%, because the generator only
  uses "write" for files. The harness never lets the router force a write.
- **Arithmetic** (v2):
  - "2+2" gets the calculator but is labeled small talk;
  - "we have 2 kids and 3 dogs" is treated as math;
  - "my flight is at 7:45 tomorrow" is treated as a task.
  - In v3 only the last of these remains; on 17 messages that is two examples, not a trend.
- **Regression noise:** the original 93 dropped from 73.1% (v1) to 71.0% (v2) and stayed there in v3. With one run of
  each, that can't be told apart from noise.

## In the harness

The details are in [architecture.md](../paratroop_harness/architecture.md#router-router-trained).

- **Registration:**
  - provider entry `smROUTER_01` → `../smROUTER_01/out/router.pt`, kind `classifier`;
  - loader `route.py:Router`;
  - the Router card, grey.
- **What it sees:** `prev` is the last turn's reply. `tools` holds only the router tools whose harness switch is on;
  `memory_recall` also needs Recall on.
- **What it does:**
  - stops the turn on small talk when it's at least 90% sure (`stop_small_talk_at`), unless the message is bare
    arithmetic;
  - gives the Planner a `route` hint;
  - forces a lookup (calculator, web_search, web_browser) when it's at least 70% sure (`force_tool_at`), never a write;
  - skips a question when P(ask) is below 0.2 (`skip_question_below`).
- **A router error ends the turn.**
- **Checks:** `checks.json` → `router` has 5 cases, graded on `intent` or `tool`. The page's score (`scores.json`) is
  4/5 from 2026-09-15 14:45, before v2 existed, with one check failing on "tool was none".

## Next (from the README)

- **Unseen message types:** code and draft requests, short thanks, sports and prices, plain knowledge questions.
- **Better stopping:** a small hand-written dev set, apart from the test set, to pick the best checkpoint instead of the
  last.
- **Free labels:** runs where a module overruled the router, to review and add.
