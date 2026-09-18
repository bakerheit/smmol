# smROUTER_01

A tiny model trained from scratch on a MacBook Pro M5 for one small brain function: reading a message and
deciding where it goes. It's the first module in
[paratroop_harness_02](../paratroop_harness_02/README.md).

For each message (plus the assistant's last reply), it gives back three things:

| Head | Choices |
|---|---|
| intent | question, task, remember, answer, statement, small_talk |
| tool | none, calculator, web_search, web_browser, file_read, file_write, memory_recall |
| ask first | true or false |

- **Size:** 1.87M parameters, a byte-level transformer (d 192, 4 layers, 6 heads, 192 bytes of input).
- **Speed:** 2.04 ms per message on the CPU. The same routing through Ministral 8B takes 1.99 s cold, and
  13.56 s once you paste in the 32 examples it needs to match this model's accuracy.
- **Accuracy:** 71.0% all-three-right on 93 hand-written messages. A prompted Ministral 8B gets 58.1% cold but
  **73.1%** shown 32 examples, so the gap this model was once credited with is supervision, not size —
  see [phase 3b](../docs/engineering/research/results.md#when-small-wins-phase-3b-the-routers-win-was-supervision-not-size-2026-09-18).
  What survives is the clock: same job, same accuracy, 6,647× faster, on a CPU.

## Switched-off tools

A tool that's switched off in the harness gets zero probability inside the model, so the router can't
pick it.

- **Why that works:** during training, half the batches switch tools off at random, and a message
  whose tool is off learns the answer `none`.
- **In practice:** with web search off, "what's the weather in Denver tomorrow" routes to `none`
  instead of `web_search`.

## Results

`out/router.pt` is the third training run (v3): v2's exact settings and the same 58,420 generated
messages, retrained in bf16 with best-epoch checkpoint selection. 5 epochs took 13.5 minutes on the
M5. It's scored on 93 hand-written messages that never appear in training (a test checks that).

| Router | Intent | Tool | Ask first | For me | All three right |
|---|---|---|---|---|---|
| **smROUTER_01 v3** (`out/router.pt`) | 87% | 81% | 90% | 98% | **71%** |
| smROUTER_01 v2 (in `out/v2/`) | 85% | 83% | 90% | 99% | 71% |
| smROUTER_01 v1, the first run (in `out/v1/`) | 87% | 82% | 92% | 98% | 73% |
| Keyword rules (`rules.py`) | 90% | 82% | 88% | 100% | 63% |
| Ministral 8B, prompted (`baseline_llm.py`) | 80% | 81% | 95% | 98% | 58% |
| smROUTER_01 v3, web tools off | 87% | 90% | 90% | 98% | 77% |

- **v3 ties v2 on the headline number** and wins two side suites (web-off 77% vs 73%, bare
  arithmetic 94% vs 82%), at 808 s instead of 860 s. bf16 and best-epoch selection changed together,
  so the two can't be separated from this pair alone.

- **It beats both baselines** on getting all three right, and it's about 1,000 times faster than
  the LLM.
- **It overfits the generator.**
  - Held-out generated messages reach 98%, but the hand-written score wobbles from epoch to epoch.
    v1's best was epoch 1 (74%), v2's epoch 4 (73%), v3's epoch 3 (72%) — but on 93 messages these are
    one-to-two message differences, inside run-to-run noise. Treat them as "no epoch clearly best".
  - It learned the templates better than it learned how people actually talk.
- **Best-epoch selection can't fix that wobble yet.** The trainer keeps the best epoch instead of the
  last, but it selects on generated held-out all-right, which saturates at 98% by epoch 4 and stops
  telling epochs apart — so v3 kept epoch 5 anyway. Selecting on the hand-written set would corrupt
  the number this table reports. And the hand-written set cannot break the tie either: its epochs sit
  one to two messages apart out of 93. Ranking them needs a *larger* held-out set, not just a harder one.
- **The web-off row isn't a better model.** With web tools off, "none" is the right tool for more
  messages, so tool accuracy is easier to get.

**Where it's wrong:**
- **Current events without a trigger word:** instead of `web_search`, "how did the Warriors do last
  night?" and "how much is a Steam Deck these days" both route to `none`.
- **Plain knowledge questions:** "who painted the Mona Lisa" routes to `web_search`.
- **Short reactions:** "haha that's great" comes out as a task, not small talk.
- **Answers to the assistant's question:** they often come out as a statement or a task.
- **Asking for code:** "write a function in javascript that…" routes to `file_write` at 98%.
  - The generator only ever uses "write" for files.
  - The harness never lets the router force a write for this reason. It can only force lookups:
    calculator, web search and web browser.

## Bare arithmetic (v2, rescored in v3)

- **What broke:** in the harness, "2+2" did nothing. v1 had never seen arithmetic without words
  around it, so it called "2+2", "2+3" and "12*7" small talk at 100%.
- **The fix:**
  - A generator for typed expressions like `2+2`, `(12 + 8) / 5` and `calc 1580.5+34946878+64`.
    They're about 5% of training.
  - `test_arithmetic.json`: 17 hand-written messages, scored on their own. 14 are expressions, some
    mid-conversation, and 3 are ordinary messages that happen to contain numbers.

| On `test_arithmetic.json` (17 messages) | All three right |
|---|---|
| smROUTER_01 v1 | 10 (59%) |
| smROUTER_01 v2 | 14 (82%) |
| **smROUTER_01 v3** | **16 (94%)** |
| Keyword rules | 4 (24%) |

- **Still wrong in v2:**
  - "2+2" now gets the calculator, but it's still labeled small talk.
  - "we have 2 kids and 3 dogs" is treated as math.
  - "my flight is at 7:45 tomorrow" is treated as a task.
- **Still wrong in v3:** only "my flight is at 7:45 tomorrow", still read as a task. The other two
  cleared up. 17 messages is a small set, so 82% → 94% is two examples, not a trend.
- **The original 93 messages dropped by 2** when v2 added arithmetic (73% to 71%), and v3 stayed at
  71%. With one run of each, that can't be told apart from training noise.
- **The harness doesn't depend on it:** a message that's nothing but arithmetic goes to Math by rule
  anyway.

## Run it

Tests (data labels, no test messages in training, switching, the rules baseline):

```bash
python3 -m unittest discover -s tests
```

Train. This writes `out/router.pt`, `out/results.json` and the table above:

```bash
python3 train.py
```

Route one message:

```bash
python3 route.py "what's 20% of 85"
```

Route an answer to the assistant's last question:

```bash
python3 route.py --prev "Which city?" "Denver"
```

Route with some tools switched off:

```bash
python3 route.py --off web_search,web_browser "what's the weather in Denver tomorrow"
```

Rebuild the Ministral baseline (uses the PC's GPU):

```bash
python3 baseline_llm.py
```

## Files

| File | What it is |
|---|---|
| `labels.json` | The intents and tools, with descriptions |
| `data.py` | The message generator: templates, slots, answer pairs, typos and casing noise |
| `test.json` | 93 hand-written, hand-labeled messages |
| `test_arithmetic.json` | 17 hand-written messages about bare arithmetic, scored separately |
| `model.py` | RouterNet, byte encoding, and `only_allowed` (switched-off tools) |
| `train.py` | Training with random tool switch-offs, scoring, the results table |
| `route.py` | `Router(path).route(text, prev, tools)`, used by the harness |
| `rules.py` | Keyword baseline |
| `baseline_llm.py` | Ministral 8B baseline through the PC gateway |
| `out/` | `router.pt`, `results.json`, `train.log`, `llm_baseline.json`, and `v1/`, `v2/` with the earlier runs |

## Next

- **Messages the generator never makes:** requests for code and drafts (task, `none`), short
  thanks and acknowledgements, sports and prices, and plain knowledge questions.
- **Give best-epoch selection a metric it can use.** The trainer now keeps the best epoch, but the
  generated held-out score saturates at 98% and can't rank epochs. Either keep a small hand-written
  dev set separate from `test.json`, or make the generated held-out set hard enough to stay
  informative past epoch 4.
- **Learn from the harness:** runs where a module overruled the router are free labels to review
  and add.
