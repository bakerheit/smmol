# paratroop_harness_02

A mixture of small cognitive modules. Each module is a small model with one job and a narrow
JSON contract. Working memory, long-term memory, math, the web and files are code, not models.

It follows [paratroop_harness_01](../paratroop_harness_01/README.md) (a pipe of functions around
a whiteboard). The design notes are in [docs/engineering/paratroop_harness/architecture.md](../docs/engineering/paratroop_harness/architecture.md).

## The cycle

```text
router -> perception -> attention -> recall -> math language -> planner -> predictor -> critic -> decision -> language
                                                    |                  ^                                   |
                                          math works out the           +------ tool result <- act <--------+
                                          problems it found                    (math does calculator steps)
```

| Module | Gets | Gives back |
|---|---|---|
| Router | the message and the last reply | what kind of message it is, which tool it needs, and whether to ask first. A small trained model, not a language model |
| Perception | the message | observations (kind, text, who, what, when, where) and a goal |
| Attention | observations | `for_me`, plus a relevance score per observation. Low scores drop out of focus |
| Recall | the goal and focus | memory queries. Code searches the store, then Recall ranks what came back |
| Math Language | a message with numbers in it | the math problems in it, which Math or the calculator works out before the Planner runs. A small trained model |
| Planner | goal, focus, memories, tool results | up to 3 candidates: `answer`, `make`, `ask` or `tool` |
| Predictor | candidates | outcome, helpfulness (0 to 1) and risk for each |
| Critic | everything so far | errors, contradictions, weak assumptions, gaps, and one question that would clear them up |
| Decision | candidates, predictions, critique | the chosen candidate |
| Math | an expression, from a calculator step or a problem | the answer, checked by the calculator. A small trained model |
| Language | the decision and what's known | the reply, or for a `make` step the thing itself (code, a draft, a list) |

- If the decision is a tool, code runs it, the result goes into working memory, and the Planner
  goes again. It stops after 3 cycles.
- With only one candidate, the Predictor and Decision are skipped.
- If the Decision names a number that isn't a candidate, the most helpful prediction wins.
- Afterwards, a rule (not a model) saves the facts, preferences and events that got attention.

## Not models

| Brain function | Here |
|---|---|
| Working memory | the structured state, saved per run as `runs/<id>.json` |
| Long-term memory | `memory/memories.jsonl`, searched with BM25. Listed with `--memories`, deleted with `--forget` or the page's Forget buttons |
| Math | `calculator`: parses the expression and walks only numbers and operators, never `eval`. It also checks the trained [Math module](#the-math-module-smmath01) |
| Web | `web_search` and `web_browser`, both through the PC's browse server, so its SSRF guard applies |
| Files | `file_system` in `workspace/`: plain file names only, no symlinks, and delete moves to `workspace/.trash/` |
| Command line | off and not built. It needs a person's approval for every command first |

## Contracts

- `contracts.json` has a JSON Schema per module. It's sent to llama.cpp as
  `response_format: {"type": "json_object", "schema": ...}`, so the server constrains the reply
  to that shape.
- Every reply is still validated, then normalized: numbers clamped, strings and lists cut to
  their limits.
- A reply that doesn't fit gets one retry with the problems spelled out. A second miss stops the
  run with "Planner broke its contract twice: ...".
- **The length caps are there for a reason.** In the first nested test, Ministral 8B filled every
  field with speculative essays in markdown and got cut off at 500 tokens after 32 s. Short
  `maxLength` limits plus one example per prompt fixed it.
- If a server rejects `response_format`, the harness remembers that and just validates afterwards.

## Guards in code

- **Tool candidates** need an enabled tool and a target with no placeholder like `[...]`. A
  file action has to be named, and an identical tool call isn't run twice.
- **File names** are plain names, and a path can never leave `workspace/`.
- **A tool step that names no tool** becomes a `make` step instead of being thrown away.
- **Questions cost the person a turn.** Code allows one only for a high-severity problem, and never
  twice in a row.
- **Long-term memory skips request details.** "User specifies programming language" is about one
  request, not about the person, so it isn't kept.
- **A message that's nothing but arithmetic goes to arithmetic** ("2+2", "12 x 7 ="), whatever the
  router or Decision made of it. smROUTER_01 called "2+2" small talk, and without a Planner nothing
  else would pick a calculator step.
- **A message about something no model can know gets looked up.** Weather, news, a price, who won:
  a short list in `LOOKUP`. The router gets first say, and this rule catches what it misses, so a
  lookup happens even with the Planner off. smROUTER_01 read "the weather in 60601" as arithmetic,
  99% sure, because of the bare number.
- **A list of links isn't an answer.** After a search on one of those questions, the top result is
  opened and its text goes to the Language module. Sites in `CLIENT_SIDE` (weather.com and friends)
  are skipped, because they're drawn in the browser and the fetched text is menus and adverts, and
  when every hit is one of them no page is opened at all. Without this the 4B invented "72°F and a
  20% chance of rain" from nothing but page titles.
- **The Language module can hand the turn back once.** It's told what tools exist, and when it needs
  something it wasn't given it writes one line, `NEED: web_browser https://...`, instead of a reply.
  The harness runs that tool and asks it to write the reply again. Lookups only, never a file write,
  never inside a `make` step, and never twice.
- **Units stuck to numbers are read past** ("322234ft", "12 kg", "$64.50") when code pulls arithmetic
  out of a message.

## Make steps

A chat asked for "a function in javascript that takes firstNumber, operator, and secondNumber and
then calculates based on those 3 arguments". Three turns later it still hadn't written it.

1. **The request:** the Planner offered two questions and a file save. The Critic called the
   operator's type and division by zero "high-severity missing information", and Decision asked
   about them.
2. **"all":** it asked again, about the same things.
3. **"nah that sounds good":**
   - The Planner finally proposed "Draft the function implementation", but typed it as a tool with no
     tool, so a guard threw it away.
   - Language, capped at 150 words, described the function and asked "Want me to draft the code now?"
4. **Along the way:** long-term memory saved "User specifies programming language" and "User wants
   function to calculate based on arguments".

**The gap:** the cycle could answer, ask or use a tool, but no step made the thing itself.

**The fix:**
- **A fourth candidate type, `make`:** write the thing now, whether it's code, a draft, a list or a
  plan.
- **Language has a `make_prompt`** with its own token limit (`make_max_tokens`, 1800).
  - It writes the thing itself.
  - It picks sensible defaults for whatever is left open. The Critic's `missing` issues arrive as
    `open_details`.
  - It names those defaults in one line.
- **Prompts:**
  - The Planner includes a make candidate for write-or-build requests.
  - The Critic doesn't call an open detail high severity when a sensible default exists, and it
    treats "sounds good" as an answer.
  - Decision prefers make.
- **Rules in code,** so none of this depends on an 8B model obeying a prompt: the no-tool guard, the
  question rules and the memory filter above.
- **The router can't force a write.** smROUTER_01 reads "write a function in javascript" as
  `file_write` (98% sure), so it may only force lookups: calculator, web_search and web_browser.

**Live check, 2026-09-15.** Every module was on Ministral 8B, run in a scratch data folder so the
page's switches weren't touched.
- **It works now.** The same message gets code on the first turn, in 82 s, through Router,
  Perception, Attention, Planner (one make step), Critic and Language. The function rejects bad
  operators and division by zero.
- **The first make prompt caused a silent default.** An unknown operator quietly became `+`. The
  prompt now says defaults decide what's supported, and bad input gets a clear error.
- **Still not fixed:**
  - **The Critic still calls "which operators" high severity.** The question rules only help when
    there's a non-question candidate to fall back on.
  - **Saving a file lost ground to make.** For "Save a shopping list", the Planner now lists a make
    step first. In 3 runs, the file save came third twice and wasn't offered once. Before the
    change, it came first.

## Run it

The page (only answers on this Mac):

```bash
python3 server.py
```

Then open http://127.0.0.1:8771.

The CLI prints each module call and its time as it goes:

```bash
python3 harness.py "What's 17.5% of 2,340?"
```

Shell pipes. Working memory (JSON) flows through, so each stage sees everything before it:

```bash
./m perception "I have a meeting this Friday" | ./m attention | ./m planner | ./m critic | ./m show
```

Other commands:

```bash
python3 harness.py --list
python3 harness.py --use critic=pc/gemma-4-12b
python3 harness.py --test planner
python3 harness.py --memories
```

Compare with v01 on the same conversations:

```bash
python3 eval.py
```

## Switches

Every module and every tool can be switched off with the checkbox on its card: modules sit around
working memory, tools sit in a row underneath, and a tool card lights up and times itself while it
runs. Choices are saved in `switches.json`. Off means the rest of the brain can't tell it exists:

- **It never runs.**
- **Prompts leave it out.** The Planner's tool menu and tool rules, and the Language module's list of
  what this assistant can reach for, are all built from whatever is on. The Planner's example uses no
  tools, and the Language module won't offer a lookup it can't do.
- **Contracts leave it out.** The Planner's allowed tool list only has tools that are on, and with
  no tools there's no `tool` candidate type at all.
- **Other modules' input leaves it out.** No critique when the Critic is off, no predictions when
  the Predictor is off, no memories when Recall or memory is off, no actions when no tools are on.
- **Fallbacks keep a reply coming:**
  - no Perception: the raw message becomes one observation;
  - no Planner: one "reply directly" candidate;
  - no Decision: the most helpful prediction, or the first candidate that doesn't ask;
  - no Language: the tool result or chosen step as is.
- **`command_line` can't be switched on** until it's built.

## The Router module (smROUTER_01)

The first module in the cycle is a small model trained here, not a prompted LLM:
[`smROUTER_01`](../smROUTER_01/README.md), about 1.9M parameters. It reads the message and the
last reply, and in milliseconds it says what kind of message it is, which tool it needs, and
whether to ask first.

- **Where it comes from:** the `smmol` provider lists trained checkpoints, so the Router's
  dropdown shows `smROUTER_01` once `../smROUTER_01/out/router.pt` exists (before that it says "not
  trained yet"). Until then, or when it's switched off, the cycle runs without it.
- **What it can see:** only the tools that are on. Tools that are off get zero probability inside
  the router.
- **What it does to the cycle:**
  - small talk it's at least 90% sure of stops the run right there;
  - its guess goes to the Planner as a hint;
  - if it's at least 70% sure a lookup is needed (calculator, web search or web browser) and nothing
    has used that tool yet, Decision is overruled to use it. For web search, the harness adds the
    search candidate itself. It never forces a file write.
  - if the chosen step is a question and the router is at least 80% sure nothing needs asking, the
    first non-question candidate wins instead.

## The Math module (smMATH01)

A module slot for a math model trained in SMMOL, like
[smMATH01](../smMATH01/README.md), which does arithmetic in its own weights.

- **Its dropdown only lists math models:** `smMATH001-a`, `smMATH01-abacus`, `smMATH01-reversed` and
  `smMATH01-plain`. Each says "not trained yet" until its checkpoint exists, and the Router's dropdown
  only lists routers.
- **Two kinds of math model:**
  - **Step models like smMATH01:** code splits the expression into single `+`, `-` or `*` steps and
    asks the model for each.
  - **Whole-expression models like [smMATH001-a](../smMATH001-a/):** they get the whole expression,
    with divisions moved last where that's the same value, and write out their work.
  - **Checking:** the calculator checks the final answer to within a cent, and the work shows up
    under the Math section of a run.
- **When it runs:** whenever a calculator step is chosen, the Math module is on, and its model is
  trained.
- **How it works for a step model like smMATH01** (a whole-expression model gets the expression in one
  piece instead, as described above): code walks the expression and hands the model one whole-number
  `+`, `-` or `*` at a time.
  - Decimals are shifted into whole numbers first, so 12.50 + 7.25 goes in as 1250 + 725.
  - A subtraction that would go below zero is flipped, and the answer negated.
- **The calculator checks it.**
  - With the calculator on, every step the model does is checked.
  - A wrong step is replaced by the exact answer and counted as a miss.
  - Division, powers, functions, negative numbers, and numbers too long for the model's window go
    straight to the calculator.
- **With the calculator off:**
  - The model's answers are used unchecked.
  - The Planner is told arithmetic means only `+ - *`.
  - Anything else fails.
- **Router + Math:** when the router is sure a message needs the calculator and no candidate has one,
  the harness pulls the arithmetic out of the message ("hey what's 48213 + 9977?" becomes
  `48213 + 9977`). With every language module off, Router and Math still answer it.
- **Bare arithmetic needs neither Router nor Planner.** The rule under Guards in code sends "2+2"
  straight to Math, even with only Math switched on.
- **The Test button is always checked,** so its score is the model's own. One check uses 8-digit
  numbers, longer than anything smMATH01 trained on.

**Live check, 2026-09-15.** It used `smMATH01-plain`, the first variant to finish training, in a
scratch data folder:

| Expression | Result | Done by |
|---|---|---|
| 48213 + 9977 | 58190 | the model, 0.07 s |
| 12.50 + 7.25 | 19.75 | the model, as 1250 + 725 |
| (250 - 48) + 1990 | 2192 | the model, two steps |
| 123 * 45 | 5535 | the model |
| 1850 / 3 | 616.6666667 | the calculator: no division in the model |
| 31415926 + 27182818 | 58598744 | the calculator: the model wrote 22643342322222822225492 |

With only Router, Math and Language switched on, "hey what's 48213 + 9977?" went Router (0.01 s) →
Math (0.07 s) → Language and came back as 58,190.

**Live check with smMATH001-a, 2026-09-15.** Router, Math Language and Math were switched on:
- **Done by the model:**
  - the 3 x 4 x 5 ft box and its cubic yards (60 and 2.22);
  - 17.5% of 2,340 (409.5);
  - a $120 price at 30% off (84).
- **Caught by the calculator:** the model got two divisions wrong. It wrote 39.99 for 212.40 ÷ 5 and 5 for
  (12 + 8) ÷ 5.
- **Too big for it:** the 322234 x 21323 x 212231 ft volume. It copied the numbers shorter and got it
  wrong, and the calculator did it instead.
- **The details** are in its [README](../smMATH001-a/README.md).

## The Math Language module (smMATH_LANGUAGE_001)

A module slot for a model trained here that reads the math problems out of a message:
[smMATH_LANGUAGE_001](../smMATH_LANGUAGE_001/README.md).

1. **It reads messages with numbers in them,** after the router and before the Planner.
2. **Its problems go into working memory** as a list:
   `[{"id": "p1", "expression", "unit", "about"}, …]`. Later problems can use `p1`, `p2` … for earlier
   answers.
3. **Each problem is worked out in order,** by the Math module when it's on and otherwise by the
   calculator. `p1` is swapped for its answer first.
4. **The answer lands next to its problem.**
   - Nobody tracks who asked. Planner and Language see the problems with their answers.
   - With Language off, the reply lists each problem, its answer and its unit.
   - Once problems are answered, the router and the arithmetic rule don't force another calculator
     step.
5. **A reader that breaks is noted in the trace and skipped;** the turn carries on.

**What it can't do yet:**
- **The volume message takes three pieces:**
  - Math Language finds `322234*21323*212231` (cubic feet);
  - smMATH01 can't multiply numbers that big, so the calculator does both steps;
  - the reply is "322234*21323*212231 = 1458238263363442 cubic feet".
- **Scores:** it gets every problem right on 78% of its hand-written test messages. Ministral gets
  82%, in about 100 times as long. Details are in its README.

## The local English model (smLANGUAGE_EN_001)

The Language card also lists `smmol/smLANGUAGE_EN_001` when
`../smLANGUAGE_EN_001/out/best.pt` exists. It loads the PyTorch checkpoint directly in this process;
no llama.cpp or OpenAI-compatible server sits in between. A small adapter turns the Language
module's working-memory payload into a short text-completion prompt. The option is only offered on
the Language card, because this foundation checkpoint cannot produce the other modules' JSON
contracts.

Loading works, but instruction following does not yet. The first checkpoint scored 0/3 on the
Language checks: it did not reliably ask the supplied question, copy the calculator's `409.5`, or
write the requested JavaScript function. That is useful evidence, not a loader failure: stage-one
pretraining learned English prose, while Paratroop needs a supervised harness-language tuning stage.
The working Ministral model remains selected by default.

The model dropdown lists the preserved `smLANGUAGE_en_SCH_001-kindergarten` checkpoint and the
promoted `smLANGUAGE_en_SCH_001-grade-01` checkpoint separately. The latter follows
`../smLANGUAGE_en_SCH_001/out/latest.safetensors`. Its **Weight configuration** panel shows the
current stage, step, parameters, tensor shape,
file size and tokenizer. Context, width, layers, heads, dropout and tied embeddings edit the source
`model_config.json` for the next training run; invalid width/head combinations are refused. Changing
tensor shapes requires training a matching checkpoint, so the panel keeps current weights separate
from next-checkpoint settings. The **Generation** panel holds live Temperature, Top K, repetition
penalty and token-limit controls.

`smLANGUAGE_en_GENERAL_001-conversation-v1` extends the first-grade school checkpoint with short
conversations rendered in the same prompt format the Harness sends. It is an experimental model:
common trained prompts improve, while unseen wording and supplied-fact copying remain unreliable.

## Settings

Click a module card and its settings are in the inspector that opens. Settings are saved in `settings.json` next to the switches, and a
setting set back to its default isn't stored.

| Module | Settings |
|---|---|
| Language models (Perception to Language) | temperature and token limit, plus the token limit for make steps on Language |
| Local school Language model | Top K and repetition penalty, declared by its selected weight set |
| Router | how sure it must be to stop on small talk (0.9), to force a lookup (0.7), and to skip a question (0.2) |
| Math | whether the calculator checks its answers. The calculator still does whatever the model can't |
| Math Language | only read messages that have numbers in them (on) |

## Chats

- **Replies stay in one conversation.** Answering a question the harness asked continues the same
  chat instead of starting over.
- **Storage:** each chat is `conversations/<id>.json`.
- **What modules see:** the last 6 turns, so an answer like "I'm the investor" arrives with the
  question it answers.
- **The page:** a thread of messages and replies, each reply with a collapsible "how it got
  there", plus New chat and Recent chats.
- **The CLI:** `--continue` (or `--conversation <id>`) keeps talking in the latest chat.
- **One turn at a time per chat.** A turn is saved before the page is told it's done, so a quick
  follow-up always sees it.

**Live check, 2026-09-15:**
1. "I have a meeting this Friday…" took 101 s and got back "help preparing, or just letting me know?"
2. "I'm the investor. A local bakery is pitching to me." took 115 s.

Perception turned the second message into "User is the investor" and "Local bakery is pitching
to User", with the goal "discuss investment opportunity with bakery". The context carried over.
The Critic then flagged three "high" issues and it asked yet another question (fund their growth,
or partner?), so over-asking is still the next thing to fix.

## First live run

2026-09-15, "What's 17.5% of 2,340?", every module on Ministral 8B on the RX 580. It took **82 s**:
10 module calls plus the calculator, and every contract fit on the first try.

| Call | Time |
|---|---|
| Perception | 9.5 s |
| Attention | 5.7 s |
| Planner | 20.9 s |
| Predictor | 15.0 s |
| Critic | 6.3 s |
| Decision | 7.4 s |
| calculator | 0.0 s → `409.5` |
| Planner | 7.6 s |
| Critic | 2.4 s |
| Language | 6.8 s |

The reply: "To find 17.5% of 2,340, multiply 2,340 by 0.175—it equals **409.5**."

## v01 vs v02

`eval.py`, 2026-09-15, everything on Ministral 8B:

| Conversation | v01 | v02 |
|---|---|---|
| Unclear Friday meeting | pass, 59 s | pass, 135 s. The Critic asked "investor or business owner?" |
| 17.5% of 2,340 | pass, 9 s | pass, 46 s (calculator) |
| Remember, then recall | fail | pass, 294 s, but it asked a needless follow-up question |
| Save a file, then read it | fail | **false pass**, 288 s: it never created the file, and memory supplied the list |
| "lol ok" | pass | pass, 9 s |
| Chicago weather today | fail (answered without searching) | fail (asked instead of searching) |

Every experiment's numbers are in [docs/engineering/research/results.md](../docs/engineering/research/results.md). The short version:
- the Critic over-asks;
- it needs today's date;
- create vs update should be checked in code;
- the grader should check what actually happened, not keywords.

## Tests

61 tests against a fake model server that plays every module, plus stubs for the router, both kinds of math model and the math reader:

```bash
python3 -m unittest discover -s tests
```

## Files

| File | What it is |
|---|---|
| `harness.py` | Modules, contracts, working memory, memory store, tools, the cycle, the CLI |
| `harness.json` | Providers, each module's model and prompt, tool switches |
| `contracts.json` | Each module's output schema |
| `checks.json` | Module checks for the Test buttons and `--test` |
| `server.py`, `page.html` | The web page. `page-classic.html` is the pre-redesign page, still served at `/classic` |
| `m` | Shell-pipe wrapper, one stage per call |
| `eval.py`, `eval.json` | v01 vs v02 on the same conversations |
| `memory/`, `workspace/`, `runs/`, `conversations/`, `choices.json`, `scores.json` | Made as it's used |

## Not done yet

- **`command_line`**, with an approval button.
- **Vector search** in long-term memory. It needs an embedding model, which means a download.
  The graph part would link memories through shared people and places.
- **Right-sizing.** Every module runs on Ministral 8B, so the "much smaller than a general model"
  claim is untested.
- **Training.** Predictor and Critic are prompted, not trained.
- **Speed.** One message is 8 or more model calls on one GPU.
