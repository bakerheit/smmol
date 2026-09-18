# Architecture

Checked on 2026-09-15 against `paratroop_harness_02/harness.py`, `harness.json`, `contracts.json`, `checks.json`,
`server.py`, `page.html` and `tests/test_harness.py`. Functions are named so you can find them; line numbers aren't
given because they move.

Contents:

1. [Overview](#1-overview)
2. [Modules](#2-modules)
3. [Contracts](#3-contracts)
4. [Switches and settings](#4-switches-and-settings)
5. [Code guards](#5-code-guards)
6. [Trained-model plumbing](#6-trained-model-plumbing)
7. [Known gaps](#7-known-gaps)

---

## 1. Overview

### What SMMOL is

SMMOL is a research workspace for small models. It has two kinds of project:

- **Models trained from scratch on the Mac** (Apple M5, PyTorch on MPS): smLLM_01, smALLM_01, smCLM_01, smROUTER_01,
  smMATH01, smMATH_LANGUAGE_001 and smMATH001-a. Each is a folder with `model.py`, `train.py`, usually `tests/`, and
  `out/` for checkpoints and results. See [smModels/](../smModels/).
- **Harnesses** that run a "brain" made of parts:
  - [paratroop_harness_01](../../../paratroop_harness_01/README.md) runs `attention | context | route | respond` as a pipe.
    Route calls `@process` and `@tool` functions, and every function writes to a shared whiteboard.
  - [paratroop_harness_02](../../../paratroop_harness_02/README.md) is the main project and the subject of this file.

The idea behind v02: each module is a small model with one job and a narrow JSON contract. Anything that doesn't need a
model is code: working memory, long-term memory, math, the web and files.

### Machines and providers

From `harness.json` → `providers`:

| Provider | `kind` | Address | Label | Used for |
|---|---|---|---|---|
| `pc` | `openai` | `http://127.0.0.1:8081` | PC · RX 580 | Prompted modules. Every one defaults to `ministral-8b` |
| `mac` | `openai` | `http://127.0.0.1:11435` | Mac · UnlimitedStudio | Another choice for prompted modules |
| `browse` | `browse` | `http://127.0.0.1:8080` | PC · browse server | `web_search` (`POST /search`) and `web_browser` (`POST /fetch`) |
| `smmol` | `local` | in-process | Mac · trained in SMMOL | Checkpoints trained in this workspace |

- An `openai` provider is up when `GET /v1/models` returns 200. Chat calls go to `POST /v1/chat/completions`.
- A `browse` provider is up when `GET /health` returns 200.
- A `local` provider is up when at least one of its listed checkpoint files exists.
- `Harness.status()` checks every provider in parallel threads and caches the answer for 15 s.

### Files and the data folder

`Harness(folder=HERE, data_dir=None)` reads config from `folder` and keeps state in `data_dir`, which defaults to
`folder`.

| Read from `folder` | Written to `data_dir`, made as it's used |
|---|---|
| `harness.json`: providers, modules, prompts, tools, examples | `memory/memories.jsonl`: long-term memory |
| `contracts.json`: a JSON Schema per module | `workspace/`: the file tool's sandbox, with `.trash/` |
| `checks.json`: cases for the Test buttons and `--test` | `runs/<id>.json`: working memory of each finished turn |
| | `conversations/<id>.json`: chats |
| | `choices.json`, `switches.json`, `settings.json`, `scores.json` |

`eval.py` and the README's live checks pass a scratch `data_dir`, so the page's switches and memory stay untouched.

### The turn cycle

`Harness.think(live)` runs one turn, in this order:

```text
record which modules and tools are on
router          on + trained                          -> stop if it's sure this is small talk
perception      or fallback: the message is one observation
attention       on                                    -> stop if not for_me
recall          Recall on + memory tool on
math_language   on + trained + the message has numbers -> solve_problems
loop, at most max_cycles (3) tool rounds:
    planner     or fallback: one "Reply directly to the message" candidate
    predictor   on, and more than one candidate
    critic      on
    decide      Decision model or a rule -> ask rule -> arithmetic rule, else router overrule
    if the pick is a tool: act, then loop again; otherwise leave the loop
language        or fallback: the plain reply
remember        memory tool on
finish          save the run and the chat turn, then mark the turn done
```

Step by step:

1. **Router.** Runs when the Router module is on and its checkpoint exists (`router_ready`). The turn stops with
   status `stopped` when `for_me` is false, `intent_p` is at least `stop_small_talk_at` (default 0.9), and the message
   isn't bare arithmetic. A router error ends the turn with status `error`.
2. **Perception.** When it's off, the message becomes one observation: `{id: "o1", kind: "request", text: message[:500]}`,
   and the goal is `message[:120]`.
3. **Attention.** Sets `relevance` on every observation. An index it didn't score gets 0.0. `for_me: false` stops the
   turn with status `stopped`.
4. **Recall.** Needs the Recall module and the `memory` tool. With an empty store it makes no model call and notes
   "long-term memory is empty". Otherwise:
   - the query prompt writes up to 3 queries;
   - BM25 searches the store with those queries plus the raw message and takes the top `memory.recall_top` (8);
   - the rank prompt scores what came back;
   - memories scoring at least 0.5 (`KEEP_MEMORY`) are kept, best first.
5. **Math Language.** Runs when `math_language_on()` (module on, checkpoint exists) and the message mentions numbers.
   The `NUMBERS` regex matches any digit or a number word like one to twelve, dozen, half, twice, double, triple,
   hundred, thousand or million. The `only_with_numbers` setting turns that condition off. A reader that fails is marked
   as an error on its trace entry, and the turn carries on with no problems. Problems it finds go to `solve_problems`
   (see [section 6](#math-language-to-answers)).
6. **Planner.** Writes up to 3 candidates. Guards clean them up before anyone else sees them
   ([section 5](#tool-candidate-guards)).
7. **Predictor.** Skipped with only one candidate (noted in `skipped`). A candidate it didn't predict gets
   `helpful: null` and the risk "no prediction given".
8. **Critic.** Issues with empty text are dropped.
9. **Decide.**
   - With one candidate, a rule picks it.
   - With the Decision module off, a rule picks the most helpful prediction, or the first candidate that doesn't ask.
   - Otherwise the Decision model picks. A choice that isn't a candidate number is replaced by the most helpful
     prediction.
   - Then `_ask_rule` runs, then `_arithmetic_rule`. If the arithmetic rule doesn't apply, `_router_overrule` runs. See
     [section 5](#5-code-guards).
   - The decision records `by`: `model`, `rule` or `router`.
10. **Act.** A calculator step goes to the Math module when `math_on()`; everything else goes to `run_tool`. A tool that
    fails doesn't end the turn: the action gets `ok: false`, and the error text becomes its result.
11. **Cycle limit.** After an act, once the turn has `max_cycles` (3) cycles, a final cycle is added with one `answer`
    candidate decided `by: "rule"`. The run note says "stopped using tools after 3 cycles and answered with what it had".
12. **Language.** Uses `make_prompt` when the chosen candidate is `make`, otherwise `prompt`. When Language is off,
    `_plain_reply` answers:
    - if no tool succeeded but Math Language problems have answers: one line per problem, `expression = answer unit`;
    - else the last successful tool result;
    - else the chosen candidate's summary;
    - else the message itself.
13. **Remember.** Saves observations that pass the memory filter ([section 5](#the-memory-filter)).
14. **Finish.** `_finish` writes `runs/<id>.json` and appends the turn to the conversation *before* it marks the live
    state finished, so a quick follow-up always sees the saved turn. A `HarnessError` ends the turn as `error` with its
    message. Any other exception is printed and ends the turn as `error` with "harness bug: ...".

Turn statuses are `running`, `done`, `stopped` and `error`. The CLI exits with 0 for `done`, 2 for `stopped` and 1 for
anything else.

### Working memory

A turn's working memory is one JSON object, made by `new_state()`. `Live` wraps it with a version counter and a
condition variable, so the page can follow along while modules write to it.

| Field | Written by | What it holds |
|---|---|---|
| `id` | start | `YYYYMMDD-HHMMSS-xxxxxx` |
| `conversation`, `history` | start | chat id, and the earlier turns from `conversations/<id>.json` |
| `message`, `today` | start | the person's message; today's date, like "Tuesday, September 15, 2026" |
| `status`, `note`, `seconds`, `started` | finish | outcome and timing |
| `switches` | think | `{modules: [...], tools: [...]}` that were on for this turn |
| `route` | router | `for_me, intent, intent_p, tool, tool_p, ask_first, ask_p, ms, tools_seen` |
| `goal`, `observations` | perception | observations get `id` (`o1`...) and `relevance` |
| `attention` | attention | `{for_me, why}` |
| `recall`, `memories` | recall | `{queries, found, note?}`; kept memories with `relevance` |
| `problems` | math_language, solve_problems | `[{id, expression, unit, about, answer, by, error, worked, math?}]` |
| `cycles` | planner and later | one entry per cycle, below |
| `reply` | language or plain reply | the text the person reads |
| `remembered` | remember | `[{id, text}]` saved this turn |
| `trace` | every model call | `{module, name, part, model, status, started, seconds, retries, error}` |

A cycle entry holds:

- `n` and `goal`;
- `candidates`: `{index, type, summary, tool, action, target, content}`;
- `dropped` and `changed`: what the guards did;
- `predictions` and `critique`;
- `decision`: `{choice, why, by}`;
- `action`: `{tool, action, target, ok, result, status, started, seconds, math?}`;
- `skipped`: steps that didn't run, and why.

### Conversations

- **Storage:** `conversations/<id>.json` is `{id, started, updated, turns: [...]}`. Each turn keeps
  `{run, message, reply, status, note, goal, asked, seconds, at}`.
- **`asked`** is true when the turn's reply came from an `ask` decision. The ask rule reads it.
- **What modules see:** `conversation_view()` gives the last 6 turns (`HISTORY_TURNS`) as
  `{person: message[:500], assistant: (reply or note)[:600]}`.
- **Ids** must match `^\d{8}-\d{6}-[0-9a-f]{6}$`, so a path can't be smuggled in. An unknown id is refused.
- **Shell pipes** don't use conversations: each `./m` run starts from a fresh state.

---

## 2. Modules

The order in `harness.json` is `Harness.order`, the order of cards around the page's circle, and the order of
`/api/state` → `modules`.

| # | id | Name | `kind` | Default model | Temp / tokens | Contract | Color |
|---|---|---|---|---|---|---|---|
| 1 | `router` | Router | `classifier` | `smmol/smROUTER_01` | n/a | none (trusted model output) | grey |
| 2 | `perception` | Perception | none | `pc/ministral-8b` | 0 / 500 | `perception` | red |
| 3 | `attention` | Attention | none | `pc/ministral-8b` | 0 / 250 | `attention` | orange |
| 4 | `recall` | Recall | none | `pc/ministral-8b` | 0 / 200 | `recall_query`, `recall_rank` | yellow |
| 5 | `planner` | Planner | none | `pc/ministral-8b` | 0.2 / 600 | `planner` | green |
| 6 | `predictor` | Predictor | none | `pc/ministral-8b` | 0 / 400 | `predictor` | teal |
| 7 | `critic` | Critic | none | `pc/ministral-8b` | 0 / 400 | `critic` | blue |
| 8 | `decision` | Decision | none | `pc/ministral-8b` | 0 / 150 | `decision` | purple |
| 9 | `math_language` | Math Language | `math_language` | `smmol/smMATH_LANGUAGE_001` | n/a | `math_language` (checked in code) | lime |
| 10 | `math` | Math | `math` | `smmol/smMATH01-abacus` | n/a | none | brown |
| 11 | `language` | Language | none | `pc/ministral-8b` | 0.4 / 500, 1800 to make | none (free text) | pink |

- "Default" is what `harness.json` says. A pick made on the page or with `--use` is saved in `choices.json` and wins.
- Temperature and token limits are defaults for the settings in a card's inspector ([section 4](#settings)).
- Prompts that contain `{today}`, `{tools}` or `{tool_rules}` get them filled in on every call (`_ask`).
- Other modules only ever see a candidate through `view()`: `{index, type, summary}`, plus `tool`, `action` and
  `target` for tool candidates. File contents are never shown to them.

In the payloads below, a key marked with * is left out when it's `None`: `drop_none` removes it, so a module never
hears about a part that's switched off.

### Router (`router`, trained)

- **Job:** read the message and the last reply, and say what kind of message it is, which tool it needs, and whether to
  ask first.
- **Call:** `Router(path).route(text, prev, tools)`. `prev` is the last turn's reply. `tools` is `router_tools()`:
  the router's tool names whose harness switch is on.
  - `ROUTER_TOOLS` maps router names to switches: `calculator`, `web_search` and `web_browser` to themselves,
    `file_read` and `file_write` to `file_system`, and `memory_recall` to `memory`.
  - `memory_recall` also needs the Recall module on.
- **Returns:** `{for_me, intent, intent_p, tool, tool_p, ask_first, ask_p, ms}`. Intents are question, task, remember,
  answer, statement and small_talk. Tools are none, calculator, web_search, web_browser, file_read, file_write and
  memory_recall.
- **Effects:**
  - stops small talk (step 1 above);
  - the Planner gets `route: {message_kind, suggested_tool, ask_first, note}`, where `suggested_tool` is the harness
    tool name and `note` says it's a fast first guess to follow unless the observations clearly say otherwise;
  - it can overrule Decision ([section 5](#the-router-only-forces-lookups)).
- The model: [models/smROUTER_01.md](../smModels/smROUTER_01.md).

### Perception (`perception`)

- **Job:** turn the message into structured observations and a goal.
- **Gets:** `{message, today, conversation}`.
- **Gives:** `observations`, up to 4, each with:
  - `kind`: one of event, fact, request, question, feeling, preference, small_talk;
  - `text` (160 characters), `who` (60), `what` (80), `when` (40), `where` (60).

  Plus `goal` (120). Code adds `id` and `relevance: null`.

### Attention (`attention`)

- **Job:** is this for the assistant, and how relevant is each observation?
- **Gets:** `{message, goal, observations: [{kind, text, when, index}], conversation}`.
- **Gives:** `for_me` (boolean), `scores` (up to 4 of `{index ≥ 1, relevance 0..1}`), `why` (160).
- Observations under 0.5 (`FOCUS`) drop out of `focused()`, which is what later modules get. If none pass, the first
  observation is used.

### Recall (`recall`)

- **Query part (`prompt`):**
  - gets `{goal, observations, conversation}`;
  - gives `queries`, up to 3 strings of 60 characters.
- **Rank part (`rank_prompt`, shown as "Recall (rank)" in the trace):**
  - gets `{goal, queries, memories: [{id, text, saved}]}`;
  - gives `ranked`, up to 8 of `{id (12), relevance 0..1}`.

### Planner (`planner`)

- **Job:** propose up to 3 next steps.
- **Gets:** `{goal, observations, memories*, actions*, last_critique*, conversation, route*, problems*}`.
- **Gives:** `goal` (120) and `candidates` (1 to 3), each with:
  - `type`: answer, make, ask or tool;
  - `summary` (160);
  - `tool`: none, calculator, web_search, web_browser or file_system;
  - `action`: none, create, read, update, delete or list;
  - `target` (300) and `content` (2000).
- **The contract shrinks with the switches** (`Harness.contract`):
  - `tool` only lists enabled tools;
  - `type` has no `tool` when no tools are enabled;
  - `action` is only `none` without `file_system`.
- **The prompt shrinks too:** `{tools}` is the menu of enabled tools and `{tool_rules}` is their "how" lines. With no
  tools, `{tool_rules}` says "No tools are available, so propose only answer, make and ask candidates."

### Predictor (`predictor`)

- **Job:** predict what each step leads to.
- **Gets:** `{goal, candidates, known: {observations, memories*, actions*, conversation}}`.
- **Gives:** `predictions`, up to 3 of `{index, outcome (160), helpful 0..1, risk (120)}`.

### Critic (`critic`)

- **Job:** find errors, contradictions, weak assumptions and missing information, plus one question that would clear
  them up.
- **Gets:** `{message, goal, observations, memories*, actions*, conversation, candidates}`. Candidates include their
  prediction when the Predictor is on.
- **Gives:** `issues`, up to 4 of `{kind, text (200), severity}`, and `question` (200).
  - `kind` is one of error, contradiction, assumption, missing.
  - `severity` is low, medium or high.
- **Who uses it:** `assumption` issues go to Language as `assumptions`. `missing` issues go to a make step as
  `open_details`. High severity feeds the ask rule.

### Decision (`decision`)

- **Job:** pick one candidate.
- **Gets:** `{candidates, critique*, actions*, conversation}`.
- **Gives:** `choice` (integer ≥ 1) and `why` (160).

### Math Language (`math_language`, trained)

- **Job:** pull the math problems out of a message.
- **Call:** `Reader(path).read(text)` returns `[{id, expression, unit, about}]`.
- **Checked in code:** the harness wraps the list as `{"problems": [...]}`, validates it against
  `contracts.json` → `math_language`, then normalizes it: up to 5 problems; `id` 4 characters, `expression` 200, `unit`
  40, `about` 60. A mismatch is an error on the trace, and the turn continues without problems.
- **Expressions** use numbers, `+ - * / **`, brackets and `sqrt()`. `p1`, `p2`... stand for earlier answers in the
  same message.
- The model: [models/smMATH_LANGUAGE_001.md](../smModels/smMATH_LANGUAGE_001.md). How answers get filled in:
  [section 6](#math-language-to-answers).

### Math (`math`, trained)

- **Job:** do arithmetic in a trained model's weights, with the calculator as a checker.
- **Runs for:** a chosen `calculator` step when `math_on()`, and every Math Language problem.
- **Two model shapes:** step models and whole-expression models ([section 6](#step-math-models-and-whole-expression-math-models)).
- **Config extras:** `tool_does` and `tool_how` describe arithmetic to the Planner while the calculator tool is off.
- The models: [models/smMATH01.md](../smModels/smMATH01.md) and [models/smMATH001-a.md](../smModels/smMATH001-a.md).

### Language (`language`)

- **Job:** write what the person reads. For a `make` step, write the thing itself.
- **Gets:** `{message, goal, conversation, decision: {type, summary, why}, question_to_ask, observations, memories*,
  tool_results*, problems*, assumptions*, open_details*}`.
  - `question_to_ask` is the Critic's question when the decision is `ask`, else empty.
  - `open_details` is only sent for make steps.
- **Gives:** free text with no contract. An empty reply gets one retry.
- **Two prompts:**
  - `prompt` answers or asks, in under 150 words.
  - `make_prompt` writes the thing, picks sensible defaults, rejects bad input instead of quietly defaulting, and names
    the defaults in one line. Its token limit is `make_max_tokens` (1800).

### Tools and stores (code, not models)

From `harness.json` → `tools`:

| Tool | Default | What the code does |
|---|---|---|
| `calculator` | on | `calc()`: parses with `ast` and walks only numbers and operators, never `eval` (details below) |
| `web_search` | on | `POST browse/search {query: target[:300], count: 5}`, formatted as numbered title, link and snippet lines |
| `web_browser` | on | Needs an `http(s)://` target. `POST browse/fetch {url, max_words: 700}`. The browse server's SSRF guard decides what's reachable |
| `file_system` | on | `FileTool` on `workspace/`: `create`, `read`, `update`, `delete`, `list` (details below) |
| `command_line` | off | `built: false`. `tool_on` is always false and `toggle` refuses it |
| `memory` | on | `kind: "store"`. Not on the Planner's menu. Switches Recall's store and Remember |

- **Calculator details:**
  - `arithmetic_text()` turns `×` into `*`, `÷` into `/`, `^` into `**` and `−` into `-`; removes thousands commas
    (`2,340`); and rewrites `17.5% of 2340` as `(17.5/100)*2340`.
  - Allowed operators: `+ - * / // % **` and unary `+ -`.
  - Allowed functions: `sqrt abs round min max floor ceil log log10 exp`. Names: `pi`, `e`.
  - Expressions longer than 200 characters are refused. So are powers with an exponent over 100 or a base over 10⁶.
  - Whole floats under 10¹⁵ print as integers; other floats use `%.10g`.
- **File tool details:**
  - Names must match `^[A-Za-z0-9][A-Za-z0-9 ._-]{0,80}$` and contain no `..`. Symlinks are refused, and the real path
    must stay inside the folder.
  - `create` fails if the file exists and `update` fails if it doesn't.
  - Files are capped at 262,144 bytes.
  - `delete` moves the file to `.trash/<timestamp>-<name>`; `list` hides dotfiles.
- **Long-term memory (`MemoryStore`):**
  - JSONL, one memory per line: `{id: "m" + 6 hex, text (500 characters), kind, source, created}`.
  - A duplicate (same text, ignoring case) isn't added. The newest 2000 are kept.
  - Search is BM25 with k1 1.2 and b 0.75. Words are lowercased, stopwords dropped, and a trailing `s` is stripped from
    words over 3 letters that don't end in `ss`.

---

## 3. Contracts

`Harness._ask()` calls a prompted module:

1. **Build the call.** The system prompt gets `{today}`, `{tools}` and `{tool_rules}` filled in. The user message is
   `Input:\n` plus the payload as JSON. The schema is `contract(name)`, narrowed by switches.
2. **Constrain the reply.** With a schema, the request carries
   `response_format: {"type": "json_object", "schema": ...}`, so llama.cpp constrains the output.
3. **Handle servers without schemas.** If a provider answers HTTP 400 to that, the harness adds it to `_no_schema`,
   retries without the schema, and from then on only validates afterwards.
4. **Validate.** `parse_json()` drops `<think>...</think>`, then takes the text from the first `{` to the last `}`.
   `validate()` checks a small JSON Schema subset: type, required, properties, items, enum, minItems.
5. **Normalize.** `normalize()` clamps numbers to minimum and maximum, cuts arrays to `maxItems` and strings to
   `maxLength`, and drops unknown keys.
6. **Retry once.** A reply that doesn't fit gets one more try, with the problems listed ("Your last reply didn't fit
   (...). Reply again with JSON that fits the contract."). If the reply was cut off (`finish_reason: "length"`), the
   token limit doubles for the retry.
7. **Give up.** A second miss raises `ContractError` "<Module> broke its contract twice: ...", which ends the turn.

Language has no schema, so its only "contract" is a non-empty reply.

Why the length caps are short: in the first nested test, Ministral 8B filled every field with long speculative text in
markdown and was cut off at 500 tokens after 32 s. Short `maxLength` limits plus one example per prompt fixed it
([harness README](../../../paratroop_harness_02/README.md#contracts)).

---

## 4. Switches and settings

### Switches

Every module and every built tool can be switched off with the checkbox on its card — modules around working
memory, tools in a grid underneath — or with `POST /api/toggle`.

- **Storage:** `data_dir/switches.json` holds `{"modules": {id: bool}, "tools": {name: bool}}`. Anything not listed
  uses its `enabled` value in `harness.json` (default true).
- **`toggle()` refuses** unknown modules, unknown tools, unbuilt tools (`command_line`), and any kind other than
  `module` or `tool`.
- **The page** won't toggle while a reply is still thinking ("Wait for this reply first.").
- **Tests:** `test_module()` refuses a switched-off module, so a 0 score never sticks to its model.

**Off means invisible.** The rest of the harness behaves as if the part doesn't exist:

| Off | What changes (verified in code and tests) |
|---|---|
| Router | No route, no `route` hint for the Planner, no overrules |
| Perception | The raw message becomes one `request` observation |
| Attention | Nothing stops the turn as "not for me"; relevance stays `null` |
| Recall | No recall; `memories` is left out of every payload |
| Math Language | Never reads; `problems` is left out of every payload |
| Planner | One candidate: "Reply directly to the message" |
| Predictor | No predictions; candidates go to Critic and Decision without them |
| Critic | No critique; `last_critique`, `critique`, `assumptions` and `open_details` are left out. The ask rule's "nothing high-severity" branch can't fire |
| Decision | Most helpful prediction, or the first candidate that doesn't ask |
| Math | Calculator steps go to `calc()` |
| Language | `_plain_reply` |
| `memory` tool | No recall and nothing saved |
| All Planner tools | No `tool` type in the contract; `actions` left out; `{tool_rules}` says no tools |
| `calculator`, Math on | Arithmetic stays on the Planner's menu, described by Math's `tool_does` ("whole-number arithmetic: + - * only, decimals allowed"). `run_tool` refuses the calculator; Math does the steps unchecked |
| `file_system` | `action` can only be `none` |

With every module off, "hello there" still gets a reply (the message itself) and makes no model calls
(`test_everything_off_still_answers`).

### Settings

Each module has knobs, shown in the card's inspector popover and stored in `data_dir/settings.json` as `{module: {key: value}}`.
They're defined in `SETTINGS`, keyed by the module's `kind`:

| Applies to | Key | Label on the page | Type | Range, step | Default |
|---|---|---|---|---|---|
| Prompted modules (no `kind`) | `temperature` | Temperature | number | 0.0–1.5, 0.1 | the module's `temperature` in `harness.json` |
| Prompted modules | `max_tokens` | Token limit | int | 50–4000, 50 | the module's `max_tokens` |
| Modules with a `make_prompt` (Language) | `make_max_tokens` | Token limit when making something | int | 200–4000, 100 | `make_max_tokens` (1800) |
| `classifier` (Router) | `stop_small_talk_at` | Stop on small talk when at least this sure | number | 0.5–1.0, 0.01 | 0.9 |
| `classifier` | `force_tool_at` | Force a lookup when at least this sure | number | 0.5–1.0, 0.01 | 0.7 |
| `classifier` | `skip_question_below` | Skip a question when the chance of needing one is below | number | 0.0–0.5, 0.01 | 0.2 |
| `math` | `check` | The calculator checks its answers (when the calculator is on) | bool | | true |
| `math_language` | `only_with_numbers` | Only read messages that have numbers in them | bool | | true |

- **`set_setting()` validates.** A bool must be a JSON boolean; a number must be in range. An error names the label and
  range, and the page shows it as a toast.
- **Defaults aren't stored.** A value set back to its default is removed from the file.
- **Where each one is read:**
  - `_ask` uses `temperature`, plus `max_tokens` for `prompt` or `make_max_tokens` for `make_prompt`;
  - `think` uses `stop_small_talk_at` and `only_with_numbers`;
  - `_router_overrule` uses `force_tool_at` and `skip_question_below`;
  - `work_math` uses `check`.
- **The page:** `/api/state` sends `settings_for(mid)` per module as `{key, label, type, min, max, step, default,
  value}`. The settings block is empty only when a module has no settings; all 11 modules have some.

---

## 5. Code guards

These rules live in code so they don't depend on an 8B model following its prompt. Each is covered by a test in
`tests/test_harness.py`.

### Tool-candidate guards

`Harness.plan()` cleans the Planner's candidates:

| Candidate | What happens | Shown on the page as |
|---|---|---|
| `type: tool` with `tool: none` | becomes a `make` step | "changed by guards: ...: named no tool, so it became a make step" |
| Any non-tool | `tool`, `action`, `target` and `content` are cleared | |
| Tool that isn't enabled | dropped | "dropped by guards: ...: tool 'x' isn't available" |
| `file_system` with `action: none` | dropped | "...: file_system needs an action" |
| No target (except `file_system list`) | dropped | "...: no target" |
| Target with a placeholder like `[...]` or `<...>` | dropped | "...: placeholder in '...'" |
| The same tool, action and target already done this turn | dropped | "...: already done" |
| Anything except `file_system create/update` | `content` cleared | |
| Nothing left | one `answer`: "Reply with what's known so far" | |

### Make steps

**The problem.** A chat asked for a JavaScript function that takes `firstNumber`, `operator` and `secondNumber`.
Three turns later it still hadn't written it:

- the Critic called operator types and division by zero high-severity;
- Decision asked, then asked again;
- the Planner's "Draft the function implementation" was typed as a tool with no tool, and a guard threw it away.

The full story is in the [harness README](../../../paratroop_harness_02/README.md#make-steps).

**In code now:**

- **A fourth candidate type, `make`.** It's in the `planner` contract's `type` enum.
- **The no-tool guard** turns a tool step with no tool into `make` instead of dropping it.
- **`speak()`** uses `make_prompt` and `make_max_tokens` for make steps, and sends the Critic's `missing` issues as
  `open_details`. The trace entry's `part` is `make`.
- **`instead_of_asking()`**, used by both question rules, prefers a `make` or `answer` candidate. It takes another
  tool step only if that's all there is.

**In prompts** (`harness.json`):

- the Planner includes a make candidate for write-or-build requests;
- the Critic doesn't call an open detail high when a sensible default exists, and it treats "yes", "all" or "sounds
  good" as an answer;
- Decision prefers make.

**Live check, 2026-09-15** (from the README):

- the same request got code on the first turn, in 82 s;
- still not fixed: the Critic still calls "which operators" high severity;
- for "Save a shopping list", the file save came third twice and wasn't offered once in 3 runs; before make existed,
  it came first.

### The ask rule

`_ask_rule(s, entry, decision)`. A question costs the person a turn.

- **Only applies** when the chosen candidate is `ask` and `instead_of_asking()` finds something else to pick.
- **Overrules the question when either:**
  - the previous turn asked (`history[-1].asked`). Why: "it asked a question last turn, so it doesn't ask another";
  - there is a critique and none of its issues is high severity. Why: "the Critic found nothing high-severity, so
    there's nothing worth asking".
- **Otherwise the question stands.** So asking needs a high-severity issue from the Critic and must not follow another
  question.
- **Two gaps:**
  - with the Critic off there's no critique, so only the "never twice in a row" part applies;
  - when every candidate is a question, nothing can replace it.

### The bare-arithmetic rule

`_arithmetic_rule(live, entry)`. A message that's nothing but arithmetic goes to arithmetic, whatever the Router or
Decision made of it.

- **What counts** (`bare_arithmetic()`):
  - it strips a leading "what's", "what is", "calc", "calculate" or "compute", units stuck to numbers (`ft`, `kg`,
    `hours`, `dollars`...), `$` signs, and trailing `= ? ! .`;
  - then it finds the longest expression the calculator can read (`find_expression()`, which reads `12x4` as `12*4`);
  - it matches only when that expression is the whole remaining text, ignoring spaces.
- **Examples from the tests:**

  | Message | Result |
  |---|---|
  | `2+2` | `2+2` |
  | `what's 48,213 + 9,977?` | `48,213 + 9,977` |
  | `12 x 7 =` | `12*7` |
  | `12ft x 3ft` | `12*3` |
  | `I have 2 kids and 3 dogs` | not bare arithmetic |
  | `hey what's 2+2 lol` | not bare arithmetic (the router's call) |

- **Fires when** there's a bare expression, `calculator` is in `enabled_tools()` (calculator on, or Math on), no
  calculator action has run this turn, and Math Language hasn't already answered a problem (`_solved`).
- **What it does:** picks the existing calculator candidate, or adds `{type: tool, summary: "Work out <expr>",
  tool: calculator, target: <expr>}`. The decision is `by: "rule"`, why "the message is just arithmetic".
- **Where it sits:** after the ask rule, and in place of the router overrule. The router's small-talk stop also skips
  bare arithmetic.
- **Why it exists:** smROUTER_01 labels "2+2" small talk, and with no Planner nothing else would pick a calculator
  step. With only Math switched on, "12 + 30 =" still gets "42" (`test_just_arithmetic_works_with_only_math_switched_on`).

### Live facts get looked up

`_lookup_rule(live, entry)`. Runs after `_router_overrule`, and only when the decision so far isn't already a tool,
so the trained router keeps first say.

- **When it fires:** `needs_lookup(message)` matches `LOOKUP`, `web_search` is enabled, and no `web_search` or
  `web_browser` has run this turn.
- **`LOOKUP`** is a deliberately narrow list of things no model can have in its weights: weather, forecast,
  raining, snowing, humidity, news, headlines, stock/share price, exchange rate, gas prices, who won, final score,
  flight status, traffic, open now, opening hours, latest version/release, release date.
- **What it does:** takes an existing `web_search` or `web_browser` candidate, or appends a new `web_search` one
  whose target is the message itself (first 300 characters).
- **Decision:** `by: "rule"`, why "no model can know this on its own, so look it up".
- **Why it exists:** with the Planner off there is one candidate, "reply directly", so nothing else proposes a
  search. smROUTER_01 also reads "Whats the weather in 60601" as `calculator` at 99%, because of the bare number,
  and the forced calculator step then finds no expression and does nothing.

### A list of links isn't an answer

`_open_page_rule(live, entry)`. Runs in the same slot, right after `_lookup_rule`.

- **When it fires:** `web_browser` is enabled and hasn't run this turn, a `web_search` succeeded this turn, and
  `needs_lookup(message)` still matches — so ordinary searches aren't hijacked.
- **What it does:** appends a `web_browser` candidate for `best_link(search result)`.
- **`best_link`** takes the first address in the result text that isn't on `CLIENT_SIDE` (weather.com,
  accuweather.com, tripadvisor, instagram, facebook, twitter/x). Those pages are drawn in the browser, so the text
  fetch returns menus and adverts. **When every hit is one of them it returns `""`** and no page is opened, so the
  reply is written from the snippets. It used to fall back to the first link; on 2026-09-15 a search for
  "Whats the weather in 60601" came back as five weather.com and accuweather URLs and the fallback fetched a page
  of adverts — exactly what the list exists to prevent.
- **Decision:** `by: "rule"`, why "a list of links isn't an answer, so read the top page".
- **Why it exists:** a search comes back as titles, links and snippets. Handed only that, Qwen3.5-4B answered
  "72°F and a 20% chance of rain" for 60601 — none of which appeared anywhere in the results. With the page text
  it answers from the National Weather Service figures the page carries, or says the numbers weren't there.

### The Language module can hand the turn back

`_hand_back(live, limit)`, called by `think()` right after `speak()`.

- **How it asks:** the Language module's prompt lists `{tools}` — whatever is switched on — and tells it that only
  `tool_results` actually happened and nothing runs after it. When it needs something it wasn't given it replies
  with one line and nothing else: `NEED: <tool> <query or https address>`, matched by the `NEED` regex.
- **Honoured** when the tool is in `FORCEABLE`, is enabled, hasn't run this turn, and the cycle limit isn't reached.
  A cycle is appended whose single candidate is that tool, `decision.by` is `"language"`, `act()` runs it, and
  `speak()` writes the reply again with the result.
- **Refused** otherwise: `speak()` runs again with a `note` saying why ("the web_search tool isn't available, so say
  what you can without it and what you'd need").
- **Never** for a `make` step, because made text can contain anything, including that line. `drop_need()` strips a
  `NEED:` line from the final reply either way, so it never reaches the person.
- **Once a turn.** A second `NEED:` after the hand-back is stripped, not honoured.

### The router only forces lookups

`_router_overrule(live, entry, decision)`. Where the small router is confident, it wins, but only for lookups.

**Forcing a tool.** All of these must hold:

- the Router is on and produced a route;
- its tool maps to a harness tool in `FORCEABLE`: `calculator`, `web_search`, `web_browser`;
- that tool is enabled;
- `tool_p ≥ force_tool_at` (0.7);
- the tool hasn't run this turn, and the current pick isn't already that tool;
- for `calculator`: Math Language hasn't already answered a problem.

Then the harness picks, in order:

1. an existing candidate for that tool;
2. for `web_search`: a new candidate searching the message itself (first 300 characters);
3. for `calculator`: a new candidate for `find_expression(message)`, if one is found;
4. otherwise nothing is forced. `web_browser` has no target to invent.

The decision is `by: "router"`, why "the router is N% sure this needs X".

**Skipping a question.** If the pick is `ask`, `ask_first` is false, and `ask_p < skip_question_below` (0.2), the
harness takes `instead_of_asking()`. Why: "the router is N% sure nothing needs asking".

**Never a write.** `file_read`, `file_write` and `memory_recall` are never forced. smROUTER_01 reads "write a function in
javascript" as `file_write` at 98%, because its training data only uses "write" for files
(`test_router_never_forces_a_file_write`).

### The memory filter

`remember()` with `worth_keeping()`. Long-term memory is for the person and their world, not the details of one
request. An observation is saved only if:

1. its `kind` is `fact`, `preference` or `event`;
2. its text doesn't match `REQUEST_ECHO`: it doesn't start with "User wants / specifies / specified / asks / asked /
   requests / requested / previously", and it doesn't contain "the assistant";
3. its text mentions "user", or its `who` names someone other than the user;
4. its relevance is `null` (Attention off) or at least 0.5.

A `when` that isn't already in the text is added as " (when: ...)". The store skips duplicates.

From `test_details_of_a_request_arent_kept_as_memories`: "User's dentist is Dr. Lee" is kept. "User specifies
programming language", "Function requires three arguments" and "User prefers the assistant's previous suggestion" are
not.

### Other limits

- **Cycle limit:** 3 tool rounds (`max_cycles`), then a forced answer.
- **Server:** localhost only, same-origin JSON POSTs, 64 KB bodies, 8000-character messages. See [api.md](api.md).
- **Not guards, but not built:** `command_line` can't be switched on.

---

## 6. Trained-model plumbing

### `LOCAL_CODE`

```python
LOCAL_CODE = {"classifier": "route.py:Router", "math": "solve.py:Solver",
              "math_language": "read.py:Reader"}
```

A module `kind` maps to `file.py:Class` inside the model's project folder. A module whose kind is in `LOCAL_CODE`:

- only offers `local` providers (`options()` sets `want = "local"`);
- only offers models whose provider entry has the same kind, so the Router's dropdown only lists routers;
- can't be tested until its checkpoint exists ("<Name>'s model isn't trained yet, so there's nothing to test").

### Provider `models` entries

```json
"smmol": {"kind": "local", "label": "Mac · trained in SMMOL", "models": {
  "smROUTER_01": {"path": "../smROUTER_01/out/router.pt", "kind": "classifier"},
  "smMATH_LANGUAGE_001": {"path": "../smMATH_LANGUAGE_001/out/reader.pt", "kind": "math_language"},
  "smMATH001-a": {"path": "../smMATH001-a/out/math.pt", "kind": "math"},
  "smMATH01-abacus": {"path": "../smMATH01/out/abacus.pt", "kind": "math"},
  "smMATH01-reversed": {"path": "../smMATH01/out/reversed.pt", "kind": "math"},
  "smMATH01-plain": {"path": "../smMATH01/out/plain.pt", "kind": "math"}
}}
```

`local_entry(p, model)` resolves an entry:

- a plain string is a path, and its kind is `classifier` (the old router form);
- `kind` defaults to `classifier`;
- `code` defaults to `LOCAL_CODE[kind]`, but an entry may set its own `code`;
- `path` is resolved relative to the harness folder.

### Offering and choosing

- **Status:** `provider_status()` lists each local model whose checkpoint file exists. With none, the error is "not
  trained yet".
- **Options:** `options(mid)` also lists untrained models of the right kind as `{up: false, note: "not trained yet"}`.
  The page shows them disabled.
- **Choosing:** `choose()` only accepts options that are up, so an untrained model can't be picked. The default in
  `harness.json` can still name one.
- **Readiness:** `local_ready(mid)` means the module's chosen provider is local and its checkpoint exists.
  - `router_ready()` gates the Router.
  - `math_on()` is module on plus ready.
  - `math_language_on()` is module on plus ready.
  - An untrained module is simply skipped.

### Loading: `load_local(checkpoint, code)`

- **Where code must live:** the project folder is `dirname(dirname(checkpoint))`. The checkpoint has to sit one folder
  below the project root (`<project>/out/<file>.pt`), and the loader file at the project root.
- **Name clashes:** projects reuse file names like `model.py`. So before importing, it takes any already-imported
  modules with the same names as the project's `.py` files out of `sys.modules`. Then it puts the project folder first
  on `sys.path` and imports the script as `smmol_<script>`.
- **Constructor:** it returns `Class(checkpoint)`. The loader gets only the checkpoint path.
- **Cleanup:** afterwards it removes the folder from `sys.path`, drops the project's modules from `sys.modules`, and puts
  back the ones it took out, so smROUTER_01's `model.py` and smMATH01's `model.py` never collide.
- **Errors:** any exception becomes `HarnessError` "couldn't load <path> (<error>)".
- **CPU only:** every loader in the workspace loads with `map_location="cpu"` and never moves to MPS, so the harness
  doesn't compete with training for the Mac's GPU.

### The cache

`Harness.local(choice)` caches loaded models in `_local_models`, keyed by `(checkpoint path, modification time)`.

- When the file's modification time changes, the old entry for that path is dropped and the model is loaded again.
  Retraining swaps the model in without restarting the server.
- **Watch out:** training scripts save their checkpoint *during* training:
  - smMATH01 every 500 steps;
  - smMATH_LANGUAGE_001 and smMATH001-a every 1000 steps.

  A page running during training uses the half-trained model. On 2026-09-15, Math Language was picked up from its step
  1000 checkpoint, which got 45% of hand-written test messages fully right. smMATH001-a's `out/math.pt` could be
  picked from its step 1000 save at 17:40, while it trained. smROUTER_01 is the exception: its
  `train.py` saves `router.pt` once, at the end. Switch a module off while its model trains.
- **Tests:** `Rig.trained()` makes an empty checkpoint file and seeds `_local_models` with a stub under that file's
  `(path, mtime)`, so nothing real is loaded.

### Math Language to answers

1. **Read.** `read_math()` stores `problems`, each with `answer: null, by: "", error: ""`.
2. **Solve.** `solve_problems()` works through them in order:
   - `fill_answers()` swaps `p1`, `p2`... for earlier answers in brackets (`p1/27` becomes `(60)/27`). Using an answer
     that isn't there raises "... uses p2 before it's worked out";
   - if `math_on()`, `work_math()` does it and `by` is `"Math"`, with its step info in `math`;
   - else if the calculator tool is on, `calc()` does it and `by` is `"calculator"`;
   - else the error is "Math and the calculator are both off".
   - One problem's error doesn't stop the others. The expression actually worked is saved as `worked`.
3. **Share.** The Planner and Language get `problems` from `_problems()`: `id, expression, unit, about, answer, error`,
   with empty fields left out. They only get it while Math Language is on.
4. **Don't redo work.** Once any problem has an answer (`_solved`), the arithmetic rule and the router's calculator
   force stand down. The Planner still runs and can still choose a tool.
5. **No Language:** `_plain_reply` writes one line per answered problem.

From `test_problems_are_read_worked_out_and_answered_with_their_units`, with only Math Language on and the calculator
doing the work:

```text
Volume of a 3ft x 4ft x 5ft box, and how many cubic yards is that?
p1  3*4*5  cubic feet   = 60           by calculator
p2  p1/27  cubic yards  = 2.222222222  by calculator, worked as (60)/27
reply: "3*4*5 = 60 cubic feet\np1/27 = 2.222222222 cubic yards"
```

### Step math models and whole-expression math models

`work_math(live, expression, check=None)` is shared by calculator steps and problems:

- `check` is "calculator tool on" AND the `math.check` setting. `test_module("math")` forces `check=True`.
- `fallback` is "calculator on" OR `check`. It decides whether the calculator may do what the model can't.
- If the model fails to load: with fallback, the error goes in `info.error` and every step goes to the calculator;
  without fallback, the turn's action fails.
- A model with `takes = "expression"` goes to `_whole_math()`. Any other model is a step model and goes through
  `calc(expression, step)`.

**Step models** (smMATH01: `Solver.fits(a, op, b)`, `Solver.answer(a, op, b)` → digit string or `None`):

- **One step at a time.** `calc()` walks the expression and hands every operation to `step()`.
- **Only `+ - *` reach the model.** Division, powers, functions and constants get "the Math module only does +, - and
  \*".
- **`whole_numbers()` shapes each step:**
  - a negative operand gets "negative numbers go to the calculator";
  - decimals become whole numbers (12.50 + 7.25 goes in as 1250 + 725, two places back; for `*` the places add up);
  - a subtraction that would go below zero is flipped, and the answer negated.
- **Too big:** `solver.fits()` false gets "too many digits for the model".
- **Checking:** the model's digits are shifted back and compared with the exact `Decimal` answer.
  - Right, or not checking: `used: "model"`.
  - Wrong or no number, with fallback: `used: "calculator"`, `misses += 1`, why "the model wrote X".
  - Without fallback: error.
- **What gets saved:** each step records `{step, asked, said, used, value, why?}`.

From `test_math_works_each_step_and_the_calculator_checks_it`, with a stub that gets every `*` wrong by one:
`(48213 + 9977) * 2 - 12.5` asks the model `48213+9977`, `58190*2` and `1163800-125`. The steps are used as model,
calculator, model; there's 1 miss, and the result is `116367.5`.

**Whole-expression models** (smMATH001-a: `takes = "expression"`, `Solver.fits(expression)`,
`Solver.solve(expression)` → `{answer, work, raw}`):

- **Shape the expression.** `plain_arithmetic()` rewrites it using only non-negative numbers, `+ - * /` and brackets,
  or returns `None` for anything else (`sqrt`, `**`, negative numbers). Divisions move last where the value is the same,
  so a model working to 2 decimal places loses nothing: `17.5% of 2,340` becomes `17.5*2340/100`.
- **Can't take it:** not plain, or `fits()` false, gets "the model only takes numbers, + - * / and brackets" or "the
  model can't take that expression". The calculator does it with fallback; without, it's an error.
- **Check it.** The exact answer is a `Fraction`. The model's `answer` is accepted if it parses and either checking is
  off or it's within 1/100 of exact, because the model works to 2 decimal places. Its `work` lines are saved in
  `info.work` and shown on the page.
- **Miss:** with fallback, the value is the exact answer written to at most 10 decimal places (`exact_text`), counted as
  a miss.
- **One consequence:** smMATH001-a cuts division to 2 places rather than rounding. A within-1/100 answer is shown as
  the model wrote it, so 1850/3 would show as the model's 2-place figure, not 616.6666667.

From `test_a_whole_expression_math_model_does_the_problems_and_the_calculator_checks_it`, with a stub that gets
anything divided by 27 wrong by 1: p1 `322234*21323*212231` = `1458238263363442` by the model; p2 is caught and becomes
`54008824569016.3703703704` from the calculator, with 1 miss.

### Calculator checking and fallback

| Calculator tool | `math.check` | Steps the model can do | Steps it can't |
|---|---|---|---|
| on | on (default) | checked; a wrong one is replaced by the exact answer and counted as a miss | calculator |
| on | off | used unchecked | calculator |
| off | ignored | used unchecked | the step fails: "can't do X (why), and the calculator is off" |

- **The Test button** always checks, so a Math score is the model's own. A check passes only when every step was
  `used: "model"` and the result matches.
- **The page** shows "N of M steps by <model>, K caught by the calculator", or "unchecked (the calculator is off)".

---

## 7. Known gaps

Verified in code, or recorded in the project READMEs.

- ~~**Router output isn't shown on the page.**~~ Fixed by the 2026-09-15 redesign: the timeline renders `route`
  (intent, tool, both confidences, ms) as its own step, and `switches` as the modules that were on for that run.
- **Math Language and Math aren't pipe stages.** `STAGES` doesn't include them, so `./m math` fails. `--test math` and
  `--test math_language` work.
- **A router error ends the turn,** while a Math Language error is only noted.
- **Half-trained checkpoints are used live** ([the cache](#the-cache)).
- **Over-asking.** The Critic still raises "high" issues for details with sensible defaults, and in the 2026-09-15
  chat check it asked another question after "I'm the investor".
- **The weather test fails in both harnesses.** In `eval.py`, v02 asked instead of searching.
- **Not built** (harness README, "Not done yet"):
  - `command_line` with an approval button;
  - vector search in long-term memory (needs an embedding model download);
  - right-sizing: every prompted module still runs on Ministral 8B;
  - trained Predictor and Critic;
  - speed: one message is 8 or more model calls on one GPU.

  See [the product whats-next](../../product/whats-next.md) for the ideas that have been discussed.
