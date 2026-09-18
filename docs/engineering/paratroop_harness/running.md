# Running the harnesses

Commands that exist in the code on 2026-09-15. Run each one from its project folder, for example
`cd ~/workspace/SMMOL/paratroop_harness_02`.

Training and testing the models is in [../smModels/running.md](../smModels/running.md).

## Before you run anything

| Command | Uses the PC's RX 580 | Leaves this Mac | Writes into `paratroop_harness_02/` |
|---|---|---|---|
| `python3 server.py`, then chatting | yes, for every prompted module that's on | PC gateway; browse server for web tools | `runs/`, `conversations/`, `memory/`, `workspace/`, and `choices.json`, `switches.json`, `settings.json`, `scores.json` when you use those controls |
| `python3 harness.py "..."` | yes | same | `runs/`, `conversations/`, `memory/`, `workspace/` |
| `./m <stage>` | yes, for prompted stages | same | `memory/` (`remember`), `workspace/` (`act`) |
| `python3 harness.py --test <module>` | for prompted modules; not for `router`, `math`, `math_language` | PC gateway for prompted modules | `scores.json` |
| `python3 harness.py --list` | no, only a status check | `GET /v1/models` and `GET /health` | nothing |
| `python3 eval.py` | yes | yes | `eval-results.json`. v02's state goes to temp folders |
| `python3 -m unittest discover -s tests` | no | no; fake servers on 127.0.0.1 | nothing; temp folders |

- **Page state belongs to the owner.** For experiments, use [a scratch data folder](#a-scratch-data-folder).
- **Dependencies:** paratroop_harness_02 uses only the Python standard library. Picking a trained model makes it import
  that model's project code, which needs PyTorch. The tests use stubs, so they don't need PyTorch.

---

## paratroop_harness_02

### The page

```bash
python3 server.py
```

Then open http://127.0.0.1:8771. It binds to 127.0.0.1 and refuses any other `Host`. To use another port:

```bash
python3 server.py --port 8780
```

What the page calls is in [api.md](api.md).

### The CLI

```bash
python3 harness.py "What's 17.5% of 2,340?"
```

- **While it runs:** each model call prints to stderr as it finishes (name, model, seconds, `ok`, `ERROR` or `retried`).
- **At the end:** it prints `conversation <id> (turn N)` to stderr and the reply to stdout.
- **Exit codes:** 0 when the turn is `done`, 2 when `stopped`, 1 for `error`.

| Flag | What it does |
|---|---|
| `message ...` | What the person said. Text piped on stdin is added after it, with a blank line between |
| `--continue` | Continue the most recent conversation |
| `--conversation ID` | Continue a conversation by its id |
| `--json` | Print all of working memory instead of the reply |
| `--list` | Providers up or DOWN, each module's model, tools on and off, the long-term memory count |
| `--use MODULE=PROVIDER/MODEL` | Swap a module's model. It must be offered and up right now, and it's saved to `choices.json` |
| `--test MODULE` | Run the module's checks from `checks.json` with its current model and save the score to `scores.json`. Exit 0 only if every check passes |
| `--memories` | List long-term memories: id, date, text |
| `--forget ID` | Delete one long-term memory |
| `--module STAGE` | Run one stage with working memory on stdin. `./m` uses this |
| `--show` | Print the reply, or else the note, from working memory on stdin |

Examples from the code and README:

```bash
python3 harness.py --continue "I'm the investor"
python3 harness.py --conversation <conversation id> "yes, the second one"
python3 harness.py --json "2+2"
python3 harness.py --list
python3 harness.py --use critic=pc/gemma-4-12b
python3 harness.py --test planner
python3 harness.py --test math
python3 harness.py --memories
python3 harness.py --forget <memory id>
```

### Shell pipes

`m` runs one stage per process. Working memory (JSON) flows down the pipe, so each stage sees what earlier stages
wrote:

```bash
./m perception "I have a meeting this Friday" | ./m attention | ./m planner | ./m critic | ./m show
```

- **Stages:** `router`, `perception`, `attention`, `recall`, `planner`, `predictor`, `critic`, `decision`, `act`,
  `language`, `remember`, plus `show`, which prints the reply.
- **The first stage** takes the message as its argument. Later stages read working memory from stdin.
- **Order matters:**
  - `attention`, `recall` and `planner` need `perception` first;
  - `predictor`, `critic`, `decision` and `act` need a `planner` cycle;
  - `act` needs a decision that picked a tool;
  - `language` needs a decision;
  - `remember` needs the `memory` tool on.
- **Refusals:** a switched-off module refuses ("<Name> is switched off"), and a stage whose tool is off refuses ("<stage>
  needs a tool that's switched off").
- **Not stages:** `math_language` and `math`. They run in full turns and under `--test`.
- **Nothing is saved as a chat.** Pipes don't start a conversation or write `runs/`.
- **See everything:** leave off `./m show` to print the whole working memory as JSON.

### A scratch data folder

State lives in a data folder that defaults to the harness folder. Give it another one to experiment without touching
the page's switches, picks, settings, scores, memory or chats. This is what `eval.py` does:

```bash
python3 - <<'EOF'
import tempfile
from harness import Harness, Live, new_state
h = Harness(data_dir=tempfile.mkdtemp(prefix="paratroop-scratch-"))
s = h.think(Live(new_state("2+2"))).state
print(s["status"], repr(s["reply"]))
EOF
```

- **Config** (`harness.json`, `contracts.json`, `checks.json`) still comes from the harness folder.
- **A new data folder starts at the defaults:** `harness.json`'s model picks, every switch at its default, no memory.
  `eval.py` copies `choices.json` in so the model picks match the page. You can copy `switches.json` or
  `settings.json` into your scratch folder the same way.
- **Chats:** `new_state(message)` has no conversation. To use chats, call `h.think(h.start(message))`, then
  `h.start(reply_text, conversation_id)`.

### Comparing with v01

```bash
python3 eval.py
python3 eval.py --only percent,memory --skip-v01
```

- **Scenarios** (`eval.json`): `meeting`, `percent`, `memory`, `files`, `small_talk`, `web`.
- **How it runs:** each scenario's messages go to both harnesses in order, as separate conversations. v02 runs in a
  fresh temp folder with a copy of `choices.json`. v01 runs from its own folder.
- **Grading** is keyword-based (`reply_contains`, `uses_web`, `stopped`...). The README calls out one false pass.
- **Output:** a table, plus `eval-results.json`.

### Tests

```bash
python3 -m unittest discover -s tests
```

61 tests in `tests/test_harness.py`:

| Class | Tests | Covers |
|---|---|---|
| `Contracts` | 3 | validate, normalize, parse_json |
| `Tools` | 3 | calculator and expression finding, the file sandbox, the memory store |
| `Thinking` | 25 | the cycle, chats, switches, contracts and retries, guards, the cycle limit, files, web, pipes, checks, make steps, the ask rule, the memory filter |
| `Routing` | 8 | the router: untrained, slot filtering, tool visibility, small talk, skipping questions, forcing search, off, never forcing writes |
| `MathModule` | 7 | step math with calculator checking, calculator off, scoring, Router plus Math, the bare-arithmetic rule |
| `MathLanguage` | 7 | problems, units, `p1` answers, whole-expression math, the number gate, what Planner and Language see, grading |
| `Settings` | 3 | saving and validation, router thresholds, unchecked math |
| `Page` | 5 | the HTTP server: chats, switches, settings, host/origin/content-type refusals, live events and Forget |

- **The fake model server** runs on 127.0.0.1. It works out which module is asking from the system prompt, answers as
  that module, and also plays the browse server.
- **Stubs** stand in for trained models: `StubRouter`, `StubMath`, `StubWholeMath`, `StubReader`.
- **Every test** uses its own temp data folder.

---

## paratroop_harness_01

Run from `~/workspace/SMMOL/paratroop_harness_01`. Details are in its [README](../../../paratroop_harness_01/README.md).

The page:

```bash
python3 server.py
```

Then open http://127.0.0.1:8770. `--port` changes the port.

The CLI:

```bash
python3 harness.py "I have a meeting this Friday to discuss an investment opportunity for our business"
python3 harness.py --pipe "context | planning" "Sam has a dentist appointment next Tuesday at 3pm"
python3 harness.py --list
python3 harness.py --use planning=pc/gemma-4-12b
python3 harness.py --test route
python3 harness.py --json "I have a meeting this Friday"
```

`--image <path>` attaches an image for `@process:vision`. The README notes there's no vision model on either machine
yet, so that step shows as skipped.

Shell pipes. Each `./pt` stage is its own process, and plain text flows through:

```bash
./pt lookup "tallest building in Chicago" | ./pt respond "say it in one sentence"
./pt "attention | context" "I have a meeting this Friday"
```

Tests (17, against fake servers):

```bash
python3 -m unittest discover -s tests
```
