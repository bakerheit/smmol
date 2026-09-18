# paratroop_harness_01

A harness that runs several small local models, each doing one brain function, joined by
pipes. The design notes are in [docs/engineering/paratroop_harness/architecture.md](../docs/engineering/paratroop_harness/architecture.md).

```text
attention | context | route | respond
```

A message drops through the stages. Route is the hub in the middle: it decides which
`@process` and `@tool` calls to make, and they run right there. Everything each function
produces lands on a shared whiteboard that the later functions read.

## The functions

| Function | Kind | Answers | Default model |
|---|---|---|---|
| Attention | stage | Is this for me? (YES passes it on, NO stops the pipe) | `pc/ministral-8b` |
| Context | stage | What is the context? Tokens like `[User][meeting][Friday]`, plus WHEN and INTENT | `pc/ministral-8b` |
| Route | the hub | Which calls to make | `pc/ministral-8b` |
| Prediction | `@process:prediction` | What response is most likely expected? | `pc/ministral-8b` |
| Planning | `@process:planning` | Numbered steps with dates | `pc/ministral-8b` |
| Vision | `@process:vision` | What's in the attached image? | none yet |
| Respond | stage | The reply the person reads | `pc/ministral-8b` |
| Look it up | `@tool:lookup` | Web facts, through the PC's browse server | `browse/browse` |
| Draw | `@tool:draw` | A picture, through the PC's Stable Diffusion | `sd/sd1.5` |

## Run it

The page (only answers on the MacBook Pro M5 itself):

```bash
python3 server.py
```

Then open http://127.0.0.1:8770. Pick models in the dropdowns, press **Test** to score a
function, send a message, and watch each step light up.

From the terminal:

```bash
python3 harness.py "I have a meeting this Friday to discuss an investment opportunity for our business"
```

```bash
python3 harness.py --pipe "context | planning" "Sam has a dentist appointment next Tuesday at 3pm"
```

Shell pipes. Each `./pt` stage is its own process, and text flows through stdout:

```bash
./pt lookup "tallest building in Chicago" | ./pt respond "say it in one sentence"
```

Other commands:

```bash
python3 harness.py --list
python3 harness.py --use planning=pc/gemma-4-12b
python3 harness.py --test route
```

## How pipes work

- Each stage's output becomes the next stage's input. Chat functions also read the
  whiteboard, so nothing is lost at a handoff.
- Attention saying NO stops the pipe.
- Route runs its calls right after itself, in order. A call that fails is marked as an error,
  and the reply still gets written.
- A call to Vision with no image, or to a function with no model picked, shows as skipped.
  A call the router makes up (like `@tool:calendar`) shows as "no such call".
- Shell pipes pass text only. The whiteboard stays inside one process.

## Swapping models

- Each dropdown lists what the servers offer right now: the PC gateway's models, the models
  UnlimitedStudio has on the MacBook Pro M5, the browse server, and Stable Diffusion. The choice is saved
  in `choices.json`.
- **Test** (or `--test`) runs that function's cases from `checks.json` and saves the score
  for that exact model in `scores.json`, so you can compare models by swapping and testing.
- **The PC runs one LLM at a time.** Giving functions different PC models makes the gateway
  swap between steps: about 5 s to load `qwen3.5-9b`, about 11 s for `gemma-4-12b` and about 50 s
  for `qwen3-coder-30b`. The family portal shares that gateway too.
- The Mac list includes `Qwen3-Coder-30B-A3B-Instruct-IQ4_XS` (16 GB). It won't fit next to
  everything else on the 16 GB Mac, so don't pick it.

## Guards in code

In the first live runs, Ministral 8B as the router ignored the rules in its prompt. It called
Vision with no image, searched for "current market trends for [user's industry]", and asked
Stable Diffusion for an infographic. So the code now:

- only offers `@process:vision` when an image is attached;
- drops tool calls with a placeholder in the argument (`[...]` or `<...>`);
- only lets a tool run when its `only_if` pattern matches the message. Draw requires words like
  draw, picture, sketch or photo.

Dropped calls are listed on the Route step so you can see what the router tried.

## Safety

- The page binds to 127.0.0.1 only. POSTs must be JSON from the same origin, and requests
  with any other Host are refused, which blocks other websites and DNS rebinding.
- Model output goes on the page as text, never as HTML. Image paths are checked against a
  strict pattern.
- `@tool:lookup` sends only a question. The browse server decides what to fetch, behind its
  SSRF guard.
- Nothing here runs code a model wrote.

## Checks so far

2026-09-15, all on `pc/ministral-8b`:

| Function | Score | Average |
|---|---|---|
| route | 3/3 | 3.0 s |
| attention | 4/4 | 0.3 s |
| context | 2/2 | 3.0 s |
| prediction | 1/1 | 4.7 s |
| planning | 1/1 | 24.8 s |
| respond | 2/2 | 1.1 s |
| lookup, draw | not run yet | |

These are small smoke checks, not benchmarks. A pass means the output had the right shape
and keywords.

## Tests

17 tests against fake servers:

```bash
python3 -m unittest discover -s tests
```

## Files

| File | What it is |
|---|---|
| `harness.py` | Core and CLI: functions, pipes, router calls, whiteboard, checks |
| `harness.json` | Providers, the flow, every function's model, prompt and guards |
| `checks.json` | Test cases per function |
| `server.py`, `page.html` | The web page |
| `pt` | Shell-pipe wrapper |
| `choices.json`, `scores.json` | Your model picks and saved scores (made on first use) |
| `runs/<id>/` | Each run's whiteboard (`run.json`) and pictures |

## Not done yet

- **Tools from the sketch aren't built yet:** `@tool:web_search` and `@tool:web_browser` as
  separate calls, `@tool:file_system` and `@tool:command_line`. The last two need a sandbox
  folder and a person's approval first. The plan is in [the harness architecture notes](../docs/engineering/paratroop_harness/architecture.md).
- **No vision model on either machine**, so the Vision slot waits. Adding one means a download
  and listing it under `vision_models` in `harness.json`.
- **Prediction and Planning are prompts on a general model**, not trained predictors.
- **Router calls run one after another.** The PC's GPU does one job at a time anyway.
