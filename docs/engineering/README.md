# SMMOL engineering docs

For developers and coding agents working in `~/workspace/SMMOL`. Everything here was checked against the code on
2026-09-15. When a project README and its code disagree, these docs follow the code and say so.

Not technical? Start with [the product docs](../product/README.md) instead.

## Layout

```text
docs/engineering/
  README.md                  this index
  paratroop_harness/         the harness: design, commands, HTTP API, adding modules
  smModels/                  one file per trained model, plus training and test commands
  research/results.md        every experiment's numbers, dated
  plans/                     agent-executable plans for work that isn't built yet
```

## paratroop_harness/

| File | What's in it | Read it when |
|---|---|---|
| [architecture.md](paratroop_harness/architecture.md) | What SMMOL is, the v02 turn cycle, working memory, every module's contract, switches, settings, the guards written in code, and how trained models plug in | You're changing the harness or need to know why it did something |
| [running.md](paratroop_harness/running.md) | Exact commands for the v02 page, CLI, shell pipes, tests and `eval.py`, plus harness_01 | You want to run a harness |
| [api.md](paratroop_harness/api.md) | The v02 server's HTTP endpoints, the state and run objects, and what the page shows | You're working on `server.py` or `page.html`, or scripting against the page |
| [adding-a-module.md](paratroop_harness/adding-a-module.md) | Step by step: a new kind of trained-model module, the way Router, Math and Math Language were added | You're wiring a new model into the harness |

## smModels/

| File | Model | Status on 2026-09-15 |
|---|---|---|
| [running.md](smModels/running.md) | Train, test and evaluate every model | |
| [smLLM_01.md](smModels/smLLM_01.md) | Tiny Shakespeare byte GPT | trained |
| [smALLM_01.md](smModels/smALLM_01.md) | Learn-by-doing gadget-world model | trained |
| [smCLM_01.md](smModels/smCLM_01.md) | Ideas reader vs words reader | trained |
| [smROUTER_01.md](smModels/smROUTER_01.md) | Router module | trained (v2), in the harness |
| [smMATH01.md](smModels/smMATH01.md) | Arithmetic in its weights, 3 variants | trained, in the harness |
| [smMATH_LANGUAGE_001.md](smModels/smMATH_LANGUAGE_001.md) | Reads the math problems out of a message | trained (finished 17:37), in the harness |
| [smMATH001-a.md](smModels/smMATH001-a.md) | Works whole expressions out step by step | trained (finished 18:09), in the harness |
| [smTOOLS_COMPUTER_CLI_01.md](smModels/smTOOLS_COMPUTER_CLI_01.md) | Plain request to shell command, 5 platforms | trained (finished 21:45), **not** in the harness; retrain planned |

Each model file covers the same things: purpose, input and output, architecture and size, how its data is made, how to
train and test it, dated results, known failure modes, and how the harness uses it.

## research/

[research/results.md](research/results.md) is one dated log of every experiment's key numbers, the baselines they were
measured against (keyword rules, Ministral 8B, the harness's regex), and the honest takeaway.

## plans/

One file per planned run, written before any code exists, with phases, gates and what to write down. A plan is
not a status page: [../product/whats-next.md](../product/whats-next.md) says what's built.

| File | What it plans | Written |
|---|---|---|
| [smCLM_02.md](plans/smCLM_02.md) | smCLM_01 on real sentences, about 2,500 words and 150 ideas, then memory that searches by meaning in paratroop_harness_02 | 2026-09-17 |
| [when-small-wins.md](plans/when-small-wins.md) | Finds the rule that predicts, before training, whether a small model will beat a prompted 8B — and runs three controls that could downgrade this repo's own headline claims | 2026-09-17 |
| [selection-metric.md](plans/selection-metric.md) | A generated validation split that holds out phrasings rather than rows, so best-checkpoint and early-stopping have a metric that can rank epochs | 2026-09-17 |
| [smRTS_01.md](plans/smRTS_01.md) | Three minimal recurrent state cells trained online with eligibility traces, no backprop through time, tested on associative recall and Tiny Shakespeare against truncated BPTT and smLLM_01. From a public repo's claims | 2026-09-17 |
| [smPERCEPTION_01.md](plans/smPERCEPTION_01.md) | A small trained model for paratroop's Perception module, judged on whether turns come out the same as with the 8B, at a fraction of the time. Checks first whether turning Perception off does just as well. Requested by lamRD | 2026-09-19 |
| [smOEM_01.md](plans/smOEM_01.md) | Organic Emulated Modeling: a model whose only goal is to copy itself and grow. Containment first: an emulated substrate inside a Docker box with no network, attacked by escape probes before any replicator runs | 2026-09-19 |

## The projects

| Folder | What it is | Write-up |
|---|---|---|
| `paratroop_harness_01/` | First harness: a pipe of functions around a shared whiteboard. Page on port 8770 | [README](../../harnesses/paratroop_harness_01/README.md) |
| `paratroop_harness_02/` | The main project: a "brain" of small modules around structured working memory. Page on port 8771, CLI, shell pipes | [README](../../harnesses/paratroop_harness_02/README.md), [architecture.md](paratroop_harness/architecture.md) |
| `smLLM_01/` | Tiny Shakespeare byte GPT | [README](../../models/smLLM_01/README.md) |
| `smALLM_01/` | Learn-by-doing model in a gadget world | [README](../../models/smALLM_01/README.md) |
| `smEFFICIENCY_01/` | Not a model: measures the MacBook Pro M5's real training-throughput ceiling, and which free optimizations (bf16, `torch.compile`, batch size) are real | [README](../../smEFFICIENCY_01/README.md) |
| `smCLM_01/` | Ideas reader vs words reader | [README](../../models/smCLM_01/README.md) |
| `smROUTER_01/` | Trained Router module | [README](../../models/smROUTER_01/README.md) |
| `smMATH01/` | Arithmetic in the model's own weights, three variants | [README](../../models/smMATH01/README.md) |
| `smMATH_LANGUAGE_001/` | Reads the math problems out of a message | [README](../../models/smMATH_LANGUAGE_001/README.md) |
| `smMATH001-a/` | Works whole expressions out step by step | [README](../../models/smMATH001-a/README.md) |
| `smTOOLS_COMPUTER_CLI_01/` | Turns a plain request into the right shell command for Ubuntu, Fedora, Arch, macOS or Windows. Runs nothing | [README](../../models/smTOOLS_COMPUTER_CLI_01/README.md), [smTOOLS_COMPUTER_CLI_01.md](smModels/smTOOLS_COMPUTER_CLI_01.md) |

## Where the project READMEs disagree with the code

Checked on 2026-09-15 at 17:43. The code is right in every case.

- **Links to `docs/harness.md`.** That file no longer exists, and there's no replacement; the v02 README and
  [architecture.md](paratroop_harness/architecture.md) cover the design. Four links point at it:
  - `paratroop_harness_02/README.md`: the intro, and the "v01 vs v02" section;
  - `paratroop_harness_01/README.md`: the intro, and "Not done yet".
- **The "cycle" section of `paratroop_harness_02/README.md`.** Its diagram and module table still list only the
  prompted modules. Router, Math Language and Math run in the cycle too; the README describes them in later sections.
- **`paratroop_harness_02/README.md`, "The Math module".** The "How it works" bullets (one `+`, `-` or `*` at a time,
  decimals shifted) only apply to step models like smMATH01. A whole-expression model like smMATH001-a gets the whole
  expression ([architecture.md](paratroop_harness/architecture.md#step-math-models-and-whole-expression-math-models)).
- **`smCLM_01/README.md`** says 287 nouns. `concepts.json` has 286 words and `out/train.log` says 286.
- **`smLLM_01/README.md`** says `model.py` is about 120 lines. It's 103.

## Ground rules

- **Leave the owner's page state alone.** `paratroop_harness_02/` holds `switches.json`, `choices.json`, `settings.json`,
  `scores.json`, `memory/`, `conversations/` and `runs/`. To experiment, give the harness a scratch data folder the way
  `eval.py` does. See [running.md](paratroop_harness/running.md#a-scratch-data-folder).
- **Share the GPUs.** Training defaults to the Mac's GPU (MPS). Prompted modules and every `baseline_llm.py` use the
  PC's RX 580 through its gateway, which the harness_01 README notes is shared with the family portal.
- **Don't train and serve on the same box.** smTOOLS_COMPUTER_CLI_01's first run took 145.6 minutes for work an idle
  M5 does in 21.1 (0.211 s/step, measured). The difference was Qwen3.5-4B-Q4_K_M holding 2.7 GB of unified memory and
  being generated from on every harness turn, on a 16 GB machine. Before a long training run, point the harness's
  prompted modules at the `pc` provider and unload the Mac's model. Training on the PC instead is not an option: the
  RX 580 is Polaris (gfx803), ROCm dropped it in 2020, PyTorch's ROCm wheels are Linux-only, and AMD's Windows HIP SDK
  starts at RDNA2.
- **Keep the best checkpoint, not the last.** smTOOLS_COMPUTER_CLI_01 peaked at step 5000 (53% exact) and fell to 47%
  by step 6000 while its saver overwrote the file every eval, so the better weights were lost. Its `train.py` now saves
  on improvement and reloads that checkpoint before the final scoring.
- **Checkpoints get picked up live.** The harness reloads a checkpoint whenever its file changes, and training scripts
  save during training, so a running page uses half-trained models. smMATH001-a's `out/math.pt` was in the Math dropdown
  from its step 1000 save at 17:40, while it trained. See
  [architecture.md](paratroop_harness/architecture.md#the-cache).
