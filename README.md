# SMMOL

**Many small models, each doing one job, trained from scratch on a laptop — measured honestly against the
alternatives.**

The popular AI assistants run one giant model in a data centre. SMMOL tests a different shape: a dozen small
specialised models, each handling one narrow "brain job", wired together by a web harness you can chat with.
One model sorts messages by intent. One pulls the arithmetic out of a sentence. One does the arithmetic.
Ordinary code does the jobs that never needed a model — exact maths, saving files, remembering facts.

Every model here was trained from scratch on one MacBook M5. Nothing was fine-tuned from a pretrained
checkpoint. The largest is 10.8M parameters.

This is a solo home research project, and the numbers below are small-sample and single-machine. They are
reported with their losses as well as their wins.

---

## Does it work? Sometimes.

Every small model is scored against two baselines on the same hand-written test set: plain keyword rules with
no model at all, and Ministral 8B prompted to do the same job. The full table with sources lives in
[the results log](docs/engineering/research/results.md#baselines-at-a-glance).

| Job | Small model here | No model (rules/regex) | Ministral 8B |
|---|---|---|---|
| Routing — intent, tool and ask-first all right (93 messages) | **71%**, 2.0 ms | 63% | 58%, 2.0 s |
| Routing bare arithmetic (17 messages) | **94%** | 24% | not run |
| Reading maths out of a message — every problem (40 messages) | 78%, 29 ms | 30% | **82%**, 3.1 s |
| Same — final answer right | 80% | 33% | **95%** |
| Plain request → shell command, exact (53 requests) | **47%**, 47 ms | 40% | 15%, 1.9 s |
| Same — right program | 64% | 60% | **79%** |
| Same — stays quiet when it isn't a terminal job (6) | **100%** | 33% | 50% |

**Read that honestly, and read the paired tests before the percentages.** These are 40–93 hand-written items, and
until 2026-09-18 not one number in this repo had an interval on it. Now they do
([significance.py](benchmarks/when_small_wins/significance.py)):

| Claim | Gap | McNemar p | 95% CI |
|---|---:|---:|---|
| Router beats a prompted 8B | +12.9 | **0.073** | [+0.2, +25.6] |
| CLI, exact command | +31.9 | **0.0026** | [+14.1, +49.7] |
| Maths reader vs the 8B | −5.0 | **0.80** | [−24.5, +14.5] |

So: **the router's headline win is not significant at 0.05** — its interval nearly touches zero. The maths reader
does **not** lose to the 8B; on 40 items the two are indistinguishable. The one genuinely solid accuracy result is
the CLI model's, which is also the one this README spent a day hedging on other grounds.

**And the opponent was never chosen.** Every baseline here is `ministral-8b` because that is what was running on
the PC; no document gives a reason. Running the same CLI test against `qwen3-coder-30b` — same prompt, same schema,
same grading, only the model name changed — puts the big model at 34.0% exact instead of 14.9%. The small model
still wins, but by **+12.8 rather than +31.9**: about 60% of the published margin was opponent choice. The
right-program picture is unchanged, 63.8% against 76.6%.

The CLI model gets the exact command right more often than either big model and names the right program less often
— it knows the shape of an answer better than it knows the tools.

**The rule this used to state has not survived contact with the item-level data.** It read: *a small model wins
when the job is narrow, has a closed output space, and can be generated in bulk.* But the CLI model has by far
the **largest** output space here — arbitrary shell strings — and posts the **largest** win; the program name it
picks comes from a closed 79-task catalog, and that is exactly where it loses. Same model, same 47 messages,
same run, opposite sign. "Closed output space" is not the axis.

What replaced it, measured rather than asserted
([write-up](docs/engineering/research/results.md), [code](benchmarks/when_small_wins/)): **two defensible
graders disagree about who is better.** Forgiving only spelling — a leading `apt update &&`, flag order — moves
the CLI lead just 4.3 points, from +31.9 to +27.7, because only two of Ministral's forty command misses were
spelling. Score by "did it name the right program" instead and the lead inverts to −14.9. The small model
reliably produces the house convention and picks the wrong tool; the 8B picks the right tool and spells it its
own way.

Finding the axis that actually predicts this, before a model is trained, is what
[when-small-wins.md](docs/engineering/plans/when-small-wins.md) is for. It also runs the control nobody has run
yet: every comparison here pits tens of thousands of labelled examples against a single zero-shot prompt.

---

## Findings worth reading even if you never run this

These are written up in full in [`docs/engineering/research/results.md`](docs/engineering/research/results.md).

- **bf16 is free on Apple silicon; `torch.compile` is a 1.7× *regression*.** Measured roofline on the M5:
  fp32 3.70 TFLOP/s, bf16 15.19 TFLOP/s. Mixed precision cost nothing in quality across seven paired
  comparisons. Compiling made it slower — the opposite of the usual CUDA result.
  → [smEFFICIENCY_01](smEFFICIENCY_01/README.md)

- **Early stopping saved more electricity than precision did.** A 20-minute run spent its last 24–49% getting
  *worse* on held-out data. Stopping at the validation minimum gave an equal model in 613 s instead of 1,200 s.

- **The same training run is 7× faster on an idle Mac** — 21.1 min vs 145.6 min, traced to 319k page-outs under
  memory pressure. Earlier wall-clock numbers in the same log are explicitly re-labelled as upper bounds
  because of it.

- **Keeping the best checkpoint only helps if some metric can tell epochs apart.** Four trainers were silently
  saving the last checkpoint instead of the best. Fixing it changed nothing for the router: its generated metric
  saturates at 98% by epoch 4, and the hand-written set cannot break the tie either — its epochs sit one to two
  messages apart out of 93, inside the run-to-run gap between two identical configs. An earlier version of this
  README called epoch 3 "a real peak"; that was reading noise, and the correction is recorded in place.

- **A negative result, kept.** [`smCONVERSATION_001/experiments/muon/FINDINGS.md`](smCONVERSATION_001/experiments/muon/FINDINGS.md)
  — the Muon optimiser came out 0.075% ahead, called out as smaller than seed variance and explicitly *not*
  worth a production switch.

- **"Small first, big when unsure" works on some jobs and hurts on others.** The maths reader and a prompted 8B
  are wrong on *zero of the same 40 items*, so handing off the messages the small model is unsure about scores
  **85.0%** against 77.5% alone and 82.5% for the 8B. But the same method costs the CLI model 6.4 points, because
  it refers to a model that scores 14.9% at that job. And the router's 2-point gain costs a **482× slowdown**
  (2 ms → 964 ms), which throws away the only thing that made it interesting. The rule is duller than it first
  looked: hand off only to a model that is actually better at the job, and only when the latency is worth it.
  An earlier version of this README stated the general claim from the maths reader alone; running the other two
  jobs refuted it. → [benchmarks/when_small_wins](benchmarks/when_small_wins/)

- **The same distribution gap has now broken four different things.** Generated held-out data saturates — 98.1%
  for the router, 99.4% for the maths reader, 97.5% for the CLI model — so it cannot rank checkpoints, cannot
  fire early stopping, and fits hand-off thresholds that over-refer on real phrasing at 32%, 48% and 64%.
  Four symptoms, one cause.

- **A file whose entire job is to say "this is not a result."**
  [`smRTS_01/out/smoke-cpu/NOT_A_RESULT.md`](smRTS_01/out/smoke-cpu/NOT_A_RESULT.md)

---

## Layout

Each `sm*` folder is one self-contained experiment: its own data generator, model, trainer, tests and README
with dated results and known failure modes.

```
docs/               engineering/ (design, per-model docs, the results log, plans)
                    product/     (the same thing in plain English)
paratroop_harness_02/   the assistant: turn cycle, modules, web page, HTTP API
paratroop_harness_01/   the previous harness, kept for comparison
smROUTER_01/        message → intent, tool, ask-first        1.87M params
smMATH01/           arithmetic in the weights, 3 variants
smMATH001-a/        works whole expressions out step by step
smMATH_LANGUAGE_001/    pulls the maths out of a sentence
smLANGUAGE_*/       byte-level language models, incl. a curriculum by school grade
smCLM_01, smCLM_02/     word → the general ideas it carries
smLLM_01/           Tiny Shakespeare byte GPT, the starting point
smALLM_01/          learn-by-doing in a gadget world
smRTS_01/           recurrent cells trained online, no backprop through time
smEFFICIENCY_01/    where the training compute actually goes
smTOOLS_COMPUTER_CLI_01/    plain request → shell command
smSOFTWARE_ENGINEERING_001/ trained on 75 permissively licensed repos
```

Start with [`docs/engineering/README.md`](docs/engineering/README.md), or
[`docs/product/README.md`](docs/product/README.md) if you'd rather skip the jargon.

---

## Running it

Python 3.9+ and PyTorch. The models train on Apple silicon via MPS, and fall back to CPU.

```bash
python3 -m unittest discover -s tests    # from inside any project folder
```

Each project's README carries its own training command. They are minutes-to-hours on an M5, not days.

Some projects compare against a larger model over HTTP — an OpenAI-compatible endpoint you run yourself.
Nothing here calls a paid API by default:

```bash
export SMMOL_LLM_URL=http://your-box:8081          # baselines and teachers
export SMMOL_PROVIDER_PC_URL=http://your-box:8081  # the harness's providers
```

Both default to `127.0.0.1`.

---

## What's deliberately not in this repo

- **Trained weights** (~1.2 GB). Two exceed GitHub's 100 MB file limit. Every project documents how to retrain.
- **Training corpora.** TinyStories and Tiny Shakespeare are fetched by each project's own script, not
  redistributed here. The 483 KB corpus of third-party source in `smSOFTWARE_ENGINEERING_001` is rebuilt
  byte-for-byte from the pinned commits and blob SHAs in its tracked `manifest.json`, alongside every upstream
  licence text.
- **The harness's live state** — conversations, memory, run transcripts. That is one person's actual usage, not
  test data. The harness recreates all of it on first run, so a fresh clone works as-is.

## Honest caveats

- One person, one machine, one run per configuration in most cases. Where a result sits inside seed noise, the
  write-up says so.
- The hand-written test sets are small — 93 messages for the router, 53 for the CLI model, 17 for arithmetic.
  A 12-point move there can be two examples.
- `smMATH_LANGUAGE_001` currently has one failing test (`test_reader.py`, a compound-unit case the trained
  checkpoint no longer extracts). It is a real, known model regression, left visible rather than tuned away.
- `smRTS_01` is mid-experiment. Its sources are hashed into run manifests and should not be edited casually.

## Licence

MIT — see [LICENSE](LICENSE), which also records the terms of the third-party data this project trains on.
