# smPERCEPTION_01 plan: a small Perception module, requested by lamRD

**Status: a request from lamRD, and a plan. Nothing here is built.** Written 2026-09-19 by the lamRD team, checked
against `paratroop_harness_02/` (`harness.json`, `contracts.json`, `checks.json`, `harness.py`),
`smMATH_LANGUAGE_001/`, [results.md](../research/results.md) and the 45 files in `paratroop_harness_02/runs/` on that
date. The runs were read for their structure and messages only. Their timings belong to lamRD's Q23, which fixes its
columns before reading any timing, so this plan doesn't read them. Each phase has deliverables, a gate, and what to
write down. A phase that fails its gate is still a result, and it goes in [results.md](../research/results.md).

**Who's asking, and why it's a request.** [lamRD](https://github.com/bakerheit/lamrd) is the research programme spun
out of this repo. Its helpers line asks whether small local models can make the 8B cheaper to use while the 8B keeps
the quality ([helpers.md](https://github.com/bakerheit/lamrd/blob/main/docs/helpers.md), questions Q23 and Q27 on its
[board](https://github.com/bakerheit/lamrd/blob/main/docs/questions.md#helpers)). One of its levers is delegation:
hand one narrow job in the agent loop to a tiny trained specialist. paratroop_harness_02 is the only agent loop either
repo owns, and building modules for it is SMMOL's work. lamRD reads SMMOL and doesn't train its models, so this is
written as a request for SMMOL to run.

Contents:

1. [The short version](#1-the-short-version)
2. [What Perception does today](#2-what-perception-does-today)
3. [The question, and what "held quality" means](#3-the-question-and-what-held-quality-means)
4. [Pre-registered predictions, and what would retire the request](#4-pre-registered-predictions-and-what-would-retire-the-request)
5. [Decisions already made](#5-decisions-already-made)
6. [Phases](#6-phases)
7. [Project layout](#7-project-layout)
8. [Ground rules that apply](#8-ground-rules-that-apply)
9. [Open questions for the owner](#9-open-questions-for-the-owner)
10. [Rough budget](#10-rough-budget)

---

## 1. The short version

**The request.** Train `smPERCEPTION_01`, a small model for paratroop's Perception module, on
[smMATH_LANGUAGE_001](../smModels/smMATH_LANGUAGE_001.md)'s recipe. Offer it in the Perception dropdown, and measure
whether turns come out the same as with Ministral 8B.

**Why Perception:**

- **It's the untested half of the harness.** The README's "Not done yet" says every prompted module runs on Ministral
  8B, so "much smaller than a general model" is untested. The three trained modules (Router, Math Language, Math) sit
  at the edges of the cycle. None of the eight prompted ones has been replaced.
- **It runs first, once per turn, with a narrow contract.** At most 4 observations, 7 kinds, short fields and a short
  goal.
- **It's shaped like the one job where a small model here held level with the 8B.** It reads structure out of a
  message, as the maths reader does. The maths reader's 77.5% against the 8B's 82.5% is p = 0.80, a tie. It's also
  the only job where small-first, 8B-when-unsure beat both models: 85.0%, handing off 32%.

**Why not the Critic.** Over-asking is the loop's best-known failure, and the Critic drives it. But the Critic's job is
a severity judgement with no good labels. The only cheap labels are the 8B's own, and distilling those would teach a
small model to over-ask too.

**The claim is about cost, not accuracy.** This is not "small beats big". The 8B's turn is the reference. The question
is whether the turn comes out the same, within a margin set now, at a fraction of the time. That's the kind of claim
that survived in this repo: every accuracy headline was downgraded, and every efficiency number held.

**Two controls run before anything is trained, and either can end the request:**

- **Is there anything to save?** lamRD's Q23 reads where the 8B's time goes in the 45 saved runs. If Perception isn't
  a real share of it, stop.
- **Does Perception matter?** With Perception off, the harness already falls back: the raw message becomes one
  observation. If turns come out the same that way, within the 8B's own rerun noise, then Perception isn't worth a
  model. It's worth a switch, and the saving is all of its time for no training. lamRD predicts that the cheapest
  helper often has no model in it (its H1). This is that test on this job.

**A correction to the pitch that led here.** Perception was described to lamRD as "mostly copying". It isn't:

- `when` is a real date, counted from today;
- `text` is a third-person rewrite;
- `goal` is free text.

So the job is split. The model writes the kinds, the fields, and the time phrase as it appears in the message. Code
turns the phrase into a date, the way the calculator works out what the maths reader writes. Whether the model can
also write `text` and `goal` well enough is part of what phase 5 measures.

## 2. What Perception does today

**In** (`harness.py`, `perceive()`): the message, today's date, and the conversation view (the last 6 turns).

**Out** (`contracts.json`): up to 4 `observations`, plus a `goal` (≤ 120 characters). Each observation has:

- a `kind`: event, fact, request, question, feeling, preference or small_talk;
- `text` (≤ 160), `who` (≤ 60), `what` (≤ 80), `when` (≤ 40) and `where` (≤ 60).

It runs on Ministral 8B on `pc`, at temperature 0, with 500 tokens.

**What each field asks of a model:**

| Field | What the prompt asks | Kind of work |
|---|---|---|
| `kind` | one of 7 labels per observation | classification, house labels |
| `who`, `what`, `where` | "come only from the message" | mostly copying; sometimes a light rewrite ("Ana, User's sister") |
| `when` | "a real date like Friday 2026-09-18, counting from today" | find the time phrase, then date arithmetic |
| `text` | "one short, self-contained sentence about the User" | a third-person rewrite |
| `goal` | "what the person most likely wants … in a few words" | short free text |
| a continuation ("yes", "I'm the investor") | fill in from the conversation | needs the last turn |

**Who reads it:**

- Attention scores each observation.
- Recall, Planner, Predictor, Critic, Decision and Language read the focused observations and the goal.
- `remember()` saves the facts, preferences and events that got attention to long-term memory, as their `text`. So
  `text` matters beyond the turn: it's what memory keeps.

**With Perception off** (`harness.py`, the `on("perception")` branch in the cycle): one observation of kind `request`
whose `text` is the message, and a goal that is the message cut to 120 characters. That fallback is arm B below.

**What exists to test it.** None of it can hold quality for a change of this size (lamRD METHOD §1):

- `checks.json` has 2 Perception checks;
- the 45 saved runs hold 39 distinct messages, and they're the owner's testing, not a sample of use;
- `eval.py` has 6 conversations.

**The one timing on record.** In the first live run, on 2026-09-15, Perception took 9.5 s of 82 s
([results.md](../research/results.md#paratroop_harness_02-first-live-run)). That's one run. Q23 is the measurement.

## 3. The question, and what "held quality" means

**The question.** Can a small trained model take Perception off the 8B, so that turns come out the same within a
margin set before the run, at a fraction of the time?

**The meter is lamRD's:** big-model cost per answer at held quality, with the helper's own cost counted in.

**Held quality is judged at the turn, not the field.** A Perception output only matters through what the turn does with
it. Two outputs that word `text` differently, but lead to the same action and the same memory write, are the same here.
Field agreement is reported too, but it doesn't decide.

**The turn outcome, fixed now.** Two turns agree when all four of these match:

- the chosen candidate's type (`answer`, `make`, `ask` or `tool`) in the first cycle;
- the tool, if there is one;
- whether the run stopped early (small talk, or "not for me");
- whether anything was written to long-term memory.

Reply wording isn't graded. See [section 9](#9-open-questions-for-the-owner).

**The noise floor is the 8B against itself.** Planner runs at temperature 0.2, so the same turn run twice with the 8B's
Perception doesn't always pick the same action. A replacement can't be asked to agree with the 8B more often than the
8B agrees with itself. That rerun agreement is measured first, in phase 2, and the margin is taken from it.

**Why the floor is taken at the turn.** Perception itself runs at temperature 0, so at the field level the 8B agrees
with its own rerun almost every time, by construction. That makes field-level self-agreement a useless floor. lamRD's
Q27, as written, uses "agreement with the 8B against the 8B's agreement with itself on a rerun". For a module at
temperature 0 that has to be taken at the turn, as here. This is noted so lamRD can correct Q27.

**The arms**, all on the same messages, with the same downstream modules and models:

| Arm | Perception | Why it's here |
|---|---|---|
| A | Ministral 8B, as today | the reference |
| A′ | the same, run again | the noise floor |
| B | off: the harness fallback | the no-model control. If B holds, stop |
| C | smPERCEPTION_01, with code for dates | the request |
| D | C first, the 8B when C is unsure | the hand-off. Run only if C falls short |

## 4. Pre-registered predictions, and what would retire the request

Written by lamRD before any run. SMMOL may revise them before phase 0's gate closes, never after. "Agreement" means
turn agreement with arm A, as defined in section 3. "The margin" is **8 points** below A–A′ agreement (the owner
confirms it, section 9).

- **P1.** Q23 puts Perception's upper bound (the share of a run that would go if Perception cost nothing) at
  **10–25%**, median over the runs where it ran.
- **P2.** A agrees with A′ on **85–95%** of turns.
- **P3.** B falls **outside** the margin. Perception changes what the turn does.
- **P4.** At the Perception boundary, C matches A's multiset of kinds on **≥ 85%** of messages. Where A filled `when`,
  C's resolved date matches it **≥ 90%** of the time.
- **P5.** C lands **inside** the margin, with Perception at least **100×** faster: tens of milliseconds against
  seconds.
- **P6.** If P5 fails, D lands inside the margin, handing off **≤ 40%** of messages. Its threshold is fitted on
  held-out templates, never on the reported set.

**Falsifiers:**

- **F1: nothing to save.** Q23 puts Perception's upper bound under **10%**. Stop before phase 1. Re-aim the request
  at the module Q23 ranks first under Q27's rule: the most 8B time, then the narrowest output.
- **F2: Perception doesn't matter.** B is inside the margin. The cheapest Perception is none:
  - recommend it off by default, which saves all of its time;
  - train nothing.

  This is a good result, and it gets written up as one.
- **F3: a small model can't hold it.** Both C and D fall outside the margin. Report which fields broke, using phase
  4's per-field table. If the breakage is in `text` and `goal`, the rewrites need a bigger model, and a split module
  (small for the kinds and fields, 8B for the rewrites) is the next plan, not this one.
- **F4: the eval can't see it.** A–A′ agreement is under **70%**. Then the turn outcome is too noisy for an 8-point
  margin at this n. Stop and fix the outcome measure before comparing anything. For example, run Planner at 0 for the
  eval, and say that it was.

## 5. Decisions already made

Not up for debate inside the run. If one turns out to be wrong, stop and say so rather than working around it.

- **Q23 gates the request.** It's lamRD's question. It runs read-only on the saved runs, with its columns fixed before
  any timing is read. This plan doesn't read the timings itself.
- **Recipe: smMATH_LANGUAGE_001's.** A byte-level GPT reader:
  - layout `input <SEP> target <END>`, with the loss on the target only;
  - greedy decoding;
  - `sure` and `weakest` from `read_with_confidence`.

  It's the recipe already proven in the harness on the closest job, and it gives arm D its confidence signal for free.
  No size search here. That's lamRD's Q28, after a saving shows.
- **Input:**
  - the assistant's last reply, at most 64 bytes, as the router takes it;
  - the message, at most 200 bytes.

  Today's date isn't an input. The model never does date arithmetic.
- **The target is a compact string, like the maths reader's:**

  ```text
  kind|who|what|when-phrase|where|text;kind|…||goal
  ```

  - `when-phrase` is copied from the message: "this Friday", "at 6pm tomorrow".
  - `resolve_when(phrase, today)` is code: tested, with no model in it. It turns the phrase into the contract's
    date form.
  - A phrase it can't parse passes through as written.
  - `parse_target()` turns the string into the contract's JSON.
- **The labels come from the 8B.** Each target is the 8B's own Perception output on a generated message, with `when`
  mapped back to the phrase that produced it.
  - This is distillation, which is what the meter asks for: the job is to do what the 8B does, more cheaply. lamRD's
    Q27 makes the same choice.
  - It keeps smPERCEPTION_01 out of any comparison that forbids distillation, such as BabyLM's.
- **The messages to label come from generators, split by template, not by row.**
  - Sources: this repo's existing generators across jobs (router, maths reader, CLI and smCONVERSATION_001's data),
    plus a new Perception generator for the kinds nothing else produces: events with dates, facts, preferences,
    feelings and continuations.
  - Every template count is written down.
  - lamRD's Q10 showed that i.i.d. generated rows measure template recall: the router scores 99% on them and 36–45%
    on held-out phrasings.
- **A leak gate with a positive control** (lamRD METHOD §4). No training message's text appears in the eval set or in
  any `test.json` here, and the gate must catch a duplicate planted on purpose. The when-small-wins k-shot run dropped
  1,226 of 50,000 pool examples this way; without the gate nothing would have shown it.
- **Select and set thresholds on held-out templates, never on the eval set.** lamRD's Q10 found held-out phrasings
  can't rank the router's epochs (tau-b 0.05 against the hand-written set), so expect checkpoint selection to be
  weak here too. Keep the last checkpoint unless the held-out-template score beats it by more than its seed noise.
- **The eval set is 300 new hand-written messages.** The 39 distinct messages from the saved runs are added and
  reported separately.
  - Sized by lamRD METHOD §1: at a discordance of 0.2–0.3, 300 paired items resolve a gap of about 5–6 points, which
    is enough for an 8-point margin.
  - A 5-point margin would need about 460.
- **The eval runs in scratch data folders** ([running.md](../paratroop_harness/running.md#a-scratch-data-folder)):
  - each message is its own chat, in a fresh folder, so no turn sees another's memories;
  - a continuation item replays its earlier turn first;
  - `file_system` is off, so no turn can create files.
- **Language is off and `max_cycles` is 1 for the end-to-end arms.** Neither changes the first cycle's action, and
  together they're a large share of a turn's time. The same settings apply to every arm.
- **Ministral stays exactly as committed**: same endpoint, model, prompts, schemas and temperatures as `harness.json`.
  The labels in phase 3 use the committed Perception prompt, unchanged.
- **A result that retires the request gets the same prominence as one that supports it.** That's F1 to F4.

## 6. Phases

### Phase 0: freeze, test first, and gate on Q23

**Deliverables**

| File | What it is |
|---|---|
| `smPERCEPTION_01/predictions.md` | P1–P6, F1–F4 and the margin, copied from section 4 and dated before any run |
| `smPERCEPTION_01/when.py` | `resolve_when(phrase, today)`, and `back_map(message, date, today)` for phase 3 |
| `smPERCEPTION_01/tests/test_when.py` | 40 hand-written phrases with their dates, across weekdays, "tomorrow", times, month names and year ends |
| `smPERCEPTION_01/tests/test_target.py` | `parse_target` round-trips the contract; a malformed target is refused, not guessed |
| `smPERCEPTION_01/tests/test_leak.py` | the leak gate, with its planted-duplicate control |

**Gate.** lamRD's Q23 result is published, and F1 doesn't fire. The tests pass.

**Write down.** Perception's Q23 upper bound, and the module Q23 ranks first.

### Phase 1: the eval set

**Deliverables**

- **`smPERCEPTION_01/eval/messages.json`.** The 300 hand-written messages and the 39 from the runs, flagged
  separately. Each item has the message, its earlier turn if it's a continuation, and who wrote it.
- **`smPERCEPTION_01/eval/README.md`.** The writing brief, with the mix fixed before writing starts:
  - at least 30 of each kind;
  - at least 40 with a time phrase;
  - at least 30 continuations;
  - at least 30 small talk.
- **Two writers**, about 150 messages each. Neither sees any model's output while writing.

**Gate.**

- The mix is met.
- The leak gate passes, and its control fires.
- No message is a near-duplicate of another, or of any `test.json` here (3-gram shingle Jaccard ≤ 0.8), and the
  near-duplicate check proves it can find a planted one.

**Write down.** Counts per kind and per writer.

### Phase 2: the floor and the no-model control. Nothing is trained

`smPERCEPTION_01/eval/run_arms.py` runs arms A, A′ and B end to end on the eval set, each message in its own scratch
folder. For each turn it stores:

- the turn outcome;
- the observations and goal;
- Perception's seconds, and the turn's seconds.

**Readouts**

- A–A′ agreement: P2, F4.
- B against A, as a paired comparison with `lib/stats` (lamRD) and the non-inferiority margin: P3, F2.

**Gate.** F4 doesn't fire. **If F2 fires, stop here.** Write it up and recommend Perception off by default. Phases 3
to 5 don't run.

**Cost.** 3 arms × 339 turns × about 65 s, which is **about 18 PC-hours**, resumable in batches.

**Write down.**

- The three agreement rates, with intervals.
- The B–A disagreements, broken down by kind and by which part of the outcome changed.
- The 10 turns where B differs from A most interestingly, a sentence each.

### Phase 3: messages and labels

**Deliverables**

| File | What it is |
|---|---|
| `smPERCEPTION_01/data.py` | the Perception generator, and the pool drawn from the existing generators, each row tagged with its template |
| `smPERCEPTION_01/label.py` | calls the committed Perception prompt on each pool message; resumable, one line per message in `data/labels.jsonl` |
| `data/split.json` | train and held-out templates, with 20% of templates held out, stratified by kind |

- **Start with 3,000 labelled messages** (about 8 PC-hours at Perception's recorded 9.5 s).
- **Check the learning curve** at 1,000, 2,000 and 3,000 on held-out templates. Label more only if it's still
  climbing by more than its seed noise.
- **`when` back-mapping.** For each label with a filled `when`, `back_map` finds the phrase in the message that
  `resolve_when` turns into that date.
  - A row that can't be mapped keeps its other fields, and its `when` is dropped from training.
  - Those rows are counted.

**Gate.**

- At least **98%** of labels are contract-valid.
- At least **90%** of filled `when` values back-map. Below that, the code path doesn't fit this job. Stop and write
  down the failures before training anything.

**Write down.** Templates per kind, rows, label failures, the back-map rate, and the PC-hours spent.

### Phase 4: train, and read at the boundary

- **Model.** Start from smMATH_LANGUAGE_001's configuration (d 256, 6 layers, 8 heads, about 4.9M parameters), with
  the context raised for the longer target: 64 + 200 bytes of input, up to 320 bytes of target.
- **Train on the M5** with the Mac's served model unloaded. Keep the checkpoint as section 5 says.
- **Score at the Perception boundary against A's output.** Do this on held-out templates and on the eval set. Use the
  eval set's arm-A outputs from phase 2; don't make new calls.
- **Readouts:**
  - kind-multiset agreement;
  - per field, exact match and token F1 for `who`, `what` and `where`;
  - the resolved date against A's `when`;
  - observation count;
  - contract validity after `parse_target`;
  - latency on the CPU and on MPS.

**Gate.** At least **98%** contract-valid on the eval set. Output that doesn't parse falls back to arm B's behaviour,
which is exactly what's being tested against, so it's allowed, but it's counted.

**Write down.**

- The per-field table: generated, held-out templates and eval set side by side. The gap between them is expected;
  record its size.
- P4's outcome.

### Phase 5: the harness, and the end-to-end comparison

**Wiring**, per [adding-a-module.md](../paratroop_harness/adding-a-module.md):

- **Loader.** Add a `LOCAL_CODE` entry for kind `perception`, pointing to `read.py:Perceiver`.
- **Dropdown.** `options()` lets the Perception module offer `smmol` models of kind `perception`. Today only
  `language` has a local special case there.
- **Dates.** `perceive()` runs `resolve_when` on each observation's phrase.
- **Failures.** Broken output is noted on the trace entry and falls back to the no-Perception behaviour, as Math
  Language does.
- **Tests.** Stubs in `tests/test_harness.py`, as for the router and the readers.
- **Page state.** Don't add it to the owner's choices. It appears in the dropdown and the owner picks it.

**Run C end to end** on the eval set: 339 turns, about 5 PC-hours, since Perception is now local.

**If P5 fails, run D:**

- Fit the threshold on held-out templates by Youden's J, using C's own right and wrong at the boundary. "Right" means
  every kind matches, and every field A filled matches after normalising.
- Freeze the threshold before the eval set is touched.
- Rerun the handed-off turns fresh. Don't copy them from arm A, so D isn't credited with A's own answers.

**Readouts**

- C against A, and D against A, with paired intervals and the non-inferiority margin: P5, P6, F3.
- Median Perception seconds, and median turn seconds.
- The hand-off rate.
- Peak memory of the harness process with and without the reader loaded, because lamRD's meter counts the helper's
  own cost.

**Write down.** The arms table (A, A′, B, C and D where run) with agreement, intervals and latency. Also which
falsifier fired.

### Phase 6: write up

- **`smPERCEPTION_01/README.md`**, in the house format: the idea, the data, the model, results, what it means, next,
  run it, files.
- **[smModels/smPERCEPTION_01.md](../smModels/)**, whatever the outcome.
- **[results.md](../research/results.md)**: dated entries for phases 2 and 5 at least, each with its falsifier's
  outcome.
- **[architecture.md](../paratroop_harness/architecture.md)**: the module table, if C or D shipped. Also the default
  switches, if F2 fired and the owner agreed.
- **This repo's engineering README**: the models table.
- **Tell lamRD.** Send the arms table for Q27, and the Q27 correction from section 3.

## 7. Project layout

```text
smPERCEPTION_01/
  README.md
  predictions.md       P1–P6, F1–F4 and the margin, dated before any run (phase 0)
  when.py              resolve_when(phrase, today) and back_map(message, date, today)
  data.py              the Perception generator and the cross-job pool, rows tagged by template
  label.py             the 8B's Perception on pool messages, resumable (phase 3)
  model.py             ReaderGPT, from smMATH_LANGUAGE_001
  train.py
  read.py              Perceiver: read() and read_with_confidence() -> the contract's JSON
  score.py             boundary scoring against the 8B's output
  eval/
    README.md          the writing brief and the kind mix
    messages.json      300 hand-written + 39 from saved runs, flagged
    run_arms.py        arms A, A′, B, C and D end to end, one scratch folder per message
  tests/               when, target round-trip, leak gate with its planted control
  data/                pool, split.json, labels.jsonl, logs/
  out/                 weights (gitignored), results.json, arms.json
```

## 8. Ground rules that apply

- **Leave the owner's page state alone.** Every harness run uses a scratch data folder. `switches.json`,
  `choices.json` and `settings.json` in `paratroop_harness_02/` aren't touched.
- **Share the PC.** Labels and arms come to about 31 PC-hours, run in resumable batches that pause when the family
  portal wants the RX 580.
- **Don't train and serve on the same box.** Phase 4 runs with the Mac's served model unloaded.
- **Checkpoints get picked up live.** Don't add `smPERCEPTION_01` to `harness.json` until phase 5. A half-trained
  reader would otherwise show up in the dropdown mid-run.
- **Select on held-out templates, report on the hand-written set.** This is the rule when-small-wins phase 4
  followed. Every threshold it fitted on i.i.d. generated rows over-referred.
- **Paired comparisons with intervals, beside the percentages** (lamRD METHOD §2 and §3). "Inside the margin" is a
  similarity claim, so it needs the margin and an interval inside it, not just a high p.

## 9. Open questions for the owner

1. **The margin.** 8 points of turn agreement is lamRD's guess at "the owner wouldn't notice". A tighter margin needs
   more messages: 5 points needs about 460 at a discordance of 0.3.
2. **The turn outcome.** Action, tool, early stop and memory write, with no reply grading. Is that what "the same
   turn" means to you? Grading replies needs a blind rater, and roughly doubles the person time.
3. **PC time.** About 18 hours for phase 2, 8 for labels and 5 for phase 5, plus D if it runs. When is the RX 580
   free?
4. **The second writer** for the eval set. That's about 3 hours each.
5. **Distillation.** smPERCEPTION_01 would learn the 8B's habits, including any you don't want, such as splitting
   one ask into several observations. Is copying the 8B the goal, or would you rather it learned a house convention
   from a generator, like the router? The second is a different plan: its reference is a hand-labelled set, not the
   8B.
6. **If F2 fires,** is Perception off by default acceptable? It changes the harness's default switches.

## 10. Rough budget

Guesses, to be corrected after phase 2.

| Phase | Machine time | Person time |
|---|---|---|
| 0: freeze, tests, Q23 gate | Q23 is lamRD's, minutes of CPU | 1–2 hours on `when.py` and the tests |
| 1: eval set | none | about 3 hours per writer, two writers |
| 2: A, A′, B end to end | about 18 PC-hours | 1 hour on `run_arms.py`, 1 hour reading |
| 3: messages and labels | about 8 PC-hours for 3,000 labels | 2–3 hours on the generator |
| 4: train and score | 15–30 M5-minutes a run, a few runs | 1 hour |
| 5: harness and C, maybe D | about 5 PC-hours, plus D's handed-off turns | 2 hours wiring, 1 hour reading |
| 6: write-up | none | 2 hours |
