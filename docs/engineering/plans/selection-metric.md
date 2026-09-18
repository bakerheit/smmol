# A selection metric that can rank epochs: the held-out-phrasings split

**Status: a plan. Nothing here is built.** Written 2026-09-17, checked against `smROUTER_01/`,
`smTOOLS_COMPUTER_CLI_01/`, `smMATH_LANGUAGE_001/`, `smMATH001-a/` and `smMATH01/` on that date. It is meant to be
handed to agents and executed phase by phase. Each phase has deliverables, a gate, and what to write down. Don't skip
a gate; a phase that fails its gate is still a result, and it goes in [results.md](../research/results.md).

Contents:

1. [The short version](#1-the-short-version)
2. [Where this stands, with the numbers](#2-where-this-stands-with-the-numbers)
3. [The question this run answers](#3-the-question-this-run-answers)
4. [Pre-registered predictions](#4-pre-registered-predictions)
5. [What would count as failure](#5-what-would-count-as-failure)
6. [Decisions already made](#6-decisions-already-made)
7. [Phases](#7-phases)
8. [Ground rules that apply](#8-ground-rules-that-apply)
9. [Why this and not the other open threads](#9-why-this-and-not-the-other-open-threads)
10. [Open questions for the owner](#10-open-questions-for-the-owner)
11. [Rough budget](#11-rough-budget)

---

## 1. The short version

Every trainer in this repo generates its own training data from templates, and every one of them holds out its
validation set by drawing more rows from the same generator with a different seed. That held-out set is therefore
i.i.d. with training, it saturates at 98–99%, and it cannot tell one epoch from another. Three things in the repo
depend on it and are broken or bent because of it:

- **The best-checkpoint fix is inert on the router.** `best_epoch` was 5 — the last — so v3 saved exactly what the
  old code would have ([results.md, 2026-09-17](../research/results.md)).
- **Early stopping never fires.** `--patience` is on ten trainers and keys off the same saturated scalar. On the
  router's five epochs the generated score goes 0.89, 0.94, 0.97, 0.98, 0.98 — monotone, so patience 3 is dead code.
  The 1.96× electricity saving measured in smEFFICIENCY_01 is not being collected on these runs.
- **One trainer already gave up and selects on its own test set.** `smTOOLS_COMPUTER_CLI_01/train.py` picks the
  checkpoint and early-stops on `standing(hand)`, where `hand` is scored on `test.json` — the 53 hand-written
  requests it reports as its headline result. Its own docstring says why: *"Only the hand-written set can tell
  checkpoints apart. The generated one saturates at 99% by step 3000."* A retrain of that model is listed as planned
  and has not run.

This plan replaces the i.i.d. held-out set with a **held-out-phrasings split**: the generator's template phrasings and
slot fillers are split in two, deterministically, and the validation set is generated only from the half training
never saw. It is still generated, still free, still never the hand-written set — but it measures generalising to new
wording instead of memorising the template list, which is the exact failure every model here has
(router 98% generated vs 71% hand-written; math reader 99% vs 78%; CLI 99% vs 47%).

The change is about 60 lines in `data.py` and 6 in `train.py` per project. It is validated on the router first,
because the router retrains in 13.5 minutes.

## 2. Where this stands, with the numbers

**The router's two curves, three runs, from `out/v1/train.log`, `out/v2/train.log`, `out/v3-train.log`:**

| Epoch | generated all-right, v1 / v2 / v3 | hand-written all-right, v1 / v2 / v3 |
|---|---|---|
| 1 | 0.90 / 0.91 / 0.89 | 0.74 / 0.63 / 0.62 |
| 2 | 0.94 / 0.94 / 0.94 | 0.71 / 0.69 / 0.70 |
| 3 | 0.97 / 0.97 / 0.97 | 0.67 / 0.70 / 0.72 |
| 4 | 0.98 / 0.98 / 0.98 | 0.72 / 0.73 / 0.70 |
| 5 | 0.98 / 0.98 / 0.98 | 0.73 / 0.71 / 0.71 |

Three different models — v1 before the arithmetic classes, v2 after, v3 with bf16 — trace the **same** generated
curve to two decimal places while their hand-written curves disagree about which epoch is best (1, 4 and 3). The
generated column is reading the generator, not the model.

**Measured, pooled over those 15 (run, epoch) points:** Kendall tau between the two columns is **0.46**, with 27 of
105 pairs tied on the generated side. Selecting on the generated column costs 0.01–0.02 of hand-written all-right per
run — small on the router, because its curve is flat after epoch 2. It is not small everywhere:

| Model | Generated held-out | Hand-written | What selection costs today |
|---|---|---|---|
| smROUTER_01 v3 | 0.984 | 0.710 | 0.01–0.02, measured above |
| smMATH_LANGUAGE_001 | 0.9915 | 0.775 | not measured |
| smMATH001-a | — (mean over categories) | — | not measured |
| smTOOLS_COMPUTER_CLI_01 | 0.991 | 0.468 | **0.06** (step 5000 scored 0.53, step 6000 0.47, and the saver kept 6000) |

**Who selects on what today** (read from each `train.py` on 2026-09-17):

| Trainer | Selects on | Honest? |
|---|---|---|
| `smROUTER_01` | generated held-out all-right, `val = generate(3000, seed+1000)` | yes, and saturated |
| `smMATH_LANGUAGE_001` | generated held-out exact, `val[:1000]` | yes, and saturated at 0.99 |
| `smMATH001-a` | mean exact over categories from `val`, whose questions are removed from the pool | yes |
| `smMATH01` | mean accuracy over ops at the trained digit length, fresh problems from a fixed seed | yes |
| `smTOOLS_COMPUTER_CLI_01` | `standing(hand)` — **the hand-written `test.json` it reports** | **no** |

That last row is the finding that makes this urgent rather than merely tidy. It was fixed on 2026-09-15, two days
before the four-trainer checkpoint audit of 2026-09-17, and it was not in that audit's scope, so the results.md
sentence "Each selects on a genuinely held-out scalar and never on the hand-written set it reports" is true of the
four and false of the fifth. Nothing is wrong with the published 46.8%: that number came from the *first* run, whose
saver was unconditional. The next run is the problem. As the code stands, a retrain publishes a
selected-on-its-own-test-set number into the README's headline table, next to Ministral's 14.9%.

## 3. The question this run answers

**Can a generated validation set that holds out phrasings, rather than rows, rank a trainer's epochs the way the
hand-written set does — without ever being the hand-written set?**

A "no" is a usable result. If the held-out-phrasings score saturates too, then nothing generated can rank these
epochs, and the honest conclusion is that best-checkpoint selection and early stopping should both be switched off on
these trainers and the last epoch reported, with the reason written down. That is a smaller repo and a truer one.

## 4. Pre-registered predictions

Registered before any run. Write the outcome next to each one.

| # | Prediction |
|---|---|
| P1 | On the router, the held-out-phrasings all-right score at epoch 5 lands **between** the i.i.d. generated score and the hand-written score: **0.82 ± 0.06** (today: 0.984 and 0.710) |
| P2 | Its per-epoch **range over epochs 3–5 is at least 0.03**, against the current metric's 0.01, so it can still tell late epochs apart |
| P3 | Pooled Kendall tau against the hand-written trajectory is **at least 0.65**, against the current metric's measured 0.46 |
| P4 | Holding 26% of phrasings out of training costs **at most 2 points** of hand-written all-right: epoch-5 score in 0.69–0.73, against v3's 0.710. One message on the 93-message set is 1.1 points, so this is inside the noise band the repo already declares |
| P5 | On a 10-epoch router run, the held-out-phrasings score **peaks and then falls**, and `--patience 3` fires before epoch 10 for the first time in this repo |
| P6 | On smTOOLS_COMPUTER_CLI_01, switching `standing()` from the hand-written set to the held-out-phrasings set costs **at most 2 points** of hand-written exact against selecting on the test set directly, i.e. it recovers most of the 6 points that the last-checkpoint saver threw away, without touching the test set |

P1 is the weakest of these and is a guess with a wide band; P3 and P6 are the ones that decide whether the change is
worth keeping.

## 5. What would count as failure

Any one of these means the approach does not work, and it is written up as such:

- **It saturates too.** The held-out-phrasings score reaches 0.97 or more by the halfway epoch and its range over the
  remaining epochs is under 0.02.
- **It does not rank better.** Pooled Kendall tau against the hand-written trajectory is under 0.60, *and* the epoch
  it picks is no better on the hand-written set than the epoch the current metric picks.
- **The split costs more than it gives.** Hand-written all-right at epoch 5 falls below 0.68 — more than 3 points
  under v3 — because training lost a quarter of its phrasings. If this is the only failure, the finding is "the
  phrasings are load-bearing training data", and the follow-on is an evaluation-only split built from *new* phrasings
  written for the purpose, which is a different and more expensive plan.
- **It is not cheap.** Scoring the held-out-phrasings set adds more than 10 s per epoch on the router.

A partial result counts. If it works on the router and saturates on the CLI model, say so per project and stop
generalising.

## 6. Decisions already made

Not up for debate inside the run. If one turns out to be wrong, stop and say so rather than working around it.

- **The hand-written test sets are never touched.** `smROUTER_01/test.json`, `test_arithmetic.json`,
  `smTOOLS_COMPUTER_CLI_01/test.json` and `smMATH_LANGUAGE_001`'s hand-written set are read-only in this work. No new
  cases, no rewording, no re-splitting. They are the only clean measurement the repo has and every result recorded
  before today was measured on them exactly as they are.
- **The split is deterministic and lives in the data module,** not in the trainer. `split(seed, hold)` is a pure
  function of the generator's own tables, so any script can reproduce the same halves without a saved file.
- **It is validated on smROUTER_01 first.** 808 s a run, five epochs, and three prior runs of per-epoch logs already
  exist to compare against. Nothing is ported to another project before the router's gate passes.
- **No hand-written dev set is written in this plan.** A second hand-written set is the obvious alternative and it
  costs hours of a person's time per project. Try the free mechanical thing first; if it fails its gate, the
  hand-written dev set is the fallback and this plan's phase 5 says so.
- **`--precision fp32 --patience 0` keeps reproducing pre-2026-09-17 behaviour.** Add
  `--val-split {iid,phrasings}` defaulting to `iid` in phase 1, flip the default to `phrasings` only at the phase 3
  gate, and record the flip's date in results.md. Every number already in results.md must still be reproducible by a
  documented flag combination.
- **One GPU job at a time on the Mac**, harness pointed at `pc` with the Mac's model unloaded before each run. The
  7× contention finding is not repeated.
- **smRTS_01 is not touched.** Its sources are sha256-hashed into Phase 2 run manifests and it is mid-experiment.

## 7. Phases

### Phase 0: the split, with no training at all

**Deliverables** — all in `smROUTER_01/`:

| File | What changes |
|---|---|
| `data.py` | New `split(seed=7, hold=0.25)`. It returns `{"phrasings": {class_index: [held indices]}, "answers": {group_index: [held indices]}, "slots": {name: [held values]}}`. Per class, hold out `max(1, round(hold * k))` phrasings when `k >= 4`, else none — class 11 has 3 phrasings and keeps all of them. Per slot, hold out `max(1, round(0.2 * len(values)))` fillers. Both draws use `random.Random(seed)` over a sorted list, so the halves do not move when Python's hash seed does. On today's tables that holds out about **52 of the 200 phrasings (26%)** and about one filler in five |
| `data.py` | `generate(n, seed=0, side="train")`. `side="train"` uses the kept phrasings and kept fillers; `side="hard"` uses only the held-out phrasings and, where a slot appears, only the held-out fillers; `side="all"` is today's behaviour, byte-for-byte, and stays the default so nothing else in the repo changes yet |
| `tests/test_split.py` | The two sides share no phrasing and no filler; every class with 4 or more phrasings appears on both sides; `generate(n, side="all")` is identical to the old `generate(n)` for seeds 0, 1, 2 and n = 500; the hard set's label distribution over (intent, tool, ask_first) is within 5 percentage points of the train set's on every label that is at least 2% of either; `without()` still drops any hard row matching `test.json` or `test_arithmetic.json` word for word |

**Gate.** `python3 -m unittest discover -s tests` passes in `smROUTER_01/`. `python3 -c "import data; print(len(data.generate(3000, side='hard')))"` returns 3000 in under 5 s. A person reads 20 random hard rows and agrees the label is right on at least 19 — the hard set is scored against, so a mislabelled hard set is worse than no hard set.

**Write down.** How many phrasings and fillers each side got, per class; the label distributions; the 20-row read.

### Phase 1: log both metrics on one router run

**Change** in `smROUTER_01/train.py`: add `--val-split {iid,phrasings}`, default `iid`. Build `val_hard =
data.without(data.generate(3000, seed=args.seed + 2000, side="hard"), held)` **always**, and score it every epoch
alongside the existing `val`. Selection and patience still key off whichever `--val-split` names, so with the default
this run reproduces v3's decisions exactly. Add `hard` to the epoch line and `best_hard`/`hard_by_epoch` to
`results.json`.

**Runs** (Mac idle, harness on `pc`, model unloaded):

```bash
cd smROUTER_01
python3 train.py --seed 0 --epochs 10 --patience 0 --out out/hard-probe-s0  2>&1 | tee out/hard-probe-s0/train.log
python3 train.py --seed 1 --epochs 10 --patience 0 --out out/hard-probe-s1  2>&1 | tee out/hard-probe-s1/train.log
```

Ten epochs, not five, because the question is whether the metric can see a peak, and five epochs of the current
recipe may not contain one. Patience off so nothing stops early and the whole trajectory is recorded. Two seeds so a
one-run rank correlation is not mistaken for a result. **Neither run writes `out/router.pt`**, so the live harness is
untouched.

**Score** with `analyse_split.py` (new, in `smROUTER_01/`): reads the per-epoch rows from both runs plus the three
historical logs, and prints, for each candidate metric, Kendall tau against the hand-written trajectory pooled over
runs, the epoch it would pick per run, and the regret — hand-written score at the picked epoch minus the best
hand-written score in that run. Candidate metrics, all three reported, **none of them chosen by which wins**:
i.i.d. generated all-right (today's), held-out-phrasings all-right, and held-out-phrasings all-right restricted to
the tool head, which is the head the hand-written set says is weakest (0.806 against intent's 0.871).

**Gate.** Held-out-phrasings all-right has a range of at least 0.03 over the last three epochs of both seeds, and
pooled Kendall tau of at least 0.60. Failing either, stop and write it up as
[section 5](#5-what-would-count-as-failure) says.

**Write down.** The full per-epoch table for both seeds and both metrics; the three taus and three regrets; wall time
per epoch with and without the extra scoring; P1, P2 and P3 marked held or falsified with their actual numbers.

### Phase 2: what the split costs training

A model trained without a quarter of its phrasings might simply be worse. Phase 1's runs already measure it, because
`side="train"` is what they trained on.

**Gate.** Hand-written all-right at epoch 5 of the seed-0 run is at least 0.69 — within 2 points of v3's 0.710 — and
the seed-1 run does not fall below 0.68. If it does, P4 is falsified and the failure clause in
[section 5](#5-what-would-count-as-failure) applies: the split becomes an evaluation-only artifact built from new
phrasings, and this plan stops here with that written down.

**Write down.** Epoch-5 and epoch-10 hand-written scores for both seeds against v1, v2 and v3's; which of the 93
messages changed side; whether the loss curve moved.

### Phase 3: make it the router's selection metric

Only after phases 1 and 2 both pass.

```bash
cd smROUTER_01
python3 train.py --seed 0 --epochs 10 --val-split phrasings --patience 3 2>&1 | tee out/v4-train.log
```

This one **does** write `out/router.pt`. Switch the Router module off on the page first; the harness reloads a
checkpoint whenever the file changes, and this trainer writes once at the end, so the window is small but real.

**Gate.**

- Hand-written all-right at the selected epoch is at least v3's 0.710. A new metric that picks a worse model is not
  an improvement, whatever its tau.
- Arithmetic all-right stays at or above v3's 0.941 — the thing v2 and v3 were built to fix does not regress.
- `best_epoch` is **not** the last epoch, or patience fired. If the trainer once again keeps the final epoch, the
  mechanism is still inert and that is the headline, not a footnote.
- `python3 -m unittest discover -s tests` passes; the harness's Router Test button still grades.

**Write down.** v4's full row next to v1, v2 and v3 in the same table results.md already uses; `best_epoch`,
`best_hard`, whether patience fired and at which epoch; the seconds saved against a full 10-epoch run; P5 marked.
Keep v3 under `out/v3/` the way v2 was kept.

### Phase 4: smTOOLS_COMPUTER_CLI_01, the one that matters

This is the reason for the plan. Port `split()` and `side=` to `smTOOLS_COMPUTER_CLI_01/data.py` over `catalog.py`'s
`say` lists: 79 tasks, 267 phrasings, so `max(1, round(0.25 * k))` per task holds out about 70 (26%), the same
fraction as the router. Slots there are paths, package names and ports, generated rather than listed; split the
generated filler pools the same way.

Then change `standing()` to read the **held-out-phrasings** score, not `test.json`, and change the early-stopping
line with it. Keep the hand-written score on the log line — it is the most informative thing in that log — but it
must not enter `standing()` or `no_improve`.

```bash
cd smTOOLS_COMPUTER_CLI_01
python3 train.py --seed 0 --steps 6000 --patience 3 2>&1 | tee out/v2-train.log
```

About 21 minutes on an idle M5 at the measured 0.211 s/step, not the 145.6 minutes the first run took under
contention.

**Gate.**

- Hand-written exact at the selected step is at least **0.51**, against the first run's published 0.468 and its
  thrown-away peak of 0.53. This is P6: selecting on the new metric should recover most of the 6 points without ever
  reading the test set.
- The step it selects is not the last step, or patience fired.
- Quiet-when-it-isn't-a-terminal-job stays at 1.00.

Failing the first gate by more than 2 points means the held-out-phrasings metric is not good enough for this model,
and the honest move is to **stop selecting altogether** on it: revert to the last step, report it, and say in the
README that this model has no usable selection metric. Do not go back to selecting on `test.json`.

**Write down.** The step-by-step table with both metrics; the selected step and its full hand-written row against the
2026-09-15 run and against Ministral; P6 marked; a dated correction in results.md noting that this trainer selected
on its own test set between 2026-09-15 and the date of this fix, that no published number was produced that way, and
which commit changed it.

### Phase 5: the other three trainers, or the fallback

Ported the same way, in this order, one per run, each gated on its own hand-written score not falling:
`smMATH_LANGUAGE_001` (generated 0.9915, the most saturated in the repo; 26 min), then `smMATH001-a` (32 min), then
`smMATH01` (13 min per variant). `smMATH01`'s existing metric is fresh problems at the trained digit length and may
already discriminate — measure before changing it, and if it does, leave it alone and record that as the one
generator in the repo that was built right the first time.

**If phase 1 failed its gate**, this phase is instead the fallback: a **200-message hand-written dev set for the
router, disjoint from `test.json` and `test_arithmetic.json`**, written by a person, checked for leakage by a test,
and used for selection only. Budget 3 hours of writing. Its gate is the same as phase 3's, and its predicted effect
is the same 2 points; it is second choice only because it does not generalise to the other four projects without
paying the same hours again.

### Phase 6: docs

- [results.md](../research/results.md): dated entries for phases 1, 3 and 4, and the correction named in phase 4.
- [smEFFICIENCY_01](../../../smEFFICIENCY_01/README.md): the early-stopping section currently claims a saving that
  these trainers were not collecting. Add what patience actually did on each.
- Each touched project's README: the new `--val-split` flag, what the held-out-phrasings set is, and the sentence
  that the hand-written set never drives a decision.
- [whats-next.md](../../product/whats-next.md): plain-English row for "the scoreboard the models are picked by".
- [../README.md](../README.md): the "Keep the best checkpoint" ground rule gains a second half — *and keep a metric
  that can still tell checkpoints apart*.

## 8. Ground rules that apply

From [the engineering README](../README.md):

- **Don't train and serve on the same box.** Point the harness's prompted modules at `pc` and unload the Mac's model
  before every run in this plan.
- **One GPU job at a time on the Mac.**
- **Checkpoints get picked up live.** Phases 1 and 2 write to `out/hard-probe-*/` so they never touch `router.pt`.
  Phase 3 does write it; switch the Router module off first.
- **Leave the owner's page state alone.** Live checks use a scratch data folder.
- **Keep the best checkpoint, not the last** — which is exactly what this plan is trying to make mean something.
- **Don't edit anything under `smRTS_01/`.**

## 9. Why this and not the other open threads

Judged by whether finishing it makes other things possible.

1. **This one.** It unblocks a queued retrain that would otherwise publish a selected-on-test number, it makes the
   best-checkpoint machinery on five trainers do something, it makes `--patience` on ten trainers capable of firing,
   and it gives every future model in the repo a free automatic proxy for the scarce resource here — hand-written
   test messages. It is about 60 lines per project and four short training runs.
2. **smMATH001-a's digit stretch.** Genuinely the best science on the list: it divides at 38% and collapses past its
   trained digit length (6.7% on 8–10 digit addition) while smMATH01's abacus variant adds 8-digit numbers at 100%,
   and the two suspects — rotary positions in `model.py` versus training numbers capped at 6 digits — have never
   been separated. A 2×2 ablation answers it in about two hours of GPU. It is second and not first only because
   nothing downstream is waiting on it, and because it would be scored by the same saturating machinery this plan
   fixes. Worth noting for whoever picks it up: `smMATH_LANGUAGE_001`'s worst bug is *also* positional
   (`90000-1` read as `9000-1`, `1000000 * 3.5` as `100000*3.5`), so one answer may explain two models.
3. **The next harness module.** 8 of 11 modules still call the mid-sized model, and the Critic is the best candidate
   — it has a closed output space and a named known problem (it over-asks). But you cannot train it honestly without
   a selection metric that is not its own test set, which is this plan. It is downstream of item 1, not a rival
   to it.
4. **smCLM_02.** Fully planned in [smCLM_02.md](smCLM_02.md) and unbuilt. It is the biggest item on the list — six
   phases, about 5,000 teacher calls — and it needs a decision and a person's hours, not another plan. When it runs,
   its phase 3 gate ("keep the best epoch by held-out precision") inherits this problem on day one.
5. **smMATH_LANGUAGE_001's failing test.** One compound-unit case in `tests/test_reader.py`, a real checkpoint
   regression deliberately left visible. Fixing it closes a ticket and unlocks nothing; it should ride along with
   that model's next retrain, which is phase 5 here.

## 10. Open questions for the owner

1. **Is 26% the right hold-out fraction?** Bigger makes the metric harder and training thinner. P4's 2-point budget
   is a guess; phase 2 measures it, and the fraction can be tuned once — on the hard set's behaviour, never on the
   hand-written score.
2. **Should `--val-split phrasings` become the default for every trainer, or stay opt-in per project?** The plan
   flips the router at phase 3 and asks again at phase 5.
3. **smTOOLS_COMPUTER_CLI_01's published 46.8%.** It came from the unconditional-saver run, so it is not a
   selected-on-test number and nothing in the repo needs retracting. Confirm that reading before phase 4 writes the
   correction.

## 11. Rough budget

Guesses, to be corrected after phase 1.

| Phase | Machine time | Person time |
|---|---|---|
| 0: the split | none | 30 min reading 20 hard rows |
| 1: both metrics, 2 seeds × 10 epochs | about 55 min on the M5 | reading two tables |
| 2: what it costs | none extra | none |
| 3: router v4 | about 27 min | a live check on the page |
| 4: the CLI model | about 25 min | reading the step table |
| 5: the other three | about 85 min | one gate read per project |
| 6: docs | none | an hour |
