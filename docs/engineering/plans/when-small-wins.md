# when-small-wins plan: the rule that says which jobs to train a small model for, and the controls that could kill it

**Status: a plan. Nothing here is built.** Written 2026-09-17, checked against `smROUTER_01/`,
`smMATH_LANGUAGE_001/`, `smTOOLS_COMPUTER_CLI_01/` and their `out/` files on that date. The item-level
numbers in [section 2](#2-what-the-scoreboard-actually-says-item-by-item) were computed while writing this and
are reproducible from files already in the repo; the commands are given. Each phase has deliverables, a gate,
and what to write down. A phase that fails its gate is still a result, and it goes in
[results.md](../research/results.md).

**Nothing in this plan trains anything.** No GPU job on the MacBook Pro M5. Phases 0-2 and 3a are pure re-analysis
of files already
committed. Phases 3b, 4 and 5 need Ministral calls on the PC and CPU-only model runs.

Contents:

1. [The short version](#1-the-short-version)
2. [What the scoreboard actually says, item by item](#2-what-the-scoreboard-actually-says-item-by-item)
3. [The question, and the rule being tested](#3-the-question-and-the-rule-being-tested)
4. [Decisions already made](#4-decisions-already-made)
5. [Phases](#5-phases)
6. [Project layout](#6-project-layout)
7. [Ground rules that apply](#7-ground-rules-that-apply)
8. [Open questions for the owner](#8-open-questions-for-the-owner)
9. [Rough budget](#9-rough-budget)

---

## 1. The short version

The README says: *"a small model wins when the job is narrow, has a closed output space, and can be generated in
bulk for training. It loses when the job needs world knowledge."*

That sentence is the most valuable claim in the repo and it has never been tested. It was written by reading three
aggregate scores. It is also, on its face, contradicted by its own scoreboard: `smTOOLS_COMPUTER_CLI_01` has by far
the **largest** output space of the three models (arbitrary shell strings, not a label set) and it posts the
**largest** win over Ministral 8B (+31.9 points on exact command). Meanwhile the program name it must choose comes
from a closed 79-task catalog, and that is exactly where it **loses** by 14.9 points — same model, same 47 messages,
same run, opposite sign.

So "closed output space" is not the axis. This plan finds the axis that is, states it as a rule that predicts the
sign of a comparison **before** the comparison is run, and then tests that rule out of sample on the four cells the
scoreboard currently leaves as "not run".

It also runs three controls that could take the thesis apart, and the plan commits in advance to publishing them:

- **3a.** Ministral's CLI failures are mostly right-program-wrong-string (30 of 43). Re-score under a normaliser. If
  the +31.9-point lead is mostly the grader rewarding the model that was trained on the answer key's conventions,
  the repo's biggest single win is not a capability win.
- **3b.** Every comparison in the repo is *58,000 labelled examples* against *one prompt*. Give Ministral 32
  retrieved examples from the same generator. If the router's 12.9-point lead closes, the flagship result is a
  supervision effect, not a size effect, and the only unambiguous surviving claim is latency.
- **4.** On the math reader, **both models are wrong on zero of 40 messages**. The union is perfect. That says the
  right architecture may be neither "small" nor "big" but "small first, big when the small one is unsure" — and the
  confidence signal needed to do it already exists in all three models.

The valuable output is one page, `benchmarks/when_small_wins/RULE.md`, that a future plan reads *before* deciding to
spend 30 minutes of M5 time training something.

## 2. What the scoreboard actually says, item by item

`results.md` reports aggregates. The per-item picture is recoverable from files already committed: every test set has
unique `text` values (verified: 93/93, 40/40, 53/53 unique), and every `out/results.json` and `out/llm_baseline.json`
stores its full `mistakes` list. An item is wrong for a model exactly when its text is in that model's mistake list.

```bash
python3 - <<'EOF'
import json, collections
for name, folder, n_note in [
    ("Router all-three-right", "smROUTER_01", None),
    ("Math reader every-problem", "smMATH_LANGUAGE_001", None),
    ("CLI exact command", "smTOOLS_COMPUTER_CLI_01", "command"),
]:
    t = json.load(open(folder + "/test.json"))
    if n_note: t = [e for e in t if e.get(n_note)]
    sw = {m["text"] for m in json.load(open(folder + "/out/results.json"))["test_mistakes"]}
    lw = {m["text"] for m in json.load(open(folder + "/out/llm_baseline.json"))["mistakes"]}
    c = collections.Counter((e["text"] in sw, e["text"] in lw) for e in t)
    print(name, len(t), dict(c))
EOF
```

Result, 2026-09-17:

| Job | n | Both right | **Small right, 8B wrong** | **Small wrong, 8B right** | Both wrong | Oracle ceiling |
|---|---:|---:|---:|---:|---:|---:|
| Router, all three right | 93 | 41 | **25** | 13 | 14 | 84.9% |
| CLI, exact command | 47 | 3 | **19** | 4 | 21 | 55.3% |
| Math reader, every problem | 40 | 24 | 7 | **9** | **0** | **100%** |
| **Total** | **180** | 68 | **51** | **26** | 35 | |

Three things fall out of that table that no aggregate shows.

**(a) The disagreement set is where the whole story lives.** 77 of 180 items separate the two models. The aggregate
margins (+12.9, +31.9, −5.0 points) are just 51−26 sliced three ways. A rule that explains 77 items explains the
scoreboard.

**(b) The two models fail in opposite ways, and it is quantified.** Classifying each CLI miss by whether the first
non-`sudo` program token matches the expected one:

| | Misses | Right program, wrong string | Wrong program |
|---|---:|---:|---:|
| Ministral 8B | 43 | **30 (70%)** | 13 |
| smTOOLS_COMPUTER_CLI_01 | 25 | 8 | **17 (68%)** |

The big model knows *which tool* and writes it in a form the grader rejects. The small model writes a
well-formed command for the *wrong tool*. Example, from `llm_baseline.json`: "get nginx onto this box" →
Ministral wrote `sudo apt update && sudo apt install -y nginx`, graded wrong against `sudo apt install -y nginx`.
That is a convention failure, not a knowledge failure, and it is scored identically to writing `dnf` on Ubuntu.

**(c) The math reader is the only job where a selector would win.** Both-wrong is zero on all 40 messages. The union
of a 4.86M-parameter model and an 8B model is perfect, while the best single model is 82.5%.
`smMATH_LANGUAGE_001/out/confidence.json` already carries a per-item confidence, and it separates right from wrong:
AUC(`weakest`) = 0.828, AUC(`sure`) = 0.778 on n=40. An oracle threshold of `weakest < 0.87` hands 8 of 40 messages
to the 8B and scores **90.0%** — above both models alone. **That 90.0% is a ceiling chosen on the same 40 items it
scores, not a result.** Phase 4 is about whether it survives an honest threshold.

## 3. The question, and the rule being tested

**The question.** Given a description of a job, before any model is trained: will a small model trained here beat a
prompted Ministral 8B on it, and by roughly how much?

**The rule this plan pre-registers**, from section 2(b). Call it the **convention/selection split**:

> A generator teaches a small model the **convention** of the answer — its exact surface form, the house label set,
> the canonical string — essentially for free, because the generator *defines* that convention. It does not teach
> **selection** — which entity out of a large inventory the answer names — except to the extent the generator
> enumerates that entity across many phrasings. A prompted big model is the mirror image: it brings selection and
> has to guess the convention.
>
> So the small model's margin on a metric tracks how much of that metric is convention and how little is selection.
> Output-space *size* is irrelevant except through this: what matters is **phrasings per output class in the
> generator**, not the number of classes.

Worked against the existing data, which is where it came from and therefore is not evidence for it:

| Metric | Convention share | Selection load | Phrasings per class | Observed margin |
|---|---|---|---|---:|
| Router, all three right | high (labels are house-defined) | low (6 intents, 6 tools) | ~10,000 examples/class | **+12.9** |
| CLI, exact command | high (canonical flag string) | mixed | ~500/(task,platform), ~3-4 templates | **+31.9** |
| CLI, right program | none (the program is the answer) | high (79 tasks × 5 platforms) | ~3-4 templates | **−14.9** |
| Math reader, every problem | low (expression form is standard) | low, but needs faithful copying of arbitrary numbers | open | **−5.0** |

The rule is only worth anything if it predicts a cell it was not fitted to. Phase 5 is that test, and its gate can
fail.

**The competing explanations**, which phase 2 must be able to distinguish and which the plan does not get to dodge:

- **H-overlap:** it is nothing to do with convention or selection; the small model wins exactly on the test items
  whose phrasing happens to resemble the generator, and loses elsewhere. All three READMEs assert this ("overfits
  the generator", 98-99% generated vs 47-78% hand-written) and none has measured it per item.
- **H-grader:** the margins are an artifact of graders written alongside the generators. Control 3a.
- **H-supervision:** there is no small-vs-big effect at all; there is a supervised-vs-prompted effect. Control 3b.
- **H-nothing:** three jobs, three unrelated causes, no axis. This is a real possible outcome and phase 2's gate
  makes it reportable rather than something to explain away.

## 4. Decisions already made

Not up for debate inside the run. If one turns out to be wrong, stop and say so rather than working around it.

- **Nothing is trained. No M5 GPU job.** Every phase is re-analysis, CPU inference, or calls to the PC's Ministral.
  This plan can run while an M5 training job belonging to someone else is running.
- **New folder `benchmarks/when_small_wins/`.** No file under any `sm*/out/` or `checkpoints/` is written or
  modified, and nothing under `smRTS_01/` is touched at all.
- **Predictions are written down before the runs**, in `benchmarks/when_small_wins/predictions.md`, and reported
  next to the results the way [smRTS_01](smRTS_01.md) does it. A prediction added after seeing a number is not a
  prediction.
- **Labels are assigned with the answer key hidden.** Phase 1's labeller sees the message and the job description,
  never who got it right.
- **The normalisation rules for control 3a are fixed and written down before any re-scoring runs.** They decide the
  headline; choosing them after seeing the numbers would be choosing the headline.
- **Thresholds in phase 4 are chosen on generated held-out data only.** The hand-written set is the reported number
  and is never selected on. This is the same rule the router's best-epoch fix already operates under.
- **Ministral stays at temperature 0**, same model, same endpoint, same JSON schema as the committed
  `baseline_llm.py` in each project. The k-shot arm changes the messages array and nothing else.
- **A control that damages the thesis gets published in `results.md` with the same prominence as one that
  supports it.** That is the point of running it.

## 5. Phases

### Phase 0: rebuild the item-level table from what is already committed

**Deliverables**

| File | What it is |
|---|---|
| `benchmarks/when_small_wins/join.py` | Reads each project's `test.json`, `out/results.json` and `out/llm_baseline.json` and writes one row per item |
| `benchmarks/when_small_wins/data/items.jsonl` | `job`, `text`, `want`, `small_right`, `llm_right`, `rules_right` (where a no-model baseline exists), and `small_got` / `llm_got` where the file stored them |
| `benchmarks/when_small_wins/tests/test_join.py` | Asserts the reconstructed per-job accuracies equal the published ones |

**Gate.** The reconstruction reproduces `results.md` exactly: router 0.710 / 0.581, math reader 0.775 / 0.825, CLI
exact 0.468 / 0.149, CLI quiet 1.000 / 0.500. Any mismatch means the join is wrong; fix it before going on. (Checked
while writing this plan: it reproduces. The test is there so it keeps reproducing.)

**Also fix the root cause.** Every `baseline_llm.py` and every trainer stores only `mistakes`. Add an `all_guesses`
list to each, so that no future item-level question needs this reconstruction. Small, mechanical, and it means
`small_got` exists for correct items too — which phase 3a needs and currently cannot have.

**Write down.** The 2×2 per job and the totals. If they differ from section 2, section 2 was wrong; say so.

### Phase 1: pre-register the labels, then label 180 items blind

**Two binary labels, decided from the message and the job description alone.** Written into
`benchmarks/when_small_wins/labels.md` with three worked examples each, before labelling starts:

- **`convention`** — would a competent human give an answer a reasonable person accepts, that this grader marks
  wrong? (Ministral's `sudo apt update && sudo apt install -y nginx`: yes. Router intent `question` vs `task` on
  "Can you add up 129, 88 and 406 for me": yes, arguably. `2+2` → `2+2`: no.)
- **`selection`** — does the correct answer name an entity that cannot be derived from the message and must be
  looked up: a program, a flag, a physical constant, a house label whose boundary is a judgement call?

**One mechanical covariate.** `overlap.py` generates 50,000 examples per job from that project's own
`data.generate(n, seed=0)` — CPU, deterministic, already in the repo — indexes 3-gram shingles, and records each test
item's maximum shingle Jaccard against the generated pool. This is H-overlap made measurable.

```bash
python3 benchmarks/when_small_wins/overlap.py --job router --n 50000 --seed 0
```

**Gate.** Two independent labellings — the owner and a second agent reading cold — agree on at least **85%** of the
180 items on each label. Disagreements are adjudicated in writing or the item is dropped, and the drop is recorded.
**If agreement is below 70% on either label, the phase fails:** an axis nobody can apply consistently is not a rule,
and that is the result. Write it up and stop.

**Write down.** Agreement per label, the adjudicated items, the marginal counts (how many items are
convention-only, selection-only, both, neither), and the overlap distribution per job.

### Phase 2: fit the smallest rule that fits, and state what would have killed it

n = 180, so this is contingency tables and at most a 3-predictor logistic regression. Anything more is fitting noise.

**Pre-registered predictions.** Base rate over the 77 disagreement items is 51:26, about 2:1 to the small model.

- **P1.** Among disagreements, items labelled `convention` and not `selection` go to the small model at **≥ 3:1**.
- **P2.** Among disagreements, items labelled `selection` go to Ministral at **≥ 2:1**.
- **P3.** `overlap` predicts `small_right` with a coefficient at least **2× its standard error**, and does **not**
  predict `llm_right` (coefficient under 1× its standard error).
- **P4.** With `convention`, `selection` and `overlap` in one model, `selection` carries a larger standardised
  coefficient on the disagreement direction than `overlap` does. (This is the substantive bet: the axis is the job,
  not just the phrasing luck.)

**Gate and falsifiers.**

- **F1 — no axis.** If P1 and P2 both land within **1.5×** of the 2:1 base rate, the labels have no predictive
  content. Then the honest write-up is: the mixed scoreboard is three jobs with three unrelated causes, the README's
  *"the pattern that holds up"* paragraph has no support and is cut, and phase 5 is cancelled. **This is a real
  possible outcome and it gets published.**
- **F2 — the generator story is wrong.** If P3 fails, then "it overfits its generator", asserted in three READMEs,
  is not visible in the hand-written sets, and those READMEs need a correction. The 99%-generated-vs-47%-handwritten
  gap would then be about difficulty, not phrasing coverage.
- **F3 — the axis is really P4 inverted.** If `overlap` dominates `selection`, the rule becomes "a small model wins
  where your generator already covers the phrasing", the binding constraint is human hours spent writing phrasing
  templates, and specialisation per se buys nothing. That is a narrower thesis than the README's and it should
  replace it.

**Write down.** The contingency tables, the regression with standard errors, which falsifier fired, and the ten
disagreement items the rule gets most wrong with a sentence each.

### Phase 3: the two controls that could take the win away

#### 3a. Is the CLI win a grader artifact? (zero inference, zero GPU)

Every miss stored a `got` string. Normalisation can only turn wrong into right, never right into wrong, so
re-scoring the stored misses is exact and complete without rerunning either model.

**The normalisation rules, fixed before running** (`benchmarks/when_small_wins/normalise.py`, with the owner's
sign-off per [section 8](#8-open-questions-for-the-owner)): a leading `sudo apt update &&` or equivalent refresh;
flag order within a single program; documented long/short flag equivalents (`-y`/`--yes`, `-l`/`--long`); quoting
style where the shell word-splits identically; trailing `.` vs `./` for the current directory. **Not** normalised:
a different program, a different subcommand, an added or removed destructive flag, or an inverted action
(`install` vs `remove`).

Score three ways and report all three: strict (today's number), normalised, and an owner pass over the remaining
misses marking "a shell user would accept this".

- **P5.** Ministral's normalised exact rises from **14.9%** to **40-55%**.
- **P6.** The small model's normalised exact rises from **46.8%** to **50-60%**.
- **P7.** The lead falls from **+31.9** points to **under +10**.

**F4 — the headline win is convention, not capability.** If the lead falls below **+5** or inverts, then the largest
win in the repo is mostly "the small model was trained on the grader's conventions". The README's CLI row is
rewritten, the result goes in `results.md`, and the general claim becomes: *a small model reliably learns the house
convention; it does not out-perform an 8B at the job.* If instead the lead stays above **+25** under normalisation,
the win is stronger than currently reported and should be stated more confidently.

#### 3b. Is it small-vs-big, or supervised-vs-prompted?

Every comparison in this repo pits 58,000-300,000 labelled examples against a single prompt. That is not a test of
model size. Give the 8B a comparable, if much smaller, dose of the same supervision.

`benchmarks/when_small_wins/kshot.py` takes each project's committed `baseline_llm.py` and changes only the
`messages` array: prepend **32 examples retrieved by BM25** over the test message from 50,000 examples drawn from
that job's own `data.generate(50000, seed=1)` — a different seed from phase 1's pool. Same system prompt, same
schema, same model, temperature 0. Examples are drawn from the generator, never from `test.json`; a unit test asserts
no retrieved example's text appears in the test set.

```bash
SMMOL_LLM_URL=http://<pc>:8081 python3 benchmarks/when_small_wins/kshot.py --job router  --k 32 --seed 1
SMMOL_LLM_URL=http://<pc>:8081 python3 benchmarks/when_small_wins/kshot.py --job mathlang --k 32 --seed 1
SMMOL_LLM_URL=http://<pc>:8081 python3 benchmarks/when_small_wins/kshot.py --job cli      --k 32 --seed 1
```

Run `--k 0` first as a reproduction check of the committed baseline, and `--k 8` as a dose-response point.

- **P8.** Router all-three-right: **58.1% → 68-75%**.
- **P9.** Math reader every-problem: **82.5% → 84-90%**.
- **P10.** CLI normalised exact: **+10 to +20 points** over its 3a number.
- **P11.** k=8 sits between k=0 and k=32 on all three jobs (a monotone dose-response; if it does not, the effect is
  not supervision and P8-P10 mean something else).

**F5 — the flagship result is a supervision effect.** If router k-shot reaches **68%** or more, the 12.9-point
accuracy win over a prompted 8B does not survive giving the 8B 32 examples. The honest claim then becomes: *a 1.87M
model matches a well-adapted 8B on routing, at 2.04 ms against 1.99 s and on hardware you own.* That is still a real
claim — it is the latency and cost claim, which is arguably the thesis's better half — but it is **not** "beats an
8B at routing by 13 points", and the README's table and its bolding must change. The plan commits to making that
change if F5 fires.

**The converse matters too.** If k=32 moves the router by under 3 points, the comparison as published is fair, the
8B genuinely cannot do this job from examples, and the thesis is in better shape than it currently claims.

**Gate.** All three jobs run at k ∈ {0, 8, 32}, the k=0 arm reproduces the committed baseline within 2 points, and
the leak test passes. Roughly 180 calls per job per k, about 1,600 calls, resumable in batches.

### Phase 4: small first, big when unsure — the constructive half

All three small models already emit confidence: `route.py` returns `intent_p` and `tool_p`, `read.py` and `cli.py`
return `sure` and `weakest` from `read_with_confidence`. None of it is stored per item today.

**Step 1.** `confidences.py` runs each small model on its own test set on the CPU (2.04 ms, 28.9 ms and 47.1 ms per
item — the whole thing is seconds) and adds the confidence columns to `items.jsonl`. Nothing under `out/` is written.

**Step 2.** Pick a threshold per job on **generated held-out data only** — each trainer already builds one — using
that data's own right/wrong labels. Then apply the frozen threshold to the hand-written set, hand those items to the
stored Ministral answer, and score.

**Step 3.** Report, per job: AUC of each confidence signal, the oracle ceiling (both-wrong count), the
generated-chosen threshold, the hand-off rate, the resulting score, and the average latency
(`(1−rate)·small_ms + rate·llm_ms`).

**Pre-registered.** The oracle numbers are known and are **not** predictions: math ceiling 100% with a best
in-sample hand-off of 90.0% at 8/40 handed off; router ceiling 84.9%; CLI exact ceiling 55.3%.

- **P12.** With a generated-chosen threshold, the math reader hand-off scores **82-88%** at a hand-off rate of
  **≤ 30%**, i.e. it matches or beats Ministral alone at an average latency under **1.0 s** against 3.13 s.
- **P13.** The router hand-off scores **72-80%**, above both 71.0% and 58.1%.
- **P14.** The CLI hand-off does **not** beat the small model alone by more than 3 points, because its oracle
  ceiling is only 55.3% and its errors are selection errors the model is confidently wrong about.

**F6 — the hand-off idea is dead.** If no generated-chosen threshold beats the better single model on **any** of the
three jobs, then small-model confidence is not usable for routing to a bigger model, and the plan says so rather
than tuning until it works. Report the AUCs either way; AUC(`weakest`) = 0.828 on math, n = 40, is the only one
measured so far and n = 40 is small.

**Why this phase is the constructive one.** If P12 and P13 hold, the thesis stops being "small beats big" — a claim
the scoreboard only half-supports — and becomes "small first, big on low confidence, beats either alone", which the
harness can actually implement and which survives F5 even if F5 fires.

### Phase 5: the out-of-sample test — predict the unfilled cells, then fill them

This is what separates a rule from a re-reading of results already in hand. `results.md`'s scoreboard has four cells
marked "not run". **Before any of them is run**, `predictions.md` records, for each, the rule's predicted Ministral
score, the predicted sign of (small − Ministral), and one sentence of reasoning from the rule.

| Cell | Small model's number | What the rule must predict |
|---|---|---|
| Router, bare arithmetic, 17 messages | 94.1% | sign and a ±10-point band |
| smCLM_01, held-out word ideas, precision@k | 0.818 | sign and a ±0.15 band |
| smMATH01 / smMATH001-a, arithmetic at trained digit lengths | 100% / 38% divide | sign per operation |
| smALLM_01, gadget world, all 8 right after 20 experiments | 83% | sign and a ±10-point band |

Then run each as a `baseline_llm.py`-shaped script against the PC's Ministral, same schema discipline. The gadget
world needs the 20 experiment results serialised as text and the 8 questions asked in one call; it is the most
harness work and is the one to drop if time runs short.

**Gate.** The rule gets the **sign** right on at least **3 of 4** cells.

**F7 — the rule has no predictive value.** Two or more sign errors and the rule is retrospective storytelling. It is
then published as a negative result: *the repo's mixed scoreboard has no rule behind it that generalises to a new
job; decide each job by running the baseline first.* That conclusion is worth having and is cheap to act on — run
`baseline_llm.py` before `train.py`, always.

**Write down.** The four predictions as written, the four results, the sign table, and each miss explained.

### Phase 6: write up, and leave one page behind

- **`benchmarks/when_small_wins/RULE.md`** — one page. The rule in three sentences, the table of what predicted what,
  the falsifiers that fired, and a four-line checklist a future plan runs before deciding to train a model. If F1 or
  F7 fired, the page says so and the checklist becomes "run the 8B baseline on 40 hand-written items first; it costs
  two minutes and you will not need this page."
- **`results.md`** — dated entries for phases 2, 3a, 3b, 4 and 5, each with its falsifier's outcome.
- **`README.md`** — the *"the pattern that holds up"* paragraph is rewritten to whatever phase 2 found, or deleted.
  If F4 fired, the CLI row gets a normalised column. If F5 fired, the table's bolding and the sentence *"the router
  beats an 8B model at routing, by 13 points"* change to the latency claim.
- **`whats-next.md`** — a line for the hand-off architecture with its real status.
- **`docs/engineering/README.md`** — the plans table gets this file.

## 6. Project layout

```text
benchmarks/when_small_wins/
  README.md
  RULE.md               the one page this whole plan exists to produce (phase 6)
  predictions.md        every P-number, written before its run
  labels.md             the two label definitions with worked examples (phase 1)
  join.py               test.json + out/*.json -> data/items.jsonl (phase 0)
  overlap.py            shingle overlap of each test item against the job's own generator
  fit.py                contingency tables and the <=3-predictor regression (phase 2)
  normalise.py          the fixed command-equivalence rules (phase 3a)
  rescore.py            strict / normalised / owner-accepted, over stored misses
  kshot.py              baseline_llm.py with 32 BM25-retrieved generator examples (phase 3b)
  confidences.py        CPU runs to add confidence columns to items.jsonl (phase 4)
  handoff.py            generated-chosen threshold, then applied to the hand-written set
  unfilled/             one baseline script per phase-5 cell
  tests/                join reproduces published scores; no retrieved example is in any test set
  data/                 items.jsonl, overlap.jsonl, generated pools (gitignored), logs/
  out/                  tables, the regression, eval.md
```

## 7. Ground rules that apply

- **Nothing under `smRTS_01/` is read-modified-written.** Its sources are sha256-hashed into Phase 2 run manifests.
  Read it if useful; do not edit it.
- **No M5 GPU job, and no training of any kind.** This plan is designed to be runnable while someone else's training
  run owns the Mac. If a phase seems to need training, it has drifted; stop.
- **No file under any `sm*/out/` or `checkpoints/` is written or modified.** Everything lands under
  `benchmarks/when_small_wins/`. The one exception is the phase-0 `all_guesses` change to the `baseline_llm.py` and
  trainer *source* files, which changes what future runs write, not what past runs wrote.
- **Teacher and baseline calls go to the PC's Ministral** over `SMMOL_LLM_URL`, in resumable batches, expecting to
  pause when the family portal wants the RX 580. Nothing here loads a model on the Mac.
- **Leave the owner's page state alone.** Nothing in this plan touches the harness's live data.
- **Select on held-out data, report on the hand-written set.** The same rule the router's best-epoch fix operates
  under, and the reason phase 4 does not get to pick its threshold on the 40 messages it reports.

## 8. Open questions for the owner

1. **The normalisation rules in 3a decide the headline.** Sign them off before any number is computed, not after.
   Specifically: does `sudo apt update && sudo apt install -y nginx` count as the same command as
   `sudo apt install -y nginx`? The answer moves Ministral's CLI score by roughly 15 points on its own.
2. **Is 32 the right k for 3b?** 32 examples is a fair-ish dose against 58,000 and fits an 8B's context. If the
   answer is "any k is unfair either way", say so and 3b becomes a dose-response curve reported without a verdict.
3. **Labelling budget.** 180 items × 2 labels, done twice independently, is about 2 hours of person time. Is that
   available, and who is the second labeller?
4. **Are you willing to publish F4 and F5 if they fire?** The repo is public. F5 in particular downgrades the
   headline sentence in the README. The plan assumes yes, because the README's existing honesty — the losses are
   already in the table — says yes. Confirm before phase 3 runs.
5. **When is the RX 580 free** for roughly 1,600 calls in phase 3b plus a few hundred in phase 5?

## 9. Rough budget

Guesses, to be corrected after phase 2.

| Phase | Machine time | Person time |
|---|---|---|
| 0: rebuild the item table | seconds (CPU) | 1 hour writing `join.py` and the `all_guesses` change |
| 1: labels and overlap | ~10 min CPU generating 150k examples | 2 hours labelling, twice |
| 2: fit the rule | seconds | 1 hour reading the tables and the ten worst items |
| 3a: normalised CLI scoring | none | 30 min fixing the rules, 30 min on the accept pass |
| 3b: k-shot Ministral | ~1,600 PC calls, 1-2 hours on the RX 580 | 1 hour writing `kshot.py` |
| 4: confidence hand-off | seconds CPU, no new LLM calls | 1 hour |
| 5: the four unfilled cells | a few hundred PC calls; smALLM_01 is the expensive one to wire | 2-3 hours, mostly smALLM_01 |
| 6: RULE.md and the doc updates | none | 2 hours |
