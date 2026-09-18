# smCLM_02 plan: ideas from real sentences, and memory that searches by meaning

**Status: a plan. Nothing here is built.** Written 2026-09-17, checked against `smCLM_01/`, `paratroop_harness_02/`
and the corpora named below on that date. It's meant to be handed to agents and executed phase by phase. Each phase
has deliverables, a gate, and what to write down. Don't skip a gate; a phase that fails its gate is still a result,
and it goes in [results.md](../research/results.md).

Contents:

1. [The short version](#1-the-short-version)
2. [Where smCLM_01 left off](#2-where-smclm_01-left-off)
3. [The two questions this run answers](#3-the-two-questions-this-run-answers)
4. [Decisions already made](#4-decisions-already-made)
5. [Phases](#5-phases)
6. [Project layout](#6-project-layout)
7. [Ground rules that apply](#7-ground-rules-that-apply)
8. [Open questions for the owner](#8-open-questions-for-the-owner)
9. [Rough budget](#9-rough-budget)

---

## 1. The short version

smCLM_01 showed a tiny model can work out what a noun means, as a bundle of named ideas, purely from how the noun is
used, in a hand-made world of 286 nouns and 79 sentence templates. smCLM_02 does the same thing on real sentences with
a vocabulary of about 2,500 words and about 150 ideas, then puts the result to work in paratroop_harness_02 as the
thing [whats-next.md](../../product/whats-next.md) calls "memory that searches by meaning": a query for "doctor" finds
the memory that only says "dentist". No downloads: the teacher is the same Ministral 8B the harness already uses, and
the model is trained here.

The run is six phases. Test sets are written by a person first, before any data is generated, and the harness
integration follows [adding-a-module.md](../paratroop_harness/adding-a-module.md) step by step.

## 2. Where smCLM_01 left off

From [smCLM_01.md](../smModels/smCLM_01.md) and `smCLM_01/out/results.json` (2026-09-15):

| | |
|---|---|
| World | 51 ideas, 286 hand-tagged nouns, 79 templates, 23 held-out nouns whose ideas are never given |
| Idea reader | a noun comes in only as its idea vector; it predicts the ideas of a hidden noun; held-out nouns start at zero and take the average prediction after each epoch |
| Result | held-out precision@k **0.818 ± 0.010**, average precision 0.932 |
| Control | a normal word model, read the right way, scores the same (0.821) |
| Weak spots | ideas no sentence shows directly: 5 of 9 found. Wrong ideas that fit the sentences: helmet → soft |
| Not done | real sentences, verbs and adjectives, a bigger inventory, the harness |

**What that means for v2.** Accuracy alone won't justify the idea reader; the word reader ties it. The reasons to build
it are the two things a word model can't give: a readable, named vector for any text, and a sensible vector for a word
it has never seen, from context alone. Both are exactly what memory search needs. v2 is designed around those two
properties, and it's scored against the honest baselines: the harness's current BM25, and a plain tag dictionary with
no model at all.

## 3. The two questions this run answers

1. **Does learning ideas from usage survive real language?** On real sentences, with a vocabulary ten times bigger,
   including verbs and adjectives, does a held-out word still get most of its ideas right?
2. **Does searching memory by ideas beat searching by words?** Against the harness's BM25 and against a no-model tag
   lookup, on a hand-written test where the query and the memory share no words.

A "no" to either is a usable result. If the tag dictionary does as well as the model on question 2, the harness should
get the dictionary; it's still a win over BM25 and it needs no checkpoint.

## 4. Decisions already made

These aren't up for debate inside the run. If one turns out to be wrong, stop and say so rather than working around it.

- **New folder `smCLM_02/`. `smCLM_01/` is untouched** and stays as the record of the first result.
- **Ideas stay named.** The inventory is human-approved and every one has a one-line meaning, like v1's
  `concepts.json`. It includes all 51 of v1's ideas, verbatim, so v1's world is still a valid test.
- **The teacher is Ministral 8B on the PC**, through `http://127.0.0.1:8081` with a strict JSON schema, the way
  `smCONVERSATION_001/teacher.py` does it. Every call is logged to JSONL. **No downloads**: no WordNet, no nltk, no
  embedding model. (WordNet's supersenses would be a cheaper inventory. Not chosen: it's a download, and 26 noun
  classes are too coarse for ideas like danger, speed or health.)
- **Tags are soft.** Every word is tagged in three teacher passes with different seeds; an idea's weight is the fraction
  of passes that gave it (0, ⅓, ⅔, 1). The reader trains against those weights with binary cross-entropy, as v1 does
  against 0/1.
- **Held-out truth is written by a person,** never by the teacher.
- **The job in the harness is memory by meaning,** plus idea tags on observations for the page. Not Attention, not the
  Planner; those are stretch items in [phase S](#phase-s-stretch-only-after-the-phase-4-gate).
- **Memory search becomes hybrid:** BM25 plus a weighted cosine over idea vectors. With the Ideas module off, search is
  pure BM25 and behaves exactly as today. "Invisible when off" is a rule of the harness, not a preference.
- **Test sets first.** The hand-written recall pairs and held-out word tags are written before any data is generated,
  and a unit test proves nothing in them appears in the training data.
- **The loader runs on the CPU.** Training may use the MacBook Pro M5's GPU, with the Mac's Ministral
  unloaded first.

## 5. Phases

### Phase 0: freeze, and write the tests first

**Why first.** Every model here so far has overfit its generator (router 98% on generated vs 71% on hand-written; math
reader 99% vs 78%). The only defence is a test written by a person, in their own words, before the generator exists.

**Deliverables**

| File | What it is |
|---|---|
| `smCLM_02/test_recall.json` | About 60 cases. Each has a `query`, a `pool` of 8 to 12 stored memories (the harness's `recall_top` is 8), and the `id` of the one that should come first. Three groups: **A, word overlap** (the query and the memory share a content word; BM25 should already win and the hybrid must not lose); **B, no overlap** (same meaning, different words: "who looks after my teeth" → "User's dentist is Dr. Lee"; "my kid" → "daughter"; "hiking" → "walks in the hills"); **C, distractors** (a pool where another memory shares words with the query but not its meaning: "Sunday is shopping day" when the query is about the dentist). Phrased unlike anything Ministral writes |
| `smCLM_02/test_words.json` | The held-out words and their ideas, filled in during phase 1 by a person, using only ideas from the approved inventory. Starts as an empty list |
| `smCLM_02/tests/test_leaks.py` | Fails if any `query` or `pool` text from `test_recall.json` appears in `data/sentences.jsonl`, or any held-out word's tags appear in `data/tags.jsonl` |

**Gate.** A second person (or a second agent reading cold) can say which memory each query should find, without
seeing the answer key, on at least 55 of the 60 cases. Cases they disagree on are rewritten or dropped.

**Write down.** How many cases per group, and who wrote them.

### Phase 1: the vocabulary and the idea inventory

**The numbers behind the targets** (measured 2026-09-17 on `smCONVERSATION_001/data/accepted/conversations.jsonl`, 248
conversations, 5,946 messages, 128,330 words):

| | |
|---|---|
| Unique content words (not function words, 3+ letters) | 7,059 |
| Used 5 or more times | 2,688 |
| Used 30 or more times (v1 needed more than 30 usages per held-out word) | 534 |
| Of v1's 286 nouns, used 10 or more times | 43 |
| Of the 600 accepted curriculum words, used 10 or more times | 118 |

So the conversation corpus alone covers a few hundred words well, and it's skewed toward planning talk (its top words
are plan, focus, step, minutes). Phase 2 fills the rest with teacher-written sentences.

**Vocabulary.** `vocabulary.py` builds `data/vocabulary.jsonl` from three sources, unioned:

1. content words used 5 or more times in the conversation corpus;
2. the 600 accepted words in `smLANGUAGE_en_SCH_001/data/curriculum.jsonl`;
3. the 286 nouns in `smCLM_01/concepts.json`.

Rules: lowercase; letters only; 3 or more letters; not in the function-word list (start from `curriculum.py`'s
`FUNCTION_WORDS`); a word that's capitalised in most of its uses is a name and is left out. Fold `-s`, `-es`, `-ed`,
`-ing` to the base form only when the base is itself in the vocabulary; otherwise keep the surface form. There's no
lemmatiser here and that's fine; write the folding rules in one function that the encoder reuses. Target: about 2,500
words. Each row: `word`, `sources`, `count`, `pos` (noun, verb, adjective, other, filled by the teacher in the tagging
pass). **Verbs and adjectives are in.** The model treats every content word the same way: in as ideas, never as a word.

**Inventory.** `ideas.json`, an object of `idea: one-line meaning`, in this order: v1's 51 verbatim, then about 100
new ones.

- `propose_ideas.py` asks the teacher, for batches of 20 vocabulary words, "what general ideas does each of these words
  carry?" from an *open* list, and tallies the answers into `data/idea_proposals.json`.
- **A person picks** the new ideas from that tally and writes each one's meaning. Aim for coverage of: domains (health,
  money, school, work, home, travel, food, technology, family, time, weather, sport, art), properties (size, speed,
  danger, temperature, texture), and roles (person, place, thing, event, action, feeling). Every idea must fit at least
  8 vocabulary words and at most 40% of them.

**Tagging.** `tag_words.py` runs three passes over the vocabulary, 20 words per call, with seeds 16001, 16002, 16003,
schema: for each word, `pos` and 1 to 10 `ideas` from an `enum` of the inventory. Output `data/tags.jsonl`:
`{"word": "...", "pos": "...", "ideas": {"health": 1.0, "person": 0.67}}`. Reject and retry a call whose words don't
match the ones sent. Log every call.

**Held-out words.** `held_out.py` picks, with a fixed seed, about 60 vocabulary words stratified by frequency band
and part of speech, plus v1's 23 held-out nouns wherever they're in the vocabulary. **A person tags all of them** in
`test_words.json`; their teacher tags are dropped from `data/tags.jsonl`. (v1's 23 already have human tags in
`concepts.json`; reuse them, extended with any new ideas that apply.)

**Review.** A random sample of 100 tagged words, checked line by line. More than 10 wrong means fix the prompt and
rerun the pass; don't hand-fix rows.

**Gate.**

- Inventory: at least 120 ideas, every one used by at least 8 words, none by more than 40%.
- Every vocabulary word has at least one idea at weight ⅔ or more.
- The 100-word sample passes.
- `test_words.json` is complete and human-written.
- `data/manifest.json` records the SHA-256 of every data file, like `smLANGUAGE_en_SCH_001` does.

**Write down.** Vocabulary size by source; inventory size; agreement between the three passes (share of ideas at
weight 1 vs ⅓); the sample's error rate and the kinds of errors.

### Phase 2: the sentences

**Target.** At least 30 usages of every vocabulary word, and around 100,000 sentences.

**Sources**, in this order:

1. **The conversation corpus,** split into sentences (about 11,800), both user and assistant turns, kept when 4 to 30
   words long.
2. **The 600 curriculum sentences.**
3. **Teacher-written sentences** for every word still under 30 usages. `generate_sentences.py`: per word, ask for 10
   sentences per call that use the word in different settings, roles and tenses; one sentence each; 6 to 25 words; no
   lists, no markdown. Structural checks: contains the word or a folded form; length band; not a duplicate, exact or
   near (reject a sentence whose 3-word shingles are 80% shared with one already kept); not in `test_recall.json`.
   Resumable in batches, every accepted batch saved before the next request, as `smLANGUAGE_en_SCH_001/generate_data.py`
   does. Expect about 2,000 words × 25 sentences, about 5,000 calls, several hours on the RX 580.

**Not in training:** v1's template sentences. They're kept only as a regression test (phase 3). Training on them would
teach template-speak.

**Held-out words** appear in the corpus like every other word. That's the point. Their tags are hidden, not their
sentences. Make sure each has at least 30 usages.

**Deliverables.** `data/sentences.jsonl` (`text`, `source`, `tokens`), updated `data/manifest.json`, and
`compile_data.py` that builds them and runs the leak test.

**Gate.** Every vocabulary word at 30 or more usages; every teacher sentence passed the structural checks;
`tests/test_leaks.py` passes; a person reads 50 random teacher sentences and finds at most 5 that a native speaker
would call wrong or nonsense.

**Write down.** Sentence counts by source, calls made, rejections by reason, wall time.

### Phase 3: the reader

**Model** (`model.py`, from v1's `ConceptModel` and `WordModel`):

- A content word in the vocabulary comes in as its idea vector (about 150 weights, 0 to 1) through a linear layer.
  Function words and unknown words come in as embeddings: the 1,500 most frequent non-content words are "frames", the
  rest are `UNK`. The idea reader never sees a content word's identity.
- Pre-norm transformer encoder, d 192, 4 layers, 6 heads, `max_len` 24. A sentence longer than 24 tokens is windowed
  around the target. About 2 to 3M parameters; print the count.
- The word reader control stays, with its two fair readings from v1 (nearest words; words that fit the same spots).

**Training** (`train.py`):

- One content word hidden per example; up to 3 examples per sentence per epoch, so short sentences aren't over-sampled.
  Loss: binary cross-entropy against the soft tags.
- Held-out words start at zero and take the average prediction wherever they were hidden, after each epoch, exactly as
  v1's `infer_ideas()` does.
- AdamW, lr 2e-3 first (v1's), 1e-3 if it's unstable; 8 epochs; batch 512; 3 seeds. Keep the best epoch by held-out
  precision, not the last. Save `out/ideas.pt` in v1's checkpoint shape plus `vocabulary`, `frames`, `ideas` (the
  inventory), `tags`, the learned held-out vectors, and the folding rules' version.
- MPS is allowed. Before a long run, point the harness's prompted modules at `pc` and unload the Mac's model
  (see [the ground rules](#7-ground-rules-that-apply)).

**Scores** (`evaluate.py`, written to `out/eval.md` and `out/results.json`):

1. **Held-out words:** precision@k and average precision against `test_words.json`, mean and SD over 3 seeds, for the
   idea reader, both word-reader readings, and most-common-ideas.
2. **Implied vs shown:** for each held-out word, split its true ideas into those some training sentence makes obvious
   and those it doesn't (v1's method), and score both.
3. **v1 regression, report only:** run the reader on v1's 80,000-sentence template world, restricted to the 51 shared
   ideas, and score the 23 v1 held-out nouns. v1's frames ("steers", "petrol") may be `UNK` here, so a drop is
   expected. Report the number; it isn't a gate.

**Gate.** Idea reader held-out precision@k of at least **0.70** averaged over 3 seeds, and at least 0.40 above
most-common-ideas. (0.70 is a guess set below v1's 0.82 because real language is messier; revise it after the first
seed and say so in the results.) If the word reader ties again, that's expected; write it down and carry on. The
reasons to keep the idea reader are readability and the unknown-word path, not this number.

**Write down.** Parameter count, seconds per seed, the table, the worst 10 held-out words with what went wrong, and
whether templates-vs-real changed the shown/implied gap.

### Phase 4: the text encoder, and recall by meaning

**The encoder** (`encode.py:Encoder`, the class the harness will load):

```python
class Encoder:
    def __init__(self, path):          # torch.load(path, map_location="cpu"), eval()
    def encode(self, text):            # -> {"vector": [float] * n_ideas, "about": [[idea, weight], ...] (top 6),
                                       #     "words": [...], "unknown": [...]}
    def similarity(self, a, b):        # cosine of two vectors
```

Steps inside `encode()`: tokenise and fold with the same function training used; a content word in the vocabulary
contributes its tag vector; an unknown content word (a name like "Lee", a word outside the vocabulary) is hidden and
its ideas are inferred from context by the reader, v1's `--teach` trick; the sentence vector is the element-wise max
over words (what v1's `--read` does), with a `--pool mean` flag for comparison; then unit-normalised. `python3
encode.py "who looks after my teeth"` prints the top ideas and the unknown words, the quickest manual check.

**Hybrid ranking**, implemented once in `evaluate.py` and copied into the harness in phase 5:

```text
score = bm25 / max_bm25_for_this_query        (0 when there's no word match)
      + ideas_weight * cosine(query, memory)   (ideas_weight defaults to 1.0)
```

Use the harness's own `MemoryStore.search` BM25 for the first term, copied not rewritten, so the numbers are the
harness's numbers.

**Scores** on `test_recall.json`, per group (A overlap, B no overlap, C distractors): recall@1, recall@3 and mean
reciprocal rank for four rankers:

| Ranker | Model? |
|---|---|
| BM25 alone (today's harness) | no |
| Tag dictionary: OR the tags of known words, unknown words contribute nothing, cosine | no |
| Idea reader: `Encoder.encode`, cosine | yes |
| Hybrid: BM25 + idea reader | yes |

Also time `encode()` on the CPU over 200 texts.

**Gate.**

- Hybrid recall@1 on group A is at least BM25's (no regression on what works today).
- Hybrid recall@1 on group B beats BM25's by at least 0.25.
- The idea reader beats the tag dictionary on group B. If it doesn't, the harness gets the dictionary instead of the
  model in phase 5, and the plan says so; it's still a win over BM25.
- `encode()` takes at most 20 ms per text on the CPU (the router takes 2 ms, the math reader 29 ms).

**Write down.** The 4 × 3 table, the group C failures (what fooled it), the speed, and the chosen `ideas_weight`.

### Phase 5: the harness

Follow [adding-a-module.md](../paratroop_harness/adding-a-module.md) in order. The specifics for this module:

| Step | What to do |
|---|---|
| 1. Loader | `smCLM_02/encode.py:Encoder`, checkpoint `smCLM_02/out/ideas.pt`, CPU only, `__main__` prints one result. Test it with a tiny random checkpoint in a temp folder |
| 2. `LOCAL_CODE` | `"ideas": "encode.py:Encoder"` |
| 3. Provider | `"smCLM_02": {"path": "../smCLM_02/out/ideas.pt", "kind": "ideas"}` under `providers.smmol.models` |
| 4. Module | `{"id": "ideas", "name": "Ideas", "color": "indigo", "kind": "ideas", "provider": "smmol", "model": "smCLM_02", "does": "Small trained model: reads the ideas in the message and in memories, so recall works by meaning"}`. Place it in `modules` after `attention` and before `recall`. That makes 12 modules; the page test's `len(state["modules"]) == 11` becomes 12 |
| 5. Contract | `"ideas": {"type": "object", "required": ["about"], "properties": {"about": {"type": "array", "maxItems": 6, "items": {"type": "object", "required": ["idea", "weight"], "properties": {"idea": {"type": "string", "maxLength": 24}, "weight": {"type": "number", "minimum": 0, "maximum": 1}}}}}}`. The raw vector stays in state and is never sent to a prompted module |
| 6. Wiring | `ideas_on()`; `read_ideas(live)` runs after Attention and before Recall, so a stopped turn never pays for it. It encodes the message and, when Perception is on, each observation's `text`. State: `"ideas": {"about": [...], "vector": [...]}` (add `"ideas": None` to `new_state()`) and `observations[i]["about"]`. A failing encoder is noted on its trace entry and the turn carries on, like Math Language |
| 6a. Memory store | `MemoryStore.add()` takes an optional `ideas` vector and stores it on the row, rounded to 3 decimals, with the checkpoint's modified time as `ideas_from`. `search(queries, top, vector=None, ideas_weight=1.0)`: with a vector, the hybrid score; without one, unchanged BM25. A row with no vector, or one whose `ideas_from` doesn't match the loaded checkpoint, is encoded during search and written back under the lock, a one-time migration |
| 6b. Recall | `recall()` passes `s["ideas"]["vector"]` and the setting when the module is on. Add `"about": [ideas as words]` to the recall query payload through a `_about(s)` helper that returns `None` when the module is off, so `drop_none` keeps it invisible. The rank prompt is unchanged |
| 6c. Remember | `remember()` encodes each saved text and stores its vector, when the module is on |
| 7. Settings | `"ideas": [("ideas_weight", "How much meaning counts next to word matches in recall", "number", 0, 3, 0.1, 1.0), ("tag_observations", "Tag each observation with its ideas", "bool", None, None, None, True)]` |
| 8. Checks | `checks.json` → `"ideas"`: cases with `message` and `about_any`, like "Who's my dentist?" → `["health", "person"]`, "Save a shopping list with eggs, milk and bread." → `["food", "shopping"]`; and recall-style cases with `stored` and `top_contains` where the query shares no word with the memory ("Who looks after my teeth?" → "lee"), seeded into a temporary store the way the existing `recall` check is. Add the no-overlap case to the `recall` checks too, so the Recall module's Test button exercises the whole path. `test_module()` branch and `grade()` |
| 9. Tests | `StubEncoder` (fixed vectors keyed by a word in the text, no PyTorch), `use_ideas()`, an `Ideas` class covering: untrained is skipped and can't be chosen; runs after attention and not on a stopped turn; observations get `about`; saved memories get vectors; a search with a vector finds the no-overlap memory; switched off means pure BM25, no `about` in any payload, stored vectors ignored; a broken encoder doesn't take the turn down; `ideas_weight` changes the ranking; `test_module()` grades it |
| 10. Page | Add `--indigo` and `--indigo-ink` to both the light and dark blocks. "How it got there" gets an Ideas section (`section("Ideas", "ideas", ...)`) with the message's top ideas, and the Recall step says for each memory whether it came by words, by meaning, or both. Optional: the memory list under the circle shows each memory's top 3 ideas |
| 11. eval.py | New scenario `memory_meaning`: `["Remember that my dentist is Dr. Lee and my appointments are on Tuesdays.", "Who looks after my teeth?"]`, expect `reply_contains ["lee"]`. The existing `memory` scenario must still pass |

**Gate.** `python3 -m unittest discover -s tests` passes in `paratroop_harness_02/` and `smCLM_02/`; the Ideas Test
button and `--test ideas` grade; `--test recall` still passes; `eval.py --only memory,memory_meaning` passes both
with the module on, and `memory` passes with it off.

**Write down.** Seconds the Ideas module adds to a turn (target: under 0.1 s), and the eval table.

### Phase 6: live check and docs

- **Live check** in a scratch data folder ([running.md](../paratroop_harness/running.md#a-scratch-data-folder)) with
  Router, Perception, Attention, Ideas, Recall and Language on, prompted modules on `pc`, the Mac's model unloaded.
  Run the two memory scenarios and five of the group B test pairs by hand through the page. Read "How it got there".
- **Docs to update:** `smCLM_02/README.md` in the house format (the idea, the data, the model, results, what it means,
  next, run it, files); [smModels/smCLM_02.md](../smModels/); [results.md](../research/results.md) with dated entries
  for phases 3, 4 and 5; [architecture.md](../paratroop_harness/architecture.md) (module table, contracts, switches,
  settings, turn cycle); the harness README (module section, settings, test count); [running.md](../smModels/running.md)
  (the project table); [models.md](../../product/smModels/models.md); [whats-next.md](../../product/whats-next.md), which
  currently says vector memory needs a download (flip its status to built or known problem, whichever is true);
  [glossary.md](../../product/glossary.md) if "Ideas module" needs a line.
- **Negative results stay.** A checkpoint that failed a gate is kept under `out/<name>/` as evidence, the way
  `smLANGUAGE_RENDER_001/out/v2/` is.

### Phase S: stretch, only after the phase 4 gate

- **Self-found ideas.** v1's first "Next" item: a sparse bottleneck (about 64 units, k-sparse) trained with the same
  masked objective but no tags, then name each unit by the words that fire it, and compare with the hand inventory. If
  the units line up with the inventory, the inventory was real; if they find things the inventory missed, add them.
- **Attention by ideas.** A code rule: relevance = cosine(observation, goal) when the Attention module is off. Cheap,
  and it would be the first non-model fallback with any judgement.
- **Ideas to the Planner.** Put observations' `about` words in the planner payload and check whether it changes any
  decision on `eval.py`. Probably it won't; write that down if so.

## 6. Project layout

```text
smCLM_02/
  README.md
  ideas.json              the inventory: idea -> one-line meaning; v1's 51 first
  test_recall.json        hand-written recall pairs (phase 0); never trained on
  test_words.json         held-out words with human tags (phases 0 and 1); never trained on
  vocabulary.py           the three sources -> data/vocabulary.jsonl; the folding rules
  teacher.py              a copy of smCONVERSATION_001/teacher.py: strict JSON, logged, retried
  propose_ideas.py        open-list proposals -> data/idea_proposals.json for the human pick
  tag_words.py            3 passes -> data/tags.jsonl
  held_out.py             picks the held-out words with a fixed seed
  generate_sentences.py   resumable teacher sentences for words under 30 usages
  compile_data.py         data/sentences.jsonl + manifest; runs the leak test
  world.py                loads the data; tokenising and folding (v1's world.py role)
  model.py                ConceptModel and WordModel, max_len 24
  train.py                3 seeds, EM refresh, best epoch -> out/ideas.pt, out/results.json
  evaluate.py             word scores, v1 regression, the 4-ranker recall table -> out/eval.md
  encode.py               Encoder (the harness loader) and a __main__
  ask.py                  v1's ask.py, adapted
  tests/                  unit tests: data rules, leaks, shapes, encoder, scoring
  data/                   vocabulary.jsonl, tags.jsonl, sentences.jsonl, manifest.json, logs/, quarantine/
  out/                    ideas.pt, results.json, train.log, eval.md
```

## 7. Ground rules that apply

From [the engineering README](../README.md):

- **Leave the owner's page state alone.** Live checks use a scratch data folder.
- **Don't train and serve on the same box.** Before phase 3, point the harness at `pc` and unload the Mac's model.
  Teacher calls in phases 1 and 2 go to the RX 580, which the family portal shares; run them in resumable batches and
  expect to pause.
- **Keep the best checkpoint, not the last.**
- **Checkpoints get picked up live.** Once `smCLM_02/out/ideas.pt` is listed in `harness.json`, a half-trained file is
  offered on the page. Switch the Ideas module off while it trains.
- **One GPU job at a time on the Mac.**

## 8. Open questions for the owner

1. **Vocabulary size.** The plan says about 2,500 words. The curriculum project defines 10,500 words but only 600 are
   accepted, so the bigger list isn't available yet. Is 2,500 enough for a first real-language run?
2. **What the memory store keeps.** The plan stores the full idea vector per memory (about 1 KB a row, about 2 MB at
   the store's 2,000 cap). The alternative is the top 8 ideas only, smaller but lossier for cosine.
3. **When the RX 580 is free** for about 5,000 teacher calls over several hours.

## 9. Rough budget

Guesses, to be corrected after phase 1.

| Phase | Machine time | Person time |
|---|---|---|
| 0: tests first | none | 2 to 3 hours writing 60 recall cases |
| 1: vocabulary and inventory | about 400 teacher calls | 1 hour picking ideas, 1 hour tagging held-out words, 1 hour reviewing the sample |
| 2: sentences | about 5,000 teacher calls, several hours on the RX 580 | 30 minutes reading 50 sentences |
| 3: the reader | 3 seeds × about 45 minutes on the M5 | reading the worst 10 words |
| 4: encoder and recall scores | minutes | reading the group C failures |
| 5: the harness | test suite | a live check |
| 6: docs | none | an hour |
