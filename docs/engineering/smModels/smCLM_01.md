# smCLM_01: ideas reader vs words reader

Project: [`smCLM_01/`](../../../smCLM_01/README.md). Checked against `world.py`, `model.py`, `train.py`, `ask.py`,
`tests/test_smclm.py`, `concepts.json`, `out/train.log` and `out/results.json` on 2026-09-15.

## Purpose

Two questions:

- Can a model learn what a noun means, as a bundle of ideas like "car = transportation, people, control", only from
  the sentences that use it?
- Does reading nouns as ideas beat reading them as words?

## Input and output

- **The world** (`concepts.json`, `world.py`):
  - 51 ideas and 286 tagged nouns, each with 1 to 8 ideas;
  - 79 sentence templates. Each slot lists the ideas a word needs and the ones it rules out (`!idea`), so a sentence
    only uses a word whose ideas fit;
  - 23 held-out nouns: car, tram, scooter, canoe, wolf, bee, mango, soup, lemonade, jacket, helmet, hammer, drum,
    violin, laptop, library, bakery, river, storm, candle, coin, nurse, grandfather. They appear in sentences, but their
    ideas are never given;
  - 4 made-up words with teach sentences: zorb, glimberry, skarn, blorp.
- **In:** a sentence of up to 12 tokens with one noun hidden (`MASK`). Other words are "frame" ids.
- **Out:**
  - the **idea reader** predicts the hidden noun's 51 ideas, each 0 to 1 (binary cross-entropy);
  - the **word reader** predicts which of the 286 nouns is hidden (cross-entropy).

## Architecture and size

`model.py`. Both readers use the same `Encoder`:

- learned positions (12), then `nn.TransformerEncoder`: pre-norm, dropout 0.1, feed-forward 4d;
- d 128, 2 layers, 4 heads.

| | Idea reader (`ConceptModel`) | Word reader (`WordModel`) |
|---|---|---|
| A noun comes in as | a linear projection of its 51-number idea vector | its own learned embedding |
| Head | 51 ideas | 286 nouns |

- **Held-out nouns** start with all-zero ideas in the idea reader. After each epoch, `infer_ideas()` sets each one's
  ideas to the average prediction wherever it was hidden.
- **Parameters:** the scripts don't print a count.

## Data

- `World.corpus(80000, seed)` makes sentences from the templates.
- **Idea reader:** trains on hidden known nouns only.
- **Word reader:** trains on every noun position.
- **Getting ideas out of the word reader,** two ways:
  - nearest words: average the ideas of the 5 known nouns closest to it in embedding space;
  - words that fit the same spots: average the ideas of the known words it would predict in the held-out word's
    positions.
- **Baseline:** the most common ideas among known nouns, with no reading.

## Train and test

```bash
python3 train.py
python3 ask.py car
python3 -m unittest discover -s tests
```

- **Training:** 80,000 sentences per seed, 8 epochs, 3 seeds, batch 512, AdamW lr 2e-3, weight decay 0.01, on the CPU.
- **Scoring on held-out words:**
  - **precision@k:** of the k ideas a word really has, how many land in its top k;
  - **average precision:** also rewards ranking the right ideas first.
- **Tests (7):** sentences only use fitting words, every idea is shown by some template, held-out words get used, teach
  sentences use known words, scoring, reader shapes, masking hides the target.
- All flags: [running.md](running.md#smclm_01).

## Results

**2026-09-15 13:27**, 3 seeds, 1,133 s on the CPU (`out/results.json`):

| Method | precision@k | Average precision |
|---|---|---|
| Idea reader | **0.818 ± 0.010** | **0.932 ± 0.006** |
| Word reader: words that fit the same spots | **0.821 ± 0.015** | **0.933 ± 0.006** |
| Word reader: nearest word vectors | 0.127 ± 0.017 | 0.189 ± 0.011 |
| Most common ideas, no reading | 0.137 | 0.209 |

- **Car, never told its ideas** (seed 0): control 0.79, transportation 0.78, vehicle 0.76, metal 0.72, machine 0.67,
  road 0.57, people 0.50. That's 7 of its 8 ideas; it missed speed. Its top 8 wrongly included heavy (0.39).
- **Shown vs implied ideas** (README):
  - ideas shown directly by some sentence were found 74 of 86 times (86%);
  - ideas only implied were found 5 of 9 times (56%): wolf → nature, bee → small and nature, violin → art,
    candle → home.
- **Made-up words** (README; 3 or 4 sentences, no retraining):
  - zorb → machine, metal, control, vehicle, transportation, people;
  - glimberry → plant, food;
  - skarn → living, animal, nature, danger, speed;
  - blorp → sound, music, wood, art.

## Known failure modes

- **Thinking in ideas didn't beat thinking in words on accuracy.** Read the right way, a normal word model scores the
  same. In this world, meaning lives in usage patterns, and both architectures pick them up.
- **Going beyond the evidence is weak** (56% on implied ideas). Examples: tram → electricity scored 0.07; laptop → work,
  bakery → food and storm → water were missed.
- **Wrong ideas that fit the sentences:** helmet → soft (0.93), hammer → sharp (0.56), grandfather → work (0.50).
- **A built-in assumption:** Claude hand-tagged the ideas and wrote the templates so that ideas show in usage. Real
  language is messier.
- **Count mismatch:** the README says 287 nouns, but `concepts.json` and `train.log` have 286.

## In the harness

Not used yet. The README's "Next" section suggests plugging it into paratroop_harness_02, so observations carry idea
vectors.
