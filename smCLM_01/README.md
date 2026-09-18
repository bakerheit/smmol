# smCLM_01

A tiny model that reads nouns only as bundles of ideas, never as words. It works out the ideas
of words it has never been told about from how they're used.

The question: can a model learn "a car is transportation, people and control" without being
told, just from sentences that use "car"? And does thinking in ideas beat thinking in words?

## The world (`concepts.json`, `world.py`)

| Piece | What's there |
|---|---|
| Ideas | 51: transportation, vehicle, machine, control, people, person, animal, food, place, danger, sound, music, soft, metal, art… |
| Words | 286 everyday nouns, each tagged with 1 to 8 ideas. For example, **car** = transportation, vehicle, machine, control, people, road, speed, metal |
| Sentences | 79 templates. Each slot says which ideas a word needs and which it rules out, like "the {person} steers the {thing with control}". A sentence only uses a word whose ideas fit, so a word's ideas show up in how it's used |
| Held out | 23 words (car, tram, scooter, canoe, wolf, bee, mango, soup, lemonade, jacket, helmet, hammer, drum, violin, laptop, library, bakery, river, storm, candle, coin, nurse, grandfather). They appear in sentences like any other word, but their ideas are never given |

**Built-in assumption:** Claude hand-tagged the ideas and wrote the templates, and the world is
designed so ideas are visible in usage. Real language is messier.

## Two readers (`model.py`)

| | Idea reader | Word reader |
|---|---|---|
| A noun comes in as | its idea vector (51 numbers from 0 to 1) | its own learned word vector |
| Trained to | predict the ideas of a hidden noun | predict which word is hidden |
| A held-out word | starts with no ideas; after each pass, its ideas become the average of what the reader predicts wherever the word was hidden | just another word |

Both are 2-layer transformers, 128 wide. Training used 80,000 sentences, 8 passes and 3 seeds,
on the Mac's CPU (19 minutes in total).

## Results

**precision@k:** of the k ideas a held-out word really has, how many land in its top k.
**Average precision** also rewards ranking the right ideas first.

| Method | precision@k | Average precision |
|---|---|---|
| Idea reader | **0.82 ± 0.01** | **0.93 ± 0.01** |
| Word reader: ideas of the words it would put in the same spots | **0.82 ± 0.01** | **0.93 ± 0.01** |
| Word reader: ideas of its nearest word vectors | 0.13 ± 0.02 | 0.19 ± 0.01 |
| Guess the most common ideas, no reading | 0.14 | 0.21 |

**Car, never told its ideas** (`python3 ask.py car`):

```text
control         ████████·· 0.79
transportation  ████████·· 0.78
vehicle         ████████·· 0.76
metal           ███████··· 0.72
machine         ███████··· 0.67
road            ██████···· 0.57
people          █████····· 0.50
like: tram, scooter, bus, canoe, taxi, van
```

That's 7 of car's 8 ideas, including all three of transportation, people and control. It missed
speed (0.23).

**Made-up words, taught with 3 or 4 sentences and no retraining** (`python3 ask.py --teach zorb`):

| Word | Sentences said | The idea reader thinks | Closest words |
|---|---|---|---|
| zorb | a driver steers it, it runs on petrol, it carries people, a mechanic fixes it | machine, metal, control, vehicle, transportation, people | tram, car, scooter |
| glimberry | a child eats it, it grows in the garden, a farmer waters it | plant, food | mango, corn, soup |
| skarn | it's dangerous, a farmer feeds it, it needs water to live, it's very fast | living, animal, nature, danger, speed | wolf, bee, lion |
| blorp | a teacher plays it, it's very loud, it's made of wood | sound, music, wood, art | violin, drum, piano |

**Ideas no sentence shows directly.** 9 of the held-out words' ideas are never needed by any
sentence those words can appear in. The only way to get them is generalizing, for example
"dangerous, fast animals usually live in nature".

| Kind of idea | Found |
|---|---|
| Shown directly by some sentence | 74 of 86 (86%) |
| Only implied | 5 of 9 (56%): wolf → nature, bee → small and nature, violin → art, candle → home |

Missed: tram → electricity (0.07), laptop → work, bakery → food, storm → water.

**Querying by ideas** (`python3 ask.py --ideas animal danger`): snake, shark, lion and bear have
their ideas given. After them come wolf (0.50) and bee (0.27), whose ideas it learned.

## What it means

- **Ideas can be learned from usage.** 82% of a held-out word's ideas land in its top picks, and
  made-up words get sensible ideas from a few sentences with no training.
- **Thinking in ideas did not beat thinking in words on accuracy.** A normal word model scores
  the same when you read ideas off the words it would put in the same spots. In this world, the
  meaning lives in usage patterns, and both architectures pick them up.
- **Where the idea reader is different:** its internal state is ideas. Those are readable, and
  another module can use them directly, for example v02's Perception handing the Planner idea
  vectors instead of text. A new word with known ideas needs zero examples.
- **The word model's own word vectors were useless for this.** At 0.13, they're no better than
  guessing common ideas.
- **Going beyond the evidence is the weak spot:** implied ideas scored 56%, and "a tram uses
  electricity" never came through.

## Next

- **Let the model find its own ideas** through a small, sparse bottleneck, then name what it
  found, instead of using a hand-written list.
- **Real sentences:** have a local model write varied sentences about the same words, and see if
  the result holds.
- **Verbs and relations as ideas too,** not just nouns.
- **Plug it into `paratroop_harness_02`,** so observations carry idea vectors.

## Run it

Train (3 seeds, about 20 minutes on the CPU):

```bash
python3 train.py
```

Look up a word:

```bash
python3 ask.py car
```

Find words by ideas; `-idea` rules one out, and `--ideas` goes last:

```bash
python3 ask.py --ideas transportation control -machine
```

Teach a new word:

```bash
python3 ask.py --teach blip "the child eats the blip" "the blip grows in the garden"
```

Read the ideas in a sentence:

```bash
python3 ask.py --read "the pilot drives people to the city"
```

Tests:

```bash
python3 -m unittest discover -s tests
```

## Files

| File | What it is |
|---|---|
| `concepts.json` | Ideas, tagged words, templates, held-out words, teach sentences |
| `world.py` | Checks the world and generates sentences |
| `model.py` | The idea reader and the word reader |
| `train.py` | Trains both readers, scores the held-out words, saves `out/models.pt` and `out/results.json` |
| `ask.py` | Word lookup, idea search, teaching new words, reading sentences |
| `tests/test_smclm.py` | 7 tests: the world keeps its rules, scoring works, masking doesn't leak |
