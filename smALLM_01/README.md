# smALLM_01

A tiny model that learns by doing. The "A" is for action. It's the second model in SMMOL,
next to [smLLM_01](../smLLM_01/README.md).

## The idea

Kids learn cause and effect by poking things. This model does the same, with no dataset:
the world is the answer key.

1. **Predict:** "if I flip C, what will everything read?"
2. **Act:** flip C.
3. **See:** read every gadget.
4. **Learn:** how wrong the guess was is the loss.

## The gadget world

- There are 8 on/off gadgets, `A` to `H`.
- **The wiring is secret and new every episode.** About 3.5 gadgets are switches that stay
  where you put them. The rest follow one or two gadgets earlier in the wiring, with copy,
  NOT, AND or OR. There are no loops.
- **Flipping means holding a gadget at a value for one step.** A switch stays flipped. Any
  other gadget springs back once it's let go.
- About 37% of flips change something besides the gadget that was flipped.

A real wiring from `demo.py`:

```text
B is a switch
G = NOT B
E = NOT G
F = G
A = B
H = F
D = F AND A
C = A OR F
```

## How it trains

- Every batch is 64 brand-new worlds with 24 random flips each, made on the spot. The data
  never repeats, so memorizing is useless, and the only thing worth learning is how to work
  out a wiring from experiments.
- Tokens look like "gadget C reads 1" and "flip C to 0". After each flip, the model predicts
  every reading. Its own flips and the random starting position aren't scored.
- It uses the same tiny GPT as smLLM_01 at a smaller size: 4.8M parameters (6 layers, 256
  wide, 8 heads) and a 225-token context. There's no dropout, since there's nothing to
  memorize.

## How it's scored

After `k` experiments in a world it has never seen, it's asked about every gadget: "flip it
now; what will all 8 read?"

| Score | Meaning |
|---|---|
| all 8 right | the whole outcome is correct |
| effects caught | of the gadgets that really changed (not counting the flipped one), how many it predicted |
| lazy guess | always guessing "only the flipped gadget changes". That's right about 61% of the time, because most flips don't reach anything else |
| random vs curious flips | random picks any gadget; curious picks the flip whose outcome the model is least sure of |

## What would count as working

1. **It learns from experiments:** effects caught rises as `k` grows.
2. **It works out the wiring:** all 8 right beats the lazy guess's 61%.
3. **Picking its own experiments helps:** curious beats random when there have only been a
   few experiments.

## Run it

Self-check the world's logic:

```bash
python3 world.py
```

Train (20 minutes by default):

```bash
python3 train.py --minutes 20
```

Score the saved model:

```bash
python3 evaluate.py
```

Watch it poke at one world, then see the secret wiring:

```bash
python3 demo.py --policy curious --seed 7
```

## Files

| File | What it does |
|---|---|
| `world.py` | Gadget world: random wirings, flipping, tokens, self-check |
| `model.py` | Tiny GPT |
| `train.py` | Learn-by-doing loop on live episodes; saves the best checkpoint to `out/ckpt.pt` |
| `evaluate.py` | The k-experiments test, random vs curious, and the lazy baseline |
| `demo.py` | One world, step by step, with the wiring revealed |
| `check_curious.py` | What kind of flips the curious picker actually chooses |
| `out/train.log`, `out/log.csv`, `out/eval.md` | Training printout, scores over time, final table |

## Not in v1

- **Hidden gadgets** (a secret cause behind two effects) and a watcher-vs-doer comparison.
- **Choosing experiments during training.** Right now the curious policy is only used at test time.
- **Dreaming:** practicing inside its own world model.

## Results

First run, 2026-09-14: 20 minutes on the M5.

- 3,864 steps, 247k brand-new worlds.
- Loss went from 0.58 to 0.134 and was still falling at the end, so more time would help.

Final test: 256 worlds it never saw, with every gadget probed, so each cell is 2,048 questions.

| experiments | random flips: all 8 right | random: effects caught | curious flips: all 8 right | curious: effects caught | lazy guess: all 8 right |
|---|---|---|---|---|---|
| 0 | 61% | 0% | 61% | 0% | 61% |
| 1 | 44% | 26% | 45% | 25% | 61% |
| 2 | 48% | 35% | 45% | 36% | 61% |
| 4 | 59% | 44% | 55% | 55% | 61% |
| 8 | 70% | 58% | 74% | 72% | 61% |
| 12 | 75% | 66% | 79% | 78% | 61% |
| 16 | 80% | 75% | 80% | 82% | 62% |
| 20 | 83% | 80% | 82% | 85% | 61% |

Against the three checks:

1. **It learns from experiments: yes.** Effects caught climbs from 0% to 80% with random
   flips and to 85% with curious ones, over 20 experiments.
2. **It works out the wiring: yes, given enough experiments.**
   - With no experiments it says "nothing else changes", exactly the lazy 61%. That's the
     right call when you know nothing.
   - After 1 to 4 experiments it's *worse* than lazy (44–59%). It starts guessing effects
     before it has enough evidence.
   - From 8 experiments on it beats lazy: 70–74% at 8, and 83% at 20.
3. **Picking its own experiments helps, partly.**
   - Curious catches more effects at 4 to 12 experiments (+11 to +14 points).
   - All 8 right only improves by 0 to 4 points, and there's no gain by 16 to 20.

**What "curious" really picks** (`check_curious.py`, 256 worlds, 20 flips each):

| | Other gadgets its flips changed | Picked a biggest-effect flip | Same gadget twice in a row | Gadgets tried |
|---|---|---|---|---|
| random | 0.83 (typical flip: 0.82) | 16% | 13% | 7.4 of 8 |
| curious | 2.01 (typical flip: 0.87) | 44% | 18% | 7.1 of 8 |

- "Least sure" mostly means "biggest lever". It still spreads its experiments around, but
  favors flips that shake a lot of the world.
- Part of the cause is probably how uncertainty gets scored: each flip is scored as if
  nothing else changed, which makes big-effect flips look uncertain.
- Big levers reveal more wiring, which may be exactly why it helps.
- Curious flips also leave the world in different states before each probe, so the two
  columns aren't perfectly apples to apples.

**The "aha" in `demo.py --seed 7`.** In this world, `B` is the master switch: everything
follows it. Flips 1 to 6 got 1 right. Once it had seen `B` work, flips 7 to 16 got 9 of 10:

```text
 5. flip B to 1  guess 01100111  saw 11101000  5 wrong
 6. flip G to 1  guess 11101010  saw 11110111  4 wrong
 7. flip B to 0  guess 00100111  saw 00100111  right
 8. flip B to 1  guess 11101000  saw 11101000  right
 9. flip F to 1  guess 11110100  saw 11111101  2 wrong
10. flip B to 0  guess 00100111  saw 00100111  right
```

**Next things to try:**

- Train longer, since the loss was still falling.
- Better curiosity: score uncertainty by sampling possible outcomes, or by two models
  disagreeing.
- Hidden gadgets, plus the watcher-vs-doer comparison.
