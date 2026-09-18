# smALLM_01: learn-by-doing gadget world

Project: [`smALLM_01/`](../../../smALLM_01/README.md). Checked against `world.py`, `model.py`, `train.py`,
`evaluate.py`, `demo.py`, `check_curious.py`, `out/train.log` and `out/eval.md` on 2026-09-15.

## Purpose

Can a tiny model learn cause and effect by poking a world, with no dataset? Each episode has secret wiring. The model
flips gadgets, predicts what everything will read, and learns from how wrong it was.

## Input and output

- **The world** (`world.py`): 8 on/off gadgets, A to H, wired in a random order with no loops.
  - The first gadget in the order is always a switch. Every later one is a switch with probability 0.35.
  - Every other gadget follows one or two gadgets earlier in the wiring, with COPY, NOT, AND or OR.
  - Flipping holds a gadget at a value for one step. A switch stays where it's put; any other gadget springs back.
- **Tokens** (vocabulary 33):
  - "gadget i reads v" is `2i + v`;
  - "flip gadget i to v" is `16 + 2i + v`;
  - `BOS` (32) starts an episode.
- **An episode:** `BOS`, the 8 starting readings, then for each flip, the flip token and the 8 readings while held.
- **Out:** the next token. After a flip, that's the model's prediction of every reading.

## Architecture and size

`TinyGPT` in `model.py`:

- the same design as [smLLM_01](smLLM_01.md), with d 256, 6 layers and 8 heads;
- no dropout, since every episode is new;
- context 225 tokens, which is `1 + 8 + 9 × 24`.

**4.8M parameters** (`out/train.log`).

## Data

- **No dataset.** Every batch is 64 brand-new worlds with 24 random flips each, made on the spot (`episodes()`).
- **Scoring:** the loss skips flip tokens (`y >= ACT`) and the 8 random starting readings. Only readings after a flip
  count.

## Train and test

```bash
python3 world.py
python3 train.py --minutes 20
python3 evaluate.py
python3 demo.py --policy curious --seed 7
python3 check_curious.py
```

- **Training:** a 20-minute budget; AdamW lr 1e-3, betas 0.9/0.99, weight decay 0.1; warmup 100 steps, then cosine by
  elapsed time.
- **Evaluation during training:** every 250 steps, 64 worlds after 8 random experiments. It keeps the checkpoint with
  the best "all 8 right".
- **`evaluate.py`:** after k experiments in unseen worlds, every possible flip is probed. Three scores:
  - **all 8 right:** every reading after the flip predicted correctly;
  - **effects caught:** of gadgets that really changed (not the flipped one), how many were predicted;
  - **lazy guess:** assume only the flipped gadget changes.
- **Flip policies:** `random`, or `curious`, which picks the flip whose outcome the model is least sure of. Uncertainty
  is scored as if nothing else changed.
- **Tests:** none, apart from `world.py`'s self-check. All flags: [running.md](running.md#smallm_01).

## Results

**2026-09-14**, 20 minutes on the M5:

- **Training:** 3,864 steps and 247.3k worlds. The best checkpoint was step 3750.
- **Loss:** 0.583 at step 250, down to 0.134 at the end, still falling.

Final test, 256 unseen worlds, 2,048 questions per cell (`out/eval.md`):

| Experiments | Random: all 8 right | Random: effects caught | Curious: all 8 right | Curious: effects caught | Lazy guess |
|---|---|---|---|---|---|
| 0 | 61% | 0% | 61% | 0% | 61% |
| 1 | 44% | 26% | 45% | 25% | 61% |
| 2 | 48% | 35% | 45% | 36% | 61% |
| 4 | 59% | 44% | 55% | 55% | 61% |
| 8 | 70% | 58% | 74% | 72% | 61% |
| 12 | 75% | 66% | 79% | 78% | 61% |
| 16 | 80% | 75% | 80% | 82% | 62% |
| 20 | 83% | 80% | 82% | 85% | 61% |

`check_curious.py` (from the README; 256 worlds, 20 flips):

| | Other gadgets its flips changed | Picked a biggest-effect flip | Same gadget twice in a row | Gadgets tried |
|---|---|---|---|---|
| random | 0.83 (typical flip 0.82) | 16% | 13% | 7.4 of 8 |
| curious | 2.01 (typical flip 0.87) | 44% | 18% | 7.1 of 8 |

## Known failure modes

- **Guesses too early.** After 1 to 4 experiments it's worse than the lazy guess (44–59% against 61%): it predicts
  effects before it has the evidence.
- **"Curious" mostly means "biggest lever."**
  - It catches more effects at 4 to 12 experiments (+11 to +14 points).
  - "All 8 right" improves by only 0 to 4 points, and there's no gain at 16 to 20.
- **Not apples to apples.** Curious flips leave the world in different states before each probe, so the two policies'
  columns can't be compared exactly.
- **Undertrained.** The loss was still falling when the 20 minutes ran out.
- **Not in v1:** hidden gadgets, choosing experiments during training, and practicing inside its own world model.

## In the harness

Not used by either harness.
