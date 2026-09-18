# smLLM_01

A tiny GPT trained from scratch on the Mac, just for fun. It's the first model in the SMMOL
series.

## What it is

| | |
|---|---|
| Tokens | bytes: one token per byte, a vocabulary of 256, so there's no tokenizer |
| Size | 10.8M parameters: 6 layers, 6 attention heads, 384 numbers per token |
| Context | 256 bytes |
| Data | Tiny Shakespeare, 1.1 MB of Shakespeare plays ([karpathy/char-rnn](https://github.com/karpathy/char-rnn)) |
| Hardware | the M5's GPU, through PyTorch MPS |

The layout copies nanoGPT's classic `shakespeare-char` setup, so results can be compared
with a well-known baseline.

## Files

| File | What it does |
|---|---|
| `model.py` | The whole model in about 100 lines: embeddings, attention, MLP, generate |
| `train.py` | Trains for a set number of minutes and saves the best checkpoint to `out/ckpt.pt` |
| `sample.py` | Loads the checkpoint and writes text |
| `data/input.txt` | Training text (the last 10% is held out) |
| `out/log.csv`, `out/train.log` | Loss over time and the full training printout |

## Run it

Train (20 minutes by default):

```bash
python3 train.py --minutes 20
```

Make it write:

```bash
python3 sample.py --prompt "JULIET:" --temp 0.8
```

## Reading the numbers

- **Loss** is how surprised the model is by the real next byte. Lower is better.
- **5.55** is pure guessing across 256 bytes, which is 8 bits per character.
- **val** is measured on the last 10% of the text, which the model never trains on.
- If train loss keeps falling while val rises, it's memorizing instead of learning.

## Knobs to try

- `--temp`: 0.5 is safe and repetitive; 1.2 is wild.
- `--minutes`: more training helps until val stops improving.
- `--layers`, `--d`, `--heads`: a bigger or smaller brain. `d` must divide evenly by `heads`.
- `--dropout`: more fights memorizing, less learns faster.

## Results

First run, 2026-09-14: 20 minutes on the M5.

| | |
|---|---|
| Steps | 1,915 (batch 64 × 256 bytes) |
| Tokens seen | 31.4M, about 31 passes over the training text |
| Speed | about 26,700 training tokens/s |
| Best val loss | **1.471** (2.12 bits/char) at step 1,500 |
| At the end | train 1.011, val 1.479 |

- It started at 5.66, which is pure guessing.
- By 2.6 minutes it was at 2.10.
- By 10 minutes it was at 1.50, and the samples already looked like a play.
- After about step 1,500, train loss kept falling but val stopped improving. That's the model
  starting to memorize this small text, and `out/ckpt.pt` keeps the step 1,500 version.
- nanoGPT's published number for this layout is about 1.47, so this matches, on a laptop.

Sample from `python3 sample.py --prompt "JULIET:" --temp 0.5 --seed 42`:

```text
JULIET:
Now, by my master's good most grace.

JULIET:
I would not be so well; but then I think
the same that is moved the common proverty.

FRIAR LAURENCE:
O my soul! O my lord, I am slain, I will not
The crown of your former and then be come a bawd.
```

Right format, real words, Shakespeare's rhythm, no actual meaning. That's what 10.8M
parameters and 1 MB of text get you.
