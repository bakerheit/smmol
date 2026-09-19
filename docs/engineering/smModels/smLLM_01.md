# smLLM_01: tiny Shakespeare byte GPT

Project: [`smLLM_01/`](../../../models/smLLM_01/README.md). Checked against `model.py`, `train.py`, `sample.py`, `out/train.log`
and `out/log.csv` on 2026-09-15.

## Purpose

The first model in SMMOL, "just for fun": a GPT trained from scratch on a MacBook Pro M5. The layout copies
nanoGPT's `shakespeare-char` setup, so the result can be compared with a well-known number.

## Input and output

- **In:** bytes. One token per byte, a vocabulary of 256, no tokenizer.
- **Out:** the next byte. `generate()` samples one byte at a time with temperature and top-k, over the last 256 bytes.

## Architecture and size

`TinyGPT` in `model.py`:

- token embedding plus learned position embedding (256 positions), then dropout;
- 6 blocks: pre-LayerNorm causal self-attention (`scaled_dot_product_attention`, 6 heads), then a 4× GELU MLP, each
  added to the residual stream;
- final LayerNorm, and an output head tied to the token embedding;
- width d 384, dropout 0.2;
- init: normal(0, 0.02), with the attention-out and MLP-out weights scaled by 1/√(2·layers).

**10.8M parameters** (`out/train.log`).

## Data

- `data/input.txt` is Tiny Shakespeare, 1,115,394 bytes (from karpathy/char-rnn).
- The first 90% (1.00M bytes) is for training; the last 10% is validation, never trained on.
- Batches are 64 random 256-byte windows. The target is the same window shifted one byte.

## Train and test

```bash
python3 train.py --minutes 20
python3 sample.py --prompt "JULIET:" --temp 0.8
```

- **The loop:** `train.py` runs for a time budget, not a step count.
  - AdamW, lr 1e-3, betas 0.9/0.99, weight decay 0.1 on matrices, gradient clip 1.0.
  - Warmup for 100 steps, then cosine decay by elapsed time down to lr/10.
- **Evaluation:** every 250 steps, loss on 20 train and 20 val batches. `out/ckpt.pt` is saved only when val improves.
  A sample prints every 1000 steps.
- **No unit tests.** All flags: [running.md](running.md#smllm_01).

## Results

**2026-09-14**, 20 minutes on the M5's GPU:

| | |
|---|---|
| Steps | 1,915 (batch 64 × 256 bytes) |
| Tokens seen | 31.4M, about 31 passes over the training text |
| Speed | about 26,100–27,000 training tokens/s |
| Start | val 5.660 (8.17 bits/char), pure guessing |
| Best val loss | **1.471 (2.12 bits/char) at step 1,500**, the saved checkpoint |
| End | train 1.011, val 1.479 |

The README gives nanoGPT's published number for this layout as about 1.47, so this matches.

## Known failure modes

- **Memorizing.** After step 1,500, train loss kept falling while val loss rose. 1 MB of text is small for 10.8M
  parameters.
- **No meaning.** Samples have the format, real words and rhythm of a play, but don't make sense.

## In the harness

Not used by either harness.
