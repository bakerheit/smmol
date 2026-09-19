# smLANGUAGE_EN_001

A small English foundation model trained from scratch. This is language pretraining: it predicts
the next byte in ordinary English prose. It is separate from `smLANGUAGE_RENDER_001`, which turns
structured state into a short reply.

## Stage one

| | |
|---|---|
| Corpus | A pinned 128 MiB prefix of TinyStories |
| License | CDLA-Sharing-1.0 |
| Model | 10.8M-parameter causal Transformer |
| Tokens | Raw UTF-8 bytes; vocabulary 256 |
| Context | 256 bytes |
| Objective | Next-byte prediction across the full sequence |
| Selection | Validation bits per byte |
| Final check | A disjoint, untouched story test split |

TinyStories is deliberately simple English. It is a sensible first pretraining stage for this
model size, but it is not broad English coverage. A later stage should mix in real prose such as
WikiText-103 instead of pretending this first run solves general language.

## Run

```bash
python3 prepare.py
python3 -m unittest -v test_language.py
python3 train.py --steps 5000
python3 evaluate.py --split test
python3 sample.py --prompt "Once upon a time"
```

Training writes an atomic `out/latest.pt` for resuming and `out/best.pt` for the lowest validation
loss. To continue an interrupted run toward a higher total step:

```bash
python3 train.py --resume --steps 10000
```

Do not use the test split to tune the run. `train.py` only reads `train.txt` and `val.txt`.

## Paratroop Harness 02

`paratroop_harness_02` can load `out/best.pt` directly as a choice on its Language card. The bridge
is in `inference.py`; it converts the harness payload into a short completion prompt and does not
fake instruction following. The pretrained checkpoint loads and generates successfully, but scored
0/3 on the harness Language checks. It needs supervised tuning on decisions, tool results, questions,
and make requests before it can replace the harness's instruction-tuned model.

## What success means

The first gate is mechanical: validation loss must fall far below the random-byte baseline of
8 bits/byte. The real gate is fixed-prompt sampling: complete words, sentence structure, stable
characters, and a locally coherent event sequence. Stage one will be biased toward short,
childlike stories; broader English needs stage two.

## Current run

Completed 2026-09-16 on PyTorch MPS. The 5,000-step run saw 81.9M training bytes and lowered
validation loss from 5.498 to 0.669 (7.93 to 0.96 bits/byte). After model selection was finished,
the sealed test split scored 0.682 loss (0.983 bits/byte) over 819,200 sequential bytes. Fixed
prompts now produce grammatical short-story prose and stable local events, with the expected
small-model weaknesses: odd object choices, repetition, and weak longer-range meaning.
