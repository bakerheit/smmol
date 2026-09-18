# Muon optimizer experiment

## Question

Can Muon reach a lower held-out byte cross-entropy than the existing AdamW setup for
`smCONVERSATION_001` under the same number of training steps and identical batches?

## Scope

This is an optimizer experiment, not a claim about Muon in general. The student is unusually small:
2.02 million parameters, four transformer blocks, width 192, and a 1,024-byte context. The experiment
uses the first 25 accepted conversations as a frozen pilot corpus while the live 1,000-conversation
generation job continues separately.

The experiment does not edit the production trainer or checkpoints.

## Controls

- Same grade-one initialization for every run.
- Same frozen corpus prefix and recorded SHA-256 hash.
- Same train/validation split.
- Same batch-index sequence for variants with the same seed.
- Same cosine schedule multiplier, warmup, batch size, step count, clipping, and evaluation set.
- Full deterministic held-out evaluation rather than sampled evaluation batches.
- Three seeds for the full run.

The comparison includes:

- `adamw`: current AdamW settings, learning rate `2e-4`.
- `muon_rms`: Muon on hidden two-dimensional block weights using Moonlight/PyTorch's
  `match_rms_adamw` update scaling. Embeddings, the tied output head, position embeddings, and
  normalization parameters remain on AdamW.
- `muon_original_10m`: Keller-style update scaling with Muon learning rate `0.01`.
- `muon_original_20m`: Keller-style update scaling with Muon learning rate `0.02`.

For the original-scaling variants, Muon weight decay is adjusted so `learning_rate * weight_decay`
matches the AdamW control. That avoids silently making weight decay 50-100 times stronger merely
because Muon uses a larger numerical learning rate.

## Run

From the repository root:

```bash
python3 smCONVERSATION_001/experiments/muon/run.py --device mps
```

For a quick wiring check:

```bash
python3 smCONVERSATION_001/experiments/muon/run.py \
  --device mps --steps 5 --seeds 16001 --variants adamw muon_rms \
  --output smCONVERSATION_001/experiments/muon/results/smoke.json
```

Raw results are written to `results/`. The runner refuses to overwrite an existing result unless
`--overwrite` is passed.

The clipping follow-up can be reproduced with:

```bash
python3 smCONVERSATION_001/experiments/muon/run.py \
  --device mps --gradient-clip 1000000 \
  --variants adamw muon_original_20m \
  --output smCONVERSATION_001/experiments/muon/results/no_effective_clip.json
```

## Sources

- Keller Jordan et al., [Muon reference implementation](https://github.com/KellerJordan/Muon), MIT.
- Liu et al., [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982).
- PyTorch's newer [`torch.optim.Muon`](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py)
  was used to cross-check update scaling. The installed PyTorch 2.8 build predates that optimizer, so
  this folder contains a small single-device implementation.

## Findings

The completed report is in [`FINDINGS.md`](FINDINGS.md). In short:

- With the production `1.0` global gradient clip, AdamW beat every Muon variant.
- The strongest Muon setup was original scaling at learning rate `0.02`.
- Without effective clipping, that Muon setup won two of three seeds and improved mean validation
  loss by `0.075%`, but the difference was too small and variable to establish a real win.
- Keep AdamW as the production default. Carry unclipped Muon `0.02` into a larger-corpus experiment.

Raw results:

- [`results/full.json`](results/full.json): four variants, three seeds, production clipping.
- [`results/no_effective_clip.json`](results/no_effective_clip.json): AdamW and the strongest Muon
  variant, three seeds, no effective clipping.
- [`results/smoke.json`](results/smoke.json): five-step wiring check only; not evidence.
