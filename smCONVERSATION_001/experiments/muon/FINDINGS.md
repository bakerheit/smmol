# Muon experiment findings

Date: 2026-09-17

## Outcome

Muon is viable for this model, but it did not earn a production switch.

Under the current trainer's global gradient clip of `1.0`, AdamW had the best mean validation loss.
When clipping was effectively disabled, original-scaling Muon at learning rate `0.02` beat AdamW on
two of three seeds and had a `0.075%` better mean loss. That difference is tiny relative to its seed
variance, so this experiment supports a larger trial rather than a claim that Muon is better.

Recommendation: keep AdamW for the current production run. For the next optimizer experiment, test
Muon `0.02` without global clipping on a larger frozen corpus, with early stopping and at least three
seeds.

## Protocol

- Hardware/backend: Apple Silicon through PyTorch MPS.
- Software: Python 3.9.6, PyTorch 2.8.0.
- Model: 2,018,688 parameters, 4 layers, width 192, 6 heads, 1,024-byte context.
- Initialization: the same grade-one checkpoint for every run.
- Corpus: the first 25 accepted conversations, SHA-256
  `13b883001ed7fd9ba16fdccb6cfcc09a1b56739e0cc25a8f59a67a414fed5e4b`.
- Data: 354 training examples and 35 domain-held-out validation examples.
- Optimization: 500 steps, batch size 4, 50 warmup steps, cosine decay, three seeds.
- Evaluation: the full held-out set at steps 0, 25, 50, 100, 200, 350, and 500.
- Pairing: variants with the same seed received identical batch indexes in identical order.
- Muon scope: hidden 2D transformer-block weights only. Embeddings, the tied output head, position
  embeddings, and normalization parameters stayed on AdamW.

## Production-clipping result

Global gradient norm limit: `1.0`.

| Optimizer | Mean best validation loss | Standard deviation | Mean bits/byte | Mean clipped steps | Difference from AdamW |
|---|---:|---:|---:|---:|---:|
| AdamW `2e-4` | **2.505784** | 0.001684 | **3.615082** | 100.0% | baseline |
| Muon RMS-matched `2e-4` | 2.519522 | 0.002668 | 3.634902 | 100.0% | +0.548% |
| Muon original `0.01` | 2.521582 | 0.002732 | 3.637874 | 89.2% | +0.630% |
| Muon original `0.02` | 2.512473 | 0.004027 | 3.624733 | 66.3% | +0.267% |

AdamW improved through step 500 in every seed. The RMS-matched and original-`0.01` Muon variants
peaked at step 350 in every seed, then regressed. Original Muon `0.02` was the strongest Muon form.

This run also exposed a broader trainer issue: a `1.0` threshold clipped every AdamW step. AdamW's
adaptive scaling made its final loss nearly insensitive to removing the clip, but Muon's momentum
and orthogonalization path reacted more strongly.

## No-effective-clipping follow-up

The threshold was set to `1,000,000`, leaving gradients unchanged. Only AdamW and the strongest Muon
variant were rerun.

| Optimizer | Seed 16001 | Seed 16002 | Seed 16003 | Mean | Standard deviation | Mean bits/byte |
|---|---:|---:|---:|---:|---:|---:|
| AdamW `2e-4` | **2.505263** | 2.504116 | 2.505273 | 2.504884 | 0.000665 | 3.613784 |
| Muon original `0.02` | 2.507890 | **2.498940** | **2.502222** | **2.503018** | 0.004528 | **3.611091** |

Muon's paired mean change was `-0.001866` loss, or `-0.075%`. The three paired differences were
`+0.002627`, `-0.005176`, and `-0.003051`; negative favors Muon. With only three seeds, the approximate
95% paired confidence interval was `[-0.011887, +0.008155]`. It crosses zero by a wide margin.

Both optimizers reached their best recorded result at step 500 in all three unclipped runs. At step
350, Muon's mean was `2.509362` versus AdamW's `2.512545`, so its slight advantage was not confined to
one final measurement.

## Speed and memory

Timing changed sharply during the experiment because another Metal workload was present early and
later released resources. Absolute timings from the main run are not comparable.

During the later paired follow-up, median throughput was:

- AdamW: 6.934 steps/second including evaluation.
- Muon: 6.721 steps/second including evaluation.

That suggests roughly 3% Muon overhead on this model, but a dedicated idle-machine timing run is
required before treating it as a firm number.

Muon keeps one momentum tensor for hidden matrices, while AdamW keeps first- and second-moment
tensors. For this model, ideal FP32 optimizer-state storage is approximately:

- AdamW: 15.40 MiB.
- Hybrid Muon/AdamW: 8.65 MiB.
- Reduction: 43.8%.

Memory is irrelevant at two million parameters, but this becomes useful if the student grows.

## What this proves

- The local Muon implementation is numerically stable on MPS for this model.
- RMS-matched Muon at the AdamW learning rate is a poor fit under the tested settings.
- Original Muon at `0.02` is the only tested Muon configuration worth carrying forward.
- Global clipping at `1.0` changes Muon's result enough that clipping must be treated as an optimizer
  hyperparameter, not a neutral safety switch.
- Muon can match AdamW closely on the pilot corpus without slowing each step much.

## What this does not prove

- That Muon wins on the final 1,000-conversation corpus.
- That the `0.02` learning rate is optimal; `0.015`, `0.025`, and `0.03` were not tested.
- That no clipping is universally safe. The 500-step pilot stayed finite, but larger data and longer
  training can behave differently.
- That the tiny mean improvement is statistically real.
- That Muon improves generated conversation quality. This experiment measured held-out byte loss,
  not human preference or multi-turn behavior.

## Next experiment

Freeze a 250-conversation corpus and compare only:

1. AdamW `2e-4`, no effective clipping.
2. Muon original `0.015`, `0.02`, and `0.025`, no effective clipping.

Use three seeds, full held-out evaluation, generated-conversation checks, and an idle-machine timing
pass. Stop a run when held-out loss fails to improve across three evaluations. Do not fold this into
the production trainer until that larger test shows a repeatable quality or compute advantage.
