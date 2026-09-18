# smEFFICIENCY_01

Not a model. A measurement of where SMMOL's training compute actually goes on the M5, and which of
three free optimizations are real.

The project's founding claim is that small local models are cheaper than one big cloud model. Every
number in [results.md](../docs/engineering/research/results.md) so far is wall-clock time and loss.
None of them says what fraction of the machine a run actually used. This measures that.

## The questions

1. What matmul throughput can this Mac actually reach, per precision? Measured, not from a spec sheet.
2. What does a real SMMOL training loop achieve against that ceiling?
3. Do mixed precision, `torch.compile`, or a bigger batch change it, and does quality survive?

## Setup

| | |
|---|---|
| Machine | Apple M5, 10 GPU cores, 16 GiB unified memory, macOS 26.6.1 |
| Software | PyTorch 2.8.0, Python 3.9.6, MPS backend |
| Model under test | smLLM_01 unchanged: 10.8M parameters, 6 layers, d 384, 6 heads, 256-byte context |
| Contention | Mac idle, no model loaded in UnlimitedStudio, one job at a time |
| FLOPs convention | nanoGPT/PaLM: `6N + 12·layers·d·ctx` per token, forward and backward |

**Fidelity check.** The fp32 measurement here is 27,761 tokens/s against smLLM_01's published
26,700, a 4% difference, and `train_precision.py` reproduces the published step-0 validation loss
of 5.660 exactly. The harness is measuring the real thing.

## 1. The machine's real ceiling

Sustained 2048³ matmuls, 50 iterations after warm-up:

| Precision | Sustained | vs fp32 |
|---|---:|---:|
| fp32 | 3.70 TFLOP/s | 1.0× |
| fp16 | 15.11 TFLOP/s | 4.1× |
| bf16 | **15.19 TFLOP/s** | **4.1×** |

Sixteen-bit arithmetic is four times the fp32 ceiling on this chip. Every model in SMMOL trains in
fp32, so every training run to date has been capped at a quarter of the machine's arithmetic.

## 2. What the training loop gets

smLLM_01, batch 64, context 256, 40 measured steps after 15 warm-up steps:

| Config | Tokens/s | Achieved | Share of its own ceiling | vs fp32 |
|---|---:|---:|---:|---:|
| fp32 eager (baseline) | 27,761 | 1.98 TFLOP/s | 54% of fp32 | 1.00× |
| **bf16 autocast** | **41,813** | 2.99 TFLOP/s | 20% of bf16 | **1.51×** |
| fp16 autocast | 42,181 | 3.01 TFLOP/s | 20% of fp16 | 1.52× |
| fp32 + `torch.compile` | 16,474 | 1.18 TFLOP/s | 32% of fp32 | 0.59× |
| bf16 + `torch.compile` | 16,375 | 1.17 TFLOP/s | 8% of bf16 | 0.59× |

- **Mixed precision is worth 1.5× and costs one line.** fp16 and bf16 are the same speed within
  noise. **Prefer bf16:** it has fp32's exponent range, so it needs no gradient scaler, while fp16
  on this stack has none available.
- **`torch.compile` is a 1.7× regression, not a win.** It also destroys the precision gain: compiled
  bf16 is no faster than compiled fp32. This is the opposite of the x86/CUDA result and is the
  single most useful negative finding here, because it is the optimization everyone reaches for
  first.
- **The ceiling share is the real story.** fp32 training already runs at 54% of the fp32 ceiling,
  which is respectable. Switching to bf16 raises throughput but drops utilization to 20%, because
  the work per kernel no longer fills the machine. Mixed precision moves this model from
  compute-bound to overhead-bound; the remaining 5× is not reachable by precision alone.

## 3. Batch size

fp32 eager, 20 measured steps each:

| Batch | Tokens/s | Steps/s |
|---:|---:|---:|
| 1 | 16,651 | 65.0 |
| 2 | 21,809 | 42.6 |
| 4 | 25,748 | 25.1 |
| 8 | 27,351 | 13.4 |
| **16** | **28,195** | 6.9 |
| 32 | 27,598 | 3.4 |
| 64 (the default) | 27,945 | 1.7 |
| 128 | 27,932 | 0.9 |

**Throughput saturates at batch 8.** Everything from 8 to 128 is the same speed within noise, so the
default batch of 64 buys no throughput over 16. That does not mean "switch to 16": above saturation,
batch size is a *statistical* choice about gradient noise per step, not a speed choice. The useful
fact is that it is free to choose it on statistical grounds, and that memory saved below batch 64
costs nothing.

Batch 1 still reaches 60% of peak here because each step processes 256 tokens of context. That is a
different regime from smRTS_01's byte-at-a-time training, where a step is one byte and launch
overhead dominates.

## 4. What was not a problem

**Batch assembly is not a bottleneck.** The Python loop in `batch()` that builds 64 slices per step
and copies them to the device measures at roughly 0% of step time. It was the obvious suspect and it
is innocent.

## 5. Does quality survive bf16?

Throughput is worthless if the model learns less per step. The test: the same 20-minute budget, the
same seed, the same schedule, validation always measured in fp32 so the number is directly
comparable to the published baseline.

| Run | Steps in 20 min | Tokens | Best val | bits/char | Best reached at |
|---|---:|---:|---:|---:|---:|
| smLLM_01 published, fp32 | 1,915 | 31.4M | **1.4707** | 2.12 | 914 s |
| smEFFICIENCY_01, bf16 | 2,935 | 48.1M | **1.4777** | 2.13 | **613 s** |

**Both runs peak at exactly step 1,500.** They reach the same quality, within 0.5%, and bf16 gets
there in **two thirds of the wall clock**. Sustained throughput over the full 20 minutes was 1.53×,
matching the microbenchmark's 1.51×.

bf16 does **not** reach a better loss, and the reason matters more than the speedup: smLLM_01 is
**data-bound, not compute-bound**. Tiny Shakespeare is 1.1 MB, and both runs start overfitting at
the same step regardless of how fast they got there. The extra 1.5× of throughput buys extra passes
over a dataset that has already given up everything it has.

```text
                fp32                      bf16
 7.5 min   val 1.701                 val 1.494
10.0 min   val 1.567                 val 1.497   <- bf16 already at baseline quality
15.0 min   val 1.492                 val 1.527   <- bf16 now overfitting
20.0 min   val 1.485 (best 1.471)    val 1.607 (best 1.478)
```

**The second finding is bigger than the first.** Both runs spend their tail getting worse. fp32
wasted its last 286 s of a 1,200 s budget (24%); bf16 wasted 587 s (49%). Neither trainer stops at
the validation minimum. Combining bf16 with stopping at the minimum gives the same model in
**613 s instead of 1,200 s — 1.96× less wall clock and electricity for an equal result.**

**Caveat, stated plainly:** one run per precision, and the fp32 baseline is the run recorded on
2026-09-14, not a fresh paired run. The 0.5% val difference is well inside what a seed change could
produce. The throughput numbers are solid; the quality claim is "indistinguishable", not "better".

## 6. What to actually do

1. **Switch training to bf16.** One `torch.autocast` line per trainer. 1.5× throughput, quality
   indistinguishable, no gradient scaler needed. Applies to every model in SMMOL.
2. **Stop at the validation minimum.** Every trainer here runs a fixed time or step budget and keeps
   going past its best checkpoint. `smLLM_01` already keeps the best checkpoint, so the quality is
   not lost — the electricity is. Early stopping is worth more than the precision change.
3. **Do not reach for `torch.compile` on this stack.** Measured 1.7× slower.
4. **Stop treating batch size as a speed knob** above 8; it is a gradient-noise choice.
5. **For smLLM_01 specifically, more compute is the wrong axis.** It is data-bound. More Shakespeare,
   or a smaller model, would move the number; a faster GPU would not.

## 7. Rolled out, 2026-09-17

Both changes were applied across the repo's trainers on the day they were measured. Every file keeps
the same flags and the same meaning: `--precision {fp32,bf16,fp16}` (bf16 by default on the GPU,
fp32 on the CPU) and `--patience N` (default 3, 0 disables). **`--precision fp32 --patience 0`
reproduces the behaviour of every run recorded before this date**, and each trainer's docstring says so.

| Trainer | bf16 | Early stopping | Selection metric patience keys off |
|---|---|---|---|
| smLLM_01 | yes | yes | val loss |
| smALLM_01 | yes | yes | held-out exact-after-8, on freshly generated worlds |
| smLANGUAGE_EN_001 | yes | yes | val loss on a disjoint `val.txt` |
| smLANGUAGE_RENDER_001 | yes | yes | the dev tuple, **not** the reported test set |
| smLANGUAGE_en_GENERAL_001 | yes | yes | validation loss on a disjoint split |
| smLANGUAGE_en_SCH_001 | yes | yes | validation loss |
| smCONVERSATION_001 | yes | yes | validation loss |
| smSOFTWARE_ENGINEERING_001 | yes | yes | validation loss |
| smROUTER_01 | yes | yes | generated held-out all-right, **not** the hand-written test set |
| smTOOLS_COMPUTER_CLI_01 | yes | yes | hand-written standing (already its checkpoint metric; see below) |
| smMATH01 | yes | **no** | none exists |
| smMATH001-a | yes | **no** | none exists |
| smMATH_LANGUAGE_001 | yes | **no** | none exists |
| smCLM_01 | **no** | **no** | skipped: CPU-only, and its held-out score is its reported result |
| smRTS_01 | **no** | **no** | excluded: its sources are sha256-hashed into run manifests, mid-experiment |

Only the forward pass and loss are autocast. Backward, gradient clipping and the optimizer step stay
in fp32, and **every evaluation runs in fp32 with no autocast**, checked with an AST scan over each
file's eval, scoring and prediction functions. So every number a trainer logs or selects on stays
comparable with the results already in [results.md](../docs/engineering/research/results.md).

**Five more free confirmations that bf16 costs nothing.** Paired smoke runs, same seed and steps:

| Trainer | bf16 val | fp32 val |
|---|---:|---:|
| smLANGUAGE_en_SCH_001 | 4.4942 | 4.4942 |
| smCONVERSATION_001 | 3.8396 | 3.8394 |
| smSOFTWARE_ENGINEERING_001 | 5.0405 | 5.0409 |
| smLANGUAGE_EN_001 | 3.2261 | 3.2260 |
| smLANGUAGE_en_GENERAL_001 | 1.0145 | 1.0144 |

### Four trainers keep the last checkpoint, not the best

Found while deciding what patience could key off, and verified by reading each file. These four have
**no best-versus-current comparison at all** — they overwrite one path every evaluation, so whatever
the last evaluation produced is what survives:

| Trainer | What it saves |
|---|---|
| smMATH01 | `save()` called unconditionally every eval; no `best` variable exists |
| smMATH001-a | unconditional `torch.save` every eval |
| smMATH_LANGUAGE_001 | unconditional `torch.save` every eval |
| smROUTER_01 | one `torch.save`, **outside** the epoch loop: always the final epoch |

This is the same failure the engineering docs record having already fixed once in
smTOOLS_COMPUTER_CLI_01, which peaked at step 5,000 (53% exact) and fell to 47% by step 6,000 while
its saver overwrote the file every eval.

**smROUTER_01 is the one that matters today.** Its own README records that it overfits its generator,
that v1 peaked at epoch 1 and v2 at epoch 4 of 5. The `router.pt` the harness loads is therefore the
last epoch, not the peak. Early stopping now stops nearer the peak, which mitigates it but does not
fix it. **Nothing was restructured** — fixing checkpointing is a separate change and an owner's call.

**One honest caveat on smTOOLS_COMPUTER_CLI_01:** its patience keys off the hand-written test set,
because that is already the metric it uses to choose a checkpoint. That is selection on the reported
set, a pre-existing weakness this change formalises rather than introduces.

## 8. The checkpoint fix, and why it did nothing yet

All four trainers above now keep the best checkpoint. Each selects on a metric that is genuinely
held out, never on the hand-written set it reports as its headline number:

| Trainer | Selects on | Held out because |
|---|---|---|
| smMATH01 | mean accuracy over ops at the trained digit length | probes are generated fresh from a fixed seed |
| smMATH001-a | mean exact score across categories | those questions are removed from the training pool |
| smMATH_LANGUAGE_001 | generated held-out exact score | never `t`, the reported hand-written set |
| smROUTER_01 | generated held-out all-right | never the reported hand-written set |

**smROUTER_01 needed a different shape of fix.** It writes `router.pt` exactly once, at the end,
which is why the harness — which reloads on file change — never picks up a half-trained router
mid-run. Saving on improvement would have broken that. It now holds the best epoch's weights in
memory and still writes the file once at the end. Same guarantee, best weights.

**A bug this introduced, caught in smoke testing.** smMATH001-a and smMATH_LANGUAGE_001 score the
model *in memory* after the loop — the last step's weights. Making the save conditional meant
`results.json` started describing a different model than the `.pt` beside it. Both now reload the
saved checkpoint before final scoring and record `best_step`. smMATH01 was unaffected: it only
prints. smROUTER_01 got the same reload for the same reason.

### The router retrain, v3 (2026-09-17)

v2's exact settings and the same 58,420 generated examples, so bf16 and best-epoch selection are the
only differences — a two-way confound, stated rather than hidden.

| | v2 | v3 |
|---|---|---|
| hand-written all-right | **0.710** | **0.710** |
| generated all-right | 0.985 | 0.984 |
| hand-written, web off | 0.731 | **0.774** |
| arithmetic suite | 0.824 | **0.941** |
| LLM-baseline suite | 0.581 | 0.581 |
| seconds | 860 | **808** |

**The fix did not fire.** `best_epoch` came out 5 — the last one — so v3 saved exactly what the old
code would have. The reason is that the selection metric is saturated: generated all-right reaches
0.98 by epoch 4 and stops discriminating. Meanwhile the hand-written set moved
0.62 → 0.70 → 0.72 → 0.70 → 0.71. **Corrected:** this first read as "a peak at epoch 3 worth catching".
It is not. One message is 1.1 points on a 93-message set, so epochs 2-5 span two messages against a
4.7-point standard error, and the same config ran 0.731 (v1) and 0.710 (v3). Neither metric can rank
these epochs — the generated one saturates, the hand-written one is too small to resolve the difference.

So the mechanism is correct and currently inert for this trainer. Catching that peak needs a harder
generated held-out set — examples the model does not ace by epoch 4 — which is separate work and an
owner's call. Selecting on the hand-written set instead would corrupt the headline number and is not
an option.

v3 kept: ties on the headline, wins two side suites, 52s faster. v2 preserved in `out/v2/`.

**One edge worth knowing:** selection uses strict `>`, so a tie keeps the earlier checkpoint. On a
healthy run that is the least-overfit model at the peak. On a run that never scores above zero it
keeps an almost-untrained checkpoint — visible in the smoke logs above, and a signal the run failed
rather than a bug.

## Run it

```bash
python3 bench.py                                      # roofline, precision, compile, batch sweep
python3 train_precision.py --precision bf16 --minutes 20 --out out/bf16-20min
```

`train_precision.py` writes only inside `smEFFICIENCY_01/out/`, so it can never overwrite
`smLLM_01/out/ckpt.pt`.

## Files

| File | What it is |
|---|---|
| `bench.py` | The measurement: empirical roofline, five training configs, batch sweep |
| `train_precision.py` | smLLM_01's trainer, verbatim, plus `--precision` and its own `--out` |
| `out/bench_results.json` | Every number above, as recorded |
| `out/bf16-20min/` | The quality run: `log.csv` and best checkpoint |
