# smRTS_01 Phase 0 results

**Result: the one-layer online traces pass; stacked online traces do not.** Phase 0 was run on 2026-09-17. The
`leaky`, `gated`, and additive `fast` cells reproduce full BPTT gradients to float64 precision for one layer while
keeping constant history storage. At two and four layers, the deliberately truncated cross-layer paths produce large
gradient errors. Phase 2 should therefore begin with one-layer online models. Deeper online models are ablations, not
part of the exact-gradient claim.

No model training was done in this phase. These are implementation, gradient, memory, and timing checks.

## Gate summary

| Gate | Result | Evidence |
|---|---:|---|
| One-layer gradient, 32 bytes | Pass | worst relative L2 error `6.40e-16`; minimum cosine rounds to `1.0` |
| One-layer carry, 128 bytes | Pass | worst relative L2 error `7.68e-16`; minimum cosine rounds to `1.0` |
| Deliberate no-carry mutant | Pass | mutant relative error `0.9565`, so the gate catches the bug |
| No double counting | Pass | traced-gradient norm ratios equal BPTT to 10 decimal places |
| Constant history memory | Pass | 10,000 steps/cell; zero live-tensor and RSS growth after warm-up |
| First-token parity | Pass | batch 1, batched stream 0, and TBPTT agree within `1e-12` |
| TBPTT controls | Pass | `delta` and `gru` forward, state-shape, and gradient smoke tests pass |
| Multi-layer exactness | Fail, as predicted | errors become material at depth 2 and can reverse direction at depth 4 |

The Phase 0 acceptance gate is one-layer exactness. That gate passes. Multi-layer bias is a measured limitation and a
promotion constraint for later phases.

## Environment

| Item | Value |
|---|---|
| Machine | Apple M5, 16 GiB unified memory |
| OS | macOS 26.6.1, arm64 |
| PyTorch | 2.8.0 |
| MPS | available |
| Gradient-check dtype | float64 |
| Gradient-check model | `dim=6`, one layer unless stated, `a=0.99` |

## Exact one-layer gradients

Each row compares the accumulated online gradient against a full, fixed-parameter BPTT unroll. “Worst parameter” is
the parameter group with the largest relative L2 error. The cosine column is the minimum over every parameter group.

| Cell | Bytes | Worst parameter | Worst relative L2 | Minimum cosine |
|---|---:|---|---:|---:|
| leaky | 32 | `cells.0.decay_logit` | `6.394e-16` | `0.9999999999999996` |
| leaky | 128 | `decoder.weight` | `5.551e-16` | `0.9999999999999994` |
| gated | 32 | `cells.0.norm.weight` | `5.030e-16` | `0.9999999999999999` |
| gated | 128 | `decoder.weight` | `5.593e-16` | `0.9999999999999996` |
| fast | 32 | `cells.0.value.weight` | `4.628e-16` | `0.9999999999999993` |
| fast | 128 | `cells.0.decay_logit` | `7.676e-16` | `0.9999999999999994` |

The 128-byte case is important: a trace implementation that only includes the current write can look plausible on a
short toy sequence. Zeroing the carried trace every step makes the leaky decay gradient miss BPTT by `0.9565`, which
confirms the long check is testing the recurrence rather than only the current byte.

## Depth bias

Logits still match full BPTT at every depth because the forward recurrence is identical. Gradients do not: the online
trainer cuts paths from lower-layer past states through later upper-layer states.

| Cell | Layers | Worst relative L2 | Minimum cosine | Parameter at minimum cosine |
|---|---:|---:|---:|---|
| leaky | 1 | `6.394e-16` | `1.0000` | `decoder.weight` |
| leaky | 2 | `0.5979` | `0.8256` | `cells.0.read.weight` |
| leaky | 4 | `0.6455` | `0.7902` | `cells.0.read.weight` |
| gated | 1 | `5.030e-16` | `1.0000` | `cells.0.decay_logit` |
| gated | 2 | `0.9534` | `0.5553` | `cells.0.value.weight` |
| gated | 4 | `2.6960` | `-0.9061` | `cells.1.decay_logit` |
| fast | 1 | `4.628e-16` | `1.0000` | `decoder.weight` |
| fast | 2 | `0.6837` | `0.7514` | `cells.0.out_norm.weight` |
| fast | 4 | `1.3191` | `-0.2287` | `cells.0.out_norm.weight` |

This is too large to wave away. The main recall comparison will use one online recurrent layer. A deeper online run
must be labelled `depth-truncated` and reported separately.

## Constant-memory check

This check uses the tiny Phase 0 model (`dim=8`, one layer; fast memory `2×4×4`) and float32. Measurements start after
500 warm-up steps. The acceptance bounds were no retained autograd graph, zero live-tensor growth, and at most 16 MiB
of RSS allocator noise.

| Cell | Steps | Time | Trace bytes | RSS growth | Tensor growth | Graph severed |
|---|---:|---:|---:|---:|---:|---:|
| leaky | 10,000 | 1.373 s | 8,224 | 0 B | 0 | yes |
| gated | 10,000 | 1.635 s | 16,416 | 0 B | 0 | yes |
| fast | 10,000 | 2.143 s | 16,512 | 0 B | 0 | yes |

Storage is constant in sequence length, but it is not free: trace memory grows linearly with independent streams.

## Matched-state promoted sizes

The promoted comparison has one recurrent layer and 384 state scalars. Vector cells use width 384. Fast and delta
use `4 heads × key_dim 8 × value_dim 12 = 384` state scalars. The constructor defaults to one layer so a later run
cannot accidentally claim multi-layer exactness. Trace sizes below are float32 and exclude model parameters,
optimizer state, activations, and the recurrent state itself.

| Cell | Parameters | State scalars/stream | Trace bytes, batch 1 | Trace bytes, batch 32 |
|---|---:|---:|---:|---:|
| leaky | 345,216 | 384 | 394,752 | 12,632,064 |
| gated | 541,824 | 384 | 787,968 | 25,214,976 |
| fast | 243,764 | 384 | 83,456 | 2,670,592 |
| delta, TBPTT | 244,788 | 384 | n/a | n/a |
| GRU, TBPTT | 837,888 | 384 | n/a | n/a |

This fixes a bad earlier comparison: `4×32×32` fast memory has 4,096 state scalars per layer and is not matched to a
384-wide vector cell. It remains available only as a larger-capacity ablation.

## Batch-1 CPU versus MPS

These are launch-overhead diagnostics, not training-throughput numbers: a tiny one-layer model, batch 1, 500 measured
steps after 30 warm-up steps, with an MPS synchronization around the timed region.

| Cell | CPU steps/s | MPS steps/s | CPU advantage |
|---|---:|---:|---:|
| leaky | 8,867 | 618 | 14.4× |
| gated | 7,661 | 600 | 12.8× |
| fast | 5,479 | 479 | 11.4× |

Use CPU for batch-1 trace debugging. This does not predict the winner for `online-batched` or TBPTT; those MPS runs
need their own end-to-end throughput measurements.

## What Phase 0 changed

- The primary capacity match is recurrent-state size, not parameter count.
- The promoted default is one layer; fast uses `heads=4`, `key_dim=8`, `value_dim=12`.
- Online exactness is limited to one recurrent layer.
- Delta and GRU stay TBPTT-only controls.
- Every recurrent state and eligibility trace lives outside the module and is detached after each byte.
- Layer-0 gated and fast cells consume one-hot bytes, avoiding an untraced learned-embedding write path.
- Half-lives are spread over the tested range instead of all starting near one byte.

## Reproduce

From the repository root:

```bash
PYTHONPATH=smRTS_01 python3 -m unittest discover -s smRTS_01/tests -p 'test_*.py' -v
PYTHONPATH=smRTS_01 python3 smRTS_01/online.py --cell leaky --steps 10000
PYTHONPATH=smRTS_01 python3 smRTS_01/online.py --cell gated --steps 10000
PYTHONPATH=smRTS_01 python3 smRTS_01/online.py --cell fast --steps 10000
```

The focused suite currently has 11 passing tests. The next deliverable is Phase 1's generated recall world; no recall
or language-training claim has been made yet.
