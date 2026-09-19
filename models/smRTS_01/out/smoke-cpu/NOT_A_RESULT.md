# Not a result

This directory is the Phase 2 **S0 smoke test**: a 0.7-minute, `dim=32`, batch-8 CPU run whose only
job was to prove `train_recall.py` runs end to end and writes its artifacts. Its numbers are
meaningless as science — the model is a fraction of the promoted size and saw 42 seconds of data.

The real Phase 2 evidence is in `../S1-leaky-tbptt-frozen128/` and `../S2-fast-tbptt-calibration/`,
and it is written up in `../../phase2_results.md`.
