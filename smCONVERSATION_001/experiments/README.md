# smCONVERSATION_001 experiments

This folder contains controlled training experiments that do not change the production trainer or
the live conversation-generation pipeline.

Each experiment must keep its protocol, source, raw results, and findings together so later runs
can be reproduced without relying on shell history.

## Experiments

- [`muon/`](muon/README.md): AdamW versus hybrid Muon/AdamW optimization on the frozen
  25-conversation pilot corpus.
