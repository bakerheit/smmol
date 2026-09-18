# smSOFTWARE_ENGINEERING_001

A 1.85M-parameter byte-level code model initialized from the accepted first-grade
`smLANGUAGE_en_SCH_001` checkpoint. This first version studies local Python syntax and short code
continuations. Its 128-byte context is far too small for repository-scale reasoning, so it is not a
coding assistant yet.

## Pilot result

Trained September 16, 2026 on 72 files from three repositories, with all three files from
`dbader/schedule` held out. Against the untouched grade-one checkpoint on identical deterministic
samples:

| Checkpoint | Held-out code loss | School loss |
|---|---:|---:|
| Grade-one base | 6.161 | 0.524 |
| Software fine-tune | 2.022 | 0.526 |

That is 67.2% lower held-out code loss with a 0.4% school-loss increase. These are byte-prediction
metrics, not proof that the model can solve programming jobs. The saved result is `out/evaluation.json`.
Direct generation is still an acceptance failure: greedy probes after a function signature collapse
to spaces, while sampled probes produce malformed Python. Keep this model out of the assistant. The
pilot proves the data and transfer pipeline; it does not yet prove useful software-engineering behavior.

## Data contract

The collector uses [GitHub's REST API](https://docs.github.com/en/rest) through the authenticated
`gh` CLI. It never scrapes GitHub
HTML and it does not treat every public repository as training material.

```text
public repo metadata
    -> public/non-fork/non-archived gate
    -> explicit SPDX license allowlist
    -> exact default-branch commit and Git tree
    -> Python path/size/generated/secret filters
    -> exact-content dedupe
    -> records.jsonl + manifest + license text
    -> repository-held-out validation split
    -> code bytes + 25% grade-one replay
    -> grade-one checkpoint fine-tune
```

The initial allowlist is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, and Unlicense. Every
record keeps repository URL, exact commit, path, Git blob SHA, content SHA-256, byte count, and SPDX
license. The detected root license text is saved beside the raw corpus. Vendor, build, generated,
binary, overlong, and obvious-secret files are rejected. Collection is serial and bounded by file,
repository, and total-byte limits.

GitHub's [license detector](https://docs.github.com/en/rest/licenses/licenses) is useful but not a legal audit: it can miss per-file or nested-license
exceptions. Review the manifest and original repositories before publishing a corpus or weights.
Keeping provenance also makes later removals and rebuilds possible.

## Build, train, and test

The checked-in source list is a small Python pilot. Collection resolves the current default-branch
heads and records their exact commits:

```bash
python3 collect_github.py
python3 validate_data.py
python3 -m unittest discover -s tests -v
python3 train.py
python3 evaluate.py
```

Add `--repo owner/name` one or more times to collect a different explicit set. Use `--query` for
bounded discovery, for example:

```bash
python3 collect_github.py --query 'language:Python stars:>5000 license:mit size:<5000'
```

The collector still rechecks visibility, fork/archive state, and license after search. It needs at
least two accepted repositories because validation holds out whole repositories, not random snippets
from the same projects.

Training writes resumable PyTorch checkpoints plus portable float32 safetensors. The architecture is
kept identical to grade one so its weights can initialize directly. Twenty-five percent of sampled training
bytes replay the accepted school corpus to reduce catastrophic forgetting. `evaluate.py` compares
the base and tuned checkpoints on held-out repositories and on school text; lower loss is better.

Try a short continuation after training:

```bash
python3 inference.py $'def clamp(value, low, high):\n    '
```

## Tradeoffs and next scale step

- The tiny byte model is cheap and fully inspectable, but 128 bytes only cover local syntax.
- Holding out repositories is harsher and more honest than holding out files from the same repo.
- A permissive-license allowlist shrinks the corpus but makes provenance and reuse less muddy.
- Exact dedupe is deterministic; a larger crawl should add near-deduplication and an opt-out ledger.
- If held-out code improves without wrecking school loss, the next model should extend context and
  capacity in a new architecture, then train from a documented grade-one weight transfer rather than
  pretending the old positional weights fit.
