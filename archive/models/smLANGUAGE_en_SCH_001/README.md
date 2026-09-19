# smLANGUAGE_en_SCH_001

A school-curriculum English model built from original word-and-sentence pairs rather than another
downloaded prose dump.

## Curriculum contract

| Level | New target words | New sentences |
|---|---:|---:|
| Preschool | 100 | 100 |
| Kindergarten | 200 | 200 |
| 1st through 12th grade | 300 through 1,400 | 300 through 1,400 |
| **Total** | **10,500** | **10,500** |

Every sentence teaches one target word. Target words are unique across the full curriculum, appear
as whole words in their sentence, and never count twice at a later grade. Sentence-length bands grow
from 3-9 words in preschool to 16-35 words in 12th grade. Those are structural gates, not proof that
a sentence is educationally correct; each generated level also needs a human sample review.

The preschool level is hand-authored. Later levels will be generated in small, resumable batches,
structurally rejected when broken, and sampled for semantic review before they become training data.

## Status

- Curriculum shape fixed: 14 levels, 10,500 words and sentences.
- Preschool: 100/100 hand-authored, validated, and accepted.
- Kindergarten: 200/200 accepted after four semantic-review rounds and a human line-by-line pass.
- 1st grade: 300/300 accepted after five semantic-review rounds and a full human line-by-line pass.
- 2nd through 12th grade: not generated yet.
- Accepted total: 600/10,500 words and 600/10,500 sentences.
- A rejected 200-row Qwen trial is preserved under `data/quarantine` and excluded from the compiled corpus.
- Model code: configurable micro GPT, open-format weight export, stage checkpoints, and a Harness adapter are built.
- Grade-one checkpoint: initialized from the kindergarten checkpoint and trained on all 600 accepted
  rows. Best validation was 0.906 loss / 1.31 bits per byte at step 700. Direct sentence probes are
  still broken, so this is measurable language-model progress but not yet a useful chat model.

## Check the data

```bash
python3 validate_data.py
python3 -m unittest discover -s tests -v
```

Generate or resume one level with the locally loaded teacher model:

```bash
python3 generate_data.py --level kindergarten --endpoint http://127.0.0.1:64190
python3 review_data.py --level kindergarten
python3 compile_data.py --allow-partial
```

`generate_data.py` saves every accepted batch before requesting another. Structural validation is
automatic. It needs the loaded model's direct llama-server address because UnlimitedStudio's public
gateway currently strips strict JSON schemas; the port is shown in the llama-server process command.
It does not pretend to judge meaning; sample review is still required before training.

`review_data.py` is a separate skeptical pass. It can repair a sentence, or remove an unsuitable
target word so generation can fill the gap. It saves the pre-review batch, every rejection, and a
summary. Passing that review still does not replace the final human sample audit.

The compiled partial corpus and its SHA-256 manifest are `data/curriculum.jsonl` and
`data/manifest.json`. `compile_data.py` without `--allow-partial` refuses to finish until all 14
levels have their exact counts.

## Configurable model and weights

`model_config.json` is the source of truth. Its `architecture` section controls vocabulary size,
context length, width, layer count, attention heads, dropout, and tied embeddings. Its `generation`
and `training` sections hold editable defaults. Invalid shapes, such as a width that is not divisible
by the head count, stop with a clear error before training.

The current `micro` shape is deliberately small while the accepted corpus is small: 1.85M trainable
parameters, 128 byte context, width 192, four layers, and six heads. Making a layer or width change
creates a new architecture; it cannot be applied to an old checkpoint whose tensor shapes differ.

Train or resume the accepted curriculum through a school stage:

```bash
python3 train.py --stage kindergarten
python3 train.py --stage kindergarten --resume
```

The trainer reads only `data/curriculum.jsonl`, so unreviewed candidates sitting in `data/levels`
cannot leak into a checkpoint. For every stage it saves:

- `checkpoints/<stage>/latest.pt`: model, optimizer, RNG, curriculum hash, and resume state;
- `checkpoints/<stage>/best.pt`: best full checkpoint for that stage;
- `checkpoints/<stage>/weights.safetensors`: portable float32 inference weights;
- `model_config.json` and `metadata.json`: exact architecture, stage, item count, step, and data hash.

The best completed stage is also copied to `out/latest.safetensors` for Paratroop Harness 02. The
safetensors reader/writer is included in this project, so loading the open-format weights does not
need a package download. `export.py` can export any full checkpoint again.

On the Harness Language card, **Model / weights** chooses the checkpoint. When the school model is
selected, **Weight configuration** shows the loaded stage, step, parameter count, tensor shape,
weight-file size, and tokenizer. Context, width, layers, heads, dropout, and tied embeddings are
editable there for the next checkpoint; Harness validates the shape and writes `model_config.json`.
Those architecture changes require retraining and never lie about changing the already-loaded
tensors. A separate **Generation** section contains Temperature, Top K, repetition penalty, and token
limits, which take effect immediately.

The files are inspectable and editable now. Before publishing them as an open-weight release, pick
an explicit redistribution license for the code, original curriculum, and weights; file format alone
does not grant redistribution rights.
