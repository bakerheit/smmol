# smLANGUAGE_RENDER_001

A small grounded reply renderer for SMMOL. It reads the structured state produced by
`paratroop_harness_02` and writes the final reply without asking a general-purpose LLM.

This is deliberately **not** the English foundation model. It does not try to learn English
as a language from a general corpus. Its narrow job is structured state in, short English
reply out. A future English model should use the separate `smLANGUAGE_EN_*` name.

## V1 boundary

V1 handles three narrow outputs:

- `answer`: grounded math, memory, file and fetched-web replies;
- `ask`: one short reason and exactly one question;
- `NEED: web_browser ...`: hand a search result back so the harness can read the page.

It does **not** handle `make` steps such as code, drafts or long plans. It also refuses
payloads that leave fewer than 300 of its 1,024 bytes for a reply. Those cases should stay
on Ministral until a later model proves it can do them.

## Model and data

- byte vocabulary: 256 bytes plus separator, end and padding tokens;
- default model: 6 layers, width 384, 6 heads, about 10.8M parameters;
- rotary positions and a key/value cache;
- only reply bytes contribute to training loss;
- generated lessons use the same payload keys the harness sends today;
- `dev.json` selects the checkpoint and `test.json` is read only after training.

The generated replies are intentionally canonical and short. Every number and URL in a
target must already occur in the input. `score.py` also rejects internal jargon, extra
questions and offers to run a lookup after the turn is over.

## Acceptance gate before harness integration

1. All offline tests pass.
2. At least 90% exact and 100% grounded on the untouched handwritten test set.
3. `NEED` and `ask` shapes are perfect.
4. No invented number or URL in generated spot checks.
5. CPU reply time is materially below the current 4–10 second Ministral Language call.
6. The harness keeps Ministral as a fallback for low confidence, oversized payloads and
   every `make` step.

## First training result

The 2026-09-16 run did not pass the gate. The best checkpoint was step 3,500: 18.2%
exact on the handwritten dev set and 0 of 11 grounded on the untouched test set. It
learned generated reply templates but did not reliably copy unseen names, numbers, URLs
or file contents. The checkpoint and results stay under `out/v2/` as negative evidence;
they are not wired into the harness.

## Run

```bash
python3 -m unittest discover -s tests
python3 train.py --steps 1000       # pilot
python3 train.py                    # full planned run
python3 speak.py payload.json
```

Training writes `out/language.pt`, `out/train.log` and `out/results.json`. The checkpoint
with the best exact score on `dev.json` is kept; the final weights are not automatically
accepted into the harness.
