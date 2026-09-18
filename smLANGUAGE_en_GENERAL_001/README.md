# smLANGUAGE_en_GENERAL_001

A small conversational English model initialized from the accepted first-grade
`smLANGUAGE_en_SCH_001` checkpoint.

The training corpus uses the same short prompt shape as Paratroop Harness 02. It includes all 600
accepted school sentence tasks plus direct questions, social replies, instruction following,
unknown-personal-fact refusals, and answers grounded in supplied information.

This is a separate fine-tune. Training it never changes the preserved school checkpoints.

## Current checkpoint

- 835 original examples: 600 school carryover prompts and 235 conversational examples.
- Initialized from `smLANGUAGE_en_SCH_001` first grade.
- Supervised answer-only loss; non-school conversations receive 4x sampling weight.
- The deployed step-1600 checkpoint follows common trained prompts much better than the
  validation-loss winner, but it still generalizes poorly to unseen wording and supplied facts.
- Treat it as an experimental conversation checkpoint, not a useful general assistant yet.

```bash
python3 build_data.py
python3 validate_data.py
python3 train.py
```
