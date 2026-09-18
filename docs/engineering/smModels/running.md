# Training and testing the models

Commands that exist in the code on 2026-09-15, with each script's defaults. Run each one from its project folder, for
example `cd ~/workspace/SMMOL/smROUTER_01`.

Harness commands are in [../paratroop_harness/running.md](../paratroop_harness/running.md).

## Before you train

- **One GPU job at a time on the MacBook Pro M5.** Training scripts pick `mps` when it's available. smCLM_01
  trains on the CPU,
  and smALLM_01 takes `--device auto`.
- **`baseline_llm.py` uses the PC's GPU.** It calls Ministral 8B at `http://127.0.0.1:8081`, which is shared.
- **Training overwrites `out/`.** Copy anything you want to keep first. smROUTER_01's first run was kept by hand in
  `out/v1/`.
- **The harness picks up checkpoints mid-training** for models it lists: smROUTER_01, smMATH01, smMATH_LANGUAGE_001 and
  smMATH001-a. Only smROUTER_01 saves once, at the end. Switch the module off on the page while its model trains.
- **PyTorch** is needed for every project here.

## What each project has

| Project | Tests | Train | Evaluate | Try it | LLM baseline |
|---|---|---|---|---|---|
| smLLM_01 | none | `train.py` | in `train.py` | `sample.py` | none |
| smALLM_01 | `world.py` self-check | `train.py` | `evaluate.py`, `check_curious.py` | `demo.py` | none |
| smCLM_01 | 7 | `train.py` | in `train.py` | `ask.py` | none |
| smROUTER_01 | 9 | `train.py` | in `train.py` | `route.py` | `baseline_llm.py` |
| smMATH01 | 8 | `train.py` | `evaluate.py` | `ask.py` | none |
| smMATH_LANGUAGE_001 | 6 | `train.py` | in `train.py` | `read.py`, `data.py` | `baseline_llm.py` |
| smMATH001-a | 6 | `train.py` | in `train.py` | `solve.py`, `work.py` | none |
| smTOOLS_COMPUTER_CLI_01 | 9 | `train.py` | in `train.py` | `cli.py`, `check.py`, `data.py` | `baseline_llm.py` |

---

## smLLM_01

```bash
python3 train.py --minutes 20
python3 sample.py --prompt "JULIET:" --temp 0.8
```

- **`train.py` flags:** `--data data/input.txt`, `--out out`, `--minutes 20`, `--batch 64`, `--lr 1e-3`,
  `--eval-every 250`, `--ctx 256`, `--d 384`, `--layers 6`, `--heads 6`, `--dropout 0.2`, `--seed 1337`.
- **It writes:** `out/ckpt.pt`, saved only when val loss improves, plus `out/log.csv`. It prints progress and samples.
- **`sample.py` flags:** `--ckpt out/ckpt.pt`, `--prompt ROMEO:`, `--tokens 500`, `--temp 0.8`, `--top-k 50`,
  `--seed`. Write `\n` in a prompt for a new line.

## smALLM_01

```bash
python3 world.py
python3 train.py --minutes 20
python3 evaluate.py
python3 demo.py --policy curious --seed 7
python3 check_curious.py
```

- **`world.py`** checks the gadget world's logic, then prints switches per world, how many flips change something
  else, and an example wiring.
- **`train.py` flags:** `--out out`, `--minutes 20`, `--batch 64`, `--steps 24` (experiments per episode), `--lr 1e-3`,
  `--eval-every 250`, `--d 256`, `--layers 6`, `--heads 8`, `--device auto`, `--seed 1337`.
  - It writes `out/ckpt.pt`, the best score after 8 experiments on 64 worlds, plus `out/log.csv`.
  - At the end it writes the full table to `out/eval.md`.
- **`evaluate.py` flags:** `--ckpt out/ckpt.pt`, `--worlds 256`, `--device auto`. It prints the k-experiments table for
  random and curious flips.
- **`demo.py` flags:** `--ckpt`, `--experiments 16`, `--policy random|curious` (default curious), `--seed 7`,
  `--device`.
- **`check_curious.py`** takes no flags. It compares what random and curious flips pick, on 256 worlds with 20 flips.

## smCLM_01

```bash
python3 train.py
python3 train.py --seeds 1 --epochs 4
python3 ask.py car
python3 ask.py --ideas transportation control -machine
python3 ask.py --teach zorb
python3 ask.py --teach blip "the child eats the blip" "the blip grows in the garden"
python3 ask.py --read "the pilot drives people to the city"
python3 -m unittest discover -s tests
```

- **`train.py` flags:** `--sentences 80000`, `--epochs 8`, `--seeds 3`, `--batch 512`, `--d 128`, `--layers 2`,
  `--heads 4`, `--max-len 12`, `--lr 2e-3`, `--out out`.
  - It runs on the CPU.
  - It writes `out/models.pt` (seed 0's readers and the learned ideas) and `out/results.json`.
- **`ask.py`:**
  - a word looks up its ideas;
  - `--ideas` finds words by ideas, `-idea` rules one out, and it must be the last option;
  - `--teach WORD [SENTENCE ...]` teaches a new word, and with no sentences it uses `concepts.json`'s teach sentences;
  - `--read` shows the ideas in a sentence;
  - `--model out/models.pt` picks the checkpoint.
- **Tests (7):** the world keeps its rules, scoring, reader shapes, and masking doesn't leak.

## smROUTER_01

```bash
python3 -m unittest discover -s tests
python3 train.py
python3 route.py "what's 20% of 85"
python3 route.py --prev "Which city?" "Denver"
python3 route.py --off web_search,web_browser "what's the weather in Denver tomorrow"
python3 baseline_llm.py
```

- **Tests (9):** generated labels are valid, answers follow questions, test messages never reach training, encoding,
  switched-off tools, random switches, network shapes, scoring, the rules baseline.
- **`train.py` flags:** `--examples 60000`, `--epochs 5`, `--batch 128`, `--lr 1e-3`, `--d 192`, `--layers 4`,
  `--heads 6`, `--max-len 192`, `--keep 0.8` (chance each tool stays on in a switched-off batch), `--device`,
  `--seed 0`, `--out out`.
  - It writes `out/router.pt` (once, at the end) and `out/results.json`.
  - It prints the table and every mistake, and adds the Ministral row if `out/llm_baseline.json` exists.
- **`route.py`:** `--prev` is the assistant's last message, and `--off` is a comma-separated list of tools that are off.
  It prints the JSON the harness gets.
- **`baseline_llm.py`** writes `out/llm_baseline.json`. It uses the PC's GPU.

## smMATH01

```bash
python3 -m unittest discover -s tests
python3 train.py
python3 train.py --variant abacus --steps 2000
python3 evaluate.py
python3 ask.py 48213+9977
python3 ask.py "1234567 * 89"
```

- **Tests (8):** problem lines in both orders, exact number lengths, place labels, answer-only scoring, the place
  shift, every variant runs, a tiny model learns 1-digit addition, the harness solver.
- **`train.py` flags:** `--variant plain|reversed|abacus|all` (default all), `--steps 8000` per variant, `--batch 256`,
  `--lr 1e-3`, `--d 256`, `--layers 6`, `--heads 8`, `--max-digits 6`, `--offset-max 25` (abacus only),
  `--eval-every 500`, `--seed 0`, `--out out`, `--device`. It writes `out/<variant>.pt` at every eval.
- **`evaluate.py` flags:** `--per 300` problems per operation and length, `--out`, `--device`.
  - It scores lengths 1 to 10 for every trained variant.
  - It writes `out/results.json` and prints the tables and some misses.
- **`ask.py`** puts one problem to every trained variant. Write it like `48213+9977`, `900-35` or `1234*56`, with the
  bigger number first for subtraction.
- **`solve.py`** has no CLI. It's the harness loader.

## smMATH_LANGUAGE_001

```bash
python3 -m unittest discover -s tests
python3 data.py
python3 train.py
python3 train.py --steps 1500
python3 read.py "what is the volume of 322234ft x 21323ft x 212231ft?"
python3 baseline_llm.py
```

- **Tests (6):** every generated target works out, targets become the harness list, hand-written messages never reach
  training, grading by value, only written problems are scored, the harness reader.
- **`data.py`** prints 20 generated messages and their targets (seed 3).
- **`train.py` flags:** `--examples 300000`, `--steps 6000`, `--batch 64`, `--lr 6e-4`, `--d 256`, `--layers 6`,
  `--heads 8`, `--eval-every 1000`, `--seed 0`, `--out out`, `--device`.
  - It writes `out/reader.pt` at every eval and `out/results.json` at the end.
  - The final table includes the harness's regex, imported from `../paratroop_harness_02/harness.py`, and the
    Ministral row if `out/llm_baseline.json` exists.
- **`read.py`** prints `{problems, ms}`. `--out` picks the folder.
- **`baseline_llm.py`** writes `out/llm_baseline.json` and `out/llm_baseline.log`. It uses the PC's GPU.

## smMATH001-a

```bash
python3 -m unittest discover -s tests
python3 work.py
python3 train.py
python3 train.py --steps 2000
python3 solve.py "(12+8)/5"
python3 solve.py "322234*21323*212231"
```

- **Tests (6):** small worked solutions, answers match exact arithmetic, generated sizes stay in bounds, only the work
  is scored, place labels, cached writing matches a full forward pass, the harness solver.
- **`work.py`** prints 8 generated expressions with their question, work and readable steps (seed 3).
- **`train.py` flags:** `--pool 300000`, `--steps 10000`, `--tokens 8192` per batch (padding included), `--lr 6e-4`,
  `--d 256`, `--layers 6`, `--heads 8`, `--offset-max 16`, `--eval-every 1000`, `--seed 0`, `--out out`, `--device`.
  - It writes `out/math.pt` at every eval and `out/results.json` at the end.
  - At the end it measures greedy writing on the CPU.
- **`solve.py`** prints the answer, the exact value, seconds, work length and readable steps. `--out` picks the folder.

## smTOOLS_COMPUTER_CLI_01

```bash
python3 -m unittest discover -s tests
python3 check.py
python3 data.py
python3 train.py --eval-every 500
python3 cli.py --platform macos "what's using port 8771"
python3 baseline_llm.py
```

- **Tests (9):** the catalog is sound, the danger rules fire, the risk rules agree, generated commands parse, no test
  request leaks into training, Windows paths stay Windows, scoring, the model and `Commander`, and checkpoint ranking.
- **`check.py`** parses every POSIX command with `bash -n`/`zsh -n`, lists risk-label disagreements and danger hits.
  It never runs a command. PowerShell is reported as "not checked".
- **`data.py`** prints generated requests with their platform and target.
- **`train.py` flags:** `--examples 200000`, `--steps 6000`, `--batch 64`, `--lr 6e-4`, `--d 256`, `--layers 6`,
  `--heads 8`, `--eval-every 1000`, `--seed 0`, `--out out`, `--device`.
  - **It keeps the best checkpoint**, ranked by `standing()`: exact commands, then right program, right risk, quiet.
    `out/cli.pt` is that checkpoint, and the final scoring reloads it, so `out/results.json` (`best_step`) describes
    the file on disk rather than the weights the loop ended on.
  - `--eval-every 500` is the recommended setting. One eval costs about 8.5 seconds.
  - The final table includes the no-model catalog lookup, and the Ministral row if `out/llm_baseline.json` exists.
- **`cli.py`** prints `{command, risk, said_risk, dangers, platform, sure, weakest, ms}`. `risk` comes from the rules
  and overrides `said_risk`. It never executes anything.
- **`baseline_llm.py`** writes `out/llm_baseline.json` and `out/llm_baseline.log`. It uses the PC's GPU.
