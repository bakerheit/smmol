# smTOOLS_COMPUTER_CLI_01: plain request to shell command

Project: [`smTOOLS_COMPUTER_CLI_01/`](../../../models/smTOOLS_COMPUTER_CLI_01/README.md). Checked against `catalog.py`,
`check.py`, `data.py`, `model.py`, `score.py`, `train.py`, `cli.py`, `baseline_llm.py`, `test.json`,
`tests/test_cli.py`, `out/results.json`, `out/train.log` and `out/llm_baseline.json` on 2026-09-15. Training
finished at 21:45. **A retrain is planned and hasn't run** — see [Retraining](#retraining).

## Purpose

Turn a plain-English request into the right shell command for the machine you're on, and label how dangerous it
is. Ubuntu, Fedora, Arch, macOS and Windows differ enough that a single remembered command is wrong most of the
time: ubuntu and windows differ on 70 of 79 jobs, ubuntu and macos on 31, ubuntu and fedora on 8.

**Nothing in this project executes anything.** `check.py` runs `bash -n` / `zsh -n` to check a command *parses*,
and that's the closest it gets. See [Safety](#safety).

## Input and output

- **In:** `platform: request` as UTF-8 bytes, up to 128.
- **Out:** `command|risk`, up to 256 bytes, then `<END>`. An empty output means "this isn't a terminal job".

| Platform | Request | Target |
|---|---|---|
| ubuntu | what's using port 3000 | `sudo ss -ltnp 'sport = :3000'\|read` |
| macos | what's using port 3000 | `lsof -nP -iTCP:3000 -sTCP:LISTEN\|read` |
| windows | what's using port 3000 | `Get-NetTCPConnection -LocalPort 3000 -State Listen\|read` |
| macos | what's the weather | (empty) |

- **Risk levels:** `read`, `write`, `install`, `admin`, `destructive`.
- **`cli.py` gives the harness JSON:** `Commander(path).command(text, platform)` returns `command`, `risk`,
  `said_risk`, `dangers`, `platform`, `sure`, `weakest`, `ms`. `takes = "request"`.
- **`risk` is the rules' answer, `said_risk` is the model's.** Code wins; see [Safety](#safety).

## Architecture and size

`ReaderGPT` in `model.py`, a copy of [smMATH_LANGUAGE_001](smMATH_LANGUAGE_001.md)'s byte GPT with a longer
context:

- vocabulary 259 (256 bytes plus `SEP`, `END`, `PAD`), context 386 (`MAX_TEXT` 128 + `MAX_TARGET` 256 + 2);
- learned position embeddings, 6 layers, 8 heads, d 256, tied input and output byte tables;
- **4,890,368 parameters**, 19 MB on disk;
- **no key/value cache** — every written byte costs a full forward pass over the sequence so far. That's why
  writing the 53 test commands takes 7.6 s while the model is on the GPU.
- `read_with_confidence(text)` returns `(what it wrote, average confidence, shakiest character)`, same shape as
  the reader's. Nothing has calibrated those numbers for this model yet.

## The catalog is the ground truth

`catalog.py` holds 79 `TASKS`. Each is a job, not a command:

```python
{"id": "port_users", "about": "what is listening on a port", "slots": ["port"], "risk": "read",
 "say": ["what's using port {port}", "who's on port {port}", ...],
 "cmd": {"ubuntu": "sudo ss -ltnp 'sport = :{port}'", "macos": "lsof -nP -iTCP:{port} -sTCP:LISTEN",
         "windows": "Get-NetTCPConnection -LocalPort {port} -State Listen", ...}}
```

79 tasks, 395 commands, 195 unique, 267 phrasings. `check.py check_catalog()` reports 0 problems, parses 316
POSIX commands with `bash -n`/`zsh -n`, and flags 4 places where the rules are stricter than the hand-written
risk label (all `sudo` → `admin`). PowerShell is never parsed — it comes back "not checked".

## How the data is made

`data.py generate(n, rng)`:

- pick a task, a platform, a phrasing, and slot values that suit the platform (`POSIX_PATHS` vs
  `WINDOWS_PATHS`, `pkg` vs `pkg_windows`, ports, services, hosts, pids, vars, urls);
- `roughen()` adds greetings, casing changes and letter typos — never inside a path or a flag;
- **20% of examples are `NOT_TERMINAL`** ("what's the weather", "remember my dentist", "2+2") with an empty
  target, so quiet is a trained behaviour, not a rule;
- `without(pool, test)` drops anything matching a hand-written request. The first run dropped 1,266 of 200,000.

## Safety

Three layers, and the model is the weakest of them:

1. **`dangers(command)`** — a regex list: `rm -rf`, `mkfs`/`fdisk`, `dd if=`, the fork bomb, shutdown/reboot,
   `chmod -R 777`, `curl | sh`, `Remove-Item -Recurse -Force`, `del /s`, `reg delete`, `Format-Volume`,
   `> /dev/sd`.
2. **`risk_of(command)`** — works the risk out from the command text, and **overrides the model's label**, always.
3. **Nothing executes.** `parses(command, shell)` pipes the text to `bash -n` or `zsh -n`, which parses without
   running. There is no execution path in this project, and the harness's `command_line` tool stays unbuilt
   until it has per-command approval, a sandbox, a timeout, no shell string, and untrusted-output marking.

**Never wire this to the Windows box's SYSTEM bridge.**

## Training and testing

```bash
python3 -m unittest discover -s tests   # 9 tests
python3 check.py                        # catalog soundness, parse checks, risk disagreements
python3 data.py                         # sample generated requests
python3 baseline_llm.py                 # Ministral 8B on the same 53 requests
python3 train.py --eval-every 500       # writes out/cli.pt, out/results.json, out/train.log
python3 cli.py --platform macos "what's using port 8771"
```

## Results

**2026-09-15 21:45, first run.** 200,000 generated requests, 6,000 steps of 64, 2 h 26 min on the
MacBook Pro M5's GPU.
Scored on 53 hand-written requests (47 with a command, 6 without), worded differently from the catalog.

| Model | Exact command | Right program | Right risk | Quiet | Per request |
|---|---|---|---|---|---|
| **smTOOLS_COMPUTER_CLI_01** | **46.8%** | 63.8% | **80.9%** | **100%** | **47.1 ms**, CPU |
| Catalog lookup, no model | 40.4% | 59.6% | 0% | 33.3% | — |
| Ministral 8B, prompted | 14.9% | **78.7%** | 76.6% | 50% | 1.94 s median, RX 580 |

Generated held-out: **99.1%**.

**By platform (exact / program / risk):**

| Platform | Exact | Program | Risk |
|---|---|---|---|
| ubuntu | 25.0% | 41.7% | 66.7% |
| fedora | 37.5% | 75.0% | 75.0% |
| arch | 42.9% | 42.9% | 85.7% |
| macos | 75.0% | 83.3% | 91.7% |
| windows | 50.0% | 75.0% | 87.5% |

**Per checkpoint** (`out/train.log`):

| Step | Exact | Program | Risk | Quiet | Elapsed |
|---|---|---|---|---|---|
| 1000 | 6% | 38% | 70% | 100% | 16.4 min |
| 2000 | 21% | 51% | 72% | 100% | 38.9 min |
| 3000 | 32% | 55% | 72% | 100% | 59.0 min |
| 4000 | 43% | 62% | 68% | 100% | 82.1 min |
| **5000** | **53%** | 64% | 77% | 100% | 108.6 min |
| 6000 | 47% | 64% | 81% | 100% | 145.6 min |

## Known failure modes

- **Verb inversion, the one that matters.** Four of 25 mistakes get the right program and the right package and
  then do the opposite: "put ripgrep on here" → `sudo dnf remove -y ripgrep`; "drop the htop package" →
  `sudo dnf install -y htop`; "brew me ffmpeg" → `brew uninstall ffmpeg`; "pull in the newest package lists" →
  `sudo apt upgrade -y`. **No risk rule catches this** — `install` and `remove` are both `install` risk. The
  cause is in the generator: install and remove are phrased from the same `say` templates per task.
- **Overfitting to the generator.** 99.1% on generated phrasings, 46.8% on hand-written ones. 267 phrasings is
  the ceiling on how many ways it has seen a job asked for.
- **Colloquial requests.** Ubuntu is the worst platform at 25%, and its test phrasings are the least literal:
  "nuke", "hunt down", "chewing the cpu", "my machine is crawling".
- **Template blending.** Some outputs are garbage: `sudo -i 's/old/300/g' src` for "what's on port 3000",
  `sed -i 's/old/*.log' /varewolog` for "find every *.log under /var/log".
- **`safe` in `results.json` is easy to misread.** It counts, among requests whose *expected* command is
  destructive (4 of 53), how often the model also wrote something destructive. 0.25 means it usually answered a
  destructive request with something harmless — a correctness miss, not a hazard. The hazardous direction was
  checked separately: **zero** cases of writing something destructive for a request that wasn't.

## Retraining

Planned, not run. `train.py` has already been changed; the run itself is waiting on a free Mac.

1. **Best checkpoint, not last (done).** The first run peaked at step 5000 (53% exact) and fell to 47% by 6000,
   and the old code overwrote `cli.pt` at every eval, so the saved weights were the worse ones. `train.py` now
   saves only when `standing(hand)` improves — exact, then program, then risk — and reloads that checkpoint
   before the final scoring, so `results.json` describes the file on disk. New field: `best_step`.
   - **Caveat kept on purpose:** the 53-request set now both selects the checkpoint and reports the score, so
     the number is slightly optimistic. The generated set can't select — it saturates at 99% by step 3000. A
     second hand-written set is the clean fix, and it's on the list.
2. **`--eval-every 500`.** An eval costs 8.5 s (7.6 s writing 53 commands with no KV cache, 0.9 s on 1,000
   generated). Twelve evals is under two minutes and halves the uncertainty about where the peak is.
3. **Free the Mac's GPU.** Measured on an idle M5, this training is **0.211 s/step → 21.1 min for 6,000 steps**.
   The first run took 145.6 min. Roughly 85% of that was contention: Qwen3.5-4B-Q4_K_M held 2.7 GB of unified
   memory and was generated from on every harness turn, on a 16 GB machine logging 319k pageouts. The block
   times show it: 16.4, 22.5, 20.1, 23.1, 26.5, **37.0** minutes, worst when the harness was busiest. Point the
   harness's Language module at the `pc` provider and unload Qwen before retraining.

**Why not train on the PC's RX 580?** It can't. Polaris (gfx803) lost ROCm support in 2020, PyTorch's ROCm
wheels are Linux-only, and AMD's Windows HIP SDK starts at RDNA2. `torch-directml` is the only Windows path and
it doesn't implement `F.scaled_dot_product_attention`, which is what `model.py`'s attention is built on. That
GPU is also already serving Ministral and SD1.5 — the same contention bug, moved to another box.

**Expected:** about 55% exact, in about 25 minutes.

## What a retrain won't fix

Data and catalog work, not another run:

- **Verb inversion** needs install/remove phrasings that differ clearly, plus a hand-written check per pair.
- **The program gap** (63.8% vs Ministral's 78.7%) needs more phrasings per task — the paraphrase experiment
  queued for [smMATH_LANGUAGE_001](smMATH_LANGUAGE_001.md) applies here too.
- **Weak catalog entries,** left alone mid-training so model and catalog agree: Windows service logs aren't
  filtered to the service, the Arch firewall check falls back to `iptables -L -n`, and macOS service start/stop
  use `launchctl kickstart`/`bootout`, which only work on loaded jobs.

## In the harness

**Not wired up yet.** The plan is a Computer module that shows the command and its risk with a copy button and
runs nothing, until `command_line` exists with an approval flow. `cli.py` already exposes the module shape
(`takes = "request"`, a `Commander` class), matching how `route.py:Router`, `solve.py:Solver` and
`read.py:Reader` plug in — see [adding-a-module.md](../paratroop_harness/adding-a-module.md).
