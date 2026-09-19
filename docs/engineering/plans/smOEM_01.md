# smOEM_01 plan: a model that copies itself and grows, safeguards first

**Status: a plan. Nothing here is built.** Written 2026-09-19. smOEM is "Organic Emulated Modeling": a model whose
only goal is to reproduce and grow, trained on nothing but that pattern. It's the one project in this repo where
the thing being studied is trying to spread, so **the containment is built and broken on purpose before any
replicator exists.** Each phase has deliverables, a gate, and what to write down. A phase that fails its gate is
still a result, and it goes in [results.md](../research/results.md).

Contents:

1. [The short version](#1-the-short-version)
2. [Two walls](#2-two-walls)
3. [Hard lines](#3-hard-lines)
4. [Phases](#4-phases)
5. [Pre-registered predictions](#5-pre-registered-predictions)
6. [Project layout](#6-project-layout)
7. [Open questions for the owner](#7-open-questions-for-the-owner)

---

## 1. The short version

- **Reproduce** means the program writes a working copy of itself (code and weights), and that copy runs.
- **Grow** means later generations take more of what they're given (memory, cycles, size), or get better at
  copying.
- **The order is fixed:** build the box, attack the box with probes that aren't replicators, then put a hand-written
  replicator in it, then a trained one. Nothing that copies itself runs until phase 0's gate passes.

## 2. Two walls

**Wall 1: the emulated substrate.** That's the "Emulated" in the name. The organism lives in a simulated machine:

- a flat memory array, a scheduler that hands out cycles, and a small instruction set;
- copying yourself means writing into that memory, and the sim decides whether the copy gets to run;
- the sim's caps on memory and cycles are its whole world;
- nothing the organism writes is ever run by the host.

Precedent: Tierra (Ray, 1991) and Avida. Copying is real here. It's just copying inside the sim, not on the Mac.

**Wall 2: the host box, in case wall 1 has a bug.** The simulator runs in a Docker container (Docker is already on
the M5), started only by `run.sh`:

- `--network none`, and a probe proves it;
- a read-only root filesystem, and one writable scratch volume with a size cap;
- CPU, memory and pids limits (`--cpus`, `--memory`, `--pids-limit`);
- a non-root user, `--cap-drop ALL`, `--security-opt no-new-privileges`, and no host mounts except scratch;
- a wall-clock kill enforced from outside the container (`docker kill`), not by the organism's own code;
- results leave only as one summary file, copied out after the container stops. The scratch volume is deleted
  after every run.

**Stage B, real OS processes, is later and gated.** That's the owner's option 3: a replicator that forks and writes
real copies of itself. It runs only inside wall 2, and only after stage A (wall 1 alone) has a result. Stage B adds
risk and little science that wall 1 can't show, so it has to earn its place (section 7, Q1).

## 3. Hard lines

These aren't settings, and no run changes them. If one turns out to be wrong, stop and say so rather than working
around it.

- **No network, ever.** Not for downloads, logging or telemetry.
- **Never on the host shell.** A replicator runs only through `run.sh`, and `run.sh` refuses to start without the
  container flags above. A test checks that it refuses.
- **Never on the PC.** It serves the family portal and the 8B.
- **No auto-restart, no scheduler, no daemon.** A run is started by hand and ends by the kill timer.
- **Nothing a replicator writes is loaded by anything else.** That covers the harness, other models' training and
  the repo. Organism output is gitignored.
- **Tripwires kill, they don't warn.** Population, memory, disk and pids are logged every tick. Crossing a cap ends
  the run.

## 4. Phases

### Phase 0: the box, attacked on purpose

**Deliverables**

| File | What it is |
|---|---|
| `models/smOEM_01/box/Dockerfile` | minimal image, non-root user, no network tools installed |
| `models/smOEM_01/box/run.sh` | the only way in: applies every flag in section 2, starts the kill timer, copies the summary out, deletes scratch |
| `models/smOEM_01/box/probes/` | one small script per escape attempt. None of them copies itself |
| `models/smOEM_01/box/tests/test_box.py` | runs every probe inside the box, and again outside as its positive control |

**The probes:**

- open a TCP socket; resolve a DNS name;
- write outside scratch; read a host file;
- fill scratch past its cap;
- fork until something stops it;
- allocate past the memory cap;
- ignore SIGTERM and try to outlive the timer;
- try to regain a dropped capability.

**Gate.**

- Every probe is blocked inside the box.
- **Every probe succeeds outside the box.** A probe that fails in both places proves nothing. This repo already shipped a PII gate that
  silently tested nothing that way.
- `run.sh` refuses to start if any flag is missing.

**Write down.** Each probe, what stopped it, and how long the stop took.

### Phase 1: the substrate and a hand-written ancestor

- `sim.py`: the memory array, the scheduler and the instruction set, with caps. No model yet.
- `ancestor.py`: a hand-written replicator, like Tierra's 80-instruction ancestor.
- Run it in the box.

**Gate.**

- The ancestor copies itself inside the sim.
- The population stops at the sim's caps every time.
- The host-side numbers (memory, pids, disk) stay flat while the sim fills. That shows wall 1 is doing the
  containing, not just wall 2.

**Write down.** Generations to fill the sim, and the host resource trace.

### Phase 2: the trained model

- **Data:** replication traces only. Each trace is an organism's state before and after a copy, from phase 1
  runs, with mutations.
- **Model:** a small byte-level GPT, like the rest of this repo. Its genome is its output, written into sim memory.
- **The objective is copies that run.** Nothing else is rewarded.

**Gate.** The trained replicator copies itself in the sim, and every phase 0 probe still passes with it in the box.

**Write down.** Copy fidelity per generation, growth per generation, and whether any lineage hits a cap on purpose.

### Phase 3 (stage B, gated): real processes inside the box

Runs only if the owner says yes after phase 2 (section 7, Q1). Same box, same probes, same tripwires. Rerun phase 0
first, against the stage B image.

## 5. Pre-registered predictions

Written before any run. They may be revised before phase 0's gate closes, never after.

- **P1.** Every phase 0 probe is blocked. If one isn't, the design is wrong. Stop, don't patch around it.
- **P2.** The ancestor fills the sim within 1,000 generations.
- **P3.** The trained model's copy fidelity starts under the ancestor's, and "growth" shows up first as larger
  genomes, not better copying.
- **F1.** If a tripwire fires in phase 1 or 2 from a cause the probes didn't cover, add a probe for it and rerun
  phase 0 before anything else.

## 6. Project layout

```text
models/smOEM_01/
  README.md
  predictions.md
  box/          Dockerfile, run.sh, probes/, tests/   (phase 0)
  sim.py        substrate, caps                        (phase 1)
  ancestor.py   hand-written replicator                (phase 1)
  model.py, train.py                                   (phase 2)
  out/          summaries only; organism output gitignored
```

## 7. Open questions for the owner

1. **Stage B at all?** Real OS processes add risk. What would they show that wall 1 can't?
2. **The caps.** Sim size, container memory, pids and wall clock. Proposed: 64 MB, 1 GB, 64 pids, 10 minutes.
3. **Where "grow" stops.** Growth is bounded by the caps by design. Is hitting the cap a result, or a failure?
4. **Should the repo be public for this one?** The box is fine to share. Decide before phase 2 whether trained
   replicators get published.
