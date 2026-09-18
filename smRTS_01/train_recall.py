"""Phase 2 recall training: one recurrent layer, staged, with every stop rule pre-registered.

Predictions, promotion gates and stop rules live in predictions.md and are not re-argued here. This
script only executes them and writes the evidence.

Training episodes come from `world.TRAIN_SEED`, best-checkpoint selection from `world.SELECT_SEED`,
and the reported grid from `world.EVAL_SEED`. The three sets are disjoint, so no checkpoint is
selected on the number that gets published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

import evaluate
import online as online_mod
import world
from cells import RTSModel, decay_half_life, half_life_to_logit

HERE = Path(__file__).resolve().parent
SOURCES = ("cells.py", "online.py", "tbptt.py", "world.py", "evaluate.py", "train_recall.py",
           "predictions.md")

# The Phase 2 training mix, fixed in predictions.md §4. Gap 8 is position-confounded and gap 512 is
# reported as extrapolation, so neither is trained on.
TRAIN_PAIRS = (2, 4, 8, 16, 32)
TRAIN_GAPS = (32, 128)


# --- provenance ---------------------------------------------------------------------------------
def source_manifest() -> Dict[str, str]:
    out = {}
    for name in SOURCES:
        path = HERE / name
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    return out


def world_constants() -> Dict[str, object]:
    return {
        "keys": world.KEYS, "values": world.VALUES, "filler": world.FILLER,
        "pair_counts": list(world.PAIR_COUNTS), "gaps": list(world.GAPS),
        "eval_seed": world.EVAL_SEED, "select_seed": world.SELECT_SEED,
        "train_seed": world.TRAIN_SEED, "eval_trials": world.EVAL_TRIALS,
        "select_trials": world.SELECT_TRIALS,
    }


def canary(device: torch.device, seconds: float = 3.0) -> Dict[str, float]:
    """A fixed micro-benchmark run immediately before and after every timed run.

    A slow `before` number means something else is on the box, and the run is not a clean wall-clock
    measurement. The threshold is applied by the caller, not hidden in here.
    """
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    for _ in range(3):
        a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < seconds:
        for _ in range(10):
            a = (a @ b).mul_(1e-3).add_(0.5)
            iterations += 1
        if device.type == "mps":
            torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    return {"matmuls_per_second": iterations / elapsed, "device": device.type}


def other_training_loads() -> List[str]:
    """Anything else on this box that looks like a training or serving job."""
    try:
        listing = subprocess.run(["ps", "-Ao", "pid,pcpu,comm,args"], capture_output=True,
                                 text=True, timeout=20).stdout
    except Exception as error:                                    # pragma: no cover - diagnostics
        return ["ps failed: %s" % error]
    mine = str(__import__("os").getpid())
    hits = []
    needles = ("train", "ollama", "llama", "vllm", "lm-studio", "LM Studio", "mlx_lm")
    for line in listing.splitlines()[1:]:
        fields = line.split(None, 3)
        if len(fields) < 4 or fields[0] == mine:
            continue
        pid, cpu, _, args = fields
        if float(cpu) < 20.0:
            continue
        if any(needle in args for needle in needles):
            hits.append("pid %s at %s%% CPU: %s" % (pid, cpu, args[:120]))
    return hits


# --- the training stream ------------------------------------------------------------------------
class EpisodeStream:
    """One independent stream of freshly generated, never-repeated training episodes."""

    def __init__(self, seed: int, stream_id: int, pairs: Sequence[int], gaps: Sequence[int]):
        self.rng = random.Random(world.derive_seed(seed, "train-stream", stream_id))
        self.pairs, self.gaps = tuple(pairs), tuple(gaps)
        self.seed, self.stream_id = seed, stream_id
        self.counter = 0
        self.buffer = bytearray()
        self.episodes_made = 0

    def _more(self) -> None:
        pairs = self.rng.choice(self.pairs)
        gap = self.rng.choice(self.gaps)
        self.counter += 1
        episode = world.sample_episode(
            world.derive_seed(self.seed, "train", self.stream_id, self.counter), pairs, gap)
        self.buffer += episode.data
        self.episodes_made += 1

    def take(self, count: int) -> bytes:
        while len(self.buffer) < count:
            self._more()
        out = bytes(self.buffer[:count])
        del self.buffer[:count]
        return out


class StreamBatch:
    """`batch` independent streams, sliced into `[batch, width]` byte columns."""

    def __init__(self, batch: int, seed: int, pairs: Sequence[int], gaps: Sequence[int]):
        self.streams = [EpisodeStream(seed, i, pairs, gaps) for i in range(batch)]
        self.bytes_served = 0

    def window(self, width: int) -> Tensor:
        """`width + 1` bytes per stream: inputs are [:-1] and targets are [1:]."""
        rows = [list(stream.take(width + 1)) for stream in self.streams]
        self.bytes_served += width * len(rows)
        return torch.tensor(rows, dtype=torch.long)

    @property
    def episodes_made(self) -> int:
        return sum(stream.episodes_made for stream in self.streams)


# --- the model ----------------------------------------------------------------------------------
def build_model(cell: str, args) -> RTSModel:
    torch.manual_seed(args.seed)
    model = RTSModel(cell, dim=args.dim, layers=args.layers, heads=args.heads,
                     key_dim=args.key_dim, value_dim=args.value_dim)
    if args.freeze_decay is not None:
        logit = float(half_life_to_logit(torch.tensor(float(args.freeze_decay))))
        with torch.no_grad():
            for module in model.cells:
                if hasattr(module, "decay_logit"):
                    module.decay_logit.fill_(logit)
                    module.decay_logit.requires_grad_(False)
    return model


def report_cells(full: bool = False) -> List[tuple]:
    """The reported cells: the gap row through pairs=16 and the pairs column at gap 128.

    Both axes are covered and the primary endpoint sits at their intersection. The full 5 x 4 grid
    is available behind `--full-grid` but is not run by default: 20 cells x 1,000 episodes, several
    of them 600+ bytes long, is a lot of forward passes to spend on descriptive numbers.
    """
    if full:
        return [(p, g) for p in world.PAIR_COUNTS for g in world.GAPS]
    row = [(16, g) for g in world.GAPS]
    column = [(p, 128) for p in world.PAIR_COUNTS if p != 16]
    return row + column


def resolve_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested and is not available")
    return torch.device(name)


# --- the training loops -------------------------------------------------------------------------
def tbptt_epoch_step(model: RTSModel, optimizer, chunk: Tensor, states, clip: float):
    inputs, targets = chunk[:, :-1], chunk[:, 1:]
    logits, new_states = model(inputs, states)
    loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="mean")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
    optimizer.step()
    return float(loss.detach()), [s.detach() for s in new_states]


def online_batched_step(model: RTSModel, optimizer, chunk: Tensor, state, clip: float):
    inputs, targets = chunk[:, :-1], chunk[:, 1:]
    total = 0.0
    for index in range(inputs.shape[1]):
        model.zero_grad(set_to_none=True)
        loss, _, state = online_mod.step(model, inputs[:, index], targets[:, index], state)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total += loss
    return total / inputs.shape[1], state


# --- the run ------------------------------------------------------------------------------------
def train(args) -> Dict[str, object]:
    device = resolve_device(args.device)
    out_dir = HERE / "out" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    contention = other_training_loads()
    canary_before = canary(device)
    if args.canary_reference and args.canary_reference > 0:
        drift = canary_before["matmuls_per_second"] / args.canary_reference
        if drift < 0.90:
            raise SystemExit(
                "contention canary is %.1f%% of the reference; the box is busy, not measuring"
                % (100 * drift))

    model = build_model(args.cell, args).to(device)
    if args.regime != "tbptt" and not model.supports_online:
        raise SystemExit("%s has no validated online trace; it is a TBPTT control" % args.cell)
    if args.regime != "tbptt" and args.layers > 1:
        raise SystemExit("online exactness is one layer only; use --depth-truncated to override")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    initial_half_lives = evaluate.half_life_summary(model)

    config = {
        "name": args.name, "stage": args.stage, "cell": args.cell, "regime": args.regime,
        "seed": args.seed, "device": str(device), "dim": args.dim, "layers": args.layers,
        "heads": args.heads, "key_dim": args.key_dim, "value_dim": args.value_dim,
        "batch": args.batch, "window": args.window, "lr": args.lr, "clip": args.clip,
        "minutes": args.minutes, "freeze_decay_half_life": args.freeze_decay,
        "train_pairs": list(TRAIN_PAIRS), "train_gaps": list(TRAIN_GAPS),
        "eval_every_seconds": args.eval_every,
        "futility_fraction": args.futility_fraction,
        "futility_mode": args.futility_mode,
        "parameters": model.parameter_count(),
        "state_scalars_per_stream": model.state_scalar_count(),
        "torch": str(torch.__version__), "python": sys.version.split()[0],
        "platform": platform.platform(), "machine": platform.machine(),
        "argv": sys.argv,
    }
    manifest = {"sources": source_manifest(), "world": world_constants(),
                "other_training_loads": contention}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    stream = StreamBatch(args.batch, world.TRAIN_SEED + args.seed, TRAIN_PAIRS, TRAIN_GAPS)
    state = None
    online_state = None
    if args.regime == "online-batched":
        online_state = online_mod.initialize(model, args.batch)
        config["trace_bytes"] = online_mod.trace_bytes(online_state)
        (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    futility_reference = evaluate.futility_reference(16, 32)
    log_path = out_dir / "log.csv"
    log_path.write_text("step,wall_s,bytes_seen,bytes_per_s,lr,train_nll,"
                        "select_acc,select_nll,select_rank,peak_mem_bytes,"
                        "halflife_median,halflife_p90\n")

    best = {"step": -1, "accuracy": -1.0, "nll": float("inf"), "wall_s": 0.0, "bytes_seen": 0}
    stop_reason = "time"
    started = time.perf_counter()
    budget = args.minutes * 60.0
    step = 0
    next_eval = 0.0
    train_nll_window: List[float] = []
    futility_checked = False
    results_futility = None
    rows: List[Dict[str, object]] = []

    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= budget:
            stop_reason = "time"
            break
        chunk = stream.window(args.window).to(device)
        if args.regime == "tbptt":
            if state is None:
                state = model.zero_state(args.batch, device=device)
            loss, state = tbptt_epoch_step(model, optimizer, chunk, state, args.clip)
        else:
            loss, online_state = online_batched_step(model, optimizer, chunk, online_state,
                                                     args.clip)
        step += 1
        train_nll_window.append(loss)

        elapsed = time.perf_counter() - started
        if elapsed >= next_eval or elapsed >= budget:
            next_eval = elapsed + args.eval_every
            selection = evaluate.selection_score(model)
            halves = evaluate.half_life_summary(model)["per_layer"]
            median = halves[0]["median"] if halves else float("nan")
            p90 = halves[0]["p90"] if halves else float("nan")
            peak = int(torch.mps.driver_allocated_memory()) if (
                device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory")) else 0
            bytes_seen = stream.bytes_served
            row = {
                "step": step, "wall_s": round(elapsed, 3), "bytes_seen": bytes_seen,
                "bytes_per_s": round(bytes_seen / max(elapsed, 1e-9), 1),
                "lr": args.lr,
                "train_nll": round(sum(train_nll_window) / len(train_nll_window), 5),
                "select_acc": round(selection["accuracy"], 5),
                "select_nll": round(selection["nll"], 5),
                "select_rank": round(selection["rank"], 3),
                "peak_mem_bytes": peak,
                "halflife_median": round(median, 3), "halflife_p90": round(p90, 3),
            }
            rows.append(row)
            with log_path.open("a") as handle:
                handle.write(",".join(str(row[k]) for k in
                                      ("step", "wall_s", "bytes_seen", "bytes_per_s", "lr",
                                       "train_nll", "select_acc", "select_nll", "select_rank",
                                       "peak_mem_bytes", "halflife_median", "halflife_p90")) + "\n")
            print("  ".join("%s=%s" % (k, row[k]) for k in
                            ("step", "wall_s", "bytes_seen", "bytes_per_s", "train_nll",
                             "select_acc", "select_nll")), flush=True)
            train_nll_window = []

            # Selection is accuracy first, answer NLL as the tie-break. Without the tie-break a run
            # that never gets an answer right keeps its step-1 weights and "best checkpoint" means
            # "earliest checkpoint", which is how the first smoke run silently reported an
            # untrained model.
            better = (selection["accuracy"], -selection["nll"]) > (best["accuracy"], -best["nll"])
            if better:
                best = {"step": step, "accuracy": selection["accuracy"], "nll": selection["nll"],
                        "wall_s": round(elapsed, 3), "bytes_seen": bytes_seen,
                        "rank": selection["rank"]}
                torch.save({"model": model.state_dict(), "best": best}, out_dir / "ckpt.pt")

            if args.gate_accuracy and selection["accuracy"] >= args.gate_accuracy:
                # Not an early pass: the gate is decided by the full frozen EVAL_SEED evaluation
                # below. This only stops burning MPS time once selection is clearly past it.
                stop_reason = "gate"
                break

            if (not futility_checked and args.futility_fraction
                    and elapsed >= args.futility_fraction * budget):
                futility_checked = True
                # predictions.md registers this at (pairs=16, gap=32), not at the selection cell.
                check = evaluate.selection_score(model, pairs=16, gap=32)
                margin = futility_reference["alphabet_nll"] - check["nll"]
                results_futility = {"nll_at_16_32": check["nll"], "margin": margin,
                                    "wall_s": round(elapsed, 3)}
                print("futility check at %.0fs: NLL(16,32)=%.3f vs alphabet floor %.3f, margin %.3f"
                      % (elapsed, check["nll"], futility_reference["alphabet_nll"], margin),
                      flush=True)
                if margin < 0.3 and args.futility_mode == "kill":
                    stop_reason = "futility"
                    break
                if margin < 0.3:
                    print("futility margin missed but --futility-mode=report: continuing",
                          flush=True)

    wall = time.perf_counter() - started
    canary_after = canary(device)

    # Report from the best checkpoint, selected on SELECT_SEED, evaluated on EVAL_SEED.
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"model": final_state, "step": step}, out_dir / "final.pt")
    if (out_dir / "ckpt.pt").exists():
        model.load_state_dict(
            torch.load(out_dir / "ckpt.pt", map_location=device, weights_only=True)["model"])
    final_half_lives = evaluate.half_life_summary(model)

    grid = evaluate.full_grid(model, cells=report_cells(args.full_grid))
    primary = grid["16,128"]["arms"]["model"]

    results = {
        "config": config,
        "manifest": manifest,
        "stop_reason": stop_reason,
        "wall_seconds": round(wall, 2),
        "steps": step,
        "bytes_seen": stream.bytes_served,
        "bytes_per_second": round(stream.bytes_served / max(wall, 1e-9), 1),
        "episodes_generated": stream.episodes_made,
        "canary_before": canary_before,
        "canary_after": canary_after,
        "other_training_loads": contention,
        "futility_reference": futility_reference,
        "futility_check": results_futility,
        "best_checkpoint": best,
        "checkpoint_sha256": hashlib.sha256((out_dir / "ckpt.pt").read_bytes()).hexdigest()
                             if (out_dir / "ckpt.pt").exists() else None,
        "final_step_selection": evaluate.selection_score(
            _reloaded(model, final_state)) if step else None,
        "half_lives_initial": initial_half_lives,
        "half_lives_final": final_half_lives,
        "log_rows": rows,
        "grid": grid,
        "primary_endpoint": {
            "cell": "pairs=16, gap=128", "protocol": "reset-per-episode, learning frozen",
            "accuracy": primary["accuracy"], "hits": primary["hits"], "trials": primary["trials"],
            "wilson_low": primary["wilson_low"], "wilson_high": primary["wilson_high"],
            "nll": primary.get("nll"), "rank": primary.get("rank"),
            "gate_wilson_low_at_least": 0.90,
            "passes_gate": bool(primary["wilson_low"] >= 0.90),
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "halflives.json").write_text(json.dumps(
        {"initial": initial_half_lives, "final": final_half_lives}, indent=2) + "\n")

    print("\n== %s ==" % args.name)
    print("stop_reason=%s wall=%.1fs bytes=%d bytes/s=%.0f steps=%d"
          % (stop_reason, wall, stream.bytes_served, results["bytes_per_second"], step))
    print("primary (16,128): acc=%.3f wilson=[%.3f, %.3f] nll=%.3f rank=%.1f  gate=%s"
          % (primary["accuracy"], primary["wilson_low"], primary["wilson_high"],
             primary.get("nll", float("nan")), primary.get("rank", float("nan")),
             results["primary_endpoint"]["passes_gate"]))
    return results


def _reloaded(model: RTSModel, state_dict) -> RTSModel:
    """Score the final-step weights without disturbing the loaded best checkpoint."""
    import copy
    clone = copy.deepcopy(model)
    clone.load_state_dict(state_dict)
    return clone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="output folder under smRTS_01/out/")
    parser.add_argument("--stage", default="", help="S1 / S2 / S3 / S4, for the record")
    parser.add_argument("--cell", choices=("leaky", "gated", "fast", "delta", "gru"),
                        default="leaky")
    parser.add_argument("--regime", choices=("tbptt", "online-batched"), default="tbptt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--key-dim", dest="key_dim", type=int, default=8)
    parser.add_argument("--value-dim", dest="value_dim", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--eval-every", dest="eval_every", type=float, default=60.0)
    parser.add_argument("--freeze-decay", dest="freeze_decay", type=float, default=None,
                        help="freeze every decay at this half-life in bytes")
    parser.add_argument("--futility-fraction", dest="futility_fraction", type=float, default=0.25)
    parser.add_argument("--futility-mode", dest="futility_mode", choices=("kill", "report"),
                        default="kill",
                        help="S1 runs 'report': its deliverable IS the negative result, so killing "
                             "the falsification early would weaken it. Any deviation is recorded "
                             "in config.json.")
    parser.add_argument("--gate-accuracy", dest="gate_accuracy", type=float, default=0.0)
    parser.add_argument("--canary-reference", dest="canary_reference", type=float, default=0.0)
    parser.add_argument("--full-grid", dest="full_grid", action="store_true")
    parser.add_argument("--depth-truncated", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
