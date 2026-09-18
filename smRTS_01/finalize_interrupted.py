"""Give an interrupted run an honest grid, without pretending the trainer finished it.

A run killed mid-flight leaves `ckpt.pt`, `config.json`, `manifest.json` and `log.csv` but no
`results.json`, `halflives.json` or `final.pt`, because those are written in the trainer's
finalisation block. This script builds those artifacts from what actually exists.

It constructs no optimizer and takes no gradient. It loads the best-by-selection checkpoint, runs
the same frozen evaluation the trainer would have run, and writes `results.json` with
`"finalized_by": "finalize_interrupted.py"` and an explicit `interrupted` block, so nothing here
can later be mistaken for a completed run. `final.pt` is NOT fabricated: the final-step weights
were never saved and are gone.

    python3 smRTS_01/finalize_interrupted.py --name S2-fast-tbptt-calibration \\
        --stop-reason user_stop --device mps
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

import evaluate
import train_recall
import world
from cells import RTSModel

HERE = Path(__file__).resolve().parent


def load_rows(path: Path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [{k: (float(v) if k != "step" else int(v)) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--stop-reason", dest="stop_reason", default="user_stop")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    out_dir = HERE / "out" / args.name
    config = json.loads((out_dir / "config.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    rows = load_rows(out_dir / "log.csv")
    if not rows:
        raise SystemExit("no log.csv rows: nothing to finalize")
    last = rows[-1]

    device = torch.device(args.device)
    model = RTSModel(config["cell"], dim=config["dim"], layers=config["layers"],
                     heads=config["heads"], key_dim=config["key_dim"],
                     value_dim=config["value_dim"]).to(device)

    # The initial half-lives are reconstructed from a fresh model built with the same constructor
    # and seed, not remembered: the trainer never wrote them for this run.
    class _Args:
        pass

    fresh_args = _Args()
    for key in ("seed", "dim", "layers", "heads", "key_dim", "value_dim"):
        setattr(fresh_args, key, config[key])
    fresh_args.freeze_decay = config.get("freeze_decay_half_life")
    initial_half_lives = evaluate.half_life_summary(train_recall.build_model(config["cell"],
                                                                            fresh_args))

    ckpt_path = out_dir / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    before = world.parameter_fingerprint(model)
    grid = evaluate.full_grid(model, cells=train_recall.report_cells())
    after = world.parameter_fingerprint(model)
    if before != after:
        raise RuntimeError("the frozen evaluation mutated parameters")

    final_half_lives = evaluate.half_life_summary(model)
    primary = grid["16,128"]["arms"]["model"]

    results = {
        "finalized_by": "finalize_interrupted.py",
        "interrupted": {
            "stop_reason": args.stop_reason,
            "what_is_missing": [
                "final.pt - the final-step weights were never written and cannot be recovered",
                "final_step_selection - the final-step model no longer exists to score",
                "canary_after - the run was killed before the closing contention canary",
                "wall_seconds/bytes_seen for the whole budget - only the last logged row is known",
            ],
            "note": args.note,
            "evaluated_checkpoint": "best-by-selection, as written by the trainer before the stop",
            "no_optimizer_constructed": True,
            "no_gradient_taken": True,
        },
        "config": config,
        "manifest": manifest,
        "stop_reason": args.stop_reason,
        "last_logged_row": last,
        "wall_seconds_at_last_log": last["wall_s"],
        "bytes_seen_at_last_log": int(last["bytes_seen"]),
        "bytes_per_second_at_last_log": last["bytes_per_s"],
        "steps_at_last_log": last["step"],
        "canary_before": None,
        "canary_after": None,
        "best_checkpoint": ckpt["best"],
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "half_lives_initial": initial_half_lives,
        "half_lives_final": final_half_lives,
        "half_lives_final_is_from": "the best-by-selection checkpoint, not the final step",
        "futility_reference": evaluate.futility_reference(16, 128),
        "log_rows": rows,
        "grid": grid,
        "evaluation_device": str(device),
        "frozen_parameter_fingerprint": before,
        "primary_endpoint": {
            "cell": "pairs=16, gap=128",
            "protocol": "reset-per-episode, learning frozen",
            "accuracy": primary["accuracy"], "hits": primary["hits"], "trials": primary["trials"],
            "wilson_low": primary["wilson_low"], "wilson_high": primary["wilson_high"],
            "nll": primary.get("nll"), "rank": primary.get("rank"),
            "gate_wilson_low_at_least": 0.90,
            "passes_gate": bool(primary["wilson_low"] >= 0.90),
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "halflives.json").write_text(json.dumps(
        {"initial": initial_half_lives, "final_from_best_checkpoint": final_half_lives,
         "finalized_by": "finalize_interrupted.py"}, indent=2) + "\n")

    print("finalized %s (stop_reason=%s)" % (args.name, args.stop_reason))
    print("primary (16,128): acc=%.3f wilson=[%.3f, %.3f] nll=%.3f rank=%.1f gate=%s"
          % (primary["accuracy"], primary["wilson_low"], primary["wilson_high"],
             primary.get("nll", float("nan")), primary.get("rank", float("nan")),
             results["primary_endpoint"]["passes_gate"]))


if __name__ == "__main__":
    main()
