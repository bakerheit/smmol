"""Pretrain smLANGUAGE_EN_001 with resumable checkpoints.

`--precision` (fp32/bf16/fp16, defaults to bf16 on mps/cuda) autocasts the forward pass and loss;
`--patience` (default 3, 0 disables) stops training when validation loss hasn't improved for that
many evals in a row. `--precision fp32 --patience 0` reproduces the previous behaviour exactly.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import torch

from model import Config, LanguageModel


HERE = Path(__file__).resolve().parent

def recorded_args(args):
    """argparse values for results.json, with paths under this project made relative.

    --out defaults to a path under HERE, which is absolute on whoever's machine ran the
    training. Recording it verbatim writes a home directory into a file that gets published
    and tells a reader nothing they need.
    """
    here = str(HERE)
    clean = {}
    for key, value in vars(args).items():
        text = str(value)
        clean[key] = os.path.relpath(text, here) if text.startswith(here) else value
    return clean

EVAL_BATCHES = 20
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def load_bytes(path):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)


def sample_batch(source, batch_size, context, generator, device):
    if len(source) <= context + 1:
        raise ValueError(f"split has {len(source)} bytes but context is {context}")
    starts = torch.randint(len(source) - context - 1, (batch_size,), generator=generator)
    offsets = torch.arange(context)
    x = source[starts[:, None] + offsets].long()
    y = source[starts[:, None] + offsets + 1].long()
    return x.to(device), y.to(device)


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint(model, optimizer, config, step, tokens_seen, best_val, args):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "train_args": recorded_args(args),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--steps", type=int, default=5000, help="total target step, including resumed steps")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--d", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--precision", choices=sorted(DTYPES), default=None,
                        help="defaults to bf16 on mps/cuda, else fp32")
    parser.add_argument("--patience", type=int, default=3,
                        help="stop after this many evals with no val improvement; 0 disables")
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1:
        parser.error("steps and batch must be positive")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if args.precision is None:
        args.precision = "bf16" if device in ("mps", "cuda") else "fp32"
    dtype = DTYPES[args.precision]
    amp = (lambda: torch.autocast(device_type=device, dtype=dtype)) if dtype is not None else nullcontext
    if args.precision == "fp16":
        print("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    latest_path = args.out / "latest.pt"
    best_path = args.out / "best.pt"

    config = Config(
        ctx=args.ctx,
        d=args.d,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    )
    resume = None
    if args.resume:
        if not latest_path.exists():
            parser.error(f"cannot resume: {latest_path} does not exist")
        resume = torch.load(latest_path, map_location="cpu", weights_only=False)
        saved_config = Config(**resume["config"])
        if saved_config != config:
            parser.error(f"checkpoint config {saved_config} does not match requested config {config}")

    train_data = load_bytes(args.data_dir / "train.txt")
    val_data = load_bytes(args.data_dir / "val.txt")
    model = LanguageModel(config)
    if resume:
        model.load_state_dict(resume["model"])
    model.to(device)
    decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    if resume:
        optimizer.load_state_dict(resume["optimizer"])

    step = resume["step"] if resume else 0
    tokens_seen = resume["tokens_seen"] if resume else 0
    best_val = resume["best_val"] if resume else float("inf")
    if step >= args.steps:
        parser.error(f"checkpoint is already at step {step}, target is {args.steps}")
    print(
        f"smLANGUAGE_EN_001 | {model.parameter_count() / 1e6:.2f}M params | {device} | "
        f"precision {args.precision} | {len(train_data) / 1e6:.1f} MB train | steps {step}->{args.steps}",
        flush=True,
    )

    train_generator = torch.Generator().manual_seed(args.seed + step)

    @torch.no_grad()
    def evaluate():
        model.eval()
        generator = torch.Generator().manual_seed(args.seed + 10_000)
        losses = []
        for _ in range(EVAL_BATCHES):
            x, y = sample_batch(val_data, args.batch, config.ctx, generator, device)
            losses.append(model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

    def learning_rate(current_step):
        if current_step < args.warmup:
            return args.lr * (current_step + 1) / max(args.warmup, 1)
        progress = (current_step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    log_path = args.out / "train.jsonl"
    started = time.time()
    model.train()
    evals_since_improved = 0
    while step < args.steps:
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = sample_batch(train_data, args.batch, config.ctx, train_generator, device)
        with amp():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        tokens_seen += args.batch * config.ctx

        should_evaluate = step == 1 or step % args.eval_every == 0 or step == args.steps
        if not should_evaluate:
            continue
        val_loss = evaluate()
        elapsed = time.time() - started
        event = {
            "step": step,
            "tokens_seen": tokens_seen,
            "elapsed_seconds_this_run": round(elapsed, 3),
            "train_loss": round(loss.item(), 6),
            "val_loss": round(val_loss, 6),
            "val_bits_per_byte": round(val_loss / math.log(2), 6),
            "learning_rate": lr,
        }
        with log_path.open("a") as log:
            log.write(json.dumps(event) + "\n")
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            evals_since_improved = 0
        else:
            evals_since_improved += 1
        payload = checkpoint(model, optimizer, config, step, tokens_seen, best_val, args)
        atomic_save(payload, latest_path)
        if improved:
            atomic_save(payload, best_path)
        speed = (args.batch * config.ctx * max(step - (resume["step"] if resume else 0), 1)) / max(elapsed, 1e-9)
        print(
            f"step {step:5d} | {tokens_seen / 1e6:7.1f}M bytes | {speed:7.0f} byte/s | "
            f"train {loss.item():.3f} | val {val_loss:.3f} ({val_loss / math.log(2):.2f} bpb)"
            f"{' | best' if improved else ''}",
            flush=True,
        )
        if args.patience and evals_since_improved >= args.patience:
            print(
                f"early stopping: val loss has not improved for {args.patience} evals, "
                f"stopping at step {step}, best val {best_val:.4f}",
                flush=True,
            )
            break

    print(f"done | step {step} | best validation {best_val:.4f} | {best_path}", flush=True)


if __name__ == "__main__":
    main()
