#!/usr/bin/env python3
"""smLLM_01's trainer, unchanged except for a precision switch and its own output folder.

Everything that affects the result is copied verbatim from `smLLM_01/train.py`: the model, the
data split, the optimizer and its two parameter groups, the warmup-then-cosine schedule, gradient
clipping, the evaluation batches, and the seed. The only additions are `--precision` and a
separate `--out`, so a run here can never overwrite `smLLM_01/out/ckpt.pt`.

**Validation is always measured in fp32**, with no autocast, whatever the training precision. That
keeps the val loss directly comparable to smLLM_01's published 1.471 (2.12 bits/char). Autocast
keeps fp32 master weights and casts only the operations, so this is a fair reading of the same
weights, not a different model.

    python3 train_precision.py --precision bf16 --minutes 20
"""
import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SMLLM = os.path.join(os.path.dirname(HERE), "models", "smLLM_01")
sys.path.insert(0, SMLLM)

from model import Config, TinyGPT  # noqa: E402

WARMUP_STEPS = 100
EVAL_BATCHES = 20
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", choices=sorted(DTYPES), default="fp32")
    ap.add_argument("--data", default=os.path.join(SMLLM, "data", "input.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "run"))
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    dtype = DTYPES[args.precision]
    if args.precision == "fp16":
        print("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.", flush=True)

    with open(args.data, "rb") as f:
        data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
    split = int(len(data) * 0.9)
    parts = {"train": data[:split], "val": data[split:]}

    c = Config(ctx=args.ctx, d=args.d, layers=args.layers, heads=args.heads, dropout=args.dropout)
    model = TinyGPT(c).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.99),
    )
    min_lr = args.lr / 10
    amp = (lambda: torch.autocast(device_type=dev, dtype=dtype)) if dtype is not None else nullcontext
    print(f"smEFFICIENCY_01: {model.param_count()/1e6:.1f}M params on {dev}, precision {args.precision}, "
          f"{args.minutes:g} min budget, val always measured in fp32", flush=True)

    def batch(part):
        src = parts[part]
        ix = torch.randint(len(src) - c.ctx - 1, (args.batch,))
        x = torch.stack([src[i:i + c.ctx] for i in ix]).long()
        y = torch.stack([src[i + 1:i + 1 + c.ctx] for i in ix]).long()
        return x.to(dev), y.to(dev)

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {p: sum(model(*batch(p))[1].item() for _ in range(EVAL_BATCHES)) / EVAL_BATCHES
               for p in ("train", "val")}
        model.train()
        return out

    def lr_at(step, frac):
        if step < WARMUP_STEPS:
            return args.lr * (step + 1) / WARMUP_STEPS
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(frac, 1.0)))

    log = open(os.path.join(args.out, "log.csv"), "w")
    log.write("step,tokens,seconds,train_loss,val_loss\n")
    best, budget, start, step = float("inf"), args.minutes * 60, time.time(), 0
    while True:
        elapsed = time.time() - start
        done = elapsed >= budget
        if step % args.eval_every == 0 or done:
            losses = evaluate()
            tokens = step * args.batch * c.ctx
            print(f"step {step:5d} | {tokens/1e6:5.1f}M tokens | {elapsed/60:5.1f} min | "
                  f"{tokens/max(elapsed, 1e-9):6.0f} tok/s | train {losses['train']:.3f} | "
                  f"val {losses['val']:.3f} ({losses['val']/math.log(2):.2f} bits/char)", flush=True)
            log.write(f"{step},{tokens},{elapsed:.1f},{losses['train']:.4f},{losses['val']:.4f}\n")
            log.flush()
            if losses["val"] < best:
                best = losses["val"]
                torch.save({"model": model.state_dict(), "config": asdict(c), "step": step,
                            "val_loss": best, "precision": args.precision}, ckpt_path)
        if done:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, elapsed / budget)
        x, y = batch("train")
        with amp():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

    log.close()
    print(f"done: {step} steps, best val {best:.3f} ({best/math.log(2):.2f} bits/char), saved {ckpt_path}")


if __name__ == "__main__":
    main()
