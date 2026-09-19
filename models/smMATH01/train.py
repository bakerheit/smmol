"""Train smMATH01 on the Mac's GPU for a fixed number of steps per variant, with fresh problems every step.

Every variant gets the same number of steps, not the same minutes, so a Mac that warms up and slows
down during a long run can't shortchange whichever variant goes last.

    python3 train.py                                # plain, then reversed, then abacus
    python3 train.py --variant abacus --steps 2000
"""
import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from evaluate import accuracy
from model import VARIANTS, Config, MathGPT
from problems import OPS, TRAIN_DIGITS, batch

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP_STEPS = 200
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def train(variant, args, dev):
    source = random.Random(args.seed)
    torch.manual_seed(args.seed)
    c = Config(variant=variant, d=args.d, layers=args.layers, heads=args.heads)
    offset = args.offset_max if variant == "abacus" else 0
    if offset + 2 * args.max_digits >= c.places:
        raise SystemExit("--offset-max is too big for %d place labels" % c.places)
    model = MathGPT(c).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98),
    )
    min_lr = args.lr / 10
    path = os.path.join(args.out, variant + ".pt")
    dtype = DTYPES[args.precision]
    amp = (lambda: torch.autocast(device_type=dev, dtype=dtype)) if dtype is not None else nullcontext
    print(f"\nsmMATH01 {variant}: {model.param_count()/1e6:.2f}M params on {dev}, precision {args.precision}, "
          f"numbers of 1 to {args.max_digits} digits, {args.steps} steps of {args.batch}", flush=True)

    def lr_at(step):
        if step < WARMUP_STEPS:
            return args.lr * (step + 1) / WARMUP_STEPS
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * step / args.steps))

    def save(step, elapsed, loss):
        torch.save({"model": model.state_dict(), "config": asdict(c), "step": step, "seconds": round(elapsed),
                    "problems": step * args.batch, "params": model.param_count(), "loss": loss,
                    "max_digits": args.max_digits, "offset_max": offset}, path)

    start, since = time.time(), 0
    running = torch.zeros((), device=dev)
    # Keep the best checkpoint, not the last. The probe below generates fresh problems from a fixed
    # seed, so it is held out from training and comparable across evals. The score we select on is
    # the mean over operations at the digit length the model actually trains on.
    best, best_step = -1.0, 0
    for step in range(args.steps + 1):
        if (step and step % args.eval_every == 0) or step == args.steps:
            elapsed = time.time() - start
            loss = running.item() / max(since, 1)
            running.zero_()
            since = 0
            quick, trained = [], []
            for digits in (args.max_digits, args.max_digits + 2):  # the longest it trains on, and 2 digits past it
                scores = [accuracy(model, op, digits, 100, random.Random(10 * digits + i), dev)[0]
                          for i, op in enumerate(OPS)]
                if digits == args.max_digits:
                    trained = scores
                quick.append("%d digits %s" % (digits, " ".join("%s%3d%%" % (op, round(100 * sc))
                                                                for op, sc in zip(OPS, scores))))
            held_out = sum(trained) / len(trained)
            improved = held_out > best
            print(f"{variant} step {step:6d} | {elapsed/60:4.1f} min | {step*args.batch/1e6:5.2f}M problems | "
                  f"loss {loss:.4f} | " + " | ".join(quick) + (" | best" if improved else ""), flush=True)
            if improved:
                best, best_step = held_out, step
                save(step, elapsed, loss)
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y, where = batch(args.batch, source, c.reverse, args.max_digits, OPS, offset)
        with amp():
            _, loss_t = model(x.to(dev), where.to(dev) if variant == "abacus" else None, y.to(dev))
        opt.zero_grad(set_to_none=True)
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss_t.detach()
        since += 1
    print(f"{variant} done: {args.steps} steps, {args.steps*args.batch/1e6:.2f}M problems in "
          f"{(time.time() - start)/60:.1f} min, saved step {best_step} "
          f"({100*best:.0f}% at {args.max_digits} digits) to {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="all", choices=VARIANTS + ("all",))
    ap.add_argument("--steps", type=int, default=8000, help="per variant")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-digits", type=int, default=TRAIN_DIGITS)
    ap.add_argument("--offset-max", type=int, default=25, help="biggest random place shift (abacus only)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--precision", choices=sorted(DTYPES), default=None,
                     help="default: bf16 on mps/cuda, fp32 elsewhere")
    args = ap.parse_args()
    if args.precision is None:
        args.precision = "bf16" if args.device in ("mps", "cuda") else "fp32"
    if args.precision == "fp16":
        print("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.", flush=True)
    os.makedirs(args.out, exist_ok=True)
    for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
        train(variant, args, args.device)
    print("\nnext: python3 evaluate.py")


if __name__ == "__main__":
    main()
