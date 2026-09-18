"""Train smLLM_01 on Tiny Shakespeare for a fixed number of minutes on the Mac's GPU.

`--precision bf16` (the default on the GPU) does the forward pass in 16-bit: measured 1.5x faster
with indistinguishable quality. `--patience` stops once validation has stopped improving, instead
of running out the clock past the best checkpoint. `--precision fp32 --patience 0` reproduces the
behaviour of runs recorded before 2026-09-17. See ../smEFFICIENCY_01/README.md for the numbers.
"""
import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from model import Config, TinyGPT

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP_STEPS = 100
EVAL_BATCHES = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "input.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
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
    ap.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None,
                    help="bf16 on the GPU by default; fp32 restores the pre-2026-09-17 behaviour")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many evals with no val improvement; 0 never stops early")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.precision is None:  # 16-bit pays off on the GPU; on the CPU it does not
        args.precision = "bf16" if dev in ("mps", "cuda") else "fp32"
    dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    if args.precision == "fp16":
        print("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.", flush=True)
    # Only the forward pass runs in 16-bit. Validation below always runs in fp32, so its number
    # stays comparable with every val loss already recorded for this project.
    amp = (lambda: torch.autocast(device_type=dev, dtype=dtype)) if dtype is not None else nullcontext
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")

    with open(args.data, "rb") as f:
        data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
    split = int(len(data) * 0.9)  # last 10% is never trained on; it tells us if we're memorizing
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
    print(f"smLLM_01: {model.param_count()/1e6:.1f}M params on {dev}, precision {args.precision}, "
          f"{len(parts['train'])/1e6:.2f}M train bytes, {args.minutes:g} min budget"
          + (f", patience {args.patience}" if args.patience else ", no early stop"), flush=True)

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

    def sample(prompt="\n", n=300):
        model.eval()
        idx = torch.tensor([list(prompt.encode())], dtype=torch.long, device=dev)
        text = bytes(model.generate(idx, n)[0].tolist()).decode("utf-8", errors="replace")
        model.train()
        return text

    def lr_at(step, frac):
        if step < WARMUP_STEPS:
            return args.lr * (step + 1) / WARMUP_STEPS
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(frac, 1.0)))

    log = open(os.path.join(args.out, "log.csv"), "w")
    log.write("step,tokens,seconds,train_loss,val_loss\n")
    best = float("inf")
    stale = 0          # evals since val last improved
    stopped_early = False
    budget = args.minutes * 60
    start = time.time()
    step = 0
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
                stale = 0
                torch.save({"model": model.state_dict(), "config": asdict(c), "step": step,
                            "val_loss": best}, ckpt_path)
            else:
                stale += 1
                if args.patience and stale >= args.patience and not done:
                    print(f"early stop: val hasn't improved in {stale} evals; best {best:.3f} "
                          f"({best/math.log(2):.2f} bits/char). Stopping at step {step}, "
                          f"{elapsed/60:.1f} of {args.minutes:g} min.", flush=True)
                    stopped_early = done = True
            if step and step % (args.eval_every * 4) == 0:
                print("----\n" + sample() + "\n----", flush=True)
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
    print(f"done{' (early stop)' if stopped_early else ''}: {step} steps, "
          f"best val {best:.3f} ({best/math.log(2):.2f} bits/char), saved {ckpt_path}")
    print("----\n" + sample("ROMEO:", 600) + "\n----", flush=True)


if __name__ == "__main__":
    main()
