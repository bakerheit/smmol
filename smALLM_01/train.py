"""Train smALLM_01 by doing: new secret wiring every episode, flip things, guess, learn from the misses.

There's no dataset. Every batch is brand-new worlds made on the spot, so the only thing
worth learning is how to figure out a wiring from experiments.
"""
import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from evaluate import load, pick_device, table, understanding
from model import Config, TinyGPT
from world import ACT, GADGETS, VOCAB, episodes

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP_STEPS = 100
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=24, help="experiments per training episode")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--precision", choices=sorted(DTYPES), default=None,
                     help="defaults to bf16 on mps/cuda, fp32 elsewhere")
    ap.add_argument("--patience", type=int, default=3,
                     help="stop if the held-out understanding score hasn't improved in N evals; 0 disables")
    args = ap.parse_args()

    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.precision is None:
        args.precision = "bf16" if dev in ("mps", "cuda") else "fp32"
    dtype = DTYPES[args.precision]
    if args.precision == "fp16":
        print("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.", flush=True)
    amp = (lambda: torch.autocast(device_type=dev, dtype=dtype)) if dtype is not None else nullcontext

    c = Config(vocab=VOCAB, ctx=1 + GADGETS + (GADGETS + 1) * args.steps,
               d=args.d, layers=args.layers, heads=args.heads)
    model = TinyGPT(c).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.99),
    )
    min_lr = args.lr / 10
    print(f"smALLM_01: {model.param_count()/1e6:.1f}M params on {dev}, precision {args.precision}, {GADGETS} gadgets, "
          f"{args.steps} experiments per episode, {args.minutes:g} min budget", flush=True)

    def lr_at(step, frac):
        if step < WARMUP_STEPS:
            return args.lr * (step + 1) / WARMUP_STEPS
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(frac, 1.0)))

    log = open(os.path.join(args.out, "log.csv"), "w")
    log.write("step,episodes,seconds,loss,exact_after_8,lazy_after_8\n")
    best, no_improve = -1.0, 0
    budget = args.minutes * 60
    start_t = time.time()
    step = 0
    loss_sum, loss_n = torch.zeros((), device=dev), 0
    while True:
        elapsed = time.time() - start_t
        done = elapsed >= budget
        if step % args.eval_every == 0 or done:
            # understanding() runs with no autocast, so this is a fair fp32 read of the same weights,
            # the same score already used to pick the best checkpoint.
            score = understanding(model, dev, "random", worlds=64, ks=(8,))[8]
            model.train()
            loss = (loss_sum / max(loss_n, 1)).item() if loss_n else float("nan")
            loss_sum, loss_n = torch.zeros((), device=dev), 0
            print(f"step {step:5d} | {step * args.batch / 1e3:6.1f}k episodes | {elapsed/60:5.1f} min | "
                  f"loss {loss:.3f} | after 8 experiments: all 8 right {score['exact']:.0%}, "
                  f"effects caught {score['effects']:.0%} (lazy guess {score['lazy']:.0%})", flush=True)
            log.write(f"{step},{step * args.batch},{elapsed:.1f},{loss:.4f},{score['exact']:.4f},{score['lazy']:.4f}\n")
            log.flush()
            if score["exact"] > best:
                best, no_improve = score["exact"], 0
                torch.save({"model": model.state_dict(), "config": asdict(c), "step": step,
                            "exact_after_8": best}, ckpt_path)
            else:
                no_improve += 1
            if args.patience and no_improve >= args.patience and not done:
                print(f"early stopping: held-out exact-after-8 hasn't improved in {args.patience} evals, "
                      f"stopping at step {step} (best {best:.0%})", flush=True)
                break
        if done:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, elapsed / budget)
        seq = episodes(args.batch, args.steps, gen).to(dev)
        x, y = seq[:, :-1], seq[:, 1:].clone()
        y[y >= ACT] = -100          # its own flips aren't something to predict
        y[:, :GADGETS] = -100       # neither is the random starting position
        with amp():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.detach()
        loss_n += 1
        step += 1

    log.close()
    model, ckpt = load(ckpt_path, dev)
    result = table(model, dev)
    with open(os.path.join(args.out, "eval.md"), "w") as f:
        f.write(f"Best checkpoint: step {ckpt['step']} of {step}, {args.minutes:g} min on {dev}\n\n{result}\n")
    print(f"done: {step} steps, best checkpoint at step {ckpt['step']}\n{result}", flush=True)


if __name__ == "__main__":
    main()
