#!/usr/bin/env python3
"""Train the first English reply renderer and score generated, dev, and untouched test cases.

`--precision` (fp32/bf16/fp16, defaults to bf16 on mps/cuda) autocasts the forward pass and loss;
`--patience` (default 3, 0 disables) stops training when the dev/generated selection metric
hasn't improved for that many evals in a row. `--precision fp32 --patience 0` reproduces the
previous behaviour exactly.
"""
import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

import data
from model import Config, LanguageGPT, batch, encode
from score import summary


HERE = os.path.dirname(os.path.abspath(__file__))

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

WARMUP_STEPS = 200
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def size(example):
    return len(encode(example["prompt"], example["reply"]))


def load_cases(name):
    with open(os.path.join(HERE, name)) as f:
        rows = json.load(f)
    return [dict(row, prompt=data.compact(row["payload"])) for row in rows]


def token_batches(examples, tokens):
    ordered, groups, group = sorted(examples, key=size), [], []
    for example in ordered:
        if group and (len(group) + 1) * size(example) > tokens:
            groups.append(group)
            group = []
        group.append(example)
    return groups + ([group] if group else [])


@torch.no_grad()
def teacher_exact(model, examples, dev, tokens=12000):
    was = model.training
    model.eval()
    right = 0
    for group in token_batches(examples, tokens):
        x, y = batch(group)
        pred = model(x.to(dev))[0].argmax(-1).cpu()
        right += int(((pred == y) | (y == -100)).all(1).sum())
    model.train(was)
    return round(right / max(1, len(examples)), 3)


@torch.no_grad()
def greedy(model, examples):
    was = model.training
    model.eval()
    replies = [model.write(e["prompt"], max_new=300).strip() for e in examples]
    model.train(was)
    return summary(examples, replies)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=int, default=120000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--tokens", type=int, default=8192, help="tokens per padded batch")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--resume", help="continue from a latest.pt checkpoint, including optimizer state")
    ap.add_argument("--precision", choices=sorted(DTYPES), default=None,
                    help="defaults to bf16 on mps/cuda, else fp32")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many evals with no improvement in the selection metric; 0 disables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.precision is None:
        args.precision = "bf16" if args.device in ("mps", "cuda") else "fp32"
    dtype = DTYPES[args.precision]
    amp = (lambda: torch.autocast(device_type=args.device, dtype=dtype)) if dtype is not None else nullcontext
    if args.precision == "fp16":
        print("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.", flush=True)
    started = time.time()
    dev_cases, test_cases = load_cases("dev.json"), load_cases("test.json")
    held = {e["prompt"] for e in dev_cases + test_cases}
    pool = [e for e in data.generate(args.pool, args.seed) if e["prompt"] not in held]
    generated_check = [e for e in data.generate(1100, args.seed + 1000) if e["prompt"] not in held][:1000]
    longest = max(map(size, pool + generated_check + dev_cases + test_cases))
    if longest > args.max_len:
        raise ValueError("longest lesson is %d bytes, over --max-len %d" % (longest, args.max_len))
    print("made %d lessons in %.1fs; longest %d bytes" % (len(pool), time.time() - started, longest), flush=True)

    resumed = torch.load(args.resume, map_location=args.device) if args.resume else None
    c = Config(**resumed["config"]) if resumed else Config(
        max_len=args.max_len, d=args.d, layers=args.layers, heads=args.heads, dropout=args.dropout)
    model = LanguageGPT(c).to(args.device)
    if resumed:
        model.load_state_dict(resumed["model"])
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98))
    if resumed and resumed.get("optimizer"):
        opt.load_state_dict(resumed["optimizer"])
    min_lr = args.lr / 10
    path = os.path.join(args.out, "language.pt")
    log = open(os.path.join(args.out, "train.log"), "w")

    def say(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    start_step = int(resumed.get("step", 0)) if resumed else 0
    say("smLANGUAGE_RENDER_001: %.2fM parameters on %s, precision %s, steps %d to %d of about %d tokens" %
        (model.param_count() / 1e6, args.device, args.precision, start_step, args.steps, args.tokens))

    def lr_at(step):
        if step < WARMUP_STEPS:
            return args.lr * (step + 1) / WARMUP_STEPS
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * step / max(1, args.steps)))

    pick = random.Random(args.seed)
    ordered = sorted(pool, key=size)
    running, since, start = torch.zeros((), device=args.device), 0, time.time()
    best = (-1.0, -1.0, -1.0)
    evals_since_improved = 0
    for step in range(start_step, args.steps + 1):
        if (step and step % args.eval_every == 0) or step == args.steps:
            gen = teacher_exact(model, generated_check, args.device)
            hand = greedy(model, dev_cases)
            standing = (hand["exact"], hand["grounded"], gen)
            say("step %5d | %5.1f min | loss %.4f | generated exact %.3f | dev exact %.3f grounded %.3f" %
                (step, (time.time() - start) / 60, running.item() / max(1, since), gen, hand["exact"], hand["grounded"]))
            running.zero_()
            since = 0
            improved = standing > best
            if improved:
                best = standing
                evals_since_improved = 0
                torch.save({"model": model.state_dict(), "config": asdict(c), "step": step,
                            "dev": {k: v for k, v in hand.items() if k != "rows"}}, path)
            else:
                evals_since_improved += 1
            torch.save({"model": model.state_dict(), "config": asdict(c), "step": step,
                        "dev": {k: v for k, v in hand.items() if k != "rows"}, "optimizer": opt.state_dict()},
                       os.path.join(args.out, "latest.pt"))
            if args.patience and evals_since_improved >= args.patience:
                say("early stopping: dev/generated selection metric hasn't improved for %d evals, "
                    "stopping at step %d, best %s" % (args.patience, step, best))
                break
        if step == args.steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        i = pick.randrange(len(ordered))
        count = max(1, args.tokens // size(ordered[i]))
        examples = ordered[max(0, i - count + 1):i + 1]
        x, y = batch(examples)
        with amp():
            _, loss = model(x.to(args.device), y.to(args.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.detach()
        since += 1
    log.close()

    ck = torch.load(path, map_location="cpu")
    final = LanguageGPT(Config(**ck["config"]))
    final.load_state_dict(ck["model"])
    final.eval()
    test = greedy(final, test_cases)
    generated = teacher_exact(final, generated_check, "cpu")
    timing_start = time.perf_counter()
    timing_chars = 0
    for example in test_cases:
        timing_chars += len(final.write(example["prompt"], max_new=300))
    timing = time.perf_counter() - timing_start
    results = {"when": time.strftime("%Y-%m-%d %H:%M"), "params": final.param_count(), "step": ck["step"],
               "seconds": round(time.time() - start, 1), "args": recorded_args(args), "generated_teacher_exact": generated,
               "dev": ck.get("dev"), "test": test,
               "ms_per_output_byte_cpu": round(1000 * timing / max(1, timing_chars), 3)}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("best step %d | test exact %.3f grounded %.3f | %.3f ms/output byte on CPU" %
          (ck["step"], test["exact"], test["grounded"], results["ms_per_output_byte_cpu"]), flush=True)
    for row in test["rows"]:
        print("%s %s\n  got:  %s\n  want: %s" %
              ("OK" if row["exact"] else "MISS", row["kind"], row["reply"], row["expected"]))


if __name__ == "__main__":
    main()
