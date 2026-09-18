#!/usr/bin/env python3
"""Train smMATH001-a on generated worked solutions, then find where it breaks.

    python3 train.py
    python3 train.py --steps 2000
"""
import argparse
import json
import math
import os
import random
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict
from fractions import Fraction

import torch

import work as W
from model import Config, MathGPT, batch

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

WARMUP_STEPS = 300
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def size(e):
    return len(e["question"]) + len(e["work"]) + 2


def token_batches(examples, tokens):
    """Groups of similar length, each about `tokens` tokens including padding."""
    ordered, groups, group = sorted(examples, key=size), [], []
    for e in ordered:
        if group and (len(group) + 1) * size(e) > tokens:
            groups.append(group)
            group = []
        group.append(e)
    return groups + ([group] if group else [])


def exact(model, examples, dev, tokens=16000):
    """Share whose whole worked solution it writes exactly (every token its top guess, so greedy writing agrees)."""
    if not examples:
        return None
    was = model.training
    model.eval()
    right = 0
    with torch.no_grad():
        for group in token_batches(examples, tokens):
            x, y, where = batch(group)
            pred = model(x.to(dev), where.to(dev))[0].argmax(-1).cpu()
            right += int(((pred == y) | (y == -100)).all(1).sum())
    model.train(was)
    return round(right / len(examples), 3)


def custom(r, n, make):
    out = []
    while len(out) < n:
        tree = make(r)
        try:
            work, answer = W.solve_tree(tree)
        except ValueError:
            continue
        out.append({"expression": W.render(tree), "question": W.question(tree), "work": work, "answer": answer})
    return out


def digits(r, lo, hi):
    d = r.randint(lo, hi)
    return str(r.randint(0 if d == 1 else 10 ** (d - 1), 10 ** d - 1))


BEYOND = {
    "add, 8-10 digit numbers": lambda r: ("+", digits(r, 8, 10), digits(r, 8, 10)),
    "subtract, 8-10 digit numbers": lambda r: ("-", digits(r, 9, 10), digits(r, 8, 8)),
    "multiply, 4-digit multiplier": lambda r: ("*", digits(r, 5, 6), digits(r, 4, 4)),
    "multiply, 5-6 digit multiplier": lambda r: ("*", digits(r, 6, 6), digits(r, 5, 6)),
    "divide, 8-9 digit dividend": lambda r: ("/", digits(r, 8, 9), digits(r, 1, 2)),
    "divide, 5-digit divisor": lambda r: ("/", digits(r, 6, 7), digits(r, 5, 5)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=300000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--tokens", type=int, default=8192, help="tokens per batch, padding included")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--offset-max", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1000)
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
    torch.manual_seed(args.seed)
    dev = args.device

    def say(line):
        print(line, flush=True)

    started = time.time()
    pool = W.generate(args.pool, args.seed)
    pool.sort(key=size)
    val = W.generate(6000, args.seed + 1000)
    held = {e["question"] for e in val}
    before = len(pool)
    pool = [e for e in pool if e["question"] not in held]
    categories = {}
    for e in val:
        for name in (e["kind"], "decimals" if e["decimals"] else None, "harness-like" if e["harness"] else None):
            if name and len(categories.setdefault(name, [])) < 300:
                categories[name].append(e)
    r = random.Random(args.seed + 7)
    beyond = {name: custom(r, 150, make) for name, make in BEYOND.items()}
    say("made %d worked solutions in %.0fs (dropped %d that matched a check question); average %.0f tokens, longest %d"
        % (len(pool), time.time() - started, before - len(pool), statistics.mean(map(size, pool)), max(map(size, pool))))

    c = Config(d=args.d, layers=args.layers, heads=args.heads)
    model = MathGPT(c).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98))
    min_lr = args.lr / 10
    path = os.path.join(args.out, "math.pt")
    dtype = DTYPES[args.precision]
    amp = (lambda: torch.autocast(device_type=dev, dtype=dtype)) if dtype is not None else nullcontext
    say("smMATH001-a: %.2fM parameters on %s, precision %s, %d steps of about %d tokens each"
        % (model.param_count() / 1e6, dev, args.precision, args.steps, args.tokens))

    def lr_at(step):
        if step < WARMUP_STEPS:
            return args.lr * (step + 1) / WARMUP_STEPS
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * step / args.steps))

    pick = random.Random(args.seed)
    running, since, start = torch.zeros((), device=dev), 0, time.time()
    # Keep the best checkpoint, not the last. `categories` comes from `val`, whose questions were
    # removed from the training pool above, so this is a genuine held-out score. We select on the
    # mean across categories, which is the one scalar that covers all of them evenly.
    best, best_step = -1.0, 0
    for step in range(args.steps + 1):
        if (step and step % args.eval_every == 0) or step == args.steps:
            quick = {name: exact(model, examples[:100], dev) for name, examples in categories.items()}
            held_out = sum(quick.values()) / len(quick)
            improved = held_out > best
            say("step %6d | %5.1f min | loss %.4f | %s%s" % (step, (time.time() - start) / 60, running.item() / max(since, 1),
                                                            " ".join("%s %.2f" % kv for kv in sorted(quick.items())),
                                                            " | best" if improved else ""))
            running.zero_()
            since = 0
            if improved:
                best, best_step = held_out, step
                torch.save({"model": model.state_dict(), "config": asdict(c), "step": step}, path)
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        i = pick.randrange(len(pool))
        count = max(1, args.tokens // size(pool[i]))
        group = pool[max(0, i - count + 1): i + 1]  # the pool is sorted, so neighbours are about as long
        x, y, where = batch(group, pick, args.offset_max)
        with amp():
            _, loss = model(x.to(dev), where.to(dev), y.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.detach()
        since += 1

    # Everything below scores the weights we kept, not the last step's, so results.json describes
    # what is actually in math.pt. The first eval always saves, so the file is there to load.
    if best_step != args.steps:
        say("keeping step %d (held-out mean %.2f), not the last step %d" % (best_step, best, args.steps))
    model.load_state_dict(torch.load(path, map_location=dev)["model"])
    results = {"when": time.strftime("%Y-%m-%d %H:%M"), "params": model.param_count(), "steps": args.steps,
               "best_step": best_step, "best_held_out": round(best, 4),
               "seconds": round(time.time() - start), "args": recorded_args(args)}
    results["trained_sizes"] = {name: exact(model, examples, dev) for name, examples in categories.items()}
    results["beyond"] = {name: exact(model, examples, dev, tokens=12000) for name, examples in beyond.items()}

    model = model.to("cpu").eval()
    spot, timings = [], []
    for name, examples in list(categories.items()) + list(beyond.items()):
        for e in examples[:15]:
            t0 = time.perf_counter()
            written = model.work(e["question"])
            timings.append((len(written), time.perf_counter() - t0))
            spot.append((name, W.final_answer(written) == e["answer"], written == e["work"]))
    results["greedy_spot_check"] = {name: {"answer_right": sum(a for n, a, _ in spot if n == name), "work_exact": sum(w for n, _, w in spot if n == name), "of": sum(1 for n, *_ in spot if n == name)}
                                    for name in dict.fromkeys(n for n, *_ in spot)}
    results["ms_per_token_cpu"] = round(1000 * sum(t for _, t in timings) / max(1, sum(n for n, _ in timings)), 2)
    volume = "322234*21323*212231"
    t0 = time.perf_counter()
    written = model.work(W.question(W.parse(volume)))
    got = W.final_answer(written)
    results["volume_example"] = {"expression": volume, "wrote": None if got is None else W.unscaled(got),
                                 "right": str(Fraction(322234 * 21323 * 212231)), "seconds": round(time.perf_counter() - t0, 1),
                                 "characters": len(written)}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)

    say("\nworked solutions written exactly right (every step), training sizes:")
    say("| kind | right |")
    say("|---|---|")
    for name, value in sorted(results["trained_sizes"].items()):
        say("| %s (%d) | %.0f%% |" % (name, len(categories[name]), 100 * value))
    say("\nbeyond the training sizes:")
    say("| kind | right |")
    say("|---|---|")
    for name, value in results["beyond"].items():
        say("| %s (%d) | %.0f%% |" % (name, len(beyond[name]), 100 * value))
    say("\ngreedy writing, spot check (answer right / whole work exact / of):")
    for name, s in results["greedy_spot_check"].items():
        say("  %-32s %2d / %2d / %2d" % (name, s["answer_right"], s["work_exact"], s["of"]))
    v = results["volume_example"]
    say("\n%s: wrote %s, right is %s (%.1fs, %d characters of work); %.2f ms per token on the CPU"
        % (v["expression"], v["wrote"], v["right"], v["seconds"], v["characters"], results["ms_per_token_cpu"]))


if __name__ == "__main__":
    main()
