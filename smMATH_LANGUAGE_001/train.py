#!/usr/bin/env python3
"""Train smMATH_LANGUAGE_001 on generated messages, then score it on hand-written ones.

    python3 train.py
    python3 train.py --steps 1500
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from data import generate, parse_target, without
from model import Config, ReaderGPT, batch
from score import score

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


def exact(model, examples, dev):
    """Share of messages whose whole target it writes exactly right (every token its top guess, so greedy agrees)."""
    was = model.training
    model.eval()
    right = 0
    with torch.no_grad():
        for s in range(0, len(examples), 128):
            x, y = batch(examples[s:s + 128])
            pred = model(x.to(dev))[0].argmax(-1).cpu()
            right += int(((pred == y) | (y == -100)).all(1).sum())
    model.train(was)
    return right / len(examples)


def read_all(model, texts):
    was = model.training
    model.eval()
    out = [[p["expression"] for p in parse_target(model.read(t))] for t in texts]
    model.train(was)
    return out


def regex_stopgap(texts):
    """The harness's own code with no model: the longest arithmetic it can find in a message, worked out."""
    sys.path.insert(0, os.path.join(HERE, "..", "paratroop_harness_02"))
    try:
        import harness
    except ImportError:
        return None
    out = []
    for t in texts:
        found = harness.find_expression(t)
        try:
            out.append([harness.calc(found)] if found else [])
        except harness.HarnessError:
            out.append([])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=int, default=300000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
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

    with open(os.path.join(HERE, "test.json")) as f:
        test = json.load(f)
    texts = [e["text"] for e in test]
    started = time.time()
    generated = generate(args.examples, args.seed)
    train = without(generated, test)
    val = without(generate(2000, args.seed + 1000), test)
    say("generated %d messages in %.0fs; dropped %d that matched a hand-written test message word for word"
        % (len(generated), time.time() - started, len(generated) - len(train)))

    c = Config(d=args.d, layers=args.layers, heads=args.heads)
    model = ReaderGPT(c).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98))
    min_lr = args.lr / 10
    path = os.path.join(args.out, "reader.pt")
    dtype = DTYPES[args.precision]
    amp = (lambda: torch.autocast(device_type=dev, dtype=dtype)) if dtype is not None else nullcontext
    say("smMATH_LANGUAGE_001: %.2fM parameters on %s, precision %s, %d steps of %d messages, %d hand-written test messages"
        % (model.param_count() / 1e6, dev, args.precision, args.steps, args.batch, len(test)))

    def lr_at(step):
        if step < WARMUP_STEPS:
            return args.lr * (step + 1) / WARMUP_STEPS
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * step / args.steps))

    pick = random.Random(args.seed)
    running, since, start = torch.zeros((), device=dev), 0, time.time()
    # Keep the best checkpoint, not the last. We select on the generated held-out exact score, never
    # on `t`: that is the hand-written test set this run reports at the end, and selecting on it would
    # corrupt the headline number.
    best, best_step = -1.0, 0
    for step in range(args.steps + 1):
        if (step and step % args.eval_every == 0) or step == args.steps:
            t, _ = score(test, read_all(model, texts))
            held_out = exact(model, val[:1000], dev)
            improved = held_out > best
            say("step %5d | %4.1f min | loss %.4f | generated: exact %.2f | hand-written: every %.2f final %.2f found %.2f quiet %.2f%s"
                % (step, (time.time() - start) / 60, running.item() / max(since, 1), held_out,
                   t["every"], t["final"], t["found"], t["quiet"], " | best" if improved else ""))
            running.zero_()
            since = 0
            if improved:
                best, best_step = held_out, step
                torch.save({"model": model.state_dict(), "config": asdict(c), "step": step}, path)
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch([train[pick.randrange(len(train))] for _ in range(args.batch)])
        with amp():
            _, loss = model(x.to(dev), y.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.detach()
        since += 1

    # Everything below scores the weights we kept, not the last step's, so results.json describes
    # what is actually in reader.pt. The first eval always saves, so the file is there to load.
    if best_step != args.steps:
        say("keeping step %d (generated held-out exact %.2f), not the last step %d" % (best_step, best, args.steps))
    model.load_state_dict(torch.load(path, map_location=dev)["model"])
    model = model.to("cpu").eval()
    results = {"when": time.strftime("%Y-%m-%d %H:%M"), "params": model.param_count(), "steps": args.steps,
               "best_step": best_step, "best_generated_exact": round(best, 4),
               "seconds": round(time.time() - start), "args": recorded_args(args), "generated_exact": exact(model, val, "cpu")}
    predictions = read_all(model, texts)
    results["test"], results["test_mistakes"] = score(test, predictions)
    timings = []
    for t in texts:
        t0 = time.perf_counter()
        model.read(t)
        timings.append(time.perf_counter() - t0)
    results["ms_per_message_cpu"] = round(1000 * statistics.median(timings), 1)
    regex = regex_stopgap(texts)
    if regex is not None:
        results["regex_test"], results["regex_mistakes"] = score(test, regex)
    llm = os.path.join(args.out, "llm_baseline.json")
    if os.path.exists(llm):
        with open(llm) as f:
            saved = json.load(f)
        results["llm_test"], results["llm_seconds"] = saved["score"], saved.get("median_seconds")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    with_math = sum(1 for e in test if e["expected"])
    say("\nhand-written test set (%d messages: %d with math, %d without):" % (len(test), with_math, len(test) - with_math))
    say("| reader | every problem right | final answer right | found the math | quiet when there's none |")
    say("|---|---|---|---|---|")
    rows = [("smMATH_LANGUAGE_001", results["test"])]
    if "regex_test" in results:
        rows.append(("harness regex stopgap (no model)", results["regex_test"]))
    if "llm_test" in results:
        rows.append(("Ministral 8B, prompted", results["llm_test"]))
    for name, r in rows:
        say("| %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% |" % (name, 100 * r["every"], 100 * r["final"], 100 * r["found"], 100 * r["quiet"]))
    say("\ngenerated held-out: whole target exactly right %.0f%%   |   %.0f ms per message on the CPU"
        % (100 * results["generated_exact"], results["ms_per_message_cpu"]))
    say("\nsmMATH_LANGUAGE_001 mistakes on the hand-written set:")
    for m in results["test_mistakes"]:
        say("  %-60s want %s got %s" % (m["text"][:60], m["want"], m["got"]))


if __name__ == "__main__":
    main()
