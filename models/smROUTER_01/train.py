#!/usr/bin/env python3
"""Train smROUTER_01 from scratch on generated messages, then score it on the hand-written test set.

While training, tools are switched off at random. When a message's tool is off, the right answer becomes
"none", so the router learns to route among only the tools that exist.

    python3 train.py
"""
import argparse
import json
import os
import random
import statistics
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import data
import rules
from model import RouterNet, encode_batch, only_allowed

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

INTENTS, TOOLS = data.LABELS["intents"], data.LABELS["tools"]
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def as_tensors(examples, max_len):
    ids, pad = encode_batch([(e["prev"], e["text"]) for e in examples], max_len)
    intent = torch.tensor([INTENTS.index(e["intent"]) for e in examples])
    tool = torch.tensor([TOOLS.index(e["tool"]) for e in examples])
    ask = torch.tensor([int(e["ask_first"]) for e in examples])
    return ids, pad, intent, tool, ask


def random_switches(tool, gen, keep):
    """Each tool stays on with probability `keep`; "none" is always there. Off tools' examples point to "none"."""
    allowed = torch.rand(len(tool), len(TOOLS), generator=gen) < keep
    allowed[:, 0] = True
    target = tool.clone()
    target[~allowed[torch.arange(len(tool)), tool]] = 0
    return allowed, target


@torch.no_grad()
def predict(model, examples, max_len, off=(), device="cpu"):
    model.eval()
    allowed = torch.tensor([t not in off for t in TOOLS])
    out = []
    for s in range(0, len(examples), 256):
        batch = examples[s:s + 256]
        ids, pad = encode_batch([(e["prev"], e["text"]) for e in batch], max_len)
        li, lt, la = model(ids.to(device), pad.to(device))
        lt = only_allowed(lt, allowed.to(device).expand_as(lt))
        for i in range(len(batch)):
            out.append({"intent": INTENTS[li[i].argmax()], "tool": TOOLS[lt[i].argmax()], "ask_first": bool(la[i].argmax())})
    model.train()
    return out


def score(examples, guesses, off=()):
    rows = {"intent": [], "tool": [], "ask_first": [], "for_me": [], "all": []}
    mistakes = []
    for e, g in zip(examples, guesses):
        want_tool = "none" if e["tool"] in off else e["tool"]
        hit = {"intent": g["intent"] == e["intent"], "tool": g["tool"] == want_tool, "ask_first": g["ask_first"] == e["ask_first"],
               "for_me": (g["intent"] == "small_talk") == (e["intent"] == "small_talk")}
        hit["all"] = hit["intent"] and hit["tool"] and hit["ask_first"]
        for k in rows:
            rows[k].append(hit[k])
        if not hit["all"]:
            mistakes.append({"prev": e["prev"], "text": e["text"], "want": [e["intent"], want_tool, e["ask_first"]],
                             "got": [g["intent"], g["tool"], g["ask_first"]]})
    return {k: round(statistics.mean(v), 3) for k, v in rows.items()}, mistakes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--keep", type=float, default=0.8, help="chance each tool stays on in a switched-off batch")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--precision", choices=sorted(DTYPES), default=None,
                     help="defaults to bf16 on mps/cuda, fp32 elsewhere")
    ap.add_argument("--patience", type=int, default=3,
                     help="stop if generated held-out score hasn't improved in N epochs; 0 disables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    say = lambda line: print(line, flush=True)  # noqa: E731
    if args.precision is None:
        args.precision = "bf16" if args.device in ("mps", "cuda") else "fp32"
    dtype = DTYPES[args.precision]
    if args.precision == "fp16":
        say("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.")
    amp = (lambda: torch.autocast(device_type=args.device, dtype=dtype)) if dtype is not None else nullcontext

    with open(os.path.join(HERE, "test.json")) as f:
        test = json.load(f)
    with open(os.path.join(HERE, "test_arithmetic.json")) as f:  # bare arithmetic like "2+2", scored on its own
        arithmetic = json.load(f)
    held = test + arithmetic
    generated = data.generate(args.examples, seed=args.seed)
    train = data.without(generated, held)
    val = data.without(data.generate(3000, seed=args.seed + 1000), held)
    say("dropped %d generated messages that matched the hand-written test set word for word" % (len(generated) - len(train)))
    ids, pad, intent, tool, ask = as_tensors(train, args.max_len)
    model = RouterNet(len(INTENTS), len(TOOLS), args.d, args.layers, args.heads, args.max_len).to(args.device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    say("smROUTER_01: %.2fM parameters on %s, precision %s, %d generated examples, %d hand-written test messages"
        % (params / 1e6, args.device, args.precision, len(train), len(test)))

    started = time.time()
    best_val, no_improve = -1.0, 0
    best_state, best_epoch = None, 0
    for epoch in range(1, args.epochs + 1):
        order, total = torch.randperm(len(train), generator=gen), 0.0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            allowed, target = (random_switches(tool[b], gen, args.keep) if torch.rand(1, generator=gen).item() < 0.5
                               else (torch.ones(len(b), len(TOOLS), dtype=torch.bool), tool[b]))
            with amp():
                li, lt, la = model(ids[b].to(args.device), pad[b].to(args.device))
                loss = (F.cross_entropy(li, intent[b].to(args.device))
                        + F.cross_entropy(only_allowed(lt, allowed.to(args.device)), target.to(args.device))
                        + 0.5 * F.cross_entropy(la, ask[b].to(args.device)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        # Both scored in fp32 (predict() carries no autocast): "val" is the generated held-out set, used
        # below for early stopping; "test" is the hand-written headline set and must never drive a decision.
        v, _ = score(val, predict(model, val, args.max_len, device=args.device))
        t, _ = score(test, predict(model, test, args.max_len, device=args.device))
        say("epoch %d  loss %.4f  generated: all-right %.2f  |  hand-written: intent %.2f tool %.2f ask %.2f all-right %.2f  (%.0fs)"
            % (epoch, total / len(train), v["all"], t["intent"], t["tool"], t["ask_first"], t["all"], time.time() - started))
        if v["all"] > best_val:
            best_val, no_improve = v["all"], 0
            best_epoch = epoch
            # Keep the best epoch's weights in memory instead of writing them now. The harness reloads
            # router.pt whenever the file changes, so this project still writes that file exactly once,
            # at the end, and it now holds the best epoch rather than the last one.
            best_state = {name: t.detach().to("cpu").clone() for name, t in model.state_dict().items()}
        else:
            no_improve += 1
        if args.patience and no_improve >= args.patience:
            say("early stopping: generated held-out all-right hasn't improved in %d epochs, stopping after epoch %d "
                "(best %.2f)" % (args.patience, epoch, best_val))
            break

    model = model.to("cpu")
    if best_state is not None:
        if best_epoch != epoch:
            say("keeping epoch %d (generated held-out all-right %.2f), not the last epoch %d"
                % (best_epoch, best_val, epoch))
        model.load_state_dict(best_state)  # everything below scores and saves the weights we keep
    results = {"when": time.strftime("%Y-%m-%d %H:%M"), "params": params, "seconds": round(time.time() - started),
               "best_epoch": best_epoch, "best_generated_all": round(best_val, 4), "epochs_run": epoch,
               "args": recorded_args(args)}
    results["generated"], _ = score(val, predict(model, val, args.max_len))
    results["test"], results["test_mistakes"] = score(test, predict(model, test, args.max_len))
    results["rules_test"], results["rules_mistakes"] = score(test, [rules.classify(e["prev"], e["text"]) for e in test])
    web_off = ("web_search", "web_browser")
    results["test_web_off"], _ = score(test, predict(model, test, args.max_len, off=web_off), off=web_off)
    results["arithmetic_test"], results["arithmetic_mistakes"] = score(arithmetic, predict(model, arithmetic, args.max_len))
    results["rules_arithmetic"], _ = score(arithmetic, [rules.classify(e["prev"], e["text"]) for e in arithmetic])
    model.eval()
    with torch.no_grad():
        timings = []
        for e in test:
            t0 = time.perf_counter()
            i, p = encode_batch([(e["prev"], e["text"])], args.max_len)
            model(i, p)
            timings.append(time.perf_counter() - t0)
    results["ms_per_message_cpu"] = round(1000 * statistics.median(timings), 2)
    llm = os.path.join(args.out, "llm_baseline.json")
    if os.path.exists(llm):
        with open(llm) as f:
            results["llm_test"] = json.load(f)["score"]

    torch.save({"state": model.state_dict(), "intents": INTENTS, "tools": TOOLS,
                "config": {k: getattr(args, k) for k in ("d", "layers", "heads", "max_len")}},
               os.path.join(args.out, "router.pt"))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    say("\nhand-written test set (%d messages):" % len(test))
    say("| router | intent | tool | ask first | for me | all three right |")
    say("|---|---|---|---|---|---|")
    rows = [("smROUTER_01", results["test"]), ("keyword rules", results["rules_test"])]
    if "llm_test" in results:
        rows.append(("Ministral 8B prompt", results["llm_test"]))
    rows.append(("smROUTER_01, web tools off", results["test_web_off"]))
    for name, r in rows:
        say("| %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% |" % (name, 100 * r["intent"], 100 * r["tool"], 100 * r["ask_first"],
                                                             100 * r["for_me"], 100 * r["all"]))
    say("\nbare arithmetic, hand-written (%d messages): all three right %.0f%% (keyword rules %.0f%%)"
        % (len(arithmetic), 100 * results["arithmetic_test"]["all"], 100 * results["rules_arithmetic"]["all"]))
    for m in results["arithmetic_mistakes"]:
        say("  %-40s want %-32s got %s" % ((m["prev"][:14] + " → " if m["prev"] else "") + m["text"], m["want"], m["got"]))
    say("\ngenerated held-out: all three right %.0f%%   |   %.2f ms per message on the CPU"
        % (100 * results["generated"]["all"], results["ms_per_message_cpu"]))
    say("\nsmROUTER_01 mistakes on the hand-written set:")
    for m in results["test_mistakes"]:
        say("  %-60s want %-32s got %s" % ((m["prev"][:20] + " → " if m["prev"] else "") + m["text"][:40], m["want"], m["got"]))


if __name__ == "__main__":
    main()
