"""Measure where smLLM_01's training compute actually goes on the M5.

Three questions:
  1. What matmul throughput can this machine actually reach? (empirical roofline, per dtype)
  2. What does the real training loop achieve, and what fraction of that roofline is it?
  3. Does mixed precision, torch.compile, or a bigger batch change it?
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "smLLM_01")
sys.path.insert(0, HERE)
from model import Config, TinyGPT  # noqa: E402

DEV = "mps"


def sync():
    torch.mps.synchronize()


def roofline(dtype, n=2048, iters=50):
    """Largest sustained matmul rate we can actually get, in TFLOP/s."""
    a = torch.randn(n, n, device=DEV, dtype=dtype)
    b = torch.randn(n, n, device=DEV, dtype=dtype)
    for _ in range(10):
        a @ b
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync()
    dt = time.perf_counter() - t
    return (2 * n ** 3 * iters) / dt / 1e12


def flops_per_token(model, cfg):
    """nanoGPT / PaLM convention: fwd+bwd, counting attention score matmuls."""
    n = sum(p.numel() for p in model.parameters())
    n_noemb = n - model.pos.weight.numel()          # tok is tied to head, keep it
    return 6 * n_noemb + 12 * cfg.layers * cfg.d * cfg.ctx


def make_data():
    with open(os.path.join(HERE, "data", "input.txt"), "rb") as f:
        raw = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
    return raw[: int(len(raw) * 0.9)]


def run(label, batch, dtype, compiled, steps=40, warmup=15, data=None, split_timing=False):
    torch.manual_seed(1337)
    cfg = Config()
    model = TinyGPT(cfg).to(DEV)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-3, betas=(0.9, 0.99))

    net = model
    if compiled:
        try:
            net = torch.compile(model)
        except Exception as exc:
            return {"label": label, "error": "compile failed: %s" % exc}

    def get_batch():
        ix = torch.randint(len(data) - cfg.ctx - 1, (batch,))
        x = torch.stack([data[i:i + cfg.ctx] for i in ix]).long()
        y = torch.stack([data[i + 1:i + 1 + cfg.ctx] for i in ix]).long()
        return x.to(DEV), y.to(DEV)

    autocast = (torch.autocast(device_type="mps", dtype=dtype)
                if dtype is not None else torch.autocast(device_type="mps", enabled=False))

    try:
        for _ in range(warmup):
            x, y = get_batch()
            with autocast:
                _, loss = net(x, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sync()
    except Exception as exc:
        return {"label": label, "error": "%s: %s" % (type(exc).__name__, str(exc)[:120])}

    data_s = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        if split_timing:
            sync(); d0 = time.perf_counter()
        x, y = get_batch()
        if split_timing:
            sync(); data_s += time.perf_counter() - d0
        with autocast:
            _, loss = net(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sync()
    wall = time.perf_counter() - t0

    tokens = steps * batch * cfg.ctx
    fpt = flops_per_token(model, cfg)
    out = {"label": label, "batch": batch,
           "dtype": "fp32" if dtype is None else str(dtype).replace("torch.", ""),
           "compiled": compiled,
           "steps_per_s": steps / wall,
           "tokens_per_s": tokens / wall,
           "tflops": tokens * fpt / wall / 1e12,
           "final_loss": float(loss.detach()),
           "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    if split_timing:
        out["data_pct"] = 100 * data_s / wall
    del model, net, opt
    torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results.json")
    args = ap.parse_args()
    data = make_data()
    results = {"roofline": {}, "configs": [], "batch_sweep": []}

    print("=== empirical roofline (2048^3 matmul) ===", flush=True)
    for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        tf = roofline(dt)
        results["roofline"][name] = tf
        print("  %-5s %7.2f TFLOP/s" % (name, tf), flush=True)

    print("\n=== training configs (smLLM_01, batch 64, ctx 256) ===", flush=True)
    configs = [
        ("fp32 eager (baseline)", 64, None, False),
        ("bf16 autocast", 64, torch.bfloat16, False),
        ("fp16 autocast", 64, torch.float16, False),
        ("fp32 + compile", 64, None, True),
        ("bf16 + compile", 64, torch.bfloat16, True),
    ]
    for label, b, dt, comp in configs:
        r = run(label, b, dt, comp, data=data, split_timing=(label.startswith("fp32 eager")))
        results["configs"].append(r)
        if "error" in r:
            print("  %-24s ERROR %s" % (label, r["error"]), flush=True)
        else:
            extra = ("  data-prep %.0f%%" % r["data_pct"]) if "data_pct" in r else ""
            print("  %-24s %8.0f tok/s  %6.2f TFLOP/s  loss %.3f%s"
                  % (label, r["tokens_per_s"], r["tflops"], r["final_loss"], extra), flush=True)

    print("\n=== batch sweep (fp32 eager) ===", flush=True)
    for b in (1, 2, 4, 8, 16, 32, 64, 128):
        r = run("batch %d" % b, b, None, False, steps=20, warmup=8, data=data)
        results["batch_sweep"].append(r)
        if "error" not in r:
            print("  batch %3d  %8.0f tok/s  %6.2f TFLOP/s  %6.1f steps/s"
                  % (b, r["tokens_per_s"], r["tflops"], r["steps_per_s"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
