"""Make smLLM_01 write. Run train.py first."""
import argparse
import os

import torch

from model import Config, TinyGPT

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "out", "ckpt.pt"))
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--temp", type=float, default=0.8, help="lower = safer, higher = weirder")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = TinyGPT(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    prompt = args.prompt.replace("\\n", "\n")
    idx = torch.tensor([list(prompt.encode())], dtype=torch.long, device=dev)
    out = model.generate(idx, args.tokens, args.temp, args.top_k)
    print(bytes(out[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
