"""Generate English text from a smLANGUAGE_EN_001 checkpoint."""

import argparse
from pathlib import Path

import torch

from model import Config, LanguageModel


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=HERE / "out" / "best.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LanguageModel(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    torch.manual_seed(args.seed)
    prompt = args.prompt.replace("\\n", "\n")
    tokens = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    output = model.generate(tokens, args.tokens, args.temperature, args.top_k)
    print(bytes(output[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
