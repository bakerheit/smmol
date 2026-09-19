"""Score a checkpoint on a sequential split and print fixed-prompt samples."""

import argparse
import math
from pathlib import Path

import torch

from model import Config, LanguageModel


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=HERE / "out" / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config(**checkpoint["config"])
    model = LanguageModel(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    data = torch.frombuffer(bytearray((args.data_dir / f"{args.split}.txt").read_bytes()), dtype=torch.uint8)

    windows = min(args.batches * args.batch, (len(data) - 1) // config.ctx)
    losses = []
    with torch.no_grad():
        for first in range(0, windows, args.batch):
            count = min(args.batch, windows - first)
            starts = (torch.arange(count) + first) * config.ctx
            offsets = torch.arange(config.ctx)
            x = data[starts[:, None] + offsets].long().to(device)
            y = data[starts[:, None] + offsets + 1].long().to(device)
            losses.append((model(x, y)[1].item(), count))
    loss = sum(value * count for value, count in losses) / sum(count for _, count in losses)
    print(
        f"{args.split}: loss {loss:.4f} | {loss / math.log(2):.3f} bits/byte | "
        f"byte perplexity {math.exp(loss):.2f} | {windows * config.ctx:,} bytes"
    )

    prompts = ("Once upon a time", "The little dog", "She opened the door and")
    torch.manual_seed(42)
    for prompt in prompts:
        tokens = torch.tensor([list(prompt.encode())], dtype=torch.long, device=device)
        output = model.generate(tokens, 220, temperature=0.7, top_k=40)
        print("\n---\n" + bytes(output[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
