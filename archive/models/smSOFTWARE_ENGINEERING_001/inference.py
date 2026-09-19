"""Generate short code continuations from a trained checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from model import Config, LanguageModel
from safetensors_io import load_file


class Generator:
    def __init__(self, checkpoint, device="auto"):
        path = Path(checkpoint)
        if path.suffix == ".safetensors":
            document = json.loads(path.with_name("model_config.json").read_text())
            state, self.metadata = load_file(path)
        else:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            document = saved["config"]
            state = saved["model"]
            self.metadata = {"stage": saved.get("stage", "unknown"), "step": str(saved.get("step", 0))}
        self.config = Config.from_dict(document)
        self.model = LanguageModel(self.config)
        self.model.load_state_dict(state)
        self.device = "mps" if device == "auto" and torch.backends.mps.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.model.to(self.device).eval()

    @torch.no_grad()
    def complete(self, prompt, max_tokens=160, temperature=0.35, top_k=24, repetition_penalty=1.08):
        tokens = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=self.device)
        generated = self.model.generate(
            tokens, max_tokens, temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        return bytes(generated[0].tolist()).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "out" / "latest.safetensors")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    print(Generator(args.checkpoint, args.device).complete(
        args.prompt, max_tokens=args.tokens, temperature=args.temperature,
    ))


if __name__ == "__main__":
    main()
