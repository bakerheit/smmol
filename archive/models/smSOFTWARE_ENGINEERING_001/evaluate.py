"""Compare the grade-one base and software fine-tune on held-out code and school text."""

import argparse
import json
import math
from pathlib import Path

import torch

from model import Config, LanguageModel
from training_data import build_splits, read_records, read_school_items, render_school


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE.parent / "smLANGUAGE_en_SCH_001" / "checkpoints" / "grade_01" / "best.pt"
DEFAULT_SCHOOL = HERE.parent / "smLANGUAGE_en_SCH_001" / "data" / "curriculum.jsonl"


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = Config.from_dict(saved["config"])
    model = LanguageModel(config)
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), config


@torch.no_grad()
def byte_loss(model, config, data, device, seed, batches=64, batch_size=32):
    source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if len(source) <= config.context_length + 1:
        raise ValueError("evaluation source is too short")
    generator = torch.Generator().manual_seed(seed)
    losses = []
    offsets = torch.arange(config.context_length)
    for _ in range(batches):
        starts = torch.randint(len(source) - config.context_length - 1, (batch_size,), generator=generator)
        x = source[starts[:, None] + offsets].long().to(device)
        y = source[starts[:, None] + offsets + 1].long().to(device)
        losses.append(model(x, y)[1].item())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "out" / "latest.pt")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--corpus", type=Path, default=HERE / "data" / "raw" / "records.jsonl")
    parser.add_argument("--school-corpus", type=Path, default=DEFAULT_SCHOOL)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    document = json.loads((HERE / "model_config.json").read_text())
    training = document["training"]
    records = read_records(args.corpus)
    school_items = read_school_items(args.school_corpus)
    _, code_validation, split = build_splits(
        records, school_items, training["seed"], training["school_replay_fraction"],
    )
    school = render_school(school_items)
    tuned, tuned_config = load_model(args.checkpoint, device)
    base, base_config = load_model(args.base, device)
    if tuned_config != base_config:
        parser.error("base and tuned architectures do not match")
    results = {"device": device, "held_out_repositories": split["validation_repositories"]}
    for label, model in (("base", base), ("tuned", tuned)):
        code = byte_loss(model, tuned_config, code_validation, device, training["seed"] + 1, args.batches)
        language = byte_loss(model, tuned_config, school, device, training["seed"] + 2, args.batches)
        results[label] = {
            "code_loss": round(code, 6),
            "code_bits_per_byte": round(code / math.log(2), 6),
            "school_loss": round(language, 6),
            "school_bits_per_byte": round(language / math.log(2), 6),
        }
    results["change"] = {
        "code_loss_percent": round(100 * (results["tuned"]["code_loss"] / results["base"]["code_loss"] - 1), 2),
        "school_loss_percent": round(100 * (results["tuned"]["school_loss"] / results["base"]["school_loss"] - 1), 2),
    }
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
