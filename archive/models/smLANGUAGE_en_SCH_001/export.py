"""Export a resumable PyTorch checkpoint as portable float32 safetensors weights."""

import argparse
import json
from pathlib import Path

import torch

from model import Config
from safetensors_io import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    Config.from_dict(saved["config"])
    output = args.out or args.checkpoint.with_suffix(".safetensors")
    metadata = {
        "model": "smLANGUAGE_en_SCH_001",
        "stage": saved.get("stage", "unknown"),
        "step": saved.get("step", 0),
        "corpus_sha256": saved.get("corpus_sha256", "unknown"),
    }
    save_file(saved["model"], output, metadata)
    output.with_name("model_config.json").write_text(json.dumps(saved["config"], indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
