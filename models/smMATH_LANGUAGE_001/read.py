"""smMATH_LANGUAGE_001 as a harness module: a message in, the math problems in it out.

    python3 read.py "what is the volume of 322234ft x 21323ft x 212231ft?"
"""
import argparse
import json
import os
import time

import torch

from data import parse_target
from model import Config, ReaderGPT

HERE = os.path.dirname(os.path.abspath(__file__))


class Reader:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu")
        self.model = ReaderGPT(Config(**ck["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.steps = ck.get("step")

    def read(self, text):
        """[{"id": "p1", "expression", "unit", "about"}, ...], or [] when there's nothing to work out."""
        return self.read_with_confidence(text)[0]

    def read_with_confidence(self, text):
        """(problems, how sure it was on average, how sure it was at its shakiest character)."""
        written, sure, weakest = self.model.read_with_confidence(text)
        return parse_target(written), sure, weakest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    reader = Reader(os.path.join(args.out, "reader.pt"))
    started = time.perf_counter()
    problems = reader.read(" ".join(args.text))
    print(json.dumps({"problems": problems, "ms": round(1000 * (time.perf_counter() - started), 1)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
