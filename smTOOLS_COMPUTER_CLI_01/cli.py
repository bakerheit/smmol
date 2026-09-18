"""smTOOLS_COMPUTER_CLI_01 as a harness module: a request in, a command line out. It never runs anything.

    python3 cli.py --platform macos "what's using port 8771"
"""
import argparse
import json
import os
import time

import torch

from check import dangers, risk_of
from data import prompt, split_target
from model import Config, ReaderGPT

HERE = os.path.dirname(os.path.abspath(__file__))
PLATFORMS = ["ubuntu", "fedora", "arch", "macos", "windows"]


class Commander:
    takes = "request"

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu")
        self.model = ReaderGPT(Config(**ck["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.steps = ck.get("step")

    def command(self, text, platform="macos"):
        """{"command", "risk", "said_risk", "dangers"}. risk comes from the rules in code, not from the model."""
        if platform not in PLATFORMS:
            raise ValueError("no platform called %r" % platform)
        started = time.perf_counter()
        written, sure, weakest = self.model.read_with_confidence(prompt(platform, text))
        command, said_risk = split_target(written)
        return {"command": command, "risk": risk_of(command) if command else "", "said_risk": said_risk,
                "dangers": dangers(command) if command else [], "platform": platform,
                "sure": sure, "weakest": weakest,
                "ms": round(1000 * (time.perf_counter() - started), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", nargs="+")
    ap.add_argument("--platform", default="macos", choices=PLATFORMS)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    out = Commander(os.path.join(args.out, "cli.pt")).command(" ".join(args.request), args.platform)
    print(json.dumps(out, indent=1))
    if not out["command"]:
        print("(nothing to run: this doesn't look like a terminal job)")


if __name__ == "__main__":
    main()
