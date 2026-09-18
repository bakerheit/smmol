"""Load smLANGUAGE_RENDER_001 and turn one harness Language payload into a reply."""
import argparse
import json
import os
import time

import torch

from data import compact
from model import Config, LanguageGPT, encode


HERE = os.path.dirname(os.path.abspath(__file__))


class Speaker:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu")
        self.model = LanguageGPT(Config(**ck["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.steps = ck.get("step")

    def fits(self, payload, reply_room=300):
        return len(encode(compact(payload))) + reply_room <= self.model.c.max_len

    def speak_with_confidence(self, payload):
        prompt = compact(payload)
        if not self.fits(payload):
            raise ValueError("payload leaves too little room for a reply")
        return self.model.write_with_confidence(prompt)

    def speak(self, payload):
        return self.speak_with_confidence(payload)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_file", help="a JSON file containing the Language payload")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    with open(args.json_file) as f:
        payload = json.load(f)
    speaker = Speaker(os.path.join(args.out, "language.pt"))
    started = time.perf_counter()
    reply, sure, weakest = speaker.speak_with_confidence(payload)
    print(reply)
    print("%.1f ms, confidence %.3f, weakest %.3f" %
          (1000 * (time.perf_counter() - started), sure, weakest))


if __name__ == "__main__":
    main()
