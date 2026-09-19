#!/usr/bin/env python3
"""Ask smROUTER_01 where a message should go.

    python3 route.py "what's the weather in Tokyo tomorrow"
    python3 route.py --prev "Which city should I check the weather for?" "Denver"
    python3 route.py --off web_search,web_browser "what's the weather in Tokyo tomorrow"
"""
import argparse
import json
import os
import time

import torch

from model import RouterNet, encode_batch, only_allowed

HERE = os.path.dirname(os.path.abspath(__file__))


class Router:
    def __init__(self, path=os.path.join(HERE, "out", "router.pt")):
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        self.intents, self.tools = ckpt["intents"], ckpt["tools"]
        self.max_len = cfg["max_len"]
        self.model = RouterNet(len(self.intents), len(self.tools), cfg["d"], cfg["layers"], cfg["heads"], cfg["max_len"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

    @torch.no_grad()
    def route(self, text, prev="", tools=None):
        """`tools` is the router tools that exist right now (None means all). The rest can't be chosen."""
        allowed = torch.tensor([t == "none" or tools is None or t in tools for t in self.tools])
        started = time.perf_counter()
        ids, pad = encode_batch([(prev or "", text)], self.max_len)
        li, lt, la = self.model(ids, pad)
        pi = torch.softmax(li[0], 0)
        pt = torch.softmax(only_allowed(lt[0], allowed), 0)
        pa = torch.softmax(la[0], 0)
        intent, tool = self.intents[pi.argmax()], self.tools[pt.argmax()]
        return {"for_me": intent != "small_talk", "intent": intent, "intent_p": round(pi.max().item(), 3),
                "tool": tool, "tool_p": round(pt.max().item(), 3), "ask_first": bool(pa[1] > 0.5),
                "ask_p": round(pa[1].item(), 3), "ms": round(1000 * (time.perf_counter() - started), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--prev", default="", help="the assistant's last message")
    ap.add_argument("--off", default="", help="comma-separated tools that are switched off")
    args = ap.parse_args()
    router = Router()
    off = {t.strip() for t in args.off.split(",") if t.strip()}
    print(json.dumps(router.route(args.text, args.prev, [t for t in router.tools if t not in off]), indent=1))


if __name__ == "__main__":
    main()
