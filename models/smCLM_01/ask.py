#!/usr/bin/env python3
"""Ask the idea reader what a word means, which words fit some ideas, or what a brand-new word means.

    python3 ask.py car
    python3 ask.py --ideas transportation control -machine
    python3 ask.py --teach zorb
    python3 ask.py --teach blip "the child eats the blip" "the blip grows in the garden"
    python3 ask.py --read "the pilot drives people to the city"
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

from model import MASK, UNK, ConceptModel, WordModel
from world import World, tokens

HERE = os.path.dirname(os.path.abspath(__file__))


def bar(score):
    return "█" * round(score * 10) + "·" * (10 - round(score * 10))


class Reader:
    def __init__(self, path):
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        self.frames, self.nouns, self.concepts = ckpt["frames"], ckpt["nouns"], ckpt["concepts"]
        self.held_out = set(ckpt["held_out"])
        self.ideas = ckpt["ideas"]
        self.max_len = cfg["max_len"]
        self.frame_id = {w: i + 3 for i, w in enumerate(self.frames)}
        self.noun_id = {w: i for i, w in enumerate(self.nouns)}
        self.idea_model = ConceptModel(len(self.frames), len(self.concepts), cfg["d"], cfg["layers"], cfg["heads"], cfg["max_len"])
        self.idea_model.load_state_dict(ckpt["idea_state"])
        self.word_model = WordModel(len(self.frames), len(self.nouns), cfg["d"], cfg["layers"], cfg["heads"], cfg["max_len"])
        self.word_model.load_state_dict(ckpt["word_state"])
        self.idea_model.eval()
        self.word_model.eval()

    def top(self, vec, n=8, floor=0.2):
        order = vec.argsort(descending=True)[:n]
        return [(self.concepts[i], vec[i].item()) for i in order if vec[i] >= floor]

    def closest(self, vec, n=6, skip=None):
        sims = F.cosine_similarity(self.ideas, vec.unsqueeze(0), dim=1)
        return [self.nouns[i] for i in sims.argsort(descending=True) if self.nouns[i] != skip][:n]

    @torch.no_grad()
    def guess_new_word(self, word, sentences):
        """Hide the new word in each sentence and ask both readers what fits there."""
        frames, ideas, nouns, is_noun, pad, target = [], [], [], [], [], []
        for sentence in sentences:
            toks = tokens(sentence)[:self.max_len]
            if word not in toks:
                print("  (skipping %r: it doesn't use %s)" % (sentence, word))
                continue
            f, x, n, isn, p = [0] * self.max_len, [[0.0] * len(self.concepts)] * self.max_len, [0] * self.max_len, [False] * self.max_len, [True] * self.max_len
            x = [list(row) for row in x]
            where = toks.index(word)
            for j, t in enumerate(toks):
                p[j] = False
                if j == where:
                    f[j] = MASK
                elif t in self.noun_id:
                    isn[j], n[j], x[j] = True, self.noun_id[t], self.ideas[self.noun_id[t]].tolist()
                else:
                    f[j] = self.frame_id.get(t, UNK)
            frames.append(f); ideas.append(x); nouns.append(n); is_noun.append(isn); pad.append(p); target.append(where)
        if not frames:
            return None, None, []
        f, x, n, isn, p, t = (torch.tensor(frames), torch.tensor(ideas), torch.tensor(nouns), torch.tensor(is_noun),
                               torch.tensor(pad), torch.tensor(target))
        idea_guess = torch.sigmoid(self.idea_model(f, x, isn, t, p)).mean(0)
        probs = torch.softmax(self.word_model(f, n, isn, t, p), dim=1).mean(0)
        word_guess = probs @ torch.tensor([[1.0 if v > 0.5 else 0.0 for v in row] for row in self.ideas.tolist()])
        swaps = [self.nouns[i] for i in probs.argsort(descending=True)[:6]]
        return idea_guess, word_guess, swaps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("word", nargs="?", help="a word to look up")
    ap.add_argument("--ideas", nargs=argparse.REMAINDER, metavar="IDEA",
                    help="ideas a word should have; put - before ideas it shouldn't (keep this option last)")
    ap.add_argument("--teach", nargs="+", metavar=("WORD", "SENTENCE"), help="a new word, then sentences that use it")
    ap.add_argument("--read", metavar="SENTENCE", help="show the ideas in a sentence")
    ap.add_argument("--model", default=os.path.join(HERE, "out", "models.pt"))
    args = ap.parse_args()
    if not os.path.exists(args.model):
        sys.exit("no trained model yet: run python3 train.py")
    r = Reader(args.model)

    if args.ideas:
        want = [c for c in args.ideas if not c.startswith("-")]
        avoid = [c[1:] for c in args.ideas if c.startswith("-")]
        unknown = [c for c in want + avoid if c not in r.concepts]
        if unknown:
            sys.exit("unknown ideas: %s\nideas: %s" % (", ".join(unknown), ", ".join(r.concepts)))
        idx = lambda names: [r.concepts.index(c) for c in names]  # noqa: E731
        fit = r.ideas[:, idx(want)].min(1).values if want else torch.ones(len(r.nouns))
        if avoid:
            fit = fit - r.ideas[:, idx(avoid)].max(1).values
        print("words that are %s%s:" % (" + ".join(want), "".join(" but not %s" % c for c in avoid)))
        for i in fit.argsort(descending=True)[:10]:
            mark = "  (ideas learned from usage)" if r.nouns[i] in r.held_out else ""
            print("  %-12s %s %.2f%s" % (r.nouns[i], bar(max(0.0, fit[i].item())), fit[i].item(), mark))
        return

    if args.teach:
        word = args.teach[0].lower()
        sentences = args.teach[1:] or World().teach.get(word, [])
        if not sentences:
            sys.exit("give some sentences that use %s" % word)
        print("teaching %r with %d sentence(s):" % (word, len(sentences)))
        for s in sentences:
            print("  • " + s)
        idea_guess, word_guess, swaps = r.guess_new_word(word, sentences)
        if idea_guess is None:
            sys.exit("none of the sentences use %s" % word)
        print("\nthe idea reader thinks a %s is:" % word)
        for c, s in r.top(idea_guess):
            print("  %-15s %s %.2f" % (c, bar(s), s))
        print("  like: %s" % ", ".join(r.closest(idea_guess)))
        print("\nthe word reader would put these words there: %s" % ", ".join(swaps))
        print("  which suggests: %s" % ", ".join("%s %.2f" % cs for cs in r.top(word_guess, 6)))
        return

    if args.read:
        toks = tokens(args.read)
        known = [t for t in toks if t in r.noun_id]
        if not known:
            sys.exit("that sentence has no words the model knows as things")
        for t in known:
            print("  %-12s %s" % (t, ", ".join("%s %.2f" % cs for cs in r.top(r.ideas[r.noun_id[t]], 5, 0.4))))
        together = torch.stack([r.ideas[r.noun_id[t]] for t in known]).max(0).values
        print("\nthe sentence is about: %s" % ", ".join(c for c, _ in r.top(together, 10, 0.5)))
        return

    if not args.word:
        ap.print_help()
        return
    word = args.word.lower()
    if word not in r.noun_id:
        sys.exit("%s isn't a word it knows. Teach it: python3 ask.py --teach %s \"a sentence that uses %s\"" % (word, word, word))
    vec = r.ideas[r.noun_id[word]]
    held = word in r.held_out
    print("%s%s" % (word, "  (never told its ideas; worked out from how it's used)" if held else "  (its ideas were given)"))
    for c, s in r.top(vec, 10):
        print("  %-15s %s %.2f" % (c, bar(s), s))
    print("  like: %s" % ", ".join(r.closest(vec, skip=word)))


if __name__ == "__main__":
    main()
