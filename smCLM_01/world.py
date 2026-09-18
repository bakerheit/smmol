"""A small made-up English world where every noun is a bundle of ideas.

A sentence only puts a noun in a slot when the noun has all of the slot's ideas (and none of
the ones it rules out). So the ideas of a word show up in how it's used, and that's the only
way a model can learn the ideas of a word it was never told about.
"""
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SLOT = re.compile(r"^\{(\w+):([^}]*)\}$")
TOKEN = re.compile(r"[a-z]+")


def tokens(sentence):
    return TOKEN.findall(sentence.lower())


class World:
    def __init__(self, path=os.path.join(HERE, "concepts.json")):
        with open(path) as f:
            data = json.load(f)
        self.about = data["concepts"]
        self.concepts = list(self.about)
        self.tags = {w: set(t) for w, t in data["words"].items()}
        unknown = sorted({c for t in self.tags.values() for c in t} - set(self.concepts))
        if unknown:
            raise ValueError("words use ideas that aren't in the list: %s" % ", ".join(unknown))
        self.nouns = sorted(self.tags)
        self.held_out = list(data["held_out"])
        missing = [w for w in self.held_out if w not in self.tags]
        if missing:
            raise ValueError("held-out words need tags for scoring: %s" % ", ".join(missing))
        self.teach = data.get("teach", {})
        self.templates = [self._parse(t) for t in data["templates"]]
        self.frames = sorted({p[1] for t in self.templates for p in t if p[0] == "word"} - set(self.tags))

    def _parse(self, text):
        parts = []
        for tok in text.split():
            m = SLOT.match(tok)
            if not m:
                parts.append(("word", tok))
                continue
            ideas = [c.strip() for c in m.group(2).split(",") if c.strip()]
            need = frozenset(c for c in ideas if not c.startswith("!"))
            avoid = frozenset(c[1:] for c in ideas if c.startswith("!"))
            unknown = (need | avoid) - set(self.concepts)
            if unknown:
                raise ValueError("template %r uses unknown ideas: %s" % (text, ", ".join(sorted(unknown))))
            fits = [w for w in self.nouns if need <= self.tags[w] and not avoid & self.tags[w]]
            if len(fits) < 2:
                raise ValueError("template %r: only %d word(s) fit slot %s" % (text, len(fits), m.group(1)))
            parts.append(("slot", m.group(1), need, avoid, fits))
        return parts

    def sentence(self, rng, detail=False):
        """One random sentence. With detail, also the (word, needed ideas, ruled-out ideas) of each slot."""
        parts = rng.choice(self.templates)
        chosen, words, filled = {}, [], []
        for part in parts:
            if part[0] == "word":
                words.append(part[1])
                continue
            _, name, need, avoid, fits = part
            if name not in chosen:
                pick = rng.choice(fits)
                for _ in range(5):  # two slots in one sentence shouldn't be the same word
                    if pick not in chosen.values():
                        break
                    pick = rng.choice(fits)
                chosen[name] = pick
                filled.append((pick, need, avoid))
            words.append(chosen[name])
        return (words, filled) if detail else words

    def corpus(self, n, seed=0):
        rng = random.Random(seed)
        return [self.sentence(rng) for _ in range(n)]

    def vector(self, word):
        return [1.0 if c in self.tags[word] else 0.0 for c in self.concepts]
