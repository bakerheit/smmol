"""Gadget world: on/off gadgets with secret wiring that changes every episode.

Some gadgets are switches: they stay where you put them. The rest follow one or two gadgets
earlier in the wiring (copy, NOT, AND, OR). The learner can flip any gadget and see what
everything reads next. Nobody ever tells it the wiring.
"""
import torch

GADGETS = 8
LETTERS = "ABCDEFGH"
SWITCH_ODDS = 0.35
COPY, NOT, AND, OR = range(4)

# Tokens: "gadget i reads v" = 2*i + v, "flip gadget i to v" = ACT + 2*i + v, BOS starts an episode.
ACT = 2 * GADGETS
BOS = 4 * GADGETS
VOCAB = BOS + 1


class Worlds:
    """A batch of secret wirings, one per episode."""

    def __init__(self, b, gen, n=GADGETS):
        rows = torch.arange(b)
        self.b, self.n, self.rows = b, n, rows
        self.order = torch.argsort(torch.rand(b, n, generator=gen), dim=1)  # wiring order, causes first
        self.switch = torch.zeros(b, n, dtype=torch.bool)
        self.p1 = torch.zeros(b, n, dtype=torch.long)
        self.p2 = torch.zeros(b, n, dtype=torch.long)
        self.gate = torch.zeros(b, n, dtype=torch.long)
        for k in range(n):
            node = self.order[:, k]
            if k == 0:
                self.switch[rows, node] = True
                continue
            first = (torch.rand(b, generator=gen) * k).long()
            if k > 1:
                second = (torch.rand(b, generator=gen) * (k - 1)).long()
                second = second + (second >= first).long()
                gate = torch.randint(0, 4, (b,), generator=gen)
            else:
                second = first
                gate = torch.randint(0, 2, (b,), generator=gen)
            self.switch[rows, node] = torch.rand(b, generator=gen) < SWITCH_ODDS
            self.p1[rows, node] = self.order[rows, first]
            self.p2[rows, node] = self.order[rows, second]
            self.gate[rows, node] = gate

    def take(self, idx):
        """The same wirings, picked or repeated by index."""
        sub = Worlds.__new__(Worlds)
        sub.b, sub.n, sub.rows = len(idx), self.n, torch.arange(len(idx))
        for name in ("order", "switch", "p1", "p2", "gate"):
            setattr(sub, name, getattr(self, name)[idx])
        return sub

    def settle(self, switches, held=None, value=None):
        """What every gadget reads, given the switch positions and one gadget being held."""
        x = switches.clone()
        rows = self.rows
        for k in range(self.n):
            node = self.order[:, k]
            a = x[rows, self.p1[rows, node]]
            c = x[rows, self.p2[rows, node]]
            g = self.gate[rows, node]
            follow = torch.where(g == COPY, a, torch.where(g == NOT, 1 - a, torch.where(g == AND, a & c, a | c)))
            v = torch.where(self.switch[rows, node], x[rows, node], follow)
            if held is not None:
                v = torch.where(held == node, value, v)
            x[rows, node] = v
        return x

    def act(self, switches, held, value):
        """Flip a gadget. A flipped switch stays put; anything else springs back once let go.

        Returns the new switch positions, what everything read while holding, and how
        everything rests after letting go (the state the next flip starts from).
        """
        reading = self.settle(switches, held, value)
        switches = switches.clone()
        stays = self.switch[self.rows, held]
        switches[self.rows, held] = torch.where(stays, value, switches[self.rows, held])
        return switches, reading, self.settle(switches)

    def describe(self, i):
        """World i's secret wiring in plain words."""
        lines = []
        for node in self.order[i].tolist():
            name = LETTERS[node]
            a, c = LETTERS[self.p1[i, node].item()], LETTERS[self.p2[i, node].item()]
            if self.switch[i, node]:
                lines.append(f"{name} is a switch")
            else:
                lines.append(f"{name} = " + [a, f"NOT {a}", f"{a} AND {c}", f"{a} OR {c}"][self.gate[i, node].item()])
        return lines


def reading_tokens(reading):
    return 2 * torch.arange(reading.shape[1]) + reading


def action_tokens(held, value):
    return (ACT + 2 * held + value)[:, None]


def random_flip(worlds, rest, gen):
    held = torch.randint(0, worlds.n, (worlds.b,), generator=gen)
    return held, 1 - rest[worlds.rows, held]


def start(worlds, gen):
    """Random switch positions, and the first look at the world."""
    switches = torch.randint(0, 2, (worlds.b, worlds.n), generator=gen)
    reading = worlds.settle(switches)
    seq = torch.cat([torch.full((worlds.b, 1), BOS), reading_tokens(reading)], dim=1)
    return switches, reading, seq


def episodes(b, steps, gen):
    """Fresh wirings and random flips: exactly what the learner sees during training."""
    w = Worlds(b, gen)
    switches, rest, seq = start(w, gen)
    parts = [seq]
    for _ in range(steps):
        held, value = random_flip(w, rest, gen)
        switches, reading, rest = w.act(switches, held, value)
        parts += [action_tokens(held, value), reading_tokens(reading)]
    return torch.cat(parts, dim=1)


if __name__ == "__main__":
    # Self-check: every reading must obey its wiring, and the wiring must have no loops.
    gen = torch.Generator().manual_seed(0)
    b = 2000
    w = Worlds(b, gen)
    switches = torch.randint(0, 2, (b, GADGETS), generator=gen)
    held = torch.randint(0, GADGETS, (b,), generator=gen)
    value = torch.randint(0, 2, (b,), generator=gen)
    x = w.settle(switches, held, value)
    rank = torch.argsort(w.order, dim=1)
    for i in range(b):
        for node in range(GADGETS):
            if node == held[i]:
                assert x[i, node] == value[i]
            elif w.switch[i, node]:
                assert x[i, node] == switches[i, node]
            else:
                p1, p2, g = w.p1[i, node].item(), w.p2[i, node].item(), w.gate[i, node].item()
                assert rank[i, p1] < rank[i, node] and (g < 2 or rank[i, p2] < rank[i, node])
                a, c = x[i, p1].item(), x[i, p2].item()
                assert x[i, node].item() == [a, 1 - a, a & c, a | c][g]
    after, _, rest = w.act(switches, held, value)
    stays = w.switch[w.rows, held]
    assert torch.equal(after[w.rows, held][stays], value[stays])
    assert torch.equal(after[w.rows, held][~stays], switches[w.rows, held][~stays])
    assert torch.equal(rest, w.settle(after))
    flip_held, flip_value = random_flip(w, rest, gen)
    assert torch.equal(w.settle(after, flip_held, flip_value)[w.rows, flip_held], 1 - rest[w.rows, flip_held])

    flip_held, flip_value = random_flip(w, w.settle(switches), gen)
    truth = w.settle(switches, flip_held, flip_value)
    same = w.settle(switches).clone()
    same[w.rows, flip_held] = flip_value
    print("world self-check passed")
    print(f"switches per world: {w.switch.float().sum(1).mean():.2f} of {GADGETS}")
    print(f"flips that change some other gadget: {(truth != same).any(1).float().mean():.0%}")
    print("example wiring:", "; ".join(w.describe(0)))
