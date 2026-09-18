"""Recall world: key-value pairs, neutral filler, one query, one answer byte.

An episode is bytes and nothing else::

    G=m;  R=q;  ...filler...  ?R q \\n

Pairs are written `k=v;` with distinct keys. Neutral filler bytes open the gap. Only the answer byte
is scored.

The gap, defined once
---------------------
With `query_at` the index of the `?` and `equals_at` the index of the queried pair's `=`::

    gap = query_at - equals_at - 1

That one line is :func:`exact_gap`, and it is the only definition used anywhere: the generator builds
to it, the parser recovers it, and the tests recount it straight off the raw bytes.

Episodes are generated fresh from a seed and never stored, the way `smALLM_01/world.py` generates wirings:
the world is the answer key, nothing is hand-labelled.

What is and is not free
-----------------------
The bytes inside the gap are the queried value, its `;`, every later pair, and filler. A pair costs four
bytes, so with `pairs` pairs and the queried pair at ordinal `q` the smallest reachable gap is::

    min_gap(pairs, q) = 2 + 4 * (pairs - 1 - q)

Filler can only make a gap longer, never shorter. So `pairs`, `gap` and `q` are not three free axes:
`q` is free exactly when `4 * (pairs - 1 - q) <= gap - 2`. Every requested cell
(`pairs` in :data:`PAIR_COUNTS`, `gap` in :data:`GAPS`) is constructible, because `q = pairs - 1` needs
only `gap >= 2`. :func:`feasible_query_indices` returns the whole feasible set and the sampler draws
uniformly *from that set* — which is **not** uniform over all positions at short gaps:

===========  ========  ==========================================
pairs        gap       feasible query ordinals
===========  ========  ==========================================
32           8         **30, 31 only** — two of thirty-two
32           32        24..31 — eight of thirty-two
any          128, 512  every ordinal
===========  ========  ==========================================

So a short-gap cell leaks position, and the leak is measured rather than denied: every episode carries
its :attr:`Episode.feasible_positions`, :func:`position_leakage` reports how much accuracy a
position-only guesser gets for free, and the `position_i` lazy baselines put it in the results table.
An explicitly requested impossible combination is rejected, never silently repaired by moving the gap.

No held-out key split
---------------------
The research memo asks for query keys reserved for evaluation. It is not implemented here and should
not be: `pairs=32` needs 32 distinct keys in a single episode, and the alphabet has 36. A holdout big
enough to matter leaves fewer than 32 keys and makes the largest cell unconstructible; a holdout of
four leaves exactly 32, so every `pairs=32` episode would use the entire training alphabet and key
diversity would be gone. Revisit only behind a larger key alphabet.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
import random
import string
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

# --- the alphabet -------------------------------------------------------------------------------
# Three disjoint printable sets plus syntax. 36 keys and 33 values clear the "at least 32" floor and
# lift the old 16-key cap that pinned `fast` on its dk=16 interference boundary.
KEYS = string.ascii_uppercase + string.digits              # 36
VALUES = string.ascii_lowercase + "!#$%&*+"                # 33
EQUALS, SEMI, QUERY, NEWLINE = b"="[0], b";"[0], b"?"[0], b"\n"[0]
SYNTAX = bytes((EQUALS, SEMI, QUERY, NEWLINE))
FILLER = "".join(sorted(set(string.printable[:95]) - set(KEYS) - set(VALUES) - set(SYNTAX.decode())))

KEY_BYTES = tuple(KEYS.encode())
VALUE_BYTES = tuple(VALUES.encode())
FILLER_BYTES = tuple(FILLER.encode())
KEY_SET, VALUE_SET, FILLER_SET = set(KEY_BYTES), set(VALUE_BYTES), set(FILLER_BYTES)

PAIR_LEN = 4        # k = v ;
TAIL_LEN = 3        # ? k answer   (the newline is scored context, not the answer)

# --- the grid -----------------------------------------------------------------------------------
PAIR_COUNTS = (2, 4, 8, 16, 32)
GAPS = (8, 32, 128, 512)
EVAL_TRIALS = 1_000
EVAL_SEED = 20260917

# Three disjoint episode universes. Selection never touches the reported set, and training never
# touches either, so "best checkpoint" is not chosen on the number that gets published.
SELECT_SEED = 20260918      # in-run monitoring and best-checkpoint selection
SELECT_TRIALS = 500
TRAIN_SEED = 20260919       # the training stream

assert len(KEY_BYTES) >= max(PAIR_COUNTS) and len(VALUE_BYTES) >= 32
assert not (KEY_SET & VALUE_SET) and not (KEY_SET & FILLER_SET) and not (VALUE_SET & FILLER_SET)
assert not (FILLER_SET & set(SYNTAX))


class WorldError(ValueError):
    """An episode that does not mechanically determine its own answer."""


# --- episodes -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Episode:
    """One generated episode. `data` is the whole truth; everything else is bookkeeping."""

    data: bytes
    pairs: Tuple[Tuple[int, int], ...]   # (key byte, value byte) in written order
    query_index: int                     # ordinal of the queried pair among `pairs`
    query_key: int
    answer: int
    answer_index: int                    # index of the answer byte inside `data`
    gap: int
    overwrite: bool
    seed: int
    feasible_positions: Tuple[int, ...] = ()   # ordinals this cell could have queried; see position_leakage

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    def text(self) -> str:
        return self.data.decode("ascii")


def exact_gap(query_at: int, equals_at: int) -> int:
    """The one definition of the gap: bytes strictly between the queried `=` and the `?`."""
    return query_at - equals_at - 1


def min_gap(pairs: int, query_index: int) -> int:
    """Smallest gap reachable with the queried pair at `query_index`: value, `;`, then later pairs."""
    return 2 + PAIR_LEN * (pairs - 1 - query_index)


def feasible_query_indices(pairs: int, gap: int, overwrite: bool = False) -> Tuple[int, ...]:
    """Every ordinal the queried pair can occupy at this exact gap.

    Overwrite episodes also need one earlier slot for the stale write, so index 0 drops out.
    """
    if pairs < 2:
        raise WorldError("pairs must be at least 2, got %d" % pairs)
    lowest = 1 if overwrite else 0
    return tuple(q for q in range(lowest, pairs) if min_gap(pairs, q) <= gap)


def check_cell(pairs: int, gap: int, overwrite: bool = False) -> None:
    """Reject an impossible (pairs, gap) cell loudly rather than bending the gap to fit."""
    if not feasible_query_indices(pairs, gap, overwrite):
        raise WorldError(
            "no query position gives gap %d with %d pairs%s: the shortest possible gap is %d"
            % (gap, pairs, " under --overwrite" if overwrite else "", min_gap(pairs, pairs - 1))
        )


def derive_seed(*parts) -> int:
    """A stable 63-bit seed from any mix of ints and strings. Same parts, same episode, any machine."""
    digest = hashlib.blake2b(b"|".join(str(p).encode() for p in parts), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


def _compose(total: int, slots: int, rng: random.Random) -> List[int]:
    """Randomly split `total` filler bytes over `slots` places."""
    if slots <= 0:
        if total:
            raise WorldError("nowhere to put %d filler bytes" % total)
        return []
    cuts = sorted(rng.randint(0, total) for _ in range(slots - 1))
    edges = [0] + cuts + [total]
    return [edges[i + 1] - edges[i] for i in range(slots)]


def _filler(count: int, rng: random.Random) -> bytes:
    """Freshly sampled every time, so a fixed padding pattern is never a shortcut."""
    return bytes(rng.choice(FILLER_BYTES) for _ in range(count))


def sample_episode(
    seed: int,
    pairs: int,
    gap: int,
    overwrite: bool = False,
    lead_filler_max: int = 0,
    query_index: Optional[int] = None,
) -> Episode:
    """One fresh episode at exactly this pair count and exactly this gap.

    `query_index` defaults to a uniform draw over :func:`feasible_query_indices`. `lead_filler_max`
    adds filler *before* the queried pair, which cannot change the gap and only blurs absolute position.
    """
    check_cell(pairs, gap, overwrite)
    rng = random.Random(seed)

    choices = feasible_query_indices(pairs, gap, overwrite)
    if query_index is None:
        query_index = rng.choice(choices)
    elif query_index not in choices:
        raise WorldError(
            "query index %d cannot reach gap %d with %d pairs; feasible: %s"
            % (query_index, gap, pairs, list(choices))
        )

    keys = rng.sample(KEY_BYTES, pairs)
    values = [rng.choice(VALUE_BYTES) for _ in range(pairs)]
    if overwrite:
        stale = rng.randrange(query_index)
        keys[stale] = keys[query_index]
        while values[stale] == values[query_index]:
            values[stale] = rng.choice(VALUE_BYTES)

    # Filler after the queried pair is forced by the gap; filler before it is free.
    post = gap - min_gap(pairs, query_index)
    after = _compose(post, pairs - query_index, rng)          # before each later pair, then before '?'
    before = _compose(rng.randint(0, lead_filler_max), query_index + 1, rng)
    slots = before + after

    out = bytearray()
    equals_at = -1
    for index in range(pairs):
        out += _filler(slots[index], rng)
        if index == query_index:
            equals_at = len(out) + 1
        out += bytes((keys[index], EQUALS, values[index], SEMI))
    out += _filler(slots[pairs], rng)

    query_at = len(out)
    answer = values[query_index]
    out += bytes((QUERY, keys[query_index], answer, NEWLINE))
    answer_index = query_at + 2

    exact = exact_gap(query_at, equals_at)
    if exact != gap:
        raise WorldError("built gap %d, wanted %d" % (exact, gap))    # pragma: no cover - guard
    return Episode(
        data=bytes(out),
        pairs=tuple(zip(keys, values)),
        query_index=query_index,
        query_key=keys[query_index],
        answer=answer,
        answer_index=answer_index,
        gap=gap,
        overwrite=overwrite,
        seed=seed,
        feasible_positions=choices,
    )


def episodes(
    count: int,
    pairs: int,
    gap: int,
    seed: int = EVAL_SEED,
    overwrite: bool = False,
    lead_filler_max: int = 0,
) -> List[Episode]:
    """`count` fresh episodes. Same arguments, same episodes: that is what makes arms paired."""
    check_cell(pairs, gap, overwrite)
    base = derive_seed(seed, pairs, gap, int(overwrite), lead_filler_max)
    return [
        sample_episode(derive_seed(base, i), pairs, gap, overwrite, lead_filler_max)
        for i in range(count)
    ]


# --- the parser: the answer, recovered from bytes alone -----------------------------------------
@dataclass(frozen=True)
class Parsed:
    """What a reader with no access to the generator can recover from an episode's bytes."""

    pairs: Tuple[Tuple[int, int], ...]
    pair_offsets: Tuple[int, ...]       # index of each pair's key byte
    query_index: int
    query_key: int
    answer: int
    answer_index: int
    gap: int
    filler_bytes: int
    duplicate_keys: Tuple[int, ...]


def _scan_body(body: bytes) -> Tuple[List[Tuple[int, int]], List[int], int]:
    """Left to right: filler is skipped, anything else must be a whole `k=v;` pair."""
    pairs: List[Tuple[int, int]] = []
    offsets: List[int] = []
    filler = 0
    i = 0
    while i < len(body):
        byte = body[i]
        if byte in FILLER_SET:
            filler += 1
            i += 1
            continue
        if byte not in KEY_SET:
            raise WorldError("byte %r at %d is neither filler nor a key" % (bytes((byte,)), i))
        if i + PAIR_LEN > len(body):
            raise WorldError("truncated pair at %d" % i)
        if body[i + 1] != EQUALS:
            raise WorldError("expected '=' at %d" % (i + 1))
        if body[i + 2] not in VALUE_SET:
            raise WorldError("byte %r at %d is not a value" % (bytes((body[i + 2],)), i + 2))
        if body[i + 3] != SEMI:
            raise WorldError("expected ';' at %d" % (i + 3))
        offsets.append(i)
        pairs.append((byte, body[i + 2]))
        i += PAIR_LEN
    return pairs, offsets, filler


def parse_episode(data: bytes, expect_overwrite: Optional[bool] = None) -> Parsed:
    """Rebuild the whole episode from its bytes, or raise. This is the validator the gate leans on."""
    if not data.endswith(b"\n"):
        raise WorldError("episode does not end with a newline")
    if data.count(NEWLINE) != 1:
        raise WorldError("episode has %d newlines, want 1" % data.count(NEWLINE))
    if data.count(QUERY) != 1:
        raise WorldError("episode has %d '?' bytes, want 1" % data.count(QUERY))

    query_at = data.index(QUERY)
    tail = data[query_at:-1]
    if len(tail) != TAIL_LEN:
        raise WorldError("want '?' key answer before the newline, got %r" % tail)
    query_key, answer = tail[1], tail[2]
    if query_key not in KEY_SET:
        raise WorldError("queried byte %r is not a key" % bytes((query_key,)))
    if answer not in VALUE_SET:
        raise WorldError("answer byte %r is not a value" % bytes((answer,)))

    pairs, offsets, filler = _scan_body(data[:query_at])
    if not pairs:
        raise WorldError("no pairs before the query")

    seen = Counter(key for key, _ in pairs)
    duplicates = tuple(sorted(key for key, n in seen.items() if n > 1))
    if expect_overwrite is True and duplicates != (query_key,):
        raise WorldError("overwrite episode must repeat exactly the queried key, got %r" % (duplicates,))
    if expect_overwrite is False and duplicates:
        raise WorldError("keys are not distinct: %r" % (duplicates,))

    bound = [i for i, (key, _) in enumerate(pairs) if key == query_key]
    if not bound:
        raise WorldError("queried key %r was never written" % bytes((query_key,)))
    query_index = bound[-1]                       # newest write wins, which is the overwrite rule
    if pairs[query_index][1] != answer:
        raise WorldError("answer byte does not match the newest value written to the queried key")

    gap = exact_gap(query_at, offsets[query_index] + 1)
    return Parsed(
        pairs=tuple(pairs),
        pair_offsets=tuple(offsets),
        query_index=query_index,
        query_key=query_key,
        answer=answer,
        answer_index=query_at + 2,
        gap=gap,
        filler_bytes=filler,
        duplicate_keys=duplicates,
    )


def answer_from_prefix(prefix: bytes) -> int:
    """The answer, from only the bytes a model has actually read: everything up to `?k`.

    This is the mechanical determinability claim. If this disagrees with the generator, the episode
    is ambiguous and the world is broken.
    """
    if len(prefix) < TAIL_LEN - 1 or prefix[-2] != QUERY:
        raise WorldError("prefix must end with '?' and the queried key")
    query_key = prefix[-1]
    if query_key not in KEY_SET:
        raise WorldError("queried byte %r is not a key" % bytes((query_key,)))
    pairs, _, _ = _scan_body(prefix[:-2])
    bound = [value for key, value in pairs if key == query_key]
    if not bound:
        raise WorldError("queried key %r was never written" % bytes((query_key,)))
    return bound[-1]


def validate(episode: Episode) -> Parsed:
    """Every claim the generator makes, re-derived from bytes and checked."""
    parsed = parse_episode(episode.data, expect_overwrite=episode.overwrite)
    if parsed.answer != episode.answer:
        raise WorldError("parsed answer disagrees with the generator")
    if parsed.answer_index != episode.answer_index:
        raise WorldError("parsed answer index disagrees with the generator")
    if parsed.gap != episode.gap:
        raise WorldError("parsed gap %d, generated gap %d" % (parsed.gap, episode.gap))
    if parsed.query_index != episode.query_index:
        raise WorldError("parsed query index disagrees with the generator")
    if len(parsed.pairs) != episode.pair_count:
        raise WorldError("parsed %d pairs, generated %d" % (len(parsed.pairs), episode.pair_count))
    if parsed.pairs != episode.pairs:
        raise WorldError("parsed pairs disagree with the generator")
    if answer_from_prefix(episode.data[: episode.answer_index]) != episode.answer:
        raise WorldError("the answer is not determined by the bytes that precede it")
    return parsed


# --- scoring ------------------------------------------------------------------------------------
def answer_mask(episode: Episode) -> torch.Tensor:
    """Next-byte scoring mask over `data[1:]`: True only where the answer byte is predicted.

    A byte model reads `data[:-1]` and predicts `data[1:]`, so the answer sits one slot earlier.
    """
    mask = torch.zeros(len(episode.data) - 1, dtype=torch.bool)
    mask[episode.answer_index - 1] = True
    return mask


def batch_tensors(batch: Sequence[Episode], pad: int = 0):
    """Ragged episodes padded into `(inputs, targets, score_mask, lengths)`.

    `score_mask` is True at exactly one position per row; padded slots are never scored.
    """
    width = max(len(e.data) for e in batch) - 1
    inputs = torch.full((len(batch), width), pad, dtype=torch.long)
    targets = torch.full((len(batch), width), pad, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.bool)
    lengths = torch.tensor([len(e.data) - 1 for e in batch], dtype=torch.long)
    for row, episode in enumerate(batch):
        span = len(episode.data) - 1
        stream = torch.tensor(list(episode.data), dtype=torch.long)
        inputs[row, :span] = stream[:-1]
        targets[row, :span] = stream[1:]
        mask[row, episode.answer_index - 1] = True
    return inputs, targets, mask, lengths


# --- lazy guesses -------------------------------------------------------------------------------
def _most_common(parsed: Parsed) -> int:
    """Most common value in the episode; ties go to the one written earliest."""
    order = {}
    for i, (_, value) in enumerate(parsed.pairs):
        order.setdefault(value, i)
    counts = Counter(value for _, value in parsed.pairs)
    return max(counts, key=lambda v: (counts[v], -order[v]))


def lazy_guessers(pairs: int) -> Dict[str, Callable[[Parsed, random.Random], int]]:
    """Every lazy guess, as a function of the parsed bytes only. No privileged access.

    `position_i` is the value written at pair ordinal `i`, which catches a model that learned
    position instead of binding. `position_0` and `position_{n-1}` coincide with `first` and `last`
    by construction; both names are reported so the table reads straight.
    """
    guessers: Dict[str, Callable[[Parsed, random.Random], int]] = {
        "uniform": lambda p, rng: rng.choice(VALUE_BYTES),
        "most_common": lambda p, rng: _most_common(p),
        "first": lambda p, rng: p.pairs[0][1],
        "last": lambda p, rng: p.pairs[-1][1],
    }
    for index in range(pairs):
        guessers["position_%d" % index] = (
            lambda p, rng, i=index: p.pairs[i][1] if i < len(p.pairs) else p.pairs[-1][1]
        )
    return guessers


def lazy_rates(
    pairs: int,
    gap: int,
    trials: int = EVAL_TRIALS,
    seed: int = EVAL_SEED,
    overwrite: bool = False,
    lead_filler_max: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Every lazy guess scored on one identical set of episodes, so the arms are paired."""
    batch = episodes(trials, pairs, gap, seed, overwrite, lead_filler_max)
    guessers = lazy_guessers(pairs)
    hits = {name: 0 for name in guessers}
    for episode in batch:
        parsed = parse_episode(episode.data, expect_overwrite=overwrite)
        rng = random.Random(derive_seed(episode.seed, "lazy"))
        for name, guess in guessers.items():
            hits[name] += int(guess(parsed, rng) == parsed.answer)
    out = {}
    for name, hit in hits.items():
        low, high = wilson(hit, trials)
        out[name] = {"hits": hit, "trials": trials, "rate": hit / trials, "low": low, "high": high}
    return out


def position_leakage(
    pairs: int,
    gap: int,
    trials: int = EVAL_TRIALS,
    seed: int = EVAL_SEED,
    overwrite: bool = False,
) -> Dict[str, object]:
    """How much accuracy a position-only guesser gets for free in this cell.

    At `pairs=32, gap=8` only ordinals 30 and 31 can be queried, so guessing "the last pair" is
    right about half the time before any binding happens. That is a property of the arithmetic, not
    of a model, so it is reported next to every result rather than argued away.
    """
    batch = episodes(trials, pairs, gap, seed, overwrite)
    counts = Counter(e.query_index for e in batch)
    feasible = feasible_query_indices(pairs, gap, overwrite)
    best = max(counts.values()) / trials if counts else 0.0
    return {
        "pairs": pairs,
        "gap": gap,
        "feasible_positions": feasible,
        "feasible_count": len(feasible),
        "all_positions": len(feasible) == pairs,
        "position_counts": dict(sorted(counts.items())),
        "best_fixed_position_rate": best,
        "chance": 1.0 / len(VALUE_BYTES),
    }


def episode_set_hash(batch: Sequence[Episode]) -> str:
    """A stable fingerprint of an ordered episode set: the proof that two arms were really paired."""
    digest = hashlib.blake2b(digest_size=16)
    for episode in batch:
        digest.update(len(episode.data).to_bytes(4, "big"))
        digest.update(episode.data)
    return digest.hexdigest()


# --- statistics ---------------------------------------------------------------------------------
def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval, the 95% default. A point estimate is not a gate."""
    if trials <= 0:
        return 0.0, 1.0
    if successes < 0 or successes > trials:
        raise ValueError("successes %d out of %d trials" % (successes, trials))
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == trials else min(1.0, centre + half)
    return low, high


# --- the evaluation protocol --------------------------------------------------------------------
RESET = "reset"                        # fresh state every episode: the primary protocol
CARRY = "carry"                        # state carried across episodes: interference, report only
RESET_AT_RANDOM = "reset-at-random"    # the control that separates interference from difficulty


@dataclass(frozen=True)
class StateProtocol:
    """When the recurrent state is cleared during an evaluation. No model, no training: just the rule."""

    name: str
    period: int = 1                    # mean episodes between resets, for RESET_AT_RANDOM

    def resets(self, count: int, seed: int) -> List[bool]:
        rng = random.Random(derive_seed(seed, self.name, self.period, "resets"))
        if self.name == RESET:
            return [True] * count
        if self.name == CARRY:
            return [i == 0 for i in range(count)]
        if self.name == RESET_AT_RANDOM:
            # Same expected reset rate as a period-`period` carry run, but at unpredictable places.
            return [i == 0 or rng.random() < 1.0 / self.period for i in range(count)]
        raise ValueError("unknown state protocol %r" % self.name)


def state_protocol(name: str, period: int = 1) -> StateProtocol:
    if name not in (RESET, CARRY, RESET_AT_RANDOM):
        raise ValueError("unknown state protocol %r" % name)
    if name == RESET_AT_RANDOM and period < 1:
        raise ValueError("period must be at least 1")
    return StateProtocol(name, period)


@dataclass(frozen=True)
class EvalPlan:
    """One (pairs, gap) cell of the paired evaluation. Learning is frozen unless it is the subject."""

    pairs: int
    gap: int
    trials: int = EVAL_TRIALS
    seed: int = EVAL_SEED
    overwrite: bool = False
    lead_filler_max: int = 0
    protocol: StateProtocol = field(default_factory=lambda: StateProtocol(RESET))
    frozen: bool = True

    @property
    def cell(self) -> Tuple[int, int]:
        return self.pairs, self.gap

    def episodes(self) -> List[Episode]:
        return episodes(self.trials, self.pairs, self.gap, self.seed, self.overwrite, self.lead_filler_max)

    def feasible_positions(self) -> Tuple[int, ...]:
        return feasible_query_indices(self.pairs, self.gap, self.overwrite)

    def episode_set_hash(self) -> str:
        """Same plan, same hash, every call. Different cell, different hash."""
        return episode_set_hash(self.episodes())


def evaluation_plans(
    trials: int = EVAL_TRIALS,
    seed: int = EVAL_SEED,
    overwrite: bool = False,
    protocol: Optional[StateProtocol] = None,
) -> List[EvalPlan]:
    """Every requested cell: 5 pair counts x 4 gaps, 1,000 paired fresh episodes each."""
    protocol = protocol or StateProtocol(RESET)
    return [
        EvalPlan(pairs=p, gap=g, trials=trials, seed=seed, overwrite=overwrite, protocol=protocol)
        for p in PAIR_COUNTS
        for g in GAPS
    ]


def parameter_fingerprint(module) -> str:
    """Hash of every parameter, so 'learning was frozen' is asserted rather than assumed."""
    digest = hashlib.blake2b(digest_size=16)
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class Outcome:
    """The result of one arm on one cell."""

    hits: int = 0
    trials: int = 0
    nll_total: float = 0.0
    rank_total: int = 0
    scored: int = 0
    by_since_reset: Dict[int, List[int]] = field(default_factory=dict)

    def record(self, correct: bool, since_reset: int, nll: Optional[float], rank: Optional[int]) -> None:
        self.hits += int(correct)
        self.trials += 1
        bucket = self.by_since_reset.setdefault(since_reset, [0, 0])
        bucket[0] += int(correct)
        bucket[1] += 1
        if nll is not None:
            self.nll_total += nll
            self.rank_total += rank
            self.scored += 1

    def summary(self, protocol: Optional["StateProtocol"] = None) -> Dict[str, object]:
        """Accuracy always; a Wilson interval only where the episodes are actually independent.

        Under CARRY and RESET_AT_RANDOM the state survives between episodes, so one episode's
        outcome depends on the ones before it. A binomial interval would understate the true spread,
        so it is omitted with a reason rather than printed and quietly trusted.
        """
        name = protocol.name if protocol is not None else RESET
        independent = name == RESET
        out = {
            "protocol": name,
            "hits": self.hits,
            "trials": self.trials,
            "accuracy": self.hits / self.trials if self.trials else 0.0,
            "wilson_low": None,
            "wilson_high": None,
            "interval_omitted_because": None,
            "accuracy_by_episodes_since_reset": {
                k: v[0] / v[1] for k, v in sorted(self.by_since_reset.items())
            },
        }
        if independent:
            out["wilson_low"], out["wilson_high"] = wilson(self.hits, self.trials)
        else:
            out["interval_omitted_because"] = (
                "state is carried across episodes under %s, so outcomes are not independent "
                "and a binomial Wilson interval would be too narrow" % name
            )
        if self.scored:
            out["nll"] = self.nll_total / self.scored
            out["rank"] = self.rank_total / self.scored
        return out


def _score(prediction, answer: int) -> Tuple[bool, Optional[float], Optional[int]]:
    """A prediction is a byte, or 256 log-probabilities and then NLL and rank come free."""
    if isinstance(prediction, int):
        return prediction == answer, None, None
    logprobs = torch.as_tensor(prediction, dtype=torch.float64).reshape(-1)
    if logprobs.numel() != 256:
        raise ValueError("expected 256 log-probabilities, got %d" % logprobs.numel())
    best = int(torch.argmax(logprobs))
    rank = 1 + int((logprobs > logprobs[answer]).sum())
    return best == answer, -float(logprobs[answer]), rank


def run_plan(
    plan: EvalPlan,
    arms: Dict[str, Callable],
    parameter_modules: Optional[Dict[str, torch.nn.Module]] = None,
) -> Dict[str, Dict[str, object]]:
    """Run every arm over the *same* episodes in the same order: paired, not merely equal-sized.

    Each arm is `f(episode, reset) -> int | 256 log-probabilities`. `reset` comes from the plan's
    state protocol, so frozen/reset, carry and reset-at-random all run through one path.
    """
    parameter_modules = parameter_modules or {}
    unknown = set(parameter_modules) - set(arms)
    if unknown:
        raise ValueError("parameter modules have no matching arm: %s" % sorted(unknown))
    before = (
        {name: parameter_fingerprint(module) for name, module in parameter_modules.items()}
        if plan.frozen
        else {}
    )

    batch = plan.episodes()
    resets = plan.protocol.resets(len(batch), plan.seed)
    outcomes = {name: Outcome() for name in arms}
    since = 0
    for episode, reset in zip(batch, resets):
        since = 0 if reset else since + 1
        for name, arm in arms.items():
            correct, nll, rank = _score(arm(episode, reset), episode.answer)
            outcomes[name].record(correct, since, nll, rank)
    if plan.frozen:
        changed = [
            name
            for name, module in parameter_modules.items()
            if parameter_fingerprint(module) != before[name]
        ]
        if changed:
            raise RuntimeError("frozen evaluation mutated parameters for arms: %s" % sorted(changed))

    fingerprint = episode_set_hash(batch)
    summaries = {}
    for name, outcome in outcomes.items():
        row = outcome.summary(plan.protocol)
        row["episode_set_hash"] = fingerprint          # identical across arms: they were paired
        row["feasible_positions"] = plan.feasible_positions()
        summaries[name] = row
    return summaries


def run_grid(
    arms: Dict[str, Callable],
    plans: Optional[Sequence[EvalPlan]] = None,
    parameter_modules: Optional[Dict[str, torch.nn.Module]] = None,
):
    """Every cell of the grid, each with its own paired episode set."""
    plans = plans or evaluation_plans()
    return {plan.cell: run_plan(plan, arms, parameter_modules) for plan in plans}


def lazy_arms(pairs: int) -> Dict[str, Callable]:
    """The lazy guesses, wrapped as evaluation arms so they run on identical episodes to a model."""
    guessers = lazy_guessers(pairs)

    def wrap(guess):
        def arm(episode: Episode, reset: bool) -> int:
            parsed = parse_episode(episode.data, expect_overwrite=episode.overwrite)
            return guess(parsed, random.Random(derive_seed(episode.seed, "lazy")))
        return arm

    return {name: wrap(guess) for name, guess in guessers.items()}


if __name__ == "__main__":
    # Self-check: every cell constructible, every gap exact, every answer readable from the bytes.
    total = 0
    for pairs in PAIR_COUNTS:
        for gap in GAPS:
            for episode in episodes(200, pairs, gap, seed=11):
                validate(episode)
                total += 1
            for episode in episodes(50, pairs, gap, seed=11, overwrite=True):
                validate(episode)
                total += 1
    print(f"world self-check passed on {total} episodes")
    print(f"alphabet: {len(KEY_BYTES)} keys, {len(VALUE_BYTES)} values, {len(FILLER_BYTES)} filler bytes")
    print(f"filler: {FILLER!r}")

    example = sample_episode(derive_seed(3), pairs=4, gap=32)
    print(f"example (pairs=4, gap=32, queried ordinal {example.query_index}):")
    print("  " + example.text().replace("\n", "\\n"))

    print("\nlazy-guess rates, 1,000 episodes per cell:")
    header = ["uniform", "most_common", "first", "last"]
    print("  pairs  gap  " + "  ".join(f"{name:>11}" for name in header))
    for pairs in PAIR_COUNTS:
        for gap in GAPS:
            rates = lazy_rates(pairs, gap, trials=EVAL_TRIALS, seed=EVAL_SEED)
            row = "  ".join(f"{rates[name]['rate']:>11.3f}" for name in header)
            print(f"  {pairs:>5}  {gap:>3}  {row}")
