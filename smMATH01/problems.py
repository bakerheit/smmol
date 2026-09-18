"""smMATH01's problems: whole-number +, - and *, written out as text and made fresh for every batch.

    48213+9977=58190$     plain: digits in the usual order
    31284+7799=09185$     reversed: every number starts at its ones digit, the order you carry in

There's no dataset to memorize. Training numbers have 1 to 6 digits; the tests go up to 10.
"""
import torch

VOCAB = "0123456789+-*=$"   # "$" ends the answer
PAD = len(VOCAB)
VOCAB_SIZE = PAD + 1
EQUALS, END = VOCAB.index("="), VOCAB.index("$")
OPS = "+-*"
OP_WEIGHTS = {"+": 0.4, "-": 0.3, "*": 0.3}
TRAIN_DIGITS = 6


def number(digits, rng):
    """A random whole number with exactly this many digits."""
    return rng.randint(0, 9) if digits == 1 else rng.randint(10 ** (digits - 1), 10 ** digits - 1)


def solve(a, op, b):
    return {"+": a + b, "-": a - b, "*": a * b}[op]


def problem(a_digits, op, b_digits, rng):
    a, b = number(a_digits, rng), number(b_digits, rng)
    if op == "-" and b > a:  # answers are never negative, so there's no minus sign to learn
        a, b = b, a
    return a, op, b


def line(a, op, b, reverse):
    """The whole line: question, answer and the end mark."""
    order = (lambda s: s[::-1]) if reverse else (lambda s: s)
    return "%s%s%s=%s$" % (order(str(a)), op, order(str(b)), order(str(solve(a, op, b))))


def question(a, op, b, reverse):
    return line(a, op, b, reverse).split("=")[0] + "="


def expected(a, op, b, reverse):
    """The answer tokens a model should write, "$" included."""
    return line(a, op, b, reverse).split("=")[1]


def readable(answer, reverse):
    """A written answer turned back into the usual digit order, for people."""
    digits = answer.rstrip("$")
    return digits[::-1] if reverse else digits


def encode(text):
    return [VOCAB.index(ch) for ch in text]


def decode(ids):
    return "".join(VOCAB[i] for i in ids if i < PAD)


def places(ids):
    """Where each digit sits inside its own number, counting from 1 (0 for anything that isn't a digit).

    The abacus variant uses this instead of the position in the line, so the third digit of a question
    number and the third digit of the answer get the same label, however long the numbers are.
    """
    is_digit = ids < 10
    at = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    last_break = torch.where(is_digit, torch.full_like(at, -1), at).cummax(dim=1).values
    return torch.where(is_digit, at - last_break, torch.zeros_like(at))


def batch(size, rng, reverse, max_digits=TRAIN_DIGITS, ops=OPS, offset_max=0):
    """(inputs, targets, places) for `size` fresh problems. Only answer tokens are scored; the rest are -100.

    With offset_max, every number in a line has its places shifted by the same random amount, so the
    labels for places past the longest training number get trained too.
    """
    rows = []
    for _ in range(size):
        op = rng.choices(ops, weights=[OP_WEIGHTS[o] for o in ops])[0]
        rows.append(encode(line(*problem(rng.randint(1, max_digits), op, rng.randint(1, max_digits), rng), reverse)))
    ids = torch.full((size, max(map(len, rows))), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
    x, y = ids[:, :-1], ids[:, 1:].clone()
    after_equals = (x == EQUALS).long().cumsum(1) > 0
    y[~after_equals | (y == PAD)] = -100
    where = places(x)
    if offset_max:
        shift = torch.tensor([[rng.randint(0, offset_max)] for _ in range(size)])
        where = torch.where(where > 0, where + shift, where)
    return x, y, where
