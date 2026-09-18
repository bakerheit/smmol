"""Worked solutions for smMATH001-a: an expression's arithmetic written out one small step at a time.

Numbers are fixed-point, scaled by 100 and written ones digit first: 12.5 is 1250, written 0521. Every digit
then lines up by place, decimals are just digits, and carries run the same way the model writes.

    (12+8)/5    question   (0021+008)/005
                work       0021+008=0002;0002/005:005*004=000002;000002-000002=0;=004
                answer     004, which is 400, which is 4

The steps come in the order the expression is worked out (brackets, then * and /, then + and -):

    A+B=C, A-B=C   C gets a trailing "-" when it's below zero, which only the last step may be
    A*B:           one line per nonzero digit of B, A*D=P (D is that digit at its place), a running sum
                   S+P=S' from the second digit on, then S>T, which drops the two extra decimal places
    A/B:           long division of A*100 by B, biggest place first: B*Q=P and R-P=R' for each nonzero
                   quotient digit Q at its place, then =Q with all of them together (cut to 2 places)
"""
import ast
import random
import re
from decimal import Decimal

SCALE = 100
OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}
SIGNS = {"+": "+", "-": "−", "*": "×", "/": "÷"}
LIMITS = {"digits": 9, "multiplier_digits": 3, "divisor_digits": 6}  # sizes a training example may have (scaled)
MAX_WORK, MAX_QUESTION = 420, 64


class TooBig(ValueError):
    """Arithmetic it could write out, but bigger than a training example may be."""


def rev(n):
    return str(n)[::-1]


def scaled(text):
    """'12.5' -> 1250."""
    whole, _, frac = str(text).partition(".")
    if not (whole.isdigit() or (whole == "" and frac)) or not (frac.isdigit() or frac == ""):
        raise ValueError("%r isn't a plain number" % text)
    if len(frac) > 2:
        raise ValueError("%s has more than 2 decimal places" % text)
    return int(whole or "0") * SCALE + int((frac + "00")[:2])


def unscaled(n, power=2):
    """1250 -> '12.5', 400 -> '4', and -300 -> '-3'. power is how many decimal places n carries."""
    return format(Decimal(n).scaleb(-power).normalize(), "f")


def parse(expression):
    """A tree of (op, left, right) with numbers as text. Only numbers with up to 2 decimal places, + - * / and brackets."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float) and node.value >= 0:
            text = repr(node.value)
            scaled(text)  # refuses 1e-05 style numbers and more than 2 decimal places
            return text
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)], walk(node.left), walk(node.right)
        raise ValueError("only numbers, + - * / and brackets")
    try:
        return walk(ast.parse(str(expression).strip(), mode="eval"))
    except SyntaxError as exc:
        raise ValueError("can't read %r (%s)" % (expression, exc))


def render(tree, number=lambda s: s, parent=None, right=False):
    """The tree as text with only the brackets it needs."""
    if isinstance(tree, str):
        return number(tree)
    op, a, b = tree
    text = render(a, number, op) + op + render(b, number, op, True)
    if parent and (PRECEDENCE[parent] > PRECEDENCE[op] or (right and PRECEDENCE[parent] == PRECEDENCE[op])):
        return "(" + text + ")"
    return text


def question(tree):
    """What the model reads: the expression with every number scaled and written ones digit first."""
    return render(tree, lambda s: rev(scaled(s)))


def nonzero(n):
    return sum(ch != "0" for ch in str(n))


def multiply(x, y, limits=None):
    a, b = (x, y) if nonzero(y) <= nonzero(x) else (y, x)  # the number with fewer nonzero digits is the multiplier
    if limits and (nonzero(b) > limits["multiplier_digits"] or len(str(a)) > limits["digits"]):
        raise TooBig("multiplication too big")
    steps, total = [], None
    for place, digit in enumerate(reversed(str(b))):
        if digit == "0":
            continue
        part = a * int(digit) * 10 ** place
        steps.append("%s*%s=%s" % (rev(a), rev(int(digit) * 10 ** place), rev(part)))
        if total is not None:
            steps.append("%s+%s=%s" % (rev(total), rev(part), rev(total + part)))
        total = part if total is None else total + part
    if total is None:
        steps.append("%s*0=0" % rev(a))
        total = 0
    steps.append("%s>%s" % (rev(total), rev(total // SCALE)))
    return "%s*%s:%s" % (rev(a), rev(b), ";".join(steps)), total // SCALE


def divide(x, y, limits=None):
    if y == 0:
        raise ValueError("division by zero")
    if limits and (len(str(y)) > limits["divisor_digits"] or len(str(x)) > limits["digits"]):
        raise TooBig("division too big")
    r, quotient, steps, top = x * SCALE, 0, [], 0
    while y * 10 ** (top + 1) <= r:
        top += 1
    for place in range(top, -1, -1):
        q = r // (y * 10 ** place)
        if q:
            part = y * q * 10 ** place
            steps.append("%s*%s=%s" % (rev(y), rev(q * 10 ** place), rev(part)))
            steps.append("%s-%s=%s" % (rev(r), rev(part), rev(r - part)))
            r -= part
            quotient += q * 10 ** place
    steps.append("=%s" % rev(quotient))
    return "%s/%s:%s" % (rev(x), rev(y), ";".join(steps)), quotient


def solve_tree(tree, limits=None):
    """(work, scaled answer). With limits, sizes past a training example's raise TooBig."""
    items = []

    def ev(t, last):
        if isinstance(t, str):
            return scaled(t)
        op, a, b = t
        x, y = ev(a, False), ev(b, False)
        if op in "+-":
            z = x + y if op == "+" else x - y
            if z < 0 and not last:
                raise ValueError("it goes below zero partway through")
            items.append("%s%s%s=%s%s" % (rev(x), op, rev(y), rev(abs(z)), "-" if z < 0 else ""))
            return z
        block, z = (multiply if op == "*" else divide)(x, y, limits)
        items.append(block)
        return z

    answer = ev(tree, True)
    return ";".join(items), answer


def final_answer(work):
    """The scaled answer a worked solution ends on, or None when it doesn't end on a number."""
    m = re.search(r"[=>](\d+)(-?)$", work.split(";")[-1] if work else "")
    if not m:
        return None
    n = int(m.group(1)[::-1])
    return -n if m.group(2) else n


def readable(work):
    """The work as lines a person can follow: usual digit order, decimals back in place."""
    def num(s, power=2):
        negative = s.endswith("-")
        n = int(s.rstrip("-")[::-1])
        return unscaled(-n if negative else n, power)

    lines, block = [], None
    for part in work.split(";") if work else []:
        try:
            if ":" in part:
                header, part = part.split(":", 1)
                op = next(ch for ch in "*/" if ch in header)
                a, b = header.split(op)
                block = op
                lines.append("%s %s %s:" % (num(a), SIGNS[op], num(b)))
            if block == "*":
                if ">" in part:
                    lines.append("  = %s" % num(part.split(">")[1]))
                    block = None
                else:
                    op = "*" if "*" in part else "+"
                    left, result = part.split("=")
                    a, b = left.split(op)
                    lines.append("  %s %s %s = %s" % ((num(a), SIGNS[op], num(b), num(result, 4)) if op == "*" else
                                                      (num(a, 4), SIGNS[op], num(b, 4), num(result, 4))))
            elif block == "/":
                if part.startswith("="):
                    lines.append("  = %s" % num(part[1:]))
                    block = None
                else:
                    op = "*" if "*" in part else "-"
                    left, result = part.split("=")
                    a, b = left.split(op)
                    lines.append("  %s %s %s = %s" % ((num(a), SIGNS[op], num(b), num(result, 4)) if op == "*" else
                                                      (num(a, 4), SIGNS[op], num(b, 4), num(result, 4))))
            else:
                op = next(ch for ch in "+-" if ch in part.split("=")[0])
                left, result = part.split("=")
                a, b = left.split(op, 1)
                lines.append("%s %s %s = %s" % (num(a), SIGNS[op], num(b), num(result)))
        except (ValueError, StopIteration, IndexError):
            lines.append(part)  # a line the model got garbled stays as it was written
    return lines


# ----- training examples -----

def number(r):
    if r.random() < 0.3:
        whole, places = (r.randint(0, 99) if r.random() < 0.4 else r.randint(0, 9999)), r.choice([1, 2])
        return "%d.%0*d" % (whole, places, r.randint(1, 10 ** places - 1))
    digits = r.choice([1, 1, 2, 2, 3, 3, 4, 5, 6])
    return str(r.randint(0 if digits == 1 else 10 ** (digits - 1), 10 ** digits - 1))


def random_tree(r, ops):
    if ops == 0:
        return number(r)
    left = r.randint(0, ops - 1)
    return r.choices("+-*/", weights=[3, 2.5, 3, 1.5])[0], random_tree(r, left), random_tree(r, ops - 1 - left)


def harness_like(r):
    """The shapes Math Language writes: volumes, percents, tax, splits, averages, ticket totals, rates, dozens."""
    def n(lo, hi):
        digits = r.randint(lo, hi)
        return str(r.randint(0 if digits == 1 else 10 ** (digits - 1), 10 ** digits - 1))

    def money():
        return "%d.%02d" % (r.randint(1, 999), r.choice([0, 25, 50, 75, 99, r.randint(1, 99)]))

    pct = r.choice(["5", "7.25", "8", "10", "12.5", "15", "17.5", "18", "20", "25", "30", "33", "40", "50"])
    k = str(r.randint(2, 12))
    m = money()
    kind = r.randrange(10)
    if kind == 0:
        return "*", ("*", n(1, 3), n(1, 3)), n(1, 3)
    if kind == 1:
        return "/", ("*", n(1, 5), pct), "100"
    if kind == 2:
        return "/", ("*", m, pct), "100"
    if kind == 3:
        return "+", m, ("/", ("*", m, pct), "100")
    if kind == 4:
        return "-", m, ("/", ("*", m, pct), "100")
    if kind == 5:
        return "/", m, k
    if kind == 6:
        return "/", ("+", ("+", n(1, 3), n(1, 3)), n(1, 3)), "3"
    if kind == 7:
        return "+", ("*", k, m), "%d.%02d" % (r.randint(1, 30), r.choice([0, 50, 99]))
    if kind == 8:
        return "/", n(2, 4), n(1, 2)
    return ("*", ("*", k, "12"), "0.%02d" % r.randint(5, 99)) if r.random() < 0.5 else ("/", ("*", k, "12"), "2")


def kind_of(tree):
    ops, decimals = set(), False

    def walk(t):
        nonlocal decimals
        if isinstance(t, str):
            decimals = decimals or "." in t
            return 0
        ops.add(t[0])
        return 1 + walk(t[1]) + walk(t[2])

    count = walk(tree)
    name = "mixed" if count > 1 else {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}[ops.pop()]
    return name, decimals


def example(r):
    while True:
        harness = r.random() < 0.3
        tree = harness_like(r) if harness else random_tree(r, r.choices([1, 2, 3], weights=[55, 30, 15])[0])
        try:
            work, answer = solve_tree(tree, LIMITS)
        except ValueError:
            continue
        q = question(tree)
        if len(work) > MAX_WORK or len(q) > MAX_QUESTION:
            continue
        kind, decimals = kind_of(tree)
        return {"expression": render(tree), "question": q, "work": work, "answer": answer, "kind": kind,
                "decimals": decimals, "harness": harness}


def generate(n, seed=0):
    r = random.Random(seed)
    return [example(r) for _ in range(n)]


if __name__ == "__main__":
    for ex in generate(8, 3):
        print(ex["expression"], "=", unscaled(ex["answer"]))
        print("   ", ex["question"], "->", ex["work"])
        for line in readable(ex["work"]):
            print("      " + line)
