"""Working out a list of problems, and scoring a reader against hand-written expectations.

A problem's expression uses numbers, + - * / ** ( ) and sqrt(), and p1, p2 ... for earlier answers in the
same message. Scoring compares the numbers the expressions work out to, so 80*15/100 and 15/100*80 are both
right.
"""
import ast
import math
import operator
import re

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
REF = re.compile(r"\bp(\d+)\b")


def evaluate(expression, answers=None):
    """The value of one expression. Raises ValueError if it isn't arithmetic or uses an answer that isn't there."""
    answers = answers or {}

    def ref(m):
        key = "p" + m.group(1)
        if key not in answers:
            raise ValueError("%s uses %s before it's worked out" % (expression, key))
        return "(%r)" % answers[key]

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 50:
                raise ValueError("that power is too big")
            return OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
            return math.sqrt(ev(node.args[0]))
        raise ValueError("%r isn't arithmetic" % expression)

    try:
        return ev(ast.parse(REF.sub(ref, expression).strip(), mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise ValueError("%r: %s" % (expression, exc))


def values(expressions):
    """Every problem's value, in order, so later problems can use earlier answers."""
    answers, out = {}, []
    for i, expression in enumerate(expressions, 1):
        out.append(evaluate(expression, answers))
        answers["p%d" % i] = out[-1]
    return out


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def grade(got, want):
    """(every problem right, final answer right). An empty want means it should find nothing."""
    if not want:
        return not got, not got
    try:
        got_values = values(got) if got else []
    except ValueError:
        return False, False
    want_values = values(want)
    every = len(got_values) == len(want_values) and all(close(a, b) for a, b in zip(got_values, want_values))
    return every, bool(got_values) and close(got_values[-1], want_values[-1])


def score(examples, predictions):
    """examples have "text" and "expected" (a list of expressions); predictions are lists of expressions."""
    rows = {"every": [], "final": [], "found": [], "quiet": []}
    mistakes = []
    for e, got in zip(examples, predictions):
        every, final = grade(got, e["expected"])
        rows["every"].append(every)
        rows["final"].append(final)
        (rows["found"] if e["expected"] else rows["quiet"]).append(bool(got) if e["expected"] else not got)
        if not every:
            mistakes.append({"text": e["text"], "want": e["expected"], "got": got})
    return {k: (round(sum(v) / len(v), 3) if v else None) for k, v in rows.items()}, mistakes
