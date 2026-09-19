#!/usr/bin/env python3
"""paratroop_harness_02: small cognitive modules with narrow contracts around a structured working memory.

One pass of thinking:

    perception -> attention -> recall -> planner -> predictor -> critic -> decision -> language
                                           ^                                  |      (answer, make, ask or tool)
                                           +------ tool result <- act <-------+   (when the decision is a tool)

Modules are small models that must answer in JSON that fits contracts.json. Working memory is
plain structured state, long-term memory is a document store, and math, the web and files are
tools, not models.

    python3 harness.py "What's 17.5% of 2,340?"
    python3 harness.py --continue "I'm the investor"        # answer in the same conversation
    ./m perception "I have a meeting this Friday" | ./m attention | ./m show
"""
import argparse
import ast
import http.client
import json
import math
import operator
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from decimal import Decimal
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("router", "perception", "attention", "recall", "planner", "predictor", "critic", "decision", "act", "language", "remember")
# smROUTER_01's tool names -> the harness switch each one depends on
ROUTER_TOOLS = {"calculator": "calculator", "web_search": "web_search", "web_browser": "web_browser",
                "file_read": "file_system", "file_write": "file_system", "memory_recall": "memory"}
FOCUS = 0.5         # attention score an observation needs to stay in focus
KEEP_MEMORY = 0.5   # recall score a memory needs to be used
RESULT_CHARS = 1500
HISTORY_TURNS = 6   # how many earlier turns of a conversation the modules see
RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
THINK = re.compile(r"<think>.*?</think>", re.S)
PLACEHOLDER = re.compile(r"\[[^\]]*\]|<[^>]*>")
# observations that only restate this request ("User specifies the language") aren't worth keeping in long-term memory
REQUEST_ECHO = re.compile(r"^user (wants|specifie[sd]|asks|asked|requests|requested|previously)\b|\bthe assistant\b", re.I)
FORCEABLE = ("calculator", "web_search", "web_browser")  # tools the router may force: lookups, never writes
LOCAL_CODE = {"classifier": "route.py:Router", "math": "solve.py:Solver",  # how each kind of model trained here is loaded
              "math_language": "read.py:Reader", "language": "inference.py:Generator"}
# arithmetic written inside a message: "what's 48213 + 9977?", "17.5% of 2,340", "12x4"
EXPRESSION = re.compile(r"\(*\d[\d,]*(?:\.\d+)?%?(?:\s*(?:[-+*/x×÷^]|%?\s*of)\s*\(*\s*\d[\d,]*(?:\.\d+)?%?\)*)+")
# units stuck to numbers ("322234ft", "12 kg", "$64.50"), which arithmetic in a message reads past
UNITS = re.compile(r"(?<=\d)\s*(?:sq\.?\s*ft|ft|feet|foot|inches|inch|yds?|yards?|cm|mm|km|m|miles?|mi|kg|g|lbs?|pounds?|oz|"
                   r"liters?|litres?|ml|gallons?|gal|hrs?|hours?|mins?|minutes?|secs?|seconds?|days?|dollars?|bucks|usd)\b\.?", re.I)
NUMBERS = re.compile(r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|half|twice|double|"
                     r"triple|hundred|thousand|million)\b", re.I)
# things no model can know from its weights: they change, or they're about right now. A short, deliberately
# narrow list, so a message that clearly needs the outside world gets a search even with no Planner running.
LOOKUP = re.compile(r"\b(weather|forecast|temperature outside|raining|snowing|humidity|wind chill|"
                    r"news|headlines?|"
                    r"stock price|share price|exchange rate|gas prices?|"
                    r"who won|final score|"
                    r"flight status|traffic (right now|on)|"
                    r"open (right )?now|opening hours|"
                    r"latest (version|release)|release date)\b", re.I)
# pages the text fetch can't read: they're drawn in the browser, so what comes back is menus and adverts
CLIENT_SIDE = ("weather.com", "accuweather.com", "tripadvisor.", "instagram.com", "facebook.com", "twitter.com", "x.com/")
# the one line the Language module writes when it wants a lookup instead of a reply: "NEED: web_search ..."
NEED = re.compile(r"^\s*NEED:\s*([a-z_]+)[\s:]+(.+?)\s*$", re.I | re.M)
# the knobs on each kind of module's card: (key, label, type, low, high, step, default)
SETTINGS = {
    None: [("temperature", "Temperature", "number", 0.0, 1.5, 0.1, 0.2), ("max_tokens", "Token limit", "int", 50, 4000, 50, 600)],
    "classifier": [("stop_small_talk_at", "Stop on small talk when at least this sure", "number", 0.5, 1.0, 0.01, 0.9),
                   ("force_tool_at", "Force a lookup when at least this sure", "number", 0.5, 1.0, 0.01, 0.7),
                   ("skip_question_below", "Skip a question when the chance of needing one is below", "number", 0.0, 0.5, 0.01, 0.2)],
    "math": [("check", "The calculator checks its answers (when the calculator is on)", "bool", None, None, None, True)],
    "math_language": [("only_with_numbers", "Only read messages that have numbers in them", "bool", None, None, None, True),
                      ("trust_above", "Ignore what it found when its shakiest character is below", "number", 0.0, 1.0, 0.05, 0.7),
                      ("sure_above", "Mark what it found unsure when its shakiest character is below", "number", 0.0, 1.0, 0.01, 0.99)],
}
WEIGHT_SETTINGS = (
    ("context_length", "Context length", "int", 8, 8192, 8),
    ("width", "Model width", "int", 32, 4096, 32),
    ("layers", "Layers", "int", 1, 64, 1),
    ("heads", "Attention heads", "int", 1, 64, 1),
    ("dropout", "Dropout", "number", 0.0, 0.9, 0.05),
    ("tie_embeddings", "Tie input/output weights", "bool", None, None, None),
)
WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and are as at be by can do does for from has have how i in is it its me my of on or our "
                 "so that the this to was we what when where which who whos why will with you your".split())


class HarnessError(Exception):
    """Something a person should read: a server down, a broken contract, a refused tool call."""


class ContractError(HarnessError):
    pass


# ----- plumbing -----

def http_json(method, url, body=None, timeout=30):
    """(status, parsed JSON or None, lower-cased headers). Raises HarnessError if unreachable."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw, head = resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as exc:
        status, raw, head = exc.code, exc.read(), exc.headers
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise HarnessError("can't reach %s (%s)" % (url, getattr(exc, "reason", exc)))
    try:
        parsed = json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        parsed = None
    return status, parsed, {k.lower(): v for k, v in (head.items() if head else [])}


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def new_id():
    return time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex()


def squash(text):
    return " ".join(str(text).split())


def today():
    return time.strftime("%A, %B %d, %Y").replace(" 0", " ")


def words(text):
    found = []
    for w in WORD.findall(str(text).lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # appointments -> appointment
        if w not in STOP:
            found.append(w)
    return found


# ----- contracts -----

def parse_json(text):
    text = THINK.sub("", str(text)).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    return json.loads(text[start:end + 1])


def validate(value, schema, path="$"):
    """Problems with `value` against a small JSON Schema subset: type, required, properties, items, enum, minItems."""
    kind = schema.get("type")
    if kind == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        ok = isinstance(value, {"object": dict, "array": list, "string": str, "boolean": bool}.get(kind, object))
    if not ok:
        return ["%s should be a %s" % (path, kind)]
    problems = []
    if "enum" in schema and value not in schema["enum"]:
        problems.append("%s must be one of %s" % (path, ", ".join(map(str, schema["enum"]))))
    if kind == "object":
        problems += ["%s.%s is missing" % (path, key) for key in schema.get("required", []) if key not in value]
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                problems += validate(value[key], sub, "%s.%s" % (path, key))
    if kind == "array":
        if len(value) < schema.get("minItems", 0):
            problems.append("%s needs at least %d item(s)" % (path, schema["minItems"]))
        for i, item in enumerate(value):
            problems += validate(item, schema.get("items", {}), "%s[%d]" % (path, i))
    return problems


def normalize(value, schema):
    """Clamp numbers into range, cut arrays to maxItems and strings to maxLength, and drop unknown keys."""
    kind = schema.get("type")
    if kind == "string" and "maxLength" in schema:
        return value[:schema["maxLength"]]
    if kind == "object":
        props = schema.get("properties", {})
        return {k: normalize(v, props[k]) for k, v in value.items() if k in props}
    if kind == "array":
        return [normalize(v, schema.get("items", {})) for v in value[:schema.get("maxItems", len(value))]]
    if kind in ("number", "integer"):
        if "minimum" in schema:
            value = max(schema["minimum"], value)
        if "maximum" in schema:
            value = min(schema["maximum"], value)
    return value


# ----- tools that aren't models -----

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "floor": math.floor,
         "ceil": math.ceil, "log": math.log, "log10": math.log10, "exp": math.exp}
NAMES = {"pi": math.pi, "e": math.e}
SYMBOLS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
MODEL_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}  # the steps a math model can be asked to do


def calc(expression, step=None):
    """Exact arithmetic without eval: parses the expression and walks only number nodes and operators.

    With step, every operation is handed to step(label, x, y, exact), which returns its value: label is "+",
    "/", "sqrt" and so on, and exact() works it out the calculator's way.
    """
    text = arithmetic_text(expression)
    if not text.strip() or len(text) > 200:
        raise HarnessError("the calculator needs a short arithmetic expression")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise HarnessError("that power is too big for the calculator")
            if step:
                return step(SYMBOLS[type(node.op)], left, right, lambda: OPS[type(node.op)](left, right))
            return OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return ev(node.operand) if isinstance(node.op, ast.UAdd) else -ev(node.operand)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS
                and not node.keywords and len(node.args) <= 3):
            args = [ev(a) for a in node.args]
            if step:
                return step(node.func.id, args, None, lambda: FUNCS[node.func.id](*args))
            return FUNCS[node.func.id](*args)
        if isinstance(node, ast.Name) and node.id in NAMES:
            return step(node.id, None, None, lambda: NAMES[node.id]) if step else NAMES[node.id]
        raise HarnessError("the calculator only does arithmetic")

    try:
        value = ev(ast.parse(text.strip(), mode="eval"))
    except SyntaxError:
        raise HarnessError("the calculator couldn't read %r" % expression)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError) as exc:
        raise HarnessError("the calculator couldn't do that (%s)" % exc)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return ("%.10g" % value) if isinstance(value, float) else str(value)


def arithmetic_text(expression):
    """How the calculator reads an expression: × and ÷ as symbols, 2,340 as 2340, 17.5% of 2340 as (17.5/100)*2340."""
    text = str(expression).replace("×", "*").replace("÷", "/").replace("^", "**").replace("−", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)                      # 2,340 -> 2340
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", text, flags=re.I)  # 17.5% of 2340
    return re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", text)


def strip_units(text):
    return UNITS.sub("", re.sub(r"\$(?=\d)", "", text))


def mentions_numbers(text):
    return bool(NUMBERS.search(text))


def find_expression(message):
    """The longest piece of a message the calculator can read, like "48213 + 9977", or "" if there isn't one.
    Units stuck to the numbers are read past, so "322234ft x 21323ft" counts."""
    for found in sorted(EXPRESSION.findall(strip_units(message)), key=len, reverse=True):
        expression = re.sub(r"(?<=[\d)])\s*x\s*(?=[\d(])", "*", found.strip().rstrip(","))
        try:
            calc(expression)
            return expression
        except HarnessError:
            continue
    return ""


def bare_arithmetic(message):
    """The expression when a message is nothing but arithmetic, like "2+2", "12 x 7 =" or "what's 48,213 + 9,977?"; else ""."""
    text = strip_units(re.sub(r"^\s*(?:what[’']?s|what is|calc(?:ulate)?|compute)\s+", "", message, flags=re.I)).strip().rstrip("=?!. ")
    expression = find_expression(text)

    def squeeze(t):
        return re.sub(r"(?<=[\d)])x(?=[\d(])", "*", re.sub(r"\s+", "", t))
    return expression if expression and squeeze(expression) == squeeze(text) else ""


def needs_lookup(message):
    """True when a message asks for something no model can have in its weights, like the weather or today's news."""
    return bool(LOOKUP.search(message))


def best_link(text):
    """The first address in a tool result worth fetching, so a list of search hits can be followed to the page.

    Sites in CLIENT_SIDE are skipped: their text arrives as menus and adverts because the page is drawn in the
    browser, and a model handed that will invent the number it was asked for. When every hit is one of them
    there is no link worth opening, so this returns "" and the reply is written from the search snippets
    instead. Falling back to the first one fetched a page of adverts, which is what it's here to avoid.
    """
    links = [l.rstrip(".,)") for l in re.findall(r"https?://[^\s<>\"']+", text or "")]
    return next((l for l in links if not any(d in l for d in CLIENT_SIDE)), "")


def language_need(reply):
    """(tool, what to look up) when the Language module handed the turn back, else None."""
    found = NEED.search(reply or "")
    return (found.group(1).lower(), found.group(2).strip()) if found else None


def drop_need(reply):
    """The reply without its NEED line, so a hand-back that can't be honoured never reaches the person."""
    return NEED.sub("", reply or "").strip()


def plain_arithmetic(expression):
    """The expression as plain numbers, + - * / and brackets, or None if it has anything else. Divisions move last
    where that's the same value ((17.5/100)*2340 becomes 17.5*2340/100), so a model working to 2 decimal places
    loses nothing along the way."""
    try:
        tree = ast.parse(arithmetic_text(expression).strip(), mode="eval").body
    except SyntaxError:
        return None

    def plain(node):
        if isinstance(node, ast.Constant):
            return type(node.value) in (int, float) and node.value >= 0
        return (isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div)
                and plain(node.left) and plain(node.right))

    def divide_last(node):
        if not isinstance(node, ast.BinOp):
            return node
        node.left, node.right = divide_last(node.left), divide_last(node.right)
        if isinstance(node.op, ast.Mult):
            for inner, other in ((node.left, node.right), (node.right, node.left)):
                if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div):  # (a/b)*c -> (a*c)/b
                    return ast.BinOp(ast.BinOp(inner.left, ast.Mult(), other), ast.Div(), inner.right)
        return node

    if not plain(tree):
        return None
    return re.sub(r"\s+", "", ast.unparse(divide_last(tree)))


def exact_value(expression):
    """A plain expression's exact value as a fraction."""
    def ev(node):
        if isinstance(node, ast.Constant):
            return Fraction(repr(node.value)) if isinstance(node.value, float) else Fraction(node.value)
        x, y = ev(node.left), ev(node.right)
        op = node.op
        return x + y if isinstance(op, ast.Add) else x - y if isinstance(op, ast.Sub) else x * y if isinstance(op, ast.Mult) else x / y
    try:
        return ev(ast.parse(expression, mode="eval").body)
    except ZeroDivisionError:
        raise HarnessError("that divides by zero")


def exact_text(value):
    """A fraction written out, to at most 10 decimal places."""
    if value.denominator == 1:
        return str(value.numerator)
    return format((Decimal(value.numerator) / Decimal(value.denominator)).quantize(Decimal("1e-10")).normalize(), "f")


def fill_answers(expression, answers):
    """p1, p2 ... swapped for the answers already worked out."""
    def swap_in(m):
        if m.group(0) not in answers:
            raise HarnessError("%s uses %s before it's worked out" % (expression, m.group(0)))
        return "(%s)" % answers[m.group(0)]
    return re.sub(r"\bp\d+\b", swap_in, expression)


def worked_values(expressions):
    """Each expression's value in order, with later ones using earlier answers."""
    answers, values = {}, []
    for i, expression in enumerate(expressions, 1):
        answers["p%d" % i] = calc(fill_answers(expression, answers))
        values.append(float(answers["p%d" % i]))
    return values


def whole_numbers(symbol, x, y):
    """x symbol y as whole numbers a math model can take: (a, b, decimal places, negate), or why it can't be.

    Decimals are shifted into whole numbers the way people do it by hand: 12.50 + 7.25 is 1250 + 725, two
    places back. A subtraction that would go below zero is flipped and the answer negated.
    """
    x, y = Decimal(repr(x)), Decimal(repr(y))
    if not (x.is_finite() and y.is_finite()) or x < 0 or y < 0:
        return "negative numbers go to the calculator"
    px, py = (max(0, -d.normalize().as_tuple().exponent) for d in (x, y))
    if symbol == "*":
        a, b, places = int(x.scaleb(px)), int(y.scaleb(py)), px + py
    else:
        places = max(px, py)
        a, b = int(x.scaleb(places)), int(y.scaleb(places))
    return (b, a, places, True) if symbol == "-" and b > a else (a, b, places, False)


def as_number(d):
    return int(d) if d == d.to_integral_value() else float(d)


def number_text(v):
    return ("%.10g" % v) if isinstance(v, float) else str(v)


class MemoryStore:
    """Long-term memory: a plain JSONL document store searched with BM25. Not a model."""

    def __init__(self, folder, keep=2000):
        self.path = os.path.join(folder, "memories.jsonl")
        self.keep = keep
        self.lock = threading.Lock()

    def all(self):
        try:
            with open(self.path) as f:
                return [json.loads(line) for line in f if line.strip()]
        except (OSError, ValueError):
            return []

    def count(self):
        return len(self.all())

    def _save(self, items):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
        os.replace(tmp, self.path)

    def add(self, text, kind, source):
        text = squash(text)[:500]
        if not text:
            return None
        with self.lock:
            items = self.all()
            if any(i["text"].lower() == text.lower() for i in items):
                return None
            item = {"id": "m" + os.urandom(3).hex(), "text": text, "kind": kind, "source": source,
                    "created": time.strftime("%Y-%m-%d %H:%M")}
            self._save((items + [item])[-self.keep:])
            return item

    def forget(self, mid):
        with self.lock:
            items = self.all()
            kept = [i for i in items if i["id"] != mid]
            if len(kept) == len(items):
                return False
            self._save(kept)
            return True

    def search(self, queries, top=8):
        items = self.all()
        if not items:
            return []
        docs = [words(i["text"]) for i in items]
        avg = sum(len(d) for d in docs) / len(docs) or 1
        df = Counter(w for d in docs for w in set(d))
        scores = [0.0] * len(docs)
        for term in set(w for q in queries for w in words(q)):
            if term not in df:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            for i, d in enumerate(docs):
                tf = d.count(term)
                if tf:
                    scores[i] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(d) / avg))
        ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
        return [dict(items[i], score=round(scores[i], 3)) for i in ranked[:top] if scores[i] > 0]


class FileTool:
    """The only files the harness may touch: plain names in one sandbox folder. Delete moves to .trash."""
    NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,80}$")

    def __init__(self, folder, max_bytes=262144):
        self.folder = folder
        self.max_bytes = max_bytes

    def _path(self, name):
        name = str(name).strip()
        if not self.NAME.match(name) or ".." in name:
            raise HarnessError("file names must be plain names like shopping-list.txt")
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, name)
        if os.path.islink(path) or os.path.dirname(os.path.realpath(path)) != os.path.realpath(self.folder):
            raise HarnessError("that file is outside the workspace")
        return path

    def do(self, action, name, content=""):
        if action == "list":
            os.makedirs(self.folder, exist_ok=True)
            names = sorted(n for n in os.listdir(self.folder) if not n.startswith("."))
            return "\n".join(names) or "(the workspace is empty)"
        path = self._path(name)
        exists = os.path.isfile(path)
        if action == "read":
            if not exists:
                raise HarnessError("there's no file called %s" % name)
            with open(path, "rb") as f:
                data = f.read(self.max_bytes + 1)
            if len(data) > self.max_bytes:
                raise HarnessError("%s is too big to read" % name)
            return data.decode("utf-8", errors="replace")
        if action in ("create", "update"):
            if action == "create" and exists:
                raise HarnessError("%s already exists; update it instead" % name)
            if action == "update" and not exists:
                raise HarnessError("there's no file called %s to update" % name)
            data = str(content).encode("utf-8")
            if len(data) > self.max_bytes:
                raise HarnessError("that's too much text for one file")
            with open(path, "wb") as f:
                f.write(data)
            return "%s %s (%d bytes)" % ("created" if action == "create" else "updated", name, len(data))
        if action == "delete":
            if not exists:
                raise HarnessError("there's no file called %s" % name)
            trash = os.path.join(self.folder, ".trash")
            os.makedirs(trash, exist_ok=True)
            shutil.move(path, os.path.join(trash, "%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), name)))
            return "moved %s to the workspace trash" % name
        raise HarnessError("file_system can't %r" % action)


# ----- working memory -----

def new_state(message, conversation=None, history=None):
    return {"id": new_id(), "conversation": conversation, "history": list(history or []), "message": message,
            "today": today(), "status": "running", "note": "",
            "goal": "", "observations": [], "attention": None, "recall": None, "memories": [], "problems": [], "math_language": None, "cycles": [],
            "reply": "", "remembered": [], "trace": [], "started": time.time(), "seconds": None}


def conversation_view(state):
    """The last few turns, kept short, so a module knows what was already said and asked."""
    return [{"person": t["message"][:500], "assistant": (t.get("reply") or t.get("note") or "")[:600]}
            for t in state.get("history", [])[-HISTORY_TURNS:]]


class ConversationStore:
    """Chats as plain JSON files, one per conversation, holding what the next turn needs."""

    def __init__(self, folder):
        self.folder = folder
        self.lock = threading.Lock()

    def _path(self, cid):
        if not RUN_ID.match(str(cid)):
            raise HarnessError("that isn't a conversation id")
        return os.path.join(self.folder, "%s.json" % cid)

    def create(self):
        now = time.strftime("%Y-%m-%d %H:%M")
        data = {"id": new_id(), "started": now, "updated": now, "turns": []}
        with self.lock:
            write_json(self._path(data["id"]), data)
        return data

    def get(self, cid):
        return read_json(self._path(cid), None)

    def append(self, cid, state):
        last = current(state)
        asked = bool(state["reply"] and last and last.get("decision") and chosen(last)["type"] == "ask")
        turn = {"run": state["id"], "message": state["message"], "reply": state["reply"], "status": state["status"],
                "note": state["note"], "goal": state["goal"], "asked": asked, "seconds": state["seconds"],
                "at": time.strftime("%Y-%m-%d %H:%M")}
        with self.lock:
            data = self.get(cid) or {"id": cid, "started": turn["at"], "turns": []}
            data["turns"].append(turn)
            data["updated"] = turn["at"]
            write_json(self._path(cid), data)

    def recent(self, n=20):
        try:
            names = [f for f in os.listdir(self.folder) if f.endswith(".json")]
        except OSError:
            return []
        names.sort(key=lambda f: os.path.getmtime(os.path.join(self.folder, f)), reverse=True)
        out = []
        for name in names[:n]:
            data = read_json(os.path.join(self.folder, name), None)
            if data and data.get("turns"):
                out.append({"id": data["id"], "first": data["turns"][0]["message"][:80], "turns": len(data["turns"]),
                            "updated": data.get("updated", "")})
        return out


class Live:
    """Working memory plus a version counter, so a page can follow along while modules write to it."""

    def __init__(self, state):
        self.state = state
        self.cond = threading.Condition()
        self.version = 0

    def apply(self, fn):
        with self.cond:
            result = fn(self.state)
            self.version += 1
            self.cond.notify_all()
            return result

    def snapshot(self):
        with self.cond:
            return self.version, json.loads(json.dumps(self.state))

    def wait(self, version, timeout):
        with self.cond:
            self.cond.wait_for(lambda: self.version != version, timeout)
            return self.version


def drop_none(d):
    return {k: v for k, v in d.items() if v is not None}


def load_local(checkpoint, code):
    """Load a model trained in SMMOL with its own code, which lives next to its out/ folder.

    code is "file.py:Class". Projects reuse file names like model.py, so a project's modules are imported
    fresh and taken back out of sys.modules afterwards, leaving other projects' same-named modules alone.
    """
    import importlib.util
    script, _, cls = code.partition(":")
    folder = os.path.dirname(os.path.dirname(checkpoint))
    names = {f[:-3] for f in os.listdir(folder) if f.endswith(".py")} if os.path.isdir(folder) else set()
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location("smmol_" + script[:-3], os.path.join(folder, script))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, cls)(checkpoint)
    except Exception as exc:
        raise HarnessError("couldn't load %s (%s)" % (checkpoint, exc))
    finally:
        sys.path.remove(folder)
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


def current(state):
    return state["cycles"][-1] if state["cycles"] else None


def focused(state):
    obs = [o for o in state["observations"] if o.get("relevance") is None or o["relevance"] >= FOCUS]
    return [{k: o[k] for k in ("kind", "text", "who", "what", "when", "where") if o.get(k)}
            for o in (obs or state["observations"][:1])]


def action_log(state):
    return [dict({k: c["action"][k] for k in ("tool", "action", "target", "ok")}, result=c["action"]["result"][:RESULT_CHARS])
            for c in state["cycles"] if c.get("action") and c["action"].get("status") == "done"]


def view(candidate):
    """What other modules need to know about a candidate."""
    shown = {"index": candidate["index"], "type": candidate["type"], "summary": candidate["summary"]}
    if candidate["type"] == "tool":
        shown.update(tool=candidate["tool"], action=candidate["action"], target=candidate["target"])
    return shown


def prediction_for(entry, index):
    return next((p for p in entry["predictions"] if p["index"] == index), None)


def chosen(entry):
    return entry["candidates"][entry["decision"]["choice"] - 1]


def instead_of_asking(candidates):
    """When a question gets overruled: say or make something, and use a tool only if that's all there is."""
    return (next((c for c in candidates if c["type"] in ("make", "answer")), None)
            or next((c for c in candidates if c["type"] != "ask"), None))


def worth_keeping(o):
    """Long-term memory is for the person and their world, not the details of one request."""
    if o["kind"] not in ("fact", "preference", "event") or REQUEST_ECHO.search(o["text"]):
        return False
    who = (o.get("who") or "").strip().lower()
    return bool(re.search(r"\buser\b", o["text"], re.I) or (who and who != "user"))


# ----- the harness -----

def provider_urls_from_env(providers):
    """Let the environment point a provider somewhere else than harness.json says.

    harness.json ships pointing at 127.0.0.1 so a fresh clone is neither broken nor carrying
    someone else's home network around. Whoever runs an engine on another box sets
    SMMOL_PROVIDER_<NAME>_URL once — SMMOL_PROVIDER_PC_URL=http://10.0.0.5:8081 — and every
    call site picks it up, because they all read through self.providers.
    """
    out = {}
    for name, cfg in providers.items():
        override = os.environ.get("SMMOL_PROVIDER_%s_URL" % name.upper())
        out[name] = dict(cfg, url=override) if override and "url" in cfg else cfg
    return out


class Harness:
    def __init__(self, folder=HERE, data_dir=None):
        self.folder = folder
        self.config = read_json(os.path.join(folder, "harness.json"), None)
        self.contracts = read_json(os.path.join(folder, "contracts.json"), None)
        if not self.config or not self.contracts:
            raise HarnessError("harness.json or contracts.json is missing in %s" % folder)
        self.providers = provider_urls_from_env(self.config["providers"])
        self.modules = {m["id"]: m for m in self.config["modules"]}
        self.order = [m["id"] for m in self.config["modules"]]
        self.tools = self.config["tools"]
        self.data_dir = data_dir or folder
        self.memory = MemoryStore(os.path.join(self.data_dir, "memory"), self.config.get("memory", {}).get("keep", 2000))
        fs = self.tools.get("file_system", {})
        self.files = FileTool(os.path.join(self.data_dir, fs.get("folder", "workspace")), fs.get("max_bytes", 262144))
        self.runs_dir = os.path.join(self.data_dir, "runs")
        self.conversations = ConversationStore(os.path.join(self.data_dir, "conversations"))
        self.lock = threading.Lock()
        self.choices = read_json(os.path.join(self.data_dir, "choices.json"), {})
        self.scores = read_json(os.path.join(self.data_dir, "scores.json"), {})
        self.checks = read_json(os.path.join(folder, "checks.json"), {})
        self.switches = read_json(os.path.join(self.data_dir, "switches.json"), {"modules": {}, "tools": {}})
        self.settings = read_json(os.path.join(self.data_dir, "settings.json"), {})
        self._local_models = {}  # (checkpoint path, modified time) -> loaded model
        self.on_event = None
        self._status = (0.0, None)
        self._no_schema = set()  # providers that rejected response_format

    # ----- models per module -----

    def module(self, mid):
        if mid not in self.modules:
            raise HarnessError("no module called %r (have: %s)" % (mid, ", ".join(self.order)))
        return self.modules[mid]

    def choice(self, mid):
        if mid in self.choices:
            return self.choices[mid]
        m = self.module(mid)
        return "%s/%s" % (m["provider"], m["model"]) if m.get("provider") and m.get("model") else None

    def provider_of(self, choice):
        name, _, model = str(choice or "").partition("/")
        if name not in self.providers or not model:
            raise HarnessError("%r isn't a provider/model pair" % choice)
        return name, self.providers[name], model

    def local_entry(self, p, model):
        """A trained model's path, kind and loading code. A plain string is the path of a router checkpoint."""
        entry = p.get("models", {}).get(model, "")
        entry = {"path": entry} if isinstance(entry, str) else dict(entry)
        entry.setdefault("kind", "classifier")
        entry.setdefault("code", LOCAL_CODE.get(entry["kind"], ""))
        entry["path"] = os.path.normpath(os.path.join(self.folder, entry.get("path", "")))
        if entry.get("training_config"):
            entry["training_config"] = os.path.normpath(os.path.join(self.folder, entry["training_config"]))
        return entry

    def local_path(self, p, model):
        return self.local_entry(p, model)["path"]

    def provider_status(self, name, timeout=4):
        p = self.providers[name]
        if p["kind"] == "local":  # models trained here; available once their checkpoint exists
            models = [{"id": m, "kind": self.local_entry(p, m)["kind"]} for m in p.get("models", {}) if os.path.isfile(self.local_path(p, m))]
            return {"up": bool(models), "models": models, "error": "" if models else "not trained yet"}
        base = p["url"].rstrip("/")
        try:
            if p["kind"] == "openai":
                status, data, _ = http_json("GET", base + "/v1/models", timeout=timeout)
                listed = data.get("data", []) if isinstance(data, dict) else []
                return {"up": status == 200, "models": [{"id": str(m["id"]), "loaded": m.get("loaded")}
                                                        for m in listed if isinstance(m, dict) and m.get("id")]}
            status, _, _ = http_json("GET", base + "/health", timeout=timeout)
            return {"up": status == 200, "models": []}
        except HarnessError as exc:
            return {"up": False, "models": [], "error": str(exc)}

    def status(self, max_age=15.0):
        with self.lock:
            stamp, cached = self._status
        if cached is not None and time.monotonic() - stamp < max_age:
            return cached
        found = {}
        threads = [threading.Thread(target=lambda n=n: found.__setitem__(n, self.provider_status(n)))
                   for n in self.providers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with self.lock:
            self._status = (time.monotonic(), found)
        return found

    def options(self, mid, status=None):
        self.module(mid)
        status = self.status() if status is None else status
        kind = self.modules[mid].get("kind")
        out = []
        for name, p in self.providers.items():
            local_kind = kind if kind in LOCAL_CODE else ("language" if mid == "language" else None)
            allowed = p["kind"] == "local" if kind in LOCAL_CODE else p["kind"] == "openai" or (
                p["kind"] == "local" and local_kind is not None
            )
            if not allowed:
                continue
            st = status.get(name) or {"up": False, "models": []}
            fits = (lambda m: self.local_entry(p, m)["kind"] == local_kind) if p["kind"] == "local" else (lambda m: True)
            for m in st["models"]:
                if fits(m["id"]):
                    out.append({"choice": "%s/%s" % (name, m["id"]), "provider": name, "label": p.get("label", name),
                                "model": m["id"], "up": st["up"], "loaded": m.get("loaded")})
            if p["kind"] == "local":  # a model still training says so, instead of looking unreachable
                listed = {m["id"] for m in st["models"]}
                out.extend({"choice": "%s/%s" % (name, m), "provider": name, "label": p.get("label", name), "model": m,
                            "up": False, "loaded": None, "note": "not trained yet"}
                           for m in p.get("models", {}) if m not in listed and fits(m))
        return out

    def choose(self, mid, choice):
        self.module(mid)
        if choice not in [o["choice"] for o in self.options(mid, self.status(max_age=0)) if o["up"]]:
            raise HarnessError("%s isn't offered for %s right now" % (choice, mid))
        with self.lock:
            self.choices[mid] = choice
            write_json(os.path.join(self.data_dir, "choices.json"), self.choices)

    # ----- switches: a module or tool that's off doesn't run and is never mentioned to any model -----

    def module_on(self, mid):
        self.module(mid)
        return bool(self.switches.get("modules", {}).get(mid, self.modules[mid].get("enabled", True)))

    def tool_on(self, name):
        cfg = self.tools.get(name)
        if not cfg or not cfg.get("built", True):
            return False
        return bool(self.switches.get("tools", {}).get(name, cfg.get("enabled", True)))

    def toggle(self, kind, name, on):
        if kind == "module":
            self.module(name)
        elif kind == "tool":
            if name not in self.tools:
                raise HarnessError("no tool called %r" % name)
            if not self.tools[name].get("built", True):
                raise HarnessError("%s isn't built yet" % name)
        else:
            raise HarnessError("you can switch a module or a tool")
        with self.lock:
            self.switches.setdefault(kind + "s", {})[name] = bool(on)
            write_json(os.path.join(self.data_dir, "switches.json"), self.switches)

    # ----- settings: knobs per module, saved next to the switches -----

    def local_setting_specs(self, mid):
        """Extra live inference knobs declared by the selected local checkpoint adapter."""
        choice = self.choice(mid)
        if not choice:
            return []
        _, provider, model = self.provider_of(choice)
        if provider["kind"] != "local":
            return []
        specs = []
        for raw in self.local_entry(provider, model).get("settings", []):
            if not isinstance(raw, dict):
                raise HarnessError("local model settings must be objects")
            required = ("key", "label", "type", "min", "max", "step", "default")
            if any(key not in raw for key in required) or raw["type"] not in ("int", "number", "bool"):
                raise HarnessError("local model has a broken setting declaration")
            specs.append(tuple(raw[key] for key in required))
        return specs

    def setting_specs(self, mid):
        mod = self.module(mid)
        specs = list(SETTINGS.get(mod.get("kind"), []))
        if mod.get("kind") is None:  # a language model's defaults come from its own config
            specs = [s[:6] + (mod.get(s[0], s[6]),) for s in specs]
            if "make_prompt" in mod:
                specs.append(("make_max_tokens", "Token limit when making something", "int", 200, 4000, 100, mod.get("make_max_tokens", 1800)))
        known = {spec[0] for spec in specs}
        specs.extend(spec for spec in self.local_setting_specs(mid) if spec[0] not in known)
        return specs

    def _spec(self, mid, key):
        spec = next((s for s in self.setting_specs(mid) if s[0] == key), None)
        if spec is None:
            raise HarnessError("%s has no setting called %s" % (self.modules[mid]["name"], key))
        return spec

    def setting(self, mid, key):
        return self.settings.get(mid, {}).get(key, self._spec(mid, key)[6])

    def settings_for(self, mid):
        return [{"key": key, "label": label, "type": kind, "min": low, "max": high, "step": step, "default": default,
                 "value": self.settings.get(mid, {}).get(key, default)}
                for key, label, kind, low, high, step, default in self.setting_specs(mid)]

    def set_setting(self, mid, key, value):
        _, label, kind, low, high, _, default = self._spec(mid, key)
        if kind == "bool":
            if not isinstance(value, bool):
                raise HarnessError("%s is on or off" % label)
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise HarnessError("%s must be a number from %s to %s" % (label, low, high))
        else:
            value = int(value) if kind == "int" else float(value)
        with self.lock:
            saved = self.settings.setdefault(mid, {})
            saved[key] = value
            if value == default:  # back at its default, so there's nothing to keep
                del saved[key]
            write_json(os.path.join(self.data_dir, "settings.json"), self.settings)

    # ----- local weight architecture: editable source config for the next checkpoint -----

    @staticmethod
    def _parameter_count(architecture):
        vocab = int(architecture["vocab_size"])
        context = int(architecture["context_length"])
        width = int(architecture["width"])
        layers = int(architecture["layers"])
        # embeddings + (attention/MLP matrices and four LayerNorm vectors) per block + final LayerNorm
        count = vocab * width + context * width + layers * (12 * width * width + 4 * width) + 2 * width
        if not architecture.get("tie_embeddings", True):
            count += vocab * width
        return count

    def weight_config(self, mid):
        """Current immutable checkpoint facts plus editable architecture for the next training run."""
        choice = self.choice(mid)
        if not choice:
            return None
        _, provider, model = self.provider_of(choice)
        if provider["kind"] != "local":
            return None
        entry = self.local_entry(provider, model)
        source_path = entry.get("training_config")
        if not source_path:
            return None
        source = read_json(source_path, None)
        if not isinstance(source, dict) or not isinstance(source.get("architecture"), dict):
            raise HarnessError("%s has no readable training architecture" % model)
        draft = source["architecture"]
        artifact_path = os.path.join(os.path.dirname(entry["path"]), "model_config.json")
        artifact = read_json(artifact_path, source)
        current = dict(artifact.get("architecture") or draft)
        metadata = read_json(os.path.join(os.path.dirname(entry["path"]), "metadata.json"), {})
        current.update({
            "parameters": self._parameter_count(current),
            "bytes": os.path.getsize(entry["path"]) if os.path.isfile(entry["path"]) else 0,
            "stage": metadata.get("stage", "unknown"),
            "step": metadata.get("step", 0),
            "format": metadata.get("format", os.path.splitext(entry["path"])[1].lstrip(".")),
        })
        editable = [{"key": key, "label": label, "type": kind, "min": low, "max": high, "step": step,
                     "value": draft.get(key)} for key, label, kind, low, high, step in WEIGHT_SETTINGS]
        keys = [spec[0] for spec in WEIGHT_SETTINGS]
        return {"current": current, "next": editable,
                "changed": any(current.get(key) != draft.get(key) for key in keys),
                "tokenizer": (source.get("tokenizer") or {}).get("kind", "unknown")}

    def set_weight_config(self, mid, key, value):
        choice = self.choice(mid)
        _, provider, model = self.provider_of(choice)
        if provider["kind"] != "local":
            raise HarnessError("pick a local weight set first")
        entry = self.local_entry(provider, model)
        path = entry.get("training_config")
        if not path:
            raise HarnessError("%s does not expose a training architecture" % model)
        spec = next((item for item in WEIGHT_SETTINGS if item[0] == key), None)
        if spec is None:
            raise HarnessError("there is no weight setting called %s" % key)
        _, label, kind, low, high, _ = spec
        if kind == "bool":
            if not isinstance(value, bool):
                raise HarnessError("%s is on or off" % label)
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise HarnessError("%s must be a number from %s to %s" % (label, low, high))
        else:
            value = int(value) if kind == "int" else float(value)
        document = read_json(path, None)
        if not isinstance(document, dict) or not isinstance(document.get("architecture"), dict):
            raise HarnessError("%s has no readable training architecture" % model)
        architecture = dict(document["architecture"])
        architecture[key] = value
        if int(architecture.get("width", 0)) % int(architecture.get("heads", 1)):
            raise HarnessError("Model width must be divisible by attention heads")
        if int(architecture.get("vocab_size", 0)) != 256:
            raise HarnessError("the byte tokenizer requires vocab_size 256")
        document["architecture"] = architecture
        write_json(path, document)

    def enabled_tools(self):
        """Tools the Planner may call. Arithmetic stays on the menu while the Math module can do it without the calculator."""
        return [n for n, t in self.tools.items() if t.get("kind", "tool") == "tool"
                and (self.tool_on(n) or (n == "calculator" and self.math_on()))]

    def _tool_text(self, name, key):
        """What the Planner is told about a tool. Arithmetic left to the Math module alone is described as what it can do."""
        if name == "calculator" and not self.tool_on(name) and self.math_on():
            return self.modules["math"].get("tool_" + key, "")
        return self.tools[name].get(key, "")

    def tool_menu(self):
        return "\n".join("- %s: %s" % (n, self._tool_text(n, "does")) for n in self.enabled_tools()) or "(none)"

    def tool_rules(self):
        tools = self.enabled_tools()
        if not tools:
            return "No tools are available, so propose only answer, make and ask candidates."
        return "For a tool candidate, set tool, action and target exactly:\n" + "\n".join(
            "- %s" % self._tool_text(n, "how") for n in tools if self._tool_text(n, "how"))

    def contract(self, name):
        """A module's contract, with every choice that points at something switched off taken out."""
        schema = json.loads(json.dumps(self.contracts[name]))
        if name == "planner":
            tools = self.enabled_tools()
            fields = schema["properties"]["candidates"]["items"]["properties"]
            fields["tool"]["enum"] = ["none"] + tools
            fields["type"]["enum"] = ["answer", "make", "ask"] + (["tool"] if tools else [])
            if "file_system" not in tools:
                fields["action"]["enum"] = ["none"]
        return schema

    # ----- the router: a small model trained here -----

    def local_ready(self, mid):
        """The module's picked model was trained here, and its checkpoint exists."""
        choice = self.choice(mid) if mid in self.modules else None
        if not choice:
            return False
        name, p, model = self.provider_of(choice)
        return p["kind"] == "local" and os.path.isfile(self.local_path(p, model))

    def router_ready(self):
        return self.local_ready("router")

    def math_on(self):
        """The Math module is switched on and has a trained model to work with."""
        return "math" in self.modules and self.module_on("math") and self.local_ready("math")

    def local(self, choice):
        """A model trained here, loaded once. A checkpoint that changes on disk (training again) is loaded again."""
        name, p, model = self.provider_of(choice)
        if p["kind"] != "local":
            raise HarnessError("%s isn't a model trained here" % choice)
        entry = self.local_entry(p, model)
        if not os.path.isfile(entry["path"]):
            raise HarnessError("%s isn't trained yet" % model)
        key = (entry["path"], os.path.getmtime(entry["path"]))
        with self.lock:
            if key not in self._local_models:
                for old in [k for k in self._local_models if k[0] == entry["path"]]:
                    del self._local_models[old]
                self._local_models[key] = load_local(entry["path"], entry["code"])
            return self._local_models[key]

    def router(self, choice):
        return self.local(choice)

    def router_tools(self):
        """The router's tool choices that exist right now. Everything else is invisible to it."""
        tools = self.enabled_tools()
        memory = self.tool_on("memory") and self.module_on("recall")
        return [rt for rt, ht in ROUTER_TOOLS.items() if (memory if ht == "memory" else ht in tools)]

    def route(self, live):
        s = live.state
        choice = self.choice("router")
        entry = {"module": "router", "name": "Router", "part": "", "model": choice, "status": "running",
                 "started": time.time(), "seconds": None, "retries": 0, "error": ""}
        live.apply(lambda st: st["trace"].append(entry))
        started = time.monotonic()
        out, error = None, ""
        try:
            prev = (s["history"][-1].get("reply") or "") if s.get("history") else ""
            out = dict(self.router(choice).route(s["message"], prev, self.router_tools()))
            out["tools_seen"] = self.router_tools()
        except HarnessError as exc:
            error = str(exc)
        except Exception as exc:  # a broken checkpoint shouldn't take the harness down with a traceback
            error = "router failed: %s" % exc
        seconds = round(time.monotonic() - started, 3)

        def close(st):
            entry.update(status="error" if error else "done", seconds=seconds, error=error)
            st["route"] = out
        live.apply(close)
        if self.on_event:
            self.on_event(entry)
        if error:
            raise HarnessError("Router " + error)

    def _route_hint(self, s):
        r = s.get("route") if self.module_on("router") else None
        if not r:
            return None
        return {"message_kind": r["intent"], "suggested_tool": ROUTER_TOOLS.get(r["tool"], "none"), "ask_first": r["ask_first"],
                "note": "a fast first guess from a small trained classifier; follow it unless the observations clearly say otherwise"}

    def _problems(self, s):
        """The problems other modules see, answers included, or None when Math Language is off or found nothing."""
        if not (self.module_on("math_language") and s.get("problems")):
            return None
        return [dict({k: p[k] for k in ("id", "expression", "unit", "about", "answer", "error") if p.get(k) not in (None, "")},
                     **({"unsure": True} if p.get("unsure") else {}))
                for p in s["problems"]]

    def _memories(self, s):
        return [m["text"] for m in s["memories"]] if self.module_on("recall") and self.tool_on("memory") else None

    def _actions(self, s):
        return action_log(s) if self.enabled_tools() else None

    def _view(self, entry, c):
        return dict(view(c), prediction=prediction_for(entry, c["index"])) if self.module_on("predictor") else view(c)

    # ----- asking a module -----

    def chat(self, choice, system, user, schema=None, max_tokens=600, temperature=0.2, timeout=600,
             local_settings=None):
        """(text, finish_reason). With a schema, llama.cpp's server constrains the reply to it."""
        name, p, model = self.provider_of(choice)
        if p["kind"] == "local":
            entry = self.local_entry(p, model)
            if entry["kind"] != "language" or schema is not None:
                raise HarnessError("%s isn't a free-text Language model" % choice)
            return self.local(choice).complete(system, user, max_tokens, temperature, **(local_settings or {})), "stop"
        body = {"model": model, "stream": False, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema is not None and name not in self._no_schema:
            body["response_format"] = {"type": "json_object", "schema": schema}
        url = p["url"].rstrip("/") + "/v1/chat/completions"
        status, data, _ = http_json("POST", url, body, timeout=timeout)
        if status == 400 and "response_format" in body:  # this server doesn't take schemas; validate afterwards instead
            self._no_schema.add(name)
            body.pop("response_format")
            status, data, _ = http_json("POST", url, body, timeout=timeout)
        try:
            choice0 = data["choices"][0]
            return THINK.sub("", str(choice0["message"]["content"] or "")), choice0.get("finish_reason")
        except (TypeError, KeyError, IndexError):
            detail = data.get("error") if isinstance(data, dict) else None
            raise HarnessError("%s gave no answer (HTTP %s%s)" % (choice, status, ": %s" % str(detail)[:200] if detail else ""))

    def _ask(self, mod, choice, prompt_key, payload, contract):
        system = (mod[prompt_key].replace("{today}", today()).replace("{tools}", self.tool_menu())
                  .replace("{tool_rules}", self.tool_rules()))
        user = "Input:\n" + json.dumps(drop_none(payload), indent=1, ensure_ascii=False)
        schema = self.contract(contract) if contract else None
        # "prompt" uses max_tokens and "make_prompt" uses make_max_tokens, so one part of a module can have its own limit
        key = prompt_key.replace("prompt", "max_tokens")
        limit = self.setting(mod["id"], key if key in {s[0] for s in self.setting_specs(mod["id"])} else "max_tokens")
        local_settings = {spec[0]: self.setting(mod["id"], spec[0]) for spec in self.local_setting_specs(mod["id"])}
        note, problems = "", []
        for attempt in range(2):
            text, finish = self.chat(choice, system, user + note, schema, limit,
                                     self.setting(mod["id"], "temperature"), local_settings=local_settings)
            if schema is None:
                if text.strip():
                    return text.strip(), attempt
                problems = ["the reply was empty"]
            else:
                try:
                    value = parse_json(text)
                    problems = validate(value, schema)
                except ValueError:
                    problems = ["the reply wasn't a JSON object"]
                if not problems:
                    return normalize(value, schema), attempt
            if finish == "length":
                problems.append("it was cut off at %d tokens" % limit)
                limit *= 2
            note = "\n\nYour last reply didn't fit (%s). Reply again%s." % (
                "; ".join(problems[:4]), " with JSON that fits the contract" if schema else "")
        raise ContractError("broke its contract twice: %s" % "; ".join(problems[:4]))

    def _module(self, live, mid, payload, contract, prompt_key="prompt"):
        mod = self.module(mid)
        choice = self.choice(mid)
        entry = {"module": mid, "name": mod["name"], "part": prompt_key[:-len("_prompt")] if prompt_key != "prompt" else "",
                 "model": choice, "status": "running", "started": time.time(), "seconds": None, "retries": 0, "error": ""}
        live.apply(lambda s: s["trace"].append(entry))
        started = time.monotonic()
        result, retries, error = None, 0, ""
        try:
            if not choice:
                raise HarnessError("no model picked")
            result, retries = self._ask(mod, choice, prompt_key, payload, contract)
        except HarnessError as exc:
            error = str(exc)
        seconds = round(time.monotonic() - started, 1)
        live.apply(lambda s: entry.update(status="error" if error else "done", seconds=seconds, retries=retries, error=error))
        if self.on_event:
            self.on_event(entry)
        if error:
            raise HarnessError("%s %s" % (mod["name"], error))
        return result

    # ----- the modules -----

    def perceive(self, live):
        s = live.state
        out = self._module(live, "perception", {"message": s["message"], "today": s["today"],
                                                "conversation": conversation_view(s)}, "perception")
        obs = [dict(o, id="o%d" % i, relevance=None) for i, o in enumerate(out["observations"], 1)]
        live.apply(lambda st: st.update(observations=obs, goal=squash(out["goal"])))

    def attend(self, live):
        s = live.state
        numbered = [dict({k: o[k] for k in ("kind", "text", "when") if o.get(k)}, index=i)
                    for i, o in enumerate(s["observations"], 1)]
        out = self._module(live, "attention", {"message": s["message"], "goal": s["goal"], "observations": numbered,
                                               "conversation": conversation_view(s)}, "attention")
        scores = {sc["index"]: sc["relevance"] for sc in out["scores"]}

        def apply(st):
            for i, o in enumerate(st["observations"], 1):
                o["relevance"] = scores.get(i, 0.0)
            st["attention"] = {"for_me": out["for_me"], "why": squash(out["why"])}
        live.apply(apply)

    def recall(self, live, memory=None):
        memory = memory or self.memory
        s = live.state
        if not memory.count():
            return live.apply(lambda st: st.update(recall={"queries": [], "found": 0, "note": "long-term memory is empty"}))
        out = self._module(live, "recall", {"goal": s["goal"], "observations": focused(s),
                                            "conversation": conversation_view(s)}, "recall_query")
        queries = [squash(q) for q in out["queries"] if squash(q)]
        found = memory.search(queries + [s["message"]], self.config.get("memory", {}).get("recall_top", 8))
        if not found:
            return live.apply(lambda st: st.update(recall={"queries": queries, "found": 0}))
        ranked = self._module(live, "recall", {"goal": s["goal"], "queries": queries,
                                               "memories": [{"id": m["id"], "text": m["text"], "saved": m["created"]} for m in found]},
                              "recall_rank", "rank_prompt")
        score = {r["id"]: r["relevance"] for r in ranked["ranked"]}
        kept = sorted((dict(m, relevance=score[m["id"]]) for m in found if score.get(m["id"], 0) >= KEEP_MEMORY),
                      key=lambda m: -m["relevance"])
        live.apply(lambda st: st.update(recall={"queries": queries, "found": len(found)}, memories=kept))

    def plan(self, live):
        s = live.state
        done = action_log(s)
        last = current(s)
        payload = {"goal": s["goal"], "observations": focused(s), "memories": self._memories(s),
                   "actions": self._actions(s), "last_critique": last["critique"] if last else None,
                   "conversation": conversation_view(s), "route": self._route_hint(s),
                   "problems": self._problems(s)}
        out = self._module(live, "planner", payload, "planner")
        enabled = self.enabled_tools()
        kept, dropped, changed = [], [], []
        for c in out["candidates"]:
            c = dict(c, summary=squash(c["summary"]), target=c["target"].strip())
            label = c["summary"] or c["type"]
            if c["type"] == "tool" and c["tool"] == "none":  # a step with no tool is something to do itself, not to throw away
                changed.append("%s: named no tool, so it became a make step" % label)
                c["type"] = "make"
            if c["type"] != "tool":
                c.update(tool="none", action="none", target="", content="")
            elif c["tool"] not in enabled:
                dropped.append("%s: tool %r isn't available" % (label, c["tool"]))
                continue
            elif c["tool"] == "file_system" and c["action"] == "none":
                dropped.append("%s: file_system needs an action" % label)
                continue
            elif not c["target"] and not (c["tool"] == "file_system" and c["action"] == "list"):
                dropped.append("%s: no target" % label)
                continue
            elif PLACEHOLDER.search(c["target"]):
                dropped.append("%s: placeholder in %r" % (label, c["target"]))
                continue
            elif any(a["tool"] == c["tool"] and a["action"] == c["action"] and a["target"] == c["target"] for a in done):
                dropped.append("%s: already done" % label)
                continue
            if c["type"] != "tool" or c["tool"] != "file_system" or c["action"] not in ("create", "update"):
                c["content"] = ""
            kept.append(c)
        if not kept:
            kept.append({"type": "answer", "summary": "Reply with what's known so far", "tool": "none",
                         "action": "none", "target": "", "content": ""})
        for i, c in enumerate(kept, 1):
            c["index"] = i
        entry = {"n": len(s["cycles"]) + 1, "goal": squash(out["goal"]) or s["goal"], "candidates": kept, "dropped": dropped,
                 "changed": changed, "predictions": [], "critique": None, "decision": None, "action": None, "skipped": []}
        live.apply(lambda st: st["cycles"].append(entry))

    def _need_cycle(self, s, what):
        if not s["cycles"]:
            raise HarnessError("run the planner before %s" % what)
        return current(s)

    def predict(self, live):
        s = live.state
        entry = self._need_cycle(s, "the predictor")
        if len(entry["candidates"]) == 1:
            return live.apply(lambda st: current(st)["skipped"].append("predictor: only one candidate"))
        out = self._module(live, "predictor", {
            "goal": entry["goal"], "candidates": [view(c) for c in entry["candidates"]],
            "known": drop_none({"observations": focused(s), "memories": self._memories(s), "actions": self._actions(s),
                                "conversation": conversation_view(s)}),
        }, "predictor")
        by = {p["index"]: p for p in out["predictions"]}
        preds = [dict(by[c["index"]], outcome=squash(by[c["index"]]["outcome"]), risk=squash(by[c["index"]]["risk"]))
                 if c["index"] in by else {"index": c["index"], "outcome": "", "helpful": None, "risk": "no prediction given"}
                 for c in entry["candidates"]]
        live.apply(lambda st: current(st).update(predictions=preds))

    def critique(self, live):
        s = live.state
        entry = self._need_cycle(s, "the critic")
        out = self._module(live, "critic", {
            "message": s["message"], "goal": entry["goal"], "observations": focused(s),
            "memories": self._memories(s), "actions": self._actions(s), "conversation": conversation_view(s),
            "candidates": [self._view(entry, c) for c in entry["candidates"]],
        }, "critic")
        result = {"issues": [dict(i, text=squash(i["text"])) for i in out["issues"] if squash(i["text"])],
                  "question": squash(out["question"])}
        live.apply(lambda st: current(st).update(critique=result))

    def decide(self, live):
        s = live.state
        entry = self._need_cycle(s, "the decision")
        cands = entry["candidates"]
        if len(cands) == 1:
            decision = {"choice": 1, "why": "only one candidate", "by": "rule"}
            live.apply(lambda st: current(st)["skipped"].append("decision: only one candidate"))
        elif not self.module_on("decision"):
            predicted = [c for c in cands if (prediction_for(entry, c["index"]) or {}).get("helpful") is not None]
            best = (max(predicted, key=lambda c: prediction_for(entry, c["index"])["helpful"]) if predicted
                    else next((c for c in cands if c["type"] != "ask"), cands[0]))
            decision = {"choice": best["index"], "by": "rule",
                        "why": "most helpful prediction" if predicted else "first candidate that doesn't ask"}
        else:
            out = self._module(live, "decision", {
                "candidates": [self._view(entry, c) for c in cands],
                "critique": entry["critique"], "actions": self._actions(s), "conversation": conversation_view(s)}, "decision")
            decision = {"choice": out["choice"], "why": squash(out["why"]), "by": "model"}
            if not 1 <= out["choice"] <= len(cands):
                best = max(cands, key=lambda c: (prediction_for(entry, c["index"]) or {}).get("helpful") or 0)
                decision = {"choice": best["index"], "by": "rule",
                            "why": "the model chose %s, which isn't a candidate, so the most helpful prediction won" % out["choice"]}
        decision = self._ask_rule(s, entry, decision)
        decision = self._arithmetic_rule(live, entry) or self._router_overrule(live, entry, decision)
        if entry["candidates"][decision["choice"] - 1]["type"] != "tool":  # the router gets first say on a lookup
            decision = self._lookup_rule(live, entry) or self._open_page_rule(live, entry) or decision
        live.apply(lambda st: current(st).update(decision=decision))
        return cands[decision["choice"] - 1]

    def _arithmetic_rule(self, live, entry):
        """A message that is nothing but arithmetic goes to arithmetic, whatever the router or Decision made of it.

        Returns None when it doesn't apply. smROUTER_01 calls "2+2" small talk, and without a Planner nothing
        else would pick a calculator step.
        """
        s = live.state
        expression = bare_arithmetic(s["message"])
        if (not expression or "calculator" not in self.enabled_tools() or any(a["tool"] == "calculator" for a in action_log(s))
                or self._solved(s)):
            return None
        cands = entry["candidates"]
        match = next((c for c in cands if c["tool"] == "calculator"), None)
        if match is None:
            match = {"index": len(cands) + 1, "type": "tool", "summary": "Work out " + expression, "tool": "calculator",
                     "action": "none", "target": expression, "content": ""}
            live.apply(lambda st: current(st)["candidates"].append(match))
        return {"choice": match["index"], "by": "rule", "why": "the message is just arithmetic"}

    def _lookup_rule(self, live, entry):
        """A message about something no model can know — the weather, the news, a price — goes to a search.

        Returns None when it doesn't apply. Without a Planner nothing else proposes a search, and smROUTER_01
        reads a bare number like a zip code as arithmetic, so this rule is what makes "the weather in 60601" work.
        """
        s = live.state
        looked_up = ("web_search", "web_browser")
        if (not needs_lookup(s["message"]) or "web_search" not in self.enabled_tools()
                or any(a["tool"] in looked_up for a in action_log(s))):
            return None
        cands = entry["candidates"]
        match = next((c for c in cands if c["tool"] in looked_up), None)
        if match is None:
            match = {"index": len(cands) + 1, "type": "tool", "summary": "Search the web for this", "tool": "web_search",
                     "action": "none", "target": squash(s["message"])[:300], "content": ""}
            live.apply(lambda st: current(st)["candidates"].append(match))
        return {"choice": match["index"], "by": "rule", "why": "no model can know this on its own, so look it up"}

    def _open_page_rule(self, live, entry):
        """A search comes back as titles and links, so when the answer is on the page, open the top one.

        Returns None when it doesn't apply. Snippets almost never carry the number someone asked for, and a model
        handed nothing but links will invent one rather than say so.
        """
        s = live.state
        log = action_log(s)
        if "web_browser" not in self.enabled_tools() or any(a["tool"] == "web_browser" for a in log):
            return None
        searched = next((a for a in reversed(log) if a["tool"] == "web_search" and a["ok"]), None)
        if not searched or not needs_lookup(s["message"]):
            return None
        link = best_link(searched["result"])
        if not link:
            return None
        cands = entry["candidates"]
        match = next((c for c in cands if c["tool"] == "web_browser"), None)
        if match is None:
            match = {"index": len(cands) + 1, "type": "tool", "summary": "Read the top result", "tool": "web_browser",
                     "action": "none", "target": link, "content": ""}
            live.apply(lambda st: current(st)["candidates"].append(match))
        return {"choice": match["index"], "by": "rule", "why": "a list of links isn't an answer, so read the top page"}

    def _ask_rule(self, s, entry, decision):
        """A question costs the person a turn. Code allows one only for a high-severity problem, and never twice in a row."""
        cands = entry["candidates"]
        instead = instead_of_asking(cands)
        if cands[decision["choice"] - 1]["type"] != "ask" or not instead:
            return decision
        if s.get("history") and s["history"][-1].get("asked"):
            why = "it asked a question last turn, so it doesn't ask another"
        elif entry["critique"] is not None and not any(i["severity"] == "high" for i in entry["critique"]["issues"]):
            why = "the Critic found nothing high-severity, so there's nothing worth asking"
        else:
            return decision
        return {"choice": instead["index"], "by": "rule", "why": why}

    def _router_overrule(self, live, entry, decision):
        """Where the small router is confident, it wins: do a lookup it's sure is needed, and skip a question it's sure isn't.
        It never forces a write: it reads "write a function" as a file write."""
        s = live.state
        r = s.get("route") if self.module_on("router") else None
        if not r:
            return decision
        cands = entry["candidates"]
        pick = cands[decision["choice"] - 1]
        want = ROUTER_TOOLS.get(r["tool"], "none")
        already = any(a["tool"] == want for a in action_log(s))
        if (want in FORCEABLE and want in self.enabled_tools() and r["tool_p"] >= self.setting("router", "force_tool_at")
                and not already and pick["tool"] != want and not (want == "calculator" and self._solved(s))):
            match = next((c for c in cands if c["tool"] == want), None)
            expression = find_expression(s["message"]) if match is None and want == "calculator" else ""
            if match is None and want == "web_search":  # a search target can come straight from the message
                match = {"index": len(cands) + 1, "type": "tool", "summary": "Search the web for this", "tool": "web_search",
                         "action": "none", "target": squash(s["message"])[:300], "content": ""}
            elif expression:  # and so can arithmetic written out in it
                match = {"index": len(cands) + 1, "type": "tool", "summary": "Work out " + expression, "tool": "calculator",
                         "action": "none", "target": expression, "content": ""}
            if match is not None and match["index"] > len(cands):
                live.apply(lambda st: current(st)["candidates"].append(match))
            if match:
                return {"choice": match["index"], "by": "router",
                        "why": "the router is %d%% sure this needs %s" % (round(100 * r["tool_p"]), want)}
        if pick["type"] == "ask" and not r["ask_first"] and r["ask_p"] < self.setting("router", "skip_question_below"):
            act = instead_of_asking(cands)
            if act:
                return {"choice": act["index"], "by": "router",
                        "why": "the router is %d%% sure nothing needs asking" % round(100 * (1 - r["ask_p"]))}
        return decision

    def act(self, live):
        s = live.state
        entry = self._need_cycle(s, "acting")
        if not entry["decision"] or chosen(entry)["type"] != "tool":
            raise HarnessError("the decision isn't a tool, so there's nothing to do")
        pick = chosen(entry)
        record = {"tool": pick["tool"], "action": pick["action"], "target": pick["target"], "ok": False, "result": "",
                  "status": "running", "started": time.time(), "seconds": None}
        live.apply(lambda st: current(st).update(action=record))
        started = time.monotonic()
        try:
            if pick["tool"] == "calculator" and self.math_on():
                result, info = self.work_math(live, pick["target"])
                live.apply(lambda st: record.update(math=info))
            else:
                result = self.run_tool(pick)
            ok = True
        except HarnessError as exc:
            result, ok = str(exc), False
        seconds = round(time.monotonic() - started, 1)
        live.apply(lambda st: record.update(ok=ok, result=result, status="done", seconds=seconds))
        if self.on_event:
            self.on_event({"module": "act", "name": "%s %s" % (pick["tool"], pick["action"] if pick["action"] != "none" else ""),
                           "model": "tool", "status": "done" if ok else "error", "seconds": seconds, "retries": 0,
                           "error": "" if ok else result})

    def run_tool(self, c):
        tool = c["tool"]
        cfg = self.tools.get(tool) or {}
        if tool not in self.enabled_tools():
            raise HarnessError("the %s tool is turned off" % tool)
        if tool == "calculator":
            if not self.tool_on("calculator"):  # arithmetic is up to the Math module right now
                raise HarnessError("the calculator tool is turned off")
            return calc(c["target"])
        if tool in ("web_search", "web_browser"):
            base = self.providers[cfg["provider"]]["url"].rstrip("/")
            if tool == "web_search":
                status, data, _ = http_json("POST", base + "/search", {"query": c["target"][:300], "count": cfg.get("count", 5)}, timeout=90)
                results = data.get("results") if isinstance(data, dict) else None
                if status != 200 or not isinstance(results, list):
                    raise HarnessError("web search failed (HTTP %s)" % status)
                lines = ["%d. %s\n   %s\n   %s" % (i, squash(r.get("title", "")), r.get("url", ""), squash(r.get("snippet", "")))
                         for i, r in enumerate([r for r in results if isinstance(r, dict)][:cfg.get("count", 5)], 1)]
                return "\n".join(lines) or "no results"
            if not re.match(r"^https?://", c["target"], re.I):
                raise HarnessError("web_browser needs a full http(s) address")
            # The browse server fetches, so its SSRF guard keeps LAN and private addresses out of reach.
            status, data, _ = http_json("POST", base + "/fetch", {"url": c["target"], "max_words": cfg.get("max_words", 700)}, timeout=90)
            if not isinstance(data, dict) or not data.get("ok"):
                raise HarnessError("couldn't read %s (%s)" % (c["target"], (data or {}).get("error") or "HTTP %s" % status))
            return str(data.get("text", ""))
        if tool == "file_system":
            return self.files.do(c["action"], c["target"], c.get("content", ""))
        raise HarnessError("no tool called %s" % tool)

    def work_math(self, live, expression, check=None):
        """A calculator step done by the Math module, one whole-number +, - or * at a time.

        With the calculator on (or check=True), every step the model does is checked, and a wrong one is
        replaced by the exact answer. Steps it can't take (division, powers, functions, negative numbers, too
        many digits) go to the calculator, or fail while the calculator is off. Returns (result, what happened).
        """
        calculator = self.tool_on("calculator")
        check = (calculator and self.setting("math", "check")) if check is None else check
        fallback = calculator or check  # the calculator does what the model can't, even when it isn't checking
        choice = self.choice("math")
        info = {"model": choice, "checked": check, "steps": [], "misses": 0}
        entry = {"module": "math", "name": self.modules["math"]["name"], "part": "", "model": choice, "status": "running",
                 "started": time.time(), "seconds": None, "retries": 0, "error": ""}
        live.apply(lambda st: st["trace"].append(entry))
        started = time.monotonic()
        result, error = None, ""
        try:
            try:
                solver = self.local(choice)
            except HarnessError as exc:
                if not fallback:
                    raise
                solver, info["error"] = None, str(exc)

            def step(label, x, y, exact):
                shown = ("%s %s %s" % (number_text(x), label, number_text(y)) if label in SYMBOLS.values()
                         else "%s(%s)" % (label, ", ".join(map(number_text, x))) if x is not None else label)
                record = {"step": shown}
                shaped = whole_numbers(label, x, y) if label in MODEL_OPS else "the Math module only does +, - and *"
                if isinstance(shaped, tuple) and solver is not None and not solver.fits(shaped[0], label, shaped[1]):
                    shaped = "too many digits for the model"
                if isinstance(shaped, str) or solver is None:
                    why = shaped if isinstance(shaped, str) else "the model didn't load"
                    if not fallback:
                        raise HarnessError("can't do %s (%s), and the calculator is off" % (shown, why))
                    record.update(used="calculator", why=why, value=exact())
                else:
                    a, b, places, negate = shaped
                    said = solver.answer(a, label, b)
                    got = None if said is None else Decimal(int(said)).scaleb(-places) * (-1 if negate else 1)
                    truth = MODEL_OPS[label](Decimal(repr(x)), Decimal(repr(y)))
                    record.update(asked="%d%s%d" % (a, label, b), said=said)
                    if got is not None and (not check or got == truth):
                        record.update(used="model", value=as_number(got))
                    elif fallback:
                        info["misses"] += 1
                        record.update(used="calculator", value=as_number(truth),
                                      why="the model wrote %s" % (said if said is not None else "no number"))
                    else:
                        raise HarnessError("wrote no number for %s, and the calculator is off" % shown)
                info["steps"].append(record)
                return record["value"]

            if solver is not None and getattr(solver, "takes", "") == "expression":
                result = self._whole_math(solver, expression, check, fallback, info)
            else:
                result = calc(expression, step)
        except HarnessError as exc:
            error = str(exc)
        seconds = round(time.monotonic() - started, 2)
        live.apply(lambda st: entry.update(status="error" if error else "done", seconds=seconds, error=error))
        if self.on_event:
            self.on_event(entry)
        if error:
            raise HarnessError("Math " + error)
        return result, info

    def _whole_math(self, solver, expression, check, fallback, info):
        """A math model that takes the whole expression and writes out its work. The calculator checks the final answer."""
        plain = plain_arithmetic(expression)
        if plain is None or not solver.fits(plain):
            why = "the model only takes numbers, + - * / and brackets" if plain is None else "the model can't take that expression"
            if not fallback:
                raise HarnessError("can't do %s (%s), and the calculator is off" % (expression, why))
            value = calc(expression)
            info["steps"].append({"step": expression, "used": "calculator", "why": why, "value": value})
            return value
        truth = exact_value(plain)
        out = solver.solve(plain)
        said = out.get("answer")
        info["work"] = list(out.get("work") or [])
        record = {"step": plain, "asked": plain, "said": said}
        try:
            got = Fraction(said) if said is not None else None
        except (ValueError, ZeroDivisionError):
            got = None
        if got is not None and (not check or abs(got - truth) < Fraction(1, 100)):  # it works to 2 decimal places
            record.update(used="model", value=said)
        elif fallback:
            info["misses"] += 1
            record.update(used="calculator", value=exact_text(truth), why="the model wrote %s" % (said if said is not None else "no number"))
        else:
            raise HarnessError("wrote no number for %s, and the calculator is off" % expression)
        info["steps"].append(record)
        return record["value"]

    # ----- Math Language: the problems in a message, worked out into working memory -----

    def math_language_on(self):
        return "math_language" in self.modules and self.module_on("math_language") and self.local_ready("math_language")

    def _solved(self, s):
        return any(p.get("answer") is not None for p in s.get("problems") or [])

    def read_math(self, live):
        """Math Language reads the message and lists its problems. A reader that breaks is noted and skipped."""
        s = live.state
        choice = self.choice("math_language")
        entry = {"module": "math_language", "name": self.modules["math_language"]["name"], "part": "", "model": choice,
                 "status": "running", "started": time.time(), "seconds": None, "retries": 0, "error": ""}
        live.apply(lambda st: st["trace"].append(entry))
        started = time.monotonic()
        problems, error, how_sure = [], "", None
        try:
            reader = self.local(choice)
            if hasattr(reader, "read_with_confidence"):  # a reader that says how sure it was
                found, sure, weakest = reader.read_with_confidence(s["message"])
                how_sure = {"sure": sure, "weakest": weakest, "trusted": True, "note": ""}
            else:
                found, how_sure = reader.read(s["message"]), None
            value = {"problems": found}
            wrong = validate(value, self.contracts["math_language"])
            if wrong:
                raise HarnessError("broke its contract: %s" % "; ".join(wrong[:3]))
            problems = [dict(p, answer=None, by="", error="") for p in normalize(value, self.contracts["math_language"])["problems"]]
            if how_sure and problems:
                # Its shakiest character is the tell: a wrong answer is usually sure everywhere but one spot.
                if how_sure["weakest"] < self.setting("math_language", "trust_above"):
                    how_sure.update(trusted=False, note="too unsure (%.2f) to use" % how_sure["weakest"])
                    problems = []
                elif how_sure["weakest"] < self.setting("math_language", "sure_above"):
                    how_sure["note"] = "not certain (%.2f)" % how_sure["weakest"]
                    problems = [dict(p, unsure=True) for p in problems]
        except HarnessError as exc:
            error = str(exc)
        except Exception as exc:  # a broken checkpoint shouldn't take the turn down
            error = "reader failed: %s" % exc
        seconds = round(time.monotonic() - started, 3)

        def close(st):
            entry.update(status="error" if error else "done", seconds=seconds, error=error)
            if how_sure:
                entry["confidence"] = how_sure["weakest"]
            st["problems"], st["math_language"] = problems, how_sure
        live.apply(close)
        if self.on_event:
            self.on_event(entry)

    def solve_problems(self, live):
        """Each problem in order: the Math module when it's on, else the calculator. Answers land next to their problem."""
        answers = {}
        for i, p in enumerate(list(live.state["problems"])):
            worked = p["expression"]
            try:
                worked = fill_answers(p["expression"], answers)
                if self.math_on():
                    result, info = self.work_math(live, worked)
                    outcome = {"answer": result, "by": "Math", "math": info}
                elif self.tool_on("calculator"):
                    outcome = {"answer": calc(worked), "by": "calculator"}
                else:
                    raise HarnessError("Math and the calculator are both off")
                answers[p["id"]] = outcome["answer"]
            except HarnessError as exc:
                outcome = {"error": str(exc)}
            live.apply(lambda st, i=i, outcome=outcome, worked=worked: st["problems"][i].update(outcome, worked=worked))

    def speak(self, live, note=""):
        s = live.state
        entry = self._need_cycle(s, "the language module")
        if not entry["decision"]:
            raise HarnessError("run the decision before the language module")
        pick = chosen(entry)
        critique = entry["critique"] or {"issues": [], "question": ""}
        make = pick["type"] == "make"
        key = "make_prompt" if make and "make_prompt" in self.module("language") else "prompt"
        text = self._module(live, "language", {
            "message": s["message"], "goal": entry["goal"], "conversation": conversation_view(s),
            "note": note or None, "tools_that_ran": [a["tool"] for a in action_log(s)] or None,
            "decision": {"type": pick["type"], "summary": pick["summary"], "why": entry["decision"]["why"]},
            "question_to_ask": critique["question"] if pick["type"] == "ask" else "",
            "observations": focused(s), "memories": self._memories(s), "tool_results": self._actions(s),
            "problems": self._problems(s),
            "assumptions": [i["text"] for i in critique["issues"] if i["kind"] == "assumption"] if entry["critique"] else None,
            "open_details": [i["text"] for i in critique["issues"] if i["kind"] == "missing"] if make and entry["critique"] else None,
        }, None, key)
        live.apply(lambda st: st.update(reply=text))

    def _hand_back(self, live, limit):
        """One lookup on the Language module's say-so: it writes "NEED: web_search the weather in 60601" instead
        of a reply, the harness runs that tool and asks it to write the reply again.

        It's the last catch for a message the router mislabelled and no rule recognised. Only lookups are allowed,
        never a write, and only once a turn.
        """
        s = live.state
        want = language_need(s["reply"])
        if not want or chosen(current(s))["type"] == "make":  # a made thing can contain anything, including that line
            return
        tool, target = want
        if tool not in FORCEABLE or tool not in self.enabled_tools():
            reason = "the %s tool isn't available, so say what you can without it and what you'd need" % tool
        elif any(a["tool"] == tool for a in action_log(s)):
            reason = "%s already ran this turn, so use the result you were given" % tool
        elif len(s["cycles"]) >= limit:
            reason = "there are no tool steps left this turn, so answer with what you have"
        else:
            reason = ""
        if reason:
            self.speak(live, reason)
        else:
            live.apply(lambda st: st["cycles"].append({
                "n": len(st["cycles"]) + 1, "goal": st["goal"], "dropped": [], "predictions": [], "critique": None,
                "candidates": [{"index": 1, "type": "tool", "summary": "Look this up: " + target, "tool": tool,
                                "action": "none", "target": target, "content": ""}],
                "decision": {"choice": 1, "by": "language", "why": "the Language module asked for %s" % tool},
                "action": None, "skipped": ["planner, predictor, critic and decision: the Language module asked for this"]}))
            self.act(live)
            self.speak(live)
        live.apply(lambda st: st.update(reply=drop_need(st["reply"])))

    def _plain_reply(self, live):
        """No Language module: say the tool result or the chosen step as it is."""
        s = live.state
        entry = current(s)
        results = [a["result"] for a in action_log(s) if a["ok"]]
        solved = [p for p in s.get("problems") or [] if p.get("answer") is not None]
        if not results and solved:  # what Math Language found, with the answers
            text = "\n".join("%s = %s%s" % (p["expression"], p["answer"], " " + p["unit"] if p.get("unit") else "") for p in solved)
        else:
            text = results[-1] if results else (chosen(entry)["summary"] if entry and entry["decision"] else s["message"])
        live.apply(lambda st: st.update(reply=text))

    def _one_candidate(self, live, summary):
        live.apply(lambda st: st["cycles"].append({
            "n": len(st["cycles"]) + 1, "goal": st["goal"], "dropped": [], "predictions": [], "critique": None,
            "decision": None, "action": None, "skipped": [],
            "candidates": [{"index": 1, "type": "answer", "summary": summary, "tool": "none", "action": "none",
                            "target": "", "content": ""}]}))

    def remember(self, live):
        """Keep the facts, preferences and events that got attention. A rule, not a model."""
        s = live.state
        saved = []
        for o in s["observations"]:
            if worth_keeping(o) and (o.get("relevance") is None or o["relevance"] >= FOCUS):
                when = o.get("when", "")
                text = o["text"] + (" (when: %s)" % when if when and when.lower() not in o["text"].lower() else "")
                item = self.memory.add(text, o["kind"], s["id"])
                if item:
                    saved.append({"id": item["id"], "text": item["text"]})
        live.apply(lambda st: st.update(remembered=saved))

    # ----- the cycle -----

    def start(self, message, conversation=None):
        """Working memory for a new turn. Without a conversation id, it starts a new conversation."""
        if conversation:
            data = self.conversations.get(conversation)
            if data is None:
                raise HarnessError("there's no conversation %s" % conversation)
        else:
            data = self.conversations.create()
        return Live(new_state(message, data["id"], data["turns"]))

    def think(self, live):
        s = live.state
        on = self.module_on
        live.apply(lambda st: st.update(switches={"modules": [m for m in self.order if on(m)],
                                                  "tools": [n for n in self.tools if self.tool_on(n)]}))
        try:
            if on("router") and self.router_ready():
                self.route(live)
                r = s["route"]
                if not r["for_me"] and r["intent_p"] >= self.setting("router", "stop_small_talk_at") and not bare_arithmetic(s["message"]):
                    self._finish(live, "stopped", "Router: nothing to reply to (%s, %d%% sure)." % (r["intent"], round(100 * r["intent_p"])))
                    return live
            if on("perception"):
                self.perceive(live)
            else:
                live.apply(lambda st: st.update(goal=squash(st["message"])[:120], observations=[{
                    "id": "o1", "kind": "request", "text": squash(st["message"])[:500], "who": "", "what": "",
                    "when": "", "where": "", "relevance": None}]))
            if on("attention"):
                self.attend(live)
                if not s["attention"]["for_me"]:
                    self._finish(live, "stopped", "Attention: not for me. %s" % s["attention"]["why"])
                    return live
            if on("recall") and self.tool_on("memory"):
                self.recall(live)
            if self.math_language_on() and (mentions_numbers(s["message"]) or not self.setting("math_language", "only_with_numbers")):
                self.read_math(live)
                if s["problems"]:
                    self.solve_problems(live)
            limit = int(self.config.get("max_cycles", 3))
            while True:
                if on("planner"):
                    self.plan(live)
                else:
                    self._one_candidate(live, "Reply directly to the message")
                if on("predictor"):
                    self.predict(live)
                if on("critic"):
                    self.critique(live)
                if self.decide(live)["type"] != "tool":
                    break
                self.act(live)
                if len(s["cycles"]) >= limit:
                    note = "stopped using tools after %d cycles and answered with what it had" % limit
                    live.apply(lambda st: st["cycles"].append({
                        "n": len(st["cycles"]) + 1, "goal": current(st)["goal"], "dropped": [], "predictions": [],
                        "candidates": [{"index": 1, "type": "answer", "summary": "Answer with what the tools found",
                                        "tool": "none", "action": "none", "target": "", "content": ""}],
                        "critique": current(st)["critique"], "decision": {"choice": 1, "why": note, "by": "rule"},
                        "action": None, "skipped": ["planner, predictor, critic and decision: cycle limit"]}))
                    live.apply(lambda st: st.update(note=note))
                    break
            if on("language"):
                self.speak(live)
                self._hand_back(live, limit)
            else:
                self._plain_reply(live)
            if self.tool_on("memory"):
                self.remember(live)
            self._finish(live, "done", s["note"])
        except HarnessError as exc:
            self._finish(live, "error", str(exc))
        except Exception as exc:  # a bug shouldn't leave a page spinning forever
            traceback.print_exc()
            self._finish(live, "error", "harness bug: %s" % exc)
        return live

    def run_stage(self, live, stage):
        """One stage on its own, for shell pipes: working memory in, working memory out."""
        s = live.state
        if stage not in STAGES:
            raise HarnessError("no stage called %r (have: %s)" % (stage, ", ".join(STAGES)))
        if stage in self.modules and not self.module_on(stage):
            raise HarnessError("%s is switched off" % self.modules[stage]["name"])
        if (stage == "act" and not self.enabled_tools()) or (stage == "remember" and not self.tool_on("memory")):
            raise HarnessError("%s needs a tool that's switched off" % stage)
        if stage in ("attention", "recall", "planner") and not s["observations"]:
            raise HarnessError("run perception before %s" % stage)
        {"router": self.route, "perception": self.perceive, "attention": self.attend, "recall": self.recall, "planner": self.plan,
         "predictor": self.predict, "critic": self.critique, "decision": self.decide, "act": self.act,
         "language": self.speak, "remember": self.remember}[stage](live)

    def _finish(self, live, status, note=""):
        final = live.snapshot()[1]
        final.update(status=status, note=note, seconds=round(time.time() - final["started"], 1))
        try:
            write_json(os.path.join(self.runs_dir, final["id"] + ".json"), final)
            if final.get("conversation"):
                self.conversations.append(final["conversation"], final)
        except (OSError, HarnessError):
            pass
        # Only now tell anyone watching that it's done, so a quick follow-up always finds this turn saved.
        live.apply(lambda st: st.update(status=status, note=note, seconds=final["seconds"]))

    # ----- checks -----

    def test_module(self, mid, on_case=None):
        """Run a module's checks with its current model and remember the score for that model."""
        self.module(mid)
        if not self.module_on(mid):  # a switched-off module can't be graded, and a 0 score would stick to its model
            raise HarnessError("%s is switched off, so there's nothing to test" % self.modules[mid]["name"])
        if self.modules[mid].get("kind") in LOCAL_CODE and not self.local_ready(mid):
            raise HarnessError("%s's model isn't trained yet, so there's nothing to test" % self.modules[mid]["name"])
        cases = self.checks.get(mid) or []
        if not cases:
            raise HarnessError("no checks written for %s" % mid)
        choice = self.choice(mid)
        if not choice:
            raise HarnessError("no model picked for %s" % mid)
        results = []
        for case in cases:
            state = new_state(case.get("message", ""))
            for key in ("goal", "observations"):
                if key in case:
                    state[key] = json.loads(json.dumps(case[key]))
            if "cycle" in case:
                base = {"n": 1, "goal": case.get("goal", ""), "candidates": [], "dropped": [], "predictions": [],
                        "critique": None, "decision": None, "action": None, "skipped": []}
                state["cycles"] = [dict(base, **json.loads(json.dumps(case["cycle"])))]
            live = Live(state)
            started = time.monotonic()
            scratch = tempfile.mkdtemp() if mid == "recall" else None
            try:
                if scratch:
                    store = MemoryStore(scratch)
                    for text in case.get("stored", []):
                        store.add(text, "fact", "check")
                    self.recall(live, store)
                elif mid == "math_language":
                    self.read_math(live)
                elif mid == "math":  # always checked, so the score is the model's own and not the calculator's
                    result, info = self.work_math(live, case["message"], check=True)
                    live.apply(lambda st: st.update(math=dict(info, result=result)))
                else:
                    self.run_stage(live, mid)
                ok, why = grade(mid, case, live.state)
            except HarnessError as exc:
                ok, why = False, str(exc)
            finally:
                if scratch:
                    shutil.rmtree(scratch, ignore_errors=True)
            result = {"input": case.get("message", "")[:80], "ok": ok, "why": why,
                      "seconds": round(time.monotonic() - started, 1)}
            results.append(result)
            if on_case:
                on_case(result)
        score = {"passed": sum(r["ok"] for r in results), "total": len(results),
                 "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 1),
                 "when": time.strftime("%Y-%m-%d %H:%M"), "results": results}
        path = os.path.join(self.data_dir, "scores.json")
        with self.lock:
            self.scores = read_json(path, self.scores)
            self.scores.setdefault(mid, {})[choice] = score
            write_json(path, self.scores)
        return choice, score

    def reload_scores(self):
        fresh = read_json(os.path.join(self.data_dir, "scores.json"), None)
        if isinstance(fresh, dict):
            with self.lock:
                self.scores = fresh


def grade(mid, case, s):
    entry = current(s)
    if mid == "router":
        r = s.get("route") or {}
        for key in ("intent", "tool"):
            if case.get(key) and r.get(key) != case[key]:
                return False, "%s was %s" % (key, r.get(key))
    elif mid == "math_language":
        got, want = [p["expression"] for p in s.get("problems") or []], case.get("expected", [])
        if len(got) != len(want):
            return False, "found %d problems (%s), wanted %d" % (len(got), "; ".join(got) or "none", len(want))
        try:
            same = all(abs(a - b) <= abs(b) * 1e-6 + 1e-9 for a, b in zip(worked_values(got), worked_values(want)))
        except HarnessError as exc:
            return False, str(exc)
        if not same:
            return False, "worked out differently: %s" % "; ".join(got)
    elif mid == "math":
        info = s.get("math") or {}
        taken = [st for st in info.get("steps", []) if st["used"] != "model"]
        if taken:
            return False, "%s: %s" % (taken[0]["step"], taken[0]["why"])
        if str(info.get("result")) != str(case["answer"]):
            return False, "got %s" % info.get("result")
    elif mid == "perception":
        kinds = [o["kind"] for o in s["observations"]]
        if case.get("kinds_any") and not set(kinds) & set(case["kinds_any"]):
            return False, "kinds were %s" % (", ".join(kinds) or "none")
        if case.get("when_filled") and not any(o.get("when") for o in s["observations"]):
            return False, "no observation has a when"
        if case.get("text_contains") and not any(case["text_contains"].lower() in o["text"].lower() for o in s["observations"]):
            return False, "no observation mentions %s" % case["text_contains"]
    elif mid == "attention":
        if s["attention"]["for_me"] is not case["for_me"]:
            return False, "for_me was %s" % s["attention"]["for_me"]
    elif mid == "recall":
        if not s["memories"] or case["top_contains"].lower() not in s["memories"][0]["text"].lower():
            return False, "top memory was %s" % (s["memories"][0]["text"] if s["memories"] else "nothing")
    elif mid == "planner":
        tools, types = [c["tool"] for c in entry["candidates"]], [c["type"] for c in entry["candidates"]]
        if case.get("tool") and case["tool"] not in tools:
            return False, "tools were %s" % ", ".join(tools)
        if case.get("type") and case["type"] not in types:
            return False, "candidates were %s" % ", ".join(types)
    elif mid == "predictor":
        if any(p.get("helpful") is None for p in entry["predictions"]) or len(entry["predictions"]) != len(entry["candidates"]):
            return False, "not every candidate got a prediction"
    elif mid == "critic":
        kinds = [i["kind"] for i in entry["critique"]["issues"]]
        if case.get("issue_kinds_any") and not set(kinds) & set(case["issue_kinds_any"]):
            return False, "issues were %s" % (", ".join(kinds) or "none")
        high = [i["text"] for i in entry["critique"]["issues"] if i["severity"] == "high"]
        if case.get("no_high") and high:
            return False, "called it high-severity: %s" % high[0]
    elif mid == "decision":
        if chosen(entry)["type"] != case["choice_type"]:
            return False, "chose a %s" % chosen(entry)["type"]
    elif mid == "language":
        reply = s["reply"].strip()
        if case.get("ends_with_question") and not reply.endswith("?"):
            return False, "didn't end with a question"
        missing = [w for w in case.get("contains", []) if w.lower() not in reply.lower()]
        if missing:
            return False, "missing %s" % ", ".join(missing)
    return True, "ok"


# ----- command line -----

def combine_request(argument, piped):
    argument, piped = (argument or "").strip(), (piped or "").strip()
    return "%s\n\n%s" % (argument, piped) if argument and piped else argument or piped


def print_event(entry):
    mark = "ok" if entry["status"] == "done" else entry["status"].upper()
    extra = " (retried)" if entry.get("retries") else ""
    line = "  %-10s %-20s %5.1fs %s%s" % (entry["name"] + (" " + entry["part"] if entry.get("part") else ""),
                                          entry["model"], entry["seconds"] or 0, mark, extra)
    sys.stderr.write(line + ("  " + entry["error"][:120] if entry.get("error") else "") + "\n")
    sys.stderr.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Think through a message with small cognitive modules.")
    ap.add_argument("message", nargs="*", help="what the person said; text piped in is added after it")
    ap.add_argument("--module", metavar="STAGE", help="run one stage with working memory on stdin: %s" % ", ".join(STAGES))
    ap.add_argument("--show", action="store_true", help="print the reply from working memory on stdin")
    ap.add_argument("--json", action="store_true", help="print all of working memory")
    ap.add_argument("--list", action="store_true", help="show each module's model and the tools")
    ap.add_argument("--use", metavar="MODULE=PROVIDER/MODEL", help="swap a module's model")
    ap.add_argument("--test", metavar="MODULE", help="run a module's checks with its current model")
    ap.add_argument("--memories", action="store_true", help="list long-term memories")
    ap.add_argument("--forget", metavar="ID", help="delete one long-term memory")
    ap.add_argument("--conversation", metavar="ID", help="continue a conversation by its id")
    ap.add_argument("--continue", dest="resume", action="store_true", help="continue the most recent conversation")
    args = ap.parse_args(argv)
    try:
        h = Harness()
        if args.list:
            status = h.status()
            for name, p in h.providers.items():
                print("%-7s %-26s %s" % (name, p.get("label", name), "up" if status[name]["up"] else "DOWN"))
            print()
            for mid in h.order:
                print("%-11s %s" % (mid, h.choice(mid)))
            print("\ntools: %s   (off: %s)" % (", ".join(h.enabled_tools()),
                                             ", ".join(n for n in h.tools if n not in h.enabled_tools()) or "none"))
            print("long-term memories: %d" % h.memory.count())
            return 0
        if args.use:
            mid, _, choice = args.use.partition("=")
            h.choose(mid.strip(), choice.strip())
            print("%s now uses %s" % (mid.strip(), choice.strip()))
            return 0
        if args.test:
            show = lambda r: sys.stderr.write("  %s %s (%.1fs): %s\n" % ("PASS" if r["ok"] else "FAIL", r["input"][:60], r["seconds"], r["why"]))  # noqa: E731
            choice, score = h.test_module(args.test, on_case=show)
            print("%s on %s: %d/%d passed, %.1fs average" % (args.test, choice, score["passed"], score["total"], score["avg_seconds"]))
            return 0 if score["passed"] == score["total"] else 1
        if args.memories:
            for m in h.memory.all():
                print("%s  %s  %s" % (m["id"], m["created"], m["text"]))
            return 0
        if args.forget:
            print("forgot" if h.memory.forget(args.forget) else "no memory with that id")
            return 0
        piped = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()
        if args.show:
            state = json.loads(piped)
            print(state.get("reply") or state.get("note") or "(no reply yet)")
            return 0
        h.on_event = print_event
        if args.module:
            if piped.strip():
                state = json.loads(piped)
                if not isinstance(state, dict) or "message" not in state:
                    raise HarnessError("stdin isn't working memory from another stage")
            else:
                if not args.message:
                    ap.error("give the first stage a message")
                state = new_state(" ".join(args.message))
            live = Live(state)
            h.run_stage(live, args.module)
            print(json.dumps(live.state, indent=1, ensure_ascii=False))
            return 0
        message = combine_request(" ".join(args.message), piped)
        if not message:
            ap.error("say something, or pipe text in")
        conversation = args.conversation
        if args.resume and not conversation:
            recent = h.conversations.recent(1)
            if not recent:
                raise HarnessError("there's no conversation to continue yet")
            conversation = recent[0]["id"]
        live = h.think(h.start(message, conversation))
        s = live.state
        sys.stderr.write("conversation %s (turn %d)\n" % (s["conversation"], len(s["history"]) + 1))
        if args.json:
            print(json.dumps(s, indent=1, ensure_ascii=False))
        elif s["reply"]:
            print(s["reply"])
        if s["status"] != "done" or s["note"]:
            sys.stderr.write("%s in %ss%s\n" % (s["status"], s["seconds"], ": " + s["note"] if s["note"] else ""))
        return {"done": 0, "stopped": 2}.get(s["status"], 1)
    except HarnessError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
