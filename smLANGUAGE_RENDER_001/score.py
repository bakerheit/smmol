"""Strict automatic checks for grounded English replies."""
import re


NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:[.,:]\d+)*%?")
URL = re.compile(r'''https?://[^\s<>`"'\],}]+''')
FORBIDDEN = re.compile(r"\b(module|json|memory store|tool result)\b", re.I)
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
STOP = frozenset("a an and are as at be by can could did do does for from had has have how i in is it its me my no "
                 "not of on or our should so that the their them there this to was we what when where which who why "
                 "will with would you your answer current data directly first give need requested information result "
                 "available provided".split())


def facts(text, pattern):
    return {x.rstrip(".,)") for x in pattern.findall(text or "")}


def content_words(text):
    return {w.lower().replace("’", "'") for w in WORD.findall(text or "")
            if len(w) >= 3 and w.lower().replace("’", "'") not in STOP}


def grade(example, reply):
    """Return (ok, problems). Exact wording is separate; these checks catch invented claims."""
    reply = (reply or "").strip()
    source = example["prompt"]
    problems = []
    if not reply:
        problems.append("empty")
    if len(reply.split()) > 150:
        problems.append("over 150 words")
    if FORBIDDEN.search(reply):
        problems.append("mentions internals")
    invented_numbers = facts(reply, NUMBER) - facts(source, NUMBER)
    if invented_numbers:
        problems.append("invented numbers: " + ", ".join(sorted(invented_numbers)))
    missing_numbers = facts(example.get("reply", ""), NUMBER) - facts(reply, NUMBER)
    if missing_numbers:
        problems.append("missing numbers: " + ", ".join(sorted(missing_numbers)))
    invented_urls = facts(reply, URL) - facts(source, URL)
    if invented_urls:
        problems.append("invented URLs")
    missing_urls = facts(example.get("reply", ""), URL) - facts(reply, URL)
    if missing_urls:
        problems.append("missing URLs")
    # A canonical target may add connective English, but every actual fact word in it also appears in the source.
    # Require those source-backed words so gibberish or a different name cannot pass merely by avoiding digits.
    required_words = content_words(example.get("reply", "")) & content_words(source)
    missing_words = required_words - content_words(reply)
    if missing_words:
        problems.append("missing facts: " + ", ".join(sorted(missing_words)))
    kind = example.get("kind")
    if kind == "need" and not re.fullmatch(r"NEED: web_browser https?://\S+", reply):
        problems.append("bad NEED line")
    if kind != "need" and re.search(r"^NEED:", reply):
        problems.append("unexpected NEED line")
    if kind == "ask" and reply.count("?") != 1:
        problems.append("ask reply needs exactly one question")
    if kind != "ask" and reply.count("?"):
        problems.append("unexpected question")
    if re.search(r"\b(I can|I(?:'|’)ll|let me) (?:look|search|check|open)\b", reply, re.I):
        problems.append("offers a lookup after the turn")
    return not problems, problems


def summary(examples, replies):
    rows = []
    for example, reply in zip(examples, replies):
        ok, problems = grade(example, reply)
        rows.append({"kind": example.get("kind"), "reply": reply, "expected": example["reply"],
                     "exact": reply.strip() == example["reply"], "grounded": ok, "problems": problems})
    n = max(1, len(rows))
    return {"exact": round(sum(r["exact"] for r in rows) / n, 3),
            "grounded": round(sum(r["grounded"] for r in rows) / n, 3), "rows": rows}
