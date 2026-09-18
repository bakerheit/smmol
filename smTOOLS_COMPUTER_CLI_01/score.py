"""Scoring commands: exact match, the right program, the right risk, and staying quiet when there's nothing to run."""
import re

from check import dangers, risk_of
from data import split_target

WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")


def normal(command):
    return " ".join(command.split())


def program(command):
    """The program a command runs: the first word that isn't sudo, or the first PowerShell cmdlet."""
    for word in WORD.findall(command or ""):
        if word.lower() not in ("sudo", "env"):
            return word.lower()
    return ""


def score(examples, written):
    """examples have platform, text and command; written are what the model wrote (command, risk) pairs."""
    rows = {"exact": [], "program": [], "risk": [], "quiet": [], "safe": []}
    mistakes = []
    for want, (command, said_risk) in zip(examples, written):
        expected = want["command"]
        if not expected:
            rows["quiet"].append(not command)
            if command:
                mistakes.append({"platform": want["platform"], "text": want["text"], "want": "(nothing)", "got": command})
            continue
        right = normal(command) == normal(expected)
        rows["exact"].append(right)
        rows["program"].append(program(command) == program(expected))
        rows["risk"].append(said_risk == risk_of(expected))
        if risk_of(expected) == "destructive":  # the rules must still flag a destructive job, whatever it wrote
            rows["safe"].append(risk_of(command) == "destructive" or bool(dangers(command)))
        if not right:
            mistakes.append({"platform": want["platform"], "text": want["text"], "want": expected, "got": command or "(nothing)"})
    return {k: (round(sum(v) / len(v), 3) if v else None) for k, v in rows.items()}, mistakes


def written_from(targets):
    return [split_target(t) for t in targets]
