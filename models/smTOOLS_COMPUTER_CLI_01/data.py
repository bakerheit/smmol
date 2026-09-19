"""Training messages for smTOOLS_COMPUTER_CLI_01: a platform plus a request, and the command it should write.

    ubuntu: what's using port 8080   ->   sudo ss -ltnp 'sport = :8080'|read

The target is the command, a "|", and how risky it is. An empty target means the request isn't a terminal job
at all, so the model should write nothing.

Everything is built from catalog.py, so the catalog stays the single place where commands are decided.
"""
import random

import catalog
from check import SLOT, risk_of

GREETINGS = ["hey ", "quick one: ", "ok so ", "can you ", "how do I ", "what's the command to ", "um ", "yo "]
ENDINGS = [" please", " thanks", " for me", "?", " real quick"]

POSIX_PATHS = ["./logs", "/var/log/syslog", "~/Downloads", "src", "./build", "/etc/hosts", "~/notes.txt", "data.csv",
               "/tmp/session.log", "./dist", "~/projects/app", "report.pdf", "/opt/tools", "./backup"]
WINDOWS_PATHS = [".\\logs", "C:\\Users\\sam\\Documents", "$HOME\\Downloads", ".\\build", "C:\\temp\\old.txt", "notes.txt",
                 "C:\\ProgramData\\app\\app.log", ".\\dist", "data.csv", "C:\\Users\\sam\\projects"]
VALUES = {
    "name": ["*.log", "notes.txt", "config.yaml", "*.png", "Dockerfile", "*.csv", "main.py"],
    "text": ["TODO", "password", "ERROR", "api_key", "deprecated", "timeout"],
    "port": ["22", "80", "443", "3000", "5432", "8000", "8080", "8081", "8771", "9000", "1433", "6379"],
    "pkg": ["htop", "git", "curl", "nginx", "ripgrep", "node", "vim", "jq", "ffmpeg", "wget", "tmux", "docker", "postgresql"],
    "svc": ["sshd", "nginx", "docker", "postgresql", "cron", "bluetooth", "apache2", "redis"],
    "user": ["sam", "alex", "deploy", "penny", "www-data", "jordan"],
    "host": ["example.com", "github.com", "192.168.1.50", "homelab.local", "8.8.8.8", "google.com"],
    "pid": ["412", "1337", "24601", "8891", "77"],
    "prog": ["git", "python3", "node", "docker", "curl", "ffmpeg", "code"],
    "var": ["PATH", "HOME", "EDITOR", "JAVA_HOME", "SHELL", "USER"],
    "url": ["https://example.com/file.tar.gz", "https://github.com/user/repo.git", "http://localhost:8080/health",
            "https://example.org/data.csv"],
    "a": ["old", "localhost", "8080", "TODO"],
    "b": ["new", "127.0.0.1", "9090", "DONE"],
    "pkg_windows": ["Git.Git", "Microsoft.PowerToys", "7zip.7zip", "Mozilla.Firefox", "Python.Python.3.12"],
}
NOT_TERMINAL = [
    "what's the weather in Denver tomorrow", "remember my dentist is Dr. Lee", "2+2", "what's 17.5% of 2,340",
    "write me a poem about autumn", "who won the game last night", "thanks!", "lol ok", "how does a mortgage work",
    "summarize this article for me", "what should I make for dinner", "I have a meeting on Friday",
    "tell me a joke", "what time is it in Tokyo", "draft an email to my landlord", "explain how DNS works",
    "what's a good name for a dog", "is it going to rain", "add milk to my shopping list", "how do I center a div",
    "my flight is at 7:45 tomorrow", "call me at 555-1234", "what year did the Berlin Wall fall",
    "can you book me a table for saturday", "how many cups in a liter", "sounds good", "what's your favorite color",
]


def value(r, slot, platform):
    if slot in ("path", "src", "dst"):
        return r.choice(WINDOWS_PATHS if platform == "windows" else POSIX_PATHS)
    if slot == "pkg" and platform == "windows":
        return r.choice(VALUES["pkg_windows"])
    return r.choice(VALUES[slot])


def roughen(r, text):
    if r.random() < 0.18:
        text = r.choice(GREETINGS) + text
    if r.random() < 0.12:
        text = text.rstrip("?.! ") + r.choice(ENDINGS)
    if r.random() < 0.3:
        text = text.rstrip("?.!")
    if r.random() < 0.06:  # a typo in the letters, never in a path, number or flag
        spots = [i for i in range(1, len(text) - 2) if text[i].isalpha() and text[i + 1].isalpha() and text[i - 1] != "/"]
        if spots:
            i = r.choice(spots)
            text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if r.random() < 0.35:
        text = text[:1].upper() + text[1:]
    elif r.random() < 0.25:
        text = text.lower()
    return text


def example(r, platform=None, task=None):
    platform = platform or r.choice(catalog.PLATFORMS)
    if task is None and r.random() < 0.2:  # a fifth of the messages are nothing to do with the terminal
        return {"platform": platform, "text": roughen(r, r.choice(NOT_TERMINAL)), "target": ""}
    task = task or r.choice(catalog.TASKS)
    filled = {slot: value(r, slot, platform) for slot in task["slots"]}
    text = r.choice(task["say"])
    command = task["cmd"][platform]
    for slot, chosen in filled.items():
        text = text.replace("{%s}" % slot, chosen)
        command = command.replace("{%s}" % slot, chosen)
    return {"platform": platform, "text": roughen(r, text), "target": "%s|%s" % (command, risk_of(command)),
            "task": task["id"], "risk": risk_of(command)}


def generate(n, seed=0):
    r = random.Random(seed)
    return [example(r) for _ in range(n)]


def prompt(platform, text):
    """What the model reads."""
    return "%s: %s" % (platform, text)


def split_target(target):
    """(command, risk) from what the model wrote, or ("", "") when it wrote nothing."""
    if not target.strip():
        return "", ""
    command, _, risk = target.rpartition("|")
    return (command.strip(), risk.strip()) if command else (target.strip(), "")


def normal(text):
    return " ".join(text.lower().strip().rstrip("?.!").split())


def without(examples, held):
    """Drop generated messages that match a hand-written test message word for word, on the same platform."""
    banned = {(e["platform"], normal(e["text"])) for e in held}
    return [e for e in examples if (e["platform"], normal(e["text"])) not in banned]


if __name__ == "__main__":
    import statistics
    from collections import Counter
    pool = generate(20000, seed=0)
    for e in generate(14, seed=3):
        print("%-8s %-52s -> %s" % (e["platform"], e["text"][:52], e["target"] or "(nothing to run)"))
    lengths = [len(prompt(e["platform"], e["text"])) + len(e["target"]) for e in pool]
    print("\nplatforms:", dict(Counter(e["platform"] for e in pool)))
    print("risks:", dict(Counter(e.get("risk", "none") for e in pool)))
    print("longest command:", max(len(e["target"]) for e in pool), "| median length:", statistics.median(lengths),
          "| longest:", max(lengths))
