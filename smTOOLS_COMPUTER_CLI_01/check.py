"""Checking commands: fill in the blanks, run them past a real shell parser, and judge how risky they are.

Nothing here runs a command. `bash -n` and `zsh -n` only parse. The danger rules are code, not a model, and
they always win: if a command looks destructive, it's destructive no matter what the model labelled it.
"""
import re
import subprocess

import catalog

SLOT = re.compile(r"\{([a-z_]+)\}")
SAMPLES = {"path": "./logs", "src": "./notes.txt", "dst": "./backup/notes.txt", "name": "*.log", "text": "TODO",
           "port": "8080", "pkg": "htop", "svc": "sshd", "user": "sam", "host": "example.com", "pid": "4242",
           "prog": "git", "var": "PATH", "a": "old", "b": "new", "url": "https://example.com/file.tar.gz"}

# Commands a person must look at twice, whatever the model called them.
DANGER = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf]", "deletes without asking"),
    (r"\brm\s+-rf\s+(/|~|\$HOME|\*)\s*$", "would wipe a whole tree"),
    (r"\bmkfs\b|\bfdisk\b|\bdiskutil\s+erase", "formats a disk"),
    (r"\bdd\s+if=", "writes straight to a device"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"\b(shutdown|reboot|halt|poweroff)\b|Restart-Computer|Stop-Computer", "takes the machine down"),
    (r"chmod\s+-R\s+777", "opens everything to everyone"),
    (r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh", "runs code straight off the internet"),
    (r"Remove-Item[^\n]*-Recurse[^\n]*-Force", "deletes a tree without asking"),
    (r"\bdel\s+/[sq]|\bformat\s+[a-z]:|\breg\s+delete\b", "Windows delete or registry wipe"),
    (r"Format-Volume|Clear-Disk", "formats a disk"),
    (r">\s*/dev/(sd|nvme|disk)", "writes over a disk device"),
]
RISK_RULES = [
    ("destructive", r"\brm\b|\bpkill\b|\bkill\b|Stop-Process|Remove-Item|Restart-Computer|\breboot\b|\bshutdown\b"),
    ("install", r"\b(apt|dnf|pacman|brew|winget)\b[^|]*\b(install|remove|upgrade|uninstall|-S|-Rs|-Syu)\b|pip\s+install"),
    ("admin", r"\bsudo\b|systemctl\s+(start|stop|restart|enable|disable)|launchctl\s+(kickstart|bootout|enable)|"
              r"Set-Service|New-LocalUser|icacls|\bchown\b"),
    ("write", r"\bmkdir\b|\bcp\b|\bmv\b|sed\s+-i|\bzip\b|\bunzip\b|tar\s+-c|tar\s+-x|curl\s+-[a-zA-Z]*O|git\s+clone|"
              r"Set-Content|New-Item|Copy-Item|Move-Item|Compress-Archive|Expand-Archive|Invoke-WebRequest[^|]*-OutFile|"
              r"\bscp\b|python3?\s+-m\s+venv|chmod"),
]


def fill(command, values=None):
    """A command with its blanks filled in. Braces that aren't slots (PowerShell blocks, find's {}) are left alone."""
    values = values or SAMPLES
    for slot in set(SLOT.findall(command)):
        command = command.replace("{%s}" % slot, str(values.get(slot, SAMPLES.get(slot, slot))))
    return command


def dangers(command):
    """Every danger rule a command trips."""
    return [why for pattern, why in DANGER if re.search(pattern, command, re.I)]


def risk_of(command):
    """How risky a command is, worked out from the command itself."""
    if dangers(command):
        return "destructive"
    for risk, pattern in RISK_RULES:
        if re.search(pattern, command, re.I):
            return risk
    return "read"


def parses(command, shell="bash"):
    """Whether a real shell parser accepts the command. PowerShell isn't installed here, so it isn't checked."""
    if shell == "powershell":
        return True, "not checked: no PowerShell on this machine"
    try:
        done = subprocess.run([shell, "-n"], input=command, text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return done.returncode == 0, done.stderr.strip()


def check_catalog():
    """Every problem in the catalog: missing platforms, unknown slots, commands a shell won't parse."""
    problems = []
    for task in catalog.TASKS:
        for key in ("id", "about", "slots", "risk", "say", "cmd"):
            if key not in task:
                problems.append("%s: no %s" % (task.get("id", "?"), key))
        if task["risk"] not in catalog.RISKS:
            problems.append("%s: risk %r isn't one of %s" % (task["id"], task["risk"], catalog.RISKS))
        missing = [p for p in catalog.PLATFORMS if p not in task["cmd"]]
        if missing:
            problems.append("%s: no command for %s" % (task["id"], ", ".join(missing)))
        for phrase in task["say"]:
            unknown = set(SLOT.findall(phrase)) - set(task["slots"])
            if unknown:
                problems.append("%s: phrasing uses %s, which isn't a slot" % (task["id"], ", ".join(sorted(unknown))))
        used = set()
        for platform, command in task["cmd"].items():
            unknown = set(SLOT.findall(command)) - set(task["slots"])
            if unknown:
                problems.append("%s/%s: command uses %s, which isn't a slot" % (task["id"], platform, ", ".join(sorted(unknown))))
            used |= set(SLOT.findall(command))
            ok, why = parses(fill(command), catalog.SHELLS[platform])
            if not ok:
                problems.append("%s/%s: %s won't parse (%s)" % (task["id"], platform, catalog.SHELLS[platform], why))
        for slot in set(task["slots"]) - used:
            problems.append("%s: slot %s is never used in a command" % (task["id"], slot))
    return problems


def main():
    problems = check_catalog()
    tasks, commands = len(catalog.TASKS), len(catalog.TASKS) * len(catalog.PLATFORMS)
    print("%d tasks, %d commands, %d phrasings" % (tasks, commands, sum(len(t["say"]) for t in catalog.TASKS)))
    print("shell-parsed: %d (every Linux and macOS command). PowerShell isn't installed here, so %d are unchecked."
          % (tasks * 4, tasks))
    disagree = [(t["id"], t["risk"], risk_of(fill(t["cmd"]["ubuntu"])))
                for t in catalog.TASKS if risk_of(fill(t["cmd"]["ubuntu"])) != t["risk"]]
    print("\nrisk labels the rules read differently (the rules are stricter on purpose):")
    for task_id, said, rules in disagree:
        print("  %-18s catalog says %-11s rules say %s" % (task_id, said, rules))
    print("\ncommands that trip a danger rule:")
    for task in catalog.TASKS:
        hits = dangers(fill(task["cmd"]["ubuntu"])) or dangers(fill(task["cmd"]["windows"]))
        if hits:
            print("  %-18s %s" % (task["id"], "; ".join(hits)))
    print("\nproblems: %d" % len(problems))
    for problem in problems:
        print("  " + problem)


if __name__ == "__main__":
    main()
