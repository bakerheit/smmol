#!/usr/bin/env python3
"""Normalise a shell command so two spellings of the same action compare equal.

Phase 3a of docs/engineering/plans/when-small-wins.md asks whether
smTOOLS_COMPUTER_CLI_01's +31.9-point lead on exact command is a capability win or
a grading artefact. Ministral loses points for writing
`sudo apt update && sudo apt install -y nginx` where the answer key says
`sudo apt install -y nginx`. Both install nginx.

The rules here are fixed BEFORE any number is computed, because they decide the
headline. They are deliberately narrow: normalisation may only forgive a difference
in *spelling*, never a difference in *what the command does*.

WHAT IS FORGIVEN
  - a leading package-index refresh (`apt update &&`, `dnf check-update ;` ...)
  - flag order within one program, including flags written after positionals
  - documented long/short equivalents, listed explicitly in EQUIVALENT_FLAGS
  - quoting style, where the shell word-splits to the same argv
  - `.` vs `./` for the current directory, and a trailing slash on a path

WHAT IS NOT FORGIVEN
  - a different program (`top` for `ps`, `tail` for `journalctl`)
  - a different subcommand (`systemctl is-active` for `systemctl status`)
  - an added or removed flag that changes output or behaviour (`-r` for `-a`,
    `--now`, `-ls`, `-l` for `-n`)
  - an added or removed pipeline stage (`df -h` vs `df -h / | awk ...`)
  - a different path or target (`./build` vs `/path/to/your/project/build`)
  - adding or removing `sudo`, which changes privilege
  - an inverted action (`install` vs `remove`)

Normalising can only turn a wrong answer right, never a right answer wrong, so
re-scoring the stored `mistakes` lists is exact and complete — neither model needs
to be run again.
"""

import re
import shlex

# Long/short pairs that are the same flag. Kept explicit rather than guessed: an
# unlisted pair is treated as a real difference.
EQUIVALENT_FLAGS = {
    "--yes": "-y",
    "--assume-yes": "-y",
    "--all": "-a",
    "--recursive": "-r",
    "--human-readable": "-h",
    "--line-number": "-n",
    "--force": "-f",
    "--quiet": "-q",
    "--verbose": "-v",
}

# A leading refresh of the package index. Dropping it cannot change what the rest
# of the line does; it only makes it more likely to succeed.
REFRESH = re.compile(
    r"^\s*(sudo\s+)?("
    r"apt(-get)?\s+update"
    r"|dnf\s+(check-update|makecache)"
    r"|yum\s+(check-update|makecache)"
    r"|pacman\s+-Sy+"
    r"|brew\s+update"
    r"|apk\s+update"
    r")\s*(&&|;)\s*",
    re.I,
)


def strip_refresh(command):
    """Drop any number of leading package-index refreshes."""
    previous = None
    while previous != command:
        previous = command
        command = REFRESH.sub("", command, count=1)
    return command.strip()


def split_pipeline(command):
    """Split on pipes and separators that are not inside quotes."""
    parts, current, quote, i = [], [], None, 0
    while i < len(command):
        ch = command[i]
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            current.append(ch)
            quote = ch
        elif ch == "|" and command[i:i + 2] != "||":
            parts.append("".join(current))
            current = []
        elif command[i:i + 2] in ("&&", "||"):
            parts.append("".join(current))
            current = []
            i += 1
        elif ch == ";":
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def canonical_path(word):
    if word in (".", "./"):
        return "."
    if len(word) > 1 and word.endswith("/"):
        return word.rstrip("/")
    if word.startswith("./") and len(word) > 2:
        return word[2:]
    return word


def normalise_stage(stage):
    """One pipeline stage to a comparable tuple: (program, subcommand, flags, args).

    Flags are sorted, so order stops mattering; positionals keep their order,
    because `cp a b` is not `cp b a`. `sudo` is preserved as part of the program
    identity, since dropping it changes privilege.
    """
    try:
        words = shlex.split(stage)
    except ValueError:
        words = stage.split()
    if not words:
        return ()

    prefix = []
    while words and words[0] in ("sudo", "doas", "command", "env"):
        prefix.append(words.pop(0))
    if not words:
        return tuple(prefix)

    program = words[0]
    rest = words[1:]

    # A leading bare word is the subcommand (git clone, systemctl status, apt install).
    subcommand = ""
    if rest and not rest[0].startswith("-"):
        if program in ("git", "systemctl", "apt", "apt-get", "dnf", "yum", "pacman",
                       "brew", "docker", "npm", "pip", "pip3", "systemd-analyze", "journalctl"):
            subcommand = rest[0]
            rest = rest[1:]

    flags, args = [], []
    for word in rest:
        if word.startswith("-") and word != "-":
            key, _, value = word.partition("=")
            key = EQUIVALENT_FLAGS.get(key, key)
            # -y -q written together as -yq is the same set as -y -q.
            if re.fullmatch(r"-[A-Za-z]{2,}", key) and not key.startswith("--"):
                flags.extend("-" + letter for letter in key[1:])
            else:
                flags.append(key + ("=" + value if value else ""))
        else:
            args.append(canonical_path(word))

    return (tuple(prefix), program, subcommand, tuple(sorted(flags)), tuple(args))


def normalise(command):
    if not command:
        return ()
    return tuple(normalise_stage(s) for s in split_pipeline(strip_refresh(command)))


def same(a, b):
    """True when two commands differ only in spelling."""
    return normalise(a) == normalise(b)


if __name__ == "__main__":
    checks = [
        # forgiven
        ("sudo apt update && sudo apt install -y nginx", "sudo apt install -y nginx", True),
        ("sudo dnf install ripgrep -y", "sudo dnf install -y ripgrep", True),
        ("grep --line-number 'TODO' src", "grep -n 'TODO' src", True),
        ("rm -rf ./build", "rm -rf build", True),
        ("ls -l -a", "ls -al", True),
        # not forgiven
        ("systemctl is-active docker", "systemctl status docker", False),
        ("uname -r", "uname -a", False),
        ("top -o %CPU -n 1 | head -n 15", "ps aux --sort=-%cpu | head -10", False),
        ("df -h / | awk 'NR==2 {print $4}'", "df -h", False),
        ("sudo systemctl enable --now postgresql", "sudo systemctl enable postgresql", False),
        ("sudo rm -rf /path/to/project/build", "rm -rf ./build", False),
        ("grep -rl 'TODO' src", "grep -rn 'TODO' src", False),
        ("apt remove nginx", "apt install nginx", False),
    ]
    bad = 0
    for a, b, want in checks:
        got = same(a, b)
        if got != want:
            bad += 1
            print("FAIL want=%s got=%s\n  %s\n  %s" % (want, got, a, b))
    print("normalise self-check: %d/%d" % (len(checks) - bad, len(checks)))
    raise SystemExit(1 if bad else 0)
