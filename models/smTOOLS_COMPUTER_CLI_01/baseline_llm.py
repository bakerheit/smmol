"""Ministral 8B on the PC, prompted to write the same commands, scored on the same hand-written requests.

    python3 baseline_llm.py
"""
import json
import os
import statistics
import time
import urllib.request

from check import risk_of
from score import score

HERE = os.path.dirname(os.path.abspath(__file__))
# Point this somewhere else with SMMOL_LLM_URL; the committed default is local.
URL = os.environ.get("SMMOL_LLM_URL", "http://127.0.0.1:8081").rstrip("/") + "/v1/chat/completions"
MODEL = "ministral-8b"
SYSTEM = (
    "You write one shell command for the platform given, and nothing else.\n"
    "Platforms: ubuntu and fedora and arch use bash, macos uses zsh with BSD tools, windows uses PowerShell.\n"
    "Use the platform's own package manager and service tools (apt/dnf/pacman/brew/winget, systemctl/launchctl/Get-Service).\n"
    "risk is one of: read (looks only), write (changes files), install, admin (needs root), destructive (hard to undo).\n"
    "If the request isn't something you'd run in a terminal, return an empty command.\n\n"
    'Example, ubuntu: "what is using port 8080" -> {"command":"sudo ss -ltnp \'sport = :8080\'","risk":"read"}\n'
    'Example, macos: "remind me to call mum" -> {"command":"","risk":""}\n\n'
    "Reply with JSON only.")
SCHEMA = {"type": "object", "required": ["command", "risk"], "properties": {
    "command": {"type": "string", "maxLength": 300}, "risk": {"type": "string", "maxLength": 12}}}


def ask(platform, text):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 200, "response_format": {"type": "json_object", "schema": SCHEMA},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "%s: %s" % (platform, text)}]}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        content = json.load(response)["choices"][0]["message"]["content"]
    said = json.loads(content[content.find("{"):content.rfind("}") + 1])
    return str(said.get("command", "")).strip(), str(said.get("risk", "")).strip()


def main():
    with open(os.path.join(HERE, "test.json")) as f:
        test = json.load(f)
    written, seconds = [], []
    for e in test:
        started = time.time()
        try:
            got = ask(e["platform"], e["text"])
        except Exception as exc:  # a timeout or broken reply counts as writing nothing
            print("failed: %s (%s)" % (e["text"], exc), flush=True)
            got = ("", "")
        seconds.append(time.time() - started)
        written.append(got)
        print("%5.1fs  %-8s %-46s %s" % (seconds[-1], e["platform"], e["text"][:46], got[0] or "(nothing)"), flush=True)
    result, mistakes = score(test, written)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "llm_baseline.json"), "w") as f:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M"), "model": MODEL, "score": result,
                   "median_seconds": round(statistics.median(seconds), 2), "mistakes": mistakes}, f, indent=1)
    print("\n%s: exact %.0f%%, right program %.0f%%, right risk %.0f%%, quiet %.0f%%, %.1fs median"
          % (MODEL, 100 * result["exact"], 100 * result["program"], 100 * (result["risk"] or 0), 100 * result["quiet"],
             statistics.median(seconds)))


if __name__ == "__main__":
    main()
