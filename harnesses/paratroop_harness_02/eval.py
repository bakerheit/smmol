#!/usr/bin/env python3
"""Run the same conversations through paratroop_harness_01 (pipes) and _02 (cognitive modules) and compare.

Each harness gets the messages of a scenario in order, as separate conversations. v02 runs in a
scratch folder, so its memory and files start empty and nothing leaks into the real ones.

    python3 eval.py                     # everything
    python3 eval.py --only percent,memory --skip-v01
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V01 = os.path.join(os.path.dirname(HERE), "paratroop_harness_01")


def load(name, folder):
    spec = importlib.util.spec_from_file_location(name, os.path.join(folder, "harness.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_v02(v02, messages, scratch):
    h = v02.Harness(HERE, data_dir=scratch)
    for name in ("choices.json",):  # use the same model picks as the real harness
        if os.path.exists(os.path.join(HERE, name)):
            shutil.copy(os.path.join(HERE, name), scratch)
            h = v02.Harness(HERE, data_dir=scratch)
    out = []
    for message in messages:
        s = h.think(v02.Live(v02.new_state(message))).state
        tools = [c["action"]["tool"] for c in s["cycles"] if c.get("action")]
        out.append({"reply": s["reply"], "status": s["status"], "note": s["note"], "seconds": s["seconds"],
                    "calls": len(s["trace"]), "tools": tools, "id": s["id"],
                    "critic": [i["kind"] + ": " + i["text"] for c in s["cycles"] if c.get("critique") for i in c["critique"]["issues"]]})
    return out


def run_v01(v01, messages):
    h = v01.Harness(V01)
    out = []
    for message in messages:
        ids = h.parse_pipe(h.flow)
        data = h.execute(v01.Run(message, "flow", ids), ids).snapshot()[1]
        final = data["final"] or {}
        out.append({"reply": final.get("text") or "", "status": data["status"], "note": data["note"], "seconds": data["seconds"],
                    "calls": sum(1 for st in data["steps"] if st["status"] != "skipped"),
                    "tools": [st["function"] for st in data["steps"] if st["kind"] in ("browse", "draw") and st["status"] == "done"],
                    "id": data["id"], "critic": []})
    return out


def grade(expect, last):
    reply = last["reply"].lower()
    failures = []
    if expect.get("stopped") and last["status"] != "stopped":
        failures.append("didn't stop")
    if not expect.get("stopped") and last["status"] != "done":
        failures.append("status %s" % last["status"])
    missing = [w for w in expect.get("reply_contains", []) if w.lower() not in reply]
    if missing:
        failures.append("reply missing %s" % ", ".join(missing))
    if expect.get("reply_contains_any") and not any(w.lower() in reply for w in expect["reply_contains_any"]):
        failures.append("reply mentions none of %s" % ", ".join(expect["reply_contains_any"]))
    if expect.get("asks_or_assumes") and "?" not in reply and "assum" not in reply:
        failures.append("neither asked nor stated an assumption")
    if expect.get("uses_web") and not set(last["tools"]) & {"web_search", "web_browser", "lookup"}:
        failures.append("didn't use the web")
    return not failures, "; ".join(failures) or "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="comma-separated scenario ids")
    ap.add_argument("--skip-v01", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(HERE, "eval.json")) as f:
        scenarios = json.load(f)
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in wanted]
    v02 = load("harness02", HERE)
    v01 = None if args.skip_v01 else load("harness01", V01)
    results = []
    for sc in scenarios:
        row = {"id": sc["id"], "about": sc["about"]}
        scratch = tempfile.mkdtemp(prefix="paratroop-eval-")
        try:
            for name, runner in (("v01", lambda: run_v01(v01, sc["messages"])) if v01 else (None, None),
                                 ("v02", lambda: run_v02(v02, sc["messages"], scratch))):
                if not name:
                    continue
                sys.stderr.write("%s %s ...\n" % (sc["id"], name))
                started = time.time()
                runs = runner()
                ok, why = grade(sc["expect"], runs[-1])
                row[name] = {"ok": ok, "why": why, "seconds": round(time.time() - started, 1), "runs": runs}
                sys.stderr.write("  %s in %.0fs: %s\n" % ("PASS" if ok else "FAIL", row[name]["seconds"], why))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        results.append(row)
    out = os.path.join(HERE, "eval-results.json")
    with open(out, "w") as f:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M"), "results": results}, f, indent=1, ensure_ascii=False)
    print("| scenario | v01 | v02 |")
    print("|---|---|---|")
    for row in results:
        cells = []
        for name in ("v01", "v02"):
            r = row.get(name)
            cells.append("%s %s, %ss" % ("PASS" if r["ok"] else "FAIL", "" if r["ok"] else "(" + r["why"] + ")", r["seconds"]) if r else "not run")
        print("| %s | %s | %s |" % (row["id"], cells[0], cells[1]))
    print("\nfull results: %s" % out)


if __name__ == "__main__":
    main()
