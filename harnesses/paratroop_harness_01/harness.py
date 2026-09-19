#!/usr/bin/env python3
"""paratroop_harness_01: several small local models, each doing one brain function, joined by pipes.

The default flow follows docs/harness.md:

    attention | context | route | respond

Attention asks "is this for me?". Context breaks the message into tokens of meaning
([User][meeting][this Friday]...). Route is the hub: it decides which @process and @tool calls
to make, and they run right there. Respond writes the reply from the shared whiteboard. Every
function's model can be swapped, and you can pipe functions yourself.

    python3 harness.py "I have a meeting this Friday to discuss an investment opportunity"
    python3 harness.py --pipe "lookup | draw" "the Chicago skyline at night"
    ./pt context "dentist next Tuesday at 3pm" | ./pt planning
"""
import argparse
import base64
import http.client
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_ON = {"openai": ("gate", "chat", "route", "vision"), "browse": ("browse",), "sd": ("draw",)}  # provider kind -> function kinds
ANSWER_KINDS = ("chat", "browse", "draw", "vision")
QUIET_KINDS = ("gate", "route")  # their outputs are decisions, not material for a reply
BOARD_CHARS = 4000
MAX_CALLS = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
THINK = re.compile(r"<think>.*?</think>", re.S)
CALL = re.compile(r"@(process|tool):([A-Za-z][\w-]*)[ \t]*([^\n]*)")
TOKEN = re.compile(r"\[([^\[\]\n]{1,60})\]")
PLACEHOLDER = re.compile(r"\[[^\]]*\]|<[^>]*>")  # "[user's industry]", "<topic>"

LOOKUP_WRITER_PROMPT = ("Turn the text into one short web search question (under 20 words) that finds what the "
                        "request needs. Reply with the question only.")
DRAW_WRITER_PROMPT = ("Turn the text into a Stable Diffusion prompt for one picture: under 40 words of comma-separated "
                      "visual details (subject, setting, style, lighting). Reply with the prompt only, no sentences "
                      "or quotes.")


class HarnessError(Exception):
    """Something a person should read: a server down, a bad pipe, a bad model choice."""


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
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def new_id():
    return time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex()


def squash(text):
    return " ".join(str(text).split())


def today():
    return time.strftime("%A, %B %d, %Y").replace(" 0", " ")


def fill(prompt):
    return str(prompt).replace("{today}", today())


def combine_request(argument, piped):
    """A shell-pipe stage's request: the words you typed, then whatever came down the pipe."""
    argument, piped = (argument or "").strip(), (piped or "").strip()
    if argument and piped:
        return "%s\n\n%s" % (argument, piped)
    return argument or piped


def whiteboard(steps, skip_text=None, budget=BOARD_CHARS):
    """Earlier results for the next function. When space runs out, the newest ones win."""
    parts, used = [], 0
    for step in reversed(steps):
        out = (step.get("output") or "").strip()
        if step.get("status") != "done" or not out or out == skip_text or step.get("kind") in QUIET_KINDS:
            continue
        room = budget - used
        if room <= 80:
            break
        block = ("[%s, step %d]\n%s" % (step["name"], step["n"], out))[:room]
        parts.append(block)
        used += len(block)
    return "\n\n".join(reversed(parts))


class Run:
    """One request's whiteboard. Every change bumps a version so a page can follow along."""

    def __init__(self, request, mode, pipe=None, image=None):
        self.cond = threading.Condition()
        self.version = 0
        self.data = {"id": new_id(), "request": request, "mode": mode, "pipe": list(pipe or []), "image": image,
                     "steps": [], "status": "running", "note": "", "final": None, "started": time.time(),
                     "seconds": None}

    @property
    def id(self):
        return self.data["id"]

    def change(self, fn):
        with self.cond:
            result = fn(self.data)
            self.version += 1
            self.cond.notify_all()
            return result

    def snapshot(self):
        with self.cond:
            return self.version, json.loads(json.dumps(self.data))

    def wait(self, version, timeout):
        with self.cond:
            self.cond.wait_for(lambda: self.version != version, timeout)
            return self.version


class Harness:
    def __init__(self, folder=HERE):
        self.folder = folder
        self.config = read_json(os.path.join(folder, "harness.json"), None)
        if not self.config:
            raise HarnessError("no readable harness.json in %s" % folder)
        self.providers = self.config["providers"]
        self.functions = {fn["id"]: fn for fn in self.config["functions"]}
        self.order = [fn["id"] for fn in self.config["functions"]]
        self.flow = self.config["flow"]
        self.router = next(fid for fid in self.order if self.functions[fid]["kind"] == "route")
        self.runs_dir = os.path.join(folder, "runs")
        self.lock = threading.Lock()
        self.choices = read_json(os.path.join(folder, "choices.json"), {})
        self.scores = read_json(os.path.join(folder, "scores.json"), {})
        self.checks = read_json(os.path.join(folder, "checks.json"), {})
        self._status = (0.0, None)

    # ----- slots and models -----

    def slots(self):
        return list(self.order)

    def kind(self, slot):
        if slot not in self.functions:
            raise HarnessError("no function called %r (have: %s)" % (slot, ", ".join(self.order)))
        return self.functions[slot]["kind"]

    def choice(self, slot):
        if slot in self.choices:
            return self.choices[slot]
        fn = self.functions[slot]
        return "%s/%s" % (fn["provider"], fn["model"]) if fn.get("provider") and fn.get("model") else None

    def provider_of(self, choice):
        name, _, model = str(choice or "").partition("/")
        if name not in self.providers or not model:
            raise HarnessError("%r isn't a provider/model pair" % choice)
        return self.providers[name], model

    def find(self, name):
        key = re.sub(r"[\s_@-]+", "", str(name).lower())
        for fid, fn in self.functions.items():
            names = [fid, fn["name"], str(fn.get("call", ""))] + list(fn.get("aliases", []))
            if key and key in (re.sub(r"[\s_@-]+", "", n.lower()) for n in names):
                return fid
        return None

    def find_call(self, space, name):
        """The function behind @process:<name> or @tool:<name>, or None."""
        want = "@%s:%s" % (space.lower(), name.lower())
        for fid, fn in self.functions.items():
            if str(fn.get("call", "")).lower() == want:
                return fid
        fid = self.find(name)
        return fid if fid and self.functions[fid].get("group") == space.lower() else None

    def parse_pipe(self, spec):
        ids = []
        for part in str(spec).split("|"):
            fid = self.find(part.strip())
            if not fid:
                raise HarnessError("no function called %r in the pipe (have: %s)"
                                   % (part.strip(), ", ".join(self.order)))
            ids.append(fid)
        return ids

    def provider_status(self, name, timeout=4):
        p = self.providers[name]
        base = p["url"].rstrip("/")
        fixed = [{"id": p.get("model", p["kind"])}]
        try:
            if p["kind"] == "openai":
                status, data, _ = http_json("GET", base + "/v1/models", timeout=timeout)
                listed = data.get("data", []) if isinstance(data, dict) else []
                models = [{"id": str(m["id"]), "loaded": m.get("loaded")}
                          for m in listed if isinstance(m, dict) and m.get("id")]
                return {"up": status == 200, "models": models}
            status, _, _ = http_json("GET", base + ("/health" if p["kind"] == "browse" else "/"), timeout=timeout)
            return {"up": status == 200, "models": fixed}
        except HarnessError as exc:
            return {"up": False, "models": [] if p["kind"] == "openai" else fixed, "error": str(exc)}

    def status(self, max_age=15.0):
        """What every provider offers right now, checked in parallel and cached briefly."""
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

    def options(self, slot, status=None):
        status = self.status() if status is None else status
        kind = self.kind(slot)
        out = []
        for name, p in self.providers.items():
            if kind not in RUNS_ON.get(p["kind"], ()):
                continue
            st = status.get(name) or {"up": False, "models": []}
            for m in st["models"]:
                if kind == "vision" and m["id"] not in p.get("vision_models", []):
                    continue  # only models known to see images
                out.append({"choice": "%s/%s" % (name, m["id"]), "provider": name, "label": p.get("label", name),
                            "model": m["id"], "up": st["up"], "loaded": m.get("loaded")})
        return out

    def choose(self, slot, choice):
        kind = self.kind(slot)
        p, _ = self.provider_of(choice)
        if kind not in RUNS_ON.get(p["kind"], ()):
            raise HarnessError("%s can't run on %s" % (slot, p.get("label", choice)))
        offered = [o["choice"] for o in self.options(slot, self.status(max_age=0)) if o["up"]]
        if choice not in offered:
            raise HarnessError("%s isn't offered for %s right now" % (choice, slot))
        with self.lock:
            self.choices[slot] = choice
            write_json(os.path.join(self.folder, "choices.json"), self.choices)

    # ----- calling models -----

    def chat(self, choice, system, user, max_tokens=900, temperature=0.3, timeout=600):
        p, model = self.provider_of(choice)
        status, data, headers = http_json("POST", p["url"].rstrip("/") + "/v1/chat/completions", {
            "model": model, "stream": False, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, timeout=timeout)
        try:
            text = data["choices"][0]["message"]["content"]
        except (TypeError, KeyError, IndexError):
            detail = data.get("error") if isinstance(data, dict) else None
            raise HarnessError("%s gave no answer (HTTP %s%s)" % (choice, status, ": %s" % str(detail)[:200] if detail else ""))
        meta = {key: headers[h] for key, h in (("served_by", "x-gateway-model"), ("waited_s", "x-gateway-waited"))
                if h in headers}
        return THINK.sub("", str(text or "")).strip(), meta

    def _rewrite(self, instruction, text, run):
        """The router's model turns a long or second-hand input into what a tool needs."""
        reply, _ = self.chat(self.choice(self.router), instruction,
                             "The request: %s\n\nText:\n%s" % (run.data["request"], text[:BOARD_CHARS]),
                             max_tokens=80, temperature=0.2)
        return squash(reply).strip('"') or squash(text)

    # ----- the functions -----

    def _gate(self, fn, choice, text, run, step, board):
        reply, meta = self.chat(choice, fill(fn["prompt"]), "Message:\n%s\n\nAnswer YES or NO." % text,
                                max_tokens=3, temperature=0)
        meta["pass"] = reply.upper().lstrip("*# ").startswith("Y")
        verdict = "YES: this is for me" if meta["pass"] else "NO: this doesn't need me"
        return {"output": verdict, "meta": meta, "forward": text}

    def _chat(self, fn, choice, text, run, step, board):
        request = run.data["request"]
        parts = []
        if board or text != request:
            parts.append("The request: %s" % request)
        if board:
            parts.append("Whiteboard (what other functions found):\n%s" % board)
        if text != request:
            parts.append("Your input:\n%s" % text)
        reply, meta = self.chat(choice, fill(fn["prompt"]), "\n\n".join(parts) if parts else text,
                                max_tokens=fn.get("max_tokens", 900), temperature=fn.get("temperature", 0.3))
        if not reply:
            raise HarnessError("%s replied with nothing" % choice)
        if fn.get("parse") == "tokens":
            meta["tokens"] = [squash(t) for t in TOKEN.findall(reply)][:24]
            for label in ("when", "intent"):
                found = re.search(r"^\W*%s:\s*(.+)$" % label, reply, re.M | re.I)
                if found:
                    meta[label] = squash(found.group(1))[:120]
        return {"output": reply, "meta": meta}

    def _route(self, fn, choice, text, run, step, board):
        board = whiteboard([s for s in run.data["steps"] if s is not step])  # all of it, context tokens included
        has_image = bool(run.data.get("image"))
        menu = "\n".join("%s  (%s)" % (f["call"], f.get("does", ""))
                         for f in (self.functions[i] for i in self.order)
                         if f.get("call") and (has_image or f["kind"] != "vision"))
        reply, meta = self.chat(choice, fill(fn["prompt"]).replace("{menu}", menu),
                                "Request:\n%s\n\nWhiteboard:\n%s\n\nImage attached: %s"
                                % (run.data["request"], board or "(empty)", "yes" if run.data.get("image") else "no"),
                                max_tokens=200, temperature=0)
        calls, unknown, dropped, seen = [], [], [], set()
        for m in CALL.finditer(reply):
            line = "@%s:%s" % (m.group(1).lower(), m.group(2).lower())
            arg = m.group(3).strip().strip('"')
            fid = self.find_call(m.group(1), m.group(2))
            if not fid:
                unknown.append(line)
                continue
            # Small routers ignore rules written in prompts, so the two that matter most are checked here.
            if PLACEHOLDER.search(arg):
                dropped.append("%s %s (placeholder in the argument)" % (line, arg))
                continue
            only_if = self.functions[fid].get("only_if")
            if only_if and not re.search(only_if, run.data["request"], re.I):
                dropped.append("%s %s (the message doesn't ask for that)" % (line, arg))
                continue
            if (fid, arg.lower()) in seen or len(calls) >= int(fn.get("max_calls", MAX_CALLS)):
                continue
            seen.add((fid, arg.lower()))
            calls.append({"function": fid, "line": line + (" " + arg if arg else ""), "arg": arg})
        meta.update({"calls": calls, "unknown": unknown, "dropped": dropped, "raw": reply[:400]})
        lines = ([c["line"] for c in calls] + ["%s (no such call)" % u for u in unknown]
                 + ["dropped: %s" % d for d in dropped])
        return {"output": "\n".join(lines) or "NONE: reply directly", "meta": meta, "forward": run.data["request"]}

    def _browse(self, fn, choice, text, run, step, board):
        p, _ = self.provider_of(choice)
        question, meta = squash(text), {}
        if len(question) > 200:
            question = self._rewrite(LOOKUP_WRITER_PROMPT, text, run)
            meta["question"] = question
        status, data, _ = http_json("POST", p["url"].rstrip("/") + "/ask", {"question": question[:2000]}, timeout=600)
        if status != 200 or not isinstance(data, dict) or not str(data.get("answer") or "").strip():
            raise HarnessError("look-up failed (HTTP %s)" % status)
        found = data["meta"].get("sources") if isinstance(data.get("meta"), dict) else None
        if isinstance(found, list):
            sources = []
            for s in found[:8]:
                url = str(s.get("url") or "") if isinstance(s, dict) else str(s)
                title = str(s.get("title") or url) if isinstance(s, dict) else url
                sources.append({"title": title[:200], "url": url})
            meta["sources"] = sources
        return {"output": str(data["answer"]).strip(), "meta": meta}

    def _draw(self, fn, choice, text, run, step, board):
        p, _ = self.provider_of(choice)
        prompt, meta = squash(text), {}
        if len(prompt) > 200 or (text == run.data["request"] and step["n"] > 1):
            prompt = self._rewrite(DRAW_WRITER_PROMPT, text, run)
            meta["rewritten_by"] = self.choice(self.router)
        prompt = prompt[:500]
        status, data, headers = http_json("POST", p["url"].rstrip("/") + "/sdapi/v1/txt2img", {
            "prompt": prompt, "negative_prompt": fn.get("negative", ""), "steps": fn.get("steps", 20),
            "cfg_scale": 7, "width": 512, "height": 512, "seed": -1, "batch_size": 1,
        }, timeout=900)
        images = data.get("images") if isinstance(data, dict) else None
        try:
            png = base64.b64decode(images[0]) if images else b""
        except (ValueError, TypeError):
            png = b""
        if status != 200 or not png.startswith(PNG_MAGIC):
            raise HarnessError("the picture maker failed (HTTP %s)" % status)
        name = "%d.png" % step["n"]
        os.makedirs(os.path.join(self.runs_dir, run.id), exist_ok=True)
        with open(os.path.join(self.runs_dir, run.id, name), "wb") as f:
            f.write(png)
        meta["prompt"] = prompt
        if "x-gateway-waited" in headers:
            meta["waited_s"] = headers["x-gateway-waited"]
        return {"output": prompt, "image": "runs/%s/%s" % (run.id, name), "meta": meta}

    def _vision(self, fn, choice, text, run, step, board):
        path = run.data.get("image")
        if not path:
            raise HarnessError("no image attached")
        try:
            with open(path, "rb") as f:
                raw = f.read(MAX_IMAGE_BYTES + 1)
        except OSError as exc:
            raise HarnessError("can't read the image (%s)" % exc)
        if len(raw) > MAX_IMAGE_BYTES:
            raise HarnessError("the image is over 8 MB")
        url = "data:%s;base64,%s" % (mimetypes.guess_type(path)[0] or "image/png", base64.b64encode(raw).decode())
        reply, meta = self.chat(choice, fill(fn["prompt"]),
                                [{"type": "text", "text": "The request: %s" % run.data["request"]},
                                 {"type": "image_url", "image_url": {"url": url}}], max_tokens=400, temperature=0.2)
        return {"output": reply, "meta": meta}

    def run_function(self, run, fid, text, why=None):
        fn = self.functions[fid]
        choice = self.choice(fid)
        step = {"n": len(run.data["steps"]) + 1, "function": fid, "name": fn["name"], "kind": fn["kind"],
                "model": choice, "why": why, "input": text, "output": "", "image": None, "meta": {},
                "status": "running", "started": time.time(), "seconds": None}
        board = whiteboard(run.data["steps"], skip_text=text)
        run.change(lambda d: d["steps"].append(step))
        started = time.monotonic()
        try:
            if not choice:
                raise HarnessError("no model picked for %s" % fn["name"])
            result, status = getattr(self, "_" + fn["kind"])(fn, choice, text, run, step, board), "done"
        except HarnessError as exc:
            result, status = {"output": str(exc)}, "error"
        seconds = round(time.monotonic() - started, 1)

        def finish(d):
            step.update(result)
            step["status"], step["seconds"] = status, seconds
        run.change(finish)
        return step

    # ----- pipes -----

    def execute(self, run, pipe_ids, on_step=None):
        try:
            self._run_pipe(run, pipe_ids, on_step)
        except HarnessError as exc:
            self._finish(run, "error", str(exc))
        except Exception as exc:  # a bug shouldn't leave a page spinning forever
            traceback.print_exc()
            self._finish(run, "error", "harness bug: %s" % exc)
        return run

    def _step(self, run, fid, text, why, on_step):
        if on_step:
            on_step({"n": len(run.data["steps"]) + 1, "name": self.functions[fid]["name"],
                     "model": self.choice(fid), "why": why}, "start")
        step = self.run_function(run, fid, text, why)
        if on_step:
            on_step(step, "end")
        return step

    def _run_pipe(self, run, ids, on_step):
        text = run.data["request"]
        for fid in ids:
            step = self._step(run, fid, text, None, on_step)
            if step["status"] == "error":
                return self._finish(run, "error", step["output"])
            if step["kind"] == "gate" and step["meta"].get("pass") is False:
                return self._finish(run, "stopped", step["output"])
            if step["kind"] == "route":
                self._run_calls(run, step, on_step)
            text = step.get("forward") or step["output"]
        self._finish(run, "done")

    def _run_calls(self, run, route_step, on_step):
        """Run what the router asked for. A call that fails is noted, and the reply still gets written."""
        for call in route_step["meta"].get("calls", []):
            fid = call["function"]
            why = "%s (asked by step %d)" % (call["line"], route_step["n"])
            if self.functions[fid]["kind"] == "vision" and not run.data.get("image"):
                self._skip(run, fid, why, "skipped: no image attached", on_step)
            elif not self.choice(fid):
                self._skip(run, fid, why, "skipped: no model picked for %s" % self.functions[fid]["name"], on_step)
            else:
                self._step(run, fid, call["arg"] or run.data["request"], why, on_step)

    def _skip(self, run, fid, why, reason, on_step):
        fn = self.functions[fid]
        step = {"n": len(run.data["steps"]) + 1, "function": fid, "name": fn["name"], "kind": fn["kind"],
                "model": self.choice(fid), "why": why, "input": "", "output": reason, "image": None, "meta": {},
                "status": "skipped", "started": time.time(), "seconds": 0.0}
        run.change(lambda d: d["steps"].append(step))
        if on_step:
            on_step(step, "end")

    def _finish(self, run, status, note=""):
        final = None
        for step in reversed(run.data["steps"]):
            if step["status"] == "done" and step["kind"] in ANSWER_KINDS:
                final = {"step": step["n"], "name": step["name"], "image": step["image"],
                         "text": "" if step["image"] else step["output"],
                         "caption": step["output"] if step["image"] else ""}
                break

        def apply(d):
            d["status"], d["note"], d["final"] = status, note, final
            d["seconds"] = round(time.time() - d["started"], 1)
        run.change(apply)
        try:
            write_json(os.path.join(self.runs_dir, run.id, "run.json"), run.snapshot()[1])
        except OSError:
            pass

    # ----- checks and scores -----

    def test_slot(self, slot, on_case=None):
        """Run a function's checks with its current model and remember the score for that model."""
        self.kind(slot)
        cases = self.checks.get(slot) or []
        if not cases:
            raise HarnessError("no checks written for %s" % slot)
        choice = self.choice(slot)
        if not choice:
            raise HarnessError("no model picked for %s" % slot)
        results = []
        for case in cases:
            text = case.get("input") or case["request"]
            run = Run(case.get("request") or text, "test")
            step = self.run_function(run, slot, text)
            ok, why = grade(case, step)
            result = {"input": text, "ok": ok, "why": why, "seconds": step["seconds"], "output": squash(step["output"])[:300]}
            results.append(result)
            if on_case:
                on_case(result)
        score = {"passed": sum(r["ok"] for r in results), "total": len(results),
                 "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 1),
                 "when": time.strftime("%Y-%m-%d %H:%M"), "results": results}
        path = os.path.join(self.folder, "scores.json")
        with self.lock:
            self.scores = read_json(path, self.scores)  # another process may have scored something meanwhile
            self.scores.setdefault(slot, {})[choice] = score
            write_json(path, self.scores)
        return choice, score

    def reload_scores(self):
        """Pick up scores written by another process, like the CLI's --test."""
        fresh = read_json(os.path.join(self.folder, "scores.json"), None)
        if isinstance(fresh, dict):
            with self.lock:
                self.scores = fresh


def grade(case, step):
    if step["status"] != "done":
        return False, step["output"]
    out = step["output"].lower()
    missing = [w for w in case.get("contains", []) if w.lower() not in out]
    if missing:
        return False, "missing %s" % ", ".join(missing)
    if case.get("contains_any") and not any(w.lower() in out for w in case["contains_any"]):
        return False, "mentions none of %s" % ", ".join(case["contains_any"])
    if "passes" in case and step["meta"].get("pass") is not case["passes"]:
        return False, "expected %s" % ("YES" if case["passes"] else "NO")
    if case.get("tokens") and len(step["meta"].get("tokens", [])) < case["tokens"]:
        return False, "only %d context tokens" % len(step["meta"].get("tokens", []))
    if case.get("calls"):
        called = [c["function"] for c in step["meta"].get("calls", [])]
        skipped = [c for c in case["calls"] if c not in called]
        if skipped:
            return False, "didn't call %s (called: %s)" % (", ".join(skipped), ", ".join(called) or "nothing")
    if case.get("numbered") and not re.search(r"^\W*1[.)]", step["output"], re.M):
        return False, "no numbered steps"
    if case.get("image") and not step.get("image"):
        return False, "no picture"
    return True, "ok"


def print_step(step, phase):
    if phase == "start":
        line = "> %d. %s on %s%s" % (step["n"], step["name"], step["model"], "   " + step["why"] if step.get("why") else "")
    else:
        mark = {"done": "ok"}.get(step["status"], step["status"].upper())
        line = "  %s in %.1fs: %s" % (mark, step["seconds"] or 0, squash(step["output"])[:160])
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a request through swappable brain functions.")
    ap.add_argument("request", nargs="*", help="what you want; text piped in on stdin is added after it")
    ap.add_argument("--pipe", help='functions in order, like "lookup | draw" (default: the flow in harness.json)')
    ap.add_argument("--image", help="attach an image for @process:vision")
    ap.add_argument("--list", action="store_true", help="show each function's model and what else it could use")
    ap.add_argument("--use", metavar="FUNCTION=PROVIDER/MODEL", help="swap a model, like planning=pc/gemma-4-12b")
    ap.add_argument("--test", metavar="FUNCTION", help="run a function's checks with its current model")
    ap.add_argument("--json", action="store_true", help="print the whole whiteboard as JSON")
    args = ap.parse_args(argv)
    try:
        h = Harness()
        if args.list:
            status = h.status()
            for name, p in h.providers.items():
                st = status.get(name, {})
                print("%-7s %-26s %s" % (name, p.get("label", name), "up" if st.get("up") else "DOWN " + st.get("error", "")))
            print()
            for slot in h.slots():
                could = ", ".join(o["choice"] for o in h.options(slot, status) if o["up"]) or "nothing yet"
                print("%-11s %-22s could use: %s" % (slot, h.choice(slot) or "(none)", could))
            return 0
        if args.use:
            slot, _, choice = args.use.partition("=")
            h.choose(slot.strip(), choice.strip())
            print("%s now uses %s" % (slot.strip(), choice.strip()))
            return 0
        if args.test:
            show = lambda r: sys.stderr.write("  %s %s (%.1fs): %s\n" % ("PASS" if r["ok"] else "FAIL", r["input"][:60], r["seconds"], r["why"]))  # noqa: E731
            choice, score = h.test_slot(args.test, on_case=show)
            print("%s on %s: %d/%d passed, %.1fs average" % (args.test, choice, score["passed"], score["total"], score["avg_seconds"]))
            return 0 if score["passed"] == score["total"] else 1
        piped = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()
        request = combine_request(" ".join(args.request), piped)
        if not request:
            ap.error("say what you want, or pipe text in")
        spec = args.pipe or h.flow
        ids = h.parse_pipe(spec)
        image = os.path.abspath(args.image) if args.image else None
        run = h.execute(Run(request, "pipe" if args.pipe else "flow", ids, image), ids, on_step=print_step)
        _, data = run.snapshot()
        if args.json:
            print(json.dumps(data, indent=2))
        elif data["final"]:
            final = data["final"]
            print(os.path.join(h.folder, final["image"]) if final["image"] else final["text"])
        if data["status"] != "done":
            sys.stderr.write("%s: %s\n" % (data["status"], data["note"]))
        return {"done": 0, "stopped": 2}.get(data["status"], 1)
    except HarnessError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
