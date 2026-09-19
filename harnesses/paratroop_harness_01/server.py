#!/usr/bin/env python3
"""The paratroop harness as a web page: the brain diagram with live model dropdowns and a whiteboard.

It only answers on 127.0.0.1.

    python3 server.py            # then open http://127.0.0.1:8770
"""
import argparse
import json
import os
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from harness import Harness, HarnessError, Run

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.html")
MAX_BODY = 64 * 1024
MAX_REQUEST_CHARS = 8000
KEEP_RUNS = 50
LOCAL_NAMES = ("127.0.0.1", "localhost", "::1")
EVENTS = re.compile(r"^/api/runs/(\d{8}-\d{6}-[0-9a-f]{6})/events$")
IMAGE = re.compile(r"^/runs/(\d{8}-\d{6}-[0-9a-f]{6})/(\d{1,3})\.png$")
CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; "
       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def summary(score):
    if not score:
        return None
    out = {k: score.get(k) for k in ("passed", "total", "avg_seconds", "when")}
    out["results"] = [{"ok": r.get("ok"), "input": str(r.get("input", ""))[:80], "why": str(r.get("why", ""))[:120]}
                      for r in score.get("results", [])]
    return out


class App:
    def __init__(self, harness):
        self.harness = harness
        self.runs = {}
        self.tests = {}
        self.lock = threading.Lock()

    def state(self):
        h = self.harness
        h.reload_scores()
        status = h.status()
        slots = []
        for slot in h.slots():
            fn = h.functions[slot]
            choice = h.choice(slot)
            with self.lock:
                test = dict(self.tests.get(slot) or {})
            slots.append({
                "slot": slot, "name": fn["name"], "group": fn.get("group", "stage"), "kind": fn["kind"],
                "call": fn.get("call"), "color": fn.get("color", "hub"), "does": fn.get("does", ""),
                "choice": choice, "options": h.options(slot, status),
                "score": summary((h.scores.get(slot) or {}).get(choice)) if choice else None,
                "has_checks": bool(h.checks.get(slot)), "testing": bool(test.get("running")),
                "test_error": test.get("error", ""),
            })
        providers = {name: {"label": p.get("label", name), "up": bool(status.get(name, {}).get("up")),
                            "error": status.get(name, {}).get("error", "")}
                     for name, p in h.providers.items()}
        return {"slots": slots, "providers": providers, "flow": h.flow, "examples": h.config.get("examples", [])}

    def start_run(self, request, pipe):
        spec = pipe.strip() or self.harness.flow
        ids = self.harness.parse_pipe(spec)
        run = Run(request, "flow" if not pipe.strip() else "pipe", ids)
        with self.lock:
            self.runs[run.id] = run
            for old in list(self.runs)[:-KEEP_RUNS]:
                del self.runs[old]
        threading.Thread(target=self.harness.execute, args=(run, ids), daemon=True).start()
        return run

    def start_test(self, slot):
        h = self.harness
        h.kind(slot)
        if not h.checks.get(slot):
            raise HarnessError("no checks written for %s" % slot)
        if not h.choice(slot):
            raise HarnessError("pick a model for %s first" % slot)
        with self.lock:
            if (self.tests.get(slot) or {}).get("running"):
                raise HarnessError("%s is already being tested" % slot)
            self.tests[slot] = {"running": True}

        def work():
            try:
                h.test_slot(slot)
                outcome = {"running": False}
            except HarnessError as exc:
                outcome = {"running": False, "error": str(exc)}
            except Exception as exc:
                traceback.print_exc()
                outcome = {"running": False, "error": "harness bug: %s" % exc}
            with self.lock:
                self.tests[slot] = outcome
        threading.Thread(target=work, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "paratroop/1"

    def log_message(self, fmt, *args):
        pass

    @property
    def app(self):
        return self.server.app

    def _local_host(self):
        host = (self.headers.get("Host") or "").strip().lower()
        name = host[1:].split("]")[0] if host.startswith("[") else host.rsplit(":", 1)[0]
        return name in LOCAL_NAMES

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._local_host():
            return self._send(403, {"error": "this page only answers on localhost"})
        path = urlsplit(self.path).path
        try:
            if path == "/":
                with open(PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8", {"Content-Security-Policy": CSP})
            if path == "/api/state":
                return self._send(200, self.app.state())
            found = EVENTS.match(path)
            if found:
                return self._events(found.group(1))
            found = IMAGE.match(path)
            if found:
                file = os.path.join(self.app.harness.runs_dir, found.group(1), found.group(2) + ".png")
                if os.path.isfile(file):
                    with open(file, "rb") as f:
                        return self._send(200, f.read(), "image/png")
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _events(self, rid):
        with self.app.lock:
            run = self.app.runs.get(rid)
        if run is None:
            return self._send(404, {"error": "no such run"})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        version = -1
        while True:
            current, data = run.snapshot()
            if current != version:
                version = current
                self.wfile.write(("data: %s\n\n" % json.dumps(data)).encode("utf-8"))
                self.wfile.flush()
                if data["status"] != "running":
                    return
            else:
                self.wfile.write(b": still working\n\n")
                self.wfile.flush()
            run.wait(version, 15)

    def do_POST(self):
        if not self._local_host():
            return self._send(403, {"error": "this page only answers on localhost"})
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc.lower() != (self.headers.get("Host") or "").lower():
            return self._send(403, {"error": "cross-site request refused"})
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            return self._send(415, {"error": "send JSON"})
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = -1
        if not 0 < size <= MAX_BODY:
            return self._send(413, {"error": "empty or too big"})
        try:
            body = json.loads(self.rfile.read(size).decode("utf-8"))
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return self._send(400, {"error": "bad JSON"})
        path = urlsplit(self.path).path
        try:
            if path == "/api/run":
                request = str(body.get("request") or "").strip()
                if not 0 < len(request) <= MAX_REQUEST_CHARS:
                    return self._send(400, {"error": "say something (up to %d characters)" % MAX_REQUEST_CHARS})
                return self._send(200, {"id": self.app.start_run(request, str(body.get("pipe") or "")).id})
            if path == "/api/choose":
                self.app.harness.choose(str(body.get("slot")), str(body.get("choice")))
                return self._send(200, self.app.state())
            if path == "/api/test":
                self.app.start_test(str(body.get("slot")))
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except HarnessError as exc:
            return self._send(400, {"error": str(exc)})


def make_server(harness, port=8770):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.app = App(harness)
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    server = make_server(Harness(), args.port)
    print("paratroop harness on http://127.0.0.1:%d" % server.server_address[1], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
