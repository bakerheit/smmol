#!/usr/bin/env python3
"""The cognitive-module harness as a web page: modules around working memory, live as it thinks.

It only answers on 127.0.0.1.

    python3 server.py            # then open http://127.0.0.1:8771
"""
import argparse
import json
import os
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from harness import Harness, HarnessError, read_json

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.html")
CLASSIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page-classic.html")  # the page before the redesign, at /classic
MAX_BODY = 64 * 1024
MAX_MESSAGE = 8000
KEEP_RUNS = 50
LOCAL_NAMES = ("127.0.0.1", "localhost", "::1")
EVENTS = re.compile(r"^/api/runs/(\d{8}-\d{6}-[0-9a-f]{6})/events$")
RUN = re.compile(r"^/api/runs/(\d{8}-\d{6}-[0-9a-f]{6})$")
CONVERSATION = re.compile(r"^/api/conversations/(\d{8}-\d{6}-[0-9a-f]{6})$")
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
        modules = []
        for mid in h.order:
            m = h.modules[mid]
            choice = h.choice(mid)
            with self.lock:
                test = dict(self.tests.get(mid) or {})
            modules.append({
                "id": mid, "name": m["name"], "color": m.get("color", "grey"), "does": m.get("does", ""), "on": h.module_on(mid),
                "choice": choice, "options": h.options(mid, status),
                "score": summary((h.scores.get(mid) or {}).get(choice)) if choice else None,
                "has_checks": bool(h.checks.get(mid)), "testing": bool(test.get("running")), "test_error": test.get("error", ""),
                "settings": h.settings_for(mid),
                "weights": h.weight_config(mid),
            })
        memories = h.memory.all()
        return {
            "modules": modules,
            "tools": [{"name": n, "label": t.get("label", n), "color": t.get("color", "grey"), "on": h.tool_on(n),
                       "built": t.get("built", True), "kind": t.get("kind", "tool"), "does": t.get("does", "")}
                      for n, t in h.tools.items()],
            "providers": {n: {"label": p.get("label", n), "up": bool(status.get(n, {}).get("up")), "error": status.get(n, {}).get("error", "")}
                          for n, p in h.providers.items()},
            "memory": {"count": len(memories),
                       "recent": [{"id": m["id"], "text": m["text"], "created": m["created"]} for m in reversed(memories[-20:])]},
            "examples": h.config.get("examples", []), "max_cycles": h.config.get("max_cycles", 3),
            "conversations": h.conversations.recent(20),
        }

    def start_run(self, message, conversation=None):
        """(run id, conversation id). One turn at a time per conversation."""
        with self.lock:
            thinking = conversation and any(l.state.get("conversation") == conversation and l.state["status"] == "running"
                                            for l in self.runs.values())
        if thinking:
            raise HarnessError("still thinking about the last message in this chat")
        live = self.harness.start(message, conversation)
        with self.lock:
            self.runs[live.state["id"]] = live
            for old in list(self.runs)[:-KEEP_RUNS]:
                if self.runs[old].state["status"] != "running":
                    del self.runs[old]
        threading.Thread(target=self.harness.think, args=(live,), daemon=True).start()
        return live.state["id"], live.state["conversation"]

    def run_state(self, rid):
        with self.lock:
            live = self.runs.get(rid)
        if live is not None:
            return live.snapshot()[1]
        return read_json(os.path.join(self.harness.runs_dir, rid + ".json"), None)

    def start_test(self, mid):
        h = self.harness
        h.module(mid)
        if not h.checks.get(mid):
            raise HarnessError("no checks written for %s" % mid)
        with self.lock:
            if (self.tests.get(mid) or {}).get("running"):
                raise HarnessError("%s is already being tested" % mid)
            self.tests[mid] = {"running": True}

        def work():
            try:
                h.test_module(mid)
                outcome = {"running": False}
            except HarnessError as exc:
                outcome = {"running": False, "error": str(exc)}
            except Exception as exc:
                traceback.print_exc()
                outcome = {"running": False, "error": "harness bug: %s" % exc}
            with self.lock:
                self.tests[mid] = outcome
        threading.Thread(target=work, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "paratroop/2"

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
            if path in ("/", "/classic"):
                with open(PAGE if path == "/" else CLASSIC, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8", {"Content-Security-Policy": CSP})
            if path == "/api/state":
                return self._send(200, self.app.state())
            found = EVENTS.match(path)
            if found:
                return self._events(found.group(1))
            found = RUN.match(path)
            if found:
                state = self.app.run_state(found.group(1))
                return self._send(200, state) if state else self._send(404, {"error": "no such run"})
            if path == "/api/conversations":
                return self._send(200, {"conversations": self.app.harness.conversations.recent(50)})
            found = CONVERSATION.match(path)
            if found:
                data = self.app.harness.conversations.get(found.group(1))
                return self._send(200, data) if data else self._send(404, {"error": "no such conversation"})
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _events(self, rid):
        with self.app.lock:
            live = self.app.runs.get(rid)
        if live is None:
            return self._send(404, {"error": "no such run"})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        version = -1
        while True:
            current, data = live.snapshot()
            if current != version:
                version = current
                self.wfile.write(("data: %s\n\n" % json.dumps(data)).encode("utf-8"))
                self.wfile.flush()
                if data["status"] != "running":
                    return
            else:
                self.wfile.write(b": still thinking\n\n")
                self.wfile.flush()
            live.wait(version, 15)

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
                message = str(body.get("message") or "").strip()
                if not 0 < len(message) <= MAX_MESSAGE:
                    return self._send(400, {"error": "say something (up to %d characters)" % MAX_MESSAGE})
                conversation = body.get("conversation")
                rid, cid = self.app.start_run(message, str(conversation) if conversation else None)
                return self._send(200, {"id": rid, "conversation": cid})
            if path == "/api/choose":
                self.app.harness.choose(str(body.get("module")), str(body.get("choice")))
                return self._send(200, self.app.state())
            if path == "/api/toggle":
                if body.get("kind") not in ("module", "tool") or not isinstance(body.get("on"), bool):
                    return self._send(400, {"error": "send kind (module or tool), id and on (true or false)"})
                self.app.harness.toggle(body["kind"], str(body.get("id")), body["on"])
                return self._send(200, self.app.state())
            if path == "/api/settings":
                self.app.harness.set_setting(str(body.get("module")), str(body.get("key")), body.get("value"))
                return self._send(200, self.app.state())
            if path == "/api/weight-config":
                self.app.harness.set_weight_config(str(body.get("module")), str(body.get("key")), body.get("value"))
                return self._send(200, self.app.state())
            if path == "/api/test":
                self.app.start_test(str(body.get("module")))
                return self._send(200, {"ok": True})
            if path == "/api/forget":
                if not self.app.harness.memory.forget(str(body.get("id"))):
                    return self._send(404, {"error": "no memory with that id"})
                return self._send(200, self.app.state())
            return self._send(404, {"error": "not found"})
        except HarnessError as exc:
            return self._send(400, {"error": str(exc)})


def make_server(harness, port=8771):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.app = App(harness)
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8771)
    args = ap.parse_args()
    server = make_server(Harness(), args.port)
    print("paratroop harness 02 on http://127.0.0.1:%d" % server.server_address[1], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
