"""paratroop_harness_01 against fake model, browse and picture servers.

    python3 -m unittest discover -s tests
"""
import base64
import http.client
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness  # noqa: E402
import server  # noqa: E402
from harness import Harness, HarnessError, Run, combine_request, whiteboard  # noqa: E402

MEETING = "I have a meeting this Friday to discuss an investment opportunity for our business"
CONTEXT_REPLY = ("TOKENS: [User][meeting][this Friday][investment opportunity][business]\n"
                 "WHEN: Friday 2026-09-18\nINTENT: prepare for an investment meeting")
PLAN_REPLY = "1. Research the business (Wednesday)\n2. Write your questions (Thursday)\n3. Bring notes (Friday)"


class Fake(BaseHTTPRequestHandler):
    """Plays every server: the gateway's models, the browse server and Stable Diffusion."""
    route_reply = None
    sd_status = 200
    seen = []

    def log_message(self, *args):
        pass

    def _send(self, code, obj, headers=None):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": "model-a", "loaded": True}, {"id": "model-b", "loaded": False}]})
        if self.path in ("/health", "/"):
            return self._send(200, {"ok": True})
        self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append((self.path, body))
        if self.path == "/v1/chat/completions":
            system, user = body["messages"][0]["content"], body["messages"][1]["content"]
            if "router in a harness" in system:
                reply = Fake.route_reply or "@process:prediction\n@process:planning\n@tool:calendar Friday\n@process:vision"
            elif "attention filter" in system:
                reply = "NO" if "lol" in user else "YES"
            elif "Context function" in system:
                reply = CONTEXT_REPLY
            elif "Prediction function" in system:
                reply = "They want help preparing for the meeting."
            elif "Planning function" in system:
                reply = PLAN_REPLY
            elif "Stable Diffusion prompt" in system:
                reply = "a red apple, studio light"
            elif "web search question" in system:
                reply = "capital of Australia"
            else:
                reply = "REPLY[%s]: %s" % (body["model"], user)
            return self._send(200, {"choices": [{"message": {"content": reply}}]}, {"X-Gateway-Model": body["model"]})
        if self.path == "/ask":
            return self._send(200, {"answer": "Canberra is the capital of Australia.",
                                    "meta": {"sources": [{"title": "Canberra", "url": "https://example.com/canberra"}]}})
        if self.path == "/sdapi/v1/txt2img":
            if Fake.sd_status != 200:
                return self._send(Fake.sd_status, {"error": "busy"})
            return self._send(200, {"images": [base64.b64encode(harness.PNG_MAGIC + b"fake").decode()]})
        self._send(404, {})


def closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Rig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.fake.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:%d" % cls.fake.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.fake.shutdown()
        cls.fake.server_close()

    def setUp(self):
        Fake.route_reply, Fake.sd_status, Fake.seen = None, 200, []
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        with open(os.path.join(ROOT, "harness.json")) as f:
            config = json.load(f)
        config["providers"] = {
            "fake": {"kind": "openai", "url": self.url, "label": "Fake", "vision_models": ["model-b"]},
            "down": {"kind": "openai", "url": "http://127.0.0.1:%d" % closed_port(), "label": "Down"},
            "browse": {"kind": "browse", "url": self.url, "label": "Browse", "model": "browse"},
            "sd": {"kind": "sd", "url": self.url, "label": "SD", "model": "sd1.5"},
        }
        for fn in config["functions"]:
            if fn["kind"] in ("gate", "chat", "route"):
                fn["provider"], fn["model"] = "fake", "model-a"
        self.write("harness.json", config)
        self.h = Harness(self.dir)

    def write(self, name, data):
        with open(os.path.join(self.dir, name), "w") as f:
            json.dump(data, f)

    def run_flow(self, request, pipe=None, image=None):
        ids = self.h.parse_pipe(pipe or self.h.flow)
        run = Run(request, "pipe" if pipe else "flow", ids, image)
        self.h.execute(run, ids)
        return run.snapshot()[1]

    def chats(self, marker):
        return [b for p, b in Fake.seen if p == "/v1/chat/completions" and marker in str(b["messages"][0]["content"])]


class Flow(Rig):
    def test_meeting_example_drops_through_every_stage(self):
        data = self.run_flow(MEETING)
        self.assertEqual(data["status"], "done")
        self.assertEqual([s["function"] for s in data["steps"]],
                         ["attention", "context", "route", "prediction", "planning", "vision", "respond"])
        context, route, vision = data["steps"][1], data["steps"][2], data["steps"][5]
        self.assertEqual(context["meta"]["tokens"], ["User", "meeting", "this Friday", "investment opportunity", "business"])
        self.assertEqual(context["meta"]["when"], "Friday 2026-09-18")
        self.assertIn("[this Friday]", self.chats("router in a harness")[0]["messages"][1]["content"])
        self.assertEqual([c["function"] for c in route["meta"]["calls"]], ["prediction", "planning", "vision"])
        self.assertEqual(route["meta"]["unknown"], ["@tool:calendar"])
        self.assertEqual((vision["status"], vision["output"]), ("skipped", "skipped: no image attached"))
        reply = data["final"]["text"]
        self.assertEqual(data["final"]["name"], "Respond")
        self.assertIn("[investment opportunity]", reply)  # Respond saw the context tokens,
        self.assertIn("Research the business", reply)     # the plan,
        self.assertIn("help preparing", reply)            # and the prediction,
        self.assertNotIn("@process", reply)               # but not the router's call list

    def test_not_for_me_stops_at_attention(self):
        data = self.run_flow("lol ok")
        self.assertEqual(data["status"], "stopped")
        self.assertEqual([s["function"] for s in data["steps"]], ["attention"])
        self.assertIsNone(data["final"])

    def test_router_calls_tools_with_their_arguments(self):
        Fake.route_reply = ("@tool:lookup capital of Australia\n@tool:draw a red apple on a table\n"
                            "@tool:lookup capital of Australia")
        data = self.run_flow("Tell me Australia's capital and draw me an apple")
        self.assertEqual(data["status"], "done")
        self.assertEqual([s["function"] for s in data["steps"]],
                         ["attention", "context", "route", "lookup", "draw", "respond"])  # the repeat is dropped
        steps = {s["function"]: s for s in data["steps"]}
        self.assertEqual(steps["lookup"]["input"], "capital of Australia")
        self.assertEqual(steps["lookup"]["meta"]["sources"][0]["url"], "https://example.com/canberra")
        self.assertEqual([b for p, b in Fake.seen if p == "/sdapi/v1/txt2img"][0]["prompt"], "a red apple on a table")
        self.assertTrue(os.path.isfile(os.path.join(self.dir, steps["draw"]["image"])))
        self.assertIn("Canberra", data["final"]["text"])

    def test_a_failed_tool_still_gets_a_reply(self):
        Fake.route_reply, Fake.sd_status = "@tool:draw a red apple", 503
        data = self.run_flow("Draw me an apple")
        self.assertEqual(data["status"], "done")
        self.assertEqual(next(s for s in data["steps"] if s["function"] == "draw")["status"], "error")
        self.assertEqual(data["final"]["name"], "Respond")

    def test_router_keeps_to_max_calls(self):
        Fake.route_reply = "\n".join("@tool:lookup question %d" % i for i in range(9))
        data = self.run_flow("Look up lots of things")
        self.assertEqual(sum(s["function"] == "lookup" for s in data["steps"]), 4)

    def test_router_drops_placeholder_and_unasked_tool_calls(self):
        Fake.route_reply = "@process:prediction\n@tool:lookup trends in [user's industry]\n@tool:draw an infographic"
        data = self.run_flow(MEETING)
        route = data["steps"][2]
        self.assertEqual([c["function"] for c in route["meta"]["calls"]], ["prediction"])
        self.assertEqual(len(route["meta"]["dropped"]), 2)
        self.assertEqual([s["function"] for s in data["steps"]],
                         ["attention", "context", "route", "prediction", "respond"])

    def test_hand_written_pipe_passes_output_along(self):
        data = self.run_flow("Sam has a dentist appointment next Tuesday", pipe="context | planning")
        self.assertEqual([s["function"] for s in data["steps"]], ["context", "planning"])
        self.assertIn("[this Friday]", self.chats("Planning function")[0]["messages"][1]["content"])
        self.assertEqual(data["final"]["name"], "Planning")

    def test_pipe_accepts_names_calls_and_aliases(self):
        self.assertEqual(self.h.parse_pipe("Attention | tokens | hub | @process:planning | reply"),
                         ["attention", "context", "route", "planning", "respond"])
        with self.assertRaises(HarnessError):
            self.h.parse_pipe("context | teleport")

    def test_vision_sends_the_image_once_a_model_is_picked(self):
        image = os.path.join(self.dir, "pic.png")
        with open(image, "wb") as f:
            f.write(harness.PNG_MAGIC + b"pixels")
        self.assertEqual(self.run_flow("What's in this?", pipe="vision", image=image)["status"], "error")
        self.h.choose("vision", "fake/model-b")
        data = self.run_flow("What's in this?", pipe="vision", image=image)
        self.assertEqual(data["status"], "done")
        content = self.chats("Describe the attached image")[-1]["messages"][1]["content"]
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))


class Models(Rig):
    def test_choose_checks_kind_and_availability_and_persists(self):
        self.h.choose("planning", "fake/model-b")
        self.assertEqual(Harness(self.dir).choice("planning"), "fake/model-b")
        for slot, choice in (("draw", "fake/model-a"), ("planning", "fake/nope"),
                             ("planning", "down/model-a"), ("vision", "fake/model-a")):
            with self.assertRaises(HarnessError, msg="%s=%s" % (slot, choice)):
                self.h.choose(slot, choice)

    def test_offline_provider_is_reported(self):
        status = self.h.status(max_age=0)
        self.assertFalse(status["down"]["up"])
        self.assertTrue(status["fake"]["up"])
        self.assertEqual([o["choice"] for o in self.h.options("vision", status)], ["fake/model-b"])

    def test_scores_are_kept_per_model(self):
        self.write("checks.json", {"context": [{"input": "meeting Friday", "contains": ["meeting"], "tokens": 4}],
                                   "route": [{"request": "draw a cat", "calls": ["draw"]}]})
        h = Harness(self.dir)
        choice, score = h.test_slot("context")
        self.assertEqual((choice, score["passed"], score["total"]), ("fake/model-a", 1, 1))
        Fake.route_reply = "@process:prediction"
        _, score = h.test_slot("route")
        self.assertEqual(score["passed"], 0)
        self.assertIn("didn't call draw", score["results"][0]["why"])
        with open(os.path.join(self.dir, "scores.json")) as f:
            self.assertEqual(sorted(json.load(f)), ["context", "route"])


class Helpers(unittest.TestCase):
    def test_combine_request(self):
        self.assertEqual(combine_request("one sentence", "long text\n"), "one sentence\n\nlong text")
        self.assertEqual(combine_request("", " piped "), "piped")

    def test_whiteboard_skips_decisions_and_keeps_the_newest(self):
        steps = [{"n": 1, "name": "Attention", "kind": "gate", "status": "done", "output": "YES"},
                 {"n": 2, "name": "Context", "kind": "chat", "status": "done", "output": "old " * 400},
                 {"n": 3, "name": "Route", "kind": "route", "status": "done", "output": "@process:planning"},
                 {"n": 4, "name": "Planning", "kind": "chat", "status": "done", "output": "1. do it"}]
        board = whiteboard(steps, budget=300)
        self.assertNotIn("YES", board)
        self.assertNotIn("@process", board)
        self.assertIn("1. do it", board)
        self.assertLessEqual(len(board), 302)


class Page(Rig):
    def setUp(self):
        super().setUp()
        self.srv = server.make_server(self.h, 0)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.host = "127.0.0.1:%d" % self.srv.server_address[1]

    def call(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection(self.host, timeout=10)
        head = {"Host": self.host}
        if body is not None:
            head["Content-Type"] = "application/json"
        head.update(headers or {})
        conn.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=head)
        resp = conn.getresponse()
        return resp.status, resp.read()

    def test_refuses_other_hosts_origins_non_json_and_odd_paths(self):
        self.assertEqual(self.call("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.call("POST", "/api/run", {"request": "hi"}, {"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.call("POST", "/api/run", {"request": "hi"}, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.call("POST", "/api/run", {"request": "hi", "pipe": "teleport"})[0], 400)
        self.assertEqual(self.call("GET", "/runs/../harness.json")[0], 404)

    def test_page_and_state(self):
        status, body = self.call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Paratroop Harness", body)
        state = json.loads(self.call("GET", "/api/state")[1])
        self.assertEqual(state["flow"], "attention | context | route | respond")
        self.assertEqual([s["group"] for s in state["slots"]].count("router"), 1)
        self.assertFalse(state["providers"]["down"]["up"])

    def test_run_streams_until_done(self):
        status, body = self.call("POST", "/api/run", {"request": MEETING}, {"Origin": "http://" + self.host})
        self.assertEqual(status, 200)
        conn = http.client.HTTPConnection(self.host, timeout=10)
        conn.request("GET", "/api/runs/%s/events" % json.loads(body)["id"], headers={"Host": self.host})
        resp = conn.getresponse()
        last = None
        for line in iter(resp.fp.readline, b""):
            if line.startswith(b"data: "):
                last = json.loads(line[6:])
                if last["status"] != "running":
                    break
        self.assertEqual((last["status"], last["final"]["name"]), ("done", "Respond"))


if __name__ == "__main__":
    unittest.main()
