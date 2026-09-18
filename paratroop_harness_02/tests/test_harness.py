"""paratroop_harness_02 against a fake model server that plays every module, plus a fake browse server.

    python3 -m unittest discover -s tests
"""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness  # noqa: E402
import server  # noqa: E402
from harness import FileTool, Harness, HarnessError, Live, MemoryStore, calc, chosen, new_state, normalize, parse_json, validate  # noqa: E402

MEETING = "I have a meeting this Friday to discuss an investment opportunity for our business"
MARKERS = [("Rank the stored memories", "recall_rank"), ("Perception module", "perception"), ("Attention module", "attention"),
           ("Recall module", "recall_query"), ("Planner module", "planner"), ("Predictor module", "predictor"),
           ("Critic module", "critic"), ("Decision module", "decision"), ("Language module", "language")]


def ob(kind, text, when=""):
    return {"kind": kind, "text": text, "who": "", "what": "", "when": when, "where": ""}


def cand(kind, summary, tool="none", action="none", target="", content=""):
    return {"type": kind, "summary": summary, "tool": tool, "action": action, "target": target, "content": content}


def default_reply(module, p):
    if module == "perception":
        msg = p["message"]
        if "lol" in msg:
            return {"observations": [ob("small_talk", "lol ok")], "goal": ""}
        if "dentist is" in msg:
            return {"observations": [ob("request", "User wants this remembered"), ob("fact", "User's dentist is Dr. Lee")], "goal": "remember the dentist"}
        if "meeting" in msg:
            return {"observations": [ob("event", "User has a meeting about an investment opportunity", "Friday 2026-09-18")], "goal": "prepare for the meeting"}
        if "%" in msg:
            return {"observations": [ob("question", "User asks what 17.5% of 2,340 is")], "goal": "work out 17.5% of 2,340"}
        return {"observations": [ob("question", msg)], "goal": "answer: " + msg}
    if module == "attention":
        small = all(o["kind"] == "small_talk" for o in p["observations"])
        return {"for_me": not small, "scores": [{"index": o["index"], "relevance": 0.9} for o in p["observations"]], "why": "test"}
    if module == "recall_query":
        return {"queries": ["dentist"]}
    if module == "recall_rank":
        return {"ranked": [{"id": m["id"], "relevance": 0.9 if "dentist" in m["text"].lower() else 0.1} for m in p["memories"]]}
    if module == "planner":
        if p["actions"]:
            return {"goal": p["goal"], "candidates": [cand("answer", "Give the result")]}
        if "%" in p["goal"]:
            return {"goal": p["goal"], "candidates": [cand("tool", "Calculate it", "calculator", "none", "17.5% of 2,340"), cand("answer", "Estimate it")]}
        if "meeting" in p["goal"]:
            return {"goal": p["goal"], "candidates": [cand("answer", "Give pitching tips"), cand("ask", "Ask which side they are on")]}
        return {"goal": p["goal"], "candidates": [cand("answer", "Answer with what's known")]}
    if module == "predictor":
        return {"predictions": [{"index": c["index"], "outcome": "fine", "helpful": 0.6, "risk": "none"} for c in p["candidates"]]}
    if module == "critic":
        if "meeting" in p["message"]:
            return {"issues": [{"kind": "assumption", "text": "Assumes they are the one pitching", "severity": "high"}],
                    "question": "Are you pitching, or being pitched?"}
        return {"issues": [], "question": ""}
    if module == "decision":
        high = any(i["severity"] == "high" for i in (p.get("critique") or {}).get("issues", []))  # no Critic, no critique
        ask = next((c["index"] for c in p["candidates"] if c["type"] == "ask"), None)
        tool = next((c["index"] for c in p["candidates"] if c["type"] == "tool"), None)
        return {"choice": ask if high and ask else tool or 1, "why": "test"}
    if module == "language":
        return "SAID " + json.dumps(p, sort_keys=True) + ((" " + p["question_to_ask"]) if p["question_to_ask"] else "")
    raise AssertionError(module)


class Fake(BaseHTTPRequestHandler):
    """Plays the gateway's models (answering as whichever module is asking) and the browse server."""
    script = {}
    seen = []
    bodies = []
    fetch_reply = {"ok": True, "url": "https://example.com", "text": "Example page text", "words": 3, "truncated": False}

    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": "model-a", "loaded": True}, {"id": "model-b", "loaded": False}]})
        if self.path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/search":
            return self._send(200, {"results": [{"title": "Chicago weather", "url": "https://weather.example/chicago", "snippet": "72°F and sunny"}]})
        if self.path == "/fetch":
            return self._send(200, Fake.fetch_reply)
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        module = next(name for marker, name in MARKERS if marker in system)
        payload = json.loads(user.split("Input:\n", 1)[1].split("\n\nYour last reply", 1)[0])
        Fake.seen.append((module, payload, "response_format" in body))
        Fake.bodies.append((module, body))
        queue = Fake.script.get(module)
        reply = queue.pop(0) if queue else default_reply(module, payload)
        content = reply if isinstance(reply, str) else json.dumps(reply)
        self._send(200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


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
        Fake.script, Fake.seen, Fake.bodies = {}, [], []
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        with open(os.path.join(ROOT, "harness.json")) as f:
            config = json.load(f)
        config["providers"] = {"fake": {"kind": "openai", "url": self.url, "label": "Fake"},
                               "browse": {"kind": "browse", "url": self.url, "label": "Browse"},
                               "smmol": {"kind": "local", "label": "Trained here", "models": {
                                   "smROUTER_01": os.path.join(self.dir, "not-trained-yet.pt"),
                                   "smMATH01-abacus": {"path": os.path.join(self.dir, "math-not-trained.pt"), "kind": "math"},
                                   "smMATH_LANGUAGE_001": {"path": os.path.join(self.dir, "reader-not-trained.pt"), "kind": "math_language"}}}}
        for m in config["modules"]:
            if m.get("kind") not in harness.LOCAL_CODE:  # modules trained here stay on their local provider
                m["provider"], m["model"] = "fake", "model-a"
        self.stubs = {}
        self.config = config
        for name in ("contracts.json", "checks.json"):
            shutil.copy(os.path.join(ROOT, name), self.dir)
        self.write_config()

    def write_config(self):
        with open(os.path.join(self.dir, "harness.json"), "w") as f:
            json.dump(self.config, f)
        self.h = Harness(self.dir)
        self.h._local_models.update(self.stubs)

    def think(self, message):
        return self.h.think(Live(new_state(message))).state

    def calls(self, module):
        return [payload for name, payload, _ in Fake.seen if name == module]

    def trained(self, name, stub, kind=None):
        """Pretend a model trained in SMMOL exists: an empty checkpoint file, played by a stub."""
        path = os.path.join(self.dir, name + ".pt")
        open(path, "wb").close()
        self.config["providers"]["smmol"]["models"][name] = {"path": path, "kind": kind} if kind else path
        self.stubs[(os.path.normpath(path), os.path.getmtime(path))] = stub
        self.write_config()
        return stub

    def use_router(self, **reply):
        return self.trained("smROUTER_01", StubRouter(dict({"for_me": True, "intent": "question", "intent_p": 0.95, "tool": "none",
                                                             "tool_p": 0.9, "ask_first": False, "ask_p": 0.05}, **reply)))

    def use_math(self):
        return self.trained("smMATH01-abacus", StubMath(), "math")

    def use_whole_math(self):
        stub = self.trained("smMATH001-a", StubWholeMath(), "math")
        self.h.choose("math", "smmol/smMATH001-a")
        return stub

    def use_reader(self, problems, weakest=1.0):
        return self.trained("smMATH_LANGUAGE_001", StubReader(problems, weakest), "math_language")

    def only(self, *keep):
        for mid in self.h.order:
            if mid not in keep:
                self.h.toggle("module", mid, False)


class Contracts(unittest.TestCase):
    SCHEMA = {"type": "object", "required": ["kind", "score", "items"], "properties": {
        "kind": {"type": "string", "enum": ["a", "b"], "maxLength": 3},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "items": {"type": "array", "maxItems": 2, "items": {"type": "integer"}}}}

    def test_validate_names_every_problem(self):
        problems = validate({"kind": "c", "score": "high", "items": [1, "two"]}, self.SCHEMA)
        self.assertEqual(len(problems), 3, problems)
        self.assertIn("$.items is missing", validate({"kind": "a", "score": 1}, self.SCHEMA))
        self.assertEqual(validate({"kind": "a", "score": 0.5, "items": [1]}, self.SCHEMA), [])
        self.assertTrue(validate({"kind": "a", "score": True, "items": []}, self.SCHEMA))  # a bool isn't a number

    def test_normalize_clamps_cuts_and_drops(self):
        out = normalize({"kind": "abcdef", "score": 7, "items": [1, 2, 3], "extra": 1}, self.SCHEMA)
        self.assertEqual(out, {"kind": "abc", "score": 1, "items": [1, 2]})

    def test_parse_json_finds_the_object(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json('<think>hmm</think> Sure: {"a": [1]}'), {"a": [1]})
        with self.assertRaises(ValueError):
            parse_json("no json here")


class Tools(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calc("17.5% of 2,340"), "409.5")
        self.assertEqual(calc("0.175 * 2340"), "409.5")
        self.assertEqual(calc("(2 + 3) * 4"), "20")
        self.assertEqual(calc("min(3, 7) + sqrt(16)"), "7")
        self.assertEqual(calc("2^10"), "1024")
        self.assertEqual(harness.find_expression("hey what's 48213 + 9977?"), "48213 + 9977")
        self.assertEqual(harness.find_expression("What's 17.5% of 2,340?"), "17.5% of 2,340")
        self.assertEqual(harness.find_expression("what is 12x4"), "12*4")
        self.assertEqual(harness.find_expression("I have 3 kids and 2 dogs"), "")
        self.assertEqual(harness.bare_arithmetic("2+2"), "2+2")
        self.assertEqual(harness.bare_arithmetic("what's 48,213 + 9,977?"), "48,213 + 9,977")
        self.assertEqual(harness.bare_arithmetic("12 x 7 ="), "12*7")
        self.assertEqual(harness.bare_arithmetic("I have 2 kids and 3 dogs"), "")
        self.assertEqual(harness.bare_arithmetic("hey what's 2+2 lol"), "")  # more than arithmetic: that's the router's call
        self.assertEqual(harness.find_expression("what is the volume of 322234ft x 21323ft x 212231ft?"), "322234*21323*212231")
        self.assertEqual(harness.bare_arithmetic("12ft x 3ft"), "12*3")
        for bad in ("__import__('os').system('ls')", "2 ** 1000000", "1/0", "open('x')", "a.b", ""):
            with self.assertRaises(HarnessError, msg=bad):
                calc(bad)

    def test_file_tool_stays_in_its_folder(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        box = os.path.join(root, "workspace")
        files = FileTool(box, max_bytes=100)
        self.assertIn("created", files.do("create", "list.txt", "eggs\nmilk"))
        self.assertEqual(files.do("read", "list.txt"), "eggs\nmilk")
        with self.assertRaises(HarnessError):
            files.do("create", "list.txt", "again")
        files.do("update", "list.txt", "eggs")
        self.assertEqual(files.do("list", ""), "list.txt")
        self.assertIn("trash", files.do("delete", "list.txt"))
        self.assertEqual(files.do("list", ""), "(the workspace is empty)")
        self.assertEqual(len(os.listdir(os.path.join(box, ".trash"))), 1)  # deleted, not destroyed
        with open(os.path.join(root, "secret.txt"), "w") as f:
            f.write("secret")
        os.symlink(os.path.join(root, "secret.txt"), os.path.join(box, "link.txt"))
        for name in ("../secret.txt", "/etc/passwd", ".hidden", "sub/dir.txt", "link.txt"):
            with self.assertRaises(HarnessError, msg=name):
                files.do("read", name)
        with self.assertRaises(HarnessError):
            files.do("create", "big.txt", "x" * 101)

    def test_memory_store(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        store = MemoryStore(root)
        first = store.add("User's dentist is Dr. Lee, appointments on Tuesdays", "fact", "t")
        store.add("User's shopping day is Sunday", "fact", "t")
        self.assertIsNone(store.add("user's dentist is dr. lee, appointments on tuesdays", "fact", "t"))  # duplicate
        found = store.search(["dentist appointment"])
        self.assertEqual(found[0]["id"], first["id"])
        self.assertEqual(store.search(["volcano"]), [])
        self.assertTrue(store.forget(first["id"]))
        self.assertEqual(store.count(), 1)


class Thinking(Rig):
    def test_math_goes_through_the_calculator(self):
        s = self.think("What's 17.5% of 2,340?")
        self.assertEqual(s["status"], "done", s["note"])
        self.assertEqual(len(s["cycles"]), 2)
        first, second = s["cycles"]
        self.assertEqual(first["candidates"][first["decision"]["choice"] - 1]["tool"], "calculator")
        self.assertEqual((first["action"]["ok"], first["action"]["result"]), (True, "409.5"))
        self.assertIn("predictor: only one candidate", second["skipped"])
        self.assertEqual(self.calls("language")[0]["tool_results"][0]["result"], "409.5")
        self.assertTrue(all(schema for name, _, schema in Fake.seen if name != "language"))  # contracts were sent
        self.assertFalse(any(schema for name, _, schema in Fake.seen if name == "language"))

    def test_follow_ups_carry_the_conversation(self):
        first = self.h.think(self.h.start(MEETING)).state
        self.assertTrue(first["reply"].endswith("Are you pitching, or being pitched?"))
        cid = first["conversation"]
        second = self.h.think(self.h.start("I'm the investor", cid)).state
        self.assertEqual((second["conversation"], second["history"][0]["message"]), (cid, MEETING))
        seen = self.calls("perception")[-1]["conversation"]
        self.assertEqual(seen[0]["person"], MEETING)
        self.assertIn("Are you pitching", seen[0]["assistant"])  # the answer arrives with the question it answers
        self.assertIn("conversation", self.calls("language")[-1])
        saved = self.h.conversations.get(cid)
        self.assertEqual([t["message"] for t in saved["turns"]], [MEETING, "I'm the investor"])
        self.assertEqual(self.h.conversations.recent(5)[0]["turns"], 2)

    def test_unknown_or_odd_conversation_ids_are_refused(self):
        with self.assertRaises(HarnessError):
            self.h.start("hi", "20260101-000000-abcdef")
        with self.assertRaises(HarnessError):
            self.h.start("hi", "../../etc/passwd")

    def test_switched_off_modules_never_run_and_are_never_mentioned(self):
        self.h.toggle("module", "predictor", False)
        self.h.toggle("module", "critic", False)
        s = self.think(MEETING)
        self.assertEqual(s["status"], "done")
        self.assertFalse({"predictor", "critic"} & {t["module"] for t in s["trace"]})
        decision = self.calls("decision")[-1]
        self.assertNotIn("critique", decision)
        self.assertFalse(any("prediction" in c for c in decision["candidates"]))
        self.assertNotIn("last_critique", self.calls("planner")[-1])
        self.assertNotIn("assumptions", self.calls("language")[-1])
        self.assertFalse(Harness(self.dir).module_on("critic"))  # the switch is saved
        with self.assertRaises(HarnessError):
            self.h.run_stage(Live(s), "critic")

    def test_switched_off_tools_vanish_from_prompts_and_contracts(self):
        for tool in ("web_search", "web_browser", "file_system"):
            self.h.toggle("tool", tool, False)
        s = self.think("What's 17.5% of 2,340?")
        self.assertEqual(s["cycles"][0]["action"]["result"], "409.5")  # the calculator still works
        _, body = next(b for b in Fake.bodies if b[0] == "planner")
        sent = json.dumps(body)
        for tool in ("web_search", "web_browser", "file_system"):
            self.assertNotIn(tool, sent)
        fields = body["response_format"]["schema"]["properties"]["candidates"]["items"]["properties"]
        self.assertEqual(fields["tool"]["enum"], ["none", "calculator"])
        self.assertEqual(fields["action"]["enum"], ["none"])
        with self.assertRaises(HarnessError):
            self.h.run_tool(cand("tool", "s", "web_search", "none", "weather"))

    def test_no_tools_means_no_tool_choices_at_all(self):
        for tool in ("calculator", "web_search", "web_browser", "file_system"):
            self.h.toggle("tool", tool, False)
        Fake.script = {"planner": [{"goal": "g", "candidates": [cand("answer", "Estimate it")]}]}
        self.think("What's 17.5% of 2,340?")
        _, body = next(b for b in Fake.bodies if b[0] == "planner")
        fields = body["response_format"]["schema"]["properties"]["candidates"]["items"]["properties"]
        self.assertEqual((fields["type"]["enum"], fields["tool"]["enum"]), (["answer", "make", "ask"], ["none"]))
        self.assertNotIn("actions", self.calls("planner")[-1])

    def test_memory_off_means_no_recall_and_nothing_saved(self):
        self.h.toggle("tool", "memory", False)
        self.think("Remember that my dentist is Dr. Lee")
        s = self.think("Who's my dentist?")
        self.assertEqual((s["remembered"], self.h.memory.count()), ([], 0))
        self.assertNotIn("recall", {t["module"] for t in s["trace"]})
        self.assertNotIn("memories", self.calls("planner")[-1])

    def test_everything_off_still_answers(self):
        for mid in self.h.order:
            self.h.toggle("module", mid, False)
        s = self.think("hello there")
        self.assertEqual((s["status"], s["trace"]), ("done", []))
        self.assertTrue(s["reply"])

    def test_unbuilt_or_unknown_things_cant_be_switched(self):
        with self.assertRaises(HarnessError):
            self.h.toggle("tool", "command_line", True)
        with self.assertRaises(HarnessError):
            self.h.toggle("module", "cerebellum", True)
        with self.assertRaises(HarnessError):
            self.h.toggle("brain", "critic", False)

    def test_small_talk_stops_at_attention(self):
        s = self.think("lol ok")
        self.assertEqual(s["status"], "stopped")
        self.assertEqual([t["module"] for t in s["trace"]], ["perception", "attention"])
        self.assertEqual(s["reply"], "")

    def test_critic_turns_a_weak_assumption_into_a_question(self):
        s = self.think(MEETING)
        self.assertEqual(s["status"], "done")
        cycle = s["cycles"][0]
        self.assertEqual(cycle["candidates"][cycle["decision"]["choice"] - 1]["type"], "ask")
        self.assertTrue(s["reply"].endswith("Are you pitching, or being pitched?"))
        self.assertEqual([m["text"] for m in s["remembered"]], ["User has a meeting about an investment opportunity (when: Friday 2026-09-18)"])

    def test_remembers_facts_and_recalls_them_later(self):
        first = self.think("Remember that my dentist is Dr. Lee")
        self.assertEqual([m["text"] for m in first["remembered"]], ["User's dentist is Dr. Lee"])  # the request itself isn't stored
        self.assertEqual(first["recall"]["note"], "long-term memory is empty")
        second = self.think("Who's my dentist?")
        self.assertEqual(second["recall"]["queries"], ["dentist"])
        self.assertEqual(second["memories"][0]["text"], "User's dentist is Dr. Lee")
        self.assertIn("User's dentist is Dr. Lee", self.calls("language")[-1]["memories"])

    def test_a_broken_contract_gets_one_retry(self):
        Fake.script = {"planner": ["sorry, here is my plan in prose"]}
        s = self.think("What's 17.5% of 2,340?")
        self.assertEqual(s["status"], "done")
        planner = [t for t in s["trace"] if t["module"] == "planner"][0]
        self.assertEqual(planner["retries"], 1)

    def test_two_broken_contracts_stop_the_run(self):
        Fake.script = {"attention": ['{"for_me": "yes"}', '{"scores": []}']}
        s = self.think("What's 17.5% of 2,340?")
        self.assertEqual(s["status"], "error")
        self.assertIn("Attention broke its contract twice", s["note"])

    def test_guards_drop_bad_tool_candidates(self):
        Fake.script = {"planner": [{"goal": "g", "candidates": [
            cand("tool", "Search it", "web_search", "none", "trends in [user's industry]"),
            cand("tool", "Save it", "file_system", "none", "notes.txt"),
            cand("tool", "Add", "calculator", "none", "")]}]}
        s = self.think("Tell me something")
        cycle = s["cycles"][0]
        self.assertEqual(len(cycle["dropped"]), 3, cycle["dropped"])
        self.assertEqual([c["type"] for c in cycle["candidates"]], ["answer"])

    def test_stops_using_tools_at_the_cycle_limit(self):
        Fake.script = {"planner": [{"goal": "g", "candidates": [cand("tool", "Add", "calculator", "none", "%d + 1" % i)]} for i in range(5)]}
        s = self.think("Keep adding")
        self.assertEqual(s["status"], "done")
        self.assertEqual(len(s["cycles"]), 4)  # 3 tool cycles, then the forced answer
        self.assertEqual(s["cycles"][-1]["decision"]["by"], "rule")
        self.assertIn("3 cycles", s["note"])

    def test_files_are_saved_in_the_workspace(self):
        Fake.script = {"planner": [{"goal": "save list", "candidates": [
            cand("tool", "Save the list", "file_system", "create", "shopping-list.txt", "eggs\nmilk\nbread")]}]}
        s = self.think("Save a shopping list with eggs, milk and bread.")
        self.assertEqual(s["status"], "done")
        with open(os.path.join(self.dir, "workspace", "shopping-list.txt")) as f:
            self.assertEqual(f.read(), "eggs\nmilk\nbread")

    def test_web_tools_go_through_the_browse_server(self):
        search = self.h.run_tool(cand("tool", "s", "web_search", "none", "Chicago weather"))
        self.assertIn("https://weather.example/chicago", search)
        self.assertEqual(self.h.run_tool(cand("tool", "b", "web_browser", "none", "https://example.com")), "Example page text")
        with self.assertRaises(HarnessError):
            self.h.run_tool(cand("tool", "b", "web_browser", "none", "file:///etc/passwd"))
        Fake.fetch_reply = {"ok": False, "error": "refused: resolves to non-public address 192.168.1.1"}
        self.addCleanup(setattr, Fake, "fetch_reply", {"ok": True, "text": "Example page text"})
        with self.assertRaisesRegex(HarnessError, "non-public"):
            self.h.run_tool(cand("tool", "b", "web_browser", "none", "http://192.168.1.1/"))

    def test_turned_off_tools_refuse(self):
        self.config["tools"]["calculator"]["enabled"] = False
        self.write_config()
        with self.assertRaises(HarnessError):
            self.h.run_tool(cand("tool", "c", "calculator", "none", "1+1"))

    def test_stages_work_one_at_a_time_like_a_shell_pipe(self):
        state = new_state("What's 17.5% of 2,340?")
        with self.assertRaises(HarnessError):
            self.h.run_stage(Live(state), "attention")
        for stage in ("perception", "attention", "recall", "planner", "predictor", "critic", "decision", "act"):
            live = Live(json.loads(json.dumps(state)))  # what crossing a pipe does to working memory
            self.h.run_stage(live, stage)
            state = live.state
        self.assertEqual(state["cycles"][0]["action"]["result"], "409.5")
        with self.assertRaises(HarnessError):
            self.h.run_stage(Live(state), "nap")

    def test_module_checks_score_per_model(self):
        choice, score = self.h.test_module("planner")
        self.assertEqual((choice, score["total"]), ("fake/model-a", 3))
        with open(os.path.join(self.dir, "scores.json")) as f:
            self.assertIn("fake/model-a", json.load(f)["planner"])

    def test_switched_off_modules_arent_tested_or_scored(self):
        self.h.toggle("module", "critic", False)
        with self.assertRaisesRegex(HarnessError, "switched off"):
            self.h.test_module("critic")
        self.assertNotIn("critic", self.h.scores)

    def test_a_request_to_make_something_gets_made_not_asked_about(self):
        Fake.script = {"planner": [{"goal": "write a JavaScript calculate function", "candidates": [
                           cand("ask", "Ask which operators it should support"), cand("tool", "Draft the function")]}],
                       "critic": [{"issues": [{"kind": "missing", "text": "Operators aren't listed", "severity": "medium"}],
                                   "question": "Which operators?"}],
                       "decision": [{"choice": 1, "why": "the operators are unclear"}]}
        s = self.think("write a function in javascript that takes firstNumber, operator, and secondNumber")
        cycle = s["cycles"][0]
        self.assertEqual([c["type"] for c in cycle["candidates"]], ["ask", "make"])  # the no-tool "tool" step was kept as a make step
        self.assertIn("became a make step", cycle["changed"][0])
        self.assertEqual((cycle["decision"]["by"], cycle["decision"]["choice"]), ("rule", 2))  # nothing high-severity to ask about
        _, body = [b for b in Fake.bodies if b[0] == "language"][-1]
        self.assertIn("make what the person asked for", body["messages"][0]["content"])
        self.assertEqual(body["max_tokens"], next(m for m in self.config["modules"] if m["id"] == "language")["make_max_tokens"])
        self.assertEqual(self.calls("language")[-1]["open_details"], ["Operators aren't listed"])
        self.assertEqual([t["part"] for t in s["trace"] if t["module"] == "language"], ["make"])

    def test_never_asks_twice_in_a_row(self):
        first = self.h.think(self.h.start(MEETING)).state
        self.assertTrue(self.h.conversations.get(first["conversation"])["turns"][0]["asked"])
        Fake.script = {"planner": [{"goal": "g", "candidates": [cand("ask", "Ask again"), cand("answer", "Give tips for both sides")]}],
                       "critic": [{"issues": [{"kind": "missing", "text": "Still unclear", "severity": "high"}], "question": "Which side?"}]}
        second = self.h.think(self.h.start("not sure yet", first["conversation"])).state
        cycle = second["cycles"][0]
        self.assertEqual((cycle["decision"]["by"], chosen(cycle)["type"]), ("rule", "answer"))
        self.assertIn("asked a question last turn", cycle["decision"]["why"])
        self.assertFalse(self.h.conversations.get(first["conversation"])["turns"][1]["asked"])

    def test_details_of_a_request_arent_kept_as_memories(self):
        Fake.script = {"perception": [{"goal": "write a function", "observations": [
            ob("fact", "User specifies programming language"), ob("fact", "Function requires three arguments"),
            ob("preference", "User prefers the assistant's previous suggestion"), ob("fact", "User's dentist is Dr. Lee")]}]}
        s = self.think("write me a function, and by the way my dentist is Dr. Lee")
        self.assertEqual([m["text"] for m in s["remembered"]], ["User's dentist is Dr. Lee"])


class StubRouter:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def route(self, text, prev="", tools=None):
        self.calls.append({"text": text, "prev": prev, "tools": list(tools or [])})
        return dict(self.reply, ms=0.1)


class StubMath:
    """Plays a math model that gets + and - right and writes every * one too big."""

    def __init__(self):
        self.asked = []

    def fits(self, a, op, b):
        return len(str(a)) + len(str(b)) <= 20

    def answer(self, a, op, b):
        self.asked.append("%d%s%d" % (a, op, b))
        return str(a + b if op == "+" else a - b if op == "-" else a * b + 1)


class StubWholeMath:
    """Plays a whole-expression math model that writes its work, and gets anything divided by 27 wrong by 1."""
    takes = "expression"

    def __init__(self):
        self.asked = []

    def fits(self, expression):
        return len(expression) <= 60

    def solve(self, expression):
        self.asked.append(expression)
        value = harness.exact_value(expression) + (1 if "/27" in expression else 0)
        return {"answer": harness.exact_text(value), "work": ["worked " + expression]}


class StubReader:
    """Plays a math reader: hands back the same problems for every message, as sure as it's told to be."""

    def __init__(self, problems, weakest=1.0):
        self.problems, self.calls, self.weakest = problems, [], weakest

    def read(self, text):
        self.calls.append(text)
        return [dict(p) for p in self.problems]

    def read_with_confidence(self, text):
        return self.read(text), 0.99, self.weakest


class StubLanguage:
    def __init__(self, reply="reply from local language model"):
        self.reply, self.calls = reply, []

    def complete(self, system, user, max_tokens, temperature, **settings):
        self.calls.append(dict({"system": system, "user": user, "max_tokens": max_tokens,
                                "temperature": temperature}, **settings))
        return self.reply


class Routing(Rig):
    def test_local_foundation_model_is_offered_only_for_language_and_can_reply(self):
        stub = self.trained("smLANGUAGE_EN_001", StubLanguage(), "language")
        self.assertIn("smmol/smLANGUAGE_EN_001", [o["choice"] for o in self.h.options("language", self.h.status(max_age=0))])
        self.assertNotIn("smmol/smLANGUAGE_EN_001", [o["choice"] for o in self.h.options("perception", self.h.status(max_age=0))])
        self.h.choose("language", "smmol/smLANGUAGE_EN_001")
        self.only("language")
        state = self.think("Tell me something.")
        self.assertEqual(state["reply"], stub.reply)
        self.assertIn('"message": "Tell me something."', stub.calls[0]["user"])

    def test_untrained_router_is_skipped_and_says_so(self):
        s = self.think("What's 17.5% of 2,340?")
        self.assertEqual(s["status"], "done")
        self.assertNotIn("router", {t["module"] for t in s["trace"]})
        offered = self.h.options("router", self.h.status(max_age=0))
        self.assertEqual([(o["choice"], o["up"], o.get("note")) for o in offered], [("smmol/smROUTER_01", False, "not trained yet")])
        with self.assertRaises(HarnessError):
            self.h.choose("router", "smmol/smROUTER_01")

    def test_router_shows_up_only_for_the_router_slot(self):
        self.use_router()
        status = self.h.status(max_age=0)
        self.assertEqual([o["choice"] for o in self.h.options("router", status)], ["smmol/smROUTER_01"])
        self.assertNotIn("smmol/smROUTER_01", [o["choice"] for o in self.h.options("planner", status)])

    def test_router_goes_first_and_only_sees_tools_that_are_on(self):
        self.h.toggle("tool", "web_browser", False)
        self.h.toggle("tool", "file_system", False)
        stub = self.use_router(tool="calculator")
        self.h.toggle("tool", "web_browser", False)  # write_config reloaded the harness; switches live in the data folder
        s = self.think("What's 17.5% of 2,340?")
        self.assertEqual(s["trace"][0]["module"], "router")
        self.assertEqual(stub.calls[0]["tools"], ["calculator", "web_search", "memory_recall"])
        self.assertEqual(self.calls("planner")[0]["route"]["suggested_tool"], "calculator")

    def test_router_stops_small_talk_on_its_own(self):
        self.use_router(for_me=False, intent="small_talk", intent_p=0.97)
        s = self.think("lol ok")
        self.assertEqual((s["status"], [t["module"] for t in s["trace"]]), ("stopped", ["router"]))

    def test_router_turns_a_needless_question_into_an_answer(self):
        self.use_router(intent="statement", ask_first=False, ask_p=0.03)
        s = self.think(MEETING)  # the fake Critic raises a high issue and the fake Decision picks "ask"
        cycle = s["cycles"][0]
        self.assertEqual((cycle["decision"]["by"], cycle["candidates"][cycle["decision"]["choice"] - 1]["type"]), ("router", "answer"))
        self.assertFalse(s["reply"].endswith("?"))

    def test_router_makes_it_search_instead_of_guessing(self):
        self.use_router(tool="web_search", tool_p=0.93)
        s = self.think("What's the weather in Chicago today?")
        first = s["cycles"][0]
        self.assertEqual(first["decision"]["by"], "router")
        self.assertEqual((first["action"]["tool"], first["action"]["target"]), ("web_search", "What's the weather in Chicago today?"))
        self.assertIn("weather.example", json.dumps(self.calls("language")[-1]["tool_results"]))

    def test_router_switched_off_leaves_no_trace(self):
        self.use_router(tool="calculator")
        self.h.toggle("module", "router", False)
        s = self.think("What's 17.5% of 2,340?")
        self.assertNotIn("router", {t["module"] for t in s["trace"]})
        self.assertNotIn("route", self.calls("planner")[0])

    def test_router_never_forces_a_file_write(self):
        self.use_router(intent="task", tool="file_write", tool_p=0.98)  # what it really says about "write a function in javascript"
        Fake.script = {"planner": [{"goal": "g", "candidates": [
                           cand("make", "Write the function"), cand("tool", "Save a draft", "file_system", "create", "draft.js", "x")]}],
                       "decision": [{"choice": 1, "why": "write it"}]}
        s = self.think("write a function in javascript")
        cycle = s["cycles"][0]
        self.assertEqual((cycle["decision"]["by"], chosen(cycle)["type"]), ("model", "make"))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "workspace", "draft.js")))


class ToolAwareness(Rig):
    """The modules know which tools exist, and a lookup can happen with nothing but the Language module on."""

    def system_prompt(self, module):
        return [b["messages"][0]["content"] for name, b in Fake.bodies if name == module][-1]

    def test_the_weather_is_looked_up_even_with_no_planner(self):
        self.only("language")  # what the page looks like with everything switched off but the reply writer
        s = self.think("What's the weather in Chicago today?")
        first = s["cycles"][0]
        self.assertEqual((first["decision"]["by"], first["action"]["tool"]), ("rule", "web_search"))
        self.assertEqual(first["action"]["target"], "What's the weather in Chicago today?")
        self.assertIn("weather.example", json.dumps(self.calls("language")[-1]["tool_results"]))

    def test_a_search_is_followed_to_the_page_and_neither_repeats(self):
        self.only("planner", "decision", "language")
        s = self.think("What's the weather in Chicago today?")
        did = [(c["action"]["tool"], c["action"]["target"]) for c in s["cycles"] if c["action"]]
        self.assertEqual(did, [("web_search", "What's the weather in Chicago today?"),
                               ("web_browser", "https://weather.example/chicago")])
        self.assertIn("Example page text", json.dumps(self.calls("language")[-1]["tool_results"]))

    def test_no_page_is_opened_when_every_hit_is_drawn_in_the_browser(self):
        self.assertEqual(harness.best_link("1. a\n   https://weather.com/x\n2. b\n   https://www.accuweather.com/y"), "")
        self.assertEqual(harness.best_link("1. a\n   https://weather.com/x\n2. b\n   https://wunderground.com/y"),
                         "https://wunderground.com/y")
        self.assertEqual(harness.best_link("no links here"), "")

    def test_the_page_isnt_opened_when_the_browser_is_off(self):
        self.only("language")
        self.h.toggle("tool", "web_browser", False)
        s = self.think("What's the weather in Chicago today?")
        self.assertEqual([c["action"]["tool"] for c in s["cycles"] if c["action"]], ["web_search"])

    def test_the_language_module_is_told_what_it_can_reach_for(self):
        self.only("language")
        self.think("say hi")
        self.assertIn("web_search: searches the web", self.system_prompt("language"))
        self.h.toggle("tool", "web_search", False)
        self.think("say hi")
        self.assertNotIn("web_search:", self.system_prompt("language"))

    def test_the_language_module_can_ask_for_a_lookup(self):
        self.only("language")
        Fake.script = {"language": ["NEED: web_search who the mayor of Chicago is", "SAID the mayor"]}
        s = self.think("Who is the mayor of Chicago right this minute?")
        handed = s["cycles"][-1]
        self.assertEqual((handed["decision"]["by"], handed["action"]["tool"]), ("language", "web_search"))
        self.assertEqual(handed["action"]["target"], "who the mayor of Chicago is")
        self.assertEqual(s["reply"], "SAID the mayor")
        self.assertIn("weather.example", json.dumps(self.calls("language")[-1]["tool_results"]))

    def test_a_hand_back_it_cant_honour_never_reaches_the_person(self):
        self.only("language")
        self.h.toggle("tool", "web_search", False)
        Fake.script = {"language": ["NEED: web_search what the mayor said", "SAID I can't look that up"]}
        s = self.think("Who is the mayor of Chicago right this minute?")
        self.assertEqual(s["reply"], "SAID I can't look that up")
        self.assertEqual(len([c for c in s["cycles"] if c["action"]]), 0)
        self.assertIn("isn't available", self.calls("language")[-1]["note"])

    def test_the_language_module_never_writes_a_file_that_way(self):
        self.only("language")
        Fake.script = {"language": ["NEED: file_system notes.txt", "SAID no"]}
        s = self.think("Who is the mayor of Chicago right this minute?")
        self.assertEqual(len([c for c in s["cycles"] if c["action"]]), 0)
        self.assertNotIn("NEED", s["reply"])

    def test_a_need_line_inside_something_it_made_is_left_alone(self):
        self.only("planner", "decision", "language")
        Fake.script = {"planner": [{"goal": "g", "candidates": [cand("make", "Write the notes")]}],
                       "decision": [{"choice": 1, "why": "make it"}],
                       "language": ["Here you go:\n\nNEED: milk\nNEED: eggs"]}
        s = self.think("write me a shopping list")
        self.assertIn("NEED: milk", s["reply"])

    def test_the_page_describes_every_tool(self):
        tools = server.App(self.h).state()["tools"]
        self.assertEqual({t["name"] for t in tools}, set(self.h.tools))
        for t in tools:
            self.assertTrue(t["label"] and t["color"] and t["does"], t)
        self.assertEqual([t["built"] for t in tools if t["name"] == "command_line"], [False])


class MathModule(Rig):
    def test_an_untrained_math_module_changes_nothing(self):
        self.h.toggle("tool", "calculator", False)
        self.assertNotIn("calculator", self.h.enabled_tools())
        offered = self.h.options("math", self.h.status(max_age=0))
        self.assertEqual([(o["choice"], o.get("note")) for o in offered], [("smmol/smMATH01-abacus", "not trained yet")])
        with self.assertRaisesRegex(HarnessError, "isn't trained yet"):
            self.h.test_module("math")

    def test_math_works_each_step_and_the_calculator_checks_it(self):
        stub = self.use_math()
        Fake.script = {"planner": [{"goal": "g", "candidates": [cand("tool", "Work it out", "calculator", "none", "(48213 + 9977) * 2 - 12.5")]}]}
        s = self.think("What's (48213 + 9977) * 2 - 12.5?")
        action = s["cycles"][0]["action"]
        self.assertEqual((action["ok"], action["result"]), (True, "116367.5"))
        self.assertEqual([st["used"] for st in action["math"]["steps"]], ["model", "calculator", "model"])  # the * was wrong
        self.assertEqual((action["math"]["misses"], action["math"]["checked"]), (1, True))
        self.assertEqual(stub.asked, ["48213+9977", "58190*2", "1163800-125"])  # 116380 - 12.5 goes in as whole numbers
        self.assertIn("math", [t["module"] for t in s["trace"]])
        self.assertEqual(self.calls("language")[-1]["tool_results"][0]["result"], "116367.5")

    def test_without_the_calculator_math_answers_alone_and_refuses_what_it_cant_do(self):
        self.use_math()
        self.h.toggle("tool", "calculator", False)
        self.assertIn("calculator", self.h.enabled_tools())  # arithmetic stays on the Planner's menu...
        self.assertIn("+ - * only", self.h.tool_menu())      # ...described as what the Math module can do
        result, info = self.h.work_math(Live(new_state("x")), "12 * 3")
        self.assertEqual((result, info["checked"], info["steps"][0]["used"]), ("37", False, "model"))  # wrong, and nobody checks
        with self.assertRaisesRegex(HarnessError, "calculator is off"):
            self.h.work_math(Live(new_state("x")), "1850 / 3")
        with self.assertRaisesRegex(HarnessError, "turned off"):
            self.h.run_tool(cand("tool", "c", "calculator", "none", "1 + 1"))

    def test_math_checks_score_the_model_not_the_calculator(self):
        self.use_math()
        choice, score = self.h.test_module("math")
        self.assertEqual((choice, score["passed"], score["total"]), ("smmol/smMATH01-abacus", 5, 6))
        self.assertIn("the model wrote 5536", [r["why"] for r in score["results"] if not r["ok"]][0])
        status = self.h.status(max_age=0)
        self.assertEqual([o["choice"] for o in self.h.options("math", status)], ["smmol/smMATH01-abacus"])
        self.assertNotIn("smmol/smMATH01-abacus", [o["choice"] for o in self.h.options("router", status)])

    def test_router_and_math_answer_arithmetic_with_every_language_model_off(self):
        self.use_router(tool="calculator", tool_p=0.97)
        stub = self.use_math()
        for mid in ("perception", "attention", "recall", "planner", "predictor", "critic", "decision", "language"):
            self.h.toggle("module", mid, False)
        s = self.think("hey what's 48213 + 9977?")
        self.assertEqual((s["status"], s["reply"]), ("done", "58190"))
        self.assertEqual(stub.asked, ["48213+9977"])
        self.assertEqual([t["module"] for t in s["trace"]], ["router", "math"])
        self.assertEqual(Fake.bodies, [])  # no language model was asked anything


    def test_just_arithmetic_goes_to_math_even_when_the_router_calls_it_small_talk(self):
        self.use_router(for_me=False, intent="small_talk", intent_p=1.0, tool="none", tool_p=1.0)  # what smROUTER_01 says about "2+2"
        stub = self.use_math()
        for mid in ("perception", "attention", "recall", "planner", "predictor", "critic", "decision", "language"):
            self.h.toggle("module", mid, False)
        s = self.think("2+2")
        self.assertEqual((s["status"], s["reply"]), ("done", "4"))
        self.assertEqual((s["cycles"][0]["decision"]["by"], s["cycles"][0]["decision"]["why"]), ("rule", "the message is just arithmetic"))
        self.assertEqual(stub.asked, ["2+2"])

    def test_just_arithmetic_works_with_only_math_switched_on(self):
        stub = self.use_math()
        for mid in self.h.order:
            if mid != "math":
                self.h.toggle("module", mid, False)
        self.assertEqual(self.think("12 + 30 =")["reply"], "42")
        self.assertEqual(stub.asked, ["12+30"])


class MathLanguage(Rig):
    VOLUME = "what is the volume of 322234ft x 21323ft x 212231ft?"

    def test_problems_are_read_worked_out_and_answered_with_their_units(self):
        self.use_reader([{"id": "p1", "expression": "3*4*5", "unit": "cubic feet", "about": "volume"},
                         {"id": "p2", "expression": "p1/27", "unit": "cubic yards", "about": "volume"}])
        self.only("math_language")
        s = self.think("Volume of a 3ft x 4ft x 5ft box, and how many cubic yards is that?")
        self.assertEqual([(p["answer"], p["by"]) for p in s["problems"]], [("60", "calculator"), ("2.222222222", "calculator")])
        self.assertEqual(s["problems"][1]["worked"], "(60)/27")
        self.assertEqual(s["reply"], "3*4*5 = 60 cubic feet\np1/27 = 2.222222222 cubic yards")
        self.assertEqual([t["module"] for t in s["trace"]], ["math_language"])

    def test_a_whole_expression_math_model_does_the_problems_and_the_calculator_checks_it(self):
        self.use_reader([{"id": "p1", "expression": "322234*21323*212231", "unit": "cubic feet", "about": "volume"},
                         {"id": "p2", "expression": "p1/27", "unit": "cubic yards", "about": "volume"}])
        whole = self.use_whole_math()
        self.only("math_language", "math")
        s = self.think(self.VOLUME)
        p1, p2 = s["problems"]
        self.assertEqual((p1["answer"], p1["by"], p1["math"]["steps"][0]["used"]), ("1458238263363442", "Math", "model"))
        self.assertEqual(p1["math"]["work"], ["worked 322234*21323*212231"])
        self.assertEqual((p2["answer"], p2["math"]["misses"]), ("54008824569016.3703703704", 1))  # its /27 was off; the calculator caught it
        self.assertEqual(whole.asked, ["322234*21323*212231", "1458238263363442/27"])
        self.assertEqual([t["module"] for t in s["trace"]], ["math_language", "math", "math"])

    def test_a_step_math_model_does_the_problems_too(self):
        self.use_reader([{"id": "p1", "expression": "48213+9977", "unit": "", "about": ""}])
        stub = self.use_math()
        self.only("math_language", "math")
        s = self.think("48213 plus 9977 please")
        self.assertEqual((s["problems"][0]["answer"], s["reply"], stub.asked), ("58190", "48213+9977 = 58190", ["48213+9977"]))

    def test_percent_is_reordered_so_the_division_comes_last(self):
        self.assertEqual(harness.plain_arithmetic("17.5% of 2,340"), "17.5*2340/100")
        self.assertEqual(harness.plain_arithmetic("(48213 + 9977) * 2"), "(48213+9977)*2")
        self.assertIsNone(harness.plain_arithmetic("sqrt(16) + 1"))
        self.assertEqual(harness.exact_text(harness.exact_value("1458238263363442/27")), "54008824569016.3703703704")

    def test_off_untrained_or_no_numbers_means_it_never_reads(self):
        self.assertEqual(self.think(self.VOLUME)["problems"], [])  # not trained in the rig
        reader = self.use_reader([{"id": "p1", "expression": "2+2", "unit": "", "about": ""}])
        self.think("tell me a joke")
        self.assertEqual(reader.calls, [])
        self.h.set_setting("math_language", "only_with_numbers", False)
        self.think("tell me a joke")
        self.assertEqual(reader.calls, ["tell me a joke"])
        self.h.toggle("module", "math_language", False)
        s = self.think("what's 2+2")
        self.assertEqual((len(reader.calls), s["problems"]), (1, []))
        self.assertNotIn("problems", self.calls("planner")[-1])

    def test_planner_and_language_see_the_answers(self):
        self.use_reader([{"id": "p1", "expression": "2340*17.5/100", "unit": "", "about": "percent"}])
        self.think("whats 17.5% of 2,340")
        self.assertEqual(self.calls("planner")[-1]["problems"][0]["answer"], "409.5")
        self.assertEqual(self.calls("language")[-1]["problems"][0]["answer"], "409.5")

    def test_an_unsure_reader_is_dropped_or_flagged(self):
        problems = [{"id": "p1", "expression": "2+2", "unit": "", "about": ""}]
        self.use_reader(problems, weakest=0.4)  # below the 0.7 floor
        self.only("math_language")
        s = self.think("what's 2+2 then")
        self.assertEqual(s["problems"], [])
        self.assertEqual((s["math_language"]["trusted"], s["math_language"]["weakest"]), (False, 0.4))
        self.assertIn("too unsure", s["math_language"]["note"])

        self.use_reader(problems, weakest=0.9)  # between the floors: kept, but flagged
        s = self.think("what's 2+2 then")
        self.assertEqual((s["problems"][0]["answer"], s["problems"][0]["unsure"]), ("4", True))
        self.assertTrue(self.h.module_on("math_language") and s["math_language"]["trusted"])

        self.h.set_setting("math_language", "trust_above", 0.95)  # a stricter floor drops it too
        s = self.think("what's 2+2 then")
        self.assertEqual(s["problems"], [])

    def test_math_language_checks_grade_by_value(self):
        self.use_reader([{"id": "p1", "expression": "322234*21323*212231", "unit": "cubic feet", "about": "volume"}])
        choice, score = self.h.test_module("math_language")
        self.assertEqual((choice, score["passed"], score["total"]), ("smmol/smMATH_LANGUAGE_001", 1, 6))


class Settings(Rig):
    def test_settings_change_what_modules_send_and_are_saved(self):
        self.h.set_setting("planner", "temperature", 0.9)
        self.h.set_setting("planner", "max_tokens", 1234)
        self.think("What's 17.5% of 2,340?")
        _, body = next(b for b in Fake.bodies if b[0] == "planner")
        self.assertEqual((body["temperature"], body["max_tokens"]), (0.9, 1234))
        self.assertEqual(Harness(self.dir).setting("planner", "max_tokens"), 1234)
        self.h.set_setting("planner", "max_tokens", 600)  # back to its default, so it isn't stored
        self.assertNotIn("max_tokens", self.h.settings.get("planner", {}))
        for bad in (("planner", "temperature", 9), ("planner", "temperature", "hot"), ("math", "check", 1),
                    ("planner", "nope", 1), ("cerebellum", "temperature", 1)):
            with self.assertRaises(HarnessError, msg=str(bad)):
                self.h.set_setting(*bad)

    def test_router_thresholds_are_settings(self):
        self.use_router(for_me=False, intent="small_talk", intent_p=0.85)
        self.assertEqual(self.think("hmm ok")["status"], "done")  # 85% is under the 90% default, so it carries on
        self.h.set_setting("router", "stop_small_talk_at", 0.8)
        self.assertEqual(self.think("hmm ok")["status"], "stopped")

    def test_local_language_declares_live_generation_knobs(self):
        name = "smLANGUAGE_en_SCH_001-kindergarten"
        path = os.path.join(self.dir, name + ".safetensors")
        open(path, "wb").close()
        knobs = [
            {"key": "top_k", "label": "Top K", "type": "int", "min": 0, "max": 256, "step": 1, "default": 40},
            {"key": "repetition_penalty", "label": "Repetition penalty", "type": "number",
             "min": 1.0, "max": 2.0, "step": 0.05, "default": 1.05},
        ]
        self.config["providers"]["smmol"]["models"][name] = {
            "path": path, "kind": "language", "settings": knobs,
        }
        stub = StubLanguage()
        self.stubs[(os.path.normpath(path), os.path.getmtime(path))] = stub
        self.write_config()
        self.h.choose("language", "smmol/" + name)
        keys = [setting["key"] for setting in self.h.settings_for("language")]
        self.assertEqual(keys, ["temperature", "max_tokens", "make_max_tokens", "top_k", "repetition_penalty"])
        self.h.set_setting("language", "top_k", 17)
        self.h.set_setting("language", "repetition_penalty", 1.2)
        self.only("language")
        self.think("Use the word bright in a sentence.")
        self.assertEqual((stub.calls[0]["top_k"], stub.calls[0]["repetition_penalty"]), (17, 1.2))
        with self.assertRaises(HarnessError):
            self.h.set_setting("language", "top_k", 300)

    def test_local_language_exposes_editable_next_checkpoint_shape(self):
        name = "school-weights"
        artifact_dir = os.path.join(self.dir, "out")
        os.makedirs(artifact_dir)
        path = os.path.join(artifact_dir, "latest.safetensors")
        with open(path, "wb") as handle:
            handle.write(b"weights")
        architecture = {"vocab_size": 256, "context_length": 128, "width": 192, "layers": 4,
                        "heads": 6, "dropout": 0.1, "tie_embeddings": True}
        document = {"architecture": architecture, "tokenizer": {"kind": "utf8_bytes", "vocab_size": 256}}
        artifact_config = os.path.join(artifact_dir, "model_config.json")
        training_config = os.path.join(self.dir, "model_config.json")
        for config_path in (artifact_config, training_config):
            with open(config_path, "w") as handle:
                json.dump(document, handle)
        with open(os.path.join(artifact_dir, "metadata.json"), "w") as handle:
            json.dump({"stage": "kindergarten", "step": 500, "format": "safetensors"}, handle)
        self.config["providers"]["smmol"]["models"][name] = {
            "path": path, "kind": "language", "training_config": training_config,
        }
        self.write_config()
        self.h.choose("language", "smmol/" + name)
        weights = self.h.weight_config("language")
        self.assertEqual((weights["current"]["stage"], weights["current"]["step"]), ("kindergarten", 500))
        self.assertEqual(weights["current"]["parameters"], 1_846_656)
        self.h.set_weight_config("language", "layers", 6)
        self.assertTrue(self.h.weight_config("language")["changed"])
        with open(training_config) as handle:
            saved = json.load(handle)
        self.assertEqual(saved["architecture"]["layers"], 6)
        with self.assertRaisesRegex(HarnessError, "divisible"):
            self.h.set_weight_config("language", "heads", 5)

    def test_math_can_answer_unchecked_while_the_calculator_still_does_the_rest(self):
        self.use_math()
        self.h.set_setting("math", "check", False)
        result, info = self.h.work_math(Live(new_state("x")), "12 * 3")
        self.assertEqual((result, info["checked"]), ("37", False))  # wrong, and nobody checked
        self.assertEqual(self.h.work_math(Live(new_state("x")), "1850 / 3")[0], "616.6666667")
        self.assertEqual([s["key"] for s in self.h.settings_for("math")], ["check"])
        self.assertEqual([s["key"] for s in self.h.settings_for("language")], ["temperature", "max_tokens", "make_max_tokens"])


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

    def wait_for(self, rid):
        conn = http.client.HTTPConnection(self.host, timeout=10)
        conn.request("GET", "/api/runs/%s/events" % rid, headers={"Host": self.host})
        resp = conn.getresponse()
        for line in iter(resp.fp.readline, b""):
            if line.startswith(b"data: "):
                last = json.loads(line[6:])
                if last["status"] != "running":
                    return last
        raise AssertionError("the event stream ended early")

    def test_replies_stay_in_one_chat(self):
        status, body = self.call("POST", "/api/run", {"message": MEETING})
        first = json.loads(body)
        self.assertEqual(self.wait_for(first["id"])["status"], "done")
        status, body = self.call("POST", "/api/run", {"message": "I'm the investor", "conversation": first["conversation"]})
        second = json.loads(body)
        self.assertEqual((status, second["conversation"]), (200, first["conversation"]))
        done = self.wait_for(second["id"])
        self.assertEqual(done["history"][0]["message"], MEETING)
        chat = json.loads(self.call("GET", "/api/conversations/%s" % first["conversation"])[1])
        self.assertEqual([t["message"] for t in chat["turns"]], [MEETING, "I'm the investor"])
        self.assertEqual(json.loads(self.call("GET", "/api/runs/%s" % first["id"])[1])["message"], MEETING)
        listed = json.loads(self.call("GET", "/api/conversations")[1])["conversations"]
        self.assertEqual(listed[0]["id"], first["conversation"])
        self.assertEqual(self.call("GET", "/api/conversations/20260101-000000-abcdef")[0], 404)
        self.assertEqual(self.call("POST", "/api/run", {"message": "hi", "conversation": "nope"})[0], 400)

    def test_switches_from_the_page(self):
        status, body = self.call("POST", "/api/toggle", {"kind": "module", "id": "critic", "on": False})
        self.assertEqual(status, 200)
        self.assertFalse(next(m for m in json.loads(body)["modules"] if m["id"] == "critic")["on"])
        self.assertEqual(self.call("POST", "/api/toggle", {"kind": "tool", "id": "command_line", "on": True})[0], 400)
        self.assertEqual(self.call("POST", "/api/toggle", {"kind": "module", "id": "critic", "on": "no"})[0], 400)

    def test_settings_from_the_page(self):
        status, body = self.call("POST", "/api/settings", {"module": "critic", "key": "temperature", "value": 0.5})
        self.assertEqual(status, 200)
        critic = next(m for m in json.loads(body)["modules"] if m["id"] == "critic")
        self.assertEqual(next(s["value"] for s in critic["settings"] if s["key"] == "temperature"), 0.5)
        self.assertEqual(self.call("POST", "/api/settings", {"module": "critic", "key": "temperature", "value": 5})[0], 400)

    def test_refuses_other_hosts_origins_and_non_json(self):
        self.assertEqual(self.call("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.call("POST", "/api/run", {"message": "hi"}, {"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.call("POST", "/api/run", {"message": "hi"}, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.call("POST", "/api/run", {"message": ""})[0], 400)

    def test_thinks_live_and_forgets_on_request(self):
        status, body = self.call("POST", "/api/run", {"message": "Remember that my dentist is Dr. Lee"}, {"Origin": "http://" + self.host})
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
        self.assertEqual(last["status"], "done")
        state = json.loads(self.call("GET", "/api/state")[1])
        self.assertEqual(state["memory"]["count"], 1)
        self.assertEqual(len(state["modules"]), 11)
        self.assertIn("command_line", [t["name"] for t in state["tools"] if not t["on"]])
        status, body = self.call("POST", "/api/forget", {"id": state["memory"]["recent"][0]["id"]})
        self.assertEqual((status, json.loads(body)["memory"]["count"]), (200, 0))


if __name__ == "__main__":
    unittest.main()
