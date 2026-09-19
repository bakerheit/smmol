"""smCLM_02 phase 1 tools: the teacher client, the proposals, the tagging, the held-out pick.

These tests never open a socket. `teacher.Teacher` is either in dry-run mode or replaced by a fake
that answers from a script, so the whole of phase 1's plumbing — batching, resuming, validating,
aggregating, stratifying — is exercised with no teacher and no PC.

What they're really guarding:

  * a dry run sends nothing and can't be mistaken for an answer;
  * a call whose words came back wrong, or with an idea outside the inventory, is thrown away
    whole, not half-kept;
  * resuming asks only for what's missing, and the aggregate is the same file every time;
  * soft weights are the fraction of passes that gave the idea;
  * held-out words keep no teacher tags, and no script ever writes a tag a person should write;
  * ideas.json starts with smCLM_01's 51, verbatim and in order, and tagging refuses a seed-only
    inventory unless it's told to.

    python3 -m unittest discover -s tests
"""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import unittest.mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import held_out  # noqa: E402
import propose_ideas  # noqa: E402
import tag_words  # noqa: E402
import teacher as teacher_module  # noqa: E402
from teacher import BudgetExceeded, DryRun, Teacher  # noqa: E402


INVENTORY = {
    "health": "to do with being well or unwell",
    "person": "is a human being",
    "money": "to do with paying and being paid",
    "food": "is eaten",
    "speed": "to do with how fast something is",
}

VOCABULARY = [
    {"word": "doctor", "count": 140, "sources": ["conversations"]},
    {"word": "apple", "count": 60, "sources": ["curriculum"]},
    {"word": "run", "count": 44, "sources": ["conversations"]},
    {"word": "quick", "count": 18, "sources": ["curriculum"]},
    {"word": "wage", "count": 7, "sources": ["conversations"]},
    {"word": "tooth", "count": 3, "sources": ["v1"]},
]


class Fake:
    """A Teacher stand-in that reads answers off a list and counts what it was asked.

    Callable entries are called with the metadata, so a test can fail a call once and pass it the
    second time — which is how the retry and resume paths get exercised without a network.
    """

    def __init__(self, answers, endpoint="http://fake", log_path=None):
        self.answers = list(answers)
        self.endpoint = endpoint
        self.log_path = log_path
        self.asked = []
        self.dry_run = False

    def ask(self, kind, system, prompt, schema, seed, temperature, max_tokens, metadata=None):
        self.asked.append({"kind": kind, "seed": seed, "metadata": metadata or {},
                           "prompt": prompt, "schema": schema})
        if not self.answers:
            raise AssertionError("the fake teacher was asked more often than the test scripted")
        answer = self.answers.pop(0)
        return answer(metadata or {}) if callable(answer) else answer


class Scratch(unittest.TestCase):
    """A temporary folder with a vocabulary and an inventory in it."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="smclm02-phase1-"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.vocabulary = self.dir / "vocabulary.jsonl"
        teacher_module.write_jsonl(self.vocabulary, VOCABULARY)
        self.ideas = self.dir / "ideas.json"
        teacher_module.write_json(self.ideas, INVENTORY)
        self.log = self.dir / "calls.jsonl"

    def words(self):
        return [row["word"] for row in VOCABULARY]


class TeacherClient(Scratch):
    def test_a_dry_run_sends_nothing_and_refuses_to_answer(self):
        client = Teacher("http://127.0.0.1:8081", "ministral-8b", self.log, dry_run=True)
        schema = {"type": "object", "properties": {}}
        with self.assertRaises(DryRun) as caught:
            client.ask("kind", "system", "prompt", schema, 1, 0.0, 10)
        body = caught.exception.body
        self.assertEqual(body["seed"], 1)
        self.assertEqual(body["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(body["messages"][1]["content"], "prompt")
        logged = teacher_module.read_jsonl(self.log)
        self.assertEqual([row["sent"] for row in logged], [False])
        self.assertTrue(logged[0]["dry_run"])

    def test_the_request_is_strict_json_with_the_schema_and_the_seed(self):
        client = Teacher("http://x/", "m", self.log)
        body = client.body("tags", "sys", "ask", {"type": "object"}, 16001, 0.7, 100)
        self.assertEqual(body["model"], "m")
        self.assertFalse(body["stream"])
        self.assertEqual(body["seed"], 16001)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_openai_uses_luna_reasoning_none_and_not_pc_only_parameters(self):
        client = Teacher("https://api.openai.com", "gpt-5.6-luna", self.log,
                         provider="openai", api_key="test-key", max_budget_usd=0.25)
        body = client.body("tags", "sys", "ask", {"type": "object"}, 16001, 0.7, 100)
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["max_completion_tokens"], 100)
        self.assertNotIn("temperature", body)
        self.assertNotIn("seed", body)
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(client.completion_url(), "https://api.openai.com/v1/chat/completions")

    def test_openai_key_is_sent_as_a_header_but_never_logged(self):
        secret = "sk-test-never-log-this"
        client = Teacher("https://api.openai.com/v1", "gpt-5.6-luna", self.log,
                         provider="openai", api_key=secret, max_budget_usd=0.25)
        captured = {}

        def reply(url, body, timeout=900, headers=None):
            captured.update({"url": url, "headers": headers, "body": body})
            return {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            }

        with unittest.mock.patch.object(teacher_module, "post_json", reply):
            answer = client.ask("k", "s", "p", {}, 1, 0.0, 10)
        self.assertEqual(answer, {"ok": True})
        self.assertEqual(captured["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer " + secret)
        self.assertNotIn(secret, self.log.read_text())
        logged = teacher_module.read_jsonl(self.log)[0]
        self.assertEqual(logged["cost_usd"], 0.000032)
        self.assertEqual(logged["budget_usd"], 0.25)

    def test_openai_missing_key_fails_before_opening_a_socket(self):
        client = Teacher("https://api.openai.com", "gpt-5.6-luna", self.log,
                         provider="openai", max_budget_usd=0.25)
        with unittest.mock.patch.object(teacher_module, "post_json") as post:
            with self.assertRaisesRegex(RuntimeError, "fresh key"):
                client.ask("k", "s", "p", {}, 1, 0.0, 10)
        post.assert_not_called()

    def test_openai_budget_cap_blocks_before_opening_a_socket(self):
        client = Teacher("https://api.openai.com", "gpt-5.6-luna", self.log,
                         provider="openai", api_key="test-key", max_budget_usd=0.000001)
        with unittest.mock.patch.object(teacher_module, "post_json") as post:
            with self.assertRaises(BudgetExceeded):
                client.ask("k", "s", "p", {}, 1, 0.0, 100)
        post.assert_not_called()
        row = teacher_module.read_jsonl(self.log)[0]
        self.assertFalse(row["sent"])
        self.assertTrue(row["budget_blocked"])

    def test_openai_dry_run_needs_no_key_and_logs_no_authorization(self):
        client = Teacher("https://api.openai.com", "gpt-5.6-luna", self.log, dry_run=True,
                         provider="openai", max_budget_usd=0.25)
        with self.assertRaises(DryRun):
            client.ask("k", "s", "p", {}, 1, 0.0, 10)
        text = self.log.read_text()
        self.assertNotIn("Authorization", text)
        self.assertNotIn("api_key", text)

    def test_a_failed_call_is_retried_and_every_attempt_is_logged(self):
        client = Teacher("http://x", "m", self.log, retries=3)
        tries = []

        def flaky(url, body, timeout=900):
            tries.append(url)
            if len(tries) < 3:
                raise ValueError("boom")
            return {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {}}

        original, teacher_module.post_json = teacher_module.post_json, flaky
        self.addCleanup(setattr, teacher_module, "post_json", original)
        client.retries = 3
        with unittest.mock.patch.object(teacher_module.time, "sleep"):
            answer = client.ask("k", "s", "p", {}, 1, 0.0, 10)
        self.assertEqual(answer, {"ok": True})
        logged = teacher_module.read_jsonl(self.log)
        self.assertEqual([row["ok"] for row in logged], [False, False, True])

    def test_thinking_tags_are_stripped_before_the_json_is_read(self):
        client = Teacher("http://x", "m", self.log)
        original = teacher_module.post_json
        self.addCleanup(setattr, teacher_module, "post_json", original)
        teacher_module.post_json = lambda url, body, timeout=900: {
            "choices": [{"message": {"content": "<think>hmm</think>{\"ok\": 1}"}}], "usage": {}}
        self.assertEqual(client.ask("k", "s", "p", {}, 1, 0.0, 10), {"ok": 1})

    def test_the_vocabulary_is_read_in_a_fixed_order_whatever_order_it_was_written(self):
        shuffled = self.dir / "shuffled.jsonl"
        teacher_module.write_jsonl(shuffled, list(reversed(VOCABULARY)))
        a = [row["word"] for row in teacher_module.load_vocabulary(self.vocabulary)]
        b = [row["word"] for row in teacher_module.load_vocabulary(shuffled)]
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))

    def test_a_missing_vocabulary_says_which_file_and_who_builds_it(self):
        with self.assertRaises(SystemExit) as caught:
            teacher_module.load_vocabulary(self.dir / "nope.jsonl")
        self.assertIn("vocabulary.py", str(caught.exception))

    def test_batches_are_a_fixed_cut_of_the_list(self):
        self.assertEqual(teacher_module.batches(list("abcde"), 2),
                         [(0, ["a", "b"]), (1, ["c", "d"]), (2, ["e"])])


class Inventory(Scratch):
    def v1(self):
        return json.loads((ROOT.parent / "smCLM_01" / "concepts.json").read_text())["concepts"]

    def test_ideas_json_starts_with_smclm_01s_fifty_one_ideas_verbatim(self):
        shipped = json.loads((ROOT / "ideas.json").read_text())
        v1 = self.v1()
        first = dict(list(shipped.items())[:teacher_module.V1_IDEAS])
        self.assertEqual(first, v1)
        self.assertEqual(list(first), list(v1), "the order changed; the plan says verbatim")

    def test_the_approved_inventory_loads_and_meets_the_gate(self):
        loaded = teacher_module.load_inventory(ROOT / "ideas.json")
        self.assertGreaterEqual(len(loaded), teacher_module.MIN_IDEAS)

    def test_a_seed_only_inventory_is_still_refused_unless_the_caller_says_so(self):
        seed = self.dir / "seed.json"
        teacher_module.write_json(seed, self.v1())
        with self.assertRaises(SystemExit) as caught:
            teacher_module.load_inventory(seed)
        self.assertIn("waiting on a person", str(caught.exception))
        self.assertEqual(len(teacher_module.load_inventory(seed, allow_partial=True)), teacher_module.V1_IDEAS)

    def test_an_idea_with_no_meaning_or_a_bad_name_is_refused(self):
        bad = self.dir / "bad.json"
        teacher_module.write_json(bad, {"health": ""})
        self.assertRaises(SystemExit, teacher_module.load_inventory, bad, True)
        teacher_module.write_json(bad, {"Health Care": "no"})
        self.assertRaises(SystemExit, teacher_module.load_inventory, bad, True)


class Proposals(Scratch):
    def answer(self, words, ideas=("health", "person")):
        return {"words": [{"word": w, "ideas": list(ideas)} for w in words]}

    def test_prompt_is_open_and_rejects_narrow_new_ideas(self):
        # open_v3: showing the inventory made Ministral cram words into it (see PROMPT_VERSION).
        prompt = propose_ideas.prompt_for(["apple", "doctor"], INVENTORY)
        self.assertNotIn("- health: to do with being well or unwell", prompt)
        self.assertIn("Do not force a word into a category", prompt)
        self.assertIn("at least 8 different words", prompt)
        self.assertIn("1 to %d ideas" % propose_ideas.MAX_IDEAS_PER_WORD, prompt)
        self.assertLessEqual(propose_ideas.MAX_IDEAS_PER_WORD, 3)

    def test_a_good_batch_is_kept_with_its_words_in_order(self):
        words = ["apple", "doctor"]
        got, problem = propose_ideas.check(self.answer(words), words)
        self.assertIsNone(problem)
        self.assertEqual(got, {"apple": ["health", "person"], "doctor": ["health", "person"]})

    def test_a_batch_that_lost_a_word_is_thrown_away_whole(self):
        got, problem = propose_ideas.check(self.answer(["apple"]), ["apple", "doctor"])
        self.assertIsNone(got)
        self.assertIn("doctor", problem)

    def test_a_batch_that_invented_a_word_is_thrown_away_whole(self):
        got, problem = propose_ideas.check(self.answer(["apple", "pear"]), ["apple", "doctor"])
        self.assertIsNone(got)
        self.assertIn("pear", problem)

    def test_a_batch_that_reordered_the_words_is_thrown_away_whole(self):
        got, problem = propose_ideas.check(self.answer(["doctor", "apple"]), ["apple", "doctor"])
        self.assertIsNone(got)
        self.assertIn("wrong", problem)

    def test_idea_names_are_folded_where_they_can_be_and_dropped_where_they_cannot(self):
        self.assertEqual(propose_ideas.clean_idea("Money"), "money")
        self.assertEqual(propose_ideas.clean_idea(" day-to-day "), "day_to_day")
        self.assertEqual(propose_ideas.clean_idea("food items"), "food_items")
        self.assertIsNone(propose_ideas.clean_idea(""))
        self.assertIsNone(propose_ideas.clean_idea("123"))
        self.assertIsNone(propose_ideas.clean_idea("x"))
        self.assertIsNone(propose_ideas.clean_idea("a" * 40))

    def test_a_word_left_with_no_usable_idea_fails_its_batch(self):
        answer = {"words": [{"word": "apple", "ideas": ["!!!"]}]}
        got, problem = propose_ideas.check(answer, ["apple"])
        self.assertIsNone(got)
        self.assertIn("apple", problem)

    def test_an_idea_that_is_just_the_word_back_is_dropped(self):
        answer = {"words": [{"word": "apple", "ideas": ["apple", "food"]}]}
        got, _ = propose_ideas.check(answer, ["apple"])
        self.assertEqual(got, {"apple": ["food"]})

    def test_a_broad_role_may_legitimately_tag_itself(self):
        answer = {"words": [{"word": "action", "ideas": ["action"]}]}
        got, problem = propose_ideas.check(answer, ["action"])
        self.assertIsNone(problem)
        self.assertEqual(got, {"action": ["action"]})

    def test_call_limit_counts_rejected_batches_too(self):
        todo = [(0, ["a"]), (1, ["b"]), (2, ["c"])]
        self.assertEqual(propose_ideas.scheduled_batches(todo, 2), todo[:2])
        self.assertEqual(propose_ideas.scheduled_batches(todo, 0), todo)

    def test_the_tally_counts_words_per_idea_and_is_the_same_file_every_time(self):
        done = {0: {"apple": ["food", "health"], "doctor": ["health", "person"]},
                1: {"run": ["speed"]}}
        first = propose_ideas.tally(done, self.words())
        second = propose_ideas.tally(dict(reversed(list(done.items()))), self.words())
        self.assertEqual(json.dumps(first), json.dumps(second))
        by_idea = {row["idea"]: row for row in first["proposals"]}
        self.assertEqual(by_idea["health"]["words"], 2)
        self.assertEqual(by_idea["health"]["examples"], ["apple", "doctor"])
        self.assertEqual(first["words_tagged"], 3)

    def test_the_tally_says_out_loud_that_a_person_picks_the_ideas(self):
        note = propose_ideas.tally({}, self.words())["human_gate"]
        self.assertIn("A person picks", note)

    def test_proposals_never_touch_ideas_json(self):
        before = (ROOT / "ideas.json").read_text()
        propose_ideas.tally({0: {"apple": ["food"]}}, self.words())
        self.assertEqual((ROOT / "ideas.json").read_text(), before)


class Tagging(Scratch):
    def tags(self, words, ideas=("health",), pos="noun"):
        return {"words": [{"word": w, "pos": pos, "ideas": list(ideas)} for w in words]}

    def test_the_schema_pins_the_ideas_and_the_pos_to_an_enum(self):
        schema = tag_words.schema_for(INVENTORY)
        word = schema["properties"]["words"]["items"]["properties"]
        self.assertEqual(word["ideas"]["items"]["enum"], sorted(INVENTORY))
        self.assertEqual(word["pos"]["enum"], list(teacher_module.POS))
        self.assertEqual(word["ideas"]["maxItems"], tag_words.MAX_IDEAS)

    def test_a_good_call_is_kept(self):
        got, problem = tag_words.check(self.tags(["apple"], ["food", "health"]), ["apple"], INVENTORY)
        self.assertIsNone(problem)
        self.assertEqual(got, {"apple": {"pos": "noun", "ideas": ["food", "health"]}})

    def test_an_idea_outside_the_inventory_throws_the_whole_call_away(self):
        answer = self.tags(["apple", "doctor"])
        answer["words"][1]["ideas"] = ["health", "sportsmanship"]
        got, problem = tag_words.check(answer, ["apple", "doctor"], INVENTORY)
        self.assertIsNone(got, "one bad idea must not leave the other word half-kept")
        self.assertIn("sportsmanship", problem)

    def test_a_pos_outside_the_enum_throws_the_call_away(self):
        answer = self.tags(["apple"], pos="Noun")
        got, problem = tag_words.check(answer, ["apple"], INVENTORY)
        self.assertIsNone(got)
        self.assertIn("pos", problem)

    def test_words_that_came_back_wrong_throw_the_call_away(self):
        for answer, sent in (
            (self.tags(["apple"]), ["apple", "doctor"]),
            (self.tags(["apple", "pear"]), ["apple", "doctor"]),
            (self.tags(["doctor", "apple"]), ["apple", "doctor"]),
            (self.tags(["apple", "apple"]), ["apple", "doctor"]),
        ):
            got, problem = tag_words.check(answer, sent, INVENTORY)
            self.assertIsNone(got, problem)

    def test_too_many_or_no_ideas_throws_the_call_away(self):
        answer = self.tags(["apple"])
        answer["words"][0]["ideas"] = []
        self.assertIsNone(tag_words.check(answer, ["apple"], INVENTORY)[0])
        answer["words"][0]["ideas"] = sorted(INVENTORY) * 3
        self.assertIsNone(tag_words.check(answer, ["apple"], INVENTORY)[0])

    def test_weights_are_the_share_of_passes_that_gave_the_idea(self):
        done = {
            (16001, 0): {"apple": {"pos": "noun", "ideas": ["food", "health"]}},
            (16002, 0): {"apple": {"pos": "noun", "ideas": ["food", "health"]}},
            (16003, 0): {"apple": {"pos": "noun", "ideas": ["food", "speed"]}},
        }
        rows, report = tag_words.aggregate(done, ["apple"], INVENTORY, set())
        self.assertEqual(rows[0]["ideas"], {"food": 1.0, "health": 0.67, "speed": 0.33})
        self.assertEqual(rows[0]["passes"], 3)
        self.assertEqual(report["words_with_no_idea_at_two_thirds"], [])

    def test_a_word_no_pass_agreed_on_is_flagged_for_the_gate_not_dropped(self):
        done = {(seed, 0): {"apple": {"pos": "noun", "ideas": [idea]}}
                for seed, idea in zip(tag_words.PASSES, ["food", "health", "speed"])}
        rows, report = tag_words.aggregate(done, ["apple"], INVENTORY, set())
        self.assertEqual(sorted(rows[0]["ideas"].values()), [0.33, 0.33, 0.33])
        self.assertEqual(report["words_with_no_idea_at_two_thirds"], ["apple"])

    def test_pos_is_the_majority_and_a_tie_breaks_the_same_way_every_time(self):
        done = {
            (16001, 0): {"run": {"pos": "verb", "ideas": ["speed"]}},
            (16002, 0): {"run": {"pos": "noun", "ideas": ["speed"]}},
            (16003, 0): {"run": {"pos": "verb", "ideas": ["speed"]}},
        }
        self.assertEqual(tag_words.aggregate(done, ["run"], INVENTORY, set())[0][0]["pos"], "verb")
        tied = {(16001, 0): {"run": {"pos": "verb", "ideas": ["speed"]}},
                (16002, 0): {"run": {"pos": "noun", "ideas": ["speed"]}}}
        first = tag_words.aggregate(tied, ["run"], INVENTORY, set())[0][0]["pos"]
        second = tag_words.aggregate(dict(reversed(list(tied.items()))), ["run"], INVENTORY,
                                     set())[0][0]["pos"]
        self.assertEqual(first, second)
        self.assertEqual(first, "noun", "ties go to the earliest pos, not to dict order")

    def test_a_word_short_of_all_three_passes_is_weighted_by_the_passes_it_got_and_counted(self):
        done = {(16001, 0): {"apple": {"pos": "noun", "ideas": ["food"]}},
                (16002, 0): {"apple": {"pos": "noun", "ideas": ["food"]}}}
        rows, report = tag_words.aggregate(done, ["apple"], INVENTORY, set())
        self.assertEqual(rows[0]["ideas"], {"food": 1.0})
        self.assertEqual(rows[0]["passes"], 2)
        self.assertEqual(report["words_short_of_all_passes"], 1)

    def test_held_out_words_get_no_row_at_all(self):
        done = {(16001, 0): {"apple": {"pos": "noun", "ideas": ["food"]},
                             "tooth": {"pos": "noun", "ideas": ["health"]}}}
        rows, report = tag_words.aggregate(done, ["apple", "tooth"], INVENTORY, {"tooth"})
        self.assertEqual([row["word"] for row in rows], ["apple"])
        self.assertEqual(report["held_out_dropped"], ["tooth"])

    def test_held_out_words_are_read_from_both_files_that_can_name_them(self):
        test_words = self.dir / "test_words.json"
        picked = self.dir / "held_out.json"
        teacher_module.write_json(test_words, {"words": [{"word": "tooth", "ideas": ["health"]}]})
        teacher_module.write_json(picked, {"words": [{"word": "wage", "band": "rare"}]})
        self.assertEqual(tag_words.held_out_words(test_words, picked), {"tooth", "wage"})
        self.assertEqual(tag_words.held_out_words(self.dir / "gone.json", picked), {"wage"})

    def test_the_aggregate_is_the_same_file_however_the_log_is_ordered(self):
        done = {
            (16001, 0): {"apple": {"pos": "noun", "ideas": ["food"]}},
            (16002, 0): {"apple": {"pos": "noun", "ideas": ["food", "health"]}},
            (16003, 1): {"run": {"pos": "verb", "ideas": ["speed"]}},
        }
        a = tag_words.aggregate(done, ["apple", "run"], INVENTORY, set())[0]
        b = tag_words.aggregate(dict(reversed(list(done.items()))), ["apple", "run"], INVENTORY,
                                set())[0]
        self.assertEqual(json.dumps(a), json.dumps(b))
        self.assertEqual([row["word"] for row in a], ["apple", "run"])

    def test_a_pass_outside_the_ones_asked_for_is_not_counted(self):
        done = {(16001, 0): {"apple": {"pos": "noun", "ideas": ["food"]}},
                (99999, 0): {"apple": {"pos": "noun", "ideas": ["speed"]}}}
        rows, _ = tag_words.aggregate(done, ["apple"], INVENTORY, set(), passes=(16001,))
        self.assertEqual(rows[0]["ideas"], {"food": 1.0})

    def test_the_prompt_lists_every_idea_with_its_meaning_and_every_word_once(self):
        prompt = tag_words.prompt_for(["apple", "doctor"], INVENTORY)
        for idea, meaning in INVENTORY.items():
            self.assertIn("- %s: %s" % (idea, meaning), prompt)
        self.assertEqual(prompt.count("\napple"), 1)


class HeldOutPick(Scratch):
    def test_pos_comes_from_the_log_by_majority_so_dropping_tags_cant_move_the_pick(self):
        for seed, pos in ((16001, "verb"), (16002, "noun"), (16003, "noun")):
            teacher_module.append_jsonl(self.log, {"pass": seed, "batch": 0, "ok": True,
                                                   "tags": {"run": {"pos": pos, "ideas": ["speed"]}}})
        teacher_module.append_jsonl(self.log, {"pass": 16001, "batch": 1, "ok": False, "why": "x"})
        self.assertEqual(held_out.pos_from_log(self.log), {"run": "noun"})

    def test_an_excluded_word_is_never_picked(self):
        self.assertIn("theo", held_out.EXCLUDE)
        picked = {r["word"] for r in json.loads((ROOT / "data" / "held_out.json").read_text())["words"]}
        self.assertFalse(picked & set(held_out.EXCLUDE))


class RepeatedIdeas(Scratch):
    def test_a_repeated_idea_is_dropped_not_the_whole_call(self):
        answer = {"words": [{"word": "apple", "pos": "noun", "ideas": ["food", "food", "health"]}]}
        tags, problem = tag_words.check(answer, ["apple"], INVENTORY)
        self.assertIsNone(problem)
        self.assertEqual(tags["apple"]["ideas"], ["food", "health"])


class Resuming(Scratch):
    """The log is the resume point, for both scripts."""

    def test_proposals_only_ask_for_the_batches_the_log_is_missing(self):
        version = propose_ideas.PROMPT_VERSION
        teacher_module.append_jsonl(self.log, {"batch": 0, "ok": True, "prompt_version": version,
                                               "proposals": {"apple": ["food"]}})
        teacher_module.append_jsonl(self.log, {"batch": 1, "ok": False, "why": "words came back wrong",
                                               "prompt_version": version})
        done = propose_ideas.done_batches(self.log)
        self.assertEqual(sorted(done), [0], "a rejected batch must be asked again")
        self.assertEqual(done[0], {"apple": ["food"]})

    def test_a_batch_from_another_prompt_version_is_asked_again(self):
        teacher_module.append_jsonl(self.log, {"batch": 0, "ok": True, "prompt_version": "reuse_v2",
                                               "proposals": {"apple": ["food"]}})
        teacher_module.append_jsonl(self.log, {"batch": 1, "ok": True,
                                               "proposals": {"doctor": ["health"]}})
        self.assertEqual(propose_ideas.done_batches(self.log), {},
                         "an old prompt's answers must not be mixed into this prompt's tally")

    def test_tagging_only_asks_for_the_calls_the_log_is_missing(self):
        teacher_module.append_jsonl(self.log, {"pass": 16001, "batch": 0, "ok": True,
                                               "tags": {"apple": {"pos": "noun", "ideas": ["food"]}}})
        teacher_module.append_jsonl(self.log, {"pass": 16002, "batch": 0, "ok": False, "why": "nope"})
        done = tag_words.done_calls(self.log)
        self.assertEqual(sorted(done), [(16001, 0)])

    def test_a_log_written_twice_does_not_double_count_a_batch(self):
        for _ in range(2):
            teacher_module.append_jsonl(self.log, {"pass": 16001, "batch": 0, "ok": True,
                                                   "tags": {"apple": {"pos": "noun",
                                                                      "ideas": ["food"]}}})
        done = tag_words.done_calls(self.log)
        rows, _ = tag_words.aggregate(done, ["apple"], INVENTORY, set())
        self.assertEqual(rows[0]["ideas"], {"food": 1.0}, "a replayed batch is still one pass")

    def test_a_whole_tagging_pass_runs_resumes_and_lands_on_the_same_file(self):
        """The loop from main(), driven by a fake: fail one call, resume, aggregate twice."""
        words = self.words()
        plan = teacher_module.batches(words, 3)
        fake = Fake([
            lambda meta: {"words": [{"word": w, "pos": "noun", "ideas": ["food"]}
                                    for w in plan[0][1]]},
            lambda meta: {"words": [{"word": "wrong", "pos": "noun", "ideas": ["food"]}]},
        ])
        kept = self.run_pass(fake, plan, 16001)
        self.assertEqual(kept, 1)
        self.assertEqual(sorted(tag_words.done_calls(self.log)), [(16001, 0)])

        fake = Fake([lambda meta: {"words": [{"word": w, "pos": "noun", "ideas": ["food"]}
                                             for w in plan[1][1]]}])
        self.run_pass(fake, plan, 16001)
        self.assertEqual([meta["batch"] for meta in (a["metadata"] for a in fake.asked)], [1],
                         "the resumed run must only ask for batch 1")
        done = tag_words.done_calls(self.log)
        rows, report = tag_words.aggregate(done, words, INVENTORY, set(), passes=(16001,))
        self.assertEqual([row["word"] for row in rows], sorted(words))
        self.assertEqual(report["words_untagged_count"], 0)

    def run_pass(self, fake, plan, seed):
        done = tag_words.done_calls(self.log)
        kept = 0
        for index, chunk in plan:
            if (seed, index) in done:
                continue
            answer = fake.ask(tag_words.KIND, "s", "p", {}, seed, 0.7, 10,
                              {"pass": seed, "batch": index})
            tags, problem = tag_words.check(answer, chunk, INVENTORY)
            if problem:
                teacher_module.append_jsonl(self.log, {"pass": seed, "batch": index, "ok": False,
                                                       "why": problem})
                continue
            teacher_module.append_jsonl(self.log, {"pass": seed, "batch": index, "ok": True,
                                                   "tags": tags})
            kept += 1
            if not fake.answers:
                break
        return kept


class HeldOut(Scratch):
    def big_vocabulary(self):
        """Enough words to have something to stratify: four bands, three parts of speech."""
        rows, pos_of = [], {}
        counts = {"high": 200, "mid": 50, "low": 15, "rare": 4}
        for band, count in counts.items():
            for pos in ("noun", "verb", "adjective"):
                for n in range(10):
                    word = "%s%s%d" % (band[:2], pos[:2], n)
                    rows.append({"word": word, "count": count})
                    pos_of[word] = pos
        return rows, pos_of

    def test_the_bands_are_read_off_the_usage_count(self):
        self.assertEqual(held_out.band_of(500), "high")
        self.assertEqual(held_out.band_of(100), "high")
        self.assertEqual(held_out.band_of(99), "mid")
        self.assertEqual(held_out.band_of(30), "mid")
        self.assertEqual(held_out.band_of(10), "low")
        self.assertEqual(held_out.band_of(0), "rare")

    def test_the_pick_is_the_same_words_every_run(self):
        rows, pos_of = self.big_vocabulary()
        first = held_out.pick(rows, pos_of)
        second = held_out.pick(list(reversed(rows)), pos_of)
        self.assertEqual([row["word"] for row in first], [row["word"] for row in second])

    def test_a_different_seed_picks_different_words(self):
        rows, pos_of = self.big_vocabulary()
        a = {row["word"] for row in held_out.pick(rows, pos_of, seed=held_out.SEED)}
        b = {row["word"] for row in held_out.pick(rows, pos_of, seed=held_out.SEED + 1)}
        self.assertNotEqual(a, b)

    def test_the_pick_covers_every_band_and_every_part_of_speech(self):
        rows, pos_of = self.big_vocabulary()
        chosen = held_out.pick(rows, pos_of)
        self.assertEqual(len(chosen), held_out.TARGET)
        self.assertEqual({row["band"] for row in chosen}, {"high", "mid", "low", "rare"})
        self.assertEqual({row["pos"] for row in chosen}, {"noun", "verb", "adjective"})
        self.assertEqual(len({row["word"] for row in chosen}), len(chosen))

    def test_a_small_corner_of_the_vocabulary_still_gets_a_word(self):
        rows, pos_of = self.big_vocabulary()
        rows.append({"word": "solo", "count": 300})
        pos_of["solo"] = "other"
        chosen = held_out.pick(rows, pos_of)
        self.assertIn("solo", [row["word"] for row in chosen])

    def test_v1s_held_out_nouns_come_in_with_their_human_tags(self):
        v1 = json.loads((ROOT.parent / "smCLM_01" / "concepts.json").read_text())
        inventory = json.loads((ROOT / "ideas.json").read_text())
        rows = held_out.v1_held_out(set(v1["held_out"]), inventory)
        self.assertEqual(len(rows), len(v1["held_out"]))
        self.assertTrue(all(row["source"] == "v1" and row["v1_ideas"] for row in rows))
        by_word = {row["word"]: row for row in rows}
        self.assertEqual(by_word["car"]["v1_ideas"], v1["words"]["car"])

    def test_a_v1_noun_that_is_not_in_this_vocabulary_is_left_out(self):
        inventory = json.loads((ROOT / "ideas.json").read_text())
        self.assertEqual(held_out.v1_held_out({"car"}, inventory)[0]["word"], "car")
        self.assertEqual(held_out.v1_held_out(set(), inventory), [])

    def test_a_v1_tag_outside_the_inventory_is_left_out(self):
        rows = held_out.v1_held_out({"car"}, {"vehicle": "a thing you ride in"})
        self.assertEqual(rows[0]["v1_ideas"], ["vehicle"])

    def test_v1s_nouns_are_not_picked_twice(self):
        rows, pos_of = self.big_vocabulary()
        rows.append({"word": "car", "count": 200})
        pos_of["car"] = "noun"
        inventory = json.loads((ROOT / "ideas.json").read_text())
        combined = held_out.combine(held_out.pick(rows, pos_of), held_out.v1_held_out({"car"},
                                                                                      inventory))
        self.assertEqual([row["word"] for row in combined].count("car"), 1)
        self.assertEqual(combined[0]["source"], "v1")

    def test_the_skeleton_leaves_the_ideas_for_a_person(self):
        rows = [{"word": "wage", "band": "rare", "pos": "noun", "source": "stratified"}]
        skeleton = held_out.skeleton(rows, with_v1_tags=True)
        self.assertEqual(skeleton[0]["ideas"], [])
        self.assertEqual(skeleton[0]["source"], "human")

    def test_the_skeleton_only_prefills_v1s_tags_when_it_is_asked_to(self):
        rows = [{"word": "car", "band": None, "pos": "noun", "source": "v1",
                 "v1_ideas": ["vehicle"]}]
        self.assertEqual(held_out.skeleton(rows, with_v1_tags=False)[0]["ideas"], [])
        prefilled = held_out.skeleton(rows, with_v1_tags=True)[0]
        self.assertEqual(prefilled["ideas"], ["vehicle"])
        self.assertEqual(prefilled["source"], "v1", "a prefilled row must say where it came from")

    def test_the_skeleton_rows_are_the_shape_test_leaks_checks(self):
        rows = [{"word": "car", "band": None, "pos": "noun", "source": "v1",
                 "v1_ideas": ["vehicle"]}]
        row = held_out.skeleton(rows, with_v1_tags=True)[0]
        self.assertEqual(set(row), {"word", "pos", "ideas", "band", "source", "note"})


class NothingIsPretendHumanTruth(unittest.TestCase):
    """The rule the whole phase turns on, checked on the files as they are on disk."""

    def test_test_words_rows_name_their_reviewer_and_say_how_they_were_made(self):
        shipped = json.loads((ROOT / "test_words.json").read_text())
        if not shipped["words"]:
            return
        self.assertTrue(shipped["filled"]["reviewed_by"], "rows without a named person reviewing them")
        for row in shipped["words"]:
            self.assertIn("person-reviewed", row["source"], row["word"])

    def test_no_phase_1_script_writes_test_words_json(self):
        for name in ("teacher.py", "propose_ideas.py", "tag_words.py", "held_out.py"):
            source = (ROOT / name).read_text()
            writes = [line for line in source.splitlines()
                      if "test_words" in line and ("write_json" in line or "open(" in line)]
            self.assertEqual(writes, [], "%s looks like it writes test_words.json" % name)

    def test_no_phase_1_script_writes_ideas_json(self):
        for name in ("propose_ideas.py", "tag_words.py", "held_out.py"):
            source = (ROOT / name).read_text()
            writes = [line for line in source.splitlines()
                      if "ideas.json" in line and "write_json" in line]
            self.assertEqual(writes, [], "%s looks like it writes ideas.json" % name)

    def test_generated_proposals_are_derived_from_the_logged_teacher_answers(self):
        """A canary may create proposals, but it still must not pretend they are human truth."""
        for name in ("tags.jsonl", "held_out.json"):
            if (ROOT / "data" / name).exists():
                # Only once a person has approved the inventory; this raises on a seed-only one.
                teacher_module.load_inventory(ROOT / "ideas.json")

        proposals = ROOT / "data" / "idea_proposals.json"
        if not proposals.exists():
            return
        log = ROOT / "data" / "logs" / "idea_proposals.jsonl"
        self.assertTrue(log.exists(), "generated proposals need their source call log")
        words = [row["word"] for row in teacher_module.load_vocabulary()]
        expected = propose_ideas.tally(propose_ideas.done_batches(log), words)
        self.assertEqual(json.loads(proposals.read_text()), expected)
        self.assertIn("A person picks", expected["human_gate"])


if __name__ == "__main__":
    unittest.main()
