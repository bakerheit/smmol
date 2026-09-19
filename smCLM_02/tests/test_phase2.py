"""smCLM_02 phase 2 tools: the corpus compiler and the sentence generator.

No socket is opened. The teacher is a fake that answers from a script, and the base corpus is passed
in, so these run in well under a second.

What they're really guarding:

  * the compiler and the leak test mean the same thing by "the same text" and "a near copy";
  * the near-copy search is exact, not a sample: it agrees with brute force;
  * a sentence is kept only if it passes every structural check, and a call can't copy itself;
  * resuming rebuilds the same state from the log, and ignores another prompt's rows;
  * an outage stops the run without charging any word a call.

    python3 -m unittest discover -s tests
"""
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import threading
import unittest
import unittest.mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import compile_data as C  # noqa: E402
import generate_sentences as G  # noqa: E402
import test_leaks as L  # noqa: E402
from teacher import read_jsonl  # noqa: E402

VOCABULARY = [{"word": w, "forms": f} for w, f in (
    ("dentist", ["dentists"]), ("tooth", []), ("clinic", []), ("walk", ["walked", "walking"]),
    ("hill", ["hills"]), ("weekend", ["weekends"]))]
KNOWN = {row["word"] for row in VOCABULARY}


class SameRules(unittest.TestCase):
    """compile_data copies the leak test's helpers; this is what keeps the copies honest."""

    SAMPLES = ["User's dentist is Dr Lee at the Bridge Street practice", "  Hello,   WORLD!! 42 ",
               "don't-stop ‘quoted’ text", "", "a b", "one two three four five"]

    def test_flat_and_shingles_match_the_leak_test(self):
        for text in self.SAMPLES:
            self.assertEqual(C.flat(text), L.flat(text), text)
            self.assertEqual(C.shingles(text), L.shingles(text), text)
        self.assertEqual(C.SHINGLE_LIMIT, L.SHINGLE_LIMIT)

    def test_the_guard_agrees_with_the_leak_test_on_every_recall_text(self):
        guard = C.LeakGuard()
        recall = json.loads((ROOT / "test_recall.json").read_text())
        for text in L.recall_texts(recall):
            self.assertTrue(guard.leaks(text), text)
            self.assertTrue(guard.leaks("  " + text.upper() + "!"), text)

    def test_the_guard_lets_an_ordinary_sentence_through(self):
        self.assertFalse(C.LeakGuard().leaks("The kettle in the office kitchen has been broken since March."))


class NearCopyIndex(unittest.TestCase):
    def test_exact_and_near_copies_are_caught_and_new_text_is_not(self):
        copies = C.NearCopies()
        copies.add("She walked her dog along the river every single morning before work")
        self.assertTrue(copies.exact("she walked her dog along the river every single morning before work!"))
        self.assertTrue(copies.near("She walked her dog along the river every single morning before school"))
        self.assertFalse(copies.near("He drove to the office early on a rainy Tuesday morning"))

    def test_the_pigeonhole_search_agrees_with_brute_force(self):
        rng = random.Random(3)
        vocab = "a b c d e f g h i j k l".split()
        kept = [" ".join(rng.choice(vocab) for _ in range(rng.randint(3, 12))) for _ in range(300)]
        copies = C.NearCopies()
        for text in kept:
            copies.add(text)
        for _ in range(500):
            text = " ".join(rng.choice(vocab) for _ in range(rng.randint(3, 12)))
            have = C.shingles(text)
            brute = any(len(have & C.shingles(k)) / len(have) >= C.SHINGLE_LIMIT for k in kept)
            self.assertEqual(copies.near(text), brute, text)


class Splitting(unittest.TestCase):
    def test_markdown_and_bullets_go_and_sentences_split(self):
        message = "**Plan:** walk first.\n- Then rest for ten minutes. After that, eat.\n1. Sleep early"
        self.assertEqual(list(C.split_sentences(message)),
                         ["Plan: walk first.", "Then rest for ten minutes.", "After that, eat.", "Sleep early"])

    def test_a_decimal_or_lowercase_after_a_stop_is_not_a_split(self):
        self.assertEqual(list(C.split_sentences("It costs 3.50 today. e.g. this stays.")),
                         ["It costs 3.50 today. e.g. this stays."])


class Checks(unittest.TestCase):
    def problem(self, text, word="dentist"):
        return G.problem_with(text, word, KNOWN)

    def test_a_good_sentence_passes_in_any_folded_form(self):
        self.assertIsNone(self.problem("My sister finally found a dentist she actually trusts."))
        self.assertIsNone(self.problem("Both dentists at the clinic were away on the same week."))

    def test_each_structural_failure_says_why(self):
        self.assertEqual(self.problem("Too short dentist."), "length")
        self.assertEqual(self.problem(" ".join(["dentist"] + ["word"] * 30)), "length")
        self.assertEqual(self.problem("- My dentist is lovely and very patient with me."), "list mark or markup")
        self.assertEqual(self.problem("My **dentist** is lovely and very patient with me."), "list mark or markup")
        self.assertEqual(self.problem("I saw the dentist today. It went well enough, honestly."),
                         "more than one sentence")
        self.assertEqual(self.problem("My dentist is kind.\nShe is also patient with me."), "more than one line")
        self.assertEqual(self.problem("The clinic opens early on most weekday mornings."), "doesn't use the word")
        self.assertEqual(self.problem(None), "not text")


class Fake:
    """A teacher that answers from a script: a list for one word, or {word: [answers]} for several.

    Keyed by word because rounds interleave words, and with --workers the calls run on threads.
    """

    def __init__(self, answers, endpoint="http://fake:1"):
        self.answers = answers if isinstance(answers, dict) else {None: list(answers)}
        self.endpoint = endpoint
        self.asked = []
        self.lock = threading.Lock()

    def request_headers(self):
        return None

    def ask(self, kind, system, prompt, schema, seed, temperature, max_tokens, metadata=None):
        metadata = metadata or {}
        with self.lock:
            self.asked.append(dict(metadata, seed=seed, prompt=prompt))
            script = self.answers.get(metadata.get("word"), self.answers.get(None))
            answer = script.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class Generating(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="smclm02-phase2-"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.log = self.dir / "sentence_calls.jsonl"
        self.patches = [
            unittest.mock.patch.object(G, "load_vocabulary", lambda: VOCABULARY),
            unittest.mock.patch.object(G.C, "base_sentences", lambda guard=None: []),
            unittest.mock.patch.object(G, "TARGET", 3),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def run_with(self, fake, *argv):
        with unittest.mock.patch.object(G, "teacher_from", lambda args, log: fake):
            return G.main(["--log", str(self.log)] + list(argv))

    def sentences(self, *texts):
        return {"sentences": list(texts)}

    def test_a_call_keeps_the_good_sentences_and_cannot_copy_itself(self):
        fake = Fake([self.sentences(
            "My dentist moved her practice across town last spring.",
            "My dentist moved her practice across town last spring!",
            "I booked the dentist for a check-up next Thursday afternoon.",
            "Short dentist.",
            "Our dentist always asks about school while cleaning my teeth.")])
        self.assertEqual(self.run_with(fake, "--words", "dentist"), 0)
        result = [r for r in read_jsonl(self.log) if r.get("result")][0]
        self.assertEqual(len(result["kept"]), 3)
        self.assertEqual(result["rejected"], {"exact copy": 1, "length": 1})
        self.assertEqual(len(fake.asked), 1, "three usages reach the target of 3; no second call")

    def test_a_word_is_asked_again_with_a_new_seed_until_it_reaches_the_target(self):
        fake = Fake([self.sentences("I finally went to the dentist after putting it off for months."),
                     self.sentences("The dentist said my teeth looked better than last year.",
                                    "A new dentist opened next to the bakery on the corner.")])
        self.run_with(fake, "--words", "dentist")
        self.assertEqual([a["call"] for a in fake.asked], [0, 1])
        self.assertEqual([a["seed"] for a in fake.asked], [G.SEED, G.SEED + 1])

    def test_a_word_that_never_gets_there_stops_at_the_call_cap(self):
        fake = Fake([self.sentences("Nothing useful comes back from this call at all today.")] * G.MAX_CALLS_PER_WORD)
        self.run_with(fake, "--words", "dentist")
        self.assertEqual(len(fake.asked), G.MAX_CALLS_PER_WORD)

    def test_resuming_rebuilds_the_state_and_ignores_another_prompts_rows(self):
        with open(self.log, "w") as f:
            f.write(json.dumps({"result": True, "word": "dentist", "call": 0, "prompt_version": G.PROMPT_VERSION,
                                "kept": ["My dentist is on holiday until the end of the month.",
                                         "We waited an hour at the dentist this morning."],
                                "rejected": {"length": 2}}) + "\n")
            f.write(json.dumps({"result": True, "word": "dentist", "call": 1, "prompt_version": "v0",
                                "kept": ["An old prompt's dentist sentence that must not count."],
                                "rejected": {}}) + "\n")
        state = G.State(VOCABULARY, self.log, base=[])
        self.assertEqual(state.calls["dentist"], 1)
        self.assertEqual(state.uses["dentist"], 2)
        self.assertEqual(state.rejected["length"], 2)
        fake = Fake([self.sentences("The dentist gave my son a sticker for sitting still.")])
        self.run_with(fake, "--words", "dentist")
        self.assertEqual([a["call"] for a in fake.asked], [1])

    def test_an_outage_stops_the_run_and_charges_no_word(self):
        down = RuntimeError("teacher failed after 3 attempts: connection refused")
        fake = Fake({"dentist": [down], "tooth": [down]})
        with unittest.mock.patch.object(G, "reachable", lambda teacher: False):
            self.assertEqual(self.run_with(fake, "--words", "dentist,tooth"), 3)
        self.assertEqual([r for r in read_jsonl(self.log) if r.get("result")], [])
        self.assertEqual(len(fake.asked), 2, "one round, then the stop")

    def test_a_model_failure_on_one_word_is_charged_and_the_run_goes_on(self):
        fake = Fake({
            "dentist": [RuntimeError("teacher failed after 3 attempts: bad JSON")]
                       + [self.sentences("Nothing useful comes back from this call at all today.")]
                       * (G.MAX_CALLS_PER_WORD - 1),
            "tooth": [self.sentences("My tooth has ached since I bit into that apple.",
                                     "The tooth fairy left a coin under her pillow last night.",
                                     "He chipped a tooth playing football in the park.")]})
        with unittest.mock.patch.object(G, "reachable", lambda teacher: True):
            self.assertEqual(self.run_with(fake, "--words", "dentist,tooth"), 0)
        results = [r for r in read_jsonl(self.log) if r.get("result")]
        self.assertEqual(results[0], dict(results[0], word="dentist", kept=[],
                                          rejected={"answer has no sentences array": 1}))
        self.assertEqual(len([r for r in results if r["word"] == "tooth"]), 1)
        self.assertEqual(len([r for r in results if r["word"] == "dentist"]), G.MAX_CALLS_PER_WORD)

    def test_a_rate_limit_is_not_charged_and_the_word_is_asked_again(self):
        fake = Fake({"dentist": [RuntimeError("teacher failed after 3 attempts: HTTP Error 429: Too Many"),
                                 self.sentences("My dentist moved her practice across town last spring.",
                                                "I booked the dentist for a check-up next Thursday.",
                                                "Our dentist always asks about school while cleaning.")]})
        with unittest.mock.patch.object(G, "reachable", lambda teacher: True):
            self.assertEqual(self.run_with(fake, "--words", "dentist"), 0)
        results = [r for r in read_jsonl(self.log) if r.get("result")]
        self.assertEqual([r["call"] for r in results], [0], "the 429 used no call")

    def test_the_spend_cap_stops_the_run_and_charges_no_word(self):
        from teacher import BudgetExceeded
        fake = Fake({"dentist": [BudgetExceeded("cap")], "tooth": [BudgetExceeded("cap")]})
        self.assertEqual(self.run_with(fake, "--words", "dentist,tooth"), 4)
        self.assertEqual([r for r in read_jsonl(self.log) if r.get("result")], [])

    def test_parallel_workers_give_the_same_corpus_as_one_worker(self):
        fillers = ("orange quietly river bought seven lamps across morning garden paper slowly "
                   "window yellow carried music winter bridge happily market stone").split()

        def script():
            out = {}
            for w in ("dentist", "tooth", "clinic", "walk", "hill", "weekend"):
                rng = random.Random(w)
                out[w] = [self.sentences(*[" ".join([w] + rng.sample(fillers, 8)) + "."
                                           for _ in range(3)]) for _ in range(G.MAX_CALLS_PER_WORD)]
            return out
        words = "dentist,tooth,clinic,walk,hill,weekend"
        self.run_with(Fake(script()), "--words", words)
        one = [r for r in read_jsonl(self.log) if r.get("result")]
        self.log.unlink()
        self.run_with(Fake(script()), "--words", words, "--workers", "4")
        many = [r for r in read_jsonl(self.log) if r.get("result")]
        self.assertEqual(one, many)

    def test_the_prompt_names_the_word_and_its_forms(self):
        prompt = G.prompt_for("walk", ["walked", "walking"])
        self.assertIn('"walk"', prompt)
        self.assertIn("walk, walked, walking", prompt)
        self.assertIn("%d to %d words" % (G.MIN_WORDS, G.MAX_WORDS), prompt)


if __name__ == "__main__":
    unittest.main()
