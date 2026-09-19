"""smCLM_02: the hand-written tests stay out of the training data, and stay well formed.

Phase 0 of docs/engineering/plans/smCLM_02.md. Two jobs:

  * the fixtures themselves are sound: `test_recall.json` has its three groups, its pools are the
    size the harness recalls, and group B really does share no word with its answer;
  * nothing in them leaks into training. No `query` or `pool` text may turn up in
    `data/sentences.jsonl`, and no held-out word from `test_words.json` may keep a row in
    `data/tags.jsonl`.

The data files don't exist until phases 1 and 2, so the leak checks skip while they're missing and
bite the moment they appear. The fixture checks always run.

    python3 -m unittest discover -s tests
"""
import json
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# The harness's tokeniser, copied from paratroop_harness_02/harness.py rather than imported, so
# these tests don't need the harness on the path. "Shares a content word" means what BM25 means.
WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and are as at be by can do does for from has have how i in is it its me my of on or our "
                 "so that the this to was we what when where which who whos why will with you your".split())

GROUPS = ("A", "B", "C")
POOL_MIN, POOL_MAX = 8, 12  # the harness's recall_top is 8
SHINGLE_LIMIT = 0.8  # phase 2's own near-duplicate rule


def words(text):
    found = []
    for w in WORD.findall(str(text).lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w not in STOP:
            found.append(w)
    return found


def flat(text):
    """A text reduced to the thing we compare: lowercase words, one space between."""
    return " ".join(WORD.findall(str(text).lower()))


def shingles(text, n=3):
    parts = flat(text).split()
    return {tuple(parts[i:i + n]) for i in range(max(1, len(parts) - n + 1))}


def load(name):
    with open(os.path.join(ROOT, name)) as f:
        return json.load(f)


def rows(path):
    """A JSONL file as a list of dicts. Missing file means an empty list, not a crash."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError as e:
                    raise AssertionError("%s line %d isn't JSON: %s" % (path, n, e))
    return out


def recall_texts(recall):
    """Every piece of text a generator must never produce: the queries and the pool memories."""
    seen = []
    for case in recall["cases"]:
        seen.append(case["query"])
        seen.extend(p["text"] for p in case["pool"])
    return seen


class RecallFixture(unittest.TestCase):
    """test_recall.json is what phase 0 promised: about 60 cases over groups A, B and C."""

    @classmethod
    def setUpClass(cls):
        cls.recall = load("test_recall.json")
        cls.cases = cls.recall["cases"]

    def test_there_are_about_sixty_cases_over_the_three_groups(self):
        self.assertGreaterEqual(len(self.cases), 55)
        counts = {g: sum(1 for c in self.cases if c["group"] == g) for g in GROUPS}
        self.assertEqual(sum(counts.values()), len(self.cases), "a case has a group outside A, B, C")
        for group, n in counts.items():
            self.assertGreaterEqual(n, 10, "group %s only has %d cases" % (group, n))

    def test_every_case_has_a_pool_the_harness_could_recall_and_an_answer_in_it(self):
        seen = set()
        for case in self.cases:
            where = case["id"]
            self.assertNotIn(where, seen, "two cases share an id")
            seen.add(where)
            self.assertTrue(case["query"].strip(), where)
            pool = case["pool"]
            self.assertTrue(POOL_MIN <= len(pool) <= POOL_MAX,
                            "%s: pool of %d, wants %d to %d" % (where, len(pool), POOL_MIN, POOL_MAX))
            ids = [p["id"] for p in pool]
            self.assertEqual(len(ids), len(set(ids)), "%s: a memory is in the pool twice" % where)
            for p in pool:
                self.assertTrue(p["text"].strip(), where)
            self.assertIn(case["answer"], ids, "%s: the answer isn't in the pool" % where)

    def test_the_answer_is_not_always_in_the_same_place(self):
        at = [[p["id"] for p in c["pool"]].index(c["answer"]) for c in self.cases]
        self.assertGreater(len(set(at)), 3, "the answer sits in too few positions; a ranker could cheat")
        for pos in set(at):
            self.assertLess(at.count(pos) / len(at), 0.5, "the answer sits at %d too often" % pos)

    def test_group_a_queries_share_a_word_with_their_answer(self):
        for case in self.cases:
            if case["group"] != "A":
                continue
            self.assertTrue(self.shared(case), "%s: group A but nothing is shared with the answer" % case["id"])

    def test_group_b_queries_share_no_word_at_all_with_their_answer(self):
        for case in self.cases:
            if case["group"] != "B":
                continue
            shared = self.shared(case)
            self.assertEqual(shared, set(),
                             "%s: group B but the query shares %s with the answer"
                             % (case["id"], ", ".join(sorted(shared))))

    def test_group_c_pools_hold_a_memory_that_shares_words_but_not_the_meaning(self):
        for case in self.cases:
            if case["group"] != "C":
                continue
            query = set(words(case["query"]))
            decoys = [p["id"] for p in case["pool"]
                      if p["id"] != case["answer"] and query & set(words(p["text"]))]
            self.assertTrue(decoys, "%s: group C but nothing else in the pool shares a word" % case["id"])

    def test_no_two_cases_are_the_same_question(self):
        asked = [flat(c["query"]) for c in self.cases]
        self.assertEqual(len(asked), len(set(asked)))

    def test_a_memory_id_always_means_the_same_memory(self):
        texts = {}
        for case in self.cases:
            for p in case["pool"]:
                self.assertEqual(texts.setdefault(p["id"], p["text"]), p["text"],
                                 "%s says something different in %s" % (p["id"], case["id"]))

    def shared(self, case):
        answer = next(p for p in case["pool"] if p["id"] == case["answer"])
        return set(words(case["query"])) & set(words(answer["text"]))


class WordsFixture(unittest.TestCase):
    """test_words.json: empty after phase 0, human-written and inventory-clean after phase 1."""

    @classmethod
    def setUpClass(cls):
        cls.words = load("test_words.json")["words"]

    def test_the_file_is_a_list_of_words_even_while_it_is_empty(self):
        self.assertIsInstance(self.words, list)

    def test_every_row_names_a_word_and_at_least_one_idea(self):
        seen = set()
        for row in self.words:
            word = row["word"]
            self.assertNotIn(word, seen, "%s is in test_words.json twice" % word)
            seen.add(word)
            self.assertRegex(word, r"^[a-z]{3,}$")
            self.assertTrue(row["ideas"], "%s has no ideas" % word)
            self.assertEqual(len(row["ideas"]), len(set(row["ideas"])), word)

    def test_every_idea_is_one_of_the_approved_ones(self):
        path = os.path.join(ROOT, "ideas.json")
        if not os.path.exists(path):
            self.skipTest("ideas.json arrives in phase 1")
        if not self.words:
            self.skipTest("test_words.json is filled in during phase 1")
        with open(path) as f:
            inventory = set(json.load(f))
        for row in self.words:
            unknown = sorted(set(row["ideas"]) - inventory)
            self.assertEqual(unknown, [], "%s uses ideas outside ideas.json: %s" % (row["word"], unknown))


class Leaks(unittest.TestCase):
    """Nothing a person wrote as a test may be trained on."""

    @classmethod
    def setUpClass(cls):
        cls.recall = load("test_recall.json")
        cls.held_out = {row["word"] for row in load("test_words.json")["words"]}
        cls.sentences = rows(os.path.join(DATA, "sentences.jsonl"))
        cls.tags = rows(os.path.join(DATA, "tags.jsonl"))

    def test_no_test_text_is_in_the_training_sentences(self):
        if not self.sentences:
            self.skipTest("data/sentences.jsonl arrives in phase 2")
        trained = {}
        for row in self.sentences:
            trained.setdefault(flat(row["text"]), row["text"])
        for text in recall_texts(self.recall):
            self.assertNotIn(flat(text), trained, "test_recall.json text is in the training data: %r" % text)

    def test_no_training_sentence_is_a_near_copy_of_a_test_text(self):
        if not self.sentences:
            self.skipTest("data/sentences.jsonl arrives in phase 2")
        wanted = [(t, shingles(t)) for t in recall_texts(self.recall)]
        for row in self.sentences:
            have = shingles(row["text"])
            for text, want in wanted:
                if not want:
                    continue
                share = len(have & want) / len(want)
                self.assertLess(share, SHINGLE_LIMIT,
                                "a training sentence is a near copy of a test text:\n  %r\n  %r"
                                % (row["text"], text))

    def test_held_out_words_keep_no_teacher_tags(self):
        if not self.tags:
            self.skipTest("data/tags.jsonl arrives in phase 1")
        if not self.held_out:
            self.skipTest("test_words.json is filled in during phase 1")
        tagged = {row["word"] for row in self.tags}
        leaked = sorted(self.held_out & tagged)
        self.assertEqual(leaked, [], "held-out words still have tags in data/tags.jsonl: %s" % leaked)


if __name__ == "__main__":
    unittest.main()
