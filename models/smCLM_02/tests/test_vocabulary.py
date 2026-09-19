"""smCLM_02 phase 1: the vocabulary rules, and the vocabulary that came out of them.

Two halves. The first builds a vocabulary from three tiny fixtures written here, so every rule -
the source union, the 5-use threshold, the function words, the proper-name exclusion, the folding -
can be checked on inputs small enough to read. The second checks `data/vocabulary.jsonl` itself,
and skips while it doesn't exist.

The folding rules get their own tests because three other files will import them (`world.py`,
`encode.py`, `generate_sentences.py`), and a quiet change to `fold()` would move the vocabulary out
from under a trained checkpoint.

    python3 -m unittest discover -s tests
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import vocabulary as v  # noqa: E402

VOCABULARY = os.path.join(ROOT, "data", "vocabulary.jsonl")


class Tokens(unittest.TestCase):
    """tokens(): lowercase, letters only, contractions cut back to their head."""

    def test_a_sentence_becomes_lowercase_words(self):
        self.assertEqual(v.tokens("The Dentist is Dr. Lee!"), ["the", "dentist", "is", "dr", "lee"])

    def test_both_kinds_of_apostrophe_are_the_same_apostrophe(self):
        self.assertEqual(v.tokens("the user’s teeth"), v.tokens("the user's teeth"))

    def test_a_contraction_keeps_its_head_and_drops_the_clitic(self):
        self.assertEqual(v.tokens("Don't, I'll, she's, we've, wasn't"),
                         ["do", "i", "she", "we", "was"])

    def test_a_possessive_is_the_word_itself(self):
        self.assertEqual(v.tokens("my mother's dentist"), ["my", "mother", "dentist"])

    def test_an_apostrophe_that_is_not_a_clitic_stays_one_word(self):
        self.assertEqual(v.tokens("o'clock"), ["oclock"])

    def test_digits_and_punctuation_are_not_words(self):
        self.assertEqual(v.tokens("2 x 3 = 6 (six?)"), ["x", "six"])

    def test_a_hyphenated_word_is_two_words(self):
        self.assertEqual(v.tokens("retro-futuristic"), ["retro", "futuristic"])

    def test_content_words_are_long_enough_and_not_grammar(self):
        self.assertTrue(v.is_content("dentist"))
        self.assertFalse(v.is_content("the"), "a function word is not content")
        self.assertFalse(v.is_content("it"), "two letters is too short")
        self.assertFalse(v.is_content("would"), "the added function words count too")


class Folding(unittest.TestCase):
    """bases() guesses, fold() only accepts a guess the vocabulary already knows."""

    KNOWN = frozenset("walk like run stop try story box dentist find state visit limit "
                      "plan share come hope new view fire".split())

    def fold(self, word):
        return v.fold(word, self.KNOWN)

    def test_the_plural_endings_fold_when_the_base_is_known(self):
        self.assertEqual(self.fold("dentists"), "dentist")
        self.assertEqual(self.fold("boxes"), "box")
        self.assertEqual(self.fold("stories"), "story")

    def test_the_ed_and_ing_endings_fold_when_the_base_is_known(self):
        self.assertEqual(self.fold("walked"), "walk")
        self.assertEqual(self.fold("walking"), "walk")
        self.assertEqual(self.fold("liked"), "like")
        self.assertEqual(self.fold("stopped"), "stop")
        self.assertEqual(self.fold("running"), "run")
        self.assertEqual(self.fold("tried"), "try")
        self.assertEqual(self.fold("planning"), "plan")

    def test_a_dropped_e_beats_a_bare_one_syllable_stem(self):
        # "stated" must not land on "stat", "fired" must not land on "fir".
        self.assertEqual(self.fold("stated"), "state")
        self.assertEqual(self.fold("coming"), "come")
        self.assertEqual(self.fold("hoping"), "hope")
        self.assertEqual(self.fold("shared"), "share")
        self.assertNotIn("stat", v.bases("stated"))

    def test_a_longer_stem_keeps_its_bare_form(self):
        self.assertEqual(self.fold("visited"), "visit")
        self.assertEqual(self.fold("limited"), "limit")

    def test_nothing_folds_when_the_base_is_not_in_the_vocabulary(self):
        self.assertEqual(self.fold("dancing"), "dancing")
        self.assertEqual(self.fold("cats"), "cats")

    def test_a_word_already_in_the_vocabulary_is_left_alone(self):
        for word in sorted(self.KNOWN):
            self.assertEqual(self.fold(word), word)

    def test_folding_follows_the_chain_to_one_word(self):
        self.assertEqual(v.fold("findings", frozenset(["find", "finding"])), "find")

    def test_folding_twice_says_what_folding_once_said(self):
        for word in "dentists boxes stories walking stated findings news cats".split():
            self.assertEqual(self.fold(self.fold(word)), self.fold(word))

    def test_the_exceptions_never_fold(self):
        self.assertEqual(self.fold("news"), "news", "news is not the plural of new")
        self.assertEqual(v.bases("news"), [])

    def test_short_words_and_double_s_endings_are_left_alone(self):
        self.assertEqual(v.bases("its"), [])
        self.assertEqual(v.bases("class"), [])

    def test_a_candidate_is_never_shorter_than_the_shortest_word_allowed(self):
        for word in "seeds going asked bees eyes ties".split():
            for candidate in v.bases(word):
                self.assertGreaterEqual(len(candidate), v.MIN_LETTERS, word)

    def test_content_tokens_are_the_folded_content_words_of_a_text(self):
        self.assertEqual(v.content_tokens("She walked the dogs to the dentists", self.KNOWN),
                         ["walk", "dogs", "dentist"])


class Build(unittest.TestCase):
    """The three sources, unioned, on fixtures small enough to check by hand."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="smclm02-vocabulary-")
        # "dentist" is used 5 times (twice as "dentists"), so folding is what gets it over the
        # line. "Lee" is always capitalised mid-sentence: a name. "Tuesday" is too, but it's a
        # calendar word and stays. "biscuit" is used once: too rare to earn a place on its own.
        messages = [
            "My dentist is kind.",
            "I saw the dentist again.",
            "Both dentists here are good.",
            "The dentists were busy on Tuesday.",
            "Lee is my dentist and Lee works on Tuesday.",
            "Lee told me to floss and I floss twice on Tuesday.",
            "On Tuesday I floss. Tuesday again. Tuesday is the day.",
            "I floss and floss and floss and floss.",
            "She had a biscuit.",
            "The Alley Cat is a shop, and the alley behind it is dark.",
            "An alley near the alley shop. The alley is dark, the alley is long.",
        ]
        cls.conversations = os.path.join(cls.dir, "conversations.jsonl")
        with open(cls.conversations, "w") as f:
            for index, text in enumerate(messages):
                f.write(json.dumps({"id": index, "messages": [{"role": "user", "content": text}]}) + "\n")
        cls.curriculum = os.path.join(cls.dir, "curriculum.jsonl")
        with open(cls.curriculum, "w") as f:
            for word in ["muffin", "the", "boxes", "box", "boot"]:
                f.write(json.dumps({"level": "preschool", "word": word, "sentence": "A %s." % word}) + "\n")
        cls.concepts = os.path.join(cls.dir, "concepts.json")
        with open(cls.concepts, "w") as f:
            json.dump({"concepts": {}, "words": {"helmet": ["thing"], "boots": ["thing"]},
                       "held_out": [], "templates": []}, f)
        cls.rows, cls.summary = v.build(cls.conversations, cls.curriculum, cls.concepts)
        cls.by_word = {row["word"]: row for row in cls.rows}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_a_word_used_enough_times_gets_in_from_the_conversations(self):
        self.assertIn("dentist", self.by_word)
        self.assertIn("conversations", self.by_word["dentist"]["sources"])

    def test_folding_is_what_gets_a_word_over_the_threshold(self):
        # three "dentist" and two "dentists" is five uses of one word, not two words under five.
        self.assertEqual(self.by_word["dentist"]["count"], 5)
        self.assertEqual(self.by_word["dentist"]["forms"], ["dentists"])
        self.assertNotIn("dentists", self.by_word)

    def test_a_rare_word_does_not_get_in_on_its_own(self):
        self.assertNotIn("biscuit", self.by_word)

    def test_a_word_capitalised_in_most_of_its_uses_is_a_name_and_is_left_out(self):
        self.assertNotIn("lee", self.by_word)

    def test_a_title_case_run_is_not_evidence_that_a_word_is_a_name(self):
        # "The Alley Cat" shouldn't cost the vocabulary the word "alley".
        self.assertIn("alley", self.by_word)

    def test_a_weekday_is_kept_even_though_it_is_always_capitalised(self):
        self.assertIn("tuesday", self.by_word)

    def test_function_words_never_get_in_from_any_source(self):
        for word in self.by_word:
            self.assertNotIn(word, v.FUNCTION_WORDS)
        self.assertNotIn("the", self.by_word, "the curriculum's function words are dropped too")

    def test_the_listed_sources_get_in_however_rare_they_are(self):
        self.assertEqual(self.by_word["muffin"]["sources"], ["curriculum"])
        self.assertEqual(self.by_word["muffin"]["count"], 0)
        self.assertEqual(self.by_word["helmet"]["sources"], ["concepts"])

    def test_a_listed_word_folds_into_its_base_and_takes_its_source_with_it(self):
        self.assertNotIn("boots", self.by_word, "boots belongs to the boot row")
        self.assertIn("concepts", self.by_word["boot"]["sources"])
        self.assertEqual(self.by_word["boot"]["forms"], ["boots"])
        self.assertEqual(self.by_word["box"]["sources"], ["curriculum"])

    def test_a_word_from_two_sources_names_both_in_the_fixed_order(self):
        rows = [row for row in self.rows if len(row["sources"]) > 1]
        for row in rows:
            self.assertEqual(row["sources"], [s for s in v.SOURCES if s in row["sources"]])

    def test_every_row_has_the_fields_the_plan_asked_for(self):
        for row in self.rows:
            self.assertEqual(sorted(row), ["count", "forms", "pos", "sources", "word"])
            self.assertTrue(row["sources"], "%s claims no source" % row["word"])
            self.assertGreaterEqual(row["count"], 0)

    def test_part_of_speech_is_an_honest_placeholder_everywhere(self):
        # The teacher fills pos in the tagging pass. Nothing here guesses it, not even for v1's
        # nouns, which really are all nouns.
        self.assertEqual({row["pos"] for row in self.rows}, {"other"})

    def test_the_rows_are_sorted_and_each_word_appears_once(self):
        words = [row["word"] for row in self.rows]
        self.assertEqual(words, sorted(words))
        self.assertEqual(len(words), len(set(words)))

    def test_no_word_in_the_vocabulary_folds_to_another_word_in_it(self):
        # The guarantee fold() relies on: a form and its base are never both rows.
        known = set(self.by_word)
        for word in known:
            self.assertEqual(v.fold(word, known), word)

    def test_the_build_is_deterministic(self):
        again, summary = v.build(self.conversations, self.curriculum, self.concepts)
        self.assertEqual(again, self.rows)
        self.assertEqual(summary["words"], self.summary["words"])

    def test_the_threshold_is_a_knob_and_moving_it_moves_the_vocabulary(self):
        loose, _ = v.build(self.conversations, self.curriculum, self.concepts, min_count=1)
        self.assertIn("biscuit", {row["word"] for row in loose})
        self.assertGreater(len(loose), len(self.rows))

    def test_the_summary_adds_up(self):
        self.assertEqual(self.summary["words"], len(self.rows))
        for name in v.SOURCES:
            self.assertEqual(self.summary["by_source"][name],
                             sum(1 for row in self.rows if name in row["sources"]))

    def test_writing_and_loading_gives_the_same_rows_back(self):
        path = os.path.join(self.dir, "out", "vocabulary.jsonl")
        sha = v.write(self.rows, path)
        self.assertEqual(len(sha), 64)
        self.assertEqual(list(v.load(path).values()), self.rows)
        self.assertEqual(v.write(self.rows, path), sha, "the same rows write the same bytes")


class TheVocabularyOnDisk(unittest.TestCase):
    """data/vocabulary.jsonl, once it's been built. Skips while it hasn't."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(VOCABULARY):
            raise unittest.SkipTest("run python3 vocabulary.py first")
        cls.rows = list(v.load(VOCABULARY).values())

    def test_it_is_about_the_size_the_plan_asked_for(self):
        self.assertTrue(2000 <= len(self.rows) <= 3200,
                        "%d words; the plan wants about 2,500" % len(self.rows))

    def test_every_word_obeys_the_rules(self):
        for row in self.rows:
            word = row["word"]
            self.assertRegex(word, r"^[a-z]{3,}$")
            self.assertNotIn(word, v.FUNCTION_WORDS)
            self.assertTrue(row["sources"])

    def test_every_source_is_really_in_there(self):
        for name in v.SOURCES:
            self.assertGreater(sum(1 for row in self.rows if name in row["sources"]), 100, name)

    def test_v1s_nouns_are_all_reachable(self):
        known = {row["word"] for row in self.rows}
        missing = [word for word in v.concept_nouns() if v.is_content(word) and v.fold(word, known) not in known]
        self.assertEqual(missing, [], "v1 nouns lost on the way in: %s" % missing[:10])

    def test_no_word_folds_to_another_word_in_the_file(self):
        known = {row["word"] for row in self.rows}
        moved = [word for word in known if v.fold(word, known) != word]
        self.assertEqual(moved, [], "these rows should have folded: %s" % moved[:10])

    def test_the_file_on_disk_is_what_the_build_produces_now(self):
        rows, _ = v.build()
        self.assertEqual(rows, self.rows, "data/vocabulary.jsonl is stale; rerun python3 vocabulary.py")


if __name__ == "__main__":
    unittest.main()
