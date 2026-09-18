"""The fixed school curriculum and structural checks for its training items."""

from collections import Counter
import re


LEVELS = [
    {"id": "preschool", "label": "Preschool", "count": 100, "min_words": 3, "max_words": 9},
    {"id": "kindergarten", "label": "Kindergarten", "count": 200, "min_words": 4, "max_words": 11},
    {"id": "grade_01", "label": "1st grade", "count": 300, "min_words": 5, "max_words": 13},
    {"id": "grade_02", "label": "2nd grade", "count": 400, "min_words": 6, "max_words": 15},
    {"id": "grade_03", "label": "3rd grade", "count": 500, "min_words": 7, "max_words": 17},
    {"id": "grade_04", "label": "4th grade", "count": 600, "min_words": 8, "max_words": 19},
    {"id": "grade_05", "label": "5th grade", "count": 700, "min_words": 9, "max_words": 21},
    {"id": "grade_06", "label": "6th grade", "count": 800, "min_words": 10, "max_words": 23},
    {"id": "grade_07", "label": "7th grade", "count": 900, "min_words": 11, "max_words": 25},
    {"id": "grade_08", "label": "8th grade", "count": 1000, "min_words": 12, "max_words": 27},
    {"id": "grade_09", "label": "9th grade", "count": 1100, "min_words": 13, "max_words": 29},
    {"id": "grade_10", "label": "10th grade", "count": 1200, "min_words": 14, "max_words": 31},
    {"id": "grade_11", "label": "11th grade", "count": 1300, "min_words": 15, "max_words": 33},
    {"id": "grade_12", "label": "12th grade", "count": 1400, "min_words": 16, "max_words": 35},
]
BY_ID = {level["id"]: level for level in LEVELS}
WORD = re.compile(r"^[a-z]+(?:-[a-z]+)?$")
TOKENS = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
FUNCTION_WORDS = frozenset(
    "a an the and or but as at by for from of to with am is are was were be been being do does did "
    "have has had can could will would shall should may might must i me my mine you your yours he him "
    "his she her hers it its we us our ours they them their theirs this that these those who whom whose "
    "which what where when why how".split()
)


def sentence_words(sentence):
    return TOKENS.findall(sentence)


def item_problems(item):
    problems = []
    level = BY_ID.get(item.get("level"))
    word = str(item.get("word") or "").strip().lower()
    sentence = str(item.get("sentence") or "").strip()
    if level is None:
        return ["unknown level"]
    if not WORD.fullmatch(word):
        problems.append("target word must be one lowercase word")
    if level["id"] != "preschool" and word in FUNCTION_WORDS:
        problems.append("target word is grammar glue, not curriculum vocabulary")
    tokens = [token.lower() for token in sentence_words(sentence)]
    if word and word not in tokens:
        problems.append("sentence does not contain the target word")
    if not sentence.endswith((".", "?", "!")):
        problems.append("sentence needs ending punctuation")
    if not level["min_words"] <= len(tokens) <= level["max_words"]:
        problems.append(
            "sentence has %d words; %s needs %d-%d"
            % (len(tokens), level["label"], level["min_words"], level["max_words"])
        )
    return problems


def corpus_problems(items, complete=False):
    problems = []
    words = Counter(str(item.get("word") or "").strip().lower() for item in items)
    sentences = Counter(" ".join(str(item.get("sentence") or "").lower().split()) for item in items)
    for index, item in enumerate(items, 1):
        problems.extend("item %d: %s" % (index, problem) for problem in item_problems(item))
    problems.extend("duplicate target word: %s" % word for word, count in words.items() if word and count > 1)
    problems.extend("duplicate sentence: %s" % sentence for sentence, count in sentences.items() if sentence and count > 1)
    if complete:
        counts = Counter(item.get("level") for item in items)
        for level in LEVELS:
            if counts[level["id"]] != level["count"]:
                problems.append(
                    "%s has %d items; needs %d" % (level["label"], counts[level["id"]], level["count"])
                )
    return problems
