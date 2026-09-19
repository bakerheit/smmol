"""smCLM_02 phase 1: the vocabulary, and the folding rules everything else reuses.

Builds `data/vocabulary.jsonl` from three sources, unioned (plan section "Phase 1"):

  1. `conversations` - content words used MIN_COUNT (5) or more times in
     `smCONVERSATION_001/data/accepted/conversations.jsonl`;
  2. `curriculum`    - the accepted words in `smLANGUAGE_en_SCH_001/data/curriculum.jsonl`;
  3. `concepts`      - the hand-tagged nouns in `smCLM_01/concepts.json`.

Rules, in the order they're applied: lowercase; letters only; 3 or more letters; not a function
word; a word that's capitalised in most of its mid-sentence uses in the corpus is a name and is
left out; `-s`, `-es`, `-ies`, `-ed`, `-ing` fold to the base form *only when the base is itself in
the vocabulary*, otherwise the surface form is kept.

There's no lemmatiser and that's deliberate. `tokens()`, `bases()` and `fold()` are the whole of the
folding rules, they live here, and `world.py`, `encode.py` and the sentence generator import them
rather than writing their own. `FOLD_VERSION` goes in the checkpoint so a rule change is visible.

Honest gaps, on purpose:

  * `pos` is `"other"` on every row. Part of speech is the teacher's job in the tagging pass
    (`tag_words.py`), not this script's; nothing here guesses it, not even for the v1 nouns, which
    really are all nouns. A `pos` that's right for some rows and a placeholder for others is worse
    than one that's honestly a placeholder everywhere.
  * `count` is occurrences in the conversation corpus only, after folding. A curriculum or v1 word
    that never turns up in conversation has `count` 0. That's a membership statement, not a claim
    that the word is unused - phase 2's job is to get every word to 30 usages.
  * `data/manifest.json` is not written here. It records the SHA-256 of *every* data file and
    belongs to `compile_data.py`; this script prints its own file's SHA-256 so the manifest can be
    filled in without re-reading the file.

    python3 vocabulary.py                 build data/vocabulary.jsonl and print the summary
    python3 vocabulary.py --dry-run       the summary only, write nothing
    python3 vocabulary.py --json          the summary as JSON
    python3 vocabulary.py --fold walking  what the folding rules do to one word
    python3 vocabulary.py --show dentist  one word's row, and the surface forms folded into it
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "vocabulary.jsonl")

CONVERSATIONS = os.path.join(WORKSPACE, "smCONVERSATION_001", "data", "accepted", "conversations.jsonl")
CURRICULUM = os.path.join(os.path.dirname(WORKSPACE), "archive", "models", "smLANGUAGE_en_SCH_001", "data", "curriculum.jsonl")
CONCEPTS = os.path.join(WORKSPACE, "smCLM_01", "concepts.json")

SOURCES = ("conversations", "curriculum", "concepts")  # fixed order, so rows are comparable
MIN_COUNT = 5  # "used 5 or more times", source 1
MIN_LETTERS = 3
NAME_SHARE = 0.5  # capitalised in at least this share of the uses that count -> a name
NAME_USES = 3  # ...and only when there are this many uses that count to judge on

# Bump this whenever tokens()/bases()/fold() change. It goes in the checkpoint next to the
# vocabulary, so a model trained under one set of rules can't be read under another by accident.
FOLD_VERSION = 1
FOLD_STEPS = 4  # "findings" -> "finding" -> "find" is two; the cap is a belt and braces

# Grammar glue. The curriculum project's list (smLANGUAGE_en_SCH_001/curriculum.py FUNCTION_WORDS)
# verbatim, then the rest of the closed class: a conversation corpus is full of "would", "there",
# "just", and none of those carry an idea worth learning.
FUNCTION_WORDS = frozenset(
    # smLANGUAGE_en_SCH_001/curriculum.py, unchanged
    "a an the and or but as at by for from of to with am is are was were be been being do does did "
    "have has had can could will would shall should may might must i me my mine you your yours he him "
    "his she her hers it its we us our ours they them their theirs this that these those who whom whose "
    "which what where when why how "
    # added here
    "about above after again against all almost also although always among any anyone anything because "
    "before below between both cannot could couldnt didnt doesnt dont down during each either else enough "
    "even ever every everyone everything few further had hadnt hasnt havent here hers herself himself "
    "however if im into isnt itself just least less let like many maybe more most much must mustnt myself "
    "near neither never new next no nor not nothing now off once one only onto other others ought our "
    "ourselves out over own perhaps please quite rather really same shant shes should shouldnt since some "
    "someone something sometimes soon still such than thats theirs then there therefore theres thing things "
    "though through thus till together too toward towards under until up upon very via wasnt way well "
    "were werent weve while whether within without wont wouldnt yet yours yourself yourselves "
    "ive youre weren theyre isn aren don doesn didn won wouldn couldn shouldn "
    "okay ok yeah yes ah oh hmm hey".split()
)

# Surface forms the folding rules must never touch, because the base they'd land on is a different
# word. Short on purpose: a candidate base is only accepted when it's already in the vocabulary, so
# most bad folds ("class" -> "clas") die on their own. These are the ones that wouldn't.
NEVER_FOLD = frozenset("news ours plus thus yes series goods means".split())

# Capitalised by convention, but plain time vocabulary, and the harness job needs them: the plan's
# own eval scenario is "my appointments are on Tuesdays". Exempt from the name rule only - they
# still have to earn their place on usage like any other corpus word.
CALENDAR = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday mondays tuesdays wednesdays thursdays "
    "fridays saturdays sundays january february march april may june july august september october "
    "november december".split()
)

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
RAW_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
APOSTROPHES = "’ʼ´"
CLITICS = ("s", "re", "ve", "ll", "d", "m", "t")
# Characters that can sit between a full stop and the next word without the word being mid-sentence.
LEADING = " \t\r\n\"'“”‘’()[]{}*_-–—•#>+"
ENDINGS = ".!?:;"
# A capital right after one of these says "a title starts here", not "this word is a name": the
# corpus is full of *Turntable Time Capsule* and "The Alley Cat".
OPENERS = "*_\"'“‘([{"
TITLE_GAP = 3  # characters allowed between two words still counted as next to each other


# ----------------------------------------------------------------------------- folding rules
# tokens() -> bases() -> fold() is the whole of it. Everything downstream imports these three.


def normalise(text):
    """Lowercase, and one kind of apostrophe."""
    text = str(text).lower()
    for ch in APOSTROPHES:
        text = text.replace(ch, "'")
    return text


def tokens(text):
    """The words of a text: lowercase, letters only, contractions cut back to their head.

    "Don't forget the user's dentist" -> ["do", "forget", "the", "user", "dentist"]. Function words
    and short words are still here; that's `content_tokens()`'s job, because the encoder needs the
    frames too.
    """
    found = []
    for match in WORD.findall(normalise(text)):
        if "'" in match:
            head, tail = match.split("'", 1)
            if tail in CLITICS:
                if tail == "t" and head.endswith("n"):  # don't, can't, wasn't
                    head = head[:-1]
                match = head
            else:
                match = head + tail
        if match:
            found.append(match)
    return found


def is_content(word):
    """A word the model takes in as ideas rather than as a word: long enough, not grammar glue."""
    return len(word) >= MIN_LETTERS and word.isalpha() and word not in FUNCTION_WORDS


def bases(word):
    """The base forms `word` could fold to, best guess first. No dictionary is consulted here.

    Conservative by design: every candidate has to be found in the vocabulary before `fold()` will
    use it, so a wrong guess costs nothing but a wrong *rule* would.
    """
    out = []

    def add(candidate):
        if len(candidate) >= MIN_LETTERS and candidate != word and candidate not in out:
            out.append(candidate)

    if word in NEVER_FOLD:
        return out
    if word.endswith("ing") and len(word) >= 6:
        stem = word[:-3]
        for candidate in _stem_first(stem, stem + "e"):
            add(candidate)  # walking -> walk, liking -> like
        if _doubled(stem):
            add(stem[:-1])  # running -> run
    if word.endswith("ed") and len(word) >= 5:
        stem = word[:-2]
        if stem.endswith("i"):
            add(stem[:-1] + "y")  # tried -> try, before the bare stem "tri"
        for candidate in _stem_first(stem, word[:-1]):
            add(candidate)  # walked -> walk, stated -> state
        if _doubled(stem):
            add(stem[:-1])  # stopped -> stop
    if word.endswith("ies") and len(word) >= 5:
        add(word[:-3] + "y")  # stories -> story
    if word.endswith("es") and len(word) >= 5:
        add(word[:-1])  # bakes -> bake
        add(word[:-2])  # boxes -> box
    if word.endswith("s") and not word.endswith("ss") and len(word) >= 4:
        add(word[:-1])  # cats -> cat
    return out


def _doubled(stem):
    return len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou"


def _stem_first(stem, with_e):
    """Which of "the stem" and "the stem with its e back" to try first, and whether to try both.

    A one-syllable stem ending consonant-vowel-consonant would have doubled that consonant if it
    were the real base - "hopping", not "hoping" - so for those the bare stem isn't offered at all,
    and "stated" can't land on "stat" or "fired" on "fir". Longer stems don't double an unstressed
    ending, so "limited" -> "limit" and "visited" -> "visit" keep the bare stem.
    """
    if not _cvc(stem):
        return [stem, with_e]
    if _syllables(stem) == 1:
        return [with_e]
    return [with_e, stem]


def _syllables(stem):
    return len(re.findall(r"[aeiouy]+", stem))


def _cvc(stem):
    """Consonant-vowel-consonant at the end, the usual sign of a dropped "e": stat(e), hop(e)."""
    if len(stem) < 3 or stem[-1] in "aeiouwxy":
        return False
    return stem[-2] in "aeiou" and stem[-3] not in "aeiou"


def fold(word, known):
    """`word` folded to its base form when the base is in `known`; the surface form otherwise.

    `known` is anything with `in`: a set of words, or a loaded vocabulary. The base wins even when
    the surface form is in `known` too, and that's safe because `build()` guarantees a vocabulary
    never holds both a form and its base - `tests/test_vocabulary.py` checks the guarantee.

    Folding follows the chain: "findings" -> "finding" -> "find", because a one-step rule would
    leave "finding" standing next to "find" as a second row for the same word. Every candidate is
    strictly shorter than the word it came from, so the chain always ends; FOLD_STEPS caps it
    anyway. Folding twice gives the same answer as folding once.
    """
    for _ in range(FOLD_STEPS):
        for candidate in bases(word):
            if candidate in known:
                word = candidate
                break
        else:
            break
    return word


def content_tokens(text, known=()):
    """The content words of a text, folded against `known`. What the encoder tokenises with."""
    return [fold(word, known) for word in tokens(text) if is_content(word)]


# ----------------------------------------------------------------------------- the sources


def read_jsonl(path):
    with open(path) as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError as error:
                raise ValueError("%s line %d isn't JSON: %s" % (path, number, error))


def conversation_texts(path=CONVERSATIONS):
    """Every message in the accepted conversations, user and assistant alike."""
    for row in read_jsonl(path):
        for message in row.get("messages") or []:
            text = message.get("content")
            if text:
                yield text


def curriculum_words(path=CURRICULUM):
    """The accepted curriculum's target words."""
    return sorted({str(row.get("word") or "").strip().lower() for row in read_jsonl(path)} - {""})


def concept_nouns(path=CONCEPTS):
    """v1's hand-tagged nouns."""
    with open(path) as f:
        return sorted(json.load(f)["words"])


def count_corpus(texts):
    """Fold-free counts of every content word, plus the evidence for the proper-name rule.

    Returns (counts, capitalised, judged). A use only counts as evidence about the word itself when
    a capital there could mean nothing else, so three kinds of use are counted and then ignored:

      * the start of a sentence, where every word is capitalised;
      * right after a quote or an emphasis mark, where a title starts;
      * next to another capitalised word, which is Title Case - *Turntable Time Capsule* would
        otherwise cost the vocabulary "turntable", "time" and "capsule".

    The cost of ignoring a use is a real name kept; the cost of counting one is a plain word lost.
    The second is worse, so the rule leans that way.
    """
    counts, capitalised, judged = Counter(), Counter(), Counter()
    for text in texts:
        text = str(text)
        found = []
        for match in RAW_WORD.finditer(text):
            word = tokens(match.group(0))
            if len(word) == 1 and is_content(word[0]):
                found.append((word[0], match.start(), match.end(), match.group(0)[0].isupper()))
        for index, (word, start, end, upper) in enumerate(found):
            counts[word] += 1
            if _sentence_start(text, start) or _after_opener(text, start):
                continue
            if _beside_a_capital(text, found, index):
                continue
            judged[word] += 1
            if upper:
                capitalised[word] += 1
    return counts, capitalised, judged


def _sentence_start(text, at):
    index = at - 1
    while index >= 0 and text[index] in LEADING:
        index -= 1
    return index < 0 or text[index] in ENDINGS


def _after_opener(text, at):
    index = at - 1
    while index >= 0 and text[index] in " \t":
        index -= 1
    return index >= 0 and text[index] in OPENERS


def _beside_a_capital(text, found, index):
    """True when the word sits in a run of capitalised words - a title, not a sentence."""
    _, start, end, _ = found[index]
    if index and found[index - 1][3] and start - found[index - 1][2] <= TITLE_GAP:
        return True
    return (index + 1 < len(found) and found[index + 1][3]
            and found[index + 1][1] - end <= TITLE_GAP)


def looks_like_a_name(word, capitalised, judged):
    """Capitalised in most of the uses that count, on enough of them to mean anything."""
    uses = judged.get(word, 0)
    return uses >= NAME_USES and capitalised.get(word, 0) / uses >= NAME_SHARE


# ----------------------------------------------------------------------------- the build


def build(conversations=CONVERSATIONS, curriculum=CURRICULUM, concepts=CONCEPTS, min_count=MIN_COUNT):
    """The vocabulary, as (rows, summary). Deterministic: same inputs, same bytes out."""
    raw, capitalised, judged = count_corpus(conversation_texts(conversations))
    listed = {word for word in curriculum_words(curriculum) if is_content(word)}
    listed |= {word for word in concept_nouns(concepts) if is_content(word)}

    # The name rule only judges words mined from the corpus. The curriculum and v1's nouns are
    # human-approved lists; "rose" being a name somewhere doesn't take the flower out of them.
    names = {word for word in raw
             if word not in listed and word not in CALENDAR
             and looks_like_a_name(word, capitalised, judged)}

    # Folding needs a vocabulary to fold against, and the vocabulary needs folded counts to be
    # built. The knot is cut once, deterministically: a base form counts as "in the vocabulary" for
    # folding if it's on one of the listed sources or the corpus uses it at all. Three "dentist"
    # and two "dentists" should be one word over the threshold, not two words under it, and that
    # only works if "dentist" is a target before its own count is in.
    targets = listed | {word for word in raw if word not in names}
    counts, folded_into = Counter(), {}
    for word, n in raw.items():
        if word in names:
            continue
        base = fold(word, targets)
        if base in names:  # never fold a real word into an excluded name
            base = word
        counts[base] += n
        if base != word:
            folded_into.setdefault(base, []).append(word)

    from_corpus = {word for word, n in counts.items() if n >= min_count and word not in names and is_content(word)}

    # The listed words fold too, against the same targets: "boots" from v1's nouns belongs to the
    # "boot" row rather than sitting beside it, or the encoder couldn't tell which row to use. The
    # membership moves with the word, so "boot" still says it came from `concepts`.
    curriculum_set = {w for w in curriculum_words(curriculum) if is_content(w)}
    concepts_set = {w for w in concept_nouns(concepts) if is_content(w)}
    from_curriculum, from_concepts = set(), set()
    for word in curriculum_set:
        from_curriculum.add(fold(word, targets))
    for word in concepts_set:
        from_concepts.add(fold(word, targets))
    for word in sorted(curriculum_set | concepts_set):
        base = fold(word, targets)
        if base != word:
            folded_into.setdefault(base, []).append(word)

    words = sorted(from_corpus | from_curriculum | from_concepts)
    rows = []
    for word in words:
        sources = []
        if word in from_corpus:
            sources.append("conversations")
        if word in from_curriculum:
            sources.append("curriculum")
        if word in from_concepts:
            sources.append("concepts")
        rows.append({
            "word": word,
            "sources": sources,
            "count": counts.get(word, 0),  # conversation-corpus uses, after folding
            "forms": sorted(set(folded_into.get(word, []))),  # surface forms this row stands for
            "pos": "other",  # the teacher fills this in tag_words.py; nothing here guesses
        })

    summary = {
        "words": len(rows),
        "fold_version": FOLD_VERSION,
        "min_count": min_count,
        "by_source": {name: sum(1 for row in rows if name in row["sources"]) for name in SOURCES},
        "source_only": {
            name: sum(1 for row in rows if row["sources"] == [name]) for name in SOURCES
        },
        "in_all_three": sum(1 for row in rows if len(row["sources"]) == 3),
        "in_two_or_more": sum(1 for row in rows if len(row["sources"]) > 1),
        "corpus": {
            "texts_scanned": sum(1 for _ in conversation_texts(conversations)),
            "content_words_seen": sum(raw.values()),
            "distinct_content_words": len(raw),
            "names_excluded": len(names),
            "surface_forms_folded": sum(len(v) for v in folded_into.values()),
            "bases_that_absorbed_a_form": len(folded_into),
        },
        "never_used_in_conversations": sum(1 for row in rows if row["count"] == 0),
        "pos_filled": sum(1 for row in rows if row["pos"] != "other"),
        "names_sample": sorted(names)[:20],
        "folded_sample": sorted((base, sorted(forms)) for base, forms in folded_into.items())[:20],
    }
    return rows, summary


def write(rows, path=OUT):
    """One JSON object per line, sorted by word. Returns the file's SHA-256."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    with open(path, "w") as f:
        f.write(body)
    return hashlib.sha256(body.encode()).hexdigest()


def load(path=OUT):
    """`data/vocabulary.jsonl` as {word: row}, in file order. What world.py and encode.py read."""
    return {row["word"]: row for row in read_jsonl(path)}


# ----------------------------------------------------------------------------- CLI


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT, help="where to write the vocabulary")
    parser.add_argument("--conversations", default=CONVERSATIONS)
    parser.add_argument("--curriculum", default=CURRICULUM)
    parser.add_argument("--concepts", default=CONCEPTS)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT,
                        help="conversation uses a word needs to earn its own place (default %d)" % MIN_COUNT)
    parser.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("--fold", metavar="WORD", help="show what the folding rules do to one word")
    parser.add_argument("--show", metavar="WORD", help="show one word's row and the forms folded into it")
    args = parser.parse_args(argv)

    rows, summary = build(args.conversations, args.curriculum, args.concepts, args.min_count)
    known = {row["word"] for row in rows}

    if args.fold:
        word = normalise(args.fold).strip()
        print("%-16s candidates: %s" % (word, ", ".join(bases(word)) or "(none)"))
        print("%-16s folds to:   %s%s" % (word, fold(word, known),
                                          "" if fold(word, known) != word else "  (kept as is)"))
        return 0

    if args.show:
        word = fold(normalise(args.show).strip(), known)
        row = next((r for r in rows if r["word"] == word), None)
        if row is None:
            print("%s isn't in the vocabulary" % args.show)
            return 1
        print(json.dumps(row, sort_keys=True))
        return 0

    sha = None if args.dry_run else write(rows, args.out)
    summary["path"] = os.path.relpath(args.out, HERE)
    summary["sha256"] = sha

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    corpus = summary["corpus"]
    print("vocabulary: %d words  (fold rules v%d, min count %d)" % (
        summary["words"], summary["fold_version"], summary["min_count"]))
    print("  by source (a word can have more than one):")
    for name in SOURCES:
        print("    %-14s %5d  (%d from this source alone)" % (
            name, summary["by_source"][name], summary["source_only"][name]))
    print("    %-14s %5d in two or more, %d in all three" % (
        "overlap", summary["in_two_or_more"], summary["in_all_three"]))
    print("  conversation corpus: %d messages, %d content-word uses, %d distinct" % (
        corpus["texts_scanned"], corpus["content_words_seen"], corpus["distinct_content_words"]))
    print("    %d words left out as names, %d surface forms folded into %d bases" % (
        corpus["names_excluded"], corpus["surface_forms_folded"], corpus["bases_that_absorbed_a_form"]))
    print("  %d words have no conversation use yet (phase 2 owes them sentences)" %
          summary["never_used_in_conversations"])
    print("  pos: 0 of %d filled - the teacher's tagging pass fills it, not this script" % summary["words"])
    if sha:
        print("  wrote %s  sha256 %s" % (summary["path"], sha))
    else:
        print("  dry run, nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
