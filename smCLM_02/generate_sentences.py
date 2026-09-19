"""smCLM_02 phase 2: teacher-written sentences for every vocabulary word still under 30 usages.

Plan section "Phase 2", source 3. Per word, one call asks for PER_CALL sentences that use the word in
different settings, roles and tenses. Every sentence has to pass the structural checks before it's
kept:

  * one plain sentence: no line breaks, no list marks or markdown, no second sentence;
  * MIN_WORDS to MAX_WORDS words (`vocabulary.tokens()`);
  * it uses the word, or a form that folds to it (`vocabulary.content_tokens()`);
  * not an exact or near copy of anything already kept, from any source (`compile_data.NearCopies`);
  * not, or nearly, a `test_recall.json` text (`compile_data.LeakGuard`).

Usages are counted over everything kept so far, the conversation and curriculum sentences included,
and a teacher sentence for "doctor" that also says "hospital" counts for both. So a word is asked
about only while it's actually short.

The work goes in rounds: each round gives every short word one call, in sorted order, CHUNK words at
a time. With `--workers` above 1 a chunk's calls run in parallel, but its answers are judged in
sorted order after they're all back, so the corpus doesn't depend on which call finished first.
`--rpm` spaces the calls out to stay under a provider's requests-a-minute limit.

Resumable and deterministic in the same way as `propose_ideas.py`. Every call's result - kept
sentences and rejections by reason - is appended to `data/logs/sentence_calls.jsonl` before the next
request goes out, and a restart rebuilds the whole state from that log. A word that can't reach the
target in MAX_CALLS_PER_WORD calls is left short and shows up in `compile_data.py`'s report; that is
a finding about the word, not something to retry forever.

The held-out words are generated for like every other word. That's the point of them: their
sentences are visible, only their tags are hidden.

    python3 generate_sentences.py --dry-run              the plan and one example request
    python3 generate_sentences.py --limit 50             50 calls, then stop; rerun to carry on
    python3 generate_sentences.py --words dentist,axe    only these words (a canary)
    python3 generate_sentences.py --summary              counts from the log, no calls
    KIMI_API_KEY=... python3 generate_sentences.py --provider kimi --workers 12 --rpm 90 --max-budget-usd 8
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import re
import sys
import threading
import time
import urllib.request

import compile_data as C
import vocabulary as V
from teacher import (BudgetExceeded, DryRun, add_teacher_arguments, append_jsonl, load_vocabulary,
                     read_jsonl, teacher_from)

KIND = "sentences"
PROMPT_VERSION = "v1"
SEED = 18001  # + the word's call number, so a retry of the same word is a different sample
PER_CALL = 10
TARGET = C.TARGET
MIN_WORDS, MAX_WORDS = 6, 25
MAX_CALLS_PER_WORD = 6
TEMPERATURE = 0.9
MAX_TOKENS = 1400

LOG = C.TEACHER_LOG
CHUNK = 48  # words whose calls are in flight together; also the most a crash can lose

SYSTEM = ("You write example sentences for a word list. Answer with JSON only, in the schema you are "
          "given.")

PROMPT = """Write {n} different English sentences that each use the word "{word}".{forms}

Rules:
- Each sentence is one ordinary sentence of {low} to {high} words, the kind a person might say or
  write in everyday life.
- Use the word in different settings, with different people, and in different tenses or roles.
- Every sentence starts differently. Do not repeat a sentence pattern.
- Plain text only: no lists, no numbering, no quotation marks around the sentence, no markdown.
- Use the word in its ordinary meaning.
"""

SCHEMA_ITEM = {"type": "string"}

LIST_MARK = re.compile(r"^\s*(?:[-*•#>]|\d+[.)])\s*")
MARKUP = re.compile(r"\*\*|__|`|[\[\]{}<>|]")
TWO_SENTENCES = re.compile(r"[.!?][\"'”’)]*\s+[\"'“‘(]?[A-Z]")


def schema(n):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["sentences"],
        "properties": {"sentences": {"type": "array", "minItems": n, "maxItems": n,
                                     "items": SCHEMA_ITEM}},
    }


def prompt_for(word, forms, n=PER_CALL):
    others = [f for f in forms if f != word]
    hint = (" Any form of it is fine: %s." % ", ".join([word] + others)) if others else ""
    return PROMPT.format(n=n, word=word, forms=hint, low=MIN_WORDS, high=MAX_WORDS)


def problem_with(text, word, known):
    """Why a sentence can't be kept, or None. Only the structural checks; copies are checked later."""
    if not isinstance(text, str):
        return "not text"
    text = text.strip()
    if "\n" in text:
        return "more than one line"
    if LIST_MARK.match(text) or MARKUP.search(text):
        return "list mark or markup"
    if TWO_SENTENCES.search(text):
        return "more than one sentence"
    n = len(V.tokens(text))
    if not MIN_WORDS <= n <= MAX_WORDS:
        return "length"
    if word not in V.content_tokens(text, known):
        return "doesn't use the word"
    return None


class State:
    """Everything kept so far, rebuilt from the base sentences and the log. Order-dependent, on purpose."""

    def __init__(self, vocabulary, log_path=LOG, guard=None, base=None):
        self.known = {row["word"] for row in vocabulary}
        self.guard = guard or C.LeakGuard()
        self.copies = C.NearCopies()
        self.uses = Counter()
        self.calls = Counter()
        self.rejected = Counter()
        self.kept = 0
        for text, _source in (C.base_sentences(self.guard) if base is None else base):
            self.keep(text)
        self.kept = 0
        for row in read_jsonl(log_path):
            if row.get("result") and row.get("prompt_version") == PROMPT_VERSION:
                self.calls[row["word"]] += 1
                self.rejected.update(row.get("rejected") or {})
                for text in row.get("kept") or ():
                    self.keep(text)

    def keep(self, text):
        self.copies.add(text)
        self.uses.update({w for w in V.content_tokens(text, self.known) if w in self.known})
        self.kept += 1

    def judge(self, answer, word):
        """(kept, rejected-by-reason) for one call's answer. Keeps as it goes, so a call can't copy itself."""
        kept, rejected = [], Counter()
        sentences = answer.get("sentences") if isinstance(answer, dict) else None
        if not isinstance(sentences, list):
            rejected["answer has no sentences array"] += 1
            return kept, rejected
        for text in sentences:
            why = problem_with(text, word, self.known)
            if why is None:
                text = text.strip()
                if self.copies.exact(text):
                    why = "exact copy"
                elif self.copies.near(text):
                    why = "near copy"
                elif self.guard.leaks(text):
                    why = "test text"
            if why:
                rejected[why] += 1
            else:
                self.keep(text)
                kept.append(text)
        return kept, rejected

    def short(self, words):
        return [w for w in words if self.uses[w] < TARGET and self.calls[w] < MAX_CALLS_PER_WORD]


def reachable(teacher, timeout=10):
    """Whether the teacher's server answers at all. Tells "the PC went away" from "this word is hard"."""
    base = teacher.endpoint[:-3] if teacher.endpoint.endswith("/v1") else teacher.endpoint
    request = urllib.request.Request(base + "/v1/models", headers=teacher.request_headers() or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except OSError:
        return False


class Pace:
    """At most `rpm` calls a minute across every thread; 0 means no limit."""

    def __init__(self, rpm):
        self.gap = 60.0 / rpm if rpm else 0.0
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if not self.gap:
            return
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next)
            self.next = at + self.gap
        time.sleep(max(0.0, at - now))


def summary(state, words):
    under = [w for w in words if state.uses[w] < TARGET]
    gave_up = [w for w in under if state.calls[w] >= MAX_CALLS_PER_WORD]
    print("teacher sentences kept: %d from %d calls" % (state.kept, sum(state.calls.values())))
    for reason, n in state.rejected.most_common():
        print("  rejected %d: %s" % (n, reason))
    print("%d of %d words under %d usages; %d of those out of calls%s"
          % (len(under), len(words), TARGET, len(gave_up),
             (": " + ", ".join(gave_up[:12])) if gave_up else ""))


def main(argv=None):
    parser = add_teacher_arguments(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--log", default=LOG)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many calls")
    parser.add_argument("--words", default=None, help="comma-separated words to do, instead of all")
    parser.add_argument("--summary", action="store_true", help="counts from the log; make no calls")
    parser.add_argument("--workers", type=int, default=1, help="calls in flight at once")
    parser.add_argument("--rpm", type=float, default=0, help="most calls a minute, across workers (0: no limit)")
    args = parser.parse_args(argv)

    vocabulary = load_vocabulary()
    forms = {row["word"]: row.get("forms") or [] for row in vocabulary}
    words = [row["word"] for row in vocabulary]
    if args.words:
        wanted = [w.strip() for w in args.words.split(",") if w.strip()]
        unknown = sorted(set(wanted) - set(words))
        if unknown:
            raise SystemExit("not in the vocabulary: %s" % ", ".join(unknown))
        words = wanted

    state = State(vocabulary, args.log)
    if args.summary:
        summary(state, words)
        return 0
    todo = state.short(words)
    print("%d words asked for, %d still short; %d calls made so far"
          % (len(words), len(todo), sum(state.calls.values())))

    teacher = teacher_from(args, args.log)
    if args.dry_run:
        word = todo[0] if todo else words[0]
        try:
            teacher.ask(KIND, SYSTEM, prompt_for(word, forms[word]), schema(PER_CALL), SEED,
                        TEMPERATURE, MAX_TOKENS, {"word": word, "call": 0, "prompt_version": PROMPT_VERSION})
        except DryRun as dry:
            print(json.dumps(dry.body, indent=2))
        print("\ndry run: nothing was sent")
        return 0

    pace = Pace(args.rpm)

    def ask(word):
        call = state.calls[word]
        pace.wait()
        try:
            return teacher.ask(KIND, SYSTEM, prompt_for(word, forms[word]), schema(PER_CALL),
                               SEED + call, TEMPERATURE, MAX_TOKENS,
                               {"word": word, "call": call, "prompt_version": PROMPT_VERSION})
        except RuntimeError as error:  # the teacher gave up after its retries, or the budget ran out
            return error

    made = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        while True:
            todo = state.short(words)
            if not todo:
                break
            for i in range(0, len(todo), CHUNK):
                # Earlier chunks this round may already have filled a word through other words' sentences.
                chunk = state.short(todo[i:i + CHUNK])
                if args.limit:
                    chunk = chunk[:max(0, args.limit - made)]
                if not chunk:
                    continue
                answers = dict(zip(chunk, pool.map(ask, chunk)))
                stop = None
                failed = {w: a for w, a in answers.items() if isinstance(a, Exception)}
                if any(isinstance(a, BudgetExceeded) for a in failed.values()):
                    stop = (4, "the spend cap is reached")
                elif failed and not reachable(teacher):
                    stop = (3, "the teacher isn't answering")
                for word in chunk:  # sorted order, whatever order the calls finished in
                    answer = answers[word]
                    call = state.calls[word]
                    if isinstance(answer, Exception):
                        # Not charged when it's the teacher's fault rather than the word's: an outage,
                        # the cap, or a rate limit. Charging those would burn every word's budget.
                        if stop or "429" in str(answer):
                            print("%s call %d: not charged: %s" % (word, call, answer))
                            continue
                        print("%s call %d: teacher failed on this word: %s" % (word, call, answer))
                        answer = {}
                    kept, rejected = state.judge(answer, word)
                    append_jsonl(args.log, {"result": True, "word": word, "call": call,
                                            "prompt_version": PROMPT_VERSION,
                                            "kept": kept, "rejected": dict(rejected)})
                    state.calls[word] += 1
                    state.rejected.update(rejected)
                    made += 1
                    print("%s call %d: kept %d, rejected %d, usages now %d"
                          % (word, call, len(kept), sum(rejected.values()), state.uses[word]), flush=True)
                if stop:
                    print("stopping: %s. Rerun to carry on." % stop[1])
                    summary(state, words)
                    return stop[0]
                if args.limit and made >= args.limit:
                    print("stopping at --limit %d; rerun to carry on" % args.limit)
                    summary(state, words)
                    return 0
    summary(state, words)
    return 0


if __name__ == "__main__":
    sys.exit(main())
