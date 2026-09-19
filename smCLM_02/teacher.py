"""The teacher client for smCLM_02: strict JSON, retried, and every call logged to JSONL.

A copy of `smCONVERSATION_001/teacher.py`, standalone (it doesn't import that project) and with two
things phase 1 of docs/engineering/plans/smCLM_02.md needs on top:

  * **a dry run.** `Teacher(..., dry_run=True).ask(...)` never opens a socket. It writes the exact
    request body it *would* have posted to the log, marked `"sent": false`, and raises `DryRun`.
    Callers catch that and stop, so a dry run can never be mistaken for an answer;
  * **the small shared bottom layer** the phase 1 scripts sit on: the file helpers (`append_jsonl`,
    `read_jsonl`, `write_json`, `write_jsonl`), `load_vocabulary()` and `batches()`. They live here
    so `propose_ideas.py`, `tag_words.py` and `held_out.py` agree on how a file is read, how the
    vocabulary is ordered, and how a run is cut into resumable batches.

The default endpoint is Ministral 8B on the PC. `--provider openai` switches to OpenAI's
Chat Completions API and reads the key only from `OPENAI_API_KEY`; `--provider kimi` switches to
Moonshot's `kimi-k2.6` with thinking off and reads the key only from `KIMI_API_KEY`. Both paid
providers get a software spend cap. Nothing here downloads anything, and API keys are never written
to the request log.

`ask()` is safe to call from several threads at once: log writes are serialised, and the spend cap
is an in-memory ledger that reserves each call's worst case before it's sent, so parallel calls
can't all see "under budget" at the same moment.

    python3 teacher.py --dry-run          # print the body of one small call, post nothing
"""

import argparse
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request


THINK = re.compile(r"<think>.*?</think>", re.S | re.I)

DEFAULT_ENDPOINT = os.environ.get("SMMOL_LLM_URL", "http://127.0.0.1:8081")
DEFAULT_MODEL = "ministral-8b"
OPENAI_ENDPOINT = "https://api.openai.com"
OPENAI_MODEL = "gpt-5.6-luna"
OPENAI_DEFAULT_BUDGET_USD = 0.25

# Standard, short-context prices per million tokens. The cap deliberately charges every input
# token at the full (not cached) rate, so caching can only make the real bill smaller.
PRICES = {
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "kimi-k2.6": {"input": 0.95, "output": 4.00},
    "kimi-k3": {"input": 3.00, "output": 15.00},
}
OPENAI_PRICES = PRICES  # the old name, kept for callers that still use it
PAID = ("openai", "kimi")
KIMI_ENDPOINT = "https://api.moonshot.ai"
KIMI_MODEL = "kimi-k2.6"
_WRITE = threading.Lock()

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LOGS = DATA / "logs"
VOCABULARY = DATA / "vocabulary.jsonl"
IDEAS = HERE / "ideas.json"

MIN_IDEAS = 120  # the phase 1 gate
V1_IDEAS = 51  # smCLM_01's inventory, the approved seed of ideas.json
POS = ("noun", "verb", "adjective", "other")


class DryRun(Exception):
    """Raised instead of answering, when a Teacher is in dry-run mode."""

    def __init__(self, body):
        super().__init__("dry run: nothing was sent to the teacher")
        self.body = body


class BudgetExceeded(RuntimeError):
    """Raised before a request that could cross the configured software spend cap."""


def append_jsonl(path, item):
    path = Path(path)
    line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _WRITE:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(line)
            handle.flush()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError("%s line %d isn't JSON: %s" % (path, number, exc))
    return rows


def write_jsonl(path, items):
    """Whole-file write through a temporary, so a killed run never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def load_vocabulary(path=None):
    """data/vocabulary.jsonl as rows sorted by word.

    Sorted, because every batch boundary in phase 1 is an index into this list: the same file must
    always cut into the same batches, whatever order `vocabulary.py` happened to write it in.
    """
    path = Path(path or VOCABULARY)
    if not path.exists():
        raise SystemExit(
            "%s doesn't exist yet. vocabulary.py builds it (phase 1 of the plan); these tools read it."
            % path)
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        word = row.get("word")
        if not isinstance(word, str) or not word:
            raise SystemExit("%s has a row with no word: %r" % (path, row))
        if word in seen:
            raise SystemExit("%s lists %r twice" % (path, word))
        seen.add(word)
    return sorted(rows, key=lambda row: row["word"])


def load_inventory(path=None, allow_partial=False):
    """ideas.json as {idea: one-line meaning}, with the human-approval gate.

    The file ships seeded with smCLM_01's 51 ideas verbatim and nothing else, because that's the
    only part of the inventory a person has already approved. The plan's phase 1 gate wants at
    least MIN_IDEAS, and the roughly 100 new ones are picked by a person from
    `data/idea_proposals.json` — `propose_ideas.py` tallies candidates, it never writes ideas.json.
    So anything that tags against the inventory refuses to run while it's still the seed, unless
    the caller says out loud that a partial inventory is what it wants.
    """
    path = Path(path or IDEAS)
    inventory = read_json(path)
    if not isinstance(inventory, dict) or not inventory:
        raise SystemExit("%s must be an object of idea -> one-line meaning" % path)
    for idea, meaning in inventory.items():
        if not re.fullmatch(r"[a-z][a-z_]{1,23}", idea):
            raise SystemExit("%s: %r isn't a usable idea name (lowercase, up to 24 characters)"
                             % (path, idea))
        if not isinstance(meaning, str) or not meaning.strip():
            raise SystemExit("%s: %r has no one-line meaning" % (path, idea))
    if len(inventory) < MIN_IDEAS and not allow_partial:
        raise SystemExit(
            "%s holds %d ideas, and the phase 1 gate wants at least %d.\n"
            "The seed is smCLM_01's %d verbatim; the new ideas are still waiting on a person to\n"
            "pick them from data/idea_proposals.json and write each one's meaning. Run\n"
            "propose_ideas.py first, or pass --allow-partial-inventory to tag against the seed."
            % (path, len(inventory), MIN_IDEAS, V1_IDEAS))
    return inventory


def batches(items, size):
    """[(index, [item, ...]), ...] — a fixed cut, so batch 7 means the same thing on every run."""
    return [(i // size, items[i:i + size]) for i in range(0, len(items), size)]


def post_json(url, body, timeout=900, headers=None):
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class Teacher:
    def __init__(self, endpoint, model, log_path, retries=3, dry_run=False, provider="pc",
                 api_key=None, max_budget_usd=None):
        if provider not in ("pc",) + PAID:
            raise ValueError("provider must be 'pc', 'openai' or 'kimi'")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.log_path = Path(log_path)
        self.retries = retries
        self.dry_run = dry_run
        self.provider = provider
        self.api_key = api_key
        self.max_budget_usd = max_budget_usd
        self._ledger = threading.Lock()
        self._spent = None  # read from the log once, then kept in memory
        self._reserved = 0.0

    def completion_url(self):
        suffix = "/chat/completions" if self.endpoint.endswith("/v1") else "/v1/chat/completions"
        return self.endpoint + suffix

    def request_headers(self):
        if self.provider in PAID and self.api_key:
            return {"Authorization": "Bearer " + self.api_key}
        return None

    def prices(self):
        for name, prices in PRICES.items():
            if self.model == name or self.model.startswith(name + "-"):
                return prices
        raise ValueError(
            "no audited price is configured for %s; add it before using a dollar cap" % self.model)

    def usage_cost(self, usage):
        """Charge a Chat Completions usage record at the configured full-token rates."""
        if self.provider not in PAID or not usage:
            return 0.0
        prices = self.prices()
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        return round((input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000,
                     9)

    def logged_cost(self):
        """Cost already recorded in this run log, read from disk once and then kept in memory."""
        with self._ledger:
            if self._spent is None:
                self._spent = sum(row["cost_usd"] for row in read_jsonl(self.log_path)
                                  if isinstance(row.get("cost_usd"), (int, float)))
            return self._spent

    def worst_case_cost(self, body):
        """Conservative upper bound for the next call, used before a socket is opened.

        A byte upper-bounds BPE tokens in the serialized payload; the extra 1,024 covers message
        framing and API-side wrappers. Output is bounded by `max_completion_tokens`.
        """
        prices = self.prices()
        input_tokens = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + 1024
        output_tokens = int(body.get("max_completion_tokens", body.get("max_tokens", 0)))
        return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000

    def check_budget(self, body):
        """Reserve this call's worst case, or refuse. Returns the reservation to settle afterwards."""
        if self.provider not in PAID or self.max_budget_usd is None:
            return 0.0
        spent = self.logged_cost()
        reserve = self.worst_case_cost(body)
        with self._ledger:
            if spent + self._reserved + reserve > self.max_budget_usd:
                raise BudgetExceeded(
                    "%s spend cap would be crossed: $%.6f logged + $%.6f in flight + $%.6f maximum "
                    "next call > $%.6f cap" % (self.provider, spent, self._reserved, reserve,
                                               self.max_budget_usd))
            self._reserved += reserve
        return reserve

    def settle(self, reserve, cost):
        """Swap a call's reservation for what it actually cost."""
        with self._ledger:
            self._reserved -= reserve
            if self._spent is not None:
                self._spent += cost

    def body(self, kind, system, prompt, schema, seed, temperature, max_tokens):
        """The request, built but not sent. A dry run prints this; `ask` posts it."""
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": kind, "strict": True, "schema": schema},
            },
        }
        if self.provider == "openai":
            # GPT-5 family models use max_completion_tokens and reasoning_effort. Omitting
            # temperature/seed also keeps this body compatible with reasoning models.
            body.update({"max_completion_tokens": max_tokens, "reasoning_effort": "none"})
        elif self.provider == "kimi":
            # kimi-k2.6 thinks by default: 2,811 hidden tokens and 75 s for ten sentences, measured
            # 2026-09-19, against 188 tokens and 6 s with it off. It only accepts temperature 1,
            # and takes no seed, so both are left out and the sampling is the server's.
            body.update({"max_tokens": max_tokens, "thinking": {"type": "disabled"}})
        else:
            body.update({
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False},
            })
        return body

    def ask(self, kind, system, prompt, schema, seed, temperature, max_tokens, metadata=None):
        body = self.body(kind, system, prompt, schema, seed, temperature, max_tokens)
        if self.dry_run:
            append_jsonl(self.log_path, {
                "kind": kind,
                "seed": seed,
                "attempt": 0,
                "metadata": metadata or {},
                "sent": False,
                "ok": False,
                "dry_run": True,
                "provider": self.provider,  # not the endpoint: it's a LAN address, and this log is published
                "body": body,
            })
            raise DryRun(body)
        if self.provider in PAID and not self.api_key:
            raise RuntimeError(
                "%s is not set. Install a fresh key locally; do not reuse a key "
                "that was pasted into chat." % ("OPENAI_API_KEY" if self.provider == "openai"
                                                else "KIMI_API_KEY"))
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                reserve = self.check_budget(body)
            except BudgetExceeded as exc:
                append_jsonl(self.log_path, {
                    "kind": kind,
                    "seed": seed,
                    "attempt": attempt,
                    "metadata": metadata or {},
                    "provider": self.provider,
                    "model": self.model,
                    "sent": False,
                    "ok": False,
                    "budget_blocked": True,
                    "error": str(exc),
                })
                raise
            started = time.time()
            data = None
            cost = 0.0
            try:
                headers = self.request_headers()
                if headers:
                    data = post_json(self.completion_url(), body, headers=headers)
                else:
                    data = post_json(self.completion_url(), body)
                text = str(data["choices"][0]["message"]["content"])
                text = THINK.sub("", text).strip()
                try:
                    parsed = json.loads(text)
                except ValueError:
                    start = text.find("{")
                    parsed, _ = json.JSONDecoder().raw_decode(text[start:])
                usage = data.get("usage") or {}
                row = {
                    "kind": kind,
                    "seed": seed,
                    "attempt": attempt,
                    "seconds": round(time.time() - started, 3),
                    "metadata": metadata or {},
                    "provider": self.provider,
                    "model": self.model,
                    "usage": usage,
                    "sent": True,
                    "ok": True,
                }
                if self.provider in PAID:
                    cost = row["cost_usd"] = self.usage_cost(usage)
                    row["budget_usd"] = self.max_budget_usd
                append_jsonl(self.log_path, row)
                return parsed
            except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
                last_error = exc
                usage = data.get("usage") or {} if isinstance(data, dict) else {}
                row = {
                    "kind": kind,
                    "seed": seed,
                    "attempt": attempt,
                    "seconds": round(time.time() - started, 3),
                    "metadata": metadata or {},
                    "provider": self.provider,
                    "model": self.model,
                    "sent": True,
                    "ok": False,
                    "error": str(exc)[:500],
                }
                if usage:
                    row["usage"] = usage
                if self.provider in PAID:
                    cost = row["cost_usd"] = self.usage_cost(usage)
                    row["budget_usd"] = self.max_budget_usd
                append_jsonl(self.log_path, row)
                if attempt < self.retries:
                    limited = isinstance(exc, urllib.error.HTTPError) and exc.code == 429
                    time.sleep(10 * attempt if limited else min(2 ** attempt, 8))
            finally:
                self.settle(reserve, cost)
        raise RuntimeError("teacher failed after %d attempts: %s" % (self.retries, last_error))


def add_teacher_arguments(parser):
    """The flags every phase 1 script shares. `--dry-run` is the default nowhere; it's always typed."""
    parser.add_argument("--provider", choices=("pc",) + PAID, default="pc")
    parser.add_argument("--endpoint", default=None,
                        help="override the provider endpoint (normally leave this unset)")
    parser.add_argument("--model", default=None,
                        help="override the provider model (normally leave this unset)")
    parser.add_argument("--max-budget-usd", type=float, default=OPENAI_DEFAULT_BUDGET_USD,
                        help="paid-provider spend cap across this log (default: $0.25)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="build every request and show the plan, but post nothing")
    return parser


def teacher_from(args, log):
    """A Teacher from the shared flags. A bare name lands in data/logs/; a path is taken as given."""
    path = Path(log)
    if len(path.parts) == 1:
        path = LOGS / path
    if args.provider == "openai":
        endpoint = args.endpoint or OPENAI_ENDPOINT
        model = args.model or OPENAI_MODEL
        api_key = os.environ.get("OPENAI_API_KEY")
        budget = args.max_budget_usd
    elif args.provider == "kimi":
        endpoint = args.endpoint or KIMI_ENDPOINT
        model = args.model or KIMI_MODEL
        api_key = os.environ.get("KIMI_API_KEY")
        budget = args.max_budget_usd
    else:
        endpoint = args.endpoint or DEFAULT_ENDPOINT
        model = args.model or DEFAULT_MODEL
        api_key = None
        budget = None
    return Teacher(endpoint, model, path, args.retries, args.dry_run, args.provider, api_key, budget)


def main():
    parser = add_teacher_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--log", default=str(LOGS / "teacher_selftest.jsonl"))
    args = parser.parse_args()
    teacher = teacher_from(args, args.log)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }
    try:
        answer = teacher.ask("selftest", "Answer with JSON.", "Say ok.", schema, 1, 0.0, 16)
    except DryRun as dry:
        print(json.dumps(dry.body, indent=2))
        print("\ndry run: nothing was sent to %s" % teacher.endpoint)
        return
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
