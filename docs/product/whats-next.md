# What's next

These are ideas that have come up in the project. **Nothing on this page is built** unless it says otherwise. Each one
has a status line so there's no confusion.

Status words:
- **Not built:** an idea only. No code exists.
- **Known problem:** something that works today but not well enough.

---

## A Learn module

**Status: not built.** Proposed on September 15, 2026. No decision to build it has been made yet.

### The idea

Today, a model only gets better when someone retrains it by hand. A Learn module would notice the assistant's own
mistakes and use them to improve the models.

### The one rule that matters

**It only learns from mistakes that were proven wrong.** Some examples of proof:

- **The calculator.** Every time the arithmetic model gets a sum wrong, the calculator already catches it. That gives a
  mistake and the right answer, with no human needed.
- **The tests.** A check that failed.
- **You.** "No, I meant…" or "that's wrong".

If it learned from its own guesses, it would teach itself its own mistakes.

### How it would work

1. **Notice.** After each reply, it writes down lessons: which part made the mistake, what it said, what was right, and
   how we know.
2. **Quick fix first.** An approved lesson goes on a short list the part checks before answering. It takes effect on
   the next message, with no training.
3. **Retrain later.** When a part has piled up enough lessons (50 was the example given), or the Mac is idle:
   - train a copy on its original lessons plus the new ones, so it doesn't forget old skills;
   - test the copy;
   - use it only if it's no worse than before and fixes most of the lessons;
   - keep the old version available to switch back to.

### Why not learn live, on every message?

- **Small models forget.** Updating them constantly tends to overwrite what they already knew. When the message sorter
  was retrained to handle bare math like "2+2", its score on the original test slipped from 73% to 71%. That might be
  noise, but it shows the risk.
- **The quick-fix list covers "right now,"** and the assistant's memory already covers facts.
- **Retraining is quick anyway.** The message sorter trains in about 14 minutes and the math reader in about 26.

### Where it would start

With the arithmetic model, because the calculator can grade it with no human involved. The message sorter would come
next, learning from the times other parts overruled it and from your corrections.

---

## A command-line tool

**Status: not built.** It already shows on the page as a faded, dashed tool card called **Command line**, marked
"todo", whose switch can't be turned on.

- **There is already a model for the hard half of it.** smTOOLS_COMPUTER_CLI_01 turns a plain request into the right
  command for Ubuntu, Fedora, Arch, macOS or Windows — see [the models](smModels/models.md#smtools_computer_cli_01-the-terminal-translator).
  It writes commands; it cannot run them, and nothing in that project can. The missing piece is the approval flow, not
  the translation.
- **What it would do:** let the assistant run programs on the computer.
- **Why it isn't built:** that's powerful and risky. The plan is that a person has to approve **every single command**
  before it runs, and it would work inside a walled-off sandbox folder.

---

## Memory that searches by meaning ("vector memory")

**Status: not built.**

- **Today:** the assistant's long-term memory finds past facts by matching words. If the words don't overlap, it can
  miss a memory that's clearly related.
- **The idea:** search by meaning instead of exact words. A search for "doctor" could then find a memory that only says
  "dentist".
- **What it needs:** a separate embedding model (explained in the [glossary](glossary.md)), which means a download.
- **A second part:** link memories together through the people and places they share.
- **A plan exists:** [smCLM_02](../engineering/plans/smCLM_02.md) (2026-09-17) proposes doing this with a model trained
  here, smCLM_01's idea reader grown to real sentences, instead of a download. Still not built.

---

## Other things on the list

| Item | Status | What it means |
|---|---|---|
| smMATH001-a: division and bigger numbers | **Known problem** | Trained on September 15, 2026. It divides correctly only 38% of the time and mostly fails on numbers bigger than it practiced on. Why it doesn't stretch to longer numbers, when smMATH01's best version does, hasn't been tested |
| Asking too many questions | **Known problem** | Code rules already stop some questions, but the part that looks for problems still flags details that have sensible defaults |
| Saving files | **Known problem** | Since the "make it now" change, the assistant offers to save a file less often when asked to save something |
| Weather and other live facts | **Known problem** | In the September 15 test, it asked a question instead of searching the web |
| Replacing the big model in more jobs | **Not built** | 8 of the 11 parts still use one mid-sized model. The Predictor and Critic, for example, aren't trained small models |
| Speed | **Known problem** | One message means 8 or more model calls on one home graphics card |
| Message sorter: real phrasing | **Not built** | Train on messages it has never seen, like code requests, short thank-yous and sports scores, and keep the version that does best on real-sounding messages |
| Math reader: long numbers | **Not built** | Help it keep count of long runs of zeros, the trick the arithmetic model already uses |
| Idea vectors (smCLM_01) in the assistant | **Not built** | Pass "bundles of ideas" between parts instead of plain text. Planned in [smCLM_02](../engineering/plans/smCLM_02.md) |
| A model that learns while it runs (smRTS_01) | **Not built** | Can a tiny byte model with an internal state instead of a context window learn to remember, training one byte at a time? Tests a public repo's claims. Planned in [smRTS_01](../engineering/plans/smRTS_01.md) |
| Training in 16-bit, and stopping when it stops improving | **Measured 2026-09-17** | The Mac does 16-bit math 4× faster than the 32-bit every model currently uses. Switching is one line and gives the same quality in two thirds of the time; stopping at the best checkpoint instead of running out the clock saves as much again. See [smEFFICIENCY_01](../../smEFFICIENCY_01/README.md) |
