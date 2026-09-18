# SMMOL, in plain English

## What it is

SMMOL is a home research project about small AI models.

The popular AI assistants today run one giant model in a data center. SMMOL tests a different idea: **lots of small
models, each doing one "brain job", running on computers at home.**

- One small model sorts messages: is this small talk, a question, a math problem?
- One pulls the math problems out of a sentence.
- One does arithmetic.
- Plain computer code does the jobs that don't need AI at all, like exact math, saving files and remembering facts.

A web page called the harness puts the pieces together into one assistant you can chat with. You can switch each piece
on or off and watch how the answer changes.

All the small models were trained from scratch on a MacBook Pro M5. Bigger AI jobs that still need a larger
model run on a gaming PC in the house.

## Why it might matter

- **Cheaper to build.** A small model trains in minutes on a laptop. The message sorter trained in about 14 minutes.
- **Faster at its one job.** The message sorter answers in about 2 thousandths of a second. The larger model on the home
  PC takes about 2 seconds for the same job.
- **Stays at home.** The models run on a Mac and a PC in the house. Only web searches and web page reads go out to the
  internet.
- **Easier to check.** Each piece has one narrow job, so it can be tested on its own, swapped for another, or
  switched off.

## Where it stands (September 15, 2026)

**What works:**
- The chat page and the "brain" of parts.
- Switching any part on or off.
- Four small models trained here and plugged into the assistant: the message sorter, the math reader, and two
  arithmetic models (one does math in its head, one writes out its work).
- A calculator that checks every answer the arithmetic model gives.

**What doesn't work well yet:**
- **It's slow.** Most parts still use one mid-sized model (Ministral 8B) on a single home graphics card. In tests, a
  full reply took from 46 seconds to more than 2 minutes.
- **The big claim isn't proven.** Only 3 of the 11 parts are small trained models so far, so "small models are
  enough" is still untested.
- **The small models make mistakes:**
  - the message sorter gets everything right on 71% of hand-written test messages;
  - the math reader gets every problem right on 78%;
  - neither arithmetic model copes with numbers bigger than the ones it practiced on.
- **It asks too many questions.** The part that looks for problems still flags too much.
- **Dividing is weak.** smMATH001-a, the model that writes out its work, gets multiplication right 97% of the time at
  the sizes it practiced, but division only 38%.

## Read next

| Page | What it covers |
|---|---|
| [paratroop_harness/the-harness.md](paratroop_harness/the-harness.md) | A tour of the chat page: the cards, on and off, settings, chatting, and reading "how it got there" |
| [smModels/models.md](smModels/models.md) | Every model in a few plain sentences, and how good it is right now |
| [whats-next.md](whats-next.md) | Ideas that have been discussed but aren't built |
| [glossary.md](glossary.md) | Plain definitions of the words these pages can't avoid |

Developers and coding agents: see [the engineering docs](../engineering/README.md).
