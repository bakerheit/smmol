# A tour of the harness page

The harness is the web page where all the parts of the assistant come together. You chat on the right, and on the left
you watch the parts work.

**Where it runs:** on the MacBook Pro M5 itself, at http://127.0.0.1:8771. It won't open from another
computer, on purpose.

**Before you start:** it's slow. Most parts still ask one mid-sized AI model on the home PC, one after another. In
tests, a full reply took from 46 seconds to more than 2 minutes.

---

## The top bar

- **The title:** "Paratroop Harness 02 · cognitive modules".
- **A row of dots**, one per computer or service the page depends on:
  - **PC · RX 580:** the home PC that runs the bigger AI model;
  - **Mac · UnlimitedStudio:** an AI app on the Mac that can stand in for the PC;
  - **PC · browse server:** does web searches and reads web pages;
  - **Mac · trained in SMMOL:** the small models trained in this project.
- **Green** means reachable. **Red** means not. Hover over a dot to see why.

## The circle of cards

- **The cards:** the 11 cards around the circle are the assistant's parts. Each one does one "brain job".
- **The center:** the dashed circle is **working memory**, the scratchpad every part reads from and writes to. It's plain
  data, not a model.
- **The circle is a route, read clockwise.** The cards sit in the order the assistant actually works through them,
  starting at Router and finishing at Language.
- **The curved lines join the parts that are switched on,** with an arrow showing the direction. A line that dips
  inside the circle is skipping over parts that are off. So with four parts on you see a four-stop route, and with
  everything on you see a full ring.
- **Faded, dashed cards are switched off,** and they have no line into the route.
- **The dashed arc labelled "loops up to 3×"** is the assistant going round again when it uses a tool: it plans,
  acts, then comes back to plan the next step. It lights up when a message actually goes round more than once.
- **While it's thinking,** the part that's working lights up, its line animates, and a timer counts the seconds.

| Card | Its job |
|---|---|
| **Router** | A small trained model. Decides what kind of message this is and which tool it needs |
| **Perception** | Turns your message into a short list of plain facts, plus what you probably want |
| **Attention** | Decides whether the message needs anything ("lol ok" doesn't), and which facts matter |
| **Recall** | Looks up things you told it before, in long-term memory |
| **Planner** | Suggests up to 3 next steps: answer, make the thing you asked for, ask you a question, or use a tool |
| **Predictor** | Guesses how helpful each step would be, and what could go wrong |
| **Critic** | Looks for mistakes, shaky assumptions and missing details |
| **Decision** | Picks one of the steps |
| **Math Language** | A small trained model. Pulls the math problems out of your message |
| **Math** | A small trained model. Does the arithmetic itself, with the calculator checking it |
| **Language** | Writes the reply you read |

### What's on each card

The card itself stays simple: the part's **name**, the **model** doing the job, a small **score pill**, and an
**on/off switch**.

**Click the card** and everything else opens in a panel:

- **on / off,** with a line explaining what switching it off means. [More below.](#turning-parts-on-and-off)
- **The model dropdown:** which model does this job. You can swap models and compare.
  - "(not trained yet)" means the model is still being built, and it can't be picked.
  - "● loaded" means that model is already warmed up on its computer.
- **The score,** like "5/6 · 0.1s" (5 of 6 checks passed, 0.1 seconds on average), **with every check listed** and
  whether it passed. **Green** means all passed, **yellow** at least half, **red** less than half. "not tested" means
  no test has run for this model yet.
- **Test:** runs those checks again with the chosen model. It says "testing…" while it runs, and the panel stays open.
- **Settings** for that part, if it has any. See [Settings](#settings).

Press Escape or click elsewhere to close it.

## Below the circle

- **Tools.** A list under the circle, one row per tool, each with the same on/off switch the parts have. They aren't
  models — they're the things the parts reach for.
  - **Calculator**: exact math;
  - **Web search** and **Web browser**: search the web and read a web page;
  - **Files**: saves and reads text files in the assistant's own folder, and never deletes for good (deleted files go
    to a trash folder);
  - **Long-term memory**: what it remembers about you;
  - **Command line**: greyed out and marked "not built yet".

  A faded row is off. While the assistant is thinking, the tool it's using lights up and counts the seconds, and
  afterwards its row says how long it took, like "ran · 0.9s".
- **Long-term memory · N.** The 20 newest things the assistant remembered about you. Each has a **Forget** button that
  deletes it.

---

## Turning parts on and off

The checkbox on any card, module or tool, switches that part off. **Off means the rest of the assistant acts
as if the part doesn't exist.** It never runs, and no other part hears about it.

The assistant still replies, using simple stand-ins:

| Switched off | What happens instead |
|---|---|
| Perception | Your whole message becomes one fact |
| Attention | Nothing gets dismissed as "not for me" |
| Recall, or the memory card | Nothing is looked up, and nothing new is saved |
| Planner | One step: reply directly |
| Predictor | No guesses about helpfulness |
| Critic | No problems flagged, so fewer reasons to ask you a question |
| Decision | A simple rule picks: the most helpful guess, or the first step that isn't a question |
| Language | The reply is the raw result, like just the number from the calculator |
| Router, Math Language, Math | They're skipped |

Why this is useful: you can see what each part adds. For example, with **only Math** switched on, "12 + 30 =" still gets
"42".

- **Saved:** your on/off choices are kept for next time.
- **Not while thinking:** you can't switch things while a reply is in progress. The page says "Wait for this reply
  first."

## Settings

Click a card and its settings are in the panel that opens, at the bottom. A change saves immediately. Under each
setting, "default" means it's at the normal value; otherwise it shows what the default is.

| Cards | Setting | What it means in plain terms |
|---|---|---|
| Perception, Attention, Recall, Planner, Predictor, Critic, Decision, Language | **Temperature** | How adventurous the writing is. Low (0) is careful and repeatable; higher is more varied |
| Same | **Token limit** | The most this part may write in one go. Too low and it gets cut off |
| Language | **Token limit when making something** | A bigger limit for when it writes the thing you asked for, like code or a list (1800 by default) |
| Router | **Stop on small talk when at least this sure** | It won't reply to "ok" or "lol" if it's at least this sure it's small talk (default 0.9, meaning 90%) |
| Router | **Force a lookup when at least this sure** | If it's this sure a message needs the calculator or a web search, that tool gets used (default 0.7). It can never force saving a file |
| Router | **Skip a question when the chance of needing one is below** | If a question seems this unlikely to be needed, the assistant answers instead of asking (default 0.2) |
| Math | **The calculator checks its answers** | On by default. Turn it off to see the math model's own answers, mistakes included |
| Math Language | **Only read messages that have numbers in them** | On by default. Off means it reads every message |

## Chatting

1. Type in the **Message** box and press **Send**, or ⌘↩ (Command+Return).
2. The example buttons under the box fill in a sample message: The Friday meeting, Percent, Remember, Recall, Save a
   file, Read it back, Not for me.
3. While it thinks, your message shows with "thinking… Planner" (or whichever part is working), and the circle lights up.

- **Replies stay in one chat.** If the assistant asks you something, just answer; it remembers the last few messages.
- **New chat** starts over.
- **Recent chats…** reopens an older one. The page also reopens your last chat when you reload it.
- **One message at a time** per chat.
- **If the page loses its live connection,** the assistant keeps working in the background. Reopen the chat in a bit.
- **"No reply needed."** means it decided your message didn't need an answer, like "lol ok".

## When it looks something up

Some questions no model can answer from memory: the weather, the news, a price, who won last night. Three things make
that work, and they're independent, so a lookup still happens with almost everything switched off:

1. **The Router** reads the message in milliseconds and says which tool it needs. When it's sure, it wins.
2. **A rule in the code** catches what the Router misses. "Whats the weather in 60601" fooled the Router — the bare
   number made it think arithmetic — so a short list of words like *weather*, *news*, *price* and *who won* sends it to
   a search anyway.
3. **It opens the top page.** A search comes back as titles and links, which almost never carry the actual number. So
   for those questions it reads the best page too, and the Language module writes the reply from the page's text.

Without step 3 the small model on the Mac read a page of weather links and answered "72°F and a 20% chance of rain" —
a number that appeared nowhere. With it, the same question comes back with the real National Weather Service figures
in about 13 seconds.

**It will also say so when it can't.** The Language module is told which tools exist and that only the results it was
handed actually happened, so it won't offer to "search that for you" — nothing runs after it. If it needs one more
page it can ask for it, and the harness fetches it and lets it write the reply again, once per message.

## Reading "How it got there"

Every reply has a folded line, **"How it got there · 82s"** (the seconds it took). Click it and the assistant's
reasoning unfolds as a **timeline**: one step per thing that happened, in the order it happened, each with a plain
headline and its own time. The **⤢** button opens the same timeline in a larger window for a proper read.

| Step | What it tells you |
|---|---|
| **Router** | What the small Router model decided in a few thousandths of a second: the kind of message, the tool it thinks is needed, and how sure it was of each |
| **Perception** | The facts it noticed, each with a tag (question, request, fact…) and a bar showing how much it mattered. When Perception is switched off the step says so honestly: "the message went in whole" |
| **Math Language** | The math problems it found, their answers, and who worked them out ("by Math" or "by calculator"). "The calculator caught 1" means the math model got one wrong and the calculator fixed it. The model's written-out working is folded underneath |
| **Recall** | What it searched long-term memory for, and which memories it used |
| **Round 1, Round 2…** | One card per round of thinking. A new round starts after a tool is used (up to 3 tool rounds) |
| **Remembered** | New things it saved about you |
| **All N model calls** | Folded: every model it asked, which model, and how many seconds each took |

Long tool output is shortened to a few lines with a **"Show all 2,591 characters"** link, so a web page doesn't bury
the rest. Only real machine output — tool results and the math working — is in monospace.

Inside a **Round** card:

- **The candidates:** the steps the Planner suggested. The chosen one has a ✓.
- **"predicted helpful 0.8":** the Predictor's guess, from 0 to 1, with the likely outcome and risk.
- **"dropped by guards" or "changed by guards":** built-in rules threw out or fixed a bad step, like a web search for
  "[your industry]".
- **"Critic:"** each problem it found, marked **low**, **medium** or **high**, and the question it would ask you.
- **"Decision (model / rule / router / language): #2 — why":** who made the final pick.
  - **model:** the Decision part chose.
  - **rule:** a built-in rule overruled it. Examples: "it asked a question last turn, so it doesn't ask another", or
    "the message is just arithmetic".
  - **router:** the small Router model was confident enough to overrule.
  - **language:** the part that writes the reply asked for one lookup before it could answer.
- **The tool line:** which tool ran, what it was given, whether it worked, and the result.
- **Math:** for arithmetic, how many steps the math model did itself, how many the calculator caught, and each step.

A tip: if a reply seems off, open "How it got there" and look at the **Critic** and **Decision** lines first. Most
surprises start there.

## The older page

The page was redesigned on 2026-09-15. The one before it is still there at
**http://127.0.0.1:8771/classic**, unchanged, if you want to compare or if something in the new one gets in your way.

## What it can't do yet

- **It asks too many questions.** The Critic still flags details that have sensible defaults.
- **Live facts:** it sometimes asks instead of searching the web for things like the weather.
- **Saving files:** it's less likely to save a file than it used to be.
- **Running programs:** `command_line` isn't built.
- More in [what's next](../whats-next.md).
