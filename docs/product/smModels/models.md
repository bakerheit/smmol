# The models

SMMOL has eight small models, all trained from scratch on a MacBook. Four of them are plugged into the assistant.

**How to read the scores:**
- **"Hand-written test messages"** are questions a person wrote on purpose, phrased differently from what the model
  practiced on. They're the fairest test.
- **"The big model"** means Ministral 8B, a freely available AI model with about 8 billion parameters, running on the
  home PC. It's what most parts of the assistant use today, so it's the main thing to beat. For size: the SMMOL models
  with a recorded size have between 1.9 and 10.8 million parameters.

All numbers are from September 14–15, 2026. Words like "parameter" are explained in the [glossary](../glossary.md).

## Summary

| Model | Nickname | In the assistant? | How good right now |
|---|---|---|---|
| smROUTER_01 | The message sorter | Yes | Fully right on 71 of 100 test messages. Beats keyword rules (63) and the big model (58), and is far faster |
| smMATH_LANGUAGE_001 | The math reader | Yes | Every problem right on 78 of 100. The big model gets 82, but takes about 100 times as long |
| smMATH01 | Math in its head | Yes | Near-perfect adding and subtracting up to 6 digits. Can't really multiply |
| smMATH001-a | Shows its work | Yes, as a Math choice | Multiplies well at the sizes it practiced (97%). Divides poorly (38%). Fails on bigger numbers |
| smLLM_01 | The Shakespeare writer | No | Looks like Shakespeare, means nothing |
| smALLM_01 | Learns by poking | No | Figures out hidden wiring after about 8 experiments |
| smTOOLS_COMPUTER_CLI_01 | The terminal translator | Not yet | Writes the exactly right command 47 times in 100, against the big model's 15. Picks the right tool less often (64 vs 79) |
| smCLM_01 | Ideas, not words | No | Guesses 82% of a word's meaning from how it's used |

---

## smROUTER_01: the message sorter

**What it does.** It reads each message first, and answers three questions in about 2 thousandths of a second:

- What kind of message is this: question, task, "remember this", small talk?
- Does it need a tool: the calculator, a web search, a saved file, memory?
- Should the assistant ask you something first?

The assistant uses its answers to skip replying to "lol ok", to use the calculator or web search when it's sure, and to
avoid needless questions.

**How good is it right now?**
- **All three answers right** on 71% of 93 hand-written test messages. Simple keyword rules get 63%. The big model gets
  58% and takes about 2 seconds per message.
- **Where it slips:**
  - news and prices without an obvious clue word ("how did the Warriors do last night?");
  - plain knowledge questions (it wants to web-search "who painted the Mona Lisa");
  - short reactions like "haha that's great";
  - answers to the assistant's own questions.
- **It thinks "write a function" means "save a file."** So the assistant has a firm rule: the sorter can make it look
  things up, but it can never make it save or change files.

## smMATH_LANGUAGE_001: the math reader

**What it does.** It reads a message like "what's 15% of 80 and 20% of 60?" and writes out the math problems in it,
without solving them. It keeps units ("cubic feet") and can chain steps ("...and how many cubic yards is that?"). The
arithmetic model or the calculator then does the actual math.

**How good is it right now?** Finished training on September 15, 2026. On 40 hand-written test messages:

| | Math reader | Big model | Simple pattern matching |
|---|---|---|---|
| Got every problem right | 78% | 82% | 30% |
| Got the final answer right | 80% | 95% | 32% |
| Stayed quiet when there was no math | 78% | 89% | 67% |
| Time per message | 0.03 seconds | 3 seconds | |

- **Close to the big model** on getting everything right, and about 100 times faster.
- **Clearly behind** on final answers and on ignoring numbers that aren't math.
- **Mistakes to know about:**
  - it drops a digit from long numbers ("90000-1" became "9000-1");
  - it turned a speed problem around;
  - it read a phone number as a subtraction;
  - it got "the weather in Denver tomorrow" as a math problem.
- **Too used to its practice examples:** it gets 99% right on those.

## smMATH01: math in its head

**What it does.** It adds, subtracts and multiplies whole numbers using only what it learned, with no calculator.
There are three versions that differ in how they read digits. Inside the assistant, the calculator checks every answer
it gives, so a wrong answer never reaches you while that check is on.

**How good is it right now?** It trained on numbers up to 6 digits long.

- **Adding and subtracting up to 6 digits:** the best version ("abacus") is right 99–100% of the time.
- **Longer numbers:** the abacus version still adds 7- and 8-digit numbers perfectly, then fades: 89% at 9 digits, 40%
  at 10. The other two versions drop to 0% at 7 digits.
- **Multiplying:** it fails. At most 55% right on 2-digit numbers, and almost never right from 3 digits up.

## smMATH001-a: shows its work

**What it does.** It takes a whole math problem, like "322234 × 21323 × 212231", and writes out the steps the way you'd
do it on paper: long multiplication, long division, then the answer. It's meant to handle the multiplying and dividing
that smMATH01 can't. It works to 2 decimal places.

**How good is it right now?** Finished training on September 15, 2026. It's scored on whether its whole written-out
working is exactly right:

- **At the sizes it practiced:** adding 100%, subtracting 99%, multiplying 97%, mixed problems 84%, the kind of
  problems the math reader writes 85%, decimals 82%.
- **Dividing:** only 38%. That's its weak spot.
- **Numbers bigger than it practiced on:** it mostly fails. 7% on adding 8- to 10-digit numbers, 0% on subtracting
  them, 21% when multiplying by a 4-digit number, and 0% on anything bigger. The "322234 × 21323 × 212231" volume
  problem came out wrong: it copied the numbers down shorter before doing any math.
- **Compared with smMATH01:** writing out the work is what made multiplication work (97%, where smMATH01 almost never
  got it right). But smMATH01's best version still copes with longer additions far better. The tests weren't
  identical, so treat the comparison as rough.
- **In the assistant,** the calculator checks its answers while that check is on, so its mistakes get caught. In a
  test, it split a $212.40 bill five ways as 39.99; the calculator caught it and used 42.48.

## smTOOLS_COMPUTER_CLI_01: the terminal translator

**What it does.** You say what you want in plain English — "what's using port 3000", "put ripgrep on here", "how much
room is left on the disk" — and it writes the command you'd type, for the computer you're actually on. The same
request needs a different command on Ubuntu, Fedora, Arch, a Mac and Windows; for 70 of its 79 jobs, Ubuntu and
Windows need completely different commands. It also labels how risky the command is: just looking, changing
something, installing, needing admin, or destructive.

**It never runs anything.** Nothing in this project can execute a command. It writes the text, and a person decides.
The assistant's command-line tool stays switched off and unbuilt until it can ask permission for every single command.

**How good is it right now?** Finished training on September 15, 2026, and scored on 53 requests written by hand,
worded differently from anything it practiced on.

| | It | A simple lookup, no model | The big model |
|---|---|---|---|
| Wrote the command exactly right | **47 in 100** | 40 | 15 |
| Reached for the right tool | 64 | 60 | **79** |
| Labelled the risk right | **81** | – | 77 |
| Said nothing when there was nothing to run | **100** | 33 | 50 |
| Time per request | **0.05 s** | – | about 2 s |

- **What it's good at:** getting the whole command character-perfect three times as often as the big model, forty
  times faster, and knowing when a message isn't a terminal job at all.
- **What it isn't:** it picks the wrong tool more often than the big model does. Knowing the right program for a
  phrasing it has never seen is where a model this small runs out of room.
- **The mistake to know about:** four times it got the right program and did the opposite thing — "put ripgrep on
  here" came out as *remove* ripgrep, and "brew me ffmpeg" as *uninstall* ffmpeg. This is exactly why a person reads
  the command before anything runs it.
- **It's also honest about danger in the direction that counts:** it never invented a destructive command for a
  request that wasn't one. And a separate set of rules, written in ordinary code, checks every command for known
  dangers and overrules whatever the model claimed.
- **It's being retrained.** It peaked partway through and then got slightly worse, and the training script was
  keeping the last version rather than the best one. That's fixed, and the next run should land a bit higher.

## smLLM_01: the Shakespeare writer

**What it does.** The first model in the project, built for fun. It read Shakespeare's plays one letter at a time and
learned to write text that looks like them.

**How good is it right now?** It writes the right format, real words and the rhythm of a play, but the lines don't
make sense. Its score matched a well-known reference result for the same setup. It isn't used in the assistant.

## smALLM_01: learns by poking

**What it does.** It learns cause and effect by experimenting, like a kid pressing buttons. Each time, it gets a new
toy world of 8 switches and lights with hidden wiring. It flips things, guesses what will happen, and learns from its
misses.

**How good is it right now?**
- **With no experiments,** it guesses "nothing else changes", which is right 61% of the time.
- **After 1 to 4 experiments,** it's actually worse than that. It jumps to conclusions.
- **From 8 experiments on,** it beats the lazy guess, reaching 83% after 20.
- **Choosing its own experiments** helps it spot effects, but barely improves getting everything right.

It isn't used in the assistant.

## smCLM_01: ideas, not words

**What it does.** It learns what a word means as a bundle of ideas. "Car" means transportation, vehicle, people and so
on, worked out only from sentences that use the word. The test: 23 words whose meanings it was never told.

**How good is it right now?**
- **82% of those words' ideas** land in its top guesses.
- **New words:** it can give sensible ideas to a made-up word from 3 or 4 sentences.
- **No clear win:** a normal word-based model did just as well when read the right way, so thinking in ideas didn't
  beat thinking in words.
- **Hidden ideas:** it's weak at ideas no sentence shows directly, finding 56% of them.
- **Not real language:** its little world was hand-made, so real language is untested.

It isn't used in the assistant yet.
