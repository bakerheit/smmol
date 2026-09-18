# Glossary

Plain meanings of the words the product pages can't avoid. Listed A to Z.

**Accuracy, or "% right".** Out of a set of test questions, how many the model got fully right. "71% right" means 71 of
every 100.

**Baseline.** Something simple to compare a new model against, so a score means something. SMMOL uses three:
hand-written keyword rules, a regular pattern-matching program with no AI, and a bigger AI model (Ministral 8B) given
instructions.

**Browse server.** A small program on the home PC that does web searches and reads web pages for the assistant. It also
blocks addresses inside the home network.

**Calculator check.** When the arithmetic model does a sum, the calculator does it too. If they disagree, the
calculator's answer is used and the mistake is counted.

**Checkpoint.** A saved copy of a model at one moment in training, stored as a file. Training saves new checkpoints as
it goes, and the assistant always uses the latest one, even if training isn't finished.

**Cloud model.** An AI model that runs in a company's data center, reached over the internet. SMMOL's models are the
opposite: local.

**Contract.** The exact shape a part's answer must have, like a form with fixed boxes and length limits. If a part's
answer doesn't fit, it gets one more try, and then the reply stops with an error.

**Embedding model.** A model that turns a piece of text into a list of numbers that captures its meaning, so texts that
mean similar things get similar numbers. Vector memory would need one.

**Generated data.** Practice examples made by a program instead of written by people. It's quick to make hundreds of
thousands of them, but they can all sound alike, so models can get too used to them.

**GPU (graphics card).** The chip that makes AI training and answering fast. A MacBook Pro M5's GPU trains
SMMOL's models. The
home PC's graphics card (an RX 580) runs the bigger model, and the family's other AI tools share it.

**Hand-written test messages.** Test questions a person wrote on purpose, phrased differently from the practice
examples. They show how a model does on real-sounding messages.

**Harness.** The program that connects all the parts into one assistant and shows them on a web page. The current one is
paratroop_harness_02.

**Held-out.** Kept away from the model during training, so a test isn't just memory.

**Keyword rules.** A simple non-AI approach: "if the message contains 'weather', use web search." Used as a baseline.

**Language model (LLM).** An AI model that reads and writes text by predicting the next piece of text, over and over.
ChatGPT-style assistants are large language models.

**Local.** Running on computers in the house, not over the internet.

**Long-term memory.** Facts the assistant saves about you between chats, like "your dentist is Dr. Lee". It's a plain
file, not a model. You can see and delete memories on the page.

**Loss.** The number a model tries to shrink during training. It measures how wrong its guesses are. Lower is better.

**Ministral 8B.** A mid-sized, freely available language model that runs on the home PC. "8B" means about 8 billion
parameters. Most parts of the assistant use it today, and it's the main baseline.

**Model.** A program that learned its behavior from examples instead of being written rule by rule.

**Module.** One part of the assistant with one job, shown as a card on the page. Examples: the Router sorts messages,
and the Critic looks for problems. Some modules are small trained models; others are the bigger model given
instructions.

**Overfitting.** When a model gets very good at its practice examples but worse at real ones, like a student who
memorizes the practice test.

**Parameter.** One of the adjustable numbers inside a model. Training tunes them. More parameters means a bigger model:
SMMOL's router has about 1.9 million, and Ministral 8B has about 8 billion.

**Prompt, or prompted model.** Giving a general-purpose model written instructions for a job instead of training a model
for it. Most of the assistant's parts are prompted today.

**Router.** The part that reads a message first and decides where it should go: small talk, a question, a math problem,
a web search, and so on.

**Small talk.** Messages that need nothing back, like "ok", "lol" or "thanks".

**Temperature.** A setting on text-writing parts. Low means careful and predictable; high means more varied and more
likely to wander.

**Token.** The small chunk of text a model reads or writes at a time. For SMMOL's own models it's usually a single
character or digit.

**Token limit.** The most a part is allowed to write in one go. It keeps answers short and stops a part from rambling.

**Tool.** Something the assistant can use that isn't a model: the calculator, web search, a web page reader, the file
saver, and long-term memory.

**Training.** Showing a model many examples and adjusting its parameters a little each time, so it gets better at its
job.

**Vector memory.** A kind of memory search that finds things by meaning instead of exact words. Not built yet.

**Weights.** Another word for a model's parameters. "Does math in its weights" means the model worked the answer out
itself, without a calculator.

**Working memory.** The assistant's scratchpad for one message: what it noticed, what it remembered, the options it
considered, and what it decided. It's plain data, not a model, and you can read it under "how it got there".
