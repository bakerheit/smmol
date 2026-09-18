"""Made-up but varied messages with router labels.

Each example is: prev (the assistant's last message, often empty), text (what the person sent), intent, tool and
ask_first. Names, numbers and phrasings are mixed and roughed up (typos, lower case, missing punctuation) so the
router can't just memorize the templates. The test set in test.json is written separately and never generated here.
"""
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "labels.json")) as f:
    LABELS = json.load(f)

SLOTS = {
    "city": ["Chicago", "Denver", "Tokyo", "Paris", "Austin", "Seattle", "Lagos", "Mumbai", "Toronto", "Berlin", "Sydney",
             "Boston", "Phoenix", "Nashville", "Dublin", "Madrid", "Portland", "Atlanta", "Oslo", "Lima", "Cleveland", "Tampa"],
    "day": ["today", "tomorrow", "this weekend", "tonight", "on Saturday", "this afternoon", "next week"],
    "topic": ["the election", "electric cars", "the stock market", "space launches", "the NBA playoffs", "AI chips",
              "the housing market", "hurricane season", "the World Cup", "interest rates", "the new iPhone", "the climate summit"],
    "team": ["the Cubs", "the Lakers", "the Packers", "Arsenal", "the Yankees", "the Bears", "Real Madrid", "the Celtics"],
    "store": ["Target", "Costco", "the DMV", "Home Depot", "the library", "Trader Joe's", "the post office", "Walgreens"],
    "product": ["a PS5", "an iPad", "a Tesla Model 3", "a Dyson vacuum", "a Nintendo Switch", "AirPods", "a KitchenAid mixer"],
    "stock": ["Apple", "Nvidia", "Tesla", "Microsoft", "Amazon", "bitcoin", "gold"],
    "movie": ["Dune", "the new Pixar movie", "the Barbie movie", "the new Marvel movie", "Oppenheimer"],
    "person": ["Sam", "Mia", "Jordan", "Priya", "Luis", "Grandma", "Aunt Rose", "Coach Dan", "Dr. Patel", "Ms. Kim", "Ben"],
    "relation": ["dentist", "doctor", "vet", "landlord", "boss", "son's teacher", "daughter's coach", "mechanic",
                 "accountant", "babysitter", "plumber"],
    "attr": ["wifi password", "blood type", "shoe size", "license plate", "locker combination", "favorite color",
             "anniversary", "insurance company", "gym schedule", "parking spot"],
    "event": ["dentist appointment", "flight", "parent-teacher conference", "job interview", "soccer game", "doctor's visit",
              "team offsite", "haircut", "car service", "birthday dinner"],
    "when": ["tomorrow", "this Friday", "next Tuesday", "on Saturday", "at 3pm", "tonight", "next week", "on the 12th",
             "Monday morning", "after lunch"],
    "list": ["shopping list", "packing list", "to-do list", "grocery list", "reading list", "gift list", "chore list"],
    "item": ["eggs", "milk", "bread", "bananas", "coffee", "batteries", "sunscreen", "socks", "dish soap", "rice",
             "apples", "tape", "a phone charger", "paper towels", "cat food", "onions", "yogurt"],
    "note": ["the garage code is 4417", "the plumber comes Thursday", "Mia likes the blue one", "we owe Sam twenty bucks",
             "the recipe needs two cups of flour", "the car is due for an oil change", "Ben's recital is at 6"],
    "file": ["notes", "groceries", "trip-plan", "ideas", "budget", "recipes", "errands"],
    "feeling": ["tired", "stressed", "excited", "nervous", "burned out", "happy", "a little sick", "bored"],
    "concept": ["compound interest", "photosynthesis", "how vaccines work", "black holes", "inflation", "how a mortgage works",
                "machine learning", "the electoral college", "how batteries work"],
    "activity": ["a job interview", "running a 5k", "a first date", "public speaking", "learning guitar", "saving money",
                 "sleeping better", "potty training"],
    "writing": ["a poem about autumn", "a thank-you note to my neighbor", "an email to my boss asking for Friday off",
                "a toast for my brother's wedding", "a short story about a dragon", "a birthday message for my mom",
                "a cover letter for a barista job"],
    "plan": ["a birthday party for my son", "a weekend in Chicago", "meals for the week", "a baby shower",
             "a road trip to the coast", "a garage sale"],
    "thing": ["dinner", "a date night", "a science project", "a team name", "a Halloween costume", "rainy day activities"],
    "unit": ["5 miles in kilometers", "70 degrees Fahrenheit in Celsius", "3 cups in milliliters", "150 pounds in kilograms",
             "12 inches in centimeters", "2 hours in seconds"],
    "time": ["7pm", "noon", "6:30", "9am", "8:15", "half past five"],
    "domain": ["example.com", "nytimes.com", "bbc.co.uk", "en.wikipedia.org", "theverge.com", "medium.com", "github.com",
               "bonappetit.com", "news.ycombinator.com", "consumerreports.org"],
    "word": ["best", "guide", "review", "2026", "deals", "how", "to", "budget", "family", "recipe", "news", "update"],
}


def pick_items(r):
    picks = r.sample(SLOTS["item"], r.randint(2, 4))
    return ", ".join(picks[:-1]) + " and " + picks[-1]


def url(r):
    domain = r.choice(SLOTS["domain"])
    prefix = r.choice(["", "www."]) if domain.count(".") == 1 else ""
    path = "-".join(r.choice(SLOTS["word"]) for _ in range(r.randint(2, 4)))
    return "https://%s%s/%s" % (prefix, domain, path)


def number(r):
    n = r.choice([r.randint(12, 99), r.randint(100, 999), r.randint(1000, 9999)])
    return "{:,}".format(n) if n >= 1000 and r.random() < 0.5 else str(n)


def money(r):
    return "$%d%s" % (r.randint(12, 480), r.choice(["", ".50", ".99", ".25"]))


def expression(r):
    """Bare arithmetic the way people type it into a chat box: 2+2, 48,213 - 9977, (12 + 8) * 3, 3.5 x 4."""
    def operand():
        n = r.choice([r.randint(0, 9), r.randint(10, 99), r.randint(100, 9999), r.randint(10000, 99999999)])
        if r.random() < 0.15:
            return "%d.%d" % (n, r.randint(1, 99))
        return "{:,}".format(n) if n >= 1000 and r.random() < 0.3 else str(n)
    ops = r.choice([["+"], ["-"], ["*"], ["x"], ["/"], ["+", "-"], ["*", "+"]])
    parts = [operand()]
    for _ in range(r.choice([1, 1, 1, 2])):
        parts += [r.choice(ops), operand()]
    if len(parts) == 5 and r.random() < 0.3:
        parts = ["("] + parts[:3] + [")"] + parts[3:]
    return (" " if r.random() < 0.5 else "").join(parts).replace("( ", "(").replace(" )", ")")


FILL = {name: (lambda r, values=values: r.choice(values)) for name, values in SLOTS.items()}
FILL.update(items=pick_items, url=url, num=number, money=money, expr=expression, n=lambda r: str(r.randint(2, 9)),
            a=lambda r: str(r.randint(2, 99)), b=lambda r: str(r.randint(2, 99)), c=lambda r: str(r.randint(2, 99)),
            pct=lambda r: r.choice(["15", "17.5", "20", "8", "12.5", "33", "7.25", "18"]), code=lambda r: str(r.randint(1000, 9999)))

# (intent, tool, ask_first, weight, phrasings)
CLASSES = [
    ("question", "calculator", False, 1.0, [
        "what's {pct}% of {num}", "what is {pct} percent of {num}?", "calculate {a} times {b}", "how much is {a} x {b}",
        "{a} * {b} = ?", "what's {num} divided by {n}", "whats {a} plus {b} plus {c}", "if I save {money} a week how much is that in a year",
        "what's the tip on {money} at {pct}%", "split {money} between {n} people", "convert {unit}", "how many seconds are in {n} days",
        "what's {num} minus {a}", "if a jacket is {money} and it's {pct}% off what do I pay", "how much is {pct}% tax on {money}",
        "what's {a} squared", "square root of {num}", "average of {a}, {b} and {c}", "add up {a}, {b} and {c} for me",
        "how much would {n} tickets at {money} each cost"]),
    ("question", "calculator", False, 0.8, [  # nothing but the arithmetic
        "{expr}", "{expr} =", "{expr}=", "{expr}?", "{expr} = ?", "what's {expr}", "whats {expr}", "calc {expr}", "{expr} equals"]),
    ("question", "calculator", True, 0.25, [
        "split the bill evenly", "what's the tip?", "how much do we each owe", "calculate the total for me", "what's 20% of it?",
        "how much is that with tax"]),
    ("question", "web_search", False, 1.2, [
        "what's the weather in {city} {day}", "is it going to rain in {city} tomorrow", "weather forecast for {city}",
        "latest news about {topic}", "what's going on with {topic}", "did {team} win last night", "what was the score of the {team} game",
        "how much does {product} cost right now", "what's {stock} trading at", "current price of {stock}",
        "what time does {store} close today", "is {store} open on sunday", "when does {movie} come out", "is {movie} still in theaters",
        "what time is it in {city}", "look up reviews for {product}", "find me a good pizza place in {city}",
        "search the web for {topic}", "any traffic on the way to {city} right now", "who's playing in {city} this weekend",
        "are flights to {city} cheap this month", "what's happening with {topic} today"]),
    ("question", "web_search", True, 0.3, [
        "what's the weather like today?", "is it going to rain later", "when does it close?", "how much does it cost right now?",
        "is the game on tonight?", "what's the score?"]),
    ("task", "web_browser", False, 0.3, [
        "summarize {url}", "can you read {url} and give me the main points", "tl;dr {url}", "pull the key points out of {url}"]),
    ("question", "web_browser", False, 0.3, [
        "what does this article say {url}", "{url} what's this about?", "is the price still the same on {url}",
        "what's the recipe on {url}"]),
    ("task", "file_write", False, 1.0, [
        "save a {list} with {items}", "add {items} to my {list}", "put {item} on the {list}", "write down that {note}",
        "make a note that {note}", "create a file called {file} with {items}", "start a {list}: {items}",
        "update my {list} with {item}", "jot this down: {note}", "save this to {file}: {note}", "remove {item} from my {list}",
        "take {item} off the {list}", "add a line to {file}: {note}"]),
    ("task", "file_write", True, 0.3, [
        "save that", "write it down", "make a list", "add it to my list", "put that in a file", "note that down for me"]),
    ("question", "file_read", False, 0.8, [
        "what's on my {list}", "read me my {list}", "show me the {list}", "what did I put on the {list}", "open {file}",
        "what's in {file}", "do I have {item} on my {list}?", "what files do I have", "list my notes",
        "what did I write in {file}", "is {item} on the {list} already"]),
    ("question", "memory_recall", False, 1.0, [
        "who's my {relation}", "what's my {attr}", "when is my {event}", "what did I tell you about {person}",
        "do you remember my {attr}", "when's {person}'s birthday", "remind me what my {attr} is", "where did I say I parked",
        "what am I allergic to", "what time is my {event}", "did I tell you where my {event} is?",
        "what's the name of my {relation}", "what did I say {person} likes"]),
    ("question", "memory_recall", True, 0.2, [
        "what did I say about it?", "when is that again?", "what was the name again?"]),
    ("remember", "none", False, 0.8, [
        "remember that my {relation} is {person}", "don't forget {person}'s birthday is {when}",
        "keep in mind I'm allergic to {item}", "my {attr} is {code}, remember that", "please remember I park in spot {n}",
        "I want you to remember that {note}", "remember: {note}", "remember my {event} is {when}"]),
    ("statement", "none", False, 1.0, [
        "I have a {event} {when}", "{person} is visiting {when}", "I'm feeling {feeling} today", "we're thinking about moving to {city}",
        "I just got back from {city}", "my {event} got moved to {when}", "work has been crazy this week",
        "I have a meeting {when} to discuss {topic}", "we adopted a dog", "I start a new job {when}", "my {relation} quit",
        "{person} and I are going to {city} {when}"]),
    ("question", "none", False, 1.2, [
        "why is the sky blue", "how do I boil an egg", "can you explain {concept}", "what is {concept}", "how does {concept} work",
        "what are some tips for {activity}", "what should I name my new puppy", "is it bad to drink coffee at night",
        "what's a good gift for {person}", "how long should I steep green tea", "what rhymes with orange", "who wrote Hamlet",
        "what's the capital of Australia", "how do I get a stain out of a shirt"]),
    ("task", "none", False, 1.2, [
        "write {writing}", "help me plan {plan}", "draft {writing}", "give me ideas for {thing}", "make me a workout plan",
        "come up with a name for {thing}", "rewrite this to sound nicer: {note}", "translate good morning into Spanish",
        "tell me a joke", "brainstorm ideas for {thing}", "plan a 3 day trip to {city}", "make a bedtime story about a robot"]),
    ("task", "none", True, 0.6, [
        "book a table for {when}", "remind me later", "send the email", "call him back", "order the usual", "buy tickets for the show",
        "schedule the meeting", "set that up for {when}", "reply to her", "cancel it", "go ahead and do it"]),
    ("small_talk", "none", False, 0.8, [
        "ok", "okay", "k", "lol", "lmao", "haha", "hahaha nice", "thanks", "thank you!", "thx", "ty", "cool", "nice", "great",
        "got it", "sounds good", "np", "nvm", "bye", "cool cool", "alright", "ok thanks", "perfect", "awesome", "👍", "love it"]),
]

# (the assistant's question, the person's answers, the tool the answer leads to)
ANSWERS = [
    ("Which city should I check the weather for?", ["{city}", "in {city}", "{city} please"], "web_search"),
    ("What should I call the file?", ["call it {file}", "{file}", "name it {file}"], "file_write"),
    ("What should go on the list?", ["{items}", "just {item}"], "file_write"),
    ("Want me to save that to a file?", ["yes", "yes please", "sure", "yeah do it"], "file_write"),
    ("Should I look that up online?", ["yes", "please do", "sure", "yeah"], "web_search"),
    ("How much was the bill?", ["{money}", "it was {money}"], "calculator"),
    ("How many people are splitting it?", ["{n}", "{n} of us"], "calculator"),
    ("Are you pitching, or is someone pitching to you?", ["I'm the investor", "they're pitching to me", "I'm pitching"], "none"),
    ("What time works for you?", ["{time}", "around {time}", "any time after {time}"], "none"),
    ("Which restaurant do you want?", ["the Thai place", "somewhere near {city}", "the usual spot"], "none"),
    ("Do you want tips for preparing, or just a reminder?", ["tips please", "just a reminder", "both"], "none"),
    ("Who is the email for?", ["my {relation}", "{person}"], "none"),
    ("Which {relation} do you mean?", ["the new one", "{person}"], "memory_recall"),
    ("Should I remember that for next time?", ["yes", "yes please", "no thanks"], "none"),
    ("How many days is the trip?", ["{n} days", "{n}"], "none"),
]

ASSISTANT_SAID = ["Done! I saved your {list}.", "The forecast for {city} is sunny and 72°F.", "It comes to {money}.",
                  "Your {relation} is {person}.", "Here are a few ideas for dinner.", "I added {item} to your {list}.",
                  "Got it, I'll remember that.", "Here's a short poem about autumn."]
UNRELATED_QUESTIONS = ["Which city should I check the weather for?", "What should I call the file?", "Want me to save that?",
                       "How many people are coming?", "What time works for you?"]
GREETINGS = ["hey ", "hi ", "ok so ", "um ", "yo ", "hey there, "]
ENDINGS = [" please", " thanks", " pls", "!!", " asap", " thx"]


def render(r, template):
    values = {}

    def sub(m):
        key = m.group(1)
        if key not in values:
            values[key] = FILL[key](r)
        return values[key]
    return re.sub(r"\{(\w+)\}", sub, template)


def roughen(r, text, intent):
    if intent not in ("small_talk", "answer"):
        if r.random() < 0.1:
            text = r.choice(GREETINGS) + text
        if r.random() < 0.08:
            text = text.rstrip("?.!") + r.choice(ENDINGS)
    if r.random() < 0.07 and len(text) > 6:
        i = r.randrange(1, len(text) - 2)
        text = text[:i] + text[i + 1] + text[i] + text[i + 2:] if r.random() < 0.5 else text[:i] + text[i + 1:]
    if r.random() < 0.3:
        text = text.rstrip("?.!")
    if r.random() < 0.4:
        text = text[:1].upper() + text[1:]
    elif r.random() < 0.2:
        text = text.lower()
    return text


def example(r):
    if r.random() < 0.12:
        question, answers, tool = r.choice(ANSWERS)
        return {"prev": render(r, question), "text": roughen(r, render(r, r.choice(answers)), "answer"),
                "intent": "answer", "tool": tool, "ask_first": False}
    intent, tool, ask, _, phrasings = r.choices(CLASSES, weights=[c[3] for c in CLASSES])[0]
    prev, roll = "", r.random()
    if roll < 0.2:
        prev = render(r, r.choice(ASSISTANT_SAID))
    elif roll < 0.3 and tool != "none" and not ask:
        prev = r.choice(UNRELATED_QUESTIONS)  # a brand-new request after a question isn't an answer
    return {"prev": prev, "text": roughen(r, render(r, r.choice(phrasings)), intent),
            "intent": intent, "tool": tool, "ask_first": ask}


def generate(n, seed=0):
    r = random.Random(seed)
    return [example(r) for _ in range(n)]


def normal(text):
    return " ".join(text.lower().strip().rstrip("?.!").split())


def without(examples, held):
    """Drop generated messages that match a held-out message word for word, so the test set stays unseen."""
    banned = {normal(e["text"]) for e in held}
    return [e for e in examples if normal(e["text"]) not in banned]


if __name__ == "__main__":
    for ex in generate(12, 3):
        print(json.dumps(ex))
