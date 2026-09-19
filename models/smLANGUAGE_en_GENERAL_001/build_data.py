"""Build deterministic, original short-conversation training data."""

from collections import Counter
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCHOOL_CORPUS = HERE.parent.parent / "archive" / "models" / "smLANGUAGE_en_SCH_001" / "data" / "curriculum.jsonl"
OUTPUT = HERE / "data" / "conversations.jsonl"
MANIFEST = HERE / "data" / "manifest.json"


def main():
    examples = []
    counts = Counter()

    def add(category, user, assistant, known=None):
        counts[category] += 1
        item = {
            "id": "%s-%04d" % (category, counts[category]),
            "category": category,
            "user": user,
            "assistant": assistant,
        }
        if known:
            item["known"] = known
        examples.append(item)

    with SCHOOL_CORPUS.open() as handle:
        for line in handle:
            item = json.loads(line)
            add("school_sentence", "Use the word %s in a sentence." % item["word"], item["sentence"])

    facts = [
        ("What color is the sky on a clear day?", "What color does a clear daytime sky look?", "A clear daytime sky looks blue."),
        ("How many days are in a week?", "How long is one week in days?", "There are seven days in a week."),
        ("How many months are in a year?", "How long is one year in months?", "There are twelve months in a year."),
        ("What do bees make?", "What sweet food comes from bees?", "Bees make honey."),
        ("Where do fish live?", "What is a fish's usual home?", "Fish live in water."),
        ("What do plants need to grow?", "What helps a plant grow?", "Plants need water, light, air, and nutrients to grow."),
        ("What does a thermometer measure?", "What can a thermometer tell us?", "A thermometer measures temperature."),
        ("What is frozen water called?", "What do we call water after it freezes?", "Frozen water is called ice."),
        ("What happens when ice gets warm?", "What does warm ice turn into?", "Ice melts into liquid water when it gets warm."),
        ("Why do people wear coats in winter?", "How does a winter coat help?", "A coat helps keep a person warm in cold weather."),
        ("What do we use an umbrella for?", "How can an umbrella help in rain?", "An umbrella helps keep rain off you."),
        ("What do birds use wings for?", "How do wings help a bird?", "Birds use their wings to fly and balance."),
        ("What does a clock show?", "What can you learn from a clock?", "A clock shows the time."),
        ("What does a map show?", "How does a map help someone?", "A map shows places and helps people find their way."),
        ("What is a baby dog called?", "What do you call a young dog?", "A baby dog is called a puppy."),
        ("What is a baby cat called?", "What do you call a young cat?", "A baby cat is called a kitten."),
        ("What shape has three sides?", "Which shape has exactly three sides?", "A triangle has three sides."),
        ("What shape has four equal sides?", "Which shape has four sides of equal length?", "A square has four equal sides."),
        ("Which planet do people live on?", "What is the name of our home planet?", "People live on Earth."),
        ("What gives Earth light during the day?", "What shines on Earth in daytime?", "The Sun gives Earth light during the day."),
        ("What do roots do for a plant?", "How do roots help a plant?", "Roots hold a plant in place and take in water."),
        ("Why do we wash our hands?", "How does handwashing help?", "Washing our hands removes dirt and helps stop germs from spreading."),
        ("What should you do before crossing a street?", "How should you check a street before crossing?", "Stop and look both ways before crossing a street."),
        ("What does a librarian do?", "Who helps people find books in a library?", "A librarian helps people find and borrow books."),
        ("What does a firefighter do?", "How does a firefighter help people?", "A firefighter puts out fires and helps keep people safe."),
        ("What does a doctor do?", "How does a doctor help?", "A doctor helps care for people who are sick or hurt."),
        ("What does a farmer grow?", "What can farmers produce on a farm?", "Farmers can grow crops and care for animals."),
        ("What can you use to cut paper?", "Which tool is used for cutting paper?", "You can use scissors to cut paper."),
        ("What can you use to write on paper?", "Name a tool used for writing on paper.", "You can write on paper with a pencil or pen."),
        ("What does a refrigerator do?", "Why is food kept in a refrigerator?", "A refrigerator keeps food cold."),
        ("Why do we sleep?", "How does sleep help the body?", "Sleep gives the body and brain time to rest."),
        ("What is rain?", "Where does rain come from?", "Rain is water that falls from clouds."),
        ("What is snow?", "What falls as snow in cold weather?", "Snow is made of ice crystals that fall from clouds."),
        ("What makes a shadow?", "How is a shadow formed?", "A shadow forms when an object blocks light."),
        ("What is a sentence?", "How can you describe a sentence?", "A sentence is a group of words that expresses a complete thought."),
        ("What is a question?", "What does a question do?", "A question asks for information."),
        ("What is an answer?", "What does an answer provide?", "An answer responds to a question."),
        ("What does fragile mean?", "How should you treat something fragile?", "Fragile means easily broken, so handle it gently."),
        ("What does enormous mean?", "Is something enormous big or small?", "Enormous means very large."),
        ("What does ancient mean?", "How old is something described as ancient?", "Ancient means very old."),
    ]
    for first, second, answer in facts:
        add("fact", first, answer)
        add("fact", second, answer)

    social = [
        ("Hello!", "Hello! How can I help?"),
        ("Good morning.", "Good morning!"),
        ("Good night.", "Good night! Sleep well."),
        ("How are you?", "I'm ready to help. How are you?"),
        ("Thank you.", "You're welcome!"),
        ("Thanks for helping me.", "You're welcome. I'm glad I could help."),
        ("I'm sorry.", "Thank you for saying that."),
        ("I made a mistake.", "That's okay. Mistakes can help us learn."),
        ("I feel sad.", "I'm sorry you're feeling sad. Do you want to talk about it?"),
        ("I feel nervous.", "It's okay to feel nervous. Taking a slow breath may help."),
        ("I am excited!", "That sounds exciting!"),
        ("I did it!", "Nice work!"),
        ("Can you help me?", "Yes. Tell me what you need help with."),
        ("I do not understand.", "That's okay. Tell me which part is confusing."),
        ("Please say that again.", "Sure. What would you like me to repeat?"),
        ("Can you make that shorter?", "Yes. Send me the text you want shortened."),
        ("Can you explain it simply?", "Yes. Tell me what you want explained."),
        ("What should we talk about?", "We can talk about something you enjoy or a question you have."),
        ("Tell me a joke.", "Why did the pencil cross the page? To get to the other side!"),
        ("Goodbye.", "Goodbye!"),
        ("See you later.", "See you later!"),
        ("Are you listening?", "Yes. I'm listening."),
        ("That answer was wrong.", "Thanks for telling me. Let's correct it."),
        ("Stay on topic.", "Okay. I will answer the question directly."),
        ("Answer in one sentence.", "Okay. I will use one sentence."),
        ("Do not guess.", "Okay. If I do not know, I will say so."),
        ("Be clear.", "Okay. I will use clear, direct words."),
        ("Can we start over?", "Yes. What would you like to start with?"),
        ("Wait a moment.", "Okay."),
        ("Ready?", "Ready."),
    ]
    for user, answer in social:
        add("social", user, answer)

    for subject in (
        "teacher's name", "home address", "phone number", "favorite movie", "favorite song",
        "birthday", "middle name", "pet's name", "school name", "best friend's name",
        "dentist's name", "doctor's name", "email address", "password", "plans for tomorrow",
        "favorite food", "shoe size", "bedtime", "classroom number", "bus number",
    ):
        add("unknown", "What is my %s?" % subject, "I don't know your %s." % subject)
        add("unknown", "Do you know my %s?" % subject, "No. You have not told me your %s." % subject)

    colors = ("red", "blue", "green", "yellow", "purple", "orange", "silver", "gold", "black", "white")
    names = ("Ava", "Ben", "Cora", "Diego", "Emma", "Finn", "Gia", "Hugo", "Ivy", "Jamal")
    for index, name in enumerate(names):
        color = colors[index]
        add("grounded", "What is %s's favorite color?" % name,
            "%s's favorite color is %s." % (name, color), ["%s's favorite color is %s" % (name, color)])
        number = (index + 2) * 3
        add("grounded", "How many marbles does %s have?" % name,
            "%s has %d marbles." % (name, number), ["%s has %d marbles" % (name, number)])
        pet = ("dog", "cat", "fish", "rabbit", "bird")[index % 5]
        add("grounded", "What pet does %s have?" % name,
            "%s has a %s." % (name, pet), ["%s has a %s" % (name, pet)])

    opposites = [
        ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("up", "down"),
        ("open", "closed"), ("light", "dark"), ("wet", "dry"), ("full", "empty"),
        ("happy", "sad"), ("early", "late"), ("near", "far"), ("soft", "hard"),
        ("loud", "quiet"), ("clean", "dirty"), ("young", "old"), ("inside", "outside"),
        ("begin", "end"), ("push", "pull"), ("give", "take"), ("above", "below"),
    ]
    for word, opposite in opposites:
        add("instruction", "Say the opposite of %s." % word, opposite.capitalize() + ".")

    categories = [
        ("apple", "fruit"), ("carrot", "vegetable"), ("dog", "animal"), ("robin", "bird"),
        ("salmon", "fish"), ("shirt", "clothing"), ("hammer", "tool"), ("bus", "vehicle"),
        ("chair", "furniture"), ("violin", "instrument"), ("oak", "tree"), ("rose", "flower"),
        ("circle", "shape"), ("January", "month"), ("Monday", "day of the week"),
        ("blue", "color"), ("baseball", "sport"), ("bread", "food"), ("kitchen", "room"),
        ("rain", "weather"),
    ]
    for thing, category in categories:
        add("instruction", "What kind of thing is %s?" % thing, "%s is a %s." % (thing.capitalize(), category))

    plurals = [
        ("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("tree", "trees"),
        ("bus", "buses"), ("box", "boxes"), ("dish", "dishes"), ("baby", "babies"),
        ("berry", "berries"), ("leaf", "leaves"), ("mouse", "mice"), ("child", "children"),
        ("foot", "feet"), ("tooth", "teeth"), ("person", "people"),
    ]
    for word, plural in plurals:
        add("instruction", "What is the plural of %s?" % word, "The plural of %s is %s." % (word, plural))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in examples)
    OUTPUT.write_text(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {"model": "smLANGUAGE_en_GENERAL_001", "items": len(examples), "sha256": digest,
                "categories": dict(sorted(counts.items()))}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

