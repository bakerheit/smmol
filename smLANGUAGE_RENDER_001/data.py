"""Generated grounded-reply lessons for smLANGUAGE_RENDER_001.

The payload shape matches what paratroop_harness_02 gives its Language module. Replies
are deliberately concise and canonical: v1 is a renderer, not an open-ended writer.
"""
import json
import random
import re
import string


FIRST = ["Alex", "Avery", "Casey", "Devon", "Emery", "Harper", "Jamie", "Jordan", "Kai", "Morgan",
         "Noah", "Parker", "Quinn", "Riley", "Robin", "Sam", "Taylor", "Terry", "Val", "Zoe"]
LAST = ["Bennett", "Brooks", "Chen", "Diaz", "Ellis", "Flores", "Grant", "Hayes", "Kim", "Lee",
        "Moreno", "Nguyen", "Ortiz", "Patel", "Reed", "Rivera", "Singh", "Turner", "Wells", "Young"]
CITIES = ["Atlanta", "Austin", "Baltimore", "Boston", "Chicago", "Cleveland", "Dallas", "Dayton", "Denver",
          "Detroit", "Houston", "Las Vegas", "Miami", "Nashville", "New York", "Orlando", "Phoenix", "Portland",
          "Raleigh", "Richmond", "Seattle", "St. Louis", "Tampa", "Tulsa"]
ROLES = ["accountant", "attorney", "dentist", "doctor", "eye doctor", "insurance contact", "mechanic",
         "pharmacist", "physical therapist", "project contact", "realtor", "tax preparer", "veterinarian"]
FILE_WORDS = ["budget", "chores", "garden", "hardware", "ideas", "packing", "projects", "repairs", "shopping",
              "supplies", "tasks", "travel", "weekend", "workout"]
ITEM_WORDS = ["batteries", "beans", "bolts", "bread", "brushes", "charger", "coffee", "filters", "gloves", "jacket",
              "milk", "nails", "paint", "rice", "screws", "socks", "sugar", "tape", "tomatoes", "washers"]


def clean_text(value, limit):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def compact(payload):
    """Stable, bounded JSON used as the model prompt and later by the harness adapter."""
    out = {}
    for key, limit in (("message", 240), ("goal", 160), ("note", 160), ("question_to_ask", 200)):
        if payload.get(key):
            out[key] = clean_text(payload[key], limit)
    if payload.get("decision"):
        d = payload["decision"]
        out["decision"] = {k: clean_text(d.get(k), 160) for k in ("type", "summary", "why") if d.get(k)}
    if payload.get("conversation"):
        out["conversation"] = [{k: clean_text(t.get(k), 200) for k in ("message", "reply") if t.get(k)}
                               for t in payload["conversation"][-2:]]
    if payload.get("observations"):
        out["observations"] = [{k: clean_text(o.get(k), 180) for k in ("kind", "text", "who", "what", "when", "where")
                                if o.get(k)} for o in payload["observations"][:4]]
    if payload.get("memories"):
        out["memories"] = [clean_text(m, 220) for m in payload["memories"][:3]]
    if payload.get("problems"):
        out["problems"] = [{k: clean_text(p.get(k), 160) for k in ("expression", "answer", "unit", "about", "error")
                            if p.get(k) not in (None, "")} for p in payload["problems"][:5]]
    if payload.get("tool_results"):
        out["tool_results"] = [{k: (bool(a.get(k)) if k == "ok" else clean_text(a.get(k), 520 if k == "result" else 180))
                                for k in ("tool", "action", "target", "ok", "result") if a.get(k) not in (None, "")}
                               for a in payload["tool_results"][-2:]]
    for key in ("assumptions", "open_details"):
        if payload.get(key):
            out[key] = [clean_text(x, 180) for x in payload[key][:3]]
    return json.dumps(out, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def base(message, kind="answer", summary="Answer directly", why="The requested information is available"):
    return {"message": message, "goal": message, "decision": {"type": kind, "summary": summary, "why": why}}


def lesson(kind, payload, reply):
    return {"kind": kind, "payload": payload, "prompt": compact(payload), "reply": reply}


def person(r, titled=False):
    name = f"{r.choice(FIRST)} {r.choice(LAST)}"
    return ("Dr. " + name.split()[1]) if titled else name


def file_name(r):
    return f"{r.choice(FILE_WORDS)}-{r.choice(FILE_WORDS)}.{r.choice(['txt', 'md'])}"


def item_list(r):
    a, b, c = r.sample(ITEM_WORDS, 3)
    return f"{a}, {b}, and {c}"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def math_case(r):
    a, b = r.randint(2, 9999), r.randint(2, 9999)
    op = r.choice(["+", "-", "*"])
    if op == "-" and b > a:
        a, b = b, a
    answer = {"+": a + b, "-": a - b, "*": a * b}[op]
    unit = r.choice(["", " dollars", " miles", " items"])
    message = r.choice([f"What's {a} {op} {b}?", f"work out {a}{op}{b}", f"Can you calculate {a} {op} {b} for me?"])
    p = base(message, summary="Give the calculated result")
    p["problems"] = [{"expression": f"{a}{op}{b}", "answer": str(answer), "unit": unit.strip()}]
    reply = f"{a} {op} {b} = {answer}{unit}."
    return lesson("math", p, reply)


def percent_case(r):
    pct, total = r.choice([5, 10, 15, 20, 25, 30, 40, 50, 75]), r.randrange(20, 2001, 20)
    answer = pct * total / 100
    shown = str(int(answer)) if answer.is_integer() else str(round(answer, 2))
    message = f"What's {pct}% of {total}?"
    p = base(message, summary="Give the percentage result")
    p["problems"] = [{"expression": f"{pct}/100*{total}", "answer": shown, "about": "percentage"}]
    return lesson("math", p, f"{pct}% of {total} is {shown}.")


def memory_case(r):
    role = r.choice(ROLES)
    name = person(r, titled=role in ("dentist", "doctor", "eye doctor", "veterinarian"))
    message = r.choice([f"Who's my {role}?", f"What was my {role}'s name?", f"Remind me who my {role} is."])
    fact = f"Your {role} is {name}."
    p = base(message, summary="Answer from the remembered fact")
    p["memories"] = [fact]
    return lesson("memory", p, fact)


def ask_case(r):
    subject, question, reason = r.choice([
        ("trip", "Which city are you traveling to?", "I need the destination first."),
        ("meeting", "What time should the meeting start?", "I need the start time first."),
        ("reminder", "When should I remind you?", "I need a time for the reminder."),
        ("delivery", "What address should it go to?", "I need the delivery address."),
        ("file", "What should the file be called?", "I need a file name."),
        ("package", "Who should receive the package?", "I need the recipient."),
        ("reservation", "How many people is the reservation for?", "I need the party size."),
        ("appointment", "Which day works for you?", "I need a day for the appointment."),
        ("purchase", "What is the spending limit?", "I need a budget first."),
        ("message", "Who should I address the message to?", "I need the recipient's name."),
        ("report", "Which date range should the report cover?", "I need a date range."),
    ])
    message = f"Help me with this {subject}."
    p = base(message, kind="ask", summary="Ask for the missing detail", why=reason)
    p["question_to_ask"] = question
    return lesson("ask", p, f"{reason} {question}")


def file_case(r):
    name, items = file_name(r), item_list(r)
    action = r.choice(["create", "update", "read", "delete"])
    if action == "read":
        message, result, reply = f"Read {name}", items, f"`{name}` contains: {items}."
    elif action == "delete":
        message, result, reply = f"Delete {name}", f"Moved {name} to trash", f"Moved `{name}` to the trash."
    else:
        verb = "Save" if action == "create" else "Update"
        message, result = f"{verb} {items} in {name}", f"{action}d {name}"
        reply = f"Saved {items} in `{name}`."
    p = base(message, summary=f"Confirm the file {action}")
    p["tool_results"] = [{"tool": "file_system", "action": action, "target": name, "ok": True, "result": result}]
    return lesson("file", p, reply)


def file_error_case(r):
    name = file_name(r)
    error = r.choice(["file already exists", "file does not exist", "the name is not allowed"])
    message = f"Save my notes in {name}"
    p = base(message, summary="Explain that the save failed")
    p["tool_results"] = [{"tool": "file_system", "action": "create", "target": name, "ok": False, "result": error}]
    return lesson("file_error", p, f"I couldn't save `{name}`: {error}.")


def web_case(r):
    city, temp = r.choice(CITIES), r.randint(35, 96)
    condition = r.choice(["sunny", "cloudy", "rainy", "clear", "partly cloudy"])
    message = f"What's the weather in {city} right now?"
    fact = f"It is {temp}°F and {condition} in {city}."
    p = base(message, summary="Answer with the current weather result")
    p["tool_results"] = [{"tool": "web_browser", "action": "none", "target": f"https://weather.example/{slug(city)}",
                          "ok": True, "result": fact}]
    return lesson("web", p, fact)


def need_case(r):
    city = r.choice(CITIES)
    url = f"https://weather.example/{slug(city)}/today"
    message = f"What's the temperature in {city} right now?"
    p = base(message, summary="Answer with current temperature")
    p["tool_results"] = [{"tool": "web_search", "action": "none", "target": message, "ok": True,
                          "result": f"1. {city} weather today\n   {url}\n   Current conditions and forecast"}]
    return lesson("need", p, f"NEED: web_browser {url}")


def direct_case(r):
    event = r.choice(["budget review", "design review", "dentist appointment", "flight", "inspection", "interview",
                      "parent meeting", "project kickoff", "service visit", "train"])
    day = r.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    time = r.choice(["9:00 AM", "10:30 AM", "2:00 PM", "4:15 PM"])
    message = r.choice([f"When is the {event}?", f"What time is the {event}?", f"Remind me when the {event} happens."])
    fact = f"The {event} is {day} at {time}."
    p = base(message, summary="Answer from the known schedule")
    p["observations"] = [{"kind": "fact", "text": fact}]
    return lesson("direct", p, fact)


def assumption_case(r):
    zip_code, city = r.choice([("02108", "Boston, Massachusetts"), ("60601", "Chicago, Illinois"), ("97205", "Portland, Oregon")])
    temp = r.randint(35, 96)
    message = f"What's the weather in {zip_code}?"
    p = base(message, summary="Answer with the current weather result")
    p["assumptions"] = [f"{zip_code} means {city}"]
    p["tool_results"] = [{"tool": "web_browser", "action": "none", "target": "https://weather.example/current",
                          "ok": True, "result": f"The current temperature in {city} is {temp}°F."}]
    return lesson("assumption", p, f"Assumption: {zip_code} means {city}.\nThe current temperature is {temp}°F.")


def unavailable_case(r):
    topic = r.choice(["the current temperature", "today's stock price", "the latest score", "the flight status"])
    message = f"What is {topic}?"
    p = base(message, summary="Explain that current data is unavailable", why="No current result was provided")
    return lesson("unavailable", p, "I don't have current data for that.")


def copy_case(r):
    """Teach attention to copy novel source-backed facts instead of memorizing a small name catalog."""
    label = r.choice(["access code", "booking code", "case number", "confirmation code", "device code", "order code"])
    letters = "".join(r.choice(string.ascii_uppercase) for _ in range(3))
    code = f"{letters}-{r.randint(100, 999)}"
    message = r.choice([f"What's the {label}?", f"Remind me of the {label}.", f"What was that {label} again?"])
    fact = f"The {label} is {code}."
    p = base(message, summary="Answer from the known fact")
    p["observations"] = [{"kind": "fact", "text": fact}]
    return lesson("copy", p, fact)


MAKERS = [math_case, percent_case, memory_case, ask_case, file_case, file_error_case,
          web_case, need_case, direct_case, assumption_case, unavailable_case, copy_case]


def generate(n, seed=0):
    r = random.Random(seed)
    return [MAKERS[i % len(MAKERS)](r) for i in range(n)]


if __name__ == "__main__":
    for example in generate(12):
        print(example["kind"], example["prompt"], "=>", example["reply"])
