"""Messages that may have arithmetic in them, and the problems a reader should pull out.

Each example is {"text", "target"}. Targets are compact, so the model has less to write:

    322234*21323*212231|cubic feet|volume;p1/27|cubic yards|volume

Problems are separated by ";", and each one is expression|unit|about. Expressions only use numbers,
+ - * / ** ( ) and sqrt(), and p1, p2 ... stand for earlier answers in the same message. An empty target
means there's nothing to work out. read.py turns a target into the JSON the harness expects.

The hand-written test messages in test.json are phrased differently on purpose, and any generated message
that matches one word for word is dropped from training.
"""
import random

from score import values

NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                11: "eleven", 12: "twelve"}
GREETINGS = ["hey ", "quick q: ", "ok so ", "um ", "yo ", "hi, ", "can you tell me "]
LENGTHS = [("ft", "feet"), ("m", "meters"), ("in", "inches"), ("cm", "centimeters"), ("yd", "yards")]
ITEMS = ["tickets", "pizzas", "shirts", "bags of mulch", "notebooks", "boxes of cereal", "gallons of paint", "lunches"]
CITIES = ["Chicago", "Denver", "Tokyo", "Austin", "Seattle", "Boston", "Oslo", "Lima"]
CONVERSIONS = [("miles", "kilometers", "*1.609344"), ("kilometers", "miles", "*0.621371"), ("pounds", "kilograms", "*0.45359237"),
               ("kilograms", "pounds", "*2.20462"), ("inches", "centimeters", "*2.54"), ("feet", "meters", "*0.3048"),
               ("hours", "minutes", "*60"), ("hours", "seconds", "*3600"), ("days", "hours", "*24"), ("days", "seconds", "*86400"),
               ("cups", "milliliters", "*236.588"), ("gallons", "liters", "*3.78541")]
SHORT = {"kilometers": "km", "miles": "mi", "pounds": "lbs", "kilograms": "kg", "inches": "in", "centimeters": "cm",
         "feet": "ft", "meters": "m", "milliliters": "ml", "liters": "L", "hours": "hrs", "minutes": "min", "seconds": "sec"}


def target(*problems):
    return ";".join("|".join(p) for p in problems)


def parse_target(text):
    """A target as the harness's problem list: [{"id": "p1", "expression", "unit", "about"}, ...]."""
    problems = []
    for part in (p for p in text.split(";") if p.strip()):
        expression, unit, about = (part.split("|") + ["", ""])[:3]
        problems.append({"id": "p%d" % (len(problems) + 1), "expression": expression.strip(), "unit": unit.strip(),
                         "about": about.strip()})
    return problems


def fill(r, forms, **values_):
    return r.choice(forms).format(**values_)


def listing(r, items):
    return ", ".join(items[:-1]) + r.choice([" and ", ", and ", ", "]) + items[-1]


# ----- numbers: (as typed in the message, as the expression writes it) -----

def whole(r, digits=None):
    d = digits or r.choice([1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8])
    n = r.randint(0 if d == 1 else 10 ** (d - 1), 10 ** d - 1)
    return ("{:,}".format(n) if n >= 1000 and r.random() < 0.35 else str(n)), str(n)


def small(r, lo=2, hi=12, words=0.25):
    k = r.randint(lo, hi)
    return (NUMBER_WORDS[k] if k in NUMBER_WORDS and r.random() < words else str(k)), str(k)


def decimal(r):
    s = "%d.%d" % (r.randint(0, 999), r.randint(1, 99))
    return s, s


def money(r, lo=1, hi=2500):
    dollars = r.randint(lo, hi)
    cents = r.choice(["", "", "", ".50", ".99", ".25", ".75", ".%02d" % r.randint(1, 99)])
    if dollars == 0 and not cents:
        cents = ".%02d" % r.randint(5, 95)
    shown = ("{:,}".format(dollars) if dollars >= 1000 and r.random() < 0.5 else str(dollars)) + cents
    return r.choice(["$" + shown, "$" + shown, "$" + shown, shown + " dollars", shown + " bucks"]), "%d%s" % (dollars, cents)


def percent(r):
    p = r.choice(["5", "7.25", "8", "10", "12.5", "15", "17.5", "18", "20", "22", "25", "30", "33", "40", "50", "65"])
    return r.choice([p + "%", p + "%", p + " percent"]), p


def operand(r):
    """A number for arithmetic in words: mostly whole, sometimes a decimal or a spelled-out small number."""
    roll = r.random()
    if roll < 0.12:
        return decimal(r)
    if roll < 0.22:
        return small(r, 2, 12, words=1.0)
    return whole(r)


# ----- kinds of messages -----

def bare(r):
    ops = r.choice(["+", "-", "*", "x", "/", "+-", "*+"])
    typed, expr = [], []
    for i in range(r.choice([2, 2, 2, 3])):
        if i:
            op = r.choice(ops)
            typed.append(r.choice(["x", "x", "×"]) if op == "x" else op)
            expr.append("*" if op == "x" else op)
        t, e = whole(r) if r.random() < 0.85 else decimal(r)
        typed.append(t)
        expr.append(e)
    if len(typed) == 5 and r.random() < 0.3:
        typed, expr = ["("] + typed[:3] + [")"] + typed[3:], ["("] + expr[:3] + [")"] + expr[3:]
    shown = (" " if r.random() < 0.5 else "").join(typed).replace("( ", "(").replace(" )", ")")
    forms = ["{e}", "{e}", "{e} =", "{e}=", "{e}?", "what's {e}", "whats {e}", "what is {e}", "calc {e}", "{e} equals"]
    return fill(r, forms, e=shown), target(("".join(expr), "", ""))


def dimensions(r, count):
    short, plural = r.choice(LENGTHS)
    dims = [whole(r, r.choice([1, 2, 2, 3, 3, 4, 5, 6])) for _ in range(count)]
    style = r.random()
    if style < 0.5:  # 322234ft x 21323ft x 212231ft
        glue = "" if r.random() < 0.7 else " "
        text = r.choice([" x ", "x", " × ", " by ", " * "]).join(t + glue + short for t, _ in dims)
    elif style < 0.8:  # 12 feet by 15 feet
        text = r.choice([" by ", " x "]).join("%s %s" % (t, plural) for t, _ in dims)
    else:  # 3 x 4 x 5 ft
        text = r.choice([" x ", "x", " by "]).join(t for t, _ in dims) + r.choice([" ", ""]) + short
    return text, [e for _, e in dims], plural


def volume(r):
    text, dims, plural = dimensions(r, 3)
    forms = ["what is the volume of {d}?", "volume of {d}", "whats the volume of a {d} box", "how many cubic {u} is {d}",
             "{d}, volume?", "a tank that's {d}, what's the volume", "calculate the volume: {d}", "how much space is {d}"]
    return fill(r, forms, d=text, u=plural), target(("*".join(dims), "cubic " + plural, "volume"))


def area(r):
    text, dims, plural = dimensions(r, 2)
    forms = ["area of a {d} room", "what's the area of {d}", "how many square {u} is {d}", "{d} area?",
             "floor space of a {d} kitchen"]
    return fill(r, forms, d=text, u=plural), target(("*".join(dims), "square " + plural, "area"))


def percent_of(r):
    (pt, pe), is_money = percent(r), r.random() < 0.3
    nt, ne = money(r) if is_money else whole(r)
    forms = ["what's {p} of {n}", "what is {p} of {n}?", "{p} of {n}", "calculate {p} of {n}", "how much is {p} of {n}",
             "{p} of {n} is what"]
    return fill(r, forms, p=pt, n=nt), target((ne + "*" + pe + "/100", "dollars" if is_money else "", "percent"))


def tip(r):
    (mt, me), (pt, pe) = money(r, 8, 900), percent(r)
    forms = ["what's a {p} tip on {m}", "tip on a {m} bill at {p}", "how much is a {p} tip for {m}", "{p} tip on {m}?",
             "{m} dinner, {p} tip, how much is the tip"]
    return fill(r, forms, p=pt, m=mt), target((me + "*" + pe + "/100", "dollars", "tip"))


def tax(r):
    (mt, me), (pt, pe) = money(r, 5, 3000), percent(r)
    if r.random() < 0.5:
        forms = ["how much is {p} tax on {m}", "what's the tax on {m} at {p}", "{p} sales tax on {m}"]
        return fill(r, forms, p=pt, m=mt), target((me + "*" + pe + "/100", "dollars", "tax"))
    forms = ["{m} plus {p} tax", "what's {m} with {p} tax", "total for {m} after {p} sales tax", "how much is {m} including {p} tax"]
    return fill(r, forms, p=pt, m=mt), target((me + "+" + me + "*" + pe + "/100", "dollars", "total"))  # divide last: fixed-point math loses nothing


def discount(r):
    (mt, me), (pt, pe) = money(r, 5, 3000), percent(r)
    forms = ["{m} jacket at {p} off, what do I pay", "{p} off {m}", "a {m} TV is {p} off, how much is it now",
             "price after {p} off {m}", "{m} with a {p} discount"]
    return fill(r, forms, p=pt, m=mt), target((me + "-" + me + "*" + pe + "/100", "dollars", "price"))


def split(r):
    (mt, me), (kt, ke) = money(r, 20, 5000), small(r, 2, 12, words=0.35)
    forms = ["split {m} between {k} people", "{m} rent and {k} roommates, what's my share", "{k} of us are splitting a {m} bill",
             "divide {m} {k} ways", "{m} split {k} ways?", "{k} people chipping in on {m}, how much each"]
    return fill(r, forms, m=mt, k=kt), target((me + "/" + ke, "dollars", "each"))


def words_op(r):
    (at, ae), (bt, be) = operand(r), operand(r)
    op = r.choice("+-*/")
    forms = {"+": ["{a} plus {b}", "add {a} and {b}", "what's {a} plus {b}", "{a} and {b} added together", "sum of {a} and {b}"],
             "-": ["{a} minus {b}", "subtract {b} from {a}", "what's {a} take away {b}", "{a} less {b}"],
             "*": ["{a} times {b}", "multiply {a} by {b}", "what is {a} multiplied by {b}", "{a} times {b}?"],
             "/": ["{a} divided by {b}", "divide {a} by {b}", "what's {a} over {b}", "how many times does {b} go into {a}"]}[op]
    return fill(r, forms, a=at, b=bt), target((ae + op + be, "", ""))


def add_up(r):
    is_money = r.random() < 0.4
    nums = [money(r, 1, 900) if is_money else whole(r, r.choice([1, 2, 2, 3, 3, 4])) for _ in range(r.choice([3, 3, 4, 5]))]
    forms = ["add up {l}", "what's the total of {l}", "sum {l}", "total of {l}?", "can you add {l}", "{l}, what's that all together"]
    return (fill(r, forms, l=listing(r, [t for t, _ in nums])),
            target(("+".join(e for _, e in nums), "dollars" if is_money else "", "total")))


def average(r):
    nums = [whole(r, r.choice([1, 2, 2, 3])) for _ in range(r.choice([2, 3, 3, 4, 5]))]
    forms = ["average of {l}", "what's the average of {l}", "mean of {l}", "{l}, what's the average", "what do {l} average out to"]
    return (fill(r, forms, l=listing(r, [t for t, _ in nums])),
            target(("(" + "+".join(e for _, e in nums) + ")/" + str(len(nums)), "", "average")))


def power_root(r):
    kind = r.randrange(4)
    if kind == 0:
        t, e = whole(r, r.choice([1, 2, 2, 3, 4]))
        return fill(r, ["{a} squared", "what's {a} squared", "square {a}", "{a} squared is what"], a=t), target((e + "**2", "", ""))
    if kind == 1:
        t, e = whole(r, r.choice([1, 2, 2, 3]))
        return fill(r, ["{a} cubed", "what's {a} cubed", "{a} to the third power"], a=t), target((e + "**3", "", ""))
    if kind == 2:
        t, e = whole(r, r.choice([1, 2, 3, 4, 5]))
        forms = ["square root of {n}", "what's the square root of {n}", "sqrt of {n}", "sqrt({n})", "√{n}"]
        return fill(r, forms, n=t), target(("sqrt(" + e + ")", "", ""))
    (t, e), k = whole(r, r.choice([1, 2])), str(r.randint(2, 9))
    forms = ["{a} to the power of {k}", "{a}^{k}", "{a} to the {k}th power", "{a} raised to {k}"]
    return fill(r, forms, a=t, k=k), target((e + "**" + k, "", ""))


def per_unit(r):
    (kt, ke), (mt, me), item = small(r, 2, 40, words=0.2), money(r, 1, 300), r.choice(ITEMS)
    text = fill(r, ["{k} {i} at {m} each", "{k} {i} for {m} apiece", "how much for {k} {i} at {m} each",
                    "what do {k} {i} cost at {m} each", "{k} {i}, {m} each, total?"], k=kt, i=item, m=mt)
    expr = ke + "*" + me
    if r.random() < 0.3:
        ft, fe = money(r, 1, 40)
        text += r.choice([" plus a {f} fee", " and a {f} service fee", " with {f} shipping"]).format(f=ft)
        expr += "+" + fe
    return text, target((expr, "dollars", "total"))


def savings(r):
    (mt, me), (period, factor) = money(r, 1, 900), r.choice([("a week", "52"), ("every week", "52"), ("a month", "12"),
                                                           ("per month", "12"), ("a day", "365"), ("every day", "365")])
    forms = ["if I save {m} {p} how much is that in a year", "saving {m} {p}, how much after a year",
             "{m} {p} for a year is how much", "how much is {m} {p} over a year"]
    return fill(r, forms, m=mt, p=period), target((me + "*" + factor, "dollars", "year"))


def travel(r):
    st, se = small(r, 20, 80, words=0)
    if r.random() < 0.6:
        (dt, de), (u, speed) = whole(r, r.choice([2, 3, 4])), r.choice([("miles", "mph"), ("km", "km/h")])
        forms = ["how long to drive {d} {u} at {s} {v}", "{d} {u} at {s} {v}, how many hours", "how many hours to go {d} {u} at {s} {v}",
                 "time to cover {d} {u} going {s} {v}"]
        return fill(r, forms, d=dt, u=u, s=st, v=speed), target((de + "/" + se, "hours", "time"))
    ht, he = small(r, 2, 12, words=0.3)
    forms = ["how far do I go in {h} hours at {s} mph", "driving {s} mph for {h} hours, how far", "{h} hours at {s} mph is how many miles"]
    return fill(r, forms, h=ht, s=st), target((he + "*" + se, "miles", "distance"))


def fuel(r):
    (gt, ge), (dt, de) = small(r, 15, 60, words=0), whole(r, r.choice([2, 3, 4]))
    forms = ["my car gets {g} mpg, how many gallons for {d} miles", "{d} miles at {g} miles per gallon, how much gas",
             "how many gallons to go {d} miles at {g} mpg", "gas needed for {d} miles if I get {g} mpg"]
    return fill(r, forms, g=gt, d=dt), target((de + "/" + ge, "gallons", "fuel"))


def convert(r):
    if r.random() < 0.2:
        tt, te = whole(r, r.choice([1, 2, 3]))
        if r.random() < 0.5:
            forms = ["convert {t}°F to celsius", "{t} fahrenheit in celsius", "what's {t} F in C", "{t} degrees F to C"]
            return fill(r, forms, t=tt), target(("(" + te + "-32)*5/9", "°C", "temperature"))
        forms = ["convert {t}°C to fahrenheit", "{t} celsius in fahrenheit", "what's {t} C in F", "{t} degrees C to F"]
        return fill(r, forms, t=tt), target((te + "*9/5+32", "°F", "temperature"))
    a, b, factor = r.choice(CONVERSIONS)
    nt, ne = whole(r, r.choice([1, 2, 3, 4])) if r.random() < 0.7 else decimal(r)

    def name(word):
        return SHORT.get(word, word) if r.random() < 0.3 else word
    forms = ["convert {n} {a} to {b}", "{n} {a} in {b}", "how many {b} is {n} {a}", "{n} {a} to {b}?", "{n} {a} is how many {b}"]
    return fill(r, forms, n=nt, a=name(a), b=name(b)), target((ne + factor, b, "conversion"))


def number_words(r):
    kind = r.randrange(4)
    if kind == 0:
        (kt, ke), (mt, me), item = small(r, 1, 6, words=0.5), money(r, 0, 5), r.choice(["eggs", "donuts", "roses", "cookies"])
        if ke == "1":
            return fill(r, ["a dozen {i} at {m} each", "one dozen {i}, {m} apiece"], i=item, m=mt), target(("12*" + me, "dollars", "total"))
        return (fill(r, ["{k} dozen {i} at {m} each", "{k} dozen {i} for {m} apiece, total?"], k=kt, i=item, m=mt),
                target((ke + "*12*" + me, "dollars", "total")))
    if kind == 1:
        t, e = whole(r)
        return fill(r, ["half of {n}", "what's half of {n}", "{n} cut in half"], n=t), target((e + "/2", "", ""))
    if kind == 2:
        kt, ke = small(r, 2, 10, words=0.5)
        return fill(r, ["what's half of {k} dozen", "half of {k} dozen"], k=kt), target((ke + "*12/2", "", ""))
    t, e = whole(r)
    word, factor = r.choice([("double", "2"), ("twice", "2"), ("triple", "3")])
    return fill(r, ["{w} {n}", "what's {w} {n}", "{w} {n}?"], w=word, n=t), target((e + "*" + factor, "", ""))


SIMPLE = [percent_of, words_op, split, tip, add_up, power_root, bare]


def several(r):
    parts = [r.choice(SIMPLE)(r) for _ in range(r.choice([2, 2, 3]))]
    joiner = r.choice([" and ", ", and ", "; ", " and also ", ". also "])
    return joiner.join(t.rstrip("?.! =") for t, _ in parts) + r.choice(["", "?"]), ";".join(g for _, g in parts)


def follow_up(r):
    kind = r.randrange(5)
    if kind == 0:
        dims, unit = [whole(r, r.choice([1, 2, 2, 3])) for _ in range(3)], r.choice(["ft", " ft", " feet"])
        d = r.choice([" x ", "x", " by "]).join(t + unit for t, _ in dims)
        forms = ["volume of {d}, and how many cubic yards is that?", "{d} in cubic feet and cubic yards",
                 "what's the volume of {d} and what is that in cubic yards"]
        return fill(r, forms, d=d), target(("*".join(e for _, e in dims), "cubic feet", "volume"), ("p1/27", "cubic yards", "volume"))
    if kind == 1:
        (at, ae), (bt, be), (ct, ce) = whole(r, r.choice([1, 2])), whole(r, r.choice([1, 2])), small(r, 1, 5, words=0)
        u, long_ = r.choice([("m", "meters"), ("ft", "feet")])
        forms = ["area of a {a}{u} by {b}{u} garden, and the volume if it's {c}{u} deep",
                 "a {a}{u} x {b}{u} pool: floor area, and the volume at {c}{u} deep",
                 "{a}{u} by {b}{u} patio, area and then volume for a {c}{u} thick slab"]
        return (fill(r, forms, a=at, b=bt, c=ct, u=u),
                target((ae + "*" + be, "square " + long_, "area"), ("p1*" + ce, "cubic " + long_, "volume")))
    if kind == 2:
        (mt, me), (kt, ke), (tt, te) = money(r, 20, 900), small(r, 2, 8, words=0.3), money(r, 1, 20)
        forms = ["split {m} between {k} people and add a {t} tip each", "{k} of us owe {m}, what's each share, and with {t} extra each?",
                 "divide {m} {k} ways then add {t} per person"]
        return (fill(r, forms, m=mt, k=kt, t=tt),
                target((me + "/" + ke, "dollars", "each"), ("p1+" + te, "dollars", "each with tip")))
    if kind == 3:
        (pt, pe), (nt, ne), (ct, ce) = percent(r), whole(r), small(r, 2, 99, words=0)
        then, second = r.choice([("then take away {c}", "p1-{c}"), ("then subtract {c}", "p1-{c}"), ("then add {c}", "p1+{c}"),
                                 ("then double it", "p1*2"), ("then halve it", "p1/2"), ("then divide that by {c}", "p1/{c}")])
        forms = ["what's {p} of {n}, " + then, "{p} of {n}, " + then]
        return fill(r, forms, p=pt, n=nt, c=ct), target((ne + "*" + pe + "/100", "", "percent"), (second.format(c=ce), "", ""))
    (at, ae), (bt, be), (ct, ce) = whole(r, r.choice([1, 2, 3])), whole(r, r.choice([1, 2, 3])), small(r, 2, 99, words=0)
    first, fop = r.choice([("{a} times {b}", "*"), ("{a} plus {b}", "+"), ("multiply {a} by {b}", "*"), ("add {a} and {b}", "+")])
    then, top = r.choice([("then divide that by {c}", "/"), ("then subtract {c}", "-"), ("and then times {c}", "*"), ("then add {c}", "+")])
    return (first + ", " + then).format(a=at, b=bt, c=ct), target((ae + fop + be, "", ""), ("p1" + top + ce, "", ""))


def no_math(r):
    k, j = small(r, 1, 9, words=0.3)[0], small(r, 1, 9, words=0.3)[0]
    forms = [
        "we've got {k} kids and {j} cats", "my train leaves at {hh}:{mm}", "you can reach me at {phone}", "she was born in {year}",
        "remember my {code_name} is {code}", "room {room} please", "the meeting is on the {nth}", "I ran {run} miles today",
        "see you at {clock}", "{days} more days until vacation!", "is it going to rain on the {nth}", "my zip code is {zip}",
        "write a poem about {k} cats", "is it cold in {city} right now", "haha ok", "ok cool", "thanks!", "sounds good",
        "who won the game last night", "what year did the Berlin Wall fall", "set an alarm for {h} am", "top {top} movies of {year}",
        "my son is {k} and my daughter is {j}", "order {k} pizzas for the party", "the score was {s1}-{s2}", "we won {s1} to {s2}",
        "my number is {phone}", "I have {k} meetings at {h}pm", "it's {temp} degrees out", "flight {flight} is delayed",
        "apartment {room}B", "iOS {ver} is out", "chapter {k} was great", "my kid is in grade {k}", "{k} out of {j} stars"]
    return fill(r, forms, k=k, j=j, hh=r.randint(1, 12), mm="%02d" % r.choice([0, 5, 15, 30, 45, 50]),
                phone=r.choice(["%03d-%04d" % (r.randint(200, 999), r.randint(0, 9999)),
                                "%03d-%03d-%04d" % (r.randint(200, 999), r.randint(200, 999), r.randint(0, 9999))]),
                year=r.randint(1940, 2026), code_name=r.choice(["locker combo", "wifi password", "pin", "bike lock code"]),
                code=r.randint(1000, 999999), room=r.randint(100, 999), nth=r.choice(["3rd", "12th", "21st", "30th"]),
                run=r.choice(["3", "5", "6.2", "13.1"]), clock=r.choice(["7", "7pm", "6:30", "noon"]), days=r.randint(2, 30),
                zip="%05d" % r.randint(10000, 99999), city=r.choice(CITIES), h=r.randint(5, 9), top=r.choice([5, 10, 20]),
                s1=r.randint(0, 40), s2=r.randint(0, 40), temp=r.randint(-10, 105), flight="%s%d" % (r.choice(["UA", "DL", "AA"]), r.randint(10, 2999)),
                ver=r.randint(14, 20)), ""


KINDS = [(bare, 1.0), (volume, 0.7), (area, 0.5), (percent_of, 0.7), (tip, 0.5), (tax, 0.5), (discount, 0.4), (split, 0.6),
         (words_op, 0.8), (add_up, 0.5), (average, 0.4), (power_root, 0.4), (per_unit, 0.5), (savings, 0.4), (travel, 0.4),
         (fuel, 0.3), (convert, 0.5), (number_words, 0.4), (several, 0.8), (follow_up, 0.6), (no_math, 1.6)]


def typo(r, text):
    spots = [i for i in range(1, len(text) - 2) if text[i].isalpha() and text[i + 1].isalpha()]
    if not spots:
        return text
    i = r.choice(spots)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:] if r.random() < 0.5 else text[:i] + text[i + 1:]


def roughen(r, text):
    """Greetings, dropped punctuation, a letter typo now and then, and odd casing. Digits are never touched."""
    if r.random() < 0.12:
        text = r.choice(GREETINGS) + text
    if r.random() < 0.3:
        text = text.rstrip("?.!")
    if r.random() < 0.06:
        text = typo(r, text)
    if r.random() < 0.35:
        text = text[:1].upper() + text[1:]
    elif r.random() < 0.2:
        text = text.lower()
    return text


def example(r):
    while True:
        make = r.choices([k for k, _ in KINDS], weights=[w for _, w in KINDS])[0]
        text, tgt = make(r)
        try:
            values([p["expression"] for p in parse_target(tgt)])  # every target must work out, so nothing broken is learned
        except ValueError:
            continue
        return {"text": roughen(r, text), "target": tgt}


def generate(n, seed=0):
    r = random.Random(seed)
    return [example(r) for _ in range(n)]


def normal(text):
    return " ".join(text.lower().strip().rstrip("?.!").split())


def without(examples, held):
    """Drop generated messages that match a hand-written test message word for word."""
    banned = {normal(e["text"]) for e in held}
    return [e for e in examples if normal(e["text"]) not in banned]


if __name__ == "__main__":
    for ex in generate(20, 3):
        print("%-70s  ->  %s" % (ex["text"], ex["target"] or "(nothing)"))
