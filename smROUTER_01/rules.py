"""The baseline to beat: hand-written keyword rules for the same labels."""
import re

SMALL_TALK = re.compile(r"^\W*((ok(ay)?|k+|lol|lmao|ha(ha)+|thanks?( you)?|thx|ty|cool|nice|great|got it|sounds good|np|nvm|"
                        r"never ?mind|bye|alright|perfect|awesome|love it|that'?s (great|awesome|cool|nice)|👍)\W*)+$", re.I)
CALC = re.compile(r"\d\s*%|\bpercent\b|\d\s*[-+*/x×]\s*\d|\b(tip|split|divided|times|plus|minus|squared|square root|average|"
                  r"convert|how many (seconds|ounces|inches|cups|minutes))\b", re.I)
WEB = re.compile(r"\b(weather|forecast|rain|news|latest|score|price|trading|stock|open|close|tonight|right now|these days|"
                 r"last night|near|reviews?|flights?|traffic|in theaters|come out)\b", re.I)
WRITE = re.compile(r"\b(save|write (it |that )?down|jot|make a note|add .+ to|put .+ on|create a file|start a|update my|"
                   r"remove .+ from|take .+ off|make a (packing |shopping |grocery )?list)\b", re.I)
READ = re.compile(r"\b(what'?s on my|what'?s in|read (me |back )?my|show (me )?(my|the)|open (my )?|what did i (put|write)|"
                  r"on the .*list|which files|what files|list my notes)\b", re.I)
REMEMBER = re.compile(r"\b(remember (that|this|my|:)|don'?t forget|keep in mind|please remember|want you to remember)\b", re.I)
RECALL = re.compile(r"\b(my|did i|i tell you|i say)\b.*\?|^\W*(who'?s|what'?s|when'?s|when is|where did i|what am i|remind me what)\b.*\bmy\b", re.I)
QUESTION = re.compile(r"\?\s*$|^\W*(what|who|when|where|why|how|is|are|do|does|did|can|could|will|should|any)\b", re.I)
TASK = re.compile(r"^\W*(write|draft|help|make|give|tell|plan|come up|brainstorm|rewrite|translate|turn|book|remind|send|"
                  r"order|call|buy|schedule|set|reply|cancel|go ahead|summarize|create|add|save|put|jot|take)\b", re.I)


def classify(prev, text):
    t = text.strip()
    if prev.strip().endswith("?") and not QUESTION.search(t):
        intent = "answer"
    elif SMALL_TALK.match(t):
        return {"intent": "small_talk", "tool": "none", "ask_first": False}
    elif REMEMBER.search(t):
        return {"intent": "remember", "tool": "none", "ask_first": False}
    elif TASK.search(t):
        intent = "task"
    elif QUESTION.search(t):
        intent = "question"
    else:
        intent = "statement"
    if re.search(r"https?://", t):
        tool = "web_browser"
    elif CALC.search(t):
        tool = "calculator"
    elif WRITE.search(t):
        tool = "file_write"
    elif READ.search(t):
        tool = "file_read"
    elif RECALL.search(t):
        tool = "memory_recall"
    elif WEB.search(t):
        tool = "web_search"
    else:
        tool = "none"
    return {"intent": intent, "tool": tool, "ask_first": False}
