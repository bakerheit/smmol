"""Data contracts, structural validation, and deterministic corpus utilities."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
THINK_RE = re.compile(r"</?think>", re.I)
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)(?:\$\s*)?\d+(?:[.,]\d+)?(?:\s*%|\b)")
STAGE_DIRECTION_RE = re.compile(
    r"(?:^|\s)\*(?:laughs?|sighs?|nods?|smiles?|pauses?|shrugs?|gestures?|thinks?|looks?|kicks?|"
    r"leans?|raises?|types?|waves?|grins?|chuckles?|gasps?|whispers?|clears? (?:his |her |their )?throat)"
    r"(?:\s+[^*\n]{0,80})?\*(?:\s|$)|^\s*\[[^\]\n]{1,160}\]\s*$",
    re.I | re.M,
)
REMOTE_ACTION_RE = re.compile(
    r"\bi(?:'ll| will| can| am going to| already| just)?\s+"
    r"(?:call|phone|visit|drive|come over|pick up|buy|purchase|mail|ship|bring|meet|contact)\b",
    re.I,
)
GENERIC_AI = (
    "as an ai language model",
    "i don't have personal experiences",
    "i cannot provide real-time",
)
META_CHAT = (
    "end of conversation",
    "end of turn",
    "assistant's turn",
    "user's turn",
    "json structure",
    "requested json",
    "exactly 20 turns",
    "exactly 20 messages",
    "this dataset",
)
HIGH_RISK_TOPIC_RE = re.compile(
    r"\b(?:allerg(?:y|ies|en|ens|ic)|gluten[- ]?free|nut[- ]?free|food safety|medical|medication|"
    r"dietary|diagnosis|legal advice|lawsuit|tax advice|investment advice)\b",
    re.I,
)


def load_config(path=HERE / "pipeline_config.json"):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError("%s line %d is broken JSON" % (path, line_number)) from exc
    return rows


def append_jsonl(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def words(text):
    return WORD_RE.findall(str(text))


def normalized(text):
    return " ".join(word.lower() for word in words(text))


def word_set(outline):
    value = str(outline.get("title", ""))
    return {word for word in normalized(value).split() if len(word) > 2}


def jaccard(first, second):
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def outline_problems(outline, existing, config):
    problems = []
    required = ("domain", "title", "scenario", "user_persona", "assistant_mode", "goal", "resolution", "facts", "skills")
    missing = [key for key in required if not outline.get(key)]
    if missing:
        return ["missing " + ", ".join(missing)]
    if outline["domain"] not in config["domains"]:
        problems.append("unknown domain")
    if HIGH_RISK_TOPIC_RE.search("%s %s" % (outline.get("title", ""), outline.get("scenario", ""))):
        problems.append("high-risk topic is outside this corpus")
    if "remote text helper" not in str(outline.get("assistant_mode", "")).lower():
        problems.append("assistant mode must describe a remote text helper")
    if not isinstance(outline["facts"], list) or not 2 <= len(outline["facts"]) <= 6:
        problems.append("facts must contain 2-6 items")
    if not isinstance(outline["skills"], list) or not 2 <= len(outline["skills"]) <= 3:
        problems.append("skills must contain 2-3 items")
    elif any(skill not in config["conversation_skills"] for skill in outline["skills"]):
        problems.append("unknown conversation skill")
    if len(words(outline["scenario"])) < 8 or len(words(outline["goal"])) < 4:
        problems.append("scenario or goal is too thin")
    limits = {
        "title": (3, 12), "scenario": (10, 100), "user_persona": (6, 60),
        "assistant_mode": (6, 100), "goal": (6, 40), "resolution": (6, 50),
    }
    for key, (minimum, maximum) in limits.items():
        count = len(words(outline[key]))
        if not minimum <= count <= maximum:
            problems.append("%s must contain %d-%d words" % (key, minimum, maximum))
    for key in ("scenario", "goal", "resolution"):
        if not str(outline.get(key, "")).rstrip().endswith((".", "!", "?")):
            problems.append("%s must end with a complete sentence" % key)
    if isinstance(outline.get("facts"), list) and any(not 3 <= len(words(fact)) <= 40 for fact in outline["facts"]):
        problems.append("each fact must contain 3-40 words")
    if isinstance(outline.get("facts"), list) and any(
        not str(fact).rstrip().endswith((".", "!", "?")) for fact in outline["facts"]
    ):
        problems.append("each fact must end with a complete sentence")
    source_text = " ".join([
        str(outline.get("scenario", "")), str(outline.get("goal", "")),
        " ".join(str(fact) for fact in outline.get("facts", [])),
    ]).lower()
    source_numbers = {match.group(0).replace(" ", "") for match in NUMBER_RE.finditer(source_text)}
    resolution_numbers = {
        match.group(0).replace(" ", "")
        for match in NUMBER_RE.finditer(str(outline.get("resolution", "")).lower())
    }
    missing_numbers = sorted(resolution_numbers - source_numbers)
    if missing_numbers:
        problems.append("resolution introduces unsupported numbers: %s" % ", ".join(missing_numbers))
    title = normalized(outline["title"])
    if any(normalized(item["title"]) == title for item in existing):
        problems.append("duplicate title")
    candidate = word_set(outline)
    for item in existing:
        if jaccard(candidate, word_set(item)) >= config["outline_similarity_limit"]:
            problems.append("near-duplicate topic")
            break
    return problems


def message_length_band(message, config):
    count = len(words(message["content"]))
    if count <= config["short_max_words"]:
        return "short"
    if count <= config["medium_max_words"]:
        return "medium"
    return "long"


def conversation_problems(conversation, outline, config):
    problems = []
    messages = conversation.get("messages")
    if not isinstance(messages, list):
        return ["messages is not a list"]
    expected = int(outline["message_count"])
    if len(messages) != expected:
        problems.append("expected %d messages, got %d" % (expected, len(messages)))
    if not config["min_messages"] <= len(messages) <= config["max_messages"]:
        problems.append("message count is outside the configured range")
    seen = set()
    bands = Counter()
    word_counts = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append("message %d is not an object" % index)
            continue
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            problems.append("message %d should have role %s" % (index, expected_role))
        content = str(message.get("content") or "").strip()
        count = len(words(content))
        if not content or count == 0:
            problems.append("message %d is empty" % index)
            continue
        word_counts.append(count)
        if count > config["long_max_words"]:
            problems.append("message %d exceeds %d words" % (index, config["long_max_words"]))
        if len(content) >= 355:
            problems.append("message %d touches the schema cap and may be truncated" % index)
        if content.endswith(("—", "-", ",", ":", ";", "...")):
            problems.append("message %d ends with an unfinished thought" % index)
        if THINK_RE.search(content):
            problems.append("message %d contains a think tag" % index)
        if STAGE_DIRECTION_RE.search(content):
            problems.append("message %d contains stage directions" % index)
        lowered = content.lower()
        if any(phrase in lowered for phrase in GENERIC_AI):
            problems.append("message %d contains generic AI boilerplate" % index)
        if any(phrase in lowered for phrase in META_CHAT) or "```" in content:
            problems.append("message %d contains generation meta-talk" % index)
        if expected_role == "assistant" and REMOTE_ACTION_RE.search(content):
            problems.append("message %d makes the remote assistant perform an outside action" % index)
        key = normalized(content)
        if key in seen:
            problems.append("message %d repeats an earlier message" % index)
        seen.add(key)
        bands[message_length_band({"content": content}, config)] += 1
    for band in ("short", "medium", "long"):
        minimum = config["min_%s_messages" % band]
        if bands[band] < minimum:
            problems.append("needs %d %s messages, got %d" % (minimum, band, bands[band]))
    if word_counts and max(word_counts) - min(word_counts) < 10:
        problems.append("message lengths do not vary enough")
    return problems


def conversation_hash(conversation):
    canonical = json.dumps(conversation.get("messages", []), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def corpus_report(conversations, config):
    domains = Counter()
    skills = Counter()
    lengths = Counter()
    openings = Counter()
    hashes = Counter()
    messages = 0
    word_counts = []
    problems = []
    for row in conversations:
        outline = row.get("outline") or {}
        domains[outline.get("domain", "missing")] += 1
        skills.update(outline.get("skills") or [])
        hashes[conversation_hash(row)] += 1
        current = conversation_problems(row, outline, config)
        if current:
            problems.append({"id": row.get("id"), "problems": current})
        for message in row.get("messages") or []:
            messages += 1
            count = len(words(message.get("content", "")))
            word_counts.append(count)
            lengths[message_length_band(message, config)] += 1
            if message.get("role") == "assistant":
                openings[" ".join(normalized(message.get("content", "")).split()[:4])] += 1
    duplicate_conversations = sum(count - 1 for count in hashes.values() if count > 1)
    common_openings = [{"opening": key, "count": count} for key, count in openings.most_common(20) if key]
    return {
        "conversations": len(conversations),
        "messages": messages,
        "assistant_targets": sum(len(row.get("messages", [])) // 2 for row in conversations),
        "domains": dict(sorted(domains.items())),
        "skills": dict(sorted(skills.items())),
        "length_bands": dict(sorted(lengths.items())),
        "message_words": {
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "mean": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
        },
        "duplicate_conversations": duplicate_conversations,
        "structural_failures": problems[:50],
        "common_assistant_openings": common_openings,
    }
