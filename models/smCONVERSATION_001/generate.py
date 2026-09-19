"""Generate, structurally validate, review, and accept Ministral conversations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

from conversation_data import (
    HERE, append_jsonl, conversation_hash, conversation_problems, corpus_report,
    load_config, outline_problems, read_jsonl, STAGE_DIRECTION_RE, write_json,
)
from teacher import Teacher


OUTLINES = HERE / "data" / "raw" / "outlines.jsonl"
GENERATED = HERE / "data" / "raw" / "generated.jsonl"
OUTLINE_REJECTIONS = HERE / "data" / "review" / "outline_rejections.jsonl"
REJECTIONS = HERE / "data" / "review" / "rejections.jsonl"
ACCEPTED = HERE / "data" / "accepted" / "conversations.jsonl"
CALLS = HERE / "data" / "logs" / "teacher_calls.jsonl"
MANIFEST = HERE / "data" / "manifest.json"

DOMAIN_SKILLS = {
    "creative_brainstorming": ["brainstorming", "clarification", "summarization"],
    "decision_making": ["compare_options", "clarification", "summarization"],
    "entertainment_discussion": ["compare_options", "callback_to_earlier_detail", "summarization"],
    "everyday_planning": ["multi_step_planning", "clarification", "callback_to_earlier_detail"],
    "event_planning": ["multi_step_planning", "compare_options", "summarization"],
    "family_scheduling": ["multi_step_planning", "clarification", "emotional_attunement"],
    "game_and_story_ideas": ["brainstorming", "instruction_following", "callback_to_earlier_detail"],
    "hobby_projects": ["multi_step_planning", "troubleshooting", "clarification"],
    "home_organization": ["multi_step_planning", "compare_options", "callback_to_earlier_detail"],
    "learning_reflection": ["emotional_attunement", "question_answering", "summarization"],
    "personal_preferences": ["clarification", "compare_options", "summarization"],
    "shopping_comparisons": ["compare_options", "clarification", "summarization"],
    "social_communication": ["emotional_attunement", "polite_disagreement", "summarization"],
    "software_project_planning": ["multi_step_planning", "troubleshooting", "clarification"],
    "study_habits": ["teaching", "multi_step_planning", "callback_to_earlier_detail"],
    "time_management": ["multi_step_planning", "clarification", "summarization"],
    "travel_preferences": ["compare_options", "clarification", "callback_to_earlier_detail"],
    "volunteer_coordination": ["multi_step_planning", "instruction_following", "summarization"],
    "workplace_communication": ["polite_disagreement", "emotional_attunement", "summarization"],
    "writing_revision": ["correction_and_repair", "instruction_following", "summarization"],
}

SKILL_GUIDANCE = {
    "brainstorming": "Offer several distinct options before narrowing them with the user.",
    "callback_to_earlier_detail": "Later reuse one specific preference or constraint stated earlier.",
    "clarification": "Ask an early, useful question before settling on advice; one strong clarification is enough.",
    "compare_options": "Compare the main options using the user's actual constraints and name the trade-offs.",
    "correction_and_repair": "Acknowledge a user correction and visibly update the advice.",
    "emotional_attunement": "Briefly recognize the user's feeling without turning the chat into therapy.",
    "honest_uncertainty": "State what is uncertain and avoid inventing a confident answer.",
    "instruction_following": "Follow every explicit format, limit, and exclusion the user gives.",
    "multi_step_planning": "Build a concrete ordered plan whose steps fit the supplied constraints.",
    "polite_disagreement": "Push back gently on one weak choice and explain why using known facts.",
    "question_answering": "Answer the user's actual questions directly before adding useful context.",
    "summarization": "Near the end, recap the decision and its two or three main reasons.",
    "teaching": "Explain one idea in plain language and check that it helps.",
    "topic_shift": "Move naturally to a related subproblem that helps finish the same goal.",
    "troubleshooting": "Test likely causes in a sensible order and adapt to the reported result.",
}


def skills_for_domain(domain):
    return list(DOMAIN_SKILLS[domain])


def skill_instructions(skills):
    return "\n".join("- %s: %s" % (skill, SKILL_GUIDANCE[skill]) for skill in skills)


def array_schema(items, count):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": items,
            }
        },
    }


def outline_schema(config, count):
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["domain", "title"],
        "properties": {
            "domain": {"type": "string", "enum": config["domains"]},
            "title": {"type": "string"},
        },
    }
    return array_schema(item, count)


def canonical_outline(raw):
    title = " ".join(str(raw["title"]).split()).strip(" .")
    title = title[:240]
    title = title.replace("/", " or ")
    title_words = title.split()[:12]
    trailing = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
    while title_words and (title_words[-1].lower().strip("'\"") in trailing or title_words[-1].isdigit()):
        title_words.pop()
    title = " ".join(title_words).strip(" .,:;-_")
    return {
        "domain": raw["domain"],
        "title": title,
        "scenario": "The user wants help with %s and will share the relevant preferences and constraints." % title.lower(),
        "user_persona": "A person with a specific everyday goal and clear personal preferences.",
        "assistant_mode": "Remote text helper that asks useful questions and stays grounded in the chat.",
        "goal": "Reach a practical decision or plan that fits the user's stated needs.",
        "resolution": "The user leaves with a clear next step based only on details established in the conversation.",
        "facts": [
            "The user supplies the relevant personal preferences and constraints during the conversation.",
            "The assistant has no live access and must not invent outside facts.",
        ],
        "skills": skills_for_domain(raw["domain"]),
    }


def message_schema(count):
    utterance = {"type": "string", "minLength": 1, "maxLength": 360}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["turns"],
        "properties": {
            "turns": {
                "type": "array", "minItems": count // 2, "maxItems": count // 2,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["user", "assistant"],
                    "properties": {
                        "user": utterance,
                        "assistant": utterance,
                    },
                },
            }
        },
    }


def review_schema():
    score = {"type": "integer", "minimum": 1, "maximum": 5}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["accept", "reasons", "scores"],
        "properties": {
            "accept": {"type": "boolean"},
            "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "scores": {
                "type": "object", "additionalProperties": False,
                "required": ["coherence", "naturalness", "factuality", "skill_coverage"],
                "properties": {
                    "coherence": score,
                    "naturalness": score,
                    "factuality": score,
                    "skill_coverage": score,
                },
            },
        },
    }


def requested_domains(existing, config, count):
    totals = Counter(item["domain"] for item in existing)
    indexed = {domain: index for index, domain in enumerate(config["domains"])}
    ranked = sorted(config["domains"], key=lambda domain: (totals[domain], indexed[domain]))
    return [ranked[index % len(ranked)] for index in range(count)]


def generate_outlines(teacher, config, target):
    outlines = read_jsonl(OUTLINES)
    rejected = Counter()
    attempts = 0
    while len(outlines) < target:
        count = min(config["outline_batch"], target - len(outlines))
        domains = requested_domains(outlines, config, count)
        existing_titles = [item["title"] for item in outlines[-120:]]
        prompt = """Create %d distinct topic titles for a conversational language-model corpus.

Requested domains, one plan for each entry in this exact order:
%s

Return one concrete, low-risk everyday topic for each requested domain, in the exact order given. Each title must be
4-10 words and specific enough to inspire a full conversation. Describe the user's task, choice, project, discussion,
or revision—not a broad subject. Keep topics meaningfully different from each other. Avoid current events, private
real-person data, medical/legal/financial directives, specialist safety advice, and scenarios that only swap a city,
food, or person's name.

Good titles: "Planning a quiet apartment book swap"; "Debugging a flaky local test".
Bad titles: "Books"; "Help me"; "Weekend plans in Chicago".

Recent accepted titles that must not be repeated or lightly paraphrased:
%s

Return only the requested JSON.""" % (
            count,
            json.dumps(domains, ensure_ascii=False),
            json.dumps(existing_titles, ensure_ascii=False),
        )
        seed = config["seed"] + 100000 + attempts
        reply = teacher.ask(
            "conversation_outlines",
            "You design varied, realistic conversation datasets. Specificity and diversity matter more than cleverness.",
            prompt, outline_schema(config, count), seed, 0.8, max(200, min(1200, 120 * count)),
            {"target": target, "existing": len(outlines), "domains": domains},
        )
        candidates = reply.get("items") if isinstance(reply, dict) else None
        if not isinstance(candidates, list):
            raise RuntimeError("outline teacher returned no items")
        added = 0
        for raw in candidates:
            if len(outlines) >= target or not isinstance(raw, dict):
                continue
            candidate = canonical_outline(raw)
            problems = outline_problems(candidate, outlines, config)
            if problems:
                append_jsonl(OUTLINE_REJECTIONS, {
                    "candidate": candidate, "problems": problems, "seed": seed,
                })
                rejected.update(problems)
                continue
            serial = len(outlines) + 1
            choices = (config["max_messages"] - config["min_messages"]) // 2 + 1
            candidate["id"] = "conversation-%04d" % serial
            candidate["message_count"] = config["min_messages"] + 2 * ((serial - 1) % choices)
            candidate["generation"] = {"model": config["model"], "seed": seed, "prompt_version": 9}
            append_jsonl(OUTLINES, candidate)
            outlines.append(candidate)
            added += 1
        attempts += 1
        print("outlines | asked %d | kept %d | %d/%d%s" % (
            len(candidates), added, len(outlines), target,
            " | rejects %s" % dict(rejected.most_common(4)) if added == 0 else "",
        ), flush=True)
        if attempts > target * 3 or (attempts > 10 and added == 0):
            raise RuntimeError("outline generation stalled: %s" % dict(rejected))
    return outlines


def conversation_part_prompt(outline, config, feedback, prior_turns, part_number, pair_count):
    required_skills = outline["skills"]
    part_short = max(1, config["min_short_messages"] // 2)
    part_medium = max(2, config["min_medium_messages"] // 2)
    part_long = max(2, config["min_long_messages"] // 2)
    phase = (
        "This is part 1 of 2. Establish the need, demonstrate any required clarification or initial comparison, and begin useful progress. Do not resolve it yet."
        if part_number == 1 else
        "This is part 2 of 2. Continue directly from the prior messages, finish every required skill not yet shown, and make the final assistant message land on the plan's resolution. Do not restart or add a new closing question."
    )
    finish_rules = []
    if "summarization" in required_skills:
        finish_rules.append("The final assistant reply must briefly recap the decision and its main reasons.")
    if "multi_step_planning" in required_skills:
        finish_rules.append("Include a clear ordered sequence of at least three practical steps before the close.")
    if "polite_disagreement" in required_skills:
        finish_rules.append("At one point, explicitly acknowledge the user's view and gently explain a different recommendation.")
    return """Write exactly %d user/assistant turn pairs for one part of a conversation from this plan:
%s

Prior turn pairs (empty for part 1):
%s

Phase:
%s

Required conversation skills (show each clearly and naturally):
%s
%s

Hard requirements:
- Exactly %d turn pairs, each containing one user message then one assistant message.
- Keep the user's persona and the assistant's mode distinct and natural.
- The assistant is a remote text helper. It never claims a shared past, personal relationship, physical presence,
  phone call, purchase, job, or access to people and places in the scenario.
- Write plain chat messages only. Do not role-play physical actions, use stage directions in asterisks or brackets,
  label turns, narrate body language, include markdown fences, or talk about the JSON task.
- The conversation needs an arc: establish the need, make progress, adapt to the user, and reach a useful close.
  Do not force a correction, callback, or topic shift unless it is listed as a required skill.
- Use only the listed scenario facts as given facts. Do not invent deadlines, measurements, prices, personal history,
  causes, policies, or product capabilities. A speaker may state a new preference or observation, but the assistant
  must not turn it into a verified external fact. Do not claim live/current knowledge.
- Never call ordinary dairy milk lactose-free, label food allergen-safe without checking its package, or give meat
  cooking time as proof of safety without a doneness check. Avoid hard technical classifications unless the plan
  supplies them and they are plainly correct.
- Move naturally toward the plan's resolution. Do not substitute a different technical solution.
- Every message must be a complete thought of one or two sentences. Keep most turns between 4 and 30 words; a few
  useful replies may reach %d words. Stay well below 350 characters. Never chop a sentence, trail off, or end with a
  dash, comma, colon, semicolon, or ellipsis.
- In this part, include at least %d short messages (8 words or fewer), %d medium messages (9-18 words), and %d long
  messages (19-%d words). Spread them across both speakers instead of clustering them.
- Vary message length naturally: the shortest and longest messages must differ by at least 10 words.
- Vary openings and sentence shapes. Do not use AI boilerplate, stage directions, analysis tags, or mention this dataset.
- A short reply must still make sense in context; a long reply must earn its length.
- Resolve the user's stated goal by the final assistant reply. Advice may be conditional, but the ending cannot add a
  brand-new question or reopen the whole problem.
%s

Return turn pairs as {"turns":[{"user":"...","assistant":"..."}]}, only in the requested JSON.""" % (
        pair_count,
        json.dumps(outline, ensure_ascii=False, indent=1),
        json.dumps(prior_turns, ensure_ascii=False, indent=1),
        phase,
        skill_instructions(required_skills),
        "\n".join("- " + rule for rule in finish_rules),
        pair_count,
        config["long_max_words"],
        part_short,
        part_medium,
        part_long,
        config["long_max_words"],
        ("A previous draft was rejected. Fix these concrete issues: " + "; ".join(feedback)) if feedback else "",
    )


def review_prompt(record, config):
    return """Review this synthetic conversation as a practical dataset editor.

Plan:
%s

Conversation:
%s

The assistant is a remote text helper, not a person inside the scenario. Treat claims of shared memories, relatives,
physical presence, or actions outside the chat as material persona failures.

Judge factuality against both the supplied plan and basic common knowledge; the plan itself may contain a mistake.
Reject clear contradictions or clearly false claims, but do not
invent specialist objections or reject reasonable uncertainty over debatable details. A helper may recommend that the
user contact someone; that is not the helper performing the action. Ordinary paraphrases of the plan are fine. Reject it for any material
problem: contradiction, unsupported scenario fact, clear factual error, unsafe directive, unnatural self-dialogue,
repeated filler, assistant dominating every turn, broken persona,
generic advice that ignores the user, a missing required skill from the plan, or an ending that does not resolve the arc. Do not reward mere grammaticality.
Do not reject synonyms, harmless elaboration introduced naturally by either speaker, minor style preferences, or a
missed optional opportunity. "correction_and_repair" may be the assistant acknowledging and adapting to a user-corrected
detail; it does not require an external factual error. "topic_shift" means moving to a related subproblem, not abandoning
the user's chosen direction. "clarification" can occur once and need not chase every imaginable ambiguity. Do not demand
extra alternatives, quantification, or specialist detail absent from the plan. The skills are coverage goals, not a
box-checking script: score 3 when most are present and 4 when all are handled naturally. Accept solid usable data even
when it is not perfect. Every rejection reason
must quote a concrete material flaw from the conversation.
The assistant may recommend that the user try, test, check, download, draft, contact, or physically do something; that
is normal text advice, not the assistant claiming an outside capability. Do not demand every imaginable clarification,
measurement, tool, schedule, or implementation detail. One useful clarification can satisfy the skill. Do not reject
reasonable user-provided details merely because they first appear during the conversation. Give factuality below 4 only
for a concrete material falsehood, contradiction, unsafe instruction, or invented claim—not for an optional omission,
debatable generalization, or a harmless recommendation. Remain strict about actual medical, allergen, legal, and safety claims.
Creative suggestions, names, examples, metaphors, hypothetical options, and proposed plans are not factual claims and
do not need to appear in the plan first. Brainstorming and planning assistants are expected to introduce useful new
ideas. Treat ordinary user approval such as "that works," "perfect," or "I like it" as confirmation; do not demand a
formal declaration. The facts constraint applies to claims about the external world, not to clearly framed suggestions.
For factuality, explicitly check technical labels, dietary claims, invented capabilities, quantities, and cooking
safety. Ordinary dairy milk is not lactose-free. A cooking time alone does not prove meat is safely cooked. Never call
a recommendation unsafe merely because it was not the user's first preference or because it lacks an optional example.
Score coherence, naturalness, factuality, and required-skill coverage from 1 to 5. Accept when coherence, naturalness,
and skill coverage are at least %d, factuality is at least %d, and the average is at least %.2f. Give short, actionable reasons when rejecting. Return
only the requested JSON.""" % (
        json.dumps(record["outline"], ensure_ascii=False, indent=1),
        json.dumps(record["messages"], ensure_ascii=False, indent=1),
        config["review_min_score"],
        config["review_min_factuality"],
        config["review_min_average"],
    )


def accepted_review(review, config):
    if not isinstance(review, dict):
        return False
    scores = review.get("scores") or {}
    values = [scores.get(key, 0) for key in (
        "coherence", "naturalness", "factuality", "skill_coverage"
    )]
    return (
        scores.get("factuality", 0) >= config["review_min_factuality"]
        and all(scores.get(key, 0) >= config["review_min_score"] for key in (
            "coherence", "naturalness", "skill_coverage"
        ))
        and sum(values) / len(values) >= config["review_min_average"]
    )


def flatten_turns(turns):
    messages = []
    if not isinstance(turns, list):
        return None
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        messages.append({"role": "user", "content": turn.get("user")})
        messages.append({"role": "assistant", "content": turn.get("assistant")})
    return messages


def sanitize_messages(messages):
    """Remove harmless role-play gestures while keeping raw generations on disk."""
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue
        content = STAGE_DIRECTION_RE.sub(" ", str(message.get("content") or ""))
        cleaned.append({"role": message.get("role"), "content": " ".join(content.split())})
    return cleaned


def write_manifest(config):
    accepted = read_jsonl(ACCEPTED)
    raw = ACCEPTED.read_bytes() if ACCEPTED.exists() else b""
    manifest = {
        "format_version": 1,
        "model": "smCONVERSATION_001",
        "teacher": config["model"],
        "endpoint": config["endpoint"],
        "prompt_version": 9,
        "normalization": {"strip_stage_directions": True, "version": 1},
        "sha256": hashlib.sha256(raw).hexdigest(),
        "report": corpus_report(accepted, config),
        "files": {
            "outlines": str(OUTLINES.relative_to(HERE)),
            "outline_rejections": str(OUTLINE_REJECTIONS.relative_to(HERE)),
            "generated": str(GENERATED.relative_to(HERE)),
            "rejections": str(REJECTIONS.relative_to(HERE)),
            "accepted": str(ACCEPTED.relative_to(HERE)),
            "teacher_calls": str(CALLS.relative_to(HERE)),
        },
    }
    write_json(MANIFEST, manifest)
    return manifest


def generate_conversations(teacher, config, target):
    buffer = 3 if target < 5 else max(15, target // 5)
    outline_target = min(config["target_outlines"], target + buffer)
    outlines = generate_outlines(teacher, config, outline_target)
    accepted = read_jsonl(ACCEPTED)
    accepted_ids = {item["id"] for item in accepted}
    accepted_hashes = {conversation_hash(item) for item in accepted}
    rejections = read_jsonl(REJECTIONS)
    generated = read_jsonl(GENERATED)
    generated_by_key = {(item["id"], item["attempt"]): item for item in generated}
    rejected_by_key = {}
    for item in rejections:
        key = (item["id"], item["attempt"])
        raw = generated_by_key.get(key)
        if item.get("stage") == "structure" and raw is not None:
            cleaned_raw = dict(raw, messages=sanitize_messages(raw.get("messages")))
            if not conversation_problems(cleaned_raw, raw["outline"], config):
                continue
        if item.get("stage") == "review" and accepted_review(item.get("review"), config):
            continue
        rejected_by_key[key] = item

    for outline in outlines:
        if len(accepted) >= target:
            break
        if outline["id"] in accepted_ids:
            continue
        feedback = []
        earlier = [item for item in rejections if item["id"] == outline["id"]]
        if earlier:
            feedback = earlier[-1].get("problems") or earlier[-1].get("review", {}).get("reasons") or []
        for attempt in range(1, config["generation_attempts_per_outline"] + 1):
            key = (outline["id"], attempt)
            if key in rejected_by_key:
                prior = rejected_by_key[key]
                feedback = prior.get("problems") or prior.get("review", {}).get("reasons") or feedback
                continue
            record = generated_by_key.get(key)
            if record is None:
                seed = config["seed"] + 200000 + int(outline["id"].split("-")[-1]) * 10 + attempt
                total_pairs = outline["message_count"] // 2
                part_sizes = [total_pairs // 2, total_pairs - total_pairs // 2]
                prior_turns = []
                for part_number, pair_count in enumerate(part_sizes, 1):
                    reply = teacher.ask(
                        "conversation_dialogue_part",
                        "You write concise, realistic two-person chat. Finish every message cleanly and keep the remote helper inside the chat.",
                        conversation_part_prompt(
                            outline, config, feedback, prior_turns, part_number, pair_count
                        ),
                        message_schema(pair_count * 2), seed + part_number * 1000000, 0.65, 1600,
                        {"id": outline["id"], "attempt": attempt, "part": part_number},
                    )
                    part_turns = reply.get("turns") if isinstance(reply, dict) else None
                    if not isinstance(part_turns, list):
                        prior_turns = None
                        break
                    prior_turns.extend(part_turns)
                record = {
                    "id": outline["id"],
                    "attempt": attempt,
                    "outline": outline,
                    "messages": flatten_turns(prior_turns),
                    "generation": {"model": config["model"], "seed": seed, "prompt_version": 9},
                }
                append_jsonl(GENERATED, record)
                generated_by_key[key] = record
            candidate_record = dict(record, messages=sanitize_messages(record.get("messages")))
            problems = conversation_problems(candidate_record, outline, config)
            digest = conversation_hash(candidate_record)
            if digest in accepted_hashes:
                problems.append("duplicates an accepted conversation")
            if problems:
                rejection = {"id": outline["id"], "attempt": attempt, "stage": "structure", "problems": problems}
                append_jsonl(REJECTIONS, rejection)
                rejections.append(rejection)
                rejected_by_key[key] = rejection
                feedback = problems
                print("reject %s attempt %d | structure | %s" % (
                    outline["id"], attempt, "; ".join(problems[:3])
                ), flush=True)
                continue
            review_seed = config["seed"] + 300000 + int(outline["id"].split("-")[-1]) * 10 + attempt
            review = teacher.ask(
                "conversation_review",
                "You are a balanced, evidence-based conversation-dataset reviewer. Accept solid usable chats and judge only what is actually written.",
                review_prompt(candidate_record, config), review_schema(), review_seed, 0, 1000,
                {"id": outline["id"], "attempt": attempt},
            )
            if not accepted_review(review, config):
                rejection = {"id": outline["id"], "attempt": attempt, "stage": "review", "review": review}
                append_jsonl(REJECTIONS, rejection)
                rejections.append(rejection)
                rejected_by_key[key] = rejection
                feedback = review.get("reasons") or ["review scores were below four"]
                print("reject %s attempt %d | review | %s" % (
                    outline["id"], attempt, "; ".join(feedback[:3])
                ), flush=True)
                continue
            accepted_record = dict(candidate_record)
            accepted_record["review"] = review
            append_jsonl(ACCEPTED, accepted_record)
            accepted.append(accepted_record)
            accepted_ids.add(outline["id"])
            accepted_hashes.add(digest)
            print("accept %s | %d/%d | %d messages" % (
                outline["id"], len(accepted), target, len(candidate_record["messages"])
            ), flush=True)
            break
    manifest = write_manifest(config)
    if len(accepted) < target:
        raise RuntimeError(
            "accepted %d/%d after exhausting %d outlines; inspect rejections before extending the cap"
            % (len(accepted), target, len(outlines))
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "pipeline_config.json")
    parser.add_argument("--target", type=int, help="accepted target; default is the 25-conversation pilot")
    args = parser.parse_args()
    config = load_config(args.config)
    target = args.target if args.target is not None else config["pilot_accepted"]
    if not 1 <= target <= config["target_accepted"]:
        parser.error("target must be between 1 and %d" % config["target_accepted"])
    teacher = Teacher(config["endpoint"], config["model"], CALLS)
    started = time.time()
    manifest = generate_conversations(teacher, config, target)
    print(json.dumps(manifest["report"], indent=2, sort_keys=True))
    print("done in %.1f minutes" % ((time.time() - started) / 60), file=sys.stderr)


if __name__ == "__main__":
    main()
