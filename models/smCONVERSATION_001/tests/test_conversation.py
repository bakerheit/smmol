import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from conversation_data import conversation_problems, corpus_report, load_config, outline_problems
from model import Config, LanguageModel
from generate import accepted_review, canonical_outline, sanitize_messages, skills_for_domain
from train import initialize_from_grade_one
from training_data import build_sets, conversation_examples, encode_supervised


CONFIG = load_config()


def outline(identifier="conversation-0001", domain="creative_brainstorming", count=20):
    return {
        "id": identifier,
        "domain": domain,
        "title": "Repairing a faded neighborhood mural",
        "scenario": "A volunteer asks how to repair a faded outdoor mural before a small community event.",
        "user_persona": "A practical volunteer who knows paint but not conservation.",
        "assistant_mode": "Remote text helper who explains art tradeoffs plainly and stays inside the chat.",
        "goal": "Choose a safe repair plan and divide the work across two weekends.",
        "resolution": "Inspect a small test patch first, then schedule the repair across the two available weekends.",
        "facts": ["The wall is brick.", "The original paint is acrylic."],
        "skills": ["clarification", "multi_step_planning", "callback_to_earlier_detail"],
        "message_count": count,
    }


def valid_messages(count=20):
    rows = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        if index < 4:
            text = "Short reply %d." % index
        elif index < 12:
            text = "This medium message has enough separate words to fit its intended length band %d." % index
        else:
            text = " ".join(["This", "long", "message", "develops", "the", "specific", "conversation", "with"] +
                            ["useful"] * 28 + ["detail", str(index)])
        rows.append({"role": role, "content": text})
    return rows


class ContractTests(unittest.TestCase):
    def test_domain_skills_are_bounded_and_known(self):
        for domain in CONFIG["domains"]:
            skills = skills_for_domain(domain)
            self.assertTrue(2 <= len(skills) <= 3)
            self.assertEqual(len(skills), len(set(skills)))
            self.assertTrue(set(skills) <= set(CONFIG["conversation_skills"]))

    def test_generated_titles_are_compacted_cleanly(self):
        plan = canonical_outline({
            "domain": "decision_making",
            "title": "Evaluating three budget-friendly outdoor gear setups for weekend hiking with a 1",
        })
        self.assertEqual(plan["title"], "Evaluating three budget-friendly outdoor gear setups for weekend hiking")
        self.assertLessEqual(len(plan["title"].split()), 12)

    def test_review_uses_scores_and_requires_strong_average(self):
        review = {
            "accept": False,
            "scores": {"coherence": 4, "naturalness": 4, "factuality": 4, "skill_coverage": 3},
        }
        self.assertTrue(accepted_review(review, CONFIG))
        review["scores"]["naturalness"] = 2
        self.assertFalse(accepted_review(review, CONFIG))

    def test_valid_conversation_passes_and_role_break_is_rejected(self):
        plan = outline()
        conversation = {"id": plan["id"], "outline": plan, "messages": valid_messages()}
        self.assertEqual(conversation_problems(conversation, plan, CONFIG), [])
        conversation["messages"][3]["role"] = "user"
        self.assertIn("message 3 should have role assistant", conversation_problems(conversation, plan, CONFIG))

    def test_outline_dedupe_catches_close_repeat(self):
        first = outline()
        second = dict(first, title="Repairing the faded neighborhood mural")
        self.assertIn("near-duplicate topic", outline_problems(second, [first], CONFIG))

    def test_outline_rejects_numbers_invented_by_resolution(self):
        plan = outline()
        plan["resolution"] = "The user chooses the option that costs $720 for 18 months."
        problems = outline_problems(plan, [], CONFIG)
        self.assertTrue(any("unsupported numbers" in problem for problem in problems))

    def test_outline_rejects_incomplete_fields(self):
        plan = outline()
        plan["resolution"] = "The user chooses the smaller option and then"
        self.assertIn("resolution must end with a complete sentence", outline_problems(plan, [], CONFIG))

    def test_outline_rejects_high_risk_topics(self):
        plan = outline()
        plan["title"] = "Planning a nut-free office lunch"
        self.assertIn("high-risk topic is outside this corpus", outline_problems(plan, [], CONFIG))

    def test_corpus_report_counts_assistant_targets(self):
        plan = outline()
        report = corpus_report([{"id": plan["id"], "outline": plan, "messages": valid_messages()}], CONFIG)
        self.assertEqual((report["conversations"], report["messages"], report["assistant_targets"]), (1, 20, 10))
        self.assertFalse(report["structural_failures"])

    def test_roleplay_meta_and_remote_actions_are_rejected(self):
        plan = outline()
        conversation = {"id": plan["id"], "outline": plan, "messages": valid_messages()}
        conversation["messages"][0]["content"] = "*kicks the chair* I guess we should start now."
        conversation["messages"][1]["content"] = "I'll call the shop and buy that for you tomorrow."
        conversation["messages"][2]["content"] = "This is the end of conversation, exactly 20 messages."
        problems = conversation_problems(conversation, plan, CONFIG)
        self.assertIn("message 0 contains stage directions", problems)
        self.assertIn("message 1 makes the remote assistant perform an outside action", problems)
        self.assertIn("message 2 contains generation meta-talk", problems)

    def test_stage_directions_are_stripped_without_touching_raw_messages(self):
        raw = [{"role": "user", "content": "Well, *laughs softly* that changed things."}]
        cleaned = sanitize_messages(raw)
        self.assertEqual(cleaned[0]["content"], "Well, that changed things.")
        self.assertIn("*laughs softly*", raw[0]["content"])

    def test_markdown_emphasis_is_not_stripped_as_stage_direction(self):
        raw = [{"role": "assistant", "content": "I like *Pixel Crust* best."}]
        self.assertEqual(sanitize_messages(raw), raw)


class TrainingTests(unittest.TestCase):
    def conversation(self, identifier, domain):
        plan = outline(identifier, domain)
        return {"id": identifier, "outline": plan, "messages": valid_messages()}

    def test_each_assistant_message_becomes_one_target(self):
        examples = conversation_examples(self.conversation("conversation-0001", "creative_brainstorming"), 1024, 6)
        self.assertEqual(len(examples), 10)
        self.assertTrue(all(example["prefix"].endswith(b"Assistant: ") for example in examples))
        inputs, targets = encode_supervised(examples[-1], 1024)
        self.assertEqual((len(inputs), len(targets)), (1024, 1024))
        self.assertTrue(all(value == -100 for value in targets[:len(examples[-1]["prefix"]) - 1]))
        self.assertTrue(any(value >= 0 for value in targets))

    def test_validation_holds_out_whole_domains(self):
        conversations = [
            self.conversation("conversation-0001", "creative_brainstorming"),
            self.conversation("conversation-0002", "study_habits"),
            self.conversation("conversation-0003", "travel_preferences"),
        ]
        school = [{"word": "cat", "sentence": "The cat naps here."}]
        train, validation, metadata = build_sets(conversations, school, 1024, seed=4, replay_fraction=0.25)
        validation_domains = set(metadata["validation_domains"])
        self.assertTrue(validation_domains)
        self.assertTrue(all(item["domain"] in validation_domains for item in validation))
        self.assertTrue(all(item["domain"] not in validation_domains or item["domain"] == "school_replay" for item in train))
        self.assertGreater(metadata["school_replay_examples"], 0)

    def test_grade_one_transfer_expands_only_positions(self):
        torch.manual_seed(4)
        base_config = Config(context_length=16, width=48, layers=1, heads=3, dropout=0)
        base = LanguageModel(base_config)
        torch.manual_seed(9)
        target_config = Config(context_length=32, width=48, layers=1, heads=3, dropout=0)
        target = LanguageModel(target_config)
        old_tail = target.position_embedding.weight[16:].detach().clone()
        transfer = initialize_from_grade_one(target, {"config": {"architecture": base_config.to_dict()}, "model": base.state_dict()})
        self.assertTrue(torch.equal(target.position_embedding.weight[:16], base.position_embedding.weight))
        self.assertTrue(torch.equal(target.position_embedding.weight[16:], old_tail))
        self.assertEqual(transfer["expanded"][0]["new_positions"], 16)


if __name__ == "__main__":
    unittest.main()
