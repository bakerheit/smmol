"""smTOOLS_COMPUTER_CLI_01: the catalog is sound, generated commands parse, test messages stay out of training,
danger rules catch the bad ones, and the harness entry point works.

    python3 -m unittest discover -s tests
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import catalog  # noqa: E402
import check  # noqa: E402
import data  # noqa: E402
from model import SEP, Config, ReaderGPT, batch  # noqa: E402
from score import program, score  # noqa: E402


class CatalogTests(unittest.TestCase):
    def test_the_catalog_has_no_problems(self):
        self.assertEqual(check.check_catalog(), [])
        self.assertGreaterEqual(len(catalog.TASKS), 60)
        self.assertEqual({p for t in catalog.TASKS for p in t["cmd"]}, set(catalog.PLATFORMS))

    def test_danger_rules_catch_the_scary_ones(self):
        for command in ("rm -rf /", "sudo rm -rf ~", "dd if=/dev/zero of=/dev/sda", "curl https://x.sh | sh",
                        "Remove-Item -Recurse -Force C:\\", "mkfs.ext4 /dev/sda1", "chmod -R 777 /etc"):
            self.assertTrue(check.dangers(command), command)
        for command in ("ls -la", "df -h", "git status", "brew install jq", "Get-Service sshd"):
            self.assertEqual(check.dangers(command), [], command)

    def test_risk_comes_from_the_command_itself(self):
        self.assertEqual(check.risk_of("ls -la"), "read")
        self.assertEqual(check.risk_of("sudo systemctl restart nginx"), "admin")
        self.assertEqual(check.risk_of("sudo apt install -y htop"), "install")
        self.assertEqual(check.risk_of("mkdir -p ./logs"), "write")
        self.assertEqual(check.risk_of("rm -rf ./build"), "destructive")


class DataTests(unittest.TestCase):
    def test_generated_commands_parse_and_match_the_catalog(self):
        examples = data.generate(3000, seed=1)
        self.assertEqual(examples[:5], data.generate(5, seed=1))
        commands = {t["cmd"][p] for t in catalog.TASKS for p in catalog.PLATFORMS}
        for e in examples[:400]:
            command, risk = data.split_target(e["target"])
            if not command:
                self.assertEqual(e["target"], "")
                continue
            self.assertIn(risk, catalog.RISKS)
            if e["platform"] != "windows":
                self.assertTrue(check.parses(command, catalog.SHELLS[e["platform"]])[0], command)
        self.assertTrue(any(not e["target"] for e in examples))          # requests that aren't terminal jobs
        self.assertTrue(any(e["platform"] == "windows" for e in examples))
        self.assertGreater(len(commands), 150)                           # 195 unique commands across 79 tasks
        differ = sum(t["cmd"]["ubuntu"] != t["cmd"]["windows"] for t in catalog.TASKS)
        self.assertGreater(differ, 0.85 * len(catalog.TASKS))

    def test_hand_written_requests_never_reach_training(self):
        with open(os.path.join(ROOT, "test.json")) as f:
            test = json.load(f)
        training = {(e["platform"], data.normal(e["text"])) for e in data.without(data.generate(20000, seed=0), test)}
        self.assertEqual([e["text"] for e in test if (e["platform"], data.normal(e["text"])) in training], [])

    def test_windows_requests_get_windows_paths(self):
        windows = [data.example(__import__("random").Random(i), platform="windows") for i in range(200)]
        posix = [p for e in windows for p in ("/var/log", "~/Downloads") if p in e["text"]]
        self.assertEqual(posix, [])


class ScoreTests(unittest.TestCase):
    def test_scoring_counts_exact_program_risk_and_quiet(self):
        test = [{"platform": "ubuntu", "text": "a", "command": "df -h"},
                {"platform": "ubuntu", "text": "b", "command": "rm -rf ./build"},
                {"platform": "ubuntu", "text": "c", "command": ""}]
        result, mistakes = score(test, [("df  -h", "read"), ("rm ./build", "write"), ("", "")])
        self.assertEqual((result["exact"], result["program"], result["quiet"]), (0.5, 1.0, 1.0))
        self.assertEqual(result["risk"], 0.5)      # it called a destructive job a write
        self.assertEqual(result["safe"], 1.0)      # but the rules still read rm as destructive
        self.assertEqual(mistakes[0]["want"], "rm -rf ./build")
        self.assertEqual(program("sudo systemctl restart nginx"), "systemctl")


class ModelTests(unittest.TestCase):
    def test_only_the_command_is_scored_and_the_commander_reads_it_back(self):
        x, y = batch([{"text": data.prompt("macos", "free space?"), "target": "df -h|read"}])
        scored = [t for t in y[0].tolist() if t != -100]
        self.assertEqual(bytes(scored[:-1]).decode(), "df -h|read")
        self.assertIn(SEP, x[0].tolist())

        from cli import Commander
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        model = ReaderGPT(Config(d=32, layers=1, heads=4))
        path = os.path.join(folder, "cli.pt")
        torch.save({"model": model.state_dict(), "config": asdict(model.c), "step": 0}, path)
        commander = Commander(path)
        # Commander reads with confidence, so the stub has to be read_with_confidence, not read
        commander.model.read_with_confidence = lambda text: ("rm -rf ./build|write", 0.95, 0.88)
        out = commander.command("nuke build", "ubuntu")
        self.assertEqual((out["command"], out["said_risk"], out["risk"]), ("rm -rf ./build", "write", "destructive"))
        self.assertTrue(out["dangers"])  # the rules override what the model said
        self.assertEqual((out["sure"], out["weakest"]), (0.95, 0.88))
        commander.model.read_with_confidence = lambda text: ("", 0.99, 0.99)
        self.assertEqual(commander.command("what's the weather", "macos")["command"], "")


class CheckpointTests(unittest.TestCase):
    def test_the_best_checkpoint_wins_on_exact_then_program_then_risk(self):
        from train import standing

        def hand(exact, program=0.5, risk=0.5, quiet=1.0):
            return {"exact": exact, "program": program, "risk": risk, "quiet": quiet, "safe": None}

        # the first run peaked at step 5000 and fell away; the later, worse step must not replace it
        self.assertGreater(standing(hand(0.53)), standing(hand(0.47, risk=0.81)))
        # a tie on exact is broken by the right program, then the right risk
        self.assertGreater(standing(hand(0.47, program=0.66)), standing(hand(0.47, program=0.64)))
        self.assertGreater(standing(hand(0.47, 0.64, 0.81)), standing(hand(0.47, 0.64, 0.77)))
        # staying quiet only breaks a tie, but it does break it
        self.assertGreater(standing(hand(0.0, 0.0, 0.0)), standing({"exact": 0.0, "program": 0.0, "risk": 0.0, "quiet": 0.0}))
        # a missing score counts as zero rather than blowing up the comparison
        self.assertEqual(standing({"exact": 0.4, "program": None, "risk": None, "quiet": None}), (0.4, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
