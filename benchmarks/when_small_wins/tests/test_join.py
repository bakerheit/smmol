#!/usr/bin/env python3
"""The reconstruction must keep reproducing the published numbers, and the
normaliser must keep refusing to forgive a behaviour change.

    python3 -m unittest discover -s tests

If a test here fails after someone retrains a model, that is correct: the
published numbers moved and results.md needs updating, not this file.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import join  # noqa: E402
from normalise import same  # noqa: E402


class Reconstruction(unittest.TestCase):
    """Phase 0's gate: the item-level join must add back up to results.md."""

    @classmethod
    def setUpClass(cls):
        cls.cli = join.cli_rows()
        cls.router = join.router_rows()
        cls.mathlang = join.mathlang_rows()

    def close(self, got, want, what):
        self.assertIsNotNone(got, "%s did not reconstruct at all" % what)
        self.assertAlmostEqual(got, want, places=2, msg="%s: got %.3f, published %.3f" % (what, got, want))

    def test_cli_exact_matches_published(self):
        self.close(join.rate(self.cli, "small_right", "command"), 0.468, "cli exact, small")
        self.close(join.rate(self.cli, "llm_right", "command"), 0.149, "cli exact, 8B")

    def test_cli_quiet_matches_published(self):
        # The trap: a miss on a say-nothing item stores the literal "(nothing)" in
        # `want`. Treat that as truthy and the 8B lands at 1.000 here and 0.085 above.
        self.close(join.rate(self.cli, "small_right", "quiet"), 1.000, "cli quiet, small")
        self.close(join.rate(self.cli, "llm_right", "quiet"), 0.500, "cli quiet, 8B")

    def test_router_matches_published(self):
        self.close(join.rate(self.router, "small_right"), 0.710, "router all-three, small")
        self.close(join.rate(self.router, "llm_right"), 0.581, "router all-three, 8B")

    def test_mathlang_matches_published(self):
        self.close(join.rate(self.mathlang, "small_right"), 0.775, "mathlang every, small")

    def test_every_test_item_appears_once(self):
        for name, rows, n in (("cli", self.cli, 53), ("router", self.router, 93), ("mathlang", self.mathlang, 40)):
            self.assertEqual(len(rows), n, "%s should have %d items" % (name, n))

    def test_mathlang_union_is_perfect(self):
        """Both models wrong on zero items — the fact the hand-off idea rests on."""
        scored = [r for r in self.mathlang if r["llm_right"] is not None]
        if not scored:
            self.skipTest("no committed 8B baseline for mathlang")
        neither = [r for r in scored if not r["small_right"] and not r["llm_right"]]
        self.assertEqual(neither, [], "some item both models miss: %s" % [r["text"] for r in neither])


class Normaliser(unittest.TestCase):
    """Normalising may forgive spelling. It may never forgive a different action."""

    def test_forgives_spelling(self):
        for a, b in [
            ("sudo apt update && sudo apt install -y nginx", "sudo apt install -y nginx"),
            ("sudo dnf install ripgrep -y", "sudo dnf install -y ripgrep"),
            ("grep --line-number 'TODO' src", "grep -n 'TODO' src"),
            ("rm -rf ./build", "rm -rf build"),
            ("ls -l -a", "ls -al"),
        ]:
            self.assertTrue(same(a, b), "should forgive: %r vs %r" % (a, b))

    def test_refuses_behaviour_changes(self):
        for a, b in [
            ("systemctl is-active docker", "systemctl status docker"),   # subcommand
            ("uname -r", "uname -a"),                                     # different output
            ("grep -rl 'TODO' src", "grep -rn 'TODO' src"),               # different output
            ("top -o %CPU -n 1", "ps aux --sort=-%cpu"),                  # different program
            ("df -h / | awk 'NR==2 {print $4}'", "df -h"),                # extra stage
            ("sudo systemctl enable --now postgresql", "sudo systemctl enable postgresql"),
            ("sudo rm -rf /path/to/project/build", "rm -rf ./build"),     # different target
            ("apt remove nginx", "apt install nginx"),                    # inverted
            ("rm -rf build", "sudo rm -rf build"),                        # privilege
        ]:
            self.assertFalse(same(a, b), "should NOT forgive: %r vs %r" % (a, b))

    def test_normalising_never_breaks_a_correct_answer(self):
        """A command is always equal to itself, so re-scoring can only add points."""
        rows = [r for r in join.cli_rows() if r["kind"] == "command"]
        for r in rows:
            self.assertTrue(same(r["want"], r["want"]), "not self-equal: %r" % r["want"])


if __name__ == "__main__":
    unittest.main()
