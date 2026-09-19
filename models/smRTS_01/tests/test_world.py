"""Phase 1 gate for smRTS_01: every answer readable from the bytes, every gap exact, axes independent."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import world  # noqa: E402
from world import (  # noqa: E402
    CARRY,
    EVAL_TRIALS,
    FILLER_BYTES,
    FILLER_SET,
    GAPS,
    KEY_SET,
    PAIR_COUNTS,
    RESET,
    RESET_AT_RANDOM,
    SYNTAX,
    VALUE_SET,
    Episode,
    WorldError,
)

CELLS = [(p, g) for p in PAIR_COUNTS for g in GAPS]
SMALL = 60          # episodes per cell where the check is cheap
SEED = 4242


def episode_at(pairs=4, gap=32, seed=SEED, **kw) -> Episode:
    return world.sample_episode(world.derive_seed(seed), pairs, gap, **kw)


class Alphabet(unittest.TestCase):
    """Three disjoint printable sets, big enough for the grid."""

    def test_sizes_clear_the_floor(self):
        self.assertGreaterEqual(len(world.KEY_BYTES), 32)
        self.assertGreaterEqual(len(world.VALUE_BYTES), 32)
        self.assertGreaterEqual(len(world.KEY_BYTES), max(PAIR_COUNTS))
        self.assertGreater(len(FILLER_BYTES), 8)

    def test_sets_are_disjoint_and_printable(self):
        for a, b in ((KEY_SET, VALUE_SET), (KEY_SET, FILLER_SET), (VALUE_SET, FILLER_SET)):
            self.assertEqual(a & b, set())
        for group in (KEY_SET, VALUE_SET, FILLER_SET):
            self.assertEqual(group & set(SYNTAX), set())
            self.assertTrue(all(32 <= b <= 126 for b in group))


class Determinism(unittest.TestCase):
    """Fresh from a seed, never stored, and the same seed is the same episode."""

    def test_same_seed_same_bytes(self):
        for pairs, gap in CELLS:
            a = world.episodes(12, pairs, gap, seed=7)
            b = world.episodes(12, pairs, gap, seed=7)
            self.assertEqual([e.data for e in a], [e.data for e in b])

    def test_different_seed_different_episodes(self):
        a = world.episodes(64, 8, 128, seed=1)
        b = world.episodes(64, 8, 128, seed=2)
        self.assertNotEqual([e.data for e in a], [e.data for e in b])

    def test_episodes_within_a_run_are_fresh(self):
        batch = world.episodes(500, 8, 128, seed=3)
        self.assertGreater(len({e.data for e in batch}), 490)

    def test_derive_seed_is_stable_and_bounded(self):
        self.assertEqual(world.derive_seed("a", 1), world.derive_seed("a", 1))
        self.assertNotEqual(world.derive_seed("a", 1), world.derive_seed("a", 2))
        self.assertLess(world.derive_seed("a", 1), 2 ** 63)


class Gaps(unittest.TestCase):
    """The gap is the byte count between the queried pair's '=' and the '?', exactly."""

    def test_every_cell_is_constructible_and_exact(self):
        for pairs, gap in CELLS:
            for episode in world.episodes(SMALL, pairs, gap, seed=11):
                world.validate(episode)
                self.assertEqual(episode.gap, gap)
                self.assertEqual(episode.pair_count, pairs)

    def test_gap_counted_straight_from_the_raw_bytes(self):
        """Recount by hand: find the queried '=' and the '?' and subtract."""
        for pairs, gap in CELLS:
            for episode in world.episodes(20, pairs, gap, seed=12):
                parsed = world.parse_episode(episode.data)
                equals = parsed.pair_offsets[parsed.query_index] + 1
                self.assertEqual(episode.data[equals], ord("="))
                question = episode.data.index(ord("?"))
                self.assertEqual(question - equals - 1, gap)
                self.assertEqual(len(episode.data[equals + 1:question]), gap)

    def test_min_gap_is_the_real_floor(self):
        for pairs in PAIR_COUNTS:
            for q in range(pairs):
                floor = world.min_gap(pairs, q)
                built = world.sample_episode(world.derive_seed(pairs, q), pairs, floor, query_index=q)
                self.assertEqual(built.gap, floor)
                self.assertEqual(built.query_index, q)
                with self.assertRaises(WorldError):
                    world.sample_episode(world.derive_seed(1), pairs, floor - 1, query_index=q)

    def test_impossible_combinations_are_rejected_not_bent(self):
        with self.assertRaises(WorldError):
            world.check_cell(4, 1)
        with self.assertRaises(WorldError):
            world.sample_episode(world.derive_seed(9), pairs=32, gap=8, query_index=0)
        with self.assertRaises(WorldError):
            world.sample_episode(world.derive_seed(9), pairs=2, gap=8, overwrite=True, query_index=0)
        with self.assertRaises(WorldError):
            world.feasible_query_indices(1, 128)
        # ...and a rejection never comes back with a different gap attached.
        try:
            world.sample_episode(world.derive_seed(9), pairs=32, gap=8, query_index=0)
        except WorldError as exc:
            self.assertIn("gap 8", str(exc))


class IndependentAxes(unittest.TestCase):
    """Pair count and gap are selectable independently, with the queried position still random."""

    def test_all_twenty_cells_exist(self):
        self.assertEqual(len(CELLS), 20)
        for pairs, gap in CELLS:
            world.check_cell(pairs, gap)
            episode = world.episodes(1, pairs, gap, seed=13)[0]
            self.assertEqual((episode.pair_count, episode.gap), (pairs, gap))

    def test_gap_does_not_move_when_pair_count_does(self):
        for gap in GAPS:
            for pairs in PAIR_COUNTS:
                built = world.episodes(SMALL, pairs, gap, seed=14)
                self.assertEqual({e.gap for e in built}, {gap})
                self.assertEqual({e.pair_count for e in built}, {pairs})

    def test_pair_count_does_not_move_when_gap_does(self):
        for pairs in PAIR_COUNTS:
            for gap in GAPS:
                built = world.episodes(SMALL, pairs, gap, seed=15)
                self.assertEqual({e.pair_count for e in built}, {pairs})

    def test_query_position_is_random_over_every_feasible_slot(self):
        for pairs, gap in CELLS:
            feasible = set(world.feasible_query_indices(pairs, gap))
            seen = {e.query_index for e in world.episodes(600, pairs, gap, seed=16)}
            self.assertEqual(seen, feasible, f"pairs={pairs} gap={gap}")

    def test_the_short_gap_corner_is_documented_not_claimed_uniform(self):
        """pairs=32, gap=8 can only query ordinals 30 and 31. Say so, measure it, move on."""
        self.assertEqual(world.feasible_query_indices(32, 8), (30, 31))
        leak = world.position_leakage(32, 8, trials=400, seed=40)
        self.assertEqual(leak["feasible_positions"], (30, 31))
        self.assertEqual(leak["feasible_count"], 2)
        self.assertFalse(leak["all_positions"])
        self.assertEqual(set(leak["position_counts"]), {30, 31})
        self.assertGreater(leak["best_fixed_position_rate"], 0.4)
        self.assertGreater(leak["best_fixed_position_rate"], leak["chance"] * 10)
        far = world.position_leakage(32, 512, trials=400, seed=40)
        self.assertTrue(far["all_positions"])
        self.assertEqual(far["feasible_count"], 32)
        self.assertLess(far["best_fixed_position_rate"], 0.10)

    def test_every_episode_carries_its_feasible_positions(self):
        for pairs, gap in CELLS:
            feasible = world.feasible_query_indices(pairs, gap)
            for episode in world.episodes(20, pairs, gap, seed=41):
                self.assertEqual(episode.feasible_positions, feasible)
                self.assertIn(episode.query_index, episode.feasible_positions)

    def test_long_gaps_reach_every_position_and_short_ones_say_why_not(self):
        """At gap >= 128 the position is fully free; at gap 8 only the tail can reach it."""
        for pairs in PAIR_COUNTS:
            for gap in (128, 512):
                self.assertEqual(world.feasible_query_indices(pairs, gap), tuple(range(pairs)))
            self.assertEqual(
                world.feasible_query_indices(pairs, 8), tuple(range(max(0, pairs - 2), pairs))
            )
            self.assertEqual(
                world.feasible_query_indices(pairs, 32), tuple(range(max(0, pairs - 8), pairs))
            )

    def test_lead_filler_moves_position_without_moving_the_gap(self):
        plain = world.episodes(40, 8, 128, seed=17, lead_filler_max=0)
        padded = world.episodes(40, 8, 128, seed=17, lead_filler_max=64)
        self.assertEqual({e.gap for e in padded}, {128})
        self.assertEqual({e.pair_count for e in padded}, {8})
        for episode in padded:
            world.validate(episode)
        self.assertGreater(
            sum(len(e.data) for e in padded), sum(len(e.data) for e in plain)
        )


class Structure(unittest.TestCase):
    """Distinct keys, k=v; syntax, and filler that cannot be mistaken for a pair."""

    def test_keys_are_distinct(self):
        for pairs, gap in CELLS:
            for episode in world.episodes(SMALL, pairs, gap, seed=18):
                keys = [k for k, _ in episode.pairs]
                self.assertEqual(len(set(keys)), pairs)
                self.assertEqual(world.parse_episode(episode.data).duplicate_keys, ())

    def test_syntax_bytes_appear_only_where_they_belong(self):
        for pairs, gap in CELLS:
            for episode in world.episodes(20, pairs, gap, seed=19):
                data = episode.data
                self.assertEqual(data.count(ord("=")), pairs)
                self.assertEqual(data.count(ord(";")), pairs)
                self.assertEqual(data.count(ord("?")), 1)
                self.assertEqual(data.count(ord("\n")), 1)
                self.assertTrue(data.endswith(b"\n"))
                for i, byte in enumerate(data):
                    if byte == ord("="):
                        self.assertIn(data[i - 1], KEY_SET)
                        self.assertIn(data[i + 1], VALUE_SET)
                        self.assertEqual(data[i + 2], ord(";"))

    def test_filler_cannot_parse_as_a_pair(self):
        body = bytes(FILLER_BYTES) * 40
        pairs, offsets, count = world._scan_body(body)
        self.assertEqual((pairs, offsets, count), ([], [], len(body)))
        # No run of filler, however long or however shuffled, can contain '=' or ';'.
        rng = random.Random(5)
        for _ in range(200):
            run = bytes(rng.choice(FILLER_BYTES) for _ in range(rng.randint(0, 30)))
            self.assertNotIn(ord("="), run)
            self.assertNotIn(ord(";"), run)
            self.assertNotIn(ord("?"), run)
            self.assertEqual(world._scan_body(run), ([], [], len(run)))

    def test_filler_is_freshly_sampled_not_a_fixed_pattern(self):
        runs = []
        for episode in world.episodes(200, 4, 512, seed=20):
            runs.append(bytes(b for b in episode.data if b in FILLER_SET))
        self.assertGreater(len(set(runs)), 190)
        pool = Counter(b for run in runs for b in run)
        self.assertEqual(set(pool), set(FILLER_BYTES))          # every filler byte gets used

    def test_only_filler_sits_between_the_pairs(self):
        for episode in world.episodes(80, 8, 512, seed=21):
            parsed = world.parse_episode(episode.data)
            body = episode.data[: episode.data.index(ord("?"))]
            spans = set()
            for offset in parsed.pair_offsets:
                spans.update(range(offset, offset + world.PAIR_LEN))
            for i, byte in enumerate(body):
                if i not in spans:
                    self.assertIn(byte, FILLER_SET)


class Determinable(unittest.TestCase):
    """Every answer is mechanically determinable from the bytes that precede it."""

    def test_prefix_alone_gives_the_answer(self):
        for pairs, gap in CELLS:
            for episode in world.episodes(SMALL, pairs, gap, seed=22):
                prefix = episode.data[: episode.answer_index]
                self.assertEqual(world.answer_from_prefix(prefix), episode.answer)
                self.assertEqual(prefix[-2], ord("?"))
                self.assertEqual(prefix[-1], episode.query_key)

    def test_validate_accepts_generated_episodes_and_nothing_else(self):
        episode = episode_at(pairs=8, gap=128)
        world.validate(episode)
        broken = [
            episode.data[:-1],                                      # no newline
            episode.data + b"x\n",                                  # two newlines
            episode.data.replace(b"?", b"??", 1),                   # two queries
            episode.data[: episode.answer_index - 1]                # queried byte is not a key
            + bytes((episode.answer,))
            + episode.data[episode.answer_index:],
            episode.data[: episode.answer_index] + b"A\n",          # answer is a key, not a value
            episode.data[: episode.answer_index] + bytes((episode.answer,)),  # no newline at all
            episode.data.replace(b"=", b":", 1),                    # broken syntax
            episode.data[:1] + episode.data[2:],                    # a pair loses its '='
        ]
        for data in broken:
            with self.assertRaises(WorldError):
                world.parse_episode(data)

    def test_a_wrong_answer_byte_is_caught(self):
        episode = episode_at(pairs=8, gap=128)
        other = next(v for v in world.VALUE_BYTES if v != episode.answer)
        data = episode.data[: episode.answer_index] + bytes((other,)) + b"\n"
        with self.assertRaises(WorldError):
            world.parse_episode(data)

    def test_querying_an_unwritten_key_is_caught(self):
        episode = episode_at(pairs=4, gap=32)
        written = {k for k, _ in episode.pairs}
        stranger = next(k for k in world.KEY_BYTES if k not in written)
        data = episode.data[: episode.answer_index - 1] + bytes((stranger, episode.answer)) + b"\n"
        with self.assertRaises(WorldError):
            world.parse_episode(data)
        with self.assertRaises(WorldError):
            world.answer_from_prefix(data[: episode.answer_index])

    def test_a_stray_key_byte_in_the_filler_is_caught(self):
        """Filler is disjoint from keys, so a key byte outside a pair must not parse."""
        episode = episode_at(pairs=4, gap=128)
        cut = next(i for i, b in enumerate(episode.data) if b in FILLER_SET)
        stray = next(k for k in world.KEY_BYTES if k not in {p[0] for p in episode.pairs})
        data = episode.data[:cut] + bytes((stray,)) + episode.data[cut:]
        self.assertIn(data[cut + 1], FILLER_SET)      # the stray key is not followed by '='
        with self.assertRaises(WorldError):
            world.parse_episode(data)

    def test_fuzzing_single_bytes_never_yields_a_different_silent_answer(self):
        episode = episode_at(pairs=4, gap=64)
        rng = random.Random(6)
        for _ in range(400):
            index = rng.randrange(len(episode.data))
            data = bytearray(episode.data)
            data[index] = rng.randrange(256)
            try:
                parsed = world.parse_episode(bytes(data))
            except WorldError:
                continue
            # If it still parses, the answer it reports must be the one its own bytes bind.
            self.assertEqual(
                parsed.answer, world.answer_from_prefix(bytes(data)[: parsed.answer_index])
            )


class Scoring(unittest.TestCase):
    """The answer byte only. Everything else in the stream is context."""

    def test_mask_marks_exactly_the_answer(self):
        for episode in world.episodes(40, 8, 128, seed=23):
            mask = world.answer_mask(episode)
            self.assertEqual(int(mask.sum()), 1)
            self.assertEqual(len(mask), len(episode.data) - 1)
            targets = torch.tensor(list(episode.data[1:]))
            self.assertEqual(int(targets[mask][0]), episode.answer)

    def test_batch_tensors_line_up_and_never_score_padding(self):
        batch = world.episodes(16, 4, 8, seed=24) + world.episodes(16, 32, 512, seed=24)
        inputs, targets, mask, lengths = world.batch_tensors(batch)
        self.assertEqual(inputs.shape, targets.shape)
        self.assertEqual(mask.shape, inputs.shape)
        self.assertTrue(torch.equal(mask.sum(1), torch.ones(len(batch), dtype=torch.long)))
        for row, episode in enumerate(batch):
            span = lengths[row].item()
            self.assertEqual(span, len(episode.data) - 1)
            self.assertEqual(int(targets[row][mask[row]][0]), episode.answer)
            self.assertTrue(bool(mask[row, span:].sum() == 0))
            self.assertEqual(bytes(inputs[row, :span].tolist()), episode.data[:-1])
            self.assertEqual(bytes(targets[row, :span].tolist()), episode.data[1:])


class LazyGuesses(unittest.TestCase):
    """Lazy rates match the construction, on the same episodes a model would see."""

    def test_the_named_guesses_are_all_there(self):
        names = world.lazy_guessers(8)
        for name in ("uniform", "most_common", "first", "last"):
            self.assertIn(name, names)
        for index in range(8):
            self.assertIn("position_%d" % index, names)

    def test_positional_guess_is_right_exactly_when_it_matches_the_query(self):
        for pairs, gap in CELLS:
            batch = world.episodes(SMALL, pairs, gap, seed=25)
            guessers = world.lazy_guessers(pairs)
            for episode in batch:
                parsed = world.parse_episode(episode.data)
                rng = random.Random(0)
                hit = guessers["position_%d" % parsed.query_index](parsed, rng)
                self.assertEqual(hit, episode.answer)
                self.assertEqual(guessers["first"](parsed, rng), parsed.pairs[0][1])
                self.assertEqual(guessers["last"](parsed, rng), parsed.pairs[-1][1])
                self.assertEqual(guessers["position_0"](parsed, rng), guessers["first"](parsed, rng))

    def test_last_pair_wins_exactly_as_often_as_the_query_sits_last(self):
        for pairs, gap in CELLS:
            batch = world.episodes(400, pairs, gap, seed=26)
            guessers = world.lazy_guessers(pairs)
            tail, hits = 0, 0
            for episode in batch:
                parsed = world.parse_episode(episode.data)
                tail += int(parsed.query_index == pairs - 1)
                hits += int(guessers["last"](parsed, random.Random(0)) == episode.answer)
            self.assertGreaterEqual(hits, tail)          # a tie on values can only add hits

    def test_short_gaps_leak_and_it_is_visible(self):
        """At gap 8 only the last positions can be queried, so 'last' is a strong lazy guess."""
        near = world.lazy_rates(32, 8, trials=400, seed=27)["last"]["rate"]
        far = world.lazy_rates(32, 512, trials=400, seed=27)["last"]["rate"]
        self.assertGreater(near, 0.4)
        self.assertLess(far, 0.15)

    def test_uniform_sits_at_one_over_the_value_alphabet(self):
        rates = world.lazy_rates(8, 128, trials=EVAL_TRIALS, seed=28)["uniform"]
        chance = 1.0 / len(world.VALUE_BYTES)
        self.assertGreater(rates["rate"], chance * 0.5)
        self.assertLess(rates["rate"], chance * 2.0)
        self.assertLessEqual(rates["low"], rates["rate"])
        self.assertGreaterEqual(rates["high"], rates["rate"])

    def test_most_common_beats_uniform_at_high_pair_counts(self):
        rates = world.lazy_rates(32, 128, trials=EVAL_TRIALS, seed=29)
        self.assertGreater(rates["most_common"]["rate"], rates["uniform"]["rate"])
        self.assertLess(rates["most_common"]["rate"], 0.5)

    def test_every_lazy_rate_is_reported_with_an_interval(self):
        rates = world.lazy_rates(4, 32, trials=200, seed=30)
        for name, row in rates.items():
            self.assertEqual(row["trials"], 200)
            self.assertLessEqual(row["low"], row["rate"], name)
            self.assertGreaterEqual(row["high"], row["rate"], name)

    def test_lazy_arms_score_the_same_episodes_as_lazy_rates(self):
        plan = world.EvalPlan(pairs=8, gap=128, trials=200, seed=31)
        rates = world.lazy_rates(8, 128, trials=200, seed=31)
        result = world.run_plan(plan, world.lazy_arms(8))
        for name, row in rates.items():
            self.assertEqual(result[name]["hits"], row["hits"], name)


class Wilson(unittest.TestCase):
    """A point estimate is not a gate."""

    def test_textbook_interval(self):
        low, high = world.wilson(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)

    def test_the_phase_two_threshold_is_exactly_919_of_1000(self):
        self.assertGreaterEqual(world.wilson(919, 1000)[0], 0.90)
        self.assertLess(world.wilson(918, 1000)[0], 0.90)
        self.assertLess(world.wilson(915, 1000)[0], 0.90)

    def test_exact_boundaries_are_clamped(self):
        self.assertEqual(world.wilson(1, 1)[1], 1.0)      # the audit's 0.9999999999999999
        self.assertGreater(world.wilson(1, 1)[0], 0.0)    # ...but 1/1 is not evidence of p=1
        self.assertEqual(world.wilson(0, 10)[0], 0.0)
        self.assertEqual(world.wilson(10, 10)[1], 1.0)
        self.assertEqual(world.wilson(1000, 1000)[1], 1.0)

    def test_bounds_and_containment(self):
        for hits, trials in ((0, 10), (10, 10), (1, 1000), (999, 1000), (0, 0)):
            low, high = world.wilson(hits, trials)
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
            self.assertLessEqual(low, high)
            if trials:
                self.assertLessEqual(low, hits / trials)
                self.assertGreaterEqual(high, hits / trials)

    def test_interval_narrows_with_more_trials(self):
        narrow = world.wilson(500, 1000)
        wide = world.wilson(50, 100)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_nonsense_counts_raise(self):
        with self.assertRaises(ValueError):
            world.wilson(11, 10)
        with self.assertRaises(ValueError):
            world.wilson(-1, 10)


class Evaluation(unittest.TestCase):
    """1,000 paired fresh episodes per cell, learning frozen, state reset per episode."""

    def test_the_grid_is_twenty_cells_of_a_thousand(self):
        plans = world.evaluation_plans()
        self.assertEqual(len(plans), 20)
        self.assertEqual({p.cell for p in plans}, set(CELLS))
        for plan in plans:
            self.assertEqual(plan.trials, EVAL_TRIALS)
            self.assertTrue(plan.frozen)
            self.assertEqual(plan.protocol.name, RESET)

    def test_a_cell_really_yields_a_thousand_valid_episodes(self):
        plan = world.EvalPlan(pairs=16, gap=128)
        batch = plan.episodes()
        self.assertEqual(len(batch), EVAL_TRIALS)
        for episode in batch:
            world.validate(episode)
        self.assertGreater(len({e.data for e in batch}), 995)

    def test_paired_episode_sets_share_a_hash_and_cells_do_not(self):
        plan = world.EvalPlan(pairs=8, gap=128, trials=50)
        self.assertEqual(plan.episode_set_hash(), plan.episode_set_hash())
        self.assertEqual(
            plan.episode_set_hash(), world.EvalPlan(pairs=8, gap=128, trials=50).episode_set_hash()
        )
        others = [
            world.EvalPlan(pairs=8, gap=512, trials=50),
            world.EvalPlan(pairs=16, gap=128, trials=50),
            world.EvalPlan(pairs=8, gap=128, trials=51),
            world.EvalPlan(pairs=8, gap=128, trials=50, seed=99),
            world.EvalPlan(pairs=8, gap=128, trials=50, overwrite=True),
        ]
        hashes = {plan.episode_set_hash()} | {o.episode_set_hash() for o in others}
        self.assertEqual(len(hashes), 1 + len(others))

    def test_every_arm_reports_the_same_episode_set_hash(self):
        plan = world.EvalPlan(pairs=4, gap=32, trials=30)
        result = world.run_plan(
            plan, {"oracle": lambda e, r: e.answer, "dud": lambda e, r: world.VALUE_BYTES[0]}
        )
        self.assertEqual(result["oracle"]["episode_set_hash"], result["dud"]["episode_set_hash"])
        self.assertEqual(result["oracle"]["episode_set_hash"], plan.episode_set_hash())
        again = world.run_plan(plan, {"oracle": lambda e, r: e.answer})
        self.assertEqual(again["oracle"]["episode_set_hash"], result["oracle"]["episode_set_hash"])
        elsewhere = world.run_plan(
            world.EvalPlan(pairs=4, gap=512, trials=30), {"oracle": lambda e, r: e.answer}
        )
        self.assertNotEqual(
            elsewhere["oracle"]["episode_set_hash"], result["oracle"]["episode_set_hash"]
        )

    def test_arms_are_paired_and_cells_are_not(self):
        first = world.EvalPlan(pairs=8, gap=128, trials=50).episodes()
        again = world.EvalPlan(pairs=8, gap=128, trials=50).episodes()
        self.assertEqual([e.data for e in first], [e.data for e in again])
        other = world.EvalPlan(pairs=8, gap=512, trials=50).episodes()
        self.assertNotEqual([e.data for e in first], [e.data for e in other])
        reseeded = world.EvalPlan(pairs=8, gap=128, trials=50, seed=99).episodes()
        self.assertNotEqual([e.data for e in first], [e.data for e in reseeded])

    def test_a_shorter_run_is_a_prefix_of_a_longer_one(self):
        short = world.EvalPlan(pairs=4, gap=32, trials=10).episodes()
        long = world.EvalPlan(pairs=4, gap=32, trials=40).episodes()
        self.assertEqual([e.data for e in short], [e.data for e in long[:10]])

    def test_two_arms_see_byte_identical_episodes(self):
        seen = {"a": [], "b": []}

        def arm(name):
            def run(episode, reset):
                seen[name].append(episode.data)
                return episode.answer if name == "a" else world.VALUE_BYTES[0]
            return run

        plan = world.EvalPlan(pairs=4, gap=32, trials=40)
        result = world.run_plan(plan, {"a": arm("a"), "b": arm("b")})
        self.assertEqual(seen["a"], seen["b"])
        self.assertEqual(result["a"]["accuracy"], 1.0)
        self.assertEqual(result["a"]["wilson_low"], world.wilson(40, 40)[0])
        self.assertLess(result["b"]["accuracy"], 1.0)

    def test_run_grid_covers_every_cell(self):
        plans = [world.EvalPlan(pairs=p, gap=g, trials=5) for p, g in CELLS]
        grid = world.run_grid({"oracle": lambda e, r: e.answer}, plans)
        self.assertEqual(set(grid), set(CELLS))
        for cell, row in grid.items():
            self.assertEqual(row["oracle"]["hits"], 5, cell)

    def test_logprob_arms_report_nll_and_rank(self):
        plan = world.EvalPlan(pairs=4, gap=32, trials=20)

        def confident(episode, reset):
            logits = torch.full((256,), -40.0, dtype=torch.float64)
            logits[episode.answer] = 0.0
            return torch.log_softmax(logits, dim=0)

        def hopeless(episode, reset):
            return torch.full((256,), -math.log(256), dtype=torch.float64)

        result = world.run_plan(plan, {"confident": confident, "hopeless": hopeless})
        self.assertEqual(result["confident"]["accuracy"], 1.0)
        self.assertLess(result["confident"]["nll"], 1e-6)
        self.assertEqual(result["confident"]["rank"], 1.0)
        self.assertAlmostEqual(result["hopeless"]["nll"], math.log(256), places=6)
        self.assertEqual(result["hopeless"]["rank"], 1.0)      # a flat tie has nothing strictly above

    def test_bad_prediction_shapes_raise(self):
        plan = world.EvalPlan(pairs=4, gap=32, trials=2)
        with self.assertRaises(ValueError):
            world.run_plan(plan, {"short": lambda e, r: torch.zeros(10)})

    def test_frozen_plan_rejects_parameter_mutation(self):
        module = torch.nn.Linear(2, 2)

        def mutating_arm(episode, reset):
            with torch.no_grad():
                module.weight.add_(1.0)
            return episode.answer

        plan = world.EvalPlan(pairs=4, gap=32, trials=2)
        with self.assertRaisesRegex(RuntimeError, "frozen evaluation mutated parameters"):
            world.run_plan(plan, {"bad": mutating_arm}, parameter_modules={"bad": module})

    def test_parameter_modules_must_name_an_arm(self):
        plan = world.EvalPlan(pairs=4, gap=32, trials=2)
        with self.assertRaisesRegex(ValueError, "no matching arm"):
            world.run_plan(
                plan,
                {"oracle": lambda e, r: e.answer},
                parameter_modules={"typo": torch.nn.Linear(2, 2)},
            )


class StateProtocols(unittest.TestCase):
    """Frozen / reset / carry / reset-at-random, represented without training anything."""

    def test_reset_is_primary_and_reports_an_interval(self):
        plan = world.EvalPlan(pairs=4, gap=32, trials=40)
        self.assertEqual(plan.protocol.name, RESET)
        self.assertTrue(plan.frozen)
        row = world.run_plan(plan, {"oracle": lambda e, r: e.answer})["oracle"]
        self.assertEqual(row["protocol"], RESET)
        self.assertEqual((row["wilson_low"], row["wilson_high"]), world.wilson(40, 40))
        self.assertIsNone(row["interval_omitted_because"])

    def test_dependent_protocols_omit_the_binomial_interval_with_a_reason(self):
        for name, period in ((CARRY, 1), (RESET_AT_RANDOM, 8)):
            plan = world.EvalPlan(
                pairs=4, gap=32, trials=40, protocol=world.state_protocol(name, period)
            )
            row = world.run_plan(plan, {"oracle": lambda e, r: e.answer})["oracle"]
            self.assertEqual(row["protocol"], name)
            self.assertIsNone(row["wilson_low"], name)
            self.assertIsNone(row["wilson_high"], name)
            self.assertIn("not independent", row["interval_omitted_because"])
            self.assertEqual(row["accuracy"], 1.0)      # accuracy is still reported

    def test_reset_clears_every_episode(self):
        self.assertEqual(world.state_protocol(RESET).resets(5, 1), [True] * 5)

    def test_carry_clears_only_at_the_start(self):
        self.assertEqual(world.state_protocol(CARRY).resets(5, 1), [True, False, False, False, False])

    def test_reset_at_random_is_deterministic_and_hits_its_rate(self):
        protocol = world.state_protocol(RESET_AT_RANDOM, period=10)
        first = protocol.resets(2000, 1)
        self.assertEqual(first, protocol.resets(2000, 1))
        self.assertNotEqual(first, protocol.resets(2000, 2))
        self.assertTrue(first[0])
        rate = sum(first) / len(first)
        self.assertGreater(rate, 0.07)
        self.assertLess(rate, 0.14)

    def test_unknown_protocols_and_bad_periods_raise(self):
        with self.assertRaises(ValueError):
            world.state_protocol("whenever")
        with self.assertRaises(ValueError):
            world.state_protocol(RESET_AT_RANDOM, period=0)
        with self.assertRaises(ValueError):
            world.StateProtocol("whenever").resets(3, 1)

    def test_episodes_since_reset_is_reported(self):
        plan = world.EvalPlan(pairs=4, gap=32, trials=6, protocol=world.state_protocol(CARRY))
        result = world.run_plan(plan, {"oracle": lambda e, r: e.answer})
        buckets = result["oracle"]["accuracy_by_episodes_since_reset"]
        self.assertEqual(sorted(buckets), list(range(6)))

    def test_the_reset_flag_reaches_the_arm(self):
        flags = []
        plan = world.EvalPlan(pairs=4, gap=32, trials=6, protocol=world.state_protocol(CARRY))

        def arm(episode, reset):
            flags.append(reset)
            return episode.answer

        world.run_plan(plan, {"arm": arm})
        self.assertEqual(flags, [True, False, False, False, False, False])

    def test_frozen_learning_is_assertable(self):
        module = torch.nn.Linear(4, 4)
        before = world.parameter_fingerprint(module)
        with torch.no_grad():
            module(torch.zeros(1, 4))
        self.assertEqual(before, world.parameter_fingerprint(module))
        with torch.no_grad():
            module.weight.add_(1.0)
        self.assertNotEqual(before, world.parameter_fingerprint(module))


class Overwrite(unittest.TestCase):
    """The variant is off by default; when it is on, the newest value is right."""

    def test_off_by_default(self):
        self.assertFalse(world.EvalPlan(pairs=4, gap=32).overwrite)
        self.assertFalse(episode_at().overwrite)
        self.assertFalse(any(p.overwrite for p in world.evaluation_plans()))
        for episode in world.episodes(20, 8, 128, seed=32):
            self.assertEqual(world.parse_episode(episode.data).duplicate_keys, ())

    def test_the_queried_key_is_written_twice_and_the_newest_wins(self):
        for pairs, gap in CELLS:
            for episode in world.episodes(SMALL, pairs, gap, seed=33, overwrite=True):
                world.validate(episode)
                parsed = world.parse_episode(episode.data, expect_overwrite=True)
                self.assertEqual(parsed.duplicate_keys, (episode.query_key,))
                written = [i for i, (k, _) in enumerate(parsed.pairs) if k == episode.query_key]
                self.assertEqual(len(written), 2)
                stale, newest = written
                self.assertLess(stale, newest)
                self.assertEqual(newest, episode.query_index)
                self.assertEqual(parsed.pairs[newest][1], episode.answer)
                self.assertNotEqual(parsed.pairs[stale][1], episode.answer)
                self.assertEqual(episode.gap, gap)
                self.assertEqual(len(parsed.pairs), pairs)

    def test_the_gap_is_measured_from_the_newest_write(self):
        for episode in world.episodes(40, 8, 128, seed=34, overwrite=True):
            parsed = world.parse_episode(episode.data, expect_overwrite=True)
            equals = parsed.pair_offsets[parsed.query_index] + 1
            self.assertEqual(episode.data.index(ord("?")) - equals - 1, 128)

    def test_overwrite_never_queries_the_first_pair(self):
        for pairs, gap in CELLS:
            self.assertNotIn(0, world.feasible_query_indices(pairs, gap, overwrite=True))
            for episode in world.episodes(30, pairs, gap, seed=35, overwrite=True):
                self.assertGreaterEqual(episode.query_index, 1)

    def test_a_plain_episode_is_rejected_as_an_overwrite_and_the_reverse(self):
        plain = episode_at(pairs=8, gap=128)
        with self.assertRaises(WorldError):
            world.parse_episode(plain.data, expect_overwrite=True)
        doubled = world.sample_episode(world.derive_seed(36), 8, 128, overwrite=True)
        with self.assertRaises(WorldError):
            world.parse_episode(doubled.data, expect_overwrite=False)

    def test_overwrite_episodes_still_answer_from_their_prefix(self):
        for episode in world.episodes(60, 16, 128, seed=37, overwrite=True):
            self.assertEqual(
                world.answer_from_prefix(episode.data[: episode.answer_index]), episode.answer
            )

    def test_stale_value_fools_a_first_write_guesser(self):
        """The whole point: the older binding must be a live wrong answer."""
        misses = 0
        batch = world.episodes(200, 8, 128, seed=38, overwrite=True)
        for episode in batch:
            parsed = world.parse_episode(episode.data, expect_overwrite=True)
            stale = next(v for k, v in parsed.pairs if k == episode.query_key)
            misses += int(stale != episode.answer)
        self.assertEqual(misses, len(batch))


if __name__ == "__main__":
    unittest.main(verbosity=2)
