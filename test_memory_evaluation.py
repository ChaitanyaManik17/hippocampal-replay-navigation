import json
import tempfile
import unittest
from pathlib import Path
from random import Random

from maze import MOVES, GridMaze
from memory_evaluation import (
    AssayConfig, CueNoise, RecallTrial, ReplayAgentMemory, Route, TransitionTableMemory, audit_rollouts,
    corrupt_cue, cue_channels, degradation_point, interference_scores, learning_curves, load_rollouts, main,
    navigation_table, overlap_measures, present_cue, run_assays, run_trial, sample_disjoint_routes,
    sample_overlapping_routes, score_checkpoints, score_fixture_recall, score_recall, seed_results,
    summarize_assays, synthetic_factory, valid_continuations,
)
from parallel_fixtures import FIXTURE_PATH, ScriptedRecall, TraceAgent

ROOT = Path(__file__).parent
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class SpyProbe:
    """Records exactly what the evaluator passes to recall()."""

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def recall(self, cue_channels, **timing):
        self.calls.append((list(cue_channels), timing))
        return self.outputs


def rollout(success, steps, *, seed=11, phase="original", step=0, start="1:1", shortest=4, agent="q"):
    return {"map_id": "loop-7x7-v1", "agent": agent, "seed": seed, "phase": phase, "phase_step": step,
            "environment_step": step, "start": start, "success": success, "rollout_steps": steps,
            "shortest_steps": shortest, "path_efficiency": shortest / steps if success else None, "collisions": 0}


class RecallScoringTest(unittest.TestCase):
    def test_fixture_variants_score_exact_none_and_partial(self):
        scores = {r["variant"]: (r["correct"], r["missed"], r["wrong"]) for r in score_fixture_recall()}
        self.assertEqual(scores, {"intact": (3, 0, 0), "recurrent_off": (0, 3, 0), "one_error": (2, 1, 0)})

    def test_probe_receives_only_cue_and_timing(self):
        case = FIXTURE["recall_case"]
        probe = SpyProbe(case["scripted_outputs"]["intact"])
        trial = RecallTrial(tuple(case["cue_channels"]), case["cue_ms"], case["recall_bins"], case["bin_ms"])
        run_trial(probe, trial, [case["expected_locations"]], num_observations=49)
        (channels, timing), = probe.calls
        self.assertEqual(channels, [36, 37])
        self.assertEqual(set(timing), {"cue_ms", "recall_bins", "bin_ms", "recurrence"})
        self.assertNotIn(case["expected_locations"], timing.values())

    def test_scripted_recall_input_boundary(self):
        probe = ScriptedRecall([10, 11, 12])
        present_cue(probe, RecallTrial((36, 37), recurrence=False), num_observations=49)
        self.assertEqual(probe.calls, [((36, 37), 25, 3, 25, False)])

    def test_absent_bins_are_misses_and_wrong_bins_are_errors(self):
        score = score_recall([10, None, 99], [[10, 11, 12]])
        self.assertEqual((score.correct, score.missed, score.wrong, score.bins), (1, 1, 1, 3))
        self.assertAlmostEqual(score.accuracy, 1 / 3)
        self.assertFalse(score.exact)

    def test_ambiguous_cue_scores_best_valid_continuation(self):
        score = score_recall([5, 6, 7], [[5, 9, 9], [5, 6, 7]])
        self.assertTrue(score.exact)
        self.assertEqual(score.valid_continuations, 2)

    def test_malformed_probe_output_is_rejected(self):
        trial = RecallTrial((36,), recall_bins=3)
        for bad in ([1, 2], [1, 2, 49], [1, True, 2], [1, 2.0, 3]):
            with self.assertRaises(ValueError):
                present_cue(SpyProbe(bad), trial, num_observations=49)
        with self.assertRaises(ValueError):
            score_recall([1, 2], [[1, 2, 3]])


class CueNoiseTest(unittest.TestCase):
    def test_codebook_matches_fixture(self):
        self.assertEqual(cue_channels(9, (0, 1)), FIXTURE["recall_case"]["cue_channels"])

    def test_missing_and_spurious_channels(self):
        true = cue_channels(9)
        noisy = corrupt_cue(true, CueNoise(missing=0.5, spurious=3), Random(1), 49)
        self.assertEqual(len(set(noisy) & set(true)), 2)
        spurious = [ch for ch in noisy if ch not in true]
        self.assertEqual(len(spurious), 3)
        self.assertTrue(all(ch // 4 != 9 for ch in spurious))
        self.assertEqual(noisy, corrupt_cue(true, CueNoise(missing=0.5, spurious=3), Random(1), 49))
        self.assertEqual(corrupt_cue(true, CueNoise(missing=1.0), Random(1), 49), [])

    def test_invalid_noise(self):
        with self.assertRaises(ValueError):
            CueNoise(missing=1.5)


class RouteTest(unittest.TestCase):
    def test_actions_and_transitions(self):
        route = Route("r", ((1, 1), (1, 2), (2, 2)))
        self.assertEqual(route.observations, (8, 9, 16))
        self.assertEqual(route.actions, (3, 1))
        last = route.transitions()[-1]
        self.assertTrue(last.terminated)
        self.assertEqual(last.reward, 1.0)
        with self.assertRaises(ValueError):
            Route("bad", ((1, 1), (1, 3)))

    def test_route_actions_replay_on_open_cells_of_real_maze(self):
        route = Route("loop", ((1, 1), (1, 2), (1, 3), (1, 4)))
        maze = GridMaze("loop-7x7-v1", seed=0)
        obs = maze.reset(start=(1, 1))
        for action, expected in zip(route.actions, route.observations[1:]):
            obs, *_ = maze.step(action)
            self.assertEqual(obs, expected)

    def test_disjoint_routes_share_nothing(self):
        routes = sample_disjoint_routes(6, 6, seed=3)
        cells = [c for r in routes for c in r.cells]
        self.assertEqual(len(cells), len(set(cells)))
        self.assertEqual(overlap_measures(routes)["shared_cell_fraction"], 0.0)

    def test_overlapping_routes_share_the_requested_segment(self):
        for count, shared in ((2, 1), (2, 3), (3, 2), (3, 4)):
            routes = sample_overlapping_routes(count, 6, shared, seed=5)
            common = set.intersection(*(set(r.cells) for r in routes))
            self.assertEqual(len(common), shared)
            self.assertEqual(len({r.observations[0] for r in routes}), count)
            m = overlap_measures(routes)
            self.assertAlmostEqual(m["shared_transition_fraction"], (shared - 1) / 5 if shared else 0)
        with self.assertRaises(ValueError):
            sample_overlapping_routes(3, 6, 1, seed=5)

    def test_shared_prefix_gives_two_valid_continuations(self):
        a = Route("a", ((0, 0), (0, 1), (0, 2)))
        b = Route("b", ((0, 0), (1, 0), (2, 0)))
        self.assertEqual(valid_continuations([a, b], a, 1, 2), [(1, 2), (7, 14)])


class NavigationScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_rollouts(ROOT / "baseline_evaluation.csv")
        cls.checkpoints = score_checkpoints(cls.rows)
        cls.seeds = seed_results(cls.checkpoints)

    def test_committed_baseline_passes_audit(self):
        self.assertEqual(len(self.rows), 4500)
        self.assertEqual(audit_rollouts(self.rows, max_steps=150), [])

    def test_committed_baseline_reproduces_reported_table(self):
        table = {r["agent"]: r for r in navigation_table(self.seeds)}
        self.assertAlmostEqual(table["random"]["blocked_final_success"], 5 / 30)
        self.assertEqual(table["q_learning"]["blocked_final_success"], 1.0)
        self.assertAlmostEqual(table["dqn"]["blocked_final_success"], 0.7)
        self.assertEqual(table["q_learning"]["blocked_initial_success"], 0.0)
        self.assertAlmostEqual(table["random"]["original_final_success"], 22 / 30)
        self.assertEqual((table["q_learning"]["recovered_seeds"], table["q_learning"]["median_recovery_step"]),
                         ("6/10", 5500))
        self.assertEqual((table["dqn"]["recovered_seeds"], table["dqn"]["median_recovery_step"]), ("7/10", 3500))
        self.assertEqual(table["dqn"]["blocked_final_rollouts"], 30)

    def test_seed_results_match_committed_summary(self):
        summary = json.loads((ROOT / "baseline_summary.json").read_text())
        mine = {(s["agent"], s["seed"]): s for s in self.seeds}
        for agent in ("q_learning", "dqn"):
            for record in summary[agent]:
                ours = mine[agent, record["seed"]]
                self.assertEqual(ours["recovery_step"], record["recovery_step"])
                self.assertAlmostEqual(ours["blocked_final_success"], record["blocked_final_success"])

    def test_failures_stay_in_denominator(self):
        rows = [rollout(1, 4), rollout(0, 150, start="2:1", shortest=5), rollout(0, 150, start="3:1", shortest=6)]
        (checkpoint,) = score_checkpoints(rows)
        self.assertEqual((checkpoint.successes, checkpoint.rollouts), (1, 3))
        self.assertEqual(checkpoint.mean_successful_efficiency, 1.0)
        self.assertAlmostEqual(learning_curves([checkpoint])[0]["mean_success"], 1 / 3)

    def test_audit_flags_inconsistent_rows(self):
        bad_efficiency = {**rollout(0, 150), "path_efficiency": 0.5}
        bad_bfs = rollout(1, 4, phase="blocked")  # blocked route from 1:1 is 12 actions
        early_failure = rollout(0, 20)
        problems = audit_rollouts([bad_efficiency, bad_bfs, early_failure], max_steps=150)
        self.assertEqual(len(problems), 3)
        self.assertIn("failed rollout has a path efficiency", problems[0])
        self.assertIn("!= BFS 12", problems[1])
        self.assertIn("ended before the 150-action limit", problems[2])

    def test_censored_recovery(self):
        rows = [rollout(s, 12 if s else 150, phase="blocked", step=step, shortest=12)
                for step, s in ((0, 0), (250, 1), (500, 1), (750, 0))]
        self.assertIsNone(seed_results(score_checkpoints(rows))[0]["recovery_step"])


class AssayTest(unittest.TestCase):
    cfg = AssayConfig(seeds=(11, 17), noise_draws=2, load_levels=(1, 3), shared_levels=(0, 3),
                      missing_levels=(0.0, 1.0), spurious_levels=(0, 4), interference_exposures=(0, 40))

    @classmethod
    def setUpClass(cls):
        cls.rows = run_assays(synthetic_factory, cls.cfg, source="test")
        cls.summary = summarize_assays(cls.rows)

    def level(self, assay, level, when=""):
        return next(s for s in self.summary if s["assay"] == assay and s["level"] == level and s["when"] == when)

    def test_recurrence_off_is_all_misses(self):
        self.assertEqual(self.level("recurrence", "recurrence=on")["mean_accuracy"], 1.0)
        off = self.level("recurrence", "recurrence=off")
        self.assertEqual((off["mean_accuracy"], off["mean_miss_rate"]), (0.0, 1.0))

    def test_load_reports_total_exposure_and_flat_curve(self):
        self.assertEqual(self.level("memory_load", "routes=3")["total_exposure"], 60)
        self.assertIsNone(degradation_point(self.summary, "memory_load")["first_drop"])

    def test_degradation_detected(self):
        self.assertEqual(degradation_point(self.summary, "cue_missing")["first_drop"], "missing=1,spurious=0")

    def test_interference_control_and_exposure(self):
        scores = {r["exposure"]: r["interference_score"] for r in interference_scores(self.summary)}
        self.assertEqual(scores[0], 0.0)
        self.assertLess(scores[40], 0.0)

    def test_every_row_carries_source_and_seed(self):
        self.assertTrue(all(r["source"] == "test" and r["seed"] in (11, 17) for r in self.rows))

    def test_replay_agent_adapter_uses_replay_interface(self):
        agent = TraceAgent([3])
        routes = sample_disjoint_routes(2, 4, seed=1)
        ReplayAgentMemory(agent).learn(routes, exposure_per_route=3)
        replays = [e for e in agent.events if e[0] == "replay"]
        self.assertEqual(len(replays), 6)
        self.assertEqual(len(replays[0][1]), 3)

    def test_stand_in_recall_is_frozen(self):
        model = TransitionTableMemory(49, 0)
        model.learn(sample_disjoint_routes(2, 5, seed=2), exposure_per_route=5)
        before = {k: dict(v) for k, v in model.successors.items()}
        model.recall([0, 1], cue_ms=25, recall_bins=4, bin_ms=25, recurrence=True)
        self.assertEqual(before, {k: dict(v) for k, v in model.successors.items()})


class CommandLineTest(unittest.TestCase):
    def test_evidence_command_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["evidence", "--csv", str(ROOT / "baseline_evaluation.csv"), "--output", tmp])
            for name in ("scoring_evidence.png", "navigation_table.md", "navigation_seeds.csv", "recall_fixture.csv"):
                self.assertTrue((Path(tmp) / name).stat().st_size > 0, name)
            self.assertIn("synthetic", (Path(tmp) / "recall_fixture.csv").read_text())


if __name__ == "__main__":
    unittest.main()
