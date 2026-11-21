import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiment import evaluate
from maze import GridMaze, Transition
from parallel_fixtures import FIXTURE_PATH, TraceAgent
from replay_runner import (
    GuardedMaze, OfflineMazeAccess, ReplayRunConfig, audit, run, run_offline_session,
    trace_factory, train_condition,
)
from replay_scheduler import (
    ACTIVE_MODES, MODES, AdaptivePriority, ReplayBudget, ReplayBuffer, ReplayScheduler, is_contiguous,
    successor_rate,
)


def fixture_buffer() -> ReplayBuffer:
    return ReplayBuffer.from_fixture(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


class StepTouchingAgent(TraceAgent):
    """Misbehaving agent that tries to step the maze during replay."""

    def __init__(self, maze):
        super().__init__([0])
        self.maze = maze

    def replay(self, sequence, *, step_ms):
        self.maze.step(0)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.buffer = fixture_buffer()
        self.budget = ReplayBudget(transitions=24, snippet_length=4, step_ms=25)
        self.scheduler = ReplayScheduler(self.budget, seed=3)
        self.recorded = {(s.episode_id, s.time_index): s.transition for s in self.buffer.steps()}

    def test_fixture_loads_in_order(self):
        self.assertEqual([eid for eid, _ in self.buffer.episodes],
                         ["short-route", "wall-and-timeout", "blocked-detour"])
        self.assertEqual(len(self.buffer), 18)

    def test_all_modes_match_duration_and_active_modes_match_transitions(self):
        plans = {mode: self.scheduler.plan(mode, self.buffer, session) for session in range(20) for mode in MODES}
        for session in range(20):
            for mode in MODES:
                plan = self.scheduler.plan(mode, self.buffer, session)
                self.assertEqual(plan.duration_ms, self.budget.duration_ms, mode)
                expected = 0 if mode == "rest" else self.budget.transitions
                self.assertEqual(plan.replayed_transitions, expected, mode)
                self.assertEqual(plan.update_count, expected, mode)
        self.assertEqual(plans["rest"].snippets, ())

    def test_ordered_snippets_stay_within_episodes(self):
        for mode in ("ordered", "adaptive"):
            for session in range(200):
                for snippet in self.scheduler.plan(mode, self.buffer, session).snippets:
                    self.assertTrue(snippet.is_contiguous(), (mode, session, snippet))
                    self.assertEqual({s.episode_id for s in snippet.steps}, {snippet.source_episode})
                    self.assertLessEqual(len(snippet.steps), self.budget.snippet_length)

    def test_terminal_rewards_and_flags_preserved(self):
        saw_goal = saw_timeout = False
        for mode in ACTIVE_MODES:
            for session in range(200):
                for snippet in self.scheduler.plan(mode, self.buffer, session).snippets:
                    for step in snippet.steps:
                        self.assertEqual(step.transition, self.recorded[(step.episode_id, step.time_index)])
                        saw_goal |= step.transition.terminated and step.transition.reward == 1.0
                        saw_timeout |= step.transition.truncated and not step.transition.terminated
        self.assertTrue(saw_goal and saw_timeout)

    def test_random_mode_is_independent_single_transitions(self):
        plan = self.scheduler.plan("random", self.buffer, 0)
        self.assertTrue(all(len(s.steps) == 1 for s in plan.snippets))
        orderly = self.scheduler.plan("ordered", self.buffer, 0)
        self.assertLess(successor_rate(plan), successor_rate(orderly))

    def test_seeded_sampling_is_reproducible(self):
        other = ReplayScheduler(self.budget, seed=3)
        for mode in MODES:
            self.assertEqual(self.scheduler.plan(mode, self.buffer, 5), other.plan(mode, self.buffer, 5))
        self.assertNotEqual(self.scheduler.plan("ordered", self.buffer, 5).snippets,
                            self.scheduler.plan("ordered", self.buffer, 6).snippets)

    def test_adaptive_prefers_success_and_ties_are_equal(self):
        windows = self.scheduler.candidate_windows(self.buffer)
        weights = self.scheduler.priorities(self.buffer, windows)
        by_episode = {}
        for (_, steps), w in zip(windows, weights):
            by_episode.setdefault(steps[0].episode_id, []).append(w)
        # The timeout episode never reached the goal and is the lowest priority.
        self.assertLess(max(by_episode["wall-and-timeout"]), min(by_episode["short-route"]))
        # Interior detour windows see identical inputs and get identical weights.
        interior = by_episode["blocked-detour"][1:]
        self.assertTrue(all(abs(w - interior[0]) < 1e-12 for w in interior))

    def test_uniform_priority_reduces_adaptive_to_ordered(self):
        flat = ReplayScheduler(self.budget, AdaptivePriority(0, 0, 0, floor=1.0), seed=3)
        for session in range(10):
            a = flat.plan("adaptive", self.buffer, session)
            o = flat.plan("ordered", self.buffer, session)
            self.assertEqual([s.steps for s in a.snippets], [s.steps for s in o.snippets])

    def test_short_episodes_remain_eligible(self):
        windows = self.scheduler.candidate_windows(self.buffer)
        self.assertIn(("wall-and-timeout", 2), {(s[0].episode_id, len(s)) for _, s in windows})
        self.assertEqual(len(windows), 1 + 1 + 9)

    def test_empty_buffer_rests_for_matched_duration(self):
        plan = self.scheduler.plan("ordered", ReplayBuffer(), 0)
        self.assertEqual((plan.replayed_transitions, plan.rest_ms), (0, self.budget.duration_ms))

    def test_contiguity_rejects_cross_episode_and_gaps(self):
        steps = self.buffer.steps()
        self.assertTrue(is_contiguous(steps[0:4]))
        self.assertFalse(is_contiguous(steps[3:5]))     # goal arrival then next episode
        self.assertFalse(is_contiguous([steps[6], steps[8]]))  # gap inside detour


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.buffer = fixture_buffer()
        self.scheduler = ReplayScheduler(ReplayBudget(24, 4, 25), seed=0)

    def test_trace_agent_receives_exactly_the_plan(self):
        for mode in MODES:
            agent = TraceAgent([0])
            plan = self.scheduler.plan(mode, self.buffer, 0)
            run_offline_session(agent, plan, None, condition=mode, seed=0, environment_step=0,
                                buffer_transitions=len(self.buffer))
            kinds = [e[0] for e in agent.events]
            if mode == "rest":
                self.assertEqual(agent.events, [("rest", 600)])
            else:
                self.assertEqual(set(kinds), {"replay"})
                self.assertEqual([e[1] for e in agent.events], [s.transitions for s in plan.snippets])
                self.assertEqual(sum(len(e[1]) * e[2] for e in agent.events), 600)
            self.assertNotIn("act", kinds)
            self.assertNotIn("observe", kinds)

    def test_guard_blocks_maze_access_offline(self):
        guard = GuardedMaze(GridMaze("loop-7x7-v1", 0))
        guard.reset()
        plan = self.scheduler.plan("ordered", self.buffer, 0)
        with self.assertRaises(OfflineMazeAccess):
            run_offline_session(StepTouchingAgent(guard), plan, guard, condition="ordered", seed=0,
                                environment_step=0, buffer_transitions=18)
        self.assertEqual(guard.offline_calls, 1)
        self.assertFalse(guard.offline)
        guard.step(3)  # usable again after the session

    def test_evaluation_does_not_write_buffer(self):
        buffer = ReplayBuffer()
        buffer.start_episode("x")
        buffer.record(Transition(8, 3, -0.005, 9, False, False), phase="original", map_id="loop-7x7-v1",
                      collision=False)
        rows = list(evaluate(TraceAgent([3]), GridMaze("loop-7x7-v1", 0), "ordered", 0, "original", 0, 0))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(buffer), 1)

    def config(self, **overrides):
        base = dict(map_id="loop-7x7-v1", conditions=list(MODES), seeds=[11], max_steps=30,
                    pretrain_steps=120, adapt_steps=60, checkpoint_every=60, replay_every=40,
                    replay_transitions=20, snippet_length=5)
        base.update(overrides)
        return ReplayRunConfig(**base)

    def test_online_budget_and_matched_sessions(self):
        config = self.config()
        sessions_by_condition = {}
        evals = {}
        for condition in MODES:
            agent = trace_factory(49, 11)
            logs, sessions = [], []
            rows = list(train_condition(agent, config, condition, 11, log_rows=logs, session_rows=sessions))
            self.assertEqual(sum(e[0] == "observe" for e in agent.events), 180)
            self.assertEqual(rows[-1].environment_step, 180)
            self.assertTrue(all(s["maze_calls_offline"] == 0 for s in sessions))
            sessions_by_condition[condition] = sessions
            evals[condition] = [(r.phase, r.phase_step, r.success, r.rollout_steps) for r in rows]
        steps = {c: [s["environment_step"] for s in v] for c, v in sessions_by_condition.items()}
        self.assertEqual(len({tuple(v) for v in steps.values()}), 1)
        self.assertEqual(steps["rest"], [40, 80, 120, 160])
        for condition, sessions in sessions_by_condition.items():
            self.assertTrue(all(s["offline_ms"] == 500 for s in sessions))
            expected = 0 if condition == "rest" else 20
            self.assertTrue(all(s["replayed_transitions"] == expected for s in sessions))
        # TraceAgent does not learn, so identical evaluations show replay never leaked into the maze.
        self.assertEqual(len({tuple(v) for v in evals.values()}), 1)

    def test_blocked_phase_evaluates_before_training(self):
        rows = list(train_condition(trace_factory(49, 11), self.config(conditions=["rest"]), "rest", 11,
                                    log_rows=[], session_rows=[]))
        first_blocked = next(r for r in rows if r.phase == "blocked")
        self.assertEqual((first_blocked.phase_step, first_blocked.environment_step), (0, 120))

    def test_cli_outputs_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run(self.config(seeds=[11, 17]), out / "run", trace_factory)
            for name in ("evaluation.csv", "replay_log.csv", "offline_sessions.csv", "config.json"):
                self.assertTrue((out / "run" / name).exists(), name)
            with (out / "run" / "replay_log.csv").open() as file:
                log = list(csv.DictReader(file))
            ordered = [r for r in log if r["condition"] in ("ordered", "adaptive")]
            self.assertTrue(ordered and all(r["contiguous"] == "1" for r in ordered))
            table = audit(out / "audit")
            self.assertEqual([r["condition"] for r in table], list(MODES))
            self.assertEqual({r["offline_ms"] for r in table}, {600})
            self.assertFalse(any("FAIL" in str(v) for r in table for v in r.values()))


if __name__ == "__main__":
    unittest.main()
