import json
import unittest

from maze import GridMaze, Transition, shortest_distance
from parallel_fixtures import FIXTURE_PATH, ScriptedRecall, TraceAgent, build_fixtures


class ParallelFixtureTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_committed_data_reproduces(self):
        self.assertEqual(self.data, build_fixtures())

    def test_capture_includes_goal_collision_timeout_and_valid_detour(self):
        original, collision, detour = self.data["episodes"]
        self.assertEqual([len(e["transitions"]) for e in (original, collision, detour)], [4, 2, 12])
        self.assertTrue(original["transitions"][-1]["terminated"])
        self.assertEqual(original["transitions"][-1]["reward"], 1.0)
        self.assertTrue(collision["transitions"][0]["collision"])
        self.assertTrue(collision["transitions"][-1]["truncated"])
        self.assertFalse(collision["transitions"][-1]["terminated"])
        self.assertTrue(detour["transitions"][-1]["terminated"])
        blocked_maze = GridMaze(detour["map_id"], 11, blocked=True)
        self.assertEqual(len(detour["transitions"]), shortest_distance(blocked_maze, tuple(detour["start"])))

    def test_trace_records_online_and_offline_calls_separately(self):
        trace = TraceAgent([3])
        transition = Transition(8, 3, -0.005, 9, False, False)
        trace.begin_episode()
        trace.act(8, training=True)
        trace.observe(transition)
        trace.replay([transition], step_ms=25)
        trace.rest(25)
        self.assertEqual(
            [event[0] for event in trace.events], ["begin_episode", "act", "observe", "replay", "rest"]
        )

    def test_scoring_target_is_not_passed_to_scripted_probe(self):
        case = self.data["recall_case"]
        probe = ScriptedRecall(case["scripted_outputs"]["one_error"])
        decoded = probe.recall(
            case["cue_channels"], cue_ms=case["cue_ms"],
            recall_bins=case["recall_bins"], bin_ms=case["bin_ms"], recurrence=True,
        )
        self.assertEqual(sum(a == b for a, b in zip(decoded, case["expected_locations"])), 2)
        self.assertEqual(probe.calls, [((36, 37), 25, 3, 25, True)])


if __name__ == "__main__":
    unittest.main()
