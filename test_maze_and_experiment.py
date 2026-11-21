import csv
import tempfile
import unittest
from pathlib import Path

from baselines import DQN, QLearning
from experiment import RunConfig, run
from maze import Action, GridMaze, Transition, shortest_distance


class MazeTests(unittest.TestCase):
    def test_seeded_start_sequence_repeats(self):
        first = GridMaze("loop-7x7-v1", 11)
        second = GridMaze("loop-7x7-v1", 11)
        self.assertEqual([first.reset() for _ in range(20)], [second.reset() for _ in range(20)])

    def test_collision_and_truncation(self):
        maze = GridMaze("corridor-v1", 0, max_steps=1)
        observation = maze.reset()
        self.assertEqual(maze.step(Action.UP), (observation, -0.05, False, True, {"collision": True}))
        with self.assertRaises(RuntimeError):
            maze.step(Action.RIGHT)

    def test_goal_wins_over_time_limit(self):
        maze = GridMaze("corridor-v1", 0, max_steps=4)
        maze.reset()
        for _ in range(3):
            _, reward, terminated, truncated, _ = maze.step(Action.RIGHT)
            self.assertEqual((reward, terminated, truncated), (-0.005, False, False))
        _, reward, terminated, truncated, _ = maze.step(Action.RIGHT)
        self.assertEqual((reward, terminated, truncated), (1.0, True, False))

    def test_blocked_route_is_longer_and_ids_stay_stable(self):
        original = GridMaze("loop-7x7-v1", 0)
        modified = GridMaze("loop-7x7-v1", 0, blocked=True)
        self.assertEqual(shortest_distance(original, (1, 1)), 4)
        self.assertEqual(shortest_distance(modified, (1, 1)), 12)
        self.assertTrue(all(shortest_distance(modified, start) is not None for start in modified.spec.starts))
        self.assertEqual(original.reset(start=(1, 2)), modified.reset(start=(1, 2)))
        self.assertFalse(original.step(Action.RIGHT)[4]["collision"])
        self.assertTrue(modified.step(Action.RIGHT)[4]["collision"])


class IntegrationTests(unittest.TestCase):
    def test_dqn_optimizes_online_and_keeps_evaluation_frozen(self):
        model = DQN(7, seed=11, decay_steps=100)
        before = [param.detach().clone() for param in model.online.parameters()]
        for _ in range(64):
            model.observe(Transition(1, Action.RIGHT, 1.0, 2, True, False))
        self.assertEqual(model.gradient_steps, 1)
        self.assertTrue(any(not model.torch.equal(p, b) for p, b in zip(model.online.parameters(), before)))
        for _ in range(3):
            self.assertIn(model.act(1, training=False), range(4))
        self.assertEqual(len(model.buffer), 64)
        self.assertEqual(model.gradient_steps, 1)

    def test_initial_blocked_checkpoint_preserves_training_budget(self):
        with tempfile.TemporaryDirectory() as root:
            path = run(RunConfig("loop-7x7-v1", ["q_learning"], [11], 16, 17, 19, 10), Path(root))
            with path.open(newline="") as file:
                rows = list(csv.DictReader(file))
        checkpoints = {(r["phase"], int(r["phase_step"]), int(r["environment_step"])) for r in rows}
        self.assertEqual(checkpoints, {
            ("original", 0, 0), ("original", 10, 10), ("original", 17, 17),
            ("blocked", 0, 17), ("blocked", 10, 27), ("blocked", 19, 36),
        })
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(not r["path_efficiency"] for r in rows if r["success"] == "0"))

    def test_timeout_preserves_bootstrap_while_goal_ends_it(self):
        model = QLearning(5, 11, 100)
        model.values[2][3] = 1.0
        model.observe(Transition(1, 3, -0.005, 2, False, True))
        self.assertAlmostEqual(model.values[1][3], 0.15 * (-0.005 + 0.97))
        model.observe(Transition(1, 1, 1.0, 2, True, False))
        self.assertAlmostEqual(model.values[1][1], 0.15)


if __name__ == "__main__":
    unittest.main()
