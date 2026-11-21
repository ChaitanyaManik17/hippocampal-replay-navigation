"""Standalone acceptance for Task 2 and the real model/runner handoff."""

import copy
import json
import pickle
import unittest
from dataclasses import replace

import numpy as np

from maze import GridMaze, Transition
from parallel_fixtures import FIXTURE_PATH
from snn_model import RewardSTDP, SNNConfig, SpikingAgent, make_agent


def pair_response(case, *, pulse_ms=None):
    config = SNNConfig()
    synapse = RewardSTDP(1, 1, [0], [0], [0.5], config, 1.0)
    pulse = case["pulse_ms"] if pulse_ms is None else pulse_ms
    for time in range(1, pulse + 1):
        synapse.advance([time == case["pre_ms"]], [time == case["post_ms"]])
    synapse.reward(case["reward"])
    return float(synapse.weights[0] - 0.5), float(synapse.eligibility[0])


class PlasticityTests(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads(FIXTURE_PATH.read_text())["plasticity_cases"]

    def test_supplied_pair_signs_and_analytic_decay(self):
        c = SNNConfig()
        for case in self.cases:
            with self.subTest(case=case["id"]):
                delta, eligibility = pair_response(case)
                self.assertEqual(int(np.sign(delta)), case["expected_delta_sign"])
                amplitude = c.a_plus if case["pre_ms"] < case["post_ms"] else -c.a_minus
                expected = amplitude * np.exp(-abs(case["post_ms"] - case["pre_ms"]) / c.pair_tau_ms)
                expected *= np.exp(-(case["pulse_ms"] - max(case["pre_ms"], case["post_ms"])) / c.eligibility_tau_ms)
                self.assertAlmostEqual(eligibility, expected, places=12)
                self.assertAlmostEqual(delta, c.learning_rate * case["reward"] * expected, places=12)

    def test_late_reward_is_smaller_by_trace_decay(self):
        early, _ = pair_response(self.cases[0])
        late, _ = pair_response(self.cases[0], pulse_ms=720)
        self.assertGreater(late, 0)
        self.assertLess(late, early)
        self.assertAlmostEqual(late / early, np.exp(-1), places=12)

    def test_no_reward_no_weight_change_and_hard_bounds(self):
        synapse = RewardSTDP(1, 1, [0], [0], [0.5], SNNConfig(), 1)
        for t in range(100):
            synapse.advance([t % 10 == 0], [t % 10 == 1])
        self.assertEqual(synapse.weights[0], 0.5)
        synapse.reward(1e6)
        self.assertEqual(synapse.weights[0], 1)
        synapse.reward(-1e6)
        self.assertEqual(synapse.weights[0], 0)

    def test_simultaneous_pair_has_no_artificial_ordering(self):
        synapse = RewardSTDP(1, 1, [0], [0], [0.5], SNNConfig(), 1)
        synapse.advance([True], [True])
        synapse.reward(1)
        self.assertEqual(synapse.weights[0], 0.5)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent(49, 11)
        data = json.loads(FIXTURE_PATH.read_text())
        self.sequence = [Transition(**{k: row[k] for k in Transition.__dataclass_fields__})
                         for row in data["episodes"][0]["transitions"]]

    def weights(self, agent=None):
        agent = agent or self.agent
        return agent.recurrent.weights.copy(), agent.readout.weights.copy()

    def assert_weights_equal(self, before, agent=None):
        for old, new in zip(before, self.weights(agent)):
            np.testing.assert_array_equal(old, new)

    def test_place_code_and_reset_preserve_learned_state(self):
        self.assertEqual(self.agent.cue_channels(9), (36, 37, 38, 39))
        self.agent.replay(self.sequence)
        before = self.weights()
        rng_before = copy.deepcopy(self.agent.rng.bit_generator.state)
        self.agent.act(9, training=True)
        self.agent.begin_episode()
        self.assert_weights_equal(before)
        self.assertFalse(self.agent.v_exc.any())
        self.assertFalse(self.agent.readout.eligibility.any())
        self.assertFalse(self.agent.recurrent.pre_trace.any())
        self.assertIsNone(self.agent._pending)
        # begin_episode does not rewind RNG (the preceding act consumed draws).
        self.assertNotEqual(rng_before, self.agent.rng.bit_generator.state)

    def test_training_and_frozen_evaluation_are_reproducible_and_isolated(self):
        maze = GridMaze("corridor-v1", 11, max_steps=20)
        original = self.agent
        before_weights = self.weights()
        observation = maze.reset()
        action = original.act(observation, training=True)
        state = pickle.dumps(original)
        evaluation = original.evaluation_copy()
        for _ in range(10):
            self.assertIn(evaluation.act(observation, training=False), range(4))
        self.assertEqual(pickle.dumps(original), state)
        self.assert_weights_equal(before_weights, evaluation)
        with self.assertRaises(RuntimeError):
            evaluation.replay(self.sequence)
        with self.assertRaises(RuntimeError):
            evaluation.observe(self.sequence[0])
        with self.assertRaises(RuntimeError):
            evaluation.act(observation, training=True)
        next_observation, reward, terminated, truncated, _ = maze.step(action)
        original.observe(Transition(observation, action, reward, next_observation, terminated, truncated))
        self.assertEqual(original.online_updates, 1)

    def test_replay_changes_both_weight_sets_and_accounts_exact_duration(self):
        before = self.weights()
        t = self.agent.simulated_ms
        self.agent.replay(self.sequence, step_ms=25)
        self.assertEqual(self.agent.simulated_ms - t, 100)
        self.assertEqual((self.agent.offline_ms, self.agent.replay_updates, self.agent.online_updates), (100, 4, 0))
        for old, new in zip(before, self.weights()):
            self.assertTrue(np.any(old != new))
        self.assertFalse(self.agent.readout.eligibility.any())  # final transition terminated

    def test_rest_freezes_weights_and_only_decays_eligibility(self):
        self.agent.replay(self.sequence[:2])
        before = self.weights()
        eligibility = self.agent.readout.eligibility.copy()
        before_time = self.agent.simulated_ms
        self.agent.rest(50)
        self.assert_weights_equal(before)
        np.testing.assert_allclose(self.agent.readout.eligibility,
                                   eligibility * np.exp(-50 / self.agent.config.eligibility_tau_ms))
        self.assertEqual(self.agent.simulated_ms - before_time, 50)
        self.assertEqual(self.agent.replay_updates, 2)

    def test_snippets_and_terminal_flags_clear_cross_episode_traces(self):
        a = self.agent
        b = copy.deepcopy(a)
        a.v_exc.fill(0.99)
        a.recurrent.eligibility.fill(9)
        a.readout.eligibility.fill(-9)
        a.replay(self.sequence[:1])
        b.replay(self.sequence[:1])
        self.assert_weights_equal(self.weights(a), b)
        terminal = replace(self.sequence[0], terminated=False, truncated=True)
        a.replay([terminal])
        self.assertFalse(a.readout.eligibility.any())

    def test_only_the_taken_motor_pool_spikes_during_association(self):
        self.agent.record_spikes = True
        chosen = replace(self.sequence[0], action=2)
        self.agent.replay([chosen])
        motor = [neuron for _, pop, neuron in self.agent.spike_events if pop == "A"]
        self.assertTrue(motor)
        self.assertEqual({n // self.agent.config.action_pool_size for n in motor}, {2})

    def test_recall_has_no_expected_route_and_preserves_full_training_state(self):
        self.agent.replay(self.sequence)
        self.agent.act(9, training=True)
        before = pickle.dumps(self.agent)
        for recurrence in (True, False):
            result = self.agent.recall([36, 37], cue_ms=25, recall_bins=3, bin_ms=25, recurrence=recurrence)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(x is None or x in range(49) for x in result))
            self.assertEqual(pickle.dumps(self.agent), before)
        self.assertEqual(self.agent.recall([], recall_bins=3), [None, None, None])

    def test_recurrence_switch_removes_actual_postcue_excitation(self):
        # A controlled edge set routes one cue population into a distinct one.
        # This checks the probe mechanism, not learned sequence recall accuracy.
        self.agent.recurrent = RewardSTDP(
            self.agent.n_exc, self.agent.n_exc,
            np.repeat(np.arange(36, 40), 4), np.tile(np.arange(40, 44), 4),
            np.full(16, 0.6), self.agent.config, 0.6,
        )
        probe = self.agent.evaluation_copy()
        probe.exc_spikes[36:40] = True
        enabled, _ = probe._window(np.zeros(probe.n_exc), 1, learn=False, recurrence=True)
        probe.begin_episode()
        probe.exc_spikes[36:40] = True
        disabled, _ = probe._window(np.zeros(probe.n_exc), 1, learn=False, recurrence=False)
        self.assertEqual(int(enabled[40:44].sum()), 4)
        self.assertEqual(int(disabled.sum()), 0)

    def test_inhibition_limits_activity_and_weights_remain_finite(self):
        self.agent.record_spikes = True
        # A diffuse cue drives ten place populations, recruiting feedback onto
        # other cells. Fully synchronous drive can mask feedback in refractory time.
        enabled, _ = self.agent._window(self.agent._cue(range(40)), 200, learn=False)
        self.assertTrue(any(pop == "I" for _, pop, _ in self.agent.spike_events))
        without_feedback = SpikingAgent(49, 11, replace(self.agent.config, inhibitory_to_excitatory=0))
        disabled, _ = without_feedback._window(without_feedback._cue(range(40)), 200, learn=False)
        self.assertLess(int(enabled.sum()), int(disabled.sum()))
        for _ in range(10):
            self.agent.replay(self.sequence)
        for synapses in (self.agent.recurrent, self.agent.readout):
            self.assertTrue(np.all(np.isfinite(synapses.weights)))
            self.assertTrue(np.all((synapses.weights >= 0) & (synapses.weights <= synapses.maximum)))

    def test_input_errors_are_atomic(self):
        before = pickle.dumps(self.agent)
        bad = replace(self.sequence[0], action=4)
        with self.assertRaises(ValueError):
            self.agent.replay([self.sequence[0], bad])
        self.assertEqual(pickle.dumps(self.agent), before)
        for cue in ([-1], [196], [True]):
            with self.assertRaises(ValueError):
                self.agent.recall(cue)
        for value in (0, 1, 2.5, True):
            with self.assertRaises(ValueError):
                self.agent.replay(self.sequence, step_ms=value)

    def test_runner_handoff_all_conditions_and_blockage(self):
        from replay_runner import ReplayRunConfig, train_condition
        from replay_scheduler import MODES

        config = ReplayRunConfig(
            map_id="loop-7x7-v1", conditions=list(MODES), seeds=[11], max_steps=8,
            pretrain_steps=8, adapt_steps=4, checkpoint_every=4,
            replay_every=4, replay_transitions=4, snippet_length=2,
        )
        for mode in MODES:
            agent = make_agent(49, 11)
            logs, sessions = [], []
            rows = list(train_condition(agent, config, mode, 11, log_rows=logs, session_rows=sessions))
            self.assertEqual(agent.online_updates, 12)
            self.assertEqual(agent.offline_ms, 300)
            self.assertEqual(agent.replay_updates, 0 if mode == "rest" else 12)
            self.assertTrue(all(s["maze_calls_offline"] == 0 for s in sessions))
            blocked = next(r for r in rows if r.phase == "blocked")
            self.assertEqual((blocked.phase_step, blocked.environment_step), (0, 8))

    def test_checkpoint_frequency_does_not_change_real_model_training(self):
        from replay_runner import ReplayRunConfig, train_condition

        results = []
        for every in (2, 8):
            config = ReplayRunConfig("corridor-v1", ["rest"], [11], 8, 8, 0, every, 4, 4, 2)
            agent = make_agent(21, 11)
            list(train_condition(agent, config, "rest", 11, log_rows=[], session_rows=[]))
            results.append(pickle.dumps(agent))
        self.assertEqual(*results)


if __name__ == "__main__":
    unittest.main()
