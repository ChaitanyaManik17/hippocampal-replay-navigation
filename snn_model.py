"""Ketan's LIF agent and reward-gated, sparse pair-STDP implementation.

Only integer place cues and observed transitions enter the model. There is no
map, planner, value table, or replay buffer here. All times are milliseconds;
voltages and positive synaptic jumps are dimensionless (threshold = 1).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from numbers import Integral
from typing import Sequence

import numpy as np

from maze import Transition


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class SNNConfig:
    """Explicit simulation constants, also saved with the pilot artifacts."""

    decision_ms: int = 25
    learning_ms: int = 25
    membrane_tau_ms: float = 10.0
    refractory_ms: int = 2
    pair_tau_ms: float = 20.0
    eligibility_tau_ms: float = 600.0
    a_plus: float = 1.0
    a_minus: float = 1.05
    eligibility_limit: float = 10.0
    learning_rate: float = 0.015
    recurrent_probability: float = 0.08
    recurrent_initial: float = 0.12
    recurrent_max: float = 0.6
    readout_initial: float = 0.25
    readout_max: float = 1.0
    place_drive: float = 2.0
    teaching_drive: float = 4.0
    excitatory_to_inhibitory: float = 0.12
    inhibitory_to_excitatory: float = 0.18
    action_pool_size: int = 4
    inhibitory_size: int = 4
    epsilon_start: float = 0.8
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 2000

    def __post_init__(self):
        for name in ("decision_ms", "learning_ms", "refractory_ms", "action_pool_size",
                     "inhibitory_size", "epsilon_decay_steps"):
            _integer(getattr(self, name), name)
        if self.learning_ms < 2:
            raise ValueError("learning_ms must allow both transition cues (>= 2)")
        for f in fields(self):
            value = getattr(self, f.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{f.name} must be finite and nonnegative")
        for name in ("membrane_tau_ms", "pair_tau_ms", "eligibility_tau_ms", "eligibility_limit",
                     "recurrent_max", "readout_max"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
        if not 0 <= self.recurrent_probability <= 1:
            raise ValueError("recurrent_probability must be in [0, 1]")
        if self.recurrent_initial > self.recurrent_max or self.readout_initial > self.readout_max:
            raise ValueError("initial weights cannot exceed bounds")


class RewardSTDP:
    """Sparse edges with per-neuron pair traces and per-edge eligibility.

    A pre spike subtracts A_minus * previous post trace; a post spike adds
    A_plus * previous pre trace. Same-tick spikes do not pair with each other.
    Only reward() changes weights. The network and single-pair diagnostic use
    this exact implementation, including exponential decay and clipping.
    """

    def __init__(self, n_pre: int, n_post: int, sources, targets, weights,
                 config: SNNConfig, maximum: float):
        self.sources = np.asarray(sources, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=float).copy()
        self.config, self.maximum = config, maximum
        if not (self.sources.ndim == self.targets.ndim == self.weights.ndim == 1
                and self.sources.shape == self.targets.shape == self.weights.shape):
            raise ValueError("one source, target, and weight per edge required")
        if (np.any(self.sources < 0) or np.any(self.sources >= n_pre)
                or np.any(self.targets < 0) or np.any(self.targets >= n_post)):
            raise ValueError("edge index outside population")
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0) or np.any(self.weights > maximum):
            raise ValueError("weights outside finite excitatory bounds")
        self.pre_trace = np.zeros(n_pre)
        self.post_trace = np.zeros(n_post)
        self.eligibility = np.zeros(len(self.weights))
        self._pair_decay = np.exp(-1.0 / config.pair_tau_ms)
        self._eligibility_decay = np.exp(-1.0 / config.eligibility_tau_ms)

    def reset(self) -> None:
        self.pre_trace.fill(0)
        self.post_trace.fill(0)
        self.eligibility.fill(0)

    def advance(self, pre_spikes, post_spikes, *, learn: bool = True) -> None:
        self.pre_trace *= self._pair_decay
        self.post_trace *= self._pair_decay
        self.eligibility *= self._eligibility_decay
        if not learn:
            return
        pre = np.asarray(pre_spikes, dtype=bool)
        post = np.asarray(post_spikes, dtype=bool)
        self.eligibility += (
            self.config.a_plus * self.pre_trace[self.sources] * post[self.targets]
            - self.config.a_minus * self.post_trace[self.targets] * pre[self.sources]
        )
        np.clip(self.eligibility, -self.config.eligibility_limit,
                self.config.eligibility_limit, out=self.eligibility)
        self.pre_trace += pre
        self.post_trace += post

    def reward(self, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("reward must be finite")
        self.weights += self.config.learning_rate * value * self.eligibility
        np.clip(self.weights, 0, self.maximum, out=self.weights)

    def current(self, pre_spikes) -> np.ndarray:
        return np.bincount(self.targets,
                           weights=self.weights * pre_spikes[self.sources],
                           minlength=len(self.post_trace))


class SpikingAgent:
    """Four place neurons per observation, sparse E->E, global I, four motor pools.

    Evaluation rolls out on evaluation_copy(), as required by the shared runner.
    Direct act(training=False) freezes weights but advances that instance's
    transient state and RNG. recall() always uses its own isolated frozen copy.
    """

    channels_per_observation = 4

    def __init__(self, num_observations: int, seed: int, config: SNNConfig | None = None):
        self.num_observations = _integer(num_observations, "num_observations")
        self.config = config or SNNConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_exc = 4 * self.num_observations
        self.n_action = 4 * self.config.action_pool_size
        mask = self.rng.random((self.n_exc, self.n_exc)) < self.config.recurrent_probability
        np.fill_diagonal(mask, False)
        sources, targets = np.nonzero(mask)
        self.recurrent = RewardSTDP(
            self.n_exc, self.n_exc, sources, targets,
            np.full(len(sources), self.config.recurrent_initial), self.config, self.config.recurrent_max,
        )
        sources = np.repeat(np.arange(self.n_exc), self.n_action)
        targets = np.tile(np.arange(self.n_action), self.n_exc)
        weights = self.config.readout_initial * self.rng.uniform(0.9, 1.1, len(sources))
        weights = np.minimum(weights, self.config.readout_max)
        self.readout = RewardSTDP(self.n_exc, self.n_action, sources, targets, weights,
                                  self.config, self.config.readout_max)
        self.place_gain = self.rng.uniform(0.9, 1.1, self.n_exc)
        self._membrane_decay = np.exp(-1.0 / self.config.membrane_tau_ms)
        self.frozen = False
        self.online_updates = 0
        self.replay_updates = 0
        self.offline_ms = 0
        self.simulated_ms = 0
        self.record_spikes = False
        self.spike_events: list[tuple[int, str, int]] = []
        self.begin_episode()

    def begin_episode(self) -> None:
        """Clear voltages, spikes, refractory counters, and traces; retain weights/RNG."""
        self.v_exc = np.zeros(self.n_exc)
        self.v_inh = np.zeros(self.config.inhibitory_size)
        self.v_action = np.zeros(self.n_action)
        self.ref_exc = np.zeros(self.n_exc, dtype=int)
        self.ref_inh = np.zeros(self.config.inhibitory_size, dtype=int)
        self.ref_action = np.zeros(self.n_action, dtype=int)
        self.exc_spikes = np.zeros(self.n_exc, dtype=bool)
        self.inh_spikes = np.zeros(self.config.inhibitory_size, dtype=bool)
        self.action_spikes = np.zeros(self.n_action, dtype=bool)
        self.recurrent.reset()
        self.readout.reset()
        self.last_action_counts = np.zeros(4, dtype=int)
        self._pending: tuple[int, int] | None = None

    def cue_channels(self, observation: int) -> tuple[int, ...]:
        observation = _integer(observation, "observation", 0)
        if observation >= self.num_observations:
            raise ValueError("observation outside stable place codebook")
        return tuple(range(4 * observation, 4 * observation + 4))

    def _cue(self, channels: Sequence[int]) -> np.ndarray:
        drive = np.zeros(self.n_exc)
        for channel in channels:
            channel = _integer(channel, "cue channel", 0)
            if channel >= self.n_exc:
                raise ValueError("cue channel outside stable place codebook")
            drive[channel] = self.config.place_drive * self.place_gain[channel]
        return drive

    def _lif(self, voltage, refractory, drive, jumps):
        held = refractory > 0
        refractory[:] = np.maximum(refractory - 1, 0)
        voltage *= self._membrane_decay
        voltage += (1 - self._membrane_decay) * drive + jumps
        np.maximum(voltage, 0, out=voltage)
        voltage[held] = 0
        spikes = voltage >= 1.0
        voltage[spikes] = 0
        refractory[spikes] = self.config.refractory_ms
        return spikes

    def _tick(self, drive, *, learn: bool, recurrence: bool = True, teach: int | None = None):
        previous_exc = self.exc_spikes
        feedback = self.config.inhibitory_to_excitatory * int(self.inh_spikes.sum())
        recurrent_current = self.recurrent.current(previous_exc) if recurrence else np.zeros(self.n_exc)
        self.exc_spikes = self._lif(self.v_exc, self.ref_exc, drive, recurrent_current - feedback)
        self.inh_spikes = self._lif(
            self.v_inh, self.ref_inh, 0.0,
            self.config.excitatory_to_inhibitory * int(previous_exc.sum()),
        )
        motor_drive = np.zeros(self.n_action)
        motor_current = self.readout.current(previous_exc)
        if teach is not None:
            # Efference copy of the action actually taken, including exploration.
            # Other motor pools are silenced during association so they cannot
            # create eligibility for an action that was not taken.
            if teach >= 0:
                selected = slice(teach * self.config.action_pool_size, (teach + 1) * self.config.action_pool_size)
                motor_drive[selected] = self.config.teaching_drive
            motor_current.fill(0)
            self.v_action[motor_drive == 0] = 0
        self.action_spikes = self._lif(self.v_action, self.ref_action, motor_drive, motor_current)
        self.recurrent.advance(self.exc_spikes, self.exc_spikes, learn=learn)
        self.readout.advance(self.exc_spikes, self.action_spikes, learn=learn)
        self.simulated_ms += 1
        if self.record_spikes:
            for population, spikes in (("E", self.exc_spikes), ("I", self.inh_spikes), ("A", self.action_spikes)):
                self.spike_events.extend((self.simulated_ms, population, int(i)) for i in np.flatnonzero(spikes))
        return self.exc_spikes, self.action_spikes

    def _window(self, drive, duration: int, *, learn: bool, recurrence: bool = True,
                teach: int | None = None):
        exc_counts, action_counts = np.zeros(self.n_exc, dtype=int), np.zeros(self.n_action, dtype=int)
        for _ in range(duration):
            exc, action = self._tick(drive, learn=learn, recurrence=recurrence, teach=teach)
            exc_counts += exc
            action_counts += action
        return exc_counts, action_counts

    def act(self, observation: int, *, training: bool = False) -> int:
        if training and self.frozen:
            raise RuntimeError("cannot train an evaluation copy")
        if self._pending is not None:
            raise RuntimeError("observe the pending training action before acting again")
        drive = self._cue(self.cue_channels(observation))
        self.v_action.fill(0)
        self.ref_action.fill(0)
        _, counts = self._window(drive, self.config.decision_ms, learn=False)
        self.last_action_counts = counts.reshape(4, self.config.action_pool_size).sum(axis=1)
        epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * max(
            0.0, 1.0 - self.online_updates / self.config.epsilon_decay_steps,
        )
        if training and self.rng.random() < epsilon:
            action = int(self.rng.integers(4))
        else:
            best = np.flatnonzero(self.last_action_counts == self.last_action_counts.max())
            action = int(self.rng.choice(best))
        if training:
            self._pending = (int(observation), action)
        return action

    def _validate_transition(self, transition: Transition) -> None:
        self.cue_channels(transition.observation)
        self.cue_channels(transition.next_observation)
        if _integer(transition.action, "action", 0) >= 4:
            raise ValueError("action must be in [0, 3]")
        if not np.isfinite(transition.reward):
            raise ValueError("reward must be finite")
        if not isinstance(transition.terminated, bool) or not isinstance(transition.truncated, bool):
            raise ValueError("transition end flags must be booleans")

    def _associate(self, transition: Transition, duration: int) -> None:
        # Present current then next location. Delaying the motor teaching drive
        # lets current-location spikes precede motor spikes. Reward arrives only
        # after both phases. The stored next observation is experience, not a target.
        first = duration // 2
        self.v_action.fill(0)
        self.ref_action.fill(0)
        self._window(self._cue(self.cue_channels(transition.observation)), first, learn=True, teach=-1)
        self._window(self._cue(self.cue_channels(transition.next_observation)), duration - first,
                     learn=True, teach=transition.action)
        self.recurrent.reward(transition.reward)
        self.readout.reward(transition.reward)

    def observe(self, transition: Transition) -> None:
        if self.frozen:
            raise RuntimeError("cannot update an evaluation copy")
        self._validate_transition(transition)
        if self._pending != (transition.observation, transition.action):
            raise ValueError("transition must match the last training action")
        self._associate(transition, self.config.learning_ms)
        self.online_updates += 1
        self._pending = None
        if transition.done:
            self.begin_episode()

    def replay(self, sequence: Sequence[Transition], *, step_ms: int = 25) -> None:
        if self.frozen:
            raise RuntimeError("cannot replay into an evaluation copy")
        step_ms = _integer(step_ms, "step_ms", 2)
        transitions = tuple(sequence)
        for transition in transitions:
            self._validate_transition(transition)
        if self._pending is not None:
            raise RuntimeError("observe the pending action before offline work")
        self.begin_episode()
        for transition in transitions:
            self._associate(transition, step_ms)
            self.replay_updates += 1
            self.offline_ms += step_ms
            if transition.done:
                self.begin_episode()

    def rest(self, duration_ms: int) -> None:
        duration_ms = _integer(duration_ms, "duration_ms", 0)
        if self._pending is not None:
            raise RuntimeError("observe the pending action before offline work")
        self._window(np.zeros(self.n_exc), duration_ms, learn=False)
        self.offline_ms += duration_ms

    def evaluation_copy(self) -> SpikingAgent:
        clone = copy.deepcopy(self)
        clone.frozen = True
        clone.record_spikes = False
        clone.spike_events = []
        clone.begin_episode()
        return clone

    def recall(self, cue_channels: Sequence[int], *, cue_ms: int = 25,
               recall_bins: int = 3, bin_ms: int = 25, recurrence: bool = True) -> list[int | None]:
        cue_ms = _integer(cue_ms, "cue_ms")
        recall_bins = _integer(recall_bins, "recall_bins")
        bin_ms = _integer(bin_ms, "bin_ms")
        if not isinstance(recurrence, bool):
            raise ValueError("recurrence must be a boolean")
        drive = self._cue(cue_channels)
        probe = self.evaluation_copy()
        probe._window(drive, cue_ms, learn=False, recurrence=recurrence)
        decoded = []
        for _ in range(recall_bins):
            counts, _ = probe._window(np.zeros(self.n_exc), bin_ms, learn=False, recurrence=recurrence)
            places = counts.reshape(self.num_observations, 4).sum(axis=1)
            best = np.flatnonzero(places == places.max())
            decoded.append(int(best[0]) if places.max() > 0 and len(best) == 1 else None)
        return decoded


def make_agent(num_observations: int, seed: int) -> SpikingAgent:
    """Factory accepted by replay_runner --agent snn_model:make_agent."""
    return SpikingAgent(num_observations, seed)
