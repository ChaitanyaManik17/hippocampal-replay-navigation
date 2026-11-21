"""Offline replay sampling for the CS 525 hippocampal replay project.

Four offline conditions share one budget object so they can be compared fairly:

* ``rest``      silent rest for the same simulated duration, no replayed transitions
* ``random``    real recorded transitions drawn independently (one transition per call)
* ``ordered``   contiguous within-episode snippets drawn uniformly over windows
* ``adaptive``  contiguous within-episode snippets drawn by priority
                (success of the source episode, rarity of visited cells, recency)

Every active mode replays exactly ``budget.transitions`` recorded transitions and
spends exactly ``budget.duration_ms`` of simulated time. ``rest`` spends the same
duration. Sampling never touches an environment: it only reads the buffer.
Recorded transitions are passed through unchanged, so rewards and both end flags
are preserved bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random
from typing import Iterable, Mapping, Sequence

from maze import Transition

MODES = ("rest", "random", "ordered", "adaptive")
ACTIVE_MODES = ("random", "ordered", "adaptive")


# --------------------------------------------------------------------------- storage


@dataclass(frozen=True)
class RecordedStep:
    """One real training transition plus the runner metadata kept beside it."""

    episode_id: str
    time_index: int
    phase: str
    map_id: str
    collision: bool
    transition: Transition

    @property
    def is_goal(self) -> bool:
        return self.transition.terminated and self.transition.reward > 0


class ReplayBuffer:
    """Episode-structured store of real experience, oldest episode first.

    Only the training loop writes here. Evaluation code must never call
    ``record``; the runner enforces that by construction.
    """

    def __init__(self, max_episodes: int | None = None):
        if max_episodes is not None and max_episodes < 1:
            raise ValueError("max_episodes must be positive when given")
        self.max_episodes = max_episodes
        self._episodes: list[tuple[str, list[RecordedStep]]] = []
        self._open: str | None = None

    # writing ---------------------------------------------------------------
    def start_episode(self, episode_id: str) -> None:
        if any(eid == episode_id for eid, _ in self._episodes):
            raise ValueError(f"duplicate episode id: {episode_id}")
        self._episodes.append((episode_id, []))
        self._open = episode_id
        if self.max_episodes is not None:
            while len(self._episodes) > self.max_episodes:
                self._episodes.pop(0)

    def record(
        self, transition: Transition, *, phase: str, map_id: str, collision: bool,
    ) -> RecordedStep:
        if self._open is None:
            raise RuntimeError("start_episode before recording")
        episode_id, steps = self._episodes[-1]
        if steps and steps[-1].transition.done:
            raise RuntimeError(f"episode {episode_id} already ended")
        step = RecordedStep(episode_id, len(steps), phase, map_id, collision, transition)
        steps.append(step)
        if transition.done:
            self._open = None
        return step

    def close_episode(self) -> None:
        """Stop writing to the current episode without inventing an end flag."""
        self._open = None

    def add_episode(self, episode_id: str, steps: Iterable[RecordedStep]) -> None:
        self.start_episode(episode_id)
        self._episodes[-1][1].extend(steps)
        self._open = None

    @classmethod
    def from_fixture(cls, data: Mapping) -> "ReplayBuffer":
        """Load ``parallel_fixtures.json`` episodes in their committed order."""
        buffer = cls()
        for episode in data["episodes"]:
            steps = [
                RecordedStep(
                    episode["episode_id"], row["time_index"], episode["phase"],
                    episode["map_id"], bool(row["collision"]),
                    Transition(
                        int(row["observation"]), int(row["action"]), float(row["reward"]),
                        int(row["next_observation"]), bool(row["terminated"]), bool(row["truncated"]),
                    ),
                )
                for row in episode["transitions"]
            ]
            buffer.add_episode(episode["episode_id"], steps)
        return buffer

    # reading ---------------------------------------------------------------
    @property
    def episodes(self) -> list[tuple[str, tuple[RecordedStep, ...]]]:
        return [(eid, tuple(steps)) for eid, steps in self._episodes if steps]

    def steps(self) -> list[RecordedStep]:
        return [step for _, steps in self.episodes for step in steps]

    def observation_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for step in self.steps():
            counts[step.transition.observation] = counts.get(step.transition.observation, 0) + 1
        return counts

    def __len__(self) -> int:
        return sum(len(steps) for _, steps in self._episodes)


# --------------------------------------------------------------------------- plans


@dataclass(frozen=True)
class ReplayBudget:
    """Shared offline allowance for one session.

    ``transitions`` is the number of recorded transitions every active mode
    replays. ``snippet_length`` is the maximum ordered snippet length.
    Each replayed transition occupies ``step_ms`` of simulated time, so all
    modes (including rest) spend ``transitions * step_ms``.
    """

    transitions: int
    snippet_length: int
    step_ms: int = 25

    def __post_init__(self):
        if min(self.transitions, self.snippet_length, self.step_ms) < 1:
            raise ValueError("budget values must be positive")

    @property
    def duration_ms(self) -> int:
        return self.transitions * self.step_ms


@dataclass(frozen=True)
class AdaptivePriority:
    """Priority = floor + success*S + rarity*R + recency*C, then raised to alpha.

    S  1.0 if the source episode reached the goal, else 0.0
    R  mean over the window of 1/sqrt(visit count of each observation in the buffer)
    C  exp(-age / recency_tau), age = number of episodes recorded after the source
    """

    success: float = 1.0
    rarity: float = 1.0
    recency: float = 1.0
    recency_tau: float = 5.0
    alpha: float = 1.0
    floor: float = 0.05

    def __post_init__(self):
        if min(self.success, self.rarity, self.recency) < 0:
            raise ValueError("priority weights must be nonnegative")
        if self.recency_tau <= 0 or self.alpha < 0 or self.floor <= 0:
            raise ValueError("recency_tau and floor must be positive, alpha nonnegative")


@dataclass(frozen=True)
class Snippet:
    steps: tuple[RecordedStep, ...]
    priority: float | None = None

    @property
    def transitions(self) -> tuple[Transition, ...]:
        return tuple(step.transition for step in self.steps)

    @property
    def source_episode(self) -> str:
        return self.steps[0].episode_id

    @property
    def start_index(self) -> int:
        return self.steps[0].time_index

    @property
    def end_index(self) -> int:
        return self.steps[-1].time_index

    def is_contiguous(self) -> bool:
        return is_contiguous(self.steps)


@dataclass(frozen=True)
class OfflinePlan:
    mode: str
    session: int
    seed: int
    budget: ReplayBudget
    snippets: tuple[Snippet, ...] = ()
    rest_ms: int = 0
    candidates: int = 0

    @property
    def replayed_transitions(self) -> int:
        return sum(len(s.steps) for s in self.snippets)

    @property
    def replay_ms(self) -> int:
        return self.replayed_transitions * self.budget.step_ms

    @property
    def duration_ms(self) -> int:
        return self.replay_ms + self.rest_ms

    @property
    def update_count(self) -> int:
        """Reward-gated update opportunities offered to the model (one per replayed step)."""
        return self.replayed_transitions


def is_contiguous(steps: Sequence[RecordedStep]) -> bool:
    """Same episode, consecutive time indices, and a valid observation chain."""
    for before, after in zip(steps, steps[1:]):
        if before.episode_id != after.episode_id:
            return False
        if after.time_index != before.time_index + 1:
            return False
        if before.transition.next_observation != after.transition.observation:
            return False
        if before.transition.done:
            return False
    return True


# --------------------------------------------------------------------------- scheduler


@dataclass
class ReplayScheduler:
    budget: ReplayBudget
    priority: AdaptivePriority = field(default_factory=AdaptivePriority)
    seed: int = 0

    def rng(self, session: int) -> Random:
        # String seeds are hashed deterministically by Random (version 2), so a
        # (seed, session) pair always gives the same stream on any machine.
        # The stream does not depend on the mode name.
        return Random(f"replay:{self.seed}:{session}")

    def plan(self, mode: str, buffer: ReplayBuffer, session: int = 0) -> OfflinePlan:
        if mode not in MODES:
            raise ValueError(f"unknown replay mode {mode!r}; choose from {MODES}")
        if mode == "rest":
            return OfflinePlan(mode, session, self.seed, self.budget, rest_ms=self.budget.duration_ms)
        if len(buffer) == 0:
            # Nothing recorded yet: fall back to rest so the session still lasts
            # the matched duration. The log marks it with zero transitions.
            return OfflinePlan(mode, session, self.seed, self.budget, rest_ms=self.budget.duration_ms)
        rng = self.rng(session)
        if mode == "random":
            pool = buffer.steps()
            picks = [pool[rng.randrange(len(pool))] for _ in range(self.budget.transitions)]
            snippets = tuple(Snippet((step,)) for step in picks)
            return OfflinePlan(mode, session, self.seed, self.budget, snippets, candidates=len(pool))
        windows = self.candidate_windows(buffer)
        weights = self.priorities(buffer, windows) if mode == "adaptive" else [1.0] * len(windows)
        snippets = self._fill(windows, weights, rng, keep_priority=mode == "adaptive")
        return OfflinePlan(mode, session, self.seed, self.budget, snippets, candidates=len(windows))

    # ordered machinery -------------------------------------------------------
    def candidate_windows(self, buffer: ReplayBuffer) -> list[tuple[int, tuple[RecordedStep, ...]]]:
        """Every contiguous window, in a fixed order: oldest episode first, then start index.

        Episodes at least ``snippet_length`` long contribute every length-L window.
        Shorter episodes contribute one window holding the whole episode, so short
        goal arrivals and timeouts stay eligible.
        """
        length = self.budget.snippet_length
        windows = []
        for episode_index, (_, steps) in enumerate(buffer.episodes):
            if len(steps) <= length:
                windows.append((episode_index, steps))
            else:
                for start in range(len(steps) - length + 1):
                    windows.append((episode_index, steps[start:start + length]))
        return windows

    def priorities(
        self, buffer: ReplayBuffer, windows: Sequence[tuple[int, tuple[RecordedStep, ...]]],
    ) -> list[float]:
        p = self.priority
        episodes = buffer.episodes
        newest = len(episodes) - 1
        reached_goal = [any(step.is_goal for step in steps) for _, steps in episodes]
        counts = buffer.observation_counts()
        scores = []
        for episode_index, steps in windows:
            success = 1.0 if reached_goal[episode_index] else 0.0
            rarity = sum(1 / math.sqrt(counts[s.transition.observation]) for s in steps) / len(steps)
            recency = math.exp(-(newest - episode_index) / p.recency_tau)
            score = p.floor + p.success * success + p.rarity * rarity + p.recency * recency
            scores.append(score ** p.alpha)
        return scores

    def _fill(self, windows, weights, rng: Random, *, keep_priority: bool) -> tuple[Snippet, ...]:
        """Draw windows with replacement until the transition budget is met exactly.

        Ties: windows are enumerated in a fixed order and drawn by cumulative
        weight, so equal priorities give equal probability and a given seed
        always picks the same windows. If the last draw would overshoot, it is
        cut to its first ``remaining`` steps, which keeps it contiguous.
        """
        remaining = self.budget.transitions
        total = sum(weights)
        snippets = []
        while remaining > 0:
            index = rng.choices(range(len(windows)), cum_weights=_cumulative(weights), k=1)[0]
            steps = windows[index][1][:remaining]
            priority = weights[index] / total if keep_priority else None
            snippets.append(Snippet(steps, priority))
            remaining -= len(steps)
        return tuple(snippets)


def _cumulative(weights: Sequence[float]) -> list[float]:
    out, running = [], 0.0
    for weight in weights:
        running += weight
        out.append(running)
    return out


def successor_rate(plan: OfflinePlan) -> float | None:
    """Fraction of adjacent replayed transitions (across the whole plan) that are true successors.

    Ordered modes score near 1, broken only at snippet boundaries. Independent
    random replay scores near chance. Returns None when fewer than two transitions.
    """
    flat = [step for snippet in plan.snippets for step in snippet.steps]
    pairs = list(zip(flat, flat[1:]))
    if not pairs:
        return None
    return sum(is_contiguous(pair) for pair in pairs) / len(pairs)
