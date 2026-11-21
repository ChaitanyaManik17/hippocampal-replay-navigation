"""Conventional RL baselines sharing the same observation and transition API."""

from collections import deque
from random import Random
from typing import Protocol

from maze import Transition

ACTION_COUNT = 4


class Agent(Protocol):
    def act(self, observation: int, *, training: bool) -> int: ...
    def observe(self, transition: Transition) -> None: ...


class RandomAgent:
    def __init__(self, seed: int):
        self.training_rng = Random(seed)
        self.evaluation_seed = seed + 1
        self.evaluation_rng = Random(self.evaluation_seed)

    def begin_evaluation(self) -> None:
        self.evaluation_rng.seed(self.evaluation_seed)

    def act(self, observation: int, *, training: bool) -> int:
        rng = self.training_rng if training else self.evaluation_rng
        return rng.randrange(ACTION_COUNT)

    def observe(self, transition: Transition) -> None:
        pass


def epsilon_at(step: int, decay_steps: int) -> float:
    return max(0.05, 1.0 - 0.95 * step / decay_steps)


class QLearning:
    def __init__(self, num_observations: int, seed: int, decay_steps: int):
        self.values = [[0.0] * ACTION_COUNT for _ in range(num_observations)]
        self.rng = Random(seed)
        self.decay_steps = decay_steps
        self.steps = 0

    def act(self, observation: int, *, training: bool) -> int:
        if training and self.rng.random() < epsilon_at(self.steps, self.decay_steps):
            return self.rng.randrange(ACTION_COUNT)
        values = self.values[observation]
        return max(range(ACTION_COUNT), key=values.__getitem__)

    def observe(self, transition: Transition) -> None:
        old = self.values[transition.observation][transition.action]
        future = 0.0 if transition.terminated else max(self.values[transition.next_observation])
        self.values[transition.observation][transition.action] += (
            0.15 * (transition.reward + 0.97 * future - old)
        )
        self.steps += 1


class DQN:
    """CPU DQN with one-hot position, uniform replay, and a target network."""

    def __init__(self, num_observations: int, seed: int, decay_steps: int):
        import torch
        from torch import nn

        self.torch = torch
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = nn.Sequential(
                nn.Linear(num_observations, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, ACTION_COUNT),
            )
        from copy import deepcopy

        self.target = deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=0.001)
        self.loss = nn.SmoothL1Loss()
        self.cues = torch.eye(num_observations)
        self.buffer: deque[Transition] = deque(maxlen=2000)
        self.rng = Random(seed)
        self.batch_rng = Random(seed + 1)
        self.decay_steps = decay_steps
        self.steps = 0
        self.gradient_steps = 0

    def act(self, observation: int, *, training: bool) -> int:
        if training and self.rng.random() < epsilon_at(self.steps, self.decay_steps):
            return self.rng.randrange(ACTION_COUNT)
        with self.torch.no_grad():
            return int(self.online(self.cues[observation]).argmax().item())

    def observe(self, transition: Transition) -> None:
        self.buffer.append(transition)
        self.steps += 1
        if len(self.buffer) < 64 or self.steps % 4:
            return
        torch = self.torch
        batch = self.batch_rng.sample(list(self.buffer), 64)
        states = self.cues[[item.observation for item in batch]]
        next_states = self.cues[[item.next_observation for item in batch]]
        actions = torch.tensor([item.action for item in batch]).unsqueeze(1)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32)
        continuation = torch.tensor([not item.terminated for item in batch])
        prediction = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            expected = rewards + 0.97 * continuation * self.target(next_states).amax(1)
        self.optimizer.zero_grad(set_to_none=True)
        self.loss(prediction, expected).backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % 50 == 0:
            self.target.load_state_dict(self.online.state_dict())


def make_agent(name: str, num_observations: int, seed: int, decay_steps: int) -> Agent:
    if name == "random":
        return RandomAgent(seed)
    if name == "q_learning":
        return QLearning(num_observations, seed, decay_steps)
    if name == "dqn":
        return DQN(num_observations, seed, decay_steps)
    raise ValueError(f"unknown baseline: {name}")
