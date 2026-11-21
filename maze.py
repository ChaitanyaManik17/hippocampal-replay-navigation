"""Deterministic four-action mazes and evaluation-only shortest distances."""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from random import Random

Cell = tuple[int, int]


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class MapSpec:
    id: str
    rows: tuple[str, ...]
    starts: tuple[Cell, ...]
    goal: Cell
    blockage: Cell | None = None


MAPS = {
    spec.id: spec
    for spec in (
        MapSpec("corridor-v1", ("#######", "#.....#", "#######"), ((1, 1),), (1, 5)),
        MapSpec(
            "t-maze-v1",
            ("#######", "#.....#", "###.###", "###.###", "#######"),
            ((3, 3),), (1, 5),
        ),
        MapSpec(
            "loop-7x7-v1",
            ("#######", "#.....#", "#.###.#", "#.###.#", "#.###.#", "#.....#", "#######"),
            ((1, 1), (2, 1), (3, 1)), (1, 5), (1, 3),
        ),
    )
}


@dataclass(frozen=True)
class Transition:
    observation: int
    action: int
    reward: float
    next_observation: int
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class GridMaze:
    """Encode position as row * width + col, with stable IDs after blockage."""

    def __init__(self, map_id: str, seed: int, max_steps: int = 150, *, blocked: bool = False):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.spec = MAPS[map_id]
        if blocked and self.spec.blockage is None:
            raise ValueError(f"{map_id} has no blocked variant")
        self.blocked = blocked
        self.max_steps = max_steps
        self.rng = Random(seed)
        self.width = len(self.spec.rows[0])
        self.num_observations = self.width * len(self.spec.rows)
        self.position: Cell | None = None
        self.elapsed = 0
        self.finished = True
        self.open_cells = frozenset(
            (row, col)
            for row, line in enumerate(self.spec.rows)
            for col, char in enumerate(line)
            if char == "." and (not blocked or (row, col) != self.spec.blockage)
        )

    def encode(self, cell: Cell) -> int:
        return cell[0] * self.width + cell[1]

    def reset(self, *, seed: int | None = None, start: Cell | None = None) -> int:
        if seed is not None:
            self.rng.seed(seed)
        start = self.rng.choice(self.spec.starts) if start is None else start
        if start not in self.open_cells or start == self.spec.goal:
            raise ValueError("start must be an open cell other than the goal")
        self.position = start
        self.elapsed = 0
        self.finished = False
        return self.encode(start)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        if self.finished or self.position is None:
            raise RuntimeError("reset the maze before stepping")
        if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(MOVES):
            raise ValueError("action must be an integer in [0, 3]")
        dr, dc = MOVES[action]
        candidate = (self.position[0] + dr, self.position[1] + dc)
        collision = candidate not in self.open_cells
        if not collision:
            self.position = candidate
        self.elapsed += 1
        terminated = self.position == self.spec.goal
        truncated = self.elapsed >= self.max_steps and not terminated
        self.finished = terminated or truncated
        reward = 1.0 if terminated else (-0.05 if collision else -0.005)
        return self.encode(self.position), reward, terminated, truncated, {"collision": collision}


def shortest_distance(maze: GridMaze, start: Cell) -> int | None:
    """Benchmark path length without exposing a planner to the agent."""
    if start not in maze.open_cells:
        return None
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        cell, distance = queue.popleft()
        if cell == maze.spec.goal:
            return distance
        for dr, dc in MOVES:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if neighbor in maze.open_cells and neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None
