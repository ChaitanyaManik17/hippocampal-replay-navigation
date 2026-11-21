"""Run paired baseline experiments and save one row per frozen rollout."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from baselines import Agent, make_agent
from maze import MAPS, Cell, GridMaze, Transition, shortest_distance


@dataclass(frozen=True)
class RunConfig:
    map_id: str
    agents: list[str]
    seeds: list[int]
    max_steps: int
    pretrain_steps: int
    adapt_steps: int
    checkpoint_every: int

    def __post_init__(self):
        if self.map_id not in MAPS:
            raise ValueError(f"unknown map: {self.map_id}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("provide unique seeds")
        allowed = {"random", "q_learning", "dqn"}
        if not self.agents or len(set(self.agents)) != len(self.agents) or set(self.agents) - allowed:
            raise ValueError("provide unique supported baseline names")
        if min(self.max_steps, self.pretrain_steps, self.checkpoint_every) <= 0:
            raise ValueError("max_steps, pretrain_steps and checkpoint_every must be positive")
        if self.adapt_steps < 0 or (self.adapt_steps and MAPS[self.map_id].blockage is None):
            raise ValueError("adapt_steps must be nonnegative and requires a blocked map variant")


@dataclass(frozen=True)
class EvaluationRow:
    map_id: str
    agent: str
    seed: int
    phase: str
    phase_step: int
    environment_step: int
    start: str
    success: int
    rollout_steps: int
    shortest_steps: int
    path_efficiency: float | None
    collisions: int


def rollout(agent: Agent, maze: GridMaze, start: Cell) -> tuple[bool, int, int]:
    observation = maze.reset(start=start)
    collisions = 0
    while True:
        action = agent.act(observation, training=False)
        observation, _, terminated, truncated, info = maze.step(action)
        collisions += int(info["collision"])
        if terminated or truncated:
            return terminated, maze.elapsed, collisions


def evaluate(
    agent: Agent, maze: GridMaze, name: str, seed: int, phase: str,
    phase_step: int, environment_step: int,
):
    reset_evaluation = getattr(agent, "begin_evaluation", None)
    if reset_evaluation is not None:
        reset_evaluation()
    for start in maze.spec.starts:
        optimal = shortest_distance(maze, start)
        if optimal is None:
            raise ValueError(f"goal unreachable from {start}")
        success, steps, collisions = rollout(agent, maze, start)
        yield EvaluationRow(
            maze.spec.id, name, seed, phase, phase_step, environment_step,
            f"{start[0]}:{start[1]}", int(success), steps, optimal,
            optimal / steps if success else None, collisions,
        )


def train_phase(
    agent: Agent, config: RunConfig, name: str, seed: int, *, blocked: bool,
):
    phase = "blocked" if blocked else "original"
    budget = config.adapt_steps if blocked else config.pretrain_steps
    offset = config.pretrain_steps if blocked else 0
    maze = GridMaze(config.map_id, seed, config.max_steps, blocked=blocked)
    eval_maze = GridMaze(config.map_id, seed, config.max_steps, blocked=blocked)
    yield from evaluate(agent, eval_maze, name, seed, phase, 0, offset)
    observation = maze.reset()
    for step in range(1, budget + 1):
        action = agent.act(observation, training=True)
        next_observation, reward, terminated, truncated, _ = maze.step(action)
        transition = Transition(observation, action, reward, next_observation, terminated, truncated)
        agent.observe(transition)
        observation = maze.reset() if transition.done and step < budget else next_observation
        if step % config.checkpoint_every == 0 or step == budget:
            yield from evaluate(agent, eval_maze, name, seed, phase, step, offset + step)


def run(config: RunConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    csv_path = output_dir / "evaluation.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(EvaluationRow.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for seed in config.seeds:
            for name in config.agents:
                maze = GridMaze(config.map_id, seed, config.max_steps)
                agent = make_agent(name, maze.num_observations, seed, config.pretrain_steps)
                for row in train_phase(agent, config, name, seed, blocked=False):
                    writer.writerow(asdict(row))
                if config.adapt_steps:
                    for row in train_phase(agent, config, name, seed, blocked=True):
                        writer.writerow(asdict(row))
                file.flush()
                print(f"Finished {name}, seed {seed}", flush=True)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = RunConfig(**json.loads(args.config.read_text()))
    print(run(config, args.output))


if __name__ == "__main__":
    main()
