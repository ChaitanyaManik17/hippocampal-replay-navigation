"""Deterministic development inputs for the four CS 525 work packages."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, Sequence

from maze import Action, GridMaze, Transition

FIXTURE_PATH = Path(__file__).with_name("parallel_fixtures.json")
SCHEMA_VERSION = 1


class ReplayAgent(Protocol):
    """Model boundary exercised by the synthetic agent and the eventual SNN."""

    def begin_episode(self) -> None: ...
    def act(self, observation: int, *, training: bool) -> int: ...
    def observe(self, transition: Transition) -> None: ...
    def replay(self, sequence: Sequence[Transition], *, step_ms: int) -> None: ...
    def rest(self, duration_ms: int) -> None: ...


class TraceAgent:
    """Runner test double that records calls and cycles through a supplied action script."""

    def __init__(self, actions: Sequence[int]):
        if not actions or any(action not in range(4) for action in actions):
            raise ValueError("provide a nonempty sequence of maze actions")
        self.actions = tuple(actions)
        self.events: list[tuple] = []
        self._cursor = 0

    def begin_episode(self) -> None:
        self._cursor = 0
        self.events.append(("begin_episode",))

    def act(self, observation: int, *, training: bool) -> int:
        action = self.actions[self._cursor % len(self.actions)]
        self._cursor += 1
        self.events.append(("act", observation, training))
        return action

    def observe(self, transition: Transition) -> None:
        self.events.append(("observe", transition))

    def replay(self, sequence: Sequence[Transition], *, step_ms: int) -> None:
        self.events.append(("replay", tuple(sequence), step_ms))

    def rest(self, duration_ms: int) -> None:
        self.events.append(("rest", duration_ms))


class RecallProbe(Protocol):
    def recall(
        self, cue_channels: Sequence[int], *, cue_ms: int, recall_bins: int,
        bin_ms: int, recurrence: bool,
    ) -> Sequence[int | None]:
        """Return one decoded location or None per bin after cue offset."""
        ...


class ScriptedRecall:
    """Probe test double with fixed decoded outputs and an auditable input boundary."""

    def __init__(self, outputs: Sequence[int | None]):
        self.outputs = tuple(outputs)
        self.calls: list[tuple] = []

    def recall(
        self, cue_channels: Sequence[int], *, cue_ms: int, recall_bins: int,
        bin_ms: int, recurrence: bool,
    ) -> Sequence[int | None]:
        if recall_bins != len(self.outputs):
            raise ValueError("fixture output length must match recall_bins")
        self.calls.append((tuple(cue_channels), cue_ms, recall_bins, bin_ms, recurrence))
        return self.outputs


def _record_episode(
    episode_id: str, actions: Sequence[Action], *, blocked: bool = False,
) -> dict:
    """Record a deliberately scripted route through the real environment."""
    maze = GridMaze("loop-7x7-v1", seed=11, max_steps=len(actions), blocked=blocked)
    start = maze.spec.starts[0]
    observation = maze.reset(start=start)
    rows = []
    for index, action in enumerate(actions):
        next_observation, reward, terminated, truncated, info = maze.step(action)
        transition = Transition(observation, int(action), reward, next_observation, terminated, truncated)
        rows.append({"time_index": index, **asdict(transition), "collision": info["collision"]})
        observation = next_observation
        if transition.done:
            break
    return {
        "episode_id": episode_id,
        "map_id": maze.spec.id,
        "phase": "blocked" if blocked else "original",
        "start": list(start),
        "step_limit": len(actions),
        "transitions": rows,
    }


def build_fixtures() -> dict:
    right, down, up = Action.RIGHT, Action.DOWN, Action.UP
    original = _record_episode("short-route", [right] * 4)
    continuation = [row["next_observation"] for row in original["transitions"]][1:]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "synthetic_development_fixtures",
        "observation_count": 49,
        "channels_per_observation": 4,
        "episodes": [
            original,
            _record_episode("wall-and-timeout", [up, right], blocked=True),
            _record_episode("blocked-detour", [down] * 4 + [right] * 4 + [up] * 4, blocked=True),
        ],
        "plasticity_cases": [
            {"id": "causal", "pre_ms": 10, "post_ms": 20, "pulse_ms": 120,
             "reward": 1, "expected_delta_sign": 1},
            {"id": "reverse", "pre_ms": 20, "post_ms": 10, "pulse_ms": 120,
             "reward": 1, "expected_delta_sign": -1},
            {"id": "zero-reward", "pre_ms": 10, "post_ms": 20, "pulse_ms": 120,
             "reward": 0, "expected_delta_sign": 0},
        ],
        "recall_case": {
            "id": "partial-route-cue",
            "cue_channels": [36, 37],
            "cue_ms": 25,
            "recall_bins": 3,
            "bin_ms": 25,
            "expected_locations": continuation,
            "scripted_outputs": {
                "intact": continuation,
                "recurrent_off": [None, None, None],
                "one_error": [10, None, 12],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="regenerate the committed fixture")
    mode.add_argument("--check", action="store_true", help="compare committed data with the environment")
    args = parser.parse_args()
    generated = json.dumps(build_fixtures(), indent=2) + "\n"
    if args.write:
        FIXTURE_PATH.write_text(generated, encoding="utf-8")
    elif args.check:
        if FIXTURE_PATH.read_text(encoding="utf-8") != generated:
            parser.exit(1, "Fixture differs from generated data. Check the contract and source.\n")
        print(f"Verified fixture schema v{SCHEMA_VERSION}: 3 maze episodes, 3 plasticity cases, 1 recall case.")
    else:
        print(generated, end="")


if __name__ == "__main__":
    main()
