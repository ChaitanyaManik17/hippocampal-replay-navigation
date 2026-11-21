"""Run online maze training with interleaved offline replay, and audit the replay controls.

Two entry points:

    python -m replay_runner --audit --output runs/replay_audit
        Four-condition audit on the synthetic fixture episodes with TraceAgent.
        Writes audit.csv, audit.md and replay_log.csv. No maze is stepped.

    python -m replay_runner --config replay_smoke_config.json --output runs/replay_demo
        Online training on GridMaze with offline sessions every ``replay_every``
        real actions. ``--agent trace`` uses the stand-in; ``--agent module:factory``
        loads the real model, where ``factory(num_observations, seed)`` returns a
        ReplayAgent (see parallel_fixtures.ReplayAgent).

Budget rules (identical for every condition):
* ``environment_step`` counts training-environment actions only.
* Offline sessions happen at the same real-action counts in every condition and
  each lasts ``replay_transitions * step_ms`` simulated milliseconds.
* After an offline session the runner starts a fresh training episode, in every
  condition including rest, so the interruption itself is matched.
* Evaluation uses a separate maze and a separate agent: a snapshot of the
  training agent taken at the checkpoint (``agent.evaluation_copy()`` if the
  model provides it, else ``copy.deepcopy``). ``begin_episode()`` is called on
  the snapshot before every rollout. The training agent's weights, transient
  state and RNG are never touched, so training is identical for any
  evaluation schedule. Evaluation never calls ``observe`` or writes the buffer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from experiment import EvaluationRow, rollout
from maze import shortest_distance
from maze import MAPS, GridMaze, Transition
from parallel_fixtures import FIXTURE_PATH, ReplayAgent, TraceAgent
from replay_scheduler import (
    MODES, AdaptivePriority, OfflinePlan, ReplayBudget, ReplayBuffer, ReplayScheduler, successor_rate,
)


# --------------------------------------------------------------------------- maze guard


class OfflineMazeAccess(RuntimeError):
    """Raised if anything steps or resets the maze during an offline session."""


class GuardedMaze:
    """Pass-through GridMaze wrapper that refuses all interaction while offline."""

    def __init__(self, maze: GridMaze):
        self.maze = maze
        self.offline = False
        self.offline_calls = 0  # attempted calls while offline (should stay 0)
        self.online_steps = 0

    def reset(self, **kwargs) -> int:
        self._check("reset")
        return self.maze.reset(**kwargs)

    def step(self, action: int):
        self._check("step")
        self.online_steps += 1
        return self.maze.step(action)

    def _check(self, name: str) -> None:
        if self.offline:
            self.offline_calls += 1
            raise OfflineMazeAccess(f"maze.{name} called during offline replay")

    @contextmanager
    def offline_session(self) -> Iterator[None]:
        self.offline = True
        try:
            yield
        finally:
            self.offline = False

    def __getattr__(self, name):
        return getattr(self.maze, name)


# --------------------------------------------------------------------------- logs


LOG_FIELDS = [
    "condition", "seed", "session", "environment_step", "call_index", "kind",
    "source_episode", "phase", "start_time_index", "end_time_index", "length",
    "observations", "actions", "rewards", "terminated", "truncated",
    "priority", "contiguous", "duration_ms",
]

SESSION_FIELDS = [
    "condition", "seed", "session", "environment_step", "replay_calls", "rest_calls",
    "replayed_transitions", "update_count", "offline_ms", "buffer_transitions",
    "candidates", "goal_transitions", "successor_rate", "maze_calls_offline",
]


def _joined(values) -> str:
    return " ".join(str(v) for v in values)


def run_offline_session(
    agent: ReplayAgent, plan: OfflinePlan, guard: GuardedMaze | None, *,
    condition: str, seed: int, environment_step: int, buffer_transitions: int,
) -> tuple[list[dict], dict]:
    """Deliver one plan to the agent. Returns per-call log rows and a session summary."""
    rows: list[dict] = []
    before = guard.offline_calls if guard else 0
    session = guard.offline_session() if guard else _null_context()
    with session:
        for call_index, snippet in enumerate(plan.snippets):
            agent.replay(snippet.transitions, step_ms=plan.budget.step_ms)
            t = snippet.transitions
            rows.append({
                "condition": condition, "seed": seed, "session": plan.session,
                "environment_step": environment_step, "call_index": call_index, "kind": "replay",
                "source_episode": snippet.source_episode, "phase": snippet.steps[0].phase,
                "start_time_index": snippet.start_index, "end_time_index": snippet.end_index,
                "length": len(t),
                "observations": _joined(x.observation for x in t),
                "actions": _joined(x.action for x in t),
                "rewards": _joined(x.reward for x in t),
                "terminated": _joined(int(x.terminated) for x in t),
                "truncated": _joined(int(x.truncated) for x in t),
                "priority": "" if snippet.priority is None else f"{snippet.priority:.6f}",
                "contiguous": int(snippet.is_contiguous()),
                "duration_ms": len(t) * plan.budget.step_ms,
            })
        if plan.rest_ms:
            agent.rest(plan.rest_ms)
            rows.append({
                "condition": condition, "seed": seed, "session": plan.session,
                "environment_step": environment_step, "call_index": len(plan.snippets),
                "kind": "rest", "length": 0, "duration_ms": plan.rest_ms,
            })
    rate = successor_rate(plan)
    summary = {
        "condition": condition, "seed": seed, "session": plan.session,
        "environment_step": environment_step,
        "replay_calls": len(plan.snippets), "rest_calls": int(plan.rest_ms > 0),
        "replayed_transitions": plan.replayed_transitions, "update_count": plan.update_count,
        "offline_ms": plan.duration_ms, "buffer_transitions": buffer_transitions,
        "candidates": plan.candidates,
        "goal_transitions": sum(s.is_goal for sn in plan.snippets for s in sn.steps),
        "successor_rate": "" if rate is None else f"{rate:.3f}",
        "maze_calls_offline": (guard.offline_calls - before) if guard else 0,
    }
    return rows, summary


@contextmanager
def _null_context():
    yield


# --------------------------------------------------------------------------- online runs


@dataclass(frozen=True)
class ReplayRunConfig:
    map_id: str
    conditions: list[str]
    seeds: list[int]
    max_steps: int
    pretrain_steps: int
    adapt_steps: int
    checkpoint_every: int
    replay_every: int
    replay_transitions: int
    snippet_length: int
    step_ms: int = 25
    buffer_episodes: int | None = None
    priority: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.map_id not in MAPS:
            raise ValueError(f"unknown map: {self.map_id}")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions) \
                or set(self.conditions) - set(MODES):
            raise ValueError(f"conditions must be unique names from {MODES}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("provide unique seeds")
        if min(self.max_steps, self.pretrain_steps, self.checkpoint_every, self.replay_every) <= 0:
            raise ValueError("step counts must be positive")
        if self.adapt_steps < 0 or (self.adapt_steps and MAPS[self.map_id].blockage is None):
            raise ValueError("adapt_steps must be nonnegative and requires a blocked map variant")
        ReplayBudget(self.replay_transitions, self.snippet_length, self.step_ms)
        AdaptivePriority(**self.priority)

    def scheduler(self, seed: int) -> ReplayScheduler:
        return ReplayScheduler(
            ReplayBudget(self.replay_transitions, self.snippet_length, self.step_ms),
            AdaptivePriority(**self.priority), seed,
        )


AgentFactory = Callable[[int, int], ReplayAgent]


def evaluation_snapshot(agent: ReplayAgent) -> ReplayAgent:
    """Separate evaluation agent carrying the current learned weights.

    A model can define ``evaluation_copy()`` to build a lighter frozen copy
    (for example, sharing read-only weight arrays). Otherwise the whole agent
    is deep-copied, which also copies its RNG state, so draws made during
    evaluation never advance the training agent's generator.
    """
    make_copy = getattr(agent, "evaluation_copy", None)
    snapshot = make_copy() if make_copy is not None else copy.deepcopy(agent)
    if snapshot is agent:
        raise RuntimeError("evaluation_copy() must return a separate agent")
    return snapshot


def evaluate_frozen(
    agent: ReplayAgent, maze: GridMaze, name: str, seed: int, phase: str,
    phase_step: int, environment_step: int,
) -> Iterator[EvaluationRow]:
    """Frozen rollouts from every start on a snapshot; same rows as experiment.evaluate."""
    evaluator = evaluation_snapshot(agent)
    for start in maze.spec.starts:
        optimal = shortest_distance(maze, start)
        if optimal is None:
            raise ValueError(f"goal unreachable from {start}")
        evaluator.begin_episode()
        success, steps, collisions = rollout(evaluator, maze, start)
        yield EvaluationRow(
            maze.spec.id, name, seed, phase, phase_step, environment_step,
            f"{start[0]}:{start[1]}", int(success), steps, optimal,
            optimal / steps if success else None, collisions,
        )


def trace_factory(num_observations: int, seed: int) -> ReplayAgent:
    """Deterministic stand-in: a fixed loop of moves that eventually reaches the loop-maze goal."""
    return TraceAgent([3, 3, 3, 3, 0, 1, 1, 3])


def train_condition(
    agent: ReplayAgent, config: ReplayRunConfig, condition: str, seed: int,
    *, log_rows: list, session_rows: list,
) -> Iterator[EvaluationRow]:
    """Train one agent in one replay condition, yielding frozen evaluation rows."""
    scheduler = config.scheduler(seed)
    buffer = ReplayBuffer(config.buffer_episodes)
    session = 0
    environment_step = 0
    for blocked in (False, True):
        if blocked and not config.adapt_steps:
            break
        phase = "blocked" if blocked else "original"
        budget = config.adapt_steps if blocked else config.pretrain_steps
        maze = GuardedMaze(GridMaze(config.map_id, seed, config.max_steps, blocked=blocked))
        eval_maze = GridMaze(config.map_id, seed, config.max_steps, blocked=blocked)
        # Evaluate the inherited policy before any training action on this map.
        yield from evaluate_frozen(agent, eval_maze, condition, seed, phase, 0, environment_step)
        episode = 0

        def new_episode() -> int:
            nonlocal episode
            agent.begin_episode()
            buffer.start_episode(f"{phase}-{episode:05d}")
            episode += 1
            return maze.reset()

        observation = new_episode()
        for phase_step in range(1, budget + 1):
            action = agent.act(observation, training=True)
            next_observation, reward, terminated, truncated, info = maze.step(action)
            environment_step += 1
            transition = Transition(observation, action, reward, next_observation, terminated, truncated)
            agent.observe(transition)
            buffer.record(transition, phase=phase, map_id=config.map_id, collision=info["collision"])
            observation = next_observation
            last = phase_step == budget
            offline = phase_step % config.replay_every == 0
            if offline:
                plan = scheduler.plan(condition, buffer, session)
                rows, summary = run_offline_session(
                    agent, plan, maze, condition=condition, seed=seed,
                    environment_step=environment_step, buffer_transitions=len(buffer),
                )
                log_rows.extend(rows)
                session_rows.append(summary)
                session += 1
            if not last and (transition.done or offline):
                buffer.close_episode()
                observation = new_episode()
            if phase_step % config.checkpoint_every == 0 or last:
                yield from evaluate_frozen(agent, eval_maze, condition, seed, phase, phase_step, environment_step)


def load_factory(spec: str) -> AgentFactory:
    if spec == "trace":
        return trace_factory
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("--agent must be 'trace' or 'module:factory'")
    return getattr(importlib.import_module(module_name), attr)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n", restval="")
        writer.writeheader()
        writer.writerows(rows)


def run(config: ReplayRunConfig, output_dir: Path, factory: AgentFactory) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    eval_path = output_dir / "evaluation.csv"
    log_rows: list[dict] = []
    session_rows: list[dict] = []
    num_observations = GridMaze(config.map_id, 0).num_observations
    with eval_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(EvaluationRow.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for seed in config.seeds:
            for condition in config.conditions:
                agent = factory(num_observations, seed)
                for row in train_condition(
                    agent, config, condition, seed, log_rows=log_rows, session_rows=session_rows,
                ):
                    writer.writerow(asdict(row))
                file.flush()
                print(f"Finished {condition}, seed {seed}", flush=True)
    _write_csv(output_dir / "replay_log.csv", LOG_FIELDS, log_rows)
    _write_csv(output_dir / "offline_sessions.csv", SESSION_FIELDS, session_rows)
    return eval_path


# --------------------------------------------------------------------------- audit


AUDIT_FIELDS = [
    "condition", "replay_calls", "rest_calls", "replayed_transitions", "update_count",
    "offline_ms", "short-route", "wall-and-timeout", "blocked-detour", "goal_transitions",
    "within_episode_order", "successor_rate", "flags_preserved", "agent_received_plan",
    "maze_calls_offline",
]


def audit(
    output_dir: Path, *, seed: int = 0, transitions: int = 24, snippet_length: int = 4,
    step_ms: int = 25,
) -> list[dict]:
    """Four-condition audit on the synthetic fixture episodes (development evidence only)."""
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    buffer = ReplayBuffer.from_fixture(data)
    recorded = {(s.episode_id, s.time_index): s.transition for s in buffer.steps()}
    scheduler = ReplayScheduler(ReplayBudget(transitions, snippet_length, step_ms), seed=seed)
    # A real maze wrapped in the guard: any offline access would raise and be counted.
    guard = GuardedMaze(GridMaze("loop-7x7-v1", seed, blocked=True))
    table, log_rows = [], []
    for mode in MODES:
        agent = TraceAgent([0])
        plan = scheduler.plan(mode, buffer, session=0)
        rows, summary = run_offline_session(
            agent, plan, guard, condition=mode, seed=seed, environment_step=0,
            buffer_transitions=len(buffer),
        )
        log_rows.extend(rows)
        # Measure what the agent actually received, not what the plan claims.
        replay_events = [event for event in agent.events if event[0] == "replay"]
        sent = [event[1] for event in replay_events]
        replayed_ms = sum(len(event[1]) * event[2] for event in replay_events)
        rested = sum(event[1] for event in agent.events if event[0] == "rest")
        per_episode = {eid: 0 for eid, _ in buffer.episodes}
        for snippet in plan.snippets:
            per_episode[snippet.source_episode] += len(snippet.steps)
        ordered = mode in ("ordered", "adaptive")
        table.append({
            "condition": mode,
            "replay_calls": summary["replay_calls"],
            "rest_calls": summary["rest_calls"],
            "replayed_transitions": summary["replayed_transitions"],
            "update_count": summary["update_count"],
            "offline_ms": replayed_ms + rested,
            **per_episode,
            "goal_transitions": summary["goal_transitions"],
            "within_episode_order": (
                ("pass" if all(s.is_contiguous() for s in plan.snippets) else "FAIL") if ordered else "n/a"
            ),
            "successor_rate": summary["successor_rate"] or "n/a",
            "flags_preserved": "pass" if all(
                recorded[(st.episode_id, st.time_index)] == st.transition
                for sn in plan.snippets for st in sn.steps
            ) else "FAIL",
            "agent_received_plan": "pass" if tuple(sent) == tuple(s.transitions for s in plan.snippets) else "FAIL",
            "maze_calls_offline": summary["maze_calls_offline"],
        })
    priority_rows = priority_table(scheduler, buffer)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "audit.csv", AUDIT_FIELDS, table)
    _write_csv(output_dir / "replay_log.csv", LOG_FIELDS, log_rows)
    _write_csv(output_dir / "priorities.csv", PRIORITY_FIELDS, priority_rows)
    (output_dir / "audit.md").write_text(
        _audit_markdown(table, seed, transitions, snippet_length, step_ms)
        + _priority_markdown(priority_rows, scheduler)
    )
    return table


PRIORITY_FIELDS = [
    "source_episode", "start_time_index", "length", "reached_goal",
    "ordered_prob", "adaptive_prob", "ordered_empirical", "adaptive_empirical",
]


def priority_table(scheduler: ReplayScheduler, buffer: ReplayBuffer, sessions: int = 2000) -> list[dict]:
    """Analytic window probabilities next to empirical draw shares over many seeded sessions."""
    windows = scheduler.candidate_windows(buffer)
    weights = scheduler.priorities(buffer, windows)
    total = sum(weights)
    key = {(steps[0].episode_id, steps[0].time_index): i for i, (_, steps) in enumerate(windows)}
    draws = {"ordered": [0] * len(windows), "adaptive": [0] * len(windows)}
    for mode in draws:
        for session in range(sessions):
            for snippet in scheduler.plan(mode, buffer, session).snippets:
                draws[mode][key[(snippet.source_episode, snippet.start_index)]] += 1
    rows = []
    for i, (episode_index, steps) in enumerate(windows):
        rows.append({
            "source_episode": steps[0].episode_id, "start_time_index": steps[0].time_index,
            "length": len(steps), "reached_goal": int(any(s.is_goal for s in buffer.episodes[episode_index][1])),
            "ordered_prob": f"{1 / len(windows):.3f}", "adaptive_prob": f"{weights[i] / total:.3f}",
            "ordered_empirical": f"{draws['ordered'][i] / sum(draws['ordered']):.3f}",
            "adaptive_empirical": f"{draws['adaptive'][i] / sum(draws['adaptive']):.3f}",
        })
    return rows


def _priority_markdown(rows, scheduler: ReplayScheduler) -> str:
    p = scheduler.priority
    lines = [
        "## Window probabilities",
        "",
        f"Adaptive weights: success {p.success}, rarity {p.rarity}, recency {p.recency}"
        f" (tau {p.recency_tau} episodes), alpha {p.alpha}, floor {p.floor}.",
        "Empirical columns are draw shares over 2,000 seeded sessions. The last draw of a session",
        "may be cut short to fit the budget; it still counts as a draw of its window here.",
        "",
        "| " + " | ".join(PRIORITY_FIELDS) + " |",
        "|" + "---|" * len(PRIORITY_FIELDS),
    ]
    lines += ["| " + " | ".join(str(r[f]) for f in PRIORITY_FIELDS) + " |" for r in rows]
    lines.append("")
    return "\n".join(lines)


def _audit_markdown(table, seed, transitions, snippet_length, step_ms) -> str:
    lines = [
        "# Replay scheduler audit (synthetic fixtures)",
        "",
        "Source: `parallel_fixtures.json` episodes and `TraceAgent`. This is a development check of the",
        "sampler and runner, not a model result.",
        "",
        f"Seed {seed}; budget {transitions} transitions per active session; max snippet length",
        f"{snippet_length}; {step_ms} ms per replayed transition; matched duration {transitions * step_ms} ms.",
        "Columns `short-route`, `wall-and-timeout`, `blocked-detour` count replayed transitions per source episode.",
        "",
        "| " + " | ".join(AUDIT_FIELDS) + " |",
        "|" + "---|" * len(AUDIT_FIELDS),
    ]
    for row in table:
        lines.append("| " + " | ".join(str(row[f]) for f in AUDIT_FIELDS) + " |")
    lines += ["", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit", action="store_true", help="run the fixture audit")
    group.add_argument("--config", type=Path, help="online run configuration (JSON)")
    parser.add_argument("--agent", default="trace", help="'trace' or module:factory")
    parser.add_argument("--seed", type=int, default=0, help="audit sampling seed")
    args = parser.parse_args()
    if args.audit:
        table = audit(args.output, seed=args.seed)
        print((args.output / "audit.md").read_text())
        if any("FAIL" in str(v) for row in table for v in row.values()):
            parser.exit(1, "Audit failed.\n")
    else:
        config = ReplayRunConfig(**json.loads(args.config.read_text()))
        print(run(config, args.output, load_factory(args.agent)))


if __name__ == "__main__":
    main()
