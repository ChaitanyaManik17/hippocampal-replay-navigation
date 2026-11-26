"""Frozen navigation scoring and cue-only recall assays (Task 4, Charvi Mayank Shastri).

The evaluator owns the ground truth. Probes and policies receive only what they
would receive in an experiment: a policy gets position IDs, a recall probe gets
cue channels and timing. Expected continuations, shortest paths and true cells
stay in this module.

Commands (run from the repository root):

    python -m memory_evaluation evidence --output runs/scoring
        Navigation scores from baseline_evaluation.csv, ScriptedRecall fixture
        scores, and the evidence figure. Recall rows are labeled synthetic.

    python -m memory_evaluation navigation runs/replay_demo/evaluation.csv --output runs/nav --max-steps 30
        Score any rollout CSV in the experiment.EvaluationRow schema and save
        tables plus navigation_curves.png. Baseline and replay runs are told
        apart by the config.json the runner wrote beside the CSV (or --kind),
        because both use the agent ID ``random`` with different meanings.

    python -m memory_evaluation assays --output runs/memory [--model module:factory]
        Recurrence, cue-noise, memory-load, route-similarity and interference
        sweeps. ``factory(num_observations, seed)`` returns a MemoryModel. Without
        --model the synthetic TransitionTableMemory stand-in is used and every
        output row is labeled with that source.

Scoring rules:
* Navigation success is pooled over every rollout, so failures and timeouts stay
  in the denominator. Path efficiency is averaged over successful rollouts only.
* Recall is scored per bin after cue offset. ``None`` is a miss; a wrong location
  is an error; both count against accuracy. When a cue admits several valid
  continuations the best-matching one is scored, and the count is logged.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import Callable, Iterable, Protocol, Sequence

from maze import MAPS, MOVES, Cell, GridMaze, Transition, shortest_distance
from parallel_fixtures import FIXTURE_PATH, RecallProbe, ScriptedRecall

CHANNELS_PER_OBSERVATION = 4
SYNTHETIC_SOURCE = "synthetic: TransitionTableMemory stand-in (not the SNN)"

# =========================================================================== navigation

ROLLOUT_FIELDS = (
    "map_id", "agent", "seed", "phase", "phase_step", "environment_step", "start",
    "success", "rollout_steps", "shortest_steps", "path_efficiency", "collisions",
)


def load_rollouts(path: Path) -> list[dict]:
    """Read and type a rollout CSV written by experiment.py or replay_runner.py."""
    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        missing = set(ROLLOUT_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"rollout CSV lacks columns: {sorted(missing)}")
        rows = []
        for raw in reader:
            rows.append({
                "map_id": raw["map_id"], "agent": raw["agent"], "seed": int(raw["seed"]),
                "phase": raw["phase"], "phase_step": int(raw["phase_step"]),
                "environment_step": int(raw["environment_step"]), "start": raw["start"],
                "success": int(raw["success"]), "rollout_steps": int(raw["rollout_steps"]),
                "shortest_steps": int(raw["shortest_steps"]),
                "path_efficiency": float(raw["path_efficiency"]) if raw["path_efficiency"] else None,
                "collisions": int(raw["collisions"]),
            })
    if not rows:
        raise ValueError("no rollout rows")
    return rows


def _parse_start(text: str) -> Cell:
    row, col = text.split(":")
    return int(row), int(col)


def audit_rollouts(rows: Iterable[dict], *, max_steps: int | None = 150) -> list[str]:
    """Return a list of inconsistencies; an empty list means the rows pass.

    Checks against the evaluator's own BFS (never shown to the policy) and the
    rule that failed rollouts carry no path efficiency.
    """
    problems = []
    bfs_cache: dict[tuple, int | None] = {}
    for index, row in enumerate(rows):
        where = f"row {index} ({row['agent']} seed {row['seed']} {row['phase']}@{row['phase_step']})"
        key = (row["map_id"], row["phase"], row["start"])
        if key not in bfs_cache:
            blocked = row["phase"] == "blocked"
            maze = GridMaze(row["map_id"], seed=0, blocked=blocked)
            bfs_cache[key] = shortest_distance(maze, _parse_start(row["start"]))
        if row["shortest_steps"] != bfs_cache[key]:
            problems.append(f"{where}: shortest_steps {row['shortest_steps']} != BFS {bfs_cache[key]}")
        if row["success"] not in (0, 1):
            problems.append(f"{where}: success must be 0 or 1")
        if row["success"]:
            if row["path_efficiency"] is None:
                problems.append(f"{where}: success without path efficiency")
            elif not math.isclose(row["path_efficiency"], row["shortest_steps"] / row["rollout_steps"]):
                problems.append(f"{where}: efficiency != shortest / steps")
            if row["rollout_steps"] < row["shortest_steps"]:
                problems.append(f"{where}: successful path shorter than BFS")
        else:
            if row["path_efficiency"] is not None:
                problems.append(f"{where}: failed rollout has a path efficiency")
            if max_steps is not None and row["rollout_steps"] != max_steps:
                problems.append(f"{where}: failure ended before the {max_steps}-action limit")
    return problems


@dataclass(frozen=True)
class Checkpoint:
    """One frozen evaluation of one seed: all starts pooled."""
    map_id: str
    agent: str
    seed: int
    phase: str
    phase_step: int
    environment_step: int
    rollouts: int
    successes: int
    mean_rollout_steps: float
    mean_successful_efficiency: float | None
    collisions: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.rollouts


def score_checkpoints(rows: Iterable[dict]) -> list[Checkpoint]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["map_id"], row["agent"], row["seed"], row["phase"], row["phase_step"]].append(row)
    checkpoints = []
    for (map_id, agent, seed, phase, step), members in groups.items():
        efficiencies = [r["path_efficiency"] for r in members if r["success"]]
        checkpoints.append(Checkpoint(
            map_id, agent, seed, phase, step, members[0]["environment_step"],
            rollouts=len(members), successes=sum(r["success"] for r in members),
            mean_rollout_steps=statistics.mean(r["rollout_steps"] for r in members),
            mean_successful_efficiency=statistics.mean(efficiencies) if efficiencies else None,
            collisions=sum(r["collisions"] for r in members),
        ))
    phase_order = {"original": 0, "blocked": 1}
    checkpoints.sort(key=lambda c: (c.agent, c.seed, phase_order.get(c.phase, 9), c.phase, c.phase_step))
    return checkpoints


def recovery_step(checkpoints: Sequence[Checkpoint], *, window: int = 3) -> int | None:
    """First blocked checkpoint that opens ``window`` consecutive all-start successes.

    ``None`` means the run did not meet the criterion within budget (censored).
    """
    blocked = sorted((c for c in checkpoints if c.phase == "blocked"), key=lambda c: c.phase_step)
    for i in range(len(blocked) - window + 1):
        if all(c.successes == c.rollouts for c in blocked[i:i + window]):
            return blocked[i].phase_step
    return None


def seed_results(checkpoints: Sequence[Checkpoint], *, window: int = 3) -> list[dict]:
    by_run: dict[tuple, list[Checkpoint]] = defaultdict(list)
    for c in checkpoints:
        by_run[c.agent, c.seed].append(c)
    results = []
    for (agent, seed), runs in sorted(by_run.items()):
        def endpoint(phase, last):
            members = sorted((c for c in runs if c.phase == phase), key=lambda c: c.phase_step)
            return (members[-1] if last else members[0]) if members else None
        record = {"agent": agent, "seed": seed}
        for label, phase, last in (
            ("original_final", "original", True), ("blocked_initial", "blocked", False),
            ("blocked_final", "blocked", True),
        ):
            point = endpoint(phase, last)
            record[f"{label}_successes"] = point.successes if point else None
            record[f"{label}_rollouts"] = point.rollouts if point else None
            record[f"{label}_success"] = point.success_rate if point else None
        final = endpoint("blocked", True)
        record["blocked_final_successful_efficiency"] = final.mean_successful_efficiency if final else None
        record["recovery_step"] = recovery_step(runs, window=window) if final else None
        results.append(record)
    return results


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    reference: bool = False  # True: no learning, so recovery is not applicable
    dashes: tuple = ()        # distinct patterns keep coincident curves visible
    marker: str = "o"


@dataclass(frozen=True)
class RunKind:
    """How to label and score one rollout table.

    The agent ID ``random`` means different things in the two kinds: a
    non-learning random-action reference in baseline runs, and the random
    transition replay condition in replay runs. Labels, colours and the
    recovery rule therefore come from the run kind, never from the ID alone.
    """
    name: str
    column: str
    series: dict[str, Series]

    def check(self, agents: Iterable[str]) -> None:
        unknown = sorted(set(agents) - set(self.series))
        if unknown:
            raise ValueError(f"{self.name} run cannot contain agent(s) {unknown}; "
                             f"expected a subset of {list(self.series)}")

    def label(self, agent: str) -> str:
        return self.series[agent].label

    def order(self, agents: Iterable[str]) -> list[str]:
        present = set(agents)
        return [a for a in self.series if a in present]


BASELINE = RunKind("baseline", "Agent", {
    "q_learning": Series("Q-learning", "#2a78d6"),
    "dqn": Series("DQN", "#eb6834"),
    "random": Series("Random action", "#1baf7a", reference=True),
})
REPLAY = RunKind("replay", "Condition", {
    "rest": Series("Silent rest", "#2a78d6", marker="o"),
    "random": Series("Random replay", "#eb6834", dashes=(6, 2), marker="s"),
    "ordered": Series("Ordered replay", "#1baf7a", dashes=(2, 2), marker="^"),
    "adaptive": Series("Adaptive replay", "#eda100", dashes=(6, 2, 1, 2), marker="D"),
})
RUN_KINDS = {kind.name: kind for kind in (BASELINE, REPLAY)}


def detect_run_kind(csv_path: Path, agents: Iterable[str], kind: str | None = None) -> RunKind:
    """Decide whether a rollout table holds baseline or replay runs.

    Order of evidence: an explicit ``kind``; the ``config.json`` the runner wrote
    beside the CSV (``agents`` for experiment.py, ``conditions`` for
    replay_runner.py); then agent IDs that occur in only one kind. ``random``
    occurs in both, so a table whose only evidence is ``random`` is rejected.
    """
    agents = set(agents)
    if kind is not None:
        resolved = RUN_KINDS[kind]
    else:
        resolved = None
        sidecar = Path(csv_path).with_name("config.json")
        if sidecar.exists():
            config = json.loads(sidecar.read_text(encoding="utf-8"))
            if "conditions" in config and "agents" not in config:
                resolved = REPLAY
            elif "agents" in config and "conditions" not in config:
                resolved = BASELINE
        if resolved is None:
            votes = {k.name for k in RUN_KINDS.values()
                     for a in agents if a in k.series and sum(a in o.series for o in RUN_KINDS.values()) == 1}
            if len(votes) != 1:
                raise ValueError(
                    f"cannot tell whether {csv_path} holds baseline or replay runs (agents {sorted(agents)}); "
                    "pass --kind baseline or --kind replay")
            resolved = RUN_KINDS[votes.pop()]
    resolved.check(agents)
    return resolved


def navigation_table(seeds: Sequence[dict], *, kind: RunKind) -> list[dict]:
    """Agent-level table. Success pools rollouts across seeds (failures included).

    Recovery is N/A only for series the run kind marks as a reference.
    """
    kind.check(s["agent"] for s in seeds)
    table = []
    for agent in kind.order(s["agent"] for s in seeds):
        runs = [s for s in seeds if s["agent"] == agent]
        row = {"run_kind": kind.name, "agent": agent, "label": kind.label(agent), "seeds": len(runs)}
        for label in ("original_final", "blocked_initial", "blocked_final"):
            successes = [s[f"{label}_successes"] for s in runs if s[f"{label}_successes"] is not None]
            rollouts = [s[f"{label}_rollouts"] for s in runs if s[f"{label}_rollouts"] is not None]
            row[f"{label}_success"] = sum(successes) / sum(rollouts) if rollouts else None
            row[f"{label}_rollouts"] = sum(rollouts)
        if kind.series[agent].reference:
            row["recovered_seeds"] = None
            row["median_recovery_step"] = None
        else:
            recovered = [s["recovery_step"] for s in runs if s["recovery_step"] is not None]
            row["recovered_seeds"] = f"{len(recovered)}/{len(runs)}"
            row["median_recovery_step"] = statistics.median(recovered) if recovered else None
        table.append(row)
    return table


def learning_curves(checkpoints: Sequence[Checkpoint]) -> list[dict]:
    """Mean and SD of per-seed success at each (agent, phase, phase_step)."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for c in checkpoints:
        groups[c.agent, c.phase, c.phase_step].append(c.success_rate)
    return [
        {"agent": a, "phase": p, "phase_step": s, "seeds": len(v), "mean_success": statistics.mean(v),
         "sd_success": statistics.stdev(v) if len(v) > 1 else 0.0}
        for (a, p, s), v in sorted(groups.items())
    ]


# =========================================================================== cues and recall scoring


def cue_channels(observation: int, local: Sequence[int] = range(CHANNELS_PER_OBSERVATION)) -> list[int]:
    """Channels for one position under the shared codebook ``4 * observation + local``."""
    if any(not 0 <= ch < CHANNELS_PER_OBSERVATION for ch in local):
        raise ValueError("local channels must be in 0-3")
    return [CHANNELS_PER_OBSERVATION * observation + ch for ch in local]


def channel_observation(channel: int) -> int:
    return channel // CHANNELS_PER_OBSERVATION


@dataclass(frozen=True)
class CueNoise:
    """``missing``: fraction of true cue channels dropped. ``spurious``: channels added
    from positions outside the cue."""
    missing: float = 0.0
    spurious: int = 0

    def __post_init__(self):
        if not 0.0 <= self.missing <= 1.0 or self.spurious < 0:
            raise ValueError("missing must be in [0, 1] and spurious >= 0")

    @property
    def label(self) -> str:
        return f"missing={self.missing:g},spurious={self.spurious}"


def corrupt_cue(channels: Sequence[int], noise: CueNoise, rng: Random, num_observations: int) -> list[int]:
    """Drop and add channels. Only the corrupted list goes to the probe."""
    channels = sorted(set(channels))
    keep = len(channels) - round(noise.missing * len(channels))
    kept = sorted(rng.sample(channels, keep))
    cued = {channel_observation(ch) for ch in channels}
    pool = [ch for ch in range(num_observations * CHANNELS_PER_OBSERVATION) if channel_observation(ch) not in cued]
    if noise.spurious > len(pool):
        raise ValueError("more spurious channels requested than available")
    return sorted(kept + rng.sample(pool, noise.spurious))


@dataclass(frozen=True)
class RecallTrial:
    """Everything the probe is allowed to see."""
    cue_channels: tuple[int, ...]
    cue_ms: int = 25
    recall_bins: int = 3
    bin_ms: int = 25
    recurrence: bool = True


@dataclass(frozen=True)
class RecallScore:
    decoded: tuple[int | None, ...]
    scored_against: tuple[int, ...]
    valid_continuations: int
    correct: int
    missed: int
    wrong: int

    @property
    def bins(self) -> int:
        return self.correct + self.missed + self.wrong

    @property
    def accuracy(self) -> float:
        return self.correct / self.bins

    @property
    def exact(self) -> bool:
        return self.correct == self.bins


def score_recall(decoded: Sequence[int | None], valid: Sequence[Sequence[int]]) -> RecallScore:
    """Score decoded bins against the best-matching valid continuation.

    Absent bins (``None``) are misses. Ties keep the first listed continuation.
    """
    if not valid:
        raise ValueError("at least one valid continuation is required")
    decoded = tuple(decoded)
    best = None
    for expected in valid:
        if len(expected) != len(decoded):
            raise ValueError("decoded and expected lengths differ")
        correct = sum(d == e for d, e in zip(decoded, expected))
        if best is None or correct > best[0]:
            best = (correct, tuple(expected))
    correct, expected = best
    missed = sum(d is None for d in decoded)
    return RecallScore(decoded, expected, len(valid), correct, missed, len(decoded) - correct - missed)


def present_cue(probe: RecallProbe, trial: RecallTrial, *, num_observations: int) -> tuple[int | None, ...]:
    """Call the probe with only the cue and timing; validate its output shape."""
    decoded = tuple(probe.recall(
        list(trial.cue_channels), cue_ms=trial.cue_ms, recall_bins=trial.recall_bins,
        bin_ms=trial.bin_ms, recurrence=trial.recurrence,
    ))
    if len(decoded) != trial.recall_bins:
        raise ValueError(f"probe returned {len(decoded)} bins, expected {trial.recall_bins}")
    for value in decoded:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                  or not 0 <= value < num_observations):
            raise ValueError(f"decoded bin {value!r} is not a location ID or None")
    return decoded


def run_trial(probe: RecallProbe, trial: RecallTrial, valid: Sequence[Sequence[int]], *,
              num_observations: int) -> RecallScore:
    return score_recall(present_cue(probe, trial, num_observations=num_observations), valid)


def score_fixture_recall(fixture: dict | None = None) -> list[dict]:
    """Score the committed ScriptedRecall cases. Expected: 3/3, 0/3, 2/3."""
    fixture = fixture or json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    case = fixture["recall_case"]
    num_observations = fixture["observation_count"]
    expected = [case["expected_locations"]]
    rows = []
    for variant, outputs in case["scripted_outputs"].items():
        trial = RecallTrial(tuple(case["cue_channels"]), case["cue_ms"], case["recall_bins"],
                            case["bin_ms"], recurrence=variant != "recurrent_off")
        probe = ScriptedRecall(outputs)
        score = run_trial(probe, trial, expected, num_observations=num_observations)
        rows.append({
            "source": "synthetic: ScriptedRecall fixture", "case": case["id"], "variant": variant,
            "cue_channels": " ".join(map(str, trial.cue_channels)), "recurrence": trial.recurrence,
            "decoded": " ".join("-" if d is None else str(d) for d in score.decoded),
            "correct": score.correct, "missed": score.missed, "wrong": score.wrong, "bins": score.bins,
            "accuracy": score.accuracy, "exact": score.exact,
        })
    return rows


# =========================================================================== routes for memory assays


@dataclass(frozen=True)
class Route:
    """A learned route: a self-avoiding sequence of 4-neighbour cells."""
    route_id: str
    cells: tuple[Cell, ...]
    width: int = 7

    def __post_init__(self):
        for a, b in zip(self.cells, self.cells[1:]):
            if (b[0] - a[0], b[1] - a[1]) not in MOVES:
                raise ValueError(f"{self.route_id}: {a} -> {b} is not a single move")
        if len(set(self.cells)) != len(self.cells):
            raise ValueError(f"{self.route_id}: route revisits a cell")

    @property
    def observations(self) -> tuple[int, ...]:
        return tuple(r * self.width + c for r, c in self.cells)

    @property
    def actions(self) -> tuple[int, ...]:
        return tuple(MOVES.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(self.cells, self.cells[1:]))

    def transitions(self, *, goal_reward: float = 1.0, step_reward: float = -0.005) -> list[Transition]:
        """Route as maze.Transition rows for models that learn through replay()."""
        obs, last = self.observations, len(self.actions) - 1
        return [
            Transition(obs[i], a, goal_reward if i == last else step_reward, obs[i + 1], i == last, False)
            for i, a in enumerate(self.actions)
        ]


def arena_cells(width: int = 7, height: int = 7) -> frozenset[Cell]:
    """Open grid with the same 49-ID codebook as the 7 x 7 maze (assay-only geometry)."""
    return frozenset((r, c) for r in range(height) for c in range(width))


def _neighbours(cell: Cell, cells: frozenset[Cell]) -> list[Cell]:
    return [(cell[0] + dr, cell[1] + dc) for dr, dc in MOVES if (cell[0] + dr, cell[1] + dc) in cells]


def _walk(length: int, free: set[Cell], cells: frozenset[Cell], rng: Random,
          starts: Sequence[Cell] | None = None, budget: int = 2000) -> list[Cell] | None:
    """Randomised depth-first self-avoiding walk of ``length`` cells inside ``free``."""
    if length == 0:
        return []
    candidates = [c for c in (starts if starts is not None else sorted(free)) if c in free]
    rng.shuffle(candidates)
    calls = 0

    def extend(path):
        nonlocal calls
        calls += 1
        if len(path) == length:
            return path
        if calls > budget:
            return None
        options = [n for n in _neighbours(path[-1], cells) if n in free and n not in path]
        rng.shuffle(options)
        for n in options:
            found = extend(path + [n])
            if found:
                return found
        return None

    for start in candidates[:20]:
        found = extend([start])
        if found:
            return found
    return None


def sample_disjoint_routes(count: int, length: int, seed: int, *, cells: frozenset[Cell] | None = None,
                           width: int = 7, prefix: str = "route", attempts: int = 200) -> list[Route]:
    """``count`` routes that share no cells (route similarity fixed at zero)."""
    cells = cells or arena_cells(width)
    rng = Random(seed)
    for _ in range(attempts):
        free, routes = set(cells), []
        for i in range(count):
            path = _walk(length, free, cells, rng)
            if path is None:
                break
            free -= set(path)
            routes.append(Route(f"{prefix}-{i}", tuple(path), width))
        if len(routes) == count:
            return routes
    raise RuntimeError(f"could not place {count} disjoint routes of {length} cells")


def sample_overlapping_routes(count: int, length: int, shared: int, seed: int, *,
                              cells: frozenset[Cell] | None = None, width: int = 7,
                              attempts: int = 500) -> list[Route]:
    """Routes with unique heads, a common middle segment of ``shared`` cells, and unique tails.

    The cue (route start) is unique, so each cue has one valid continuation, but
    a memoryless chain cannot tell which tail follows the shared segment. A cell
    has four neighbours, so at most three routes can enter and leave one segment,
    and two when the segment is a single crossing cell.
    """
    if shared == 0:
        return sample_disjoint_routes(count, length, seed, cells=cells, width=width, prefix="sim")
    if count > 3 or (shared == 1 and count > 2):
        raise ValueError("a shared segment admits at most three routes (two when it is one cell)")
    head = (length - shared + 1) // 2
    tail = length - shared - head
    if head < 1:
        raise ValueError("each route needs at least one unique head cell for the cue")
    cells = cells or arena_cells(width)
    rng = Random(seed)
    for _ in range(attempts):
        free = set(cells)
        segment = _walk(shared, free, cells, rng)
        if segment is None:
            continue
        free -= set(segment)
        heads, tails = [], []
        for _ in range(count):
            h = _walk(head, free, cells, rng, starts=_neighbours(segment[0], cells))
            if h is None:
                break
            free -= set(h)
            heads.append(list(reversed(h)))
        if len(heads) < count:
            continue
        for _ in range(count):
            t = _walk(tail, free, cells, rng, starts=_neighbours(segment[-1], cells)) if tail else []
            if t is None:
                break
            free -= set(t)
            tails.append(t)
        if len(tails) == count and (tail or count == 1):
            return [Route(f"sim-{i}", tuple(h + segment + t), width) for i, (h, t) in enumerate(zip(heads, tails))]
    raise RuntimeError(f"could not place {count} routes sharing {shared} cells")


def overlap_measures(routes: Sequence[Route]) -> dict:
    """Mean pairwise fraction of shared cells and of shared directed transitions."""
    pairs = [(a, b) for i, a in enumerate(routes) for b in routes[i + 1:]]
    if not pairs:
        return {"shared_cell_fraction": 0.0, "shared_transition_fraction": 0.0}
    cell_f, trans_f = [], []
    for a, b in pairs:
        cell_f.append(len(set(a.cells) & set(b.cells)) / len(a.cells))
        ta, tb = set(zip(a.cells, a.cells[1:])), set(zip(b.cells, b.cells[1:]))
        trans_f.append(len(ta & tb) / max(1, len(ta)))
    return {"shared_cell_fraction": statistics.mean(cell_f), "shared_transition_fraction": statistics.mean(trans_f)}


def valid_continuations(routes: Sequence[Route], route: Route, cue_positions: int, recall_bins: int) -> list[tuple[int, ...]]:
    """Continuations of every learned route that begins with the same cued positions."""
    cue = route.observations[:cue_positions]
    found = []
    for other in routes:
        obs = other.observations
        if obs[:cue_positions] == cue and len(obs) >= cue_positions + recall_bins:
            continuation = obs[cue_positions:cue_positions + recall_bins]
            if continuation not in found:
                found.append(continuation)
    if not found:
        raise ValueError(f"{route.route_id} is too short for {recall_bins} recall bins")
    return found


# =========================================================================== models under test


class MemoryModel(RecallProbe, Protocol):
    """What the assays need: learn routes (weights may change), then frozen recall."""

    def learn(self, routes: Sequence[Route], *, exposure_per_route: int) -> None: ...


class ReplayAgentMemory:
    """Adapter for a ReplayAgent that also implements recall(): routes are learned
    through guided replay of their transitions, ``exposure_per_route`` times each."""

    def __init__(self, agent, *, step_ms: int = 25):
        self.agent, self.step_ms = agent, step_ms

    def learn(self, routes, *, exposure_per_route):
        for _ in range(exposure_per_route):
            for route in routes:
                self.agent.replay(route.transitions(), step_ms=self.step_ms)

    def recall(self, cue_channels, *, cue_ms, recall_bins, bin_ms, recurrence):
        return self.agent.recall(cue_channels, cue_ms=cue_ms, recall_bins=recall_bins,
                                 bin_ms=bin_ms, recurrence=recurrence)


class TransitionTableMemory:
    """SYNTHETIC stand-in used only to exercise the assay code.

    Decodes the cue by channel majority, then follows the most-exposed learned
    successor of each position (first-order, no capacity limit). With recurrence
    off it is silent after cue offset. Its numbers are not spiking-model results.
    """

    source = SYNTHETIC_SOURCE

    def __init__(self, num_observations: int, seed: int):
        self.num_observations = num_observations
        self.successors: dict[int, Counter] = defaultdict(Counter)

    def learn(self, routes, *, exposure_per_route):
        for route in routes:
            obs = route.observations
            for a, b in zip(obs, obs[1:]):
                self.successors[a][b] += exposure_per_route

    def recall(self, cue_channels, *, cue_ms, recall_bins, bin_ms, recurrence):
        votes = Counter(channel_observation(ch) for ch in cue_channels)
        if not votes or not recurrence:
            return [None] * recall_bins
        top = max(votes.values())
        # Most-voted position; lowest ID breaks ties.
        position = min(o for o, v in votes.items() if v == top)
        decoded = []
        for _ in range(recall_bins):
            nxt = self.successors.get(position)
            if not nxt:
                decoded.append(None)
                continue
            best = max(nxt.values())
            position = min(o for o, c in nxt.items() if c == best)
            decoded.append(position)
        return decoded


def synthetic_factory(num_observations: int, seed: int) -> TransitionTableMemory:
    return TransitionTableMemory(num_observations, seed)


# =========================================================================== assays


@dataclass
class AssayConfig:
    seeds: tuple[int, ...] = (11, 17, 23)
    num_observations: int = 49
    width: int = 7
    route_length: int = 6
    cue_positions: int = 1
    recall_bins: int = 5
    cue_ms: int = 25
    bin_ms: int = 25
    exposure_per_route: int = 20
    noise_draws: int = 5
    base_route_count: int = 4
    fixed_noise: dict = field(default_factory=lambda: {"missing": 0.25, "spurious": 0})
    missing_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    spurious_levels: tuple[int, ...] = (0, 2, 4, 6, 8)
    load_levels: tuple[int, ...] = (1, 2, 4, 6)
    similarity_count: int = 2
    shared_levels: tuple[int, ...] = (0, 1, 2, 3, 4)
    interference_exposures: tuple[int, ...] = (0, 10, 20, 40)
    interference_routes: int = 4
    degradation_tolerance: float = 0.10

    @classmethod
    def from_json(cls, path: Path) -> "AssayConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown assay settings: {sorted(unknown)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


def _score_routes(model: RecallProbe, routes: Sequence[Route], learned: Sequence[Route], cfg: AssayConfig,
                  noise: CueNoise, rng: Random, *, recurrence: bool = True, draws: int | None = None) -> list[dict]:
    """Probe every route; noise draws repeat only when the cue is actually corrupted."""
    draws = draws or (cfg.noise_draws if (noise.missing or noise.spurious) else 1)
    rows = []
    for route in routes:
        true_cue = [ch for o in route.observations[:cfg.cue_positions] for ch in cue_channels(o)]
        valid = valid_continuations(learned, route, cfg.cue_positions, cfg.recall_bins)
        for draw in range(draws):
            cue = corrupt_cue(true_cue, noise, rng, cfg.num_observations)
            trial = RecallTrial(tuple(cue), cfg.cue_ms, cfg.recall_bins, cfg.bin_ms, recurrence)
            score = run_trial(model, trial, valid, num_observations=cfg.num_observations)
            rows.append({
                "route_id": route.route_id, "draw": draw, "recurrence": recurrence,
                "cue_size": len(cue), "valid_continuations": score.valid_continuations,
                "correct": score.correct, "missed": score.missed, "wrong": score.wrong, "bins": score.bins,
            })
    return rows


def _tag(rows, **meta):
    return [{**meta, **row} for row in rows]


def run_assays(factory: Callable[[int, int], MemoryModel], cfg: AssayConfig, *, source: str) -> list[dict]:
    """Run every sweep for every seed. One factor varies per assay."""
    fixed = CueNoise(**cfg.fixed_noise)
    clean = CueNoise()
    out = []
    for seed in cfg.seeds:
        base = sample_disjoint_routes(cfg.base_route_count, cfg.route_length, seed, width=cfg.width)

        # Recurrence on/off: same trained network, clean cue.
        model = factory(cfg.num_observations, seed)
        model.learn(base, exposure_per_route=cfg.exposure_per_route)
        for recurrence in (True, False):
            out += _tag(_score_routes(model, base, base, cfg, clean, Random(seed), recurrence=recurrence),
                        assay="recurrence", level=f"recurrence={'on' if recurrence else 'off'}",
                        level_value=int(recurrence), seed=seed, routes=len(base))

        # Cue noise: fixed routes and network, sweep missing, then spurious.
        for missing in cfg.missing_levels:
            noise = CueNoise(missing=missing)
            out += _tag(_score_routes(model, base, base, cfg, noise, Random(seed * 1000 + 1)),
                        assay="cue_missing", level=noise.label, level_value=missing, seed=seed, routes=len(base))
        for spurious in cfg.spurious_levels:
            noise = CueNoise(spurious=spurious)
            out += _tag(_score_routes(model, base, base, cfg, noise, Random(seed * 1000 + 2)),
                        assay="cue_spurious", level=noise.label, level_value=spurious, seed=seed, routes=len(base))

        # Memory load: fresh fixed-size network per load, fixed exposure per route and fixed cue noise.
        for load in cfg.load_levels:
            routes = sample_disjoint_routes(load, cfg.route_length, seed + 7919 * load, width=cfg.width)
            m = factory(cfg.num_observations, seed)
            m.learn(routes, exposure_per_route=cfg.exposure_per_route)
            out += _tag(_score_routes(m, routes, routes, cfg, fixed, Random(seed * 1000 + 3)),
                        assay="memory_load", level=f"routes={load}", level_value=load, seed=seed, routes=load,
                        total_exposure=load * cfg.exposure_per_route, cue_noise=fixed.label)

        # Route similarity: fixed count, vary shared middle segment.
        for shared in cfg.shared_levels:
            routes = sample_overlapping_routes(cfg.similarity_count, cfg.route_length, shared,
                                               seed + 104729 * (shared + 1), width=cfg.width)
            m = factory(cfg.num_observations, seed)
            m.learn(routes, exposure_per_route=cfg.exposure_per_route)
            out += _tag(_score_routes(m, routes, routes, cfg, fixed, Random(seed * 1000 + 4)),
                        assay="route_similarity", level=f"shared_cells={shared}", level_value=shared,
                        seed=seed, routes=len(routes), cue_noise=fixed.label, **overlap_measures(routes))

        # Interference: learn targets, score, expose to other routes, score again.
        interferers = sample_overlapping_interferers(base, cfg, seed)
        overlap = overlap_measures(list(base) + list(interferers))
        for exposure in cfg.interference_exposures:
            m = factory(cfg.num_observations, seed)
            m.learn(base, exposure_per_route=cfg.exposure_per_route)
            before = _score_routes(m, base, base, cfg, clean, Random(seed))
            if exposure:
                m.learn(interferers, exposure_per_route=exposure)
            after = _score_routes(m, base, base, cfg, clean, Random(seed))
            for when, rows in (("before", before), ("after", after)):
                out += _tag(rows, assay="interference", level=f"exposure={exposure}", level_value=exposure,
                            seed=seed, routes=len(base), when=when, interfering_routes=len(interferers),
                            **{f"target_interferer_{k}": v for k, v in overlap.items()})
    return [{"source": source, **row} for row in out]


def sample_overlapping_interferers(targets: Sequence[Route], cfg: AssayConfig, seed: int) -> list[Route]:
    """Routes that start on a target's second cell and leave it by a different successor,
    so each interferer competes with one target transition."""
    cells = arena_cells(cfg.width)
    rng = Random(seed + 31337)
    routes = []
    for i, target in enumerate(targets[:cfg.interference_routes]):
        pivot, target_next = target.cells[1], target.cells[2]
        free = set(cells) - {target_next, target.cells[0]}
        for _ in range(200):
            path = _walk(cfg.route_length, free, cells, rng, starts=[pivot])
            if path and path[1] != target_next:
                routes.append(Route(f"interferer-{i}", tuple(path), cfg.width))
                break
    return routes


def summarize_assays(rows: Sequence[dict], *, tolerance: float = 0.10) -> list[dict]:
    """Seed-level accuracy (bins pooled over routes and draws), then mean and SD across seeds."""
    seed_bins: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    meta = {}
    for r in rows:
        key = (r["source"], r["assay"], r.get("when", ""), r["level_value"], r["level"], r["seed"])
        tally = seed_bins[key]
        tally[0] += r["correct"]; tally[1] += r["missed"]; tally[2] += r["wrong"]; tally[3] += r["bins"]
        meta.setdefault(key[:5], {k: r[k] for k in ("shared_cell_fraction", "shared_transition_fraction",
                                                   "total_exposure") if k in r})
    by_level: dict[tuple, list] = defaultdict(list)
    for key, (c, m, w, b) in seed_bins.items():
        by_level[key[:5]].append((key[5], c / b, m / b, w / b))
    summary = []
    when_order = {"": 0, "before": 1, "after": 2}
    for key in sorted(by_level, key=lambda k: (k[0], k[1], when_order.get(k[2], 3), k[3])):
        source, assay, when, value, level = key
        seeds = sorted(by_level[key])
        acc = [s[1] for s in seeds]
        summary.append({
            "source": source, "assay": assay, "when": when, "level": level, "level_value": value,
            "seeds": len(seeds), "mean_accuracy": statistics.mean(acc),
            "sd_accuracy": statistics.stdev(acc) if len(acc) > 1 else 0.0,
            "mean_miss_rate": statistics.mean(s[2] for s in seeds),
            "mean_error_rate": statistics.mean(s[3] for s in seeds),
            "seed_accuracies": " ".join(f"{s[0]}:{s[1]:.3f}" for s in seeds),
            **meta[key],
        })
    return summary


def degradation_point(summary: Sequence[dict], assay: str, *, tolerance: float = 0.10, when: str = "") -> dict:
    """First level whose mean accuracy falls more than ``tolerance`` below the first level."""
    levels = [s for s in summary if s["assay"] == assay and s["when"] == when]
    levels.sort(key=lambda s: s["level_value"])
    if not levels:
        return {"assay": assay, "reference": None, "first_drop": None, "note": "not run"}
    reference = levels[0]["mean_accuracy"]
    for s in levels[1:]:
        if s["mean_accuracy"] < reference - tolerance:
            return {"assay": assay, "reference": levels[0]["level"], "first_drop": s["level"],
                    "note": f"accuracy {reference:.2f} -> {s['mean_accuracy']:.2f}"}
    return {"assay": assay, "reference": levels[0]["level"], "first_drop": None,
            "note": f"no decline beyond {tolerance:.2f} over {levels[0]['level']} .. {levels[-1]['level']}"}


def interference_scores(summary: Sequence[dict]) -> list[dict]:
    before = {s["level_value"]: s for s in summary if s["assay"] == "interference" and s["when"] == "before"}
    after = {s["level_value"]: s for s in summary if s["assay"] == "interference" and s["when"] == "after"}
    return [
        {"exposure": e, "before": before[e]["mean_accuracy"], "after": after[e]["mean_accuracy"],
         "interference_score": after[e]["mean_accuracy"] - before[e]["mean_accuracy"]}
        for e in sorted(before) if e in after
    ]


# =========================================================================== output


def _fmt(value, pct=False):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{100 * value:.1f}%" if pct else f"{value:g}"
    return str(value)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n", restval="")
        writer.writeheader()
        writer.writerows(rows)


def navigation_markdown(table: Sequence[dict], source: str, kind: RunKind) -> str:
    lines = [
        f"Source: {source} ({kind.name} runs). Success pools every rollout across seeds; "
        "failures stay in the denominator.", "",
        f"| {kind.column} | ID | Seeds | Original final | Blocked initial | Blocked final "
        "| Recovered seeds | Median recovery step |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in table:
        lines.append(
            f"| {r['label']} | `{r['agent']}` | {r['seeds']} | {_fmt(r['original_final_success'], True)} "
            f"| {_fmt(r['blocked_initial_success'], True)} | {_fmt(r['blocked_final_success'], True)} "
            f"| {_fmt(r['recovered_seeds'])} | {_fmt(r['median_recovery_step'])} |")
    return "\n".join(lines) + "\n"


def memory_markdown(summary: Sequence[dict], cfg: AssayConfig) -> str:
    source = summary[0]["source"] if summary else "none"
    lines = [
        "# Combined memory table", "",
        f"**Source: {source}.**",
        "Accuracy = correct bins / all bins after cue offset; absent bins are misses. "
        "Seed accuracy pools routes and noise draws; the table shows mean ± SD across seeds.", "",
        f"Fixed settings: {cfg.route_length}-cell routes, cue = first {cfg.cue_positions} position(s) "
        f"(4 channels each), {cfg.recall_bins} recall bins of {cfg.bin_ms} ms, {cfg.exposure_per_route} "
        f"exposures per route, network size {cfg.num_observations} positions, seeds {list(cfg.seeds)}.", "",
        "| Assay | Phase | Level | Mean accuracy | SD | Miss rate | Error rate | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in summary:
        notes = []
        if "total_exposure" in s:
            notes.append(f"total exposure {s['total_exposure']}")
        if "shared_transition_fraction" in s:
            notes.append(f"shared cells {s['shared_cell_fraction']:.2f}, transitions {s['shared_transition_fraction']:.2f}")
        lines.append(f"| {s['assay']} | {s['when'] or '-'} | {s['level']} | {s['mean_accuracy']:.3f} "
                     f"| {s['sd_accuracy']:.3f} | {s['mean_miss_rate']:.3f} | {s['mean_error_rate']:.3f} "
                     f"| {'; '.join(notes)} |")
    lines += ["", "## Where accuracy starts to fall", "",
              f"First level more than {cfg.degradation_tolerance:.2f} below the sweep's first level.", ""]
    for assay in ("cue_missing", "cue_spurious", "memory_load", "route_similarity"):
        d = degradation_point(summary, assay, tolerance=cfg.degradation_tolerance)
        lines.append(f"- **{assay}**: {d['first_drop'] or 'no decline observed'} ({d['note']})")
    lines += ["", "## Interference (after − before)", ""]
    for r in interference_scores(summary):
        lines.append(f"- exposure {r['exposure']}: {r['before']:.3f} → {r['after']:.3f} "
                     f"(score {r['interference_score']:+.3f})")
    return "\n".join(lines) + "\n"


INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _plot_navigation_axes(axes, curves: Sequence[dict], kind: RunKind) -> dict[str, list[str]]:
    """Draw original and blocked learning curves; return the labels drawn per phase."""
    drawn: dict[str, list[str]] = {}
    for ax, phase, title in ((axes[0], "original", "Original maze"), (axes[1], "blocked", "Preferred route blocked")):
        ends = []
        drawn[phase] = []
        for agent in kind.order(c["agent"] for c in curves):
            spec = kind.series[agent]
            label, color = spec.label, spec.color
            pts = [c for c in curves if c["agent"] == agent and c["phase"] == phase]
            if not pts:
                continue
            x = [p["phase_step"] for p in pts]
            y = [p["mean_success"] for p in pts]
            sd = [p["sd_success"] for p in pts]
            ax.fill_between(x, [max(0, a - b) for a, b in zip(y, sd)], [min(1, a + b) for a, b in zip(y, sd)],
                            color=color, alpha=0.12, linewidth=0)
            line, = ax.plot(x, y, color=color, linewidth=2, label=label,
                            marker=spec.marker if len(x) < 8 else None, markersize=5, markerfacecolor="none")
            if spec.dashes:
                line.set_dashes(spec.dashes)
            ends.append([y[-1], f"{label} {100 * y[-1]:.0f}%", x[-1]])
            drawn[phase].append(label)
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):  # keep direct labels at least 0.08 apart
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.08)
        for y_label, text, x_end in ends:
            ax.annotate(text, (x_end, y_label), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Real maze actions in this phase", fontsize=8, color=MUTED)
        ax.set_ylim(-0.02, 1.12)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.3)
        _style(ax)
    axes[0].set_ylabel("Frozen goal success\n(mean ± SD across seeds)", fontsize=8, color=MUTED)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=8, frameon=False,
                   handlelength=3)
    return drawn


def plot_navigation(curves: Sequence[dict], kind: RunKind, path: Path, *, source: str) -> dict[str, list[str]]:
    """Two-panel learning-curve figure for any rollout table; returns the labels drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    drawn = _plot_navigation_axes(axes, curves, kind)
    fig.suptitle(f"Frozen navigation · {kind.name} runs · {source}", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return drawn


def plot_evidence(curves: Sequence[dict], recall_rows: Sequence[dict], path: Path, *, nav_source: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    _plot_navigation_axes(axes[:2], curves, BASELINE)

    ax = axes[2]
    names = [r["variant"] for r in recall_rows]
    left = [0] * len(recall_rows)
    for key, color, hatch, label in (("correct", "#2a78d6", None, "Correct"), ("wrong", "#eb6834", None, "Wrong"),
                                     ("missed", "#fcfcfb", "///", "Missed (None)")):
        vals = [r[key] for r in recall_rows]
        ax.barh(names, vals, left=left, color=color, hatch=hatch, edgecolor="#8a8984" if hatch else "#fcfcfb",
                linewidth=1.5, height=0.55, label=label)
        left = [a + b for a, b in zip(left, vals)]
    for i, r in enumerate(recall_rows):
        ax.text(r["bins"] + 0.08, i, f"{r['correct']}/{r['bins']}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, recall_rows[0]["bins"] + 0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Recall bins after cue offset", fontsize=8, color=MUTED)
    ax.set_title("SYNTHETIC ScriptedRecall fixture", fontsize=10, color=INK, loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=7, frameon=False)
    _style(ax)
    ax.grid(False)
    fig.suptitle(f"Scoring evidence · navigation: {nav_source} · recall: scripted test double, not model output",
                 fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_assays(summary: Sequence[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (("cue_missing", "Missing cue fraction"), ("cue_spurious", "Spurious cue channels"),
              ("memory_load", "Learned routes"), ("route_similarity", "Shared cells between routes"))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for ax, (assay, xlabel) in zip(axes, panels):
        pts = sorted((s for s in summary if s["assay"] == assay), key=lambda s: s["level_value"])
        x = [s["level_value"] for s in pts]
        y = [s["mean_accuracy"] for s in pts]
        sd = [s["sd_accuracy"] for s in pts]
        ax.errorbar(x, y, yerr=sd, color="#2a78d6", linewidth=2, marker="o", markersize=5, capsize=3)
        ax.set_xticks(x)
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
        ax.set_ylim(-0.02, 1.05)
        _style(ax)
    axes[0].set_ylabel("Recall accuracy after cue offset", fontsize=8, color=MUTED)
    source = summary[0]["source"] if summary else ""
    fig.suptitle(f"Memory assays · source: {source}", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =========================================================================== CLI


def score_navigation(csv_path: Path, output: Path, *, max_steps: int | None, source: str,
                     kind: str | None = None, plot: bool = True) -> tuple[list[dict], RunKind]:
    """Audit, score and write tables (and the learning-curve figure) for one rollout CSV."""
    rows = load_rollouts(csv_path)
    run_kind = detect_run_kind(csv_path, {r["agent"] for r in rows}, kind)
    problems = audit_rollouts(rows, max_steps=max_steps)
    if problems:
        raise ValueError("rollout audit failed:\n" + "\n".join(problems[:20]))
    checkpoints = score_checkpoints(rows)
    seeds = seed_results(checkpoints)
    table = navigation_table(seeds, kind=run_kind)
    curves = learning_curves(checkpoints)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "navigation_checkpoints.csv",
              [{**asdict(c), "success_rate": c.success_rate} for c in checkpoints])
    write_csv(output / "navigation_seeds.csv",
              [{"run_kind": run_kind.name, "label": run_kind.label(s["agent"]), **s} for s in seeds])
    write_csv(output / "navigation_curves.csv", curves)
    (output / "navigation_table.md").write_text(
        f"Rollouts: {len(rows)}; checkpoints: {len(checkpoints)}; audit: passed.\n\n"
        + navigation_markdown(table, source, run_kind), encoding="utf-8")
    if plot:
        plot_navigation(curves, run_kind, output / "navigation_curves.png", source=source)
    return curves, run_kind


def _load_factory(spec: str):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name or "factory")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evidence", help="baseline navigation + fixture recall + figure")
    ev.add_argument("--csv", type=Path, default=Path("baseline_evaluation.csv"))
    ev.add_argument("--output", type=Path, required=True)
    nav = sub.add_parser("navigation", help="score a rollout CSV")
    nav.add_argument("csv", type=Path)
    nav.add_argument("--output", type=Path, required=True)
    nav.add_argument("--max-steps", type=int, default=None, help="check failures ran to this limit")
    nav.add_argument("--source", default=None, help="label for tables and figure (default: CSV name)")
    nav.add_argument("--kind", choices=sorted(RUN_KINDS),
                     help="baseline or replay; default reads config.json beside the CSV")
    asy = sub.add_parser("assays", help="memory sweeps")
    asy.add_argument("--output", type=Path, required=True)
    asy.add_argument("--model", help="module:factory returning a MemoryModel; default is the synthetic stand-in")
    asy.add_argument("--config", type=Path)
    args = parser.parse_args(argv)

    if args.command == "evidence":
        source = f"{args.csv.name} (conventional RL baselines)"
        curves, _ = score_navigation(args.csv, args.output, max_steps=150, source=source,
                                     kind="baseline", plot=False)
        recall = score_fixture_recall()
        write_csv(args.output / "recall_fixture.csv", recall)
        plot_evidence(curves, recall, args.output / "scoring_evidence.png", nav_source=args.csv.name)
        print((args.output / "navigation_table.md").read_text(encoding="utf-8"))
        for r in recall:
            print(f"[synthetic] {r['variant']}: {r['correct']}/{r['bins']} correct, {r['missed']} missed")
    elif args.command == "navigation":
        score_navigation(args.csv, args.output, max_steps=args.max_steps,
                         source=args.source or args.csv.name, kind=args.kind)
        print((args.output / "navigation_table.md").read_text(encoding="utf-8"))
    else:
        cfg = AssayConfig.from_json(args.config) if args.config else AssayConfig()
        if args.model:
            factory, source = _load_factory(args.model), f"model {args.model}"
        else:
            factory, source = synthetic_factory, SYNTHETIC_SOURCE
        rows = run_assays(factory, cfg, source=source)
        summary = summarize_assays(rows, tolerance=cfg.degradation_tolerance)
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / "memory_assays.csv", rows)
        write_csv(args.output / "memory_summary.csv", summary)
        (args.output / "memory_table.md").write_text(memory_markdown(summary, cfg), encoding="utf-8")
        plot_assays(summary, args.output / "memory_assays.png")
        print((args.output / "memory_table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
