"""Create a two-panel recovery figure and seed-level summary from rollout CSV."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

LABELS = {"random": "Random", "q_learning": "Q-learning", "dqn": "DQN"}
COLORS = {"random": "#797A85", "q_learning": "#007773", "dqn": "#A53E23"}


def load_groups(csv_path: Path):
    groups = defaultdict(list)
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            key = (row["agent"], int(row["seed"]), row["phase"], int(row["phase_step"]))
            groups[key].append(row)
    if not groups:
        raise ValueError("no rollout data to analyze")
    return groups


def success(rows):
    return statistics.mean(int(row["success"]) for row in rows)


def summarize(groups):
    summary = {}
    for name in sorted({key[0] for key in groups}):
        seeds = sorted({key[1] for key in groups if key[0] == name})
        per_seed = []
        for seed in seeds:
            original_steps = sorted(key[3] for key in groups if key[:3] == (name, seed, "original"))
            blocked_steps = sorted(key[3] for key in groups if key[:3] == (name, seed, "blocked"))
            record = {
                "seed": seed,
                "original_final_success": success(groups[name, seed, "original", original_steps[-1]]),
            }
            if blocked_steps:
                final = groups[name, seed, "blocked", blocked_steps[-1]]
                efficiencies = [float(row["path_efficiency"]) for row in final if row["path_efficiency"]]
                # Three consecutive checkpoints at all starts. Missing recovery is censored.
                recovery = next(
                    (triplet[0] for triplet in zip(blocked_steps, blocked_steps[1:], blocked_steps[2:])
                     if name != "random" and all(success(groups[name, seed, "blocked", step]) == 1 for step in triplet)),
                    None,
                )
                record.update(
                    blocked_initial_success=success(groups[name, seed, "blocked", blocked_steps[0]]),
                    blocked_final_success=success(final),
                    blocked_final_successful_path_efficiency=(
                        statistics.mean(efficiencies) if efficiencies else None
                    ),
                    recovery_step=recovery,
                )
            per_seed.append(record)
        summary[name] = per_seed
    return summary


def plot(groups, figure_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = [phase for phase in ("original", "blocked") if any(key[2] == phase for key in groups)]
    fig, axes = plt.subplots(1, len(phases), figsize=(5.7 * len(phases), 3.7), squeeze=False)
    for axis, phase in zip(axes[0], phases):
        by_agent_step = defaultdict(list)
        for (name, _, row_phase, step), records in groups.items():
            if row_phase == phase:
                by_agent_step[name, step].append(success(records))
        for name in LABELS:
            steps = sorted(step for agent, step in by_agent_step if agent == name)
            if not steps:
                continue
            samples = [by_agent_step[name, step] for step in steps]
            means = [statistics.mean(values) for values in samples]
            deviations = [statistics.stdev(values) if len(values) > 1 else 0 for values in samples]
            axis.plot(steps, means, label=LABELS[name], color=COLORS[name], linewidth=2)
            axis.fill_between(
                steps, [max(0, m - d) for m, d in zip(means, deviations)],
                [min(1, m + d) for m, d in zip(means, deviations)],
                color=COLORS[name], alpha=0.12, linewidth=0,
            )
        axis.set(
            title="Original maze" if phase == "original" else "Preferred route blocked",
            xlabel="Real maze actions in this phase",
            ylabel="Goal success (mean across seeds)",
            ylim=(-0.02, 1.02),
        )
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = load_groups(args.csv)
    (args.output / "summary.json").write_text(json.dumps(summarize(groups), indent=2) + "\n")
    plot(groups, args.output / "blocked_route_recovery.png")


if __name__ == "__main__":
    main()
