"""Reproduce Task 2 spike/plasticity evidence and a fixed-budget online pilot.

python -m snn_diagnostics --config snn_pilot_config.json --output snn_evidence
No replay scheduler or memory scorer is needed for this standalone deliverable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from maze import GridMaze, Transition
from parallel_fixtures import FIXTURE_PATH
from snn_model import RewardSTDP, SNNConfig, SpikingAgent


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plasticity_diagnostic(config: SNNConfig) -> tuple[list[dict], list[dict]]:
    fixtures = json.loads(FIXTURE_PATH.read_text())
    cases = fixtures["plasticity_cases"] + [dict(fixtures["plasticity_cases"][0], id="late-reward", pulse_ms=720)]
    rows, summary = [], []
    for case in cases:
        synapse = RewardSTDP(1, 1, [0], [0], [0.5], config, 1)
        for time in range(1, case["pulse_ms"] + 1):
            pre, post = time == case["pre_ms"], time == case["post_ms"]
            synapse.advance([pre], [post])
            reward = case["reward"] if time == case["pulse_ms"] else 0
            if reward:
                synapse.reward(reward)
            rows.append({"case": case["id"], "time_ms": time, "pre_spike": int(pre),
                         "post_spike": int(post), "reward": reward,
                         "eligibility": float(synapse.eligibility[0]), "weight": float(synapse.weights[0])})
        delta = float(synapse.weights[0] - 0.5)
        summary.append({**case, "measured_delta": delta, "measured_sign": int(np.sign(delta)),
                        "passed": int(np.sign(delta)) == case["expected_delta_sign"]})
    return rows, summary


def evaluate(agent: SpikingAgent, config: dict, seed: int, checkpoint: int) -> list[dict]:
    # The snapshot owns all evaluation voltages, traces, RNG draws, and weights.
    frozen = agent.evaluation_copy()
    maze = GridMaze(config["map_id"], seed, config["max_steps"])
    rows = []
    for rollout in range(config["evaluation_rollouts"]):
        frozen.begin_episode()
        start = maze.spec.starts[rollout % len(maze.spec.starts)]
        observation = maze.reset(start=start)
        collisions = 0
        reward_sum = 0.0
        for length in range(1, maze.max_steps + 1):
            action = frozen.act(observation, training=False)
            observation, reward, terminated, truncated, info = maze.step(action)
            collisions += int(info["collision"])
            reward_sum += reward
            if terminated or truncated:
                break
        rows.append({"seed": seed, "environment_step": checkpoint, "rollout": rollout,
                     "start": f"{start[0]}:{start[1]}", "success": int(terminated),
                     "length": length, "collisions": collisions, "return": reward_sum,
                     "terminated": int(terminated), "truncated": int(truncated)})
    return rows


def run_pilot(config: dict, model_config: SNNConfig) -> tuple[list[dict], list[dict], list[dict]]:
    evaluation, training, summaries = [], [], []
    for seed in config["seeds"]:
        maze = GridMaze(config["map_id"], seed, config["max_steps"])
        agent = SpikingAgent(maze.num_observations, seed, model_config)
        evaluation.extend(evaluate(agent, config, seed, 0))
        observation = maze.reset()
        agent.begin_episode()
        episode, length, collisions, reward_sum = 0, 0, 0, 0.0
        for step in range(1, config["training_steps"] + 1):
            action = agent.act(observation, training=True)
            next_observation, reward, terminated, truncated, info = maze.step(action)
            agent.observe(Transition(observation, action, reward, next_observation, terminated, truncated))
            observation = next_observation
            length += 1
            collisions += int(info["collision"])
            reward_sum += reward
            if terminated or truncated or step == config["training_steps"]:
                training.append({"seed": seed, "episode": episode, "environment_step": step,
                                 "success": int(terminated), "length": length, "collisions": collisions,
                                 "return": reward_sum, "terminated": int(terminated), "truncated": int(truncated),
                                 "budget_cutoff": int(not (terminated or truncated))})
                if step < config["training_steps"]:
                    agent.begin_episode()
                    observation = maze.reset()
                    episode += 1
                    length, collisions, reward_sum = 0, 0, 0.0
            if step % config["checkpoint_every"] == 0 or step == config["training_steps"]:
                rows = evaluate(agent, config, seed, step)
                evaluation.extend(rows)
                print(f"seed={seed} real_actions={step} frozen_success={sum(r['success'] for r in rows)}/{len(rows)}", flush=True)
        initial = [r for r in evaluation if r["seed"] == seed and r["environment_step"] == 0]
        final = [r for r in evaluation if r["seed"] == seed and r["environment_step"] == config["training_steps"]]
        train_rows = [r for r in training if r["seed"] == seed]
        summaries.append({"seed": seed, "training_actions": agent.online_updates,
                          "offline_ms": agent.offline_ms, "replay_updates": agent.replay_updates,
                          "training_episodes_including_cutoff": len(train_rows),
                          "training_successes": sum(r["success"] for r in train_rows),
                          "initial_successes": sum(r["success"] for r in initial),
                          "final_successes": sum(r["success"] for r in final),
                          "evaluation_rollouts_per_checkpoint": len(final),
                          "final_success_rate": sum(r["success"] for r in final) / len(final),
                          "final_mean_length_all_rollouts": float(np.mean([r["length"] for r in final]))})
    return evaluation, training, summaries


def plot_diagnostic(output: Path, spike_rows: list[dict], plasticity: list[dict], summary: list[dict], pilot: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {"E": "#2463a8", "I": "#cf5b25", "A": "#30835b"}
    offsets = {"E": 0, "I": 196, "A": 200}
    for pop in colors:
        rows = [r for r in spike_rows if r["population"] == pop]
        axes[0, 0].scatter([r["time_ms"] for r in rows], [r["neuron"] + offsets[pop] for r in rows],
                           s=6, color=colors[pop], label=pop)
    axes[0, 0].axvline(100, color="gray", linestyle=":")
    axes[0, 0].set_xlim(0, 150)
    axes[0, 0].set(title="Real LIF spikes: guided fixture then silent rest", xlabel="Simulated time (ms)", ylabel="Neuron (E / I / action)")
    axes[0, 0].legend(loc="lower right")
    for case in ("causal", "reverse", "zero-reward"):
        rows = [r for r in plasticity if r["case"] == case]
        axes[0, 1].plot([r["time_ms"] for r in rows], [r["eligibility"] for r in rows], label=case,
                        linestyle="--" if case == "zero-reward" else "-")
    axes[0, 1].axvline(120, color="gray", linestyle=":", label="reward pulse")
    axes[0, 1].set(title="Single-synapse supplied spike-pair cases", xlabel="Time (ms)", ylabel="Eligibility")
    axes[0, 1].legend()
    axes[1, 0].bar([r["id"] for r in summary], [r["measured_delta"] for r in summary], color="#2463a8")
    axes[1, 0].axhline(0, color="black", linewidth=0.7)
    axes[1, 0].set(title="Weight change at reward (late pulse: 720 ms)", ylabel="Delta weight")
    for seed in sorted({r["seed"] for r in pilot}):
        rows = [r for r in pilot if r["seed"] == seed]
        steps = sorted({r["environment_step"] for r in rows})
        success = [np.mean([r["success"] for r in rows if r["environment_step"] == s]) for s in steps]
        axes[1, 1].plot(steps, success, marker="o", label=f"seed {seed}")
    axes[1, 1].set(title="Online-only corridor pilot; frozen rollouts", xlabel="Real training actions", ylabel="Goal success", ylim=(-0.05, 1.05))
    axes[1, 1].legend()
    fig.suptitle("Task 2: model and plasticity acceptance evidence", fontsize=15)
    fig.savefig(output / "spike_plasticity_diagnostic.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("snn_pilot_config.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/snn_pilot"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model_config = SNNConfig(**config.get("model", {}))
    if (len(config["seeds"]) != len(set(config["seeds"])) or not config["seeds"]
            or any(not isinstance(config[k], int) or isinstance(config[k], bool) or config[k] <= 0
                   for k in ("training_steps", "max_steps", "checkpoint_every", "evaluation_rollouts"))):
        raise ValueError("unique seeds and positive integer budgets are required")
    args.output.mkdir(parents=True, exist_ok=True)
    plasticity, pairs = plasticity_diagnostic(model_config)
    fixture = json.loads(FIXTURE_PATH.read_text())
    agent = SpikingAgent(49, 11, model_config)
    agent.record_spikes = True
    sequence = [Transition(**{k: row[k] for k in Transition.__dataclass_fields__})
                for row in fixture["episodes"][0]["transitions"]]
    agent.replay(sequence)
    agent.rest(50)
    spikes = [{"time_ms": time, "population": pop, "neuron": neuron} for time, pop, neuron in agent.spike_events]
    write_csv(args.output / "spikes.csv", spikes)
    write_csv(args.output / "plasticity_trace.csv", plasticity)
    evaluation, training, summaries = run_pilot(config, model_config)
    write_csv(args.output / "pilot_evaluation.csv", evaluation)
    write_csv(args.output / "pilot_training_episodes.csv", training)
    payload = {
        "kind": "real_snn_online_pilot_and_fixture_driven_diagnostic",
        "config": {**config, "model": asdict(model_config)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("snn_model.py", "snn_diagnostics.py", "maze.py", "parallel_fixtures.json")},
        "plasticity_cases": pairs,
        "late_to_early_delta_ratio": pairs[-1]["measured_delta"] / pairs[0]["measured_delta"],
        "pilot_seeds": summaries,
        "limitations": ["One corridor, three seeds, online only; no replay benefit claim.",
                        "Spike-pair stimuli and guided diagnostic route are supplied fixtures.",
                        "Frozen pilot rollouts use the real LIF model; every failure remains in the denominator.",
                        "Cue-only recall performance and integrated stress assays are not measured by this pilot."],
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_diagnostic(args.output, spikes, plasticity, pairs, evaluation)


if __name__ == "__main__":
    main()
