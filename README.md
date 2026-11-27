# Hippocampal replay for adaptive navigation

Rutgers CS 525 Neural Computation. This repository starts with Chaitanya's maze environment, conventional reinforcement learning baselines, and route blockage experiment. The team will use these as a reference for its recurrent spiking network and hippocampal replay experiments.

![Goal success in the original and blocked maze across ten seeds](blocked_route_recovery.png)

[Baseline report](baseline_report.md) · [Team integration contract](integration.md) · [Rollout data](baseline_evaluation.csv) · [Seed-level summary](baseline_summary.json)

## Work in parallel

The [parallel implementation plan](parallel_implementation_plan.md) gives each teammate a component, validation checks, evidence, and a report section. The [shareable PDF](Hippocampal_Replay_Parallel_Implementation_Plan.pdf) describes the same assignments.

Run `python -m parallel_fixtures --check` and `python -m unittest test_parallel_fixtures -v` from the repository root. The fixture JSON includes recorded maze episodes, spike timing cases, and a scripted recall case. The stand-in agent and recall probe let the runner and evaluator pass standalone checks before the real model is connected.

## Run the demo

Tested with Python 3.12 and CPU PyTorch on Linux (x86-64). The requirements file uses the PyTorch CPU wheel index.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python -m experiment --config smoke_config.json --output runs/demo
python -m analysis runs/demo/evaluation.csv --output runs/demo
```

The final command writes `blocked_route_recovery.png` and `summary.json`. The demo runs one seed at a short budget to verify the pipeline. Use the full configuration to reproduce the committed experiment.

```bash
python -m experiment --config blocked_route_config.json --output runs/full
python -m analysis runs/full/evaluation.csv --output runs/full
```

Regenerate the figure directly from the committed rollout data with `python -m analysis baseline_evaluation.csv --output runs/replot`.

## Ketan's spiking model (Task 2)

`snn_model.py` implements the LIF model, reward-gated STDP, replay/rest boundary, and frozen cue-only probe. See the [model report](snn_report.md), [diagnostic](snn_evidence/spike_plasticity_diagnostic.png), and [seed-level evidence](snn_evidence/summary.json). The online corridor pilot uses three fixed seeds and retains all failed evaluations.

For standalone model work, Python 3.12+ with the smaller dependency file is sufficient; PyTorch is only needed for the existing DQN tests. The committed SNN evidence was generated on macOS arm64 with Python 3.14.4 and NumPy 2.3.5.

```bash
python -m pip install -r requirements-snn.txt
python -m unittest test_snn_model test_parallel_fixtures test_replay -v
python -m snn_diagnostics --config snn_pilot_config.json --output runs/snn_pilot
python -m replay_runner --config replay_smoke_config.json --agent snn_model:make_agent --output runs/snn_replay_smoke
```

The last command checks the real model with all four replay conditions and route blockage. Its small budget is an integration check, not a replay performance study. For frozen rollouts use `agent.evaluation_copy()` and call `begin_episode()` on that copy before each rollout. `recall()` isolates its own state automatically.

## Repository contents

| File | Purpose |
| --- | --- |
| `maze.py` | Versioned corridor, T-maze, and 7 × 7 loop. Stable position IDs, rewards, collision handling, and evaluation-only BFS. |
| `baselines.py` | Random agent, tabular Q-learning, and CPU PyTorch DQN. |
| `experiment.py` | Paired baseline training and frozen evaluation before and after the route changes. |
| `analysis.py` | Rollout aggregation, per-seed recovery summary, and plot generation. |
| `blocked_route_config.json` | Exact experiment configuration for the committed results. |
| `test_maze_and_experiment.py` | Environment, learning, and experiment regression tests. |
| `snn_model.py` | Place-coded recurrent LIF agent, inhibitory pool, action pools, eligibility traces, frozen recall. |
| `test_snn_model.py` | Spike-pair acceptance, evaluation isolation, replay accounting, and real runner handoff tests. |
| `snn_diagnostics.py`, `snn_pilot_config.json` | Reproducible spike/plasticity diagnostic and three-seed online corridor pilot. |
| `snn_report.md`, `snn_evidence/` | Model methods, stability/limitations, raw pilot outcomes, and diagnostic figure. |

The baseline results above cover conventional agents; the SNN pilot is reported separately. The recurrent SNN and replay scheduler are implemented. Integrated replay comparisons and cue-noise, memory-load, route-similarity, and retention assays remain shared team work. DQN uses conventional experience replay internally.
