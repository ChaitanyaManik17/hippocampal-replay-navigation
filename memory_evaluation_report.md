# Evaluation methods, metrics, and limitations
**Scoring, memory assays, and figures · Charvi Mayank Shastri · Rutgers CS 525**

`memory_evaluation.py` scores every result in the project. It has two jobs: scoring frozen navigation from rollout tables, and running cue-only recall assays on any model that exposes `recall()`. The evaluator keeps all ground truth. Policies see position IDs. Recall probes see only cue channels and timing. Expected continuations, true cue cells and BFS distances stay inside the scoring code. The tests check this boundary.

## Navigation scoring

The input is any CSV in the `experiment.EvaluationRow` schema. `experiment.py` and `replay_runner.py` both write this format, so baselines and replay conditions go through the same code.

- **Run kind.** The agent ID `random` means two different things. In baseline runs it is the non-learning random-action reference. In replay runs it is the random-transition replay condition. The evaluator decides which kind a CSV holds from the `config.json` the runner wrote next to it (`agents` from `experiment.py`, `conditions` from `replay_runner.py`). Without that file it uses IDs that appear in only one kind, and `--kind baseline|replay` sets it explicitly. A table whose only ID is `random` is rejected rather than guessed. Labels, plot styles and the recovery rule come from the run kind. Only the random-action baseline gets N/A for recovery; Random replay uses the same criterion as the other replay conditions.
- **Audit first.** Before scoring, each row is checked against the evaluator's own BFS for its map phase and start. Successful rollouts must have efficiency = shortest / steps and can't be shorter than BFS. Failed rollouts must have no efficiency, and with a known action limit they must have run to it. Scoring stops if any row fails.
- **Checkpoint success** = successes / rollouts over all starts. Timeouts stay in the denominator.
- **Path efficiency** is averaged over successful rollouts only and reported separately, so an agent can't look efficient by failing.
- **Learning curves** show the mean ± SD of per-seed checkpoint success against *real maze actions*. Offline replay time is never on this axis.
- **Endpoints** are pooled over all rollouts of all seeds: original-final, blocked-initial (before any blocked training action) and blocked-final.
- **Recovery** is the first blocked checkpoint that starts a run of three consecutive checkpoints with every start successful. Seeds that never meet this are reported as censored (`k/N recovered`), not dropped. The median is taken over recovered seeds only and labeled that way. The random-action baseline is a non-learning reference, so its recovery is N/A.

**Validation on the committed baseline.** All 4,500 rollouts (1,500 checkpoints) pass the audit. The scorer reproduces `baseline_report.md` and every per-seed recovery step in `baseline_summary.json`:

| Agent | Original final | Blocked initial | Blocked final | Recovered seeds | Median recovery step |
| --- | ---: | ---: | ---: | ---: | ---: |
| DQN | 100.0% | 0.0% | 70.0% | 7/10 | 3,500 |
| Q-learning | 100.0% | 0.0% | 100.0% | 6/10 | 5,500 |
| Random action | 73.3% | 16.7% | 16.7% | N/A | N/A |

Source: `baseline_evaluation.csv`, conventional RL baselines on one layout.

## Recall scoring

A cue is a set of channels under the shared codebook `4 * observation + local_channel`. The probe returns one decoded location or `None` per bin after cue offset.

- **Per-bin scoring.** Each bin is correct, a **miss** (`None`) or an **error** (wrong location). Accuracy = correct / all bins, so silence counts against the model. Miss and error rates are reported separately because they mean different things: no activity versus the wrong activity.
- **Output validation.** A probe that returns the wrong number of bins, a non-integer, a boolean or an out-of-range ID is rejected rather than scored.
- **Ambiguous cues.** If the cue matches the start of several learned routes, every valid continuation is listed and the best-matching one is scored. The number of valid continuations is logged per trial.
- **Fixture check (synthetic).** `ScriptedRecall` on the committed `partial-route-cue` case scores 3/3 for intact, 0/3 with 3 misses for recurrent-off, and 2/3 with 1 miss for one-error, matching `integration.md`.

![Scoring evidence](runs/scoring/scoring_evidence.png)

*Left and middle panels: baseline navigation from the committed CSV. Right panel: synthetic ScriptedRecall test double, not model output. Regenerate with `python -m memory_evaluation evidence --output runs/scoring`.*

## Memory assays

Each assay changes one factor at a time and keeps everything else fixed. Routes are self-avoiding walks of 4-neighbour moves in a 7 × 7 grid that uses the same 49-ID codebook as the maze. Each route can be turned into `maze.Transition` rows, so a model can learn it through `replay()`. The cue is the route's first position (all four channels). Recall covers the next five positions. Each seed pools accuracy over routes and noise draws, and tables give the mean ± SD across seeds.

| Assay | What varies | What stays fixed |
| --- | --- | --- |
| Recurrence | excitatory recurrence on / off | trained network, routes, clean cue |
| Cue missing | fraction of true cue channels removed | routes, network, spurious = 0 |
| Cue spurious | channels added from positions outside the cue | routes, network, missing = 0 |
| Memory load | number of learned routes (fresh network per level) | network size, exposures per route, cue noise; total exposure is logged |
| Route similarity | cells in a shared middle segment | route count, route length, exposures, cue noise; measured shared-cell and shared-transition fractions are logged |
| Interference | exposure to interfering routes after learning | targets, network; each interferer leaves a target's second cell by a different successor, and exposure 0 is the control |

The **interference score** is accuracy after exposure minus accuracy before it, measured on the same trained model. **Degradation** is reported as the first level whose mean accuracy falls more than 0.10 below the sweep's first level. If no level does, the report says "no decline observed over the tested range".

The integrated model plugs in through `python -m memory_evaluation assays --model module:factory --output runs/memory`. `factory(num_observations, seed)` returns an object with `learn(routes, exposure_per_route=...)` and `recall(...)`. `ReplayAgentMemory` turns any `ReplayAgent` that also has `recall()` into such an object by replaying each route's transitions.

## Combined memory table

**To be filled from the integrated SNN run.** The SNN (`snn_model.py`) is not in the repository yet, so this table has no model results. The pipeline was checked end to end with `TransitionTableMemory`, a first-order lookup table that is labeled synthetic in every output row. The stand-in shows that the sweeps, degradation detection and interference scoring work. It has no capacity limit and no spiking dynamics, so its curves say nothing about the hypothesis. Pipeline check output: `runs/memory/memory_table.md`.

| Assay | Levels | Integrated SNN accuracy (mean ± SD) | Accuracy starts to fall at |
| --- | --- | --- | --- |
| Recurrence | on / off | pending integration | — |
| Cue missing | 0, 0.25, 0.5, 0.75, 1.0 | pending integration | pending |
| Cue spurious | 0, 2, 4, 6, 8 channels | pending integration | pending |
| Memory load | 1, 2, 4, 6 routes | pending integration | pending |
| Route similarity | 0–4 shared cells (2 routes) | pending integration | pending |
| Interference | exposure 0, 10, 20, 40 | pending integration | — |

## Limitations

- Navigation results come from one map layout with three evaluation starts, so per-checkpoint success can only be 0, 1/3, 2/3 or 1. Treat the seed, not the rollout, as the unit of comparison.
- The recovery criterion (three consecutive perfect checkpoints) is a choice. Runs that recover in the last two checkpoints are censored, as for four Q-learning seeds.
- Memory routes live on an open 7 × 7 grid that uses the maze's codebook, not on the loop maze. The loop has only 16 open cells, too few for the load sweep. These routes test sequence memory over place codes, not navigable paths.
- Grid geometry limits the similarity assay. A shared segment can be entered and left by at most three routes, or two if the segment is a single crossing cell. The default fixes the route count at two.
- Spurious channels are spread across random positions. A spurious cue concentrated on one competing position would be a harder test and is not run by default.
- The evaluator can't confirm by itself that `recall()` leaves weights unchanged. That check belongs to the model's own tests.
- Fixture and stand-in outputs only validate the evaluator. Claims about spiking replay or memory need the integrated model runs.
