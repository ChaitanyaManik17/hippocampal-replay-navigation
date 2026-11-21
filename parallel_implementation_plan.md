# Hippocampal replay for adaptive navigation

Rutgers CS 525 Neural Computation · Parallel implementation plan

Each person owns one component, one validation suite, one evidence artifact, and one report section. The task budgets are comparable by deliverable. Debugging and simulation runtime can vary.

## Independent development

- Work in separate branches and keep the module names below distinct. Everyone can start from the existing maze, baseline data, and test fixtures in this repository.
- Run your component against the reference inputs listed under your task. Standalone acceptance uses your own code and the committed fixtures. Each teammate can finish and open their change independently.
- After all four packages are ready, run a shared integration check with the real model, scheduler, and evaluator. The combined scientific results are produced during that check.

## Task 1 · Chaitanya Manik — Maze, baselines, and shared fixtures

- **Build:** Own `maze.py`, `baselines.py`, `experiment.py`, and the shared interfaces and fixtures. Deliver the corridor, T-maze, and 7 × 7 original and blocked maze, plus random, Q-learning, and DQN baselines.
- **Validate:** Verify stable observation IDs, reward and terminal behavior, seeded starts, collision and timeout cases, a legal detour, and deterministic fixture regeneration. Check that the training and recall stand-ins exercise their interfaces.
- **Evidence:** Deliver the ten-seed blocked-route result, the recovery figure, and the fixture checks. The committed baseline results contain 4,500 evaluation rollouts.
- **Write:** Own the objective, environment, baseline methods, and blocked-route baseline results in `baseline_report.md`. Keep the README and demo commands current.
- **Start with:** Existing maze and baseline runner, `baseline_evaluation.csv`, and the PyTorch DQN reference below.

## Task 2 · Ketan Singh — Spiking model and plasticity

- **Build:** Own `snn_model.py`. Implement place input, recurrent excitatory LIF neurons with inhibition, four action pools, and reward-gated STDP with eligibility traces. Expose the online, offline, and cue-only methods from `integration.md`.
- **Validate:** Use the three supplied spike-pair cases to check causal, reversed, and zero-reward behavior. Check that a late reward has a smaller effect after trace decay, and that evaluation leaves learned weights unchanged. Run a small online corridor or T-maze pilot using `GridMaze` directly.
- **Evidence:** Save one spike and plasticity diagnostic, plus pilot success from three fixed seeds. Keep unsuccessful runs in the result.
- **Write:** Own the model architecture, action decoding, plasticity, and stability notes.
- **Start with:** `parallel_fixtures.json` → `plasticity_cases`, and the existing maze. The pilot needs only online steps.

## Task 3 · Dwiti Milind Choksi — Replay scheduler and runner

- **Build:** Own `replay_scheduler.py` and `replay_runner.py`. Implement silent rest, random recorded transitions, uniform ordered snippets, and adaptive ordered snippets. Use seeded sampling and log source episode, order, rewards, offline time, and update count.
- **Validate:** Use `TraceAgent` and the recorded fixture episodes to check all four modes. Verify that ordered snippets remain within episodes, terminal rewards are preserved, and offline work never steps the maze. Match offline durations and total replayed transitions across active modes.
- **Evidence:** Save one four-condition audit table with sampled counts, duration, update count, and an ordering check. Use the synthetic fixtures for this deliverable.
- **Write:** Own replay sampling, controls, and budget accounting. Document the priorities and tie behavior.
- **Start with:** `parallel_fixtures.json` → `episodes`, and `TraceAgent` from `parallel_fixtures.py`.

## Task 4 · Charvi Mayank Shastri — Scoring, memory assays, and figures

- **Build:** Own `memory_evaluation.py`. Implement frozen navigation scoring and cue-only recall scoring. Include missing and spurious cue conditions, a recurrence-off comparison, and a before/after interference score.
- **Validate:** Use `ScriptedRecall` to verify exact match, partial match, and silent failure. Withhold the ground truth from the probe and count absent decoded bins as misses. Validate navigation aggregation on the committed baseline CSV and keep failures in the denominator.
- **Evidence:** Save one reproducible scoring figure or table fed by the baseline data and explicitly labeled synthetic recall cases.
- **Write:** Own evaluation methods, metrics, and limitations. Fill the combined memory table from actual model runs during integration.
- **Start with:** `baseline_evaluation.csv`, `parallel_fixtures.json` → `recall_case`, and `ScriptedRecall` from `parallel_fixtures.py`.

## Shared integration and submission

1. Use the versioned interface in `integration.md`. Verify fixture regeneration with `python -m parallel_fixtures --check`.
2. Swap the stand-ins for the real model in the runner and evaluator. Use one seed to verify reset, transition handling, silent rest, replay logs, frozen recall, and route blockage.
3. Run the four replay conditions with matched real maze-action budgets. Keep a seed-level result for each run and include failures. Use the same starts across conditions and separate replay computation from real interactions.
4. Combine the real navigation and recall figures with the baseline report. Compare against the course submission rubric. Each person reviews their own module, outputs, and report section.

The fixture outputs are development checks. Only runs with the integrated spiking model support claims about spiking replay or memory.

## References

- [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Brian2 reward-modulated STDP example](https://brian2.readthedocs.io/en/2.11.0/examples/frompapers.Izhikevich_2007.html)
- [Florian (2007), Reinforcement Learning Through Modulation of STDP](https://www.florian.io/papers/2007_Florian_Modulated_STDP.pdf)
- [Foster and Wilson (2006), Reverse replay of behavioural sequences](https://www.nature.com/articles/nature04587)
