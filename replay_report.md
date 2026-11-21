# Replay scheduler and runner

Owner: Dwiti Milind Choksi (Task 3). Code: `replay_scheduler.py`, `replay_runner.py`. Tests: `test_replay.py`.
Evidence: `replay_audit/` (synthetic fixtures with `TraceAgent`). Nothing in this section is a spiking-model result.

## What the runner does

Training alternates real maze actions with offline sessions. Every `replay_every` real actions, the runner pauses, asks the scheduler for a plan, and delivers it through `agent.replay(sequence, step_ms=...)` or `agent.rest(duration_ms)`. While a session runs, the training maze is wrapped by `GuardedMaze`, which raises `OfflineMazeAccess` on any `step` or `reset` and counts the attempt. After each session the runner calls `begin_episode()` and resets the maze. This happens in every condition, rest included, so the interruption is matched.

Real experience goes into a `ReplayBuffer` organised by episode. Each entry is the six-field `maze.Transition` plus runner metadata (episode ID, time index, phase, map ID, collision). Evaluation runs on a separate maze *and* a separate agent. At each checkpoint the runner takes a snapshot of the training agent with its current weights: `agent.evaluation_copy()` if the model defines it, otherwise `copy.deepcopy(agent)`. It calls `begin_episode()` on the snapshot before every rollout. The training agent's weights, transient state and RNG are never touched, so the training action sequence for a seed does not depend on how often evaluation runs (`test_evaluation_interval_does_not_change_training` checks intervals of 10, 25 and 180 steps in all four conditions). Evaluation never calls `observe` and never writes to the buffer. Rows use the same schema as `experiment.evaluate`. On the route change, the runner builds a `blocked=True` maze, keeps the agent and buffer, and evaluates before the first blocked training action (phase step 0).

## Replay conditions

| Condition | What is delivered | Calls per session |
|---|---|---|
| `rest` | `rest(T × step_ms)`, no transitions | 1 |
| `random` | T recorded transitions drawn uniformly with replacement from all steps, one transition per `replay` call | T |
| `ordered` | Contiguous within-episode windows drawn uniformly with replacement until T transitions are reached | about T / L |
| `adaptive` | The same windows as `ordered`, drawn with probability proportional to priority | about T / L |

T is `replay_transitions` and L is `snippet_length`. Random transitions go out as separate calls so the model resets its short-lived traces between independent samples. Packing unrelated transitions into one snippet would let spurious sequence structure leak into the control.

**Windows.** An episode with at least L steps contributes every length-L window. A shorter episode contributes one window with the whole episode, so short goal arrivals and timeouts stay eligible. An episode still open when a session starts can be sampled, since its recorded prefix is contiguous.

**Exact budget fill.** Windows are drawn until the running total reaches T. If the last draw would go over, it is cut to its first `remaining` steps. That keeps it contiguous, but it can drop a terminal step at the end of the window.

## Budget accounting

- `environment_step` counts training-maze actions only, so it is identical across conditions at every checkpoint.
- Every condition spends `T × step_ms` of offline simulated time per session. Active modes replay exactly T recorded transitions. Rest replays none.
- `update_count` is the number of replayed transitions, meaning reward-gated update opportunities offered to the model (0 for rest). Replay *calls* are logged separately because they differ by design: T for random, about T/L for ordered and adaptive.
- If a session falls before anything has been recorded, the plan becomes a rest of the matched duration and is logged with zero transitions.

Logs: `replay_log.csv` has one row per call, with source episode, phase, time-index span, observations, actions, rewards, both end flags, priority, contiguity flag and duration. `offline_sessions.csv` has one row per session, with calls, transitions, update count, offline ms, buffer size, candidates, goal transitions replayed, successor rate and attempted maze calls.

## Adaptive priorities and tie behaviour

For window *w* from episode *e*:

    score(w) = floor + a_s · S(e) + a_r · R(w) + a_c · C(e)
    P(w)     ∝ score(w) ^ alpha

- **S (success):** 1 if episode *e* ended at the goal, else 0.
- **R (rarity):** the mean over the window of 1/√(visits to that observation in the buffer).
- **C (recency):** exp(−age / τ), where age is the number of episodes recorded after *e*.

Defaults: a_s = a_r = a_c = 1, τ = 5 episodes, alpha = 1, floor = 0.05. The floor keeps every window possible. With all three weights at 0, adaptive reduces exactly to ordered (this is tested).

**Ties.** Windows are listed in a fixed order (oldest episode first, then start index) and drawn by cumulative weight with `random.choices`. Equal priorities therefore get equal probability. No argmax is involved, so no tie-breaking rule can bias selection. In the fixture audit the eight interior detour windows tie at 0.097 each.

**Seeding.** Each session uses `Random("replay:{seed}:{session}")`. The stream depends on seed and session but not on mode, so ordered and adaptive share common random numbers and differ only through their weights. Runs are reproducible across machines.

## Fixture audit (synthetic, development evidence)

Seed 0, T = 24, L = 4, 25 ms per transition, matched duration 600 ms. Regenerate with `python -m replay_runner --audit --output replay_audit`.

| Condition | Calls | Transitions | Updates | Offline ms | short-route | wall-and-timeout | blocked-detour | Goal steps | Within-episode order | Successor rate | Maze calls offline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rest | 1 (rest) | 0 | 0 | 600 | 0 | 0 | 0 | 0 | n/a | n/a | 0 |
| random | 24 | 24 | 24 | 600 | 3 | 1 | 20 | 3 | n/a | 0.043 | 0 |
| ordered | 6 | 24 | 24 | 600 | 0 | 0 | 24 | 1 | pass | 0.826 | 0 |
| adaptive | 6 | 24 | 24 | 600 | 0 | 0 | 24 | 1 | pass | 0.826 | 0 |

Every replayed transition is identical to the recorded one, so rewards and end flags are preserved. `TraceAgent` received exactly the planned sequences. Successor rate is the fraction of adjacent replayed transitions that are true successors. In ordered modes it is broken only at snippet boundaries: 19 of 23 pairs, which is the 18 within-snippet pairs plus one boundary where consecutive windows happened to line up. For random replay it is near chance.

One session of six draws cannot show the priority effect. `replay_audit/audit.md` adds per-window probabilities and empirical draw shares over 2,000 sessions. The timeout episode, which never reached the goal, drops from 0.091 under uniform sampling to 0.044 under adaptive sampling.

## Limitations

- These results come from fixtures and a stand-in agent. Whether replay helps the spiking model has to be measured in the integrated runs.
- A deep copy at every checkpoint is simple and safe, but it can be slow for a large network. The SNN can define `evaluation_copy()` to return a lighter frozen copy. It must return a separate object whose RNG and transient state are independent of the training agent's.
- Uniform window sampling weights long episodes more because they have more windows. The same is true of the adaptive baseline before its priorities are applied.
- Replay is forward-only. Reverse replay (Foster and Wilson, 2006) would be a natural extra condition.
