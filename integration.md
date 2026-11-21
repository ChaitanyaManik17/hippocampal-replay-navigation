# Team integration contract

Use `maze.py` as the shared environment implementation. A teammate can wrap this interface for their agent or runner.

## Observation and transitions

- Build `GridMaze(map_id, seed, max_steps=150, blocked=False)`. The versioned identifiers are `corridor-v1`, `t-maze-v1`, and `loop-7x7-v1`. The last has a blocked variant. Use a seeded default start for training and explicit starts for paired evaluation.
- `reset(seed=None, start=None)` returns the integer observation `row * maze.width + col`. Its range is `0 <= observation < maze.num_observations`. Preserve the position-to-cue codebook across map changes. If you corrupt cues for an assay, keep the underlying true cell within the evaluator.
- `step(action)` returns `(next_observation, reward, terminated, truncated, info)`. The action order is up, down, left, right (0 through 3). `info["collision"]` reports an observed invalid move. Never pass map geometry or BFS results to the policy for action selection.
- `Transition(observation, action, reward, next_observation, terminated, truncated)` is the baseline representation. Record the agent's actual input cue in the teammate replay format and attach episode ID, time index, and map phase. Store both end flags so time limits are distinguishable from goal arrivals.

## Agent and runner responsibilities

- Baselines implement `act(observation, training=...)` and `observe(transition)`. A wrapper can call the SNN's episode reset, cue encoder, spiking action method, and reward update through the same boundary.
- Dwiti's runner owns the replay schedule and storage. Count only training-environment actions in `environment_step`, and record offline updates separately. Evaluate in a separate environment without learning or replay-buffer writes. For a stateful agent, reset transient state at independent rollout boundaries and isolate it from ongoing training.
- To change the route, make a new environment with `blocked=True`, preserving the policy and learned state. Evaluate the original policy immediately after the change, before taking any blocked training action.
- Log map ID, agent/condition, seed, phase, phase step, cumulative environment step, start, success, rollout length, shortest length, path efficiency, and collisions. Keep failed rollouts in the data and leave successful-path efficiency blank for them.

## Handoffs

Ketan receives stable cue IDs and action order. Dwiti receives the transition schema and example runner. Charvi receives a paired rollout table, map phase labels, and the evaluation-only shortest-path helper. Use distinct output folders for new models or experimental conditions.
