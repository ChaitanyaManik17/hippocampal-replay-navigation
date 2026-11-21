# Replay scheduler audit (synthetic fixtures)

Source: `parallel_fixtures.json` episodes and `TraceAgent`. This is a development check of the
sampler and runner, not a model result.

Seed 0; budget 24 transitions per active session; max snippet length
4; 25 ms per replayed transition; matched duration 600 ms.
Columns `short-route`, `wall-and-timeout`, `blocked-detour` count replayed transitions per source episode.

| condition | replay_calls | rest_calls | replayed_transitions | update_count | offline_ms | short-route | wall-and-timeout | blocked-detour | goal_transitions | within_episode_order | successor_rate | flags_preserved | agent_received_plan | maze_calls_offline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rest | 0 | 1 | 0 | 0 | 600 | 0 | 0 | 0 | 0 | n/a | n/a | pass | pass | 0 |
| random | 24 | 0 | 24 | 24 | 600 | 3 | 1 | 20 | 3 | n/a | 0.043 | pass | pass | 0 |
| ordered | 6 | 0 | 24 | 24 | 600 | 0 | 0 | 24 | 1 | pass | 0.826 | pass | pass | 0 |
| adaptive | 6 | 0 | 24 | 24 | 600 | 0 | 0 | 24 | 1 | pass | 0.826 | pass | pass | 0 |

## Window probabilities

Adaptive weights: success 1.0, rarity 1.0, recency 1.0 (tau 5.0 episodes), alpha 1.0, floor 0.05.
Empirical columns are draw shares over 2,000 seeded sessions. The last draw of a session
may be cut short to fit the budget; it still counts as a draw of its window here.

| source_episode | start_time_index | length | reached_goal | ordered_prob | adaptive_prob | ordered_empirical | adaptive_empirical |
|---|---|---|---|---|---|---|---|
| short-route | 0 | 4 | 1 | 0.091 | 0.083 | 0.091 | 0.082 |
| wall-and-timeout | 0 | 2 | 0 | 0.091 | 0.044 | 0.088 | 0.046 |
| blocked-detour | 0 | 4 | 1 | 0.091 | 0.093 | 0.095 | 0.091 |
| blocked-detour | 1 | 4 | 1 | 0.091 | 0.097 | 0.091 | 0.100 |
| blocked-detour | 2 | 4 | 1 | 0.091 | 0.097 | 0.090 | 0.098 |
| blocked-detour | 3 | 4 | 1 | 0.091 | 0.097 | 0.089 | 0.094 |
| blocked-detour | 4 | 4 | 1 | 0.091 | 0.097 | 0.097 | 0.102 |
| blocked-detour | 5 | 4 | 1 | 0.091 | 0.097 | 0.095 | 0.101 |
| blocked-detour | 6 | 4 | 1 | 0.091 | 0.097 | 0.087 | 0.097 |
| blocked-detour | 7 | 4 | 1 | 0.091 | 0.097 | 0.089 | 0.095 |
| blocked-detour | 8 | 4 | 1 | 0.091 | 0.097 | 0.087 | 0.094 |
