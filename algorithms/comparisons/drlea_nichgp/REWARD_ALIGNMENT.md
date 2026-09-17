# Reward alignment

## Routing Agent

For the FCFS task, `expected_risk_finish` is the mean read-only risk-finish
prediction over currently legal idle VMs. For selected VM `a`:

```text
completion = expected_risk_finish - selected_risk_finish
deadline = min(alpha_deadline * (task_deadline - selected_risk_finish), 0)
energy = -beta_energy * actual_incremental_fuzzy_energy / energy_normalizer
slack = slack_weight * global_risk_slack_improvement / time_normalizer
```

The energy baseline is the environment's
`fuzzy_energy_mean + 1.0 * fuzzy_energy_std` immediately before assignment;
the after value is read immediately after that assignment. Reward estimation
does not mutate state.

| Mode | Components |
|---|---|
| `completion_only` | completion |
| `deadline_energy` | completion + deadline + energy |
| `deadline_energy_slack` | completion + deadline + energy + slack |

Defaults are `alpha_deadline=1`, `beta_energy=0.05`,
`energy_normalizer=1000 J`, `slack_weight=0.1`, `time_normalizer=300 s`.
`select_reward_mode` compares variants using validation metrics and the common
tuple. Test metrics are rejected as a tuning source by the strict seed split.

## Sequencing Agent

Global risk slack is mean `deadline - predicted_workflow_risk_finish` over
arrived unfinished workflows. SA receives
`clip((after_slack - before_slack) / 300, -3, 3)`. The two values bracket only
the chosen rule/task and frozen RA assignment.

Final scientific comparison never uses reward; it uses fuzzy workflow
lateness and fuzzy energy from the shared simulator.
