"""Stage-1 flat global-VM routing agent and training loop."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .config import config_for_scenario, protocol_artifact_identity
from .checkpointing import (
    experiment_manifest,
    portable_path,
    prepare_output,
    write_csv,
    write_json,
)
from .d3qn import MaskedD3QN
from .env_adapter import CEWSEnvAdapter
from .features import routing_state
from .metrics import aggregate_seed_metrics, comparison_key, episode_metrics
from .rewards import routing_reward


class RoutingAgent(MaskedD3QN):
    expected_role = "routing"

    @classmethod
    def create(cls, config, seed: int):
        num_vms = sum(config.cloud_vms_per_host) + sum(
            config.edge_vms_per_host
        )
        return cls(
            8 + 10 * num_vms,
            num_vms,
            config.routing,
            seed,
            role="routing",
        )


def run_routing_episode(
    config,
    agent: RoutingAgent,
    seed: int,
    *,
    training: bool,
) -> tuple[dict, list[dict]]:
    adapter = CEWSEnvAdapter(config, int(seed))
    adapter.reset()
    task_id = adapter.advance_until_actionable()
    trace = []
    while task_id is not None:
        state, mask = routing_state(
            adapter, task_id, config.normalization
        )
        expected = adapter.expected_risk_finish(task_id)
        before_slack = adapter.aggregate_risk_slack()
        action, selection = agent.choose_action(
            state, mask, training=training
        )
        execution = adapter.execute(task_id, action)
        after_slack = adapter.aggregate_risk_slack()
        reward, reward_fields = routing_reward(
            expected_risk_finish=expected,
            selected_risk_finish=execution["risk_finish"],
            task_deadline=adapter.task_deadline(task_id),
            incremental_fuzzy_energy=execution[
                "incremental_fuzzy_energy_actual"
            ],
            risk_slack_improvement=after_slack - before_slack,
            config=config.reward,
        )
        next_task = adapter.advance_until_actionable()
        done = next_task is None
        if done:
            next_state = np.zeros(agent.state_dim, dtype=np.float32)
            next_mask = np.zeros(agent.action_dim, dtype=np.float32)
        else:
            next_state, next_mask = routing_state(
                adapter, next_task, config.normalization
            )
        loss = None
        if training:
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
                mask,
                next_mask,
            )
            loss = agent.learn()
        trace.append(
            {
                "task_id": int(task_id),
                "global_vm_id": int(action),
                "selection": selection,
                "reward": reward,
                "loss": loss,
                **reward_fields,
            }
        )
        task_id = next_task
    return episode_metrics(adapter), trace


def evaluate_routing(config, agent, seeds) -> dict:
    rows = []
    for scenario in config.training_scenarios:
        scenario_config = config_for_scenario(config, scenario)
        for seed in seeds:
            row = run_routing_episode(
                scenario_config, agent, int(seed), training=False
            )[0]
            row["scenario_id"] = scenario
            rows.append(row)
    return aggregate_seed_metrics(rows)


def train_routing(
    config,
    *,
    checkpoint_path: str | Path | None = None,
) -> tuple[RoutingAgent, Path, list[dict]]:
    started = time.perf_counter()
    started_cpu = time.process_time()
    output = prepare_output(config)
    path = Path(checkpoint_path or output / "ra.pt")
    agent = RoutingAgent.create(config, config.algorithm_seed)
    best_key = None
    history = []
    interaction_count = 0
    for episode in range(config.ra_episodes):
        train_seed = config.train_seeds[
            episode % len(config.train_seeds)
        ]
        training_scenario = config.training_scenarios[
            episode % len(config.training_scenarios)
        ]
        episode_config = config_for_scenario(config, training_scenario)
        train_metrics, trace = run_routing_episode(
            episode_config, agent, train_seed, training=True
        )
        interaction_count += len(trace)
        validation_due = (
            (episode + 1) % config.validation_interval == 0
            or episode + 1 == config.ra_episodes
        )
        validation = None
        saved = False
        if validation_due:
            validation = evaluate_routing(
                config, agent, config.validation_seeds
            )
            key = comparison_key(validation)
            saved = best_key is None or key < best_key
            if saved:
                best_key = key
                agent.save(
                    path,
                    metrics=validation,
                    protocol_identity=protocol_artifact_identity(config),
                )
        history.append(
            {
                "episode": episode + 1,
                "train_seed": int(train_seed),
                "training_scenario": training_scenario,
                "mean_reward": float(
                    np.mean([row["reward"] for row in trace])
                )
                if trace
                else 0.0,
                "epsilon": float(agent.epsilon),
                "saved": bool(saved),
                "validation_performed": bool(validation_due),
                **{
                    f"validation_{name}": (
                        float(validation[name]) if validation is not None else None
                    )
                    for name in (
                        "deadline_violation_rate",
                        "max_fuzzy_lateness",
                        "mean_fuzzy_lateness",
                        "fuzzy_energy_score",
                    )
                },
                "train_fuzzy_energy_score": train_metrics[
                    "fuzzy_energy_score"
                ],
            }
        )
    best_agent = RoutingAgent.load(
        path,
        expected_protocol_identity=protocol_artifact_identity(config),
    )
    write_json(output / "ra_history.json", history)
    write_csv(output / "ra_history.csv", history)
    write_json(
        output / "ra_manifest.json",
        experiment_manifest(
            config,
            stage="ra",
            elapsed_seconds=time.perf_counter() - started,
            extra={
                "checkpoint": portable_path(path),
                "best_comparison_key": list(best_key or ()),
                "environment_interaction_count": interaction_count,
                "validation_call_count": sum(
                    row["validation_performed"] for row in history
                ),
                "compute_device": str(agent.device),
                "process_cpu_seconds": float(
                    time.process_time() - started_cpu
                ),
            },
        ),
    )
    return best_agent, path, history
