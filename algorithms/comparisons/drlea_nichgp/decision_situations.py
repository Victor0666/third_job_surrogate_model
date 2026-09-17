"""Collect global-ready situations using a frozen routing agent."""

from __future__ import annotations

from dataclasses import dataclass

from .config import config_for_scenario
from .env_adapter import CEWSEnvAdapter
from .features import routing_state
from .gp_features import task_gp_terminals


@dataclass(frozen=True)
class DecisionSituation:
    seed: int
    current_time: float
    task_ids: tuple[int, ...]
    terminals: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "current_time": self.current_time,
            "task_ids": list(self.task_ids),
            "terminals": [list(values) for values in self.terminals],
        }


def collect_decision_situations(
    config,
    routing_agent,
    *,
    seeds,
    minimum_ready: int | None = None,
    target_size: int | None = None,
) -> list[DecisionSituation]:
    threshold = int(
        config.gp.decision_ready_threshold
        if minimum_ready is None
        else minimum_ready
    )
    target = int(
        config.gp.target_situations
        if target_size is None
        else target_size
    )
    situations = []
    for index, seed in enumerate(seeds):
        scenario = config.training_scenarios[
            index % len(config.training_scenarios)
        ]
        scenario_config = config_for_scenario(config, scenario)
        adapter = CEWSEnvAdapter(scenario_config, int(seed))
        adapter.reset()
        task_id = adapter.advance_until_actionable()
        while task_id is not None:
            ready = adapter.ready_tasks()
            if len(ready) >= threshold:
                situations.append(
                    DecisionSituation(
                        seed=int(seed),
                        current_time=adapter.current_time,
                        task_ids=tuple(ready),
                        terminals=tuple(
                            tuple(
                                task_gp_terminals(
                                    adapter,
                                    ready_task,
                                    config.normalization,
                                    ready,
                                ).tolist()
                            )
                            for ready_task in ready
                        ),
                    )
                )
                if len(situations) >= target:
                    return situations
            task_id = adapter.select_fcfs_task()
            state, mask = routing_state(
                adapter, task_id, config.normalization
            )
            vm_id, _kind = routing_agent.choose_action(
                state, mask, training=False
            )
            adapter.execute(task_id, vm_id)
            task_id = adapter.advance_until_actionable()
    if not situations:
        raise RuntimeError(
            "No GP decision situation met the configured ready threshold"
        )
    return situations
