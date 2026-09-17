"""Spawn-safe CPU process pool for deterministic GP fitness evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
import multiprocessing

import torch

from .config import config_for_scenario
from .gp_primitives import GPProgram
from .metrics import aggregate_seed_metrics
from .routing_agent import RoutingAgent


_WORKER_CONFIGS = {}
_WORKER_ROUTING = None


def _init_worker(
    config,
    online_state,
    state_dim: int,
    action_dim: int,
    agent_config,
    agent_seed: int,
    threads_per_worker: int,
) -> None:
    global _WORKER_CONFIGS, _WORKER_ROUTING
    threads = max(1, int(threads_per_worker))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    inference_config = replace(agent_config, replay_capacity=1)
    routing = RoutingAgent(
        int(state_dim),
        int(action_dim),
        inference_config,
        int(agent_seed),
        device="cpu",
        role="routing",
    )
    routing.online.load_state_dict(online_state)
    routing.online.eval()
    _WORKER_ROUTING = routing
    _WORKER_CONFIGS = {
        scenario: config_for_scenario(config, scenario)
        for scenario in config.training_scenarios
    }


def _run_job(job):
    index, program_payload, scenario, seed = job
    if _WORKER_ROUTING is None:
        raise RuntimeError("GP evaluation worker was not initialized")
    program = GPProgram.from_dict(program_payload)
    try:
        from .niching_gp import run_rule_episode

        row = run_rule_episode(
            _WORKER_CONFIGS[str(scenario)],
            _WORKER_ROUTING,
            program,
            int(seed),
        )
        row["scenario_id"] = str(scenario)
        return index, row, None
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        return index, None, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class EvaluationBatch:
    results: list[dict | None]
    failures: list[list[str]]
    episode_count: int


class GPEvaluationPool:
    """Evaluate independent program/scenario/seed episodes in fixed order."""

    def __init__(
        self,
        config,
        routing_agent,
        *,
        workers: int = 1,
        threads_per_worker: int = 1,
    ):
        self.config = config
        self.routing_agent = routing_agent
        self.workers = max(1, int(workers))
        self.threads_per_worker = max(1, int(threads_per_worker))
        self.executor = None

    def __enter__(self):
        if self.workers > 1:
            if str(self.routing_agent.device) != "cpu":
                raise ValueError(
                    "parallel GP evaluation requires a CPU routing agent"
                )
            online_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.routing_agent.online.state_dict().items()
            }
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(
                    self.config,
                    online_state,
                    self.routing_agent.state_dim,
                    self.routing_agent.action_dim,
                    self.routing_agent.config,
                    self.routing_agent.seed,
                    self.threads_per_worker,
                ),
            )
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        return False

    @staticmethod
    def _unique_programs(programs) -> list[GPProgram]:
        unique = {}
        for program in programs:
            unique.setdefault(program.tokens, program)
        return list(unique.values())

    def evaluate(self, programs, seeds) -> EvaluationBatch:
        programs = self._unique_programs(programs)
        seed_values = tuple(int(seed) for seed in seeds)
        if not programs:
            return EvaluationBatch([], [], 0)
        if self.executor is None:
            return self._evaluate_serial(programs, seed_values)

        jobs = []
        locations = []
        index = 0
        for program_index, program in enumerate(programs):
            for scenario_index, scenario in enumerate(
                self.config.training_scenarios
            ):
                for seed_index, seed in enumerate(seed_values):
                    jobs.append(
                        (
                            index,
                            program.to_dict(),
                            scenario,
                            seed,
                        )
                    )
                    locations.append(
                        (
                            program_index,
                            scenario_index,
                            seed_index,
                        )
                    )
                    index += 1
        try:
            records = list(
                self.executor.map(_run_job, jobs, chunksize=1)
            )
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "GP worker pool failed; reduce --workers or check memory"
            ) from exc

        rows = [[] for _ in programs]
        failures = [[] for _ in programs]
        for (job_index, row, error), location in zip(records, locations):
            if int(job_index) >= len(jobs):
                raise RuntimeError("GP worker returned an invalid job index")
            program_index, _scenario_index, _seed_index = location
            if error is None:
                rows[program_index].append(row)
            else:
                failures[program_index].append(str(error))
        results = [
            None if failures[index] else aggregate_seed_metrics(rows[index])
            for index in range(len(programs))
        ]
        return EvaluationBatch(results, failures, len(jobs))

    def _evaluate_serial(self, programs, seeds) -> EvaluationBatch:
        from .niching_gp import evaluate_program

        results = []
        failures = []
        episode_count = 0
        episodes_per_program = (
            len(self.config.training_scenarios) * len(seeds)
        )
        for program in programs:
            try:
                results.append(
                    evaluate_program(
                        self.config,
                        self.routing_agent,
                        program,
                        seeds,
                    )
                )
                failures.append([])
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                results.append(None)
                failures.append([f"{type(exc).__name__}: {exc}"])
            episode_count += episodes_per_program
        return EvaluationBatch(results, failures, episode_count)
