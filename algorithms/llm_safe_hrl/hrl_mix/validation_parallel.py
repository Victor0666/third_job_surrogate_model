# -*- coding: utf-8 -*-
"""验证评估的进程级并行池。

每 ``validation_interval`` 轮，训练主循环要在固定 source 场景（以及课程学习
开启时的各训练场景）上，对全部 validation seed 各跑一个确定性 episode。这些
episode 之间没有任何共享可变状态：

- 每个 seed 现建一个独立环境，seed 完全决定环境；
- 三个 agent 只被只读推理调用（``deterministic=True`` 关掉 epsilon 分支，
  ``count_step=False`` 关掉步数计数），不写回任何 agent 状态；
- 汇总只发生在
  :func:`hrl_mix.train_eval.aggregate_seed_results`，它按 seed 顺序接收结果。

因此把 seed 分发到常驻工作进程后，返回值与串行执行逐位一致。

**设备约束**：worker 必须与父进程使用同一设备。验证结果直接决定 best
checkpoint 的选择（:mod:`hrl_mix.model_selection` 的可行性优先比较键），
而 masked argmax 在 cpu 与 cuda 之间存在最后一两个 ulp 的差异，极小概率翻转
选择。本模块因此把父进程的 ``device`` 原样传给每个 worker，不做任何降级。

**审计开关**：设 ``SAFE_HRL_VALIDATION_PARALLEL_AUDIT=1`` 后，每批 seed 会再
串行跑一遍并逐位比对（墙钟类字段除外）。正式长跑前建议在目标机器、目标设备
上先跑一次带审计的验证，确认并行路径无偏差。
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import torch

from base.d3qn_agent import D3QNAgent
from hrl_mix.train_eval import (
    assert_seed_results_identical,
    evaluate_one_seed,
)

#: 打开逐位审计的环境变量。
AUDIT_ENV_VAR = "SAFE_HRL_VALIDATION_PARALLEL_AUDIT"

#: 工作进程内缓存的环境类、agent 副本与权重版本。
_WORKER_STATE: dict = {}


@dataclass(frozen=True)
class AgentReplicaSpec:
    """重建一个只做推理的 agent 副本所需的全部构造参数。

    只包含影响 ``select_action`` 结果的字段。学习率、replay 容量之类与推理
    无关的参数在 worker 里取默认值，``buffer_size`` 固定为 1，避免每个 worker
    白白预分配整容量的 replay 数组。
    """

    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    head_hidden_dims: tuple[int, ...] | None
    observation_schema_version: str
    safe_rl_enabled: bool

    @classmethod
    def from_agent(cls, agent: D3QNAgent) -> "AgentReplicaSpec":
        head = agent.head_hidden_dims
        return cls(
            input_dim=int(agent.input_dim),
            output_dim=int(agent.output_dim),
            hidden_dims=tuple(int(v) for v in agent.hidden_dims),
            head_hidden_dims=(
                tuple(int(v) for v in head) if head else None
            ),
            observation_schema_version=str(
                agent.observation_schema_version
            ),
            safe_rl_enabled=bool(agent.safe_rl_enabled),
        )

    def build(self, device: str) -> D3QNAgent:
        return D3QNAgent(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            head_hidden_dims=self.head_hidden_dims,
            device=device,
            observation_schema_version=(
                self.observation_schema_version
            ),
            safe_rl_enabled=self.safe_rl_enabled,
            buffer_size=1,
        )


_LAYERS = ("vm", "host", "manager")


def _agent_weights(agent: D3QNAgent) -> dict:
    """取出决定推理结果的全部张量与标量。"""
    payload = {
        "online": {
            key: value.detach().cpu()
            for key, value in agent.online.state_dict().items()
        },
        "lagrange_multiplier": float(agent.lagrange_multiplier),
    }
    q_c_online = getattr(agent, "q_c_online", None)
    if q_c_online is not None:
        payload["q_c_online"] = {
            key: value.detach().cpu()
            for key, value in q_c_online.state_dict().items()
        }
    return payload


def _apply_weights(agent: D3QNAgent, payload: dict, device: str) -> None:
    agent.online.load_state_dict(
        {
            key: value.to(device)
            for key, value in payload["online"].items()
        }
    )
    agent.online.eval()
    agent.lagrange_multiplier = float(payload["lagrange_multiplier"])
    q_c_payload = payload.get("q_c_online")
    q_c_online = getattr(agent, "q_c_online", None)
    if q_c_payload is not None and q_c_online is not None:
        q_c_online.load_state_dict(
            {
                key: value.to(device)
                for key, value in q_c_payload.items()
            }
        )
        q_c_online.eval()


def _init_worker(env_module, env_qualname, specs, device, torch_threads):
    """建一次环境类引用与三个 agent 骨架，之后每批只换权重。"""
    # 每个 worker 单线程，避免 N 个进程各开满 BLAS 线程互相抢核。
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(int(torch_threads))
    torch.set_num_threads(int(torch_threads))

    import importlib

    module = importlib.import_module(env_module)
    _WORKER_STATE["env_cls"] = getattr(module, env_qualname)
    _WORKER_STATE["device"] = device
    _WORKER_STATE["agents"] = {
        layer: specs[layer].build(device) for layer in _LAYERS
    }
    _WORKER_STATE["weights_version"] = None


def _run_seed_job(payload):
    """在 worker 内跑一个 seed 的评估 episode。"""
    (
        job_index,
        weights_path,
        weights_version,
        env_kwargs,
        seed,
        return_safety_metrics,
    ) = payload
    device = _WORKER_STATE["device"]
    agents = _WORKER_STATE["agents"]
    if _WORKER_STATE["weights_version"] != weights_version:
        weights = torch.load(weights_path, map_location="cpu", weights_only=False)
        for layer in _LAYERS:
            _apply_weights(agents[layer], weights[layer], device)
        _WORKER_STATE["weights_version"] = weights_version
    result = evaluate_one_seed(
        _WORKER_STATE["env_cls"],
        env_kwargs,
        agents["vm"],
        agents["host"],
        agents["manager"],
        seed,
        return_safety_metrics=return_safety_metrics,
    )
    return job_index, result


def resolve_worker_count(requested, *, job_count):
    """把 ``--validation-workers`` 解析成实际进程数。

    ``0`` 表示自动：取 CPU 数减 2 与本批作业数的较小值。``1`` 表示严格串行，
    此时完全不创建进程池，行为与改动前一模一样。
    """
    requested = int(requested)
    if requested < 0:
        raise ValueError("validation worker count must be non-negative")
    if requested == 0:
        available = os.cpu_count() or 1
        requested = max(1, available - 2)
    return max(1, min(requested, max(1, int(job_count))))


class ValidationEvaluationPool:
    """常驻工作进程池，按 seed 分发验证 episode。

    权重通过一个临时文件在批次之间广播：一批验证只写一次，worker 只在版本号
    变化时重新加载。这比把 state_dict 塞进每个作业的 payload 便宜得多。
    """

    def __init__(
        self,
        env_cls,
        vm_agent,
        host_agent,
        manager_agent,
        *,
        workers,
        device,
        weights_path,
    ):
        self._env_cls = env_cls
        self._agents = {
            "vm": vm_agent,
            "host": host_agent,
            "manager": manager_agent,
        }
        self._device = str(device)
        self._weights_path = str(weights_path)
        self._weights_version = 0
        self._workers = int(workers)
        self.audit_enabled = os.environ.get(AUDIT_ENV_VAR, "") == "1"
        specs = {
            layer: AgentReplicaSpec.from_agent(agent)
            for layer, agent in self._agents.items()
        }
        # spawn：跨平台一致，且不会让子进程继承父进程的 CUDA 上下文。
        self._executor = ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(
                env_cls.__module__,
                env_cls.__qualname__,
                specs,
                self._device,
                1,
            ),
        )
        # 训练中途异常退出时 ProcessPoolExecutor 自己的 atexit 会回收 worker，
        # 但共享权重文件不归它管，这里补一个幂等的清理。
        atexit.register(self._remove_weights_file)

    def _remove_weights_file(self) -> None:
        try:
            os.remove(self._weights_path)
        except OSError:
            pass

    @property
    def worker_count(self) -> int:
        return self._workers

    def publish_weights(self) -> None:
        """把当前三个 agent 的推理权重写到共享文件并推进版本号。

        主循环每次验证前调用一次；同一次验证里的多个场景共用同一份权重。
        """
        payload = {
            layer: _agent_weights(agent)
            for layer, agent in self._agents.items()
        }
        torch.save(payload, self._weights_path)
        self._weights_version += 1

    def evaluate_jobs(self, jobs, *, return_safety_metrics):
        """并行跑完一批 ``(env_kwargs, seed)`` 作业，按入参顺序返回结果。

        课程学习开启时一次验证要跑多个场景，把场景与 seed 一起摊平成作业表
        才能填满 worker——只按 seed 并行的话，场景之间仍然是串行的。
        """
        jobs = list(jobs)
        if self._weights_version == 0:
            raise RuntimeError(
                "publish_weights() must be called before evaluate_jobs()"
            )
        payloads = [
            (
                index,
                self._weights_path,
                self._weights_version,
                env_kwargs,
                int(seed),
                bool(return_safety_metrics),
            )
            for index, (env_kwargs, seed) in enumerate(jobs)
        ]
        results = [None] * len(payloads)
        # 结果按 job_index 写回，与完成顺序无关——聚合顺序必须是入参顺序。
        try:
            for job_index, result in self._executor.map(_run_seed_job, payloads):
                results[job_index] = result
        except BrokenProcessPool as exc:
            # 裸的 BrokenProcessPool 只说"worker 没了"，不说为什么。这里把
            # spawn 模式下最常见的两个死因说清楚，否则排查要从头猜。
            raise RuntimeError(
                "validation worker process died. With the spawn start method "
                "the most common causes are (1) the entry point runs training "
                "at module import time instead of under an "
                "`if __name__ == \"__main__\":` guard, which makes every "
                "worker re-run the whole training script, and (2) running out "
                "of memory with "
                f"{self._workers} workers. Re-run with --validation-workers 1 "
                "to fall back to the serial path."
            ) from exc
        if self.audit_enabled:
            # 审计模式：同一批作业在父进程里再串行跑一遍，逐位比对。
            serial = [
                evaluate_one_seed(
                    self._env_cls,
                    env_kwargs,
                    self._agents["vm"],
                    self._agents["host"],
                    self._agents["manager"],
                    seed,
                    return_safety_metrics=bool(return_safety_metrics),
                )
                for env_kwargs, seed in jobs
            ]
            assert_seed_results_identical(results, serial)
        return results

    def evaluate_seeds(self, env_kwargs, seeds, *, return_safety_metrics):
        """并行跑完一批 seed，按入参顺序返回单 seed 结果。"""
        return self.evaluate_jobs(
            [(env_kwargs, seed) for seed in seeds],
            return_safety_metrics=return_safety_metrics,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._remove_weights_file()

    def __enter__(self) -> "ValidationEvaluationPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
