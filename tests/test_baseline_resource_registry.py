"""Formal comparison adapters must consume the canonical scenario registry."""

from __future__ import annotations

import pytest

from algorithms.comparisons.drlea_nichgp.config import build_config, config_for_scenario
from algorithms.comparisons.fcfs.train_fcfs import build_fcfs_protocol
from algorithms.comparisons.fuzzy_common.protocol import FuzzyComparisonProtocol
from algorithms.llm_safe_hrl.scenario_registry import SCENARIO_REGISTRY


SCENARIOS = ("SS", "MS", "LS", "SM", "MM", "LM", "SL", "ML", "LL")


def _topology(value):
    return (
        int(value.num_cloud_hosts),
        int(value.num_edge_hosts),
        tuple(value.cloud_vms_per_host),
        tuple(value.edge_vms_per_host),
    )


def _mapping_topology(value):
    return (
        int(value["num_cloud_hosts"]),
        int(value["num_edge_hosts"]),
        tuple(value["cloud_vms_per_host"]),
        tuple(value["edge_vms_per_host"]),
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_formal_fuzzy_baselines_use_registry_topology(scenario):
    """IRWS/MARL/PD3QN share this protocol and its environment kwargs."""
    spec = SCENARIO_REGISTRY[scenario]
    protocol = FuzzyComparisonProtocol(
        scenario=scenario,
        ddl_setting="T",
        train_seeds=(1,),
        validation_seeds=(101,),
        test_seeds=(201,),
        workflows_per_episode=1,
    )
    assert _mapping_topology(
        protocol.environment_kwargs(1, require_deadline_cache=False)
    ) == _topology(spec)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_formal_fcfs_and_drlea_use_registry_topology(scenario):
    spec = SCENARIO_REGISTRY[scenario]
    fcfs = build_fcfs_protocol(
        scenario, "T", workflows_per_episode=1
    ).environment_kwargs(201, require_deadline_cache=False)
    source = {"S": "SS", "M": "SM", "L": "SL"}[spec.resource_scale.resource_code]
    drlea = config_for_scenario(
        build_config(
            source,
            "T",
            0,
            smoke=True,
            protocol="single",
            source_scenario=source,
        ),
        scenario,
    )

    assert _mapping_topology(fcfs) == _topology(spec)
    assert _topology(drlea) == _topology(spec)
