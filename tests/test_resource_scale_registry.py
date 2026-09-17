"""Canonical resource-scale and retained edge-capability tests."""

from __future__ import annotations

import pytest

from algorithms.llm_safe_hrl.scenario_registry import (
    RESOURCE_SCALE_REGISTRY,
    SCENARIO_REGISTRY,
    apply_scenario_to_problem_config,
)
from common.resource_opt import create_cluster


EXPECTED_RESOURCE_SCALES = {
    "S": (3, 25, (9, 8, 8)),
    "M": (6, 50, (9, 9, 8, 8, 8, 8)),
    "L": (9, 75, (9, 9, 9, 8, 8, 8, 8, 8, 8)),
}


@pytest.mark.parametrize(
    ("resource_code", "expected_hosts", "expected_vms", "expected_distribution"),
    [
        (code, values[0], values[1], values[2])
        for code, values in EXPECTED_RESOURCE_SCALES.items()
    ],
)
def test_canonical_resource_scale_counts_and_distribution(
    resource_code,
    expected_hosts,
    expected_vms,
    expected_distribution,
):
    scale = RESOURCE_SCALE_REGISTRY[resource_code]

    assert scale.total_hosts == expected_hosts
    assert scale.total_vms == expected_vms
    assert scale.vms_per_host == expected_distribution
    assert len(scale.host_types) == expected_hosts
    assert scale.num_cloud_hosts + scale.num_edge_hosts == expected_hosts
    assert sum(scale.cloud_vms_per_host) + sum(scale.edge_vms_per_host) == expected_vms


@pytest.mark.parametrize(
    ("scenario_ids", "resource_code"),
    [
        (("SS", "MS", "LS"), "S"),
        (("SM", "MM", "LM"), "M"),
        (("SL", "ML", "LL"), "L"),
    ],
)
def test_scenarios_share_their_canonical_resource_scale(
    scenario_ids,
    resource_code,
):
    scale = RESOURCE_SCALE_REGISTRY[resource_code]

    for scenario_id in scenario_ids:
        scenario = SCENARIO_REGISTRY[scenario_id]
        assert scenario.resource_code == resource_code
        assert scenario.resource_scale is scale
        assert scenario.total_hosts == scale.total_hosts
        assert scenario.total_vms == scale.total_vms
        assert scenario.vms_per_host == scale.vms_per_host
        assert scenario.host_types == scale.host_types


@pytest.mark.parametrize("scenario_id", tuple(SCENARIO_REGISTRY))
def test_scenario_materialization_uses_canonical_resource_mapping(scenario_id):
    configured = apply_scenario_to_problem_config(
        {"dataset": {}, "resources": {}},
        scenario_id,
    )
    scenario = SCENARIO_REGISTRY[scenario_id]
    resources = configured["resources"]

    assert resources == scenario.resource_scale.resource_mapping()
    combined_distribution = tuple(resources["cloud_vms_per_host"]) + tuple(
        resources["edge_vms_per_host"]
    )
    assert combined_distribution == scenario.vms_per_host
    assert resources["num_cloud_hosts"] + resources["num_edge_hosts"] == (
        scenario.total_hosts
    )
    assert sum(combined_distribution) == scenario.total_vms


@pytest.mark.parametrize("resource_code", ("S", "M", "L"))
def test_shared_cluster_constructor_retains_edge_hosts_and_power_models(
    resource_code,
):
    scale = RESOURCE_SCALE_REGISTRY[resource_code]
    hosts, vms = create_cluster(**scale.resource_mapping(), fuzzy_seed=0)

    assert len(hosts) == scale.total_hosts
    assert len(vms) == scale.total_vms
    assert tuple(len(hosts[host_id].vm_ids) for host_id in sorted(hosts)) == (
        scale.vms_per_host
    )
    assert tuple(hosts[host_id].server_type for host_id in sorted(hosts)) == (
        scale.host_types
    )
    assert {host.server_type for host in hosts.values()} == {"cloud", "edge"}
    assert all(callable(host.power_model) for host in hosts.values())
    assert all(host.power(0.0) > 0.0 for host in hosts.values())

