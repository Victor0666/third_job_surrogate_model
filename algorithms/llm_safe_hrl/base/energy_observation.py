"""Candidate-relative features and placement diagnostics (no energy model)."""
import numpy as np

ENERGY_OBSERVATION_VERSION = "energy_relative_v1"
ENERGY_DIAGNOSTIC_FIELDS = (
    "mean_host_energy_regret", "mean_host_energy_regret_norm",
    "mean_vm_energy_regret", "mean_vm_energy_regret_norm",
    "mean_selected_vm_energy_rank", "best_energy_vm_selection_rate",
    "global_best_energy_vm_selection_rate",
)
PLACEMENT_ENERGY_FIELDS = (
    "host_energy_regret", "host_energy_regret_norm",
    "vm_energy_regret", "vm_energy_regret_norm",
    "selected_vm_energy_rank", "selected_best_energy_vm",
    "selected_global_best_energy_vm",
)


def relative_energy(values, legal):
    """Illegal slots are 1; equal/single legal candidates are 0."""
    values = np.asarray(values, dtype=float)
    legal = np.asarray(legal, dtype=bool)
    result = np.ones(values.shape, dtype=np.float32)
    if legal.any():
        candidates = values[legal]
        if not np.isfinite(candidates).all():
            raise ValueError("Legal candidate energy must be finite")
        result[legal] = np.clip(
            (candidates - candidates.min()) / (np.ptp(candidates) + 1e-9), 0, 1
        )
    return result


def placement_energy_diagnostics(energies, host_indices, selected_index):
    """Input contains legal VM global indices only; ties share rank 1."""
    local = [energies[i] for i in host_indices if i in energies]
    selected = energies[selected_index]
    global_best, global_worst = min(energies.values()), max(energies.values())
    host_best = min(local)
    host_regret = max(0.0, host_best - global_best)
    vm_regret = max(0.0, selected - host_best)
    # A common global range makes normalized Host + VM regrets additive.
    denominator = global_worst - global_best + 1e-9
    return dict(zip(PLACEMENT_ENERGY_FIELDS, (
        host_regret, host_regret / denominator,
        vm_regret, vm_regret / denominator,
        1 + sum(value < selected for value in local),
        int(selected == host_best), int(selected == global_best),
    )))
