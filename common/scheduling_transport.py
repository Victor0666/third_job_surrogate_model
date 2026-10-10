"""Physical timing shared by every workflow scheduler (seconds)."""

import math


COMMUNICATION_MODEL_VERSION = "combined_input_compute_output_v1"


def execution_window(route_time, input_time, available_time, compute_time, output_time):
    values = tuple(map(float, (route_time, input_time, available_time, compute_time, output_time)))
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("transport times must be finite and non-negative")
    route, incoming, available, compute, outgoing = values
    start = max(route, available)
    arrival = start + incoming
    return arrival, start, arrival + compute + outgoing
