"""Physical timing shared by every workflow scheduler (seconds)."""

import math


COMMUNICATION_MODEL_VERSION = "input_transfer_then_compute_output_v1"


def execution_window(route_time, input_time, available_time, compute_time, output_time):
    values = tuple(map(float, (route_time, input_time, available_time, compute_time, output_time)))
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("transport times must be finite and non-negative")
    route, incoming, available, compute, outgoing = values
    arrival = route + incoming
    start = max(arrival, available)
    return arrival, start, start + compute + outgoing
