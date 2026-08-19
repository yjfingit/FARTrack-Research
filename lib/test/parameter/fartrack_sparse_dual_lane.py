"""Fixed parameters for entropy-conditioned dual-lane memory."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.dual_lane_enabled = os.environ.get("FARTRACK_DUAL_LANE_ENABLED", "1") == "1"
    params.stable_entropy_threshold = float(
        os.environ.get("FARTRACK_STABLE_ENTROPY_THRESHOLD", "0.4000612944364548")
    )
    return params
