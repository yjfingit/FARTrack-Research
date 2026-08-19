"""Parameters for isolated branch-disagreement causal mixing."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.branch_motion_enabled = os.environ.get("FARTRACK_BRANCH_MOTION_ENABLED", "1") != "0"
    params.branch_motion_threshold = float(os.environ.get(
        "FARTRACK_BRANCH_MOTION_THRESHOLD", "0.0"
    ))
    params.branch_motion_alpha = float(os.environ.get("FARTRACK_BRANCH_MOTION_ALPHA", "0.10"))
    return params
