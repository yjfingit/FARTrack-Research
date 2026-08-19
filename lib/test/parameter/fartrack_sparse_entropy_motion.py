"""Parameters for the isolated entropy-motion candidate."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.entropy_motion_enabled = os.environ.get("FARTRACK_ENTROPY_MOTION_ENABLED", "1") != "0"
    # These values are overwritten only through this isolated parameter file
    # during the calibration schedule; they are fixed before selection.
    params.entropy_motion_threshold = float(os.environ.get(
        "FARTRACK_ENTROPY_MOTION_THRESHOLD", "0.4000612944364548"
    ))
    params.entropy_motion_alpha = float(os.environ.get("FARTRACK_ENTROPY_MOTION_ALPHA", "0.25"))
    return params
