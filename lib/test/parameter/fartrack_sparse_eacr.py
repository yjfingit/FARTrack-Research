"""Pre-registered parameters for the EACR node-13 calibration experiment."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.eacr_entropy_threshold = 0.47618
    params.eacr_min_mean_age = 40
    # This only supports the implementation-parity test; the registered
    # candidate always runs with the default enabled value.
    params.eacr_enabled = os.environ.get("FARTRACK_EACR_ENABLED", "1") != "0"
    params.save_all_boxes = False
    return params
