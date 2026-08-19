"""Parameters for pre-registered node-17 CSPP selection evaluation."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.cspp_enabled = os.environ.get("FARTRACK_CSPP_ENABLED", "1") != "0"
    params.cspp_threshold = float(os.environ.get("FARTRACK_CSPP_THRESHOLD", "0.0259431"))
    return params
