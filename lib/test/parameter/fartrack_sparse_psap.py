"""Parameters for preregistered node-14 PSAP calibration and selection."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.psap_enabled = os.environ.get("FARTRACK_PSAP_ENABLED", "1") != "0"
    params.psap_threshold = float(os.environ.get("FARTRACK_PSAP_THRESHOLD", "inf"))
    params.psap_high_alpha = float(os.environ.get("FARTRACK_PSAP_HIGH_ALPHA", "0.5"))
    params.psap_low_alpha = float(os.environ.get("FARTRACK_PSAP_LOW_ALPHA", "0.5"))
    params.psap_log_dir = os.environ.get("FARTRACK_PSAP_LOG_DIR") or None
    return params
