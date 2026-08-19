"""Parameters for passive frozen-model localization reliability diagnostics."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.reliability_log_dir = os.environ.get(
        "FARTRACK_RELIABILITY_LOG_DIR",
        "/root/autodl-tmp/experiment/.research-assets/output/node7_reliability/logs",
    )
    return params
