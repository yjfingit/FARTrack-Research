"""Parameters for disagreement-conditioned template-quality experiments."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def _flag(name, default):
    return os.environ.get(name, "1" if default else "0") != "0"


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.dctq_enabled = _flag("FARTRACK_DCTQ_ENABLED", True)
    params.dctq_threshold = float(os.environ.get(
        "FARTRACK_DCTQ_THRESHOLD", "0.0037756040692329407"))
    params.dctq_level = int(os.environ.get("FARTRACK_DCTQ_LEVEL", "1"))
    params.dctq_static = _flag("FARTRACK_DCTQ_STATIC", False)
    return params
