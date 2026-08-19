"""Parameters for pre-registered rare two-view arbitration."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def _env_flag(name, default):
    return os.environ.get(name, str(int(default))).strip().lower() not in {"0", "false", "no", "off"}


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.two_view_enabled = _env_flag("FARTRACK_TWO_VIEW_ENABLED", True)
    params.two_view_disagreement_threshold = float(
        os.environ.get("FARTRACK_TWO_VIEW_THRESHOLD", "0.006562486290931702")
    )
    params.two_view_expansion = float(os.environ.get("FARTRACK_TWO_VIEW_EXPANSION", "1.1"))
    params.two_view_log_dir = os.environ.get("FARTRACK_TWO_VIEW_LOG_DIR", "")
    return params
