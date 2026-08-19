"""Parameters for the isolated delayed-search research tracker."""

import os

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    params.delayed_search_enabled = os.environ.get("FARTRACK_DELAYED_SEARCH_ENABLED", "1") == "1"
    params.delayed_search_expansion_factor = float(
        os.environ.get("FARTRACK_DELAYED_SEARCH_FACTOR", "1.10")
    )
    # Filled from node7's public calibration partition q85 statistic.  This
    # literal is intentionally checked against the preselection manifest.
    params.delayed_search_disagreement_threshold = 0.0037756040692329407
    return params
