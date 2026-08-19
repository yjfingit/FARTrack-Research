"""Parameters for recent-consensus template rebinding."""

from lib.test.parameter.fartrack_sparse_research import parameters as baseline_parameters


def parameters(yaml_name: str):
    params = baseline_parameters(yaml_name)
    params.recent_consensus_window = 30
    return params
