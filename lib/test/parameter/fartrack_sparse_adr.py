"""Parameters for appearance-diverse reservoir rebinding."""

from lib.test.parameter.fartrack_sparse_research import parameters as baseline_parameters


def parameters(yaml_name: str):
    params = baseline_parameters(yaml_name)
    params.appearance_reservoir_capacity = 4
    return params
