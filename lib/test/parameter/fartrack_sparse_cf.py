"""Parameters for the counterfactual template-pool verifier."""

from lib.test.parameter.fartrack_sparse_research import parameters as base_parameters


def parameters(yaml_name: str):
    params = base_parameters(yaml_name)
    # Fixed before B_dev exists. These are not selected using B_test.
    params.cf_group_size = 2
    params.cf_min_accepted = 5
    params.cf_max_disagreement = 0.075
    return params
