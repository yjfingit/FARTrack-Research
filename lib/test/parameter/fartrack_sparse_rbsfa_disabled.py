"""Identity-control parameters for RBSFA parity checks."""

from lib.test.parameter.fartrack_sparse_rbsfa import parameters as _parameters


def parameters(yaml_name: str):
    params = _parameters(yaml_name)
    params.rbsfa_enabled = False
    return params
