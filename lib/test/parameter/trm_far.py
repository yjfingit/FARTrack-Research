"""Parameters for the transactional-memory research tracker."""

from lib.test.parameter.fartrack_sparse_research import parameters as _baseline_parameters


def parameters(yaml_name: str):
    params = _baseline_parameters(yaml_name)
    params.memory_controller = "transactional_evidence_v1"
    return params
