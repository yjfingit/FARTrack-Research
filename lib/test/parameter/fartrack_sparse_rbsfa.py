"""Parameters for the reliability-gated sparse-flow research tracker."""

from lib.test.parameter.fartrack_sparse_research import parameters as _parameters


def parameters(yaml_name: str):
    params = _parameters(yaml_name)
    params.rbsfa_enabled = True
    # q95 of the branch-disagreement signal, estimated only on node-7's
    # calibration partition (sequence identifier modulo 3 equals 1).
    params.rbsfa_threshold = 0.006562486290931702
    params.rbsfa_blend = 0.25
    params.rbsfa_min_inliers = 6
    params.rbsfa_fb_threshold = 1.0
    return params
