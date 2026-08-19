from lib.test.parameter.fartrack_sparse_research import parameters as baseline_parameters


def parameters(yaml_name: str):
    params = baseline_parameters(yaml_name)
    params.template_sampling_method = "exponential"
    return params
