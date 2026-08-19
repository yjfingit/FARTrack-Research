"""Research-only FARTrackSparse parameters with an immutable shared checkpoint."""

import os

from lib.config.fartrack_sparse.config import cfg, update_config_from_file
from lib.test.utils import TrackerParams


def parameters(yaml_name: str):
    params = TrackerParams()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    yaml_file = os.path.join(project_root, 'experiments', 'fartrack_sparse', f'{yaml_name}.yaml')
    update_config_from_file(yaml_file)
    # PRETRAIN_PTH is only used by the training initializer. Inference loads
    # the frozen released checkpoint below and must not depend on the author's
    # private teacher path embedded in the YAML.
    cfg.MODEL.PRETRAIN_PTH = ''
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = '/root/autodl-tmp/experiment/.research-assets/checkpoints/FARTrackSparse_ep0015.pth.tar'
    params.save_all_boxes = False
    return params
