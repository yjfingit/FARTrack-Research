"""Inference parameters for trained CTQA checkpoints on B_dev only."""

import copy
import os

from lib.config.fartrack_sparse.config import cfg, update_config_from_file
from lib.test.utils import TrackerParams


SUPPORTED_CONFIG = "fartrack_sparse_224_ctqa_bdev"


def parameters(yaml_name: str):
    if yaml_name != SUPPORTED_CONFIG:
        raise ValueError(f"CTQA parameters only support {SUPPORTED_CONFIG}, got {yaml_name}")
    checkpoint = os.environ.get("FARTRACK_CTQA_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("Set FARTRACK_CTQA_CHECKPOINT to a trained CTQA checkpoint")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    params = TrackerParams()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    yaml_file = os.path.join(project_root, 'experiments', 'fartrack_sparse', f'{yaml_name}.yaml')
    # Evaluation processes commonly instantiate a tracker once, but isolate
    # config mutation here so a baseline parameter module cannot inherit CTQA.
    params.cfg = copy.deepcopy(cfg)
    update_config_from_file(yaml_file, base_cfg=params.cfg)
    params.template_factor = params.cfg.TEST.TEMPLATE_FACTOR
    params.template_size = params.cfg.TEST.TEMPLATE_SIZE
    params.search_factor = params.cfg.TEST.SEARCH_FACTOR
    params.search_size = params.cfg.TEST.SEARCH_SIZE
    params.checkpoint = checkpoint
    params.save_all_boxes = False
    return params
