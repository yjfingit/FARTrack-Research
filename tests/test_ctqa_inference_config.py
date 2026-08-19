"""No-dataset checks for the CTQA inference configuration and entrypoint."""

from copy import deepcopy
from pathlib import Path

from lib.config.fartrack_sparse.config import cfg, update_config_from_file
from lib.test.tracker.fartrack_sparse_ctqa import get_tracker_class
from lib.test.tracker.fartrack_sparse import FARTrackSparse


def test_ctqa_bdev_configuration_enables_only_the_adapter():
    local = deepcopy(cfg)
    config = Path(__file__).resolve().parents[1] / "experiments/fartrack_sparse/fartrack_sparse_224_ctqa_bdev.yaml"
    update_config_from_file(str(config), base_cfg=local)
    assert local.MODEL.TRAJECTORY_QUERY_ADAPTER.ENABLED is True
    assert local.MODEL.TRAJECTORY_QUERY_ADAPTER.HIDDEN_DIM == 192
    assert local.TEST.SEARCH_SIZE == 224
    assert local.TEST.TEMPLATE_SIZE == 112


def test_ctqa_tracker_entrypoint_reuses_audited_sparse_tracker():
    assert get_tracker_class() is FARTrackSparse
