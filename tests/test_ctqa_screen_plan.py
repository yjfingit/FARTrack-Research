"""Prevent unintentional divergence between CTQA and matched control screen."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments" / "fartrack_sparse"


def test_matched_screen_configs_only_differ_at_ctqa_feature_flag():
    control = yaml.safe_load((EXPERIMENTS / "fartrack_sparse_224_continue_got10k_screen.yaml").read_text())
    ctqa = yaml.safe_load((EXPERIMENTS / "fartrack_sparse_224_ctqa_got10k_screen.yaml").read_text())
    assert control["DATA"]["TRAIN"]["DATASETS_NAME"] == ["GOT10K_train_full"]
    assert control["DATA"]["VAL"]["DATASETS_NAME"] == []
    assert ctqa["DATA"]["TRAIN"]["DATASETS_NAME"] == ["GOT10K_train_full"]
    assert ctqa["DATA"]["VAL"]["DATASETS_NAME"] == []

    control_flag = control["MODEL"]["TRAJECTORY_QUERY_ADAPTER"].pop("ENABLED")
    ctqa_flag = ctqa["MODEL"]["TRAJECTORY_QUERY_ADAPTER"].pop("ENABLED")
    assert control_flag is False
    assert ctqa_flag is True
    assert control == ctqa
