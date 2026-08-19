"""Dedicated environment for CTQA training on GOT-10k training only.

Before launching a CTQA run, copy this file to ``lib/train/admin/local.py``
in the isolated experiment worktree.  It deliberately exposes no validation,
LaSOT, or test paths, so the training configuration cannot silently consume a
development or held-out evaluation split.
"""


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = "/root/autodl-tmp/experiment/.research-assets/runs/ctqa_got10k_train"
        self.tensorboard_dir = self.workspace_dir + "/tensorboard"
        self.pretrained_networks = self.workspace_dir + "/pretrained_networks"
        self.got10k_dir = "/root/autodl-tmp/experiment/.research-assets/data/got10k/train"
