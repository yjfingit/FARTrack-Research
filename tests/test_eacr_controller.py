import torch

from lib.test.tracker.fartrack_sparse_eacr import EACRController


def test_eacr_trigger_and_one_frame_reset():
    controller = EACRController(0.47618, 40)
    controller.set_alarm(torch.tensor(0.48), 39)
    assert not controller.consume_alarm(torch.device("cpu")).item()
    controller.set_alarm(torch.tensor(0.48), 40)
    assert controller.consume_alarm(torch.device("cpu")).item()
    assert not controller.consume_alarm(torch.device("cpu")).item()


def test_eacr_disabled_never_triggers():
    controller = EACRController(0.47618, 40, enabled=False)
    controller.set_alarm(torch.tensor(0.99), 999)
    assert not controller.consume_alarm(torch.device("cpu")).item()
