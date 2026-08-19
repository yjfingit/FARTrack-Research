import torch

from lib.test.tracker.fartrack_sparse_cf import CounterfactualAgreementVerifier


def test_retain_only_preserves_requested_template_keys():
    verifier = CounterfactualAgreementVerifier(num_template=5)
    base = torch.ones((1, 445, 445), dtype=torch.bool)
    masked = verifier.retain_only(base, (0, 4))
    assert masked[:, :, :49].all()
    assert not masked[:, :, 49:196].any()
    assert masked[:, :, 196:245].all()
    assert masked[:, :, 245:].all()


def test_normalized_l1_is_zero_for_identical_predictions():
    box = torch.tensor([0.1, 0.2, 0.5, 0.6])
    assert CounterfactualAgreementVerifier.normalized_l1(box, box).item() == 0.0
