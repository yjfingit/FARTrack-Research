import torch

from lib.test.tracker.fartrack_sparse_cf import AcceptedTemplatePool, CounterfactualAgreementVerifier


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


def test_accepted_pool_survives_rejections_then_later_acceptance():
    anchor = torch.zeros((1, 3, 112, 112))
    pool = AcceptedTemplatePool(anchor, num_template=5)
    first = torch.ones_like(anchor)
    masks = [torch.ones((1, 49), dtype=torch.bool) for _ in range(4)]
    pool.accept(first, masks)
    # Rejections call no pool mutation, regardless of their absolute frame ids.
    for _ in range(20):
        templates, attention_mask = pool.fartrack_inputs()
        assert len(templates) == 5
        assert attention_mask.shape == (1, 445, 445)
        assert torch.equal(templates[-1], first)
    later = torch.full_like(anchor, 2)
    pool.accept(later, masks)
    templates, attention_mask = pool.fartrack_inputs()
    assert len(templates) == 5
    assert attention_mask.shape == (1, 445, 445)
    assert torch.equal(templates[-2], first)
    assert torch.equal(templates[-1], later)
