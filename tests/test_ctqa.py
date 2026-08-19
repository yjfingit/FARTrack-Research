"""CPU unit checks for the causal trajectory query adapter."""

import torch

from lib.models.fartrack_sparse.base_backbone import CausalTrajectoryQueryAdapter


def test_ctqa_is_a_zero_effect_but_trainable_adapter():
    torch.manual_seed(7)
    embedding = torch.nn.Embedding(605, 16)
    adapter = CausalTrajectoryQueryAdapter(embed_dim=16, hidden_dim=12)
    history = torch.randint(0, 600, (2, 12))

    bias = adapter(history, embedding)
    assert bias.shape == (2, 4, 16)
    assert torch.count_nonzero(bias) == 0

    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    loss = (bias * torch.randn_like(bias)).sum()
    loss.backward()
    assert adapter.output.weight.grad is not None
    assert torch.count_nonzero(adapter.output.weight.grad) > 0
    optimizer.step()

    assert torch.count_nonzero(adapter(history, embedding)) > 0


def test_ctqa_rejects_ambiguous_history_length():
    adapter = CausalTrajectoryQueryAdapter(embed_dim=8, hidden_dim=8)
    embedding = torch.nn.Embedding(20, 8)
    try:
        adapter(torch.ones(1, 11, dtype=torch.long), embedding)
    except ValueError as error:
        assert "Expected trajectory tokens" in str(error)
    else:
        raise AssertionError("Malformed trajectory history must not be silently accepted")
