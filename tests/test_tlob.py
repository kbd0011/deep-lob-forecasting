"""Forward-shape, gradient, and config tests for TLOB (CPU, tiny tensors)."""
import pytest
import torch
from src.models.tlob import TLOB, TLOBBlock


def test_forward_shape_matches_deeplob_contract():
    model = TLOB(dim=40, depth=2, heads=4)
    out = model(torch.randn(4, 1, 100, 40))
    assert out.shape == (4, 3)


def test_single_sample():
    model = TLOB(dim=24, depth=1, heads=4)
    out = model(torch.randn(1, 1, 100, 40))
    assert out.shape == (1, 3)


def test_backward_produces_gradients():
    model = TLOB(dim=24, depth=1, heads=4)
    x = torch.randn(3, 1, 100, 40)
    y = torch.randint(0, 3, (3,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_block_rejects_indivisible_head_count():
    with pytest.raises(ValueError):
        TLOBBlock(dim=40, heads=3)  # 40 not divisible by 3
