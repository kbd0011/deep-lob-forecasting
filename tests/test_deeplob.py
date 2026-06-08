"""Forward-shape and gradient tests for the DeepLOB model (CPU, tiny tensors)."""
import torch
from src.models.deeplob import DeepLOB


def test_forward_shape():
    model = DeepLOB()
    x = torch.randn(8, 1, 100, 40)  # (batch, 1, time, 40 LOB features)
    out = model(x)
    assert out.shape == (8, 3)      # logits over {down, stationary, up}


def test_handles_single_sample():
    model = DeepLOB()
    out = model(torch.randn(1, 1, 100, 40))
    assert out.shape == (1, 3)


def test_backward_produces_gradients():
    model = DeepLOB()
    x = torch.randn(4, 1, 100, 40)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
