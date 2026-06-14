import pytest


torch = pytest.importorskip("torch", reason="PyTorch is required for loss tests")

from cvio_asl_ldam.losses.asl_ldam import ASLLDAMLoss


def test_asl_ldam_returns_finite_scalar():
    criterion = ASLLDAMLoss()
    logits = torch.randn(4, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    loss = criterion(logits, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
