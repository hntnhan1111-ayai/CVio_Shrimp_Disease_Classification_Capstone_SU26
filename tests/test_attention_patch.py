import pytest


torch = pytest.importorskip("torch", reason="PyTorch is required for attention tests")

from cvio_asl_ldam.attention.simam_dcfr import SimAMDCFR


def test_simam_dcfr_preserves_shape():
    module = SimAMDCFR(channels=16)
    tensor = torch.randn(2, 16, 8, 8)
    output = module(tensor)
    assert output.shape == tensor.shape
    assert torch.isfinite(output).all()
