"""Shape sanity checks for custom attention modules."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from attention_modules import (  # noqa: E402
    AttentionGate,
    CALiteSpatialGate,
    CESALite,
    LowFPCBAMLite,
    NAMAttention,
    TripletAttention,
)


def check_single_input_module(module_cls, shape, *args):
    x = torch.randn(*shape)
    module = module_cls(shape[1], *args)
    module.eval()
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape, f"{module_cls.__name__}: expected {x.shape}, got {y.shape}"


def check_attention_gate(shape):
    x = torch.randn(*shape)
    gate = torch.randn(shape[0], max(shape[1] * 2, 8), max(shape[2] // 2, 1), max(shape[3] // 2, 1))
    module = AttentionGate(shape[1], gate.shape[1])
    module.eval()
    with torch.no_grad():
        y = module([x, gate])
    assert y.shape == x.shape, f"AttentionGate: expected {x.shape}, got {y.shape}"


def main():
    dummy_shapes = [
        (2, 64, 80, 80),
        (2, 128, 40, 40),
        (2, 256, 20, 20),
    ]

    single_modules = [
        CESALite,
        TripletAttention,
        CALiteSpatialGate,
        NAMAttention,
        LowFPCBAMLite,
    ]

    for shape in dummy_shapes:
        for module_cls in single_modules:
            check_single_input_module(module_cls, shape)
        check_attention_gate(shape)

    print("All custom attention shape sanity checks passed.")


if __name__ == "__main__":
    main()
