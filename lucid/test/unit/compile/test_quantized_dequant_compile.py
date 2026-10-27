"""Compile × static-quant (dequantize path) — the int8 weight must compile.

A converted static-quant ``Linear`` / ``Conv`` (sidecar design: int8 weight +
``dequantize`` → float ``F.linear`` each forward) failed to compile on Metal and
silently fell back to eager.  Two gaps, both fixed 2026-07-13:

1. The int8 weight is a graph feed that ``dequantize`` immediately casts to
   float, but ``I8`` was missing from the compile-path dtype maps (MpsBuilder /
   CompiledExecutable feed placeholders + the ``astype`` emitter) → the int8
   feed threw "dtype not supported".  Added ``I8`` / ``I16`` → ``MPSDataTypeInt8``
   / ``…Int16``.
2. Once int8 fed, the graph was still a *mixed-device* trace: a HistogramObserver
   derives the activation range from host floats, so ``convert`` baked CPU
   activation qparams into an otherwise-Metal module.  Fixed by baking qparams on
   the module's device (``nn.quantized._utils._module_device``).

Note: a group-size-divisible Linear on Metal instead routes to the *fused* MLX
``quantized_matmul`` path, which has no emitter and correctly stays eager (see
``test_quantized_matmul_fallback.py``); this file exercises the dequantize path
(non-divisible ``in_features`` / Conv) that genuinely compiles.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.quantization as Q


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:
        return False


# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = [
    pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable"),
    pytest.mark.usefixtures("inference"),
]


def _prep_convert(model: nn.Module, x: lucid.Tensor) -> nn.Module:
    model = model.to("metal").eval()
    qm = Q.prepare(model, Q.get_default_qconfig()).to("metal")
    for _ in range(5):
        qm(x)
    return Q.convert(qm)


def test_dequant_linear_compiles_and_reacts_to_input() -> None:
    """in_features=48 (not group-divisible) → dequantize path → must compile."""
    lucid.manual_seed(0)
    x = lucid.randn(8, 48, device="metal")
    cm = _prep_convert(nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 12)), x)
    assert type(cm[0]).__name__ == "Linear"  # dequantize sidecar, not the MLX GEMM

    comp = lucid.compile(cm)
    x2 = lucid.randn(8, 48, device="metal")
    c1, c2 = comp(x), comp(x2)

    assert len(list(comp._eager_only.snapshot())) == 0  # compiled, not eager
    assert len(comp._cache) == 1
    assert float((c1 - c2).abs().max().item()) > 1e-4  # reacts to new input
    assert float((comp(x) - cm(x)).abs().max().item()) < 1e-4  # matches eager


def test_dequant_conv_compiles() -> None:
    """Quantized Conv2d always uses the dequantize path — it must compile too."""
    lucid.manual_seed(1)
    x = lucid.randn(2, 3, 16, 16, device="metal")
    cm = _prep_convert(nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()), x)

    comp = lucid.compile(cm)
    comp(x)
    comp(lucid.randn(2, 3, 16, 16, device="metal"))
    assert len(list(comp._eager_only.snapshot())) == 0
    assert len(comp._cache) == 1
    assert float((comp(x) - cm(x)).abs().max().item()) < 1e-4


def test_converted_module_is_single_device() -> None:
    """Every buffer of a Metal-converted quantized module lives on Metal (no
    CPU-stranded qparam that would strand the trace across two devices)."""
    lucid.manual_seed(2)
    x = lucid.randn(4, 48, device="metal")
    cm = _prep_convert(nn.Sequential(nn.Linear(48, 16)), x)
    assert {str(b.device) for _, b in cm.named_buffers()} == {"device('metal')"}
