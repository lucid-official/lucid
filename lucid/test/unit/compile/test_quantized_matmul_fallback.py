"""Compile × quantized-GEMM correctness regression.

The fused low-precision GEMM (``quantized_matmul``) is a Pattern-B op that calls
MLX directly with no autograd/tracer wiring.  It was therefore invisible to the
compile tracer, so a compiled quantized layer baked its *activation-dependent*
result as a trace-time constant: ``compiled(x2)`` silently returned the answer
for ``x1`` (maxdiff 0.0 between two different inputs, ~2.8 vs eager).

Fixed 2026-07-13 by recording the op in the active tracer (``on_op_enter`` +
``on_op_io`` with a non-empty input list).  It has no MPSGraph emitter, so the
builder now aborts the graph and the whole signature falls back cleanly to
eager — correct output.  ``dequantize`` is deliberately left untraced: its
inputs are constant weights, so baking is harmless and it stays compilable.
"""

import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.quantized as nnq
from lucid.quantization import _qgemm


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:
        return False


# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = [
    pytest.mark.skipif(
        not (backends.quantized.available and _metal_ok()),
        reason="engine lacks quantized ops or Metal unavailable",
    ),
    pytest.mark.usefixtures("inference"),
]


def _maxdiff(a: object, b: object) -> float:
    return float((a - b).abs().max().item())


def test_compiled_quantized_matmul_reacts_to_new_input() -> None:
    """The regression: two different inputs must give two different outputs."""
    lucid.manual_seed(0)
    qlin = nnq.QuantizedLinearMLX.from_float(nn.Linear(512, 256).to("metal")).to("metal")
    x1 = lucid.randn(4, 512, device="metal")
    x2 = lucid.randn(4, 512, device="metal")

    cq = lucid.compile(qlin)
    c1, c2 = cq(x1), cq(x2)

    # Before the fix c1 == c2 (both baked to x1's answer).
    assert _maxdiff(c1, c2) > 1e-3
    # And each compiled output matches the eager reference for its own input.
    assert _maxdiff(c2, qlin(x2)) < 1e-4
    assert _maxdiff(c1, qlin(x1)) < 1e-4


def test_compiled_quantized_matmul_falls_back_to_eager() -> None:
    """No emitter for the fused GEMM → the signature is marked eager-only."""
    lucid.manual_seed(1)
    qlin = nnq.QuantizedLinearMLX.from_float(nn.Linear(256, 128).to("metal")).to("metal")
    cq = lucid.compile(qlin)
    cq(lucid.randn(8, 256, device="metal"))

    assert len(list(cq._eager_only.snapshot())) == 1  # aborted → eager
    assert len(cq._cache) == 0  # no real executable cached


def test_multilayer_quantized_mlp_correct_under_compile() -> None:
    """Every quantized layer falls back, but the composed model stays correct."""

    class QMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nnq.QuantizedLinearMLX.from_float(
                nn.Linear(256, 512).to("metal"), relu=True
            ).to("metal")
            self.l2 = nnq.QuantizedLinearMLX.from_float(nn.Linear(512, 128).to("metal")).to("metal")

        def forward(self, x: object) -> object:
            return self.l2(self.l1(x))

    lucid.manual_seed(2)
    m = QMLP().to("metal")
    xa = lucid.randn(8, 256, device="metal")
    xb = lucid.randn(8, 256, device="metal")
    cm = lucid.compile(m)

    assert _maxdiff(cm(xa), cm(xb)) > 1e-3
    assert _maxdiff(cm(xb), m(xb)) < 1e-4


def test_dequantize_path_still_compiles() -> None:
    """Constant-weight dequantize must remain compilable (not forced to eager)."""

    class DeqLin(nn.Module):
        def __init__(self, lin: nn.Linear) -> None:
            super().__init__()
            wq, sc, bi = _qgemm.quantize(lin.weight, group_size=64, bits=8)
            self.register_buffer("wq", wq)
            self.register_buffer("sc", sc)
            self.register_buffer("bi", bi)
            self.register_buffer("b", lin.bias.data)

        def forward(self, x: object) -> object:
            w = _qgemm.dequantize(self.wq, self.sc, self.bi, group_size=64, bits=8)
            return x @ w.mT + self.b

    lucid.manual_seed(3)
    m = DeqLin(nn.Linear(512, 256).to("metal")).to("metal")
    x1 = lucid.randn(4, 512, device="metal")
    x2 = lucid.randn(4, 512, device="metal")
    cm = lucid.compile(m)
    c1, c2 = cm(x1), cm(x2)

    assert _maxdiff(c1, c2) > 1e-3
    assert _maxdiff(c2, m(x2)) < 1e-4
    assert len(cm._cache) == 1  # a real executable was built
    assert len(list(cm._eager_only.snapshot())) == 0
