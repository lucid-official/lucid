"""``nn.FusedLinear`` exports to Core ML (LCD-450).

Export traces under ``no_grad``, so a ``FusedLinear`` records the fused
inference kernel — ``_fused_linear_relu`` / ``_fused_linear_gelu`` — which
had no MIL translation and refused the export. It goes out as MIL's
``linear`` followed by ``relu``, or ``gelu`` in the tanh mode the kernel
computes.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


class _Mlp(nn.Module):
    """A transformer feed-forward block made of the two fused layers."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.FusedLinear(8, 32, activation="gelu")
        self.down = nn.FusedLinear(32, 4, activation="relu")

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.down(self.up(x))


def test_the_trace_records_the_fused_kernels() -> None:
    model = _Mlp().eval()
    x = lucid.randn(2, 5, 8)
    with lucid.no_grad(), lucid.compile._tracing(watch=[x]) as tracer:
        model(x)
    assert [op.name for op in tracer.graph.ops] == ["_fused_linear_gelu", "_fused_linear_relu"]


def test_a_fused_linear_block_predicts_as_eager(tmp_path: object) -> None:
    lucid.manual_seed(0)
    model = _Mlp().eval()
    example = lucid.randn(2, 5, 8)
    exported = cml.export(model, example, f"{tmp_path}/mlp.mlpackage")
    for x in (example, lucid.randn(2, 5, 8) * 3):
        with lucid.no_grad():
            want = model(x)
        got = exported.predict(x)
        assert isinstance(got, lucid.Tensor)
        assert got.shape == want.shape == (2, 5, 4)
        scale = max(1.0, float(want.abs().max().item()))
        assert float((got - want).abs().max().item()) <= 1e-4 * scale
