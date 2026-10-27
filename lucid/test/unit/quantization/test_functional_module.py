"""``lucid.quantization`` Phase-3 — FloatFunctional / QFunctional (residuals)."""

import contextlib
from collections.abc import Iterator

import numpy as np

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.nn.quantized as nnq
import lucid.quantization as Q


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
    prev = backends.quantized.engine
    backends.quantized.engine = "reference"
    try:
        yield
    finally:
        backends.quantized.engine = prev


class TestFloatFunctional:
    def test_passthrough_before_prepare(self) -> None:
        lucid.manual_seed(0)
        ff = nnq.FloatFunctional()
        a, b = lucid.randn(4, 4), lucid.randn(4, 4)
        assert np.allclose(ff.add(a, b).numpy(), (a + b).numpy())
        assert np.allclose(ff.mul(a, b).numpy(), (a * b).numpy())
        assert np.allclose(ff.add_relu(a, b).numpy(), F.relu(a + b).numpy())
        assert np.allclose(ff.cat([a, b], dim=0).numpy(), lucid.cat([a, b], dim=0).numpy())

    def test_scalar_ops(self) -> None:
        lucid.manual_seed(1)
        ff = nnq.FloatFunctional()
        a = lucid.randn(3, 3)
        assert np.allclose(ff.add_scalar(a, 2.0).numpy(), (a + 2.0).numpy())
        assert np.allclose(ff.mul_scalar(a, 3.0).numpy(), (a * 3.0).numpy())


class TestQFunctionalResidual:
    class _ResBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nnq.QuantStub()
            self.fc1 = nn.Linear(64, 64)
            self.fc2 = nn.Linear(64, 64)
            self.skip = nnq.FloatFunctional()
            self.dq = nnq.DeQuantStub()

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            x = self.q(x)
            y = self.fc2(F.relu(self.fc1(x)))
            return self.dq(self.skip.add(x, y))

    def _run(self) -> float:
        lucid.manual_seed(0)
        m = self._ResBlock().eval()
        x = lucid.randn(16, 64)
        yf = m(x).numpy()
        prepared = Q.prepare(m, Q.get_default_qconfig_mapping())
        for _ in range(10):
            prepared(lucid.randn(16, 64))
        qm = Q.convert(prepared)
        assert isinstance(qm.skip, nnq.QFunctional)  # residual add is quantized
        yq = qm(x).numpy()
        return float(np.abs(yf - yq).max() / (np.abs(yf).max() + 1e-9))

    def test_residual_reference_path(self) -> None:
        with _reference_engine():
            err = self._run()
        assert err < 0.05

    def test_residual_default_path(self) -> None:
        # auto engine (MLX-routed Linear on Metal, or dequant elsewhere) — the
        # QuantizedLinearMLX device round-trip keeps the skip-add on one device.
        err = self._run()
        assert err < 0.05
