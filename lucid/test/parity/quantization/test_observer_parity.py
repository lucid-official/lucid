"""Parity: ``lucid.quantization`` observers vs the reference framework.

Compares the ``(scale, zero_point)`` produced after an identical
calibration pass — the qparam derivation is the contract that must match.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.quantization as Q
from lucid.test._helpers.compare import assert_close


def _ref_observers(ref: Any) -> Any:
    """Import the reference observer classes (ao namespace)."""
    return ref.ao.quantization.observer


@pytest.mark.parity
class TestObserverParity:
    def test_minmax_per_tensor_affine(self, ref: Any) -> None:
        obs = _ref_observers(ref)
        rng = np.random.default_rng(0)
        batches = [rng.standard_normal((4, 8)).astype(np.float32) for _ in range(3)]

        lo = Q.MinMaxObserver(qscheme=Q.per_tensor_affine, qdtype=Q.quint8)
        to = obs.MinMaxObserver(dtype=ref.quint8, qscheme=ref.per_tensor_affine)
        for b in batches:
            lo(lucid.tensor(b.copy()))
            to(ref.tensor(b.copy()))
        ls, lz = lo.calculate_qparams()
        ts, tz = to.calculate_qparams()
        # Per-tensor qparams: lucid keeps a scalar, the reference a shape-(1,)
        # tensor — compare the scalar values.
        assert ls.item() == pytest.approx(ts.item(), abs=1e-6)
        assert abs(lz.item() - tz.item()) <= 1

    def test_per_channel_symmetric(self, ref: Any) -> None:
        obs = _ref_observers(ref)
        rng = np.random.default_rng(1)
        w = rng.standard_normal((6, 10)).astype(np.float32)

        lp = Q.PerChannelMinMaxObserver(ch_axis=0, qscheme=Q.per_channel_symmetric, qdtype=Q.qint8)
        tp = obs.PerChannelMinMaxObserver(
            ch_axis=0, dtype=ref.qint8, qscheme=ref.per_channel_symmetric
        )
        lp(lucid.tensor(w.copy()))
        tp(ref.tensor(w.copy()))
        ls, _ = lp.calculate_qparams()
        ts, _ = tp.calculate_qparams()
        assert_close(ls, ts, atol=1e-6)

    def test_moving_average(self, ref: Any) -> None:
        obs = _ref_observers(ref)
        rng = np.random.default_rng(2)
        batches = [rng.standard_normal((4, 8)).astype(np.float32) for _ in range(4)]

        le = Q.MovingAverageMinMaxObserver(
            averaging_constant=0.1, qscheme=Q.per_tensor_affine, qdtype=Q.quint8
        )
        te = obs.MovingAverageMinMaxObserver(
            averaging_constant=0.1, dtype=ref.quint8, qscheme=ref.per_tensor_affine
        )
        for b in batches:
            le(lucid.tensor(b.copy()))
            te(ref.tensor(b.copy()))
        ls, _ = le.calculate_qparams()
        ts, _ = te.calculate_qparams()
        assert ls.item() == pytest.approx(ts.item(), abs=1e-6)
