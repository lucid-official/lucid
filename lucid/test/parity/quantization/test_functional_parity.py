"""Parity: ``lucid.quantization`` primitives vs the reference framework.

Random inputs are used deliberately: with a generic scale, ``x / scale``
never lands exactly on an ``N.5`` tie (a measure-zero event in float), so
every rounding mode agrees and the comparison is exact up to ``1e-4``.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.quantization as Q
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestQuantizeParity:
    def test_quantize_int_codes_per_tensor(self, ref: Any) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 16)).astype(np.float32)
        scale, zp = 0.05, 0
        lc = Q.quantize(lucid.tensor(data.copy()), scale, float(zp), Q.qint8)
        rc = ref.quantize_per_tensor(ref.tensor(data.copy()), scale, zp, ref.qint8).int_repr()
        assert np.array_equal(lc.numpy().astype(np.int64), rc.numpy().astype(np.int64))

    def test_dequantize_per_tensor(self, ref: Any) -> None:
        rng = np.random.default_rng(1)
        data = rng.standard_normal((4, 8)).astype(np.float32)
        scale, zp = 0.05, 0
        q = Q.quantize(lucid.tensor(data.copy()), scale, float(zp), Q.qint8)
        l = Q.dequantize(q, scale, float(zp))
        r = ref.quantize_per_tensor(ref.tensor(data.copy()), scale, zp, ref.qint8).dequantize()
        assert_close(l, r, atol=1e-4)


@pytest.mark.parity
class TestFakeQuantizeParity:
    def test_per_tensor(self, ref: Any) -> None:
        rng = np.random.default_rng(2)
        data = rng.standard_normal((3, 16)).astype(np.float32)
        scale, zp, qmin, qmax = 0.037, 0, -128, 127
        l = Q.fake_quantize(lucid.tensor(data.copy()), scale, float(zp), qmin, qmax)
        r = ref.fake_quantize_per_tensor_affine(ref.tensor(data.copy()), scale, zp, qmin, qmax)
        assert_close(l, r, atol=1e-4)

    def test_per_tensor_unsigned_zero_point(self, ref: Any) -> None:
        rng = np.random.default_rng(3)
        data = rng.standard_normal((5, 7)).astype(np.float32)
        scale, zp, qmin, qmax = 0.02, 128, 0, 255
        l = Q.fake_quantize(lucid.tensor(data.copy()), scale, float(zp), qmin, qmax)
        r = ref.fake_quantize_per_tensor_affine(ref.tensor(data.copy()), scale, zp, qmin, qmax)
        assert_close(l, r, atol=1e-4)

    def test_per_channel(self, ref: Any) -> None:
        rng = np.random.default_rng(4)
        data = rng.standard_normal((4, 8)).astype(np.float32)
        scale_np = np.array([0.02, 0.03, 0.05, 0.017], dtype=np.float32)
        # Reference per-channel fake-quant rejects int64 zero-points; use int32.
        zp_np = np.zeros(4, dtype=np.int32)
        qmin, qmax = -128, 127
        l = Q.fake_quantize(
            lucid.tensor(data.copy()),
            lucid.tensor(scale_np.copy()),
            lucid.tensor(zp_np.astype(np.float32).copy()),
            qmin,
            qmax,
            ch_axis=0,
        )
        r = ref.fake_quantize_per_channel_affine(
            ref.tensor(data.copy()),
            ref.tensor(scale_np.copy()),
            ref.tensor(zp_np.copy()),
            0,
            qmin,
            qmax,
        )
        assert_close(l, r, atol=1e-4)

    def test_ste_gradient_matches_reference(self, ref: Any) -> None:
        rng = np.random.default_rng(5)
        # Scale by 3 so a good fraction of values saturate — exercises the mask.
        data = (rng.standard_normal((2, 10)).astype(np.float32)) * 3.0
        scale, zp, qmin, qmax = 0.05, 0, -128, 127

        xg = lucid.tensor(data.copy(), requires_grad=True)
        Q.fake_quantize(xg, scale, float(zp), qmin, qmax).sum().backward()

        xr = ref.tensor(data.copy(), requires_grad=True)
        ref.fake_quantize_per_tensor_affine(xr, scale, zp, qmin, qmax).sum().backward()

        assert_close(xg.grad, xr.grad, atol=1e-6)
