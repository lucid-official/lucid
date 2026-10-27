"""``lucid.quantization`` Phase-0 primitives — schemes, dtypes, quant math, STE."""

import numpy as np
import pytest

import lucid
import lucid.quantization as Q


class TestQDtype:
    def test_ranges(self) -> None:
        assert (Q.qint8.quant_min, Q.qint8.quant_max) == (-128, 127)
        assert (Q.quint8.quant_min, Q.quint8.quant_max) == (0, 255)
        assert (Q.qint4.quant_min, Q.qint4.quant_max) == (-8, 7)
        assert Q.qint8.signed and not Q.quint8.signed

    def test_storage_dtypes_exist(self) -> None:
        # Every quantized dtype must name a real lucid storage dtype.
        for qd in (Q.qint8, Q.quint8, Q.qint32, Q.qint4):
            assert isinstance(getattr(lucid, qd.storage), lucid.dtype)

    def test_repr(self) -> None:
        assert repr(Q.qint8) == "lucid.quantization.qint8"


class TestQScheme:
    def test_predicates(self) -> None:
        assert Q.per_channel_symmetric.is_per_channel
        assert not Q.per_tensor_affine.is_per_channel
        assert Q.per_tensor_symmetric.is_symmetric
        assert not Q.per_channel_affine.is_symmetric


class TestQuantizeDequantize:
    def test_round_trip_on_grid(self) -> None:
        # Values that sit exactly on the grid must survive quant→dequant.
        x = lucid.tensor([-1.0, -0.3, 0.0, 0.7, 2.0])
        scale, zp = 0.05, 0.0
        q = Q.quantize(x, scale, zp, Q.qint8)
        assert q.dtype is lucid.int8
        dq = Q.dequantize(q, scale, zp)
        assert np.allclose(dq.numpy(), x.numpy(), atol=1e-6)

    def test_saturation(self) -> None:
        # Values beyond the grid clamp to quant_min / quant_max.
        x = lucid.tensor([-1000.0, 1000.0])
        q = Q.quantize(x, 0.1, 0.0, Q.qint8)
        assert q.numpy().tolist() == [-128, 127]

    def test_affine_zero_point(self) -> None:
        # Unsigned affine: real 0 maps to the zero_point code.
        x = lucid.tensor([0.0])
        q = Q.quantize(x, 0.02, 128.0, Q.quint8)
        assert q.item() == 128


class TestFakeQuantize:
    def test_equals_dequant_of_quant(self) -> None:
        x = lucid.tensor([-1.0, -0.31, 0.0, 0.72, 2.0])
        s, z = 0.05, 0.0
        fq = Q.fake_quantize(x, s, z, Q.qint8.quant_min, Q.qint8.quant_max)
        dq = Q.dequantize(Q.quantize(x, s, z, Q.qint8), s, z)
        assert np.allclose(fq.numpy(), dq.numpy(), atol=1e-6)

    def test_ste_gradient_per_tensor(self) -> None:
        # Gradient is 1 inside the grid, 0 where the value saturates.
        xg = lucid.tensor([-100.0, 0.2, 0.5, 100.0], requires_grad=True)
        Q.fake_quantize(xg, 0.05, 0.0, Q.qint8.quant_min, Q.qint8.quant_max).sum().backward()
        assert xg.grad.numpy().tolist() == [0.0, 1.0, 1.0, 0.0]

    def test_ste_gradient_per_channel(self) -> None:
        w = lucid.tensor([[1.0, 2.0, 300.0], [-400.0, 0.5, 0.1]], requires_grad=True)
        scale = lucid.tensor([0.02, 0.03])
        zp = lucid.tensor([0.0, 0.0])
        Q.fake_quantize(
            w, scale, zp, Q.qint8.quant_min, Q.qint8.quant_max, ch_axis=0
        ).sum().backward()
        # Cells whose code saturates (300, -400) get zero gradient.
        assert w.grad.numpy().tolist() == [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]


class TestPerChannel:
    def test_round_trip(self) -> None:
        w = lucid.tensor([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.1]])
        # Scales chosen so no channel saturates (max |code| < 128).
        scale = lucid.tensor([0.03, 0.05])
        zp = lucid.tensor([0.0, 0.0])
        q = Q.quantize(w, scale, zp, Q.qint8, ch_axis=0)
        dq = Q.dequantize(q, scale, zp, ch_axis=0)
        # Non-saturating reconstruction error is bounded by half a step per channel.
        err = np.abs(dq.numpy() - w.numpy())
        bound = np.array([[0.03 / 2], [0.05 / 2]]) + 1e-6
        assert (err <= bound).all()


class TestCalculateQParams:
    def test_affine_includes_zero(self) -> None:
        # Affine range is extended to include 0 -> min side maps below zp.
        scale, zp = Q.calculate_qparams(-1.0, 3.0, Q.per_tensor_affine, Q.quint8)
        assert scale.item() == pytest.approx(4.0 / 255.0, rel=1e-5)
        assert 0.0 <= zp.item() <= 255.0

    def test_symmetric_zero_point_pinned(self) -> None:
        scale, zp = Q.calculate_qparams(-2.5, 2.0, Q.per_tensor_symmetric, Q.qint8)
        assert zp.item() == 0.0
        assert scale.item() > 0.0

    def test_degenerate_range_floored(self) -> None:
        # A constant tensor (min == max == 0) must not divide by zero.
        scale, _ = Q.calculate_qparams(0.0, 0.0, Q.per_tensor_affine, Q.qint8)
        assert scale.item() > 0.0


class TestSymmetricExtremum:
    """The STE mask is read off ``round(x * (1 / s))``, as the reference does (LCD-393).

    ``s = amax / 127.5``, so a channel's extremum sits on the ``127.5`` rounding
    boundary.  Divided, ``x / s`` lands exactly on it and rounds off the grid;
    multiplied by the float32 reciprocal it lands a hair to either side, and
    which side decides whether the gradient there survives.  ``quantize`` reads
    the same codes, so a converted model meets the rounding it trained against.
    """

    def test_mask_follows_the_reciprocal_rounding(self, device: str) -> None:
        data = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32)
        obs = Q.PerChannelMinMaxObserver(ch_axis=1)
        obs(lucid.tensor(data, device=device))
        scale, zero_point = obs.calculate_qparams()
        x = lucid.tensor(data, device=device, requires_grad=True)
        Q.fake_quantize(x, scale, zero_point, -128, 127, ch_axis=1).sum().backward()
        assert x.grad is not None

        inv = np.float32(1.0) / scale.numpy()
        codes = np.round(data * inv)
        expected = ((codes >= -128) & (codes <= 127)).astype(np.float32)
        assert (expected == 0.0).any(), "no extremum on the boundary — tests nothing"
        np.testing.assert_array_equal(x.grad.numpy(), expected)
        q = Q.quantize(x.detach(), scale, zero_point, Q.qint8, ch_axis=1)
        np.testing.assert_array_equal(q.numpy(), np.clip(codes, -128, 127))
