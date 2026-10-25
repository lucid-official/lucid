"""Reduction ops — sum / mean / max / min / argmax / cumsum / std / var / etc."""

import numpy as np
import pytest

import lucid

# ── basic full-tensor reductions ────────────────────────────────────────


class TestSum:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert lucid.sum(t).item() == 10.0

    def test_along_dim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        np.testing.assert_array_equal(lucid.sum(t, dim=0).numpy(), [4.0, 6.0])
        np.testing.assert_array_equal(lucid.sum(t, dim=1).numpy(), [3.0, 7.0])

    def test_keepdim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        out = lucid.sum(t, dim=0, keepdim=True)
        assert out.shape == (1, 2)


class TestMean:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([2.0, 4.0, 6.0], device=device)
        assert lucid.mean(t).item() == 4.0

    def test_along_dim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        np.testing.assert_array_equal(lucid.mean(t, dim=0).numpy(), [2.0, 3.0])


class TestMax:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 5.0, 2.0], device=device)
        assert lucid.max(t).item() == 5.0

    def test_along_dim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 4.0], [3.0, 2.0]], device=device)
        vals, idx = lucid.max(t, dim=0)
        np.testing.assert_array_equal(vals.numpy(), [3.0, 4.0])
        np.testing.assert_array_equal(idx.numpy(), [1, 0])
        assert vals is lucid.max(t, dim=0).values or np.array_equal(
            lucid.max(t, dim=0).values.numpy(), [3.0, 4.0]
        )
        np.testing.assert_array_equal(lucid.max(t, dim=0).indices.numpy(), [1, 0])


class TestMin:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 5.0, 2.0], device=device)
        assert lucid.min(t).item() == 1.0

    def test_along_dim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 4.0], [3.0, 2.0]], device=device)
        vals, idx = lucid.min(t, dim=0)
        np.testing.assert_array_equal(vals.numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(idx.numpy(), [0, 1])
        np.testing.assert_array_equal(lucid.min(t, dim=0).values.numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(lucid.min(t, dim=0).indices.numpy(), [0, 1])


class TestProd:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert lucid.prod(t).item() == 24.0


class TestArgmax:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 5.0, 2.0], device=device)
        assert int(lucid.argmax(t).item()) == 2

    def test_along_dim(self, device: str) -> None:
        t = lucid.tensor([[1.0, 4.0], [3.0, 2.0]], device=device)
        out = lucid.argmax(t, dim=0).numpy()
        np.testing.assert_array_equal(out, [1, 0])


class TestArgmin:
    def test_full(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 5.0, 2.0], device=device)
        assert int(lucid.argmin(t).item()) == 1


class TestCumsum:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        np.testing.assert_array_equal(
            lucid.cumsum(t, dim=0).numpy(), [1.0, 3.0, 6.0, 10.0]
        )

    def test_int_promotes_to_int64_no_overflow(self, device: str) -> None:
        # Like sum, an integer cumsum promotes to int64 so it doesn't silently
        # overflow int32 mid-accumulation.
        t = lucid.tensor([2_000_000_000] * 3, dtype=lucid.int32, device=device)
        out = lucid.cumsum(t, dim=0)
        assert out.dtype == lucid.int64
        np.testing.assert_array_equal(
            out.numpy(), [2_000_000_000, 4_000_000_000, 6_000_000_000]
        )


class TestCumprod:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        np.testing.assert_array_equal(
            lucid.cumprod(t, dim=0).numpy(), [1.0, 2.0, 6.0, 24.0]
        )

    def test_int_promotes_to_int64(self, device: str) -> None:
        t = lucid.tensor([100, 100, 100], dtype=lucid.int32, device=device)
        out = lucid.cumprod(t, dim=0)
        assert out.dtype == lucid.int64
        np.testing.assert_array_equal(out.numpy(), [100, 10_000, 1_000_000])


class TestStdVar:
    def test_var(self, device: str) -> None:
        # population variance of [0, 1, 2] = 2/3.
        t = lucid.tensor([0.0, 1.0, 2.0], device=device)
        v = lucid.var(t, correction=0).item()
        assert abs(v - 2.0 / 3.0) < 1e-5

    def test_std(self, device: str) -> None:
        t = lucid.tensor([0.0, 1.0, 2.0], device=device)
        s = lucid.std(t, correction=0).item()
        assert abs(s - np.sqrt(2.0 / 3.0)) < 1e-5

    def test_var_corrected(self, device: str) -> None:
        # Bessel-corrected variance of [0, 1, 2] = 1.
        t = lucid.tensor([0.0, 1.0, 2.0], device=device)
        v = lucid.var(t, correction=1).item()
        assert abs(v - 1.0) < 1e-5


class TestTrace:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        assert lucid.trace(t).item() == 5.0


class TestLogsumexp:
    def test_known(self, device: str) -> None:
        # logsumexp([0, 0, 0]) = log 3.
        t = lucid.tensor([0.0, 0.0, 0.0], device=device)
        assert abs(lucid.logsumexp(t, dim=0).item() - np.log(3.0)) < 1e-5


# ── nan-safe reductions ─────────────────────────────────────────────────


class TestNanReductions:
    def test_nansum(self, device: str) -> None:
        t = lucid.tensor([1.0, float("nan"), 3.0], device=device)
        assert lucid.nansum(t).item() == 4.0

    def test_nanmean(self, device: str) -> None:
        t = lucid.tensor([2.0, float("nan"), 4.0], device=device)
        assert lucid.nanmean(t).item() == 3.0


# ── shape coverage across reduction axes ────────────────────────────────


@pytest.mark.parametrize("shape", [(2, 3, 4), (5,), (1, 1, 1)])
@pytest.mark.parametrize("dim", [0, -1])
def test_sum_shape_inference(shape: tuple[int, ...], dim: int, device: str) -> None:
    t = lucid.zeros(*shape, device=device)
    out = lucid.sum(t, dim=dim)
    expected_shape = list(shape)
    expected_shape.pop(dim if dim >= 0 else dim + len(shape))
    assert out.shape == tuple(expected_shape)


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((300, 16), 1),  # row_sum / row_var path (O == 1, M >= 256, I <= 4096)
        ((16, 64), 0),  # krk_sum / krk_var path (O == 1, M == 16, I == 64)
        ((4, 8, 16, 16), (0, 2, 3)),  # OMI channel reduction (O == 4, M == 8, I == 256)
    ],
)
def test_half_reductions_accumulate_in_float32(
    device: str, dtype: lucid.Dtype, shape: tuple[int, ...], dim: int | tuple[int, ...]
) -> None:
    rng = np.random.default_rng(7)
    x_np = rng.uniform(0.5, 1.5, size=shape).astype(np.float32)
    x = lucid.tensor(x_np, dtype=dtype, device=device)
    x_f32 = x.to(lucid.float32)

    for fn in (lucid.sum, lucid.mean):
        got = fn(x, dim=dim)
        want = fn(x_f32, dim=dim).to(dtype)
        assert got.dtype == dtype
        np.testing.assert_allclose(
            got.to(lucid.float32).numpy(),
            want.to(lucid.float32).numpy(),
            rtol=2e-3 if dtype == lucid.float16 else 1.6e-2,
            atol=1e-3,
        )

    got_var = lucid.var(x, dim=dim, correction=1)
    want_var = lucid.var(x_f32, dim=dim, correction=1).to(dtype)
    assert got_var.dtype == dtype
    np.testing.assert_allclose(
        got_var.to(lucid.float32).numpy(),
        want_var.to(lucid.float32).numpy(),
        rtol=4e-3 if dtype == lucid.float16 else 2.5e-2,
        atol=1e-4,
    )
