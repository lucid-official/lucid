"""Fused single-pass Metal bounds-checked gather (LCD-277).

Covers ``gpu_take_along_axis`` and ``gpu_take`` in
``lucid/_C/backend/gpu/AxisIndex.h`` across all Metal source dtypes, index
dtypes, axes, non-contiguous views, empty shapes, 64-bit out-of-range
indices, and zero-host-sync isolation under Policy B.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal device not available")

_FLOAT_DTYPES = [lucid.float32, lucid.float16, lucid.bfloat16]
_INT_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]
_INDEX_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]


@pytest.mark.parametrize("src_dtype", _FLOAT_DTYPES)
@pytest.mark.parametrize("idx_dtype", _INDEX_DTYPES)
def test_gather_float_dtypes_in_and_out_of_range(
    src_dtype: lucid.dtype, idx_dtype: lucid.dtype
) -> None:
    src = lucid.arange(12, dtype=src_dtype, device="metal").reshape(3, 4)
    idx = lucid.tensor([[0, -1, 4, -5], [2, 3, -4, 99]], dtype=idx_dtype, device="metal")
    out = lucid.gather(src, 1, idx).to(lucid.float32).tolist()
    assert out[0][0] == 0.0
    assert out[0][1] == 3.0
    assert math.isnan(out[0][2])
    assert math.isnan(out[0][3])
    assert out[1][0] == 6.0
    assert out[1][1] == 7.0
    assert out[1][2] == 4.0
    assert math.isnan(out[1][3])


@pytest.mark.parametrize("src_dtype", _INT_DTYPES)
@pytest.mark.parametrize("idx_dtype", _INDEX_DTYPES)
def test_gather_integer_dtypes_fill_zero_out_of_range(
    src_dtype: lucid.dtype, idx_dtype: lucid.dtype
) -> None:
    src = (lucid.arange(8, dtype=src_dtype, device="metal") + 10).reshape(2, 4)
    idx = lucid.tensor([[1, 4, -1], [-5, 0, 3]], dtype=idx_dtype, device="metal")
    got = lucid.gather(src, 1, idx).tolist()
    assert got == [[11, 0, 13], [0, 14, 17]]


def test_gather_bool_and_complex64_out_of_range() -> None:
    b_src = lucid.tensor([[True, False, True]], dtype=lucid.bool_, device="metal")
    b_idx = lucid.tensor([[0, 2, 3, -4]], dtype=lucid.int64, device="metal")
    assert lucid.gather(b_src, 1, b_idx).tolist() == [[True, True, False, False]]

    c_src = lucid.tensor([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=lucid.complex64, device="metal")
    c_idx = lucid.tensor([[1, -2, 5, -3]], dtype=lucid.int64, device="metal")
    c_out = lucid.gather(c_src, 1, c_idx).tolist()[0]
    assert c_out[0] == complex(3.0, -4.0)
    assert c_out[1] == complex(1.0, 2.0)
    assert math.isnan(c_out[2].real) and math.isnan(c_out[2].imag)
    assert math.isnan(c_out[3].real) and math.isnan(c_out[3].imag)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_gather_3d_axes_and_64bit_bounds(axis: int) -> None:
    src = lucid.arange(24, dtype=lucid.float32, device="metal").reshape(2, 3, 4)
    huge = 2**40
    neg_huge = -(2**40)
    idx = lucid.zeros((2, 3, 4), dtype=lucid.int64, device="metal")
    idx_np = np.zeros((2, 3, 4), dtype=np.int64)
    idx_np[0, 0, 0] = 1
    idx_np[0, 0, 1] = -1
    idx_np[0, 0, 2] = huge
    idx_np[0, 0, 3] = neg_huge
    idx = lucid.tensor(idx_np, dtype=lucid.int64, device="metal")

    out = lucid.gather(src, axis, idx).numpy()
    ax = axis if axis >= 0 else axis + 3
    coord_pos = [0, 0, 0]
    coord_pos[ax] = 1
    assert out[0, 0, 0] == src.numpy()[tuple(coord_pos)]

    coord_neg = [0, 0, 1]
    coord_neg[ax] = src.shape[ax] - 1
    assert out[0, 0, 1] == src.numpy()[tuple(coord_neg)]
    assert np.isnan(out[0, 0, 2])
    assert np.isnan(out[0, 0, 3])


@pytest.mark.parametrize("src_dtype", [*_FLOAT_DTYPES, lucid.complex64])
def test_embedding_take_dtypes_and_strict_negative_bounds(
    src_dtype: lucid.dtype,
) -> None:
    weight = lucid.arange(12, dtype=lucid.float32, device="metal").reshape(4, 3).to(src_dtype)
    idx = lucid.tensor([[2, -1], [4, 2**40]], dtype=lucid.int64, device="metal")
    out = _wrap(_C_engine.nn.embedding(_unwrap(weight), _unwrap(idx), -1))
    assert out.shape == (2, 2, 3)
    if src_dtype == lucid.complex64:
        vals = out.tolist()
        assert vals[0][0] == [6.0 + 0.0j, 7.0 + 0.0j, 8.0 + 0.0j]
        assert all(math.isnan(z.real) and math.isnan(z.imag) for z in vals[0][1])
        assert all(math.isnan(z.real) and math.isnan(z.imag) for z in vals[1][0])
        assert all(math.isnan(z.real) and math.isnan(z.imag) for z in vals[1][1])
    else:
        vals_f = out.to(lucid.float32).tolist()
        assert vals_f[0][0] == [6.0, 7.0, 8.0]
        assert all(math.isnan(v) for v in vals_f[0][1])
        assert all(math.isnan(v) for v in vals_f[1][0])
        assert all(math.isnan(v) for v in vals_f[1][1])


def test_non_contiguous_views_and_empty_outputs() -> None:
    base = lucid.arange(20, dtype=lucid.float32, device="metal").reshape(4, 5)
    view = base.T[:, 1:4]  # (5, 3), non-contiguous
    idx = lucid.tensor([[0, 2, 3], [-1, -3, -4]], dtype=lucid.int64, device="metal")
    out = lucid.gather(view, 1, idx).tolist()
    want_row0 = [view.tolist()[0][0], view.tolist()[0][2], math.nan]
    want_row1 = [view.tolist()[1][2], view.tolist()[1][0], math.nan]
    assert out[0][:2] == want_row0[:2] and math.isnan(out[0][2])
    assert out[1][:2] == want_row1[:2] and math.isnan(out[1][2])

    empty_idx = lucid.zeros((4, 0), dtype=lucid.int64, device="metal")
    empty_out = lucid.gather(base, 1, empty_idx)
    assert empty_out.shape == (4, 0)

    empty_emb = F.embedding(lucid.zeros((0, 3), dtype=lucid.int64, device="metal"), base)
    assert empty_emb.shape == (0, 3, 5)


def test_fused_gather_and_sink_take_zero_host_syncs() -> None:
    x = lucid.arange(24, dtype=lucid.float32, device="metal").reshape(4, 6)
    idx_g = lucid.tensor([[0, 6, -1], [2, -7, 2**40]], dtype=lucid.int64, device="metal")
    idx_s = lucid.tensor([1, 4, -1, 99], dtype=lucid.int64, device="metal")
    lucid.eval(x, idx_g, idx_s)

    before = _C_engine.host_sync_count()
    g_out = lucid.gather(x, 1, idx_g)
    s_out = lucid.index_select(x, 0, idx_s)
    e_out = F.embedding(lucid.tensor([0, 4, -1], dtype=lucid.int64, device="metal"), x)
    x_copy = x.clone()
    x_copy[idx_s] = 0.0
    lucid.eval(g_out, s_out, e_out, x_copy)
    assert _C_engine.host_sync_count() == before
