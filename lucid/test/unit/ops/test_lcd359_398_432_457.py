"""Regression tests for LCD-359, LCD-398, LCD-432, LCD-438, and LCD-457."""

import math
from types import ModuleType

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
import lucid.nn.functional as F
from lucid.test.unit.stability.test_grid_sample_far_coordinates import (
    _kinds,
    _stand_in,
)

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------------------
# LCD-457: Empty tensor overlap check and dense zero-size strides
# ---------------------------------------------------------------------------


_EMPTY_PERMUTES = [
    ((2, 0, 3), (2, 1, 0)),
    ((2, 0, 3), (0, 2, 1)),
    ((4, 3, 0, 2), (3, 1, 2, 0)),
    ((0, 5), (1, 0)),
]


@pytest.mark.parametrize("shape,perm", _EMPTY_PERMUTES)
def test_empty_permuted_view_inplace_ops_succeed(
    ref: ModuleType,
    device: str,
    shape: tuple[int, ...],
    perm: tuple[int, ...],
) -> None:
    x = lucid.zeros(*shape, device=device).permute(*perm)
    rx = ref.zeros(*shape).permute(*perm)
    x.mul_(2.0)
    x.add_(1.0)
    rx.mul_(2.0)
    rx.add_(1.0)
    assert x.shape == tuple(rx.shape)
    assert x.numel() == 0


@pytest.mark.parametrize("shape,perm", _EMPTY_PERMUTES)
def test_empty_permuted_view_keeps_its_strides_through_a_write(
    ref: ModuleType,
    device: str,
    shape: tuple[int, ...],
    perm: tuple[int, ...],
) -> None:
    x = lucid.zeros(*shape, device=device).permute(*perm)
    rx = ref.zeros(*shape).permute(*perm)
    x.mul_(2.0)
    rx.mul_(2.0)
    assert x.stride() == rx.stride()


def test_dropped_base_view_keeps_its_strides_through_a_write(
    ref: ModuleType,
) -> None:
    y = lucid.arange(6.0).reshape(2, 3).t()
    ry = ref.arange(6.0).reshape(2, 3).t()
    y.mul_(2.0).add_(1.0)
    ry.mul_(2.0).add_(1.0)
    assert y.stride() == ry.stride() == (1, 3)
    assert y.is_contiguous() == ry.is_contiguous() is False
    assert y.tolist() == ry.tolist()

    # A view of a dropped intermediate in an autograd graph also keeps its
    # strided layout across an in-place write and propagates gradients.
    x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    rx = ref.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    v = (x + 1.0).t()
    rv = (rx + 1.0).t()
    v.mul_(3.0).erfc_()
    rv.mul_(3.0).erfc_()
    assert v.stride() == rv.stride() == (1, 3)
    assert v.reshape(-1).tolist() == pytest.approx(rv.reshape(-1).tolist(), rel=1e-5, abs=1e-5)
    v.sum().backward()
    rv.sum().backward()
    assert x.grad is not None and rx.grad is not None
    assert x.grad.reshape(-1).tolist() == pytest.approx(
        rx.grad.reshape(-1).tolist(), rel=1e-5, abs=1e-5
    )


@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (3, 0),
        (0, 3),
        (3, 0, 2),
        (2, 0, 3, 4),
        (2, 3, 0, 1, 4),
    ],
)
def test_dense_empty_tensor_strides_match_reference(
    ref: ModuleType,
    device: str,
    shape: tuple[int, ...],
) -> None:
    x = lucid.zeros(*shape, device=device)
    rx = ref.zeros(*shape)
    assert x.stride() == rx.stride()
    assert x.is_contiguous() == rx.is_contiguous()
    for axis in range(len(shape) + 1):
        assert x.unsqueeze(axis).stride() == rx.unsqueeze(axis).stride()


# ---------------------------------------------------------------------------
# LCD-432: frexp and ldexp at extreme magnitudes and special values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_name", ["float32", "float64", "float16", "bfloat16"])
def test_frexp_matches_reference_across_full_range(
    ref: ModuleType,
    device: str,
    dtype_name: str,
) -> None:
    if device == "metal" and dtype_name == "float64":
        pytest.skip("Metal does not support float64")
    l_dt = getattr(lucid, dtype_name)
    r_dt = getattr(ref, dtype_name)
    if dtype_name in ("float16", "bfloat16"):
        vals = [
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1e-7,
            -1e-7,
            1e-4,
            60000.0,
            -60000.0,
            INF,
            -INF,
            NAN,
        ]
    else:
        vals = [
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.401298464324817e-45,
            -1.401298464324817e-45,
            1e-40,
            -1e-40,
            1.5e-39,
            -1.5e-39,
            1e-38,
            1e38,
            -1e38,
            3e38,
            -3e38,
            INF,
            -INF,
            NAN,
        ]
    lx = lucid.tensor(vals, dtype=l_dt, device=device)
    rx = ref.tensor(vals, dtype=r_dt)
    lm, le = lucid.frexp(lx)
    rm, re = ref.frexp(rx)
    round_trip = lucid.ldexp(lm, le)
    want_rt = ref.ldexp(rm, re)

    assert lm.dtype == l_dt
    assert le.dtype == lucid.int32
    assert le.tolist() == re.tolist()

    lm_list = lm.to(dtype=lucid.float32).tolist()
    rm_list = rm.to(dtype=ref.float32).tolist()
    for g, w in zip(lm_list, rm_list, strict=True):
        if math.isnan(w):
            assert math.isnan(g)
        elif math.isinf(w):
            assert g == w
        elif w == 0.0:
            assert g == 0.0
            assert math.copysign(1.0, g) == math.copysign(1.0, w)
        else:
            assert g == pytest.approx(w, rel=1e-6, abs=1e-7)

    for g, w in zip(round_trip.tolist(), want_rt.tolist(), strict=True):
        if math.isnan(w):
            assert math.isnan(g)
        elif math.isinf(w):
            assert g == w
        elif w == 0.0:
            assert g == 0.0
            assert math.copysign(1.0, g) == math.copysign(1.0, w)
        else:
            assert g == pytest.approx(w, rel=1e-6, abs=0.0)


@pytest.mark.parametrize(
    "m_vals,e_vals",
    [
        ([0.5, -0.75, 0.5877471566200256, 1.0], [128, 127, 127, -126]),
        ([1e30, -1e30, 0.5, -0.5, -0.0, 0.0], [-200, -200, -130, -148, 50, -50]),
    ],
)
def test_ldexp_extreme_exponents_and_gradients(
    ref: ModuleType,
    device: str,
    m_vals: list[float],
    e_vals: list[int],
) -> None:
    lm = lucid.tensor(m_vals, dtype=lucid.float32, device=device)
    le = lucid.tensor(e_vals, dtype=lucid.int32, device=device)
    rm = ref.tensor(m_vals, dtype=ref.float32)
    re = ref.tensor(e_vals, dtype=ref.int32)
    got = lucid.ldexp(lm, le)
    want = ref.ldexp(rm, re)
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-6, atol=0.0)
    for g, w in zip(got.tolist(), want.tolist(), strict=True):
        if w == 0.0:
            assert math.copysign(1.0, g) == math.copysign(1.0, w)

    # Differentiable in both arguments (first and second derivatives).
    x = lucid.tensor([0.75, -1.25], dtype=lucid.float32, device=device, requires_grad=True)
    k = lucid.tensor([-4.0, 0.0], dtype=lucid.float32, device=device, requires_grad=True)
    y = lucid.ldexp(x, k)
    gx, gk = lucid.autograd.grad(y.sum(), [x, k], create_graph=True)
    (g2k,) = lucid.autograd.grad(gk.sum(), [k])
    np.testing.assert_allclose(gx.numpy(), np.array([2.0**-4.0, 1.0], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(
        gk.numpy(),
        np.array(
            [0.75 * (2.0**-4.0) * math.log(2.0), -1.25 * math.log(2.0)],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        g2k.numpy(),
        np.array(
            [0.75 * (2.0**-4.0) * (math.log(2.0) ** 2), -1.25 * (math.log(2.0) ** 2)],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# LCD-438 & LCD-398: grid_sample far-coordinate unnormalise & clipped NaN grad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1, 3, 5), (1, 1, 3, 4, 5)])
@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "point",
    [
        [2e38, 0.25],
        [-2e38, 0.25],
        [0.25, 2e38],
        [1e10, INF],
        [INF, 1e10],
        [-1e10, -INF],
    ],
)
def test_grid_sample_near_float_max_and_mixed_infinite_coords(
    ref: ModuleType,
    device: str,
    shape: tuple[int, ...],
    mode: str,
    padding_mode: str,
    align_corners: bool,
    point: list[float],
) -> None:
    if len(shape) == 5 and mode == "bicubic":
        pytest.skip("5-D grid_sample supports bilinear and nearest only")
    coords = point if len(shape) == 4 else [point[0], point[1], 0.25]
    n_elem = math.prod(shape)
    img_vals = [float(i + 1) for i in range(n_elem)]
    grid_shape = (1, 1, 1, 2) if len(shape) == 4 else (1, 1, 1, 1, 3)

    lx = lucid.tensor(img_vals, dtype=lucid.float32, device=device).reshape(*shape)
    lx.requires_grad_(True)
    lg = lucid.tensor(coords, dtype=lucid.float32, device=device).reshape(*grid_shape)
    lg.requires_grad_(True)
    ly = F.grid_sample(lx, lg, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    ly.sum().backward()

    def reference(at: list[float]) -> tuple[np.ndarray, ...]:
        rx = ref.tensor(img_vals, dtype=ref.float32).reshape(*shape)
        rg = ref.tensor(at, dtype=ref.float32).reshape(*grid_shape)
        rx.requires_grad_(True)
        rg.requires_grad_(True)
        ry = ref.nn.functional.grid_sample(
            rx, rg, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )
        ry.sum().backward()
        return ry.detach().numpy(), rx.grad.numpy(), rg.grad.numpy()

    def check(got: tuple[np.ndarray, ...], want: tuple[np.ndarray, ...]) -> None:
        for g, w in zip(got, want, strict=True):
            np.testing.assert_allclose(g, w, atol=1e-5, equal_nan=True)

    got = (ly.numpy(), lx.grad.numpy(), lg.grad.numpy())
    # The 5-D cells where Lucid departs from the reference on purpose are the
    # far-coordinate matrix's named departures: the reference's 3-D kernel
    # reads a reflected non-finite position at different places forward and
    # backward ("nan-split"), and overflows a far zeros-padded position
    # ("overflow-wrap").  Lucid is held to the reference at that matrix's
    # stand-in coordinate, and keeps the reference's forward under nan-split.
    kinds = {k for k in _kinds(shape, mode, padding_mode, align_corners, coords) if k}
    if not kinds:
        check(got, reference(coords))
        return
    if kinds == {"nan-split"}:
        check(got[:1], reference(coords)[:1])
    check(got, reference(_stand_in(shape, mode, padding_mode, align_corners, coords)))


@pytest.mark.parametrize("bad_mode,bad_pad", [(2, 0), (-1, 0), (0, 2), (0, -1)])
def test_engine_grid_sample_refuses_unlowered_mode_or_padding(
    device: str,
    bad_mode: int,
    bad_pad: int,
) -> None:
    x = lucid.zeros(1, 1, 4, 4, device=device)._impl
    g = lucid.zeros(1, 2, 2, 2, device=device)._impl
    with pytest.raises(_C_engine.InvalidArgument):
        _C_engine.nn.grid_sample(x, g, bad_mode, bad_pad, False)


# ---------------------------------------------------------------------------
# LCD-359: Engine unique_op (NaN, -0.0, all real/bool dtypes) & scatter_op dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_name",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
)
def test_engine_unique_supports_all_real_and_bool_dtypes(
    ref: ModuleType,
    device: str,
    dtype_name: str,
) -> None:
    if device == "metal" and dtype_name == "float64":
        pytest.skip("Metal does not support float64")
    l_dt = getattr(lucid, dtype_name)
    r_dt = getattr(ref, dtype_name)
    data = [True, False, True, True] if dtype_name == "bool" else [3, 1, 3, -1, 2, 1, 0, -1]
    lx = lucid.tensor(data, dtype=l_dt, device=device)
    rx = ref.tensor(data, dtype=r_dt)
    got_eng = lucid.Tensor.__new_from_impl__(_C_engine.unique(lx._impl))
    got_pub = lucid.unique(lx)
    want = ref.unique(rx)
    assert got_eng.dtype == l_dt
    assert got_eng.device == lx.device
    assert got_eng.tolist() == want.tolist()
    assert got_pub.tolist() == want.tolist()


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize(
    "vals",
    [
        [NAN, 2.0, 1.0, NAN, -1.0, 1.0, 2.0],
        [-0.0, 0.0, 1.0, -0.0],
        [0.0, -0.0, 1.0, 0.0],
        [INF, -INF, NAN, 0.0, INF, -INF],
    ],
)
def test_engine_unique_orders_nans_last_and_preserves_first_zero_sign(
    ref: ModuleType,
    device: str,
    dtype_name: str,
    vals: list[float],
) -> None:
    if device == "metal" and dtype_name == "float64":
        pytest.skip("Metal does not support float64")
    l_dt = getattr(lucid, dtype_name)
    r_dt = getattr(ref, dtype_name)
    lx = lucid.tensor(vals, dtype=l_dt, device=device)
    rx = ref.tensor(vals, dtype=r_dt)
    got_eng = lucid.Tensor.__new_from_impl__(_C_engine.unique(lx._impl))
    got_pub = lucid.unique(lx)
    want = ref.unique(rx)
    for got in (got_eng, got_pub):
        g_list = got.to(dtype=lucid.float32).tolist()
        w_list = want.to(dtype=ref.float32).tolist()
        assert len(g_list) == len(w_list)
        for g, w in zip(g_list, w_list, strict=True):
            if math.isnan(w):
                assert math.isnan(g)
            else:
                assert g == w
                if w == 0.0:
                    assert math.copysign(1.0, g) == math.copysign(1.0, w)


def test_engine_scatter_refuses_mismatched_src_dtype(device: str) -> None:
    base = lucid.zeros(4, dtype=lucid.float32, device=device)._impl
    idx = lucid.tensor([0, 2], dtype=lucid.int64, device=device)._impl
    src = lucid.ones(2, dtype=lucid.float16, device=device)._impl
    with pytest.raises(_C_engine.DtypeMismatch):
        _C_engine.scatter(base, 0, idx, src)
