"""Matmul, BLAS, linear, conv, and einsum-contraction ops refuse mixed dtypes
outside autocast and cast eligible floating operands inside autocast (LCD-575).
"""

from collections.abc import Callable
from typing import Any

import pytest

import lucid
import lucid.nn.functional as F

_DEVICES = [
    "cpu",
    pytest.param(
        "metal",
        marks=pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device"),
    ),
]


def _make(
    lib: Any,
    shape: tuple[int, ...],
    dtype_name: str,
    device: str = "cpu",
) -> Any:
    dt = getattr(lib, dtype_name)
    if lib is lucid:
        return lucid.ones(shape, dtype=dt, device=device)
    dev = "mps" if device == "metal" else "cpu"
    return lib.ones(shape, dtype=dt, device=dev)


# (name, shapes, call(lib, F_mod, *tensors), refuses_outside_autocast)
_TABLE: list[tuple[str, list[tuple[int, ...]], Callable[..., Any], bool]] = [
    ("matmul", [(2, 3), (3, 2)], lambda lib, fn, a, b: a @ b, True),
    ("mm", [(2, 3), (3, 2)], lambda lib, fn, a, b: lib.mm(a, b), True),
    ("bmm", [(2, 2, 3), (2, 3, 2)], lambda lib, fn, a, b: lib.bmm(a, b), True),
    ("mv", [(2, 3), (3,)], lambda lib, fn, a, b: lib.mv(a, b), True),
    ("vdot", [(3,), (3,)], lambda lib, fn, a, b: lib.vdot(a, b), True),
    (
        "addmm",
        [(2, 2), (2, 3), (3, 2)],
        lambda lib, fn, c, a, b: lib.addmm(c, a, b),
        True,
    ),
    (
        "baddbmm",
        [(2, 2, 2), (2, 2, 3), (2, 3, 2)],
        lambda lib, fn, c, a, b: lib.baddbmm(c, a, b),
        True,
    ),
    (
        "addbmm",
        [(2, 2), (2, 2, 3), (2, 3, 2)],
        lambda lib, fn, c, a, b: lib.addbmm(c, a, b),
        True,
    ),
    ("addmv", [(2,), (2, 3), (3,)], lambda lib, fn, c, a, b: lib.addmv(c, a, b), True),
    ("addr", [(2, 3), (2,), (3,)], lambda lib, fn, c, a, b: lib.addr(c, a, b), False),
    ("outer", [(2,), (3,)], lambda lib, fn, a, b: a.outer(b), False),
    (
        "einsum_contract",
        [(2, 3), (3, 2)],
        lambda lib, fn, a, b: (
            lib.einops.einsum("ij,jk->ik", a, b) if lib is lucid else lib.einsum("ij,jk->ik", a, b)
        ),
        True,
    ),
    (
        "einsum_outer",
        [(2,), (3,)],
        lambda lib, fn, a, b: (
            lib.einops.einsum("i,j->ij", a, b) if lib is lucid else lib.einsum("i,j->ij", a, b)
        ),
        False,
    ),
    (
        "tensordot",
        [(2, 3), (3, 2)],
        lambda lib, fn, a, b: lib.tensordot(a, b, dims=1),
        True,
    ),
    ("linear", [(2, 3), (4, 3)], lambda lib, fn, x, w: fn.linear(x, w), True),
    (
        "linear_bias",
        [(2, 3), (4, 3), (4,)],
        lambda lib, fn, x, w, b: fn.linear(x, w, b),
        True,
    ),
    (
        "bilinear",
        [(2, 3), (2, 4), (5, 3, 4)],
        lambda lib, fn, x1, x2, w: fn.bilinear(x1, x2, w),
        True,
    ),
    ("conv1d", [(1, 2, 4), (3, 2, 2)], lambda lib, fn, x, w: fn.conv1d(x, w), True),
    (
        "conv2d",
        [(1, 2, 4, 4), (3, 2, 2, 2)],
        lambda lib, fn, x, w: fn.conv2d(x, w),
        True,
    ),
    (
        "conv3d",
        [(1, 2, 4, 4, 4), (3, 2, 2, 2, 2)],
        lambda lib, fn, x, w: fn.conv3d(x, w),
        True,
    ),
    (
        "conv_transpose1d",
        [(1, 2, 4), (2, 3, 2)],
        lambda lib, fn, x, w: fn.conv_transpose1d(x, w),
        True,
    ),
    (
        "conv_transpose2d",
        [(1, 2, 4, 4), (2, 3, 2, 2)],
        lambda lib, fn, x, w: fn.conv_transpose2d(x, w),
        True,
    ),
    (
        "conv_transpose3d",
        [(1, 2, 4, 4, 4), (2, 3, 2, 2, 2)],
        lambda lib, fn, x, w: fn.conv_transpose3d(x, w),
        True,
    ),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    ("name", "shapes", "call", "refuses"),
    _TABLE,
    ids=[row[0] for row in _TABLE],
)
@pytest.mark.parametrize(
    ("first_dt", "rest_dt"),
    [("float16", "float32"), ("float32", "float16"), ("bfloat16", "float32")],
)
def test_mixed_float_dtypes_outside_autocast(
    device: str,
    name: str,
    shapes: list[tuple[int, ...]],
    call: Callable[..., Any],
    refuses: bool,
    first_dt: str,
    rest_dt: str,
) -> None:
    dtypes = [first_dt] + [rest_dt] * (len(shapes) - 1)
    args = [_make(lucid, s, dt, device) for s, dt in zip(shapes, dtypes, strict=True)]
    if refuses:
        with pytest.raises(TypeError, match="DtypeMismatch"):
            call(lucid, F, *args)
    else:
        out = call(lucid, F, *args)
        assert out.dtype == lucid.float32


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "shapes", "call", "refuses"),
    _TABLE,
    ids=[row[0] for row in _TABLE],
)
def test_mixed_float_dtypes_match_reference_outside_and_inside_autocast(
    ref: Any,
    name: str,
    shapes: list[tuple[int, ...]],
    call: Callable[..., Any],
    refuses: bool,
) -> None:
    dtypes = ["float16"] + ["float32"] * (len(shapes) - 1)
    l_args = [_make(lucid, s, dt, "cpu") for s, dt in zip(shapes, dtypes, strict=True)]
    r_args = [_make(ref, s, dt, "cpu") for s, dt in zip(shapes, dtypes, strict=True)]

    if refuses:
        with pytest.raises(TypeError, match="DtypeMismatch"):
            call(lucid, F, *l_args)
        with pytest.raises(RuntimeError):
            call(ref, ref.nn.functional, *r_args)
    else:
        l_out = call(lucid, F, *l_args)
        r_out = call(ref, ref.nn.functional, *r_args)
        assert str(l_out.dtype).split(".")[-1] == str(r_out.dtype).split(".")[-1]

    # Under CPU bfloat16 autocast, LOWER ops cast both f16 and f32 operands to
    # bfloat16 and succeed, while KEEP ops (vdot, addmv, bilinear) still refuse.
    with lucid.amp.autocast("cpu", dtype=lucid.bfloat16):
        try:
            l_ac = call(lucid, F, *l_args)
            l_ok: str | None = str(l_ac.dtype).split(".")[-1]
        except TypeError:
            l_ok = None

    with ref.autocast("cpu", dtype=ref.bfloat16):
        try:
            r_ac = call(ref, ref.nn.functional, *r_args)
            r_ok: str | None = str(r_ac.dtype).split(".")[-1]
        except RuntimeError:
            r_ok = None

    assert l_ok == r_ok


@pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
@pytest.mark.parametrize(
    ("name", "shapes", "call", "refuses"),
    [row for row in _TABLE if row[0] != "vdot"],
    ids=[row[0] for row in _TABLE if row[0] != "vdot"],
)
def test_mixed_float_dtypes_succeed_under_metal_autocast(
    name: str,
    shapes: list[tuple[int, ...]],
    call: Callable[..., Any],
    refuses: bool,
) -> None:
    dtypes = ["float16"] + ["float32"] * (len(shapes) - 1)
    args = [_make(lucid, s, dt, "metal") for s, dt in zip(shapes, dtypes, strict=True)]
    with lucid.amp.autocast("metal", dtype=lucid.float16):
        out = call(lucid, F, *args)
    expected = (
        lucid.float32
        if name in ("bilinear", "tensordot", "conv_transpose3d", "outer", "einsum_outer")
        else lucid.float16
    )
    assert out.dtype == expected
