"""Reference parity for ``roll`` and shape-sequence adapters (LCD-348).

``lucid.roll`` and ``Tensor.roll`` share ``_roll_adapter``, which normalizes
``shifts`` (``int | Sequence[int]``) and ``dims`` (``int | Sequence[int] | None``)
before calling ``_C_engine.roll``.  Without ``dims``, elements roll in
flattened order and reshape back.  Sibling shape/axis adapters (``flip``,
``tile``, ``Tensor.repeat``, ``permute``, ``movedim``, ``reshape``, ``view``,
``expand``, ``broadcast_to``) share ``_int_list`` / ``_to_axes`` so scalar
``SupportsIndex`` values (including ``np.int64``) work everywhere and ``bool``
or ``None`` is refused with ``TypeError``.
"""

from collections.abc import Callable, Sequence
from types import ModuleType

import numpy as np
import pytest

import lucid


def _values(shape: tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape)) if shape else 1
    return (np.arange(n, dtype=np.float32) + 1.0).reshape(shape)


def _assert_roll_parity(
    ref: ModuleType,
    device: str,
    shape: tuple[int, ...],
    shifts: int | Sequence[int],
    dims: int | Sequence[int] | None = None,
    *,
    pass_dims: bool = True,
) -> None:
    vals = _values(shape)
    w_vals = _values(shape) * 0.5 + 0.25

    # 1. Free function: lucid.roll vs ref.roll
    x_fn = lucid.tensor(vals, device=device, requires_grad=True)
    r_fn = ref.tensor(vals, requires_grad=True)
    if pass_dims:
        out_fn = lucid.roll(x_fn, shifts, dims)
        ref_fn = ref.roll(r_fn, shifts, dims)
    else:
        out_fn = lucid.roll(x_fn, shifts)
        ref_fn = ref.roll(r_fn, shifts)

    assert tuple(out_fn.shape) == tuple(ref_fn.shape)
    np.testing.assert_allclose(out_fn.numpy(), ref_fn.detach().numpy(), rtol=1e-6, atol=1e-6)

    (out_fn * lucid.tensor(w_vals, device=device)).sum().backward()
    (ref_fn * ref.tensor(w_vals)).sum().backward()
    assert x_fn.grad is not None
    np.testing.assert_allclose(x_fn.grad.numpy(), r_fn.grad.numpy(), rtol=1e-6, atol=1e-6)

    # 2. Tensor method: Tensor.roll vs ref Tensor.roll
    x_m = lucid.tensor(vals, device=device, requires_grad=True)
    r_m = ref.tensor(vals, requires_grad=True)
    if pass_dims:
        out_m = x_m.roll(shifts, dims)
        ref_m = r_m.roll(shifts, dims)
    else:
        out_m = x_m.roll(shifts)
        ref_m = r_m.roll(shifts)

    assert tuple(out_m.shape) == tuple(ref_m.shape)
    np.testing.assert_allclose(out_m.numpy(), ref_m.detach().numpy(), rtol=1e-6, atol=1e-6)

    (out_m * lucid.tensor(w_vals, device=device)).sum().backward()
    (ref_m * ref.tensor(w_vals)).sum().backward()
    assert x_m.grad is not None
    np.testing.assert_allclose(x_m.grad.numpy(), r_m.grad.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 4)], ids=str)
@pytest.mark.parametrize(
    ("shifts", "dims"),
    [
        (1, 0),
        (-1, -1),
        (2, 0),
        ((1,), (0,)),
        ([2], [-1]),
    ],
    ids=["int-pos", "int-neg", "int-wrap", "tuple-1", "list-1"],
)
def test_roll_int_and_single_axis_parity(
    ref: ModuleType,
    device: str,
    shape: tuple[int, ...],
    shifts: int | Sequence[int],
    dims: int | Sequence[int],
) -> None:
    _assert_roll_parity(ref, device, shape, shifts, dims, pass_dims=True)


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 3, 4)], ids=str)
@pytest.mark.parametrize("shifts", [0, 1, -2, 7, (1,), [-2]], ids=str)
def test_roll_dims_none_flattens_and_restores_shape(
    ref: ModuleType,
    device: str,
    shape: tuple[int, ...],
    shifts: int | Sequence[int],
) -> None:
    _assert_roll_parity(ref, device, shape, shifts, None, pass_dims=False)
    _assert_roll_parity(ref, device, shape, shifts, None, pass_dims=True)


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
@pytest.mark.parametrize(
    ("shifts", "dims"),
    [
        ((1, -1), (0, 1)),
        ([2, -1], [0, -1]),
        ((1, 2), (0, 0)),
        ((1, -1), (0, -2)),
    ],
    ids=["tuple-2d", "list-neg-dim", "repeated-dim", "aliased-neg-dim"],
)
def test_roll_multi_axis_sequence_parity(
    ref: ModuleType,
    device: str,
    shape: tuple[int, ...],
    shifts: Sequence[int],
    dims: Sequence[int],
) -> None:
    _assert_roll_parity(ref, device, shape, shifts, dims, pass_dims=True)


@pytest.mark.parity
def test_roll_keyword_arguments_parity(ref: ModuleType, device: str) -> None:
    x = lucid.arange(6.0, device=device, requires_grad=True).reshape(2, 3)
    r = ref.arange(6.0, requires_grad=True).reshape(2, 3)

    got_fn = lucid.roll(input=x, shifts=1, dims=0)
    got_m = x.roll(shifts=1, dims=0)
    want = ref.roll(input=r, shifts=1, dims=0)
    np.testing.assert_allclose(got_fn.numpy(), want.detach().numpy())
    np.testing.assert_allclose(got_m.numpy(), want.detach().numpy())

    got_none = x.roll(shifts=-2, dims=None)
    want_none = r.roll(shifts=-2, dims=None)
    np.testing.assert_allclose(got_none.numpy(), want_none.detach().numpy())


@pytest.mark.parity
@pytest.mark.parametrize(
    ("shifts", "dims"),
    [
        (1, (0, 1)),
        ((1, 2), 0),
        ((1, 2), (0,)),
        ((1, 2), None),
        ((), (0,)),
    ],
    ids=[
        "int-shift-seq-dims",
        "seq-shifts-int-dim",
        "len-2-vs-1",
        "seq-shifts-no-dims",
        "empty-shifts",
    ],
)
def test_roll_mismatched_shifts_and_dims_refused(
    ref: ModuleType,
    device: str,
    shifts: int | Sequence[int],
    dims: int | Sequence[int] | None,
) -> None:
    x = lucid.zeros(2, 3, device=device)
    r = ref.zeros(2, 3)
    with pytest.raises(RuntimeError):
        ref.roll(r, shifts, dims)
    with pytest.raises(ValueError, match="roll"):
        lucid.roll(x, shifts, dims)
    with pytest.raises(ValueError, match="roll"):
        x.roll(shifts, dims)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("_label", "call_ref", "call_lucid"),
    [
        (
            "flip-int",
            lambda r, t: t.flip(0).numpy(),
            lambda x: (lucid.flip(x, 0), x.flip(0)),
        ),
        (
            "flip-np-int",
            lambda r, t: t.flip(int(np.int64(-1))).numpy(),
            lambda x: (lucid.flip(x, np.int64(-1)), x.flip(np.int64(-1))),
        ),
        (
            "flip-varargs",
            lambda r, t: t.flip(0, 1).numpy(),
            lambda x: (lucid.flip(x, (0, 1)), x.flip(0, 1)),
        ),
        (
            "tile-int",
            lambda r, t: t.tile(2).numpy(),
            lambda x: (lucid.tile(x, 2), x.tile(2)),
        ),
        (
            "tile-np-int",
            lambda r, t: t.tile(int(np.int64(2))).numpy(),
            lambda x: (lucid.tile(x, np.int64(2)), x.tile(np.int64(2))),
        ),
        (
            "tile-varargs",
            lambda r, t: t.tile(2, 3).numpy(),
            lambda x: (lucid.tile(x, (2, 3)), x.tile(2, 3)),
        ),
        (
            "repeat-varargs",
            lambda r, t: t.repeat(2, 3).numpy(),
            lambda x: (x.repeat((2, 3)), x.repeat(2, 3)),
        ),
        (
            "repeat-np-int",
            lambda r, t: t.repeat(2, 3).numpy(),
            lambda x: (
                x.repeat((np.int64(2), np.int64(3))),
                x.repeat(np.int64(2), np.int64(3)),
            ),
        ),
        (
            "permute-varargs",
            lambda r, t: t.permute(1, 0).numpy(),
            lambda x: (lucid.permute(x, (1, 0)), x.permute(1, 0)),
        ),
        (
            "movedim-np-int",
            lambda r, t: r.movedim(t, 0, 1).numpy(),
            lambda x: (
                lucid.movedim(x, np.int64(0), np.int64(1)),
                x.movedim(np.int64(0), np.int64(1)),
            ),
        ),
        (
            "rot90-tuple",
            lambda r, t: r.rot90(t, 1, (0, 1)).numpy(),
            lambda x: (lucid.rot90(x, 1, (0, 1)), lucid.rot90(x, 1, [0, 1])),
        ),
    ],
)
def test_sibling_shape_ops_int_and_sequence_spellings(
    ref: ModuleType,
    device: str,
    _label: str,
    call_ref: Callable[[ModuleType, object], np.ndarray],
    call_lucid: Callable[[lucid.Tensor], tuple[lucid.Tensor, lucid.Tensor]],
) -> None:
    del _label  # used for pytest parametrization id
    vals = _values((2, 3))
    x = lucid.tensor(vals, device=device)
    r = ref.tensor(vals)
    want = call_ref(ref, r)
    got_a, got_b = call_lucid(x)
    np.testing.assert_array_equal(got_a.numpy(), want)
    np.testing.assert_array_equal(got_b.numpy(), want)


def _call_untyped(fn: object, *args: object) -> object:
    assert callable(fn)
    return fn(*args)


@pytest.mark.parametrize(
    "bad_call",
    [
        lambda x: _call_untyped(lucid.roll, x, None),
        lambda x: _call_untyped(x.roll, None),
        lambda x: x.roll(True, 0),
        lambda x: x.roll(1, True),
        lambda x: lucid.flip(x, True),
        lambda x: x.flip(True),
        lambda x: _call_untyped(lucid.flip, x, None),
        lambda x: lucid.tile(x, True),
        lambda x: x.tile(True),
        lambda x: _call_untyped(lucid.tile, x, None),
        lambda x: x.repeat(True, True),
        lambda x: _call_untyped(x.repeat, None),
        lambda x: _call_untyped(lucid.permute, x, None),
        lambda x: x.permute(True, False),
        lambda x: lucid.movedim(x, True, 0),
        lambda x: _call_untyped(lucid.movedim, x, None, 0),
    ],
)
def test_bool_and_none_refused_in_shape_sequence_ops(
    device: str,
    bad_call: Callable[[lucid.Tensor], object],
) -> None:
    x = lucid.arange(6.0, device=device).reshape(2, 3)
    with pytest.raises(TypeError):
        bad_call(x)
