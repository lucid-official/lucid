"""The Python side resolves a dim by the engine's rule (LCD-406).

The engine's ``wrap_dim`` gives a 0-d tensor one notional axis.  Python
wrappers that read ``shape[dim]``, built a permutation or checked a dim
list kept copies of the rule with the raw rank, and each copy failed
differently: ``list index out of range`` from argsort and swapaxes, a
silent no-op from squeeze on an axis past the last one.  They now share
:mod:`lucid._ops._dims`, which is held here to the engine itself, and the
ops that had no rule at all — ``roll`` without dims, ``normalize`` over
several dims — are held to the reference.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._ops._dims import _wrap_dim, _wrap_dims


@pytest.mark.parametrize("ndim", [0, 1, 2, 3])
def test_the_python_rule_is_the_engines(ndim):
    """Every dim the engine takes, and only those, wraps to its answer."""
    x = _C_engine.zeros([2] * ndim, _C_engine.F32, _C_engine.CPU)
    for dim in range(-ndim - 2, ndim + 2):
        try:
            # ``cumsum`` keeps the shape, so the wrapped dim is the one the
            # engine reports back by refusing or not; ``flip`` would too.
            _C_engine.cumsum(x, dim)
        except IndexError:
            with pytest.raises(IndexError, match=f"axis out of range: got {dim}"):
                _wrap_dim(dim, ndim, "op")
            continue
        assert _wrap_dim(dim, ndim, "op") == dim % max(ndim, 1)


@pytest.mark.parametrize(
    ("dims", "ndim"), [((0, -1), 0), ((1, -1), 2), ((0, 0), 3)], ids=str
)
def test_an_axis_named_twice_is_refused(dims, ndim):
    with pytest.raises(RuntimeError, match="appears multiple times"):
        _wrap_dims(dims, ndim, "op")


_PAST_THE_LAST_AXIS = {
    "squeeze": lambda lib, t, d: lib.squeeze(t, d),
    "squeeze_list": lambda lib, t, d: lib.squeeze(t, [d]),
    "take_along_dim": lambda lib, t, d: lib.take_along_dim(
        t, lib.zeros(t.shape, dtype=lib.int64), dim=d
    ),
    "split": lambda lib, t, d: lib.split(t, 1, dim=d),
    "chunk": lambda lib, t, d: lib.chunk(t, 1, dim=d),
    "quantile": lambda lib, t, d: lib.quantile(t, 0.5, dim=d),
    "nanquantile": lambda lib, t, d: lib.nanquantile(t, 0.5, dim=d),
    "sort": lambda lib, t, d: lib.sort(t, dim=d),
    "sort_descending": lambda lib, t, d: lib.sort(t, dim=d, descending=True),
    "argsort": lambda lib, t, d: lib.argsort(t, dim=d),
    "argsort_descending": lambda lib, t, d: lib.argsort(t, dim=d, descending=True),
    "topk_smallest": lambda lib, t, d: lib.topk(t, 1, dim=d, largest=False),
    "swapaxes": lambda lib, t, d: lib.swapaxes(t, d, 0),
    "glu": lambda lib, t, d: lib.nn.functional.glu(t, d),
    "diff": lambda lib, t, d: lib.diff(t, dim=d),
    "Tensor.diff": lambda lib, t, d: t.diff(dim=d),
    "linalg.cross": lambda lib, t, d: lib.linalg.cross(t, t, dim=d),
    "unique": lambda lib, t, d: lib.unique(t, dim=d),
    "unique_consecutive": lambda lib, t, d: lib.unique_consecutive(t, dim=d),
    "fft": lambda lib, t, d: lib.fft.fft(t, dim=d),
    "fftshift": lambda lib, t, d: lib.fft.fftshift(t, dim=d),
}


@pytest.mark.parametrize("shape", [(), (3,), (2, 1, 3)], ids=str)
@pytest.mark.parametrize("op", list(_PAST_THE_LAST_AXIS))
def test_an_axis_past_the_last_is_refused(ref, device, op, shape):
    call = _PAST_THE_LAST_AXIS[op]
    bad = max(len(shape), 1)
    for dim in (bad, -bad - 1):
        with pytest.raises(
            IndexError,
            match=f"axis out of range: got {dim}|Dimension specified as {dim}|FFT axis {dim}",
        ):
            call(lucid, lucid.zeros(shape, device=device), dim)
        if (
            op
            in (
                "split",
                "chunk",
                "glu",
                "diff",
                "Tensor.diff",
                "unique",
                "unique_consecutive",
                "fft",
                "fftshift",
            )
            and not shape
        ):
            continue
        with pytest.raises(IndexError):
            call(ref, ref.zeros(shape), dim)


_STRAGGLERS = {
    "Tensor.diff": lambda lib, d: lib.tensor([1.0, 2.0, 4.0]).diff(dim=d),
    "linalg.cross": lambda lib, d: lib.linalg.cross(
        lib.ones(3, 3), lib.ones(3, 3), dim=d
    ),
}


@pytest.mark.parametrize(
    ("op", "dim"),
    [
        ("Tensor.diff", 1),
        ("Tensor.diff", -2),
        ("linalg.cross", 2),
        ("linalg.cross", -3),
    ],
)
def test_a_straggler_refuses_an_axis_past_the_last(ref, op, dim):
    call = _STRAGGLERS[op]
    with pytest.raises(IndexError):
        call(ref, dim)
    with pytest.raises(IndexError, match=f"axis out of range: got {dim}"):
        call(lucid, dim)


def test_squeeze_refuses_an_axis_named_twice(ref):
    with pytest.raises(RuntimeError, match="appears multiple times"):
        ref.squeeze(ref.zeros(1, 3), (0, -2))
    with pytest.raises(RuntimeError, match="appears multiple times"):
        lucid.squeeze(lucid.zeros(1, 3), (0, -2))


def _values(shape):
    n = int(np.prod(shape)) if shape else 1
    return (np.arange(n, dtype=np.float32) + 1.0).reshape(shape)


def _compare(ref, device, shape, call, rtol=1e-5):
    values = _values(shape)
    x = lucid.tensor(values, device=device, requires_grad=True)
    xr = ref.tensor(values, requires_grad=True)
    got, want = call(lucid, F, x), call(ref, ref.nn.functional, xr)
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=rtol, atol=1e-6)
    weights = _values(tuple(want.shape))
    (got * lucid.tensor(weights, device=device)).sum().backward()
    (want * ref.tensor(weights)).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), xr.grad.numpy(), rtol=rtol, atol=1e-6)


_ROLLS = {
    "no_dims": lambda lib, F, t: lib.roll(t, 2),
    "no_dims_negative": lambda lib, F, t: lib.roll(t, -1),
    "no_dims_method": lambda lib, F, t: t.roll(1),
    "int_dims": lambda lib, F, t: lib.roll(t, 1, 0),
    "tuple_dims": lambda lib, F, t: lib.roll(t, (1,), (-1,)),
}


@pytest.mark.parametrize("shape", [(), (5,), (2, 3, 4)], ids=str)
@pytest.mark.parametrize("call", list(_ROLLS))
def test_roll_matches_reference(ref, device, call, shape):
    if not shape and call in ("int_dims", "tuple_dims"):
        # The reference refuses a 0-d tensor's dim for ``roll``: only the
        # flat roll reaches it.
        with pytest.raises(IndexError):
            _ROLLS[call](ref, None, ref.tensor(1.5))
        with pytest.raises(IndexError):
            _ROLLS[call](lucid, None, lucid.tensor(1.5, device=device))
        return
    _compare(ref, device, shape, _ROLLS[call])


@pytest.mark.parametrize(
    ("shifts", "dims"), [((1, 2), None), ((1, 2), (0,))], ids=["no-dims", "misaligned"]
)
def test_roll_refuses_shifts_that_do_not_align(ref, shifts, dims):
    with pytest.raises(RuntimeError):
        ref.roll(ref.zeros(2, 3), shifts, dims)
    with pytest.raises(ValueError):
        lucid.roll(lucid.zeros(2, 3), shifts, dims)


_NORMALIZE_DIMS = [1, -1, (1,), (0, -1), [0, 1], None]


@pytest.mark.parametrize("p", [2.0, 1.0])
@pytest.mark.parametrize("dim", _NORMALIZE_DIMS, ids=str)
@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)], ids=str)
def test_normalize_takes_every_dim_spelling(ref, device, shape, dim, p):
    _compare(ref, device, shape, lambda lib, F, t: F.normalize(t, p=p, dim=dim))


@pytest.mark.parametrize("dim", [0, -1, (0,), None], ids=str)
def test_normalize_on_a_0d_tensor(ref, device, dim):
    _compare(ref, device, (), lambda lib, F, t: F.normalize(t, dim=dim))


@pytest.mark.parametrize("shape", [(), (2, 3)], ids=str)
def test_glu_refuses_an_odd_halved_dim(ref, device, shape):
    """An odd split broadcast its unequal halves into a wrong answer."""
    with pytest.raises(RuntimeError):
        ref.nn.functional.glu(ref.ones(shape), -1)
    with pytest.raises(ValueError, match="must have an even size"):
        F.glu(lucid.ones(shape, device=device), -1)


def test_glu_matches_reference(ref, device):
    # The gate's slope at 8 is ~3e-4, where float32 sigmoid' differs in
    # the fourth digit between the two libraries.
    _compare(ref, device, (2, 4), lambda lib, F, t: F.glu(t, -1), rtol=1e-3)


_BOOL_DIMS = {
    "normalize": lambda lib, t: lib.nn.functional.normalize(t, dim=True),
    "sum": lambda lib, t: lib.sum(t, dim=True),
    "sum_tuple": lambda lib, t: lib.sum(t, dim=(True,)),
    "squeeze": lambda lib, t: lib.squeeze(t, True),
    "roll_shifts": lambda lib, t: lib.roll(t, True, 0),
    "roll_dims": lambda lib, t: lib.roll(t, 1, True),
}


@pytest.mark.parametrize("op", list(_BOOL_DIMS))
def test_a_bool_dim_is_refused(ref, op):
    """``dim=True`` used to pass as axis 1."""
    with pytest.raises(TypeError):
        _BOOL_DIMS[op](ref, ref.ones(2, 3))
    with pytest.raises(TypeError):
        _BOOL_DIMS[op](lucid, lucid.ones(2, 3))


def test_roll_takes_any_index_type(ref, device):
    """A NumPy integer is an index for shifts and dims alike."""
    shift, axis = np.int64(1), np.int64(-1)
    _compare(ref, device, (2, 3), lambda lib, F, t: lib.roll(t, shift, axis))
    _compare(ref, device, (2, 3), lambda lib, F, t: lib.roll(t, (shift,), (axis,)))


# ── LCD-424: diff, linalg.cross, take_along_dim, quantile, sort ─────────────


@pytest.mark.parametrize("dim", [0, -1])
def test_diff_refuses_a_0d_tensor_and_negative_order(ref, device, dim):
    with pytest.raises(RuntimeError):
        ref.diff(ref.tensor(1.0), dim=dim)
    with pytest.raises(RuntimeError, match="at least one-dimensional"):
        lucid.diff(lucid.tensor(1.0, device=device), dim=dim)
    with pytest.raises(RuntimeError, match="non-negative"):
        lucid.tensor([1.0, 2.0], device=device).diff(n=-1, dim=dim)


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_diff_matches_reference(ref, device, n, dim):
    _compare(ref, device, (3, 4), lambda lib, F, t: t.diff(n=n, dim=dim))


@pytest.mark.parametrize("dim", [0, -1, 1, -2])
def test_cross_refuses_0d_and_non_3_axes(ref, device, dim):
    with pytest.raises(IndexError):
        ref.linalg.cross(ref.tensor(1.0), ref.tensor(1.0), dim=dim)
    with pytest.raises(IndexError):
        lucid.linalg.cross(
            lucid.tensor(1.0, device=device), lucid.tensor(1.0, device=device), dim=dim
        )
    if dim in (0, -1, 1, -2):
        with pytest.raises(RuntimeError):
            ref.linalg.cross(ref.ones(2, 2), ref.ones(2, 2), dim=dim)
        with pytest.raises(RuntimeError, match="must have length 3"):
            lucid.linalg.cross(
                lucid.ones(2, 2, device=device),
                lucid.ones(2, 2, device=device),
                dim=dim,
            )


def test_take_along_dim_on_a_0d_tensor(ref, device):
    x = lucid.tensor(5.0, device=device, requires_grad=True)
    xr = ref.tensor(5.0, requires_grad=True)
    got = lucid.take_along_dim(
        x, lucid.tensor(0, dtype=lucid.int64, device=device), dim=None
    )
    want = ref.take_along_dim(xr, ref.tensor(0, dtype=ref.int64), dim=None)
    assert tuple(got.shape) == tuple(want.shape) == (1,)
    np.testing.assert_allclose(got.numpy(), want.detach().numpy())
    got.sum().backward()
    want.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), xr.grad.numpy())
    for dim in (0, -1, 1, -2):
        with pytest.raises(IndexError):
            ref.take_along_dim(ref.tensor(5.0), ref.tensor(0), dim=dim)
        with pytest.raises(IndexError):
            lucid.take_along_dim(
                lucid.tensor(5.0, device=device),
                lucid.tensor(0, dtype=lucid.int64, device=device),
                dim=dim,
            )


@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        ((0,), None),
        ((0,), 0),
        ((0,), -1),
        ((0, 1), None),
        ((0, 1), 0),
        ((0, 1), 1),
        ((2, 0), 0),
        ((2, 0), 1),
    ],
    ids=str,
)
@pytest.mark.parametrize("op", ["quantile", "nanquantile"])
def test_quantile_refuses_an_empty_tensor(ref, device, op, shape, dim):
    with pytest.raises(RuntimeError, match="non-empty"):
        getattr(ref, op)(ref.zeros(shape), 0.5, dim=dim)
    with pytest.raises((RuntimeError, ValueError), match="non-empty"):
        getattr(lucid, op)(lucid.zeros(shape, device=device), 0.5, dim=dim)


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dim", [None, 0, -1], ids=str)
@pytest.mark.parametrize("op", ["quantile", "nanquantile"])
def test_quantile_on_a_0d_tensor(ref, device, op, dim, keepdim):
    _compare(
        ref,
        device,
        (),
        lambda lib, F, t: getattr(lib, op)(t, 0.5, dim=dim, keepdim=keepdim),
    )
    nan_got = getattr(lucid, op)(
        lucid.tensor(float("nan"), device=device), 0.5, dim=dim, keepdim=keepdim
    )
    assert nan_got.shape == () and np.isnan(nan_got.numpy())


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dim", [0, -1])
def test_sort_and_argsort_on_a_0d_tensor(ref, device, dim, descending):
    _compare(
        ref,
        device,
        (),
        lambda lib, F, t: lib.sort(t, dim=dim, descending=descending)[0],
    )
    _compare(
        ref,
        device,
        (),
        lambda lib, F, t: t.sort(dim=dim, descending=descending)[0],
    )
    x = lucid.tensor(1.5, device=device)
    sort_res = lucid.sort(x, dim=dim, descending=descending)
    assert sort_res.indices.shape == () and sort_res.indices.dtype == lucid.int64
    assert sort_res.indices.item() == 0
    assert x.sort(dim=dim, descending=descending).indices.item() == 0
    got_idx = lucid.argsort(x, dim=dim, descending=descending)
    want_idx = ref.argsort(ref.tensor(1.5), dim=dim, descending=descending)
    assert got_idx.shape == () and got_idx.dtype == lucid.int64
    assert got_idx.item() == want_idx.item() == 0


def test_matrix_norm_refuses_out_of_range_dims(ref, device) -> None:
    """LCD-263: ``linalg.matrix_norm`` wraps its axis pair through ``_wrap_dim``
    instead of ``% rank``."""
    x = lucid.ones(2, 3, device=device)
    xr = ref.ones(2, 3)
    for bad_dim in ((0, 2), (-3, 1)):
        with pytest.raises(IndexError, match="axis out of range"):
            lucid.linalg.matrix_norm(x, dim=bad_dim)
        with pytest.raises(IndexError):
            ref.linalg.matrix_norm(xr, dim=bad_dim)
