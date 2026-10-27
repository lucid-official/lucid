"""Every engine op that takes a dim follows the one rule (LCD-423).

The rule has two halves, both in ``autograd/Helpers``: ``wrap_dim`` resolves
one signed dim (0-d answers to 0 and -1, anything else is an
``IndexError``), and ``distinct_dims`` refuses a list naming one axis twice
with the reference's ``RuntimeError``.  ``flatten`` wrapped its start and
end by hand — a 0-d input took any dim, an out-of-range one was a
``ValueError`` — and the reductions' ``normalize_axes`` deduplicated a
repeated dim without a word.  The FFTs left the duplicate to MLX, which
refused it with its own ``ValueError``.

The public reductions refuse a duplicate in Python before the engine sees
it, so the engine half is called directly here as well.
"""

import numpy as np
import pytest

import lucid
import lucid.fft
import lucid.linalg
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap


def _outcome(fn):
    try:
        return fn(), None
    except (IndexError, RuntimeError, ValueError) as err:
        return None, err


def _same_refusal(got, want):
    """``got`` is the error the reference's ``want`` is, by builtin class."""
    assert got is not None, f"Lucid accepted what the reference refused: {want}"
    if isinstance(want, IndexError):
        assert isinstance(got, IndexError), got
    else:
        assert isinstance(got, RuntimeError) and not isinstance(got, IndexError), got


# ── flatten ──────────────────────────────────────────────────────────────────

_FLATTEN_DIMS = [-4, -3, -2, -1, 0, 1, 2, 3]


@pytest.mark.parametrize("end", _FLATTEN_DIMS)
@pytest.mark.parametrize("start", _FLATTEN_DIMS)
@pytest.mark.parametrize("shape", [(), (3,), (2, 3, 4)], ids=str)
def test_flatten_matches_reference(ref, device, shape, start, end):
    x = lucid.arange(int(np.prod(shape)), dtype=lucid.float32, device=device).reshape(shape)
    xr = ref.arange(int(np.prod(shape)), dtype=ref.float32).reshape(shape)
    want, want_err = _outcome(lambda: ref.flatten(xr, start, end))
    got, got_err = _outcome(lambda: lucid.flatten(x, start, end))
    if want_err is not None:
        _same_refusal(got_err, want_err)
        return
    assert got_err is None, got_err
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_array_equal(got.numpy(), want.numpy())
    _, method_err = _outcome(lambda: x.flatten(start, end))
    assert method_err is None, method_err


# ── a repeated dim ───────────────────────────────────────────────────────────

# engine reduction -> the reference op with the same reduction.
_ENGINE_REDUCTIONS = {
    "sum": "sum",
    "mean": "mean",
    "max": "amax",
    "min": "amin",
    "prod": "sum",  # the reference's prod takes one dim; any list op refuses alike
    "var": "var",
    "std": "std",
    "logsumexp": "logsumexp",
}

_REPEATS = {(2, 3, 4): [(0, 0), (1, -2), (0, -3), (2, 0, -1)], (): [(0, 0), (0, -1)]}
_REPEAT_CASES = [(shape, dims) for shape, all_dims in _REPEATS.items() for dims in all_dims]


@pytest.mark.parametrize("shape,dims", _REPEAT_CASES, ids=str)
@pytest.mark.parametrize("op", list(_ENGINE_REDUCTIONS))
def test_engine_reduction_refuses_a_repeated_dim(ref, device, op, shape, dims):
    x = lucid.ones(shape, device=device)
    want_err = _outcome(lambda: getattr(ref, _ENGINE_REDUCTIONS[op])(ref.ones(shape), dims))[1]
    assert isinstance(want_err, RuntimeError)
    with pytest.raises(RuntimeError, match="appears multiple times") as info:
        getattr(_C_engine, op)(x._impl, list(dims), False)
    _same_refusal(info.value, want_err)


@pytest.mark.parametrize("op", list(_ENGINE_REDUCTIONS))
def test_engine_reduction_distinct_dims_still_reduce(ref, device, op):
    x = lucid.arange(24, dtype=lucid.float32, device=device).reshape(2, 3, 4)
    got = _wrap(getattr(_C_engine, op)(x._impl, [-1, 0], False))
    assert tuple(got.shape) == (3,)


_PUBLIC_REDUCTIONS = [
    "sum",
    "mean",
    "nansum",
    "nanmean",
    "amax",
    "amin",
    "logsumexp",
    "var",
    "std",
    "count_nonzero",
    "var_mean",
    "std_mean",
    "all",
    "any",
]


@pytest.mark.parametrize("shape,dims", _REPEAT_CASES, ids=str)
@pytest.mark.parametrize("op", _PUBLIC_REDUCTIONS)
def test_public_reduction_refuses_a_repeated_dim(ref, device, op, shape, dims):
    want_err = _outcome(lambda: getattr(ref, op)(ref.ones(shape), dims))[1]
    assert want_err is not None
    with pytest.raises(RuntimeError, match="appears multiple times"):
        getattr(lucid, op)(lucid.ones(shape, device=device), dims)


def test_public_reduction_repeated_dim_is_a_runtime_error(device):
    with pytest.raises(RuntimeError, match="appears multiple times"):
        lucid.sum(lucid.ones(2, 3, device=device), (0, 0))


@pytest.mark.parametrize(
    "call",
    [
        lambda lib, x: lib.fft.fftn(x, dim=(0, 0)),
        lambda lib, x: lib.fft.fft2(x, dim=(1, -1)),
        lambda lib, x: lib.fft.ifftn(x, dim=(0, -2)),
        lambda lib, x: lib.fft.rfftn(x, dim=(1, 1)),
        lambda lib, x: lib.fft.irfftn(x, dim=(0, 0)),
    ],
    ids=["fftn", "fft2", "ifftn", "rfftn", "irfftn"],
)
def test_fft_refuses_a_repeated_dim(ref, device, call):
    want_err = _outcome(lambda: call(ref, ref.ones(2, 3)))[1]
    got_err = _outcome(lambda: call(lucid, lucid.ones(2, 3, device=device)))[1]
    _same_refusal(got_err, want_err)


def test_vector_norm_refuses_a_repeated_dim_in_the_engine(device):
    x = lucid.ones(2, 3, device=device)
    with pytest.raises(RuntimeError, match="appears multiple times"):
        _C_engine.linalg.norm(x._impl, 2.0, [0, -2], False)


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("dims", [(0, 0), (0, -2)], ids=str)
def test_rot90_refuses_one_axis_twice(device, dims, k):
    with pytest.raises(RuntimeError, match="expected rotation dims to be different"):
        lucid.rot90(lucid.ones(2, 3, device=device), k, dims)


@pytest.mark.parametrize("k", [0, 1])
def test_rot90_refuses_bad_dims_length_or_input_rank(ref, device, k):
    for shape, dims in [((2, 3), (0,)), ((2, 3), (0, 1, 2)), ((3,), (0, 1)), ((), (0, 1))]:
        want_err = _outcome(lambda: ref.rot90(ref.ones(shape), k, dims))[1]
        got_err = _outcome(lambda: lucid.rot90(lucid.ones(shape, device=device), k, dims))[1]
        _same_refusal(got_err, want_err)


# ── the single-dim ops ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "case",
    [
        ("squeeze", (), lambda E, t: E.squeeze(t, 0), lambda r, t: r.squeeze(t, 0)),
        ("squeeze", (), lambda E, t: E.squeeze(t, -1), lambda r, t: r.squeeze(t, -1)),
        ("squeeze", (), lambda E, t: E.squeeze(t, 1), lambda r, t: r.squeeze(t, 1)),
        ("squeeze", (1, 3), lambda E, t: E.squeeze(t, -3), lambda r, t: r.squeeze(t, -3)),
        ("unsqueeze", (), lambda E, t: E.unsqueeze(t, 1), lambda r, t: r.unsqueeze(t, 1)),
        ("unsqueeze", (), lambda E, t: E.unsqueeze(t, -2), lambda r, t: r.unsqueeze(t, -2)),
        ("unsqueeze", (2,), lambda E, t: E.unsqueeze(t, 2), lambda r, t: r.unsqueeze(t, 2)),
        ("unsqueeze", (2,), lambda E, t: E.unsqueeze(t, 3), lambda r, t: r.unsqueeze(t, 3)),
        ("stack", (2,), lambda E, t: E.stack([t, t], 2), lambda r, t: r.stack([t, t], 2)),
        ("stack", (2,), lambda E, t: E.stack([t, t], -3), lambda r, t: r.stack([t, t], -3)),
        (
            "unflatten",
            (6,),
            lambda E, t: E.unflatten(t, 1, [6]),
            lambda r, t: r.unflatten(t, 1, (6,)),
        ),
        (
            "unflatten",
            (),
            lambda E, t: E.unflatten(t, 0, [1]),
            lambda r, t: r.unflatten(t, 0, (1,)),
        ),
        ("unfold", (), lambda E, t: E.unfold_dim(t, 0, 1, 1), lambda r, t: t.unfold(0, 1, 1)),
        ("unfold", (), lambda E, t: E.unfold_dim(t, -1, 1, 3), lambda r, t: t.unfold(-1, 1, 3)),
        ("unfold", (), lambda E, t: E.unfold_dim(t, 1, 1, 1), lambda r, t: t.unfold(1, 1, 1)),
        ("unfold", (), lambda E, t: E.unfold_dim(t, 0, 2, 1), lambda r, t: t.unfold(0, 2, 1)),
        ("unfold", (4,), lambda E, t: E.unfold_dim(t, -2, 2, 1), lambda r, t: t.unfold(-2, 2, 1)),
    ],
    ids=lambda c: c[0],
)
def test_single_dim_op_matches_reference(ref, device, case):
    _, shape, call, ref_call = case
    n = int(np.prod(shape))
    x = lucid.arange(n, dtype=lucid.float32, device=device).reshape(shape)
    xr = ref.arange(n, dtype=ref.float32).reshape(shape)
    want, want_err = _outcome(lambda: ref_call(ref, xr))
    got, got_err = _outcome(lambda: _wrap(call(_C_engine, x._impl)))
    if want_err is not None:
        _same_refusal(got_err, want_err)
        return
    assert got_err is None, got_err
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_array_equal(got.numpy(), want.numpy())
