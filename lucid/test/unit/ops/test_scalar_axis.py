"""A dim argument on a 0-d tensor (LCD-249).

The reference gives a 0-d tensor one notional axis: ``dim`` 0 and -1 name
it, and the answer is 0-d again, with or without ``keepdim``.  Every
reduction refused it ("normalize_axes: axis out of range: 0 for ndim=0"),
and so did sort, argsort, topk, kthvalue, the cumulative scans, softmax,
flip, transpose, movedim, gather, index_select, the scatters and
``F.normalize``, each through its own copy of the axis rule.  The rule
now has one owner (``wrap_dim`` in the engine), and the ops that need a
real axis run on the scalar viewed as shape (1,).  ``narrow`` is not here:
the reference refuses a 0-d input to it too.

Every op that takes a dim is held here to the reference on a 0-d input:
each dim spelling, with and without ``keepdim``, value, dtype and shape,
and the gradient where the op has one.  A spelling the reference refuses
by signature (a tuple where it takes an int) is not compared.  A dim past
the notional axis must be refused by the engine's rule, not by a wrapper
that happens to index past a shape: the Python wrappers resolve a dim by
the engine's rule too (``lucid._ops._dims``), so they are held to the same
matrix.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg


def _first(out):
    return out[0] if isinstance(out, tuple) else out


def _like(lib, t, value):
    """An index or src operand on ``t``'s device."""
    dtype = lib.int64 if isinstance(value, (int, list)) else t.dtype
    return lib.tensor(value, dtype=dtype, device=t.device)


def _reduce(name):
    def call(lib, t, d, k):
        fn = getattr(lib, name)
        return fn(t) if d is None else fn(t, dim=d, keepdim=k)

    return call


def _along(name):
    def call(lib, t, d, k):
        del k  # no keepdim in the signature
        return getattr(lib, name)(t, dim=d)

    return call


# name -> (call(lib, tensor, dim, keepdim), differentiable)
_OPS = {
    **{
        name: (_reduce(name), True)
        for name in ("sum", "mean", "prod", "amax", "amin", "nansum", "nanmean")
    },
    "max": (lambda lib, t, d, k: _first(_reduce("max")(lib, t, d, k)), True),
    "min": (lambda lib, t, d, k: _first(_reduce("min")(lib, t, d, k)), True),
    "logsumexp": (lambda lib, t, d, k: lib.logsumexp(t, dim=d, keepdim=k), True),
    "var": (lambda lib, t, d, k: lib.var(t, dim=d, keepdim=k, correction=0), True),
    "std": (lambda lib, t, d, k: lib.std(t, dim=d, keepdim=k, correction=0), True),
    "vector_norm": (lambda lib, t, d, k: lib.linalg.vector_norm(t, dim=d, keepdim=k), True),
    "count_nonzero": (lambda lib, t, d, k: lib.count_nonzero(t, dim=d), False),
    "argmax": (lambda lib, t, d, k: lib.argmax(t, dim=d, keepdim=k), False),
    "argmin": (lambda lib, t, d, k: lib.argmin(t, dim=d, keepdim=k), False),
    "cumsum": (_along("cumsum"), True),
    "cumprod": (_along("cumprod"), True),
    "cummax": (lambda lib, t, d, k: _first(_along("cummax")(lib, t, d, k)), True),
    "cummin": (lambda lib, t, d, k: _first(_along("cummin")(lib, t, d, k)), True),
    "sort": (lambda lib, t, d, k: _first(_along("sort")(lib, t, d, k)), True),
    "argsort": (_along("argsort"), False),
    "topk": (lambda lib, t, d, k: lib.topk(t, 1, dim=d)[0], True),
    "topk_indices": (lambda lib, t, d, k: lib.topk(t, 1, dim=d)[1], False),
    "topk_k0": (lambda lib, t, d, k: lib.topk(t, 0, dim=d)[0], True),
    "kthvalue": (lambda lib, t, d, k: _first(lib.kthvalue(t, 1, dim=d, keepdim=k)), True),
    "softmax": (_along("softmax"), True),
    "log_softmax": (_along("log_softmax"), True),
    "flip": (lambda lib, t, d, k: lib.flip(t, dims=(d,) if isinstance(d, int) else d), True),
    "argsort_descending": (lambda lib, t, d, k: lib.argsort(t, dim=d, descending=True), False),
    "topk_smallest": (lambda lib, t, d, k: lib.topk(t, 1, dim=d, largest=False)[0], True),
    "quantile": (lambda lib, t, d, k: lib.quantile(t, 0.5, dim=d, keepdim=k), True),
    "transpose": (lambda lib, t, d, k: t.transpose(d, -1), True),
    "swapaxes": (lambda lib, t, d, k: lib.swapaxes(t, d, 0), True),
    "movedim": (lambda lib, t, d, k: lib.movedim(t, d, 0), True),
    "gather": (lambda lib, t, d, k: lib.gather(t, d, _like(lib, t, 0)), True),
    "index_select": (lambda lib, t, d, k: lib.index_select(t, d, _like(lib, t, [0])), True),
    "index_select_0d_index": (lambda lib, t, d, k: lib.index_select(t, d, _like(lib, t, 0)), True),
    "scatter": (lambda lib, t, d, k: lib.scatter(t, d, _like(lib, t, 0), _like(lib, t, 2.0)), True),
    "scatter_add": (
        lambda lib, t, d, k: lib.scatter_add(t, d, _like(lib, t, 0), _like(lib, t, 2.0)),
        True,
    ),
    "scatter_reduce_amax": (
        lambda lib, t, d, k: lib.scatter_reduce(t, d, _like(lib, t, 0), _like(lib, t, 2.0), "amax"),
        True,
    ),
    "normalize": (lambda lib, t, d, k: lib.nn.functional.normalize(t, dim=d), True),
}

_DIMS = [0, -1, (0,), None]


def _signature_refuses(ref, call, d, k):
    try:
        call(ref, ref.tensor(1.5), d, k)
    except TypeError:
        return True
    return False


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dim", _DIMS, ids=str)
@pytest.mark.parametrize("op", list(_OPS))
def test_matches_reference(ref, device, op, dim, keepdim):
    call, differentiable = _OPS[op]
    if _signature_refuses(ref, call, dim, keepdim):
        pytest.skip("the reference's signature does not take this dim spelling")
    x = lucid.tensor(1.5, device=device, requires_grad=differentiable)
    xr = ref.tensor(1.5, requires_grad=differentiable)
    got = call(lucid, x, dim, keepdim)
    want = call(ref, xr, dim, keepdim)
    assert tuple(got.shape) == tuple(want.shape) == ()
    assert str(got.dtype).rsplit(".", 1)[-1] == str(want.dtype).rsplit(".", 1)[-1]
    np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=1e-6)
    if not differentiable:
        return
    got.sum().backward()
    want.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), xr.grad.numpy(), rtol=1e-6)


@pytest.mark.parametrize("dim", [1, -2])
@pytest.mark.parametrize("op", list(_OPS))
def test_an_axis_past_the_notional_one_is_refused(ref, device, op, dim):
    """Refused by the engine's axis rule, not by a wrapper tripping over it."""
    call, _ = _OPS[op]
    with pytest.raises(IndexError):
        call(ref, ref.tensor(1.5), dim, False)
    with pytest.raises(IndexError, match=r"axis out of range: got -?\d"):
        call(lucid, lucid.tensor(1.5, device=device), dim, False)


def test_an_integer_scan_widens_as_on_any_rank(ref, device):
    got = lucid.cumsum(lucid.tensor(3, dtype=lucid.int32, device=device), dim=0)
    want = ref.cumsum(ref.tensor(3, dtype=ref.int32), dim=0)
    assert got.dtype == lucid.int64 and want.dtype == ref.int64
    assert got.item() == want.item() == 3


@pytest.mark.parametrize(
    "call",
    [
        lambda lib, t: lib.argsort(t, dim=-1),
        lambda lib, t: lib.argsort(t, dim=0),
        lambda lib, t: lib.topk(t, 2, dim=-1)[1],
    ],
    ids=["argsort", "argsort-dim0", "topk"],
)
def test_sort_indices_are_int64(ref, device, call):
    """The index dtype of every index-returning op, as the reference's."""
    values = np.array([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]], dtype=np.float32)
    got = call(lucid, lucid.tensor(values, device=device))
    want = call(ref, ref.tensor(values))
    assert got.dtype == lucid.int64
    np.testing.assert_array_equal(got.numpy(), want.numpy())
