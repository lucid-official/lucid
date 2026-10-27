"""Reductions over a dim tuple that mixes signs (LCD-36).

``logsumexp`` sorted the raw signed axes and squeezed them in that order,
so ``dim=(-3, 2)`` on a 3-D input squeezed axis 2 first and then asked for
axis -3 of a 2-D tensor: "squeeze: axis out of range".  It now goes through
the one axis normaliser every reduction kernel uses.  Every reduction that
takes a dim tuple is held to the same rule here: each ordering of each
signed spelling of one, two and three distinct axes of a 3-D input, with
and without ``keepdim``, against the reference — forward and, where the
op is differentiable, backward.
"""

import itertools

import numpy as np
import pytest

import lucid
import lucid.linalg

_NDIM = 3
_SHAPE = (2, 3, 4)


def _top_level(name, pick=None):
    """``lib.<name>(t, dim=d, keepdim=k)`` on either library, optionally indexed."""

    def call(lib, t, d, k):
        out = getattr(lib, name)(t, dim=d, keepdim=k)
        return out if pick is None else out[pick]

    return call


# name -> (call(lib, tensor, dims, keepdim), differentiable)
_OPS = {
    **{
        name: (_top_level(name), True)
        for name in (
            "sum",
            "mean",
            "amax",
            "amin",
            "var",
            "std",
            "logsumexp",
            "nansum",
            "nanmean",
        )
    },
    "var_mean": (_top_level("var_mean", pick=0), True),
    "std_mean": (_top_level("std_mean", pick=0), True),
    "vector_norm": (
        lambda lib, t, d, k: lib.linalg.vector_norm(t, dim=d, keepdim=k),
        True,
    ),
    # No keepdim in either signature.
    "count_nonzero": (lambda lib, t, d, k: lib.count_nonzero(t, dim=d), False),
}


def _signed_dim_tuples(arity):
    """Every ordering of every signed spelling of ``arity`` distinct axes."""
    signed = list(range(-_NDIM, _NDIM))
    return [
        dims
        for dims in itertools.permutations(signed, arity)
        if len({d % _NDIM for d in dims}) == arity
    ]


@pytest.mark.parametrize("arity", [1, 2, 3])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("op", list(_OPS))
def test_matches_reference(ref, device, op, keepdim, arity):
    call, differentiable = _OPS[op]
    rng = np.random.default_rng(36)
    values = rng.standard_normal(_SHAPE).astype(np.float32)
    failures = []
    for dims in _signed_dim_tuples(arity):
        x = lucid.tensor(values, device=device, requires_grad=differentiable)
        xr = ref.tensor(values, requires_grad=differentiable)
        try:
            y = call(lucid, x, dims, keepdim)
        except (IndexError, RuntimeError, ValueError) as exc:
            failures.append(f"{dims}: {type(exc).__name__}: {exc}")
            continue
        yr = call(ref, xr, dims, keepdim)
        if tuple(y.shape) != tuple(yr.shape):
            failures.append(f"{dims}: shape {tuple(y.shape)} != {tuple(yr.shape)}")
            continue
        if not np.allclose(y.detach().numpy(), yr.detach().numpy(), rtol=1e-5, atol=1e-5):
            failures.append(f"{dims}: forward values differ")
            continue
        if not differentiable:
            continue
        upstream = np.linspace(-1.0, 2.0, max(1, yr.numel())).reshape(tuple(yr.shape))
        y.backward(lucid.tensor(upstream.tolist(), device=device))
        yr.backward(ref.tensor(upstream.tolist()))
        if not np.allclose(x.grad.numpy(), xr.grad.numpy(), rtol=1e-4, atol=1e-5):
            failures.append(f"{dims}: gradient differs")
    assert not failures, failures
