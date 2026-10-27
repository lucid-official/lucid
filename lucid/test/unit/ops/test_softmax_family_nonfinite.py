"""The softmax family on rows whose max is not finite (LCD-286).

A max-shift reduction subtracts the row max before ``exp``.  When that max
is infinite, ``inf - inf`` is NaN and poisons the row, so each op has to
decide what an infinite max means — and the two devices decided differently:

* ``logsumexp`` answered NaN on both devices for a row holding +inf (the
  reference: +inf) and for an all -inf row (the reference: -inf).  The
  reference shifts an infinite max by 0; so does Lucid now.
* ``log_softmax`` on Metal went through MLX's guarded ``logsumexp`` and
  answered ``[-inf, nan, -inf]`` for ``[1, inf, 2]``, where the reference
  and the CPU shift by the raw max and answer all NaN — as ``softmax``
  does on every device.  Its backward read the -inf back as a probability
  of 0 and passed finite gradients, where the reference's are NaN.

Every row is checked against the reference in its own dtype, forward and
backward, and on both devices, so the two devices agree because each
agrees with the reference.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

_INF, _NAN = math.inf, math.nan

_ROWS = {
    "finite": [1.0, 2.0, 3.0],
    "pos_inf": [1.0, _INF, 2.0],
    "pos_and_neg_inf": [_INF, 1.0, -_INF],
    "two_pos_inf": [_INF, _INF, 1.0],
    "neg_inf_finite_max": [-_INF, 1.0, 2.0],
    "all_neg_inf": [-_INF, -_INF, -_INF],
    "nan": [1.0, _NAN, 2.0],
    "nan_leading": [_NAN, 1.0, 2.0],
    "nan_and_pos_inf": [1.0, _NAN, _INF],
}

_OPS = {
    "softmax": (lambda x, d: F.softmax(x, d), lambda r, x, d: r.softmax(x, d)),
    "log_softmax": (
        lambda x, d: F.log_softmax(x, d),
        lambda r, x, d: r.log_softmax(x, d),
    ),
    "logsumexp": (
        lambda x, d: lucid.logsumexp(x, d),
        lambda r, x, d: r.logsumexp(x, d),
    ),
}

_DTYPES = {"float32": 1e-6, "float64": 1e-12, "float16": 2e-3, "bfloat16": 2e-2}


def _cases():
    devices = ["cpu", "metal"] if metal_available() else ["cpu"]
    for device in devices:
        for dtype in _DTYPES:
            if device == "metal" and dtype == "float64":
                continue
            for op in _OPS:
                for row in _ROWS:
                    yield pytest.param(device, dtype, op, row, id=f"{device}-{dtype}-{op}-{row}")


def _as_f64(t):
    return t.detach().float().numpy().astype(np.float64)


def _inputs(ref, row, dtype, layout, device, requires_grad=False):
    values = np.array(_ROWS[row], dtype=np.float64)
    # ``leading_axis`` puts the row down a column, so the reduction runs
    # over a strided axis (the CPU's inner > 1 path) beside a finite column.
    dim = -1
    if layout == "leading_axis":
        values = np.stack([values, np.arange(3, dtype=np.float64)], axis=1)
        dim = 0
    x = lucid.tensor(
        values, dtype=getattr(lucid, dtype), device=device, requires_grad=requires_grad
    )
    xr = ref.tensor(values).to(getattr(ref, dtype)).requires_grad_(requires_grad)
    return x, xr, dim


def _check(got, want, dtype):
    tol = _DTYPES[dtype]
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want)), (got, want)
    assert np.array_equal(np.isinf(got) & (got > 0), np.isinf(want) & (want > 0)), (
        got,
        want,
    )
    assert np.array_equal(np.isinf(got) & (got < 0), np.isinf(want) & (want < 0)), (
        got,
        want,
    )
    finite = np.isfinite(want)
    np.testing.assert_allclose(got[finite], want[finite], rtol=tol, atol=tol)


@pytest.mark.parametrize(("device", "dtype", "op", "row"), list(_cases()))
@pytest.mark.parametrize("layout", ["last_axis", "leading_axis"])
def test_matches_reference(ref, device, dtype, op, row, layout):
    x, xr, dim = _inputs(ref, row, dtype, layout, device)
    ours, theirs = _OPS[op]
    _check(_as_f64(ours(x, dim)), _as_f64(theirs(ref, xr, dim)), dtype)


@pytest.mark.parametrize(("device", "dtype", "op", "row"), list(_cases()))
@pytest.mark.parametrize("layout", ["last_axis", "leading_axis"])
def test_gradient_matches_reference(ref, device, dtype, op, row, layout):
    # A non-uniform upstream gradient, so ``sum(g)`` and ``sum(y * g)`` in
    # the backward formulas are neither zero nor proportional to the output.
    x, xr, dim = _inputs(ref, row, dtype, layout, device, requires_grad=True)
    ours, theirs = _OPS[op]
    y = ours(x, dim)
    yr = theirs(ref, xr, dim)
    # ``tolist`` because a 0-d ndarray given a dtype comes back with shape
    # (1,) from ``lucid.tensor`` — the reduced ``logsumexp`` output is 0-d.
    g = np.linspace(-1.0, 2.0, int(np.prod(yr.shape))).reshape(tuple(yr.shape)).tolist()
    y.backward(lucid.tensor(g, dtype=getattr(lucid, dtype), device=device))
    yr.backward(ref.tensor(g, dtype=ref.float64).to(getattr(ref, dtype)))
    _check(_as_f64(x.grad), _as_f64(xr.grad), dtype)


def test_cross_entropy_nonfinite_and_parallel_layer_norm_determinism(ref) -> None:
    """Guard for LCD-154: cross_entropy on non-finite rows and bitwise-deterministic
    multi-chunk CPU layer_norm / rms_norm backward."""
    logits = [
        [_INF, 1.0, 2.0],
        [-_INF, -_INF, -_INF],
        [-_INF, 1.0, 2.0],
        [1.0, 2.0, 3.0],
    ]
    target = [1, 0, 2, 1]
    x = lucid.tensor(logits, dtype=lucid.float32, requires_grad=True)
    t = lucid.tensor(target, dtype=lucid.int64)
    xr = ref.tensor(logits, dtype=ref.float32, requires_grad=True)
    tr = ref.tensor(target, dtype=ref.int64)
    loss = F.cross_entropy(x, t, reduction="none")
    rloss = ref.nn.functional.cross_entropy(xr, tr, reduction="none")
    _check(_as_f64(loss), _as_f64(rloss), "float32")
    loss.sum().backward()
    rloss.sum().backward()
    _check(_as_f64(x.grad), _as_f64(xr.grad), "float32")

    lucid.manual_seed(0)
    ln_x = lucid.randn(512, 256, requires_grad=True)
    ln_g = lucid.randn(512, 256)
    ln_w = lucid.randn(256, requires_grad=True)
    ln_b = lucid.randn(256, requires_grad=True)
    y1 = F.layer_norm(ln_x, (256,), ln_w, ln_b)
    y1.backward(ln_g)
    dx1 = ln_x.grad.numpy().copy()
    dw1 = ln_w.grad.numpy().copy()
    db1 = ln_b.grad.numpy().copy()
    ln_x.grad = None
    ln_w.grad = None
    ln_b.grad = None
    y2 = F.layer_norm(ln_x, (256,), ln_w, ln_b)
    y2.backward(ln_g)
    assert np.array_equal(y1.detach().numpy(), y2.detach().numpy())
    assert np.array_equal(dx1, ln_x.grad.numpy())
    assert np.array_equal(dw1, ln_w.grad.numpy())
    assert np.array_equal(db1, ln_b.grad.numpy())
