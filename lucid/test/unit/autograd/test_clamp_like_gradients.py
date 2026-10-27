"""Composites that stand in for a clamp keep the clamp's gradient (LCD-436).

``maximum``/``minimum`` split a tie between their operands, as the reference
does, so a composite that used them as a floor — the one-sided ``clamp``,
``fmax``/``fmin``, a hinge, the guard that keeps a special function's
losing branch finite — handed only half the derivative to an input sitting
exactly on the bound.  Each now follows the rule of the op it stands for:

- ``clamp`` (scalar or tensor bounds): a tie goes to ``x``; at a NaN on
  either side nobody gets a gradient.
- ``fmax``/``fmin``: a tie goes to the first operand; a NaN's gradient goes
  to the other side.
- the special functions: the analytic derivative at the cutoff.

Values are planted in 16-element tensors, the reference's vectorised path.
"""

import math

import pytest

import lucid

_NAN = float("nan")
_INF = float("inf")
_BASE = [
    0.5,
    -1.0,
    2.0,
    0.25,
    -3.0,
    0.0,
    0.75,
    -0.5,
    1.25,
    0.1,
    -2.0,
    0.3,
    0.6,
    -0.7,
    0.9,
    1.1,
]
_SLOT = 5


def _planted(value, base=_BASE):
    v = list(base)
    v[_SLOT] = value
    return v


def _same(got, want, rel=1e-5, abs_=1e-6):
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if math.isnan(w):
            assert math.isnan(g), (i, got, want)
        else:
            assert g == pytest.approx(w, rel=rel, abs=abs_), (i, got, want)


def _grads(lib, call, columns, **kw):
    ts = [lib.tensor(c, requires_grad=True, **kw) for c in columns]
    call(lib, *ts).sum().backward()
    return [t.grad.tolist() for t in ts]


def _graph_grads(call, columns, device):
    ts = [lucid.tensor(c, device=device, requires_grad=True) for c in columns]
    return [g.tolist() for g in lucid.autograd.grad(call(lucid, *ts).sum(), ts, create_graph=True)]


# ── clamp with a scalar bound ───────────────────────────────────────────────

_SCALAR_CLAMPS = {
    "min": lambda lib, x: lib.clamp(x, min=0.0),
    "max": lambda lib, x: lib.clamp(x, max=0.0),
    "method_min": lambda lib, x: x.clamp(min=0.0),
    "both": lambda lib, x: lib.clamp(x, -0.0, 0.0),
}


@pytest.mark.parametrize("value", [0.0, -0.0, _NAN, _INF, -_INF, 1.0, -1.0], ids=str)
@pytest.mark.parametrize("name", list(_SCALAR_CLAMPS))
def test_scalar_clamp_gradient(ref, device, name, value):
    call = _SCALAR_CLAMPS[name]
    want = _grads(ref, call, [_planted(value)])
    _same(_grads(lucid, call, [_planted(value)], device=device)[0], want[0])
    _same(_graph_grads(call, [_planted(value)], device)[0], want[0])


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "int32"])
def test_scalar_clamp_keeps_values_across_dtypes(ref, device, dtype_name):
    vals = [-3, -1, 0, 1, 2, 5, 7, -8] * 2
    x = lucid.tensor(vals, dtype=getattr(lucid, dtype_name), device=device)
    r = ref.tensor(vals, dtype=getattr(ref, dtype_name))
    for kw in ({"min": 0}, {"max": 2}, {"min": -1, "max": 3}):
        got = lucid.clamp(x, **kw)
        want = ref.clamp(r, **kw)
        assert got.tolist() == want.tolist(), kw
        assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1], kw


# ── clamp with tensor bounds ────────────────────────────────────────────────

# (x, lo, hi) planted triples, including lo == hi, lo > hi and every NaN.
_TRIPLES = [
    (0.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
    (2.0, 1.0, 2.0),
    (3.0, 1.0, 2.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0),
    (0.0, 2.0, 1.0),
    (1.5, 2.0, 1.0),
    (3.0, 2.0, 1.0),
    (_NAN, 1.0, 2.0),
    (0.0, _NAN, 2.0),
    (0.0, 1.0, _NAN),
]
_LO_BASE = [v - 1.0 for v in _BASE]
_HI_BASE = [v + 1.0 for v in _BASE]

_TENSOR_CLAMPS = {
    "both": lambda lib, x, lo, hi: lib.clamp(x, lo, hi),
    "min": lambda lib, x, lo, hi: lib.clamp(x, min=lo) + hi * 0,
    "max": lambda lib, x, lo, hi: lib.clamp(x, max=hi) + lo * 0,
}


@pytest.mark.parametrize("triple", _TRIPLES, ids=str)
@pytest.mark.parametrize("name", list(_TENSOR_CLAMPS))
def test_tensor_bound_clamp_matches_reference(ref, device, name, triple):
    call = _TENSOR_CLAMPS[name]
    columns = [
        _planted(triple[0]),
        _planted(triple[1], _LO_BASE),
        _planted(triple[2], _HI_BASE),
    ]
    rx, rlo, rhi = (ref.tensor(c) for c in columns)
    lx, llo, lhi = (lucid.tensor(c, device=device) for c in columns)
    _same(call(lucid, lx, llo, lhi).tolist(), call(ref, rx, rlo, rhi).tolist())
    want = _grads(ref, call, columns)
    for got, exp in zip(_grads(lucid, call, columns, device=device), want, strict=True):
        _same(got, exp)
    for got, exp in zip(_graph_grads(call, columns, device), want, strict=True):
        _same(got, exp)


def test_tensor_bound_clamp_broadcasts(ref, device):
    """A per-channel bound collects every row's share."""
    vals = [[0.0, 1.0, 2.0, 3.0], [1.0, -1.0, 2.5, 0.5], [2.0, 2.0, 0.0, 9.0]]

    def run(lib, **kw):
        x = lib.tensor(vals, requires_grad=True, **kw)
        lo = lib.tensor([1.0, 0.0, 1.0, 0.5], requires_grad=True, **kw)
        hi = lib.tensor(2.0, requires_grad=True, **kw)
        lib.clamp(x, lo, hi).sum().backward()
        return x.grad.tolist(), lo.grad.tolist(), hi.grad.tolist()

    want = run(ref)
    got = run(lucid, device=device)
    assert got == want


# ── fmax / fmin ─────────────────────────────────────────────────────────────

_PAIRS = [
    (1.0, 1.0),
    (-0.0, 0.0),
    (0.0, -0.0),
    (_INF, _INF),
    (_NAN, 1.0),
    (1.0, _NAN),
    (_NAN, _NAN),
    (2.0, 1.0),
    (1.0, 2.0),
]


@pytest.mark.parametrize("pair", _PAIRS, ids=str)
@pytest.mark.parametrize("op", ["fmax", "fmin"])
def test_fmax_fmin_gradient(ref, device, op, pair):
    columns = [_planted(pair[0]), _planted(pair[1], [v * 0.5 - 0.1 for v in _BASE])]

    def call(lib, a, b):
        return getattr(lib, op)(a, b)

    want = _grads(ref, call, columns)
    for got, exp in zip(_grads(lucid, call, columns, device=device), want, strict=True):
        _same(got, exp)
    for got, exp in zip(_graph_grads(call, columns, device), want, strict=True):
        _same(got, exp)


# ── losses built on a hinge ─────────────────────────────────────────────────


def test_triplet_margin_hinge_at_zero(ref, device):
    """``margin + d_ap - d_an`` exactly 0 is clamp_min's tie: the anchor
    keeps the whole gradient."""
    a = [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]] * 4
    p = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [3.0, 2.0]] * 4
    n = [[2.0, 0.0], [1.0, 3.0], [0.0, 4.0], [3.0, 4.0]] * 4

    def run(lib, **kw):
        ta = lib.tensor(a, requires_grad=True, **kw)
        loss = lib.nn.functional.triplet_margin_loss(
            ta, lib.tensor(p, **kw), lib.tensor(n, **kw), margin=1.0, eps=0.0
        )
        loss.backward()
        return ta.grad.reshape(-1).tolist()

    _same(run(lucid, device=device), run(ref), rel=1e-4, abs_=1e-5)


# ── special functions at their branch cutoffs ───────────────────────────────

_CUTOFFS = {
    "erfcx": (lambda lib, x: lib.special.erfcx(x), 2.0),
    "i0": (lambda lib, x: lib.i0(x), 3.75),
    "i0e": (lambda lib, x: lib.special.i0e(x), 3.75),
    "i1": (lambda lib, x: lib.special.i1(x), 3.75),
    "i1e": (lambda lib, x: lib.special.i1e(x), 3.75),
    "log_ndtr_low": (lambda lib, x: lib.special.log_ndtr(x), -1.0),
    "lgamma_half": (lambda lib, x: lib.lgamma(x), 0.5),
}


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_CUTOFFS))
def test_special_function_derivative_at_cutoff(ref, device, name, dtype_name):
    if device == "metal" and dtype_name == "float64":
        pytest.skip("Metal has no float64")
    call, cutoff = _CUTOFFS[name]
    vals = _planted(cutoff, [abs(v) + 0.5 for v in _BASE])

    def run(lib, **kw):
        x = lib.tensor(vals, dtype=getattr(lib, dtype_name), requires_grad=True, **kw)
        call(lib, x).sum().backward()
        return x.grad[_SLOT].item()

    assert run(lucid, device=device) == pytest.approx(run(ref), rel=2e-3, abs=1e-5)


# The reference's Bessel J has no backward; its derivative is held to the
# identities J0' = -J1 and J1' = J0 - J1 / x, read off the reference.
_BESSEL = {
    "bessel_j0": (
        lambda lib, x: lib.special.bessel_j0(x),
        lambda ref, x: -ref.special.bessel_j1(x),
    ),
    "bessel_j1": (
        lambda lib, x: lib.special.bessel_j1(x),
        lambda ref, x: ref.special.bessel_j0(x) - ref.special.bessel_j1(x) / x,
    ),
}


@pytest.mark.parametrize("cutoff", [3.0, 5.0, 8.0])
@pytest.mark.parametrize("name", list(_BESSEL))
def test_bessel_derivative_at_cutoff(ref, device, name, cutoff):
    call, derivative = _BESSEL[name]
    vals = _planted(cutoff, [abs(v) + 0.5 for v in _BASE])
    x = lucid.tensor(vals, device=device, requires_grad=True)
    call(lucid, x).sum().backward()
    want = derivative(ref, ref.tensor(vals))[_SLOT].item()
    assert x.grad[_SLOT].item() == pytest.approx(want, rel=2e-3, abs=1e-5)
