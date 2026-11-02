"""maximum / minimum gradients at ties, NaN, infinity and signed zero (LCD-436).

The reference splits a tie between the operands, ``-0`` against ``+0`` and
``inf`` against ``inf`` included, and when either side is NaN it gives
*both* operands the whole gradient: its formula masks a loser out with a
strict comparison, and a NaN loses no comparison.  Lucid sent a tie wholly
to the first operand and dropped a NaN pair's gradient altogether; the tie
rule also doubled ``hypot``'s gradient where ``|a| == |b|``.

The oracle is the reference's vectorised path: every pair is planted in a
16-element tensor (see ``debug-reference-scalar-vs-vector-path``).  Each
case runs through ``backward()``, ``grad(create_graph=True)`` and a second
derivative on both devices.
"""

import math

import pytest

import lucid

_NAN = float("nan")
_INF = float("inf")

_WIDTH = 16
_SLOT = 5
# Background values that never tie between the operands.
_BASE_A = [0.5, -1.0, 2.0, 0.25, -3.0, 0.0, 0.75, -0.5, 1.25, 0.1, -2.0, 0.3, 0.6, -0.7, 0.9, 1.1]
_BASE_B = [1.5, -2.0, 0.5, 1.25, -1.0, 0.0, -0.75, 0.5, 0.25, 0.9, -0.2, 0.8, -0.6, 0.7, 1.9, -1.1]
# Distinct weights so a gradient landing on the wrong slot shows.
_WEIGHTS = [float(i + 1) for i in range(_WIDTH)]

_PAIRS = {
    "tie": (1.0, 1.0),
    "nan_left": (_NAN, 1.0),
    "nan_right": (1.0, _NAN),
    "nan_both": (_NAN, _NAN),
    "inf_tie": (_INF, _INF),
    "ninf_tie": (-_INF, -_INF),
    "neg_zero_left": (-0.0, 0.0),
    "neg_zero_right": (0.0, -0.0),
    "inf_wins": (_INF, 1.0),
    "ninf_loses": (1.0, -_INF),
    "strict": (2.0, 1.0),
}
_OPS = ("maximum", "minimum")


def _skip_f64_on_metal(device, dtype_name):
    if device == "metal" and dtype_name == "float64":
        pytest.skip("Metal has no float64")


def _operands(lib, pair, dtype, **kw):
    a, b = list(_BASE_A), list(_BASE_B)
    a[_SLOT], b[_SLOT] = pair
    x = lib.tensor(a, dtype=dtype, requires_grad=True, **kw)
    y = lib.tensor(b, dtype=dtype, requires_grad=True, **kw)
    w = lib.tensor(_WEIGHTS, dtype=dtype, **kw)
    return x, y, w


def _same(got, want):
    assert len(got) == len(want)
    for i, (g, v) in enumerate(zip(got, want, strict=True)):
        if math.isnan(v):
            assert math.isnan(g), (i, got, want)
        else:
            assert g == pytest.approx(v, rel=1e-5, abs=1e-6), (i, got, want)


def _reference_first(ref, op, pair, dtype_name):
    x, y, w = _operands(ref, pair, getattr(ref, dtype_name))
    (getattr(ref, op)(x, y) * w).sum().backward()
    return x.grad.tolist(), y.grad.tolist()


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("pair", list(_PAIRS))
@pytest.mark.parametrize("op", _OPS)
def test_first_derivative_matches_reference(ref, device, op, pair, dtype_name):
    _skip_f64_on_metal(device, dtype_name)
    want_x, want_y = _reference_first(ref, op, _PAIRS[pair], dtype_name)
    dtype = getattr(lucid, dtype_name)

    x, y, w = _operands(lucid, _PAIRS[pair], dtype, device=device)
    (getattr(lucid, op)(x, y) * w).sum().backward()
    _same(x.grad.tolist(), want_x)
    _same(y.grad.tolist(), want_y)

    x, y, w = _operands(lucid, _PAIRS[pair], dtype, device=device)
    gx, gy = lucid.autograd.grad((getattr(lucid, op)(x, y) * w).sum(), (x, y), create_graph=True)
    _same(gx.tolist(), want_x)
    _same(gy.tolist(), want_y)
    # The masks are constants of the inputs: a linear loss leaves the
    # gradient detached, as in the reference.
    assert not gx.requires_grad and not gy.requires_grad


def _second(lib, op, pair, dtype, **kw):
    x, y, w = _operands(lib, pair, dtype, **kw)
    out = getattr(lib, op)(x, y)
    gx, gy = lib.autograd.grad((out * out * w).sum(), (x, y), create_graph=True)
    hx, hy = lib.autograd.grad((gx + gy).sum(), (x, y))
    return hx.tolist(), hy.tolist()


# A squared output takes the infinities to inf * 0 inside the reference's
# own second-derivative formula; the finite and NaN pairs carry the rule.
_FINITE = [name for name, pair in _PAIRS.items() if not any(math.isinf(v) for v in pair)]


@pytest.mark.parametrize("pair", _FINITE)
@pytest.mark.parametrize("op", _OPS)
def test_second_derivative_matches_reference(ref, device, op, pair):
    want = _second(ref, op, _PAIRS[pair], ref.float32)
    got = _second(lucid, op, _PAIRS[pair], lucid.float32, device=device)
    _same(got[0], want[0])
    _same(got[1], want[1])


@pytest.mark.parametrize("op", _OPS)
def test_broadcast_bound_collects_its_share(ref, device, op):
    """A 0-d operand broadcast against a tensor sums its share of every
    tie: ``maximum(x, 0)`` at ``x == 0`` hands the bound half."""
    vals = [0.0, 1.0, -1.0, 0.0, -0.0, 2.0, -2.0, 0.0] * 2

    def run(lib, **kw):
        x = lib.tensor(vals, requires_grad=True, **kw)
        z = lib.tensor(0.0, requires_grad=True, **kw)
        getattr(lib, op)(x, z).sum().backward()
        return x.grad.tolist(), z.grad.item()

    want_x, want_z = run(ref)
    got_x, got_z = run(lucid, device=device)
    _same(got_x, want_x)
    assert got_z == pytest.approx(want_z)


def test_hypot_at_equal_magnitudes(ref, device):
    """``hypot`` takes the maximum and the minimum of ``|a|`` and ``|b|``:
    with a tie sent wholly to ``a`` both reached it, doubling ``a``'s
    gradient and leaving ``b`` none."""

    def run(lib, **kw):
        a = lib.tensor([3.0, -2.0, 0.5, 4.0] * 4, requires_grad=True, **kw)
        b = lib.tensor([-3.0, 2.0, 0.5, 1.0] * 4, requires_grad=True, **kw)
        lib.hypot(a, b).sum().backward()
        return a.grad.tolist(), b.grad.tolist()

    want = run(ref)
    got = run(lucid, device=device)
    _same(got[0], want[0])
    _same(got[1], want[1])


_REDUCE_TIES = {
    "amax": lambda lib, x: lib.amax(x.reshape(4, 4), 1),
    "amin": lambda lib, x: lib.amin(x.reshape(4, 4), 1),
    "max_all": lambda lib, x: x.max(),
    "min_all": lambda lib, x: x.min(),
}
# ``max(dim)`` / ``min(dim)`` are the odd ones out: the whole gradient goes
# to the returned (first) extremum, not an even split.
for _op in ("max", "min"):
    for _dim in (0, 1, -1):
        for _keep in (False, True):
            _REDUCE_TIES[f"{_op}_dim{_dim}_keep{_keep}"] = (
                lambda lib, x, _op=_op, _dim=_dim, _keep=_keep: getattr(x.reshape(4, 4), _op)(
                    dim=_dim, keepdim=_keep
                )[0]
            )


@pytest.mark.parametrize("name", list(_REDUCE_TIES))
def test_reduction_ties_follow_the_reference(ref, device, name):
    """``amax``/``amin`` and the full ``max``/``min`` split a tie evenly
    across every extremum; ``max(dim)``/``min(dim)`` send it to the returned
    index.  Either way the gradient is the reference's."""
    vals = [5.0, 5.0, 1.0, 5.0, -2.0, -2.0, -2.0, 0.0, -0.0, 0.0, -1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    call = _REDUCE_TIES[name]

    def run(lib, **kw):
        x = lib.tensor(vals, requires_grad=True, **kw)
        call(lib, x).sum().backward()
        return x.grad.tolist()

    _same(run(lucid, device=device), run(ref))


# 3-d block with ties on every axis, signed zeros and NaN included.
_TIE_BLOCK = [
    [[1.0, 3.0, 3.0, 3.0], [2.0, 3.0, 0.0, 3.0], [3.0, 1.0, 3.0, 1.0]],
    [[-0.0, 0.0, -0.0, 2.0], [1.0, 1.0, 1.0, 1.0], [_NAN, 2.0, _NAN, 2.0]],
]


def _dim_extremum(lib, op, dim, keepdim, **kw):
    x = lib.tensor(_TIE_BLOCK, requires_grad=True, **kw)
    values, indices = getattr(x, op)(dim=dim, keepdim=keepdim)
    weights = lib.arange(float(values.numel()), **kw).reshape(*values.shape) + 1.0
    (values * weights).sum().backward()
    return x.grad.flatten().tolist(), indices.flatten().tolist()


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("op", ["max", "min"])
def test_dim_extremum_gradient_follows_the_returned_index(ref, device, op, dim, keepdim):
    """Weighted, so a gradient credited to a sibling tie shows."""
    want_grad, want_idx = _dim_extremum(ref, op, dim, keepdim)
    got_grad, got_idx = _dim_extremum(lucid, op, dim, keepdim, device=device)
    assert got_idx == want_idx
    _same(got_grad, want_grad)


@pytest.mark.parametrize("op", ["max", "min"])
def test_dim_extremum_values_are_the_indexed_elements(device, op):
    x = lucid.tensor(_TIE_BLOCK, device=device)
    values, indices = getattr(x, op)(dim=2)
    gathered = lucid.gather(x, 2, indices.unsqueeze(-1)).squeeze(-1)
    _same(values.flatten().tolist(), gathered.flatten().tolist())


def test_dim_extremum_on_a_scalar_and_an_integer_tensor(device):
    assert lucid.tensor(3.0, device=device).max(dim=0)[1].item() == 0
    values, indices = lucid.tensor([[1, 4, 4], [7, 0, 7]], device=device).max(dim=1)
    assert values.tolist() == [4, 7]
    assert indices.tolist() == [1, 0]


_TIE_POOL_SHAPE = {1: (1, 1, 7), 2: (1, 1, 5, 5), 3: (1, 1, 4, 4, 4)}

_POOL_CALLS = {
    "dilated": lambda F, n, x: getattr(F, f"max_pool{n}d")(x, 2, stride=1, dilation=2),
    "adaptive": lambda F, n, x: getattr(F, f"adaptive_max_pool{n}d")(x, (2,) * n),
    "adaptive_uneven": lambda F, n, x: getattr(F, f"adaptive_max_pool{n}d")(x, (3,) * n),
}


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("kind", list(_POOL_CALLS))
def test_pool_fallback_ties_go_to_the_first_maximum(ref, device, kind, n):
    """The dilated and uneven-adaptive max pools reduce windows in Python;
    on an all-equal input every window is a tie, and the gradient follows
    the first-maximum index ``return_indices`` reports."""
    shape = _TIE_POOL_SHAPE[n]

    def run(lib, **kw):
        x = lib.ones(*shape, requires_grad=True, **kw)
        _POOL_CALLS[kind](lib.nn.functional, n, x).sum().backward()
        return x.grad.flatten().tolist()

    _same(run(lucid, device=device), run(ref))


@pytest.mark.parametrize("name", ["cummax", "cummin"])
@pytest.mark.parametrize("dim", [0, 1])
def test_cumulative_extreme_values_gradient_follows_its_indices(ref, device, name, dim):
    """A running extreme's value is a copy of the element its index names,
    the later of a tie: ``cummax([1, 3, 3])`` credits all three."""
    block = [[1.0, 3.0, 3.0, 2.0], [2.0, 2.0, 5.0, 5.0], [2.0, 1.0, 1.0, 2.0]]

    def run(lib, **kw):
        x = lib.tensor(block, requires_grad=True, **kw)
        values, indices = getattr(lib, name)(x, dim)
        (values * (lib.arange(12.0, **kw).reshape(3, 4) + 1.0)).sum().backward()
        return x.grad.flatten().tolist(), indices.flatten().tolist()

    want_grad, want_idx = run(ref)
    got_grad, got_idx = run(lucid, device=device)
    assert got_idx == want_idx
    _same(got_grad, want_grad)
