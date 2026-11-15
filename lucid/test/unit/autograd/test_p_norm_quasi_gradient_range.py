"""LCD-1017: the quasi-norm gradient where ``N / m`` overflows float32 and the gradient does not.

``PNormBackward`` of an order below 1 (``quasi_grad``, LCD-1009) formed the slope's norm factor
``(N / m)^(1-p)`` as a power of the ratio ``N / m`` of the norm to the slice's largest magnitude.
The ratio is up to ``n^(1/p)`` (1e40 for 1e4 equal entries at ``p = 0.1``), past float32's
3.4e38, while the gradient ``n^((1-p)/p)`` of equal entries (1e36) is a good float: the gradient
came back ``inf`` on both streams, the norm itself right.  The factor is now ``(d q^(1-p)) q'^(1-p)``
with ``q = N / M``, ``q' = M / m`` and ``M = sqrt(N m)``, each factor of the ratio's square root,
and the exponent ``1 - p`` split into its float32 head and the tail the head loses (a power of a
base near 1e20 turns the rounding of the exponent alone into 20 ulp).

The closed form is plain Python float64.  For ``n`` equal entries the gradient is ``n^((1-p)/p)``
at every magnitude (``N = c n^(1/p)``, ``(c / N)^(p-1)``), so the cases below pick the magnitudes
at which the norm is finite and the cotangent keeps the gradient finite.
"""

import math
import struct

import pytest

import lucid
import lucid.linalg
from lucid.autograd import grad

F32_MAX = 3.4028234663852886e38


def _ulp(v: float) -> float:
    """float32 spacing at ``v``."""
    return 2.0 ** (math.frexp(abs(v))[1] - 24)


def _equal_case_fits(p: float, n: int, magnitude: float, cotangent: float) -> bool:
    """The norm and the gradient are finite float32 numbers (with margin) at these arguments."""
    norm = magnitude * float(n) ** (1.0 / p)
    gradient = cotangent * float(n) ** ((1.0 - p) / p)
    return norm < 0.5 * F32_MAX and 1e-30 < gradient < 0.5 * F32_MAX


#: ``N / m = n^(1/p)`` reaches 1e30 and up (the ratio overflows at 3.4e38), plus anchors below it.
_RATIO_CASES = [(0.05, 100), (0.1, 1_000), (0.1, 10_000), (0.1, 100_000), (0.2, 100_000)]
_ANCHORS = [(0.25, 100_000), (0.5, 100_000), (0.75, 100_000)]
EQUAL_CASES = [
    (p, n, magnitude, cotangent)
    for p, n in _RATIO_CASES + _ANCHORS
    for magnitude in (1e-20, 1e-5, 1e-3, 1.0, 1e10)
    for cotangent in (1.0, 1e-10)
    if _equal_case_fits(p, n, magnitude, cotangent)
]


def _id(case: tuple[float, int, float, float]) -> str:
    p, n, magnitude, cotangent = case
    return f"p{p}-n{n}-x{magnitude:g}-ct{cotangent:g}"


def _ulps(got: list[float], want: float) -> float:
    return max(abs(g - want) / _ulp(want) for g in got)


#: float32 ulp against float64: the powers and the division that make up the factor, with the
#: rounding of ``1 - p`` the exponent split restores (it was 15 .. 40 ulp at ``p = 0.1``).
EQUAL_ULPS = 6.0


#: Ten different magnitudes in a slice: the gradient inherits ``(1 - p)`` of the forward value's
#: error, which on Metal is 6e-7 (5 ulp) at ``p = 0.1`` for such a slice (the CPU's is 0.1).
UNEQUAL_ULPS = 16.0


@pytest.mark.parametrize("case", EQUAL_CASES, ids=_id)
def test_the_gradient_of_many_equal_entries_is_the_closed_form(
    device: str, case: tuple[float, int, float, float]
) -> None:
    p, n, magnitude, cotangent = case
    x = lucid.full((1, n), magnitude, device=device, requires_grad=True)
    norm = lucid.linalg.vector_norm(x, ord=p, dim=-1)
    assert math.isfinite(norm.item())
    (g,) = grad(norm.sum() * cotangent, [x])
    got = g.reshape(n).tolist()
    assert all(v - v == 0.0 for v in got), f"non-finite entry: {got[0]}"
    want = cotangent * float(n) ** ((1.0 - p) / p)
    assert _ulps(got, want) <= EQUAL_ULPS, f"{got[0]} vs {want}"


@pytest.mark.parametrize("case", EQUAL_CASES[:: max(1, len(EQUAL_CASES) // 12)], ids=_id)
def test_the_recorded_gradient_equals_the_eager_one(
    device: str, case: tuple[float, int, float, float]
) -> None:
    p, n, magnitude, cotangent = case
    x = lucid.full((1, n), magnitude, device=device, requires_grad=True)
    loss = lucid.linalg.vector_norm(x, ord=p, dim=-1).sum() * cotangent
    (eager,) = grad(loss, [x], retain_graph=True)
    (recorded,) = grad(loss, [x], create_graph=True)
    want = cotangent * float(n) ** ((1.0 - p) / p)
    assert _ulps(recorded.reshape(n).tolist(), want) <= EQUAL_ULPS
    assert recorded.reshape(n).tolist() == eager.reshape(n).tolist()


def _closed_gradient(values: list[float], p: float) -> list[float]:
    """``sign(x) (|x| / N)^(p-1)`` with ``N = top * s^(1/p)`` in units of the largest entry."""
    top = max(abs(v) for v in values)
    norm_factor = sum((abs(v) / top) ** p for v in values) ** ((1.0 - p) / p)
    return [
        0.0 if v == 0.0 else math.copysign((abs(v) / top) ** (p - 1.0) * norm_factor, v)
        for v in values
    ]


def _assert_ulps(got: list[float], want: list[float], tolerance: float) -> None:
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g - g == 0.0, f"entry {i} is {g}, want {w}"
        if w == 0.0:
            assert g == 0.0, f"entry {i} is {g}, want 0"
        else:
            assert abs(g - w) <= tolerance * _ulp(w), f"entry {i}: {g} vs {w}"


@pytest.mark.parametrize(("p", "n"), [(0.1, 10_000), (0.05, 50), (0.1, 1_000)])
def test_unequal_entries_keep_every_gradient_entry(device: str, p: float, n: int) -> None:
    """Ten distinct magnitudes and both signs: the slope ``(|x| / m)^(p-1)`` differs per entry."""
    values = [(1e-3 if i % 2 == 0 else -1e-3) * (1.0 + i % 10) for i in range(n)]
    x = lucid.tensor(values, device=device, requires_grad=True)
    norm = lucid.linalg.vector_norm(x, ord=p)
    assert math.isfinite(norm.item())
    (g,) = grad(norm, [x])
    _assert_ulps(g.tolist(), _closed_gradient(x.tolist(), p), UNEQUAL_ULPS)


@pytest.mark.parametrize("dim", [-1, 1, (1,)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_every_row_has_its_own_scale(device: str, dim: int | tuple[int], keepdim: bool) -> None:
    """The reduced norm and the largest magnitude are per row, in different decades."""
    n = 10_000
    rows = [[1e-3] * n, [1e-6 * (1.0 + i % 3) for i in range(n)]]
    x = lucid.tensor(rows, device=device, requires_grad=True)
    norm = lucid.linalg.vector_norm(x, ord=0.1, dim=dim, keepdim=keepdim)
    (g,) = grad(norm.sum(), [x])
    got = g.tolist()
    for row, values in zip(got, x.tolist(), strict=True):
        _assert_ulps(row, _closed_gradient(values, 0.1), UNEQUAL_ULPS)


def test_a_leading_axis_reduction_keeps_every_column(device: str) -> None:
    n = 10_000
    columns = [[1e-3 * (1.0 + (i + j) % 4) for j in range(3)] for i in range(n)]
    x = lucid.tensor(columns, device=device, requires_grad=True)
    (g,) = grad(lucid.linalg.vector_norm(x, ord=0.1, dim=0).sum(), [x])
    got, values = g.tolist(), x.tolist()
    for j in range(3):
        _assert_ulps(
            [row[j] for row in got], _closed_gradient([row[j] for row in values], 0.1), UNEQUAL_ULPS
        )


def test_a_reduction_over_every_axis_keeps_the_gradient(device: str) -> None:
    x = lucid.full((100, 100), 1e-3, device=device, requires_grad=True)
    (g,) = grad(lucid.linalg.vector_norm(x, ord=0.1), [x])
    _assert_ulps(g.reshape(10_000).tolist(), [10_000.0**9] * 10_000, EQUAL_ULPS)


@pytest.mark.parametrize("p", [0.05, 0.1])
def test_a_zero_entry_beside_many_live_ones_keeps_gradient_zero(device: str, p: float) -> None:
    n = 100 if p == 0.05 else 10_000
    x = lucid.full((n,), 1e-3 if p == 0.1 else 1e-5, device=device)
    x[0] = 0.0
    x.requires_grad_(True)
    (g,) = grad(lucid.linalg.vector_norm(x, ord=p), [x])
    got = g.tolist()
    assert got[0] == 0.0
    _assert_ulps(got[1:], [float(n - 1) ** ((1.0 - p) / p)] * (n - 1), EQUAL_ULPS)


def _closed_hvp(values: list[float], w: list[float], p: float) -> list[float]:
    """``H w = (p-1)/N (|r|^(p-2) w - s <s, w>)`` for ``r = |x| / N`` and ``s`` the gradient."""
    top = max(abs(v) for v in values)
    norm = top * sum((abs(v) / top) ** p for v in values) ** (1.0 / p)
    slope = _closed_gradient(values, p)
    dot = sum(a * b for a, b in zip(slope, w, strict=True))
    return [
        (p - 1.0) / norm * ((abs(v) / norm) ** (p - 2.0) * b - a * dot)
        for v, a, b in zip(values, slope, w, strict=True)
    ]


@pytest.mark.parametrize(("p", "n"), [(0.1, 1_000), (0.1, 10_000), (0.2, 100_000)])
def test_the_second_derivative_where_the_ratio_overflows_is_the_closed_form(
    device: str, p: float, n: int
) -> None:
    """The recorded gradient is the same graph: its Hessian-vector product was ``inf`` as well."""
    values = [1e-3] * n
    w = [1e-6 * ((i % 7) - 3.0) for i in range(n)]
    x = lucid.tensor(values, device=device, requires_grad=True)
    (g,) = grad(lucid.linalg.vector_norm(x, ord=p), [x], create_graph=True)
    (hw,) = grad(g, [x], grad_outputs=lucid.tensor(w, device=device))
    want = _closed_hvp(x.tolist(), lucid.tensor(w).tolist(), p)
    scale = max(abs(v) for v in want)
    for i, (got, ref) in enumerate(zip(hw.tolist(), want, strict=True)):
        assert got - got == 0.0, f"entry {i} is {got}, want {ref}"
        assert abs(got - ref) <= 1e-5 * scale, f"entry {i}: {got} vs {ref}"


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_one_live_entry_in_a_long_slice_has_exactly_the_gradient_of_its_sign(
    device: str, sign: float
) -> None:
    """``N = m``: the split of the ratio (the slice is long enough to need it) is exact there."""
    n = 10_000
    x = lucid.zeros((n,), device=device)
    x[7] = sign * 2.5
    x.requires_grad_(True)
    norm = lucid.linalg.vector_norm(x, ord=0.1)
    assert norm.item() == 2.5
    (g,) = grad(norm, [x])
    got = g.tolist()
    assert got[7] == sign
    assert all(v == 0.0 for i, v in enumerate(got) if i != 7)


def test_a_float64_slice_has_the_closed_form_where_float32_overflows() -> None:
    """The ratio is 1e100 for 1e5 entries at ``p = 0.05``; float64 holds it and its gradient."""
    n = 100_000
    x = lucid.full((1, n), 1.0, device="cpu", dtype=lucid.float64, requires_grad=True)
    norm = lucid.linalg.vector_norm(x, ord=0.05, dim=-1)
    (g,) = grad(norm.sum(), [x])
    want = float(n) ** 19.0
    assert abs(norm.item() - float(n) ** 20.0) <= 1e-12 * norm.item()
    assert max(abs(v - want) for v in g.reshape(n).tolist()) <= 1e-12 * want


#: Both signs of the tail ``(1 - p) - float32(1 - p)`` of the exponent split: its power of an
#: infinite base is ``inf`` for a positive tail and 0 for a negative one, and ``inf * 0`` is NaN.
NON_FINITE_ORDERS = [0.1, 0.15, 0.2, 0.3, 0.7]


def _exponent_tail(p: float) -> float:
    exponent = 1.0 - p
    return exponent - struct.unpack("f", struct.pack("f", exponent))[0]


def test_the_non_finite_orders_have_exponent_tails_of_both_signs() -> None:
    tails = [_exponent_tail(p) for p in NON_FINITE_ORDERS]
    assert any(t > 0.0 for t in tails) and any(t < 0.0 for t in tails)


@pytest.mark.parametrize("p", NON_FINITE_ORDERS)
@pytest.mark.parametrize("device_name", ["cpu", "metal"])
def test_an_overflowing_norm_gives_an_infinite_gradient_not_a_nan(
    device_name: str, p: float
) -> None:
    """The norm itself is ``inf`` in float32; its gradient is not recoverable from it, and the
    split of the exponent must not turn the ``inf`` into the NaN of ``inf * 0``."""
    x = lucid.full((1, 1_000), 1e38, device=device_name, requires_grad=True)
    norm = lucid.linalg.vector_norm(x, ord=p, dim=-1)
    assert norm.item() == math.inf
    (g,) = grad(norm.sum(), [x])
    assert set(g.reshape(1_000).tolist()) == {math.inf}


@pytest.mark.parametrize("p", NON_FINITE_ORDERS)
@pytest.mark.parametrize("device_name", ["cpu", "metal"])
def test_an_infinite_entry_leaves_the_finite_entries_infinite_not_a_nan(
    device_name: str, p: float
) -> None:
    """The norm is ``inf``; the other entries' slope times an infinite scale is ``inf``, as it was
    before the exponent was split (the infinite entry itself is ``0 * inf``)."""
    x = lucid.full((1_000,), 1.0, device=device_name)
    x[0] = math.inf
    x.requires_grad_(True)
    norm = lucid.linalg.vector_norm(x, ord=p)
    assert norm.item() == math.inf
    (g,) = grad(norm, [x])
    assert set(g.tolist()[1:]) == {math.inf}
