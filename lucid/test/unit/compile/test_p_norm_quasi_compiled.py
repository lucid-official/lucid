"""LCD-1015: a compiled graph takes a range-safe quasi-norm ``0 < p < 1``, as eager Metal does.

LCD-1009 gave eager Metal a fused kernel for ``vector_norm(x, ord=p)`` with ``0 < p < 1``;
the compiled ``p_norm`` emitter (LCD-955) took ``p > 1`` only, so a trace of a quasi-norm
recorded the plain chain ``(sum |x|^p)^(1/p)``.  That chain kept the value and the first
gradient in range, but not accurate: MPSGraph's ``pow`` is ``exp2(y log2 x)`` in float32,
whose error grows with ``|y log2 x|``, so a slice at 1e+-25 lost up to 20 ulp at ``p = 0.75``
and 28 at ``p = 0.1`` where a slice at 1 lost one.  The emitter now lowers ``p_norm`` below
1 to the eager kernel's form: terms ``(|x| / d)^p`` (the power of a rescaled ratio where the
division flushes, ``p < 1/2``), the root ``(d h) h`` with ``h = s^(1/(2p))`` (``s^(1/p)`` alone
leaves float32 for 1e4 equal entries at ``p = 0.1``), and a VJP that is ``quasi_grad``: slope
``sign(x) (|x| / d)^(p-1) s^((1-p)/p)`` with 0 at a zero entry.

A compiled function cannot take a second derivative (``autograd.grad(create_graph=True)``
inside a trace makes the call run eager, ``test_autograd_inside_trace.py``), so the
second-derivative range loss LCD-1009 fixed is not reachable from a compiled graph.

Every case compares the compiled call with the eager one *and* with a float64 closed form,
and asserts that an executable was built: a trace that fell back to eager would agree with
eager trivially.  The ulp bounds are against the float64 value: MPSGraph's ``pow`` and
float32 sum cannot match eager's double-float kernel below ``p = 1/4`` (the root multiplies
the sum's rounding by ``1/p``), so the bound loosens as ``p`` shrinks.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
import lucid.nn as nn
from lucid.compile import _tracing, compiled_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

#: The orders the plain chain was least accurate at, and the ones the eager tests use.
ORDERS = [0.1, 0.25, 0.5, 0.75, 0.9]
#: Float32's rounding is 1.19e-7 relative; the bounds are in those units, against float64.
ULP = 1.1920929e-07
#: The most the compiled value is allowed from the float64 value, by order.
VALUE_ULP = {0.1: 6.0, 0.25: 6.0, 0.5: 3.0, 0.75: 4.0, 0.9: 4.0}
#: The same for the gradient, entry by entry.
GRAD_ULP = {0.1: 8.0, 0.25: 4.0, 0.5: 3.0, 0.75: 4.0, 0.9: 4.0}
FLT_MAX = 3.4028235e38
BASE = np.array(
    [[3.0, -4.0, 1.0, 2.0, 0.5], [0.25, 0.5, -2.0, 7.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
    dtype=np.float32,
)
#: Float32's smallest normal is 1.18e-38 and Metal flushes below it.
MAGNITUDES = [1e-30, 1e-25, 1e-15, 1e-5, 1.0, 1e5, 1e15, 1e25, 1e30]


def _metal(a: np.ndarray, **kw: object) -> lucid.Tensor:
    return lucid.tensor(np.asarray(a, dtype=np.float32), device=COMPILE_DEVICE, **kw)


def _truth(a: np.ndarray, p: float, axis: object = None, keepdims: bool = False) -> np.ndarray:
    """The float64 norm of the float32 values, computed relative to the largest."""
    x = np.abs(a.astype(np.float32).astype(np.float64))
    top = x.max(axis=axis, keepdims=True)
    safe = np.where(top > 0, top, 1.0)
    out = safe * ((x / safe) ** p).sum(axis=axis, keepdims=True) ** (1.0 / p)
    if keepdims:
        return out
    return out.reshape(np.sum(a, axis=axis).shape) if axis is not None else out.reshape(())


def _truth_grad(a: np.ndarray, p: float) -> np.ndarray:
    """``sign(x) (|x| / N)^(p-1)`` of each row of a 2-D ``a`` (0 at a zero entry), float64."""
    x = a.astype(np.float32).astype(np.float64)
    top = np.abs(x).max(axis=-1, keepdims=True)
    norm = top * ((np.abs(x) / top) ** p).sum(axis=-1, keepdims=True) ** (1.0 / p)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(x == 0.0, 0.0, np.sign(x) * (np.abs(x) / norm) ** (p - 1.0))


def _ulps(got: np.ndarray, want: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    rel = np.abs(got - want) / np.where(want == 0, 1.0, np.abs(want))
    return float(rel.max()) / ULP


def _compiled(
    fn: Callable[[lucid.Tensor], lucid.Tensor], warm: lucid.Tensor, x: lucid.Tensor
) -> lucid.Tensor:
    """``fn(x)`` through a built executable: ``warm`` traces it, ``x`` replays it."""
    cm = lucid.compile(fn)
    with lucid.no_grad():
        cm(warm)
        out = cm(x)
    info = cm.cache_info()
    assert info["entries"] >= 1 and not info["eager_only"], info
    return out


class _Direct(nn.Module):
    """``vector_norm(z * x)`` over the last axis with ``x`` = 1: ``z.grad`` is the norm's gradient."""

    def __init__(self, z: np.ndarray, p: float, dim: object = -1, keepdim: bool = False) -> None:
        super().__init__()
        self.p, self.dim, self.keepdim = p, dim, keepdim
        self.z = nn.Parameter(_metal(z))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return LA.vector_norm(self.z * x, self.p, self.dim, self.keepdim)


def _loss(out: lucid.Tensor) -> lucid.Tensor:
    return out.sum()


def _compiled_grad(
    z: np.ndarray, p: float, dim: object = -1, keepdim: bool = False
) -> tuple[np.ndarray, float]:
    """The gradient of ``sum(vector_norm(z, p))`` from a compiled step, and its loss."""
    model = _Direct(z, p, dim, keepdim)
    model.zero_grad()
    loss = compiled_step(model, _metal(np.ones_like(z)), _loss)
    return model.z.grad.numpy(), float(loss.item())


def _eager_grad(z: np.ndarray, p: float, dim: object = -1, keepdim: bool = False) -> np.ndarray:
    leaf = _metal(z, requires_grad=True)
    LA.vector_norm(leaf, p, dim, keepdim).sum().backward()
    return leaf.grad.numpy()


def _recorded_ops(fn: Callable[[lucid.Tensor], lucid.Tensor], x: lucid.Tensor) -> list[str]:
    with _tracing(watch=((x,), {})) as tracer:
        fn(x)
    return [node.name for node in tracer.graph.ops]


# ── what a trace records ──────────────────────────────────────────────────────


@pytest.mark.parametrize("p", ORDERS)
def test_a_trace_records_the_p_norm_node_not_the_plain_chain(p: float) -> None:
    names = _recorded_ops(lambda t: LA.vector_norm(t, p, -1), _metal(BASE))
    assert "p_norm" in names
    assert "pow_scalar" not in names


def test_a_zero_dim_input_records_the_plain_chain() -> None:
    """The emitter has no 0-d variant: a scalar in a trace would fall back to eager."""
    scalar = lucid.tensor(1.5, device=COMPILE_DEVICE)
    names = _recorded_ops(lambda t: LA.vector_norm(t, 0.5), scalar)
    assert "p_norm" not in names and "pow_scalar" in names


def test_a_half_input_reaches_the_op_widened() -> None:
    """The kernel is float32: a half input is cast up, reduced by ``p_norm``, and cast back."""
    half = _metal(BASE).to(lucid.float16)
    assert _recorded_ops(lambda t: LA.vector_norm(t, 0.5, -1), half) == [
        "astype",
        "p_norm",
        "astype",
    ]


# ── values: orders x magnitudes ───────────────────────────────────────────────


@pytest.mark.parametrize("p", ORDERS)
def test_vector_norm_across_magnitudes(p: float) -> None:
    cm = lucid.compile(lambda t: LA.vector_norm(t, p, -1))
    checked = 0
    with lucid.no_grad():
        cm(_metal(BASE))
        for magnitude in MAGNITUDES:
            arr = BASE * np.float32(magnitude)
            want = _truth(arr, p, 1)
            if np.any(want > 0.9 * FLT_MAX):
                continue  # the norm itself leaves float32: ``test_a_norm_past_float32_is_inf``
            got = cm(_metal(arr)).numpy()
            eager = LA.vector_norm(_metal(arr), p, -1).numpy()
            assert _ulps(got, want) < VALUE_ULP[p], (magnitude, _ulps(got, want))
            assert _ulps(got, eager) < VALUE_ULP[p] + 3.0, magnitude
            checked += 1
    assert checked >= 5
    assert cm.cache_info()["entries"] >= 1 and not cm.cache_info()["eager_only"]


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5])
def test_a_norm_past_float32_is_inf(p: float) -> None:
    arr = np.full((1, 4), 3e37, dtype=np.float32)
    got = _compiled(lambda t: LA.vector_norm(t, p, -1), _metal(BASE[:1, :4]), _metal(arr))
    assert math.isinf(got.item())
    assert math.isinf(LA.vector_norm(_metal(arr), p, -1).item())


# ── gradients: orders x magnitudes ────────────────────────────────────────────


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_gradient_of_a_compiled_step_is_the_closed_form(p: float, magnitude: float) -> None:
    """The slope is homogeneous of degree 0, so it must not depend on the scale at all."""
    arr = BASE * np.float32(magnitude)
    got, loss = _compiled_grad(arr, p)
    want = _truth_grad(arr, p)
    assert np.all(np.isfinite(got))
    assert _ulps(got, want) < GRAD_ULP[p], _ulps(got, want)
    assert _ulps(got, _eager_grad(arr, p)) < GRAD_ULP[p] + 4.0
    if np.isfinite(loss):
        assert _ulps(loss, float(_truth(arr, p, 1).sum())) < 3.0 * VALUE_ULP[p]


# ── wide slices ───────────────────────────────────────────────────────────────

#: A slice whose smallest entries are 40 or more decades below its largest: the ratio underflows
#: float32 but ``r^p > r`` is not negligible below 1 (eager's ``WIDE_SLICES``).
WIDE_SLICES = [
    np.array([1e20] + [1e-20] * 1000, dtype=np.float32),
    np.array([1e10] + [1e-30] * 10, dtype=np.float32),
    np.array([3e38] + [1e-10] * 10, dtype=np.float32),
    np.array([1e20] + [1e-20] * 1_000_000, dtype=np.float32),
]


@pytest.mark.parametrize("p", [0.05] + ORDERS + [0.2])
@pytest.mark.parametrize("index", range(len(WIDE_SLICES)))
def test_small_entries_beside_a_huge_one_keep_their_powers(index: int, p: float) -> None:
    """The ratio of such an entry flushes to 0 (the p > 1 emitter would lose 61% of the norm at
    ``p = 0.1``); the term is the power of a rescaled ratio, so it still counts."""
    arr = WIDE_SLICES[index][None, :]
    want = float(_truth(arr, p, 1)[0])
    got = _compiled(lambda t: LA.vector_norm(t, p, -1), _metal(BASE), _metal(arr)).item()
    if want > FLT_MAX:
        assert math.isinf(got)
    else:
        assert got == pytest.approx(want, rel=2e-6, abs=0)


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("index", [0, 1, 3])
def test_the_gradient_of_the_small_entries_beside_a_huge_one(index: int, p: float) -> None:
    """Eager flushes ``|x| / d`` below 1.18e-38 and reads such an entry as a zero entry (gradient
    0, a documented limit); the compiled slope keeps them, and they are the closed form."""
    arr = WIDE_SLICES[index][None, :]
    got, _ = _compiled_grad(arr, p)
    want = _truth_grad(arr, p)
    live = np.isfinite(want) & (np.abs(want) < 0.9 * FLT_MAX)
    assert np.all(np.isfinite(got[live]))
    assert _ulps(got[live], want[live]) < GRAD_ULP[p] + 4.0


#: Per order: the value and the gradient may be this many ulp from float64 on a spike beside
#: entries 8 to 35 decades below it.  The sum of those terms is rounded in float32 and the root
#: multiplies that by ``1/p``, so ``p <= 0.2`` is looser (the plain chain: 7 to 12 ulp there).
RANDOM_WIDE_ULP = {0.1: (24.0, 36.0), 0.25: (8.0, 10.0), 0.5: (2.0, 3.0), 0.75: (2.0, 4.0)}


@pytest.mark.parametrize("p", list(RANDOM_WIDE_ULP))
@pytest.mark.parametrize(
    ("n", "spike", "lo", "hi"),
    [(1000, 1e20, -30, -18), (100, 1e30, -25, -10), (5000, 1e10, -35, -25)],
)
def test_a_spike_beside_entries_of_random_scales(
    p: float, n: int, spike: float, lo: int, hi: int
) -> None:
    rng = np.random.default_rng(5)
    value_bound, grad_bound = RANDOM_WIDE_ULP[p]
    for _ in range(3):
        arr = (10.0 ** rng.uniform(lo, hi, size=n) * rng.choice([-1.0, 1.0], size=n)).astype(
            np.float32
        )
        arr[0] = spike
        want = float(_truth(arr[None, :], p, 1)[0])
        if want > 0.9 * FLT_MAX:
            continue
        got = _compiled(lambda t: LA.vector_norm(t, p, -1), _metal(BASE), _metal(arr[None, :]))
        assert _ulps(got.item(), want) < value_bound
        grad, _ = _compiled_grad(arr[None, :], p)
        want_grad = _truth_grad(arr[None, :], p)
        live = np.abs(want_grad) < 0.9 * FLT_MAX
        assert _ulps(grad[live], want_grad[live]) < grad_bound


# ── a root past float32 whose product is in range ─────────────────────────────


@pytest.mark.parametrize(
    ("p", "count", "value", "want"),
    [(0.1, 10000, 1e-3, 1e37), (0.05, 100, 1e-5, 1e35), (0.1, 10000, 1e-2, 1e38)],
)
def test_a_root_past_float32_whose_product_is_in_range_is_not_inf(
    p: float, count: int, value: float, want: float
) -> None:
    """``(count * value^p)^(1/p) = value * count^(1/p)``: ``count^(1/p)`` alone is 1e40."""
    arr = np.full((1, count), value, dtype=np.float32)
    got = _compiled(lambda t: LA.vector_norm(t, p, -1), _metal(BASE[:1, :1]), _metal(arr))
    assert got.item() == pytest.approx(want, rel=2e-6, abs=0)


@pytest.mark.parametrize(("p", "count", "value"), [(0.1, 10000, 1e-3), (0.1, 10000, 1e-2)])
def test_the_gradient_at_a_root_past_float32_is_the_closed_form(
    p: float, count: int, value: float
) -> None:
    """The slope ``(x / N)^(p-1) = (1e-40)^(-0.9)`` is 1e36, an ordinary float: the compiled VJP
    takes ``s^((1-p)/p)`` rather than forming ``N / d = 1e40``.  The eager side is covered by
    LCD-1017 (``test_the_eager_gradient_at_a_root_past_float32_is_finite`` below)."""
    arr = np.full((1, count), value, dtype=np.float32)
    got, _ = _compiled_grad(arr, p)
    want = _truth_grad(arr, p)
    assert np.all(np.isfinite(got))
    assert _ulps(got, want) < GRAD_ULP[p]


def test_the_eager_gradient_at_a_root_past_float32_is_finite() -> None:
    arr = np.full((1, 10000), 1e-3, dtype=np.float32)
    assert np.all(np.isfinite(_eager_grad(arr, 0.1)))


# ── zero, NaN and infinite entries ────────────────────────────────────────────

NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("magnitude", [1e-30, 1.0, 1e30])
def test_a_zero_entry_in_a_live_slice_has_gradient_zero(p: float, magnitude: float) -> None:
    """``|x|^(p-1)`` is infinite at 0; its gradient there is 0, as the reference's."""
    arr = np.array([[3.0, 0.0, -4.0, 1.0, 0.0, 2.0], [0.0, 0.5, 0.0, 7.0, -1.0, 0.0]])
    arr = (arr * magnitude).astype(np.float32)
    got, _ = _compiled_grad(arr, p)
    assert np.all(np.isfinite(got))
    assert np.all(got[arr == 0.0] == 0.0)
    assert _ulps(got, _truth_grad(arr, p)) < GRAD_ULP[p]
    live = np.ascontiguousarray(arr[:, :5])
    value = _compiled(lambda t: LA.vector_norm(t, p, -1), _metal(BASE[:2, :]), _metal(live))
    assert _ulps(value.numpy(), _truth(live, p, 1)) < VALUE_ULP[p]


SPECIAL = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, INF, 2.0, 0.0],
        [INF, INF, 1.0, 1.0],
        [1.0, NAN, 2.0, 3.0],
        [INF, NAN, 0.0, 0.0],
        [-INF, 1.0, 1.0, 1.0],
        [3.0, -4.0, 0.0, 0.0],
        [NAN, NAN, NAN, NAN],
        [-0.0, 0.0, -0.0, 0.0],
    ],
    dtype=np.float32,
)


@pytest.mark.parametrize("p", ORDERS)
def test_zero_inf_and_nan_slices_answer_as_eager_does(p: float) -> None:
    got = _compiled(
        lambda t: LA.vector_norm(t, p, 1), _metal(np.ones_like(SPECIAL)), _metal(SPECIAL)
    )
    want = LA.vector_norm(_metal(SPECIAL), p, 1).numpy()
    np.testing.assert_array_equal(np.isnan(got.numpy()), np.isnan(want))
    np.testing.assert_array_equal(np.isinf(got.numpy()), np.isinf(want))
    finite = np.isfinite(want)
    assert _ulps(got.numpy()[finite], want[finite]) < VALUE_ULP[p] + 3.0
    assert got.numpy()[0] == 0.0 and got.numpy()[8] == 0.0


@pytest.mark.parametrize("p", ORDERS)
def test_the_gradient_of_zero_and_nan_slices(p: float) -> None:
    """A zero slice has the subgradient 0 and a NaN slice keeps its NaN, as eager's."""
    zero, _ = _compiled_grad(np.zeros((2, 5), dtype=np.float32), p)
    assert np.all(zero == 0.0)
    nan_row = np.array([[NAN, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    nan, _ = _compiled_grad(nan_row, p)
    assert np.all(np.isnan(nan))
    assert np.all(np.isnan(_eager_grad(nan_row, p)))


@pytest.mark.parametrize("p", ORDERS)
def test_the_gradient_with_an_infinite_entry_has_eagers_nan_pattern(p: float) -> None:
    row = np.array([[1.0, INF, 2.0, 0.5]], dtype=np.float32)
    got, _ = _compiled_grad(row, p)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(_eager_grad(row, p)))
    np.testing.assert_array_equal(np.isinf(got), np.isinf(_eager_grad(row, p)))


# ── layouts ───────────────────────────────────────────────────────────────────

LAYOUTS = [
    ((7,), None),
    ((5, 9), 1),
    ((5, 9), 0),
    ((4, 6, 5), (0, 2)),
    ((4, 6, 5), (1,)),
    ((3, 4, 5, 6), (1, 2)),
    ((2, 3, 4, 5, 6), (1, 3)),
    ((2, 3, 4, 5, 6), (1, 2)),
    ((2, 3, 4, 5, 6), (0, 4)),
    ((2, 3, 4, 5, 6), None),
    ((2, 3, 4, 5, 6), ()),
]


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("p", [0.25, 0.5, 0.9])
@pytest.mark.parametrize(("shape", "dim"), LAYOUTS)
def test_every_layout_and_axis_set(
    shape: tuple[int, ...], dim: object, p: float, keepdim: bool
) -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal(shape).astype(np.float32)
    axis = None if dim in (None, ()) else dim if isinstance(dim, int) else tuple(dim)
    cm = lucid.compile(lambda t: LA.vector_norm(t, p, dim=dim, keepdim=keepdim))
    with lucid.no_grad():
        cm(_metal(base))
        for magnitude in (1e-25, 1.0, 1e25):
            arr = base * np.float32(magnitude)
            got = cm(_metal(arr)).numpy()
            want = _truth(arr, p, axis, keepdim)
            assert got.shape == want.shape
            assert _ulps(got, want) < VALUE_ULP[p] + 2.0
    assert cm.cache_info()["entries"] >= 1 and not cm.cache_info()["eager_only"]


class _Scaled(nn.Module):
    """``vector_norm(x * w)``: the gradient reaches ``w`` through the norm."""

    def __init__(self, shape: tuple[int, ...], p: float, dim: object, keepdim: bool) -> None:
        super().__init__()
        self.p, self.dim, self.keepdim = p, dim, keepdim
        self.w = nn.Parameter(_metal(np.linspace(0.5, 2.0, int(np.prod(shape))).reshape(shape)))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return LA.vector_norm(x * self.w, self.p, self.dim, self.keepdim)


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("p", [0.25, 0.75])
@pytest.mark.parametrize(("shape", "dim"), LAYOUTS)
def test_the_gradient_of_a_compiled_step_equals_eager_at_every_layout(
    shape: tuple[int, ...], dim: object, p: float, keepdim: bool
) -> None:
    rng = np.random.default_rng(1)
    for magnitude in (1e-25, 1.0, 1e25):
        arr = (rng.standard_normal(shape) * magnitude).astype(np.float32)
        model = _Scaled(shape, p, dim, keepdim)
        eager = _Scaled(shape, p, dim, keepdim)
        x = _metal(arr)
        model.zero_grad()
        compiled_step(model, x, _loss)
        _loss(eager(x)).backward()
        assert np.all(np.isfinite(model.w.grad.numpy()))
        assert _ulps(model.w.grad.numpy(), eager.w.grad.numpy()) < GRAD_ULP[p] + 4.0


# ── dynamic batch, a rank-5 producer ──────────────────────────────────────────


@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
def test_dynamic_batch_serves_other_batch_sizes(magnitude: float) -> None:
    rng = np.random.default_rng(1)
    cm = lucid.compile(lambda t: LA.vector_norm(t, 0.25, -1), dynamic=True)
    with lucid.no_grad():
        cm(_metal(rng.standard_normal((4, 6))))
        for batch in (4, 7, 2):
            arr = rng.standard_normal((batch, 6)).astype(np.float32) * np.float32(magnitude)
            assert _ulps(cm(_metal(arr)).numpy(), _truth(arr, 0.25, 1)) < VALUE_ULP[0.25]
    assert cm.cache_info()["entries"] >= 1 and not cm.cache_info()["eager_only"]


@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
def test_a_rank_5_producer_feeds_the_norm_at_its_canonical_rank(magnitude: float) -> None:
    """A lead of 80 is past the rank canonicalisation's threshold: ``t * 2`` is rewritten
    and ``p_norm`` reads the view of its result, as ``sum`` does."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((80, 2, 3, 4, 5)).astype(np.float32)
    arr = base * np.float32(magnitude)
    got = _compiled(lambda t: LA.vector_norm(t * 2.0, 0.5, (1, 3)), _metal(base), _metal(arr))
    assert _ulps(got.numpy(), _truth(arr * np.float32(2.0), 0.5, (1, 3))) < VALUE_ULP[0.5] + 2.0


@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
def test_a_rank_5_input_keeps_its_gradient(magnitude: float) -> None:
    """The VJP works at the canonical lower rank and leaves it again."""
    rng = np.random.default_rng(4)
    shape = (80, 2, 3, 4, 5)
    arr = (rng.standard_normal(shape) * magnitude).astype(np.float32)
    model = _Scaled(shape, 0.5, (1, 3), False)
    eager = _Scaled(shape, 0.5, (1, 3), False)
    model.zero_grad()
    compiled_step(model, _metal(arr), _loss)
    _loss(eager(_metal(arr))).backward()
    assert _ulps(model.w.grad.numpy(), eager.w.grad.numpy()) < GRAD_ULP[0.5] + 4.0
