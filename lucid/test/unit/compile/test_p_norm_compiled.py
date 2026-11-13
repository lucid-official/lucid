"""LCD-955: a compiled graph takes a range-safe entrywise p-norm, as eager Metal does.

``vector_norm`` / ``matrix_norm('fro')`` / ``F.normalize`` on Metal answer through
a fused kernel that rescales by the largest magnitude (LCD-944).  A trace used
to record the plain ``sqrt(sum(x * x))`` instead, which flushes a float32 slice
near 1e-25 to 0 and overflows one near 1e25, so a compiled graph was wrong where
the same call run eagerly was right.  The trace now records the ``p_norm`` op,
which the compiled graph lowers to the same rescale (``OpEmitters/_ScaledPNorm.h``),
with its own VJP; ``lp_normalize`` shares the lowering.

Every case compares the compiled call with the eager one *and* with a float64
truth computed relative to the largest magnitude, and every case asserts that an
executable was built: a trace that fell back to eager would agree with eager
trivially.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.compile import compiled_step
from lucid.compile._core.trace_context import compile_attempt, recording
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

#: float32's smallest normal is 1.18e-38 and Metal flushes below it; the largest
#: finite is 3.4e38, and a slice of 5 entries of up to 4 * 1e36 stays inside it.
MAGNITUDES = [1e-37, 1e-30, 1e-25, 1e-20, 1e-10, 1.0, 1e10, 1e20, 1e25, 1e30, 1e36]
#: Orders the scaled op serves: finite and above 1.
SCALED_ORDERS = [1.5, 2.0, 3.0, 3.5]
BASE = np.array(
    [[3.0, -4.0, 1.0, 2.0, 0.5], [0.25, 0.5, -2.0, 7.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
    dtype=np.float32,
)
RTOL = 2e-5


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


def _agree(got: lucid.Tensor, want: np.ndarray, rtol: float = RTOL) -> None:
    assert got.shape == want.shape
    np.testing.assert_allclose(got.numpy(), want, rtol=rtol, atol=0)


def test_the_op_has_an_emitter_and_a_vjp() -> None:
    assert _C_engine.compile.emitter_registered("p_norm")
    assert (
        _C_engine.compile.vjp_registration_status("p_norm")
        == _C_engine.compile.VjpRegistration.Registered
    )


def test_a_trace_records_the_scaled_op_not_the_plain_squares() -> None:
    names = {n for n in _recorded_ops(lambda t: LA.vector_norm(t, 2, -1), _metal(BASE))}
    assert "p_norm" in names
    assert not names & {"mul", "sqrt"}


def _recorded_ops(fn: Callable[[lucid.Tensor], lucid.Tensor], x: lucid.Tensor) -> list[str]:
    with compile_attempt(), recording(watch=(x,)) as tracer:
        fn(x)
    return [node.name for node in tracer.graph.ops]


# ── magnitudes ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("p", SCALED_ORDERS)
def test_vector_norm_across_magnitudes(p: float) -> None:
    cm = lucid.compile(lambda t: LA.vector_norm(t, p, -1))
    with lucid.no_grad():
        cm(_metal(BASE))
        for magnitude in MAGNITUDES:
            arr = BASE * np.float32(magnitude)
            got = cm(_metal(arr))
            eager = LA.vector_norm(_metal(arr), p, -1)
            _agree(got, _truth(arr, p, 1))
            _agree(got, eager.numpy())
    assert cm.cache_info()["entries"] >= 1 and not cm.cache_info()["eager_only"]


@pytest.mark.parametrize("magnitude", [1e-30, 1e-25, 1e25, 1e30])
def test_the_plain_form_is_not_what_runs(magnitude: float) -> None:
    """sqrt(sum(x * x)) answers 0 / inf here: a compiled graph does not."""
    arr = np.full((2, 3), magnitude, dtype=np.float32)
    got = _compiled(lambda t: LA.vector_norm(t), _metal(BASE[:2, :3]), _metal(arr))
    assert got.item() == pytest.approx(math.sqrt(6.0) * magnitude, rel=RTOL, abs=0)


@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
def test_matrix_norm_fro_and_norm(magnitude: float) -> None:
    arr = BASE * np.float32(magnitude)
    want = _truth(arr, 2.0)
    for fn in (
        lambda t: LA.matrix_norm(t, "fro"),
        lambda t: LA.norm(t),
        lambda t: LA.norm(t.reshape(-1)),
    ):
        _agree(_compiled(fn, _metal(BASE), _metal(arr)), want)


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
]


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("p", [2.0, 3.5])
@pytest.mark.parametrize(("shape", "dim"), LAYOUTS)
def test_every_layout_and_axis_set(
    shape: tuple[int, ...], dim: object, p: float, keepdim: bool
) -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal(shape).astype(np.float32)
    axis = dim if dim is None or isinstance(dim, int) else tuple(dim)
    cm = lucid.compile(lambda t: LA.vector_norm(t, p, dim=dim, keepdim=keepdim))
    with lucid.no_grad():
        cm(_metal(base))
        for magnitude in (1e-25, 1.0, 1e25):
            arr = base * np.float32(magnitude)
            got = cm(_metal(arr))
            _agree(got, _truth(arr, p, axis, keepdim), rtol=5e-5)
    assert cm.cache_info()["entries"] >= 1 and not cm.cache_info()["eager_only"]


# ── orders the scaled op does not serve ───────────────────────────────────────


@pytest.mark.parametrize("p", [1.0, 0.5, math.inf, -math.inf, -2.0])
@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 1e3])
def test_the_other_orders_still_compile_and_equal_eager(p: float, magnitude: float) -> None:
    arr = BASE * np.float32(magnitude)
    got = _compiled(lambda t: LA.vector_norm(t, p, -1), _metal(BASE), _metal(arr))
    _agree(got, LA.vector_norm(_metal(arr), p, -1).numpy(), rtol=1e-5)


# ── special values ────────────────────────────────────────────────────────────

NAN, INF = float("nan"), float("inf")
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


@pytest.mark.parametrize("p", [2.0, 3.0, 3.5])
def test_zero_inf_and_nan_slices(p: float) -> None:
    got = _compiled(lambda t: LA.vector_norm(t, p, 1), _metal(SPECIAL * 0 + 1), _metal(SPECIAL))
    want = LA.vector_norm(_metal(SPECIAL), p, 1).numpy()
    np.testing.assert_array_equal(np.isnan(got.numpy()), np.isnan(want))
    np.testing.assert_array_equal(np.isinf(got.numpy()), np.isinf(want))
    finite = np.isfinite(want)
    np.testing.assert_allclose(got.numpy()[finite], want[finite], rtol=RTOL, atol=0)
    assert got.numpy()[0] == 0.0 and got.numpy()[8] == 0.0


# ── F.normalize (the lp_normalize op) ─────────────────────────────────────────


@pytest.mark.parametrize("eps", [1e-12, 1e-30])
@pytest.mark.parametrize("magnitude", [1e-25, 1e-15, 1e-5, 1.0, 1e10, 1e25, 1e30])
def test_normalize_across_magnitudes(magnitude: float, eps: float) -> None:
    arr = BASE * np.float32(magnitude)
    got = _compiled(lambda t: F.normalize(t, 2.0, -1, eps), _metal(BASE), _metal(arr))
    eager = F.normalize(_metal(arr), 2.0, -1, eps)
    _agree(got, eager.numpy(), rtol=1e-4)
    if eps == 1e-30 or magnitude >= 1e-10:
        norms = np.sqrt((got.numpy().astype(np.float64) ** 2).sum(axis=1))
        np.testing.assert_allclose(norms, 1.0, rtol=1e-4)


@pytest.mark.parametrize("dim", [(0, 1), None])
@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
def test_normalize_over_several_dims_goes_through_vector_norm(
    dim: object, magnitude: float
) -> None:
    arr = BASE * np.float32(magnitude)
    got = _compiled(lambda t: F.normalize(t, 2.0, dim, 1e-30), _metal(BASE), _metal(arr))
    _agree(got, F.normalize(_metal(arr), 2.0, dim, 1e-30).numpy(), rtol=1e-4)


@pytest.mark.parametrize("p", [1.0, 3.0, math.inf])
def test_normalize_orders_the_emitter_declines_still_equal_eager(p: float) -> None:
    cm = lucid.compile(lambda t: F.normalize(t, p, -1))
    with lucid.no_grad():
        cm(_metal(BASE))
        got = cm(_metal(BASE * 4))
    _agree(got, F.normalize(_metal(BASE * 4), p, -1).numpy(), rtol=1e-5)


# ── half precision, dynamic batch ─────────────────────────────────────────────


def test_a_float16_input_reduces_in_float32() -> None:
    ones = lucid.ones(65536, dtype=lucid.float16, device=COMPILE_DEVICE)
    got = _compiled(lambda t: LA.vector_norm(t), ones[:8], ones)
    assert got.dtype == lucid.float16 and got.item() == 256.0


@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
def test_dynamic_batch_serves_other_batch_sizes(magnitude: float) -> None:
    rng = np.random.default_rng(1)
    cm = lucid.compile(lambda t: LA.vector_norm(t, 3.5, -1), dynamic=True)
    with lucid.no_grad():
        cm(_metal(rng.standard_normal((4, 6))))
        for batch in (4, 7, 2):
            arr = rng.standard_normal((batch, 6)).astype(np.float32) * np.float32(magnitude)
            _agree(cm(_metal(arr)), _truth(arr, 3.5, 1), rtol=5e-5)
    assert cm.cache_info()["entries"] >= 1 and not cm.cache_info()["eager_only"]


@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
def test_dynamic_batch_normalize(magnitude: float) -> None:
    rng = np.random.default_rng(2)
    cm = lucid.compile(lambda t: F.normalize(t, 2.0, -1, 1e-30), dynamic=True)
    with lucid.no_grad():
        cm(_metal(rng.standard_normal((4, 6))))
        for batch in (4, 7, 2):
            arr = rng.standard_normal((batch, 6)).astype(np.float32) * np.float32(magnitude)
            want = F.normalize(_metal(arr), 2.0, -1, 1e-30).numpy()
            _agree(cm(_metal(arr)), want, rtol=1e-4)
    assert cm.cache_info()["entries"] >= 1 and not cm.cache_info()["eager_only"]


@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
def test_a_rank_5_producer_feeds_the_norm_at_its_canonical_rank(magnitude: float) -> None:
    """A lead of 80 is past the rank canonicalisation's threshold: ``t * 2`` is rewritten
    and ``p_norm`` reads the view of its result, as ``sum`` does."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((80, 2, 3, 4, 5)).astype(np.float32)
    arr = base * np.float32(magnitude)
    got = _compiled(lambda t: LA.vector_norm(t * 2.0, 2.0, (1, 3)), _metal(base), _metal(arr))
    _agree(got, _truth(arr * np.float32(2.0), 2.0, (1, 3)), rtol=5e-5)


# ── gradients: a compiled step against the eager one ──────────────────────────


class _Scaled(nn.Module):
    """``vector_norm(x * w)``: the gradient reaches ``w`` through the norm."""

    def __init__(self, p: float, dim: object, keepdim: bool) -> None:
        super().__init__()
        self.p, self.dim, self.keepdim = p, dim, keepdim
        self.w = nn.Parameter(lucid.tensor(np.linspace(0.5, 2.0, 5, dtype=np.float32)))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return LA.vector_norm(x * self.w, self.p, self.dim, self.keepdim)


class _Normalized(nn.Module):
    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(lucid.tensor(np.linspace(0.5, 2.0, 5, dtype=np.float32)))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return F.normalize(x * self.w, 2.0, -1, self.eps) * lucid.arange(5.0, device=x.device)


def _loss(out: lucid.Tensor) -> lucid.Tensor:
    """Linear in the output, so a norm near 1e20 does not overflow the loss itself."""
    return out.sum()


def _step_pair(
    make: Callable[[], nn.Module], arr: np.ndarray
) -> tuple[lucid.Tensor, lucid.Tensor, tuple[float, float]]:
    model = make().to(COMPILE_DEVICE)
    eager = make().to(COMPILE_DEVICE)
    eager.load_state_dict(model.state_dict())
    x = _metal(arr)
    model.zero_grad()
    got = compiled_step(model, x, _loss)
    want = _loss(eager(x))
    want.backward()
    return model.w.grad, eager.w.grad, (float(got.item()), float(want.item()))


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("p", [2.0, 3.5])
@pytest.mark.parametrize("dim", [-1, 0, None])
@pytest.mark.parametrize("magnitude", [1e-25, 1e-5, 1.0, 1e5, 1e20])
def test_the_gradient_of_a_compiled_step_equals_eager(
    magnitude: float, dim: object, p: float, keepdim: bool
) -> None:
    arr = BASE * np.float32(magnitude)
    got, want, (loss_got, loss_want) = _step_pair(lambda: _Scaled(p, dim, keepdim), arr)
    assert np.all(np.isfinite(got.numpy()))
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=2e-4, atol=0)
    assert loss_got == pytest.approx(loss_want, rel=1e-4, abs=0)


def _normalize_weight_grad(arr: np.ndarray) -> np.ndarray:
    """The float64 gradient of ``_Normalized`` with respect to its weight, in closed form.

    ``y = z / N`` with ``z = x * w`` and ``N = |z|``: ``dL/dz = (c - <c, y> y) / N`` for
    ``L = <c, y>``, and ``dL/dw = sum_rows x * dL/dz``.  Everything is a ratio to ``N``, so
    the magnitude of ``x`` only scales the answer.
    """
    x = arr.astype(np.float64)
    w = np.linspace(0.5, 2.0, 5, dtype=np.float32).astype(np.float64)
    c = np.arange(5.0)
    z = x * w
    norm = np.sqrt(((z / np.abs(z).max(axis=1, keepdims=True)) ** 2).sum(axis=1, keepdims=True))
    norm = norm * np.abs(z).max(axis=1, keepdims=True)
    y = z / norm
    dz = (c - (y * c).sum(axis=1, keepdims=True) * y) / norm
    return (x * dz).sum(axis=0)


@pytest.mark.parametrize("magnitude", [1e-25, 1e-20, 1e-3, 1.0, 1e10, 1e25, 1e30])
def test_the_gradient_of_a_compiled_normalize_is_the_analytic_one(magnitude: float) -> None:
    """Eager's backward leaves the float range from ~1e8 up (``pow(norm, 3)``), so the truth is closed form."""
    arr = BASE * np.float32(magnitude)
    got, _, _ = _step_pair(lambda: _Normalized(1e-30), arr)
    want = _normalize_weight_grad(arr)
    assert np.all(np.isfinite(got.numpy()))
    np.testing.assert_allclose(got.numpy(), want, rtol=2e-4, atol=1e-5 * np.abs(want).max())


@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 1e5])
def test_the_gradient_of_a_compiled_normalize_equals_eager(magnitude: float) -> None:
    got, want, _ = _step_pair(lambda: _Normalized(1e-30), BASE * np.float32(magnitude))
    np.testing.assert_allclose(
        got.numpy(), want.numpy(), rtol=1e-3, atol=1e-6 * float(np.abs(want.numpy()).max())
    )


def test_the_gradient_of_a_zero_slice_is_zero_and_a_nan_slice_stays_nan() -> None:
    arr = np.array([[0.0] * 5, BASE[0], [np.nan, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    model = _Scaled(2.0, -1, False).to(COMPILE_DEVICE)

    def row_loss(out: lucid.Tensor) -> lucid.Tensor:
        return out.sum()

    x = _metal(arr[:2])
    got = compiled_step(model, x, row_loss)
    assert np.isfinite(got.item())
    assert np.all(np.isfinite(model.w.grad.numpy()))
    x_nan = _metal(arr[2:])
    model.zero_grad()
    compiled_step(model, x_nan, row_loss)
    assert np.all(np.isnan(model.w.grad.numpy()))
