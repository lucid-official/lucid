"""``cdist`` / ``pdist`` with ``p=2`` keep tiny and huge magnitudes (LCD-952).

The matmul form ``|a|^2 + |b|^2 - 2 a.b`` squares the coordinates, and a float32
square leaves the range long before the distance does: rows of ``1e-20`` and
``3e-20`` are 4e-20 apart, but their squares (1e-40) are subnormal, which Metal
flushes to zero, and rows of ``1e25`` and ``3e25`` square to ``inf`` and cancel
to ``nan``.  Both forms now divide their operands by a largest magnitude first
and multiply the root by it afterwards.

What the tests pin:

* the reference's contract — which ``compute_mode`` takes which form (the
  difference form up to 25 rows in both operands, the matmul form past that);
* the answers at tiny and huge magnitudes, on both devices, every mode, with
  ``abs=0`` (``pytest.approx``'s default floor of 1e-12 would pass a 1e-20
  answer of 0);
* first and second derivatives, which the constant divisor must not touch;
* inside the window nothing changes: normal magnitudes are bit-identical to the
  unscaled forms, and zero / ``nan`` / infinite rows answer what they did.
"""

import math
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid._types import DTypeLike

_MM = "use_mm_for_euclid_dist"
_EXACT = "donot_use_mm_for_euclid_dist"
_DEFAULT = "use_mm_for_euclid_dist_if_necessary"
_MODES = [_DEFAULT, _MM, _EXACT]

#: Tiny, and huge, float32 magnitudes whose squares leave the range and whose
#: distances do not.  ``1e-35`` and ``1e37`` sit next to the range's two ends.
_TINY = [1e-20, 1e-30, 1e-35]
_HUGE = [1e25, 1e30, 1e37]
_EXTREMES = _TINY + _HUGE


def _t(
    a: np.ndarray, device: str = "cpu", *, requires_grad: bool = False, dtype: DTypeLike = None
) -> lucid.Tensor:
    return lucid.tensor(a, device=device, requires_grad=requires_grad, dtype=dtype)


def _truth(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The pairwise Euclidean distance of the *stored* values, in float64."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return np.asarray(np.sqrt(((a64[..., :, None, :] - b64[..., None, :, :]) ** 2).sum(-1)))


def _points(rows: int, cols: int, magnitude: float, seed: int) -> np.ndarray:
    """Entries of magnitude ``0.5 m`` to ``2 m`` with random signs: none subnormal, none tied."""
    rng = np.random.default_rng(seed)
    signed = rng.uniform(0.5, 2.0, (rows, cols)) * rng.choice([-1.0, 1.0], (rows, cols))
    return (signed * magnitude).astype(np.float32)


def _two_rows(magnitude: float) -> np.ndarray:
    """Rows ``1*m`` and ``3*m`` in four coordinates: ``4*m`` apart."""
    return np.array([[1.0] * 4, [3.0] * 4], dtype=np.float32) * np.float32(magnitude)


def _near_duplicates(rows: int) -> np.ndarray:
    """Rows of one seed's first ``rows``, each moved by 1e-3: the two forms round differently."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((rows, 8)).astype(np.float32) * 10.0
    return base + rng.standard_normal((rows, 8)).astype(np.float32) * 1e-3


def _close(got: object, want: object, rel: float = 1e-5) -> None:
    """``got`` equals ``want`` to ``rel``, with no absolute floor: a tiny answer must be right."""
    got_arr = np.asarray(got, dtype=np.float64)
    want_arr = np.asarray(want, dtype=np.float64)
    assert got_arr.shape == want_arr.shape
    assert np.isfinite(got_arr).all(), got_arr
    np.testing.assert_allclose(got_arr, want_arr, rtol=rel, atol=0.0)


# ── The issue's repro, every mode ────────────────────────────────────────────


@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("magnitude", _EXTREMES, ids=lambda m: f"{m:g}")
def test_two_points_keep_their_distance(device: str, magnitude: float, compute_mode: str) -> None:
    rows = _two_rows(magnitude)
    x = _t(rows, device)
    want = 4.0 * float(rows[0, 0])
    got = lucid.cdist(x[:1], x[1:], compute_mode=compute_mode).item()
    assert got == pytest.approx(want, rel=1e-5, abs=0)
    assert lucid.cdist(x[:1], x[1:], 2.0, compute_mode).item() == pytest.approx(
        want, rel=1e-5, abs=0
    )


@pytest.mark.parametrize("magnitude", _EXTREMES, ids=lambda m: f"{m:g}")
def test_pdist_keeps_the_distance(device: str, magnitude: float) -> None:
    rows = _two_rows(magnitude)
    got = F.pdist(_t(rows, device)).item()
    assert got == pytest.approx(4.0 * float(rows[0, 0]), rel=1e-5, abs=0)


@pytest.mark.parametrize("magnitude", _EXTREMES, ids=lambda m: f"{m:g}")
@pytest.mark.parametrize("rows", [2, 25, 26, 40], ids=lambda n: f"n{n}")
def test_pdist_at_extreme_magnitude_across_the_form_switch(
    device: str, magnitude: float, rows: int
) -> None:
    """Up to 25 rows the difference form answers, past that the matmul form."""
    x = _points(rows, 8, magnitude, seed=rows)
    want = _truth(x, x)[np.triu_indices(rows, 1)]
    _close(F.pdist(_t(x, device)).tolist(), want, rel=2e-4)


@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("magnitude", [1e-20, 1e-35, 1e25, 1e37], ids=lambda m: f"{m:g}")
@pytest.mark.parametrize("shape", [(30, 40), (40, 30), (5, 40), (3, 4)], ids=str)
def test_cdist_rectangular_at_extreme_magnitude(
    device: str, magnitude: float, compute_mode: str, shape: tuple[int, int]
) -> None:
    a = _points(shape[0], 8, magnitude, seed=1)
    b = _points(shape[1], 8, magnitude, seed=2)
    got = lucid.cdist(_t(a, device), _t(b, device), compute_mode=compute_mode)
    _close(got.tolist(), _truth(a, b), rel=2e-4)


@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("magnitude", [1e-200, 1e-150, 1e200, 1e300], ids=lambda m: f"{m:g}")
def test_float64_keeps_the_distance(
    device_cpu_only: str, magnitude: float, compute_mode: str
) -> None:
    rows = np.array([[1.0] * 4, [3.0] * 4]) * magnitude
    x = _t(rows, dtype=lucid.float64)
    got = lucid.cdist(x[:1], x[1:], compute_mode=compute_mode).item()
    assert got == pytest.approx(4.0 * magnitude, rel=1e-12, abs=0)


# ── Mixed magnitudes ─────────────────────────────────────────────────────────

_BATCH_MAGNITUDES = (1e-20, 1.0, 1e25)


@pytest.mark.parametrize("compute_mode", _MODES)
def test_batch_entries_are_scaled_independently(device: str, compute_mode: str) -> None:
    """A tiny batch entry beside a huge one keeps its distances: one divisor per entry."""
    a = np.stack([_points(6, 4, m, seed=i) for i, m in enumerate(_BATCH_MAGNITUDES)])
    b = np.stack([_points(5, 4, m, seed=10 + i) for i, m in enumerate(_BATCH_MAGNITUDES)])
    got = lucid.cdist(_t(a, device), _t(b, device), compute_mode=compute_mode)
    _close(got.tolist(), _truth(a, b), rel=2e-4)


@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("batched", ["x1", "x2"])
def test_a_broadcast_operand_joins_every_batch_entry(
    device: str, compute_mode: str, batched: str
) -> None:
    """One un-batched operand is compared with entries of every magnitude."""
    batch = np.stack([_points(6, 4, m, seed=i) for i, m in enumerate(_BATCH_MAGNITUDES)])
    single = _points(5, 4, 1e-20, seed=7)
    if batched == "x1":
        got = lucid.cdist(_t(batch, device), _t(single, device), compute_mode=compute_mode)
        want = _truth(batch, single)
    else:
        got = lucid.cdist(_t(single, device), _t(batch, device), compute_mode=compute_mode)
        want = _truth(single, batch)
    _close(got.tolist(), want, rel=2e-4)


@pytest.mark.parametrize("compute_mode", [_DEFAULT, _EXACT])
def test_every_pair_keeps_its_own_scale_in_the_difference_form(
    device: str, compute_mode: str
) -> None:
    """Rows from ``1e-20`` to ``1e25`` in one matrix: each pair is divided by its own largest."""
    scales = [1e-20, 1e-20, 1.0, 1e25, 1e25]
    x = np.stack([_points(1, 4, s, seed=i)[0] for i, s in enumerate(scales)])
    got = lucid.cdist(_t(x, device), _t(x, device), compute_mode=compute_mode)
    _close(got.tolist(), _truth(x, x), rel=1e-5)


# ── Which mode takes which form (the reference's contract) ───────────────────


@pytest.mark.parametrize(
    ("p_rows", "r_rows", "form"),
    [
        (3, 5, _EXACT),
        (25, 25, _EXACT),
        (25, 3, _EXACT),
        (3, 25, _EXACT),
        (26, 3, _MM),
        (3, 26, _MM),
        (26, 26, _MM),
        (40, 40, _MM),
    ],
    ids=str,
)
def test_the_default_mode_switches_to_the_matmul_form_past_25_rows(
    device: str, p_rows: int, r_rows: int, form: str
) -> None:
    x1, x2 = _t(_near_duplicates(p_rows), device), _t(_near_duplicates(r_rows), device)
    default = lucid.cdist(x1, x2).numpy()
    exact = lucid.cdist(x1, x2, compute_mode=_EXACT).numpy()
    mm = lucid.cdist(x1, x2, compute_mode=_MM).numpy()
    assert not np.array_equal(exact, mm), "the two forms must differ here for the test to bite"
    np.testing.assert_array_equal(default, exact if form == _EXACT else mm)
    np.testing.assert_array_equal(lucid.cdist(x1, x2, 2.0, _DEFAULT).numpy(), default)


@pytest.mark.parametrize("compute_mode", [_MM, _EXACT])
def test_an_explicit_mode_ignores_the_row_count(device: str, compute_mode: str) -> None:
    """Three rows each: the default would take the difference form, and a mode overrides it."""
    x1, x2 = _t(_near_duplicates(3), device), _t(_near_duplicates(3), device)
    default = lucid.cdist(x1, x2).numpy()
    chosen = lucid.cdist(x1, x2, compute_mode=compute_mode).numpy()
    assert np.array_equal(chosen, default) == (compute_mode == _EXACT)


def test_the_difference_form_does_not_cancel(device: str) -> None:
    """Two points 1e-3 apart at magnitude 10: the matmul form's noise is the distance."""
    base = np.full((1, 8), 10.0, dtype=np.float32)
    near = base + np.float32(1e-3)
    want = float(_truth(base, near)[0, 0])
    x1, x2 = _t(base, device), _t(near, device)
    assert lucid.cdist(x1, x2).item() == pytest.approx(want, rel=1e-4, abs=0)
    assert lucid.cdist(x1, x2, compute_mode=_EXACT).item() == pytest.approx(want, rel=1e-4, abs=0)


def test_the_diagonal_of_a_set_with_itself_is_exactly_zero_up_to_25_rows(device: str) -> None:
    x = _t(_near_duplicates(25), device)
    assert np.diag(lucid.cdist(x, x).numpy()).tolist() == [0.0] * 25


def test_pdist_takes_the_default_mode(device: str) -> None:
    """``pdist`` is the strict upper triangle of ``cdist(x, x)`` in the default mode."""
    for rows in (6, 25, 26, 33):
        x = _t(_near_duplicates(rows), device)
        full = lucid.cdist(x, x).numpy()
        np.testing.assert_array_equal(F.pdist(x).numpy(), full[np.triu_indices(rows, 1)])


# ── Mixed dtypes are refused whatever the mode picks ─────────────────────────


@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("p", [2.0, 1.0, 3.0, math.inf, 0.0], ids=str)
def test_mixed_dtypes_are_refused_in_every_mode(device: str, compute_mode: str, p: float) -> None:
    """The difference form promotes and the matmul form refuses; the choice must not decide."""
    a = _t(_points(3, 4, 1.0, seed=1), device)
    b = _t(_points(5, 4, 1.0, seed=2), device, dtype=lucid.float16)
    with pytest.raises(RuntimeError, match="same dtype"):
        lucid.cdist(a, b, p, compute_mode)


# ── Derivatives are the unscaled form's ──────────────────────────────────────


def _grads(
    device: str, mode: str, magnitude: float, rows: tuple[int, int] = (6, 5)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The value over ``magnitude`` and the two gradients of ``sum(w * cdist(a, b))``."""
    scale = np.float32(magnitude)
    a = _t(_points(rows[0], 3, 1.0, seed=4) * scale, device, requires_grad=True)
    b = _t(_points(rows[1], 3, 1.0, seed=5) * scale, device, requires_grad=True)
    out = lucid.cdist(a, b, compute_mode=mode)
    (out * _t(_points(rows[0], rows[1], 1.0, seed=6), device)).sum().backward()
    assert a.grad is not None and b.grad is not None
    return out.numpy() / scale, a.grad.numpy(), b.grad.numpy()


@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("rows", [(6, 5), (30, 28)], ids=str)
@pytest.mark.parametrize("magnitude", [1e-20, 1e-30, 1e25, 1e30], ids=lambda m: f"{m:g}")
def test_the_gradient_does_not_depend_on_the_scale(
    device: str, compute_mode: str, magnitude: float, rows: tuple[int, int]
) -> None:
    """``d|x|/dx`` is degree 0: the gradient at ``c x`` is the gradient at ``x``."""
    value, ga, gb = _grads(device, compute_mode, 1.0, rows)
    scaled_value, scaled_ga, scaled_gb = _grads(device, compute_mode, magnitude, rows)
    for got, want in ((scaled_ga, ga), (scaled_gb, gb), (scaled_value, value)):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5 * float(np.abs(want).max()))


@pytest.mark.parametrize("compute_mode", [_MM, _EXACT])
@pytest.mark.parametrize("magnitude", [1e-20, 1e25], ids=lambda m: f"{m:g}")
def test_the_second_derivative_scales_as_the_inverse(
    device: str, compute_mode: str, magnitude: float
) -> None:
    """``d²|x|/dx²`` is degree -1: ``c * H(c x) = H(x)``."""
    a_base, b_base = _points(6, 3, 1.0, seed=4), _points(5, 3, 1.0, seed=5)
    v, w = _points(6, 5, 1.0, seed=6), _points(6, 3, 1.0, seed=7)

    def hessian_vector(scale: float) -> np.ndarray:
        s = np.float32(scale)
        a = _t(a_base * s, device, requires_grad=True)
        dist = lucid.cdist(a, _t(b_base * s, device), compute_mode=compute_mode)
        (g,) = lucid.autograd.grad((dist * _t(v, device)).sum(), [a], create_graph=True)
        (h,) = lucid.autograd.grad((g * _t(w, device)).sum(), [a])
        return h.numpy().astype(np.float64) * scale

    got, want = hessian_vector(magnitude), hessian_vector(1.0)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5 * float(np.abs(want).max()))


@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("magnitude", [1e-20, 1.0, 1e25], ids=lambda m: f"{m:g}")
def test_coincident_rows_have_a_zero_gradient_not_nan(
    device: str, compute_mode: str, magnitude: float
) -> None:
    rows = _points(4, 3, magnitude, seed=8)
    a = _t(rows, device, requires_grad=True)
    lucid.cdist(a, _t(rows[:2], device), compute_mode=compute_mode).sum().backward()
    assert a.grad is not None
    assert np.isfinite(a.grad.numpy()).all()


# ── Zero, nan and infinite rows answer what they did ─────────────────────────


@pytest.mark.parametrize("compute_mode", _MODES)
def test_all_zero_operands_are_zero_distance(device: str, compute_mode: str) -> None:
    zeros = _t(np.zeros((4, 3), dtype=np.float32), device, requires_grad=True)
    other = _t(np.zeros((5, 3), dtype=np.float32), device)
    out = lucid.cdist(zeros, other, compute_mode=compute_mode)
    assert out.tolist() == [[0.0] * 5] * 4
    out.sum().backward()
    assert zeros.grad is not None
    assert np.isfinite(zeros.grad.numpy()).all()


@pytest.mark.parametrize("compute_mode", _MODES)
def test_a_zero_row_beside_tiny_rows_is_their_norm(device: str, compute_mode: str) -> None:
    a = _points(4, 3, 1e-25, seed=1)
    a[0] = 0.0
    b = _points(3, 3, 1e-25, seed=2)
    got = lucid.cdist(_t(a, device), _t(b, device), compute_mode=compute_mode)
    _close(got.tolist(), _truth(a, b), rel=2e-4)


@pytest.mark.parametrize("magnitude", [1e-20, 1.0, 1e25], ids=lambda m: f"{m:g}")
def test_a_nan_row_poisons_only_its_own_pairs_in_the_difference_form(
    device: str, magnitude: float
) -> None:
    a, b = _points(4, 3, magnitude, seed=1), _points(5, 3, magnitude, seed=2)
    a[1, 2] = np.nan
    got = lucid.cdist(_t(a, device), _t(b, device), compute_mode=_EXACT).numpy()
    assert np.isnan(got[1]).all()
    _close(np.delete(got, 1, axis=0), np.delete(_truth(np.nan_to_num(a), b), 1, axis=0))


@pytest.mark.parametrize("magnitude", [1e-20, 1.0, 1e25], ids=lambda m: f"{m:g}")
def test_an_infinite_row_is_infinitely_far_in_the_difference_form(
    device: str, magnitude: float
) -> None:
    a, b = _points(4, 3, magnitude, seed=1), _points(5, 3, magnitude, seed=2)
    a[2, 0] = np.inf
    got = lucid.cdist(_t(a, device), _t(b, device), compute_mode=_EXACT).numpy()
    assert np.isposinf(got[2]).all()
    _close(np.delete(got, 2, axis=0), np.delete(_truth(np.nan_to_num(a, posinf=0.0), b), 2, axis=0))


@pytest.mark.parametrize("magnitude", [1.0, 1e-25, 1e25], ids=lambda m: f"{m:g}")
def test_the_matmul_form_answers_what_it_did_beside_a_nan_or_infinite_row(
    device: str, magnitude: float
) -> None:
    """A non-finite entry leaves its batch entry to the unscaled form, bit for bit."""
    a, b = _points(30, 5, magnitude, seed=1), _points(28, 5, magnitude, seed=2)
    a[3, 1] = np.nan
    a[9, 0] = np.inf
    b[4, 2] = -np.inf
    x1, x2 = _t(a, device), _t(b, device)
    got = lucid.cdist(x1, x2, compute_mode=_MM).numpy()
    np.testing.assert_array_equal(got, _plain_matmul_form(x1, x2).numpy())
    assert np.isnan(got[3]).all()


# ── Inside the window nothing changes ────────────────────────────────────────


def _plain_matmul_form(x1: lucid.Tensor, x2: lucid.Tensor) -> lucid.Tensor:
    """The matmul form as it was before the scaling, written out with public ops."""
    x1_sq = (x1 * x1).sum(dim=-1, keepdim=True)
    x2_sq = (x2 * x2).sum(dim=-1, keepdim=True)
    sq = x1_sq + x2_sq.swapaxes(-1, -2) - 2.0 * lucid.matmul(x1, x2.swapaxes(-1, -2))
    return lucid.sqrt(lucid.clamp(sq, 0.0, float("inf")))


def _plain_difference_form(x1: lucid.Tensor, x2: lucid.Tensor) -> lucid.Tensor:
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    return lucid.sqrt((diff * diff).sum(dim=-1))


@pytest.mark.parametrize("magnitude", [1.0, 1e-4, 1e4, 1e8, 1e-8], ids=lambda m: f"{m:g}")
@pytest.mark.parametrize(
    "shapes",
    [((40, 8), (30, 8)), ((2, 40, 8), (2, 30, 8)), ((6, 8), (3, 6, 8))],
    ids=str,
)
def test_normal_magnitudes_are_bit_identical_to_the_unscaled_forms(
    device: str, magnitude: float, shapes: tuple[tuple[int, ...], tuple[int, ...]]
) -> None:
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(shapes[0]) * magnitude).astype(np.float32)
    b = (rng.standard_normal(shapes[1]) * magnitude).astype(np.float32)
    x1, x2 = _t(a, device), _t(b, device)
    np.testing.assert_array_equal(
        lucid.cdist(x1, x2, compute_mode=_MM).numpy(), _plain_matmul_form(x1, x2).numpy()
    )
    np.testing.assert_array_equal(
        lucid.cdist(x1, x2, compute_mode=_EXACT).numpy(), _plain_difference_form(x1, x2).numpy()
    )


def test_the_same_bits_with_a_gradient_to_carry(device: str) -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((40, 8)).astype(np.float32)
    b = rng.standard_normal((30, 8)).astype(np.float32)
    plain = lucid.cdist(_t(a, device), _t(b, device), compute_mode=_MM).numpy()
    tracked = lucid.cdist(_t(a, device, requires_grad=True), _t(b, device), compute_mode=_MM)
    np.testing.assert_array_equal(tracked.numpy(), plain)


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=str)
@pytest.mark.parametrize("compute_mode", _MODES)
def test_half_inputs_keep_their_dtype(device: str, dtype: DTypeLike, compute_mode: str) -> None:
    """Half precision is not scaled (no window fits its range) and answers in its own dtype."""
    a, b = _points(6, 4, 1.0, seed=1), _points(5, 4, 1.0, seed=2)
    x1, x2 = _t(a, device, dtype=dtype), _t(b, device, dtype=dtype)
    got = lucid.cdist(x1, x2, compute_mode=compute_mode)
    assert got.dtype == dtype
    _close(got.to(lucid.float32).tolist(), _truth(a, b), rel=3e-2)


@pytest.mark.parametrize("compute_mode", _MODES)
def test_empty_operands_keep_their_shapes(device: str, compute_mode: str) -> None:
    a, b = _points(3, 4, 1.0, seed=1), _points(5, 4, 1.0, seed=2)
    cases = [(a[:0], b, (0, 5)), (a, b[:0], (3, 0)), (a[:, :0], b[:, :0], (3, 5))]
    for x1, x2, shape in cases:
        got = lucid.cdist(_t(x1, device), _t(x2, device), compute_mode=compute_mode)
        assert tuple(got.shape) == shape
        assert not np.any(got.numpy())


# ── A compile trace takes the same route as eager ────────────────────────────


@pytest.mark.parametrize("compute_mode", [_MM, _EXACT])
def test_a_traced_cdist_rescales_at_replay(device: str, compute_mode: str) -> None:
    """The divisor is computed in the graph, so a trace of normal data handles tiny data."""
    fn = lucid.compile(lambda p, q: lucid.cdist(p, q, compute_mode=compute_mode))
    fn(_t(_points(6, 4, 1.0, seed=1), device), _t(_points(5, 4, 1.0, seed=2), device))
    for magnitude in (1e-20, 1e25):
        a, b = _points(6, 4, magnitude, seed=3), _points(5, 4, magnitude, seed=4)
        got = fn(_t(a, device), _t(b, device))
        _close(got.tolist(), _truth(a, b), rel=2e-4)


# ── Parity with the reference ────────────────────────────────────────────────


def _ref_cdist(ref: ModuleType, a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    return np.asarray(ref.cdist(ref.tensor(a), ref.tensor(b), compute_mode=mode).numpy())


@pytest.mark.parity
@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("rows", [(5, 7), (25, 25), (26, 5), (5, 26), (30, 40)], ids=str)
def test_matches_the_reference_at_normal_magnitude(
    ref: ModuleType, device: str, compute_mode: str, rows: tuple[int, int]
) -> None:
    a, b = _points(rows[0], 8, 1.0, seed=1), _points(rows[1], 8, 1.0, seed=2)
    got = lucid.cdist(_t(a, device), _t(b, device), compute_mode=compute_mode).numpy()
    np.testing.assert_allclose(got, _ref_cdist(ref, a, b, compute_mode), rtol=1e-4, atol=0.0)


@pytest.mark.parity
@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("magnitude", _EXTREMES, ids=lambda m: f"{m:g}")
def test_matches_the_reference_wherever_it_is_accurate(
    ref: ModuleType, device: str, compute_mode: str, magnitude: float
) -> None:
    """The reference loses these answers (``0`` / ``inf`` / ``nan``, or digits to a subnormal).

    Where its answer is the true distance, so is this one; where it is not, this
    one still is.
    """
    rows = _two_rows(magnitude)
    truth = float(_truth(rows[:1], rows[1:])[0, 0])
    got = lucid.cdist(_t(rows[:1], device), _t(rows[1:], device), compute_mode=compute_mode).item()
    assert got == pytest.approx(truth, rel=1e-5, abs=0)
    theirs = float(_ref_cdist(ref, rows[:1], rows[1:], compute_mode)[0, 0])
    if math.isfinite(theirs) and theirs == pytest.approx(truth, rel=1e-5, abs=0):
        assert got == pytest.approx(theirs, rel=1e-5, abs=0)


@pytest.mark.parity
@pytest.mark.parametrize("rows", [2, 5, 25, 26, 33], ids=lambda n: f"n{n}")
@pytest.mark.parametrize("p", [2.0, 1.0, math.inf], ids=str)
def test_pdist_matches_the_reference(ref: ModuleType, device: str, rows: int, p: float) -> None:
    x = _points(rows, 6, 1.0, seed=rows)
    want = ref.nn.functional.pdist(ref.tensor(x), p).numpy()
    np.testing.assert_allclose(F.pdist(_t(x, device), p).numpy(), want, rtol=1e-4, atol=0.0)


@pytest.mark.parity
@pytest.mark.parametrize("magnitude", [1e-20, 1e-30], ids=lambda m: f"{m:g}")
def test_pdist_at_tiny_magnitude_matches_the_reference_cpu(
    ref: ModuleType, device: str, magnitude: float
) -> None:
    """The reference's CPU ``pdist`` keeps ``1e-20`` to a few digits and flushes ``1e-30``."""
    rows = _two_rows(magnitude)
    want = float(ref.nn.functional.pdist(ref.tensor(rows)).item())
    truth = 4.0 * float(rows[0, 0])
    got = F.pdist(_t(rows, device)).item()
    assert got == pytest.approx(truth, rel=1e-5, abs=0)
    if want == pytest.approx(truth, rel=1e-5, abs=0):
        assert got == pytest.approx(want, rel=1e-5, abs=0)


@pytest.mark.parity
@pytest.mark.parametrize("compute_mode", [_DEFAULT, _EXACT])
def test_near_coincident_points_match_the_reference(
    ref: ModuleType, device: str, compute_mode: str
) -> None:
    """Where the reference takes the difference form, so does this: no cancellation."""
    a = _near_duplicates(6)
    b = a + np.float32(1e-3)
    got = lucid.cdist(_t(a, device), _t(b, device), compute_mode=compute_mode).numpy()
    np.testing.assert_allclose(got, _ref_cdist(ref, a, b, compute_mode), rtol=1e-4, atol=0.0)


@pytest.mark.parity
@pytest.mark.parametrize("compute_mode", _MODES)
@pytest.mark.parametrize("rows", [(6, 5), (30, 28)], ids=str)
def test_gradients_match_the_reference(
    ref: ModuleType, device: str, compute_mode: str, rows: tuple[int, int]
) -> None:
    a, b = _points(rows[0], 3, 1.0, seed=4), _points(rows[1], 3, 1.0, seed=5)
    weight = _points(rows[0], rows[1], 1.0, seed=6)
    ra, rb = ref.tensor(a, requires_grad=True), ref.tensor(b, requires_grad=True)
    (ref.cdist(ra, rb, compute_mode=compute_mode) * ref.tensor(weight)).sum().backward()
    _, ga, gb = _grads(device, compute_mode, 1.0, rows)
    np.testing.assert_allclose(ga, ra.grad.numpy(), rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(gb, rb.grad.numpy(), rtol=1e-3, atol=1e-5)
