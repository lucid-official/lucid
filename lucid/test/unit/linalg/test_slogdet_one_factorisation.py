"""``slogdet``'s gradient comes from the factorisation its value came from (LCD-954).

``slogdet`` factors ``Aᵀ`` and reads the sign, ``log|det|`` and the ``singular``
flag off it.  A matrix with a duplicated row factors, as ``Aᵀ``, to a last pivot
of rounding noise, so the flag says regular and ``log|det|`` is a finite
``-18``.  The gradient then inverted ``A`` itself -- a second, independent
factorisation, which met an exact zero -- and ``backward`` raised ``LinAlgError``
for an input ``forward`` had answered.  The gradient is now solved from the
factors the value was read from, so the two cannot disagree about which matrix is
singular: a matrix counted regular has a (large) finite gradient, one counted
singular keeps its ``cof / det`` infinities, and ``backward`` never raises.
"""

import math
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available


def _placements() -> list[object]:
    metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")
    return [
        pytest.param("cpu", "float32", id="cpu-float32"),
        pytest.param("cpu", "float64", id="cpu-float64"),
        pytest.param("metal", "float32", id="metal-float32", marks=metal),
    ]


def _base() -> lucid.Tensor:
    lucid.manual_seed(0)
    return lucid.randn(4, 4, dtype=lucid.float64)


def _duplicate_row() -> lucid.Tensor:
    a = _base()
    a[3] = a[1]
    return a


def _power_of_two_row() -> lucid.Tensor:
    a = _base()
    a[3] = 2.0 * a[1]
    return a


def _inexact_row_multiples() -> lucid.Tensor:
    a = _base()
    a[3] = 0.1 * a[0] + 0.3 * a[1]
    return a


def _inexact_column_multiples() -> lucid.Tensor:
    a = _base()
    a[:, 3] = 0.1 * a[:, 0] + 0.3 * a[:, 1]
    return a


def _duplicate_column() -> lucid.Tensor:
    a = _base()
    a[:, 3] = a[:, 1]
    return a


def _power_of_two_column() -> lucid.Tensor:
    a = _base()
    a[:, 3] = 4.0 * a[:, 1]
    return a


def _rank_one() -> lucid.Tensor:
    return lucid.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [4.0, 8.0, 12.0]], dtype=lucid.float64)


def _zero() -> lucid.Tensor:
    return lucid.zeros(3, 3, dtype=lucid.float64)


#: Rank-deficient, but ``Aᵀ``'s last pivot is rounding noise: the value is finite.
_NOISE_PIVOT = {
    "duplicate row": _duplicate_row,
    "power-of-two row": _power_of_two_row,
    "inexact row multiples": _inexact_row_multiples,
    "inexact column multiples": _inexact_column_multiples,
}

#: ``Aᵀ``'s elimination reaches an exact zero pivot: the value is ``-inf``.
_EXACT_ZERO_PIVOT = {
    "duplicate column": _duplicate_column,
    "power-of-two column": _power_of_two_column,
    "rank one": _rank_one,
    "zero": _zero,
}


def _tensor(rows: lucid.Tensor, device: str, dtype: str) -> lucid.Tensor:
    return rows.to(getattr(lucid, dtype)).to(device).clone()


def _eye_like(A: lucid.Tensor) -> lucid.Tensor:
    return lucid.eye(int(A.shape[-1]), dtype=A.dtype, device=A.device)


def _values(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.to("cpu").reshape(-1).tolist()]


def _grad_of(A: lucid.Tensor) -> list[float]:
    """``∂ log|det A| / ∂A`` of a ``requires_grad`` copy of ``A``, flat."""
    leaf = A.detach().clone().requires_grad_()
    lucid.linalg.slogdet(leaf)[1].sum().backward()
    assert leaf.grad is not None
    return _values(leaf.grad)


@pytest.mark.parametrize(("device", "dtype"), _placements())
def test_the_issue_repro_backward_no_longer_raises(device: str, dtype: str) -> None:
    lucid.manual_seed(0)
    a = lucid.randn(4, 4, dtype=getattr(lucid, dtype))
    a[3] = a[1]
    A = a.to(device).requires_grad_()
    sign, logabs = lucid.linalg.slogdet(A)
    assert abs(sign.item()) == 1.0 and math.isfinite(logabs.item())
    logabs.backward()
    assert A.grad is not None
    assert all(math.isfinite(v) for v in _values(A.grad))


@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("kind", list(_NOISE_PIVOT))
def test_a_matrix_with_a_finite_value_has_a_finite_gradient(
    kind: str, device: str, dtype: str
) -> None:
    A = _tensor(_NOISE_PIVOT[kind](), device, dtype).requires_grad_()
    sign, logabs = lucid.linalg.slogdet(A)
    assert abs(sign.item()) == 1.0 and math.isfinite(logabs.item()), "not a noise-pivot input"
    logabs.backward()
    assert A.grad is not None
    got = _values(A.grad)
    assert all(math.isfinite(v) for v in got)
    # Rank n - 1 with a noise pivot: the inverse is huge, not zero.
    assert max(abs(v) for v in got) > 1e6


@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("kind", list(_NOISE_PIVOT))
def test_the_gradient_is_solved_from_the_factors_of_a_transpose(
    kind: str, device: str, dtype: str
) -> None:
    # The forward's own factorisation, through the public API: its solve against
    # the identity is ``A⁻ᵀ``, and it is the gradient to the last bit.
    A = _tensor(_NOISE_PIVOT[kind](), device, dtype)
    LU, pivots = lucid.linalg.lu_factor(A.mT)
    want = lucid.linalg.lu_solve(LU, pivots, _eye_like(A))
    assert _grad_of(A) == _values(want)


@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("kind", list(_EXACT_ZERO_PIVOT))
def test_an_exactly_singular_matrix_keeps_its_infinite_gradient(
    kind: str, device: str, dtype: str
) -> None:
    A = _tensor(_EXACT_ZERO_PIVOT[kind](), device, dtype)
    _, logabs = lucid.linalg.slogdet(A)
    assert logabs.item() == -math.inf, "not an exact-zero-pivot input"
    # cof / det: ``±inf`` along the cofactor, NaN where it is zero -- never a finite
    # number, and never an error.
    assert not any(math.isfinite(v) for v in _grad_of(A))


def _mixed_batch(device: str, dtype: str) -> lucid.Tensor:
    lucid.manual_seed(11)
    regular = lucid.randn(2, 4, 4, dtype=lucid.float64) + 3.0 * lucid.eye(4, dtype=lucid.float64)
    members = [
        regular[0],
        _duplicate_row(),
        _duplicate_column(),
        _power_of_two_row(),
        regular[1],
        _inexact_column_multiples(),
    ]
    return lucid.stack(members).to(getattr(lucid, dtype)).to(device)


@pytest.mark.parametrize(("device", "dtype"), _placements())
def test_a_batch_mixing_regular_noise_pivot_and_exactly_singular_members(
    device: str, dtype: str
) -> None:
    A = _mixed_batch(device, dtype).requires_grad_()
    _, logabs = lucid.linalg.slogdet(A)
    finite = [math.isfinite(v) for v in _values(logabs)]
    assert finite == [True, True, False, True, True, True]
    logabs.sum().backward()
    assert A.grad is not None
    grad = A.grad.to("cpu")
    for i in (0, 1, 3, 4, 5):
        assert all(math.isfinite(v) for v in _values(grad[i])), i
    assert not any(math.isfinite(v) for v in _values(grad[2]))
    # A regular member's gradient is its own inverse transpose, whoever shares the batch.
    scale = max(abs(v) for v in _values(grad[0]))
    rtol = 1e-4 if dtype == "float32" else 1e-10
    for i in (0, 4):
        want = lucid.linalg.inv(A.detach()[i].to("cpu")).mT
        assert max(abs(g - w) for g, w in zip(_values(grad[i]), _values(want), strict=True)) <= (
            rtol * scale
        ), i


@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("kind", list(_NOISE_PIVOT))
def test_the_second_derivative_at_a_matrix_with_a_finite_value(
    kind: str, device: str, dtype: str
) -> None:
    A = _tensor(_NOISE_PIVOT[kind](), device, dtype).requires_grad_()
    (g,) = lucid.autograd.grad(lucid.linalg.slogdet(A)[1], [A], create_graph=True)
    weights = lucid.arange(16, dtype=A.dtype, device=device).reshape(4, 4)
    (h,) = lucid.autograd.grad((g * weights).sum(), [A])
    assert all(math.isfinite(v) for v in _values(h))
    assert max(abs(v) for v in _values(h)) > 1e12


def _second_derivative(batch: lucid.Tensor) -> lucid.Tensor:
    """``∂/∂A Σ w ⊙ ∂(Σ log|det|)/∂A`` for a fixed ramp ``w``, through ``create_graph``."""
    A = batch.clone().requires_grad_()
    (g,) = lucid.autograd.grad(lucid.linalg.slogdet(A)[1].sum(), [A], create_graph=True)
    n = int(A.shape[-1])
    weights = lucid.arange(n * n, dtype=A.dtype).reshape(n, n)
    (h,) = lucid.autograd.grad((g * weights).sum(), [A])
    return h


def test_a_batch_members_second_derivative_ignores_a_noise_pivot_neighbour() -> None:
    lucid.manual_seed(5)
    regular = lucid.randn(3, 3, dtype=lucid.float64) + 3.0 * lucid.eye(3, dtype=lucid.float64)
    noisy = lucid.randn(3, 3, dtype=lucid.float64)
    noisy[2] = noisy[0]
    alone = _second_derivative(regular.unsqueeze(0))
    together = _second_derivative(lucid.stack([regular, noisy]))
    assert all(math.isfinite(v) for v in _values(together))
    assert lucid.allclose(together[0], alone[0], rtol=1e-9, atol=0.0)


def test_the_gradient_of_a_batch_differentiates_twice() -> None:
    lucid.manual_seed(6)
    A = lucid.randn(2, 3, 3, dtype=lucid.float64) + 3.0 * lucid.eye(3, dtype=lucid.float64)
    assert lucid.autograd.gradgradcheck(lambda a: lucid.linalg.slogdet(a)[1], [A.requires_grad_()])


@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("kind", ["inexact row multiples", "duplicate row"])
def test_logdet_of_a_matrix_with_a_finite_value_backpropagates(
    kind: str, device: str, dtype: str
) -> None:
    # A positive determinant gives its log; a negative one NaN, and the gradient
    # through the NaN branch is a finite zero -- the zero ``backward`` once raised on.
    A = _tensor(_NOISE_PIVOT[kind](), device, dtype).requires_grad_()
    negative = lucid.linalg.slogdet(A.detach())[0].item() < 0
    out = lucid.logdet(A)
    assert math.isnan(out.item()) == negative
    out.backward()
    assert A.grad is not None
    got = _values(A.grad)
    assert all(math.isfinite(v) for v in got)
    if negative:
        assert all(v == 0.0 for v in got)


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("kind", list(_NOISE_PIVOT))
def test_the_gradient_is_the_references(
    kind: str, device: str, dtype: str, ref: ModuleType
) -> None:
    rows = _NOISE_PIVOT[kind]().tolist()
    A = _tensor(lucid.tensor(rows, dtype=lucid.float64), device, dtype).requires_grad_()
    A_ref = ref.tensor(rows, dtype=getattr(ref, dtype), requires_grad=True)
    sign, logabs = lucid.linalg.slogdet(A)
    want_sign, want_log = ref.linalg.slogdet(A_ref)
    assert sign.item() == float(want_sign.detach())
    assert math.isclose(logabs.item(), float(want_log.detach()), rel_tol=1e-4)
    logabs.backward()
    want_log.backward()
    assert A.grad is not None
    got, want = _values(A.grad), [float(v) for v in A_ref.grad.reshape(-1).tolist()]
    scale = max(abs(w) for w in want)
    assert max(abs(g - w) for g, w in zip(got, want, strict=True)) <= 1e-4 * scale


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("kind", list(_NOISE_PIVOT))
def test_the_second_derivative_is_the_references(
    kind: str, device: str, dtype: str, ref: ModuleType
) -> None:
    rows = _NOISE_PIVOT[kind]().tolist()
    A = _tensor(lucid.tensor(rows, dtype=lucid.float64), device, dtype).requires_grad_()
    A_ref = ref.tensor(rows, dtype=getattr(ref, dtype), requires_grad=True)
    weights = lucid.arange(16, dtype=A.dtype, device=device).reshape(4, 4)
    weights_ref = ref.arange(16, dtype=getattr(ref, dtype)).reshape(4, 4)
    (g,) = lucid.autograd.grad(lucid.linalg.slogdet(A)[1], [A], create_graph=True)
    (h,) = lucid.autograd.grad((g * weights).sum(), [A])
    (g_ref,) = ref.autograd.grad(ref.linalg.slogdet(A_ref)[1], [A_ref], create_graph=True)
    (h_ref,) = ref.autograd.grad((g_ref * weights_ref).sum(), [A_ref])
    got, want = _values(h), [float(v) for v in h_ref.reshape(-1).tolist()]
    scale = max(abs(w) for w in want)
    assert max(abs(x - w) for x, w in zip(got, want, strict=True)) <= 1e-3 * scale


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _placements())
def test_the_mixed_batch_matches_the_reference_where_it_is_finite(
    device: str, dtype: str, ref: ModuleType
) -> None:
    A = _mixed_batch(device, dtype).requires_grad_()
    A_ref = ref.tensor(_mixed_batch("cpu", "float64").tolist(), dtype=getattr(ref, dtype))
    A_ref.requires_grad_()
    lucid.linalg.slogdet(A)[1].sum().backward()
    ref.linalg.slogdet(A_ref)[1].sum().backward()
    assert A.grad is not None
    for i in (0, 1, 3, 4, 5):
        got, want = _values(A.grad[i]), [float(v) for v in A_ref.grad[i].reshape(-1).tolist()]
        scale = max(abs(w) for w in want)
        assert max(abs(g - w) for g, w in zip(got, want, strict=True)) <= 1e-4 * scale, i


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("kind", ["inexact row multiples", "duplicate row"])
def test_logdet_has_the_references_value_and_gradient(
    kind: str, device: str, dtype: str, ref: ModuleType
) -> None:
    rows = _NOISE_PIVOT[kind]().tolist()
    A = _tensor(lucid.tensor(rows, dtype=lucid.float64), device, dtype).requires_grad_()
    A_ref = ref.tensor(rows, dtype=getattr(ref, dtype), requires_grad=True)
    out, out_ref = lucid.logdet(A), ref.logdet(A_ref)
    assert math.isnan(out.item()) == math.isnan(float(out_ref.detach()))
    out.backward()
    out_ref.backward()
    assert A.grad is not None
    got, want = _values(A.grad), [float(v) for v in A_ref.grad.reshape(-1).tolist()]
    scale = max([abs(w) for w in want] + [1.0])
    assert max(abs(g - w) for g, w in zip(got, want, strict=True)) <= 1e-4 * scale
