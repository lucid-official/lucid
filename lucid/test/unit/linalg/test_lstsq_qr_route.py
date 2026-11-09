"""LCD-941: ``lstsq`` takes a QR route for a full-rank ``A``.

The reference's default driver solves a full-rank system through a QR
factorisation; Lucid answered every matrix through the pseudo-inverse (two
SVDs) and was 4-17x slower.  ``gels`` and ``gelsy`` now try QR first, decide
full rank from a conditioning estimate, and send what the estimate does
not clear to the SVD route.  What must not change is the
answer — the minimum-norm solution, the residuals / rank / singular values
each driver returns, the derivative in both operands — and the rank of a
rank-deficient matrix, which the SVD route decides.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
import lucid.linalg._lstsq as _lstsq
from lucid.autograd import grad

_SHAPES = [(6, 3), (3, 6), (5, 5), (40, 7), (7, 40), (2, 6, 3), (3, 4, 9), (2, 5, 5)]
_DRIVERS = [None, "gels", "gelsy", "gelsd", "gelss"]


def _system(shape: tuple[int, ...], k: int = 2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape), rng.standard_normal((*shape[:-2], shape[-2], k))


def _dtype(device: str) -> tuple[lucid.dtype, float]:
    return (lucid.float32, 2e-3) if device == "metal" else (lucid.float64, 1e-9)


class _Svd:
    """Records whether the SVD route ran; it is the only route that ranks."""

    calls = 0


@pytest.fixture
def svd_calls(monkeypatch: pytest.MonkeyPatch) -> type[_Svd]:
    _Svd.calls = 0
    original = _lstsq._svd_route

    def counted(*args: object, **kwargs: object) -> object:
        _Svd.calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_lstsq, "_svd_route", counted)
    return _Svd


# ── the answer is the reference's, for every driver and shape ────────────────


@pytest.mark.parity
@pytest.mark.parametrize("driver", _DRIVERS)
@pytest.mark.parametrize("shape", _SHAPES)
def test_every_output_matches_the_reference(
    ref: ModuleType, device: str, shape: tuple[int, ...], driver: str | None
) -> None:
    dt, tol = _dtype(device)
    a_np, b_np = _system(shape)
    got = LA.lstsq(
        lucid.tensor(a_np, dtype=dt, device=device),
        lucid.tensor(b_np, dtype=dt, device=device),
        driver=driver,
    )
    rdt = ref.float32 if device == "metal" else ref.float64
    # Metal's own default driver is ``gels``, the one a GPU backend defaults to.
    want = ref.linalg.lstsq(
        ref.tensor(a_np, dtype=rdt),
        ref.tensor(b_np, dtype=rdt),
        driver=driver or ("gels" if device == "metal" else "gelsy"),
    )
    for g, w in zip(got, want, strict=True):
        assert tuple(g.shape) == tuple(w.shape)
        if g.numel():
            np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=tol, atol=tol)


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (2, 6, 3)])
def test_a_vector_right_hand_side_keeps_its_shape(
    ref: ModuleType, device: str, shape: tuple[int, ...]
) -> None:
    dt, tol = _dtype(device)
    a_np, _ = _system(shape)
    b_np = np.random.default_rng(1).standard_normal((*shape[:-2], shape[-2]))
    got = LA.lstsq(
        lucid.tensor(a_np, dtype=dt, device=device), lucid.tensor(b_np, dtype=dt, device=device)
    )
    rdt = ref.float32 if device == "metal" else ref.float64
    want = ref.linalg.lstsq(ref.tensor(a_np, dtype=rdt), ref.tensor(b_np, dtype=rdt))
    assert tuple(got.solution.shape) == tuple(want.solution.shape)
    np.testing.assert_allclose(got.solution.numpy(), want.solution.numpy(), rtol=tol, atol=tol)


def test_a_right_hand_side_of_batch_one_is_broadcast_over_a_batch_of_matrices(device: str) -> None:
    dt, tol = _dtype(device)
    a_np, _ = _system((4, 6, 3))
    b_np = np.random.default_rng(2).standard_normal((1, 6, 2))
    got = LA.lstsq(
        lucid.tensor(a_np, dtype=dt, device=device), lucid.tensor(b_np, dtype=dt, device=device)
    )
    want = np.stack([np.linalg.lstsq(a, b_np[0], rcond=None)[0] for a in a_np])
    np.testing.assert_allclose(got.solution.numpy(), want, rtol=tol, atol=tol)
    # Metal's default driver, ``gels``, reports no rank.
    assert tuple(got.rank.shape) == ((4,) if device == "cpu" else (0,))


# ── rank deficiency is still decided by the SVD ──────────────────────────────


def _deficient(kind: str) -> np.ndarray:
    rng = np.random.default_rng(5)
    if kind == "duplicate-column":
        a = rng.standard_normal((8, 4))
        a[:, 2] = a[:, 0]
        return a
    if kind == "low-rank-tall":
        return rng.standard_normal((9, 2)) @ rng.standard_normal((2, 5))
    if kind == "low-rank-wide":
        return rng.standard_normal((4, 2)) @ rng.standard_normal((2, 9))
    if kind == "zero":
        return np.zeros((5, 3))
    # Singular values down to 1e-16: below the cutoff, so rank deficient, but not exactly.
    u, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    v, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    return (u * np.logspace(0, -16, 6)) @ v.T


@pytest.mark.parity
@pytest.mark.parametrize("driver", [None, "gelsy", "gelsd"])
@pytest.mark.parametrize(
    "kind", ["duplicate-column", "low-rank-tall", "low-rank-wide", "zero", "graded"]
)
def test_a_rank_deficient_matrix_is_ranked_and_solved_like_the_reference(
    ref: ModuleType, device_cpu_only: str, kind: str, driver: str | None
) -> None:
    a_np = _deficient(kind)
    b_np = np.random.default_rng(6).standard_normal((a_np.shape[0], 2))
    got = LA.lstsq(lucid.tensor(a_np), lucid.tensor(b_np), driver=driver)
    want = ref.linalg.lstsq(ref.tensor(a_np), ref.tensor(b_np), driver=driver or "gelsy")
    assert int(got.rank.item()) == int(want.rank.item())
    # The graded matrix keeps singular values near 1e-13: its solution is the
    # ill-conditioned one, agreeing to the conditioning, not to the digit.
    tol = 1e-2 if kind == "graded" else 1e-6
    np.testing.assert_allclose(got.solution.numpy(), want.solution.numpy(), rtol=tol, atol=1e-8)


@pytest.mark.parametrize("kind", ["duplicate-column", "low-rank-tall", "zero"])
def test_a_rank_deficient_matrix_never_reaches_the_qr_answer(
    svd_calls: type[_Svd], kind: str
) -> None:
    a_np = _deficient(kind)
    lucid.linalg.lstsq(lucid.tensor(a_np), lucid.tensor(np.ones((a_np.shape[0], 1))))
    assert svd_calls.calls == 1


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (5, 5), (2, 6, 3)])
@pytest.mark.parametrize("driver", [None, "gels", "gelsy"])
def test_a_full_rank_matrix_never_runs_an_svd(
    svd_calls: type[_Svd], device: str, shape: tuple[int, ...], driver: str | None
) -> None:
    dt, _ = _dtype(device)
    a_np, b_np = _system(shape)
    LA.lstsq(
        lucid.tensor(a_np, dtype=dt, device=device),
        lucid.tensor(b_np, dtype=dt, device=device),
        driver=driver,
    )
    assert svd_calls.calls == 0


@pytest.mark.parametrize("driver", ["gelsd", "gelss"])
def test_the_singular_value_drivers_always_take_the_svd_route(
    svd_calls: type[_Svd], driver: str
) -> None:
    a_np, b_np = _system((6, 3))
    got = LA.lstsq(lucid.tensor(a_np), lucid.tensor(b_np), driver=driver)
    assert svd_calls.calls == 1
    assert tuple(got.singular_values.shape) == (3,)


def test_a_rcond_above_the_condition_number_sends_a_full_rank_matrix_to_the_svd(
    svd_calls: type[_Svd],
) -> None:
    a_np = np.diag([1.0, 1e-3, 1.0])
    b = lucid.tensor(np.ones((3, 1)))
    got = LA.lstsq(lucid.tensor(a_np), b, rcond=1e-2)
    assert svd_calls.calls == 1
    assert int(got.rank.item()) == 2


# ── a batch splits by the estimate, and comes back in order ───────────────


@pytest.mark.parity
def test_one_deficient_matrix_in_a_batch_is_the_only_one_ranked_by_svd(
    ref: ModuleType, svd_calls: type[_Svd], device_cpu_only: str
) -> None:
    rng = np.random.default_rng(8)
    a_np = rng.standard_normal((5, 6, 3))
    a_np[1, :, 2] = a_np[1, :, 0]
    a_np[3] = a_np[3, :, :1] @ rng.standard_normal((1, 3))
    b_np = rng.standard_normal((5, 6, 2))
    got = LA.lstsq(lucid.tensor(a_np), lucid.tensor(b_np))
    want = ref.linalg.lstsq(ref.tensor(a_np), ref.tensor(b_np))
    assert svd_calls.calls == 1
    assert got.rank.tolist() == want.rank.tolist() == [3, 2, 3, 1, 3]
    np.testing.assert_allclose(got.solution.numpy(), want.solution.numpy(), rtol=1e-6, atol=1e-8)


def test_a_non_finite_matrix_in_a_batch_is_nan_and_its_neighbours_are_not(device: str) -> None:
    a_np = np.stack([np.eye(3), np.full((3, 3), np.nan), 2 * np.eye(3)]).astype(np.float32)
    b_np = np.ones((3, 3, 1), dtype=np.float32)
    got = LA.lstsq(lucid.tensor(a_np, device=device), lucid.tensor(b_np, device=device))
    sol = got.solution.numpy()
    np.testing.assert_allclose(sol[0], 1.0, atol=1e-5)
    assert np.isnan(sol[1]).all()
    np.testing.assert_allclose(sol[2], 0.5, atol=1e-5)


# ── the derivative is the pseudo-inverse's, in both operands, twice ──────────


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (5, 5), (2, 6, 3), (2, 3, 6)])
def test_the_qr_route_gradient_equals_the_svd_routes(shape: tuple[int, ...]) -> None:
    a_np, b_np = _system(shape)
    w = np.random.default_rng(9).standard_normal((*shape[:-2], shape[-1], 2))
    grads = []
    for driver in ("gelsy", "gelsd"):
        a = lucid.tensor(a_np, requires_grad=True)
        b = lucid.tensor(b_np, requires_grad=True)
        (LA.lstsq(a, b, driver=driver).solution * lucid.tensor(w)).sum().backward()
        assert a.grad is not None and b.grad is not None
        grads.append((a.grad.numpy(), b.grad.numpy()))
    for qr_grad, svd_grad in zip(grads[0], grads[1], strict=True):
        np.testing.assert_allclose(qr_grad, svd_grad, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (5, 5)])
def test_a_gradient_in_one_operand_alone_is_still_recorded(shape: tuple[int, ...]) -> None:
    a_np, b_np = _system(shape)
    only_b = lucid.tensor(b_np, requires_grad=True)
    LA.lstsq(lucid.tensor(a_np), only_b).solution.sum().backward()
    assert only_b.grad is not None and np.isfinite(only_b.grad.numpy()).all()
    only_a = lucid.tensor(a_np, requires_grad=True)
    LA.lstsq(only_a, lucid.tensor(b_np)).solution.sum().backward()
    assert only_a.grad is not None and np.isfinite(only_a.grad.numpy()).all()


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (5, 5)])
def test_the_qr_route_differentiates_twice_like_the_svd_route(shape: tuple[int, ...]) -> None:
    a_np, b_np = _system(shape)
    w = np.random.default_rng(10).standard_normal((shape[-1], 2))
    out = []
    for driver in ("gelsy", "gelsd"):
        a = lucid.tensor(a_np, requires_grad=True)
        b = lucid.tensor(b_np)
        loss = (LA.lstsq(a, b, driver=driver).solution * lucid.tensor(w)).sum()
        (g,) = grad(loss, [a], create_graph=True)
        (h,) = grad((g * g).sum(), [a])
        out.append(h.numpy())
    np.testing.assert_allclose(out[0], out[1], rtol=1e-7, atol=1e-8)


def test_gradcheck_through_the_qr_route() -> None:
    a_np, b_np = _system((5, 3))
    a = lucid.tensor(a_np, requires_grad=True)
    b = lucid.tensor(b_np, requires_grad=True)
    assert lucid.autograd.gradcheck(lambda x, y: LA.lstsq(x, y).solution.sum(), [a, b])


# ── the condition estimate ───────────────────────────────────────────────────────


def _kappa(a: np.ndarray) -> float:
    s = np.linalg.svd(a, compute_uv=False)
    return float(s[0] / s[-1])


def _estimate(a: np.ndarray) -> float:
    route = _lstsq._QrLeastSquares.of(lucid.tensor(a))
    return float(_lstsq._estimate_condition(route._r).item())


@pytest.mark.parametrize("kind", ["gauss", "graded-1e3", "graded-1e7", "hilbert-8", "kahan"])
def test_the_condition_estimate_stays_within_the_margin_of_the_truth(kind: str) -> None:
    rng = np.random.default_rng(11)
    n = 40
    if kind == "gauss":
        a = rng.standard_normal((n, n))
    elif kind.startswith("graded"):
        u, _ = np.linalg.qr(rng.standard_normal((n, n)))
        v, _ = np.linalg.qr(rng.standard_normal((n, n)))
        a = (u * np.logspace(0, -int(kind[-1]), n)) @ v.T
    elif kind == "hilbert-8":
        a = 1.0 / (np.arange(8)[:, None] + np.arange(8)[None, :] + 1)
    else:
        c = 0.285
        a = np.diag(np.sqrt(1 - c * c) ** np.arange(n)) @ (
            np.triu(-c * np.ones((n, n)), 1) + np.eye(n)
        )
    ratio = _estimate(a) / _kappa(a)
    # Below 1/_ESTIMATE_MARGIN the certificate could clear a rank-deficient matrix.
    assert 1.0 / _lstsq._ESTIMATE_MARGIN < ratio < 1.5


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_the_condition_estimate_does_not_depend_on_scale(scale: float) -> None:
    a = np.random.default_rng(12).standard_normal((12, 12))
    lo = lucid.tensor((a * scale).astype(np.float32))
    hi = lucid.tensor(a.astype(np.float32))
    est = lambda t: float(
        _lstsq._estimate_condition(_lstsq._QrLeastSquares.of(t)._r).item()
    )  # noqa: E731
    assert est(lo) == pytest.approx(est(hi), rel=1e-3)


def test_a_singular_triangle_has_no_finite_estimate() -> None:
    r = lucid.tensor(np.triu(np.ones((4, 4))) * np.array([1.0, 1.0, 0.0, 1.0]))
    assert not np.isfinite(float(_lstsq._estimate_condition(r).item()))


# ── the estimate's blind spot ────────────────────────────────────────────────
#
# An inverse iteration finds the smallest singular direction only from a start
# with a component along it.  Two starts of different structure make that a
# matter of construction; a matrix whose smallest right singular vector is
# orthogonal to the ones in use is the case no start choice removes.


def _starts(n: int) -> np.ndarray:
    return _lstsq._start_vectors(n, lucid.zeros(n, n, dtype=lucid.float64)).numpy()


def _near_deficient(n: int, rcond: float, orthogonal_to: np.ndarray) -> np.ndarray:
    """``σ = (1, …, 1, 10 s, s)`` with ``s = 0.4 rcond``; ``V``'s last column avoids ``orthogonal_to``."""
    rng = np.random.default_rng(5)
    basis, _ = np.linalg.qr(orthogonal_to)
    w = rng.standard_normal(n)
    w -= basis @ (basis.T @ w)
    w /= np.linalg.norm(w)
    q, _ = np.linalg.qr(np.concatenate([w[:, None], rng.standard_normal((n, n - 1))], axis=1))
    v = np.concatenate([q[:, 1:], q[:, :1]], axis=1)
    s = 0.4 * rcond
    u, _ = np.linalg.qr(rng.standard_normal((n + 5, n)))
    return (u * np.concatenate([np.ones(n - 2), [10 * s, s]])) @ v.T


def _ranks(a: np.ndarray, rcond: float) -> tuple[int, int]:
    b = lucid.tensor(np.random.default_rng(6).standard_normal((a.shape[0], 1)))
    default = LA.lstsq(lucid.tensor(a), b, rcond=rcond)
    exact = LA.lstsq(lucid.tensor(a), b, rcond=rcond, driver="gelsd")
    return int(default.rank.item()), int(exact.rank.item())


def test_a_start_orthogonal_to_one_of_the_two_is_still_ranked_right() -> None:
    n, rcond = 20, 1e-6
    ramp = _starts(n)[:, :1]
    a = _near_deficient(n, rcond, ramp)
    assert _ranks(a, rcond) == (n - 1, n - 1)
    alternating = _starts(n)[:, 1:]
    a = _near_deficient(n, rcond, alternating)
    assert _ranks(a, rcond) == (n - 1, n - 1)


@pytest.mark.xfail(
    strict=True,
    reason="LCD-945: a smallest singular vector orthogonal to both estimate starts is missed",
)
def test_a_matrix_orthogonal_to_both_starts_is_ranked_like_gelsd() -> None:
    n, rcond = 20, 1e-6
    a = _near_deficient(n, rcond, _starts(n))
    default, exact = _ranks(a, rcond)
    assert default == exact == n - 1
