"""The linear-solve family reads its right-hand side one way, and differentiates.

``solve``, ``lu_solve`` and ``solve_triangular`` share one contract, owned by
the engine (``solve_rhs_contract``): ``B`` is ``(*, n, k)``, or a 1-D ``(n,)``
vector, and the batch axes of ``A`` and ``B`` broadcast.  ``solve`` alone also
reads a ``B`` of exactly ``A.shape[:-1]`` as one vector per matrix, as the
reference does; for ``lu_solve`` and ``solve_triangular`` that ``B`` is a
matrix (the reference has no such reading for them).  Every other ``B`` is refused with
``ShapeMismatch`` before a backend sees it.  Before, each backend took the
batch count from ``A`` and the column count from ``B``'s last axis, so a
single ``A`` against a batch of ``B`` solved only the first batch, a ``B``
with the wrong row count was solved anyway, and a batch of vectors had its
batch axis read as the column count.

All three record a gradient in every tensor input (``lu_solve`` in the
factor and ``B``), and ``det`` differentiates at a singular matrix — its
gradient there is the adjugate's transpose, not an inverse failure.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

N = 3
_OPS = ("solve", "lu_solve", "solve_triangular")

# (A batch, B shape) — every legal way to spell the right-hand side.
_GOOD: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "vector": ((), (N,)),
    "matrix": ((), (N, 2)),
    "batched": ((4,), (4, N, 2)),
    "single A, batched B": ((), (4, N, 2)),
    "batched A, single B": ((2,), (N, 2)),
    "broadcast batch": ((2, 1), (3, N, 2)),
    "vector against a batch": ((2,), (N,)),
    # ``solve`` only: one vector per matrix.  The others refuse it (2 rows).
    "batched vector": ((2,), (2, N)),
    # Batch size equal to n, so B == A.shape[:-1] is also a valid matrix:
    # ``solve`` reads vectors, ``lu_solve`` / ``solve_triangular`` a matrix
    # broadcast over the batch.
    "ambiguous: A batch = n": ((N,), (N, N)),
}
_SOLVE_ONLY = {"batched vector"}

# (A batch, B shape) — shapes the reference refuses.
_BAD: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "0-d": ((), ()),
    "vector of the wrong length": ((), (N + 2,)),
    "wrong row count": ((), (N + 2, 2)),
    "batch that does not broadcast": ((2,), (3, N, 2)),
    "batched vector of another batch": ((2,), (4, N)),
}


def _devices() -> list[str]:
    return ["cpu", "metal"] if metal_available() else ["cpu"]


def _well_conditioned(batch: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((*batch, N, N)) + 3.0 * np.eye(N)


def _is_vector(op: str, a_batch: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return len(b) == 1 or (op == "solve" and b == (*a_batch, N))


_GOOD_PAIRS = [
    (op, case) for op in _OPS for case in _GOOD if op == "solve" or case not in _SOLVE_ONLY
]
_BAD_PAIRS = [(op, case) for op in _OPS for case in _BAD] + [
    (op, case) for op in _OPS if op != "solve" for case in _SOLVE_ONLY
]


def _shapes(case: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return _GOOD[case] if case in _GOOD else _BAD[case]


def _oracle(A: np.ndarray, B: np.ndarray, vector: bool) -> np.ndarray:
    """The reference reading of ``solve``, spelled out in numpy."""
    rhs = B[..., None] if vector else B
    batch = np.broadcast_shapes(A.shape[:-2], rhs.shape[:-2])
    X = np.linalg.solve(
        np.broadcast_to(A, (*batch, N, N)), np.broadcast_to(rhs, (*batch, N, rhs.shape[-1]))
    )
    return X[..., 0] if vector else X


def _call(op: str, A: lucid.Tensor, B: lucid.Tensor) -> lucid.Tensor:
    """The op's solution of ``A X = B`` — each op is handed what it reads."""
    if op == "solve":
        return LA.solve(A, B)
    if op == "solve_triangular":
        return LA.solve_triangular(A, B, upper=True)
    LU, piv = LA.lu_factor(A)
    return LA.lu_solve(LU, piv, B)


def _operand(op: str, A: np.ndarray) -> np.ndarray:
    return np.triu(A) if op == "solve_triangular" else A


def _tol(device: str) -> float:
    return 1e-10 if device == "cpu" else 1e-4


def _dtype(device: str) -> lucid.dtype:
    return lucid.float64 if device == "cpu" else lucid.float32


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(("op", "case"), _GOOD_PAIRS)
def test_forward_matches_the_reference_reading(op: str, case: str, device: str) -> None:
    a_batch, b_shape = _GOOD[case]
    A = _operand(op, _well_conditioned(a_batch, 0))
    B = np.random.default_rng(1).standard_normal(b_shape)
    dt = _dtype(device)
    X = _call(
        op, lucid.tensor(A, dtype=dt, device=device), lucid.tensor(B, dtype=dt, device=device)
    )
    want = _oracle(A, B, _is_vector(op, a_batch, b_shape))
    assert X.shape == want.shape
    np.testing.assert_allclose(X.numpy(), want, rtol=_tol(device), atol=_tol(device))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(("op", "case"), _BAD_PAIRS)
def test_a_malformed_rhs_is_refused(op: str, case: str, device: str) -> None:
    a_batch, b_shape = _shapes(case)
    dt = _dtype(device)
    A = lucid.tensor(_operand(op, _well_conditioned(a_batch, 0)), dtype=dt, device=device)
    B = lucid.zeros(*b_shape, dtype=dt, device=device)
    with pytest.raises(_C_engine.ShapeMismatch, match=op):
        _call(op, A, B)


def test_lu_solve_refuses_pivots_of_another_shape() -> None:
    LU, piv = LA.lu_factor(lucid.tensor(_well_conditioned((2,), 0)))
    with pytest.raises(_C_engine.ShapeMismatch, match="pivots"):
        _C_engine.linalg.lu_solve(
            LU._impl, piv[0]._impl, lucid.ones(2, N, 1, dtype=lucid.float64)._impl
        )


# ── gradients ───────────────────────────────────────────────────────────────


def _gradcheck_case(op: str, case: str) -> tuple[Callable[..., lucid.Tensor], list[lucid.Tensor]]:
    a_batch, b_shape = _GOOD[case]
    A = lucid.tensor(_operand(op, _well_conditioned(a_batch, 2)))
    B = lucid.tensor(np.random.default_rng(3).standard_normal(b_shape))
    out_shape = _oracle(A.numpy(), B.numpy(), _is_vector(op, a_batch, b_shape)).shape
    W = lucid.tensor(np.random.default_rng(4).standard_normal(out_shape))
    # gradcheck differentiates the inputs that require grad, and only those.
    B.requires_grad_()
    if op == "lu_solve":
        LU, piv = LA.lu_factor(A)
        LU = LU.detach().requires_grad_()
        return (lambda lu, b: (LA.lu_solve(lu, piv, b) * W).sum()), [LU, B]
    return (lambda a, b: (_call(op, a, b) * W).sum()), [A.requires_grad_(), B]


@pytest.mark.parametrize(("op", "case"), _GOOD_PAIRS)
def test_gradients_match_finite_differences(op: str, case: str) -> None:
    fn, inputs = _gradcheck_case(op, case)
    assert lucid.autograd.gradcheck(fn, inputs, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("op", _OPS)
def test_second_derivatives_match_finite_differences(op: str) -> None:
    fn, inputs = _gradcheck_case(op, "broadcast batch")
    assert lucid.autograd.gradgradcheck(fn, inputs, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize(("op", "case"), _GOOD_PAIRS)
def test_metal_gradients_agree_with_cpu(op: str, case: str) -> None:
    a_batch, b_shape = _GOOD[case]
    A_np = _operand(op, _well_conditioned(a_batch, 2))
    B_np = np.random.default_rng(3).standard_normal(b_shape)

    def grads(device: str) -> list[np.ndarray]:
        dt = _dtype(device)
        A = lucid.tensor(A_np, dtype=dt, device=device)
        B = lucid.tensor(B_np, dtype=dt, device=device).requires_grad_()
        if op == "lu_solve":
            LU, piv = LA.lu_factor(A)
            first = LU.detach().requires_grad_()
            X = LA.lu_solve(first, piv, B)
        else:
            first = A.requires_grad_()
            X = _call(op, first, B)
        X.sum().backward()
        return [t.grad.to("cpu").numpy().astype(np.float64) for t in (first, B)]

    for got, want in zip(grads("metal"), grads("cpu"), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", _devices())
def test_a_solve_through_a_transposed_view_reads_the_view(device: str) -> None:
    dt = _dtype(device)
    A = _well_conditioned((), 5)
    B = np.random.default_rng(6).standard_normal((N, 2))
    X = LA.solve(
        lucid.tensor(A, dtype=dt, device=device).mT, lucid.tensor(B, dtype=dt, device=device)
    )
    np.testing.assert_allclose(
        X.numpy(), np.linalg.solve(A.T, B), rtol=_tol(device), atol=_tol(device)
    )


@pytest.mark.parametrize("op", _OPS)
def test_an_empty_rhs_gives_an_empty_solution(op: str) -> None:
    A = lucid.tensor(_operand(op, _well_conditioned((), 0)))
    assert _call(op, A, lucid.zeros(N, 0, dtype=lucid.float64)).shape == (N, 0)
    assert _call(op, A, lucid.zeros(0, N, 2, dtype=lucid.float64)).shape == (0, N, 2)


def test_kl_between_an_unbatched_and_a_batched_mvn_matches_the_reference(ref: Any) -> None:
    """``kl_divergence`` solves ``Lq`` (batch (n,)) against ``Lp`` (n, n).

    ``Lp`` has the shape ``Lq.shape[:-1]``; read as a batch of vectors it
    gave a wrong, partly negative KL.  It is a matrix broadcast over q's
    batch, as the reference reads it.
    """
    from lucid.distributions import MultivariateNormal, kl_divergence

    Lp = np.linalg.cholesky(np.eye(N) * 2 + 0.1)
    Lq = np.linalg.cholesky(np.stack([np.eye(N) * (i + 1) + 0.1 for i in range(N)]))
    p = MultivariateNormal(lucid.zeros(N, dtype=lucid.float64), scale_tril=lucid.tensor(Lp))
    q = MultivariateNormal(lucid.zeros(N, N, dtype=lucid.float64), scale_tril=lucid.tensor(Lq))
    dist = ref.distributions
    want = dist.kl_divergence(
        dist.MultivariateNormal(ref.zeros(N, dtype=ref.float64), scale_tril=ref.tensor(Lp)),
        dist.MultivariateNormal(ref.zeros(N, N, dtype=ref.float64), scale_tril=ref.tensor(Lq)),
    )
    got = kl_divergence(p, q)
    assert got.shape == tuple(want.shape)
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-10, atol=1e-12)


# ── det / slogdet at a singular matrix ──────────────────────────────────────

# Rank n-1 (the gradient is the nonzero adjugate) and rank n-2 (it is zero).
_SINGULAR = {
    "rank 1 of 2": [[1.0, 2.0], [2.0, 4.0]],
    "rank 2 of 3": [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]],
    "rank 1 of 3": [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
    "zero": [[0.0] * 3] * 3,
}


def _cofactor(A: np.ndarray) -> np.ndarray:
    n = A.shape[-1]
    C = np.empty_like(A)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A, i, 0), j, 1)
            C[i, j] = (-1) ** (i + j) * (np.linalg.det(minor) if minor.size else 1.0)
    return C


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("case", list(_SINGULAR))
def test_det_gradient_at_a_singular_matrix_is_the_cofactor(case: str, device: str) -> None:
    M = np.array(_SINGULAR[case])
    A = lucid.tensor(M, dtype=_dtype(device), device=device).requires_grad_()
    LA.det(A).backward()
    np.testing.assert_allclose(A.grad.numpy(), _cofactor(M), atol=_tol(device) * 10)


def test_det_gradient_on_a_batch_with_one_singular_matrix() -> None:
    M = np.stack([np.array(_SINGULAR["rank 2 of 3"]), _well_conditioned((), 7)])
    assert lucid.autograd.gradcheck(
        lambda a: (LA.det(a) * lucid.tensor([1.0, -2.0], dtype=lucid.float64)).sum(),
        [lucid.tensor(M, requires_grad=True)],
    )


@pytest.mark.parametrize("case", ["rank 1 of 2", "rank 2 of 3", "zero"])
def test_det_differentiates_twice_at_a_singular_matrix(case: str) -> None:
    """The cofactor is a polynomial in A, so its own gradient exists there too."""
    assert lucid.autograd.gradgradcheck(
        LA.det, [lucid.tensor(np.array(_SINGULAR[case]), requires_grad=True)]
    )


def test_det_still_differentiates_twice_at_an_invertible_matrix() -> None:
    assert lucid.autograd.gradgradcheck(
        LA.det, [lucid.tensor(_well_conditioned((2,), 8), requires_grad=True)]
    )


def _singular_slogdet(device: str) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    A = lucid.tensor(_SINGULAR["rank 1 of 2"], dtype=_dtype(device), device=device)
    A.requires_grad_()
    sign, logabs = LA.slogdet(A)
    logabs.backward()
    return A, sign, logabs


@pytest.mark.parametrize("device", _devices())
def test_slogdet_gradient_at_a_singular_matrix_is_inf_along_the_adjugate(device: str) -> None:
    """``g · cof(A) / det(A)``: infinite wherever the cofactor is nonzero."""
    A, sign, logabs = _singular_slogdet(device)
    assert float(logabs.item()) == -np.inf
    assert float(sign.item()) == 0.0
    grad = A.grad.numpy()
    assert np.isinf(grad).all()
    pattern = np.sign(grad)
    assert (pattern == np.sign(_cofactor(np.array(_SINGULAR["rank 1 of 2"])))).all() or (
        pattern == -np.sign(_cofactor(np.array(_SINGULAR["rank 1 of 2"])))
    ).all()


@pytest.mark.parametrize("device", ["cpu", *(["metal"] if metal_available() else [])])
def test_slogdet_at_a_singular_matrix_keeps_the_signed_zero(device: str) -> None:
    """The reference: det = -0.0 (one row swap), so the grad is [-inf, inf, inf, -inf]."""
    A, _, _ = _singular_slogdet(device)
    np.testing.assert_array_equal(A.grad.numpy(), [[-np.inf, np.inf], [np.inf, -np.inf]])


def test_slogdet_gradient_at_an_invertible_matrix_is_the_inverse_transpose() -> None:
    M = _well_conditioned((2,), 9)
    A = lucid.tensor(M).requires_grad_()
    LA.slogdet(A)[1].sum().backward()
    np.testing.assert_allclose(A.grad.numpy(), np.linalg.inv(M).swapaxes(-1, -2), rtol=1e-10)
