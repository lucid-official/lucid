"""A factorisation that fails raises ``lucid.linalg.LinAlgError`` on every
device and variant, and its ``*_ex`` twin reports the same ``info``.

The defect class (LCD-206, LCD-244): the factorisations did not own their
failure detection or its type.  Metal ``cholesky(upper=True)`` skipped the
check the lower path ran and returned ``[[1, 2], [0, 1]]`` for a matrix
that is not positive definite, so ``cholesky_ex`` reported ``info = 0``
for it.  Every failure that was caught came out as a bare ``LucidError``,
the type of the shape refusals too, so ``lucid.linalg`` told them apart
with ``type(err) is LucidError``.  ``lu_factor`` never raised at all.

Now the backend raises ``LinAlgError`` — a ``LucidError`` and so a
``RuntimeError``, as the reference raises it — from the one status reader
both streams share, and the Metal stream reads Cholesky's ``info`` off its
factor.  ``lu_factor`` and ``ldl_factor`` finish on a singular matrix, so
their kernels return LAPACK's own ``info`` (reading it back off the factor
was wrong for non-finite input).  A non-square matrix is a
``ShapeMismatch``, never a ``LinAlgError``.  The grid below is every
factorisation x {succeeds, fails} x device x {one matrix, a batch}, held
to the reference: the same inputs raise, the same ``info`` comes back, and
the answers agree where there is one.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
from lucid._C import engine as _C_engine

_SPD = np.array([[4.0, 2.0], [2.0, 3.0]])
_SPD2 = np.array([[5.0, 1.0], [1.0, 2.0]])
_NOT_PD = np.array([[1.0, 2.0], [2.0, 1.0]])  # potrf fails at order 2
_REGULAR = np.array([[2.0, 1.0], [4.0, 7.0]])
_REGULAR2 = np.array([[3.0, 1.0], [1.0, 2.0]])
_SINGULAR = np.array([[1.0, 2.0], [2.0, 4.0]])  # getrf: U[2, 2] == 0
_INDEFINITE = np.array([[0.0, 1.0], [1.0, 0.0]])  # LDL takes a 2 x 2 block
_D_SINGULAR = np.array([[1.0, 0.0], [0.0, 0.0]])  # sytrf: D[2, 2] == 0

#: Inputs per family: (succeeds, a second success, fails).
_INPUTS = {
    "cholesky": (_SPD, _SPD2, _NOT_PD),
    "lu": (_REGULAR, _REGULAR2, _SINGULAR),
    "ldl": (_SPD, _INDEFINITE, _D_SINGULAR),
}

#: A raising op: ``fn(linalg, A)`` with ``linalg`` Lucid's or the
#: reference's, and the family its inputs come from.
_RAISING: dict[str, tuple[Callable[[Any, Any], Any], str]] = {
    "cholesky": (lambda M, a: M.cholesky(a), "cholesky"),
    "cholesky_upper": (lambda M, a: M.cholesky(a, upper=True), "cholesky"),
    "inv": (lambda M, a: M.inv(a), "lu"),
    "solve": (lambda M, a: M.solve(a, _rhs(M, a)), "lu"),
    "matrix_power_minus_one": (lambda M, a: M.matrix_power(a, -1), "lu"),
    "lu_factor": (lambda M, a: M.lu_factor(a), "lu"),
    "ldl_factor": (lambda M, a: M.ldl_factor(a), "ldl"),
}

#: An ``_ex`` op: ``fn(linalg, A, check_errors)`` returning outputs whose
#: last is ``info``; and whether a failed matrix's outputs are the
#: reference's (a completed factorisation) or left undefined (Lucid
#: zero-fills them, the reference leaves the partial factor).
_EX: dict[str, tuple[Callable[[Any, Any, bool], Any], str, bool]] = {
    "cholesky_ex": (lambda M, a, c: M.cholesky_ex(a, check_errors=c), "cholesky", False),
    "cholesky_ex_upper": (
        lambda M, a, c: M.cholesky_ex(a, upper=True, check_errors=c),
        "cholesky",
        False,
    ),
    "inv_ex": (lambda M, a, c: M.inv_ex(a, check_errors=c), "lu", False),
    "solve_ex": (lambda M, a, c: M.solve_ex(a, _rhs(M, a), check_errors=c), "lu", False),
    "lu_factor_ex": (lambda M, a, c: M.lu_factor_ex(a, check_errors=c), "lu", True),
    "ldl_factor_ex": (lambda M, a, c: M.ldl_factor_ex(a, check_errors=c), "ldl", True),
}


def _rhs(M: Any, a: Any) -> Any:
    """A right-hand side of ones for ``a``, in ``a``'s framework."""
    shape = tuple(a.shape[:-1]) + (1,)
    if M is LA:
        return lucid.ones(*shape, dtype=a.dtype, device=a.device)
    return a.new_ones(shape)


def _operand(family: str, fails: bool, batched: bool) -> np.ndarray:
    good, good2, bad = _INPUTS[family]
    if not batched:
        return bad if fails else good
    # The failure is the second matrix, so a check that read only the
    # first — or only the last — would miss one of the two batch tests.
    return np.stack([good, bad, good2]) if fails else np.stack([good, good2])


def _both(a: np.ndarray, device: str, ref: Any) -> tuple[Any, Any]:
    return (lucid.tensor(a, dtype=lucid.float32, device=device), ref.tensor(a, dtype=ref.float32))


def _np(x: Any) -> np.ndarray:
    if isinstance(x, lucid.Tensor):
        return np.asarray(x.to("cpu").numpy(), dtype=np.float64)
    return np.asarray(x.detach().numpy(), dtype=np.float64)


_CELLS = [
    pytest.param(
        fails, batched, id=f"{'fails' if fails else 'succeeds'}-{'batch' if batched else 'one'}"
    )
    for fails in (False, True)
    for batched in (False, True)
]


# ── the class ─────────────────────────────────────────────────────────────────


def test_linalg_error_is_the_engine_class_and_a_runtime_error() -> None:
    assert LA.LinAlgError is _C_engine.LinAlgError
    assert issubclass(LA.LinAlgError, _C_engine.LucidError)
    assert issubclass(LA.LinAlgError, RuntimeError)
    # Not an argument refusal: ``except ValueError`` must not swallow it.
    assert not issubclass(LA.LinAlgError, ValueError)
    assert LA.LinAlgError.__module__ == _C_engine.__name__


def test_the_reference_class_is_a_runtime_error_too(ref: Any) -> None:
    assert issubclass(ref.linalg.LinAlgError, RuntimeError)


def test_a_shape_refusal_is_not_a_linalg_error(device: str) -> None:
    wide = lucid.ones(2, 3, device=device)
    with pytest.raises(_C_engine.ShapeMismatch) as info:
        LA.cholesky(wide)
    assert not isinstance(info.value, LA.LinAlgError)
    # The ``_ex`` variants let it through rather than report ``info``, all
    # with the one refusal the kernels raise.
    for ex in (LA.cholesky_ex, LA.inv_ex, LA.ldl_factor_ex):
        with pytest.raises(_C_engine.ShapeMismatch):
            ex(wide)
    with pytest.raises(_C_engine.ShapeMismatch):
        LA.solve_ex(wide, lucid.ones(2, 1, device=device))


# ── the raising variants ──────────────────────────────────────────────────────


@pytest.mark.parametrize(("fails", "batched"), _CELLS)
@pytest.mark.parametrize("op", list(_RAISING))
def test_raising_ops_fail_where_the_reference_does(
    op: str, fails: bool, batched: bool, device: str, ref: Any
) -> None:
    fn, family = _RAISING[op]
    a = _operand(family, fails, batched)
    mine, theirs = _both(a, device, ref)
    if fails:
        with pytest.raises(RuntimeError):
            fn(ref.linalg, theirs)
        with pytest.raises(LA.LinAlgError) as info:
            fn(LA, mine)
        if batched:
            # The failed matrix is named — the second of three.
            assert "(Batch element 1): " in str(info.value)
        return
    got, want = fn(LA, mine), fn(ref.linalg, theirs)
    got = got if isinstance(got, tuple) else (got,)
    want = want if isinstance(want, tuple) else (want,)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(_np(g), _np(w), rtol=1e-5, atol=1e-5)


def test_both_triangles_refuse_a_matrix_that_is_not_pd(device: str) -> None:
    """The LCD-206 repro, on each device: both triangles refuse it."""
    a = lucid.tensor(_NOT_PD.tolist(), device=device)
    for upper in (False, True):
        with pytest.raises(LA.LinAlgError, match=r"leading minor of order 2"):
            LA.cholesky(a, upper=upper)


# ── the _ex variants ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(("fails", "batched"), _CELLS)
@pytest.mark.parametrize("op", list(_EX))
def test_ex_ops_report_the_reference_info(
    op: str, fails: bool, batched: bool, device: str, ref: Any
) -> None:
    fn, family, failed_outputs_defined = _EX[op]
    a = _operand(family, fails, batched)
    mine, theirs = _both(a, device, ref)
    got, want = fn(LA, mine, False), fn(ref.linalg, theirs, False)
    got_info, want_info = got[-1], want[-1]
    assert got_info.dtype is lucid.int32
    np.testing.assert_array_equal(_np(got_info), _np(want_info))
    ok = _np(want_info) == 0
    for g, w in zip(got[:-1], want[:-1], strict=True):
        g, w = _np(g), _np(w)
        if not failed_outputs_defined and not ok.all():
            g, w = g[ok], w[ok]  # only the matrices that succeeded
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)
    if fails:
        with pytest.raises(LA.LinAlgError):
            fn(LA, mine, True)
    else:
        fn(LA, mine, True)


@pytest.mark.parametrize("n", [3, 8, 70])
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_info_matches_the_reference_past_the_blocked_size(
    n: int, upper: bool, device: str, ref: Any
) -> None:
    """``info`` is the order of the first failing leading minor, on both
    triangles, including past LAPACK's block size."""
    rng = np.random.default_rng(n)
    q = rng.standard_normal((4, n, n))
    a = q @ q.transpose(0, 2, 1) + n * np.eye(n)
    for b, k in enumerate((0, n // 2, n - 1)):
        a[b + 1, k, k] = -abs(a[b + 1, k, k])  # fails exactly at order k + 1
    mine, theirs = _both(a, device, ref)
    _, got = LA.cholesky_ex(mine, upper=upper)
    _, want = ref.linalg.cholesky_ex(theirs, upper=upper)
    np.testing.assert_array_equal(_np(got), _np(want))
    with pytest.raises(LA.LinAlgError, match=r"\(Batch element 1\): .*leading minor of order 1 "):
        LA.cholesky(mine, upper=upper)


# ── lu_solve on the factors of a singular matrix ──────────────────────────────


def test_lu_solve_through_lu_factor_ex_matches_the_reference(device: str, ref: Any) -> None:
    """``lu_factor`` refuses a singular matrix; ``lu_factor_ex`` hands its
    factors on, and back substitution through the zero pivot gives the
    IEEE result the reference gives."""
    mine, theirs = _both(_SINGULAR, device, ref)
    lu, piv, info = LA.lu_factor_ex(mine)
    assert int(info) == 2
    x = _np(LA.lu_solve(lu, piv, lucid.ones(2, 1, device=device)))
    rlu, rpiv, _ = ref.linalg.lu_factor_ex(theirs)
    want = _np(ref.linalg.lu_solve(rlu, rpiv, ref.ones(2, 1)))
    for where in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(where(x), where(want))


def test_lu_still_factors_a_singular_matrix(device: str) -> None:
    """``lu`` (P, L, U) is defined for every matrix, as the reference's is."""
    P, L, U = LA.lu(lucid.tensor(_SINGULAR.tolist(), device=device))
    np.testing.assert_allclose(_np(P @ L @ U), _SINGULAR, atol=1e-6)


# ── non-finite input: the info is LAPACK's own ────────────────────────────────


def _symmetric_draws(seed: int, count: int) -> list[np.ndarray]:
    """Small symmetric matrices with zeros, NaN and infinity in them."""
    rng = np.random.default_rng(seed)
    values = np.array([0.0, 1.0, -1.0, 2.0, 0.5, -3.0, np.nan, np.inf])
    p = [0.35, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.07]
    out = []
    while len(out) < count:
        n = int(rng.integers(1, 6))
        m = rng.choice(values, size=(n, n), p=p)
        s = np.tril(m) + np.tril(m, -1).T
        if not np.isfinite(s).all():
            out.append(s)
    return out


def test_ldl_info_is_sytrf_s_on_non_finite_input(device: str, ref: Any) -> None:
    """The LCD-244 review repro: D's diagonal says nothing about ``info``
    here (``sytrf`` gave up on the infinity at step 2, not at the zero of
    step 1), so ``info`` must come from LAPACK itself."""
    a = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, np.inf], [1.0, np.inf, 0.0]])
    mine, theirs = _both(a, device, ref)
    _, _, info = LA.ldl_factor_ex(mine)
    _, _, want = ref.linalg.ldl_factor_ex(theirs)
    assert int(info) == int(want) == 2
    with pytest.raises(LA.LinAlgError, match=r"D\[2,2\]"):
        LA.ldl_factor(mine)


def test_ldl_info_agrees_wherever_the_factorisations_do(device: str, ref: Any) -> None:
    """On non-finite input two LAPACKs may pivot differently — then the
    factors differ and so may ``info``.  Wherever the pivots agree, the
    ``info`` must too."""
    compared = 0
    for a in _symmetric_draws(7, 300):
        mine, theirs = _both(a, device, ref)
        _, piv, info = LA.ldl_factor_ex(mine)
        _, rpiv, rinfo = ref.linalg.ldl_factor_ex(theirs)
        if _np(piv).tolist() != _np(rpiv).tolist():
            continue
        compared += 1
        assert int(info) == int(rinfo), a.tolist()
    assert compared > 250


def test_lu_info_matches_the_reference_on_non_finite_input(device: str, ref: Any) -> None:
    rng = np.random.default_rng(11)
    values = np.array([0.0, 0.0, 1.0, -1.0, 2.0, np.nan, np.inf, 0.5, -3.0])
    p = [0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.03, 0.06, 0.06]
    for _ in range(300):
        shape = tuple(int(d) for d in rng.integers(1, 6, size=2))
        a = rng.choice(values, size=shape, p=p)
        mine, theirs = _both(a, device, ref)
        _, _, info = LA.lu_factor_ex(mine)
        _, _, want = ref.linalg.lu_factor_ex(theirs)
        assert int(info) == int(want), a.tolist()
