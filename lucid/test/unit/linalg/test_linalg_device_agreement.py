"""Linalg ops answer degenerate input with one policy on every device.

The defect class (CHA-227): the GPU backend re-implemented the CPU
backend's LAPACK loops and dropped their ``info`` policy.  Metal
``solve_triangular`` returned ``b`` unchanged for a singular triangle,
the CPU raised "LAPACK numerical failure (info=2)", and the reference
returns ``[-inf, inf]``.  The CPU backend owns each policy now and the
Metal stream delegates to it, and a singular triangle gets the IEEE
result of substitution, as the reference gives it.

These tests check every engine-backed ``lucid.linalg`` op against that
on the inputs a policy has to decide on:

* a singular matrix, a singular triangle, a NaN, an infinity;
* a batch whose first matrix fails and whose second does not.  A
  value-returning op answers each matrix on its own, and a refusing op
  refuses the whole batch.  ``ldl_factor`` returned the failed matrix's
  output block unwritten, because the CPU checked ``info`` only once,
  after the loop;
* a batch of two matrices whose pivots differ (CHA-192: Metal
  ``ldl_factor`` uploaded the pivots without their batch dimensions);
* the empty extents: ``0 x 0``, an empty batch, and no right-hand sides.

``test_devices_agree`` checks that Metal gives the CPU's kind of answer:
both raise the same error class, or both return the same shapes, with
NaN and ±inf in the same places and the finite values close.
``test_matches_the_reference`` checks Lucid's kind against the reference
on each device.  The known gaps are strict xfails in ``_REFERENCE_GAPS``,
so a fix makes the test fail until its entry is removed.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
from lucid.test._fixtures.devices import device_dtype_params

_NAN, _INF = float("nan"), float("inf")

#: The degenerate inputs: ``A`` and the column count of its right-hand side.
_CASES: dict[str, tuple[np.ndarray, int]] = {
    "singular": (np.array([[1.0, 2.0], [2.0, 4.0]]), 1),
    "singular_triangle": (np.array([[1.0, 2.0], [0.0, 0.0]]), 1),
    "nan": (np.array([[_NAN, 2.0], [3.0, 4.0]]), 1),
    "inf": (np.array([[_INF, 2.0], [3.0, 4.0]]), 1),
    "mixed_batch": (
        np.array([[[1.0, 2.0], [0.0, 0.0]], [[2.0, 1.0], [0.0, 4.0]]]),
        1,
    ),
    # Symmetric, the second indefinite: its LDLᵀ takes a 2 x 2 pivot block,
    # so the two matrices' pivots differ (CHA-192: Metal returned the first
    # matrix's pivots, then whatever memory followed them).
    "indefinite_batch": (
        np.array([[[2.0, 1.0], [1.0, 4.0]], [[0.0, 1.0], [1.0, 0.0]]]),
        1,
    ),
    "empty": (np.zeros((0, 0)), 1),
    "empty_batch": (np.zeros((0, 2, 2)), 1),
    "no_rhs": (np.array([[2.0, 1.0], [0.0, 4.0]]), 0),
}

#: An op: ``fn(linalg, a, b)`` — ``linalg`` is ``lucid.linalg`` or the
#: reference's — and whether it takes a right-hand side.
_Op = tuple[Callable[[Any, Any, Any], Any], bool]


def _lu_solve(M: Any, a: Any, b: Any) -> Any:
    # The factors of a singular matrix come from ``lu_factor_ex``:
    # ``lu_factor`` refuses them.
    lu, piv, _ = M.lu_factor_ex(a)
    return M.lu_solve(lu, piv, b)


_OPS: dict[str, _Op] = {
    "solve_triangular": (lambda M, a, b: M.solve_triangular(a, b, upper=True), True),
    "solve_triangular_lower": (
        lambda M, a, b: M.solve_triangular(a.mT, b, upper=False),
        True,
    ),
    "solve_triangular_unit": (
        lambda M, a, b: M.solve_triangular(a, b, upper=True, unitriangular=True),
        True,
    ),
    "solve_triangular_right": (
        lambda M, a, b: M.solve_triangular(a, b.mT, upper=True, left=False),
        True,
    ),
    "lu_factor": (lambda M, a, b: M.lu_factor(a), False),
    "lu_factor_ex": (lambda M, a, b: M.lu_factor_ex(a), False),
    "lu_solve": (_lu_solve, True),
    "lu": (lambda M, a, b: M.lu(a), False),
    "lstsq": (lambda M, a, b: M.lstsq(a, b)[0], True),
    "ldl_factor": (lambda M, a, b: M.ldl_factor(a), False),
    "ldl_factor_ex": (lambda M, a, b: M.ldl_factor_ex(a), False),
    "inv": (lambda M, a, b: M.inv(a), False),
    "solve": (lambda M, a, b: M.solve(a, b), True),
    "det": (lambda M, a, b: M.det(a), False),
    "slogdet": (lambda M, a, b: M.slogdet(a), False),
    "cholesky": (lambda M, a, b: M.cholesky(a), False),
    "cholesky_upper": (lambda M, a, b: M.cholesky(a, upper=True), False),
    "qr": (lambda M, a, b: M.qr(a), False),
    "svdvals": (lambda M, a, b: M.svdvals(a), False),
    "eigvalsh": (lambda M, a, b: M.eigvalsh(a), False),
    "eigvals": (lambda M, a, b: M.eigvals(a), False),
    "pinv": (lambda M, a, b: M.pinv(a), False),
    "matrix_power_minus_one": (lambda M, a, b: M.matrix_power(a, -1), False),
}

#: Ops whose output order is not part of the contract (eigenvalues).
_UNORDERED = {"eigvals"}

#: (op, case) pairs where Lucid's kind differs from the reference.  Each one
#: is a known gap outside the solve_triangular / lu_factor policy that
#: CHA-227 settled.
_REFERENCE_GAPS: dict[tuple[str, str], str] = {
    ("lstsq", "inf"): "lstsq uses the SVD minimum-norm solution (gelss: zeros), "
    "the reference defaults to QR with pivoting (gelsy: [nan, 0])",
    ("eigvals", "nan"): "eigvals raises on a non-finite input, the reference " "returns NaN",
    ("eigvals", "inf"): "eigvals raises on a non-finite input, the reference " "returns NaN",
}

_Kind = tuple[str, Any]


def _params(with_gaps: bool) -> list[Any]:
    params = []
    for op, (_, takes_b) in _OPS.items():
        for case in _CASES:
            if case == "no_rhs" and not takes_b:
                continue  # without a right-hand side it is a regular matrix
            gap = _REFERENCE_GAPS.get((op, case)) if with_gaps else None
            marks = [pytest.mark.xfail(strict=True, reason=gap)] if gap else []
            params.append(pytest.param(op, case, marks=marks, id=f"{op}-{case}"))
    return params


def _operands(case: str) -> tuple[np.ndarray, np.ndarray]:
    a, cols = _CASES[case]
    return a, np.ones(a.shape[:-1] + (cols,))


def _leaves(out: Any) -> list[Any]:
    return list(out) if isinstance(out, (tuple, list)) else [out]


def _as_real(x: np.ndarray, unordered: bool) -> np.ndarray:
    """``x`` as float64, a complex value split into its two parts."""
    if np.iscomplexobj(x):
        if unordered:
            x = np.sort_complex(x)
        return np.stack([x.real, x.imag]).astype(np.float64)
    if unordered:
        x = np.sort(x, axis=-1)
    return np.asarray(x, dtype=np.float64)


def _kind(call: Callable[[], Any], to_numpy: Callable[[Any], Any], op: str) -> _Kind:
    """What the call answered: an error class, or its outputs as arrays.

    Only the errors a numerical policy raises are caught; anything else is a
    bug in the call itself and fails the test.
    """
    try:
        out = call()
    except (RuntimeError, ValueError) as err:
        return ("raise", type(err).__name__)
    unordered = op in _UNORDERED
    return ("value", [_as_real(to_numpy(x), unordered) for x in _leaves(out)])


def _lucid_kind(op: str, case: str, device: str, dtype: lucid.dtype) -> _Kind:
    a, b = _operands(case)
    fn = _OPS[op][0]
    return _kind(
        lambda: fn(
            LA,
            lucid.tensor(a, dtype=dtype, device=device),
            lucid.tensor(b, dtype=dtype, device=device),
        ),
        lambda t: t.to("cpu").numpy(),
        op,
    )


def _assert_same_kind(
    got: _Kind, want: _Kind, *, error_class: bool, rtol: float, atol: float
) -> None:
    assert got[0] == want[0], f"answered {got}, expected {want}"
    if got[0] == "raise":
        if error_class:
            assert got[1] == want[1]
        return
    assert len(got[1]) == len(want[1])
    for g, w in zip(got[1], want[1]):
        assert g.shape == w.shape
        for where in (np.isnan, np.isposinf, np.isneginf):
            np.testing.assert_array_equal(where(g), where(w), err_msg=where.__name__)
        finite = np.isfinite(g)
        np.testing.assert_allclose(g[finite], w[finite], rtol=rtol, atol=atol)


@pytest.mark.parametrize(("op", "case"), _params(with_gaps=False))
def test_devices_agree(op: str, case: str, cross_device_pair: tuple[str, str]) -> None:
    """Metal answers with the CPU backend's policy, not one of its own."""
    cpu, metal = cross_device_pair
    _assert_same_kind(
        _lucid_kind(op, case, metal, lucid.float32),
        _lucid_kind(op, case, cpu, lucid.float32),
        error_class=True,
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), device_dtype_params([lucid.float32, lucid.float64]))
@pytest.mark.parametrize(("op", "case"), _params(with_gaps=True))
def test_matches_the_reference(
    op: str, case: str, device: str, dtype: lucid.dtype, ref: Any
) -> None:
    """Lucid's kind of answer is the reference's, on every device."""
    a, b = _operands(case)
    ref_dtype = ref.float64 if dtype is lucid.float64 else ref.float32
    fn = _OPS[op][0]
    want = _kind(
        lambda: fn(
            ref.linalg,
            ref.tensor(a, dtype=ref_dtype),
            ref.tensor(b, dtype=ref_dtype),
        ),
        lambda t: t.numpy(),
        op,
    )
    tol = 1e-4 if dtype is lucid.float32 else 1e-10
    _assert_same_kind(
        _lucid_kind(op, case, device, dtype),
        want,
        error_class=False,  # the reference's error classes are its own
        rtol=tol,
        atol=tol,
    )


def test_singular_triangle_gives_the_ieee_result(device: str) -> None:
    """The CHA-227 repro: back substitution through a zero pivot."""
    a = lucid.tensor([[1.0, 2.0], [0.0, 0.0]], device=device)
    b = lucid.tensor([[1.0], [1.0]], device=device)
    x = LA.solve_triangular(a, b, upper=True).to("cpu").numpy()
    np.testing.assert_array_equal(x, np.array([[-_INF], [_INF]], np.float32))
    # 0 / 0 is NaN, and it propagates up the substitution.
    b0 = lucid.tensor([[1.0], [0.0]], device=device)
    assert np.isnan(LA.solve_triangular(a, b0, upper=True).to("cpu").numpy()).all()


@pytest.mark.parametrize("failing", [0, 1])
def test_a_batch_refusal_names_the_failing_matrix(failing: int, device: str) -> None:
    """``info`` is checked per matrix, so the refusal says which one failed —
    including a failure ahead of the last matrix, which used to go through."""
    batch = np.stack([np.array([[2.0, 1.0], [1.0, 4.0]])] * 3)
    batch[failing] = [[1.0, 0.0], [0.0, 0.0]]
    with pytest.raises(LA.LinAlgError, match=rf"\(Batch element {failing}\): D\[2,2\] is zero"):
        LA.ldl_factor(lucid.tensor(batch, dtype=lucid.float32, device=device))


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params([lucid.float32, lucid.float64]))
def test_pinv_all_inf_is_nan_while_single_inf_is_zero(device: str, dtype: lucid.dtype) -> None:
    """``pinv([[inf, inf], [inf, inf]])`` is all-NaN while a single leading
    ``inf`` gives zeros, matching LAPACK ``gesdd`` on both devices."""
    batch = lucid.tensor(
        [[[_INF, _INF], [_INF, _INF]], [[_INF, 2.0], [3.0, 4.0]]],
        dtype=dtype,
        device=device,
    )
    out = LA.pinv(batch).to("cpu").numpy()
    assert np.isnan(out[0]).all()
    np.testing.assert_array_equal(out[1], np.zeros((2, 2), dtype=out.dtype))
