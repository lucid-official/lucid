"""The triangular, LU and LDL solves refuse the operands the reference refuses (LCD-1083).

``solve``, ``solve_ex`` and ``lstsq`` read a 1-D right-hand side as a vector.
``solve_triangular``, ``lu_solve`` and ``ldl_solve`` do not: the reference takes a
matrix ``B`` of two or more dimensions there, and Lucid answered a vector anyway
(``solve_triangular(A, ones(3), upper=True)`` returned a ``(3,)`` solution where the
reference raises).  Each refusal here is held to the reference's exception class and
message, on every device and dtype the op holds; the spellings that stay legal are held
to the reference's result shape, so a guard that over-refuses fails as loudly as one that
under-refuses.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
from lucid.test._fixtures.devices import device_dtype_params

_REAL = (lucid.float32, lucid.float64)
_COMPLEX = (lucid.complex64, lucid.complex128)
_REAL_PLACEMENTS = device_dtype_params(_REAL)
_PLACEMENTS = device_dtype_params(_REAL + _COMPLEX)

N = 3


def _array(shape: tuple[int, ...], dtype_name: str) -> np.ndarray:
    """Operands whose values rarely matter: a refusal is raised before they are read.

    The accepted spellings do read them, so a matrix is ``3 I`` (its leading block, when
    not square) — full rank, and triangular whichever triangle is read.
    """
    np_dtype = np.complex128 if dtype_name.startswith("complex") else np.float64
    if len(shape) < 2:
        return np.ones(shape, dtype=np_dtype)
    return np.broadcast_to(3 * np.eye(shape[-2], shape[-1], dtype=np_dtype), shape).copy()


def _tensor(lib: ModuleType, shape: tuple[int, ...], dtype_name: str, device: str) -> Any:
    arr = _array(shape, dtype_name)
    if lib is lucid:
        return lucid.tensor(arr, dtype=getattr(lucid, dtype_name), device=device)
    return lib.tensor(arr, dtype=getattr(lib, dtype_name))


def _pivots(lib: ModuleType, batch: tuple[int, ...], device: str) -> Any:
    """The identity permutation, in LAPACK's 1-based ``int32``."""
    arr = np.broadcast_to(np.arange(1, N + 1, dtype=np.int32), (*batch, N)).copy()
    if lib is lucid:
        return lucid.tensor(arr, dtype=lucid.int32, device=device)
    return lib.tensor(arr, dtype=lib.int32)


@dataclass(frozen=True)
class _Case:
    """One call: the operands' shapes, and the options the op takes."""

    name: str
    a: tuple[int, ...]
    b: tuple[int, ...]
    options: dict[str, Any] = field(default_factory=dict)


def _solve_triangular(lib: ModuleType, case: _Case, name: str, device: str) -> Any:
    a, b = _tensor(lib, case.a, name, device), _tensor(lib, case.b, name, device)
    return lib.linalg.solve_triangular(a, b, upper=True, **case.options)


def _lu_solve(lib: ModuleType, case: _Case, name: str, device: str) -> Any:
    lu, b = _tensor(lib, case.a, name, device), _tensor(lib, case.b, name, device)
    return lib.linalg.lu_solve(lu, _pivots(lib, case.a[:-2], device), b, **case.options)


def _ldl_solve(lib: ModuleType, case: _Case, name: str, device: str) -> Any:
    ld, b = _tensor(lib, case.a, name, device), _tensor(lib, case.b, name, device)
    return lib.linalg.ldl_solve(ld, _pivots(lib, case.a[:-2], device), b, **case.options)


_Call = Callable[[ModuleType, _Case, str, str], Any]

#: Operand shapes the reference refuses for ``solve_triangular``, in its order of checks:
#: the matrix's rank, that it is square, the right-hand side's rank, its extent, the batch.
_TRIANGULAR_REFUSALS = [
    _Case("B 1-D", (N, N), (N,)),
    _Case("B 0-d", (N, N), ()),
    _Case("B 1-D, left=False", (N, N), (N,), {"left": False}),
    _Case("B 1-D against a batch of A", (2, N, N), (N,)),
    _Case("A 1-D", (N,), (N, 1)),
    _Case("A 0-d", (), (N, 1)),
    _Case("A 1-D, left=False", (N,), (N, 1), {"left": False}),
    _Case("A 1-D, B 1-D: A is read first", (N,), (N,)),
    _Case("A 0-d, B 0-d: A is read first", (), ()),
    _Case("A wide", (N, N + 1), (N, 2)),
    _Case("A tall", (N + 1, N), (N + 1, 2)),
    _Case("A wide, left=False", (N, N + 1), (N, 2), {"left": False}),
    _Case("A batched, wide", (2, N, N + 1), (2, N, 2)),
    _Case("A wide, B 1-D: squareness is read first", (N, N + 1), (N,)),
    _Case("B has too many rows", (N, N), (N + 1, 2)),
    _Case("B has too few rows", (N, N), (N - 1, 2)),
    _Case("B has too many columns, left=False", (N, N), (2, N + 1), {"left": False}),
    _Case("B holds one vector per matrix", (2, N, N), (2, N)),
    _Case("extent and batch both wrong: the extent is read first", (2, N, N), (5, N + 1, 2)),
    _Case("batch, B is longer", (2, N, N), (4, N, N, 2)),
    _Case("batch, A is longer", (2, N, N, N), (5, N, N, 2)),
    _Case("batch, last axis disagrees first", (4, 2, N, N), (5, N, N, 2)),
    _Case("batch, left=False", (N, N, N), (2, 2, N), {"left": False}),
]

_LU_REFUSALS = [
    _Case("B 1-D", (N, N), (N,)),
    _Case("B 0-d", (N, N), ()),
    _Case("B 1-D, left=False", (N, N), (N,), {"left": False}),
    _Case("B 1-D, adjoint", (N, N), (N,), {"adjoint": True}),
    _Case("B 1-D against a batch of LU", (2, N, N), (N,)),
]

_LDL_REFUSALS = [
    _Case("B 1-D", (N, N), (N,)),
    _Case("B 0-d", (N, N), ()),
    _Case("B 1-D against a batch of LD", (2, N, N), (N,)),
]

_REFUSALS: list[tuple[str, _Call, _Case]] = [
    *[("solve_triangular", _solve_triangular, c) for c in _TRIANGULAR_REFUSALS],
    *[("lu_solve", _lu_solve, c) for c in _LU_REFUSALS],
    *[("ldl_solve", _ldl_solve, c) for c in _LDL_REFUSALS],
]
_REFUSAL_PARAMS = [pytest.param(op, call, c, id=f"{op}-{c.name}") for op, call, c in _REFUSALS]


def _reference_refusal(call: _Call, case: _Case, name: str, ref: ModuleType) -> Exception:
    """What the reference raises for ``case`` — a failure of the test if it answers instead."""
    try:
        call(ref, case, name, "cpu")
    except (RuntimeError, IndexError) as err:
        return err
    raise AssertionError(f"the reference answered {case.name!r}; it is not a refusal")


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
@pytest.mark.parametrize(("op", "call", "case"), _REFUSAL_PARAMS)
def test_refusal_matches_the_reference(
    ref: ModuleType, op: str, call: _Call, case: _Case, device: str, dtype: lucid.dtype
) -> None:
    name = str(dtype).removeprefix("lucid.")
    want = _reference_refusal(call, case, name, ref)
    with pytest.raises(type(want)) as got:
        call(lucid, case, name, device)
    assert str(got.value) == str(want), op


# Spellings the reference accepts, and the shape it answers with: the guard must not
# turn them away, and the ops whose reference reads a vector must keep reading one.
_TRIANGULAR_ACCEPTED = [
    _Case("matrix", (N, N), (N, 2)),
    _Case("one column", (N, N), (N, 1)),
    _Case("no columns", (N, N), (N, 0)),
    _Case("left=False", (N, N), (2, N), {"left": False}),
    _Case("batched", (4, N, N), (4, N, 2)),
    _Case("single A against a batch of B", (N, N), (4, N, 2)),
    _Case("batch against a single B", (2, N, N), (N, 2)),
    _Case("broadcast batch", (2, 1, N, N), (3, N, 2)),
    _Case("B equal to A.shape[:-1] is a matrix", (N, N, N), (N, N)),
    _Case("empty system", (0, 0), (0, 2)),
    _Case("empty batch", (0, N, N), (N, 2)),
    _Case("left=False, batch", (2, N, N), (2, 5, N), {"left": False}),
]


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
@pytest.mark.parametrize("case", _TRIANGULAR_ACCEPTED, ids=lambda c: c.name)
def test_solve_triangular_still_answers_a_matrix_rhs(
    ref: ModuleType, case: _Case, device: str, dtype: lucid.dtype
) -> None:
    name = str(dtype).removeprefix("lucid.")
    want = _solve_triangular(ref, case, name, "cpu")
    got = _solve_triangular(lucid, case, name, device)
    assert got.shape == tuple(want.shape)
    assert got.dtype == dtype


def _solve(lib: ModuleType, case: _Case, name: str, device: str) -> Any:
    a, b = _tensor(lib, case.a, name, device), _tensor(lib, case.b, name, device)
    return lib.linalg.solve(a, b)


def _solve_ex(lib: ModuleType, case: _Case, name: str, device: str) -> Any:
    a, b = _tensor(lib, case.a, name, device), _tensor(lib, case.b, name, device)
    return lib.linalg.solve_ex(a, b).result


def _lstsq(lib: ModuleType, case: _Case, name: str, device: str) -> Any:
    a, b = _tensor(lib, case.a, name, device), _tensor(lib, case.b, name, device)
    return lib.linalg.lstsq(a, b).solution


_VECTOR_RHS = [
    _Case("B 1-D", (N, N), (N,)),
    _Case("B 1-D against a batch of A", (2, N, N), (N,)),
    _Case("one vector per matrix", (2, N, N), (2, N)),
    _Case("B a matrix", (N, N), (N, 2)),
]
_VECTOR_OPS: list[tuple[str, _Call, list[_Case]]] = [
    ("solve", _solve, _VECTOR_RHS),
    ("solve_ex", _solve_ex, _VECTOR_RHS),
    ("lstsq", _lstsq, [_Case("B 1-D", (4, N), (4,)), _Case("B a matrix", (4, N), (4, 2))]),
]
_VECTOR_PARAMS = [
    pytest.param(call, c, id=f"{op}-{c.name}") for op, call, cases in _VECTOR_OPS for c in cases
]


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _REAL_PLACEMENTS)
@pytest.mark.parametrize(("call", "case"), _VECTOR_PARAMS)
def test_solve_solve_ex_and_lstsq_still_read_a_vector_rhs(
    ref: ModuleType, call: _Call, case: _Case, device: str, dtype: lucid.dtype
) -> None:
    """The reference reads a 1-D ``B`` as a vector for these three, so the shape rule is theirs alone."""
    name = str(dtype).removeprefix("lucid.")
    want = call(ref, case, name, "cpu")
    got = call(lucid, case, name, device)
    assert got.shape == tuple(want.shape)


def test_a_vector_rhs_is_one_column_away_from_a_legal_call() -> None:
    """The refusal's remedy works: ``b.unsqueeze(-1)`` solves, and ``.squeeze(-1)`` undoes it."""
    a = lucid.triu(lucid.ones(N, N, dtype=lucid.float64)) + lucid.eye(N, dtype=lucid.float64)
    b = lucid.tensor([1.0, 2.0, 3.0], dtype=lucid.float64)
    x = LA.solve_triangular(a, b.unsqueeze(-1), upper=True).squeeze(-1)
    np.testing.assert_allclose(x.numpy(), np.linalg.solve(np.triu(a.numpy()), b.numpy()))
    with pytest.raises(RuntimeError, match="at least 2 dimensions"):
        LA.solve_triangular(a, b, upper=True)
