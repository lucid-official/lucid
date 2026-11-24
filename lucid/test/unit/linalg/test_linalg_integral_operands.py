"""Every route of ``lucid.linalg`` turns a bool or integer operand away where the reference does (LCD-1105).

An op can have more than one road to its answer: ``householder_product`` hands one matrix to the
engine's ``orgqr`` and composes a batch, or fewer reflectors, from other ops; ``inv`` returns an
empty matrix without calling the engine; ``solve`` reads a shape before the engine reads a dtype.
A refusal that sits on one road only is answered on the others, and a test that calls the op once
-- the table of ``test_integer_refusal_table`` does, per op -- cannot tell.  This file calls each
road (2-D, batched, empty, a mode or a driver that routes elsewhere) once per dtype:

* ``answers`` is the dtypes the reference answers, and every other cell is a refusal.  A cell where
  Lucid answers what the reference refuses, or refuses what it answers, fails.
* ``kind`` is how the reference turns the refused cells away: a ``"guard"`` (a ``RuntimeError``
  that is not a ``NotImplementedError``) or a ``"kernel"`` (a ``NotImplementedError``, a kernel
  with no instantiation for the dtype).  Every refused cell is held to it exactly, with one
  exception: the engine's factorisations raise ``NotImplementedError('only F32/F64')`` for a bool
  where the reference's guard is a plain ``RuntimeError``.  That class gap is recorded per op in
  ``lucid/test/audit/bool_refusal_baseline.json``, and a bool cell of an op the baseline lists is
  held to ``RuntimeError`` only -- so the day the baseline loses the op, its cells become exact.
* ``-m parity`` replays every route on the reference, so the table cannot drift from it, and checks
  that an answered cell has the reference's result dtype.

Every operand is a matrix of ones: a refusal is raised before the values are read, and the answered
roads (``matrix_power``, ``multi_dot``, ``cross``, ``vander``, ``diagonal``) do not need them.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import pytest

import lucid
from lucid.test.unit.ops._dtype_sweep import AUDIT_DIR

pytestmark = pytest.mark.filterwarnings("ignore")

Kind = Literal["guard", "kernel"]
#: ``M(*shape)`` builds an operand of the dtype under test; ``la`` is the framework's ``linalg``.
Operand = Callable[..., object]
Call = Callable[[ModuleType, Operand], object]

_DTYPES = ("bool", "int8", "int16", "int32", "int64")
_ALL = frozenset(_DTYPES)
_INTEGERS = frozenset(_DTYPES[1:])
_NONE: frozenset[str] = frozenset()
_INF = float("inf")


@dataclass(frozen=True)
class Route:
    """One call of a ``linalg`` op, and how the reference answers it for each dtype."""

    name: str
    call: Call
    #: The dtypes the reference answers.
    answers: frozenset[str] = _NONE
    #: How it refuses the rest.
    kind: Kind = "guard"


def _guard(name: str, call: Call) -> Route:
    return Route(name, call)


def _kernel(name: str, call: Call) -> Route:
    return Route(name, call, kind="kernel")


ROUTES: tuple[Route, ...] = (
    # ── factorisations the reference guards ────────────────────────────────────────────────────
    _guard("cholesky", lambda la, M: la.cholesky(M(3, 3))),
    _guard("cholesky-upper", lambda la, M: la.cholesky(M(3, 3), upper=True)),
    _guard("cholesky-batched", lambda la, M: la.cholesky(M(2, 3, 3))),
    _guard("cholesky_ex", lambda la, M: la.cholesky_ex(M(2, 3, 3))),
    _guard("inv", lambda la, M: la.inv(M(3, 3))),
    _guard("inv-batched", lambda la, M: la.inv(M(2, 3, 3))),
    # An empty matrix is its own inverse and never reaches the engine.
    _guard("inv-empty", lambda la, M: la.inv(M(0, 0))),
    _guard("inv-empty-batch", lambda la, M: la.inv(M(2, 0, 0))),
    _guard("inv_ex", lambda la, M: la.inv_ex(M(3, 3))),
    _guard("inv_ex-batched", lambda la, M: la.inv_ex(M(2, 3, 3))),
    _guard("det", lambda la, M: la.det(M(3, 3))),
    _guard("det-batched", lambda la, M: la.det(M(2, 3, 3))),
    _guard("det-empty", lambda la, M: la.det(M(0, 0))),
    _guard("slogdet", lambda la, M: la.slogdet(M(3, 3))),
    _guard("slogdet-batched", lambda la, M: la.slogdet(M(2, 3, 3))),
    _guard("eig", lambda la, M: la.eig(M(3, 3))),
    _guard("eig-batched", lambda la, M: la.eig(M(2, 3, 3))),
    _guard("eigvals", lambda la, M: la.eigvals(M(3, 3))),
    _guard("eigvals-batched", lambda la, M: la.eigvals(M(2, 3, 3))),
    _guard("svd", lambda la, M: la.svd(M(4, 3))),
    _guard("svd-full", lambda la, M: la.svd(M(4, 3), full_matrices=True)),
    _guard("svd-batched", lambda la, M: la.svd(M(2, 4, 3))),
    _guard("svd-empty", lambda la, M: la.svd(M(0, 3))),
    _guard("svdvals", lambda la, M: la.svdvals(M(4, 3))),
    _guard("svdvals-batched", lambda la, M: la.svdvals(M(2, 4, 3))),
    _guard("qr", lambda la, M: la.qr(M(4, 3))),
    _guard("qr-complete", lambda la, M: la.qr(M(4, 3), mode="complete")),
    _guard("qr-r", lambda la, M: la.qr(M(4, 3), mode="r")),
    _guard("qr-wide", lambda la, M: la.qr(M(3, 4))),
    _guard("qr-batched", lambda la, M: la.qr(M(2, 4, 3))),
    _guard("qr-empty", lambda la, M: la.qr(M(0, 0))),
    _guard("pinv", lambda la, M: la.pinv(M(4, 3))),
    _guard("pinv-batched", lambda la, M: la.pinv(M(2, 4, 3))),
    _guard("pinv-hermitian", lambda la, M: la.pinv(M(3, 3), hermitian=True)),
    _guard("pinv-rtol", lambda la, M: la.pinv(M(4, 3), rtol=0.1)),
    _guard("matrix_rank", lambda la, M: la.matrix_rank(M(3, 3))),
    _guard("matrix_rank-batched", lambda la, M: la.matrix_rank(M(2, 3, 3))),
    _guard("matrix_rank-hermitian", lambda la, M: la.matrix_rank(M(3, 3), hermitian=True)),
    _guard("matrix_rank-tol", lambda la, M: la.matrix_rank(M(3, 3), atol=0.1, rtol=0.0)),
    _guard("cond", lambda la, M: la.cond(M(3, 3))),
    _guard("cond-fro", lambda la, M: la.cond(M(3, 3), "fro")),
    _guard("cond-nuc", lambda la, M: la.cond(M(3, 3), "nuc")),
    _guard("cond-1", lambda la, M: la.cond(M(3, 3), 1)),
    _guard("cond-inf", lambda la, M: la.cond(M(3, 3), _INF)),
    _guard("cond-batched", lambda la, M: la.cond(M(2, 3, 3))),
    _guard("matrix_exp", lambda la, M: la.matrix_exp(M(3, 3))),
    _guard("matrix_exp-batched", lambda la, M: la.matrix_exp(M(2, 3, 3))),
    _guard("matrix_exp-empty", lambda la, M: la.matrix_exp(M(0, 0))),
    _guard("ldl_factor", lambda la, M: la.ldl_factor(M(3, 3))),
    _guard("ldl_factor-batched", lambda la, M: la.ldl_factor(M(2, 3, 3))),
    _guard("ldl_factor-hermitian", lambda la, M: la.ldl_factor(M(3, 3), hermitian=True)),
    _guard("ldl_factor_ex", lambda la, M: la.ldl_factor_ex(M(2, 3, 3))),
    # ── norms ──────────────────────────────────────────────────────────────────────────────────
    _guard("matrix_norm", lambda la, M: la.matrix_norm(M(3, 3))),
    _guard("matrix_norm-nuc", lambda la, M: la.matrix_norm(M(3, 3), "nuc")),
    _guard("matrix_norm-2", lambda la, M: la.matrix_norm(M(3, 3), 2)),
    _guard("matrix_norm-1", lambda la, M: la.matrix_norm(M(3, 3), 1)),
    _guard("matrix_norm-inf", lambda la, M: la.matrix_norm(M(3, 3), _INF)),
    _guard("matrix_norm-batched", lambda la, M: la.matrix_norm(M(2, 3, 3))),
    _guard("norm-vector", lambda la, M: la.norm(M(3))),
    _guard("norm-matrix", lambda la, M: la.norm(M(3, 3))),
    _guard("norm-ord-0", lambda la, M: la.norm(M(3), 0)),
    _guard("norm-ord-inf", lambda la, M: la.norm(M(3), _INF)),
    _guard("vector_norm", lambda la, M: la.vector_norm(M(3))),
    _guard("vector_norm-1", lambda la, M: la.vector_norm(M(3), 1)),
    _guard("vector_norm-3", lambda la, M: la.vector_norm(M(3), 3)),
    _guard("vector_norm-inf", lambda la, M: la.vector_norm(M(3), _INF)),
    _guard("vector_norm-dim", lambda la, M: la.vector_norm(M(3, 3), 2, dim=0)),
    _guard("vector_norm-empty", lambda la, M: la.vector_norm(M(0), 2)),
    _guard("vecdot", lambda la, M: la.vecdot(M(3), M(3))),
    _guard("vecdot-dim", lambda la, M: la.vecdot(M(3, 3), M(3, 3), dim=0)),
    # ── linear systems the reference guards ────────────────────────────────────────────────────
    # ``solve`` judges ``A``'s dtype first: the shape cells that follow are in
    # ``test_integer_refusal_table`` (``solve-not-square``, ``solve-left-false-vector``).
    _guard("solve", lambda la, M: la.solve(M(3, 3), M(3, 2))),
    _guard("solve-batched", lambda la, M: la.solve(M(2, 3, 3), M(2, 3, 2))),
    _guard("solve-vector", lambda la, M: la.solve(M(3, 3), M(3))),
    _guard("solve-left-false", lambda la, M: la.solve(M(3, 3), M(2, 3), left=False)),
    _guard("solve_ex", lambda la, M: la.solve_ex(M(3, 3), M(3, 2))),
    _guard("solve_ex-batched", lambda la, M: la.solve_ex(M(2, 3, 3), M(2, 3, 2))),
    _guard("solve_ex-left-false", lambda la, M: la.solve_ex(M(3, 3), M(2, 3), left=False)),
    _guard("tensorsolve", lambda la, M: la.tensorsolve(M(3, 3), M(3))),
    _guard("tensorsolve-nd", lambda la, M: la.tensorsolve(M(2, 3, 6), M(2, 3))),
    _guard("tensorinv", lambda la, M: la.tensorinv(M(4, 2, 2), ind=1)),
    _guard("lstsq", lambda la, M: la.lstsq(M(3, 3), M(3, 2))),
    _guard("lstsq-vector", lambda la, M: la.lstsq(M(3, 3), M(3))),
    _guard("lstsq-batched", lambda la, M: la.lstsq(M(2, 4, 3), M(2, 4, 2))),
    _guard("lstsq-wide", lambda la, M: la.lstsq(M(3, 4), M(3, 2))),
    _guard("lstsq-gels", lambda la, M: la.lstsq(M(4, 3), M(4, 2), driver="gels")),
    _guard("lstsq-gelsd", lambda la, M: la.lstsq(M(4, 3), M(4, 2), driver="gelsd")),
    # Given an ``rcond`` the reference's refusal is a kernel's: the class depends on the call.
    _kernel("lstsq-rcond", lambda la, M: la.lstsq(M(4, 3), M(4, 2), rcond=0.1)),
    _kernel("lstsq-rcond-vector", lambda la, M: la.lstsq(M(3, 3), M(3), rcond=0.1)),
    _kernel("lstsq-rcond-batched", lambda la, M: la.lstsq(M(2, 4, 3), M(2, 4, 2), rcond=0.1)),
    _kernel(
        "lstsq-rcond-gelss", lambda la, M: la.lstsq(M(4, 3), M(4, 2), rcond=0.1, driver="gelss")
    ),
    # ── what the reference has no kernel for ───────────────────────────────────────────────────
    _kernel("eigh", lambda la, M: la.eigh(M(3, 3))),
    _kernel("eigh-upper", lambda la, M: la.eigh(M(3, 3), UPLO="U")),
    _kernel("eigh-batched", lambda la, M: la.eigh(M(2, 3, 3))),
    _kernel("eigvalsh", lambda la, M: la.eigvalsh(M(3, 3))),
    _kernel("eigvalsh-batched", lambda la, M: la.eigvalsh(M(2, 3, 3))),
    _kernel("lu", lambda la, M: la.lu(M(3, 3))),
    _kernel("lu-no-pivot", lambda la, M: la.lu(M(3, 3), pivot=False)),
    _kernel("lu-batched", lambda la, M: la.lu(M(2, 3, 3))),
    _kernel("lu-rectangular", lambda la, M: la.lu(M(4, 3))),
    _kernel("lu_factor", lambda la, M: la.lu_factor(M(3, 3))),
    _kernel("lu_factor-batched", lambda la, M: la.lu_factor(M(2, 3, 3))),
    _kernel("lu_factor-rectangular", lambda la, M: la.lu_factor(M(4, 3))),
    _kernel("lu_factor_ex", lambda la, M: la.lu_factor_ex(M(2, 3, 3))),
    _kernel("solve_triangular", lambda la, M: la.solve_triangular(M(3, 3), M(3, 2), upper=True)),
    _kernel(
        "solve_triangular-lower", lambda la, M: la.solve_triangular(M(3, 3), M(3, 2), upper=False)
    ),
    _kernel(
        "solve_triangular-left-false",
        lambda la, M: la.solve_triangular(M(3, 3), M(2, 3), upper=True, left=False),
    ),
    _kernel(
        "solve_triangular-unit",
        lambda la, M: la.solve_triangular(M(3, 3), M(3, 2), upper=True, unitriangular=True),
    ),
    _kernel(
        "solve_triangular-batched",
        lambda la, M: la.solve_triangular(M(2, 3, 3), M(2, 3, 2), upper=True),
    ),
    # ``householder_product``: the engine's ``orgqr`` takes one matrix with a reflector per
    # column; a batch, fewer reflectors or none are composed in Python.  Every road refuses.
    _kernel("householder_product", lambda la, M: la.householder_product(M(3, 3), M(3))),
    _kernel("householder_product-tall", lambda la, M: la.householder_product(M(4, 3), M(3))),
    _kernel("householder_product-fewer", lambda la, M: la.householder_product(M(4, 3), M(2))),
    _kernel(
        "householder_product-square-fewer", lambda la, M: la.householder_product(M(3, 3), M(2))
    ),
    _kernel("householder_product-none", lambda la, M: la.householder_product(M(4, 3), M(0))),
    _kernel(
        "householder_product-batched", lambda la, M: la.householder_product(M(2, 4, 3), M(2, 3))
    ),
    _kernel(
        "householder_product-batched-fewer",
        lambda la, M: la.householder_product(M(2, 4, 3), M(2, 2)),
    ),
    _kernel("householder_product-empty", lambda la, M: la.householder_product(M(0, 0), M(0))),
    # ── what the reference answers ─────────────────────────────────────────────────────────────
    # An integer matrix power is integer arithmetic; the bool one has no matmul kernel, and a
    # negative power is an inverse.
    Route("matrix_power-0", lambda la, M: la.matrix_power(M(3, 3), 0), _ALL),
    Route("matrix_power-1", lambda la, M: la.matrix_power(M(3, 3), 1), _ALL),
    Route("matrix_power-2", lambda la, M: la.matrix_power(M(3, 3), 2), _INTEGERS, "kernel"),
    Route("matrix_power-3", lambda la, M: la.matrix_power(M(3, 3), 3), _INTEGERS, "kernel"),
    Route(
        "matrix_power-batched", lambda la, M: la.matrix_power(M(2, 3, 3), 2), _INTEGERS, "kernel"
    ),
    _guard("matrix_power-negative", lambda la, M: la.matrix_power(M(3, 3), -1)),
    Route(
        "multi_dot", lambda la, M: la.multi_dot([M(3, 3), M(3, 3), M(3, 3)]), _INTEGERS, "kernel"
    ),
    Route(
        "multi_dot-vector", lambda la, M: la.multi_dot([M(3), M(3, 3), M(3)]), _INTEGERS, "kernel"
    ),
    Route("cross", lambda la, M: la.cross(M(3), M(3)), _INTEGERS, "kernel"),
    Route("cross-dim", lambda la, M: la.cross(M(3, 3), M(3, 3), dim=0), _INTEGERS, "kernel"),
    Route("vander", lambda la, M: la.vander(M(3)), _INTEGERS),
    Route("vander-n", lambda la, M: la.vander(M(3), N=4), _INTEGERS),
    Route("diagonal", lambda la, M: la.diagonal(M(3, 3)), _ALL),
    Route("diagonal-offset", lambda la, M: la.diagonal(M(3, 3), offset=1), _ALL),
)

_CASES = [(route, dtype) for route in ROUTES for dtype in _DTYPES]
_IDS = [f"{route.name}-{dtype}" for route, dtype in _CASES]

_LUCID_DTYPES = {
    "bool": lucid.bool_,
    "int8": lucid.int8,
    "int16": lucid.int16,
    "int32": lucid.int32,
    "int64": lucid.int64,
}


def _is_kernel(error: BaseException) -> bool:
    return isinstance(error, NotImplementedError)


_RECORDED: dict[str, dict[str, str | list[str]]] = json.loads(
    (AUDIT_DIR / "bool_refusal_baseline.json").read_text()
)["deviations"]
#: The ``linalg`` ops whose bool is refused as a kernel where the reference's is a guard.
_BOOL_CLASS_GAPS = frozenset(
    name.removeprefix("lucid.linalg.")
    for name, entry in _RECORDED.items()
    if name.startswith("lucid.linalg.") and entry["kind"] == "class"
)


def _holds_class(route: Route, dtype: str, error: BaseException, *, on_lucid: bool) -> bool:
    """Whether ``error`` is the class ``route`` turns ``dtype`` away with.

    Exactly, except Lucid's bool of an op the baseline records (``_BOOL_CLASS_GAPS``): a
    ``RuntimeError`` is enough there.  The reference is always held to the exact class.
    """
    if not isinstance(error, RuntimeError):
        return False
    if route.kind == "kernel":
        return _is_kernel(error)
    if on_lucid and dtype == "bool" and route.name.partition("-")[0] in _BOOL_CLASS_GAPS:
        return True
    return not _is_kernel(error)


def test_every_route_name_is_unique() -> None:
    assert len({route.name for route in ROUTES}) == len(ROUTES)


@pytest.mark.parametrize(("route", "dtype"), _CASES, ids=_IDS)
def test_lucid_answers_and_refuses_where_the_reference_does(
    device: str, route: Route, dtype: str
) -> None:
    def operand(*shape: int) -> object:
        return lucid.ones(*shape, dtype=_LUCID_DTYPES[dtype], device=device)

    if dtype in route.answers:
        assert route.call(lucid.linalg, operand) is not None
        return
    with pytest.raises(RuntimeError) as caught:
        route.call(lucid.linalg, operand)
    assert _holds_class(route, dtype, caught.value, on_lucid=True), (
        type(caught.value),
        str(caught.value),
    )


def _dtype_names(result: object) -> tuple[str, ...]:
    """The dtypes of a result -- a tensor, or a tuple of tensors -- spelled alike by both frameworks."""
    if isinstance(result, tuple):
        return tuple(name for part in result for name in _dtype_names(part))
    return (str(getattr(result, "dtype")).rsplit(".", 1)[-1],)


@pytest.mark.parity
@pytest.mark.parametrize(("route", "dtype"), _CASES, ids=_IDS)
def test_the_reference_does_what_the_table_says(ref: ModuleType, route: Route, dtype: str) -> None:
    def operand(*shape: int) -> object:
        return ref.ones(*shape, dtype=getattr(ref, dtype))

    if dtype in route.answers:
        assert route.call(ref.linalg, operand) is not None
        return
    with pytest.raises(RuntimeError) as caught:
        route.call(ref.linalg, operand)
    assert _holds_class(route, dtype, caught.value, on_lucid=False), (
        type(caught.value),
        str(caught.value),
    )


_ANSWERED = [(route, dtype) for route, dtype in _CASES if dtype in route.answers]


@pytest.mark.parity
@pytest.mark.parametrize(
    ("route", "dtype"), _ANSWERED, ids=[f"{route.name}-{dtype}" for route, dtype in _ANSWERED]
)
def test_an_answered_cell_has_the_reference_result_dtype(
    ref: ModuleType, device: str, route: Route, dtype: str
) -> None:
    want = route.call(ref.linalg, lambda *s: ref.ones(*s, dtype=getattr(ref, dtype)))
    got = route.call(
        lucid.linalg, lambda *s: lucid.ones(*s, dtype=_LUCID_DTYPES[dtype], device=device)
    )
    assert _dtype_names(got) == _dtype_names(want)
