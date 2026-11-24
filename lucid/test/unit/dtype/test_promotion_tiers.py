"""A 0-d tensor beside a dimensioned one: the dtype each op answers in (LCD-1060).

``a * b`` ranks its operands as they arrive, so a 0-d ``int16`` beside an
``int8`` vector computes in ``int8`` and wraps ``63 * 63``.  ``kron`` and the
stacking ops lift every operand to rank >= 1 *before* they combine them, so
the reference ranks the lifted tensors, none of them 0-d, and answers
``int16``.  ``block_diag`` is the other way round: it settles the dtype on the
operands as given.  ``kron`` and ``hstack`` / ``vstack`` once used the
``a * b`` rule and cast the 0-d tensor down.  The owner is
:class:`lucid._promotion.Tiers`; each op names the ranking it follows.

This pins the class, not one op: every op that meets a 0-d tensor with a
dimensioned one, over the 11 x 11 dtype lattice, on both devices, against the
reference.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import device_supports

_DTYPES = [
    lucid.bool_,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.complex64,
    lucid.complex128,
]
_INTEGERS = {lucid.int8, lucid.int16, lucid.int32, lucid.int64}
_COMPLEX = {lucid.complex64, lucid.complex128}

#: ``(lib, zero_dim, dimensioned) -> result`` — ``lib`` is Lucid or the
#: reference, so one lambda drives both.  ``dimensioned`` has shape ``(1,)``,
#: the one length every op here accepts beside a 0-d tensor.
_Call = Callable[[Any, Any, Any], Any]

#: Every op that takes a 0-d tensor beside a dimensioned one and promotes.
_SWEEP: dict[str, _Call] = {
    "kron(0d, nd)": lambda lib, z, d: lib.kron(z, d),
    "kron(nd, 0d)": lambda lib, z, d: lib.kron(d, z),
    "Tensor.kron": lambda lib, z, d: z.kron(d),
    "hstack(0d, nd)": lambda lib, z, d: lib.hstack([z, d]),
    "hstack(nd, 0d)": lambda lib, z, d: lib.hstack([d, z]),
    "vstack(0d, nd)": lambda lib, z, d: lib.vstack([z, d]),
    "vstack(nd, 0d)": lambda lib, z, d: lib.vstack([d, z]),
    "dstack(0d, nd)": lambda lib, z, d: lib.dstack([z, d]),
    "dstack(nd, 0d)": lambda lib, z, d: lib.dstack([d, z]),
    "cat": lambda lib, z, d: lib.cat([z.reshape(1), d]),
    "stack": lambda lib, z, d: lib.stack([z, d[0]]),
    "block_diag(0d, nd)": lambda lib, z, d: lib.block_diag(z, d),
    "block_diag(nd, 0d)": lambda lib, z, d: lib.block_diag(d, z),
    "where(c, 0d, nd)": lambda lib, z, d: lib.where(d == d, z, d),
    "where(c, nd, 0d)": lambda lib, z, d: lib.where(d == d, d, z),
    "mul": lambda lib, z, d: z * d,
}


def _value(dt: lucid.dtype) -> bool | int | float | complex:
    """A value whose square overflows int8 and is inexact in float16."""
    if dt is lucid.bool_:
        return True
    if dt in _INTEGERS:
        return 63
    return 63 + 0j if dt in _COMPLEX else 63.0


def _name(dt: Any) -> str:
    """A dtype of either framework, by its bare name."""
    return "bool" if dt is lucid.bool_ else str(dt).removeprefix("lucid.").rsplit(".", 1)[-1]


def _dtype_in(lib: ModuleType, dt: lucid.dtype) -> Any:
    """``dt`` as ``lib`` spells it — the reference names a dtype by ``_name``."""
    return dt if lib is lucid else getattr(lib, _name(dt))


def _operands(lib: ModuleType, zero: lucid.dtype, dim: lucid.dtype, device: str) -> tuple[Any, Any]:
    """A 0-d tensor of ``zero`` and a length-1 tensor of ``dim``, in ``lib``."""
    kwargs: dict[str, Any] = {"device": device} if lib is lucid else {}
    z = lib.tensor(_value(zero), dtype=_dtype_in(lib, zero), **kwargs)
    d = lib.tensor([_value(dim)], dtype=_dtype_in(lib, dim), **kwargs)
    return z, d


def test_dimensioned_ranking_ignores_the_zero_dim_tier() -> None:
    """``Tiers.DIMENSIONED`` is the plain lattice join, over every dtype pair."""
    for da in _DTYPES:
        for db in _DTYPES:
            zero = lucid.zeros((), dtype=da)._impl
            dim = lucid.zeros(1, dtype=db)._impl
            for operands in ((zero, dim), (dim, zero)):
                got = _promotion.result_type(*operands, tiers=_promotion.Tiers.DIMENSIONED)
                assert got == _promotion.promote_types(da._engine, db._engine), (da, db)


def test_by_rank_ranking_is_the_default_and_still_ranks_zero_dim_lower() -> None:
    zero = lucid.zeros((), dtype=lucid.int16)._impl
    dim = lucid.zeros(1, dtype=lucid.int8)._impl
    by_rank = _promotion.result_type(zero, dim)
    assert by_rank == lucid.int8._engine
    assert by_rank == _promotion.result_type(zero, dim, tiers=_promotion.Tiers.BY_RANK)
    assert _promotion.result_type(zero, dim, tiers=_promotion.Tiers.DIMENSIONED) == (
        lucid.int16._engine
    )


def test_dimensioned_ranking_keeps_python_numbers_in_their_own_tier() -> None:
    zero = lucid.zeros((), dtype=lucid.int16)._impl
    dim = lucid.zeros(1, dtype=lucid.int8)._impl
    tiers = _promotion.Tiers.DIMENSIONED
    assert _promotion.result_type(zero, dim, 7, tiers=tiers) == lucid.int16._engine
    lifted = _promotion.result_type(zero, dim, 2.5, tiers=tiers)
    assert lifted == lucid.get_default_dtype()._engine


@pytest.mark.parametrize("zero_first", [True, False])
def test_kron_of_a_zero_dim_int16_beside_an_int8_vector_does_not_wrap(zero_first: bool) -> None:
    a = lucid.tensor(63, dtype=lucid.int16)
    b = lucid.tensor([63], dtype=lucid.int8)
    for out in (
        lucid.kron(a, b) if zero_first else lucid.kron(b, a),
        a.kron(b) if zero_first else b.kron(a),
    ):
        assert out.dtype is lucid.int16
        assert out.tolist() == [3969]


@pytest.mark.parametrize("join", [lucid.hstack, lucid.vstack])
def test_stacks_lift_a_zero_dim_tensor_before_they_rank_it(join: Callable[..., Any]) -> None:
    a = lucid.tensor(63, dtype=lucid.int16)
    b = lucid.tensor([7], dtype=lucid.int8)
    for out in (join([a, b]), join([b, a])):
        assert out.dtype is lucid.int16


def test_block_diag_ranks_the_operands_as_they_are_given() -> None:
    """Unlike ``kron``, the reference settles ``block_diag`` before it reshapes."""
    a = lucid.tensor(63, dtype=lucid.int16)
    b = lucid.tensor([7], dtype=lucid.int8)
    for out in (lucid.block_diag(a, b), lucid.block_diag(b, a)):
        assert out.dtype is lucid.int8
    assert lucid.block_diag(a.reshape(1), b).dtype is lucid.int16


@pytest.mark.parametrize(
    ("da", "db"),
    [
        (lucid.float32, lucid.float16),
        (lucid.float32, lucid.bfloat16),
        (lucid.float64, lucid.float32),
        (lucid.int16, lucid.int8),
    ],
)
@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_linalg_cross_refuses_mixed_dtypes(
    da: lucid.dtype, db: lucid.dtype, shape: tuple[int, ...]
) -> None:
    """The engine reads a 0-d float32 beside half as a scalar and answered half.

    A single vector and a batch refuse alike, with the engine's ``DtypeMismatch``
    — the class the batch already raised.
    """
    a = lucid.ones(shape, dtype=da)
    b = lucid.ones(shape, dtype=db)
    for x, y in ((a, b), (b, a)):
        with pytest.raises(_C_engine.DtypeMismatch):
            lucid.linalg.cross(x, y)


@pytest.mark.parity
@pytest.mark.parametrize("op", list(_SWEEP))
def test_zero_dim_beside_dimensioned_matches_the_reference(
    op: str, device: str, ref: ModuleType
) -> None:
    """Where the reference answers, so does Lucid, at the same dtype and values.

    A cell Lucid cannot run for want of a kernel (``NotImplementedError``) is
    left out; the complex32 the reference answers for float16 beside complex is
    the documented deviation (Lucid has no complex32 and answers complex64).
    """
    call = _SWEEP[op]
    problems: list[str] = []
    compared = 0
    for zero in _DTYPES:
        for dim in _DTYPES:
            if not (device_supports(device, zero) and device_supports(device, dim)):
                continue
            label = f"{_name(zero)}[0d] , {_name(dim)}[nd]"
            try:
                want = call(ref, *_operands(ref, zero, dim, "cpu"))
            except (RuntimeError, TypeError, NotImplementedError):
                continue  # the reference refuses: nothing to match
            want_dtype = _name(want.dtype)
            if want_dtype == "complex32":
                continue
            try:
                got = call(lucid, *_operands(lucid, zero, dim, device))
            except NotImplementedError:
                continue
            compared += 1
            if _name(got.dtype) != want_dtype:
                problems.append(f"{label}: dtype {_name(got.dtype)}, reference {want_dtype}")
            elif got.tolist() != want.tolist():
                problems.append(f"{label}: {got.tolist()}, reference {want.tolist()}")
    assert compared > 0
    assert not problems, f"{len(problems)} cell(s):\n  " + "\n  ".join(problems[:20])


@pytest.mark.parity
def test_linalg_cross_refuses_wherever_the_reference_does(ref: ModuleType) -> None:
    """Every mixed-dtype pair: the reference refuses, and so does Lucid."""
    problems: list[str] = []
    for da in _DTYPES:
        for db in _DTYPES:
            if da is db:
                continue
            ref_args = [ref.tensor([_value(dt)] * 3, dtype=_dtype_in(ref, dt)) for dt in (da, db)]
            try:
                ref.linalg.cross(*ref_args)
            except RuntimeError:
                pass
            else:
                problems.append(f"{_name(da)} , {_name(db)}: the reference answers")
                continue
            args = [lucid.tensor([_value(dt)] * 3, dtype=dt) for dt in (da, db)]
            with pytest.raises(_C_engine.DtypeMismatch):
                lucid.linalg.cross(*args)
    assert not problems, "\n".join(problems)


def _inner(lib: Any, z: Any, d: Any) -> Any:
    return lib.linalg.inner(z, d) if lib is lucid else lib.inner(z, d)


def _lerp(lib: Any, z: Any, d: Any) -> Any:
    return lib.lerp(d, d, z)


def _column_stack(lib: Any, z: Any, d: Any) -> Any:
    return lib.column_stack([z, d])


@pytest.mark.parity
@pytest.mark.parametrize(
    ("call", "zero", "dim"),
    [
        pytest.param(
            _inner,
            lucid.int16,
            lucid.int8,
            marks=pytest.mark.xfail(
                strict=True,
                reason="LCD-1060 FOUND: inner with a 0-d operand is a product in the reference "
                "and ranks by tier; Lucid hands the engine's single-dtype kernel both and refuses",
            ),
            id="inner",
        ),
        pytest.param(
            _lerp,
            lucid.float32,
            lucid.float16,
            marks=pytest.mark.xfail(
                strict=True,
                reason="LCD-1060 FOUND: lerp's 0-d weight ranks below the half-precision "
                "operands in the reference; Lucid refuses the dtype pair",
            ),
            id="lerp",
        ),
        pytest.param(
            _column_stack,
            lucid.int16,
            lucid.int8,
            marks=pytest.mark.xfail(
                strict=True,
                reason="LCD-1060 FOUND: column_stack does not lift a 0-d input to a column",
            ),
            id="column_stack",
        ),
    ],
)
def test_zero_dim_gaps_outside_the_ranking_fix(
    call: _Call, zero: lucid.dtype, dim: lucid.dtype, ref: ModuleType
) -> None:
    """Deferred: the same 0-d-beside-dimensioned pairing, for ops the fix does not own."""
    want = call(ref, *_operands(ref, zero, dim, "cpu"))
    got = call(lucid, *_operands(lucid, zero, dim, "cpu"))
    assert _name(got.dtype) == _name(want.dtype)
    assert got.tolist() == want.tolist()
