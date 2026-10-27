"""Every spelling of a binary op takes its dtype from ``lucid._promotion``.

``a + b``, ``lucid.add(a, b)``, ``a.add(b)`` and ``a += b`` used to read
four hand-synced tables, so ``lucid.add(c128, c64)`` and ``c128 + c64``
disagreed, a 0-d operand widened a half tensor, and the bitwise dunders
skipped promotion altogether (LCD-237, LCD-164, LCD-290).  This sweeps the
whole 11 x 11 dtype lattice across the three operand tiers — dimensioned
tensors, 0-d tensors and Python numbers — for every binary operator and
checks that

* every spelling of an op agrees with every other, in dtype and value;
* the dtype is ``lucid.result_type`` of the operands (true division moves
  an integral result to the default float, a comparison answers bool);
* an op either answers in every spelling or refuses in every spelling;
* ``lucid.result_type`` is the reference's ``result_type`` cell for cell,
  but for complex32, which Lucid lacks and answers complex64 for;
* no other module under ``lucid/`` holds a dtype kind/width table.
"""

import ast
import operator
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import lucid

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
_SCALARS: list[bool | int | float | complex] = [True, 2, 2.5, 1 + 2j]

#: Not representable on the Metal stream.
_CPU_ONLY = {lucid.float64, lucid.complex128}

_INTEGRAL = {lucid.bool_, lucid.int8, lucid.int16, lucid.int32, lucid.int64}
_COMPLEX = {lucid.complex64, lucid.complex128}


def _make(dt: lucid.dtype, zero_dim: bool, device: str) -> lucid.Tensor:
    if dt is lucid.bool_:
        value: Any = True if zero_dim else [True, True]
        out = lucid.tensor(value, device=device)
    else:
        value = 2 if zero_dim else [3, 2]
        out = lucid.tensor(value, dtype=dt, device=device)
    assert out.ndim == (0 if zero_dim else 1)
    return out


Operand = lucid.Tensor | bool | int | float | complex


def _cells(device: str) -> Iterator[tuple[str, Operand, Operand]]:
    """Every (label, a, b) of the sweep on ``device``."""
    dtypes = [d for d in _DTYPES if device == "cpu" or d not in _CPU_ONLY]
    tiers = [("dim", False), ("0d", True)]
    for da in dtypes:
        for db in dtypes:
            for ta, za in tiers:
                for tb, zb in tiers:
                    yield (
                        f"{da}[{ta}] , {db}[{tb}]",
                        _make(da, za, device),
                        _make(db, zb, device),
                    )
        for ta, za in tiers:
            for s in _SCALARS:
                yield f"{da}[{ta}] , {s!r}", _make(da, za, device), s
                yield f"{s!r} , {da}[{ta}]", s, _make(da, za, device)


class _Op:
    """One binary op and its spellings."""

    def __init__(
        self,
        dunder: Callable[[Any, Any], Any],
        fn: Callable[[Any, Any], Any],
        method: str,
        supported: Callable[[lucid.dtype], bool | None],
        *,
        inplace: str | None = None,
        iop: Callable[[Any, Any], Any] | None = None,
        true_div: bool = False,
        compare: bool = False,
    ) -> None:
        self.dunder = dunder
        self.fn = fn
        self.method = method
        self.supported = supported
        self.inplace = inplace
        self.iop = iop
        self.true_div = true_div
        self.compare = compare


def _any(_: lucid.dtype) -> bool:
    return True


def _integral(d: lucid.dtype) -> bool:
    return d in _INTEGRAL


def _integer(d: lucid.dtype) -> bool:
    return d in _INTEGRAL and d is not lucid.bool_


def _real(d: lucid.dtype) -> bool:
    return d not in _COMPLEX


_OPS: dict[str, _Op] = {
    "+": _Op(operator.add, lucid.add, "add", _any, inplace="add_", iop=operator.iadd),
    "-": _Op(
        operator.sub,
        lucid.sub,
        "sub",
        lambda d: d is not lucid.bool_,
        inplace="sub_",
        iop=operator.isub,
    ),
    "*": _Op(operator.mul, lucid.mul, "mul", _any, inplace="mul_", iop=operator.imul),
    "/": _Op(
        operator.truediv,
        lucid.div,
        "div",
        _any,
        inplace="div_",
        iop=operator.itruediv,
        true_div=True,
    ),
    "//": _Op(
        operator.floordiv,
        lucid.floor_divide,
        "floor_divide",
        lambda d: d is not lucid.bool_ and d not in _COMPLEX,
    ),
    "%": _Op(
        operator.mod,
        lucid.remainder,
        "remainder",
        lambda d: d is not lucid.bool_ and d not in _COMPLEX,
    ),
    "&": _Op(operator.and_, lucid.bitwise_and, "bitwise_and", _integral),
    "|": _Op(operator.or_, lucid.bitwise_or, "bitwise_or", _integral),
    "^": _Op(operator.xor, lucid.bitwise_xor, "bitwise_xor", _integral),
    "<<": _Op(operator.lshift, lucid.bitwise_left_shift, "bitwise_left_shift", _integer),
    ">>": _Op(operator.rshift, lucid.bitwise_right_shift, "bitwise_right_shift", _integer),
    "==": _Op(operator.eq, lucid.eq, "eq", _any, compare=True),
    "!=": _Op(operator.ne, lucid.ne, "ne", _any, compare=True),
    "<": _Op(operator.lt, lucid.lt, "lt", _real, compare=True),
    "<=": _Op(operator.le, lucid.le, "le", _real, compare=True),
    ">": _Op(operator.gt, lucid.gt, "gt", _real, compare=True),
    ">=": _Op(operator.ge, lucid.ge, "ge", _real, compare=True),
}


#: What an op raises when it refuses a dtype (engine errors derive from these).
_REFUSALS = (RuntimeError, TypeError, ValueError)


def _outcome(call: Callable[[], Any]) -> tuple[str, Any, Any]:
    try:
        out = call()
    except _REFUSALS as e:
        return ("refused", type(e).__name__, None)
    return ("ok", out.dtype, out.tolist())


def _spellings(op: _Op, a: Operand, b: Operand) -> dict[str, Callable[[], Any]]:
    calls: dict[str, Callable[[], Any]] = {
        "operator": lambda: op.dunder(a, b),
        "lucid.fn": lambda: op.fn(a, b),
    }
    if isinstance(a, lucid.Tensor):
        calls["Tensor.method"] = lambda: getattr(a, op.method)(b)
    return calls


def _inplace_spellings(op: _Op, a: Operand, b: Operand) -> dict[str, Callable[[], Any]]:
    """The in-place forms, where the result has ``a``'s shape."""
    if not isinstance(a, lucid.Tensor) or op.inplace is None:
        return {}
    if isinstance(b, lucid.Tensor) and b.ndim > a.ndim:
        return {}
    iop, name = op.iop, op.inplace
    assert iop is not None

    def by_method() -> Any:
        x = a.clone()
        getattr(x, name)(b)
        return x

    def by_operator() -> Any:
        x = a.clone()
        return iop(x, b)

    return {f"Tensor.{name}": by_method, "augmented": by_operator}


def _expected(op: _Op, a: Operand, b: Operand) -> lucid.dtype:
    rt = lucid.result_type(a, b)
    if op.true_div and rt in _INTEGRAL:
        return lucid.get_default_dtype()
    return rt


def _is_bool_operand(x: Operand) -> bool:
    return isinstance(x, bool) or (isinstance(x, lucid.Tensor) and x.dtype is lucid.bool_)


@pytest.mark.parametrize("symbol", list(_OPS))
def test_every_spelling_answers_at_the_promoted_dtype(symbol: str, device: str) -> None:
    op = _OPS[symbol]
    problems: list[str] = []
    for label, a, b in _cells(device):
        expected = _expected(op, a, b)
        supported = (
            False
            if symbol == "-" and (_is_bool_operand(a) or _is_bool_operand(b))
            else op.supported(expected)
        )
        out_dtype = lucid.bool_ if op.compare else expected
        outcomes = {k: _outcome(c) for k, c in _spellings(op, a, b).items()}
        inplace = {k: _outcome(c) for k, c in _inplace_spellings(op, a, b).items()}
        kinds = {o[0] for o in outcomes.values()}
        if supported is None:
            if len(kinds) != 1:
                problems.append(f"{label}: spellings disagree: {outcomes}")
                continue
            supported = kinds == {"ok"}
        if not supported:
            all_outcomes = {**outcomes, **inplace}
            if {o[0] for o in all_outcomes.values()} != {"refused"}:
                problems.append(f"{label}: expected a refusal, got {all_outcomes}")
            continue
        if kinds != {"ok"}:
            problems.append(f"{label}: some spellings refused: {outcomes}")
            continue
        dtypes = {k: o[1] for k, o in outcomes.items()}
        if set(dtypes.values()) != {out_dtype}:
            problems.append(f"{label}: expected {out_dtype}, got {dtypes}")
            continue
        values = {repr(o[2]) for o in outcomes.values()}
        if len(values) != 1:
            problems.append(f"{label}: spellings disagree on values: {outcomes}")
            continue
        if inplace and isinstance(a, lucid.Tensor):
            if not lucid.can_cast(out_dtype, a.dtype):
                for k, o in inplace.items():
                    if o[:2] != ("refused", "RuntimeError"):
                        problems.append(f"{label}: {k} gave {o[:2]}, not a refusal")
            else:
                want_inplace_val = repr(op.dunder(a, b).to(a.dtype).tolist())
                for k, o in inplace.items():
                    if o[0] != "ok" or o[1] != a.dtype or repr(o[2]) != want_inplace_val:
                        problems.append(
                            f"{label}: {k} expected ('ok', {a.dtype}, {want_inplace_val}), got {o}"
                        )
    assert not problems, f"{len(problems)} cell(s):\n  " + "\n  ".join(problems[:30])


def _ref_operand(ref: ModuleType, x: Operand) -> Any:
    if not isinstance(x, lucid.Tensor):
        return x
    name = "bool" if x.dtype is lucid.bool_ else str(x.dtype).removeprefix("lucid.")
    return ref.tensor(x.tolist(), dtype=getattr(ref, name))


def _ref_dtype_name(ref: ModuleType, d: Any) -> str:
    if d == ref.complex32:
        return "complex64"  # Lucid has no complex32; the documented deviation.
    return str(d).rsplit(".", 1)[-1]


def _lucid_dtype_name(d: lucid.dtype) -> str:
    return "bool" if d is lucid.bool_ else str(d).removeprefix("lucid.")


@pytest.mark.parity
def test_result_type_is_the_references(ref: ModuleType) -> None:
    problems: list[str] = []
    for label, a, b in _cells("cpu"):
        if not isinstance(a, lucid.Tensor) and not isinstance(b, lucid.Tensor):
            continue
        want = _ref_dtype_name(ref, ref.result_type(_ref_operand(ref, a), _ref_operand(ref, b)))
        got = _lucid_dtype_name(lucid.result_type(a, b))
        if got != want:
            problems.append(f"{label}: lucid {got}, reference {want}")
    assert not problems, f"{len(problems)} cell(s):\n  " + "\n  ".join(problems[:30])


@pytest.mark.parity
@pytest.mark.parametrize("symbol", [s for s, o in _OPS.items() if not o.compare])
def test_operator_dtype_is_the_references(symbol: str, ref: ModuleType) -> None:
    """Where both answer, the operator's dtype is the reference's."""
    op = _OPS[symbol]
    problems: list[str] = []
    for label, a, b in _cells("cpu"):
        try:
            want_dt = op.dunder(_ref_operand(ref, a), _ref_operand(ref, b)).dtype
        except _REFUSALS:  # a reference refusal leaves nothing to compare
            continue
        if want_dt == ref.complex32:
            continue
        try:
            got = op.dunder(a, b).dtype
        except _REFUSALS:  # refusals are pinned by the sweep above
            continue
        want = _ref_dtype_name(ref, want_dt)
        if _lucid_dtype_name(got) != want:
            problems.append(f"{label}: lucid {got}, reference {want}")
    assert not problems, f"{len(problems)} cell(s):\n  " + "\n  ".join(problems[:30])


@pytest.mark.parity
@pytest.mark.parametrize("symbol", [s for s, o in _OPS.items() if o.inplace])
def test_inplace_promotion_matches_reference(symbol: str, ref: ModuleType) -> None:
    op = _OPS[symbol]
    problems: list[str] = []
    for label, a, b in _cells("cpu"):
        if not isinstance(a, lucid.Tensor):
            continue
        ra = _ref_operand(ref, a)
        rb = _ref_operand(ref, b)
        try:
            op.dunder(ra, rb)
        except _REFUSALS:
            continue
        try:
            getattr(ra.clone(), op.inplace)(rb)
            ref_ok = True
        except RuntimeError:
            ref_ok = False
        for k, call in _inplace_spellings(op, a, b).items():
            o = _outcome(call)
            if not ref_ok:
                if o[:2] != ("refused", "RuntimeError"):
                    problems.append(f"{label} ({k}): expected RuntimeError, got {o[:2]}")
            elif o[0] == "ok" and o[1] != a.dtype:
                problems.append(f"{label} ({k}): mutated dtype from {a.dtype} to {o[1]}")
    assert not problems, f"{len(problems)} cell(s):\n  " + "\n  ".join(problems[:30])


def test_promote_types_and_can_cast_read_the_same_lattice(ref: ModuleType) -> None:
    for da in _DTYPES:
        for db in _DTYPES:
            p = lucid.promote_types(da, db)
            assert p == lucid.promote_types(db, da)
            assert p == lucid.result_type(_make(da, False, "cpu"), _make(db, False, "cpu"))
            assert lucid.can_cast(da, db) == ref.can_cast(
                getattr(ref, _lucid_dtype_name(da)),
                getattr(ref, _lucid_dtype_name(db)),
            )


# ── one table ────────────────────────────────────────────────────────────────

#: Each dtype spelling a table key may use, and the category it belongs to.
_KEY_CATEGORY = {
    **dict.fromkeys(["Bool", "bool_"], "bool"),
    **dict.fromkeys(["I8", "I16", "I32", "I64", "int8", "int16", "int32", "int64"], "int"),
    **dict.fromkeys(
        ["F16", "BF16", "F32", "F64", "float16", "bfloat16", "float32", "float64"],
        "float",
    ),
    **dict.fromkeys(["C64", "C128", "complex64", "complex128"], "complex"),
}
_OWNER = Path("lucid/_promotion.py")


def _key_name(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.rsplit(".", 1)[-1]
    return None


def _is_int(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int


def _is_rank(node: ast.expr) -> bool:
    """A category (an int) or a (category, width) pair of ints or names."""
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        return all(_is_int(e) or isinstance(e, ast.Name) for e in node.elts)
    return _is_int(node)


def _is_promotion_table(node: ast.AST) -> bool:
    """A dict that ranks dtypes across categories: four or more dtype keys
    from at least two of bool / int / float / complex, each mapped to a
    category or a (category, width) pair.  An integer-only width table —
    packing bytes, checking a range — ranks nothing and is not one."""
    if not isinstance(node, ast.Dict):
        return False
    names = [_key_name(k) for k in node.keys]
    categories = [_KEY_CATEGORY[n] for n in names if n in _KEY_CATEGORY]
    return (
        len(categories) >= 4 and len(set(categories)) >= 2 and all(_is_rank(v) for v in node.values)
    )


def test_no_other_module_holds_a_kind_width_table() -> None:
    root = Path(lucid.__file__).resolve().parent
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent)
        if rel == _OWNER or rel.parts[:2] == ("lucid", "test"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found += [f"{rel}:{n.lineno}" for n in ast.walk(tree) if _is_promotion_table(n)]
    assert not found, (
        "dtype kind/width tables outside lucid/_promotion.py — promote "
        "through lucid._promotion instead: " + ", ".join(found)
    )


def test_the_owner_still_holds_its_table() -> None:
    # The scan above is only meaningful while it can recognise a table.
    root = Path(lucid.__file__).resolve().parent.parent
    tree = ast.parse((root / _OWNER).read_text(encoding="utf-8"))
    assert sum(_is_promotion_table(n) for n in ast.walk(tree)) == 1


def test_bool_beside_python_bool_skips_astype_copy(device: str) -> None:
    from lucid import _dispatch, _promotion

    assert not _promotion._lifts(_promotion.BOOL, _promotion.BOOL)
    t = lucid.tensor([True, False], device=device)
    d = t._impl.dtype
    dt = _promotion.beside_number(t._impl, d, True)
    assert dt is d
    for opmath in (False, True):
        lhs, _ = _dispatch._binary_operands(t, True, opmath=opmath)
        assert lhs is t._impl
        _, rhs = _dispatch._binary_operands(False, t, opmath=opmath)
        assert rhs is t._impl
    assert (t & True).tolist() == [True, False]
    assert (t | False).tolist() == [True, False]
    assert (t + True).tolist() == [True, True]


def test_mixed_device_zero_dim_opmath_raises_device_mismatch(
    device_gpu_only: str,
) -> None:
    from lucid._C import engine as _C_engine

    for half_dt in (lucid.float16, lucid.bfloat16):
        half_c = lucid.tensor([1.5, 2.5], dtype=half_dt, device="cpu")
        scalar_m = lucid.tensor(2.0, dtype=lucid.float32, device=device_gpu_only)
        for fn in (operator.mul, operator.truediv, operator.floordiv):
            with pytest.raises(_C_engine.DeviceMismatch):
                fn(half_c, scalar_m)
            with pytest.raises(_C_engine.DeviceMismatch):
                fn(scalar_m, half_c)
    half_m = lucid.tensor([1e4], dtype=lucid.float16, device=device_gpu_only)
    tiny_c = lucid.tensor(1e-5, dtype=lucid.float32, device="cpu")
    assert (half_m * tiny_c).item() == pytest.approx(0.1, rel=1e-2)


def test_try_unwrap_and_unique_models_all() -> None:
    from lucid import _promotion
    import lucid.models as models

    t = lucid.tensor([1.0, 2.0])
    p = lucid.nn.Parameter(lucid.tensor([1.0]))
    assert _promotion.try_unwrap(t) is t._impl
    assert _promotion.try_unwrap(p) is p._impl
    assert _promotion.try_unwrap(t._impl) is t._impl
    assert _promotion.try_unwrap(42) is None

    class _BogusImpl:
        _impl = "not-a-tensor-impl"

    assert _promotion.try_unwrap(_BogusImpl()) is None
    assert len(models.__all__) == len(set(models.__all__))


def test_wrap_axis_errors_name_the_calling_op() -> None:
    from lucid._C import engine as _C_engine

    x1 = lucid.tensor([1.0, 2.0])._impl
    x2 = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])._impl
    idx1 = lucid.tensor([0])._impl
    scalar = lucid.tensor(1.0)._impl

    calls: list[tuple[str, Callable[[], object]]] = [
        ("sort", lambda: _C_engine.sort(x1, 5)),
        ("argsort", lambda: _C_engine.argsort(x1, 5)),
        ("argmax", lambda: _C_engine.argmax(x1, 5, False)),
        ("argmin", lambda: _C_engine.argmin(x1, 5, False)),
        ("topk", lambda: _C_engine.topk(x1, 1, 5)),
        ("repeat", lambda: _C_engine.repeat(x1, 2, 5)),
        ("repeat", lambda: _C_engine.repeat(scalar, 2, 0)),
        ("concatenate", lambda: _C_engine.concatenate([x1, x1], 5)),
        ("split", lambda: _C_engine.split(x1, 1, 5)),
        ("split_at", lambda: _C_engine.split_at(x1, [1], 5)),
        ("unbind", lambda: _C_engine.unbind(x1, 5)),
        ("roll", lambda: _C_engine.roll(x1, [1], [5])),
        ("gather", lambda: _C_engine.gather(x1, idx1, 5)),
        ("diagonal", lambda: _C_engine.diagonal(x2, 0, 5, 1)),
        ("diagonal", lambda: _C_engine.diagonal(x2, 0, 0, 5)),
    ]
    for op_name, call in calls:
        with pytest.raises(IndexError, match=rf"^{op_name}: "):
            call()
