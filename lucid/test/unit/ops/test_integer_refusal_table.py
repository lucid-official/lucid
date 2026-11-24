"""The bool and integer refusals of the Python entry points, pinned row by row (LCD-1026).

``lucid._ops._bool_refusal.ROWS`` is the one table of what the Python entry points -- the
composites, the adapters, the losses -- turn away: a row per ``(op, role)`` says, for bool and
each signed integer, the class the reference raises, or that the reference answers.  This file is
the test of the table:

* every row has a call (a *pin*), and every pin names a row -- a refusal cannot be added and left
  unpinned, and a row cannot be dropped unnoticed;
* each pin is replayed on the reference for bool, int8, int16, int32 and int64 (``-m parity``): it
  answers where the row says it answers, and raises the row's class where the row refuses -- a
  verdict *or* a class that drifts from the reference's fails;
* each pin is replayed on Lucid, on both devices, and does the same;
* the table is well formed: one owner per ``(op, dtype)`` against the engine's schemas, one
  spelling of a dtype in a message across the two layers, and a float input pays for no lookup.

The classes, named after how the reference refuses:

``kernel``
    ``NotImplementedError``: a kernel with no instantiation for the dtype.
``guard``
    a ``RuntimeError`` that is not a ``NotImplementedError``: a check on the input dtype.
``value``
    a ``ValueError``: a check Python makes on an argument.

A pin that says nothing about its dtypes takes them from its row.  A call whose answer differs
from the row's -- a loss beside a floating target, an operand of another dtype -- names the dtypes
it refuses and the class.  Every Lucid refusal is a ``RuntimeError``, as the reference's are.
"""

from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._ops import _bool_refusal as _table
from lucid._ops._bool_refusal import BOOL_REFUSED, ROWS, Rule
from lucid.test._helpers.dtype_refusal_pins import DTYPES, PINNED, PINS, Pin

pytestmark = pytest.mark.filterwarnings("ignore")

#: The class each Lucid error names, as the reference's classes are told apart.
_LABEL: dict[type[BaseException], str] = {
    _C_engine.NotImplementedError: "kernel",
    _C_engine.DtypeMismatch: "guard",
    _C_engine.InvalidArgument: "value",
}

_VECTOR = [1, 0, 1, 1]
_MATRIX = [[1, 0, 1], [0, 0, 1]]


_LUCID_DTYPES: dict[str, lucid.dtype] = {
    "bool": lucid.bool_,
    "int8": lucid.int8,
    "int16": lucid.int16,
    "int32": lucid.int32,
    "int64": lucid.int64,
    "float16": lucid.float16,
    "float32": lucid.float32,
    "float64": lucid.float64,
    "complex64": lucid.complex64,
}


def _reference_dtype(ref: ModuleType, name: str) -> object:
    """The reference framework's dtype of the same name (``bool`` is spelled alike)."""
    return getattr(ref, name)


def _refused_dtypes(pin: Pin) -> frozenset[str]:
    """The dtypes ``pin``'s call is refused for: its own say, or its row's."""
    if pin.refused is not None:
        return pin.refused
    rules = ROWS[(pin.op, pin.role)].by_code()
    return frozenset(name for name, rule in zip(DTYPES, rules, strict=True) if rule is not None)


def _kind_of(pin: Pin, dtype: str) -> str:
    """The class the reference raises for ``pin`` on ``dtype``."""
    if isinstance(pin.kind, str):
        return pin.kind
    if pin.kind is not None and dtype in pin.kind:
        return pin.kind[dtype]
    rule = ROWS[(pin.op, pin.role)].by_code()[DTYPES.index(dtype)]
    assert rule is not None, (pin.op, pin.role, dtype)
    return _LABEL[rule.error]


def _lucid_matches(kind: str, error: BaseException) -> bool:
    """Whether ``error`` is a Lucid refusal of class ``kind``."""
    if not isinstance(error, RuntimeError):
        return False
    if kind == "kernel":
        return isinstance(error, NotImplementedError)
    if kind == "value":
        return isinstance(error, ValueError)
    return not isinstance(error, NotImplementedError)


def _reference_matches(kind: str, error: BaseException) -> bool:
    """Whether ``error`` is the reference's refusal of class ``kind``."""
    if kind == "kernel":
        return isinstance(error, NotImplementedError)
    if kind == "value":
        return isinstance(error, ValueError)
    return isinstance(error, RuntimeError) and not isinstance(error, NotImplementedError)


_CASES = [(name, dtype) for name in PINS for dtype in DTYPES]


@pytest.mark.parametrize(("name", "dtype"), _CASES)
def test_lucid_does_what_the_table_says(device: str, name: str, dtype: str) -> None:
    pin = PINS[name]
    x = lucid.tensor(_VECTOR, dtype=_LUCID_DTYPES[dtype], device=device)
    m = lucid.tensor(_MATRIX, dtype=_LUCID_DTYPES[dtype], device=device)
    if dtype not in _refused_dtypes(pin):
        assert pin.call(lucid, x, m) is not None
        return
    kind = _kind_of(pin, dtype)
    with pytest.raises(RuntimeError) as caught:
        pin.call(lucid, x, m)
    assert _lucid_matches(kind, caught.value), (kind, type(caught.value), str(caught.value))


@pytest.mark.parity
@pytest.mark.parametrize(("name", "dtype"), _CASES)
def test_reference_does_what_the_table_says(ref: ModuleType, name: str, dtype: str) -> None:
    pin = PINS[name]
    x = ref.tensor(_VECTOR, dtype=_reference_dtype(ref, dtype))
    m = ref.tensor(_MATRIX, dtype=_reference_dtype(ref, dtype))
    if dtype not in _refused_dtypes(pin):
        assert pin.call(ref, x, m) is not None
        return
    kind = _kind_of(pin, dtype)
    with pytest.raises(Exception) as caught:
        pin.call(ref, x, m)
    assert _reference_matches(kind, caught.value), (kind, type(caught.value), str(caught.value))


# ── the table is pinned, and well formed ────────────────────────────────────────────────────────


def test_every_row_of_the_table_has_a_pin_and_every_pin_a_row() -> None:
    assert set(ROWS) <= PINNED, f"rows with no pin: {sorted(set(ROWS) - PINNED)}"
    # A pin with no row is an op the engine's schema owns, asked about by a Python entry that
    # reaches the engine on some paths only (``grid_sample``'s image).
    engine = _table.engine_refusals()
    rowless = PINNED - set(ROWS)
    assert all(role == "" and op in engine for op, role in rowless), sorted(rowless)
    for name, pin in PINS.items():
        if (pin.op, pin.role) in rowless:
            assert pin.refused is not None and isinstance(pin.kind, str), name


def test_a_pin_that_refuses_what_its_row_answers_says_so_on_purpose() -> None:
    # An override that restates the row hides a stale pin: it must differ from what the row says.
    for name, pin in PINS.items():
        if pin.refused is None or (pin.op, pin.role) not in ROWS:
            continue
        row_refused = frozenset(
            dtype
            for dtype, rule in zip(DTYPES, ROWS[(pin.op, pin.role)].by_code(), strict=True)
            if rule is not None
        )
        assert pin.refused != row_refused, f"{name} restates its row"


def test_the_rows_are_the_bool_table_the_old_consumers_read() -> None:
    primary = {op for (op, role), row in ROWS.items() if role == "" and row.on_bool is not None}
    assert set(BOOL_REFUSED) == primary
    for op in BOOL_REFUSED:
        assert BOOL_REFUSED[op] == ROWS[(op, "")].on_bool


def test_an_op_has_one_owner_per_dtype() -> None:
    # The engine owns the dtypes its schema declares and Python the rest: ``rms_norm`` and
    # ``embedding_bag`` are the engine's for integers and the entry's for bool, and an entry that
    # reaches the engine on some paths only asks the schema (``refuse_declared``) rather than
    # declaring the dtype a second time.  No cell is in both.
    engine = _table.engine_refusals()
    both: set[tuple[str, str]] = set()
    for (op, role), row in ROWS.items():
        if role != "" or op not in engine:
            continue
        for dtype, mine, theirs in zip(DTYPES, row.by_code(), engine[op], strict=True):
            if mine is not None and theirs is not None:
                both.add((op, dtype))
    assert not both, sorted(both)


def test_a_rule_names_the_cast_of_its_class() -> None:
    hints = {rule.hint for row in ROWS.values() for rule in row.by_code() if rule is not None}
    engine_hints = {
        rule.hint
        for rules in _table.engine_refusals().values()
        for rule in rules
        if rule is not None
    }
    assert engine_hints <= {_table._FLOATING, _table._INTEGER, _table._NUMBER}
    assert hints <= {
        _table._FLOATING,
        _table._INTEGER,
        _table._NUMBER,
        _table._INT64,
        _table._INT32_OR_64,
        _table._BOOL_MASK,
    }


def test_a_row_sets_one_integer_slot() -> None:
    rule = Rule(_table._KERNEL, _table._FLOATING)
    with pytest.raises(ValueError, match="one of on_ints, on_narrow and on_short"):
        _table.Row("x", on_ints=rule, on_narrow=rule)
    with pytest.raises(ValueError, match="one of on_ints, on_narrow and on_short"):
        _table.Row("x", on_narrow=rule, on_short=rule)
    with pytest.raises(ValueError, match="one of on_ints, on_narrow and on_short"):
        _table.Row("x", on_ints=rule, on_short=rule)


def test_a_dtype_is_refused_by_its_width() -> None:
    # ``local_response_norm`` answers int64 and refuses the narrower integers and bool.
    rules = ROWS[("local_response_norm", "")].by_code()
    assert [rule is not None for rule in rules] == [True, True, True, True, False]
    # ``one_hot`` is the same by width, with its own cast.
    assert {rule.hint for rule in ROWS[("one_hot", "")].by_code() if rule} == {_table._INT64}
    # ``gather`` reads int32 and int64, so only bool, int8 and int16 are refused.
    rules = ROWS[("gather", "index")].by_code()
    assert [rule is not None for rule in rules] == [True, True, True, False, False]
    assert {rule.hint for rule in rules if rule} == {_table._INT32_OR_64}


def test_a_role_with_no_row_reads_the_primary_one() -> None:
    error = _table.refusal("huber_loss", 1, "target")
    assert isinstance(error, _C_engine.NotImplementedError)
    assert "an int8 target" in str(error)


def test_an_in_place_twin_takes_the_row_of_its_op() -> None:
    assert isinstance(_table.refusal("pow_", 0), _C_engine.NotImplementedError)
    assert str(_table.refusal("celu_", 3)).startswith("celu_ does not accept int32")


def test_an_integer_is_told_to_cast_where_its_row_says() -> None:
    assert str(_table.refusal("quantile", 3)) == (
        "quantile does not accept int32 — cast to a floating dtype, e.g. x.float()"
    )
    assert str(_table.refusal("cross_entropy", 4, "target", index=True)).endswith(
        "cast to an integer dtype, e.g. x.long()"
    )
    assert "a bool target" in str(_table.refusal("cross_entropy", 0, "target"))


def test_a_refusal_the_table_does_not_hold_is_a_lookup_error() -> None:
    with pytest.raises(KeyError, match="no refusal row"):
        _table.refusal("not_an_op", 0)
    with pytest.raises(KeyError, match="does not refuse"):
        _table.refusal("cov", 3)


def test_a_float_pays_for_no_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    # The check runs on every float tensor of every op that carries it; it must not reach the
    # table for one.
    def boom(*args: object) -> None:
        raise AssertionError("a float input reached the table")

    monkeypatch.setattr(_table, "_rule", boom)
    x = lucid.tensor([1.0, 2.0])
    _table.refuse("quantile", x)
    _table.refuse("quantile", x, "weight")
    _table.refuse("quantile", 3.0)
    _table.refuse("quantile", None)
    _table.refuse_common("gammainc", x, x)
    _table.refuse_mixed("prelu", x, x)
    _table.refuse_declared("embedding_bag", x)
    _table.refuse_complex("histogram", x)
    _table.refuse_bool("cov", x)


def test_a_dtype_the_row_answers_pays_for_one_membership_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An int64 ``take`` index, an int32 ``gather`` index and an empty int8 one all pass without the
    # rule being read: the row's refused codes answer, and only a refusal builds the rule.
    def boom(*args: object) -> None:
        raise AssertionError("an answered dtype reached the rule")

    monkeypatch.setattr(_table, "_rule", boom)
    _table.refuse("take", lucid.tensor([0, 1]), "index")
    _table.refuse("gather", lucid.tensor([0, 1], dtype=lucid.int32), "index")
    _table.refuse("gather", lucid.zeros(0, dtype=lucid.int8), "index")


def test_a_row_that_answers_an_empty_operand_lets_it_through_whatever_its_dtype() -> None:
    assert ROWS[("gather", "index")].answers_empty
    assert not ROWS[("take", "index")].answers_empty
    for dtype in (lucid.bool_, lucid.int8, lucid.int16):
        _table.refuse("gather", lucid.zeros(0, 2, dtype=dtype), "index")
        with pytest.raises(RuntimeError):
            _table.refuse("gather", lucid.zeros(1, 2, dtype=dtype), "index")
    with pytest.raises(RuntimeError):
        _table.refuse("take", lucid.zeros(0, dtype=lucid.int8), "index")


def test_a_row_names_the_codes_it_refuses() -> None:
    assert ROWS[("gather", "index")].refused_codes() == {0, 1, 2}
    assert ROWS[("take", "index")].refused_codes() == {0, 1, 2, 3}
    assert ROWS[("local_response_norm", "")].refused_codes() == {0, 1, 2, 3}
    assert ROWS[("quantile", "")].refused_codes() == {0, 1, 2, 3, 4}


def test_what_the_reference_answers_is_not_refused() -> None:
    # ``refuse`` answers an operand its row has no rule for.
    ints = lucid.tensor([1, 2])
    _table.refuse("cov", ints)
    _table.refuse("masked_select", lucid.tensor([True, False]), "mask")
    _table.refuse_common("gammainc", ints, lucid.tensor([1.0, 2.0]))
    _table.refuse_common("searchsorted", ints, lucid.tensor([True, False]))
    with pytest.raises(NotImplementedError):
        _table.refuse_common("gammainc", ints, lucid.tensor([True, False]))
    with pytest.raises(NotImplementedError):
        _table.refuse_common("searchsorted", lucid.tensor([True]), lucid.tensor([False]))


def test_a_refused_in_place_celu_leaves_the_tensor_as_it_was(device: str) -> None:
    x = lucid.tensor(_VECTOR, dtype=lucid.int32, device=device)
    version = x._impl.version
    with pytest.raises(NotImplementedError, match="celu does not accept int32"):
        lucid.nn.functional.celu_(x)
    assert x.tolist() == _VECTOR
    assert x._impl.version == version


# ── what the reference passes through is passed through ─────────────────────────────────────────

#: ``feature_alpha_dropout`` draws a channel mask only when something is dropped; at eval, or at
#: ``p == 0``, the reference hands the input back whatever its dtype, so the refusal stays on the
#: path that draws (the engine's dropout rows sit inside ``training && p > 0`` for the same reason).
_NOTHING_DROPPED = {"eval": {"p": 0.5, "training": False}, "p0": {"p": 0.0, "training": True}}


@pytest.mark.parametrize("case", list(_NOTHING_DROPPED))
@pytest.mark.parametrize("dtype", DTYPES)
def test_feature_alpha_dropout_returns_an_input_nothing_is_dropped_from(
    device: str, dtype: str, case: str
) -> None:
    x = lucid.tensor([[[1, 0], [1, 1]]], dtype=_LUCID_DTYPES[dtype], device=device)
    out = lucid.nn.functional.feature_alpha_dropout(x, **_NOTHING_DROPPED[case])
    assert out.dtype == x.dtype
    assert out.tolist() == x.tolist()


@pytest.mark.parametrize("dtype", DTYPES)
def test_feature_alpha_dropout_refuses_what_it_drops_from(device: str, dtype: str) -> None:
    x = lucid.tensor([[[1, 0], [1, 1]]], dtype=_LUCID_DTYPES[dtype], device=device)
    with pytest.raises(RuntimeError, match=f"does not accept {dtype}") as caught:
        lucid.nn.functional.feature_alpha_dropout(x, p=0.5, training=True)
    assert not isinstance(caught.value, NotImplementedError)


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_NOTHING_DROPPED))
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_reference_returns_what_it_drops_nothing_from(
    ref: ModuleType, dtype: str, case: str
) -> None:
    x = ref.tensor([[[1, 0], [1, 1]]], dtype=_reference_dtype(ref, dtype))
    out = ref.nn.functional.feature_alpha_dropout(x, **_NOTHING_DROPPED[case])
    assert out.dtype == x.dtype
    assert out.tolist() == x.tolist()


# ── a mask is a bool, and nothing else ──────────────────────────────────────────────────────────

_NOT_BOOL = ("int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex64")


@pytest.mark.parametrize("dtype", _NOT_BOOL)
def test_masked_select_refuses_a_mask_of_any_dtype_but_bool(device: str, dtype: str) -> None:
    if device == "metal" and dtype == "float64":
        pytest.skip("Metal holds no float64")
    values = lucid.tensor([1.0, 2.0, 3.0], device=device)
    mask = lucid.tensor([1, 0, 1], dtype=_LUCID_DTYPES[dtype], device=device)
    with pytest.raises(RuntimeError, match=f"does not accept (an|a) {dtype} mask") as caught:
        lucid.masked_select(values, mask)
    assert not isinstance(caught.value, NotImplementedError)
    assert lucid.masked_select(values, mask.bool()).tolist() == [1.0, 3.0]


@pytest.mark.parametrize("dtype", _NOT_BOOL)
def test_the_method_and_the_function_refuse_a_mask_alike(dtype: str) -> None:
    values = lucid.tensor([1.0, 2.0, 3.0])
    mask = lucid.tensor([1, 0, 1], dtype=_LUCID_DTYPES[dtype])
    with pytest.raises(RuntimeError, match="does not accept"):
        values.masked_select(mask)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _NOT_BOOL)
def test_the_reference_refuses_a_mask_of_any_dtype_but_bool(ref: ModuleType, dtype: str) -> None:
    values = ref.tensor([1.0, 2.0, 3.0])
    mask = ref.tensor([1, 0, 1], dtype=_reference_dtype(ref, dtype))
    with pytest.raises(RuntimeError) as caught:
        ref.masked_select(values, mask)
    assert not isinstance(caught.value, NotImplementedError)


# ── a real-valued kernel is not handed a complex tensor ─────────────────────────────────────────


@pytest.mark.parametrize("op", ["histogram", "histogramdd"])
def test_a_histogram_refuses_a_complex_input_where_it_used_to_cast_it(device: str, op: str) -> None:
    # ``histogram`` cast a complex input to the default float, dropping its imaginary part.
    x = lucid.tensor(
        [[1 + 1j, 2 + 0j], [3 - 1j, 4 + 2j]], dtype=_LUCID_DTYPES["complex64"], device=device
    )
    with pytest.raises(NotImplementedError, match=f"{op} does not accept complex64"):
        getattr(lucid, op)(x, 2)


@pytest.mark.parity
@pytest.mark.parametrize("op", ["histogram", "histogramdd"])
def test_the_reference_refuses_a_complex_histogram_input(ref: ModuleType, op: str) -> None:
    x = ref.tensor([[1 + 1j, 2 + 0j], [3 - 1j, 4 + 2j]])
    with pytest.raises(NotImplementedError):
        getattr(ref, op)(x, 2)


# ── what a refusal tells the caller to cast to ──────────────────────────────────────────────────


def test_a_mask_is_told_to_cast_to_bool_not_to_float(device: str) -> None:
    # A float cast would invert the meaning of a 1 / 0 keep mask: ``attn_mask`` and
    # ``masked_select``'s mask are told to ``.bool()`` it.
    mask = lucid.tensor([[1, 0], [1, 1]], dtype=lucid.int32, device=device)
    q = lucid.ones(1, 2, 3, device=device)
    with pytest.raises(RuntimeError, match=r"cast to a bool dtype, e\.g\. mask\.bool\(\)"):
        lucid.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask)
    with pytest.raises(RuntimeError, match=r"cast to a bool dtype, e\.g\. mask\.bool\(\)"):
        lucid.masked_select(q, mask.reshape(1, 2, 2)[..., :1])
