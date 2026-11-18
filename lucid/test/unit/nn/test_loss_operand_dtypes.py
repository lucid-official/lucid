"""Every loss decides its operand dtypes at its door, the same on every
device (LCD-366).

Each loss used to leave its operands' dtypes to whichever op touched them
first. ``nll_loss``, ``multi_margin_loss`` and ``multilabel_margin_loss``
took an integer input on the CPU and returned an integer or a float loss.
On Metal the same call raised ``ValueError`` ("cannot store nan in an
integer tensor"). The audit's dtype axis recorded all three as
asymmetric. The other losses drifted from the reference in both
directions:

- ``mse_loss`` scored an integer pair, which the reference refuses;
- ``margin_ranking_loss`` and ``poisson_nll_loss`` refused the integer
  operands that the reference takes;
- ``huber_loss`` read a float16 target as float32 on the CPU.

The rule now lives in one table, ``_OPERAND_DTYPES``, which holds each
loss's input kind and target kind. Every loss checks it before it
computes. These tests sweep that table:

- every loss in ``nn.functional`` has a row;
- for each dtype the call is refused with a ``TypeError`` naming the loss
  and the dtype, on the CPU and on Metal alike, or it computes the value
  of the same loss on float32 operands -- a bool is refused with the
  reference's ``RuntimeError`` and a cast to make (LCD-1024);
- under ``-m parity``, each verdict is the reference's, and so is each
  accepted value and its dtype.

There are three sweeps. In ``input``, the input is set to each dtype and
the floating companions follow it, as the audit's dtype axis builds its
calls. ``mixed`` keeps those companions at float32. ``target`` keeps the
input at float32 and sets the target to each dtype. Class indices stay
``int64`` except in ``target``.
"""

import math
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn.functional as F
from lucid.nn.functional import loss as _loss_module
from lucid.nn.functional._loss_checks import _FLOATING_RESULT, _OPERAND_DTYPES
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.nn.test_loss_boundary_validation import losses_with

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]

DTYPES = ("int8", "int32", "int64", "bool", "float16", "float32")
SWEEPS = ("input", "mixed", "target")

_INTEGERS = frozenset({"int8", "int32", "int64"})
_FLOATS = frozenset({"float16", "float32"})
_ACCEPTS: dict[str, frozenset[str]] = {
    "floating": _FLOATS,
    "real": _FLOATS | _INTEGERS,
    "numeric": _FLOATS | _INTEGERS | {"bool"},
    "index": _INTEGERS,
}

#: The engine's CTC kernel has no float16 path, and neither has the
#: reference's CPU one: an unsupported dtype, not a wrong one.
_NOT_IMPLEMENTED = frozenset({("ctc_loss", "float16")})

# Integer-valued data, so every integer and bool cast is exact.
_X = [[1.0, -2.0, 3.0, 0.0], [2.0, 1.0, -1.0, -3.0], [0.0, 2.0, 1.0, -1.0]]
_Y = [[0.0, -1.0, 2.0, 1.0], [1.0, 2.0, -2.0, -1.0], [1.0, 1.0, 0.0, -2.0]]
_Z = [[2.0, 0.0, 1.0, -1.0], [0.0, -1.0, 1.0, 2.0], [-1.0, 0.0, 2.0, 1.0]]
_POS = [[1.0, 2.0, 3.0, 1.0], [2.0, 1.0, 1.0, 3.0], [1.0, 1.0, 2.0, 2.0]]
_LOGP = [[-v for v in row] for row in _POS]
_BIN = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
_PROB = [[0.25, 0.5, 0.75, 0.5], [0.5, 0.25, 0.25, 0.75], [0.75, 0.5, 0.5, 0.25]]
# A bool label is True for both signs, so it reads as 1 everywhere.
_SIGN = [1.0, -1.0, 1.0]
_SIGNS = [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]]
_CLASS = [0, 1, 3]
_LABELS = [[0, 1, -1, 0], [3, -1, 0, 0], [2, 1, 0, -1]]
_CTC = [[[-1.5] * 5] * 2] * 6

#: ``make(values, dtype)`` builds one tensor of either framework.
Make = Callable[[object, str], object]
Call = tuple[list[object], dict[str, object]]


def _call(name: str, make: Make, x: str, t: str, idx: str, aux: str) -> Call:
    """The positional arguments and options of an ordinary call of loss
    ``name``: its input(s) at ``x``, its target at ``t`` (``idx`` for class
    indices) and any other floating operand at ``aux``."""
    calls: dict[str, Callable[[], Call]] = {
        "mse_loss": lambda: ([make(_X, x), make(_Y, t)], {}),
        "l1_loss": lambda: ([make(_X, x), make(_Y, t)], {}),
        "smooth_l1_loss": lambda: ([make(_X, x), make(_Y, t)], {}),
        "huber_loss": lambda: ([make(_X, x), make(_Y, t)], {}),
        "cross_entropy": lambda: ([make(_X, x), make(_CLASS, idx)], {}),
        "nll_loss": lambda: ([make(_LOGP, x), make(_CLASS, idx)], {}),
        "multi_margin_loss": lambda: ([make(_X, x), make(_CLASS, idx)], {}),
        "multilabel_margin_loss": lambda: ([make(_X, x), make(_LABELS, idx)], {}),
        "binary_cross_entropy": lambda: ([make(_PROB, x), make(_BIN, t)], {}),
        "binary_cross_entropy_with_logits": lambda: ([make(_X, x), make(_BIN, t)], {}),
        "kl_div": lambda: ([make(_LOGP, x), make(_PROB, t)], {"reduction": "batchmean"}),
        "hinge_embedding_loss": lambda: ([make(_X, x), make(_SIGNS, t)], {}),
        "soft_margin_loss": lambda: ([make(_X, x), make(_SIGNS, t)], {}),
        "multilabel_soft_margin_loss": lambda: ([make(_X, x), make(_BIN, t)], {}),
        "poisson_nll_loss": lambda: ([make(_POS, x), make(_POS, t)], {}),
        "gaussian_nll_loss": lambda: ([make(_X, x), make(_Y, t), make(_POS, aux)], {}),
        "margin_ranking_loss": lambda: (
            [make(_X[0][:3], x), make(_Y[0][:3], x), make(_SIGN, t)],
            {},
        ),
        "cosine_embedding_loss": lambda: ([make(_X, x), make(_Y, x), make(_SIGN, t)], {}),
        "triplet_margin_loss": lambda: ([make(_X, x), make(_Y, x), make(_Z, x)], {}),
        "triplet_margin_with_distance_loss": lambda: ([make(_X, x), make(_Y, x), make(_Z, x)], {}),
        "ctc_loss": lambda: (
            [
                make(_CTC, x),
                make([[1, 2], [3, 4]], "int64"),
                make([6, 6], "int64"),
                make([2, 2], "int64"),
            ],
            {},
        ),
    }
    return calls[name]()


def _dtypes(name: str, sweep: str, dtype: str) -> tuple[str, str, str, str]:
    """``(input, target, class index, other operand)`` dtypes of one cell."""
    if sweep == "input":
        return dtype, dtype, "int64", dtype
    if sweep == "mixed":
        return dtype, "float32", "int64", "float32"
    return "float32", dtype, dtype, "float32"


def _expected(name: str, sweep: str, dtype: str) -> bool:
    """Whether ``_OPERAND_DTYPES`` takes the cell — read off the table, so a
    row that changes is a row these tests then hold the loss to."""
    x, t, idx, aux = _dtypes(name, sweep, dtype)
    input_kind, target_kind = _OPERAND_DTYPES[name]
    target = idx if target_kind == "index" else t
    ok = x in _ACCEPTS[input_kind]
    if target_kind:
        ok = ok and target in _ACCEPTS[target_kind]
    if name in _FLOATING_RESULT:
        ok = ok and (x in _FLOATS or target in _FLOATS)
    if name == "gaussian_nll_loss":
        ok = ok and aux in _FLOATS
    if name == "poisson_nll_loss":
        # Each operand takes a bool, but the loss subtracts the product of the two.
        ok = ok and not (x == "bool" and target == "bool")
    return ok


def _sweep_cells() -> list[tuple[str, str, str]]:
    return [
        (name, sweep, dtype)
        for name in sorted(_OPERAND_DTYPES)
        for sweep in SWEEPS
        for dtype in DTYPES
        if sweep == "input" or _OPERAND_DTYPES[name][1]
    ]


CELLS = _sweep_cells()


def _is_index(values: object) -> bool:
    """Whether ``values`` are class indices or lengths (Python ints), not
    data (Python floats)."""
    while isinstance(values, list):
        values = values[0]
    return isinstance(values, int)


def _data(values: object, dtype: str, device: str, *, widen: bool = False) -> lucid.Tensor:
    """``values`` as a Lucid tensor of ``dtype`` on ``device``; with ``widen``, cast on to float32
    — the same values, computed in float32."""
    d = lucid.bool_ if dtype == "bool" else getattr(lucid, dtype)
    t = lucid.tensor(values, device=device).to(dtype=d)
    return t.float() if widen else t


def _lucid_make(device: str, *, as_float32: bool = False) -> Make:
    """A ``make`` for Lucid tensors on ``device``.  With ``as_float32`` each
    data operand is built at its dtype and then cast to float32 — the same
    values, computed in float32 — and indices are left alone."""

    def make(values: object, dtype: str) -> object:
        return _data(values, dtype, device, widen=as_float32 and not _is_index(values))

    return make


def _run(name: str, make: Make, dtypes: tuple[str, str, str, str]) -> object:
    """Loss ``name`` called on operands built by ``make`` at ``dtypes``."""
    args, options = _call(name, make, *dtypes)
    return getattr(F, name)(*args, **options)


def _close(got: float, want: float, dtype: str) -> bool:
    tol = 2e-2 if "float16" in dtype else 1e-5
    return abs(got - want) <= tol * max(1.0, abs(want))


def test_every_loss_has_a_dtype_rule() -> None:
    public = {
        name
        for name in F.__all__
        if getattr(getattr(F, name, None), "__module__", "") == _loss_module.__name__
    }
    assert public == set(_OPERAND_DTYPES)
    assert set(losses_with("reduction")) <= set(_OPERAND_DTYPES)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("name", "sweep", "dtype"), CELLS)
def test_the_door_decides(name: str, sweep: str, dtype: str, device: str) -> None:
    """Refused with a ``TypeError`` naming the loss and the dtype (a bool, with
    the reference's ``RuntimeError``), or computed as on float32 operands —
    the same verdict on each device."""
    make = _lucid_make(device)
    dtypes = _dtypes(name, sweep, dtype)
    if (name, dtype) in _NOT_IMPLEMENTED and sweep == "input":
        with pytest.raises(NotImplementedError):
            _run(name, make, dtypes)
        return
    if not _expected(name, sweep, dtype) and dtype == "bool":
        # A bool is refused as every op refuses one (LCD-1024): the reference's class, and a cast.
        with pytest.raises(RuntimeError, match=f"{name} does not accept") as caught:
            _run(name, make, dtypes)
        assert "cast to" in str(caught.value)
        return
    if not _expected(name, sweep, dtype):
        with pytest.raises(TypeError, match=name) as caught:
            _run(name, make, dtypes)
        assert f"lucid.{dtype}" in str(caught.value)
        return
    out = _run(name, make, dtypes)
    assert isinstance(out, lucid.Tensor) and out.is_floating_point()
    want = _run(name, _lucid_make(device, as_float32=True), dtypes)
    assert isinstance(want, lucid.Tensor)
    got_v, want_v = float(out.item()), float(want.item())
    assert math.isfinite(got_v)
    assert _close(got_v, want_v, dtype), (got_v, want_v)


# ── against the reference ─────────────────────────────────────────────────

#: Cells Lucid takes and the reference refuses, each for a stated reason.
#: Every other cell has the reference's verdict.
_LENIENT: dict[tuple[str, str, str], str] = {
    **{
        (name, sweep, "float16"): (
            "the reference's CPU kernel has no float16 path; its GPU kernel "
            "has one, and Lucid follows that"
        )
        for name in ("multi_margin_loss", "multilabel_margin_loss")
        for sweep in ("input", "mixed")
    },
    **{
        (name, "target", width): (
            "Lucid takes class indices of any integer width; the reference " "only int64"
        )
        for name in ("cross_entropy", "nll_loss", "multi_margin_loss", "multilabel_margin_loss")
        for width in ("int8", "int32")
    },
    **{
        ("binary_cross_entropy", sweep, "float16"): (
            "the reference wants the target at the input's exact dtype; Lucid "
            "computes the two in the input's"
        )
        for sweep in ("mixed", "target")
    },
}

#: Cells both take, where the loss comes back at another dtype.
_OTHER_DTYPE: dict[tuple[str, str, str], str] = {
    ("binary_cross_entropy_with_logits", "target", "float16"): (
        "the reference rounds the float32 loss to the target's float16; Lucid "
        "keeps the float32 the two promote to"
    )
}


def _name(dtype: object) -> str:
    """A dtype's bare name, ``float32`` for either framework's."""
    return str(dtype).rsplit(".", 1)[-1]


@pytest.mark.parity
@pytest.mark.parametrize(("name", "sweep", "dtype"), CELLS)
def test_the_rule_is_the_references(ref: ModuleType, name: str, sweep: str, dtype: str) -> None:
    def make_ref(values: object, d: str) -> object:
        return ref.tensor(values).to(getattr(ref, d))

    dtypes = _dtypes(name, sweep, dtype)
    args, options = _call(name, make_ref, *dtypes)
    try:
        want = getattr(ref.nn.functional, name)(*args, **options)
    except (RuntimeError, TypeError, NotImplementedError):
        want = None
    try:
        got = _run(name, _lucid_make("cpu"), dtypes)
    except (TypeError, NotImplementedError):
        got = None

    cell = (name, sweep, dtype)
    if cell in _LENIENT:
        assert want is None and got is not None, _LENIENT[cell]
        return
    assert (got is None) == (want is None), (got, want)
    if got is None:
        return
    assert isinstance(got, lucid.Tensor)
    if cell not in _OTHER_DTYPE:
        assert _name(got.dtype) == _name(want.dtype)
    assert _close(float(got.item()), float(want.item()), dtype)


# ── what the table does not hold (LCD-1046) ───────────────────────────────
#
# ``_OPERAND_DTYPES`` is the rule for an input and a target at the dtypes a loss is usually
# called with, and the sweeps above read each of its cells against the reference.  Three
# things were not in it, and had no check of their own:
#
# - a class ``weight``: the reference reads it at the input's dtype, so a bool, integer,
#   float16 or float64 weight beside float32 logits is a dtype ``RuntimeError`` -- but only
#   for class indices; a target of class probabilities multiplies the weight in, and takes
#   any dtype;
# - the length tensors of ``ctc_loss``, which must be integral (a bool or floating one is a
#   ``RuntimeError``, and a Python list of anything but ints a ``TypeError``);
# - the pair and the options of ``poisson_nll_loss``: a bool beside another dtype is a number
#   (``exp`` of it is a float), refused only where the reference subtracts one.

_WEIGHT = [1.0, 0.0, 1.0, 1.0]
_WEIGHT_DTYPES = ("bool", "int8", "int64", "float16", "float32")
_WEIGHT_LOSSES = ("cross_entropy", "nll_loss", "multi_margin_loss")


def _weighted(
    functional: ModuleType, name: str, make: Make, weight: str, *, probabilities: bool = False
) -> tuple[float, str]:
    """``(value, dtype name)`` of loss ``name`` of ``functional`` on float32 logits and a class
    ``weight`` of dtype ``weight``; the target is class probabilities with ``probabilities``,
    class indices without."""
    logits = make(_LOGP if name == "nll_loss" else _X, "float32")
    target = make(_PROB, "float32") if probabilities else make(_CLASS, "int64")
    out = getattr(functional, name)(logits, target, weight=make(_WEIGHT, weight))
    return float(out.item()), _name(out.dtype)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("weight", _WEIGHT_DTYPES)
@pytest.mark.parametrize("name", _WEIGHT_LOSSES)
def test_a_class_weight_is_read_at_the_inputs_dtype(name: str, weight: str, device: str) -> None:
    make = _lucid_make(device)
    if weight == "float32":
        assert math.isfinite(_weighted(F, name, make, weight)[0])
        return
    with pytest.raises(RuntimeError) as caught:
        _weighted(F, name, make, weight)
    message = str(caught.value)
    if weight == "bool":
        assert f"{name} does not accept a bool weight" in message and "cast to" in message
    else:
        assert f"{name}: expected weight of the input's dtype" in message
        assert f"lucid.{weight}" in message
    # The engine's dtype class is a ``TypeError`` as well, so a caller that caught one still does.
    assert isinstance(caught.value, TypeError)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("weight", _WEIGHT_DTYPES)
def test_class_probabilities_take_a_weight_of_any_dtype(weight: str, device: str) -> None:
    make = _lucid_make(device)
    got, _ = _weighted(F, "cross_entropy", make, weight, probabilities=True)
    want, _ = _weighted(F, "cross_entropy", make, "float32", probabilities=True)
    assert _close(got, want, "float32")


_WEIGHT_CELLS = [
    (name, weight, probabilities)
    for name in _WEIGHT_LOSSES
    for weight in (*_WEIGHT_DTYPES, "float64")
    for probabilities in (False, True)
    if name == "cross_entropy" or not probabilities
]


@pytest.mark.parity
@pytest.mark.parametrize(("name", "weight", "probabilities"), _WEIGHT_CELLS)
def test_a_class_weight_is_the_references(
    ref: ModuleType, name: str, weight: str, probabilities: bool
) -> None:
    def make_ref(values: object, d: str) -> object:
        return ref.tensor(values).to(getattr(ref, d))

    try:
        want = _weighted(ref.nn.functional, name, make_ref, weight, probabilities=probabilities)
    except RuntimeError:
        want = None
    try:
        got = _weighted(F, name, _lucid_make("cpu"), weight, probabilities=probabilities)
    except RuntimeError:
        got = None
    assert (got is None) == (want is None), (got, want)
    if got is not None and want is not None:
        assert got[1] == want[1]
        assert _close(got[0], want[0], "float32")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", ("bool", "int8", "int64", "float16", "float32"))
def test_pos_weight_is_a_number_not_a_mask(dtype: str, device: str) -> None:
    """``pos_weight - 1`` is a subtraction, which the reference refuses on a bool."""

    def run(*, widen: bool) -> lucid.Tensor:
        return F.binary_cross_entropy_with_logits(
            _data(_X, "float32", device),
            _data(_BIN, "float32", device),
            pos_weight=_data([1.0, 2.0, 1.0, 1.0], dtype, device, widen=widen),
        )

    if dtype == "bool":
        with pytest.raises(RuntimeError, match="does not accept a bool pos_weight") as caught:
            run(widen=False)
        assert "cast to" in str(caught.value)
        return
    assert _close(float(run(widen=False).item()), float(run(widen=True).item()), dtype)


_CTC_INTEGERS = ("int8", "int32", "int64")
_CTC_LENGTHS = ("input_lengths", "target_lengths")


def _ctc(functional: ModuleType, make: Make, **lengths: object) -> float:
    """``ctc_loss`` of ``functional`` with the lengths it is handed by keyword, the rest ordinary."""
    options: dict[str, object] = {
        "input_lengths": make([6, 6], "int64"),
        "target_lengths": make([2, 2], "int64"),
        **lengths,
    }
    out = functional.ctc_loss(make(_CTC, "float32"), make([[1, 2], [3, 4]], "int64"), **options)
    return float(out.item())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _CTC_INTEGERS)
@pytest.mark.parametrize("which", _CTC_LENGTHS)
def test_ctc_lengths_of_any_integer_width_are_taken(which: str, dtype: str, device: str) -> None:
    make = _lucid_make(device)
    values = [6, 6] if which == "input_lengths" else [2, 2]
    assert _close(_ctc(F, make, **{which: make(values, dtype)}), _ctc(F, make), "float32")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", ("bool", "float16", "float32"))
@pytest.mark.parametrize("which", _CTC_LENGTHS)
def test_ctc_lengths_must_be_integral(which: str, dtype: str, device: str) -> None:
    make = _lucid_make(device)
    values = [6, 6] if which == "input_lengths" else [2, 2]
    with pytest.raises(RuntimeError) as caught:
        _ctc(F, make, **{which: make(values, dtype)})
    message = str(caught.value)
    if dtype == "bool":
        assert f"ctc_loss does not accept a bool {which}" in message
        assert "cast to an integer dtype" in message
    else:
        assert f"ctc_loss: {which} must be integral, got lucid.{dtype}" in message
    assert isinstance(caught.value, TypeError)


_NOT_INTEGERS: dict[str, list[object]] = {
    "floats": [6.0, 6.0],
    "bools": [True, True],
    "strings": ["6", "6"],
}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("entries", list(_NOT_INTEGERS))
@pytest.mark.parametrize("which", _CTC_LENGTHS)
def test_a_ctc_length_list_holds_python_integers(which: str, entries: str, device: str) -> None:
    make = _lucid_make(device)
    with pytest.raises(TypeError, match=f"ctc_loss: {which} must hold integers"):
        _ctc(F, make, **{which: _NOT_INTEGERS[entries]})


@pytest.mark.parametrize("device", _DEVICES)
def test_ctc_length_sequences_of_integers_are_taken(device: str) -> None:
    make = _lucid_make(device)
    assert _ctc(F, make, input_lengths=(6, 6), target_lengths=[2, 2]) == _ctc(F, make)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ("bool", "int8", "int32", "int64", "float16", "float32"))
@pytest.mark.parametrize("which", _CTC_LENGTHS)
def test_ctc_length_dtypes_are_the_references(ref: ModuleType, which: str, dtype: str) -> None:
    values = [6, 6] if which == "input_lengths" else [2, 2]

    def make_ref(v: object, d: str) -> object:
        return ref.tensor(v).to(getattr(ref, d))

    def accepted(functional: ModuleType, make: Make) -> bool:
        try:
            _ctc(functional, make, **{which: make(values, dtype)})
        except RuntimeError:
            return False
        return True

    assert accepted(F, _lucid_make("cpu")) == accepted(ref.nn.functional, make_ref)


@pytest.mark.parity
@pytest.mark.parametrize("entries", list(_NOT_INTEGERS))
@pytest.mark.parametrize("which", _CTC_LENGTHS)
def test_ctc_length_list_is_the_references(ref: ModuleType, which: str, entries: str) -> None:
    def make_ref(v: object, d: str) -> object:
        return ref.tensor(v).to(getattr(ref, d))

    with pytest.raises(TypeError):
        _ctc(ref.nn.functional, make_ref, **{which: _NOT_INTEGERS[entries]})


_POISSON_DTYPES = ("bool", "int64", "float16", "float32")
_POISSON_CELLS = [
    (x, t, log_input, full)
    for x in _POISSON_DTYPES
    for t in _POISSON_DTYPES
    for log_input in (True, False)
    for full in (False, True)
]


def _poisson_refused(x: str, t: str, log_input: bool, full: bool) -> bool:
    """The reference subtracts a bool operand in these cells and refuses it: the product of a
    bool input and a bool target, a bool input that is not a log, and a bool target's Stirling term.
    """
    return (x == "bool" and (t == "bool" or not log_input)) or (t == "bool" and full)


def _poisson_dtype(x: str, t: str) -> str:
    """What the loss comes back as: ``exp`` / ``log`` of an input that is not floating gives the
    default floating dtype, which a float16 target does not narrow; a float16 input beside anything
    but float32 stays float16."""
    return "float16" if x == "float16" and t != "float32" else "float32"


def _poisson(
    x: str, t: str, log_input: bool, full: bool, device: str, *, widen: bool = False
) -> lucid.Tensor:
    return F.poisson_nll_loss(
        _data(_POS, x, device, widen=widen),
        _data(_POS, t, device, widen=widen),
        log_input=log_input,
        full=full,
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("x", "t", "log_input", "full"), _POISSON_CELLS)
def test_poisson_refuses_the_bool_operands_it_subtracts(
    x: str, t: str, log_input: bool, full: bool, device: str
) -> None:
    if _poisson_refused(x, t, log_input, full):
        with pytest.raises(RuntimeError, match="poisson_nll_loss does not accept") as caught:
            _poisson(x, t, log_input, full, device)
        assert "cast to a floating dtype" in str(caught.value)
        return
    got = _poisson(x, t, log_input, full, device)
    assert _name(got.dtype) == _poisson_dtype(x, t)
    want = _poisson(x, t, log_input, full, device, widen=True)
    assert _close(float(got.item()), float(want.item()), f"{x}{t}")


@pytest.mark.parity
@pytest.mark.parametrize(("x", "t", "log_input", "full"), _POISSON_CELLS)
def test_poisson_dtype_rules_are_the_references(
    ref: ModuleType, x: str, t: str, log_input: bool, full: bool
) -> None:
    try:
        want = ref.nn.functional.poisson_nll_loss(
            ref.tensor(_POS).to(getattr(ref, x)),
            ref.tensor(_POS).to(getattr(ref, t)),
            log_input=log_input,
            full=full,
        )
    except RuntimeError:
        want = None
    assert (want is None) == _poisson_refused(x, t, log_input, full)
    if want is not None:
        assert _name(want.dtype) == _poisson_dtype(x, t)
