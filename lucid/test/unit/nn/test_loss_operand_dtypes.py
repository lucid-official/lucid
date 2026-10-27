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
  of the same loss on float32 operands;
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


def _lucid_make(device: str, *, as_float32: bool = False) -> Make:
    """A ``make`` for Lucid tensors on ``device``.  With ``as_float32`` each
    data operand is built at its dtype and then cast to float32 — the same
    values, computed in float32 — and indices are left alone."""

    def make(values: object, dtype: str) -> object:
        d = lucid.bool_ if dtype == "bool" else getattr(lucid, dtype)
        t = lucid.tensor(values, device=device).to(dtype=d)
        return t if not as_float32 or _is_index(values) else t.float()

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
    """Refused with a ``TypeError`` naming the loss and the dtype, or
    computed as on float32 operands — the same verdict on each device."""
    make = _lucid_make(device)
    dtypes = _dtypes(name, sweep, dtype)
    if (name, dtype) in _NOT_IMPLEMENTED and sweep == "input":
        with pytest.raises(NotImplementedError):
            _run(name, make, dtypes)
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

#: Cells the reference takes and Lucid refuses.
_STRICTER: dict[tuple[str, str, str], str] = {
    ("poisson_nll_loss", "mixed", "bool"): (
        "a bool rate is refused whatever the target's dtype; the reference "
        "refuses it only beside a bool target"
    )
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
    if cell in _STRICTER:
        assert want is not None and got is None, _STRICTER[cell]
        return
    assert (got is None) == (want is None), (got, want)
    if got is None:
        return
    assert isinstance(got, lucid.Tensor)
    if cell not in _OTHER_DTYPE:
        assert _name(got.dtype) == _name(want.dtype)
    assert _close(float(got.item()), float(want.item()), dtype)
