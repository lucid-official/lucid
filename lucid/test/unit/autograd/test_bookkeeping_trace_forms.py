"""A compile recording leaves out the eager graph's bookkeeping (LCD-570a).

Four internal ``Function``s — ``_Snapshot``, ``_Overwritten``,
``_TruncQuotient``, ``_RoundDecimals`` — and ``_take``'s version bump exist
only so an eager backward does not refuse an in-place write.  A compile
cannot differentiate through their nodes, so in grad mode a recording takes
their plain ops instead (``Function._trace_form``): the ops a recording
under ``no_grad`` takes, plus one graph-carrying copy (``contiguous``) per
operand the node reads, so the plain ops save nothing the traced code may
write — an eager backward run inside the recording still sees what eager
sees.  No ``detach`` (a gradient sink), no ``_make_view`` (no emitter), and
no call noted as a custom backward.  Values and ``requires_grad`` stay
eager's.

Only a class defined in Lucid is trusted with these markers; one set on a
user's class is ignored.
"""

import difflib
import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._ops.composite._rounding import _RoundDecimals, _TruncQuotient
from lucid._ops.composite.indexing import _Snapshot
from lucid._tensor._indexing import _take
from lucid._tensor.tensor import Tensor
from lucid.autograd import Function
from lucid.autograd._overwritten import _Overwritten
from lucid.autograd.checkpoint import checkpoint
from lucid.autograd.function import (
    FunctionMeta,
    _is_exact_derivative,
    _lucid_defined,
    _trace_form_of,
)
from lucid.compile._core.custom_backward import _APPLIED, CustomBackwardInTrace
from lucid.compile._core.fallback import CompileFallbackWarning

type _Case = Callable[[nn.Linear, Tensor], tuple[Tensor, Tensor]]


def _trunc(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    q = lucid.div(lin(x), 0.3, rounding_mode="trunc")
    return q * 2.0, q


def _round(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    r = lucid.round(lin(x), decimals=1)
    return r * 2.0, r


def _fill(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    h.fill_(2.0)
    return h * 2.0, h


def _zero(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    h.zero_()
    return h * 2.0 + lin(x), h


def _fill_view(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    v = h[0]
    v.fill_(2.0)
    return h * 2.0, v


def _setitem_whole(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    h[...] = lin(x) * 3.0
    return h * 2.0, h


def _setitem_partial(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    h[0] = lin(x)[1]
    return h * 2.0, h


def _setitem_view(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    v = h[1:]
    v[0] = lin(x)[0]
    return h * 2.0, v


def _index_fill(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    h.index_fill_(0, lucid.tensor([1], device=x.device.type), 3.0)
    return h * 2.0, h


def _index_put(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    h.index_put_((lucid.tensor([1], device=x.device.type),), lin(x)[0])
    return h * 2.0, h


def _relu_inplace(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    F.relu(h, inplace=True)
    return h * 2.0, h


def _relu_inplace_view(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    v = h[1:]
    F.relu(v, inplace=True)
    return h * 2.0, v


def _relu_inplace_with_view(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    v = h[1:]
    F.relu(h, inplace=True)
    return v * 2.0, h


_CASES: dict[str, _Case] = {
    "trunc_div": _trunc,
    "round_decimals": _round,
    "fill_": _fill,
    "zero_": _zero,
    "fill_view": _fill_view,
    "setitem_whole": _setitem_whole,
    "setitem_partial": _setitem_partial,
    "setitem_view": _setitem_view,
    "index_fill_": _index_fill,
    "index_put_": _index_put,
    "relu_inplace": _relu_inplace,
    "relu_inplace_view": _relu_inplace_view,
    "relu_inplace_with_view": _relu_inplace_with_view,
}

#: Ops the ``no_grad`` recording has that the grad-mode one lacks, and not
#: for bookkeeping: the ``no_grad`` branch writes the scatter's result into
#: the destination's own buffer (``assign_from``, or ``assign_inplace`` once
#: the scatter's flattening view of it, kept alive by the tracer, aliases
#: it), which on Metal records a ``reshape`` of the view result.  Grad mode
#: scatters into a copy and rebinds the destination to the result
#: (``_take``), recording nothing.
_PLAIN_ONLY: dict[tuple[str, str], list[str]] = {
    ("index_put_", "metal"): ["reshape"],
    ("setitem_partial", "metal"): ["reshape"],
}

#: Ops only the grad-mode recording has, and not for bookkeeping either: a
#: whole write of a value with a graph makes the value a buffer of its own
#: before ``t`` takes it (``__setitem__``), a copy the ``no_grad`` branch
#: makes into ``t``'s buffer without recording it.
_RECORDED_ONLY: dict[str, list[str]] = {"setitem_whole": ["contiguous"]}

#: How many operands each bookkeeping trace form reads through a copy: the
#: tensors its node's plain op would otherwise save.
_OPERANDS_READ: dict[type[Function], int] = {
    _Snapshot: 1,
    _RoundDecimals: 1,
    _TruncQuotient: 2,
    _Overwritten: 0,
}


def _differences(plain: list[str], recorded: list[str]) -> tuple[list[str], list[str]]:
    """The ops only ``recorded`` has, and the ops only ``plain`` has."""
    matcher = difflib.SequenceMatcher(a=plain, b=recorded, autojunk=False)
    extra: list[str] = []
    missing: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            missing += plain[i1:i2]
            extra += recorded[j1:j2]
    return extra, missing


def _setup(device: str) -> tuple[nn.Linear, Tensor]:
    lucid.manual_seed(0)
    lin = nn.Linear(4, 4).to(device)
    return lin, lucid.randn(3, 4, device=device)


def _record(
    case: _Case, lin: nn.Linear, x: Tensor, *, grad: bool
) -> tuple[list[str], int, tuple[Tensor, Tensor]]:
    mode = lucid.enable_grad() if grad else lucid.no_grad()
    with mode, lucid.compile._tracing(watch=(x,), module=lin) as tracer:
        result = case(lin, x)
    names = [str(op.name) for op in tracer.graph.ops]
    return names, len(_APPLIED.get(tracer, [])), result


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_grad_mode_recording_takes_the_no_grad_ops(
    monkeypatch: pytest.MonkeyPatch, device: str, name: str
) -> None:
    # Each bookkeeping trace form reads graph-carrying copies of the operands
    # its node reads — one ``contiguous`` each that ``no_grad`` lacks.
    copies = 0

    def counting(klass: type[Function]) -> Callable[..., object]:
        form = _trace_form_of(klass)
        assert form is not None

        def counted(*args: object) -> object:
            nonlocal copies
            copies += _OPERANDS_READ[klass]
            return form(*args)

        return counted

    lin, x = _setup(device)
    plain, _, _ = _record(_CASES[name], lin, x, grad=False)
    for klass in _OPERANDS_READ:
        monkeypatch.setattr(klass, "_trace_form", staticmethod(counting(klass)))
    recorded, noted, _ = _record(_CASES[name], lin, x, grad=True)
    assert noted == 0, "a bookkeeping node was noted as a custom backward"
    assert "detach" not in recorded and "_make_view" not in recorded
    extra, missing = _differences(plain, recorded)
    assert sorted(extra) == ["contiguous"] * copies + _RECORDED_ONLY.get(name, [])
    assert missing == _PLAIN_ONLY.get((name, device), [])


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_recording_sees_what_eager_sees(device: str, name: str) -> None:
    case = _CASES[name]
    lin, x = _setup(device)
    with lucid.enable_grad():
        want_out, want_t = case(lin, x)
    _, _, (got_out, got_t) = _record(case, lin, x, grad=True)
    assert got_out.tolist() == want_out.tolist()
    assert got_t.tolist() == want_t.tolist()
    assert (got_out.requires_grad, got_t.requires_grad) == (
        want_out.requires_grad,
        want_t.requires_grad,
    )
    assert (got_t.grad_fn is None) == (want_t.grad_fn is None)


def test_take_bumps_the_version_but_records_no_view(device: str) -> None:
    t = lucid.zeros(3, device=device) * 1.0
    old = t._impl
    before = old.version
    _take(t, (lucid.ones(3, device=device) * 1.0)._impl)
    assert old.version == before + 1
    t = lucid.zeros(3, device=device) * 1.0
    old = t._impl
    before = old.version
    with lucid.compile._tracing(watch=(t,)) as tracer:
        _take(t, (t + 1.0)._impl)
    assert [str(op.name) for op in tracer.graph.ops] == ["full", "add"]
    assert old.version == before + 1


# ── an eager backward inside a recording still runs ────────────────────────
#
# A recorded call may differentiate something itself — a force field
# ``-grad(E(x))``, a gradient penalty, a ``jvp``.  The node lets a write
# follow its read; the plain ops a recording takes in its place must too, or
# the inner backward refuses that write (``VersionMismatch``) before the
# compile can classify the call and run it eagerly.


def _activation(name: str) -> _Case:
    fn = getattr(F, name)

    def case(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
        h = lin(x) * 1.0
        fn(h, inplace=True)
        return h * 2.0, h

    return case


def _trunc_then_write(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    q = lucid.div(h, 0.3, rounding_mode="trunc")
    h.mul_(2.0)
    return q + h, h


def _trunc_divisor_written(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    b = x * 1.0 + 3.0
    q = lucid.div(h, b, rounding_mode="trunc")
    b.mul_(2.0)
    return q + h, b


def _round_then_write(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    r = lucid.round(h, decimals=1)
    h.mul_(2.0)
    return r + h, h


def _fill_then_write(lin: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    h = lin(x) * 1.0
    h.fill_(2.0)
    h.mul_(3.0)
    return h * x, h


_INNER_CASES: dict[str, _Case] = {
    **_CASES,
    **{
        f"{name}_inplace": _activation(name)
        for name in ("relu", "leaky_relu", "hardtanh", "elu", "silu")
    },
    "trunc_then_write": _trunc_then_write,
    "trunc_divisor_written": _trunc_divisor_written,
    "round_then_write": _round_then_write,
    "fill_then_write": _fill_then_write,
}


def _inner_grad(case: _Case, lin: nn.Linear, x: Tensor) -> list[object]:
    with lucid.enable_grad():
        out, _ = case(lin, x)
        (g,) = lucid.autograd.grad(out.sum() + (x * x).sum(), (x,))
    return g.tolist()


@pytest.mark.parametrize("name", sorted(_INNER_CASES))
def test_an_inner_backward_runs_in_a_recording(device: str, name: str) -> None:
    case = _INNER_CASES[name]
    lin, x = _setup(device)
    x.requires_grad_(True)
    want = _inner_grad(case, lin, x)
    with lucid.compile._tracing(watch=(x,), module=lin):
        got = _inner_grad(case, lin, x)
    assert got == want


class _ForceField(nn.Module):
    """``-dE/dx`` of an energy whose hidden layer is written in place."""

    def __init__(self, write: str) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 1)
        self._case = _INNER_CASES[write]

    def forward(self, x: Tensor) -> Tensor:
        with lucid.enable_grad():
            xx = x.detach().requires_grad_(True)
            out, _ = self._case(self.a, xx)
            (g,) = lucid.autograd.grad(self.b(out).sum(), (xx,))
        return -g


_PUBLIC_WRITES = ("relu_inplace", "silu_inplace", "trunc_then_write", "fill_")


def _is_classified_refusal(error: RuntimeError) -> bool:
    """A refusal the compile names, not an inner backward's ``VersionMismatch``."""
    if isinstance(error, _C_engine.VersionMismatch):
        return False
    return "backward pass ran inside" in str(error) or isinstance(
        error, CustomBackwardInTrace
    )


@pytest.mark.parametrize("write", _PUBLIC_WRITES)
def test_a_compiled_force_field_matches_eager(device: str, write: str) -> None:
    lucid.manual_seed(0)
    model = _ForceField(write).to(device).eval()
    x = lucid.randn(3, 4, device=device)
    want = model(x).tolist()
    compiled = lucid.compile(model)
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        assert compiled(x).tolist() == want


@pytest.mark.parametrize("write", _PUBLIC_WRITES)
def test_a_gradient_penalty_step_is_refused_by_name(device: str, write: str) -> None:
    lucid.manual_seed(0)
    lin = nn.Linear(4, 4).to(device)
    case = _INNER_CASES[write]
    x = lucid.randn(3, 4, device=device).requires_grad_(True)

    def penalised(out: Tensor) -> Tensor:
        (g,) = lucid.autograd.grad(out.sum(), (x,), create_graph=True)
        return out.sum() + (g * g).sum()

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = lin

        def forward(self, t: Tensor) -> Tensor:
            return case(self.lin, t)[0]

    try:
        lucid.compile.compiled_step(_Model(), x, penalised, use_grad_mode=True)
    except RuntimeError as error:
        assert _is_classified_refusal(error), repr(error)


# ── outside a recording nothing changes ────────────────────────────────────


def _eager_overwrite(device: str, kind: str) -> tuple[object, ...]:
    w = lucid.linspace(-1.0, 1.0, 6, device=device).reshape(2, 3).requires_grad_(True)
    h = w * 1.5
    if kind == "fill_view":
        t = h[0]
        before = t._impl.version
        t.fill_(2.0)
    else:
        t = h
        before = t._impl.version
        if kind == "zero_":
            t.zero_()
        else:
            t.fill_(2.0)
    seen = (
        t.tolist(),
        h.tolist(),
        t.requires_grad,
        t.grad_fn is not None,
        t.is_leaf,
        t._impl.version - before,
    )
    (h * 3.0 + w).sum().backward()
    assert w.grad is not None
    return (*seen, w.grad.tolist())


_ROW0 = [2.0, 2.0, 2.0]
_ROW1 = [0.30000001192092896, 0.9000000357627869, 1.5]


@pytest.mark.parametrize(
    ("kind", "want"),
    [
        # Values, the values of the tensor written, requires_grad, grad_fn,
        # is_leaf, version steps and w's gradient — what eager gave before
        # ``_Overwritten`` stopped detaching its output.
        ("fill_", ([_ROW0, _ROW0], [_ROW0, _ROW0], True, True, False, 1)),
        ("zero_", ([[0.0] * 3] * 2, [[0.0] * 3] * 2, True, True, False, 1)),
        ("fill_view", (_ROW0, [_ROW0, _ROW1], True, True, False, 4)),
    ],
)
def test_an_eager_overwrite_is_unchanged(
    device: str, kind: str, want: tuple[object, ...]
) -> None:
    got = _eager_overwrite(device, kind)
    assert got[:-1] == want
    grad = [[1.0] * 3, [5.5] * 3] if kind == "fill_view" else [[1.0] * 3] * 2
    assert got[-1] == grad


# ── only Lucid's own classes are trusted ───────────────────────────────────


class _Claims(Function):
    """A user ``Function`` claiming both markers, which must be ignored."""

    _trace_form = staticmethod(lambda x: x)
    _exact_derivative = True

    @staticmethod
    def forward(ctx: object, x: Tensor) -> Tensor:
        del ctx
        return x * 2.0

    @staticmethod
    def backward(ctx: object, grad: Tensor) -> Tensor:
        del ctx
        return grad * 2.0


@pytest.mark.parametrize(
    ("module", "trusted"),
    [
        ("lucid", True),
        ("lucid.autograd.checkpoint", True),
        ("lucid.test", False),
        ("lucid.test.unit.autograd.x", False),
        ("lucidity", False),
        ("mypkg.lucid", False),
    ],
)
def test_the_trust_rule(module: str, trusted: bool) -> None:
    klass = type("_Probe", (), {"__module__": module})
    assert _lucid_defined(klass) is trusted


def test_a_users_markers_are_ignored() -> None:
    assert _trace_form_of(_Claims) is None
    assert not _is_exact_derivative(_Claims)
    x = lucid.ones(3, requires_grad=True)
    with lucid.enable_grad(), lucid.compile._tracing(watch=(x,)) as tracer:
        y = _Claims.apply(x)
    assert isinstance(y, Tensor)
    assert y.tolist() == [2.0, 2.0, 2.0]
    assert len(_APPLIED.get(tracer, [])) == 1, "the call was not noted"
    assert y.grad_fn is not None


def test_lucids_bookkeeping_classes_are_trusted() -> None:
    for klass in _OPERANDS_READ:
        assert _trace_form_of(klass) is not None, klass.__name__
        assert not _is_exact_derivative(klass), klass.__name__


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_the_checkpoint_function_is_an_exact_derivative(
    monkeypatch: pytest.MonkeyPatch, use_reentrant: bool
) -> None:
    # ``checkpoint`` builds its class per call: catch it as it is made.
    made: list[type] = []
    init = FunctionMeta.__init__

    def record(
        cls: type, name: str, bases: tuple[type, ...], dct: dict[str, object]
    ) -> None:
        init(cls, name, bases, dct)
        made.append(cls)

    monkeypatch.setattr(FunctionMeta, "__init__", record)
    x = lucid.ones(2, requires_grad=True)
    checkpoint(lambda t: t * 3.0, x, use_reentrant=use_reentrant)
    (klass,) = (k for k in made if k.__name__ == "_CheckpointFn")
    assert issubclass(klass, Function)
    assert _is_exact_derivative(klass)
    assert _trace_form_of(klass) is None
