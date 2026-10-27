"""A write into a tensor a custom ``Function`` saved is caught, as for a built-in op.

``ctx.save_for_backward`` kept the tensors and nothing else, so a write
between the forward and the backward went unnoticed and ``backward`` read
the values written after the forward, in silence, on both devices:

    out = Mul.apply(w, buf).sum(); buf.fill_(5.0); out.backward()  ->  w.grad == 5

A built-in op records each saved tensor's version and its backward refuses
a mismatch (``VersionMismatch``); the reference refuses this one too.  The
context now records the versions once ``forward`` has returned and checks
them when ``backward`` reads ``ctx.saved_tensors``, with the engine's
message.  An in-place op under grad mode on a tensor that already has a
graph is caught as well, now that it moves the version (near the bottom).
"""

from collections.abc import Callable

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.autograd import Function
from lucid.autograd.graph import allow_mutation_on_saved_tensors

# ``ctx.saved_tensors`` raises ``VersionMismatch`` inside ``backward``, and the
# engine hands a Python ``backward``'s exception to the caller as it was
# raised (test_custom_function_exception_type.py), so that is the type a
# ``backward()`` caller gets.
RAISED: type[Exception] = _C_engine.VersionMismatch


class _Mul(Function):
    """``w * buf``, saving both."""

    @staticmethod
    def forward(ctx, w, buf):
        ctx.save_for_backward(w, buf)
        return w * buf

    @staticmethod
    def backward(ctx, grad):
        w, buf = ctx.saved_tensors
        return grad * buf, grad * w


class _Exp(Function):
    """``exp(x)``, saving its own output."""

    @staticmethod
    def forward(ctx, x):
        y = x.exp()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return grad * y


# Every writer the tests run between the forward and the backward.
WRITES: dict[str, Callable[[lucid.Tensor, str], object]] = {
    "fill_": lambda t, d: t.fill_(5.0),
    "zero_": lambda t, d: t.zero_(),
    "copy_": lambda t, d: t.copy_(lucid.full((3,), 5.0, device=d)),
    "index_fill_": lambda t, d: t.index_fill_(0, lucid.tensor([0], device=d), 5.0),
    # Writes into ``t`` itself, as the reference does.  It used to give ``t``
    # a new impl and leave the saved one as it was, so backward ran on the
    # old values where the reference refuses (API-02).
    "setitem": lambda t, d: t.__setitem__(1, 7.0),
}


def _saved(device: str) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    """``w``, a buffer ``_Mul`` saved, and the loss built from them."""
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    buf = lucid.tensor([1.0, 1.0, 1.0], device=device)
    return w, buf, _Mul.apply(w, buf).sum()


# ── writes that are refused ──────────────────────────────────────────────────


@pytest.mark.parametrize("write", list(WRITES), ids=list(WRITES))
def test_a_write_into_a_saved_input_is_refused(write: str, device: str) -> None:
    w, buf, out = _saved(device)
    WRITES[write](buf, device)
    with pytest.raises(RAISED, match=r"VersionMismatch \(_Mul input 1\)"):
        out.backward()
    assert w.grad is None


def test_a_write_under_no_grad_is_refused(device: str) -> None:
    # ``no_grad`` says not to record the write; it still changes the values
    # backward would read.
    _, buf, out = _saved(device)
    with lucid.no_grad():
        buf.add_(1.0)
    with pytest.raises(RAISED, match=r"VersionMismatch \(_Mul input 1\)"):
        out.backward()


def test_a_write_into_a_saved_leaf_is_refused(device: str) -> None:
    # An optimiser-style update of a parameter before the backward that
    # needs its old value.
    w, _, out = _saved(device)
    with lucid.no_grad():
        w.add_(1.0)
    with pytest.raises(RAISED, match=r"VersionMismatch \(_Mul input 0\)"):
        out.backward()


def test_a_write_into_a_saved_output_is_refused(device: str) -> None:
    x = lucid.tensor([0.0, 1.0], requires_grad=True, device=device)
    y = _Exp.apply(x)
    with lucid.no_grad():
        y.mul_(2.0)
    # Seeded at ``y`` itself: an op consuming ``y`` records its version too
    # and would refuse first, before ``_Exp``'s own check ran.
    with pytest.raises(RAISED, match=r"VersionMismatch \(_Exp output 0\)"):
        y.backward(lucid.ones_like(y))


def test_a_write_into_the_base_of_a_saved_view_is_refused(device: str) -> None:
    # The view shares the base's buffer/family, and with it the base's count.
    w = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    base = lucid.ones(4, device=device)
    out = _Mul.apply(w, base[0:2]).sum()
    base.fill_(5.0)
    with pytest.raises(RAISED, match=r"VersionMismatch \(_Mul input 1\)"):
        out.backward()


def test_create_graph_reads_through_the_same_check(device: str) -> None:
    # ``grad(create_graph=True)`` runs ``backward`` on its own engine path.
    w, buf, out = _saved(device)
    buf.fill_(5.0)
    with pytest.raises(RAISED, match=r"VersionMismatch \(_Mul input 1\)"):
        lucid.autograd.grad(out, [w], create_graph=True)


# ── the error itself ─────────────────────────────────────────────────────────


def _captured(device: str):
    """The context of one ``Mul`` call, its ``buf``, and a tensor ``forward``
    made and saved — read directly, without the engine in between."""
    seen = []

    class Mul(Function):
        @staticmethod
        def forward(ctx, w, buf):
            tmp = buf * 2.0
            ctx.save_for_backward(w, buf, tmp)
            ctx.tmp = tmp
            seen.append(ctx)
            return w * buf

        @staticmethod
        def backward(ctx, grad):
            return grad, None

    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    buf = lucid.ones(3, device=device)
    Mul.apply(w, buf)
    return seen[0], buf, seen[0].tmp


def test_the_error_is_the_engines_and_worded_as_the_engine_words_it(device: str) -> None:
    ctx, buf, _ = _captured(device)
    buf.fill_(5.0)
    with pytest.raises(_C_engine.VersionMismatch) as custom:
        _ = ctx.saved_tensors

    # The engine's message for the same write after the built-in ``mul``.
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    buf2 = lucid.ones(3, device=device)
    out = (w * buf2).sum()
    buf2.fill_(5.0)
    with pytest.raises(_C_engine.VersionMismatch) as engine:
        out.backward()

    assert str(custom.value) == str(engine.value).replace("(mul ", "(Mul ")


def test_a_saved_intermediate_is_named_by_its_slot(device: str) -> None:
    ctx, _, tmp = _captured(device)
    tmp.fill_(0.0)
    with pytest.raises(_C_engine.VersionMismatch, match=r"\(Mul saved tensor 2\)"):
        _ = ctx.saved_tensors


# ── paths that are unaffected ────────────────────────────────────────────────


def test_no_write_gives_the_gradient(device: str) -> None:
    w, _, out = _saved(device)
    out.backward()
    assert w.grad is not None
    assert w.grad.tolist() == [1.0, 1.0, 1.0]


def test_a_write_into_a_tensor_nothing_saved_still_runs(device: str) -> None:
    w, _, out = _saved(device)
    unrelated = lucid.ones(3, device=device)
    unrelated.fill_(5.0)
    out.backward()
    assert w.grad is not None
    assert w.grad.tolist() == [1.0, 1.0, 1.0]


def test_a_write_inside_forward_is_what_backward_reads(device: str) -> None:
    # The versions are taken once ``forward`` has returned, so a ``forward``
    # that saves a tensor and then writes it is not refused for its own
    # write — the reference saves at the same point.
    class ScaleAfterSave(Function):
        @staticmethod
        def forward(ctx, w, buf):
            scale = buf * 2.0
            ctx.save_for_backward(scale)
            scale.add_(1.0)
            return w * scale

        @staticmethod
        def backward(ctx, grad):
            (scale,) = ctx.saved_tensors
            return grad * scale, None

    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    ScaleAfterSave.apply(w, lucid.ones(3, device=device)).sum().backward()
    assert w.grad is not None
    assert w.grad.tolist() == [3.0, 3.0, 3.0]


def test_a_backward_that_never_reads_the_saved_tensors_runs(device: str) -> None:
    # The check is where the values are read, as in the reference.
    class NoRead(Function):
        @staticmethod
        def forward(ctx, w, buf):
            ctx.save_for_backward(buf)
            return w * 2.0

        @staticmethod
        def backward(ctx, grad):
            return grad * 2.0, None

    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    buf = lucid.ones(3, device=device)
    out = NoRead.apply(w, buf).sum()
    buf.fill_(5.0)
    out.backward()
    assert w.grad is not None
    assert w.grad.tolist() == [2.0, 2.0, 2.0]


def test_a_second_backward_with_a_retained_graph_runs(device: str) -> None:
    w, _, out = _saved(device)
    out.backward(retain_graph=True)
    out.backward()
    assert w.grad is not None
    assert w.grad.tolist() == [2.0, 2.0, 2.0]


def test_the_opt_out_lets_backward_read_the_new_values(device: str) -> None:
    w, buf, out = _saved(device)
    with allow_mutation_on_saved_tensors():
        buf.fill_(5.0)
        out.backward()
    assert w.grad is not None
    assert w.grad.tolist() == [5.0, 5.0, 5.0]


def test_saved_tensors_read_inside_forward_are_the_saved_objects(device: str) -> None:
    class ReadInForward(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            (same,) = ctx.saved_tensors
            assert same is x
            return x * 2.0

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * 2.0 + x * 0.0

    x = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    ReadInForward.apply(x).sum().backward()
    assert x.grad is not None
    assert x.grad.tolist() == [2.0, 2.0]


# ── an in-place op under grad mode ───────────────────────────────────────────
#
# An in-place op on a tensor that already has a graph makes that tensor its
# output (the "adopt" path).  That path used to leave the version alone,
# because the op saved the pre-write state in a buffer of its own; a custom
# Function's context holds the tensor and read the new values.  The version
# moves there now (CHA-32), and the reference refuses all three.


def test_a_grad_mode_mul_into_a_saved_output_is_refused(device: str) -> None:
    x = lucid.tensor([0.0, 1.0], requires_grad=True, device=device)
    y = _Exp.apply(x)
    y.mul_(2.0)  # was: x.grad == 4 * exp(x), the right answer is 2 * exp(x)
    with pytest.raises(RAISED, match=r"VersionMismatch \(_Exp output 0\)"):
        y.sum().backward()


def test_a_grad_mode_exp_into_a_saved_output_is_refused(device: str) -> None:
    x = lucid.tensor([0.0, 1.0], requires_grad=True, device=device)
    y = _Exp.apply(x)
    y.exp_()  # was: backward read exp(exp(x)) where it saved exp(x)
    with pytest.raises(RAISED, match=r"VersionMismatch \(_Exp output 0\)"):
        y.sum().backward()


def test_a_grad_mode_relu_into_a_saved_input_is_refused(device: str) -> None:
    x = lucid.tensor([-1.0, 2.0], requires_grad=True, device=device)
    w = lucid.tensor([1.0, 1.0], requires_grad=True, device=device)
    h = x * 1.0
    y = _Mul.apply(w, h)
    h.relu_()  # was: w.grad == relu(x) == [0, 2], the right answer is [-1, 2]
    with pytest.raises(RAISED, match=r"VersionMismatch \(_Mul input 1\)"):
        y.sum().backward()


# ── against the reference ────────────────────────────────────────────────────


def _ref_mul(ref):
    class RefMul(ref.autograd.Function):
        @staticmethod
        def forward(ctx, w, buf):
            ctx.save_for_backward(w, buf)
            return w * buf

        @staticmethod
        def backward(ctx, grad):
            w, buf = ctx.saved_tensors
            return grad * buf, grad * w

    return RefMul


# What runs, under no_grad, between the forward and the backward.
SCENARIOS: dict[str, Callable[[object, object], object]] = {
    "nothing": lambda w, buf: None,
    "buf.fill_": lambda w, buf: buf.fill_(5.0),
    "buf.zero_": lambda w, buf: buf.zero_(),
    "buf.add_": lambda w, buf: buf.add_(1.0),
    "w.mul_": lambda w, buf: w.mul_(2.0),
    "other": lambda w, buf: (w * 0.0).fill_(1.0),
}


def _outcome(make, apply, no_grad, scenario: str):
    """The error message backward raised, or ``w.grad``, for one scenario."""
    w = make([1.0, 2.0, 3.0], True)
    buf = make([1.0, 1.0, 1.0], False)
    out = apply(w, buf).sum()
    with no_grad():
        SCENARIOS[scenario](w, buf)
    try:
        out.backward()
    except RuntimeError as e:
        return str(e)
    return w.grad.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=list(SCENARIOS))
def test_refuses_where_the_reference_refuses(scenario: str, device: str, ref) -> None:
    got = _outcome(
        lambda v, rg: lucid.tensor(v, requires_grad=rg, device=device),
        _Mul.apply,
        lucid.no_grad,
        scenario,
    )
    want = _outcome(
        lambda v, rg: ref.tensor(v, requires_grad=rg), _ref_mul(ref).apply, ref.no_grad, scenario
    )
    if isinstance(want, str):
        assert "modified by an inplace operation" in want
        assert isinstance(got, str) and "VersionMismatch (_Mul input" in got
    else:
        assert got == want
