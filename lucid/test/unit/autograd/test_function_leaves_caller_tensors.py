"""A custom ``Function`` leaves the caller's tensors as they were.

Settling ``forward``'s outputs changes the tensors it is applied to: the
node is registered on a differentiable output, and a marked one is
detached.  Both were applied to whatever ``forward`` returned, and an input
returned as-is *is* the caller's tensor:

* ``ctx.mark_non_differentiable(m)`` on an input ``m`` returned as-is
  turned the caller's ``m.requires_grad`` to ``False`` (LCD-543), whether
  ``m`` was passed positionally or by keyword.  ``apply`` now hands back
  the detached alias :meth:`Tensor.detach` makes — a new tensor over the
  same storage, as the reference hands back.
* an unmarked input returned as-is has the node registered on itself, and
  becomes a non-leaf with the ``Function``'s ``grad_fn`` — needing a
  gradient even when it had needed none; ``checkpoint`` of a segment that
  passes an input through does the same to the caller's leaf.  That case
  is pinned below (LCD-558): the reference hands back a view, and the
  views Lucid can make do not keep the gradients right — a detached alias
  is a gradient sink to a compiled step, and an in-place write to the
  input re-derives a plain view from the write, dropping the custom
  ``backward`` (``test_a_write_to_the_input_keeps_the_custom_gradient``).
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn.functional as F
from lucid.autograd import Function
from lucid.autograd.checkpoint import checkpoint

#: What ``forward`` returns beside ``x * 2`` — built from the input ``m``.
_KINDS: dict[str, Callable[[object], object]] = {
    "input as-is": lambda m: m,
    "view of the input": lambda m: m.view(-1),
    "fresh tensor": lambda m: m * 3,
}

#: Every case: what is returned, whether it is marked non-differentiable,
#: and whether the caller's ``m`` needs a gradient.
_CASES = [
    (kind, marked, needs_grad)
    for kind in _KINDS
    for marked in (True, False)
    for needs_grad in (True, False)
]


def _case_id(case: tuple[str, bool, bool]) -> str:
    kind, marked, needs_grad = case
    return (
        f"{kind}-{'marked' if marked else 'unmarked'}-"
        f"{'m needs grad' if needs_grad else 'm needs no grad'}"
    )


def _function(lib: ModuleType, kind: str, marked: bool) -> type:
    """A ``Function`` returning ``x * 2`` and something built from ``m``."""

    class Pair(lib.autograd.Function):
        @staticmethod
        def forward(ctx, x, m):
            out = _KINDS[kind](m)
            if marked:
                ctx.mark_non_differentiable(out)
            return x * 2, out

        @staticmethod
        def backward(ctx, grad_y, grad_out):
            del ctx
            return grad_y * 2, grad_out * 3

    return Pair


def _inputs(lib: ModuleType, needs_grad: bool, device: str) -> tuple[object, object]:
    x = lib.ones(4, requires_grad=True, device=device)
    m = lib.ones(4, requires_grad=needs_grad, device=device)
    return x, m


def _state(t: object) -> tuple[bool, bool, bool, int]:
    """What a caller can observe about one of its tensors."""
    return t.requires_grad, t.grad_fn is None, t.is_leaf, t._version


def _flags(out: object, m: object) -> tuple[bool, bool, bool, bool]:
    """What ``apply`` hands back for ``m``'s slot."""
    return out is m, out.requires_grad, out.grad_fn is None, out.is_leaf


def _grads(lib: ModuleType, case: tuple[str, bool, bool], device: str) -> tuple:
    """Gradients that reach ``x`` and ``m`` through every output that has a graph."""
    kind, marked, needs_grad = case
    x, m = _inputs(lib, needs_grad, device)
    y, out = _function(lib, kind, marked).apply(x, m)
    loss = y.sum() + out.sum() if out.requires_grad else y.sum()
    loss.backward()
    return (x.grad.tolist(), None if m.grad is None else m.grad.tolist())


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_the_callers_tensors_are_left_alone(case: tuple[str, bool, bool], device: str) -> None:
    kind, marked, needs_grad = case
    x, m = _inputs(lucid, needs_grad, device)
    x_before, m_before = _state(x), _state(m)
    x_impl, m_impl = x._impl, m._impl

    y, out = _function(lucid, kind, marked).apply(x, m)

    assert _state(x) == x_before and _state(m) == m_before
    assert x._impl is x_impl and m._impl is m_impl
    assert out is not m and out._impl is not m_impl


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_a_marked_output_carries_no_gradient(case: tuple[str, bool, bool], device: str) -> None:
    kind, marked, needs_grad = case
    x, m = _inputs(lucid, needs_grad, device)
    y, out = _function(lucid, kind, marked).apply(x, m)
    assert y.requires_grad and y.grad_fn is not None
    if marked:
        assert not out.requires_grad and out.grad_fn is None and out.is_leaf
    else:
        assert out.requires_grad and out.grad_fn is not None and not out.is_leaf


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_gradients_reach_the_inputs(case: tuple[str, bool, bool], device: str) -> None:
    kind, marked, needs_grad = case
    x_grad, m_grad = _grads(lucid, case, device)
    assert x_grad == [2.0] * 4
    if not needs_grad:
        assert m_grad is None
    else:
        # A marked output's gradient reaches ``backward`` as zeros.
        assert m_grad == ([0.0] * 4 if marked else [3.0] * 4)


#: The reference keeps ``requires_grad`` on a marked view of an input that
#: needs a gradient, because a view taken under its ``no_grad`` needs one;
#: in Lucid such a view needs none, and a marked output carries no gradient
#: (``test_a_marked_output_carries_no_gradient``), as the reference's own
#: documentation for the marker says.
_REFERENCE_NO_GRAD_VIEW = ("view of the input", True, True)


@pytest.mark.parity
@pytest.mark.parametrize("case", [c for c in _CASES if c != _REFERENCE_NO_GRAD_VIEW], ids=_case_id)
def test_the_outputs_match_the_reference(
    ref: ModuleType, case: tuple[str, bool, bool], device: str
) -> None:
    # Nothing here depends on where a tensor lives, so the reference runs
    # on its CPU against each Lucid device.
    kind, marked, needs_grad = case
    seen = {}
    for lib, dev in ((lucid, device), (ref, "cpu")):
        x, m = _inputs(lib, needs_grad, dev)
        y, out = _function(lib, kind, marked).apply(x, m)
        seen[lib] = (_flags(out, m), _flags(y, x), _state(x), _state(m))
    assert seen[lucid] == seen[ref]
    assert _grads(lucid, case, device) == _grads(ref, case, "cpu")


@pytest.mark.parity
def test_the_reference_divergence_is_its_no_grad_view(ref: ModuleType) -> None:
    # The one case left out above differs from the reference exactly as a
    # view taken under ``no_grad`` does.
    answers = {}
    for lib in (lucid, ref):
        m = lib.ones(4, requires_grad=True)
        with lib.no_grad():
            view = m.view(-1)
        _, out = _function(lib, "view of the input", True).apply(lib.ones(4, requires_grad=True), m)
        answers[lib] = (out.requires_grad, view.requires_grad)
    assert answers[lucid] == (False, False)
    assert answers[ref] == (True, True)


def _untracked(lib: ModuleType, mode: str, marked: bool):
    """Return ``m`` as-is from a call that records no node."""
    x = lib.ones(4, requires_grad=(mode == "under no_grad"))
    m = lib.ones(4, requires_grad=(mode == "under no_grad"))
    before = _state(m)
    fn = _function(lib, "input as-is", marked)
    if mode == "under no_grad":
        with lib.no_grad():
            _, out = fn.apply(x, m)
    else:
        _, out = fn.apply(x, m)
    return m, before, out


_UNTRACKED = [
    (mode, marked) for mode in ("under no_grad", "no input needs grad") for marked in (True, False)
]


def _untracked_id(case: tuple[str, bool]) -> str:
    mode, marked = case
    return f"{mode}-{'marked' if marked else 'unmarked'}"


@pytest.mark.parametrize("case", _UNTRACKED, ids=_untracked_id)
def test_an_untracked_call_hands_back_a_new_tensor(case: tuple[str, bool]) -> None:
    m, before, out = _untracked(lucid, *case)
    assert _state(m) == before
    assert out is not m and not out.requires_grad and out.grad_fn is None
    # Over the input's storage, as the alias the tracked call hands back.
    assert out.tolist() == m.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("case", _UNTRACKED, ids=_untracked_id)
def test_an_untracked_call_matches_the_reference(ref: ModuleType, case: tuple[str, bool]) -> None:
    answers = {}
    for lib in (lucid, ref):
        m, before, out = _untracked(lib, *case)
        answers[lib] = (_state(m) == before, _flags(out, m))
    assert answers[lucid] == answers[ref]


def _keyword_function(kind: str, marked: bool) -> type:
    """``_function``'s ``Function`` taking ``m`` by keyword, which gets no gradient."""

    class Keyword(Function):
        @staticmethod
        def forward(ctx, x, *, m):
            out = _KINDS[kind](m)
            if marked:
                ctx.mark_non_differentiable(out)
            return x * 2, out

        @staticmethod
        def backward(ctx, grad_y, grad_out):
            del ctx, grad_out
            return grad_y * 2

    return Keyword


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_a_keyword_tensor_is_left_alone(case: tuple[str, bool, bool], device: str) -> None:
    kind, marked, needs_grad = case
    x, m = _inputs(lucid, needs_grad, device)
    before, m_impl = _state(m), m._impl
    _, out = _keyword_function(kind, marked).apply(x, m=m)
    assert _state(m) == before and m._impl is m_impl
    assert out is not m and out._impl is not m_impl
    # Handed back as the same tensor passed positionally would be.
    x_pos, m_pos = _inputs(lucid, needs_grad, device)
    _, positional = _function(lucid, kind, marked).apply(x_pos, m_pos)
    assert _flags(out, m) == _flags(positional, m_pos)


#: A write through what ``apply`` handed back for an input returned as-is:
#: whether ``m`` needs a gradient, whether it was marked, and whether the
#: call was recorded or made under ``no_grad`` (the write is made after it,
#: with grad mode on).
_WRITES = [
    (needs_grad, marked, mode)
    for needs_grad in (True, False)
    for marked in (True, False)
    for mode in ("recorded", "under no_grad")
]


def _write_id(case: tuple[bool, bool, str]) -> str:
    needs_grad, marked, mode = case
    return (
        f"{'m needs grad' if needs_grad else 'm needs no grad'}-"
        f"{'marked' if marked else 'unmarked'}-{mode}"
    )


def _write_through(lib: ModuleType, case: tuple[bool, bool, str], device: str):
    """``out.add_(1)`` on what came back: refused, or what it did to ``m``."""
    needs_grad, marked, mode = case
    x, m = _inputs(lib, needs_grad, device)
    fn = _function(lib, "input as-is", marked)
    if mode == "under no_grad":
        with lib.no_grad():
            _, out = fn.apply(x, m)
    else:
        _, out = fn.apply(x, m)
    version = m._version
    try:
        out.add_(1)
    except RuntimeError:
        return "refused"
    # The write reaches ``m``'s storage; how far its count moves is not
    # compared — Lucid's in-place ops bump it twice.
    return m.tolist(), m._version > version, out.requires_grad, _state(m)[:3]


@pytest.mark.parametrize("case", [c for c in _WRITES if c[1]], ids=_write_id)
def test_a_write_through_a_marked_alias(case: tuple[bool, bool, str], device: str) -> None:
    needs_grad, _, _ = case
    values, moved, out_needs_grad, m_state = _write_through(lucid, case, device)
    # A detached alias: the write reaches the caller's storage, and the
    # caller's tensor keeps its place in autograd.
    assert values == [2.0] * 4 and moved and not out_needs_grad
    assert m_state == (needs_grad, True, True)


@pytest.mark.parity
@pytest.mark.parametrize("case", _WRITES, ids=_write_id)
def test_a_write_through_what_came_back_matches_the_reference(
    ref: ModuleType, case: tuple[bool, bool, str], device: str
) -> None:
    assert _write_through(lucid, case, device) == _write_through(ref, case, "cpu")


def _passthrough(lib: ModuleType, scale: float) -> type:
    """``forward`` returns its input as-is; ``backward`` scales the gradient."""

    class Scale(lib.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            del ctx
            return t

        @staticmethod
        def backward(ctx, grad):
            del ctx
            return grad * scale

    return Scale


def _write_to_the_input(lib: ModuleType, write: str, device: str):
    """Write the input in place after the call; the gradient that reaches ``w``."""
    w = lib.ones(3, requires_grad=True, device=device)
    if write == "add_ after a triple":
        h = w * 1
        out = _passthrough(lib, 3.0).apply(h)
        h.add_(0.0)
    else:
        h = w * 2
        out = _passthrough(lib, -1.0).apply(h)
        relu = F.relu if lib is lucid else lib.nn.functional.relu
        relu(h, inplace=True)
    try:
        out.sum().backward()
    except RuntimeError:
        return "refused"
    return w.grad.tolist()


#: What the custom ``backward`` gives, through ``h = w * 1`` and ``w * 2``.
_CUSTOM_GRADIENT = {"add_ after a triple": [3.0] * 3, "relu_ after a gradient reversal": [-2.0] * 3}


@pytest.mark.parametrize("write", sorted(_CUSTOM_GRADIENT))
def test_a_write_to_the_input_keeps_the_custom_gradient(write: str, device: str) -> None:
    # The reference refuses the write (the input is its output view's base);
    # Lucid, until LCD-558, keeps the custom backward — never the plain
    # view's gradient (``[1, 1, 1]`` / ``[2, 2, 2]``) a view alias gave.
    got = _write_to_the_input(lucid, write, device)
    assert got == "refused" or got == _CUSTOM_GRADIENT[write]


@pytest.mark.parity
@pytest.mark.parametrize("write", sorted(_CUSTOM_GRADIENT))
def test_a_write_to_the_input_is_refused_as_by_the_reference(ref: ModuleType, write: str) -> None:
    assert _write_to_the_input(ref, write, "cpu") == "refused"
    assert _write_to_the_input(lucid, write, "cpu") == "refused"


def _saving(lib: ModuleType) -> type:
    """Save the input and return it as-is."""

    class Saving(lib.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            ctx.save_for_backward(t)
            return t

        @staticmethod
        def backward(ctx, grad):
            (t,) = ctx.saved_tensors
            return grad * t

    return Saving


def _write_to_a_saved_input(lib: ModuleType, device: str) -> str:
    x = lib.ones(3, requires_grad=True, device=device)
    h = x * 1
    out = _saving(lib).apply(h)
    with lib.no_grad():
        out.add_(1)
    try:
        out.sum().backward()
    except RuntimeError:
        return "refused"
    return "differentiated"


def test_a_write_to_a_saved_input_is_caught(device: str) -> None:
    # ``backward`` sees the write instead of reading the new values as the
    # saved ones.
    assert _write_to_a_saved_input(lucid, device) == "refused"


@pytest.mark.parity
def test_a_write_to_a_saved_input_matches_the_reference(ref: ModuleType, device: str) -> None:
    assert _write_to_a_saved_input(lucid, device) == _write_to_a_saved_input(ref, "cpu")


class _Twice(Function):
    @staticmethod
    def forward(ctx, t):
        del ctx
        return t, t

    @staticmethod
    def backward(ctx, grad_a, grad_b):
        del ctx
        return grad_a + grad_b


def test_a_non_leaf_input_keeps_its_grad_fn(device: str) -> None:
    x = lucid.ones(3, requires_grad=True, device=device)
    h = x * 1
    producer = h.grad_fn
    out = _passthrough(lucid, 3.0).apply(h)
    assert h.grad_fn is producer and out is not h


def test_a_non_leaf_input_returned_as_is_gets_the_custom_gradient(device: str) -> None:
    x = lucid.ones(3, requires_grad=True, device=device)
    _passthrough(lucid, 3.0).apply(x * 1).sum().backward()
    assert x.grad.tolist() == [3.0] * 3


def test_an_input_returned_twice_is_two_tensors(device: str) -> None:
    x = lucid.ones(3, requires_grad=True, device=device)
    a, b = _Twice.apply(x)
    assert a is not b and a is not x and b is not x
    assert x.is_leaf and x.grad_fn is None


def test_checkpoint_passing_an_input_through_keeps_the_leaf(device: str) -> None:
    x = lucid.ones(3, requires_grad=True, device=device)
    w = lucid.ones(3, requires_grad=True, device=device)
    passed, _ = checkpoint(lambda a, b: (a, b * 2), x, w, use_reentrant=True)
    assert x.is_leaf and x.grad_fn is None and passed is not x


def test_checkpoint_passing_an_input_through_gets_its_gradient(device: str) -> None:
    x = lucid.ones(3, requires_grad=True, device=device)
    w = lucid.ones(3, requires_grad=True, device=device)
    passed, doubled = checkpoint(lambda a, b: (a, b * 2), x, w, use_reentrant=True)
    (passed.sum() * 3 + doubled.sum()).backward()
    assert x.grad.tolist() == [3.0] * 3
    assert w.grad.tolist() == [2.0] * 3
