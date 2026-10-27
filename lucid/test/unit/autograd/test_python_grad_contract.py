"""Every gradient Python hands back to the engine is checked against its slot.

A gradient slot is read as the tensor it stands for — that tensor's dtype,
device and shape.  A module backward hook's returned tuple went in unchecked
(CHA-226): ``nn.Linear(64, 64)`` with a full hook returning ``(ones(1),)`` left
``x.grad`` at 1.6e8 on CPU — a read past the end of a one-element buffer —
and raised MLX's ValueError on Metal; a float16 gradient handed to a non-leaf
had its bits taken for float32; a tuple of the wrong length was cut or padded
silently; and an entry that was not a tensor became ``None``.

The check has one owner, ``autograd/TensorHooks``: a tensor hook's result,
a module hook's tuple and ``Tensor.grad =`` all go through its converter and
its kind check.  This file holds every entry point to the same contract:

* a value of another shape (short or long), dtype or device is refused with
  ShapeMismatch / DtypeMismatch / DeviceMismatch;
* a value that is neither a tensor nor ``None`` is a TypeError;
* a module hook's tuple of the wrong length is a RuntimeError naming both
  counts;

and in each case nothing is applied: the gradient the entry point would have
written stays as it was, and a module's barrier starts the next pass empty.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import metal_available

_SHAPE = (2, 3)

# What each kind of wrong value is refused with.
_KIND_ERRORS: dict[str, type[BaseException]] = {
    "short": _C_engine.ShapeMismatch,
    "long": _C_engine.ShapeMismatch,
    "dtype": _C_engine.DtypeMismatch,
    "device": _C_engine.DeviceMismatch,
    "non-tensor": TypeError,
}
_CASES = [*_KIND_ERRORS, "count"]


def _other(device: str) -> str:
    return "cpu" if device == "metal" else "metal"


def _wrong(case: str, device: str) -> object:
    """One gradient for a ``_SHAPE`` float32 slot on ``device``, wrong as ``case`` says."""
    if case == "short":
        return lucid.ones(1, device=device)
    if case == "long":
        return lucid.ones(4, 3, device=device)
    if case == "dtype":
        return lucid.ones(*_SHAPE, device=device, dtype=lucid.float16)
    if case == "device":
        return lucid.ones(*_SHAPE, device=_other(device))
    assert case == "non-tensor"
    return 3


class _Twice(nn.Module):
    """No parameters, so a refused hook leaves no gradient anywhere."""

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x * 2.0


def _engine_hook(t: lucid.Tensor, hook: Callable[[lucid.Tensor], object]) -> None:
    """Install ``hook`` through the engine's tensor-hook entry point.

    The runner hands back whatever the hook returned, a tensor unwrapped, so
    the engine's own check is what the test sees.
    """

    def runner(grad: _C_engine.TensorImpl) -> object:
        result = hook(_wrap(grad))
        return _unwrap(result) if isinstance(result, lucid.Tensor) else result

    _C_engine._tensor_hook_runner(_unwrap(t), lambda: runner)


@dataclass
class _Armed:
    """A leaf whose gradient the entry point writes, and the call that fails."""

    x: lucid.Tensor
    trigger: Callable[[], None]
    grad_before: list[float] | None = None


def _leaf(device: str) -> lucid.Tensor:
    return lucid.ones(*_SHAPE, device=device, requires_grad=True)


def _arm_tensor_hook(device: str, case: str) -> _Armed:
    x = _leaf(device)
    y = x * 1.0
    if case == "count":
        _engine_hook(y, lambda g: (g, g))
    else:
        bad = _wrong(case, device)
        _engine_hook(y, lambda g: bad)
    return _Armed(x, lambda: (y * 2.0).sum().backward())


def _arm_leaf_hook(device: str, case: str) -> _Armed:
    x = _leaf(device)
    if case == "count":
        _engine_hook(x, lambda g: (g, g))
    else:
        bad = _wrong(case, device)
        _engine_hook(x, lambda g: bad)
    return _Armed(x, lambda: (x * 2.0).sum().backward())


def _arm_full_hook(device: str, case: str) -> _Armed:
    x = _leaf(device)
    m = _Twice()
    if case == "count":
        m.register_full_backward_hook(lambda mod, gi, go: (gi[0], gi[0]))
    else:
        bad = _wrong(case, device)
        m.register_full_backward_hook(lambda mod, gi, go: (bad,))
    # A non-leaf input: the replaced gradient goes through a kernel next.
    out = m(x * 1.0)
    return _Armed(x, lambda: out.sum().backward())


def _arm_pre_hook(device: str, case: str) -> _Armed:
    x = _leaf(device)
    m = _Twice()
    if case == "count":
        m.register_full_backward_pre_hook(lambda mod, go: (go[0], go[0]))
    else:
        bad = _wrong(case, device)
        m.register_full_backward_pre_hook(lambda mod, go: (bad,))
    out = m(x * 1.0)
    return _Armed(x, lambda: out.sum().backward())


def _arm_grad_assignment(device: str, case: str) -> _Armed:
    x = _leaf(device)
    x.grad = lucid.full(_SHAPE, 5.0, device=device)
    bad: object = (x.grad, x.grad) if case == "count" else _wrong(case, device)

    def assign() -> None:
        x.grad = bad  # type: ignore[assignment]

    return _Armed(x, assign, grad_before=[5.0] * 6)


_ENTRIES: dict[str, Callable[[str, str], _Armed]] = {
    "tensor-hook": _arm_tensor_hook,
    "leaf-hook": _arm_leaf_hook,
    "module-full-hook": _arm_full_hook,
    "module-pre-hook": _arm_pre_hook,
    "grad-assignment": _arm_grad_assignment,
}

# A tuple where one gradient belongs is not a tensor; a module hook's tuple
# of the wrong length is a count error, as the reference words it.
_COUNT_ERRORS: dict[str, tuple[type[BaseException], str]] = {
    "tensor-hook": (TypeError, "tensor or None"),
    "leaf-hook": (TypeError, "tensor or None"),
    "module-full-hook": (
        RuntimeError,
        "invalid number of grad_input, got 2, but expected 1",
    ),
    "module-pre-hook": (
        RuntimeError,
        "invalid number of grad_output, got 2, but expected 1",
    ),
    "grad-assignment": (TypeError, "Tensor"),
}


def _flat(t: lucid.Tensor | None) -> list[float] | None:
    return None if t is None else [float(v) for v in t.reshape(-1).tolist()]


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("entry", list(_ENTRIES))
def test_a_wrong_gradient_is_refused_and_nothing_is_applied(
    entry: str, case: str, device: str
) -> None:
    if case == "device" and not metal_available():
        pytest.skip("needs a second device")
    armed = _ENTRIES[entry](device, case)
    if case == "count":
        error, match = _COUNT_ERRORS[entry]
        with pytest.raises(error, match=match):
            armed.trigger()
    else:
        with pytest.raises(_KIND_ERRORS[case]):
            armed.trigger()
    assert _flat(armed.x.grad) == armed.grad_before


# ── the reported case ─────────────────────────────────────────────────────────


def test_the_reported_short_gradient(device: str) -> None:
    m = nn.Linear(64, 64).to(device)
    x = lucid.randn(2, 64, device=device, requires_grad=True)
    m.register_full_backward_hook(lambda mod, gi, go: (lucid.ones(1, device=device),))
    with pytest.raises(_C_engine.ShapeMismatch, match="grad_input\\[0\\]"):
        m(x).sum().backward()
    assert x.grad is None


def test_a_float16_gradient_for_a_non_leaf_is_refused(device: str) -> None:
    """Read as float32, its bits came back as ``[64.1, 64.1, 0, 0]``."""
    m = _Twice()
    m.register_full_backward_hook(lambda mod, gi, go: (gi[0].to(lucid.float16),))
    x = lucid.ones(4, device=device, requires_grad=True)
    with pytest.raises(_C_engine.DtypeMismatch):
        m(x * 1.0).sum().backward()
    assert x.grad is None


# ── module hooks: the rest of the tuple contract ─────────────────────────────


def test_a_refused_hook_is_the_last_one_to_run(device: str) -> None:
    """The next hook is never handed a tuple the engine refused."""
    seen: list[int] = []
    m = _Twice()
    m.register_full_backward_hook(lambda mod, gi, go: (gi[0], gi[0]))
    m.register_full_backward_hook(lambda mod, gi, go: seen.append(len(gi)))
    x = _leaf(device)
    with pytest.raises(RuntimeError, match="invalid number of grad_input"):
        m(x).sum().backward()
    assert seen == []


def test_chained_replacements_are_each_checked(device: str) -> None:
    """A good first result reaches the second hook; a bad second one is refused."""
    seen: list[list[float] | None] = []
    m = _Twice()
    m.register_full_backward_hook(lambda mod, gi, go: (gi[0] * 3.0,))

    def second(mod: object, gi: tuple[lucid.Tensor, ...], go: object) -> object:
        seen.append(_flat(gi[0]))
        return (gi[0].to(lucid.float16),)

    m.register_full_backward_hook(second)
    x = _leaf(device)
    with pytest.raises(_C_engine.DtypeMismatch):
        m(x).sum().backward()
    assert seen == [[6.0] * 6]
    assert x.grad is None


def test_a_bare_tensor_is_not_a_tuple(device: str) -> None:
    m = _Twice()
    m.register_full_backward_hook(lambda mod, gi, go: gi[0])
    x = _leaf(device)
    with pytest.raises(TypeError, match="tuple of grad_input"):
        m(x).sum().backward()
    assert x.grad is None


def test_a_list_is_a_tuple(device: str) -> None:
    m = _Twice()
    m.register_full_backward_hook(lambda mod, gi, go: [g * 0.5 for g in gi])
    x = _leaf(device)
    m(x).sum().backward()
    assert _flat(x.grad) == [1.0] * 6


def test_a_gradient_for_an_input_that_takes_none_is_refused(device: str) -> None:
    class Masked(nn.Module):
        def forward(self, x: lucid.Tensor, mask: lucid.Tensor) -> lucid.Tensor:
            return x * mask

    m = Masked()
    m.register_full_backward_hook(lambda mod, gi, go: (gi[0], lucid.ones(*_SHAPE, device=device)))
    x = _leaf(device)
    with pytest.raises(RuntimeError, match="grad_input\\[1\\].*where none flows"):
        m(x, lucid.ones(*_SHAPE, device=device)).sum().backward()
    assert x.grad is None


def test_a_module_whose_inputs_take_no_gradient_hands_its_hooks_none(
    device: str,
) -> None:
    """One ``None`` per positional argument, as the reference passes them —
    and a hook may hand them back, but not put a gradient in their place."""

    class Scaled(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(lucid.ones(*_SHAPE, device=device))

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return x * self.w

    seen: list[tuple[object, ...]] = []
    m = Scaled()
    m.register_full_backward_hook(lambda mod, gi, go: seen.append(gi) or gi)
    frozen = lucid.ones(*_SHAPE, device=device)
    m(frozen).sum().backward()
    assert seen == [(None,)]

    m2 = Scaled()
    m2.register_full_backward_hook(lambda mod, gi, go: (lucid.ones(*_SHAPE, device=device),))
    with pytest.raises(RuntimeError, match="where none flows"):
        m2(frozen).sum().backward()


def test_a_refused_pass_leaves_the_barrier_empty(device: str) -> None:
    """The failed pass's gradients do not reach the next one."""
    calls: list[int] = []
    refuse = [True]

    def hook(mod: object, gi: tuple[lucid.Tensor, ...], go: object) -> object:
        calls.append(1)
        return (gi[0].to(lucid.float16),) if refuse[0] else None

    m = _Twice()
    m.register_full_backward_hook(hook)
    x = _leaf(device)
    out = m(x * 1.0).sum()
    with pytest.raises(_C_engine.DtypeMismatch):
        out.backward(retain_graph=True)
    assert x.grad is None
    refuse[0] = False
    out.backward()
    assert calls == [1, 1]
    assert _flat(x.grad) == [2.0] * 6


def test_a_kept_gradient_is_not_added_into(device: str) -> None:
    """The engine adds into what flows on in place; a hook may keep what it
    was handed, so a CPU gradient leaves the barrier as a copy."""
    kept: list[lucid.Tensor] = []

    class Id(nn.Module):
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return x * 1.0

    m = Id()
    m.register_full_backward_hook(lambda mod, gi, go: kept.append(gi[0]))
    x = lucid.ones(3, device=device, requires_grad=True)
    w = lucid.ones(3, device=device, requires_grad=True)
    h = x * w
    ((h * 2.0).sum() + m(h).sum()).backward()
    assert _flat(kept[0]) == [1.0, 1.0, 1.0]
    assert _flat(x.grad) == [3.0, 3.0, 3.0]


def test_a_replacement_keeps_its_graph_under_create_graph(device: str) -> None:
    m = _Twice()
    m.register_full_backward_hook(lambda mod, gi, go: (gi[0] * 3.0,))
    x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    (g,) = lucid.autograd.grad((m(x * x)).sum(), x, create_graph=True)
    assert _flat(g) == [12.0, 24.0]  # 3 * 2 * 2x
    (h,) = lucid.autograd.grad(g.sum(), x)
    assert _flat(h) == [12.0, 12.0]


def test_a_wrong_dtype_is_refused_under_create_graph(device: str) -> None:
    m = _Twice()
    m.register_full_backward_pre_hook(lambda mod, go: (go[0].to(lucid.float16),))
    x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    with pytest.raises(_C_engine.DtypeMismatch):
        lucid.autograd.grad(m(x * x).sum(), x, create_graph=True)


# ── custom Function: the engine end of its backward ──────────────────────────


def test_a_backward_that_hands_the_engine_a_non_tensor_is_a_type_error() -> None:
    """``_python_node._validate`` holds a Function's gradients to its inputs;
    the engine still refuses a value that is no tensor at all, rather than
    taking it for ``None``."""
    x = lucid.ones(2, requires_grad=True)
    out = lucid.ones(2) * 1.0
    node = _C_engine._PythonBackwardNode()
    node.ctx = _C_engine.FunctionCtx()
    node.backward_fn = lambda ctx, *grads: [3]
    _C_engine._register_python_backward_node([_unwrap(out)], node, [_unwrap(x)])
    with pytest.raises(TypeError, match="backward result must be a tensor or None, got int"):
        out.sum().backward()
    assert x.grad is None


def test_save_for_backward_refuses_a_non_tensor() -> None:
    ctx = _C_engine.FunctionCtx()
    ctx.save_for_backward(_unwrap(lucid.ones(1)), None)
    with pytest.raises(TypeError, match="save_for_backward must be a tensor or None, got str"):
        ctx.save_for_backward("x")  # type: ignore[arg-type]
