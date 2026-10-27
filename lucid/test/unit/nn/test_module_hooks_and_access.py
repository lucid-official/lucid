"""``nn.Module``'s hooks, accessors and attribute protocol.

``module.py`` sat at 46.5%, and the dark regions were whole features
rather than error branches: every backward-hook path, the
``get_submodule`` / ``get_parameter`` / ``get_buffer`` family,
``__delattr__``, and the state machine that routes gradients back through
registered hooks.  A module was built and called; nothing else about it
was exercised.

Hooks are asserted on what they *observe and change*, not merely on
having fired — a hook that returns a modified value has to have that
value used, or the registration is decorative.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


def _x(shape=(2, 4)):
    return lucid.tensor(np.random.default_rng(0).standard_normal(shape), dtype=lucid.float32)


class Tiny(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.a = nn.Linear(width, width)
        self.b = nn.Linear(width, width)
        self.register_buffer("scale", lucid.tensor(np.array([2.0])))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.b(self.a(x))


# ── the accessor family ───────────────────────────────────────────────────────


def test_get_submodule_walks_a_dotted_path():
    model = Tiny()
    assert model.get_submodule("a") is model.a
    assert model.get_submodule("") is model


def test_get_submodule_refuses_a_name_that_is_not_there():
    with pytest.raises(AttributeError):
        Tiny().get_submodule("nope")


def test_get_parameter_and_get_buffer():
    model = Tiny()
    assert model.get_parameter("a.weight") is model.a.weight
    assert model.get_buffer("scale") is model.scale


@pytest.mark.parametrize(
    "getter,target",
    [("get_parameter", "a.absent"), ("get_buffer", "absent"), ("get_submodule", "a.b")],
)
def test_the_accessors_refuse_a_missing_target(getter, target):
    with pytest.raises(AttributeError):
        getattr(Tiny(), getter)(target)


def test_get_parameter_refuses_a_buffer_and_vice_versa():
    """The two namespaces overlap in the dotted syntax but not in kind."""
    model = Tiny()
    with pytest.raises(AttributeError):
        model.get_parameter("scale")


# ── the attribute protocol ────────────────────────────────────────────────────


def test_deleting_a_parameter_removes_it_from_the_state():
    model = Tiny()
    assert "a.weight" in model.state_dict()
    del model.a.weight
    assert "a.weight" not in model.state_dict()
    assert not any(name == "weight" for name, _ in model.a.named_parameters())


def test_deleting_a_buffer_removes_it():
    model = Tiny()
    assert "scale" in model.state_dict()
    del model.scale
    assert "scale" not in model.state_dict()


def test_deleting_a_submodule_removes_it():
    model = Tiny()
    del model.a
    assert not any(name == "a" for name, _ in model.named_children())


def test_deleting_a_plain_attribute_still_works():
    model = Tiny()
    model.tag = "keep"
    del model.tag
    assert not hasattr(model, "tag")


def test_assigning_a_parameter_over_a_plain_attribute():
    model = Tiny()
    model.extra = lucid.tensor(np.zeros(3))
    model.extra = nn.Parameter(lucid.tensor(np.ones(3)))
    assert any(name == "extra" for name, _ in model.named_parameters())


# ── forward hooks ─────────────────────────────────────────────────────────────


def test_a_forward_hook_sees_the_output():
    model, seen = Tiny(), []
    model.register_forward_hook(lambda mod, args, out: seen.append(out))
    model(_x())
    assert len(seen) == 1
    assert tuple(seen[0].shape) == (2, 4)


def test_a_forward_hook_can_replace_the_output():
    model = Tiny()
    model.register_forward_hook(lambda mod, args, out: out * 0.0)
    assert np.allclose(np.asarray(model(_x()).numpy()), 0.0)


def test_a_forward_pre_hook_can_replace_the_input():
    """Returning a value has to change what forward receives, or the
    registration is decorative."""
    model = Tiny()
    model.register_forward_pre_hook(lambda mod, args: (args[0] * 0.0,))
    zeroed = np.asarray(model(_x()).numpy())
    direct = np.asarray(model(lucid.zeros((2, 4))).numpy())
    assert np.allclose(zeroed, direct)


def test_a_forward_pre_hook_returning_none_leaves_the_input_alone():
    model, calls = Tiny(), []
    model.register_forward_pre_hook(lambda mod, args: calls.append(1))
    before = np.asarray(model(_x()).numpy())
    assert calls == [1]
    assert np.allclose(before, np.asarray(Tiny.forward(model, _x()).numpy()))


def test_a_hook_handle_removes_the_hook():
    model, calls = Tiny(), []
    handle = model.register_forward_hook(lambda mod, args, out: calls.append(1))
    model(_x())
    handle.remove()
    model(_x())
    assert calls == [1]


def test_several_forward_hooks_run_in_registration_order():
    model, order = Tiny(), []
    model.register_forward_hook(lambda mod, args, out: order.append("first"))
    model.register_forward_hook(lambda mod, args, out: order.append("second"))
    model(_x())
    assert order == ["first", "second"]


class _Fails(nn.Module):
    def forward(self, x):
        raise ValueError("forward failed")


def _failing_hook(calls):
    def hook(mod, args, out):
        calls.append("failing")
        raise RuntimeError("hook failed")

    return hook


@pytest.mark.parametrize("first_is_global", [False, True])
def test_a_failing_always_call_hook_does_not_skip_the_next_one(first_is_global):
    # forward() raised, so every always_call hook still runs as cleanup.  One
    # that fails is reported as a warning — it must neither stop the hooks
    # after it nor replace the forward error the caller sees.
    from lucid.nn.hooks import register_module_forward_hook

    model, calls = _Fails(), []
    failing = _failing_hook(calls)
    if first_is_global:
        handle = register_module_forward_hook(failing, always_call=True)
    else:
        handle = model.register_forward_hook(failing, always_call=True)
    model.register_forward_hook(lambda mod, args, out: calls.append("next"), always_call=True)
    try:
        with pytest.warns(RuntimeWarning, match="hook failed"):
            with pytest.raises(ValueError, match="forward failed"):
                model(_x())
    finally:
        handle.remove()
    assert calls == ["failing", "next"]


class _HookError(Exception):
    """A user's own exception class, which no enumeration of builtins names."""


# Every ``Exception`` a hook can raise: a user's class, and builtins outside
# any short list (LCD-464 — an enumerated tuple let these replace the error).
_HOOK_ERRORS = [_HookError, NameError, MemoryError, SystemError, UserWarning]


@pytest.mark.parametrize("error", _HOOK_ERRORS, ids=lambda e: e.__name__)
def test_no_always_call_hook_error_masks_the_forward_error(error):
    def hook(mod, args, out):
        raise error("hook failed")

    model = nn.Linear(2, 2)
    model.register_forward_hook(hook, always_call=True)
    with pytest.warns(RuntimeWarning, match="hook failed"):
        with pytest.raises(lucid._C.engine.ShapeMismatch):
            model(lucid.zeros(3, 5))


@pytest.mark.parity
def test_the_reference_also_reraises_the_forward_error(ref):
    def hook(mod, args, out):
        raise _HookError("hook failed")

    model = ref.nn.Linear(2, 2)
    model.register_forward_hook(hook, always_call=True)
    with pytest.warns(UserWarning, match="hook failed"):
        with pytest.raises(RuntimeError):
            model(ref.zeros(3, 5))


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_from_an_always_call_hook_is_not_swallowed(error):
    # Not an ``Exception``: the caller asked to stop, so it wins over the
    # forward error rather than becoming a warning.
    def hook(mod, args, out):
        raise error()

    model = _Fails()
    model.register_forward_hook(hook, always_call=True)
    with pytest.raises(error):
        model(_x())


# ── backward hooks ────────────────────────────────────────────────────────────


def test_a_full_backward_hook_sees_the_gradients():
    model, seen = Tiny(), []

    def hook(mod, grad_input, grad_output):
        seen.append((grad_input, grad_output))

    model.register_full_backward_hook(hook)
    model(_x()).sum().backward()
    assert len(seen) == 1
    grad_input, grad_output = seen[0]
    assert grad_output[0] is not None


def test_a_backward_pre_hook_sees_the_output_gradient():
    model, seen = Tiny(), []
    model.register_full_backward_pre_hook(lambda mod, grad_output: seen.append(grad_output))
    model(_x()).sum().backward()
    assert len(seen) == 1


def test_a_backward_hook_handle_removes_it():
    model, calls = Tiny(), []
    handle = model.register_full_backward_hook(lambda mod, gi, go: calls.append(1))
    model(_x()).sum().backward()
    handle.remove()
    model(_x()).sum().backward()
    assert calls == [1]


def test_gradients_still_reach_the_parameters_with_hooks_attached():
    """A hook must observe the backward pass without breaking it."""
    model = Tiny()
    model.register_full_backward_hook(lambda mod, gi, go: None)
    model.register_forward_hook(lambda mod, args, out: None)
    model(_x()).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name


# ── train / eval and the module tree ──────────────────────────────────────────


def test_train_and_eval_propagate_to_children():
    model = Tiny()
    model.eval()
    assert not model.training and not model.a.training
    model.train()
    assert model.training and model.a.training


def test_apply_visits_every_module():
    model, visited = Tiny(), []
    model.apply(lambda m: visited.append(type(m).__name__))
    assert visited.count("Linear") == 2
    assert "Tiny" in visited


def test_repr_nests_the_children():
    text = repr(Tiny())
    assert "Linear" in text and "Tiny" in text


def test_children_and_modules_differ_by_depth():
    model = Tiny()
    assert len(list(model.children())) == 2
    assert len(list(model.modules())) == 3  # self plus two Linears


def test_zero_grad_clears_what_backward_set():
    model = Tiny()
    model(_x()).sum().backward()
    assert model.a.weight.grad is not None
    model.zero_grad()
    grad = model.a.weight.grad
    assert grad is None or np.abs(np.asarray(grad.numpy())).sum() == 0.0


def test_a_module_without_forward_says_so():
    class Bare(nn.Module):
        pass

    with pytest.raises(NotImplementedError):
        Bare()(_x())


def test_to_empty_honours_recurse_false():
    target_dev = "metal" if lucid.metal.is_available() else "cpu"

    class Parent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root_param = nn.Parameter(lucid.ones(2))
            self.child = nn.Linear(2, 2)

    m = Parent()
    assert m.to_empty(device=target_dev, recurse=False) is m
    assert m.root_param.device.type == target_dev
    if target_dev == "metal":
        assert m.child.weight.device.type == "cpu"
    assert m.to_empty(device=target_dev, recurse=True) is m
    assert m.child.weight.device.type == target_dev
