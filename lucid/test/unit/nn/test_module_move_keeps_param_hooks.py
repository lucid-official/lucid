"""A parameter keeps its gradient hooks when the module moves it (LCD-962).

A hook registered with ``param.register_hook`` or
``param.register_post_accumulate_grad_hook`` lives in the engine, on the
parameter's ``TensorImpl``.  ``Module.to`` / ``.half()`` / ``.metal()`` give
the *same* ``Parameter`` object new data, and the engine impl behind it was
swapped for one over the new storage — the hooks stayed on the discarded impl,
so the parameter's gradient reached ``.grad`` unhooked.  The one owner is the
data swap (``lucid._dispatch._swap_data``, behind ``t.data = v`` and every
``Module._apply``): it now hands the runners to the impl that replaces the old
one, so the hook lists, their order and every handle already issued follow.

The table is every way a module's parameters get new data x what a hook must
still do afterwards.  The reference framework keeps a parameter's hooks across
``.to()`` / ``.half()`` / ``.data =`` and loses them across ``deepcopy`` and an
``assign=True`` load (new Parameter objects); both are pinned below.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
from lucid.autograd._hooks import _has_active_tensor_hooks
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._core.signature import model_has_active_hooks
from lucid.test._fixtures.devices import metal_available

Tensor = lucid.Tensor


def _nothing(m: nn.Module) -> object:
    return m


@dataclass(frozen=True)
class Move:
    """One way to give a module's parameters new data.

    ``prep`` runs before any hook is registered (it puts the module where the
    move starts); ``apply`` is the move under test.
    """

    apply: Callable[[nn.Module], object]
    prep: Callable[[nn.Module], object] = _nothing
    metal: bool = False


def _set_data(change: Callable[[Tensor], Tensor]) -> Callable[[nn.Module], object]:
    def apply(m: nn.Module) -> None:
        for p in m.parameters():
            p.data = change(p.data)

    return apply


_MOVES: dict[str, Move] = {
    "to(float64)": Move(lambda m: m.to(lucid.float64)),
    "double()": Move(lambda m: m.double()),
    "half()": Move(lambda m: m.half()),
    "bfloat16()": Move(lambda m: m.bfloat16()),
    "float() from float64": Move(lambda m: m.float(), prep=lambda m: m.double()),
    "type(float64)": Move(lambda m: m.type(lucid.float64)),
    "to(metal)": Move(lambda m: m.to("metal"), metal=True),
    "metal()": Move(lambda m: m.metal(), metal=True),
    "to(metal, float16)": Move(lambda m: m.to("metal", lucid.float16), metal=True),
    "to_empty(metal)": Move(lambda m: m.to_empty(device="metal"), metal=True),
    "to(cpu) from metal": Move(lambda m: m.to("cpu"), prep=lambda m: m.metal(), metal=True),
    "cpu() from metal": Move(lambda m: m.cpu(), prep=lambda m: m.metal(), metal=True),
    "half() on metal": Move(lambda m: m.half(), prep=lambda m: m.metal(), metal=True),
    "to(metal) then to(cpu)": Move(lambda m: m.to("metal").to("cpu"), metal=True),
    ".data = cast": Move(_set_data(lambda t: t.double())),
    ".data = moved": Move(_set_data(lambda t: t.to("metal")), metal=True),
}

_MOVE_PARAMS = [
    pytest.param(
        move,
        id=name,
        marks=pytest.mark.skipif(move.metal and not metal_available(), reason="needs Metal"),
    )
    for name, move in _MOVES.items()
]


def _linear(move: Move) -> nn.Linear:
    lucid.manual_seed(0)
    m = nn.Linear(3, 2)
    move.prep(m)
    return m


def _backward(m: nn.Linear) -> Tensor:
    """Zero the gradients, run a batch of two through ``m`` and return ``weight.grad``."""
    m.zero_grad()
    x = lucid.ones(2, 3, device=m.weight.device, dtype=m.weight.dtype)
    m(x).sum().backward()
    assert m.weight.grad is not None
    return m.weight.grad


def _values(t: Tensor) -> list[float]:
    return t.to("cpu").to(lucid.float32).reshape(-1).tolist()


# The gradient of ``sum(x @ W.T + b)`` w.r.t. every weight entry is the batch
# size: 2.
_PLAIN = 2.0


@pytest.mark.parametrize("move", _MOVE_PARAMS)
def test_a_hook_still_modifies_the_gradient_after_the_move(move: Move) -> None:
    m = _linear(move)
    m.weight.register_hook(lambda g: g * 3.0)
    move.apply(m)
    assert _has_active_tensor_hooks(m.weight)
    assert _values(_backward(m)) == [_PLAIN * 3.0] * 6
    # The bias had no hook and still has none.
    assert m.bias is not None and not _has_active_tensor_hooks(m.bias)
    assert m.bias.grad is not None and _values(m.bias.grad) == [2.0, 2.0]


@pytest.mark.parametrize("move", _MOVE_PARAMS)
def test_the_handle_issued_before_the_move_removes_the_hook_after_it(move: Move) -> None:
    m = _linear(move)
    handle = m.weight.register_hook(lambda g: g * 0.0)
    move.apply(m)
    assert _values(_backward(m)) == [0.0] * 6
    handle.remove()
    assert not _has_active_tensor_hooks(m.weight)
    assert _values(_backward(m)) == [_PLAIN] * 6
    handle.remove()  # still idempotent


@pytest.mark.parametrize("move", _MOVE_PARAMS)
def test_two_hooks_keep_their_registration_order(move: Move) -> None:
    m = _linear(move)
    m.weight.register_hook(lambda g: g * 2.0)
    m.weight.register_hook(lambda g: g + 1.0)
    move.apply(m)
    # (2 * 2) + 1, not (2 + 1) * 2.
    assert _values(_backward(m)) == [5.0] * 6


@pytest.mark.parametrize("move", _MOVE_PARAMS)
def test_a_hook_removed_before_the_move_stays_removed(move: Move) -> None:
    m = _linear(move)
    gone = m.weight.register_hook(lambda g: g * 0.0)
    kept = m.weight.register_hook(lambda g: g * 5.0)
    gone.remove()
    move.apply(m)
    assert _values(_backward(m)) == [_PLAIN * 5.0] * 6
    kept.remove()
    assert not _has_active_tensor_hooks(m.weight)
    assert _values(_backward(m)) == [_PLAIN] * 6


@pytest.mark.parametrize("move", _MOVE_PARAMS)
def test_every_hook_removed_before_the_move_leaves_no_hook(move: Move) -> None:
    m = _linear(move)
    m.weight.register_hook(lambda g: g * 0.0).remove()
    move.apply(m)
    assert not _has_active_tensor_hooks(m.weight)
    assert _values(_backward(m)) == [_PLAIN] * 6


@pytest.mark.parametrize("move", _MOVE_PARAMS)
def test_a_hook_registered_after_the_move_joins_the_carried_one(move: Move) -> None:
    m = _linear(move)
    m.weight.register_hook(lambda g: g * 2.0)
    move.apply(m)
    after = m.weight.register_hook(lambda g: g + 1.0)
    assert _values(_backward(m)) == [5.0] * 6
    after.remove()
    assert _values(_backward(m)) == [4.0] * 6


@pytest.mark.parametrize("move", _MOVE_PARAMS)
def test_a_post_accumulate_hook_follows_the_move(move: Move) -> None:
    m = _linear(move)
    seen: list[Tensor] = []
    handle = m.weight.register_post_accumulate_grad_hook(seen.append)
    move.apply(m)
    _backward(m)
    assert len(seen) == 1 and seen[0] is m.weight
    handle.remove()
    _backward(m)
    assert len(seen) == 1


@pytest.mark.parametrize("move", _MOVE_PARAMS)
def test_a_frozen_parameter_keeps_its_hook_and_stays_frozen(move: Move) -> None:
    m = _linear(move)
    m.weight.register_hook(lambda g: g * 3.0)
    m.weight.register_post_accumulate_grad_hook(lambda p: None)
    m.weight.requires_grad_(False)
    move.apply(m)
    assert not m.weight.requires_grad
    x = lucid.ones(2, 3, device=m.weight.device, dtype=m.weight.dtype)
    m(x).sum().backward()
    assert m.weight.grad is None
    m.weight.requires_grad_(True)
    assert _values(_backward(m)) == [_PLAIN * 3.0] * 6


@pytest.mark.parametrize("move", _MOVE_PARAMS)
def test_the_parameter_is_the_same_object_and_keeps_its_gradient_state(move: Move) -> None:
    m = _linear(move)
    weight = m.weight
    weight.register_hook(lambda g: g * 3.0)
    move.apply(m)
    assert m.weight is weight and isinstance(weight, nn.Parameter) and weight.requires_grad
    assert _values(_backward(m)) == [_PLAIN * 3.0] * 6
    # A move after a backward carries the gradient and the hook both.
    move.apply(m)
    assert _values(_backward(m)) == [_PLAIN * 3.0] * 6


def test_a_chain_of_moves_keeps_the_hook() -> None:
    m = nn.Linear(3, 2)
    m.weight.register_hook(lambda g: g * 3.0)
    for step in (lambda: m.double(), lambda: m.half(), lambda: m.float(), lambda: m.bfloat16()):
        step()
        assert _values(_backward(m)) == [_PLAIN * 3.0] * 6


@pytest.mark.skipif(not metal_available(), reason="needs Metal")
def test_a_round_trip_through_metal_keeps_the_hook() -> None:
    m = nn.Linear(3, 2)
    m.weight.register_hook(lambda g: g * 3.0)
    for target in ("metal", "cpu", "metal", "cpu"):
        m.to(target)
        assert _values(_backward(m)) == [_PLAIN * 3.0] * 6


def test_hooks_on_every_parameter_of_a_tree_follow() -> None:
    m = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    seen: list[str] = []
    m[0].weight.register_hook(lambda g: (seen.append("first"), g)[1])
    m[1].weight.register_hook(lambda g: (seen.append("second"), g)[1])
    m[1].bias.register_hook(lambda g: (seen.append("bias"), g)[1])
    m.double()
    m(lucid.ones(2, 3, dtype=lucid.float64)).sum().backward()
    assert sorted(seen) == ["bias", "first", "second"]


def test_an_unhooked_module_gets_no_hook_from_a_move() -> None:
    m = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    m.double().half()
    assert not model_has_active_hooks(m)
    assert all(not _has_active_tensor_hooks(p) for p in m.parameters())


def test_buffers_move_and_hold_their_values() -> None:
    bn = nn.BatchNorm1d(3)
    bn.weight.register_hook(lambda g: g * 0.0)
    bn(lucid.randn(4, 3))
    mean, var = _values(bn.running_mean), _values(bn.running_var)
    count = bn.num_batches_tracked
    assert count is not None
    bn.double()
    assert bn.running_mean.dtype == lucid.float64 and bn.running_var.dtype == lucid.float64
    assert _values(bn.running_mean) == pytest.approx(mean)
    assert _values(bn.running_var) == pytest.approx(var)
    assert bn.num_batches_tracked is not None and bn.num_batches_tracked.dtype == count.dtype
    assert not _has_active_tensor_hooks(bn.running_mean)
    bn(lucid.randn(4, 3, dtype=lucid.float64)).sum().backward()
    assert bn.weight.grad is not None and _values(bn.weight.grad) == [0.0] * 3


@pytest.mark.skipif(not metal_available(), reason="needs Metal")
def test_a_module_with_an_integer_parameter_moves_without_touching_its_hooks() -> None:
    class Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(lucid.ones(3))
            self.index = nn.Parameter(lucid.arange(3), requires_grad=False)

    m = Holder()
    m.weight.register_hook(lambda g: g * 3.0)
    index_dtype = m.index.dtype
    m.metal()
    assert m.index.dtype == index_dtype
    assert m.weight.device.type == "metal" and m.index.device.type == "metal"
    (m.weight * lucid.ones(3, device="metal")).sum().backward()
    assert m.weight.grad is not None and _values(m.weight.grad) == [3.0] * 3


def test_an_optimizer_built_before_the_move_still_steps_the_moved_parameter() -> None:
    m = nn.Linear(3, 2)
    m.weight.register_hook(lambda g: g * 3.0)
    opt = lucid.optim.SGD(m.parameters(), lr=1.0)
    m.double()
    before = _values(m.weight)
    _backward(m)
    opt.step()
    # The hooked gradient (2 * 3) is what the step applies.
    assert _values(m.weight) == pytest.approx([b - 6.0 for b in before])


def test_load_state_dict_copy_keeps_the_hook_and_assign_replaces_the_parameter() -> None:
    donor = nn.Linear(3, 2)
    m = nn.Linear(3, 2)
    weight = m.weight
    weight.register_hook(lambda g: g * 3.0)
    m.load_state_dict(donor.state_dict())
    assert m.weight is weight
    assert _values(_backward(m)) == [_PLAIN * 3.0] * 6
    m.load_state_dict(donor.double().state_dict())  # copied into float32 in place
    assert m.weight is weight and m.weight.dtype == lucid.float32
    assert _values(_backward(m)) == [_PLAIN * 3.0] * 6
    # ``assign=True`` puts the loaded tensors in as new Parameters, as the
    # reference does, so the old object's hook is not on them.
    m.load_state_dict(donor.state_dict(), assign=True)
    assert m.weight is not weight
    assert not _has_active_tensor_hooks(m.weight)


def test_deepcopy_does_not_copy_a_hook_and_leaves_the_original_hooked() -> None:
    import copy

    m = nn.Linear(3, 2)
    m.weight.register_hook(lambda g: g * 3.0)
    twin = copy.deepcopy(m)
    assert not _has_active_tensor_hooks(twin.weight)
    assert _values(_backward(twin)) == [_PLAIN] * 6
    assert _values(_backward(m)) == [_PLAIN * 3.0] * 6


class _Doubling(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * 2.0

    def right_inverse(self, y: Tensor) -> Tensor:
        return y / 2.0


def test_a_parametrization_keeps_the_hook_on_the_original() -> None:
    """``right_inverse`` puts new data in the original through ``.data =``."""
    from lucid.nn.utils import parametrize

    m = nn.Linear(3, 2)
    m.weight.register_hook(lambda g: g * 0.0)
    parametrize.register_parametrization(m, "weight", _Doubling())
    m(lucid.ones(2, 3)).sum().backward()
    original = m.parametrizations.weight.original
    assert original.grad is not None and _values(original.grad) == [0.0] * 6
    original.grad = None
    m.weight = lucid.ones(2, 3)
    m(lucid.ones(2, 3)).sum().backward()
    assert original.grad is not None and _values(original.grad) == [0.0] * 6


@pytest.mark.skipif(not metal_available(), reason="needs Metal")
def test_a_moved_hooked_parameter_still_sends_a_compiled_call_to_eager() -> None:
    """The compile side reads the same engine state: a hook that moved is a hook."""
    m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    m[0].weight.register_hook(lambda g: g * 0.0)
    m.to("metal")
    assert model_has_active_hooks(m)
    opt = lucid.optim.SGD(m.parameters(), lr=0.1)
    step = lucid.compile.fused_step(m, lambda y: (y * y).sum(), opt)
    before = _values(m[0].weight)
    with pytest.warns(CompileFallbackWarning, match="gradient hook"):
        step(lucid.randn(4, 8, device="metal"))
    assert _values(m[0].weight) == before  # the hook zeroed the gradient, eagerly
    assert _values(m[2].weight) != [0.0] * 32
    assert step.cache_info()["entries"] == 0

    compiled = lucid.compile(m)
    out = compiled(lucid.randn(4, 8, device="metal"))
    for p in m.parameters():
        p.grad = None
    out.sum().backward()
    assert m[0].weight.grad is not None and _values(m[0].weight.grad) == [0.0] * 64
