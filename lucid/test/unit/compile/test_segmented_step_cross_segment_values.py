"""A value that crosses segments outside their arguments is not frozen (LCD-902).

A segmented step traces each segment on its own and runs executables after.
A value a later segment reads from a stash an earlier one wrote — a dict, a
list, a module attribute, a closure cell — is a constant of that executable,
and the write that stored it ran only while the earlier segment was traced.
Compiled, call 1 read call 0's value and no gradient reached the segment that
made it.  The step now refuses: it runs eagerly, and equals eager.

Every way to carry a value across a segment boundary is a row of one table,
run through a segmented step and an eager twin over three steps with the
batch changing each time, so a frozen value shows.  The controls — a skip that
stays inside one segment, a constant read from two segments, a residual
module — must stay compiled.
"""

import types
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.compile import make_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

_WIDTH = 4


class _Fn(nn.Module):
    """A parameter-free stage built from one function."""

    def __init__(self, fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self._fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self._fn(x)


def _linear() -> nn.Linear:
    return nn.Linear(_WIDTH, _WIDTH)


def _stage(writes: Callable[[lucid.Tensor], object], reads: Callable[[lucid.Tensor], lucid.Tensor]):
    """Linear, a stage that stores, two Linears, a stage that reads, Linear:
    the writer and the reader are in different segments for 2 and for 3."""

    def store(x: lucid.Tensor) -> lucid.Tensor:
        writes(x)
        return x

    return [_linear(), _Fn(store), _linear(), _linear(), _Fn(reads), _linear()]


def _dict_stash() -> list[nn.Module]:
    box: dict[str, lucid.Tensor] = {}
    return _stage(lambda x: box.__setitem__("v", x), lambda x: x + box["v"])


def _list_stash() -> list[nn.Module]:
    store: list[lucid.Tensor] = []

    def put(x: lucid.Tensor) -> None:
        store[:] = [x]

    return _stage(put, lambda x: x + store[0])


def _closure_cell() -> list[nn.Module]:
    cell = lucid.zeros(5, _WIDTH, device=COMPILE_DEVICE)

    def put(x: lucid.Tensor) -> None:
        nonlocal cell
        cell = x

    def get(x: lucid.Tensor) -> lucid.Tensor:
        return x + cell

    return _stage(put, get)


def _attribute() -> list[nn.Module]:
    holder = types.SimpleNamespace()
    return _stage(lambda x: setattr(holder, "kept", x), lambda x: x + holder.kept)


def _module_attribute() -> list[nn.Module]:
    class Keeper(nn.Module):
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            self.kept = x
            return x

    class Reader(nn.Module):
        def __init__(self, keeper: Keeper) -> None:
            super().__init__()
            object.__setattr__(self, "keeper", keeper)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return x + self.keeper.kept

    keeper = Keeper()
    return [_linear(), keeper, _linear(), _linear(), Reader(keeper), _linear()]


def _non_differentiable() -> list[nn.Module]:
    box: dict[str, lucid.Tensor] = {}
    return _stage(
        lambda x: box.__setitem__("m", (x > 0.0).to(x.dtype)), lambda x: x * box["m"] + 1.0
    )


def _detached() -> list[nn.Module]:
    box: dict[str, lucid.Tensor] = {}
    return _stage(lambda x: box.__setitem__("v", x.detach()), lambda x: x * box["v"])


def _the_batch() -> list[nn.Module]:
    box: dict[str, lucid.Tensor] = {}

    def keep(x: lucid.Tensor) -> lucid.Tensor:
        box["v"] = x
        return x

    return [_Fn(keep), _linear(), _linear(), _linear(), _Fn(lambda x: x + box["v"]), _linear()]


def _a_boundary_input() -> list[nn.Module]:
    """The stage stores its own input, which is the previous segment's output."""
    box: dict[str, lucid.Tensor] = {}

    def keep(x: lucid.Tensor) -> lucid.Tensor:
        box["v"] = x
        return x * 1.0

    return [_linear(), _linear(), _Fn(keep), _linear(), _linear(), _Fn(lambda x: x + box["v"])]


def _two_readers() -> list[nn.Module]:
    box: dict[str, lucid.Tensor] = {}

    def keep(x: lucid.Tensor) -> lucid.Tensor:
        box["v"] = x
        return x

    return [
        _linear(),
        _Fn(keep),
        _linear(),
        _Fn(lambda x: x + box["v"]),
        _linear(),
        _Fn(lambda x: x * box["v"]),
    ]


# Crossing a boundary, whichever way: each must run eagerly and equal eager.
CROSSING = {
    "dict": _dict_stash,
    "list": _list_stash,
    "closure_cell": _closure_cell,
    "namespace_attribute": _attribute,
    "module_attribute": _module_attribute,
    "non_differentiable": _non_differentiable,
    "detached": _detached,
    "the_batch": _the_batch,
    "a_boundary_input": _a_boundary_input,
    "two_readers": _two_readers,
}


def _same_segment() -> list[nn.Module]:
    """The stash and its reader are neighbours: one segment for ``segments=2``."""
    box: dict[str, lucid.Tensor] = {}

    def keep(x: lucid.Tensor) -> lucid.Tensor:
        box["v"] = x
        return x

    return [_linear(), _Fn(keep), _Fn(lambda x: x + box["v"]), _linear(), _linear()]


def _constant_in_two_segments() -> list[nn.Module]:
    """A tensor made before the call, read by a stage of each segment."""
    scale = lucid.tensor([0.5, 1.0, 1.5, 2.0], device=COMPILE_DEVICE)
    return [_linear(), _Fn(lambda x: x * scale), _linear(), _linear(), _Fn(lambda x: x * scale)]


def _residual_module() -> list[nn.Module]:
    class Residual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a, self.b = _linear(), _linear()

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return x + self.b(self.a(x))

    return [_linear(), Residual(), Residual(), _linear()]


def _plain() -> list[nn.Module]:
    return [_linear(), nn.ReLU(), _linear(), nn.Tanh(), _linear(), _linear()]


# Nothing here crosses a boundary: every row stays compiled, at the segment
# counts that keep it so (the skip's two stages share a segment for 2 only).
ORDINARY = {
    "skip_inside_one_segment": (_same_segment, [2]),
    "constant_in_two_segments": (_constant_in_two_segments, [2, 3]),
    "residual_modules": (_residual_module, [2, 3]),
    "plain": (_plain, [2, 3]),
}


def _build(make: Callable[[], list[nn.Module]]) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(*make()).to(COMPILE_DEVICE)


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    return (y * y).mean()


def _step_against_twin(
    make: Callable[[], list[nn.Module]], segments: int
) -> tuple[Callable[..., lucid.Tensor], list[tuple[float, float, float]]]:
    """Three steps, a new batch each: ``(the step, [(loss, eager loss, worst
    gradient difference)])``."""
    compiled, twin = _build(make), _build(make)
    step = make_step(compiled, _loss, segments=segments)
    lucid.manual_seed(1)
    rows = []
    for i in range(3):
        x = metal_tensor(5, _WIDTH) * (1 + i)
        for p in (*compiled.parameters(), *twin.parameters()):
            p.grad = None
        got = step(x)
        got.backward()
        want = _loss(twin(x))
        want.backward()
        worst = max(
            float(np.abs(a.grad.numpy() - b.grad.numpy()).max())
            for a, b in zip(compiled.parameters(), twin.parameters(), strict=True)
        )
        rows.append((float(got.item()), float(want.item()), worst))
    return step, rows


@pytest.mark.parametrize("segments", [2, 3])
@pytest.mark.parametrize("name", CROSSING)
def test_a_value_crossing_segments_runs_eagerly_and_equals_eager(name: str, segments: int) -> None:
    step, rows = _step_against_twin(CROSSING[name], segments)
    for i, (loss, want, worst) in enumerate(rows):
        assert loss == pytest.approx(want, rel=1e-4), f"{name} step {i}: loss"
        assert worst < 1e-4, f"{name} step {i}: gradient differs by {worst}"
    assert step.eager_only["flag"], f"{name}: a frozen value must refuse the compile"


@pytest.mark.parametrize(
    "name, segments", [(name, k) for name, (_, counts) in ORDINARY.items() for k in counts]
)
def test_a_model_that_crosses_nothing_stays_compiled_and_equals_eager(
    name: str, segments: int
) -> None:
    step, rows = _step_against_twin(ORDINARY[name][0], segments)
    for i, (loss, want, worst) in enumerate(rows):
        assert loss == pytest.approx(want, rel=1e-4), f"{name} step {i}: loss"
        assert worst < 1e-4, f"{name} step {i}: gradient differs by {worst}"
    assert not step.eager_only["flag"], f"{name}: nothing crosses, the step must stay compiled"
    assert any(seg_cache for seg_cache in step.cache)


def test_the_refusal_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    monkeypatch.setattr("lucid.compile._core.fallback._NOTICED", set())
    with pytest.warns(UserWarning, match="segment 2 reads a value segment 0 computed"):
        _step_against_twin(_dict_stash, 3)


def test_call_values_name_a_held_value_to_every_segment_but_its_maker() -> None:
    """What a segment made, or was handed, may be read again by that segment
    only; the ledger needs no tracer once it has recorded one."""
    from lucid._dispatch import _unwrap
    from lucid.compile import _tracing
    from lucid.compile._core.call_values import CallValues

    lin = _linear().to(COMPILE_DEVICE)
    x = metal_tensor(3, _WIDTH)
    kept: dict[str, lucid.Tensor] = {}
    values = CallValues()
    with _tracing() as maker:
        kept["v"] = lin(x)
        lin(x) + 1.0  # made and dropped: nothing holds it to be read again
    values.made_by(0, maker)
    del maker
    values.hand(2, (_unwrap(x),))

    with _tracing() as reads_the_value:
        kept["v"] + 1.0
    assert values.foreign_read(0, reads_the_value) is None, "its maker may read it again"
    assert values.foreign_read(1, reads_the_value) == 0, "any other segment may not"

    with _tracing() as reads_the_input:
        x + 1.0
    assert values.foreign_read(2, reads_the_input) is None
    assert values.foreign_read(3, reads_the_input) == 2
