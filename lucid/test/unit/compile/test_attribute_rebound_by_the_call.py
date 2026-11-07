"""A tensor attribute the call rebinds, read by the calls after it (LCD-903).

A trace runs the forward once, so the Python that rebinds an attribute
(``self.h = y.detach()``) runs once too, at trace time.  The executable it
becomes is fed the tensor the attribute held then and never looks again, and a
replay does not rebind anything.  Both halves went wrong without a word:

* a state read and then rebound (an EMA kept as a plain attribute, a recurrent
  state, the previous step's feature) answered from the tensor it held at trace
  time on every call after the second: call 3 of a ``make_step`` read -2.19815
  where eager read 2.92519;
* a state the call first stores (``h`` starting as ``None``) was keyed as
  ``None``, and the executable for that key never stores it, so a model that
  starts a sequence again reads the branch of its first call for good.

The owner is ``held_writes``, which every entry that compiles a call asks once
the trace is over (:class:`~lucid.compile._core.rebound_slots.SlotWatch`): the
call that rebinds such an attribute stands as the call and its signature runs
eagerly.  ``compiled_step`` traces every call afresh and reads the attributes as
they are, so it needs no verdict.

Every case runs four calls against a twin that runs eagerly and compares the
loss after each.  What a rebinding is not (writing into the old tensor, keeping
a tensor nobody reads, an attribute rebuilt before it is read) must still
compile.
"""

from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.compile._entry.function import compiled_step
from lucid.compile._entry.fused_step import fused_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.utils.cache import StaticCache


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

Tensor = lucid.Tensor
WIDTH = 4
CALLS = 4
TOLERANCE = 3e-4


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield


class _Stateful(nn.Module):
    """``combine`` is the state the case keeps between calls."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: Tensor) -> Tensor:
        return self.combine(self.linear(x), x)

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError


class _Recurrent(_Stateful):
    """A recurrent state, ``None`` until the first call stores it (the card's case)."""

    def __init__(self) -> None:
        super().__init__()
        self.h: Tensor | None = None

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        out = y if self.h is None else y + self.h
        self.h = y.detach()
        return out


class _RecurrentGraph(_Stateful):
    """The state keeps its graph: only a call that never differentiates may."""

    def __init__(self) -> None:
        super().__init__()
        self.h: Tensor | None = None

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        out = y if self.h is None else y + self.h
        self.h = y
        return out


class _EmaAttribute(_Stateful):
    """An EMA held as a plain attribute, there from the start, rebound each call."""

    def __init__(self) -> None:
        super().__init__()
        self.ema = lucid.zeros(WIDTH, device=COMPILE_DEVICE)

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        out = y + self.ema
        self.ema = 0.9 * self.ema + 0.1 * y.detach().mean(0)
        return out


class _PreviousInput(_Stateful):
    """The previous call's argument, kept as it came."""

    def __init__(self) -> None:
        super().__init__()
        self.prev: Tensor | None = None

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        out = y if self.prev is None else y - self.prev
        self.prev = x
        return out


class _CacheDict(_Stateful):
    """A dict of tensors the module keeps: read, then stored again."""

    def __init__(self) -> None:
        super().__init__()
        self.cache: dict[str, Tensor] = {}

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        out = y + self.cache["h"] if "h" in self.cache else y
        self.cache["h"] = y.detach()
        return out


class _CacheList(_Stateful):
    """A history list: the last item is read, and a new one appended."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[Tensor] = []

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        out = y + self.history[-1] if self.history else y
        self.history.append(y.detach())
        return out


class _ClearsIt(_Stateful):
    """Reads the state and drops it."""

    def __init__(self) -> None:
        super().__init__()
        self.h: Tensor | None = lucid.ones(WIDTH, device=COMPILE_DEVICE)

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        out = y if self.h is None else y + self.h
        self.h = None
        return out


class _CopiesIntoTheOld(_Stateful):
    """Writing into the tensor the attribute holds: a write the entry carries."""

    def __init__(self) -> None:
        super().__init__()
        self.h = lucid.zeros(WIDTH, device=COMPILE_DEVICE)

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        out = y + self.h
        self.h.copy_(y.detach().mean(0))
        return out


class _EmaInPlace(_Stateful):
    def __init__(self) -> None:
        super().__init__()
        self.ema = lucid.zeros(WIDTH, device=COMPILE_DEVICE)

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        out = y + self.ema
        self.ema.mul_(0.9).add_(0.1 * y.detach().mean(0))
        return out


class _KeepsForLater(_Stateful):
    """Stores a tensor nothing in the call ever reads."""

    def __init__(self) -> None:
        super().__init__()
        self.last: Tensor | None = None

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        self.last = y.detach()
        return y * 2.0


class _BuiltBeforeItIsRead(_Stateful):
    """Rebinds an attribute to a tensor made from the input, then reads that one."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = lucid.ones(WIDTH, device=COMPILE_DEVICE)

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        self.gate = x.detach().mean(0) * 0.5 + 1.0
        return y * self.gate


class _ReadsAnAttribute(_Stateful):
    """An attribute read and never rebound by the call."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = lucid.full((WIDTH,), 1.5, device=COMPILE_DEVICE)

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        return y * self.scale


class _LazyConstant(_Stateful):
    """A constant built on the first call and kept, as a mask cache is."""

    def __init__(self) -> None:
        super().__init__()
        self.mask: Tensor | None = None

    def combine(self, y: Tensor, x: Tensor) -> Tensor:
        if self.mask is None:
            self.mask = lucid.ones(WIDTH, device=COMPILE_DEVICE) * 0.5
        return y * self.mask


#: ``(model, whether its state may carry a graph)``.  A state that keeps the graph
#: of the call that made it cannot be differentiated through a second time.
REBINDING: dict[str, type[_Stateful]] = {
    "recurrent": _Recurrent,
    "ema_attribute": _EmaAttribute,
    "previous_input": _PreviousInput,
    "cache_dict": _CacheDict,
    "cache_list": _CacheList,
    "clears_it": _ClearsIt,
}

#: Nothing here is a rebinding a later call reads, so each must still compile.
NOT_REBINDING: dict[str, type[_Stateful]] = {
    "copies_into_the_old": _CopiesIntoTheOld,
    "ema_in_place": _EmaInPlace,
    "keeps_for_later": _KeepsForLater,
    "built_before_it_is_read": _BuiltBeforeItIsRead,
    "reads_an_attribute": _ReadsAnAttribute,
    "lazy_constant": _LazyConstant,
}


def _build(case: type[_Stateful]) -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(case(), nn.Tanh(), nn.Linear(WIDTH, WIDTH)).to(COMPILE_DEVICE)


def _batches() -> list[Tensor]:
    lucid.manual_seed(1)
    return [lucid.randn(6, WIDTH, device=COMPILE_DEVICE) * (1 + i) for i in range(CALLS)]


def _loss(y: Tensor) -> Tensor:
    return (y * y).mean()


class _Entry:
    """One way to call a model; subclasses say how to run it and what they cached."""

    grad = True
    steps_itself = False

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        self.model, self.optimizer = model, optimizer

    def call(self, x: Tensor) -> float:
        raise NotImplementedError

    def compiled(self) -> bool | None:
        """Whether an executable is cached; ``None`` when the entry has none to say."""
        return None


class _MakeStep(_Entry):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.step = lucid.compile.make_step(model, _loss)

    def call(self, x: Tensor) -> float:
        loss = self.step(x)
        value = float(loss.item())
        loss.backward()
        return value

    def compiled(self) -> bool:
        return len(self.step.cache) > 0


class _Segmented(_Entry):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.step = lucid.compile.make_step(model, _loss, segments=3)

    def call(self, x: Tensor) -> float:
        loss = self.step(x)
        value = float(loss.item())
        loss.backward()
        return value


class _CompiledStep(_Entry):
    def call(self, x: Tensor) -> float:
        return float(compiled_step(self.model, x, _loss).item())


class _FusedStep(_Entry):
    steps_itself = True

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.step = fused_step(model, _loss, optimizer)

    def call(self, x: Tensor) -> float:
        return float(self.step(x).item())

    def compiled(self) -> bool:
        return self.step.cache_info()["entries"] > 0


class _ModuleNoGrad(_Entry):
    grad = False

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.module = lucid.compile(model)

    def call(self, x: Tensor) -> float:
        with lucid.no_grad():
            return float(_loss(self.module(x)).item())

    def compiled(self) -> bool:
        return self.module.cache_info()["entries"] > 0


ENTRIES: dict[str, type[_Entry]] = {
    "make_step": _MakeStep,
    "segmented": _Segmented,
    "compiled_step": _CompiledStep,
    "fused_step": _FusedStep,
    "module_no_grad": _ModuleNoGrad,
}


def _eager_call(model: nn.Module, x: Tensor, grad: bool) -> float:
    if not grad:
        with lucid.no_grad():
            return float(_loss(model(x)).item())
    loss = _loss(model(x))
    value = float(loss.item())
    loss.backward()
    return value


def _run(case: type[_Stateful], entry_name: str) -> _Entry:
    """Run ``entry_name`` against an eager twin, comparing the loss after each call."""
    model, twin = _build(case), _build(case)
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    twin_optimizer = optim.SGD(twin.parameters(), lr=0.05)
    entry = ENTRIES[entry_name](model, optimizer)
    for i, batch in enumerate(_batches()):
        optimizer.zero_grad()
        twin_optimizer.zero_grad()
        got = entry.call(batch.clone())
        want = _eager_call(twin, batch.clone(), entry.grad)
        assert abs(got - want) / max(1.0, abs(want)) < TOLERANCE, f"call {i}: {got} vs {want}"
        if entry.grad:
            if not entry.steps_itself:
                optimizer.step()
            twin_optimizer.step()
    return entry


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize("case", list(REBINDING))
def test_an_attribute_the_call_rebinds_is_read_as_it_is_now(case: str, entry: str) -> None:
    _run(REBINDING[case], entry)


@pytest.mark.parametrize("entry", ["make_step", "segmented", "fused_step", "module_no_grad"])
@pytest.mark.parametrize("case", list(NOT_REBINDING))
def test_what_is_not_a_rebinding_still_compiles(case: str, entry: str) -> None:
    ran = _run(NOT_REBINDING[case], entry)
    assert ran.compiled() in (True, None), "refusing every attribute would pass the comparison"


def test_a_state_that_keeps_its_graph() -> None:
    """Only a call that does not differentiate may keep the graph of the one before."""
    _run(_RecurrentGraph, "module_no_grad")


def _closure_model() -> Callable[[Tensor], Tensor]:
    """A function over a box of tensors: the state is the closure's, not a module's."""
    lucid.manual_seed(0)
    linear = nn.Linear(WIDTH, WIDTH).to(COMPILE_DEVICE)
    box: dict[str, Tensor] = {}

    def forward(x: Tensor) -> Tensor:
        y = linear(x)
        out = y + box["h"] if "h" in box else y
        box["h"] = y.detach()
        return out

    return forward


def test_a_tensor_a_closure_keeps_is_read_as_it_is_now() -> None:
    compiled, eager = lucid.compile(_closure_model()), _closure_model()
    with lucid.no_grad():
        for i, x in enumerate(_batches()):
            got = float(_loss(compiled(x)).item())
            want = float(_loss(eager(x)).item())
            assert abs(got - want) / max(1.0, abs(want)) < TOLERANCE, f"call {i}: {got} vs {want}"


def test_a_sequence_that_starts_again_reads_its_state() -> None:
    """The ``None`` branch of the first call is not the call of a model that kept a state."""
    model, twin = _build(_Recurrent), _build(_Recurrent)
    compiled = lucid.compile(model)
    batches = _batches()
    with lucid.no_grad():
        for sequence in range(2):
            model[0].h = twin[0].h = None
            for i, x in enumerate(batches):
                got = float(_loss(compiled(x)).item())
                want = float(_loss(twin(x)).item())
                assert abs(got - want) / max(1.0, abs(want)) < TOLERANCE, (
                    f"sequence {sequence} call {i}: {got} vs {want}"
                )


@pytest.mark.parametrize("case", ["recurrent", "cache_dict"])
def test_a_step_that_cannot_stand_puts_the_slots_back_for_the_eager_call(case: str) -> None:
    """A cumulative batch norm traces without its buffers, so the trace-time call
    cannot stand as the call: the slots it rebound go back, and eager rebinds once."""
    lucid.manual_seed(0)

    def build() -> nn.Module:
        lucid.manual_seed(0)
        return nn.Sequential(
            REBINDING[case](), nn.BatchNorm1d(WIDTH, momentum=None), nn.Linear(WIDTH, WIDTH)
        ).to(COMPILE_DEVICE)

    model, twin = build(), build()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    twin_optimizer = optim.SGD(twin.parameters(), lr=0.05)
    step = lucid.compile.make_step(model, _loss)
    for i, batch in enumerate(_batches()):
        optimizer.zero_grad()
        twin_optimizer.zero_grad()
        loss = step(batch.clone())
        got = float(loss.item())
        loss.backward()
        want = _eager_call(twin, batch.clone(), True)
        assert abs(got - want) / max(1.0, abs(want)) < TOLERANCE, f"call {i}: {got} vs {want}"
        optimizer.step()
        twin_optimizer.step()


class _DecodeRecurrent(nn.Module):
    """A decode forward that keeps the previous step's hidden state on the module."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)
        self.h: Tensor | None = None

    def forward(self, hidden: Tensor, cache: StaticCache, position: Tensor) -> Tensor:
        y = self.linear(hidden)
        keys, _ = cache.update(
            y.reshape(1, 1, 1, WIDTH), y.reshape(1, 1, 1, WIDTH), 0, {"cache_position": position}
        )
        out = y if self.h is None else y + self.h
        self.h = y.detach()
        return out + keys.mean()


def _decode(compiled: bool) -> list[Tensor]:
    lucid.manual_seed(0)
    model = _DecodeRecurrent().to(COMPILE_DEVICE)
    cache = StaticCache(8)

    def forward(hidden: Tensor, position: Tensor) -> Tensor:
        return model(hidden, cache, position)

    step = compiled_decode_step(forward, cache) if compiled else forward
    outs: list[Tensor] = []
    with lucid.no_grad():
        for pos in range(CALLS + 1):
            hidden = lucid.ones(1, 1, WIDTH, device=COMPILE_DEVICE) * (0.1 * (pos + 1))
            outs.append(step(hidden, lucid.tensor([pos], device=COMPILE_DEVICE).long()))
    return outs


def test_a_decode_step_reads_the_attribute_it_rebinds() -> None:
    for i, (got, want) in enumerate(zip(_decode(True), _decode(False), strict=True)):
        assert float((got - want).abs().max().item()) < TOLERANCE, f"step {i}"
