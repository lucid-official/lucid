"""What a refused trace leaves behind: the eager call that follows is the call (LCD-990).

A trace runs the forward's Python once.  When nothing compiles, an eager call follows and
runs that Python again, so whatever the first run left in the module is what the second
starts from.  The tensors the trace wrote in place were put back (the tracer's
``restore_outside_writes``); a plain attribute was not, unless a rebound tensor happened to
be found beside it.  A forward that rebuilds a cache once and says so in a stamp
(``if self._stamp != 0: rebuild; self._stamp = 0``) then found the stamp set and the cache
not: the eager rerun skipped the rebuild and read the cache as it was before the trace, 33.73
off the eager twin on the first call.  A counter moved twice, a list grew by two.

The owner is the attempt: an entry that runs the call eagerly after it says so
(``CompileAttempt.concluded(None)``), and :meth:`SlotWatch.restore` puts every slot back, scalar
and container as much as tensor.  A trace that stands as the call, and one that compiles, keep
what the trace-time call did: it was the call.

Every case runs three calls against an eager twin and compares the output, and the module's
whole observable state, after each.  The class is the product of

* what refuses the trace: a host read, a read of ``Tensor._version``, a backward pass;
* how the rebuild lands: rebinding the cache, writing into it, or growing a second cache by
  concatenation (a rebinding that changes the shape, which a registered buffer takes through its
  registry and not through ``__dict__``);
* where the cache is kept: a registered buffer, or a plain tensor attribute;
* the entry that runs the call: ``lucid.compile`` of a module, ``make_step``, a segmented
  ``make_step`` (the module in its first and in its last segment), ``fused_step``, and a
  compiled decode step.
"""

import warnings
from collections.abc import Callable, Iterator
from typing import Final

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._core.rebound_slots import SlotWatch
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.compile._entry.fused_step import fused_step
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.utils.cache import StaticCache

pytestmark = pytest.mark.skipif(not metal_available(), reason="compiled executables run on Metal")

Tensor = lucid.Tensor
WIDTH: Final = 4
CALLS: Final = 3
TOLERANCE: Final = 3e-4

REFUSALS: Final = ("host_read", "version_read", "backward")
WRITES: Final = ("rebind", "copy_", "grow")
#: What a trace that is not refused compiles: a cache grown by concatenation is not one of them.
COMPILED_WRITES: Final = ("rebind", "copy_")
TARGETS: Final = ("buffer", "attribute")


@pytest.fixture(autouse=True)
def _fresh_session() -> Iterator[None]:
    _C_engine.compile.session_cache_clear()
    yield


class _Stamped(nn.Module):
    """A cache rebuilt once from the parameter, guarded by a plain stamp.

    ``refuse`` is what the trace is refused by (``"none"`` for the control that compiles),
    ``write`` how the rebuild lands, and ``target`` where the cache is kept.  The counters
    beside it are the rest of the state a call moves: a scalar, a list, a set and a dict.
    ``grow`` leaves ``scale`` alone and appends a row to a second cache, ``grown``.
    """

    def __init__(self, refuse: str, write: str, target: str) -> None:
        super().__init__()
        self.refuse, self.write = refuse, write
        self.w = nn.Parameter(lucid.randn(WIDTH, WIDTH))
        if target == "buffer":
            self.register_buffer("scale", lucid.ones(WIDTH), persistent=False)
            self.register_buffer("grown", lucid.zeros(1, WIDTH), persistent=False)
        else:
            self.scale = lucid.ones(WIDTH, device=COMPILE_DEVICE)
            self.grown = lucid.zeros(1, WIDTH, device=COMPILE_DEVICE)
        self._stamp = -1
        self.calls = 0
        self.log: list[int] = []
        self.seen: set[int] = set()
        self.table: dict[str, int] = {}

    def _refused(self) -> None:
        if self.refuse == "host_read":
            float(self.w.sum())
        elif self.refuse == "version_read":
            self.w._version
        elif self.refuse == "backward":
            with lucid.enable_grad():
                lucid.autograd.grad((self.w * self.w).sum(), self.w)

    def forward(self, x: Tensor) -> Tensor:
        self._refused()
        if self._stamp != 0:
            built = self.w.detach().sum(0).abs() + 1.0
            if self.write == "rebind":
                self.scale = built
            elif self.write == "copy_":
                self.scale.copy_(built)
            else:
                self.grown = lucid.cat([self.grown, built[None]], 0)
            self._stamp = 0
        self.calls += 1
        self.log.append(self.calls)
        self.seen.add(self.calls % 2)
        self.table["n"] = self.calls
        return x @ self.w * self.scale


def _state(stamped: _Stamped) -> tuple[object, ...]:
    """What a call of the module leaves that is exact: its stamp, scalar, list, set and dict,
    and how many rows the grown cache has."""
    return (
        stamped._stamp,
        stamped.calls,
        list(stamped.log),
        set(stamped.seen),
        dict(stamped.table),
        stamped.grown.shape[0],
    )


def _same_state(got: _Stamped, want: _Stamped, when: str, *, counts: bool = True) -> None:
    """The exact state and the cache, which a compiled run computes to a rounding of eager's.

    ``counts`` is off for an entry whose compile runs the forward's Python twice, which
    moves a counter twice (a segmented step traces the forward again for its backward).
    """
    exact, wanted = _state(got), _state(want)
    if not counts:
        exact, wanted = exact[:1], wanted[:1]
    assert exact == wanted, f"state {when}"
    gap = float((got.scale - want.scale).abs().max().item())
    assert gap <= TOLERANCE, f"cache {when}: {gap} off"


def _stamped_of(model: nn.Module) -> _Stamped:
    found = [m for m in model.modules() if isinstance(m, _Stamped)]
    assert len(found) == 1
    return found[0]


def _loss(y: Tensor) -> Tensor:
    return (y * y).mean()


def _batches() -> list[Tensor]:
    lucid.manual_seed(1)
    return [lucid.randn(6, WIDTH, device=COMPILE_DEVICE) * (1 + i) for i in range(CALLS)]


type _Build = Callable[[_Stamped], nn.Module]


def _alone(stamped: _Stamped) -> nn.Module:
    return nn.Sequential(nn.Linear(WIDTH, WIDTH), stamped, nn.Tanh(), nn.Linear(WIDTH, WIDTH))


def _first_segment(stamped: _Stamped) -> nn.Module:
    return nn.Sequential(stamped, nn.Tanh(), nn.Linear(WIDTH, WIDTH), nn.Tanh())


def _last_segment(stamped: _Stamped) -> nn.Module:
    return nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, WIDTH), stamped)


def _model(build: _Build, refuse: str, write: str, target: str) -> nn.Module:
    lucid.manual_seed(0)
    return build(_Stamped(refuse, write, target)).to(COMPILE_DEVICE)


class _Entry:
    """One way to call a model: how it runs a call, and how many executables it holds."""

    grad = True
    steps_itself = False
    #: Whether a call that compiles moves the module's counters once, as eager does.
    counts_once = True
    build: _Build = staticmethod(_alone)

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        self.model, self.optimizer = model, optimizer

    def call(self, x: Tensor) -> float:
        raise NotImplementedError

    def cached(self) -> int:
        raise NotImplementedError


class _Module(_Entry):
    grad = False

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.compiled = lucid.compile(model)

    def call(self, x: Tensor) -> float:
        with lucid.no_grad():
            return float(_loss(self.compiled(x)).item())

    def cached(self) -> int:
        return int(self.compiled.cache_info()["entries"])


class _MakeStep(_Entry):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.step = lucid.compile.make_step(model, _loss)

    def call(self, x: Tensor) -> float:
        loss = self.step(x)
        value = float(loss.item())
        loss.backward()
        return value

    def cached(self) -> int:
        return len(self.step.cache)


class _Segmented(_MakeStep):
    """A two-segment step; ``holder`` is the segment the stamped module sits in."""

    holder = 0
    counts_once = False

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        _Entry.__init__(self, model, optimizer)
        self.step = lucid.compile.make_step(model, _loss, segments=2)

    def cached(self) -> int:
        return len(self.step.cache[self.holder])


class _SegmentedFirst(_Segmented):
    build = staticmethod(_first_segment)


class _SegmentedLast(_Segmented):
    build = staticmethod(_last_segment)
    holder = 1


class _Fused(_Entry):
    steps_itself = True

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.step = fused_step(model, _loss, optimizer)

    def call(self, x: Tensor) -> float:
        return float(self.step(x).item())

    def cached(self) -> int:
        return int(self.step.cache_info()["entries"])


ENTRIES: Final[dict[str, type[_Entry]]] = {
    "module": _Module,
    "make_step": _MakeStep,
    "segmented_first": _SegmentedFirst,
    "segmented_last": _SegmentedLast,
    "fused_step": _Fused,
}


def _eager(model: nn.Module, x: Tensor, grad: bool) -> float:
    if not grad:
        with lucid.no_grad():
            return float(_loss(model(x)).item())
    loss = _loss(model(x))
    value = float(loss.item())
    loss.backward()
    return value


def _run(entry_name: str, refuse: str, write: str, target: str) -> _Entry:
    """Run the entry against an eager twin, comparing the loss and the state after each call.

    A call that is refused runs the Python once in the trace and once eagerly, so every entry
    compares the counters then; one that compiles is compared as the entry counts it.
    """
    kind = ENTRIES[entry_name]
    counts = refuse != "none" or kind.counts_once
    model, twin = (_model(kind.build, refuse, write, target) for _ in range(2))
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    twin_optimizer = optim.SGD(twin.parameters(), lr=0.05)
    entry = kind(model, optimizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        for i, batch in enumerate(_batches()):
            optimizer.zero_grad()
            twin_optimizer.zero_grad()
            got = entry.call(batch.clone())
            want = _eager(twin, batch.clone(), entry.grad)
            assert abs(got - want) / max(1.0, abs(want)) < TOLERANCE, f"call {i}: {got} vs {want}"
            _same_state(_stamped_of(model), _stamped_of(twin), f"after call {i}", counts=counts)
            if entry.grad:
                if not entry.steps_itself:
                    optimizer.step()
                twin_optimizer.step()
    return entry


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("refuse", REFUSALS)
@pytest.mark.parametrize("entry", list(ENTRIES))
def test_a_refused_trace_leaves_the_module_as_one_eager_call_would(
    entry: str, refuse: str, write: str, target: str
) -> None:
    ran = _run(entry, refuse, write, target)
    assert ran.cached() == 0, "the scenario must be refused, or it compares nothing"


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("write", COMPILED_WRITES)
@pytest.mark.parametrize("entry", list(ENTRIES))
def test_a_trace_that_compiles_keeps_what_its_call_did(entry: str, write: str, target: str) -> None:
    """The control: nothing refuses it, so the executable runs after the trace-time call, whose
    stamp and counters stand.  Putting them back would run the rebuild and the counting twice."""
    ran = _run(entry, "none", write, target)
    assert ran.cached() > 0, "the control must compile, or it proves nothing"


@pytest.mark.parametrize("target", TARGETS)
def test_the_scenario_catches_a_stamp_left_behind(
    target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the restore the eager rerun finds the stamp set: the guard has teeth."""
    monkeypatch.setattr(SlotWatch, "restore", lambda self: None)
    with pytest.raises(AssertionError):
        _run("module", "host_read", "copy_", target)


# -- a step several segments of which are attempts ---------------------------------------


def test_a_segment_refused_after_a_compiled_one_leaves_both_as_one_eager_step_would() -> None:
    """The first segment compiles and runs; the second is refused, and the whole step reruns
    eagerly.  The first segment's Python ran in its trace (twice: the backward traces the forward
    again) and runs again in the rerun: its counters went back past its own attempt."""

    def build() -> nn.Sequential:
        lucid.manual_seed(0)
        return nn.Sequential(
            _Stamped("none", "copy_", "buffer"),
            nn.Tanh(),
            _Stamped("host_read", "copy_", "buffer"),
            nn.Tanh(),
        ).to(COMPILE_DEVICE)

    model, twin = build(), build()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    twin_optimizer = optim.SGD(twin.parameters(), lr=0.05)
    step = lucid.compile.make_step(model, _loss, segments=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        for i, batch in enumerate(_batches()):
            optimizer.zero_grad()
            twin_optimizer.zero_grad()
            loss = step(batch.clone())
            got = float(loss.item())
            loss.backward()
            want = _eager(twin, batch.clone(), True)
            assert abs(got - want) / max(1.0, abs(want)) < TOLERANCE, f"call {i}: {got} vs {want}"
            for part in (0, 2):
                _same_state(model[part], twin[part], f"of segment {part} after call {i}")
            optimizer.step()
            twin_optimizer.step()
    assert len(step.cache[1]) == 0, "the scenario must be refused in the second segment"


# -- the decode step -------------------------------------------------------------------


class _Decoding(nn.Module):
    """A decode forward whose state is a :class:`_Stamped`'s, with a cache to write."""

    def __init__(self, stamped: _Stamped) -> None:
        super().__init__()
        self.stamped = stamped

    def forward(self, hidden: Tensor, cache: StaticCache, position: Tensor) -> Tensor:
        y = self.stamped(hidden)
        keys, _ = cache.update(
            y.reshape(1, 1, 1, WIDTH), y.reshape(1, 1, 1, WIDTH), 0, {"cache_position": position}
        )
        return y + keys.mean()


def _decode(refuse: str, write: str, target: str, compiled: bool) -> tuple[list[Tensor], _Stamped]:
    lucid.manual_seed(0)
    model = _Decoding(_Stamped(refuse, write, target)).to(COMPILE_DEVICE)
    cache = StaticCache(8)

    def forward(hidden: Tensor, position: Tensor) -> Tensor:
        return model(hidden, cache, position)

    step = compiled_decode_step(forward, cache) if compiled else forward
    outs: list[Tensor] = []
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        for pos in range(CALLS + 1):
            hidden = lucid.ones(1, 1, WIDTH, device=COMPILE_DEVICE) * (0.1 * (pos + 1))
            outs.append(step(hidden, lucid.tensor([pos], device=COMPILE_DEVICE).long()))
            if compiled:
                assert not step.cache, "the scenario must be refused"
    return outs, model.stamped


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("refuse", REFUSALS)
def test_a_refused_decode_step_leaves_the_module_as_one_eager_call_would(
    refuse: str, write: str, target: str
) -> None:
    got, stamped = _decode(refuse, write, target, compiled=True)
    want, twin = _decode(refuse, write, target, compiled=False)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert float((a - b).abs().max().item()) < TOLERANCE, f"step {i}"
    _same_state(stamped, twin, "after the last step")
