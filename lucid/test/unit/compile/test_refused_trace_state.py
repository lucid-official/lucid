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

The same driver then checks the rest of what a trace-time call leaves that eager does not (LCD-992):

* a registered buffer, or an attribute, grown by every call (``cat``): its trace stands as the call
  and is never replayed (a replay would hand every call the first one's history);
* a ``.backward()`` with no ``inputs=`` in the forward: ``.grad`` of the parameters and of the
  call's own tensors goes back with the rest, or the eager rerun accumulates into it twice;
* a plan that compiled and then failed its first run, which the call answers eagerly: the Python
  the trace ran goes back first, in every entry (the segmented ones too, forward and backward);
* a segmented compile, which traces the forward twice (the second for its backward) and moves the
  forward's counters once, as eager does.
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
from lucid.test.unit.compile.test_first_run_fallback import _inject
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


def _same_state(got: _Stamped, want: _Stamped, when: str) -> None:
    """The exact state and the cache, which a compiled run computes to a rounding of eager's."""
    assert _state(got) == _state(want), f"state {when}"
    gap = float((got.scale - want.scale).abs().max().item())
    assert gap <= TOLERANCE, f"cache {when}: {gap} off"


def _inner[M: nn.Module](model: nn.Module, kind: type[M]) -> M:
    """The one module of ``kind`` in ``model``'s tree."""
    found = [m for m in model.modules() if isinstance(m, kind)]
    assert len(found) == 1
    return found[0]


def _stamped_of(model: nn.Module) -> _Stamped:
    return _inner(model, _Stamped)


def _loss(y: Tensor) -> Tensor:
    return (y * y).mean()


def _batches() -> list[Tensor]:
    lucid.manual_seed(1)
    return [lucid.randn(6, WIDTH, device=COMPILE_DEVICE) * (1 + i) for i in range(CALLS)]


type _Build = Callable[[nn.Module], nn.Module]
type _Make = Callable[[], nn.Module]
#: Compares the module of a call with its eager twin's, after it: what a case is about.
type _Compare = Callable[[nn.Module, nn.Module, str], None]


def _alone(inner: nn.Module) -> nn.Module:
    return nn.Sequential(nn.Linear(WIDTH, WIDTH), inner, nn.Tanh(), nn.Linear(WIDTH, WIDTH))


def _first_segment(inner: nn.Module) -> nn.Module:
    return nn.Sequential(inner, nn.Tanh(), nn.Linear(WIDTH, WIDTH), nn.Tanh())


def _last_segment(inner: nn.Module) -> nn.Module:
    return nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, WIDTH), inner)


def _model(build: _Build, make: _Make) -> nn.Module:
    lucid.manual_seed(0)
    return build(make()).to(COMPILE_DEVICE)


class _Entry:
    """One way to call a model: how it runs a call, and how many executables it holds."""

    grad = True
    steps_itself = False
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


def _drive(entry_name: str, make: _Make, compare: _Compare) -> _Entry:
    """Run the entry against an eager twin, comparing the loss and ``compare`` after each call."""
    kind = ENTRIES[entry_name]
    model, twin = (_model(kind.build, make) for _ in range(2))
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
            compare(model, twin, f"after call {i}")
            if entry.grad:
                if not entry.steps_itself:
                    optimizer.step()
                twin_optimizer.step()
    return entry


def _same_stamped(model: nn.Module, twin: nn.Module, when: str) -> None:
    _same_state(_stamped_of(model), _stamped_of(twin), when)


def _run(entry_name: str, refuse: str, write: str, target: str) -> _Entry:
    """:func:`_drive` over a :class:`_Stamped`, whose state is compared exactly."""
    return _drive(entry_name, lambda: _Stamped(refuse, write, target), _same_stamped)


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
    monkeypatch.setattr(SlotWatch, "restore", lambda self, *, grads=True: None)
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
    """A decode forward over ``inner``, with a cache to write."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, hidden: Tensor, cache: StaticCache, position: Tensor) -> Tensor:
        y = self.inner(hidden)
        keys, _ = cache.update(
            y.reshape(1, 1, 1, WIDTH), y.reshape(1, 1, 1, WIDTH), 0, {"cache_position": position}
        )
        return y + keys.mean()


def _decode(make: _Make, compiled: bool, *, refused: bool) -> tuple[list[Tensor], nn.Module]:
    """Three decode steps over ``make()``, compiled or eager: each output, and the module.

    ``refused``: that no step may leave an executable (a trace that is refused makes none).
    """
    lucid.manual_seed(0)
    model = _Decoding(make()).to(COMPILE_DEVICE)
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
            if compiled and refused:
                assert not step.cache, "the scenario must be refused"
    return outs, model.inner


def _same_decode(make: _Make, compare: _Compare, *, refused: bool = True) -> None:
    """A compiled decode step against an eager one: every output, then ``compare``."""
    got, model = _decode(make, compiled=True, refused=refused)
    want, twin = _decode(make, compiled=False, refused=refused)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert float((a - b).abs().max().item()) < TOLERANCE, f"step {i}"
    compare(model, twin, "after the last step")


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("refuse", REFUSALS)
def test_a_refused_decode_step_leaves_the_module_as_one_eager_call_would(
    refuse: str, write: str, target: str
) -> None:
    _same_decode(lambda: _Stamped(refuse, write, target), _same_stamped)


# -- LCD-992: a buffer replaced by a tensor of another shape ------------------------------


class _Growing(nn.Module):
    """A history that every call grows by a row (``cat``), which the output reads.

    A registered buffer takes the replacement through its registry and not through ``__dict__``;
    a write into the old buffer cannot change its shape, so ``Module.__setattr__`` leaves it a
    replacement.  Compiled, the executable was fed the history it read at trace time, and the
    module kept the first call's.
    """

    def __init__(self, target: str) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.randn(WIDTH, WIDTH))
        if target == "buffer":
            self.register_buffer("hist", lucid.zeros(1, WIDTH), persistent=False)
        else:
            self.hist = lucid.zeros(1, WIDTH, device=COMPILE_DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        self.hist = lucid.cat([self.hist, x.detach().reshape(-1, WIDTH).mean(0, keepdim=True)], 0)
        return x @ self.w + self.hist.sum()


def _same_history(model: nn.Module, twin: nn.Module, when: str) -> None:
    got, want = _inner(model, _Growing).hist, _inner(twin, _Growing).hist
    assert got.shape == want.shape, f"history {when}: {tuple(got.shape)} vs {tuple(want.shape)}"
    gap = float((got - want).abs().max().item())
    assert gap <= TOLERANCE, f"history {when}: {gap} off"


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry", list(ENTRIES))
def test_a_history_grown_by_every_call_is_never_replayed(entry: str, target: str) -> None:
    """The trace that grew it stands as the call, and the signature runs eagerly: every call
    finds the history one row longer, as eager's does.  Replayed, call 1 was 0.66 off."""
    ran = _drive(entry, lambda: _Growing(target), _same_history)
    assert ran.cached() == 0, "a rebinding no write can carry must not compile"


@pytest.mark.parametrize("target", TARGETS)
def test_a_decode_step_that_grows_a_history_is_never_replayed(target: str) -> None:
    _same_decode(lambda: _Growing(target), _same_history)


class _Table(nn.Module):
    """A table built once from constants and kept: the same values however often it is built."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.randn(WIDTH, WIDTH))
        self.register_buffer("table", None, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.table is None:
            self.table = lucid.arange(WIDTH, device=COMPILE_DEVICE).float() * 0.1
        return x @ self.w + self.table


def test_a_table_built_once_from_constants_still_compiles() -> None:
    """What a replay would build is what the trace built: no verdict, an executable."""
    ran = _drive("module", _Table, lambda model, twin, when: None)
    assert ran.cached() > 0


# -- LCD-992: a backward pass in the forward accumulates into .grad -----------------------


class _Backwarding(nn.Module):
    """A forward that runs a backward pass of its own: ``.backward()`` with no ``inputs=``
    accumulates into the gradient of every leaf it reaches, ``w`` here."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.randn(WIDTH, WIDTH))

    def forward(self, x: Tensor) -> Tensor:
        with lucid.enable_grad():
            (self.w * self.w).sum().backward()
        return x @ self.w


def _same_grads(model: nn.Module, twin: nn.Module, when: str) -> None:
    got, want = _inner(model, _Backwarding).w.grad, _inner(twin, _Backwarding).w.grad
    assert got is not None and want is not None, f"gradient {when}"
    gap = float((got - want).abs().max().item())
    assert gap <= TOLERANCE * (1.0 + float(want.abs().max().item())), f"gradient {when}: {gap} off"


@pytest.mark.parametrize("entry", ["module", "make_step", "segmented_first", "segmented_last"])
def test_a_backward_in_a_refused_forward_accumulates_once(entry: str) -> None:
    """The trace ran the forward's backward, and the eager rerun runs it again: ``.grad`` went
    back with the rest of the module's state, or it held twice what eager's did."""
    ran = _drive(entry, _Backwarding, _same_grads)
    assert ran.cached() == 0, "a backward pass in the forward must refuse the trace"


def test_a_backward_in_a_refused_decode_step_accumulates_once() -> None:
    _same_decode(_Backwarding, _same_grads)


class _BackwardToArgument(nn.Module):
    """Backward from the forward into its own input, a leaf the module does not hold."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.randn(WIDTH, WIDTH))

    def forward(self, x: Tensor) -> Tensor:
        with lucid.enable_grad():
            (x * x).sum().backward()
        return x @ self.w


def test_a_refused_trace_puts_back_the_gradient_of_an_argument() -> None:
    model, twin = (_model(lambda inner: inner, _BackwardToArgument) for _ in range(2))
    compiled = lucid.compile(model)
    x, y = (lucid.ones(6, WIDTH, device=COMPILE_DEVICE).requires_grad_() for _ in range(2))
    with warnings.catch_warnings(), lucid.no_grad():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        compiled(x)
        twin(y)
    assert x.grad is not None and y.grad is not None
    assert float((x.grad - y.grad).abs().max().item()) <= TOLERANCE, "accumulated twice"
    assert compiled.cache_info()["entries"] == 0


def test_a_gradient_already_held_goes_back_to_its_values() -> None:
    """A gradient the call found is not zeroed or doubled: it is as it was before the trace."""
    model, twin = (_model(_alone, _Backwarding) for _ in range(2))
    for m in (_inner(model, _Backwarding), _inner(twin, _Backwarding)):
        m.w.grad = lucid.ones(WIDTH, WIDTH, device=COMPILE_DEVICE) * 3.0
    compiled = lucid.compile(model)
    x = lucid.randn(6, WIDTH, device=COMPILE_DEVICE)
    with warnings.catch_warnings(), lucid.no_grad():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        compiled(x)
        twin(x)
    _same_grads(model, twin, "after the call")


# -- LCD-992: a plan that compiled and failed its first run -------------------------------

#: The engine call each entry's first run goes through, which the case makes fail.
FIRST_RUNS: Final = {
    "module": "run_executable",
    "make_step": "run_executable",
    "segmented_first": "run_executable",
    "segmented_last": "run_executable",
    "fused_step": "run_executable_inplace",
}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("write", COMPILED_WRITES)
@pytest.mark.parametrize("entry", list(ENTRIES))
def test_a_first_run_the_engine_fails_leaves_the_module_as_one_eager_call_would(
    entry: str, write: str, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trace ran the forward's Python and the executable it made never ran: the eager answer
    runs it again over what the trace left, so that goes back first."""
    calls = _inject(monkeypatch, FIRST_RUNS[entry], {0})
    _run(entry, "none", write, target)
    assert len(calls) > 1, "the failure was never reached, or nothing ran after it"


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("write", COMPILED_WRITES)
def test_a_decode_step_whose_first_run_fails_leaves_the_module_as_eager_would(
    write: str, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject(monkeypatch, "run_executable", {0})
    _same_decode(lambda: _Stamped("none", write, target), _same_stamped, refused=False)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(("entry", "fails_at"), [("segmented_first", 3), ("segmented_last", 2)])
def test_a_segmented_backward_recompute_does_not_move_the_forward_state_again(
    entry: str, fails_at: int, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forward ran compiled; its backward executable's first run fails, in the caller's
    ``loss.backward()``, and the segment's gradients are taken eagerly from its input: a second
    run of the forward's Python, whose effects are not the call's."""
    _inject(monkeypatch, "run_executable", {fails_at})
    _run(entry, "none", "copy_", target)


def test_a_first_run_failure_gives_the_eager_call_the_random_stream_eager_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Put back with the module: the attempt's streams are restored when it ends, and a run that
    failed may have drawn from them since."""
    lucid.manual_seed(5)
    model = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Dropout(0.5)).to(COMPILE_DEVICE)
    compiled = lucid.compile(model)
    x = lucid.randn(6, WIDTH, device=COMPILE_DEVICE)
    drawn: list[object] = []

    def draw_then_fail(*args: object) -> object:
        drawn.append(lucid.rand(1, device=COMPILE_DEVICE))  # a run's own draw before it fails
        raise RuntimeError("run_executable: MPSGraph failed: injected")

    monkeypatch.setattr(_C_engine.compile, "run_executable", draw_then_fail)
    lucid.manual_seed(7)
    with warnings.catch_warnings(), lucid.no_grad():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        got = compiled(x)
    lucid.manual_seed(7)
    with lucid.no_grad():
        want = model(x)
    assert drawn, "the failure was never reached"
    assert float((got - want).abs().max().item()) <= TOLERANCE


# -- LCD-992: a segmented compile traces the forward twice ---------------------------------


class _Counting(nn.Module):
    """A forward that counts its own runs: eager moves it once per call."""

    def __init__(self) -> None:
        super().__init__()
        self.runs = 0
        self.w = nn.Parameter(lucid.randn(WIDTH, WIDTH))

    def forward(self, x: Tensor) -> Tensor:
        self.runs += 1
        return x @ self.w


def _same_runs(model: nn.Module, twin: nn.Module, when: str) -> None:
    assert _inner(model, _Counting).runs == _inner(twin, _Counting).runs, f"runs {when}"


@pytest.mark.parametrize("entry", ["segmented_first", "segmented_last"])
def test_a_segmented_compile_moves_a_forward_counter_once_per_call(entry: str) -> None:
    """The backward executable is built from a second trace of the forward.  That run starts from
    where the call began, so the counter holds one run, not two."""
    ran = _drive(entry, _Counting, _same_runs)
    assert ran.cached() > 0, "the scenario must compile, or it proves nothing"
