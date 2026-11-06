"""An in-place write on a tensor the call computed, under every entry (LCD-895).

Recording a graph, an in-place op runs its forward against a stand-in for its
target (``inplace::snapshot``), so that its node keeps the values the target is
about to lose.  The tracer was not told, so it took the stand-in for a tensor
from outside: the op read a feed pinned at the trace-time value, and a feed that
requires grad besides.  ``y = x * 2; y += x`` answered from the first batch on
every later call, and the gradient went through the trace-time call's graph:
``make_step`` raised "backward a second time" from call 1, a segmented step and
``fused_step`` trained on a wrong gradient from call 0.  ``no_grad`` makes no
stand-in, which is why a module entry there was right.

Each case runs several calls with different inputs against a twin running
eagerly and compares the loss and the gradient (the parameters, for
``fused_step``, which steps them itself).  The first call agrees on much of this
either way.  Where the trace can read the value from the op that computed it, the
case also asserts that the entry stayed compiled: refusing every in-place write
would pass the comparison.
"""

from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
from lucid.compile._entry.function import compiled_step
from lucid.compile._entry.fused_step import fused_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

Tensor = lucid.Tensor
Write = Callable[[Tensor, Tensor], Tensor]
Step = Callable[[Tensor], float]
WIDTH = 4
CALLS = 3
TOLERANCE = 3e-4


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield


def _setitem(y: Tensor, key: object, value: Tensor) -> Tensor:
    y[key] = value
    return y


#: Every write takes the tensor to write and the input of the call.
WRITES: dict[str, Write] = {
    "mul_scalar": lambda y, x: y.mul_(2.0),
    "mul_tensor": lambda y, x: y.mul_(x),
    "mul_itself": lambda y, x: y.mul_(y),
    "add_": lambda y, x: y.add_(x),
    "sub_": lambda y, x: y.sub_(x),
    "div_": lambda y, x: y.div_(2.0),
    "chain": lambda y, x: y.mul_(2.0).add_(x),
    "iadd": lambda y, x: y.__iadd__(x),
    "iadd_itself": lambda y, x: y.__iadd__(y),
    "imul": lambda y, x: y.__imul__(x),
    "isub": lambda y, x: y.__isub__(x),
    "idiv": lambda y, x: y.__itruediv__(2.0),
    "unary_clamp": lambda y, x: y.clamp_(-1.0, 1.0),
    "unary_neg": lambda y, x: y.neg_(),
    "unary_exp": lambda y, x: y.mul_(0.1).exp_(),
    "copy": lambda y, x: y.copy_(x * 3.0),
    "copy_of_itself": lambda y, x: y.copy_(y * 3.0),
    "fill": lambda y, x: y.fill_(2.5),
    "zero": lambda y, x: y.zero_(),
    "masked_fill": lambda y, x: y.masked_fill_(x > 0, 1.5),
    "slice_assign": lambda y, x: _setitem(y, (slice(0, 2),), x[0:2] * 3.0),
    "column_assign": lambda y, x: _setitem(y, (slice(None), 1), x[:, 2] * 3.0),
    "list_assign": lambda y, x: _setitem(y, ([0, 2],), x[[1, 3]] * 3.0),
    "view_mul": lambda y, x: (y[:2].mul_(2.0), y)[1],
    "view_add": lambda y, x: (y[:, 1:3].add_(x[:, :2]), y)[1],
    "view_iadd": lambda y, x: (y[1:].__iadd__(x[1:]), y)[1],
    "relu_inplace": lambda y, x: F.relu(y, inplace=True),
}

#: What the written tensor is: computed by an op, a copy, or a factory's.
MADE: dict[str, Callable[[Tensor], Tensor]] = {
    "op": lambda x: x * 2.0,
    "clone": lambda x: x.clone(),
    "factory": lambda x: lucid.ones(*x.shape, device=x.device),
}


class _Writes(nn.Module):
    def __init__(self, write: Write, made: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.write, self.made = write, made

    def forward(self, x: Tensor) -> Tensor:
        return self.write(self.made(x), x) * 2.0


class _ReadsBeforeAndAfter(nn.Module):
    """Both values of the tensor reach the answer."""

    def forward(self, x: Tensor) -> Tensor:
        y = x * 2.0
        before = y * 3.0
        y += x
        return before + y


class _WritesAViewReadAfter(nn.Module):
    """The view written is read afterwards: it holds the write, in the graph too."""

    def forward(self, x: Tensor) -> Tensor:
        y = x * 2.0
        v = y[:2]
        v += x[:2]
        return v.sum() + y * 2.0


class _ReadsAViewFromBefore(nn.Module):
    """A view from before a write through its base has no id for the write: eager."""

    def forward(self, x: Tensor) -> Tensor:
        y = x * 2.0
        v = y[:2]
        y += x
        return v.sum() + y


class _WritesItsArgument(nn.Module):
    """In a monolithic step this is an op's output, in a segment an argument."""

    def forward(self, x: Tensor) -> Tensor:
        x += x * 2.0
        return x * 2.0


class _KeepsIt(nn.Module):
    """The call makes the tensor and keeps it: the trace stands as the call."""

    def forward(self, x: Tensor) -> Tensor:
        y = x * 2.0
        y += x
        self.kept = y.detach()
        return y * 2.0


def _sequential(middle: nn.Module) -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(WIDTH, WIDTH), middle, nn.Linear(WIDTH, WIDTH)).to(
        COMPILE_DEVICE
    )


def _batches() -> list[Tensor]:
    lucid.manual_seed(1)
    return [lucid.randn(6, WIDTH, device=COMPILE_DEVICE) * (1 + i) for i in range(CALLS)]


def _loss(y: Tensor) -> Tensor:
    return (y * y).mean()


def _close(a: Tensor | None, b: Tensor | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    scale = max(1.0, float(b.abs().max().item()))
    return float((a - b).abs().max().item()) / scale < TOLERANCE


class _Entry:
    """One way to call a model; subclasses say how to run it and how to read it."""

    compares_grads = True
    steps_itself = False

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        self.model, self.optimizer = model, optimizer

    def call(self, x: Tensor) -> float:
        raise NotImplementedError

    def stayed_compiled(self) -> bool | None:
        """Whether every call ran compiled; ``None`` when the entry cannot say."""
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

    def stayed_compiled(self) -> bool:
        return len(self.step.eager_only) == 0 and len(self.step.cache) == 1


class _Segmented(_Entry):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.step = lucid.compile.make_step(model, _loss, segments=3)

    def call(self, x: Tensor) -> float:
        loss = self.step(x)
        value = float(loss.item())
        loss.backward()
        return value

    def stayed_compiled(self) -> bool:
        return not self.step.eager_only["flag"]


class _CompiledStep(_Entry):
    def call(self, x: Tensor) -> float:
        return float(compiled_step(self.model, x, _loss).item())


class _FusedStep(_Entry):
    compares_grads = False
    steps_itself = True

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.step = fused_step(model, _loss, optimizer)

    def call(self, x: Tensor) -> float:
        return float(self.step(x).item())


class _ModuleGrad(_Entry):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.compiled = lucid.compile(model)

    def call(self, x: Tensor) -> float:
        loss = _loss(self.compiled(x))
        value = float(loss.item())
        loss.backward()
        return value


class _ModuleNoGrad(_Entry):
    compares_grads = False

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__(model, optimizer)
        self.compiled = lucid.compile(model)

    def call(self, x: Tensor) -> float:
        with lucid.no_grad():
            return float(_loss(self.compiled(x)).item())


ENTRIES: dict[str, type[_Entry]] = {
    "make_step": _MakeStep,
    "segmented": _Segmented,
    "compiled_step": _CompiledStep,
    "fused_step": _FusedStep,
    "module_grad": _ModuleGrad,
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


def _run(middle: Callable[[], nn.Module], entry_name: str) -> _Entry:
    """Run ``entry_name`` against an eager twin, comparing after each call."""
    model, twin = _sequential(middle()), _sequential(middle())
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    twin_optimizer = optim.SGD(twin.parameters(), lr=0.01)
    entry = ENTRIES[entry_name](model, optimizer)
    grad = entry_name != "module_no_grad"
    for i, x in enumerate(_batches()):
        optimizer.zero_grad()
        twin_optimizer.zero_grad()
        got = entry.call(x)
        want = _eager_call(twin, x, grad)
        assert abs(got - want) / max(1.0, abs(want)) < TOLERANCE, f"call {i}: {got} vs {want}"
        if entry.compares_grads:
            for p, q in zip(model.parameters(), twin.parameters(), strict=True):
                assert _close(p.grad, q.grad), f"call {i}: gradient"
        if grad and not entry.steps_itself:
            optimizer.step()
        if grad:
            twin_optimizer.step()
        for p, q in zip(model.parameters(), twin.parameters(), strict=True):
            assert _close(p.detach(), q.detach()), f"call {i}: parameters"
    return entry


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize("write", list(WRITES))
def test_a_write_on_what_an_op_computed(write: str, entry: str) -> None:
    ran = _run(lambda: _Writes(WRITES[write], MADE["op"]), entry)
    assert ran.stayed_compiled() in (True, None)


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize("write", list(WRITES))
def test_a_write_on_a_copy_of_the_input(write: str, entry: str) -> None:
    _run(lambda: _Writes(WRITES[write], MADE["clone"]), entry)


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize("write", list(WRITES))
def test_a_write_on_a_factorys_tensor(write: str, entry: str) -> None:
    _run(lambda: _Writes(WRITES[write], MADE["factory"]), entry)


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize(
    "middle",
    [_ReadsBeforeAndAfter, _WritesAViewReadAfter, _ReadsAViewFromBefore, _WritesItsArgument],
    ids=lambda c: c.__name__.strip("_"),
)
def test_what_reads_around_the_write(middle: type[nn.Module], entry: str) -> None:
    _run(middle, entry)


@pytest.mark.parametrize("entry", list(ENTRIES))
def test_a_kept_tensor_written_in_the_graph(entry: str) -> None:
    _run(_KeepsIt, entry)


@pytest.mark.parametrize("grad", [True, False], ids=["grad", "no_grad"])
def test_the_write_reads_the_op_that_computed_the_value(grad: bool) -> None:
    """The recording itself: no feed appears for the value the write reads."""
    from lucid.compile._core.trace_context import recording

    x = _batches()[0]
    linear = nn.Linear(WIDTH, WIDTH).to(COMPILE_DEVICE)
    with recording(grad_mode=grad) as tracer:
        y = linear(x) * 2.0
        computed = int(tracer.lookup_id(y._impl))
        feeds = set(tracer.external_feeds)
        y += x
    add = next(op for op in tracer.graph.ops if op.name == "add")
    assert computed in list(add.inputs)
    assert set(tracer.external_feeds) == feeds


# -- a value the tracer was not told of ----------------------------------------


class _ComputesUntraced(nn.Module):
    """Computes under the tracer's nose: the guard for whatever does this next."""

    def forward(self, x: Tensor) -> Tensor:
        tracer = _C_engine.compile.current_tracer()
        _C_engine.compile.set_current_tracer(None)
        try:
            y = x * 2.0
        finally:
            _C_engine.compile.set_current_tracer(tracer)
        return y * 3.0


@pytest.mark.parametrize("entry", ["make_step", "fused_step", "compiled_step"])
def test_a_computed_value_the_trace_cannot_name_runs_eagerly(entry: str) -> None:
    """Refused, not frozen: with no id to read, the trace cannot stand for the call."""
    ran = _run(_ComputesUntraced, entry)
    assert ran.stayed_compiled() in (False, None)


def test_a_leaf_of_the_model_is_no_computed_value() -> None:
    """A parameter's ``AccumulateGrad`` is younger than the trace, and it is a feed."""
    ran = _run(lambda: nn.Linear(WIDTH, WIDTH), "make_step")
    assert ran.stayed_compiled()


def test_an_input_computed_before_the_trace_is_no_computed_value() -> None:
    """An eager trunk's output requires grad and has a node: older than the trace."""
    lucid.manual_seed(0)
    trunk = nn.Linear(WIDTH, WIDTH).to(COMPILE_DEVICE)
    head = nn.Sequential(nn.Linear(WIDTH, WIDTH)).to(COMPILE_DEVICE)
    step = lucid.compile.make_step(head, _loss)
    for x in _batches():
        loss = step(trunk(x))
        loss.backward()
    assert len(step.eager_only) == 0
    assert all(p.grad is not None for p in trunk.parameters())
