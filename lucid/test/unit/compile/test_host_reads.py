"""A host read of a traced value runs the call eagerly, whichever API made it.

``.item()``, ``.tolist()``, ``.numpy()``, ``bool(t)``, ``float(t)`` and
``int(t)`` hand a tensor's values to Python, where the recording cannot
follow what is done with them. Traced once, the value read becomes a
constant and every later call answers the first one's question — no
crash, no warning (LCD-337).

The tracer marks such a trace unsupported, and the compiled call runs
eagerly with a ``CompileFallbackWarning``. It used to see a read only of a
value an op had already consumed, so reading an *argument* before any op
touched it (``sum(x.tolist())`` first thing in forward) still froze; the
arguments are now declared to the tracer up front.

The property is checked, not the mechanism: two inputs eager tells apart,
and the compiled second answer has to be eager's.
"""

import warnings
from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile import CompiledModule
from lucid.compile._core import fallback
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.compile._host_reads import (
    CONSTANT_READS,
    DERIVED_VALUES,
    FUSED_KERNELS,
    HOST_READS,
    STATE_READS,
    Apply,
    inputs,
)


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


@pytest.fixture(autouse=True)
def _fresh_notices(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    _C_engine.compile.session_cache_clear()
    yield


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", list(HOST_READS))
def test_a_host_read_runs_eagerly_and_follows_the_input(name: str) -> None:
    """Between the calls both the input and the weights change."""
    first, second = inputs()
    reference = Apply(HOST_READS[name]).eval()
    want_first = reference(first)
    reference.change_state()
    want_second = reference(second)
    assert float((want_first - want_second).abs().max().item()) > 1e-3

    model = Apply(HOST_READS[name]).eval().to(COMPILE_DEVICE)
    compiled = lucid.compile.compile(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got_first = compiled(first.to(COMPILE_DEVICE)).to("cpu")
        model.change_state()
        got_second = compiled(second.to(COMPILE_DEVICE)).to("cpu")

    for got, want in ((got_first, want_first), (got_second, want_second)):
        assert float((got - want).abs().max().item()) <= 1e-4 * float(want.abs().max().item())
    said = [
        str(w.message) for w in caught if issubclass(w.category, fallback.CompileFallbackWarning)
    ]
    assert said and "host" in said[0], said
    assert compiled.cache_info()["eager_only"]


_HELD = Apply(HOST_READS["buffer-item"])


def _reads_a_global_model(x: lucid.Tensor) -> lucid.Tensor:
    return _HELD(x)


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("reach", ["bound-method", "closure", "global"])
@pytest.mark.parametrize("name", sorted(STATE_READS))
def test_a_host_read_of_a_reached_modules_state_runs_eagerly(name: str, reach: str) -> None:
    """The weights of a module the compiled function reaches are watched as
    a compiled module's are: the trace watched only the wrapper's own, none,
    so the first call's value was baked in (5.35 off eager)."""
    global _HELD
    model = Apply(HOST_READS[name]).eval().to(COMPILE_DEVICE)
    if reach == "global":
        _HELD = model
        compiled = lucid.compile.compile(_reads_a_global_model)
    elif reach == "closure":
        compiled = lucid.compile.compile(lambda t: model(t))
    else:
        compiled = lucid.compile.compile(model.forward)
    first, second = (t.to(COMPILE_DEVICE) for t in inputs())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fallback.CompileFallbackWarning)
        compiled(first)
        model.change_state()
        got = compiled(second)
    assert float((got - model(second)).abs().max().item()) <= 1e-5
    assert compiled.cache_info()["eager_only"]


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", list(CONSTANT_READS))
def test_a_read_that_feeds_nothing_stays_compiled(name: str) -> None:
    model = Apply(CONSTANT_READS[name]).eval()
    first, second = inputs()
    compiled = lucid.compile.compile(model.to(COMPILE_DEVICE))
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        compiled(first.to(COMPILE_DEVICE))
        got = compiled(second.to(COMPILE_DEVICE)).to("cpu")
    assert float((got - second * 3).abs().max().item()) <= 1e-5
    assert not compiled.cache_info()["eager_only"]


def _follows_the_input(
    fn: Callable[[Apply, lucid.Tensor], lucid.Tensor], device: str
) -> tuple[list[str], CompiledModule]:
    """Compile ``fn`` on ``device``, call it twice, compare with eager.

    Returns the fallback warnings raised and the compiled module.
    """
    first, second = inputs()
    reference = Apply(fn).eval()
    want = [reference(first).to("cpu"), reference(second).to("cpu")]
    assert float((want[0] - want[1]).abs().max().item()) > 1e-3

    compiled = lucid.compile.compile(Apply(fn).eval().to(device))
    with warnings.catch_warnings(record=True) as caught, lucid.no_grad():
        warnings.simplefilter("always")
        got = [compiled(t.to(device)).to("cpu") for t in (first, second)]
    for g, w in zip(got, want, strict=True):
        assert float((g - w).abs().max().item()) <= 1e-4 * float(w.abs().max().item())
    said = [
        str(w.message) for w in caught if issubclass(w.category, fallback.CompileFallbackWarning)
    ]
    return said, compiled


@pytest.mark.parametrize("name", sorted(set(HOST_READS) - STATE_READS))
def test_a_host_read_of_the_input_on_the_cpu_follows_the_input(name: str) -> None:
    """On the CPU every compiled call runs eagerly; the moves go to Metal."""
    _said, compiled = _follows_the_input(HOST_READS[name], "cpu")
    assert compiled.cache_info()["eager_only"]


@pytest.mark.parametrize("device", ["metal", "cpu"])
@pytest.mark.parametrize("name", sorted(DERIVED_VALUES))
def test_a_copy_made_outside_the_ops_follows_the_input(name: str, device: str) -> None:
    """``(x * 3).to("cpu")`` returned the first call's answer (LCD-345)."""
    said, compiled = _follows_the_input(DERIVED_VALUES[name], device)
    assert compiled.cache_info()["eager_only"]
    if device == "metal":
        assert said, "the fallback was silent"


def _ordinary(x: lucid.Tensor) -> lucid.Tensor:
    # Moves and casts that copy nothing, and a constant made on x's device:
    # none of them is a traced value copied outside the ops.
    same = x.to(x.device).to(lucid.float32).float()
    bias = lucid.tensor([0.5], device=x.device)
    return (same * 3 + bias).relu()


def test_a_function_without_a_copy_records_no_extra_op() -> None:
    x = lucid.randn(16).to(COMPILE_DEVICE)
    with lucid.compile._tracing(watch=((x,), {})) as tracer:
        _ordinary(x)
    names = [op.name for op in tracer.graph.ops]
    assert not {"to_device", "detach", "to_shared_storage"} & set(names), names
    assert not tracer.graph.unsupported

    compiled = lucid.compile.compile(_ordinary)
    second = lucid.randn(16).to(COMPILE_DEVICE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        compiled(x)
        got = compiled(second)
    assert float((got - _ordinary(second)).abs().max().item()) <= 1e-6
    assert not compiled.cache_info()["eager_only"]


def test_a_copy_of_a_constant_records_nothing() -> None:
    """``Tracer::on_derived`` leaves a tensor the trace has not seen alone."""
    x = lucid.randn(4).to(COMPILE_DEVICE)
    with lucid.compile._tracing(watch=((x,), {})) as tracer:
        # Built from Python numbers and consumed by no op, so the trace has
        # not seen it. (An op reading it would make it a feed, and a read of
        # a feed is a read of a traced value.)
        constant = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        impl = constant._impl
        _C_engine.to_device(impl, _C_engine.Device.GPU).tolist()
        impl.transfer_to_device(_C_engine.Device.GPU, False).tolist()
        impl.clone_with_grad(True).tolist()
        constant.data.tolist()
        lucid.metal.to_shared(constant).tolist()
    assert [op.name for op in tracer.graph.ops] == []
    assert not tracer.graph.unsupported


@pytest.mark.parametrize("device", ["metal", "cpu"])
@pytest.mark.parametrize("name", sorted(FUSED_KERNELS))
def test_a_fused_kernel_is_recorded_and_follows_the_input(name: str, device: str) -> None:
    """The trace records the fused kernels: a Core ML export and a compile
    trace see the kernel's output as computed from the input. On Metal the
    compile emitter translates them (LCD-450); on the CPU every compiled call
    is eager."""
    x = inputs()[0].to(device)
    with lucid.compile._tracing(watch=((x,), {})) as tracer:
        FUSED_KERNELS[name](Apply(FUSED_KERNELS[name]), x)
    assert any(op.name.startswith("_fused_linear") for op in tracer.graph.ops)
    said, compiled = _follows_the_input(FUSED_KERNELS[name], device)
    if device == "metal":
        assert not said, said
        assert not compiled.cache_info()["eager_only"]
    else:
        assert compiled.cache_info()["eager_only"]


def test_a_metal_kernel_is_recorded() -> None:
    x = inputs()[0].to(COMPILE_DEVICE)
    with lucid.compile._tracing(watch=((x,), {})) as tracer:
        DERIVED_VALUES["metal-kernel"](Apply(DERIVED_VALUES["metal-kernel"]), x)
    assert "_run_metal_kernel" in [op.name for op in tracer.graph.ops]


def test_a_cpu_constant_moved_to_metal_stays_compiled() -> None:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return x * lucid.tensor([2.0]).to(x.device)

    compiled = lucid.compile.compile(fn)
    first, second = (t.to(COMPILE_DEVICE) for t in inputs())
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        compiled(first)
        got = compiled(second)
    assert float((got - second * 2).abs().max().item()) <= 1e-5
    assert not compiled.cache_info()["eager_only"]


class _Linear(nn.Module):
    def __init__(self, move: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.move = move

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.move(self.lin(x))


def _step_twice(
    move: Callable[[lucid.Tensor], lucid.Tensor],
    loss_fn: Callable[[lucid.Tensor], lucid.Tensor],
) -> bool:
    """A compiled training step against eager, on two inputs.

    Returns whether the step ran eagerly.
    """
    lucid.manual_seed(0)
    model = _Linear(move).to(COMPILE_DEVICE)
    step = lucid.compile.make_step(model, loss_fn)
    for seed in (1, 2):
        lucid.manual_seed(seed)
        x = (lucid.randn(2, 4) * seed).to(COMPILE_DEVICE)
        model.zero_grad()
        got = step(x)
        got.backward()
        got_grads = [_grad(p) for p in model.parameters()]
        model.zero_grad()
        want = loss_fn(model(x))
        want.backward()
        scale = abs(float(want.item()))
        assert abs(float(got.item()) - float(want.item())) <= 1e-4 * scale
        for g, p in zip(got_grads, model.parameters(), strict=True):
            assert float((g - _grad(p)).abs().max().item()) <= 1e-4
    return bool(step.eager_only)


def _grad(p: lucid.Tensor) -> lucid.Tensor:
    assert p.grad is not None
    return p.grad.to("cpu")


def test_a_step_reading_a_moved_value_follows_the_input() -> None:
    """``.to()`` under grad goes through the ``to_device`` op (LCD-345)."""

    def loss_fn(out: lucid.Tensor) -> lucid.Tensor:
        rows = out.to("cpu").tolist()
        assert isinstance(rows, list)
        total = sum(v for row in rows if isinstance(row, list) for v in row)
        assert isinstance(total, float)
        return (out * total).sum()

    with pytest.warns(fallback.CompileFallbackWarning, match="host"):
        assert _step_twice(lambda y: y, loss_fn)


def test_a_step_with_a_move_that_copies_nothing_stays_compiled() -> None:
    def loss_fn(out: lucid.Tensor) -> lucid.Tensor:
        return out.square().sum()

    assert not _step_twice(lambda y: y.to(y.device).float(), loss_fn)


class _Gain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gain", lucid.tensor(2.0))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x * self.gain.item()


def test_a_segment_reading_its_buffer_on_the_host_follows_it() -> None:
    """A segment's trace watched its parameters only: the gain read on the
    first step stayed baked in after the buffer changed (1.21 for 3.15)."""
    runs = []
    for compiled in (True, False):
        lucid.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 3), _Gain(), nn.Linear(3, 2))
        model = model.to(COMPILE_DEVICE)
        x, t = (lucid.randn(2, n).to(COMPILE_DEVICE) for n in (4, 2))
        step = lucid.compile.make_step(model, nn.functional.mse_loss, segments=3)
        losses = []
        for k in range(3):
            if k == 1:
                model[1].get_buffer("gain").fill_(5.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", fallback.CompileFallbackWarning)
                loss = step(x, t) if compiled else nn.functional.mse_loss(model(x), t)
            loss.backward()
            losses.append(float(loss.item()))
        runs.append(losses)
    got, want = runs
    assert got == pytest.approx(want, rel=1e-5)
