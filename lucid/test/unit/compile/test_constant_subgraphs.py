"""A constant built inside a compiled call is one feed, not a run of ops (LCD-449).

``x + lucid.tensor([1., 2.]).to(x.device)`` used to record the literal's CPU
``contiguous`` and the transfer, so a Metal trace became mixed-device and the
whole call ran eagerly. ``lucid.compile`` now traces with
``Tracer.fold_constants``: an op whose inputs were all created inside the
current trace (``made_here``) is left out, and the first recorded op that reads
its result takes it as a feed, already on that op's device.

Only tensors built inside the trace call (CPU factories, ``lucid.tensor(...)``
literals, ``from_bytes``) start in ``made_here``. Pre-existing closure or
module tensors stay normal external feeds — ops on them are recorded in the
graph, so mutating a captured Metal tensor updates the result on the next call
without re-tracing.
"""

import json
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import FrameType

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile import CompiledModule, _tracing
from lucid.compile._core import fallback
from lucid.compile._core.symbolic_gate import graph_symbolic_safe
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    _C_engine.compile.session_cache_clear()
    yield


def _x(seed: int, *shape: int) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randn(*(shape or (4,))).to(COMPILE_DEVICE)


def _close(got: lucid.Tensor, want: lucid.Tensor) -> bool:
    return float((got - want).abs().max().item()) <= 1e-5


# Each builds a constant on the spot and moves it to the input's device.
CONSTANTS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "literal": lambda x: x + lucid.tensor([1.0, 2.0, 3.0, 4.0]).to(x.device),
    "arange": lambda x: x * lucid.arange(4).float().to(x.device),
    "ones": lambda x: x - lucid.ones(4).to(x.device),
    "zeros": lambda x: x + lucid.zeros(4).to(x.device),
    "full": lambda x: x * lucid.full((4,), 3.0).to(x.device),
    "eye-row": lambda x: x + lucid.eye(4)[2].to(x.device),
    "float-of-int-literal": lambda x: x + lucid.tensor([1, 2, 3, 4]).float().to(x.device),
    "metal-factory-chain": lambda x: x + lucid.ones(4, device=x.device) * 2,
    "metal-arange": lambda x: x + lucid.arange(4, device=x.device).float(),
    "constant-returned-too": lambda x: (x * 2, lucid.arange(4).float().to(x.device)),
}


def _compiled_twice(
    fn: Callable[[lucid.Tensor], object],
) -> tuple[CompiledModule[[lucid.Tensor], object], list[str], list[tuple[object, object]]]:
    compiled = lucid.compile.compile(fn)
    pairs: list[tuple[object, object]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for seed in (1, 2):
            x = _x(seed)
            pairs.append((compiled(x), fn(x)))
    return compiled, [str(w.message) for w in caught], pairs


def _leaves(value: object) -> list[lucid.Tensor]:
    leaves = list(value) if isinstance(value, tuple) else [value]
    return [leaf for leaf in leaves if isinstance(leaf, lucid.Tensor)]


@pytest.mark.parametrize("name", list(CONSTANTS))
def test_a_constant_built_in_the_call_stays_compiled(name: str) -> None:
    compiled, said, pairs = _compiled_twice(CONSTANTS[name])
    info = compiled.cache_info()
    assert not said, said
    assert not info["eager_only"]
    assert info["entries"] == 1
    for got, want in pairs:
        assert len(_leaves(got)) == len(_leaves(want)) >= 1
        for g, w in zip(_leaves(got), _leaves(want), strict=True):
            assert _close(g, w)


def test_the_trace_holds_one_feed_and_no_constant_op() -> None:
    x = _x(0)
    with _tracing(watch=((x,), {}), fold_constants=True) as tracer:
        CONSTANTS["arange"](x)
    assert [op.name for op in tracer.graph.ops] == ["mul"]
    devices = {str(impl.device) for impl in tracer.external_feeds.values()}
    assert devices == {"Device.GPU"}


def test_a_returned_constant_is_a_new_tensor_each_call() -> None:
    compiled = lucid.compile.compile(CONSTANTS["constant-returned-too"])
    first = _leaves(compiled(_x(1)))[1]
    first.add_(100.0)
    second = _leaves(compiled(_x(2)))[1]
    assert _close(second, lucid.arange(4).float().to(COMPILE_DEVICE))


# ── a tensor from outside the call: live, not frozen ─────────────────────────


def _assert_follows(
    compiled: Callable[[lucid.Tensor], lucid.Tensor],
    eager: Callable[[lucid.Tensor], lucid.Tensor],
    update: Callable[[], None],
    calls: Callable[[], AbstractContextManager[object]] = nullcontext,
) -> None:
    """The calls run under ``calls()``; the update between them does not."""
    x = _x(3, 2, 4)
    with calls():
        assert _close(compiled(x), eager(x))
        before = eager(x)
    update()
    with calls():
        after = eager(x)
        assert not _close(before, after)  # the update is visible to eager
        assert _close(compiled(x), after)


@pytest.mark.parametrize("grad_mode", ["no_grad", "enable_grad"])
def test_a_captured_weight_updated_by_an_optimizer_is_followed(grad_mode: str) -> None:
    """Under ``no_grad`` the executable reads ``w`` as it is now.  Under grad
    mode the call needs ``w``'s gradient, so it runs eagerly instead and
    stays differentiable (LCD-582) — following ``w`` all the same."""
    lucid.manual_seed(0)
    w = nn.Parameter(lucid.randn(3, 4).to(COMPILE_DEVICE))

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return x @ w.T

    compiled = lucid.compile.compile(fn)
    opt = optim.SGD([w], lr=0.5)

    def step() -> None:
        opt.zero_grad()
        (w * w).sum().backward()
        opt.step()

    calls = lucid.no_grad if grad_mode == "no_grad" else lucid.enable_grad
    _assert_follows(compiled, fn, step, calls)
    eager_only = compiled.cache_info()["eager_only"]
    if grad_mode == "no_grad":
        assert not eager_only
        return
    with calls():
        assert compiled(_x(3, 2, 4)).requires_grad
    assert eager_only and all(key.grad for key in eager_only)


def test_a_captured_weight_written_in_place_is_followed_without_recompiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lucid.manual_seed(0)
    w = lucid.randn(3, 4).to(COMPILE_DEVICE)

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return x @ w.T

    compiled = lucid.compile.compile(fn)
    compiles = 0
    orig_compile_for = compiled._compile_for

    def spy_compile_for(*args: object, **kwargs: object) -> object:
        nonlocal compiles
        compiles += 1
        return orig_compile_for(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(compiled, "_compile_for", spy_compile_for)

    def update() -> None:
        w.mul_(-2.0)

    _assert_follows(compiled, fn, update)
    assert compiles == 1
    assert not compiled.cache_info()["eager_only"]


def test_a_captured_cpu_tensor_moved_in_the_call_is_followed() -> None:
    table = lucid.arange(4).float()

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return x + table.to(x.device) * 2

    def update() -> None:
        table.add_(10.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_follows(lucid.compile.compile(fn), fn, update)


class _PlainAttribute(nn.Module):
    """A tensor attribute that is neither a parameter nor a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.scale = lucid.ones(3, device=COMPILE_DEVICE)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.lin(x) * self.scale.exp()


@pytest.mark.usefixtures("inference")
def test_a_module_attribute_that_is_not_state_is_followed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lucid.manual_seed(0)
    model = _PlainAttribute().to(COMPILE_DEVICE)
    assert "scale" not in dict(model.named_buffers())
    compiled = lucid.compile.compile(model)
    compiles = 0
    orig_compile_for = compiled._compile_for

    def spy_compile_for(*args: object, **kwargs: object) -> object:
        nonlocal compiles
        compiles += 1
        return orig_compile_for(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(compiled, "_compile_for", spy_compile_for)

    def update() -> None:
        model.scale.fill_(0.5)

    _assert_follows(compiled, model, update)
    assert compiles == 1
    assert not compiled.cache_info()["eager_only"]


# ── a constant written in place ───────────────────────────────────────────────


def _written_constant(x: lucid.Tensor) -> lucid.Tensor:
    c = lucid.zeros(4).to(x.device)
    c.add_(x)
    return c * 2


def _read_then_written(x: lucid.Tensor) -> lucid.Tensor:
    c = lucid.ones(4).to(x.device)
    y = x + c
    c.mul_(x)
    return y + c


def _constant_onto_constant(x: lucid.Tensor) -> lucid.Tensor:
    c = lucid.zeros(4, device=x.device)
    c.add_(lucid.ones(4, device=x.device))
    return x + c


@pytest.mark.parametrize("fn", [_written_constant, _read_then_written, _constant_onto_constant])
def test_a_constant_written_in_place_is_right_on_every_call(
    fn: Callable[[lucid.Tensor], lucid.Tensor],
) -> None:
    compiled = lucid.compile.compile(fn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for seed in (1, 2, 3, 1):
            x = _x(seed)
            assert _close(compiled(x), fn(x))
    assert not caught, [str(w.message) for w in caught]
    assert not compiled.cache_info()["eager_only"]


def test_an_outside_tensor_written_with_a_constant_is_carried_back() -> None:
    """The written value is folded; the write into ``tally`` still happens
    once per call, and the value's buffer moving into ``tally`` leaves the
    executable's feed a copy of its own."""
    tally = lucid.zeros((), device=COMPILE_DEVICE)

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        with lucid.no_grad():
            tally.add_(lucid.tensor(2.0).to(x.device))
        return x * 3

    compiled = lucid.compile.compile(fn)
    for seed in (1, 2, 3):
        x = _x(seed)
        assert _close(compiled(x), x * 3)
    assert float(tally.item()) == 6.0
    assert not compiled.cache_info()["eager_only"]


def test_a_metal_factory_stays_an_op() -> None:
    """The compiled graph makes it itself — a graph constant, not a feed."""
    x = _x(0)
    with _tracing(watch=((x,), {}), fold_constants=True) as tracer:
        x * 2.0 + lucid.ones(4, device=x.device)
    names = [op.name for op in tracer.graph.ops]
    assert "full" in names and "ones" in names
    assert len(tracer.external_feeds) == 1  # x alone


def test_a_constant_written_in_place_is_no_outside_write() -> None:
    x = _x(0)
    with _tracing(watch=((x,), {}), fold_constants=True) as tracer:
        _written_constant(x)
    assert list(tracer.outside_writes()) == []


# ── random draws ──────────────────────────────────────────────────────────────

DRAWS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "randn": lambda x: x + lucid.randn(4, device=x.device),
    "rand": lambda x: x * lucid.rand(4, device=x.device),
    "dropout-of-a-constant": lambda x: (
        x + F.dropout(lucid.ones(4, device=x.device), p=0.5, training=True)
    ),
    "randn-on-cpu-moved": lambda x: x + lucid.randn(4).to(x.device),
}


@pytest.mark.parametrize("name", list(DRAWS))
def test_a_random_draw_is_never_folded(name: str) -> None:
    x = _x(0)
    compiled = lucid.compile.compile(DRAWS[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first, second = compiled(x), compiled(x)
    assert not _close(first, second)


def test_a_draw_is_a_feed_not_a_constant() -> None:
    x = _x(0)
    with _tracing(watch=((x,), {}), fold_constants=True) as tracer:
        r = lucid.randn(4, device=x.device)
        x + r
    assert not tracer.made_here(r._impl)
    assert tracer.lookup_id(r._impl) is not None


# ── dynamic=True ──────────────────────────────────────────────────────────────


def _positions(x: lucid.Tensor) -> lucid.Tensor:
    return x + lucid.arange(x.shape[0]).float().to(x.device).reshape(-1, 1)


def test_a_constant_sized_from_the_batch_follows_every_batch() -> None:
    compiled = lucid.compile.compile(_positions, dynamic=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for batch in (1, 4, 2):
            x = _x(batch, batch, 3)
            got = compiled(x)
            assert tuple(got.shape) == (batch, 3)
            assert _close(got, _positions(x))
    assert not caught, [str(w.message) for w in caught]
    # The gate saw the batch-sized constant: per-shape executables.
    assert compiled._symbolic_resolved is False
    assert compiled.cache_info()["entries"] == 3


def test_a_fixed_constant_keeps_the_symbolic_batch() -> None:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return x + lucid.arange(3).float().to(x.device)

    compiled = lucid.compile.compile(fn, dynamic=True)
    for batch in (2, 5):
        x = _x(batch, batch, 3)
        assert _close(compiled(x), fn(x))
    assert compiled.cache_info()["entries"] == 1


def test_the_gate_refuses_a_constant_with_the_batch_in_its_shape() -> None:
    graph = type("G", (), {"ops": []})()
    assert graph_symbolic_safe(graph, 4, None, [(4,)]) is False
    assert graph_symbolic_safe(graph, 4, None, [(2, 4)]) is False
    assert graph_symbolic_safe(graph, 4, None, [(3,)]) is True


# ── every other tracer records what it did before ─────────────────────────────


def _tracing_caller() -> str:
    """The file that entered ``_tracing`` — the first frame outside it."""
    frame: FrameType | None = sys._getframe(2)
    while frame is not None and Path(frame.f_code.co_filename).name in (
        "__init__.py",
        "contextlib.py",
        "trace_context.py",
    ):
        frame = frame.f_back
    return Path(frame.f_code.co_filename).name if frame is not None else "?"


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> dict[str, set[bool]]:
    """For each file that traced while the test ran, its tracers' ``fold_constants``.

    Keyed by caller so a compile the tracer under test triggers on the side
    (the attention probe compiles through ``lucid.compile``) is told apart.
    """
    seen: dict[str, set[bool]] = {}
    install = _C_engine.compile.set_current_tracer

    def spy(tracer: _C_engine.compile.Tracer | None) -> None:
        if tracer is not None:
            seen.setdefault(_tracing_caller(), set()).add(bool(tracer.fold_constants))
        install(tracer)

    monkeypatch.setattr(_C_engine.compile, "set_current_tracer", spy)
    return seen


def test_compile_folds(installed: dict[str, set[bool]]) -> None:
    lucid.compile.compile(CONSTANTS["literal"])(_x(0))
    assert installed == {"module.py": {True}}


def test_the_optimizer_tracer_does_not_fold(installed: dict[str, set[bool]]) -> None:
    from lucid.compile import compile_optimizer

    w = nn.Parameter(lucid.randn(4).to(COMPILE_DEVICE))
    opt = compile_optimizer(optim.Adam([w], lr=0.1))
    (w * w).sum().backward()
    opt.step()
    assert installed["compiler.py"] == {False}


def test_the_decode_tracer_does_not_fold(installed: dict[str, set[bool]]) -> None:
    from lucid.test.unit.compile.test_compiled_decode import _decode, _MaskedDecoder

    lucid.manual_seed(0)
    model = _MaskedDecoder().to(COMPILE_DEVICE).eval()
    # Generation's mode: under grad mode a trainable model's step runs eagerly.
    with lucid.no_grad():
        _decode(model, compiled=True, n_steps=2)
    assert installed["decode_step.py"] == {False}


def test_the_segmented_tracers_do_not_fold(installed: dict[str, set[bool]]) -> None:
    lucid.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).to(COMPILE_DEVICE)
    target = lucid.tensor([0, 2]).to(COMPILE_DEVICE)
    step = lucid.compile.make_step(model, lambda out: F.cross_entropy(out, target), segments=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step(_x(0, 2, 4))
    assert installed["segmented_step.py"] == {False}


def test_the_core_ml_tracer_does_not_fold(installed: dict[str, set[bool]]) -> None:
    from lucid.coreml._build import trace

    model = nn.Linear(4, 3)
    trace(model, lucid.randn(2, 4))
    assert installed == {"_build.py": {False}}


_RECORDINGS = Path(__file__).with_name("_fold_off_recordings.json")


def _recorded(x: lucid.Tensor) -> lucid.Tensor:
    a = lucid.arange(4, device=x.device).float()
    e = lucid.eye(4, device=x.device)[1]
    f = lucid.full((4,), 2.0, device=x.device)
    return (x + a) * e + f + lucid.tensor([1.0, 2.0, 3.0, 4.0]).to(x.device)


@pytest.mark.parametrize("device", ["cpu", COMPILE_DEVICE])
@pytest.mark.parametrize("mode", ["coreml", "optimizer"])
def test_a_tracer_without_folding_records_what_main_did(device: str, mode: str) -> None:
    """The factory hook is a no-op with folding off: the recording is the one
    the engine made before it existed (``_fold_off_recordings.json``, taken at
    a571a6c20), op by op, id by id."""
    x = lucid.ones(4, device=device)
    with _tracing(watch=[x] if mode == "coreml" else ()) as tracer:
        if mode == "coreml":
            tracer.redraw_rng = False
        _recorded(x)
    ops = [
        [
            op.name,
            [int(i) for i in op.inputs],
            [[int(o.id), list(o.shape), str(o.dtype), str(o.device)] for o in op.outputs],
            [list(kv) for kv in sorted((k, repr(v)) for k, v in op.attrs.items())],
        ]
        for op in tracer.graph.ops
    ]
    feeds = sorted([int(k), list(v.shape)] for k, v in tracer.external_feeds.items())
    want = json.loads(_RECORDINGS.read_text())[f"{'cpu' if device == 'cpu' else 'metal'}-{mode}"]
    assert ops == want["ops"]
    assert feeds == want["feeds"]
