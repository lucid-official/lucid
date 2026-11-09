"""The running statistics a batch norm hands out reach the buffers (LCD-875).

Traced, a training-mode batch norm leaves its buffers alone and hands the new
statistics out, for the executable to return and the entry point to write
back after every run.  ``lucid.compile`` carried a forward's in-place writes
but not those: a functional ``F.batch_norm(..., training=True)`` on a model's
buffers compiled, and its statistics stayed at their first values under
``no_grad``, with no error.  With grad on it ran eagerly and was right.

A norm module in training mode compiles through the module entry the same way
(LCD-882): the decision comes from the trace, not the module tree.  Each case
runs four calls on inputs of different mean and spread against a twin running
eagerly, and compares the answer and every buffer after each.

The rest are the statistics a compiled run cannot carry, across the entry points:
ones given as arguments, which a later run would write into the first call's
tensors; ones with no feed of their own, a view or a tensor the call made, which
would vanish; and the counters, which only a training batch norm layer advances
and which a lazily built one's first call must not miss.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

Tensor = lucid.Tensor
SHAPE = (6, 4)
#: (scale, offset) per call: each batch has its own mean and spread.
CALLS = ((1.0, 0.0), (2.0, 1.0), (0.5, -2.0), (3.0, 0.5))


@pytest.fixture(autouse=True)
def _fresh_session() -> Iterator[None]:
    _C_engine.compile.session_cache_clear()
    yield


class _Normed(nn.Module):
    """A linear layer, then the norm under test."""

    def __init__(self, norm: str) -> None:
        super().__init__()
        self.lin = nn.Linear(SHAPE[-1], SHAPE[-1])
        self.gamma = nn.Parameter(lucid.full((SHAPE[-1],), 1.5))
        self.beta = nn.Parameter(lucid.full((SHAPE[-1],), -0.25))
        # Not the defaults, so that a statistic left alone shows.
        self.register_buffer("rm", lucid.arange(float(SHAPE[-1])) * 0.1)
        self.register_buffer("rv", lucid.arange(float(SHAPE[-1])) * 0.2 + 1.0)
        self.bn = MODULES[norm]() if norm in MODULES else nn.Identity()
        self.forward_with = NORMS[norm]

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with(self, self.lin(x))


def _functional(m: _Normed, y: Tensor) -> Tensor:
    return F.batch_norm(y, m.rm, m.rv, training=True)


def _functional_affine(m: _Normed, y: Tensor) -> Tensor:
    return F.batch_norm(y, m.rm, m.rv, m.gamma, m.beta, training=True, momentum=0.3)


def _functional_twice(m: _Normed, y: Tensor) -> Tensor:
    once = F.batch_norm(y, m.rm, m.rv, training=True)
    return F.batch_norm(once * 2.0 + 1.0, m.rm, m.rv, training=True)


def _functional_eval(m: _Normed, y: Tensor) -> Tensor:
    return F.batch_norm(y, m.rm, m.rv, m.gamma, m.beta, training=False)


def _module(m: _Normed, y: Tensor) -> Tensor:
    return m.bn(y)


def _module_twice(m: _Normed, y: Tensor) -> Tensor:
    return m.bn(m.bn(y) * 2.0)


def _module_2d(m: _Normed, y: Tensor) -> Tensor:
    return m.bn(y.reshape(*SHAPE, 1, 1)).reshape(*SHAPE)


def _module_3d(m: _Normed, y: Tensor) -> Tensor:
    return m.bn(y.reshape(*SHAPE, 1, 1, 1)).reshape(*SHAPE)


NORMS: dict[str, Callable[[_Normed, Tensor], Tensor]] = {
    "functional": _functional,
    "functional-affine": _functional_affine,
    "functional-twice": _functional_twice,
    "functional-eval": _functional_eval,
    "batch1d": _module,
    "batch1d-twice": _module_twice,
    "batch1d-eval": _module,
    "batch2d": _module_2d,
    "batch3d": _module_3d,
}

MODULES: dict[str, Callable[[], nn.Module]] = {
    "batch1d": lambda: nn.BatchNorm1d(SHAPE[-1]),
    "batch1d-twice": lambda: nn.BatchNorm1d(SHAPE[-1]),
    "batch1d-eval": lambda: nn.BatchNorm1d(SHAPE[-1]).eval(),
    "batch2d": lambda: nn.BatchNorm2d(SHAPE[-1]),
    "batch3d": lambda: nn.BatchNorm3d(SHAPE[-1]),
}


def _frozen(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


#: How each call is made: the grad mode, and what is done to the model first.
#: With grad on and trainable parameters the call needs gradients and runs
#: eagerly; with them frozen it compiles under grad mode too.
MODES: dict[
    str, tuple[Callable[[], AbstractContextManager[object]], Callable[[nn.Module], None]]
] = {
    "no_grad": (lucid.no_grad, lambda _m: None),
    "grad": (nullcontext, lambda _m: None),
    "grad-frozen": (nullcontext, _frozen),
}


def _model(norm: str, mode: str) -> _Normed:
    lucid.manual_seed(0)
    model = _Normed(norm).to(COMPILE_DEVICE)
    MODES[mode][1](model)
    return model


def _inputs() -> list[Tensor]:
    lucid.manual_seed(1)
    base = lucid.randn(*SHAPE, device=COMPILE_DEVICE)
    return [base * scale + offset for scale, offset in CALLS]


def _floats(t: Tensor) -> list[float]:
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def _seen(model: _Normed, run: Callable[[Tensor], Tensor], mode: str) -> list[list[float]]:
    seen: list[list[float]] = []
    for x in _inputs():
        with MODES[mode][0]():
            out = run(x)
        seen.append([*_floats(out), *(v for _, b in model.named_buffers() for v in _floats(b))])
    return seen


def _close(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        abs(g - w) <= 1e-4 * (1.0 + abs(w)) for g, w in zip(got, want, strict=True)
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("norm", NORMS)
def test_a_compiled_forward_moves_the_statistics_as_eager_does(norm: str, mode: str) -> None:
    """Statistics and counters after every call, the answer too; a functional
    batch norm compiles wherever the call needs no gradients."""
    model, twin = _model(norm, mode), _model(norm, mode)
    compiled = lucid.compile(model)
    got, want = _seen(model, compiled, mode), _seen(twin, twin, mode)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert _close(g, w), f"call {i}: {g} vs {w}"
    info = compiled.cache_info()
    runs_compiled = mode != "grad"
    assert (info["entries"] == 1 and not info["eager_only"]) is runs_compiled, info


def test_an_eval_batch_norm_leaves_its_statistics_and_counter_alone() -> None:
    """``training=False`` hands nothing out: the buffers keep their values."""
    model = _model("batch1d-eval", "no_grad")
    before = {name: _floats(b) for name, b in model.named_buffers()}
    compiled = lucid.compile(model)
    with lucid.no_grad():
        for x in _inputs():
            compiled(x)
    assert {name: _floats(b) for name, b in model.named_buffers()} == before
    assert compiled.cache_info()["entries"] == 1


def _loss(y: Tensor) -> Tensor:
    return (y * y).mean()


class _Run:
    """One entry point driving a model, and the eager equivalent of its call:
    gradients and an optimizer step, wherever the entry point takes them."""

    def __init__(
        self, build: Callable[[nn.Module], Callable[[Tensor], object]], *, trains: bool
    ) -> None:
        self.build = build
        self.trains = trains

    def eager(self, model: nn.Module, optimizer: lucid.optim.SGD | None, x: Tensor) -> None:
        if not self.trains:
            with lucid.no_grad():
                model(x)
            return
        _loss(model(x)).backward()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()


def _forward(model: nn.Module) -> Callable[[Tensor], object]:
    compiled = lucid.compile(model)

    def run(x: Tensor) -> object:
        with lucid.no_grad():
            return compiled(x)

    return run


def _stepped(segments: int) -> Callable[[nn.Module], Callable[[Tensor], object]]:
    def build(model: nn.Module) -> Callable[[Tensor], object]:
        step = lucid.compile.make_step(model, _loss, segments=segments)
        return lambda x: step(x).backward()

    return build


def _fused(model: nn.Module) -> Callable[[Tensor], object]:
    return lucid.compile.fused_step(model, _loss, lucid.optim.SGD(model.parameters(), lr=0.01))


#: Every entry point with an eager fallback that a batch norm's statistics reach.
RUNS = {
    "module": _Run(_forward, trains=False),
    "make_step": _Run(_stepped(1), trains=True),
    "segmented": _Run(_stepped(2), trains=True),
    "fused_step": _Run(_fused, trains=True),
}


def _state(model: nn.Module) -> dict[str, list[float]]:
    """Every buffer, and whatever statistics a body keeps in an attribute."""
    state = {name: _floats(b) for name, b in model.named_buffers()}
    for name, module in model.named_modules():
        kept = getattr(module, "kept", None)
        if kept is not None:
            state[f"{name}.kept"] = [*_floats(kept[0]), *_floats(kept[1])]
    return state


def _assert_state_follows_eager(make: Callable[[], nn.Module], run: _Run) -> None:
    """Build a model and its twin the same; the state must agree after every call."""
    lucid.manual_seed(0)
    model = make().to(COMPILE_DEVICE)
    lucid.manual_seed(0)
    twin = make().to(COMPILE_DEVICE)
    driven = run.build(model)
    optimizer = lucid.optim.SGD(twin.parameters(), lr=0.01) if run.build is _fused else None
    for i, x in enumerate(_inputs()):
        lucid.manual_seed(10 + i)
        driven(x)
        lucid.manual_seed(10 + i)
        run.eager(twin, optimizer, x)
        got, want = _state(model), _state(twin)
        assert got.keys() == want.keys(), f"call {i}"
        for name, values in want.items():
            assert _close(got[name], values), f"call {i}, {name}: {got[name]} vs {values}"


def _chain(body: nn.Module) -> nn.Module:
    """``body`` between two layers, so that a segmented step has two segments."""
    return nn.Sequential(nn.Linear(SHAPE[-1], SHAPE[-1]), body, nn.Linear(SHAPE[-1], SHAPE[-1]))


class _NoFeed(nn.Module):
    """A functional batch norm whose statistics have no feed of their own."""

    def __init__(self, form: str) -> None:
        super().__init__()
        self.register_buffer("rm", lucid.zeros(2, SHAPE[-1]))
        self.register_buffer("rv", lucid.ones(2, SHAPE[-1]))
        self.form = form
        self.kept: tuple[Tensor, Tensor] | None = None

    def forward(self, y: Tensor) -> Tensor:
        if self.form == "view":
            return F.batch_norm(y, self.rm[0], self.rv[0], training=True)
        if self.form == "made-and-dropped":
            mean, var = lucid.zeros_like(self.rm[0]), lucid.ones_like(self.rv[0])
            return F.batch_norm(y, mean, var, training=True)
        if self.kept is None:
            self.kept = lucid.zeros_like(self.rm[0]), lucid.ones_like(self.rv[0])
        return F.batch_norm(y, self.kept[0], self.kept[1], training=True)


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("form", ["view", "made-and-kept", "made-and-dropped"])
def test_a_statistic_with_no_feed_is_never_dropped_silently(form: str, run: str) -> None:
    """Refused to eager, so a view's bytes and a kept tensor move every call."""
    _assert_state_follows_eager(lambda: _chain(_NoFeed(form)), RUNS[run])


def test_compiled_step_refuses_a_statistic_with_no_feed() -> None:
    """It has no eager fallback: the statistic is refused loudly, not lost."""
    model = _chain(_NoFeed("view")).to(COMPILE_DEVICE)
    x = _inputs()[0]
    with pytest.raises(RuntimeError, match="running statistics"):
        lucid.compile.compiled_step(model, x, _loss)


class _NamedLikeALayer(nn.Module):
    """Buffers named like a batch norm layer's, which only its forward counts."""

    def __init__(self, adds: bool) -> None:
        super().__init__()
        self.register_buffer("running_mean", lucid.zeros(SHAPE[-1]))
        self.register_buffer("running_var", lucid.ones(SHAPE[-1]))
        self.register_buffer("num_batches_tracked", lucid.tensor(0, dtype=lucid.int64))
        self.adds = adds

    def forward(self, y: Tensor) -> Tensor:
        if self.adds:
            self.num_batches_tracked.add_(1)
        return F.batch_norm(y, self.running_mean, self.running_var, training=True)


class _ThroughAnEvaluatingLayer(nn.Module):
    """A training functional call on the buffers of a layer in eval mode."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(SHAPE[-1]).eval()

    def forward(self, y: Tensor) -> Tensor:
        return F.batch_norm(y, self.bn.running_mean, self.bn.running_var, training=True)


COUNTERS: dict[str, Callable[[], nn.Module]] = {
    "named-like-a-layer": lambda: _NamedLikeALayer(False),
    "named-like-a-layer-counting-itself": lambda: _NamedLikeALayer(True),
    "through-an-evaluating-layer": _ThroughAnEvaluatingLayer,
}


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("case", COUNTERS)
def test_only_a_training_batch_norm_layer_is_counted(case: str, run: str) -> None:
    """``F.batch_norm`` counts nothing: its statistics move, a counter does not."""
    _assert_state_follows_eager(lambda: _chain(COUNTERS[case]()), RUNS[run])


class _WritesItsLayer(nn.Module):
    """A batch norm layer whose ``running_mean`` is also written in place: before
    the norm that starts a value of its own, after it the buffer's last value is
    not the norm's."""

    def __init__(self, when: str) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(SHAPE[-1])
        self.when = when

    def forward(self, y: Tensor) -> Tensor:
        if self.when == "before":
            self.bn.running_mean.add_(0.5)
        out = self.bn(y)
        if self.when == "after":
            self.bn.running_mean.add_(0.5)
        return out


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("when", ["before", "after"])
def test_a_layer_whose_statistic_is_written_in_place_still_counts(when: str, run: str) -> None:
    _assert_state_follows_eager(lambda: _chain(_WritesItsLayer(when)), RUNS[run])


LAZY: dict[str, Callable[[], nn.Module]] = {
    "lazy-linear-and-norm": lambda: nn.Sequential(
        nn.LazyLinear(SHAPE[-1]), nn.LazyBatchNorm1d(), nn.ReLU(), nn.Linear(SHAPE[-1], SHAPE[-1])
    ),
    "lazy-norm": lambda: nn.Sequential(
        nn.Linear(SHAPE[-1], SHAPE[-1]),
        nn.LazyBatchNorm1d(),
        nn.ReLU(),
        nn.Linear(SHAPE[-1], SHAPE[-1]),
    ),
}


@pytest.mark.filterwarnings("ignore::lucid.nn.module.UninitializedParameterWarning")
@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("case", LAZY)
def test_a_lazily_built_batch_norm_counts_its_first_call(case: str, segments: int) -> None:
    """Statistics made inside the trace: the first call's update and count both
    land, whether the trace stands as the call or the call reruns eagerly."""
    _assert_state_follows_eager(LAZY[case], _Run(_stepped(segments), trains=True))


class _GivenStatistics(nn.Module):
    """A forward handed its running statistics as arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(SHAPE[-1], SHAPE[-1])

    def forward(self, x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
        return F.batch_norm(self.lin(x), mean, var, training=True)


def _pairs(shared: bool, rows: int) -> list[tuple[Tensor, Tensor]]:
    """One ``(mean, var)`` per call, or the same one every call; ``rows`` adds an
    axis, for a forward that indexes into it (a view of an argument)."""

    def pair() -> tuple[Tensor, Tensor]:
        mean = lucid.arange(float(SHAPE[-1]), device=COMPILE_DEVICE) * 0.1
        var = lucid.ones(SHAPE[-1], device=COMPILE_DEVICE)
        return (mean, var) if not rows else (mean.repeat(rows, 1), var.repeat(rows, 1))

    if shared:
        one = pair()
        return [one for _ in CALLS]
    return [pair() for _ in CALLS]


class _GivenStatisticsView(_GivenStatistics):
    """The same, reading the statistics through a view of each argument."""

    def forward(self, x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
        return super().forward(x, mean[0], var[0])


def _given_compile(model: nn.Module) -> Callable[..., object]:
    compiled = lucid.compile(model)

    def run(*args: Tensor) -> object:
        with lucid.no_grad():
            return compiled(*args)

    return run


def _given_frozen(model: nn.Module) -> Callable[..., object]:
    _frozen(model)
    return lucid.compile(model)


def _given_function(model: nn.Module) -> Callable[..., object]:
    _frozen(model)
    return lucid.compile(lambda *args: model(*args))


@pytest.mark.parametrize("shared", [False, True], ids=["a-pair-per-call", "one-pair"])
@pytest.mark.parametrize("view", [False, True], ids=["whole", "view"])
@pytest.mark.parametrize("entry", [_given_compile, _given_frozen, _given_function])
def test_statistics_given_as_arguments_land_in_that_calls_tensors(
    entry: Callable[[nn.Module], Callable[..., object]], view: bool, shared: bool
) -> None:
    """A compiled run would write the first call's pair on every call: the next
    call's arguments are other tensors, so the write refuses it, as an in-place
    one does."""
    cls = _GivenStatisticsView if view else _GivenStatistics
    lucid.manual_seed(0)
    model = cls().to(COMPILE_DEVICE)
    lucid.manual_seed(0)
    twin = cls().to(COMPILE_DEVICE)
    run = entry(model)
    got, want = _pairs(shared, 2 if view else 0), _pairs(shared, 2 if view else 0)
    for x, pair, twin_pair in zip(_inputs(), got, want, strict=True):
        run(x, *pair)
        with lucid.no_grad():
            twin(x, *twin_pair)
    for pair, twin_pair in zip(got, want, strict=True):
        assert _close(_floats(pair[0]), _floats(twin_pair[0]))
        assert _close(_floats(pair[1]), _floats(twin_pair[1]))


def _loss_with_statistics(y: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    return _loss(F.batch_norm(y, mean, var, training=True))


@pytest.mark.parametrize("shared", [False, True], ids=["a-pair-per-call", "one-pair"])
def test_a_step_loss_given_statistics_lands_in_that_calls_tensors(shared: bool) -> None:
    """``make_step`` hands the arguments after the first to the loss."""
    lucid.manual_seed(0)
    model = nn.Linear(SHAPE[-1], SHAPE[-1]).to(COMPILE_DEVICE)
    lucid.manual_seed(0)
    twin = nn.Linear(SHAPE[-1], SHAPE[-1]).to(COMPILE_DEVICE)
    step = lucid.compile.make_step(model, _loss_with_statistics)
    got, want = _pairs(shared, 0), _pairs(shared, 0)
    for x, pair, twin_pair in zip(_inputs(), got, want, strict=True):
        step(x, *pair).backward()
        _loss_with_statistics(twin(x), *twin_pair).backward()
    for pair, twin_pair in zip(got, want, strict=True):
        assert _close(_floats(pair[0]), _floats(twin_pair[0]))
        assert _close(_floats(pair[1]), _floats(twin_pair[1]))
