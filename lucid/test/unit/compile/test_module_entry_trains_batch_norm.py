"""``lucid.compile(model)`` compiles a training-mode batch norm (LCD-882).

The module entry used to run any training-mode module with
``track_running_stats`` eagerly, deciding from the module tree.  The tree cannot
see a functional ``F.batch_norm`` (LCD-875), and the trace already carries what a
norm hands out: the executable returns the new running statistics, and the entry
writes them and the ``num_batches_tracked`` counter back after every run, as
``make_step`` does.  The one update a trace cannot carry is a cumulative average
(``momentum=None``), whose factor is read from the batch count on the host: that
layer, in training, still runs eagerly.

Every case runs against a twin that runs eagerly.  The model is built twice from
the same seed; each call feeds both a batch of its own mean and spread, and the
answer and every buffer are compared after each call.  Three ways to call it,
because the module entry compiles only a forward that records no autograd graph:

* ``no_grad`` -- compiled;
* grad mode with the parameters frozen -- compiled too;
* grad mode with trainable parameters -- eager by design, so ``backward`` reaches
  the parameters; the gradients must still match the twin's and the statistics
  must not move twice.
"""

import warnings
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import override

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._entry.module import CompiledModule
from lucid.test.unit.compile import _zoo_matrix as zoo
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

Tensor = lucid.Tensor


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


@pytest.fixture(autouse=True)
def _fresh_session() -> Iterator[None]:
    _C_engine.compile.session_cache_clear()
    yield


def _frozen(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


#: How a call is made: its grad mode, what is done to the model first, and
#: whether the module entry compiles it.
MODES: dict[
    str, tuple[Callable[[], AbstractContextManager[object]], Callable[[nn.Module], None], bool]
] = {
    "no_grad": (lucid.no_grad, lambda _m: None, True),
    "grad-frozen": (nullcontext, _frozen, True),
    "grad": (nullcontext, lambda _m: None, False),
}


def _buffers(model: nn.Module) -> dict[str, np.ndarray]:
    return {n: b.numpy().copy() for n, b in model.named_buffers()}


def _grads(model: nn.Module) -> dict[str, np.ndarray]:
    return {n: p.grad.numpy().copy() for n, p in model.named_parameters() if p.grad is not None}


def _assert_close(
    got: dict[str, np.ndarray], want: dict[str, np.ndarray], what: str, rtol: float, atol: float
) -> None:
    assert got.keys() == want.keys(), f"{what}: {sorted(got.keys() ^ want.keys())}"
    for name, value in want.items():
        assert np.allclose(got[name], value, rtol=rtol, atol=atol), (
            f"{what} {name}: worst {np.abs(got[name] - value).max()} "
            f"(value scale {np.abs(value).max()})"
        )


def _compare(
    make: Callable[[], nn.Module],
    batch: Callable[[int], tuple[object, ...]],
    mode: str,
    *,
    steps: int,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> CompiledModule[..., object]:
    """Drive a compiled model and its eager twin for ``steps`` calls, in train mode."""
    guard, prepare, _ = MODES[mode]
    lucid.manual_seed(0)
    model = make().to(COMPILE_DEVICE)
    lucid.manual_seed(0)
    twin = make().to(COMPILE_DEVICE)
    prepare(model)
    prepare(twin)
    model.train()
    twin.train()
    compiled = lucid.compile(model)
    for i in range(steps):
        args = batch(i)
        # A model's own random draws (dropout, stochastic depth) are the same
        # stream for both calls: a compiled call draws what eager draws.
        lucid.manual_seed(1000 + i)
        with guard():
            got = compiled(*args)
        lucid.manual_seed(1000 + i)
        with guard():
            want = twin(*args)
        if mode == "grad":
            zoo.objective(got).backward()
            zoo.objective(want).backward()
            _assert_close(_grads(model), _grads(twin), f"call {i}, gradient", rtol, atol * 10)
        _assert_close(
            {"out": zoo._main_output(got).detach().numpy()},
            {"out": zoo._main_output(want).detach().numpy()},
            f"call {i}, output",
            rtol,
            atol,
        )
        _assert_close(_buffers(model), _buffers(twin), f"call {i}, buffer", rtol, atol)
    return compiled


def _assert_compiled(compiled: CompiledModule[..., object], mode: str) -> None:
    info = compiled.cache_info()
    if MODES[mode][2]:
        assert info["entries"] == 1 and not info["eager_only"], info
    else:
        assert info["entries"] == 0, info


def _batches(shape: tuple[int, ...]) -> Callable[[int], tuple[object, ...]]:
    def batch(i: int) -> tuple[object, ...]:
        lucid.manual_seed(100 + i)
        x = lucid.randn(*shape, device=COMPILE_DEVICE) * (1.0 + i) + 0.7 * i
        return (x,)

    return batch


# -- plain norm modules ---------------------------------------------------------------


def _conv_bn(norm: Callable[[int], nn.Module], dims: int) -> Callable[[], nn.Module]:
    conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dims]
    return lambda: nn.Sequential(conv(3, 4, 3), norm(4), nn.ReLU(), conv(4, 4, 1), norm(4))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "dims, norm",
    [
        (1, lambda c: nn.BatchNorm1d(c)),
        (2, lambda c: nn.BatchNorm2d(c)),
        (3, lambda c: nn.BatchNorm3d(c)),
        (2, lambda c: nn.BatchNorm2d(c, momentum=0.3, eps=1e-3)),
        (2, lambda c: nn.BatchNorm2d(c, affine=False)),
        (2, lambda c: nn.BatchNorm2d(c, track_running_stats=False)),
        (2, lambda c: nn.InstanceNorm2d(c, track_running_stats=True)),
        (2, lambda c: nn.InstanceNorm2d(c, momentum=0.4, affine=True, track_running_stats=True)),
        (1, lambda c: nn.InstanceNorm1d(c, track_running_stats=True)),
        (3, lambda c: nn.InstanceNorm3d(c, track_running_stats=True)),
    ],
    ids=[
        "batch1d",
        "batch2d",
        "batch3d",
        "batch2d-momentum-eps",
        "batch2d-no-affine",
        "batch2d-no-stats",
        "instance2d",
        "instance2d-momentum-affine",
        "instance1d",
        "instance3d",
    ],
)
def test_a_training_norm_module_compiles_and_moves_the_state_as_eager_does(
    dims: int, norm: Callable[[int], nn.Module], mode: str
) -> None:
    """Statistics, counters, answers (and, with grad on, gradients) after each call."""
    shape = (4, 3, *([6] * dims))
    with must_compile(mode):
        compiled = _compare(_conv_bn(norm, dims), _batches(shape), mode, steps=4)
    _assert_compiled(compiled, mode)


@contextmanager
def must_compile(mode: str) -> Iterator[None]:
    """A call that compiles raises on a fallback notice instead of passing quietly."""
    with warnings.catch_warnings():
        if MODES[mode][2]:
            warnings.simplefilter("error", CompileFallbackWarning)
        yield


@pytest.mark.parametrize("mode", ["no_grad", "grad-frozen"])
def test_the_counter_counts_every_call_of_a_shared_norm(mode: str) -> None:
    """One layer called twice in a forward advances its counter by two, once per call."""

    class Twice(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.bn = nn.BatchNorm1d(4)

        @override
        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            return self.bn(self.bn(self.lin(x)) * 2.0 + 1.0)

    with must_compile(mode):
        compiled = _compare(Twice, _batches((6, 4)), mode, steps=3)
    _assert_compiled(compiled, mode)


# -- a functional batch norm beside a module ------------------------------------------


class _Mixed(nn.Module):
    """A batch norm layer and a functional one on buffers of the model's own."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.register_buffer("rm", lucid.arange(4.0) * 0.1)
        self.register_buffer("rv", lucid.arange(4.0) * 0.2 + 1.0)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        y = self.bn(self.lin(x))
        return F.batch_norm(y, self.rm, self.rv, training=True, momentum=0.3)


@pytest.mark.parametrize("mode", ["no_grad", "grad-frozen"])
def test_a_functional_batch_norm_is_carried_beside_a_module(mode: str) -> None:
    """The decision is the trace's, so the functional one is carried like the layer's."""
    with must_compile(mode):
        compiled = _compare(_Mixed, _batches((6, 4)), mode, steps=4)
    _assert_compiled(compiled, mode)


class _Functional(nn.Module):
    """No batch norm module at all: the module tree holds nothing to find."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("rm", lucid.zeros(4))
        self.register_buffer("rv", lucid.ones(4))
        self.register_buffer("num_batches_tracked", lucid.zeros(()))

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return F.batch_norm(self.lin(x), self.rm, self.rv, training=True)


def test_a_functional_only_model_compiles_and_carries_its_statistics() -> None:
    with must_compile("no_grad"):
        compiled = _compare(_Functional, _batches((6, 4)), "no_grad", steps=4)
    _assert_compiled(compiled, "no_grad")


# -- the cumulative average: the one update a trace cannot carry ----------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dims", [1, 2])
def test_a_cumulative_batch_norm_in_training_falls_back_with_eager_values(
    dims: int, mode: str
) -> None:
    """Refused to eager, by notice, and the values are eager's."""
    layer = nn.BatchNorm1d if dims == 1 else nn.BatchNorm2d

    def norm(c: int) -> nn.Module:
        return layer(c, momentum=None)

    shape = (4, 3, *([6] * dims))
    compiled = _compare(_conv_bn(norm, dims), _batches(shape), mode, steps=4)
    info = compiled.cache_info()
    assert info["entries"] == 0, info
    if MODES[mode][2]:
        assert info["eager_only"], "a call that could compile was refused silently"


def test_a_cumulative_batch_norm_says_why_it_runs_eagerly() -> None:
    lucid.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4, momentum=None)).to(COMPILE_DEVICE)
    compiled = lucid.compile(model)
    with pytest.warns(CompileFallbackWarning, match="momentum=None"):
        with lucid.no_grad():
            compiled(_batches((6, 4))(0)[0])


def test_a_cumulative_batch_norm_in_eval_compiles() -> None:
    """Evaluating, it reads its buffers and updates nothing: nothing to refuse."""
    lucid.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4, momentum=None)).to(COMPILE_DEVICE)
    model.train()
    with lucid.no_grad():
        model(_batches((6, 4))(0)[0])
    model.eval()
    before = _buffers(model)
    compiled = lucid.compile(model)
    with must_compile("no_grad"), lucid.no_grad():
        for i in range(3):
            x = _batches((6, 4))(i)[0]
            got = compiled(x)
            assert np.allclose(got.numpy(), model(x).numpy(), rtol=1e-4, atol=1e-5)
    assert compiled.cache_info()["entries"] == 1
    _assert_close(_buffers(model), before, "buffer", 0.0, 0.0)


def test_eval_mode_is_unchanged() -> None:
    """``batch_norm_eval`` reads the statistics: they and the counter stay put."""
    lucid.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).to(COMPILE_DEVICE)
    with lucid.no_grad():
        model.train()
        model(_batches((6, 4))(0)[0])
    model.eval()
    before = _buffers(model)
    compiled = lucid.compile(model)
    with must_compile("no_grad"), lucid.no_grad():
        for i in range(3):
            x = _batches((6, 4))(i)[0]
            assert np.allclose(compiled(x).numpy(), model(x).numpy(), rtol=1e-4, atol=1e-5)
    assert compiled.cache_info()["entries"] == 1
    _assert_close(_buffers(model), before, "buffer", 0.0, 0.0)


# -- a forward that raises after a norm ------------------------------------------------


class _Boom(ValueError):
    """What the forward raises."""


class _RaisesAfter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Conv1d(4, 4, 1)
        self.bn = nn.BatchNorm1d(4)
        self.inst = nn.InstanceNorm1d(4, track_running_stats=True)
        self.raise_now = False

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        y = self.bn(self.lin(x))
        y = self.inst(y)
        if self.raise_now:
            raise _Boom("forward failed on purpose")
        return y


def test_a_forward_that_raises_after_a_norm_leaves_eagers_state_through_the_module_entry() -> None:
    """The raising call completed the trace-time call; the retraced calls after it
    neither repeat nor skip an update."""
    lucid.manual_seed(0)
    model = _RaisesAfter().to(COMPILE_DEVICE)
    lucid.manual_seed(0)
    twin = _RaisesAfter().to(COMPILE_DEVICE)
    compiled = lucid.compile(model)
    batch = _batches((6, 4, 5))
    model.raise_now = twin.raise_now = True
    with lucid.no_grad():
        with pytest.raises(_Boom):
            compiled(batch(0)[0])
        with pytest.raises(_Boom):
            twin(batch(0)[0])
    _assert_close(_buffers(model), _buffers(twin), "buffer after the raise", 1e-5, 1e-6)
    assert float(model.bn.num_batches_tracked.item()) == 1.0

    model.raise_now = twin.raise_now = False
    with must_compile("no_grad"), lucid.no_grad():
        for i in range(1, 4):
            x = batch(i)[0]
            assert np.allclose(compiled(x).numpy(), twin(x).numpy(), rtol=1e-4, atol=1e-5)
            _assert_close(_buffers(model), _buffers(twin), f"call {i}, buffer", 1e-4, 1e-5)
    assert compiled.cache_info()["entries"] == 1


# -- real zoo factories ----------------------------------------------------------------

#: Families with a batch norm somewhere in their body, each built through its
#: smallest registered factory with a shrunken config (``create_model(name,
#: **overrides)``, no invented factory: H11).  The zoo has no model with a tracked
#: ``InstanceNorm`` (ncsn's is its own), which the module cases above cover.
ZOO_FAMILIES = ("resnet", "cspnet", "efficientnet", "mobilenet")


# -- real zoo factories ----------------------------------------------------------------

#: Families with a batch norm somewhere in their body, each built through its
#: smallest registered factory with a shrunken config (``create_model(name,
#: **overrides)``, no invented factory: H11).  The zoo has no model with a tracked
#: ``InstanceNorm`` (ncsn's is its own), which the module cases above cover.
ZOO_FAMILIES = ("resnet", "cspnet", "efficientnet", "mobilenet")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", ZOO_FAMILIES)
def test_a_zoo_model_in_training_follows_eager(family: str, mode: str) -> None:
    """Statistics, counters, answers and (grad on) gradients over several calls."""
    task = "image-classification"

    def make() -> nn.Module:
        return zoo.build(family, task)[1]

    _, probe_model, probe_cfg, _ = zoo.build(family, task)

    def batch(i: int) -> tuple[object, ...]:
        args, _kwargs = zoo.default_inputs(probe_model, probe_cfg, family, 100 + i)
        return tuple(a * (1.0 + 0.5 * i) for a in args)

    with must_compile(mode):
        compiled = _compare(make, batch, mode, steps=3, rtol=2e-3, atol=2e-4)
    _assert_compiled(compiled, mode)
