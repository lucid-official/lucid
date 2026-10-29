"""Every ``lucid.compile`` entry point on the CPU: eager and equal, or refused once.

The executables run on Metal only.  A model on the CPU therefore has nothing
to compile, and each public entry point owns that case at its boundary
(LCD-659): it runs the eager step and returns what eager returns, with the
one-time ``CompileFallbackWarning``; or, where it cannot (the compiled
optimizer owns the optimizer state), it refuses up front with one error that
names Metal — never the engine's ``only Device::GPU`` from deep in a lowering.

``fused_step`` and ``compiled_step`` used to raise that engine error instead
of falling back, and ``fused_step`` took the optimizer's state into a wrapper
that could not step on the CPU.  The table below classifies *every* name in
``lucid.compile.__all__``, so a new entry point has to say which it is.
"""

import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import lucid
import lucid.compile as lc
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile._core import fallback

_STEPS = 3

#: Entry points that train on the CPU as eager does, byte for byte.
FALLS_BACK = ("fused_step", "compiled_step", "make_step", "make_step_segments", "compiled_module")
#: Entry points that refuse on the CPU with the one error naming Metal.
REFUSES = ("compile_optimizer", "save_compiled")
#: Names of ``__all__`` that need no device decision: forward-only wrappers whose
#: own CPU tests live elsewhere, loaders, tracing plumbing and result types.
NOT_AN_ENTRY = (
    "_tracing",
    "compile",
    "CompiledModule",
    "load_compiled",
    "diagnose",
    "DiagnosisReport",
    "OpInfo",
)


@pytest.fixture(autouse=True)
def _fresh_notices(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts as a process that has announced nothing yet."""
    monkeypatch.setattr(fallback, "_NOTICED", set())
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    yield


def _model() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _data() -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(1)
    return lucid.randn(6, 4), lucid.randn(6, 2)


def _sgd(m: nn.Module) -> optim.Optimizer:
    return optim.SGD(m.parameters(), lr=0.05, momentum=0.9)


def _adamw(m: nn.Module) -> optim.Optimizer:
    return optim.AdamW(m.parameters(), lr=0.01)


Trainer = Callable[[nn.Sequential, optim.Optimizer], Callable[[lucid.Tensor, lucid.Tensor], float]]


def _eager(m: nn.Sequential, opt: optim.Optimizer) -> Callable[[lucid.Tensor, lucid.Tensor], float]:
    def step(x: lucid.Tensor, t: lucid.Tensor) -> float:
        opt.zero_grad()
        loss = F.mse_loss(m(x), t)
        loss.backward()
        opt.step()
        return float(loss.item())

    return step


def _fused(m: nn.Sequential, opt: optim.Optimizer) -> Callable[[lucid.Tensor, lucid.Tensor], float]:
    fused = lc.fused_step(m, F.mse_loss, opt)
    return lambda x, t: float(fused(x, t).item())


def _compiled(
    m: nn.Sequential, opt: optim.Optimizer
) -> Callable[[lucid.Tensor, lucid.Tensor], float]:
    def step(x: lucid.Tensor, t: lucid.Tensor) -> float:
        opt.zero_grad()
        loss = lc.compiled_step(m, x, lambda out: F.mse_loss(out, t))
        opt.step()
        return float(loss.item())

    return step


def _made(segments: int) -> Trainer:
    def build(
        m: nn.Sequential, opt: optim.Optimizer
    ) -> Callable[[lucid.Tensor, lucid.Tensor], float]:
        made = lc.make_step(m, F.mse_loss, segments=segments)

        def step(x: lucid.Tensor, t: lucid.Tensor) -> float:
            opt.zero_grad()
            loss = made(x, t)
            loss.backward()
            opt.step()
            return float(loss.item())

        return step

    return build


def _via_module(
    m: nn.Sequential, opt: optim.Optimizer
) -> Callable[[lucid.Tensor, lucid.Tensor], float]:
    cm = lc.compile(m)

    def step(x: lucid.Tensor, t: lucid.Tensor) -> float:
        opt.zero_grad()
        loss = cm.step(x, t, loss_fn=F.mse_loss)
        assert isinstance(loss, lucid.Tensor)
        loss.backward()
        opt.step()
        return float(loss.item())

    return step


_TRAINERS: dict[str, Trainer] = {
    "fused_step": _fused,
    "compiled_step": _compiled,
    "make_step": _made(1),
    "make_step_segments": _made(2),
    "compiled_module": _via_module,
}


def _train(
    trainer: Trainer, make_opt: Callable[[nn.Module], optim.Optimizer]
) -> tuple[list[float], list[list[float]]]:
    m = _model()
    opt = make_opt(m)
    step = trainer(m, opt)
    x, t = _data()
    losses = [step(x, t) for _ in range(_STEPS)]
    return losses, [p.reshape(-1).tolist() for p in m.parameters()]


@pytest.mark.parametrize("make_opt", [_sgd, _adamw], ids=["sgd", "adamw"])
@pytest.mark.parametrize("entry", FALLS_BACK)
def test_cpu_step_equals_eager(
    entry: str, make_opt: Callable[[nn.Module], optim.Optimizer]
) -> None:
    """Losses and parameters after a few steps are eager's, exactly (CPU is deterministic)."""
    want = _train(_eager, make_opt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fallback.CompileFallbackWarning)
        got = _train(_TRAINERS[entry], make_opt)
    assert got == want


@pytest.mark.parametrize("entry", ["fused_step", "compiled_step"])
def test_the_fallback_is_announced_once(entry: str) -> None:
    """One ``CompileFallbackWarning`` for the kind of reason, naming the CPU."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _train(_TRAINERS[entry], _sgd)
    said = [
        str(w.message) for w in caught if issubclass(w.category, fallback.CompileFallbackWarning)
    ]
    assert len(said) == 1
    assert entry in said[0] and "cpu" in said[0] and "running eagerly" in said[0]


def test_fused_step_leaves_the_optimizer_its_own_state() -> None:
    """On the CPU nothing takes the optimizer over: its state is the eager state."""
    m = _model()
    opt = _adamw(m)
    fused = lc.fused_step(m, F.mse_loss, opt)
    x, t = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fallback.CompileFallbackWarning)
        fused(x, t)
    assert fused.optimizer is opt
    assert len(opt.state_dict()["state"]) == len(list(m.parameters()))
    assert all(int(entry["step"].item()) == 1 for entry in opt.state_dict()["state"].values())


def test_fused_step_with_an_enabled_grad_scaler_trains_on_the_cpu() -> None:
    """The scaler is stepped and updated the eager way when there is no executable."""
    from lucid.amp import GradScaler

    m = _model()
    opt = _sgd(m)
    fused = lc.fused_step(m, F.mse_loss, opt, grad_scaler=GradScaler(init_scale=1024.0))
    x, t = _data()
    before = [p.reshape(-1).tolist() for p in m.parameters()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fallback.CompileFallbackWarning)
        loss = fused(x, t)
    assert [p.reshape(-1).tolist() for p in m.parameters()] != before
    assert loss.shape == ()


def test_compile_optimizer_refuses_a_cpu_step_before_touching_state() -> None:
    """One error naming Metal, raised before the optimizer's state or grads move."""
    m = _model()
    opt = _sgd(m)
    copt = lc.compile_optimizer(opt)
    x, t = _data()
    F.mse_loss(m(x), t).backward()
    before = [p.reshape(-1).tolist() for p in m.parameters()]
    with pytest.raises(RuntimeError, match="Metal") as why:
        copt.step()
    assert "Device::GPU" not in str(why.value)
    assert [p.reshape(-1).tolist() for p in m.parameters()] == before
    assert copt.state_dict()["state"] == {}


def test_save_compiled_refuses_a_cpu_module_naming_metal(tmp_path: Path) -> None:
    cm = lc.compile(_model())
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", fallback.CompileFallbackWarning)
        cm(_data()[0])
    with pytest.raises(RuntimeError, match="Metal"):
        lc.save_compiled(cm, str(tmp_path / "artifact"))


def test_the_forward_wrappers_equal_eager_on_the_cpu() -> None:
    """``compile`` and ``diagnose`` on a CPU model: the eager output, no engine error."""
    m = _model()
    x, _ = _data()
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", fallback.CompileFallbackWarning)
        want = m(x).reshape(-1).tolist()
        assert lc.compile(m)(x).reshape(-1).tolist() == want
        lc.diagnose(m, x)


def test_every_public_entry_point_is_classified() -> None:
    """A new name in ``lucid.compile.__all__`` must say what it does on the CPU."""
    classified = {"fused_step", "compiled_step", "make_step", *REFUSES, *NOT_AN_ENTRY}
    assert set(lc.__all__) <= classified, sorted(set(lc.__all__) - classified)
