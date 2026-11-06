"""The running statistics a batch norm hands out reach the buffers (LCD-875).

Traced, a training-mode batch norm leaves its buffers alone and hands the new
statistics out, for the executable to return and the entry point to write
back after every run.  ``lucid.compile`` carried a forward's in-place writes
but not those: a functional ``F.batch_norm(..., training=True)`` on a model's
buffers compiled, and its statistics stayed at their first values under
``no_grad``, with no error.  With grad on it ran eagerly and was right.

A norm module in training mode still runs eagerly through the module entry,
by refusal (LCD-882 would carry its statistics instead): these cases guard
that the refusal holds.  Each case runs four calls on inputs of different
mean and spread against a twin running eagerly, and compares the answer and
every buffer after each.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
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
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
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

#: A norm module in training mode: the module entry refuses it, by design.
REFUSED = {"batch1d", "batch1d-twice", "batch2d", "batch3d"}


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
    runs_compiled = mode != "grad" and norm not in REFUSED
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
