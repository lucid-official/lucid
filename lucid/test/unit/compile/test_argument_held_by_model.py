"""A tensor the model holds, passed as the input, keys an entry of its own (LCD-629).

A trace of ``m(m.weight)`` reads the weight once: the argument and the
parameter share one feed, and the feed binding hands that feed to the
argument.  The key compared the arguments only with each other, so a later
call with another tensor of the same signature reused the executable and
ran with the weight replaced by that tensor — 23.6 off eager, without a
word; ``make_step`` left the weight's gradient 3.05 off and the input's
None.  Each entry point below is called with a tensor the model holds, then
with a fresh tensor of the same shape (requiring grad and not), then with
the held one again, and is held to eager.  On Metal the fresh tensor builds
an entry of its own and the held one finds its first entry again.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compiled_step, fused_step, make_step
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.filterwarnings(
    "ignore::lucid.compile._core.fallback.CompileFallbackWarning"
)

_TOL = 1e-5
_METAL = pytest.mark.skipif(not metal_available(), reason="metal unavailable")
_DEVICES = ["cpu", pytest.param("metal", marks=_METAL)]


def _values(shape: tuple[int, ...], offset: float, device: str) -> lucid.Tensor:
    """A deterministic tensor, different for each ``offset``."""
    n = int(np.prod(shape))
    data = (np.arange(n, dtype=np.float32).reshape(shape) * 0.07 + offset) % 1.1
    return lucid.tensor(data).to(device)


def _close(a: lucid.Tensor, b: lucid.Tensor) -> None:
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=_TOL, atol=_TOL)


class _Held(nn.Module):
    """Reads a parameter, a buffer and a plain tensor attribute, all ``(4, 4)``."""

    def __init__(self, device: str) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("shift", lucid.full((4, 4), 0.25))
        self.to(device)
        # A plain attribute, not a buffer: it lives in ``__dict__``.
        self.scale = _values((4, 4), 0.6, device)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return (self.lin(x) + self.shift) * self.scale


def _global_matmul(x: lucid.Tensor) -> lucid.Tensor:
    return (x @ _WEIGHT).tanh()


_WEIGHT = lucid.zeros(4, 4)


def _held_model(
    which: str, device: str
) -> tuple[Callable[[lucid.Tensor], lucid.Tensor], lucid.Tensor]:
    """What to compile, and the tensor it holds that the test passes in."""
    global _WEIGHT
    if which in ("closure", "global"):
        weight = _values((4, 4), 0.3, device)
        if which == "global":
            _WEIGHT = weight
            return _global_matmul, weight

        def closure(x: lucid.Tensor) -> lucid.Tensor:
            return (x @ weight).tanh()

        return closure, weight
    lucid.manual_seed(0)
    model = _Held(device)
    held = {
        "parameter": model.lin.weight,
        # Another tensor object over the parameter's own ``TensorImpl``.
        "parameter-contiguous": model.lin.weight.contiguous(),
        "buffer": model.shift,
        "attribute": model.scale,
    }[which]
    return model, held


_WHICH = ["parameter", "parameter-contiguous", "buffer", "attribute", "closure", "global"]


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fresh_grad", [True, False], ids=["fresh-grad", "fresh"])
@pytest.mark.parametrize("which", _WHICH)
def test_a_compiled_module_keys_a_held_tensor_apart(
    which: str, fresh_grad: bool, device: str
) -> None:
    target, held = _held_model(which, device)
    compiled = lucid.compile(target)
    fresh = _values((4, 4), 0.9, device).requires_grad_(fresh_grad)
    for x in (held, fresh, held, fresh):
        _close(compiled(x), target(x))
    if device == "metal":
        info = compiled.cache_info()
        assert info["entries"] == 2 and not info["eager_only"], info


# One training step that runs its backward: ``run(x, target) -> loss``.
_Run = Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]


def _seq(device: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4)).to(device)


def _eager(model: nn.Module) -> _Run:
    def run(x: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
        loss = F.mse_loss(model(x), target)
        loss.backward()
        return loss

    return run


def _train(model: nn.Sequential, run: _Run, fresh_grad: bool, device: str) -> list[np.ndarray]:
    """Three steps — the first layer's weight as the input, a fresh tensor,
    the weight again — and what each left: the loss and every gradient."""
    target = _values((4, 4), 0.2, device)
    fresh = _values((4, 4), 0.9, device).requires_grad_(fresh_grad)
    seen: list[np.ndarray] = []
    for x in (model[0].weight, fresh, model[0].weight):
        for p in model.parameters():
            p.grad = None
        fresh.grad = None
        seen.append(run(x, target).detach().numpy())
        seen += [p.grad.numpy() for p in model.parameters() if p.grad is not None]
        seen.append(np.array(fresh.grad is None))
        if fresh.grad is not None:
            seen.append(fresh.grad.numpy())
    return seen


def _assert_same(got: list[np.ndarray], want: list[np.ndarray]) -> None:
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=_TOL, atol=_TOL)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fresh_grad", [True, False], ids=["fresh-grad", "fresh"])
@pytest.mark.parametrize("segments", [1, 2])
def test_make_step_keys_a_parameter_input_apart(
    segments: int, fresh_grad: bool, device: str
) -> None:
    model, ref = _seq(device), _seq(device)
    step = make_step(model, F.mse_loss, segments=segments)
    _assert_same(
        _train(model, _eager_step(step), fresh_grad, device),
        _train(ref, _eager(ref), fresh_grad, device),
    )
    if device == "metal":
        if segments == 1:
            assert len(step.cache) == 2 and not step.eager_only  # type: ignore[attr-defined]
        else:
            first = step.cache[0]  # type: ignore[attr-defined]
            assert len(first) == 2 and not step.eager_only["flag"]  # type: ignore[attr-defined]


def _eager_step(step: Callable[..., lucid.Tensor]) -> _Run:
    def run(x: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
        loss = step(x, target)
        loss.backward()
        return loss

    return run


@_METAL
@pytest.mark.parametrize("fresh_grad", [True, False], ids=["fresh-grad", "fresh"])
def test_compiled_step_answers_a_parameter_input_as_eager(fresh_grad: bool) -> None:
    # It traces afresh on every call; Metal only, with no eager fallback.
    device = "metal"
    model, ref = _seq(device), _seq(device)

    def run(x: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
        # It runs the backward itself: the gradients are in ``.grad``.
        return compiled_step(model, x, lambda y: F.mse_loss(y, target))

    _assert_same(
        _train(model, run, fresh_grad, device), _train(ref, _eager(ref), fresh_grad, device)
    )


class _Shifted(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("shift", lucid.full((4, 4), 0.25))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.lin(x) + self.shift


def _fused_run(
    calls: Callable[[_Shifted], list[tuple[lucid.Tensor, lucid.Tensor]]], compiled: bool
) -> tuple[list[float], list[np.ndarray], object]:
    lucid.manual_seed(0)
    model = _Shifted().to("metal")
    opt = optim.SGD(model.parameters(), lr=0.1)
    step = fused_step(model, F.mse_loss, opt) if compiled else None
    losses = []
    for x, t in calls(model):
        if step is not None:
            losses.append(float(step(x, t).item()))
            continue
        opt.zero_grad()
        loss = F.mse_loss(model(x), t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    params = [p.detach().numpy() for p in model.parameters()]
    return losses, params, None if step is None else step.cache_info()


def _buffer_input(m: _Shifted) -> list[tuple[lucid.Tensor, lucid.Tensor]]:
    """A buffer passed as the input shares the buffer's feed."""
    t = _values((4, 4), 0.2, "metal")
    return [(m.shift, t), (_values((4, 4), 0.9, "metal"), t), (m.shift, t)]


def _same_input_twice(m: _Shifted) -> list[tuple[lucid.Tensor, lucid.Tensor]]:
    """``step(a, a)`` reads one feed, ``step(a, b)`` two: the key compared
    the inputs' shapes only."""
    del m
    a = _values((4, 4), 0.5, "metal")
    return [(a, a), (a, _values((4, 4), 0.1, "metal")), (a, a)]


_FUSED_CALLS = {"buffer-input": _buffer_input, "same-input-twice": _same_input_twice}


@_METAL
@pytest.mark.parametrize("calls", list(_FUSED_CALLS))
def test_fused_step_keys_how_its_inputs_bind(calls: str) -> None:
    losses, params, info = _fused_run(_FUSED_CALLS[calls], compiled=True)
    ref_losses, ref_params, _ = _fused_run(_FUSED_CALLS[calls], compiled=False)
    np.testing.assert_allclose(losses, ref_losses, rtol=_TOL, atol=_TOL)
    for p, rp in zip(params, ref_params, strict=True):
        np.testing.assert_allclose(p, rp, rtol=_TOL, atol=_TOL)
    assert isinstance(info, dict)
    assert info["entries"] == 2 and not info["eager_only"], info
