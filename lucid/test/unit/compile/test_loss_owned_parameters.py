"""A compiled step trains the parameters its loss owns, as eager does (LCD-721).

``make_step`` and ``compiled_step`` asked the executable for the gradients of
the model's parameters and of the call's tensors, and nothing else.  A learned
loss weight — the ``log_var`` of an uncertainty-weighted multi-task loss, a
criterion's centres — is neither: ``loss_fn`` closes over it or holds it as an
attribute of its own.  The executable read it, the loss depended on it, and its
``.grad`` stayed ``None``; an optimizer that held it stepped nothing, and the
model's weights drifted without a word.

Each case is held to eager on an identical model, on CPU and Metal: the
gradients of one ``backward``, then three Adam steps over the model and the
loss's parameters together.  ``fused_step`` has no ``.grad`` to compare (its
backward and update run inside the executable), so it is held to the
trajectory.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compiled_step, fused_step, make_step

pytestmark = pytest.mark.filterwarnings(
    "ignore::lucid.compile._core.fallback.CompileFallbackWarning"
)

_TOL = 1e-4

_LossFn = Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]


@dataclass(slots=True)
class _Loss:
    """A loss function and the parameters it owns."""

    fn: _LossFn
    params: list[nn.Parameter]


def _scalar(value: float, device: str) -> nn.Parameter:
    return nn.Parameter(lucid.zeros((), device=device) + value)


def _closure(device: str) -> _Loss:
    logs = _scalar(0.3, device)

    def fn(out: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        return F.mse_loss(out, t) * lucid.exp(-logs) + logs

    return _Loss(fn, [logs])


class _Uncertainty(nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.log_var = _scalar(-0.2, device)

    def forward(self, out: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        return F.mse_loss(out, t) * lucid.exp(-self.log_var) + self.log_var


def _attribute(device: str) -> _Loss:
    crit = _Uncertainty(device)
    return _Loss(lambda out, t: crit(out, t), list(crit.parameters()))


class _MultiTask(nn.Module):
    """Uncertainty-weighted sum of a loss per half of the output."""

    def __init__(self, device: str) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(lucid.zeros(2, device=device) + 0.1)

    def forward(self, out: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        first = F.mse_loss(out[:, :2], t[:, :2])
        second = F.mse_loss(out[:, 2:], t[:, 2:])
        lv = self.log_vars
        return first * lucid.exp(-lv[0]) + lv[0] + second * lucid.exp(-lv[1]) + lv[1]


def _multitask(device: str) -> _Loss:
    crit = _MultiTask(device)
    return _Loss(lambda out, t: crit(out, t), list(crit.parameters()))


_OWNERS: dict[str, Callable[[str], _Loss]] = {
    "closure": _closure,
    "attribute": _attribute,
    "multitask": _multitask,
}
_ENTRIES = ("make_step", "segmented", "compiled_step", "fused_step")
_GRAD_ENTRIES = ("make_step", "segmented", "compiled_step")


def _model(device: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)


def _batch(device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(1)
    return lucid.randn(16, 8, device=device), lucid.randn(16, 4, device=device)


def _close(got: lucid.Tensor | None, want: lucid.Tensor | None) -> bool:
    assert got is not None and want is not None, (got, want)
    scale = max(1.0, float(want.abs().max().item()))
    return float((got - want).abs().max().item()) <= _TOL * scale


@dataclass(slots=True)
class _Run:
    """One entry point driving ``model`` and ``loss`` on a batch."""

    model: nn.Sequential
    loss: _Loss
    forward_backward: Callable[[], lucid.Tensor]
    compiled: Callable[[], bool]

    @property
    def params(self) -> list[nn.Parameter]:
        return [*self.model.parameters(), *self.loss.params]


def _eager(owner: str, device: str) -> _Run:
    model, loss = _model(device), _OWNERS[owner](device)
    x, y = _batch(device)

    def forward_backward() -> lucid.Tensor:
        out = loss.fn(model(x), y)
        out.backward()
        return out

    return _Run(model, loss, forward_backward, lambda: True)


def _compiled(entry: str, owner: str, device: str) -> _Run:
    model, loss = _model(device), _OWNERS[owner](device)
    x, y = _batch(device)
    if entry == "compiled_step":

        def run() -> lucid.Tensor:
            return compiled_step(model, x, lambda out: loss.fn(out, y))

        return _Run(model, loss, run, lambda: True)

    step = make_step(model, loss.fn, segments=2 if entry == "segmented" else 1)
    if entry == "segmented":

        def ran() -> bool:
            return not step.eager_only["flag"]

    else:

        def ran() -> bool:
            return bool(step.cache) and not step.eager_only

    def run_step() -> lucid.Tensor:
        out = step(x, y)
        out.backward()
        return out

    return _Run(model, loss, run_step, ran)


def _fused(owner: str, device: str) -> _Run:
    model, loss = _model(device), _OWNERS[owner](device)
    x, y = _batch(device)
    step = fused_step(model, loss.fn, optim.Adam([*model.parameters(), *loss.params], lr=0.05))
    return _Run(model, loss, lambda: step(x, y), lambda: step.cache_info()["entries"] >= 1)


@pytest.mark.parametrize("entry", _GRAD_ENTRIES)
@pytest.mark.parametrize("owner", sorted(_OWNERS))
def test_one_backward_gives_the_eager_gradients(owner: str, entry: str, device: str) -> None:
    eager, run = _eager(owner, device), _compiled(entry, owner, device)
    want_loss = eager.forward_backward()
    got_loss = run.forward_backward()

    if device == "metal":
        assert run.compiled(), f"{entry} fell back to eager"
    assert _close(got_loss, want_loss)
    assert run.loss.params, "the case owns no parameter"
    for got, want in zip(run.params, eager.params, strict=True):
        assert _close(got.grad, want.grad)


def _trajectory(run: _Run, steps: int, fused: bool) -> list[float]:
    if fused:
        for _ in range(steps):
            run.forward_backward()
        return [float(p.detach().sum().item()) for p in run.params]
    opt = optim.Adam(run.params, lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        run.forward_backward()
        opt.step()
    return [float(p.detach().sum().item()) for p in run.params]


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("owner", sorted(_OWNERS))
def test_three_adam_steps_follow_eager(owner: str, entry: str, device: str) -> None:
    eager = _eager(owner, device)
    if entry == "fused_step":
        run = _fused(owner, device)
    else:
        run = _compiled(entry, owner, device)
    want = _trajectory(eager, 3, fused=False)
    got = _trajectory(run, 3, fused=entry == "fused_step")

    if device == "metal":
        assert run.compiled(), f"{entry} fell back to eager"
    # The loss's parameters must have moved: a dropped gradient leaves them put.
    start = _OWNERS[owner](device).params
    for p, s in zip(run.loss.params, start, strict=True):
        assert float((p.detach() - s.detach()).abs().max().item()) > 1e-3, (
            "the loss weight is stuck"
        )
    for g, w in zip(got, want, strict=True):
        assert abs(g - w) <= _TOL * max(1.0, abs(w))


def test_a_loss_parameter_the_loss_does_not_reach_gets_none(device: str) -> None:
    lucid.manual_seed(2)
    unused = _scalar(1.0, device)

    def fn(out: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        _metric = unused * 3.0  # traced, discarded: the loss does not depend on it
        return F.mse_loss(out, t)

    model = _model(device)
    x, y = _batch(device)
    step = make_step(model, fn)
    step(x, y).backward()

    assert unused.grad is None
    assert all(p.grad is not None for p in model.parameters())
    if device == "metal":
        assert step.cache and not step.eager_only


def test_a_loss_parameter_gradient_adds_into_dot_grad(device: str) -> None:
    """Two backwards without ``zero_grad`` leave the sum, and a held ``.grad`` sees it."""
    loss = _closure(device)
    model = _model(device)
    x, y = _batch(device)
    step = make_step(model, loss.fn)
    (logs,) = loss.params

    step(x, y).backward()
    held = logs.grad
    assert held is not None
    first = float(held.item())
    step(x, y).backward()

    assert logs.grad is not None
    assert abs(float(held.item()) - 2.0 * first) <= _TOL * max(1.0, abs(first))


def test_freezing_and_unfreezing_a_loss_parameter_is_followed(device: str) -> None:
    """The executable differentiates what required grad when it was traced; a flip retraces."""
    loss = _closure(device)
    model = _model(device)
    x, y = _batch(device)
    step = make_step(model, loss.fn)
    (logs,) = loss.params

    logs.requires_grad_(False)
    step(x, y).backward()
    assert logs.grad is None
    logs.requires_grad_(True)
    step(x, y).backward()
    assert logs.grad is not None

    eager = _eager("closure", device)
    eager.forward_backward()
    assert _close(logs.grad, eager.params[-1].grad)
    logs.requires_grad_(False)
    logs.grad = None
    step(x, y).backward()
    assert logs.grad is None


@pytest.mark.parametrize("entry", ["make_step", "compiled_step"])
def test_a_loss_tensor_computed_from_a_leaf_passes_its_gradient_on(entry: str, device: str) -> None:
    """A tensor eager ops made from a leaf, closed over by the loss, reaches the leaf."""

    def one(use_compiled: bool) -> nn.Parameter:
        model = _model(device)
        x, y = _batch(device)
        base = _scalar(0.5, device)
        scale = lucid.exp(base * 2.0)  # not a leaf: the graph behind it ends at ``base``

        def fn(out: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
            return F.mse_loss(out, t) * scale

        if not use_compiled:
            fn(model(x), y).backward()
        elif entry == "make_step":
            make_step(model, fn)(x, y).backward()
        else:
            compiled_step(model, x, lambda out: fn(out, y))
        return base

    got, want = one(True), one(False)
    assert _close(got.grad, want.grad)
