"""A parameter no gradient reaches keeps ``.grad is None`` under every step (LCD-899).

An optimizer treats a ``None`` gradient and a zero one differently: weight decay and
momentum still act on a zero.  Eager leaves a parameter ``None`` when the loss reaches
it only through a path with no gradient, and each compiled step has to say the same.
A segmented step's backward executable returns a zero slot for a target the loss does
not depend on, and the step used to hand that zero to autograd as a gradient.

The table is every way a path loses its gradient (a comparison mask, ``argmax``, an
integer cast, ``detach``, ``.data``) crossed with where the parameter sits: in the
module before the path (a mask built from its output), in the module that has the
path, or as a parameter of the module whose own use of it is the non-differentiable
one.  Each runs under ``segments=2``, ``segments=3`` and the monolithic ``make_step``,
against an eager twin: ``None`` exactly where eager's is, equal values elsewhere.
"""

from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
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
WIDTH = 4
CALLS = 3
TOLERANCE = 3e-4


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield


class _NoGradient(nn.Module):
    """``forward`` is ``x`` taken through one path that carries no gradient to ``x``."""

    def __init__(self, path: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.path = path

    def forward(self, x: Tensor) -> Tensor:
        return self.path(x)


class _OwnParameter(nn.Module):
    """Holds a parameter ``w``, which ``use`` reads as ``(x, w) -> y``."""

    def __init__(self, use: Callable[[Tensor, Tensor], Tensor]) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.arange(WIDTH, dtype=lucid.float32) - 1.0)
        self.use = use

    def forward(self, x: Tensor) -> Tensor:
        return self.use(x, self.w)


def _ones_like(x: Tensor) -> Tensor:
    return lucid.ones(*x.shape, device=x.device)


#: ``x -> y`` with no gradient to ``x`` (the module's own input is one a gradient was fed).
PATHS: dict[str, Callable[[Tensor], Tensor]] = {
    "mask": lambda x: _ones_like(x).masked_fill(x > 0, 1.5),
    "mask_inplace": lambda x: _ones_like(x).masked_fill_(x > 0, 1.5),
    "mask_beside_a_gradient": lambda x: x * 2.0 + _ones_like(x).masked_fill(x > 0, 1.5),
    "argmax": lambda x: _ones_like(x) * x.argmax(1, keepdim=True).float(),
    "argmin": lambda x: _ones_like(x) * x.argmin(1, keepdim=True).float(),
    "integer_cast": lambda x: x.int().float() + 1.0,
    "detach": lambda x: x.detach() * 2.0,
    "detach_beside_a_gradient": lambda x: x * 2.0 + x.detach(),
    "data": lambda x: x.data * 2.0,
}

#: ``(x, w) -> y`` where ``w`` is reached only through a path with no gradient,
#: or (``w_and_a_mask``) through one that has it and one that has not.
USES: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "w_in_a_mask": lambda x, w: x + (w > 0).float(),
    "w_in_an_argmax": lambda x, w: x + w.argmax(0).float(),
    "w_detached": lambda x, w: x * 2.0 + w.detach(),
    "w_data": lambda x, w: x * 2.0 + w.data,
    "w_and_a_mask": lambda x, w: x * w + (w > 0).float(),
}


def _layouts(middle: Callable[[], nn.Module]) -> dict[str, Callable[[], nn.Module]]:
    """The module under test first, in the middle and last of a three-module model."""

    def linear() -> nn.Module:
        return nn.Linear(WIDTH, WIDTH)

    def build(order: tuple[bool, ...]) -> Callable[[], nn.Module]:
        def model() -> nn.Module:
            lucid.manual_seed(0)
            mods = [middle() if under_test else linear() for under_test in order]
            return nn.Sequential(*mods).to(COMPILE_DEVICE)

        return model

    return {
        "first": build((True, False, False)),
        "middle": build((False, True, False)),
        "last": build((False, False, True)),
    }


def _models() -> dict[str, Callable[[], nn.Module]]:
    out: dict[str, Callable[[], nn.Module]] = {}
    for path_name, path in PATHS.items():
        for layout, model in _layouts(lambda p=path: _NoGradient(p)).items():
            # Last, a path with no gradient makes a loss that needs none: nothing to compare.
            if layout == "last" and not path_name.endswith("beside_a_gradient"):
                continue
            out[f"{path_name}-{layout}"] = model
    for use_name, use in USES.items():
        for layout, model in _layouts(lambda u=use: _OwnParameter(u)).items():
            out[f"{use_name}-{layout}"] = model
    return out


MODELS = _models()


def _loss(y: Tensor) -> Tensor:
    return (y * y).mean()


def _batches() -> list[Tensor]:
    lucid.manual_seed(1)
    return [lucid.randn(6, WIDTH, device=COMPILE_DEVICE) * (1 + i) for i in range(CALLS)]


def _agree(got: Tensor | None, want: Tensor | None) -> bool:
    if got is None or want is None:
        return got is None and want is None
    scale = max(1.0, float(want.abs().max().item()))
    return float((got - want).abs().max().item()) / scale < TOLERANCE


def _leaves_one_unreached(name: str) -> bool:
    """Whether eager leaves a parameter ``None`` in this case (the table's own claim)."""
    case, layout = name.rsplit("-", 1)
    if case in USES:
        return case != "w_and_a_mask"
    # Nothing before the first module has a parameter, and a gradient beside the path
    # reaches everything before it.
    return layout != "first" and not case.endswith("beside_a_gradient")


def _eager_none(twin: nn.Module) -> list[bool]:
    return [p.grad is None for p in twin.parameters()]


#: A step per entry: how to make it from a model, and whether it ran compiled.
ENTRIES: dict[str, Callable[[nn.Module], Callable[[Tensor], Tensor]]] = {
    "segments_2": lambda m: lucid.compile.make_step(m, _loss, segments=2),
    "segments_3": lambda m: lucid.compile.make_step(m, _loss, segments=3),
    "make_step": lambda m: lucid.compile.make_step(m, _loss),
}


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize("name", list(MODELS))
def test_a_parameter_eager_leaves_none_stays_none(name: str, entry: str) -> None:
    model, twin = MODELS[name](), MODELS[name]()
    step = ENTRIES[entry](model)
    nones: list[bool] = []
    for i, batch in enumerate(_batches()):
        for p in (*model.parameters(), *twin.parameters()):
            p.grad = None
        got = step(batch.clone())
        got.backward()
        want = _loss(twin(batch.clone()))
        want.backward()
        nones = _eager_none(twin)
        assert abs(float(got.item()) - float(want.item())) < TOLERANCE * 10, f"call {i}: loss"
        for j, (p, q) in enumerate(zip(model.parameters(), twin.parameters(), strict=True)):
            assert (p.grad is None) == (q.grad is None), (
                f"call {i}, parameter {j}: eager gradient is "
                f"{'None' if q.grad is None else 'a tensor'}, the step's is "
                f"{'None' if p.grad is None else 'a tensor'}"
            )
            assert _agree(p.grad, q.grad), f"call {i}, parameter {j}: gradient values"
    assert any(nones) == _leaves_one_unreached(name), "the case is not the one the table says"


def _momentum_sgd(model: nn.Module) -> optim.Optimizer:
    return optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.1)


@pytest.mark.parametrize("entry", ["segments_3", "make_step", "fused_step"])
@pytest.mark.parametrize("name", ["mask-first", "w_in_a_mask-middle", "argmax-middle"])
def test_weight_decay_and_momentum_leave_an_unreached_parameter_alone(
    name: str, entry: str
) -> None:
    """The reason it matters: a zero gradient still moves a parameter under decay."""
    model, twin = MODELS[name](), MODELS[name]()
    optimizer, twin_optimizer = _momentum_sgd(model), _momentum_sgd(twin)
    if entry == "fused_step":
        stepper = fused_step(model, _loss, optimizer)
    else:
        step = ENTRIES[entry](model)

    for batch in _batches():
        optimizer.zero_grad()
        twin_optimizer.zero_grad()
        if entry == "fused_step":
            stepper(batch.clone())
        else:
            step(batch.clone()).backward()
            optimizer.step()
        _loss(twin(batch.clone())).backward()
        twin_optimizer.step()
    for j, (p, q) in enumerate(zip(model.parameters(), twin.parameters(), strict=True)):
        assert _agree(p.detach(), q.detach()), f"parameter {j} after the steps"
