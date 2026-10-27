"""A compiled step trains the tensors the model holds now (LCD-505).

An executable is fed the tensors it was traced with.  ``make_step`` and
``fused_step`` kept their executables when the model's tensors were replaced,
and went on training the old ones without a word:

* after ``load_state_dict(..., assign=True)`` make_step answered with the
  old weights (loss 1.07 against eager's 2.55);
* after ``.half()`` then ``.float()`` the key came back to the first
  executable, which reads the tensors from before the cast;
* fused_step read and wrote the old tensors, and the live weights never
  moved.

``CompiledModule`` dropped its executables on such a replacement, but only
when it reached the model itself.  Each step now hears of a replacement on
any module of the tree (``ParamFingerprint``) and traces again.  Every case
is held to the eager step on an identical model.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step, make_step
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

_TOL = 1e-5


def _model() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 1)).to("metal")


def _data() -> tuple[lucid.Tensor, lucid.Tensor]:
    rng = np.random.default_rng(3)
    x = lucid.tensor(rng.standard_normal((8, 6)).astype(np.float32)).to("metal")
    t = lucid.tensor(rng.standard_normal((8, 1)).astype(np.float32)).to("metal")
    return x, t


def _assign_load(module: nn.Module, seed: int) -> None:
    rng = np.random.default_rng(seed)
    state = {
        k: lucid.tensor(rng.standard_normal(tuple(v.shape)).astype(np.float32)).to("metal")
        for k, v in module.state_dict().items()
    }
    module.load_state_dict(state, assign=True)


def _round_trip(module: nn.Module) -> None:
    module.half()
    module.float()


# What happens to the model between the first step and the next ones.
_CHANGES: dict[str, Callable[[nn.Sequential], None]] = {
    "assign_load": lambda m: _assign_load(m, 1),
    "half_float_round_trip": _round_trip,
    # Only one submodule is told: the step has to watch the whole tree.
    "a_submodule_half_float_round_trip": lambda m: _round_trip(m[2]),
    "a_submodule_assign_load": lambda m: _assign_load(m[0], 2),
}


def _flat(module: nn.Module) -> list[np.ndarray]:
    return [np.asarray(p.detach().numpy()) for p in module.parameters()]


def _make_step_run(model: nn.Sequential, change: Callable[[nn.Sequential], None]) -> list[float]:
    x, t = _data()
    step = make_step(model, F.mse_loss)
    opt = optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for k in range(4):
        if k == 2:
            change(model)
            # The new parameters an assign load put in place, as a training
            # loop would after a load.
            opt = optim.SGD(model.parameters(), lr=0.1)
        opt.zero_grad()
        loss = step(x, t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _eager_run(model: nn.Sequential, change: Callable[[nn.Sequential], None]) -> list[float]:
    x, t = _data()
    opt = optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for k in range(4):
        if k == 2:
            change(model)
            opt = optim.SGD(model.parameters(), lr=0.1)
        opt.zero_grad()
        loss = F.mse_loss(model(x), t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


@pytest.mark.parametrize("change", list(_CHANGES))
def test_make_step_trains_the_tensors_the_model_holds_now(change: str) -> None:
    model, ref = _model(), _model()
    losses = _make_step_run(model, _CHANGES[change])
    ref_losses = _eager_run(ref, _CHANGES[change])
    np.testing.assert_allclose(losses, ref_losses, rtol=_TOL, atol=_TOL)
    for p, rp in zip(_flat(model), _flat(ref), strict=True):
        np.testing.assert_allclose(p, rp, rtol=_TOL, atol=_TOL)


# fused_step steps the optimizer it was built with.  An assign load puts new
# Parameter objects in the tree, which that optimizer does not hold — so the
# fused step is held to the eager loop with the same optimizer, as it was
# built, across a load of one submodule.
_FUSED_CHANGES = {k: v for k, v in _CHANGES.items() if k != "assign_load"}


@pytest.mark.parametrize("change", list(_FUSED_CHANGES))
def test_fused_step_trains_the_tensors_the_model_holds_now(change: str) -> None:
    x, t = _data()
    model, ref = _model(), _model()
    step = fused_step(model, F.mse_loss, optim.SGD(model.parameters(), lr=0.1))
    ref_opt = optim.SGD(ref.parameters(), lr=0.1)
    for k in range(4):
        if k == 2:
            _FUSED_CHANGES[change](model)
            _FUSED_CHANGES[change](ref)
        loss = float(step(x, t).item())
        ref_opt.zero_grad()
        ref_loss = F.mse_loss(ref(x), t)
        ref_loss.backward()
        ref_opt.step()
        np.testing.assert_allclose(loss, float(ref_loss.item()), rtol=_TOL, atol=_TOL)
    for p, rp in zip(_flat(model), _flat(ref), strict=True):
        np.testing.assert_allclose(p, rp, rtol=_TOL, atol=_TOL)


def test_fused_step_refuses_an_optimizer_left_without_the_models_parameters() -> None:
    # After an assign load of the whole model the optimizer holds none of the
    # model's parameters.  The step trained them as detached tensors; it now
    # says there is nothing to step, as on a first call.
    x, t = _data()
    model = _model()
    step = fused_step(model, F.mse_loss, optim.SGD(model.parameters(), lr=0.1))
    step(x, t).item()
    _assign_load(model, 1)
    with pytest.raises(RuntimeError, match="no optimizer parameter"):
        step(x, t)


@pytest.mark.usefixtures("inference")
def test_a_compiled_module_answers_with_a_submodules_new_weights() -> None:
    # CompiledModule watched the model itself only: a load into one of its
    # submodules left it answering with the old weights.
    x, _ = _data()
    model = _model()
    compiled = lucid.compile(model)
    compiled(x)
    _assign_load(model[0], 2)
    np.testing.assert_allclose(compiled(x).numpy(), model(x).numpy(), rtol=_TOL, atol=_TOL)
