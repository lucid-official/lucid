"""Every optimizer's state follows its parameters through a conversion (LCD-77).

``module.double()``, ``.half()`` and ``.to(device)`` between two ``step``
calls give each parameter a new buffer of the new dtype or on the new
device.  The engine optimizers carry their state across when they rebuild
(``Optimizer._rebuild_engine_optim``); the two optimizers that keep their
state in Python — ``SparseAdam`` and ``LBFGS`` — kept theirs where it was,
and the next step mixed float32 state with float64 parameters, or CPU state
with Metal parameters, and raised ``DtypeMismatch`` / ``DeviceMismatch``.

The matrix is every optimizer × every conversion.  Two oracles:

* the uninterrupted run — a conversion that keeps the state changes the
  trajectory by rounding only, where a reset or a raise does not;
* the reference with its own state-follows-parameter rule: it re-homes
  optimizer state when that state is loaded, so the reference run converts
  the model and reloads its optimizer's ``state_dict``.
"""

import warnings
from typing import Any

import numpy as np
import pytest

import lucid

_W0 = np.array([[0.5, -1.0, 0.25], [1.5, 0.75, -0.5]], dtype=np.float32)
_B0 = np.array([0.1, -0.2], dtype=np.float32)
_X = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 1.0]], dtype=np.float32)
# SparseAdam runs over an embedding whose every row is looked up each step,
# so the reference's sparse update touches the same rows as Lucid's dense one.
_E0 = np.array([[0.5, -1.0], [1.5, 0.75], [-0.5, 0.25]], dtype=np.float32)
_IDX = [0, 1, 2, 1]
_STEPS = 4

_OPTIMIZERS: list[tuple[str, dict[str, Any]]] = [
    ("SGD", {"lr": 0.05, "momentum": 0.9}),
    ("Adam", {"lr": 0.05}),
    ("AdamW", {"lr": 0.05, "weight_decay": 0.01}),
    ("RMSprop", {"lr": 0.01}),
    ("Adagrad", {"lr": 0.05}),
    ("Adadelta", {"lr": 1.0}),
    ("Adamax", {"lr": 0.05}),
    ("RAdam", {"lr": 0.05}),
    ("NAdam", {"lr": 0.05}),
    ("ASGD", {"lr": 0.05}),
    ("Rprop", {"lr": 0.05}),
    ("SparseAdam", {"lr": 0.05}),
    ("LBFGS", {"lr": 0.1, "max_iter": 3, "history_size": 3}),
]
_PYTHON_SIDE = {"SparseAdam", "LBFGS"}

# (start device, conversion applied after steps 1 and 2)
_MOVES: dict[str, tuple[str, str]] = {
    "cpu-double": ("cpu", "double"),
    "cpu-half": ("cpu", "half"),
    "cpu-to-metal": ("cpu", "metal"),
    "metal-to-cpu": ("metal", "cpu"),
    "metal-half": ("metal", "half"),
}


_CASES = [
    pytest.param(name, kwargs, move, id=f"{name}-{move}")
    for name, kwargs in _OPTIMIZERS
    for move in _MOVES
]


def _convert(model: Any, how: str) -> None:
    if how == "double":
        model.double()
    elif how == "half":
        model.half()
    else:
        model.to(how)


class _Run:
    """One model and optimizer in ``lib`` (Lucid or the reference)."""

    def __init__(self, lib: Any, name: str, kwargs: dict[str, Any], device: str) -> None:
        self.lib = lib
        self.name = name
        if name == "SparseAdam":
            sparse = {} if lib is lucid else {"sparse": True}
            self.model = lib.nn.Embedding(3, 2, **sparse)
            self.model.weight = lib.nn.Parameter(lib.tensor(_E0.copy()))
        else:
            self.model = lib.nn.Linear(3, 2)
            self.model.weight = lib.nn.Parameter(lib.tensor(_W0.copy()))
            self.model.bias = lib.nn.Parameter(lib.tensor(_B0.copy()))
        if lib is lucid:
            self.model.to(device)
        self.opt = getattr(lib.optim, name)(self.model.parameters(), **kwargs)

    def _loss(self) -> Any:
        w = self.model.weight
        if self.name == "SparseAdam":
            x = self.lib.tensor(_IDX)
            if self.lib is lucid:
                x = x.to(w.device)
            return (self.model(x) ** 2).sum()
        x = self.lib.tensor(_X).to(w.dtype)
        if self.lib is lucid:
            x = x.to(w.device)
        return (self.model(x) ** 2).sum()

    def step(self) -> np.ndarray:
        def closure() -> Any:
            self.opt.zero_grad()
            loss = self._loss()
            loss.backward()
            return loss

        if self.name == "LBFGS":
            self.opt.step(closure)
        else:
            closure()
            self.opt.step()
        w = self.model.weight.detach()
        if self.lib is lucid:
            return w.to("cpu").to(lucid.float64).numpy().copy()
        return w.double().numpy().copy()


def _trajectory(run: _Run, convert: Any) -> np.ndarray:
    out = []
    for i in range(_STEPS):
        out.append(run.step())
        if i in (0, 1):
            convert(run)
    return np.stack(out)


def _lucid_trajectory(name: str, kwargs: dict[str, Any], move: str) -> np.ndarray:
    device, how = _MOVES[move]
    # The two conversions go there and back, so both directions are exercised.
    back = {"double": "float", "half": "float", "metal": "cpu", "cpu": "metal"}
    hops = [how, back[how]]

    def convert(r: _Run) -> None:
        target = hops.pop(0)
        if target == "float":
            r.model.float()
        else:
            _convert(r.model, target)

    run = _Run(lucid, name, kwargs, device)
    with warnings.catch_warnings():
        # Every optimizer hands its state over, so nothing may be reset.
        warnings.simplefilter("error", RuntimeWarning)
        return _trajectory(run, convert)


def _tolerance(move: str) -> float:
    return 2e-2 if "half" in move else 1e-5


@pytest.mark.parametrize(("name", "kwargs", "move"), _CASES)
def test_conversion_keeps_the_uninterrupted_trajectory(
    name: str, kwargs: dict[str, Any], move: str
) -> None:
    got = _lucid_trajectory(name, kwargs, move)
    want = _trajectory(_Run(lucid, name, kwargs, _MOVES[move][0]), lambda r: None)
    # Every step moved the weights, along the path the state dictates.
    assert np.all(np.abs(np.diff(got, axis=0)).max(axis=(1, 2)) > 0)
    np.testing.assert_allclose(got, want, rtol=0, atol=_tolerance(move))


@pytest.mark.parity
@pytest.mark.parametrize(("name", "kwargs", "move"), _CASES)
def test_conversion_follows_the_reference(
    name: str, kwargs: dict[str, Any], move: str, ref: Any
) -> None:
    _, how = _MOVES[move]
    back = {"double": "float", "half": "float"}
    hops = [how, back.get(how, how)]

    def convert(r: _Run) -> None:
        # The reference runs on the CPU; only a dtype conversion applies.
        target = hops.pop(0)
        if target in ("double", "half", "float"):
            getattr(r.model, target)()
        # Its rule: state is re-homed to the parameters when it is loaded.
        r.opt.load_state_dict(r.opt.state_dict())

    want = _trajectory(_Run(ref, name, kwargs, "cpu"), convert)
    got = _lucid_trajectory(name, kwargs, move)
    np.testing.assert_allclose(got, want, rtol=0, atol=_tolerance(move))
