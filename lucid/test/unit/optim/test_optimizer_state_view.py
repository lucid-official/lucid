"""``Optimizer.state`` as a view of where the state lives (LCD-812).

The structure against the reference and the "never stale" property across
``state_dict`` / ``load_state_dict`` are in ``test_optimizer_initial_state``;
this file covers what the view does with writes, copies and an optimizer
that keeps its state in Python.
"""

import copy
import pickle
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim


def _params() -> list[nn.Parameter]:
    return [nn.Parameter(lucid.ones(3, 2)), nn.Parameter(lucid.ones(4))]


def _step(opt: optim.Optimizer, params: list[nn.Parameter], scale: float = 1.0) -> None:
    for p in params:
        p.grad = lucid.ones(*p.shape) * scale
    opt.step()


def test_a_parameter_without_state_reads_as_empty_and_is_not_listed() -> None:
    params = _params()
    opt = optim.Adam(params, lr=0.1)
    assert len(opt.state) == 0 and params[0] not in opt.state
    assert dict(opt.state[params[0]]) == {}
    params[0].grad = lucid.ones(3, 2)
    opt.step()
    assert list(opt.state) == [params[0]]
    assert params[1] not in opt.state


def test_a_tensor_that_is_not_the_optimizers_is_not_a_key() -> None:
    opt = optim.Adam(_params(), lr=0.1)
    with pytest.raises(KeyError):
        opt.state[nn.Parameter(lucid.ones(2))]


def test_an_entry_read_before_a_step_follows_it() -> None:
    params = _params()
    opt = optim.Adam(params, lr=0.1)
    _step(opt, params)
    entry = opt.state[params[0]]
    assert int(entry["step"]) == 1
    _step(opt, params)
    assert int(entry["step"]) == 2, "the entry is read each time, not copied once"


def test_assigning_an_existing_key_writes_through_to_the_engine() -> None:
    params = _params()
    opt = optim.Adam(params, lr=0.1)
    _step(opt, params)
    opt.state[params[0]]["exp_avg"] = lucid.full((3, 2), 7.0)
    assert np.all(opt.state[params[0]]["exp_avg"].numpy() == 7.0)
    assert np.all(opt.state_dict()["state"][0]["exp_avg"] == 7.0)
    assert np.all(opt.state[params[1]]["exp_avg"].numpy() != 7.0), "only that parameter's"


def test_a_write_of_the_wrong_shape_is_refused_and_changes_nothing() -> None:
    params = _params()
    opt = optim.Adam(params, lr=0.1)
    _step(opt, params)
    before = opt.state[params[0]]["exp_avg"].numpy().copy()
    with pytest.raises(lucid._C.engine.ShapeMismatch):
        opt.state[params[0]]["exp_avg"] = lucid.ones(5)
    np.testing.assert_array_equal(opt.state[params[0]]["exp_avg"].numpy(), before)


def test_writes_that_cannot_work_say_why() -> None:
    params = _params()
    opt = optim.Adam(params, lr=0.1)
    with pytest.raises(RuntimeError, match="stepped"):
        opt.state[params[0]]["exp_avg"] = lucid.ones(3, 2)
    _step(opt, params)
    with pytest.raises(KeyError, match="no state entry"):
        opt.state[params[0]]["not_a_buffer"] = lucid.ones(3, 2)
    with pytest.raises(TypeError, match="load_state_dict"):
        del opt.state[params[0]]["exp_avg"]
    with pytest.raises(TypeError, match="load_state_dict"):
        opt.state[params[0]] = {}
    with pytest.raises(TypeError, match="load_state_dict"):
        del opt.state[params[0]]


def test_state_follows_a_conversion() -> None:
    model = nn.Linear(3, 2)
    opt = optim.Adam(model.parameters(), lr=0.1)
    model(lucid.ones(1, 3)).sum().backward()
    opt.step()
    model.double()
    model(lucid.ones(1, 3).double()).sum().backward()
    opt.step()
    assert opt.state[model.weight]["exp_avg"].dtype == lucid.float64


@pytest.mark.parametrize("how", ["deepcopy", "pickle"])
def test_a_copy_has_the_state_by_its_own_parameters_and_the_source_stays_live(how: str) -> None:
    params = _params()
    opt = optim.Adam(params, lr=0.1)
    _step(opt, params)
    twin = copy.deepcopy(opt) if how == "deepcopy" else pickle.loads(pickle.dumps(opt))
    twin_params = [p for g in twin.param_groups for p in g["params"]]
    assert all(a is not b for a, b in zip(params, twin_params, strict=True))
    for p, q in zip(params, twin_params, strict=True):
        for key, value in opt.state[p].items():
            np.testing.assert_array_equal(twin.state[q][key].numpy(), value.numpy())
    _step(opt, params)
    assert int(opt.state[params[0]]["step"]) == 2, "copying left the source's state live"
    assert int(twin.state[twin_params[0]]["step"]) == 1
    _step(twin, twin_params)
    assert int(twin.state[twin_params[0]]["step"]) == 2


class _Momentum(optim.Optimizer):
    """An optimizer with no engine, keeping its own state the way the reference's do."""

    def __init__(self, params: list[nn.Parameter]) -> None:
        super().__init__(params, {"lr": 0.1})

    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        with lucid.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    entry = self.state[p]
                    entry["buf"] = entry.get("buf", lucid.zeros_like(p)) * 0.5 + p.grad
                    entry["n"] = entry.get("n", 0) + 1
                    p.copy_(p - group["lr"] * entry["buf"])


def test_an_optimizer_that_keeps_its_state_in_python_reads_and_writes_it_by_parameter() -> None:
    params = _params()
    opt = _Momentum(params)
    _step(opt, params)
    _step(opt, params)
    assert opt.state[params[0]]["n"] == 2
    np.testing.assert_allclose(opt.state[params[0]]["buf"].numpy(), 1.5)


def test_python_held_state_round_trips_through_a_checkpoint_as_tensors() -> None:
    params = _params()
    opt = _Momentum(params)
    _step(opt, params)
    saved = opt.state_dict()
    assert saved["state"][0]["n"] == 1
    assert isinstance(saved["state"][0]["buf"], np.ndarray)

    twin_params = _params()
    twin = _Momentum(twin_params)
    twin.load_state_dict(copy.deepcopy(saved))
    assert isinstance(twin.state[twin_params[1]]["buf"], lucid.Tensor)
    np.testing.assert_array_equal(
        twin.state[twin_params[1]]["buf"].numpy(), opt.state[params[1]]["buf"].numpy()
    )
    assert twin.state[twin_params[1]]["n"] == 1


def test_python_held_state_is_refused_for_a_parameter_index_it_does_not_have() -> None:
    opt = _Momentum(_params())
    saved = opt.state_dict()
    saved["state"] = {5: {"n": 1}}
    with pytest.raises(IndexError):
        opt.load_state_dict(saved)
