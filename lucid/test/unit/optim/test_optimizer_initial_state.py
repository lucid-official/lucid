"""State an optimizer has before its first update, and scalars a checkpoint carries (LCD-747).

Two leftovers of the optimizer-state parity work (LCD-722):

* **Adagrad builds its state in the constructor.**  The reference framework
  gives every parameter ``step`` 0 and a ``sum`` filled with
  ``initial_accumulator_value`` when the optimizer is built, so its
  ``state_dict()["state"]`` is populated before the first step.  Lucid kept
  "no entry until updated" and exported ``{}``.  No other optimizer in the
  reference does this, and ``_state_load.initial_state`` is the one place that
  makes it.  A group added afterwards gets its state at its first
  update, in the reference as here.
* **A scalar in a checkpoint can be a numpy scalar** — 0-d numpy arithmetic
  hands back ``np.float32`` / ``np.int64``, not a 0-d array — and loading one
  raised ``TypeError`` out of the engine's tensor constructor.
"""

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine

# A matrix, a vector and a scalar: the scalar checks a 0-d parameter, whose
# state is 0-d like the step counter.
_SHAPES = ((3, 4), (4,), ())
_TOL = 1e-6


def _arrays(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.asarray(rng.standard_normal(s), dtype=np.float32) for s in _SHAPES]


def _as_array(v: object) -> np.ndarray:
    numpy = getattr(v, "numpy", None)
    return np.asarray(numpy() if callable(numpy) else v)


def _lucid_params(shapes: tuple[tuple[int, ...], ...] = _SHAPES, seed: int = 0) -> list[Any]:
    arrays = _arrays(seed) if shapes == _SHAPES else [np.ones(s, dtype=np.float32) for s in shapes]
    return [nn.Parameter(lucid.tensor(a.copy())) for a in arrays]


def _ref_params(ref: Any, params: list[Any]) -> list[Any]:
    return [ref.nn.Parameter(ref.from_numpy(p.numpy().copy())) for p in params]


def _feed(ref: Any, lucid_params: list[Any], ref_params: list[Any], seed: int) -> None:
    for p, rp, g in zip(lucid_params, ref_params, _arrays(100 + seed), strict=True):
        p.grad = lucid.tensor(g.copy())
        rp.grad = ref.from_numpy(g.copy())


def _assert_state_agrees(opt: Any, ref_opt: Any, what: str) -> None:
    """Every state entry the reference holds is Lucid's: same keys, shapes, values."""
    state, ref_state = opt.state_dict()["state"], ref_opt.state_dict()["state"]
    assert sorted(state) == sorted(ref_state), f"{what}: state keys"
    for idx, ref_entry in ref_state.items():
        assert set(state[idx]) == set(ref_entry), f"{what}: entry {idx}"
        for key, want in ref_entry.items():
            got, want_arr = _as_array(state[idx][key]), want.detach().numpy()
            assert got.shape == want_arr.shape, f"{what}: shape of {idx}/{key}"
            np.testing.assert_allclose(
                got.astype(np.float64),
                want_arr.astype(np.float64),
                rtol=_TOL,
                atol=_TOL,
                err_msg=f"{what}: state {idx}/{key}",
            )


# ── Adagrad's state at construction ───────────────────────────────────────

_ADAGRAD_KWARGS = [
    pytest.param({"lr": 0.1}, id="default"),
    pytest.param({"lr": 0.1, "initial_accumulator_value": 0.5}, id="initial-0.5"),
    pytest.param({"lr": 0.1, "lr_decay": 0.05, "weight_decay": 0.01}, id="decay"),
]


@pytest.mark.parametrize("kw", _ADAGRAD_KWARGS)
def test_adagrad_state_is_the_references_before_and_after_steps(
    kw: dict[str, Any], ref: Any
) -> None:
    params = _lucid_params()
    ref_params = _ref_params(ref, params)
    opt = optim.Adagrad(params, **kw)
    ref_opt = ref.optim.Adagrad(ref_params, **kw)

    state = opt.state_dict()["state"]
    assert sorted(state) == [0, 1, 2], "every parameter has an entry before the first step"
    _assert_state_agrees(opt, ref_opt, "before the first step")

    for k in range(3):
        _feed(ref, params, ref_params, k)
        opt.step()
        ref_opt.step()
        _assert_state_agrees(opt, ref_opt, f"after step {k + 1}")


def test_adagrad_state_after_add_param_group_is_the_references(ref: Any) -> None:
    params = _lucid_params()
    ref_params = _ref_params(ref, params)
    opt = optim.Adagrad(params[:2], lr=0.1, initial_accumulator_value=0.5)
    ref_opt = ref.optim.Adagrad(ref_params[:2], lr=0.1, initial_accumulator_value=0.5)

    opt.add_param_group({"params": [params[2]]})
    ref_opt.add_param_group({"params": [ref_params[2]]})
    # The new group's state starts at its first update, here and there.
    assert sorted(opt.state_dict()["state"]) == [0, 1]
    _assert_state_agrees(opt, ref_opt, "after add_param_group")

    for k in range(2):
        _feed(ref, params, ref_params, k)
        opt.step()
        ref_opt.step()
        _assert_state_agrees(opt, ref_opt, f"after step {k + 1}")

    # A group added to an optimizer that has stepped is the same.
    extra = _lucid_params(((2,),))
    ref_extra = _ref_params(ref, extra)
    opt.add_param_group({"params": extra})
    ref_opt.add_param_group({"params": ref_extra})
    _assert_state_agrees(opt, ref_opt, "after a late add_param_group")


def test_adagrad_state_follows_the_parameter_through_a_conversion() -> None:
    model = nn.Linear(3, 2)
    opt = optim.Adagrad(model.parameters(), lr=0.1, initial_accumulator_value=0.25)
    model.double()
    model(lucid.ones(1, 3).double()).sum().backward()
    opt.step()
    for entry in opt.state_dict()["state"].values():
        assert _as_array(entry["sum"]).dtype == np.float64
        assert _as_array(entry["step"]) == 1


def test_adagrad_round_trips_before_and_after_a_step() -> None:
    params = _lucid_params()
    opt = optim.Adagrad(params, lr=0.1, initial_accumulator_value=0.5)
    saved_fresh = copy.deepcopy(opt.state_dict())
    for p in params:
        p.grad = lucid.ones(p.shape)
    opt.step()
    saved_stepped = copy.deepcopy(opt.state_dict())

    twin_params = _lucid_params()
    twin = optim.Adagrad(twin_params, lr=0.1, initial_accumulator_value=0.5)
    for saved in (saved_stepped, saved_fresh):
        twin.load_state_dict(copy.deepcopy(saved))
        got = twin.state_dict()["state"]
        assert sorted(got) == sorted(saved["state"])
        for idx, entry in saved["state"].items():
            assert set(got[idx]) == set(entry)
            for key, want in entry.items():
                np.testing.assert_array_equal(_as_array(got[idx][key]), _as_array(want))


def test_adagrad_loading_an_empty_state_leaves_none_as_the_reference_does(ref: Any) -> None:
    params = _lucid_params()
    ref_params = _ref_params(ref, params)
    opt = optim.Adagrad(params, lr=0.1)
    ref_opt = ref.optim.Adagrad(ref_params, lr=0.1)
    empty = {"state": {}, "param_groups": opt.state_dict()["param_groups"]}
    ref_empty = {"state": {}, "param_groups": ref_opt.state_dict()["param_groups"]}
    opt.load_state_dict(copy.deepcopy(empty))
    ref_opt.load_state_dict(ref_empty)
    _assert_state_agrees(opt, ref_opt, "after loading an empty state")
    _feed(ref, params, ref_params, 0)
    opt.step()
    ref_opt.step()
    _assert_state_agrees(opt, ref_opt, "after the first step from nothing")


_OPTIMIZERS: list[tuple[str, dict[str, Any]]] = [
    ("SGD", {"lr": 0.1}),
    ("SGD", {"lr": 0.1, "momentum": 0.9}),
    ("Adam", {"lr": 0.05}),
    ("AdamW", {"lr": 0.05}),
    ("Adamax", {"lr": 0.05}),
    ("Adagrad", {"lr": 0.1}),
    ("Adadelta", {"lr": 0.5}),
    ("RMSprop", {"lr": 0.01, "momentum": 0.9, "centered": True}),
    ("NAdam", {"lr": 0.05}),
    ("RAdam", {"lr": 0.05}),
    ("ASGD", {"lr": 0.05, "t0": 2.0}),
    ("Rprop", {"lr": 0.05}),
]


def _id(case: tuple[str, dict[str, Any]]) -> str:
    return case[0] + "".join(f"-{k}={v}" for k, v in case[1].items())


@pytest.mark.parametrize("case", [pytest.param(c, id=_id(c)) for c in _OPTIMIZERS])
def test_an_optimizers_state_before_its_first_step_is_the_references(
    case: tuple[str, dict[str, Any]], ref: Any
) -> None:
    """Only Adagrad has state at construction — in the reference, so here."""
    name, kw = case
    params = _lucid_params()
    opt = getattr(optim, name)(params, **kw)
    ref_opt = getattr(ref.optim, name)(_ref_params(ref, params), **kw)
    _assert_state_agrees(opt, ref_opt, "before the first step")


# ── numpy scalars in a checkpoint ─────────────────────────────────────────


def _as_numpy_scalars(state: dict[int, dict[str, Any]], kind: type) -> None:
    """Make every 0-d entry of ``state`` a ``kind`` scalar (only counts, for an integer kind)."""
    for entry in state.values():
        for key, value in entry.items():
            arr = np.asarray(value)
            if arr.ndim == 0 and (kind is not np.int64 or np.issubdtype(arr.dtype, np.integer)):
                entry[key] = kind(arr[()])


def _step(opt: Any, params: list[Any], seed: int) -> None:
    for p, g in zip(params, _arrays(100 + seed), strict=True):
        p.grad = lucid.tensor(g.copy())
    opt.step()


@pytest.mark.parametrize("kind", [np.float32, np.float64, np.int64], ids=lambda k: k.__name__)
@pytest.mark.parametrize("case", [pytest.param(c, id=_id(c)) for c in _OPTIMIZERS])
def test_a_numpy_scalar_state_entry_loads_like_a_0d_array(
    case: tuple[str, dict[str, Any]], kind: type
) -> None:
    name, kw = case
    params = _lucid_params()
    opt = getattr(optim, name)(params, **kw)
    for k in range(3):
        _step(opt, params, k)
    saved = copy.deepcopy(opt.state_dict())
    as_scalars = copy.deepcopy(saved)
    _as_numpy_scalars(as_scalars["state"], kind)

    twin_params = _lucid_params()
    with lucid.no_grad():
        for p, q in zip(params, twin_params, strict=True):
            q.copy_(p)
    twin = getattr(optim, name)(twin_params, **kw)
    twin.load_state_dict(as_scalars)

    reloaded = twin.state_dict()["state"]
    assert sorted(reloaded) == sorted(saved["state"])
    for idx, entry in saved["state"].items():
        for key, want in entry.items():
            np.testing.assert_allclose(
                _as_array(reloaded[idx][key]).astype(np.float64),
                _as_array(want).astype(np.float64),
                rtol=1e-6,
                err_msg=f"{idx}/{key}",
            )
    _step(opt, params, 3)
    _step(twin, twin_params, 3)
    for a, b in zip(params, twin_params, strict=True):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-6)


def test_a_numpy_scalar_step_is_that_parameters_own_count() -> None:
    """``np.float64`` is a ``float``; it must not become one count for the whole group."""
    params = _lucid_params()
    opt = optim.Adam(params, lr=0.05)
    for k in range(2):
        _step(opt, params, k)
    saved = copy.deepcopy(opt.state_dict())
    saved["state"][0]["step"] = np.float64(2.0)
    saved["state"][1]["step"] = np.float32(7.0)
    saved["state"][2]["step"] = np.int64(4)
    opt.load_state_dict(saved)
    steps = [int(_as_array(opt.state_dict()["state"][i]["step"])) for i in range(3)]
    assert steps == [2, 7, 4]


def test_an_array_with_one_element_is_not_taken_for_a_scalar() -> None:
    """A buffer saved as a (1,) array is a buffer; its shape is checked against the parameter's."""
    params = _lucid_params()
    opt = optim.Adam(params, lr=0.05)
    _step(opt, params, 0)
    saved = copy.deepcopy(opt.state_dict())
    saved["state"][1]["exp_avg"] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(_C_engine.ShapeMismatch):
        opt.load_state_dict(saved)


# ── what `opt.state` shows ────────────────────────────────────────────────


def _assert_no_host_arrays(value: object, where: str) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _assert_no_host_arrays(v, f"{where}/{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _assert_no_host_arrays(v, f"{where}[{i}]")
    else:
        assert isinstance(value, (lucid.Tensor, int, float, bool)), (
            f"{where} is a {type(value).__name__}, not a tensor or a Python number"
        )


@pytest.mark.parametrize("case", [pytest.param(c, id=_id(c)) for c in _OPTIMIZERS])
def test_opt_state_holds_no_host_arrays_before_or_after_a_step(
    case: tuple[str, dict[str, Any]],
) -> None:
    """``opt.state`` shows the optimizer's own state; host arrays belong to checkpoints."""
    name, kw = case
    params = _lucid_params()
    opt = getattr(optim, name)(params, **kw)
    _assert_no_host_arrays(opt.state, "before the first step")
    for k in range(2):
        _step(opt, params, k)
        _assert_no_host_arrays(opt.state, f"after step {k + 1}")


def test_lbfgs_state_holds_no_host_arrays_after_a_step() -> None:
    params = _lucid_params()
    opt = optim.LBFGS(params, lr=0.1)

    def closure() -> lucid.Tensor:
        opt.zero_grad()
        loss = sum((p**2).sum() for p in params)
        loss.backward()
        return loss

    opt.step(closure)
    _assert_no_host_arrays(opt.state, "LBFGS")


def test_adagrad_seed_shows_in_state_and_follows_the_steps() -> None:
    params = _lucid_params()
    opt = optim.Adagrad(params, lr=0.1, initial_accumulator_value=0.5)
    for p in params:
        assert float(opt.state[p]["sum"].min()) == 0.5, "the seed, before the engines exist"
    _step(opt, params, 0)
    for p in params:
        assert float(opt.state[p]["sum"].min()) > 0.5, "the live sum, not the seed"
    sums = [_as_array(e["sum"]) for e in opt.state_dict()["state"].values()]
    assert all(np.all(s > 0.5) for s in sums), "the exported sum is the live one"


def test_adagrad_parameter_without_a_gradient_keeps_its_seeded_state(ref: Any) -> None:
    params = _lucid_params()
    ref_params = _ref_params(ref, params)
    opt = optim.Adagrad(params, lr=0.1, initial_accumulator_value=0.5)
    ref_opt = ref.optim.Adagrad(ref_params, lr=0.1, initial_accumulator_value=0.5)
    _feed(ref, params, ref_params, 0)
    for p, rp in zip(params[1:], ref_params[1:], strict=True):
        p.grad = None
        rp.grad = None
    opt.step()
    ref_opt.step()
    _assert_state_agrees(opt, ref_opt, "a parameter that never got a gradient")


# ── `opt.state` is live, whatever was called on the optimizer (LCD-812) ───


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _kind(value: object) -> object:
    """A state value's structure: ``"number"``, ``"tensor"`` or a list of them."""
    if isinstance(value, (list, tuple)):
        return [_kind(v) for v in value]
    return "number" if _is_number(value) else "tensor"


def _flat(value: object) -> list[np.ndarray]:
    """The arrays inside a state value (a tensor, a number or a list of them)."""
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in _flat(v)]
    return [_as_array(value)]


def _state_by_index(opt: Any, params: list[Any]) -> dict[int, dict[str, Any]]:
    index = {id(p): i for i, p in enumerate(params)}
    return {index[id(p)]: dict(opt.state[p]) for p in opt.state}


def _assert_state_is_live_and_the_references(
    opt: Any, params: list[Any], ref_opt: Any, where: str
) -> None:
    """``opt.state`` is tensors and numbers, equals ``state_dict()`` and has the reference's shape."""
    _assert_no_host_arrays(opt.state, where)
    live = _state_by_index(opt, params)
    saved = opt.state_dict()["state"]
    assert sorted(live) == sorted(saved), f"{where}: parameters with state"
    for idx, entry in saved.items():
        assert set(live[idx]) == set(entry), f"{where}: keys of {idx}"
        for key, want in entry.items():
            for got_arr, want_arr in zip(_flat(live[idx][key]), _flat(want), strict=True):
                np.testing.assert_array_equal(got_arr, want_arr, err_msg=f"{where}: {idx}/{key}")
    ref_state = ref_opt.state_dict()["state"]
    assert sorted(live) == sorted(ref_state), f"{where}: reference parameters with state"
    for idx, ref_entry in ref_state.items():
        assert set(live[idx]) == set(ref_entry), f"{where}: reference keys of {idx}"
        for key, want in ref_entry.items():
            assert _kind(live[idx][key]) == _kind(want), f"{where}: type of {idx}/{key}"


@pytest.mark.parametrize("case", [pytest.param(c, id=_id(c)) for c in _OPTIMIZERS])
def test_opt_state_is_live_through_step_state_dict_load_and_step(
    case: tuple[str, dict[str, Any]], ref: Any
) -> None:
    name, kw = case
    params = _lucid_params()
    ref_params = _ref_params(ref, params)
    opt = getattr(optim, name)(params, **kw)
    ref_opt = getattr(ref.optim, name)(ref_params, **kw)
    _assert_state_is_live_and_the_references(opt, params, ref_opt, "before the first step")

    for k in range(2):
        _feed(ref, params, ref_params, k)
        opt.step()
        ref_opt.step()
        _assert_state_is_live_and_the_references(opt, params, ref_opt, f"after step {k + 1}")

    saved = copy.deepcopy(opt.state_dict())
    ref_saved = ref_opt.state_dict()
    _assert_state_is_live_and_the_references(opt, params, ref_opt, "after state_dict")
    before = {
        i: {k: _as_array(v).copy() for k, v in e.items()}
        for i, e in _state_by_index(opt, params).items()
    }
    _feed(ref, params, ref_params, 2)
    opt.step()
    ref_opt.step()
    _assert_state_is_live_and_the_references(opt, params, ref_opt, "after state_dict and a step")
    after = _state_by_index(opt, params)
    for idx, entry in before.items():
        assert any(
            not np.array_equal(_as_array(after[idx][key]), want) for key, want in entry.items()
        ), f"parameter {idx}: opt.state is still the state at state_dict()"

    twin_params = _lucid_params()
    ref_twin_params = _ref_params(ref, twin_params)
    twin = getattr(optim, name)(twin_params, **kw)
    ref_twin = getattr(ref.optim, name)(ref_twin_params, **kw)
    twin.load_state_dict(saved)
    ref_twin.load_state_dict(ref_saved)
    _assert_state_is_live_and_the_references(twin, twin_params, ref_twin, "after load_state_dict")
    for k in range(2):
        _feed(ref, twin_params, ref_twin_params, 3 + k)
        twin.step()
        ref_twin.step()
        _assert_state_is_live_and_the_references(
            twin, twin_params, ref_twin, f"after load_state_dict and step {k + 1}"
        )


def test_lbfgs_state_is_live_and_the_references(ref: Any) -> None:
    params = _lucid_params()
    ref_params = _ref_params(ref, params)
    opt = optim.LBFGS(params, lr=0.1)
    ref_opt = ref.optim.LBFGS(ref_params, lr=0.1)

    def closure() -> lucid.Tensor:
        opt.zero_grad()
        loss = sum((p**2).sum() for p in params)
        loss.backward()
        return loss

    def ref_closure() -> Any:
        ref_opt.zero_grad()
        loss = sum((p**2).sum() for p in ref_params)
        loss.backward()
        return loss

    assert len(opt.state) == 0
    opt.step(closure)
    ref_opt.step(ref_closure)
    assert list(opt.state) == [params[0]]
    entry, ref_entry = dict(opt.state[params[0]]), ref_opt.state[ref_params[0]]
    # ``al`` is the reference's scratch buffer for the two-loop recursion.
    assert set(entry) == set(ref_entry) - {"al"}
    for key in entry:
        assert _kind(entry[key]) == _kind(ref_entry[key]), key
    _assert_no_host_arrays(opt.state, "LBFGS")
    assert dict(opt.state[params[1]]) == {}, "a parameter without state reads as empty"

    saved = opt.state_dict()
    before = int(opt.state[params[0]]["n_iter"])
    opt.step(closure)
    assert int(opt.state[params[0]]["n_iter"]) > before, (
        "opt.state follows the step after state_dict"
    )
    assert saved["state"][0]["n_iter"] == before
