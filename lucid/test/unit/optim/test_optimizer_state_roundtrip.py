"""Optimizer state survives a checkpoint, and has the reference framework's shape.

Eight engine optimizers — RMSprop, Adagrad, Adadelta, Adamax, RAdam, NAdam,
ASGD and Rprop — used to save no state at all: ``state_dict()["state"]`` was
empty, and a run resumed from a checkpoint silently restarted every running
average, accumulator and step count.  These pin the repair:

* a run saved after ``N`` steps and restored into a fresh optimizer continues
  **bit-identically** with the uninterrupted run, on every device, through
  ``state_dict``, ``copy.deepcopy`` and ``lucid.save`` / ``lucid.load``;
* the per-parameter keys and shapes are the reference framework's, and so
  are the values;
* each parameter counts its own steps, as the reference framework does: a
  frozen first parameter no longer holds the others at step 0 (which divided
  by zero in the bias correction), and a parameter unfrozen late starts its
  bias correction at step 1.

Adam, AdamW and SGD ride along: they moved to the same per-parameter step
counter.
"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine

# A matrix, a vector and a scalar: the scalar checks 0-d parameters, whose
# state is 0-d like the step counter.
_SHAPES = ((3, 4), (4,), ())
# RAdam's rectified branch starts at step 6, the first step after the save.
_N_BEFORE = 5
_N_AFTER = 3


def _arrays(seed: int, dtype: type = np.float32) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.asarray(rng.standard_normal(s), dtype=dtype) for s in _SHAPES]


def _init(dtype: type = np.float32) -> list[np.ndarray]:
    return _arrays(0, dtype)


def _grads(dtype: type = np.float32) -> list[list[np.ndarray]]:
    return [_arrays(100 + k, dtype) for k in range(_N_BEFORE + _N_AFTER)]


def _tensor(a: np.ndarray, device: str) -> lucid.Tensor:
    # Built on the CPU and moved: ``lucid.tensor(a, device=...)`` turns a 0-d
    # array into shape (1,).
    return lucid.tensor(a.copy()).to(device)


def _params(arrays: list[np.ndarray], device: str) -> list[nn.Parameter]:
    return [nn.Parameter(_tensor(a, device)) for a in arrays]


def _feed(
    params: list[nn.Parameter],
    grads: list[np.ndarray],
    device: str,
    frozen: tuple[int, ...] = (),
) -> None:
    """Give every parameter its gradient for one step; ``frozen`` get none."""
    for i, (p, g) in enumerate(zip(params, grads)):
        p.grad = None if i in frozen else _tensor(g, device)


def _values(params: list[nn.Parameter]) -> list[np.ndarray]:
    return [np.array(p.numpy(), copy=True) for p in params]


def _as_array(v: object) -> np.ndarray:
    """A state entry as an array, whether the loader hands back arrays or tensors."""
    numpy = getattr(v, "numpy", None)
    return np.asarray(numpy() if callable(numpy) else v)


def _assert_bit_identical(a: list[np.ndarray], b: list[np.ndarray], what: str) -> None:
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.dtype == y.dtype and x.shape == y.shape, f"{what}: param {i}"
        assert np.array_equal(x, y), f"{what}: param {i} differs by {np.max(np.abs(x - y))}"


# (class, kwargs) — the options that change which state an optimizer keeps,
# and weight decay, which changes the gradient the state is built from.
_CASES = [
    ("RMSprop", {}),
    ("RMSprop", {"momentum": 0.9, "weight_decay": 1e-2}),
    ("Adagrad", {"lr": 1e-1}),
    ("Adagrad", {"lr": 1e-1, "weight_decay": 1e-2}),
    ("Adadelta", {}),
    ("Adadelta", {"weight_decay": 1e-2}),
    ("Adamax", {}),
    ("Adamax", {"weight_decay": 1e-2}),
    ("RAdam", {}),
    ("RAdam", {"weight_decay": 1e-2}),
    ("NAdam", {}),
    ("NAdam", {"weight_decay": 1e-2}),
    ("ASGD", {"t0": 2.0}),  # averaging active before the save
    ("ASGD", {"weight_decay": 1e-2}),  # averaging not yet started
    ("Rprop", {}),
    ("Rprop", {"etas": (0.3, 1.5), "step_sizes": (1e-4, 1.0)}),
    # Moved to the per-parameter step counter alongside the eight.
    ("Adam", {}),
    ("Adam", {"amsgrad": True}),
    ("AdamW", {}),
    ("SGD", {"lr": 1e-2, "momentum": 0.9}),
]


def _case_id(case: tuple[str, dict[str, object]]) -> str:
    name, kw = case
    return name + "".join(f"-{k}={v}" for k, v in kw.items())


_CASE_PARAMS = [pytest.param(c, id=_case_id(c)) for c in _CASES]


def _make(name: str, kw: dict[str, object], params: list[nn.Parameter]) -> optim.Optimizer:
    return getattr(optim, name)(params, **kw)


def _resume(
    case: tuple[str, dict[str, object]],
    device: str,
    transfer: Callable[[dict[str, Any]], Any],
    dtype: type = np.float32,
) -> None:
    """Run ``_N_BEFORE`` steps, checkpoint through ``transfer``, restore into a
    fresh optimizer, then step both and require identical parameters."""
    name, kw = case
    grads = _grads(dtype)
    live = _params(_init(dtype), device)
    opt = _make(name, kw, live)
    for g in grads[:_N_BEFORE]:
        _feed(live, g, device)
        opt.step()

    saved = opt.state_dict()
    assert len(saved["state"]) == len(_SHAPES), f"{name} saved no state"
    resumed = _params(_values(live), device)
    opt2 = _make(name, kw, resumed)
    opt2.load_state_dict(transfer(saved))

    # The restored optimizer reports the state it was given.
    again = opt2.state_dict()["state"]
    for idx, entry in saved["state"].items():
        assert set(again[idx]) == set(entry)
        for key, value in entry.items():
            assert np.array_equal(_as_array(again[idx][key]), _as_array(value)), (
                idx,
                key,
            )

    for k, g in enumerate(grads[_N_BEFORE:]):
        _feed(live, g, device)
        opt.step()
        _feed(resumed, g, device)
        opt2.step()
        _assert_bit_identical(_values(live), _values(resumed), f"{name} step {_N_BEFORE + k + 1}")


@pytest.mark.parametrize("case", _CASE_PARAMS)
def test_a_restored_optimizer_continues_bit_identically(
    case: tuple[str, dict[str, object]], device: str
) -> None:
    _resume(case, device, lambda sd: sd)


@pytest.mark.parametrize(
    "case",
    [pytest.param(("NAdam", {}), id="NAdam"), pytest.param(("RAdam", {}), id="RAdam")],
)
def test_a_float64_run_round_trips_exactly(case: tuple[str, dict[str, object]]) -> None:
    # F64 parameters keep their scalar state (NAdam's mu_product) at F64.
    _resume(case, "cpu", lambda sd: sd, dtype=np.float64)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(("NAdam", {}), id="NAdam"),
        pytest.param(("RMSprop", {"momentum": 0.9}), id="RMSprop"),
    ],
)
def test_a_deep_copied_state_dict_restores(
    case: tuple[str, dict[str, object]], device: str
) -> None:
    _resume(case, device, copy.deepcopy)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(("SGD", {"momentum": 0.9}), id="SGD"),
        pytest.param(("Adam", {}), id="Adam"),
        pytest.param(("NAdam", {}), id="NAdam"),
        pytest.param(("Rprop", {}), id="Rprop"),
    ],
)
def test_a_state_dict_saved_to_disk_restores(
    case: tuple[str, dict[str, object]], device: str, tmp_path: Any
) -> None:
    path = str(tmp_path / "optim.lucid")

    def through_disk(sd: dict[str, Any]) -> Any:
        lucid.save(sd, path)
        return lucid.load(path, weights_only=True)

    _resume(case, device, through_disk)


def test_a_checkpoint_with_an_integer_step_still_restores(device: str) -> None:
    # Checkpoints written before the per-parameter counter stored ``step`` as
    # one Python int per parameter group.
    def as_int_step(sd: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(sd)
        for entry in out["state"].values():
            entry["step"] = int(_as_array(entry["step"]))
        return out

    _resume(("Adam", {}), device, as_int_step)


# ── the engine's own round trip ────────────────────────────────────────────


_EngineFactory = Callable[[list[Any]], Any]


def _engine_rmsprop_centered(momentum: float) -> _EngineFactory:
    return lambda impls: _C_engine.RMSprop(impls, lr=1e-2, momentum=momentum, centered=True)


def _engine_asgd_averaging() -> _EngineFactory:
    return lambda impls: _C_engine.ASGD(impls, lr=1e-2, t0=2.0)


@pytest.mark.parametrize(
    "make, keys",
    [
        pytest.param(
            _engine_rmsprop_centered(0.0),
            {"step", "square_avg", "grad_avg"},
            id="RMSprop-centered",
        ),
        pytest.param(
            _engine_rmsprop_centered(0.9),
            {"step", "square_avg", "momentum_buffer", "grad_avg"},
            id="RMSprop-centered-momentum",
        ),
        pytest.param(
            _engine_asgd_averaging(),
            {"step", "eta", "mu", "ax"},
            id="ASGD-averaging",
        ),
    ],
)
def test_engine_state_round_trips(make: _EngineFactory, keys: set[str], device: str) -> None:
    # state_buffers -> load_state_buffers on the engine itself, without the
    # Python loader in between.
    grads = _grads()
    live = _params(_init(), device)
    eng = make([p._impl for p in live])
    for g in grads[:_N_BEFORE]:
        _feed(live, g, device)
        eng.step()

    bufs = eng.state_buffers()
    assert {name for name, _ in bufs} == keys
    resumed = _params(_values(live), device)
    eng2 = make([p._impl for p in resumed])
    eng2.load_state_buffers(bufs)
    for g in grads[_N_BEFORE:]:
        _feed(live, g, device)
        eng.step()
        _feed(resumed, g, device)
        eng2.step()
        _assert_bit_identical(_values(live), _values(resumed), "engine round trip")


# ── per-parameter step counters ─────────────────────────────────────────────

_STEP_USERS = ["Adam", "AdamW", "Adamax", "NAdam", "RAdam"]


@pytest.mark.parametrize("name", _STEP_USERS)
def test_a_frozen_first_parameter_does_not_stall_the_others(name: str, device: str) -> None:
    # The step counter used to advance only when slot 0 updated, so freezing
    # the first parameter left every other one at step 0 — and 1 - beta1**0
    # is a zero denominator.
    params = _params(_init(), device)
    opt = _make(name, {}, params)
    for g in _grads()[:3]:
        _feed(params, g, device, frozen=(0,))
        opt.step()
    for p in params[1:]:
        assert np.all(np.isfinite(p.numpy()))
    np.testing.assert_array_equal(params[0].numpy(), _init()[0])
    state = opt.state_dict()["state"]
    assert 0 not in state
    assert [int(_as_array(state[i]["step"])) for i in (1, 2)] == [3, 3]


def _late_unfreeze_run(
    make: Callable[[list[Any]], Any],
    params: list[Any],
    grads: list[list[np.ndarray]],
    to_grad: Callable[[np.ndarray], Any],
) -> Any:
    """Step five times; parameter 0 receives no gradient for the first two."""
    opt = make(params)
    for k, g in enumerate(grads[:5]):
        for i, (p, gi) in enumerate(zip(params, g)):
            p.grad = None if (i == 0 and k < 2) else to_grad(gi)
        opt.step()
    return opt


@pytest.mark.parametrize("name", _STEP_USERS)
def test_a_late_parameter_follows_the_reference_trajectory(
    name: str, device: str, ref: Any
) -> None:
    # The reference framework counts steps per parameter: one unfrozen after
    # two steps runs its bias correction from step 1, not step 3.
    grads = _grads()
    params = _params(_init(), device)
    opt = _late_unfreeze_run(
        lambda ps: _make(name, {}, ps), params, grads, lambda g: _tensor(g, device)
    )
    ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in _init()]
    ref_opt = _late_unfreeze_run(
        lambda ps: getattr(ref.optim, name)(ps),
        ref_params,
        grads,
        lambda g: ref.from_numpy(g.copy()),
    )
    for p, rp in zip(params, ref_params):
        np.testing.assert_allclose(p.numpy(), rp.detach().numpy(), rtol=1e-5, atol=1e-6)
    steps = [int(_as_array(e["step"])) for _, e in sorted(opt.state_dict()["state"].items())]
    ref_steps = [int(e["step"]) for _, e in sorted(ref_opt.state_dict()["state"].items())]
    assert steps == ref_steps == [3, 5, 5]


@pytest.mark.parametrize("name", _STEP_USERS)
def test_uneven_step_counts_round_trip_through_the_engine(name: str, device: str) -> None:
    # The engine saves and restores each parameter's own count.
    grads = _grads()
    live = _params(_init(), device)
    opt = _late_unfreeze_run(
        lambda ps: _make(name, {}, ps), live, grads, lambda g: _tensor(g, device)
    )
    resumed = _params(_values(live), device)
    opt2 = _make(name, {}, resumed)
    engine: Any = opt._engine_optims[0]
    engine2: Any = opt2._engine_optims[0]
    engine2.load_state_buffers(engine.state_buffers())
    for g in grads[5:]:
        _feed(live, g, device)
        opt.step()
        _feed(resumed, g, device)
        opt2.step()
        _assert_bit_identical(_values(live), _values(resumed), name)


def test_uneven_step_counts_round_trip_through_state_dict(device: str) -> None:
    # Each parameter's own "step" goes back through load_state_buffers; the
    # loader used to fold them to their maximum, so uneven counts came back
    # even.
    grads = _grads()
    live = _params(_init(), device)
    opt = _late_unfreeze_run(
        lambda ps: _make("Adam", {}, ps), live, grads, lambda g: _tensor(g, device)
    )
    resumed = _params(_values(live), device)
    opt2 = _make("Adam", {}, resumed)
    opt2.load_state_dict(opt.state_dict())
    for g in grads[5:]:
        _feed(live, g, device)
        opt.step()
        _feed(resumed, g, device)
        opt2.step()
        _assert_bit_identical(_values(live), _values(resumed), "Adam")


# ── the reference framework's state layout ──────────────────────────────────

_REF_CASES = [
    *_CASE_PARAMS,
    pytest.param(("RMSprop", {"centered": True}), id="RMSprop-centered=True"),
]


@pytest.mark.parametrize("case", _REF_CASES)
def test_state_keys_shapes_and_values_match_the_reference(
    case: tuple[str, dict[str, object]], device: str, ref: Any
) -> None:
    name, kw = case
    grads = _grads()
    params = _params(_init(), device)
    opt = _make(name, kw, params)
    ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in _init()]
    ref_opt = getattr(ref.optim, name)(ref_params, **kw)
    for g in grads[:_N_BEFORE]:
        _feed(params, g, device)
        opt.step()
        for rp, gi in zip(ref_params, g):
            rp.grad = ref.from_numpy(gi.copy())
        ref_opt.step()

    state = opt.state_dict()["state"]
    ref_state = ref_opt.state_dict()["state"]
    assert sorted(state) == sorted(ref_state)
    for idx, ref_entry in ref_state.items():
        entry = state[idx]
        assert set(entry) == set(ref_entry)
        for key, ref_value in ref_entry.items():
            ours = _as_array(entry[key])
            theirs = ref_value.detach().numpy()
            assert ours.shape == theirs.shape, (idx, key)
            if key == "step":
                assert int(ours) == int(theirs)
            else:
                np.testing.assert_allclose(
                    ours, theirs, rtol=1e-4, atol=1e-6, err_msg=f"{idx}/{key}"
                )


# ── refusals ─────────────────────────────────────────────────────────────────


def test_state_for_a_differently_shaped_parameter_is_refused() -> None:
    params = _params(_init(), "cpu")
    opt = _make("NAdam", {}, params)
    _feed(params, _grads()[0], "cpu")
    opt.step()
    saved = opt.state_dict()

    other = _params([np.zeros((4, 3), np.float32), *_init()[1:]], "cpu")
    opt2 = _make("NAdam", {}, other)
    with pytest.raises(_C_engine.ShapeMismatch, match="load_state_buffers"):
        opt2.load_state_dict(saved)
