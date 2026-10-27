"""A hand edit to ``param_groups`` takes effect at the next step (CHA-210).

Manual warm-up, decay and fine-tuning schedules write the group directly —
``for g in opt.param_groups: g["lr"] = ...`` — and the reference reads the
group on every step.  Until CHA-210 only ``lr`` reached the engine, and only
through an LR scheduler: ``SGD(lr=1)`` edited to ``lr=0`` still moved by the
full gradient.  Every step now hands each edited group to its engine.

Each case runs a few steps, edits the first group, runs a few more, and
edits again where the edit sequence goes on.  Two oracles:

* the reference, given the same edits (``parity``);
* an optimizer built afresh with the edited values and loaded with the
  state of the edited one at that moment: an edit must be indistinguishable
  from constructing with the new value, bit for bit.  This one also covers
  the switches the reference cannot take mid-run (AMSGrad, centered and
  RMSprop momentum switched on raise ``KeyError`` there).

The rules a group is held to are the reference's constructor checks, once
for every optimizer, at construction and at an edit alike.
"""

import copy
import math
import pickle
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine

_SHAPES = ((3, 4), (4,), ())
# SparseAdam's reference takes sparse gradients, which a 0-d tensor has none of.
_SPARSE_SHAPES = ((3, 4), (4,))
_PHASE = 2  # steps between edits

Edits = list[dict[str, Any]]
Case = tuple[str, dict[str, Any], Edits]

# (optimizer, constructor arguments, edits applied to group 0 one phase apart)
_CASES: list[Case] = [
    # The repro: lr 1 → 0 must stop the parameter.
    ("SGD", {"lr": 1.0}, [{"lr": 0.0}]),
    ("SGD", {"lr": 0.1}, [{"weight_decay": 0.2}]),
    ("SGD", {"lr": 0.1}, [{"momentum": 0.9, "dampening": 0.5}]),
    ("SGD", {"lr": 0.1, "momentum": 0.9}, [{"momentum": 0.0, "lr": 0.0}]),
    # A buffer kept while momentum was off resumes when it is switched back on.
    ("SGD", {"lr": 0.1, "momentum": 0.9}, [{"momentum": 0.0}, {"momentum": 0.5}]),
    ("SGD", {"lr": 0.1, "momentum": 0.9}, [{"dampening": 0.3}, {"nesterov": False}]),
    ("SGD", {"lr": 0.1, "momentum": 0.9}, [{"nesterov": True}]),
    ("Adam", {"lr": 1.0}, [{"lr": 0.0}]),
    ("Adam", {"lr": 0.05}, [{"beta1": 0.5}, {"beta2": 0.9}]),
    ("Adam", {"lr": 0.05}, [{"eps": 1e-2, "weight_decay": 0.1}]),
    ("Adam", {"lr": 0.05, "amsgrad": True}, [{"amsgrad": False}, {"amsgrad": True}]),
    ("AdamW", {"lr": 0.05}, [{"lr": 0.01, "weight_decay": 0.3}]),
    ("AdamW", {"lr": 0.05}, [{"beta1": 0.6, "beta2": 0.95, "eps": 1e-3}]),
    ("AdamW", {"lr": 0.05, "amsgrad": True}, [{"amsgrad": False}]),
    ("NAdam", {"lr": 0.05}, [{"lr": 0.01}, {"beta1": 0.7}]),
    ("NAdam", {"lr": 0.05}, [{"beta2": 0.9, "eps": 1e-3, "weight_decay": 0.1}]),
    ("RAdam", {"lr": 0.05}, [{"lr": 0.01, "beta1": 0.8}]),
    # beta2 = 0.5 keeps the rectified branch reachable within a few steps.
    ("RAdam", {"lr": 0.05}, [{"beta2": 0.5, "eps": 1e-3, "weight_decay": 0.1}]),
    ("Adamax", {"lr": 0.05}, [{"lr": 0.01, "beta1": 0.7, "beta2": 0.9}]),
    ("Adamax", {"lr": 0.05}, [{"weight_decay": 0.1}]),
    ("Adamax", {"lr": 0.05}, [{"eps": 1e-3}]),
    ("RMSprop", {"lr": 0.01}, [{"lr": 0.05, "alpha": 0.9}, {"eps": 1e-3}]),
    ("RMSprop", {"lr": 0.01}, [{"weight_decay": 0.1}]),
    ("RMSprop", {"lr": 0.01, "momentum": 0.9}, [{"momentum": 0.5}, {"momentum": 0.0}]),
    (
        "RMSprop",
        {"lr": 0.01, "centered": True},
        [{"centered": False}, {"centered": True}],
    ),
    ("Rprop", {"lr": 0.01}, [{"eta_minus": 0.3, "eta_plus": 1.5}]),
    ("Rprop", {"lr": 0.01}, [{"step_min": 5e-3, "step_max": 0.02}]),
    # lr only seeds a parameter's step size, here as there.
    ("Rprop", {"lr": 0.01}, [{"lr": 0.5}]),
    ("Adagrad", {"lr": 0.1}, [{"lr": 0.05, "lr_decay": 0.1}]),
    ("Adagrad", {"lr": 0.1}, [{"weight_decay": 0.1, "eps": 1e-3}]),
    # The accumulator seed is read once, at construction, here as there.
    ("Adagrad", {"lr": 0.1}, [{"initial_accumulator_value": 0.5}]),
    ("Adadelta", {"lr": 1.0}, [{"lr": 0.5, "rho": 0.5}]),
    ("Adadelta", {"lr": 1.0}, [{"eps": 1e-3, "weight_decay": 0.1}]),
    # ASGD folds lr into the next step's eta, so its effect lags one step.
    ("ASGD", {"lr": 0.05, "t0": 2.0}, [{"lr": 0.01, "lambd": 0.1}]),
    ("ASGD", {"lr": 0.05}, [{"alpha": 0.5, "t0": 0.0}, {"weight_decay": 0.1}]),
    ("SparseAdam", {"lr": 0.05}, [{"lr": 0.01}, {"betas": (0.5, 0.9)}]),
    ("SparseAdam", {"lr": 0.05}, [{"eps": 1e-3}]),
    ("SparseAdam", {"lr": 0.05}, [{"maximize": True}, {"maximize": False}]),
]

# Switched on mid-run, which the reference refuses with a ``KeyError``: the
# new buffer starts at zero, as in a fresh optimizer given the same state.
_SWITCH_ON_CASES: list[Case] = [
    ("Adam", {"lr": 0.05}, [{"amsgrad": True}]),
    ("AdamW", {"lr": 0.05}, [{"amsgrad": True}]),
    ("RMSprop", {"lr": 0.01}, [{"centered": True}]),
    ("RMSprop", {"lr": 0.01}, [{"momentum": 0.9}]),
    ("RMSprop", {"lr": 0.01}, [{"momentum": 0.9, "centered": True}]),
]

# A reference group spells some hyper-parameters as a pair.
_PAIRS = {
    "beta1": ("betas", 0),
    "beta2": ("betas", 1),
    "eta_minus": ("etas", 0),
    "eta_plus": ("etas", 1),
    "step_min": ("step_sizes", 0),
    "step_max": ("step_sizes", 1),
}

# Agreement with the reference: a few float32 ulps, compounded over the run.
_RTOL, _ATOL = 2e-5, 2e-6


def _case_id(case: Case) -> str:
    name, kw, edits = case
    built = "".join(f"-{k}={v}" for k, v in kw.items())
    return name + built + "".join("|" + ",".join(f"{k}={v}" for k, v in e.items()) for e in edits)


def _shapes(name: str) -> tuple[tuple[int, ...], ...]:
    return _SPARSE_SHAPES if name == "SparseAdam" else _SHAPES


def _arrays(seed: int, shapes: tuple[tuple[int, ...], ...]) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.asarray(rng.standard_normal(s), dtype=np.float32) for s in shapes]


def _tensor(a: np.ndarray, device: str) -> lucid.Tensor:
    # Built on the CPU and moved: ``lucid.tensor(a, device=...)`` turns a 0-d
    # array into shape (1,).
    return lucid.tensor(a.copy()).to(device)


def _params(arrays: list[np.ndarray], device: str) -> list[nn.Parameter]:
    return [nn.Parameter(_tensor(a, device)) for a in arrays]


def _feed(params: list[nn.Parameter], grads: list[np.ndarray], device: str) -> None:
    for p, g in zip(params, grads):
        p.grad = _tensor(g, device)


def _values(params: list[nn.Parameter]) -> list[np.ndarray]:
    return [np.array(p.numpy(), copy=True) for p in params]


def _grads(name: str, step: int) -> list[np.ndarray]:
    return _arrays(100 + step, _shapes(name))


def _apply_edit(group: dict[str, Any], edit: dict[str, Any]) -> None:
    group.update(edit)


def _apply_ref_edit(group: dict[str, Any], edit: dict[str, Any]) -> None:
    for key, value in edit.items():
        pair = _PAIRS.get(key)
        if pair is None:
            group[key] = value
            continue
        name, slot = pair
        both = list(group[name])
        both[slot] = value
        group[name] = tuple(both)


def _ctor_kwargs(group: dict[str, Any]) -> dict[str, Any]:
    """The constructor arguments that build ``group``'s hyper-parameters."""
    out = {k: v for k, v in group.items() if k != "params" and k not in _PAIRS}
    for name in ("betas", "etas", "step_sizes"):
        keys = [k for k, (pair, _) in _PAIRS.items() if pair == name]
        if all(k in group for k in keys):
            out[name] = tuple(group[k] for k in keys)
    return out


def _run_lucid(case: Case, device: str) -> list[list[np.ndarray]]:
    """Parameters after every step of ``case``."""
    name, kw, edits = case
    params = _params(_arrays(0, _shapes(name)), device)
    opt = getattr(optim, name)(params, **kw)
    out: list[list[np.ndarray]] = []
    for phase in range(len(edits) + 1):
        if phase:
            _apply_edit(opt.param_groups[0], edits[phase - 1])
        for k in range(phase * _PHASE, (phase + 1) * _PHASE):
            _feed(params, _grads(name, k), device)
            opt.step()
            out.append(_values(params))
    return out


def _run_ref(case: Case, ref: Any) -> list[list[np.ndarray]]:
    name, kw, edits = case
    params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in _arrays(0, _shapes(name))]
    opt = getattr(ref.optim, name)(params, **kw)
    out: list[list[np.ndarray]] = []
    for phase in range(len(edits) + 1):
        if phase:
            _apply_ref_edit(opt.param_groups[0], edits[phase - 1])
        for k in range(phase * _PHASE, (phase + 1) * _PHASE):
            for p, g in zip(params, _grads(name, k)):
                grad = ref.from_numpy(g.copy())
                p.grad = grad.to_sparse() if name == "SparseAdam" else grad
            opt.step()
            out.append([p.detach().numpy().copy() for p in params])
    return out


@pytest.mark.parity
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=_case_id(c)) for c in _CASES],
)
def test_an_edit_takes_effect_as_in_the_reference(case: Case, device: str, ref: Any) -> None:
    ours = _run_lucid(case, device)
    theirs = _run_ref(case, ref)
    for k, (got, want) in enumerate(zip(ours, theirs)):
        for i, (a, b) in enumerate(zip(got, want)):
            np.testing.assert_allclose(
                a, b, rtol=_RTOL, atol=_ATOL, err_msg=f"param {i} after step {k + 1}"
            )


def test_the_repro_stops_moving_at_lr_zero(device: str) -> None:
    w = nn.Parameter(lucid.ones(3, device=device))
    opt = optim.SGD([w], lr=1.0)
    (w * 2).sum().backward()
    opt.step()
    opt.zero_grad()
    before = w.numpy().copy()
    opt.param_groups[0]["lr"] = 0.0
    (w * 2).sum().backward()
    opt.step()
    np.testing.assert_array_equal(w.numpy(), before)


# ── an edit equals a fresh optimizer on the same state ─────────────────────


def _transfer_state(src: optim.Optimizer, dst: optim.Optimizer) -> None:
    """Give ``dst`` the state ``src`` holds now, with ``dst``'s own groups."""
    saved = copy.deepcopy(src.state_dict())
    for mine, theirs in zip(dst.param_groups, saved["param_groups"]):
        for key, value in mine.items():
            if key != "params":
                theirs[key] = value
    dst.load_state_dict(saved)


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=_case_id(c)) for c in [*_CASES, *_SWITCH_ON_CASES]],
)
def test_an_edit_equals_a_fresh_optimizer_on_the_same_state(case: Case, device: str) -> None:
    name, kw, edits = case
    params = _params(_arrays(0, _shapes(name)), device)
    opt = getattr(optim, name)(params, **kw)
    for k in range(_PHASE):
        _feed(params, _grads(name, k), device)
        opt.step()
    for phase, edit in enumerate(edits, start=1):
        _apply_edit(opt.param_groups[0], edit)
        fresh_params = _params(_values(params), device)
        fresh = getattr(optim, name)(fresh_params, **_ctor_kwargs(opt.param_groups[0]))
        _transfer_state(opt, fresh)
        for k in range(phase * _PHASE, (phase + 1) * _PHASE):
            grads = _grads(name, k)
            _feed(params, grads, device)
            opt.step()
            _feed(fresh_params, grads, device)
            fresh.step()
            for i, (a, b) in enumerate(zip(_values(params), _values(fresh_params))):
                np.testing.assert_array_equal(a, b, err_msg=f"param {i} after step {k + 1}")


def _quadratic(params: list[nn.Parameter], target: list[lucid.Tensor]) -> lucid.Tensor:
    loss = ((params[0] - target[0]) ** 2).sum()
    for p, t in zip(params[1:], target[1:]):
        loss = loss + ((p - t) ** 2).sum()
    return loss


def test_an_lbfgs_edit_equals_a_fresh_optimizer_on_the_same_state() -> None:
    # LBFGS steps through a closure; its state round-trips on the CPU.
    init = _arrays(0, _SHAPES)
    target = [_tensor(a, "cpu") for a in _arrays(1, _SHAPES)]
    params = _params(init, "cpu")
    opt = optim.LBFGS(params, lr=1.0)

    def closure_for(ps: list[nn.Parameter], o: optim.Optimizer) -> Callable[[], lucid.Tensor]:
        def closure() -> lucid.Tensor:
            o.zero_grad()
            loss = _quadratic(ps, target)
            loss.backward()
            return loss

        return closure

    opt.step(closure_for(params, opt))
    opt.param_groups[0]["lr"] = 0.25
    fresh_params = _params(_values(params), "cpu")
    fresh = optim.LBFGS(fresh_params, lr=0.25)
    _transfer_state(opt, fresh)
    for _ in range(2):
        opt.step(closure_for(params, opt))
        fresh.step(closure_for(fresh_params, fresh))
        for a, b in zip(_values(params), _values(fresh_params)):
            np.testing.assert_array_equal(a, b)


# ── groups other than the first ────────────────────────────────────────────


@pytest.mark.parity
def test_each_group_is_read_on_its_own(device: str, ref: Any) -> None:
    init = _arrays(0, _SHAPES)
    params = _params(init, device)
    opt = optim.Adam([{"params": params[:2]}, {"params": params[2:], "lr": 0.01}], lr=0.05)
    ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in init]
    ref_opt = ref.optim.Adam(
        [{"params": ref_params[:2]}, {"params": ref_params[2:], "lr": 0.01}], lr=0.05
    )
    for k in range(3 * _PHASE):
        if k == _PHASE:
            opt.param_groups[1]["lr"] = 0.2
            ref_opt.param_groups[1]["lr"] = 0.2
        if k == 2 * _PHASE:
            opt.param_groups[0]["beta1"] = 0.5
            ref_opt.param_groups[0]["betas"] = (0.5, 0.999)
        grads = _grads("Adam", k)
        _feed(params, grads, device)
        opt.step()
        for p, g in zip(ref_params, grads):
            p.grad = ref.from_numpy(g.copy())
        ref_opt.step()
        for a, p in zip(_values(params), ref_params):
            np.testing.assert_allclose(a, p.detach().numpy(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parity
def test_an_added_group_is_read_on_every_step(device: str, ref: Any) -> None:
    init = _arrays(0, _SHAPES)
    params = _params(init, device)
    opt = optim.SGD(params[:2], lr=0.1, momentum=0.9)
    ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in init]
    ref_opt = ref.optim.SGD(ref_params[:2], lr=0.1, momentum=0.9)
    for k in range(3 * _PHASE):
        if k == _PHASE:
            # Added after the engines exist: it gets an engine of its own.
            opt.add_param_group({"params": params[2:], "lr": 0.5})
            ref_opt.add_param_group({"params": ref_params[2:], "lr": 0.5})
        if k == 2 * _PHASE:
            opt.param_groups[1]["lr"] = 0.05
            opt.param_groups[1]["momentum"] = 0.0
            ref_opt.param_groups[1]["lr"] = 0.05
            ref_opt.param_groups[1]["momentum"] = 0.0
        grads = _grads("SGD", k)
        _feed(params, grads, device)
        opt.step()
        for p, g in zip(ref_params, grads):
            p.grad = ref.from_numpy(g.copy())
        ref_opt.step()
        for a, p in zip(_values(params), ref_params):
            np.testing.assert_allclose(a, p.detach().numpy(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parity
def test_a_scheduler_and_a_hand_edit_compose(device: str, ref: Any) -> None:
    init = _arrays(0, _SHAPES)
    params = _params(init, device)
    opt = optim.SGD(params, lr=0.1, momentum=0.9)
    sched = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in init]
    ref_opt = ref.optim.SGD(ref_params, lr=0.1, momentum=0.9)
    ref_sched = ref.optim.lr_scheduler.StepLR(ref_opt, step_size=2, gamma=0.5)
    for k in range(3 * _PHASE):
        if k == _PHASE:
            opt.param_groups[0]["weight_decay"] = 0.2
            ref_opt.param_groups[0]["weight_decay"] = 0.2
        grads = _grads("SGD", k)
        _feed(params, grads, device)
        opt.step()
        sched.step()
        for p, g in zip(ref_params, grads):
            p.grad = ref.from_numpy(g.copy())
        ref_opt.step()
        ref_sched.step()
        for a, p in zip(_values(params), ref_params):
            np.testing.assert_allclose(a, p.detach().numpy(), rtol=_RTOL, atol=_ATOL)


def test_an_edit_before_the_first_step_is_built_in(device: str) -> None:
    params = _params(_arrays(0, _SHAPES), device)
    opt = optim.Adam(params, lr=1.0)
    opt.param_groups[0]["lr"] = 0.0
    before = _values(params)
    _feed(params, _grads("Adam", 0), device)
    opt.step()
    for a, b in zip(_values(params), before):
        np.testing.assert_array_equal(a, b)


def test_an_edited_value_survives_a_checkpoint(device: str) -> None:
    params = _params(_arrays(0, _SHAPES), device)
    opt = optim.RMSprop(params, lr=0.01)
    for k in range(_PHASE):
        _feed(params, _grads("RMSprop", k), device)
        opt.step()
    opt.param_groups[0]["lr"] = 0.03
    opt.param_groups[0]["momentum"] = 0.5
    saved = copy.deepcopy(opt.state_dict())
    assert saved["param_groups"][0]["lr"] == 0.03
    resumed_params = _params(_values(params), device)
    resumed = optim.RMSprop(resumed_params, lr=0.01)
    resumed.load_state_dict(saved)
    for k in range(_PHASE, 3 * _PHASE):
        grads = _grads("RMSprop", k)
        _feed(params, grads, device)
        opt.step()
        _feed(resumed_params, grads, device)
        resumed.step()
        for a, b in zip(_values(params), _values(resumed_params)):
            np.testing.assert_array_equal(a, b)


# A buffer whose feature is switched off is kept — in the state, in a
# checkpoint, and through a restore — and resumes when the feature is
# switched back on, as the reference's state keeps it.
_KEPT: list[tuple[str, dict[str, Any], str, object, object, str]] = [
    ("SGD", {"lr": 0.1, "momentum": 0.9}, "momentum", 0.0, 0.5, "momentum_buffer"),
    ("Adam", {"lr": 0.05, "amsgrad": True}, "amsgrad", False, True, "max_exp_avg_sq"),
    ("AdamW", {"lr": 0.05, "amsgrad": True}, "amsgrad", False, True, "max_exp_avg_sq"),
    ("RMSprop", {"lr": 0.01, "momentum": 0.9}, "momentum", 0.0, 0.5, "momentum_buffer"),
    ("RMSprop", {"lr": 0.01, "centered": True}, "centered", False, True, "grad_avg"),
]


@pytest.mark.parametrize(
    "name, kw, key, off, on, kept",
    [pytest.param(*c, id=f"{c[0]}-{c[2]}") for c in _KEPT],
)
def test_a_switched_off_buffer_survives_a_checkpoint_and_resumes(
    name: str,
    kw: dict[str, Any],
    key: str,
    off: object,
    on: object,
    kept: str,
    device: str,
) -> None:
    params = _params(_arrays(0, _SHAPES), device)
    opt = getattr(optim, name)(params, **kw)
    for k in range(_PHASE):
        _feed(params, _grads(name, k), device)
        opt.step()
    opt.param_groups[0][key] = off
    for k in range(_PHASE, 2 * _PHASE):
        _feed(params, _grads(name, k), device)
        opt.step()

    saved = copy.deepcopy(opt.state_dict())
    assert all(kept in entry for entry in saved["state"].values())
    # Restored into an optimizer built with the feature off.
    resumed_params = _params(_values(params), device)
    resumed = getattr(optim, name)(resumed_params, **{**kw, key: off})
    resumed.load_state_dict(saved)
    again = resumed.state_dict()["state"]
    for idx, entry in saved["state"].items():
        np.testing.assert_array_equal(again[idx][kept], entry[kept])

    opt.param_groups[0][key] = on
    resumed.param_groups[0][key] = on
    for k in range(2 * _PHASE, 4 * _PHASE):
        grads = _grads(name, k)
        _feed(params, grads, device)
        opt.step()
        _feed(resumed_params, grads, device)
        resumed.step()
        for a, b in zip(_values(params), _values(resumed_params)):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parity
@pytest.mark.parametrize(
    "name, kw, key, off, on, kept",
    [pytest.param(*c, id=f"{c[0]}-{c[2]}") for c in _KEPT],
)
def test_a_switched_off_buffer_is_in_the_state_as_in_the_reference(
    name: str,
    kw: dict[str, Any],
    key: str,
    off: object,
    on: object,
    kept: str,
    ref: Any,
) -> None:
    init = _arrays(0, _SHAPES)
    params = _params(init, "cpu")
    opt = getattr(optim, name)(params, **kw)
    ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in init]
    ref_opt = getattr(ref.optim, name)(ref_params, **kw)
    for k in range(2 * _PHASE):
        if k == _PHASE:
            opt.param_groups[0][key] = off
            ref_opt.param_groups[0][key] = off
        grads = _grads(name, k)
        _feed(params, grads, "cpu")
        opt.step()
        for p, g in zip(ref_params, grads):
            p.grad = ref.from_numpy(g.copy())
        ref_opt.step()
    state = opt.state_dict()["state"]
    ref_state = ref_opt.state_dict()["state"]
    for idx, entry in ref_state.items():
        np.testing.assert_allclose(state[idx][kept], entry[kept].numpy(), rtol=_RTOL, atol=_ATOL)


def test_a_rejected_edit_also_stops_a_checkpoint() -> None:
    params = _params(_arrays(0, _SHAPES), "cpu")
    opt = optim.Adam(params, lr=0.05)
    _step("Adam", opt, params, 0)
    opt.param_groups[0]["beta1"] = 1.5
    with pytest.raises(_C_engine.InvalidArgument, match="beta1 must be in"):
        opt.state_dict()
    opt.param_groups[0]["beta1"] = 0.9
    assert opt.state_dict()["state"]


@pytest.mark.parametrize(
    "name",
    [
        "SGD",
        "Adam",
        "AdamW",
        "LBFGS",
        "RMSprop",
        "Adagrad",
        "Adadelta",
        "Adamax",
        "RAdam",
        "NAdam",
        "ASGD",
        "Rprop",
        "SparseAdam",
    ],
)
def test_a_fresh_optimizer_pickles(name: str) -> None:
    params = _params(_arrays(0, _shapes(name)), "cpu")
    opt = getattr(optim, name)(params, lr=0.1)
    loaded = pickle.loads(pickle.dumps(opt))
    assert type(loaded) is type(opt)
    for mine, theirs in zip(loaded.param_groups, opt.param_groups):
        assert {k: v for k, v in mine.items() if k != "params"} == {
            k: v for k, v in theirs.items() if k != "params"
        }
    # The copy reads its own groups: an edit to it is checked and applied.
    loaded_params = loaded.param_groups[0]["params"]
    loaded.param_groups[0]["lr"] = 0.05
    _step(name, loaded, loaded_params, 0)
    loaded.param_groups[0]["lr"] = -1.0
    with pytest.raises(_C_engine.InvalidArgument):
        _step(name, loaded, loaded_params, 1)


# ── what a step costs ──────────────────────────────────────────────────────


class _Recorder:
    """An engine optimizer that counts the hyper-parameter hand-overs."""

    def __init__(self, engine: _C_engine.Adam) -> None:
        self._engine = engine
        self.calls: list[dict[str, float]] = []

    def set_hyperparams(self, values: dict[str, float]) -> None:
        self.calls.append(dict(values))
        self._engine.set_hyperparams(values)

    def __getattr__(self, name: str) -> object:
        return getattr(self._engine, name)


def test_a_step_after_no_edit_makes_no_hand_over(device: str) -> None:
    params = _params(_arrays(0, _SHAPES), device)
    opt = optim.Adam([{"params": params[:2]}, {"params": params[2:], "lr": 0.01}], lr=0.05)
    _feed(params, _grads("Adam", 0), device)
    opt.step()
    recorders = [_Recorder(cast(_C_engine.Adam, e)) for e in opt._engines]
    opt._engines[:] = recorders
    for k in range(1, 6):
        _feed(params, _grads("Adam", k), device)
        opt.step()
    assert [r.calls for r in recorders] == [[], []]

    # Re-assigning the same value is no edit either.
    opt.param_groups[0]["lr"] = 0.05
    opt.param_groups[1]["eps"] = 1e-3
    for k in range(6, 9):
        _feed(params, _grads("Adam", k), device)
        opt.step()
    assert recorders[0].calls == []
    assert len(recorders[1].calls) == 1
    assert recorders[1].calls[0]["eps"] == 1e-3


# ── the rules a group is held to ───────────────────────────────────────────

_ADAM_FAMILY_RULES = [
    ({"lr": -1.0}, {"lr": -1.0}, "lr must be >= 0"),
    ({"eps": -1.0}, {"eps": -1.0}, "eps must be >= 0"),
    ({"betas": (1.0, 0.999)}, {"beta1": 1.0}, "beta1 must be in [0, 1)"),
    ({"betas": (-0.1, 0.999)}, {"beta1": -0.1}, "beta1 must be in [0, 1)"),
    ({"betas": (0.9, 1.0)}, {"beta2": 1.0}, "beta2 must be in [0, 1)"),
    ({"weight_decay": -1.0}, {"weight_decay": -1.0}, "weight_decay must be >= 0"),
]

# (optimizer, base arguments, bad constructor arguments, the same as a group
#  edit, message)
_RULES: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str]] = [
    ("SGD", {"lr": 0.1}, {"lr": -1.0}, {"lr": -1.0}, "lr must be >= 0"),
    (
        "SGD",
        {"lr": 0.1},
        {"momentum": -1.0},
        {"momentum": -1.0},
        "momentum must be >= 0",
    ),
    (
        "SGD",
        {"lr": 0.1},
        {"weight_decay": -1.0},
        {"weight_decay": -1.0},
        "weight_decay must be >= 0",
    ),
    (
        "SGD",
        {"lr": 0.1},
        {"nesterov": True},
        {"nesterov": True},
        "nesterov requires momentum > 0 and dampening = 0",
    ),
    (
        "SGD",
        {"lr": 0.1, "momentum": 0.9, "nesterov": True},
        {"dampening": 0.5},
        {"dampening": 0.5},
        "nesterov requires momentum > 0 and dampening = 0",
    ),
    *[("Adam", {}, c, e, m) for c, e, m in _ADAM_FAMILY_RULES],
    *[("AdamW", {}, c, e, m) for c, e, m in _ADAM_FAMILY_RULES],
    *[("NAdam", {}, c, e, m) for c, e, m in _ADAM_FAMILY_RULES],
    *[("RAdam", {}, c, e, m) for c, e, m in _ADAM_FAMILY_RULES],
    *[("Adamax", {}, c, e, m) for c, e, m in _ADAM_FAMILY_RULES],
    ("Adagrad", {}, {"lr": -1.0}, {"lr": -1.0}, "lr must be >= 0"),
    ("Adagrad", {}, {"lr_decay": -1.0}, {"lr_decay": -1.0}, "lr_decay must be >= 0"),
    (
        "Adagrad",
        {},
        {"weight_decay": -1.0},
        {"weight_decay": -1.0},
        "weight_decay must be >= 0",
    ),
    (
        "Adagrad",
        {},
        {"initial_accumulator_value": -1.0},
        {"initial_accumulator_value": -1.0},
        "initial_accumulator_value must be >= 0",
    ),
    ("Adagrad", {}, {"eps": -1.0}, {"eps": -1.0}, "eps must be >= 0"),
    ("Adadelta", {}, {"lr": -1.0}, {"lr": -1.0}, "lr must be >= 0"),
    ("Adadelta", {}, {"rho": 1.5}, {"rho": 1.5}, "rho must be in [0, 1]"),
    ("Adadelta", {}, {"rho": -0.5}, {"rho": -0.5}, "rho must be in [0, 1]"),
    ("Adadelta", {}, {"eps": -1.0}, {"eps": -1.0}, "eps must be >= 0"),
    (
        "Adadelta",
        {},
        {"weight_decay": -1.0},
        {"weight_decay": -1.0},
        "weight_decay must be >= 0",
    ),
    ("RMSprop", {}, {"lr": -1.0}, {"lr": -1.0}, "lr must be >= 0"),
    ("RMSprop", {}, {"alpha": -1.0}, {"alpha": -1.0}, "alpha must be >= 0"),
    ("RMSprop", {}, {"eps": -1.0}, {"eps": -1.0}, "eps must be >= 0"),
    (
        "RMSprop",
        {},
        {"weight_decay": -1.0},
        {"weight_decay": -1.0},
        "weight_decay must be >= 0",
    ),
    ("RMSprop", {}, {"momentum": -1.0}, {"momentum": -1.0}, "momentum must be >= 0"),
    ("Rprop", {}, {"lr": -1.0}, {"lr": -1.0}, "lr must be >= 0"),
    (
        "Rprop",
        {},
        {"etas": (1.5, 1.2)},
        {"eta_minus": 1.5},
        "etas must satisfy 0 < eta_minus < 1 < eta_plus",
    ),
    (
        "Rprop",
        {},
        {"etas": (0.0, 1.2)},
        {"eta_minus": 0.0},
        "etas must satisfy 0 < eta_minus < 1 < eta_plus",
    ),
    (
        "Rprop",
        {},
        {"etas": (0.5, 0.9)},
        {"eta_plus": 0.9},
        "etas must satisfy 0 < eta_minus < 1 < eta_plus",
    ),
    ("ASGD", {}, {"lr": -1.0}, {"lr": -1.0}, "lr must be >= 0"),
    (
        "ASGD",
        {},
        {"weight_decay": -1.0},
        {"weight_decay": -1.0},
        "weight_decay must be >= 0",
    ),
    ("SparseAdam", {}, {"lr": 0.0}, {"lr": 0.0}, "lr must be > 0"),
    ("SparseAdam", {}, {"eps": 0.0}, {"eps": 0.0}, "eps must be > 0"),
    (
        "SparseAdam",
        {},
        {"betas": (1.0, 0.999)},
        {"betas": (1.0, 0.999)},
        "betas[0] must be in [0, 1)",
    ),
    (
        "SparseAdam",
        {},
        {"betas": (0.9, -0.1)},
        {"betas": (0.9, -0.1)},
        "betas[1] must be in [0, 1)",
    ),
    ("LBFGS", {}, {"lr": -1.0}, {"lr": -1.0}, "lr must be >= 0"),
]


def _rule_id(
    rule: tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str],
) -> str:
    name, base, bad, _, _ = rule
    return name + "".join(f"-{k}={v}" for k, v in {**base, **bad}.items())


_RULE_PARAMS = [pytest.param(r, id=_rule_id(r)) for r in _RULES]


def test_every_optimizer_has_its_rules() -> None:
    optimizers = {
        name
        for name in optim.__all__
        if name != "Optimizer"
        and isinstance(getattr(optim, name), type)
        and issubclass(getattr(optim, name), optim.Optimizer)
    }
    assert len(optimizers) == 13
    assert {rule[0] for rule in _RULES} == optimizers
    assert {case[0] for case in _CASES} == optimizers - {"LBFGS"}


def _closure_for(params: list[nn.Parameter], opt: optim.Optimizer) -> Callable[[], lucid.Tensor]:
    target = [lucid.zeros_like(p) for p in params]

    def closure() -> lucid.Tensor:
        opt.zero_grad()
        loss = _quadratic(params, target)
        loss.backward()
        return loss

    return closure


def _step(name: str, opt: optim.Optimizer, params: list[nn.Parameter], k: int) -> None:
    if name == "LBFGS":
        opt.step(_closure_for(params, opt))
        return
    _feed(params, _grads(name, k), "cpu")
    opt.step()


@pytest.mark.parametrize("rule", _RULE_PARAMS)
def test_the_constructor_rejects_a_value_the_rules_reject(
    rule: tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str],
) -> None:
    name, base, bad, _, message = rule
    params = _params(_arrays(0, _shapes(name)), "cpu")
    with pytest.raises(_C_engine.InvalidArgument, match=_match(name, message)):
        getattr(optim, name)(params, **{**base, **bad})
    # ``InvalidArgument`` is the ``ValueError`` the reference raises.
    assert issubclass(_C_engine.InvalidArgument, ValueError)


@pytest.mark.parity
@pytest.mark.parametrize("rule", _RULE_PARAMS)
def test_the_reference_rejects_it_too(
    rule: tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str], ref: Any
) -> None:
    name, base, bad, _, _ = rule
    params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in _arrays(0, _shapes(name))]
    with pytest.raises(ValueError):
        getattr(ref.optim, name)(params, **{**base, **bad})


@pytest.mark.parametrize("rule", _RULE_PARAMS)
def test_an_edit_the_rules_reject_raises_and_changes_nothing(
    rule: tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str],
) -> None:
    name, base, _, edit, message = rule
    params = _params(_arrays(0, _shapes(name)), "cpu")
    opt = getattr(optim, name)(params, **base)
    _step(name, opt, params, 0)
    group = opt.param_groups[0]
    kept = {k: group[k] for k in edit}
    group.update(edit)
    before = _values(params)
    with pytest.raises(_C_engine.InvalidArgument, match=_match(name, message)):
        _step(name, opt, params, 1)
    for a, b in zip(_values(params), before):
        np.testing.assert_array_equal(a, b)
    # Still rejected on the next step: nothing of the group was applied.
    with pytest.raises(_C_engine.InvalidArgument):
        _step(name, opt, params, 1)
    group.update(kept)
    _step(name, opt, params, 1)


@pytest.mark.parametrize("rule", _RULE_PARAMS)
def test_an_added_group_the_rules_reject_is_refused(
    rule: tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str],
) -> None:
    name, base, _, edit, message = rule
    if name == "LBFGS":
        pytest.skip("LBFGS takes a single parameter group")
    params = _params(_arrays(0, _shapes(name)), "cpu")
    opt = getattr(optim, name)(params[:1], **base)
    with pytest.raises(_C_engine.InvalidArgument, match=_match(name, message)):
        opt.add_param_group({"params": params[1:], **edit})
    assert len(opt.param_groups) == 1


def test_an_invalid_edit_before_the_first_step_builds_nothing() -> None:
    params = _params(_arrays(0, _SHAPES), "cpu")
    opt = optim.Adam([{"params": params[:2]}, {"params": params[2:]}], lr=0.1)
    opt.param_groups[1]["lr"] = -1.0
    _feed(params, _grads("Adam", 0), "cpu")
    with pytest.raises(_C_engine.InvalidArgument):
        opt.step()
    opt.param_groups[1]["lr"] = 0.1
    opt.step()
    assert len(opt._engines) == 2


def test_nan_is_rejected_where_the_reference_rejects_it() -> None:
    params = _params(_arrays(0, _SHAPES), "cpu")
    with pytest.raises(_C_engine.InvalidArgument):
        optim.Adam(params, lr=math.nan)
    # The reference's SGD compares ``lr < 0.0``, which a NaN passes.
    optim.SGD(params, lr=math.nan)


@pytest.mark.parametrize("value", [None, "fast"])
def test_a_value_of_the_wrong_type_is_a_type_error(value: object) -> None:
    params = _params(_arrays(0, _SHAPES), "cpu")
    opt = optim.SGD(params, lr=0.1)
    _step("SGD", opt, params, 0)
    opt.param_groups[0]["lr"] = value
    with pytest.raises(TypeError, match="SGD: lr must be a number"):
        _step("SGD", opt, params, 1)


def _match(name: str, message: str) -> str:
    return f"^{name}: " + message.replace("[", r"\[").replace("(", r"\(").replace(")", r"\)")
