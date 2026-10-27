"""Adagrad, RMSprop and ASGD follow the reference framework's algorithms.

Each of the three took a different path from the reference until CHA-63/64/65:

* ``Adagrad`` handed its arguments to the engine in the wrong order, so
  ``lr_decay`` became ``eps``, ``eps`` became ``initial_accumulator_value``,
  and ``lr_decay`` itself was never applied;
* ``RMSprop(centered=True)`` stored the flag in the param group but ran the
  plain update, and kept no ``grad_avg``;
* ``ASGD`` ran a fixed step size, no parameter decay, and an average that
  only started at ``t0`` under a different weight, with no ``eta`` / ``mu``.

These pin the repair: the parameter **and state** trajectory matches the
reference step for step on every device and float dtype; a run restored
from ``state_dict`` continues bit-identically; and the state survives a
parameter move (``module.to()`` / ``.double()``), which rebuilds the engine.

``SGD`` joined them with CHA-210: its first momentum step damped the
gradient into a zeroed buffer, ``buf = (1 - dampening) * g``, where the
reference starts the buffer at the gradient itself, so with
``dampening != 0`` every later step was off.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.test._fixtures.devices import device_dtype_params, metal_available

# A matrix, a vector and a scalar: the scalar checks 0-d parameters, whose
# per-parameter scalars (eta, mu) are the same shape as the parameter.
_SHAPES = ((3, 4), (4,), ())
# SparseAdam in the reference requires at least 1-D parameters.
_SPARSE_SHAPES = ((4, 3), (5,))
_STEPS = 6

_CASES: list[tuple[str, dict[str, Any]]] = [
    ("Adagrad", {"lr": 0.1}),
    ("Adagrad", {"lr": 0.1, "lr_decay": 0.05}),
    ("Adagrad", {"lr": 0.1, "initial_accumulator_value": 0.5}),
    ("Adagrad", {"lr": 0.1, "eps": 1e-3}),
    (
        "Adagrad",
        {
            "lr": 0.1,
            "lr_decay": 0.01,
            "weight_decay": 0.1,
            "eps": 1e-6,
            "initial_accumulator_value": 0.1,
        },
    ),
    (
        "Adagrad",
        {
            "lr": 0.1,
            "lr_decay": 0.05,
            "weight_decay": 0.1,
            "eps": 1e-3,
            "initial_accumulator_value": 0.5,
        },
    ),
    ("RMSprop", {"lr": 0.01}),
    ("RMSprop", {"lr": 0.01, "centered": True}),
    ("RMSprop", {"lr": 0.01, "centered": True, "momentum": 0.9}),
    # 1 - alpha above 0.5 takes the lerp's other form.
    ("RMSprop", {"lr": 0.01, "centered": True, "alpha": 0.3}),
    (
        "RMSprop",
        {
            "lr": 0.01,
            "centered": True,
            "momentum": 0.5,
            "weight_decay": 0.1,
            "alpha": 0.9,
        },
    ),
    (
        "RMSprop",
        {
            "lr": 0.05,
            "alpha": 0.7,
            "eps": 1e-3,
            "weight_decay": 0.1,
            "momentum": 0.9,
            "centered": True,
        },
    ),
    ("RMSprop", {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.1}),
    # t0 = 1e6: the average only tracks the parameter.
    ("ASGD", {"lr": 0.05}),
    ("ASGD", {"lr": 0.05, "t0": 2.0, "lambd": 0.1}),
    ("ASGD", {"lr": 0.05, "t0": 0.0, "alpha": 0.5, "lambd": 0.01}),
    ("ASGD", {"lr": 0.05, "t0": 3.0, "weight_decay": 0.1}),
    (
        "ASGD",
        {"lr": 0.05, "lambd": 0.1, "alpha": 0.5, "t0": 2.0, "weight_decay": 0.1},
    ),
    # The first momentum step starts the buffer at the gradient, undamped.
    ("SGD", {"lr": 0.1, "momentum": 0.9}),
    ("SGD", {"lr": 0.1, "momentum": 0.9, "dampening": 0.5}),
    ("SGD", {"lr": 0.1, "momentum": 0.5, "dampening": 0.3, "weight_decay": 0.1}),
    ("SGD", {"lr": 0.1, "momentum": 0.9, "nesterov": True}),
    (
        "SGD",
        {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1, "nesterov": True},
    ),
    # Exaggerated hyper-parameters across the remaining optimizers (LCD-221):
    # large eps (1e-3), non-default betas, weight_decay, and variant flags
    # expose update-rule placement differences hidden at default eps=1e-8.
    (
        "Adam",
        {
            "lr": 0.05,
            "betas": (0.7, 0.8),
            "eps": 1e-3,
            "weight_decay": 0.1,
            "amsgrad": False,
        },
    ),
    (
        "Adam",
        {
            "lr": 0.05,
            "betas": (0.7, 0.8),
            "eps": 1e-3,
            "weight_decay": 0.1,
            "amsgrad": True,
        },
    ),
    (
        "AdamW",
        {
            "lr": 0.05,
            "betas": (0.7, 0.8),
            "eps": 1e-3,
            "weight_decay": 0.1,
            "amsgrad": False,
        },
    ),
    (
        "AdamW",
        {
            "lr": 0.05,
            "betas": (0.7, 0.8),
            "eps": 1e-3,
            "weight_decay": 0.1,
            "amsgrad": True,
        },
    ),
    (
        "NAdam",
        {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1},
    ),
    # beta2 = 0.8 keeps rho_t <= 5 for early steps; beta2 = 0.5 reaches rho_t > 5.
    (
        "RAdam",
        {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1},
    ),
    (
        "RAdam",
        {"lr": 0.05, "betas": (0.7, 0.5), "eps": 1e-3, "weight_decay": 0.1},
    ),
    (
        "Adamax",
        {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1},
    ),
    (
        "Adadelta",
        {"lr": 0.5, "rho": 0.7, "eps": 1e-3, "weight_decay": 0.1},
    ),
    (
        "Rprop",
        {"lr": 0.05, "etas": (0.4, 1.5), "step_sizes": (1e-2, 0.2)},
    ),
    (
        "SparseAdam",
        {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3},
    ),
]


def _case_id(case: tuple[str, dict[str, Any]]) -> str:
    name, kw = case
    return name + "".join(f"-{k}={v}" for k, v in kw.items())


_CASE_PARAMS = [pytest.param(c, id=_case_id(c)) for c in _CASES]

_DEVICE_DTYPES = device_dtype_params([lucid.float32, lucid.float64])

# Agreement with the reference: a few ulps of each dtype.
_TOL = {lucid.float32: 1e-6, lucid.float64: 1e-12}


def _np_dtype(dtype: lucid.dtype) -> type:
    return np.float64 if dtype == lucid.float64 else np.float32


def _shapes(name: str = "") -> tuple[tuple[int, ...], ...]:
    return _SPARSE_SHAPES if name == "SparseAdam" else _SHAPES


def _arrays(
    seed: int,
    dtype: lucid.dtype,
    shapes: tuple[tuple[int, ...], ...] = _SHAPES,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.asarray(rng.standard_normal(s), dtype=_np_dtype(dtype)) for s in shapes]


def _tensor(a: np.ndarray, device: str) -> lucid.Tensor:
    # Built on the CPU and moved: ``lucid.tensor(a, device=...)`` turns a 0-d
    # array into shape (1,).
    return lucid.tensor(a.copy()).to(device)


def _params(arrays: list[np.ndarray], device: str) -> list[nn.Parameter]:
    return [nn.Parameter(_tensor(a, device)) for a in arrays]


def _feed(params: list[nn.Parameter], grads: list[np.ndarray], device: str) -> None:
    for p, g in zip(params, grads, strict=True):
        p.grad = _tensor(g, device)


def _values(params: list[nn.Parameter]) -> list[np.ndarray]:
    return [np.array(p.numpy(), copy=True) for p in params]


def _as_array(v: object) -> np.ndarray:
    numpy = getattr(v, "numpy", None)
    return np.asarray(numpy() if callable(numpy) else v)


@contextlib.contextmanager
def _ref_default_dtype(ref: Any, dtype: lucid.dtype) -> Iterator[None]:
    """Run the reference at the default dtype matching ``dtype``.

    The reference keeps per-parameter scalar state (ASGD's ``eta`` and
    ``mu``) at its default dtype; Lucid keeps it at float64 for a float64
    parameter and float32 otherwise — the reference's behaviour when its
    default dtype is the parameter's.
    """
    before = ref.get_default_dtype()
    ref.set_default_dtype(ref.float64 if dtype == lucid.float64 else ref.float32)
    try:
        yield
    finally:
        ref.set_default_dtype(before)


@pytest.mark.parametrize("case", _CASE_PARAMS)
@pytest.mark.parametrize("device, dtype", _DEVICE_DTYPES)
def test_trajectory_and_state_match_the_reference(
    case: tuple[str, dict[str, Any]], device: str, dtype: lucid.dtype, ref: Any
) -> None:
    name, kw = case
    tol = _TOL[dtype]
    shapes = _shapes(name)
    init = _arrays(0, dtype, shapes)
    params = _params(init, device)
    opt = getattr(optim, name)(params, **kw)
    with _ref_default_dtype(ref, dtype):
        ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in init]
        ref_opt = getattr(ref.optim, name)(ref_params, **kw)
        for k in range(_STEPS):
            grads = _arrays(100 + k, dtype, shapes)
            if name == "SparseAdam":
                for g in grads:
                    g[k % g.shape[0]] = 0.0
            _feed(params, grads, device)
            opt.step()
            for rp, g in zip(ref_params, grads, strict=True):
                tg = ref.from_numpy(g.copy())
                rp.grad = tg.to_sparse(1) if name == "SparseAdam" else tg
            ref_opt.step()
            for i, (ours, rp) in enumerate(zip(_values(params), ref_params, strict=True)):
                np.testing.assert_allclose(
                    ours,
                    rp.detach().numpy(),
                    rtol=tol,
                    atol=tol,
                    err_msg=f"param {i} after step {k + 1}",
                )

    state = opt.state_dict()["state"]
    ref_state = ref_opt.state_dict()["state"]
    assert sorted(state) == sorted(ref_state)
    for idx, ref_entry in ref_state.items():
        entry = state[idx]
        assert set(entry) == set(ref_entry), idx
        for key, ref_value in ref_entry.items():
            ours = _as_array(entry[key])
            theirs = (
                ref_value.detach().numpy()
                if hasattr(ref_value, "detach")
                else np.asarray(ref_value)
            )
            assert ours.shape == theirs.shape, (idx, key)
            if key == "step":
                assert int(ours) == int(theirs) == _STEPS
            else:
                np.testing.assert_allclose(ours, theirs, rtol=tol, atol=tol, err_msg=f"{idx}/{key}")


def test_adamax_adds_eps_inside_exp_inf(device: str, ref: Any) -> None:
    # LCD-221 repro: Adamax adds eps inside max(beta2 * exp_inf, |g| + eps),
    # not in the denominator m / (exp_inf + eps).
    w = nn.Parameter(lucid.zeros(2, device=device))
    opt = optim.Adamax([w], lr=0.05, eps=1e-3)
    w.grad = lucid.tensor([1e-3, 2e-3], device=device)
    opt.step()

    rw = ref.nn.Parameter(ref.zeros(2))
    ref_opt = ref.optim.Adamax([rw], lr=0.05, eps=1e-3)
    rw.grad = ref.tensor([1e-3, 2e-3])
    ref_opt.step()

    exp_inf = _as_array(opt.state_dict()["state"][0]["exp_inf"])
    np.testing.assert_allclose(exp_inf, [0.002, 0.003], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        exp_inf,
        ref_opt.state_dict()["state"][0]["exp_inf"].detach().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(w.numpy(), rw.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_adagrad_takes_initial_accumulator_value_before_eps(ref: Any) -> None:
    # Positional slot 4 is ``initial_accumulator_value`` and slot 5 is ``eps``,
    # matching the reference framework's constructor signature.
    init = _arrays(0, lucid.float32)
    params = _params(init, "cpu")
    opt = optim.Adagrad(params, 0.1, 0.0, 0.0, 0.5, 1e-3)
    group = opt.param_groups[0]
    assert group["initial_accumulator_value"] == 0.5
    assert group["eps"] == 1e-3
    ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in init]
    ref_opt = ref.optim.Adagrad(ref_params, 0.1, 0.0, 0.0, 0.5, 1e-3)
    grads = _arrays(100, lucid.float32)
    _feed(params, grads, "cpu")
    opt.step()
    for rp, g in zip(ref_params, grads):
        rp.grad = ref.from_numpy(g.copy())
    ref_opt.step()
    for ours, rp in zip(_values(params), ref_params):
        np.testing.assert_allclose(ours, rp.detach().numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(("Adagrad", {"lr": 0.1, "lr_decay": 0.05}), id="Adagrad"),
        pytest.param(("ASGD", {"lr": 0.05, "t0": 2.0, "lambd": 0.1}), id="ASGD"),
    ],
)
def test_a_scheduled_learning_rate_reaches_the_schedule(
    case: tuple[str, dict[str, Any]], device: str, ref: Any
) -> None:
    # Adagrad decays the scheduled rate; ASGD folds it into the next eta.
    name, kw = case
    init = _arrays(0, lucid.float32)
    params = _params(init, device)
    opt = getattr(optim, name)(params, **kw)
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
    ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in init]
    ref_opt = getattr(ref.optim, name)(ref_params, **kw)
    ref_sched = ref.optim.lr_scheduler.ExponentialLR(ref_opt, gamma=0.5)
    for k in range(_STEPS):
        grads = _arrays(100 + k, lucid.float32)
        _feed(params, grads, device)
        opt.step()
        sched.step()
        for rp, g in zip(ref_params, grads):
            rp.grad = ref.from_numpy(g.copy())
        ref_opt.step()
        ref_sched.step()
        for i, (ours, rp) in enumerate(zip(_values(params), ref_params)):
            np.testing.assert_allclose(
                ours,
                rp.detach().numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"param {i} after step {k + 1}",
            )


# ── state_dict round trip ──────────────────────────────────────────────────

_ROUND_TRIP = [
    ("Adagrad", {"lr": 0.1, "lr_decay": 0.05, "initial_accumulator_value": 0.1}),
    ("RMSprop", {"lr": 0.01, "centered": True}),
    ("RMSprop", {"lr": 0.01, "centered": True, "momentum": 0.9}),
    ("ASGD", {"lr": 0.05, "t0": 2.0, "lambd": 0.1}),
    ("ASGD", {"lr": 0.05}),
]


@pytest.mark.parametrize("case", [pytest.param(c, id=_case_id(c)) for c in _ROUND_TRIP])
@pytest.mark.parametrize("device, dtype", _DEVICE_DTYPES)
def test_a_restored_run_continues_bit_identically(
    case: tuple[str, dict[str, Any]], device: str, dtype: lucid.dtype
) -> None:
    name, kw = case
    before, after = 4, 4
    live = _params(_arrays(0, dtype), device)
    opt = getattr(optim, name)(live, **kw)
    for k in range(before):
        _feed(live, _arrays(100 + k, dtype), device)
        opt.step()

    saved = opt.state_dict()
    resumed = _params(_values(live), device)
    opt2 = getattr(optim, name)(resumed, **kw)
    opt2.load_state_dict(saved)
    again = opt2.state_dict()["state"]
    for idx, entry in saved["state"].items():
        assert set(again[idx]) == set(entry)
        for key, value in entry.items():
            assert np.array_equal(_as_array(again[idx][key]), _as_array(value)), (
                idx,
                key,
            )

    for k in range(before, before + after):
        grads = _arrays(100 + k, dtype)
        _feed(live, grads, device)
        opt.step()
        _feed(resumed, grads, device)
        opt2.step()
        for i, (x, y) in enumerate(zip(_values(live), _values(resumed))):
            assert np.array_equal(x, y), f"param {i} after step {k + 1}"


# ── engine rebuild on a parameter move ─────────────────────────────────────


class _Holder(nn.Module):
    """A module owning the test parameters, so ``.to()`` can move them."""

    def __init__(self, arrays: list[np.ndarray]) -> None:
        super().__init__()
        for i, a in enumerate(arrays):
            self.register_parameter(f"p{i}", nn.Parameter(_tensor(a, "cpu")))

    def forward(self) -> None:
        """Unused: the test feeds gradients directly."""


def _holder_params(holder: _Holder) -> list[nn.Parameter]:
    return [getattr(holder, f"p{i}") for i in range(len(_SHAPES))]


_MOVES = [
    pytest.param("metal", lucid.float32, id="to-metal"),
    pytest.param("cpu", lucid.float64, id="double"),
]


@pytest.mark.parametrize("case", [pytest.param(c, id=_case_id(c)) for c in _ROUND_TRIP])
@pytest.mark.parametrize("target, target_dtype", _MOVES)
def test_state_survives_a_parameter_move(
    case: tuple[str, dict[str, Any]], target: str, target_dtype: lucid.dtype
) -> None:
    if target == "metal" and not metal_available():
        pytest.skip("Metal device not available on this host")
    name, kw = case
    moved_at = 3
    init = _arrays(0, lucid.float32)

    # Uninterrupted reference run, entirely on the CPU in float32.
    still = _params(init, "cpu")
    opt_still = getattr(optim, name)(still, **kw)
    want: list[list[np.ndarray]] = []
    for k in range(_STEPS):
        _feed(still, _arrays(100 + k, lucid.float32), "cpu")
        opt_still.step()
        want.append(_values(still))

    holder = _Holder(init)
    params = _holder_params(holder)
    opt = getattr(optim, name)(params, **kw)
    for k in range(_STEPS):
        if k == moved_at:
            holder.to(device=target, dtype=target_dtype)
            params = _holder_params(holder)
        dev = target if k >= moved_at else "cpu"
        grads = _arrays(100 + k, lucid.float32)
        if k >= moved_at and target_dtype == lucid.float64:
            grads = [g.astype(np.float64) for g in grads]
        _feed(params, grads, dev)
        opt.step()
        for i, (got, exp) in enumerate(zip(_values(params), want[k])):
            np.testing.assert_allclose(
                got.astype(np.float64),
                exp.astype(np.float64),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"param {i} after step {k + 1}",
            )

    # Every state entry came along — a reset one would have restarted the
    # step count and dropped the running averages.
    state = opt.state_dict()["state"]
    keys = set(opt_still.state_dict()["state"][0])
    for idx in range(len(_SHAPES)):
        assert set(state[idx]) == keys
        assert int(_as_array(state[idx]["step"])) == _STEPS
