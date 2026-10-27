"""An optimizer step or an ``nn.init`` call inside autocast is the call outside it.

LCD-500.  Neither turns autocast off, and the reference's do not either:
every op they are built from either has no row in the policy table or has
a row that leaves a float32 input float32, so an autocast scope around
them changes nothing.  Before the table their element-wise ops were LOWER,
so ``LBFGS.step``, ``NAdam(decoupled_weight_decay=True).step``,
``SparseAdam`` and ``nn.init.trunc_normal_`` mixed a half intermediate with
a float32 one and raised ``DtypeMismatch`` — or, worse, rounded a float32
parameter update through float16.

Each case runs twice from the same seed, once inside ``autocast(metal,
float16)`` or ``autocast(cpu, bfloat16)`` and once outside, and the
parameters, the optimizer state and their dtypes must agree bit for bit.
A row that casts a float32 input (a LOWER row, or a new op an optimizer
starts to use) breaks it.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.init as init
import lucid.optim as optim
from lucid.amp import autocast
from lucid.test._fixtures.devices import metal_available

_SCOPES = [
    pytest.param(
        "metal",
        lucid.float16,
        id="metal-f16",
        marks=pytest.mark.skipif(not metal_available(), reason="Metal unavailable"),
    ),
    pytest.param("cpu", lucid.bfloat16, id="cpu-bf16"),
]

# Every optimizer in ``lucid.optim`` but SparseAdam and LBFGS, which need a
# different model and a closure, with options that reach their branches.
_OPTIMIZERS: dict[str, tuple[str, dict[str, Any]]] = {
    "SGD": ("SGD", {"lr": 0.1}),
    "SGD-nesterov-wd": (
        "SGD",
        {"lr": 0.1, "momentum": 0.9, "nesterov": True, "weight_decay": 0.01},
    ),
    "Adam": ("Adam", {"lr": 0.1}),
    "Adam-amsgrad-wd": ("Adam", {"lr": 0.1, "amsgrad": True, "weight_decay": 0.01}),
    "AdamW": ("AdamW", {"lr": 0.1}),
    "AdamW-amsgrad": ("AdamW", {"lr": 0.1, "amsgrad": True}),
    "RMSprop": ("RMSprop", {"lr": 0.1}),
    "RMSprop-centered-momentum": (
        "RMSprop",
        {"lr": 0.1, "centered": True, "momentum": 0.9, "weight_decay": 0.01},
    ),
    "Adagrad": ("Adagrad", {"lr": 0.1, "lr_decay": 0.01, "weight_decay": 0.01}),
    "Adadelta": ("Adadelta", {"lr": 0.1, "weight_decay": 0.01}),
    "Adamax": ("Adamax", {"lr": 0.1, "weight_decay": 0.01}),
    "RAdam": ("RAdam", {"lr": 0.1, "weight_decay": 0.01}),
    "NAdam": ("NAdam", {"lr": 0.1}),
    "NAdam-decoupled": ("NAdam", {"lr": 0.1, "weight_decay": 0.1, "decoupled_weight_decay": True}),
    "ASGD": ("ASGD", {"lr": 0.1, "t0": 1.0}),
    "Rprop": ("Rprop", {"lr": 0.1}),
}

_STEPS = 3


def _params_and_state(model: nn.Module, opt: optim.Optimizer) -> list[Any]:
    """Every parameter and every state value, in a fixed order."""
    out: list[Any] = [p for p in model.parameters()]
    for index in sorted(opt.state):
        for key in sorted(opt.state[index]):
            out.append((key, opt.state[index][key]))
    return out


def _assert_identical(inside: list[Any], outside: list[Any]) -> None:
    assert len(inside) == len(outside)
    for a, b in zip(inside, outside, strict=True):
        if isinstance(a, tuple):
            assert a[0] == b[0]
            a, b = a[1], b[1]
        if isinstance(a, lucid.Tensor):
            assert isinstance(b, lucid.Tensor)
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a.detach().numpy(), b.detach().numpy())
        else:
            assert a == b


def _linear_run(name: str, device: str, amp: lucid.dtype | None) -> list[Any]:
    cls, kwargs = _OPTIMIZERS[name]
    lucid.manual_seed(0)
    model = nn.Linear(4, 3).to(device)
    data = lucid.randn(5, 4, device=device)
    opt = getattr(optim, cls)(model.parameters(), **kwargs)
    for _ in range(_STEPS):
        opt.zero_grad()
        (model(data) ** 2).sum().backward()
        if amp is None:
            opt.step()
        else:
            with autocast(device_type=device, dtype=amp):
                opt.step()
    return _params_and_state(model, opt)


@pytest.mark.parametrize(("device", "amp"), _SCOPES)
@pytest.mark.parametrize("name", sorted(_OPTIMIZERS))
def test_optimizer_step_inside_autocast_is_the_step_outside(
    name: str, device: str, amp: lucid.dtype
) -> None:
    inside = _linear_run(name, device, amp)
    outside = _linear_run(name, device, None)
    _assert_identical(inside, outside)
    assert all(p.dtype == lucid.float32 for p in inside if isinstance(p, lucid.Tensor))


def _lbfgs_run(device: str, amp: lucid.dtype | None) -> list[Any]:
    lucid.manual_seed(0)
    model = nn.Linear(4, 3).to(device)
    data = lucid.randn(5, 4, device=device)
    opt = optim.LBFGS(model.parameters(), lr=0.5, max_iter=4, history_size=3)

    def closure() -> lucid.Tensor:
        # The model's forward is the caller's; only the step is under test.
        with autocast(device_type=device, enabled=False):
            opt.zero_grad()
            loss = (model(data) ** 2).sum()
            loss.backward()
        return loss

    for _ in range(2):
        if amp is None:
            opt.step(closure)
        else:
            with autocast(device_type=device, dtype=amp):
                opt.step(closure)
    return _params_and_state(model, opt)


@pytest.mark.parametrize(("device", "amp"), _SCOPES)
def test_lbfgs_step_inside_autocast_is_the_step_outside(device: str, amp: lucid.dtype) -> None:
    _assert_identical(_lbfgs_run(device, amp), _lbfgs_run(device, None))


def _sparse_adam_run(device: str, amp: lucid.dtype | None) -> list[Any]:
    lucid.manual_seed(0)
    model = nn.Embedding(10, 4).to(device)
    opt = optim.SparseAdam(model.parameters(), lr=0.1)
    for rows in ([1, 3, 3], [0, 3, 7]):
        opt.zero_grad()
        (model(lucid.tensor(rows, device=device)) ** 2).sum().backward()
        if amp is None:
            opt.step()
        else:
            with autocast(device_type=device, dtype=amp):
                opt.step()
    return _params_and_state(model, opt)


@pytest.mark.parametrize(("device", "amp"), _SCOPES)
def test_sparse_adam_step_inside_autocast_is_the_step_outside(
    device: str, amp: lucid.dtype
) -> None:
    _assert_identical(_sparse_adam_run(device, amp), _sparse_adam_run(device, None))


def test_every_optimizer_is_covered() -> None:
    covered = {cls for cls, _ in _OPTIMIZERS.values()} | {"LBFGS", "SparseAdam"}
    exported = {
        name
        for name in optim.__all__
        if isinstance(getattr(optim, name), type)
        and issubclass(getattr(optim, name), optim.Optimizer)
        and name != "Optimizer"
    }
    assert covered == exported


# ── nn.init ──────────────────────────────────────────────────────────────────

_InitFn = Callable[[lucid.Tensor], object]
_INIT: dict[str, tuple[tuple[int, ...], _InitFn]] = {
    "constant_": ((4, 4), lambda w: init.constant_(w, 0.3)),
    "constant": ((4, 4), lambda w: init.constant(w, 0.3)),
    "dirac_": ((4, 4, 3), lambda w: init.dirac_(w)),
    "dirac": ((4, 4, 3), lambda w: init.dirac(w)),
    "eye_": ((4, 4), lambda w: init.eye_(w)),
    "eye": ((4, 4), lambda w: init.eye(w)),
    "kaiming_normal_": ((4, 4), lambda w: init.kaiming_normal_(w)),
    "kaiming_normal": ((4, 4), lambda w: init.kaiming_normal(w)),
    "kaiming_uniform_": ((4, 4), lambda w: init.kaiming_uniform_(w, a=5**0.5)),
    "kaiming_uniform": ((4, 4), lambda w: init.kaiming_uniform(w)),
    "normal_": ((4, 4), lambda w: init.normal_(w, 0.1, 0.5)),
    "normal": ((4, 4), lambda w: init.normal(w)),
    "ones_": ((4, 4), lambda w: init.ones_(w)),
    "orthogonal_": ((4, 4), lambda w: init.orthogonal_(w, gain=2.0)),
    "orthogonal": ((4, 4), lambda w: init.orthogonal(w)),
    "sparse_": ((4, 4), lambda w: init.sparse_(w, 0.5)),
    "sparse": ((4, 4), lambda w: init.sparse(w, 0.5)),
    "trunc_normal_": ((4, 4), lambda w: init.trunc_normal_(w, std=0.5, a=-0.6, b=0.6)),
    "uniform_": ((4, 4), lambda w: init.uniform_(w, -0.4, 0.7)),
    "uniform": ((4, 4), lambda w: init.uniform(w)),
    "xavier_normal_": ((4, 4), lambda w: init.xavier_normal_(w)),
    "xavier_normal": ((4, 4), lambda w: init.xavier_normal(w)),
    "xavier_uniform_": ((4, 4), lambda w: init.xavier_uniform_(w, gain=1.5)),
    "xavier_uniform": ((4, 4), lambda w: init.xavier_uniform(w)),
    "zeros_": ((4, 4), lambda w: init.zeros_(w)),
}


def _init_run(name: str, device: str, amp: lucid.dtype | None) -> lucid.Tensor:
    shape, fn = _INIT[name]
    lucid.manual_seed(0)
    w = lucid.empty(*shape, device=device)
    if amp is None:
        fn(w)
    else:
        with autocast(device_type=device, dtype=amp):
            fn(w)
    return w


@pytest.mark.parametrize(("device", "amp"), _SCOPES)
@pytest.mark.parametrize("name", sorted(_INIT))
def test_init_inside_autocast_is_the_init_outside(name: str, device: str, amp: lucid.dtype) -> None:
    inside = _init_run(name, device, amp)
    outside = _init_run(name, device, None)
    assert inside.dtype == outside.dtype == lucid.float32
    np.testing.assert_array_equal(inside.numpy(), outside.numpy())


def test_every_init_function_is_covered() -> None:
    assert set(_INIT) == set(init.__all__) - {"calculate_gain"}
