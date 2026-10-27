"""CPU optimizer support and parity for float16 and bfloat16 parameters (LCD-71)."""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.test._fixtures.devices import metal_available

_CASES: list[tuple[str, dict[str, Any]]] = [
    ("SGD", {"lr": 0.1}),
    ("SGD", {"lr": 0.1, "momentum": 0.9, "dampening": 0.2, "weight_decay": 0.1}),
    ("SGD", {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1, "nesterov": True}),
    ("Adam", {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1, "amsgrad": False}),
    ("Adam", {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1, "amsgrad": True}),
    (
        "AdamW",
        {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1, "amsgrad": False},
    ),
    ("AdamW", {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1, "amsgrad": True}),
    ("Adamax", {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1}),
    ("NAdam", {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1}),
    ("RAdam", {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "weight_decay": 0.1}),
    ("RAdam", {"lr": 0.05, "betas": (0.7, 0.5), "eps": 1e-3, "weight_decay": 0.1}),
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
    ("Adadelta", {"lr": 0.5, "rho": 0.7, "eps": 1e-3, "weight_decay": 0.1}),
    (
        "RMSprop",
        {
            "lr": 0.01,
            "alpha": 0.8,
            "eps": 1e-3,
            "weight_decay": 0.1,
            "momentum": 0.9,
            "centered": True,
        },
    ),
    ("Rprop", {"lr": 0.05, "etas": (0.4, 1.5), "step_sizes": (1e-2, 0.2)}),
    ("ASGD", {"lr": 0.05, "lambd": 0.1, "alpha": 0.5, "t0": 2.0, "weight_decay": 0.1}),
    ("SparseAdam", {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3}),
    ("SparseAdam", {"lr": 0.05, "betas": (0.7, 0.8), "eps": 1e-3, "maximize": True}),
]


def _case_id(case: tuple[str, dict[str, Any]]) -> str:
    name, kw = case
    return name + "".join(f"-{k}={v}" for k, v in kw.items())


_CASE_PARAMS = [pytest.param(c, id=_case_id(c)) for c in _CASES]

_HALF_DTYPES = [
    pytest.param(lucid.float16, 1e-2, 1e-3, id="float16"),
    pytest.param(lucid.bfloat16, 2e-2, 1e-2, id="bfloat16"),
]


@pytest.mark.parametrize("dtype, rtol, atol", _HALF_DTYPES)
def test_sgd_cpu_half_repro(dtype: lucid.dtype, rtol: float, atol: float) -> None:
    p = nn.Parameter(lucid.tensor([1.0, 2.0, 3.0], dtype=dtype, device="cpu", requires_grad=True))
    loss = (p * 2.0).sum()
    loss.backward()
    optim.SGD([p], lr=0.1).step()
    assert p.dtype == dtype
    np.testing.assert_allclose(
        p.detach().float().numpy(),
        np.array([0.8, 1.8, 2.8], dtype=np.float32),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parity
@pytest.mark.parametrize("case", _CASE_PARAMS)
@pytest.mark.parametrize("dtype, rtol, atol", _HALF_DTYPES)
def test_cpu_half_optimizers_match_ref_and_metal(
    case: tuple[str, dict[str, Any]], dtype: lucid.dtype, rtol: float, atol: float, ref: Any
) -> None:
    name, kw = case
    ref_dt = ref.float16 if dtype == lucid.float16 else ref.bfloat16
    rng = np.random.default_rng(42)
    init = rng.standard_normal((4, 3)).astype(np.float32)

    p_cpu = nn.Parameter(lucid.tensor(init.copy()).to(device="cpu", dtype=dtype))
    opt_cpu = getattr(optim, name)([p_cpu], **kw)

    p_ref = ref.nn.Parameter(ref.from_numpy(init.copy()).to(dtype=ref_dt))
    opt_ref = getattr(ref.optim, name)([p_ref], **kw)

    has_metal = metal_available()
    if has_metal:
        p_gpu = nn.Parameter(lucid.tensor(init.copy()).to(device="metal", dtype=dtype))
        opt_gpu = getattr(optim, name)([p_gpu], **kw)

    for step in range(4):
        g = rng.standard_normal((4, 3)).astype(np.float32)
        if name == "SparseAdam":
            g[step % 4] = 0.0
        p_cpu.grad = lucid.tensor(g.copy()).to(device="cpu", dtype=dtype)
        opt_cpu.step()
        assert p_cpu.dtype == dtype

        tg = ref.from_numpy(g.copy()).to(dtype=ref_dt)
        p_ref.grad = tg.to_sparse(1) if name == "SparseAdam" else tg
        opt_ref.step()

        cpu_val = p_cpu.detach().float().numpy()
        ref_val = p_ref.detach().float().numpy()
        np.testing.assert_allclose(
            cpu_val,
            ref_val,
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} ({dtype}) CPU vs ref at step {step + 1}",
        )

        if has_metal:
            p_gpu.grad = lucid.tensor(g.copy()).to(device="metal", dtype=dtype)
            opt_gpu.step()
            gpu_val = p_gpu.detach().float().numpy()
            np.testing.assert_allclose(
                cpu_val,
                gpu_val,
                rtol=rtol,
                atol=atol,
                err_msg=f"{name} ({dtype}) CPU vs Metal at step {step + 1}",
            )


@pytest.mark.parametrize("case", _CASE_PARAMS)
@pytest.mark.parametrize("dtype, _rtol, _atol", _HALF_DTYPES)
def test_cpu_half_optimizer_state_dict_roundtrip(
    case: tuple[str, dict[str, Any]], dtype: lucid.dtype, _rtol: float, _atol: float
) -> None:
    name, kw = case
    rng = np.random.default_rng(7)
    init = rng.standard_normal((4, 3)).astype(np.float32)

    p1 = nn.Parameter(lucid.tensor(init.copy()).to(device="cpu", dtype=dtype))
    opt1 = getattr(optim, name)([p1], **kw)
    for step in range(2):
        g = rng.standard_normal((4, 3)).astype(np.float32)
        if name == "SparseAdam":
            g[step % 4] = 0.0
        p1.grad = lucid.tensor(g.copy()).to(device="cpu", dtype=dtype)
        opt1.step()

    sd = opt1.state_dict()
    p2 = nn.Parameter(p1.detach().clone())
    opt2 = getattr(optim, name)([p2], **kw)
    opt2.load_state_dict(sd)

    g_next = rng.standard_normal((4, 3)).astype(np.float32)
    p1.grad = lucid.tensor(g_next.copy()).to(device="cpu", dtype=dtype)
    p2.grad = lucid.tensor(g_next.copy()).to(device="cpu", dtype=dtype)
    opt1.step()
    opt2.step()

    np.testing.assert_array_equal(
        p1.detach().float().numpy(),
        p2.detach().float().numpy(),
        err_msg=f"{name} ({dtype}) state_dict roundtrip mismatch",
    )
