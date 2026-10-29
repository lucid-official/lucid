"""OpInfo differential fuzzing suite (Phase B1 & 2nd-order strided VJP).

Systematically exercises ``OPINFO_REGISTRY`` (150+ operators across unary/special
math, binary/ternary broadcast math, multi-axis reductions & scans, indexing &
view layout ops, linear algebra contractions/decompositions, FFT/spectral ops,
and ``nn.functional`` layers/losses/attention) along five orthogonal verification
axes:

1. Multi-argument & broadcast forward parity across ``(cpu, metal)`` ×
   ``(float32, float16, bfloat16)``.
2. Non-contiguous strided view forward + backward parity across ``(cpu, metal)``.
3. First-order backward VJP parity under non-uniform upstream cotangents across
   ``(cpu, metal)``.
4. Second-order Hessian-vector product (``create_graph=True``) × non-contiguous
   strided view parity across ``(cpu, metal)``.
5. Central-difference float64 numerical Jacobian verification (``gradcheck``).
"""

from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
import lucid.special as lspecial
from lucid.test._fixtures.devices import metal_available
from lucid.test._helpers.compare import assert_close
from lucid.test.parity.opinfo import (
    OPINFO_REGISTRY,
    OpInfo,
    verify_opinfo_backward,
    verify_opinfo_forward,
    verify_opinfo_gradcheck,
    verify_opinfo_second_order_strided,
    verify_opinfo_strided_views,
)

_DEVICES: tuple[str, ...] = ("cpu", "metal") if metal_available() else ("cpu",)
_DTYPES: tuple[str, ...] = ("float32", "float16", "bfloat16")
_OP_IDS: list[str] = [op.name for op in OPINFO_REGISTRY]


@pytest.mark.parity
def test_opinfo_registry_completeness() -> None:
    """Ensure the OpInfo registry covers all 16 operator families without duplicate names."""
    names = [op.name for op in OPINFO_REGISTRY]
    assert len(names) == len(set(names)), "Duplicate OpInfo names in OPINFO_REGISTRY"
    assert len(OPINFO_REGISTRY) >= 150
    families = {op.family for op in OPINFO_REGISTRY}
    expected_families = {
        "unary_math",
        "binary_math",
        "ternary_math",
        "reduction",
        "scan",
        "indexing",
        "view_layout",
        "linalg",
        "fft",
        "nn_activation",
        "nn_norm",
        "nn_layer",
        "nn_pool",
        "nn_spatial",
        "nn_attention",
        "nn_loss",
    }
    assert expected_families.issubset(families)


@pytest.mark.parity
@pytest.mark.parametrize("dtype_name", _DTYPES)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("op", OPINFO_REGISTRY, ids=_OP_IDS)
def test_opinfo_forward_parity(op: OpInfo, device: str, dtype_name: str, ref: Any) -> None:
    """Multi-argument, broadcast, and boundary-value forward parity vs reference."""
    verify_opinfo_forward(op, ref, device, dtype_name)


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("op", OPINFO_REGISTRY, ids=_OP_IDS)
def test_opinfo_backward_parity(op: OpInfo, device: str, ref: Any) -> None:
    """First-order VJP parity under non-uniform upstream cotangents."""
    verify_opinfo_backward(op, ref, device, "float32")


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("op", OPINFO_REGISTRY, ids=_OP_IDS)
def test_opinfo_strided_view_parity(op: OpInfo, device: str, ref: Any) -> None:
    """Forward and backward parity when inputs are non-contiguous strided views."""
    verify_opinfo_strided_views(op, ref, device, "float32")


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("op", OPINFO_REGISTRY, ids=_OP_IDS)
def test_opinfo_second_order_strided_parity(op: OpInfo, device: str, ref: Any) -> None:
    """Second-order HVP (create_graph=True) parity through non-contiguous strided views."""
    verify_opinfo_second_order_strided(op, ref, device, "float32")


@pytest.mark.parity
@pytest.mark.parametrize("op", OPINFO_REGISTRY, ids=_OP_IDS)
def test_opinfo_numerical_gradcheck(op: OpInfo, ref: Any) -> None:
    """Central-difference float64 numerical Jacobian verification."""
    verify_opinfo_gradcheck(op, ref)


@pytest.mark.parity
@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    op_idx=st.integers(min_value=0, max_value=len(OPINFO_REGISTRY) - 1),
    use_metal=st.booleans(),
    use_second_order=st.booleans(),
)
def test_opinfo_hypothesis_strided_fuzz(
    op_idx: int, use_metal: bool, use_second_order: bool, ref: Any
) -> None:
    """Property-based fuzzing over random OpInfo entries, devices, and strided VJP/HVP modes."""
    op = OPINFO_REGISTRY[op_idx]
    device = "metal" if (use_metal and metal_available()) else "cpu"
    if use_second_order:
        verify_opinfo_second_order_strided(op, ref, device, "float32")
    else:
        verify_opinfo_strided_views(op, ref, device, "float32")


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
def test_second_order_origin_and_metal_pool_regressions(device: str, ref: Any) -> None:
    """Regression test for 2nd-order derivatives at x=0 and Metal MaxPool2d create_graph=True."""
    # 1. Exact 2nd-order derivatives at x = 0.0 for expm1, erfc, sinc, and special.i0e.
    cases = (
        ("expm1", lucid.expm1, 1.0, 1.0),
        ("erfc", lucid.erfc, -2.0 / np.sqrt(np.pi), 0.0),
        ("sinc", lucid.sinc, 0.0, -(np.pi**2) / 3.0),
        ("special.i0e", lspecial.i0e, 0.0, 0.5),
    )
    for name, fn, expected_d1, expected_d2 in cases:
        x = lucid.tensor([0.0, 0.5], dtype=lucid.float32, device=device, requires_grad=True)
        y = fn(x)
        (g1,) = lucid.autograd.grad(y.sum(), (x,), create_graph=True)
        (g2,) = lucid.autograd.grad(g1[0], (x,))
        assert_close(
            float(g1[0].cpu().item()),
            float(expected_d1),
            atol=1e-5,
            rtol=1e-5,
            msg=f"{name} 1st derivative at 0 ({device})",
        )
        assert_close(
            float(g2[0].cpu().item()),
            float(expected_d2),
            atol=1e-5,
            rtol=1e-5,
            msg=f"{name} 2nd derivative at 0 ({device})",
        )

    # 2. Padded & strided MaxPool2d backward with create_graph=True on CPU & Metal.
    rng = np.random.default_rng(123)
    x_np = rng.uniform(-1.0, 2.0, size=(2, 2, 5, 5)).astype(np.float32)
    cot_np = rng.uniform(-1.0, 1.0, size=(2, 2, 3, 3)).astype(np.float32)
    lx = lucid.tensor(x_np, dtype=lucid.float32, device=device, requires_grad=True)
    rx = ref.tensor(x_np, dtype=ref.float32, requires_grad=True)
    l_out = F.max_pool2d(lx, kernel_size=3, stride=2, padding=1)
    r_out = ref.nn.functional.max_pool2d(rx, kernel_size=3, stride=2, padding=1)
    l_cot = lucid.tensor(cot_np, dtype=lucid.float32, device=device)
    r_cot = ref.tensor(cot_np, dtype=ref.float32)
    (lg1,) = lucid.autograd.grad((l_out * l_cot).sum(), (lx,), create_graph=True)
    (rg1,) = ref.autograd.grad((r_out * r_cot).sum(), (rx,), create_graph=True)
    assert_close(lg1, rg1, atol=1e-5, rtol=1e-5, msg=f"max_pool2d create_graph grad ({device})")
