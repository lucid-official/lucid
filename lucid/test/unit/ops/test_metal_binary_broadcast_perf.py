"""Metal binary broadcast parity, autograd, and saved-version contracts.

Verifies that passing broadcast operands directly to MLX binary primitives in
``BinaryKernel`` (without ``contiguous(broadcast_to(...))`` materialisation)
preserves forward and backward numerical parity against the CPU / reference,
and upholds the saved-tensor version contract across:

* Ops: ``+``, ``-``, ``*``, ``/``, ``**``, ``maximum``, ``minimum``, ``atan2``,
  ``==``, ``<``, ``&``, ``|``, ``^``, ``<<``, ``>>``, ``//``, ``%``
* Broadcast / layout geometries: scalar 0-d, row ``(1, N)``, col ``(M, 1)``,
  3-D broadcast, transposed view, sliced view, negative-stride (flipped) view
* ``requires_grad=False`` and ``requires_grad=True`` (including ``create_graph=True``
  and saved-tensor version checks on in-place mutation)
"""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine

BinaryFn: TypeAlias = Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]

FLOAT_OPS: dict[str, BinaryFn] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
    "maximum": lucid.maximum,
    "minimum": lucid.minimum,
    "atan2": lucid.atan2,
    "floordiv": lambda a, b: a // b,
    "remainder": lambda a, b: a % b,
}

FLOAT_COMPARE_OPS: dict[str, BinaryFn] = {
    "eq": lambda a, b: a == b,
    "lt": lambda a, b: a < b,
}

INT_OPS: dict[str, BinaryFn] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "pow": lambda a, b: a**b,
    "maximum": lucid.maximum,
    "minimum": lucid.minimum,
    "eq": lambda a, b: a == b,
    "lt": lambda a, b: a < b,
    "bitwise_and": lambda a, b: a & b,
    "bitwise_or": lambda a, b: a | b,
    "bitwise_xor": lambda a, b: a ^ b,
    "lshift": lambda a, b: a << b,
    "rshift": lambda a, b: a >> b,
    "floordiv": lambda a, b: a // b,
    "remainder": lambda a, b: a % b,
}

LAYOUT_CASES: tuple[str, ...] = (
    "scalar_0d",
    "row_1xN",
    "col_Mx1",
    "broadcast_3d",
    "transposed_view",
    "sliced_view",
    "negative_stride_view",
)


def _build_float_pair(
    layout: str, device: str, requires_grad: bool
) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    """Return ``(a_leaf, b_leaf, a_op, b_op)`` positive enough for ``pow`` / ``div``."""
    rng = np.random.default_rng(42)
    base_a = (np.abs(rng.standard_normal((4, 6))) + 0.5).astype(np.float32)

    if layout == "scalar_0d":
        raw_b = np.array(1.75, dtype=np.float32)
        a_leaf = lucid.tensor(base_a, device=device, requires_grad=requires_grad)
        b_leaf = lucid.tensor(raw_b, device=device, requires_grad=requires_grad)
        return a_leaf, b_leaf, a_leaf, b_leaf

    if layout == "row_1xN":
        raw_b = (np.abs(rng.standard_normal((1, 6))) + 0.5).astype(np.float32)
        a_leaf = lucid.tensor(base_a, device=device, requires_grad=requires_grad)
        b_leaf = lucid.tensor(raw_b, device=device, requires_grad=requires_grad)
        return a_leaf, b_leaf, a_leaf, b_leaf

    if layout == "col_Mx1":
        raw_b = (np.abs(rng.standard_normal((4, 1))) + 0.5).astype(np.float32)
        a_leaf = lucid.tensor(base_a, device=device, requires_grad=requires_grad)
        b_leaf = lucid.tensor(raw_b, device=device, requires_grad=requires_grad)
        return a_leaf, b_leaf, a_leaf, b_leaf

    if layout == "broadcast_3d":
        raw_a = (np.abs(rng.standard_normal((2, 1, 5))) + 0.5).astype(np.float32)
        raw_b = (np.abs(rng.standard_normal((1, 4, 1))) + 0.5).astype(np.float32)
        a_leaf = lucid.tensor(raw_a, device=device, requires_grad=requires_grad)
        b_leaf = lucid.tensor(raw_b, device=device, requires_grad=requires_grad)
        return a_leaf, b_leaf, a_leaf, b_leaf

    if layout == "transposed_view":
        raw_a = (np.abs(rng.standard_normal((6, 4))) + 0.5).astype(np.float32)
        raw_b = (np.abs(rng.standard_normal((6, 1))) + 0.5).astype(np.float32)
        a_leaf = lucid.tensor(raw_a, device=device, requires_grad=requires_grad)
        b_leaf = lucid.tensor(raw_b, device=device, requires_grad=requires_grad)
        return a_leaf, b_leaf, a_leaf.t(), b_leaf.t()

    if layout == "sliced_view":
        raw_a = (np.abs(rng.standard_normal((5, 8))) + 0.5).astype(np.float32)
        raw_b = (np.abs(rng.standard_normal((3, 8))) + 0.5).astype(np.float32)
        a_leaf = lucid.tensor(raw_a, device=device, requires_grad=requires_grad)
        b_leaf = lucid.tensor(raw_b, device=device, requires_grad=requires_grad)
        return a_leaf, b_leaf, a_leaf[1:5, ::2], b_leaf[1:2, 1::2]

    if layout == "negative_stride_view":
        raw_b = (np.abs(rng.standard_normal((1, 6))) + 0.5).astype(np.float32)
        a_leaf = lucid.tensor(
            np.ascontiguousarray(base_a[::-1]),
            device=device,
            requires_grad=requires_grad,
        )
        b_leaf = lucid.tensor(
            np.ascontiguousarray(raw_b[:, ::-1]),
            device=device,
            requires_grad=requires_grad,
        )
        return a_leaf, b_leaf, lucid.flip(a_leaf, dims=(0,)), b_leaf[:, ::-1]

    raise ValueError(f"unknown layout {layout!r}")


def _build_int_pair(layout: str, device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    """Return ``(a, b)`` int32 broadcast operands with small positive values."""
    rng = np.random.default_rng(99)
    base_a = rng.integers(1, 7, size=(4, 6), dtype=np.int32)

    if layout == "scalar_0d":
        return (
            lucid.tensor(base_a, device=device),
            lucid.tensor(np.array(2, dtype=np.int32), device=device),
        )
    if layout == "row_1xN":
        raw_b = rng.integers(1, 5, size=(1, 6), dtype=np.int32)
        return lucid.tensor(base_a, device=device), lucid.tensor(raw_b, device=device)
    if layout == "col_Mx1":
        raw_b = rng.integers(1, 5, size=(4, 1), dtype=np.int32)
        return lucid.tensor(base_a, device=device), lucid.tensor(raw_b, device=device)
    if layout == "broadcast_3d":
        raw_a = rng.integers(1, 7, size=(2, 1, 5), dtype=np.int32)
        raw_b = rng.integers(1, 5, size=(1, 4, 1), dtype=np.int32)
        return lucid.tensor(raw_a, device=device), lucid.tensor(raw_b, device=device)
    if layout == "transposed_view":
        raw_a = rng.integers(1, 7, size=(6, 4), dtype=np.int32)
        raw_b = rng.integers(1, 5, size=(6, 1), dtype=np.int32)
        return (
            lucid.tensor(raw_a, device=device).t(),
            lucid.tensor(raw_b, device=device).t(),
        )
    if layout == "sliced_view":
        raw_a = rng.integers(1, 7, size=(5, 8), dtype=np.int32)
        raw_b = rng.integers(1, 5, size=(3, 8), dtype=np.int32)
        return (
            lucid.tensor(raw_a, device=device)[1:5, ::2],
            lucid.tensor(raw_b, device=device)[1:2, 1::2],
        )
    if layout == "negative_stride_view":
        raw_b = rng.integers(1, 5, size=(1, 6), dtype=np.int32)
        return (
            lucid.flip(lucid.tensor(base_a, device=device), dims=(0,)),
            lucid.tensor(raw_b, device=device)[:, ::-1],
        )
    raise ValueError(f"unknown layout {layout!r}")


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("layout", LAYOUT_CASES)
@pytest.mark.parametrize("op_name", list(FLOAT_OPS))
def test_metal_float_binary_broadcast_forward_and_backward_parity(
    op_name: str, layout: str, requires_grad: bool
) -> None:
    fn = FLOAT_OPS[op_name]
    a_cpu_leaf, b_cpu_leaf, a_cpu, b_cpu = _build_float_pair(layout, "cpu", requires_grad)
    a_gpu_leaf, b_gpu_leaf, a_gpu, b_gpu = _build_float_pair(layout, "metal", requires_grad)

    out_cpu = fn(a_cpu, b_cpu)
    out_gpu = fn(a_gpu, b_gpu)

    assert out_gpu.shape == out_cpu.shape
    assert out_gpu.dtype == out_cpu.dtype
    assert out_gpu.is_contiguous()
    assert out_gpu.requires_grad == requires_grad
    np.testing.assert_allclose(out_gpu.numpy(), out_cpu.numpy(), rtol=1e-5, atol=1e-5)

    # Also check reversed operand order (broadcasting first operand against second).
    rev_cpu = fn(b_cpu, a_cpu)
    rev_gpu = fn(b_gpu, a_gpu)
    assert rev_gpu.shape == rev_cpu.shape
    assert rev_gpu.is_contiguous()
    np.testing.assert_allclose(rev_gpu.numpy(), rev_cpu.numpy(), rtol=1e-5, atol=1e-5)

    if not requires_grad:
        return

    rng = np.random.default_rng(7)
    grad_np = rng.standard_normal(tuple(out_cpu.shape)).astype(np.float32)
    out_cpu.backward(lucid.tensor(grad_np, device="cpu"))
    out_gpu.backward(lucid.tensor(grad_np, device="metal"))

    assert a_gpu_leaf.grad is not None and a_cpu_leaf.grad is not None
    assert b_gpu_leaf.grad is not None and b_cpu_leaf.grad is not None
    assert a_gpu_leaf.grad.shape == a_cpu_leaf.grad.shape
    assert b_gpu_leaf.grad.shape == b_cpu_leaf.grad.shape
    np.testing.assert_allclose(
        a_gpu_leaf.grad.numpy(), a_cpu_leaf.grad.numpy(), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        b_gpu_leaf.grad.numpy(), b_cpu_leaf.grad.numpy(), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("layout", LAYOUT_CASES)
@pytest.mark.parametrize("op_name", list(FLOAT_COMPARE_OPS))
def test_metal_float_comparison_broadcast_parity(
    op_name: str, layout: str, requires_grad: bool
) -> None:
    fn = FLOAT_COMPARE_OPS[op_name]
    _, _, a_cpu, b_cpu = _build_float_pair(layout, "cpu", requires_grad)
    _, _, a_gpu, b_gpu = _build_float_pair(layout, "metal", requires_grad)

    out_cpu = fn(a_cpu, b_cpu)
    out_gpu = fn(a_gpu, b_gpu)
    assert out_gpu.shape == out_cpu.shape
    assert out_gpu.dtype == lucid.bool_
    assert not out_gpu.requires_grad
    np.testing.assert_array_equal(out_gpu.numpy(), out_cpu.numpy())


@pytest.mark.parametrize("layout", LAYOUT_CASES)
@pytest.mark.parametrize("op_name", list(INT_OPS))
def test_metal_integer_and_bitwise_broadcast_parity(op_name: str, layout: str) -> None:
    fn = INT_OPS[op_name]
    a_cpu, b_cpu = _build_int_pair(layout, "cpu")
    a_gpu, b_gpu = _build_int_pair(layout, "metal")

    out_cpu = fn(a_cpu, b_cpu)
    out_gpu = fn(a_gpu, b_gpu)
    assert out_gpu.shape == out_cpu.shape
    assert out_gpu.dtype == out_cpu.dtype
    assert out_gpu.is_contiguous()
    np.testing.assert_array_equal(out_gpu.numpy(), out_cpu.numpy())

    rev_cpu = fn(b_cpu, a_cpu)
    rev_gpu = fn(b_gpu, a_gpu)
    assert rev_gpu.shape == rev_cpu.shape
    np.testing.assert_array_equal(rev_gpu.numpy(), rev_cpu.numpy())


@pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div", "pow", "maximum", "minimum"])
@pytest.mark.parametrize("layout", ["scalar_0d", "row_1xN", "col_Mx1", "broadcast_3d"])
def test_metal_binary_broadcast_create_graph_second_derivative(op_name: str, layout: str) -> None:
    fn = FLOAT_OPS[op_name]

    def grads(device: str) -> tuple[np.ndarray, np.ndarray]:
        a_leaf, b_leaf, a_op, b_op = _build_float_pair(layout, device, True)
        y = fn(a_op, b_op)
        da, db = lucid.autograd.grad(
            (y * y).sum(), (a_leaf, b_leaf), create_graph=True, allow_unused=True
        )
        loss2 = da.sum() + db.sum()
        loss2.backward()
        assert a_leaf.grad is not None and b_leaf.grad is not None
        return a_leaf.grad.numpy(), b_leaf.grad.numpy()

    g_a_cpu, g_b_cpu = grads("cpu")
    g_a_gpu, g_b_gpu = grads("metal")
    np.testing.assert_allclose(g_a_gpu, g_a_cpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(g_b_gpu, g_b_cpu, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("layout", ["scalar_0d", "row_1xN", "col_Mx1"])
def test_metal_binary_broadcast_saved_version_contract(layout: str) -> None:
    # 1. Writing into a broadcast operand whose values backward reads raises VersionMismatch.
    a_leaf, b_leaf, a_op, b_op = _build_float_pair(layout, "metal", True)
    y = a_op * b_op
    with lucid.no_grad():
        b_leaf.mul_(2.0)
    with pytest.raises(_C_engine.VersionMismatch):
        y.sum().backward()

    # 2. Writing into an operand whose values backward does NOT read (e.g. add,
    #    or x * c where c does not require grad so dc is not computed) is allowed.
    a_leaf2, b_leaf2, a_op2, b_op2 = _build_float_pair(layout, "metal", True)
    y_add = a_op2 + b_op2
    with lucid.no_grad():
        a_leaf2.add_(1.0)
        b_leaf2.add_(1.0)
    y_add.sum().backward()
    assert a_leaf2.grad is not None and b_leaf2.grad is not None

    a_leaf3, _, a_op3, _ = _build_float_pair(layout, "metal", True)
    _, _, _, b_const = _build_float_pair(layout, "metal", False)
    y_const_mul = a_op3 * b_const
    with lucid.no_grad():
        a_leaf3.mul_(3.0)
    y_const_mul.sum().backward()
    assert a_leaf3.grad is not None


@pytest.mark.parametrize(
    "op_name",
    ["add", "sub", "mul", "div", "pow", "maximum", "minimum", "floordiv", "remainder"],
)
@pytest.mark.parametrize("layout", ["scalar_0d", "row_1xN", "col_Mx1", "broadcast_3d"])
def test_metal_binary_broadcast_does_not_materialize_expanded_operand(
    op_name: str, layout: str
) -> None:
    fn = FLOAT_OPS[op_name]
    _, _, a_gpu, b_gpu = _build_float_pair(layout, "metal", False)
    # Force any lazy view reads on the inputs to settle before counting.
    _ = a_gpu.numpy()
    _ = b_gpu.numpy()

    before = _C_engine.memory_stats(_C_engine.Device.GPU)
    out = fn(a_gpu, b_gpu)
    after = _C_engine.memory_stats(_C_engine.Device.GPU)

    out_bytes = out.numel() * 4
    allocated_bytes = after.current_bytes - before.current_bytes
    assert allocated_bytes == out_bytes
