"""Unit tests for lucid.metal.run_kernel (Phase 18 MetalKernelRunner)."""

import numpy as np

import lucid
import lucid.metal as metal

# ── MSL sources ──────────────────────────────────────────────────────────────

_MSL_ADD1 = """
#include <metal_stdlib>
using namespace metal;
kernel void add_one(
    device const float* inp [[buffer(0)]],
    device float* out       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = inp[gid] + 1.0f;
}
"""

_MSL_VADD = """
#include <metal_stdlib>
using namespace metal;
kernel void vadd(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    c[gid] = a[gid] + b[gid];
}
"""

_MSL_RELU = """
#include <metal_stdlib>
using namespace metal;
kernel void relu(
    device const float* x [[buffer(0)]],
    device float* y       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    y[gid] = max(0.0f, x[gid]);
}
"""


class TestMetalKernelRunner:
    """Phase 18: custom MSL kernel execution via lucid.metal.run_kernel."""

    def test_single_input_add_one(self) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        y = metal.run_kernel(_MSL_ADD1, "add_one", [x], (4,), grid=(4, 1, 1), threads=(1, 1, 1))
        np.testing.assert_allclose(y.numpy(), [2.0, 3.0, 4.0, 5.0], atol=1e-6)

    def test_two_input_vector_add(self) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = lucid.tensor([10.0, 20.0, 30.0])
        c = metal.run_kernel(_MSL_VADD, "vadd", [a, b], (3,), grid=(3, 1, 1), threads=(1, 1, 1))
        np.testing.assert_allclose(c.numpy(), [11.0, 22.0, 33.0], atol=1e-6)

    def test_relu_kernel(self) -> None:
        x = lucid.tensor([-1.0, 2.0, -0.5, 0.0, 3.0])
        y = metal.run_kernel(_MSL_RELU, "relu", [x], (5,), grid=(5, 1, 1), threads=(1, 1, 1))
        np.testing.assert_allclose(y.numpy(), [0.0, 2.0, 0.0, 0.0, 3.0], atol=1e-6)

    def test_metal_input_tensor(self) -> None:
        """Metal-device input tensors are accepted without copy."""
        x = lucid.tensor([5.0, 6.0, 7.0]).to(device="metal")
        y = metal.run_kernel(_MSL_ADD1, "add_one", [x], (3,), grid=(3, 1, 1), threads=(1, 1, 1))
        np.testing.assert_allclose(y.numpy(), [6.0, 7.0, 8.0], atol=1e-6)

    def test_output_is_cpu_tensor(self) -> None:
        """run_kernel always returns a CPU tensor (SharedStorage)."""
        x = lucid.tensor([1.0])
        y = metal.run_kernel(_MSL_ADD1, "add_one", [x], (1,), grid=(1, 1, 1), threads=(1, 1, 1))
        assert y.device == lucid.device("cpu")

    def test_output_moveable_to_metal(self) -> None:
        """Output can be moved back to Metal for further GPU computation."""
        x = lucid.tensor([3.0])
        y = metal.run_kernel(_MSL_ADD1, "add_one", [x], (1,), grid=(1, 1, 1), threads=(1, 1, 1))
        y_gpu = y.to(device="metal")
        assert y_gpu.device == lucid.device("metal")
        np.testing.assert_allclose(y_gpu.cpu().numpy(), [4.0], atol=1e-6)

    def test_large_tensor(self) -> None:
        """Kernel dispatched over 1024 elements."""
        n = 1024
        x = lucid.ones(n)
        y = metal.run_kernel(_MSL_ADD1, "add_one", [x], (n,), grid=(n, 1, 1), threads=(1, 1, 1))
        np.testing.assert_allclose(y.numpy(), [2.0] * n, atol=1e-6)

    def test_run_metal_kernel_is_available(self) -> None:
        """lucid.metal.run_kernel is importable and callable."""
        assert callable(metal.run_kernel)
