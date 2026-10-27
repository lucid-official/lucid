"""``lucid.metal`` — Metal-stream specific behaviour.

Most of the real Metal validation lives in cross-device tests under
``unit/device/`` and ``unit/ops/`` — this file is the home for any
metal-only edge cases (allocation, sync, driver-level surface).
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import metal_available


@pytest.fixture(autouse=True)
def _require_metal() -> None:
    if not metal_available():
        pytest.skip("Metal not available on this host")


class TestMetalSurface:
    def test_module_present(self) -> None:
        # ``lucid.metal`` is always importable; behaviour just degrades
        # to no-op when Metal isn't there.
        assert hasattr(lucid, "metal")


class TestMetalAllocation:
    def test_zeros(self) -> None:
        t = lucid.zeros(8, device="metal")
        assert t.is_metal
        np.testing.assert_array_equal(t.numpy(), np.zeros(8))

    def test_round_trip(self) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = a.to(device="metal")
        c = b.to(device="cpu")
        np.testing.assert_array_equal(a.numpy(), c.numpy())


class TestMetalArith:
    def test_add(self) -> None:
        a = lucid.tensor([1.0, 2.0], device="metal")
        b = lucid.tensor([3.0, 4.0], device="metal")
        np.testing.assert_array_equal((a + b).numpy(), [4.0, 6.0])

    def test_matmul(self) -> None:
        a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device="metal")
        b = lucid.tensor([[5.0, 6.0], [7.0, 8.0]], device="metal")
        np.testing.assert_array_equal((a @ b).numpy(), [[19.0, 22.0], [43.0, 50.0]])


class TestMetalLazyTransposeBridge:
    # Regression: the GPU bridge used to memcpy the underlying buffer with
    # array.data<T>(), which ignores stride metadata.  Lazy transposes (e.g.
    # the deferred NHWC→NCHW perm in conv_nd_forward) left the data buffer
    # in the source layout and .numpy() returned bytes in the wrong order.
    # download_gpu_to_cpu now wraps in mlx::core::contiguous() before eval.
    def test_permute_view_matches_contiguous(self) -> None:
        t = lucid.randn(2, 3, 4, 5, device="metal")
        view = t.permute((0, 2, 3, 1))
        np.testing.assert_allclose(view.numpy(), view.contiguous().numpy())

    def test_grouped_conv2d_output_bytes(self) -> None:
        import lucid.nn.functional as F

        x = lucid.randn(1, 2, 2, 2, device="metal")
        w = lucid.randn(2, 1, 1, 1, device="metal")
        y = F.conv2d(x, w, groups=2)
        np.testing.assert_allclose(y.numpy(), y.contiguous().numpy())

    def test_grad_bytes_through_grouped_backward(self) -> None:
        import lucid.nn.functional as F

        x = lucid.randn(2, 4, 4, 4, device="metal", requires_grad=True)
        w = lucid.randn(4, 1, 3, 3, device="metal", requires_grad=True)
        F.conv2d(x, w, groups=4).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), x.grad.contiguous().numpy())

    def test_mpsgraph_kernel_on_lazy_transpose(self) -> None:
        # Regression: MpsBridge.array_to_buffer used to hand MPSGraph the
        # raw MTLBuffer of a non-contiguous MLX array (e.g. the lazy NHWC
        # transpose view left behind by conv_nd_forward).  MPSGraph reads
        # the buffer linearly in the declared logical shape order, so the
        # NHWC-laid-out bytes appeared in the wrong slots — F.gelu(conv(x))
        # came out element-shuffled and silently wrong (diff ≈ 1.9 vs
        # correct).  array_to_buffer now forces row-contiguous on entry.
        import math

        import lucid.nn as nn
        import lucid.nn.functional as F

        lucid.manual_seed(0)
        c = nn.Conv2d(3, 4, 3, padding=1).to("metal")
        x = lucid.randn(1, 3, 4, 4, device="metal")
        y = c(x)
        y_gelu = F.gelu(y)

        # Compare element-wise against host-side gelu_exact computed from
        # the same conv output.
        for i in range(y.numpy().size):
            v_in = float(y.numpy().flat[i])
            v_out = float(y_gelu.numpy().flat[i])
            expected = 0.5 * v_in * (1.0 + math.erf(v_in / math.sqrt(2.0)))
            assert abs(v_out - expected) < 1e-5, (
                f"gelu_exact mismatch at idx {i}: " f"input={v_in} got={v_out} expected={expected}"
            )
