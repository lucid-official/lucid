"""A bfloat16 convolution on the CPU is a bfloat16 convolution (LCD-385).

The CPU convolution has no half path, so it widens to float32 at the door
and rounds back on the way out.  It did that with a loop of its own that
read every 16-bit float as IEEE half and rounded the result back to
float16 bits, whatever the dtype.  A bfloat16 input was decoded as
float16 and the output was float16 bits labelled bfloat16.  Every
bfloat16 convolution on the CPU was off by up to 1e12.  The forward and
both gradients must match float32 on the same rounded values.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

_HALVES = {"float16": lucid.float16, "bfloat16": lucid.bfloat16}
_TOL = {"float16": 2e-2, "bfloat16": 6e-2}

# name -> (functional, input shape, weight shape)
_CONVS = {
    "conv1d": (F.conv1d, (2, 3, 9), (4, 3, 3)),
    "conv2d": (F.conv2d, (2, 3, 6, 5), (4, 3, 3, 2)),
    "conv3d": (F.conv3d, (1, 2, 4, 4, 3), (3, 2, 2, 3, 2)),
    "conv_transpose1d": (F.conv_transpose1d, (2, 3, 7), (3, 4, 3)),
    "conv_transpose2d": (F.conv_transpose2d, (2, 3, 4, 5), (3, 4, 3, 2)),
}


def _rounded(seed: int, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    a = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return lucid.tensor(a).to(dtype).to(lucid.float32).numpy()


def _run(name: str, arrays: list[np.ndarray], dtype: Any) -> list[np.ndarray]:
    fn = _CONVS[name][0]
    x, w, b = (lucid.tensor(a, dtype=dtype, requires_grad=True) for a in arrays)
    out = fn(x, w, b, padding=1)
    assert out.dtype == dtype
    out.sum().backward()
    assert x.grad is not None and w.grad is not None and b.grad is not None
    return [t.detach().to(lucid.float32).numpy() for t in (out, x.grad, w.grad, b.grad)]


@pytest.mark.parametrize("dtype", list(_HALVES))
@pytest.mark.parametrize("name", list(_CONVS))
def test_half_conv_matches_float32_on_cpu(name: str, dtype: str) -> None:
    _, x_shape, w_shape = _CONVS[name]
    out_channels = w_shape[1] if "transpose" in name else w_shape[0]
    half = _HALVES[dtype]
    arrays = [
        _rounded(0, x_shape, half),
        _rounded(1, w_shape, half),
        _rounded(2, (out_channels,), half),
    ]
    got = _run(name, arrays, half)
    want = _run(name, arrays, lucid.float32)
    for what, g, w in zip(
        ("output", "input grad", "weight grad", "bias grad"), got, want, strict=True
    ):
        np.testing.assert_allclose(
            g, w, rtol=0, atol=_TOL[dtype] * max(1.0, float(np.abs(w).max())), err_msg=what
        )
