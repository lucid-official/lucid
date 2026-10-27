"""Metal convolutions over channel counts that are not multiples of 16.

MLX pads such channels up to a multiple of 16 inside its Metal convolution
and fills the padding from a zero scalar it frees before the GPU reads it.
When anything claims that buffer first — a scalar written while the work is
still queued, which asynchronous evaluation makes routine — the padded
channels hold that value and the convolution is wrong: whole layers off,
some NaN.  The backend therefore aligns the channels itself, as graph ops
(``gpu_align_conv_channels``), and MLX never pads.

The first group of tests holds the alignment to the CPU answer, forward and
both gradients, across the channel counts it touches and the ones it must
leave alone.  The last one recreates the race: without the alignment it
fails most iterations on MLX 0.32.1 and later.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

# (in, out): unaligned in / aligned out (the 2-D padding path), aligned in,
# at most 4 in, and unaligned out (the 3-D path pads and slices those too).
CHANNELS = [(8, 8), (5, 16), (12, 32), (16, 16), (3, 8), (8, 20), (24, 40)]


def _conv(rank: int, transposed: bool):
    if transposed:
        return {2: F.conv_transpose2d, 3: F.conv_transpose3d}[rank]
    return {2: F.conv2d, 3: F.conv3d}[rank]


def _run(x: np.ndarray, w: np.ndarray, rank: int, transposed: bool, stride: int, device: str):
    xt = lucid.tensor(x, device=device, requires_grad=True)
    wt = lucid.tensor(w, device=device, requires_grad=True)
    y = _conv(rank, transposed)(xt, wt, stride=stride, padding=1)
    # A fixed, non-uniform cotangent, so every output position weighs in.
    g = np.cos(np.arange(int(np.prod(y.shape)), dtype=np.float32)).reshape(y.shape)
    (y * lucid.tensor(g, device=device)).sum().backward()
    return y.numpy(), xt.grad.numpy(), wt.grad.numpy()


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("cin,cout", CHANNELS)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("transposed", [False, True])
def test_metal_convolution_matches_cpu(rank, cin, cout, stride, transposed):
    rng = np.random.default_rng(cin * 100 + cout)
    size = 12 if rank == 2 else 6  # output large enough for MLX's padded path
    x = rng.standard_normal((2, cin) + (size,) * rank).astype(np.float32)
    # Transposed weights are (in, out, K...); forward ones (out, in, K...).
    wshape = (cin, cout) if transposed else (cout, cin)
    w = (rng.standard_normal(wshape + (3,) * rank) * 0.2).astype(np.float32)

    cpu = _run(x, w, rank, transposed, stride, "cpu")
    metal = _run(x, w, rank, transposed, stride, "metal")
    for name, a, b in zip(("output", "grad x", "grad w"), cpu, metal):
        assert a.shape == b.shape, f"{name}: {a.shape} vs {b.shape}"
        scale = max(float(np.abs(a).max()), 1.0)
        err = float(np.abs(a - b).max()) / scale
        assert err < 1e-4, f"{name}: rel {err:.2e}"


@pytest.mark.parametrize(
    "xshape,wshape,fn",
    [((64, 8, 32, 32), (8, 8, 3, 3), F.conv2d), ((4, 3, 8, 16, 16), (24, 3, 3, 3, 3), F.conv3d)],
    ids=["conv2d-8-8", "conv3d-3-24"],
)
def test_convolution_survives_scalars_written_while_it_is_queued(xshape, wshape, fn):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = (rng.standard_normal(wshape) * 0.2).astype(np.float32)
    ref = fn(lucid.tensor(x), lucid.tensor(w), padding=1).numpy()
    xm = lucid.tensor(x, device="metal")
    wm = lucid.tensor(w, device="metal")
    small = lucid.ones(4, device="metal")
    _C_engine.eval_tensors([xm._impl, wm._impl, small._impl])

    wrong = 0
    for _ in range(20):
        y = fn(xm, wm, padding=1)
        _C_engine.eval_tensors_async([y._impl])
        # Scalar multiplies: each writes a fresh non-zero scalar from the host
        # while the convolution waits.  Held until it is read back below.
        _scalars = [small * (1000.0 + i) for i in range(64)]
        if not np.allclose(y.numpy(), ref, atol=1e-3):
            wrong += 1
    assert wrong == 0, f"{wrong}/20 convolutions came back wrong"
