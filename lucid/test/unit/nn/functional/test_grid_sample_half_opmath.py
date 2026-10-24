"""A half ``grid_sample`` composite samples in float32 and rounds once (LCD-616).

The parts of ``grid_sample`` written in Python — bicubic, every 5-D path,
and the reflection the 4-D kernel lacks — computed a float16 or bfloat16
input's coordinates, weights and sums in that dtype.  A float16 bicubic
sample came out about 2e-2 off the float32 answer on both devices, where
the reference's MPS kernel is one rounding (5e-4) off.  Each now widens a
half pair to float32 and rounds the result once, so on the same half
inputs it is the float32 answer, rounded, and so is its gradient.

The 4-D bilinear / nearest kernel without reflection is the engine's and
is not held here.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

_PADDINGS = ("zeros", "border", "reflection")
# (rank, mode, padding): every path the composite or the reflection serves.
_PATHS = [
    *((4, "bicubic", p) for p in _PADDINGS),
    (4, "bilinear", "reflection"),
    (4, "nearest", "reflection"),
    *((5, m, p) for m in ("bilinear", "nearest") for p in _PADDINGS),
]


def _operands(
    rank: int, device: str, dtype: lucid.dtype
) -> tuple[lucid.Tensor, lucid.Tensor]:
    rng = np.random.default_rng(0)
    if rank == 4:
        x, g = rng.standard_normal((2, 3, 16, 16)), rng.random((2, 9, 9, 2))
    else:
        x, g = rng.standard_normal((1, 2, 8, 8, 8)), rng.random((1, 5, 5, 5, 3))
    # Coordinates reach past the edges, so every padding has work to do.
    as_tensor = lambda a: lucid.tensor(a.astype(np.float32), device=device).to(dtype)
    return as_tensor(x), as_tensor(g * 2.6 - 1.3)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("half", ["float16", "bfloat16"])
@pytest.mark.parametrize(("rank", "mode", "padding"), _PATHS)
def test_a_half_sample_is_the_float32_one_rounded_once(
    device: str, rank: int, mode: str, padding: str, half: str, align_corners: bool
) -> None:
    dtype = getattr(lucid, half)
    x, grid = _operands(rank, device, dtype)
    x.requires_grad_(True)
    grid.requires_grad_(True)
    kwargs = {"mode": mode, "padding_mode": padding, "align_corners": align_corners}
    out = F.grid_sample(x, grid, **kwargs)
    out.float().sum().backward()

    x32 = x.detach().float().requires_grad_(True)
    grid32 = grid.detach().float().requires_grad_(True)
    want = F.grid_sample(x32, grid32, **kwargs)
    want.sum().backward()

    assert out.dtype == dtype
    np.testing.assert_array_equal(
        out.detach().float().numpy(), want.to(dtype).float().numpy()
    )
    assert x.grad is not None and x32.grad is not None
    assert grid.grad is not None and grid32.grad is not None
    np.testing.assert_array_equal(
        x.grad.float().numpy(), x32.grad.to(dtype).float().numpy()
    )
    np.testing.assert_array_equal(
        grid.grad.float().numpy(), grid32.grad.to(dtype).float().numpy()
    )


@pytest.mark.parametrize("padding", _PADDINGS)
def test_float16_bicubic_is_within_one_rounding_of_float32(
    device: str, padding: str
) -> None:
    """The LCD-616 report's measure: 1.1e-2 (reflection) before, ~1e-3 now."""
    x, grid = _operands(4, device, lucid.float16)
    kwargs = {"mode": "bicubic", "padding_mode": padding, "align_corners": False}
    out = F.grid_sample(x, grid, **kwargs).float()
    want = F.grid_sample(x.float(), grid.float(), **kwargs)
    assert float((out - want).abs().max().item()) <= 1e-3


def test_a_float32_sample_is_not_rounded() -> None:
    x, grid = _operands(4, "cpu", lucid.float32)
    out = F.grid_sample(x, grid, mode="bicubic", padding_mode="reflection")
    assert out.dtype == lucid.float32


def test_a_mixed_pair_is_still_refused() -> None:
    """Widening a pair of two dtypes would accept what the kernel refuses."""
    x, grid = _operands(4, "cpu", lucid.bfloat16)
    with pytest.raises(TypeError):
        F.grid_sample(x, grid.float(), padding_mode="reflection")
