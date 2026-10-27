"""Every ``grid_sample`` setting against the reference, values and gradients.

``padding_mode="reflection"`` used to be mapped to ``border`` before the
kernel saw it, so it matched the reference's *border* result exactly and
its reflection result not at all — on both devices.  A setting that is
accepted has to compute what it names, so the matrix is the whole
product: every mode, every padding mode, both corner conventions, 4-D and
5-D wherever the reference accepts the pair, on every device, with the
gradient for the input and for the grid.

The grid is spread well past ``[-1, 1]`` (several reflections deep): an
interior-only grid agrees whatever the padding mode does.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._helpers.compare import assert_close

_MODES_4D = ["bilinear", "nearest", "bicubic"]
_MODES_5D = ["bilinear", "nearest"]
_PADDINGS = ["zeros", "border", "reflection"]

_CASES = [
    pytest.param(rank, mode, padding, corners, id=f"{rank}d-{mode}-{padding}-{tag}")
    for rank, modes in ((4, _MODES_4D), (5, _MODES_5D))
    for mode in modes
    for padding in _PADDINGS
    for corners, tag in ((True, "corners"), (False, "edges"))
]


def _operands(rank: int) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(rank)
    if rank == 4:
        x = rng.standard_normal((2, 3, 5, 6))
        grid = rng.standard_normal((2, 4, 7, 2)) * 1.8
    else:
        x = rng.standard_normal((2, 3, 4, 5, 6))
        grid = rng.standard_normal((2, 3, 4, 5, 3)) * 1.8
    return x.astype(np.float32), grid.astype(np.float32), rng


@pytest.mark.parity
class TestGridSampleMatrix:
    @pytest.mark.parametrize(("rank", "mode", "padding_mode", "align_corners"), _CASES)
    def test_values_and_gradients(
        self, ref: Any, device: str, rank: int, mode: str, padding_mode: str, align_corners: bool
    ) -> None:
        x, grid, rng = _operands(rank)
        kw = dict(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

        lx = lucid.tensor(x, device=device, requires_grad=True)
        lg = lucid.tensor(grid, device=device, requires_grad=True)
        rx = ref.tensor(x, requires_grad=True)
        rg = ref.tensor(grid, requires_grad=True)
        l_out = F.grid_sample(lx, lg, **kw)
        r_out = ref.nn.functional.grid_sample(rx, rg, **kw)
        assert_close(l_out, r_out.detach(), atol=1e-5, msg="value")

        # A random cotangent, so no gradient term cancels against another.
        weight = rng.standard_normal(tuple(r_out.shape)).astype(np.float32)
        (l_out * lucid.tensor(weight, device=device)).sum().backward()
        (r_out * ref.tensor(weight)).sum().backward()
        assert lx.grad is not None and lg.grad is not None
        assert_close(lx.grad, rx.grad, atol=1e-5, msg="input grad")
        assert_close(lg.grad, rg.grad, atol=1e-4, rtol=1e-4, msg="grid grad")


@pytest.mark.parity
class TestAffineGrid:
    @pytest.mark.parametrize(
        "size", [(2, 3, 5, 6), (2, 3, 4, 5, 6), (2, 1, 1, 3, 1)], ids=["2d", "3d", "3d-unit-axes"]
    )
    @pytest.mark.parametrize("align_corners", [True, False], ids=["corners", "edges"])
    # The reference warns about unit-size axes with align_corners; the
    # values are still the ones to match.
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_values_and_gradient(
        self, ref: Any, device: str, size: tuple[int, ...], align_corners: bool
    ) -> None:
        rank = len(size) - 2
        rng = np.random.default_rng(len(size))
        theta = rng.standard_normal((2, rank, rank + 1)).astype(np.float32)

        lt = lucid.tensor(theta, device=device, requires_grad=True)
        rt = ref.tensor(theta, requires_grad=True)
        l_out = F.affine_grid(lt, size, align_corners=align_corners)
        r_out = ref.nn.functional.affine_grid(rt, size, align_corners=align_corners)
        assert_close(l_out, r_out.detach(), atol=1e-5, msg="value")

        weight = rng.standard_normal(tuple(r_out.shape)).astype(np.float32)
        (l_out * lucid.tensor(weight, device=device)).sum().backward()
        (r_out * ref.tensor(weight)).sum().backward()
        assert lt.grad is not None
        assert_close(lt.grad, rt.grad, atol=1e-4, msg="theta grad")
