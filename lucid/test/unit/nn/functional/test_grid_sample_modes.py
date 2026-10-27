"""``grid_sample`` computes the setting it was given, or refuses it by name.

``padding_mode="reflection"`` was once rewritten to ``border`` before the
kernel ran, so a reflection request clamped.  The values here are worked
out by hand on a 1-row ramp, where reflection and border disagree; the
full matrix against the reference lives in the parity tier
(``lucid/test/parity/nn/test_grid_sample_parity.py``).
"""

import pytest

import lucid
import lucid.nn.functional as F

# A ramp 0, 1, 2, 3 along x.  The sample sits at x = 1.5 (normalised),
# past the right edge.
_RAMP = [[[[0.0, 1.0, 2.0, 3.0]]]]


def _sample_ramp(device: str, padding_mode: str, align_corners: bool) -> float:
    x = lucid.tensor(_RAMP, device=device)
    grid = lucid.tensor([[[[1.5, 0.0]]]], device=device)
    out = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=align_corners)
    return float(out.item())


class TestReflectionIsNotBorder:
    def test_corners(self, device: str) -> None:
        # Pixel coordinate (1.5 + 1) / 2 * 3 = 3.75, mirrored at the last
        # pixel's centre (3): 2.25.  Border would clamp to 3.
        assert _sample_ramp(device, "reflection", True) == pytest.approx(2.25)
        assert _sample_ramp(device, "border", True) == pytest.approx(3.0)

    def test_edges(self, device: str) -> None:
        # ((1.5 + 1) * 4 - 1) / 2 = 4.5, mirrored at the outer edge (3.5):
        # 2.5.  Border would clamp to 3.
        assert _sample_ramp(device, "reflection", False) == pytest.approx(2.5)
        assert _sample_ramp(device, "border", False) == pytest.approx(3.0)

    def test_volumetric(self, device: str) -> None:
        # The same ramp as a 5-D input, sampled the same way along x.
        x = lucid.tensor([_RAMP], device=device)
        grid = lucid.tensor([[[[[1.5, 0.0, 0.0]]]]], device=device)
        out = F.grid_sample(x, grid, padding_mode="reflection", align_corners=True)
        assert float(out.item()) == pytest.approx(2.25)


class TestEveryAcceptedSettingRuns:
    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    @pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
    def test_four_d(self, device: str, mode: str, padding_mode: str) -> None:
        x = lucid.randn(1, 2, 4, 5, device=device)
        grid = lucid.rand(1, 3, 3, 2, device=device) * 3.0 - 1.5
        out = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)
        assert tuple(out.shape) == (1, 2, 3, 3)

    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    def test_five_d(self, device: str, mode: str, padding_mode: str) -> None:
        x = lucid.randn(1, 2, 3, 4, 5, device=device)
        grid = lucid.rand(1, 2, 3, 3, 3, device=device) * 3.0 - 1.5
        out = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)
        assert tuple(out.shape) == (1, 2, 2, 3, 3)


class TestNearestGridGradient:
    """Nearest does not move with the grid; its grid gradient is zero, not absent."""

    @pytest.mark.parametrize("rank", [4, 5])
    def test_grid_alone_can_be_differentiated(self, device: str, rank: int) -> None:
        x = lucid.randn(*(1, 2) + (3,) * (rank - 2), device=device)
        grid = lucid.randn(*(1,) + (2,) * (rank - 2) + (rank - 2,), device=device)
        grid.requires_grad = True
        F.grid_sample(x, grid, mode="nearest").sum().backward()
        assert grid.grad is not None
        assert float(grid.grad.abs().sum().item()) == 0.0


class TestRefusedByName:
    def test_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="'cubic'"):
            F.grid_sample(lucid.randn(1, 1, 2, 2), lucid.zeros(1, 1, 1, 2), mode="cubic")

    def test_unknown_padding(self) -> None:
        with pytest.raises(ValueError, match="'circular'"):
            F.grid_sample(
                lucid.randn(1, 1, 2, 2),
                lucid.zeros(1, 1, 1, 2),
                padding_mode="circular",
            )

    def test_bicubic_volumetric(self) -> None:
        with pytest.raises(ValueError, match="bicubic"):
            F.grid_sample(lucid.randn(1, 1, 2, 2, 2), lucid.zeros(1, 1, 1, 1, 3), mode="bicubic")

    def test_rank(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            F.grid_sample(lucid.randn(1, 1, 4), lucid.zeros(1, 1, 2))

    def test_grid_that_does_not_fit_a_volume(self) -> None:
        with pytest.raises(ValueError, match="3 coordinates"):
            F.grid_sample(lucid.randn(1, 1, 2, 2, 2), lucid.zeros(1, 1, 1, 1, 2))

    def test_batch_mismatch(self) -> None:
        with pytest.raises(ValueError, match="batch"):
            F.grid_sample(lucid.randn(1, 1, 2, 2, 2), lucid.zeros(2, 1, 1, 1, 3))

    def test_dtype_mismatch(self) -> None:
        with pytest.raises(TypeError, match="dtype"):
            F.grid_sample(
                lucid.randn(1, 1, 2, 2, 2, dtype=lucid.float64),
                lucid.zeros(1, 1, 1, 1, 3),
            )


class TestAffineGridSizes:
    def test_volumetric_shape(self) -> None:
        theta = lucid.randn(2, 3, 4)
        assert tuple(F.affine_grid(theta, (2, 1, 3, 4, 5)).shape) == (2, 3, 4, 5, 3)

    def test_identity_volume_reaches_every_voxel_centre(self) -> None:
        theta = lucid.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
        x = lucid.randn(1, 2, 3, 4, 5)
        grid = F.affine_grid(theta, (1, 2, 3, 4, 5), align_corners=False)
        out = F.grid_sample(x, grid, align_corners=False)
        assert float((out - x).abs().max().item()) < 1e-5

    def test_wrong_size_length(self) -> None:
        with pytest.raises(ValueError, match="size"):
            F.affine_grid(lucid.randn(1, 2, 3), (1, 1, 2))

    def test_theta_that_does_not_fit_a_volume(self) -> None:
        with pytest.raises(ValueError, match=r"\(1, 3, 4\)"):
            F.affine_grid(lucid.randn(1, 2, 3), (1, 1, 2, 2, 2))
