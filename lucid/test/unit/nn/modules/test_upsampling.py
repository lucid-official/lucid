"""Dedicated unit coverage for ``nn`` upsampling / pixel-shuffle modules.

Forward shape + a deterministic value/round-trip check + backward — these
families had no dedicated test file before.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


class TestUpsample:
    def test_nearest_scale(self) -> None:
        out = nn.Upsample(scale_factor=2)(lucid.ones(2, 3, 8, 8))
        assert out.shape == (2, 3, 16, 16)

    def test_bilinear_scale(self) -> None:
        out = nn.Upsample(scale_factor=2, mode="bilinear")(lucid.ones(2, 3, 8, 8))
        assert out.shape == (2, 3, 16, 16)

    def test_explicit_size(self) -> None:
        out = nn.Upsample(size=(10, 12))(lucid.ones(1, 1, 4, 4))
        assert out.shape == (1, 1, 10, 12)

    def test_nearest_replicates_value(self) -> None:
        # nearest upsample of a constant tensor stays constant
        out = nn.Upsample(scale_factor=2)(lucid.full((1, 1, 2, 2), 7.0)).numpy()
        assert out.shape == (1, 1, 4, 4)
        assert (out == 7.0).all()

    def test_backward_bilinear(self) -> None:
        x = lucid.ones(1, 1, 4, 4, requires_grad=True)
        nn.Upsample(scale_factor=2, mode="bilinear")(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 1, 4, 4)

    def test_backward_nearest_2d(self) -> None:
        # nearest-mode now propagates gradients: each input pixel feeds a 2x2
        # output block under a 2x upsample, so every input gradient is 4.0.
        x = lucid.ones(1, 1, 4, 4, requires_grad=True)
        nn.Upsample(scale_factor=2, mode="nearest")(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 1, 4, 4)
        assert (x.grad == 4.0).all().item()

    def test_backward_nearest_3d(self) -> None:
        # each voxel feeds a 2x2x2 output block under a 2x upsample -> grad 8.0.
        x = lucid.ones(1, 1, 2, 2, 2, requires_grad=True)
        nn.Upsample(scale_factor=2, mode="nearest")(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 1, 2, 2, 2)
        assert (x.grad == 8.0).all().item()

    def test_backward_nearest_asymmetric(self) -> None:
        # non-integer scale: gradients stay strictly positive everywhere.
        x = lucid.ones(1, 2, 3, 3, requires_grad=True)
        nn.Upsample(scale_factor=1.5, mode="nearest")(x).sum().backward()
        assert x.grad is not None
        assert (x.grad > 0).all().item()


class TestPixelShuffleUnshuffle:
    def test_pixel_shuffle_shape(self) -> None:
        # (N, C*r^2, H, W) -> (N, C, H*r, W*r)
        out = nn.PixelShuffle(2)(lucid.ones(1, 8, 4, 4))
        assert out.shape == (1, 2, 8, 8)

    def test_pixel_unshuffle_shape(self) -> None:
        out = nn.PixelUnshuffle(2)(lucid.ones(1, 2, 8, 8))
        assert out.shape == (1, 8, 4, 4)

    def test_shuffle_unshuffle_round_trip(self) -> None:
        x = lucid.randn(2, 4, 6, 6)
        rt = nn.PixelUnshuffle(2)(nn.PixelShuffle(2)(x))
        assert rt.shape == x.shape
        import numpy as np

        np.testing.assert_allclose(rt.numpy(), x.numpy(), atol=1e-6)

    def test_backward(self) -> None:
        x = lucid.ones(1, 8, 4, 4, requires_grad=True)
        nn.PixelShuffle(2)(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 8, 4, 4)


#: Leading dims in front of ``(C, H, W)`` — none, one, several.
_LEADING = [(), (8,), (2, 3), (2, 2, 2)]


@pytest.mark.parity
@pytest.mark.parametrize("lead", _LEADING, ids=str)
@pytest.mark.parametrize("op", ["pixel_shuffle", "pixel_unshuffle"])
def test_pixel_shuffle_any_leading_dims_match_the_reference(
    ref: Any, device: str, lead: tuple[int, ...], op: str
) -> None:
    """``(*, C, H, W)`` — the reference takes any rank of at least 3."""
    shape = (*lead, 8, 3, 4) if op == "pixel_shuffle" else (*lead, 3, 6, 8)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    mine = lucid.tensor(x, device=device, requires_grad=True)
    out = getattr(F, op)(mine, 2)
    theirs = ref.from_numpy(x).requires_grad_()
    want = getattr(ref.nn.functional, op)(theirs, 2)

    assert out.shape == tuple(want.shape)
    np.testing.assert_array_equal(out.cpu().numpy(), want.detach().numpy())
    weights = np.random.default_rng(1).standard_normal(out.shape).astype(np.float32)
    (out * lucid.tensor(weights, device=device)).sum().backward()
    (want * ref.from_numpy(weights)).sum().backward()
    assert mine.grad is not None
    np.testing.assert_array_equal(mine.grad.cpu().numpy(), theirs.grad.numpy())


def test_pixel_shuffle_modules_take_any_leading_dims(device: str) -> None:
    x = lucid.randn(8, 3, 4, device=device)
    assert nn.PixelShuffle(2)(x).shape == (2, 6, 8)
    assert nn.PixelUnshuffle(2)(lucid.randn(3, 6, 8, device=device)).shape == (12, 3, 4)
    assert nn.PixelShuffle(2)(lucid.randn(2, 3, 8, 3, 4, device=device)).shape == (2, 3, 2, 6, 8)
