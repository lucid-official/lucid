"""A bool segmentation mask survives every transform that resamples it (LCD-1020).

``interpolate`` and ``grid_sample`` refuse a bool tensor, as the reference's CPU
does, so the transforms that resample a ``T.Mask`` cast a bool mask to float32 for
the sample and bring it back (``functional._sampled``).  A mask is always sampled
nearest, which copies values, so the answer is the nearest sample of the mask --
the same one the float mask gets -- and its dtype stays bool.  Before, the
resizes returned bool, and the warps and displacements leaked a float32 mask.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.utils.transforms as T

_H, _W = 24, 32

# p=1 everywhere, so the transform acts; the same seed gives the float and the bool
# mask the same random parameters.
_TRANSFORMS: dict[str, Callable[[], T.Transform]] = {
    "Resize": lambda: T.Resize(16, 20),
    "RandomResizedCrop": lambda: T.RandomResizedCrop(16, 20, p=1.0),
    "RandomScale": lambda: T.RandomScale(scale_limit=0.4, p=1.0),
    "Rotate": lambda: T.Rotate(limit=30, p=1.0),
    "SafeRotate": lambda: T.SafeRotate(limit=30, p=1.0),
    "Perspective": lambda: T.Perspective(scale=(0.05, 0.1), p=1.0),
    "ElasticTransform": lambda: T.ElasticTransform(alpha=20.0, sigma=8.0, p=1.0),
    "GridDistortion": lambda: T.GridDistortion(p=1.0),
    "OpticalDistortion": lambda: T.OpticalDistortion(distort_limit=0.1, p=1.0),
    "RandomGridShuffle": lambda: T.RandomGridShuffle(grid=(3, 3), p=1.0),
}


def _blobs(device: str) -> lucid.Tensor:
    """A ``(1, H, W)`` bool mask with structure a nearest sample can get wrong."""
    rows = lucid.arange(_H, device=device).reshape(_H, 1)
    cols = lucid.arange(_W, device=device).reshape(1, _W)
    mask = ((rows // 4 + cols // 5) % 2 == 0) | ((rows - 12) ** 2 + (cols - 16) ** 2 < 30)
    return mask.reshape(1, _H, _W)


@pytest.mark.parametrize("name", list(_TRANSFORMS))
def test_a_bool_mask_stays_bool_and_is_the_nearest_sample(name: str, device: str) -> None:
    mask = _blobs(device)
    transform = _TRANSFORMS[name]()
    image = T.Image(lucid.rand(3, _H, _W, device=device))

    lucid.manual_seed(7)
    got = transform({"image": image, "mask": T.Mask(mask)})["mask"]
    lucid.manual_seed(7)
    want = transform({"image": image, "mask": T.Mask(mask.to(lucid.float32))})["mask"]

    assert isinstance(got, T.Mask) and isinstance(want, T.Mask)
    assert got.data.dtype == lucid.bool_, (name, got.data.dtype)
    assert got.data.shape == want.data.shape
    assert got.data.tolist() == (want.data > 0.5).tolist()


@pytest.mark.parametrize("name", list(_TRANSFORMS))
def test_a_bool_mask_is_resampled_not_dropped(name: str, device: str) -> None:
    # The transform moved something: a mask that came back untouched (or empty) would pass the
    # comparison above against a float mask that did the same.
    mask = _blobs(device)
    lucid.manual_seed(7)
    got = _TRANSFORMS[name]()(T.Mask(mask)).data
    assert bool(got.any()) and not bool(got.all())
