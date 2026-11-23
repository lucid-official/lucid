"""An integer label mask survives every transform that resamples it (LCD-1026).

``interpolate`` and ``grid_sample`` refuse an integer tensor, as the reference's CPU does (it has
no kernel for one), so the transforms that resample a ``T.Mask`` sample an integer label mask as
float32 and bring it back (``functional._sampled``): rounded, and in the dtype it came in.  A
mask is always sampled nearest, which copies values, so the answer is the nearest sample of the
labels -- the one the float mask gets -- and its dtype stays what the caller gave.  float32 holds
every integer below ``2**24``, so the labels come back unchanged up to there.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms.functional import _sampled

_H, _W = 24, 32

# p=1 everywhere, so the transform acts; the same seed gives the float and the integer mask the
# same random parameters.  The bool twin of this file is ``test_bool_mask``.
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

_INTEGERS = ("int8", "int16", "int32", "int64")


def _labels(device: str, dtype: str) -> lucid.Tensor:
    """A ``(1, H, W)`` label mask of classes 0..5 in blocks a nearest sample can get wrong."""
    rows = lucid.arange(_H, device=device).reshape(_H, 1)
    cols = lucid.arange(_W, device=device).reshape(1, _W)
    classes = (rows // 4 + cols // 5) % 6
    return classes.to(getattr(lucid, dtype)).reshape(1, _H, _W)


@pytest.mark.parametrize("dtype", _INTEGERS)
@pytest.mark.parametrize("name", list(_TRANSFORMS))
def test_an_integer_mask_keeps_its_dtype_and_is_the_nearest_sample(
    name: str, dtype: str, device: str
) -> None:
    mask = _labels(device, dtype)
    transform = _TRANSFORMS[name]()
    image = T.Image(lucid.rand(3, _H, _W, device=device))

    lucid.manual_seed(7)
    got = transform({"image": image, "mask": T.Mask(mask)})["mask"]
    lucid.manual_seed(7)
    want = transform({"image": image, "mask": T.Mask(mask.to(lucid.float32))})["mask"]

    assert isinstance(got, T.Mask) and isinstance(want, T.Mask)
    assert got.data.dtype == getattr(lucid, dtype), (name, got.data.dtype)
    assert got.data.shape == want.data.shape
    assert got.data.tolist() == want.data.round().to(getattr(lucid, dtype)).tolist()


@pytest.mark.parametrize("name", list(_TRANSFORMS))
def test_an_integer_mask_is_resampled_and_keeps_its_classes(name: str, device: str) -> None:
    # The transform moved something, and no class was invented on the way: a nearest sample only
    # copies the labels it was given.
    mask = _labels(device, "int64")
    lucid.manual_seed(7)
    got = _TRANSFORMS[name]()(T.Mask(mask)).data
    assert set(got.unique().tolist()) <= set(mask.unique().tolist()) | {0}
    assert len(set(got.unique().tolist())) > 1


@pytest.mark.parametrize("dtype", _INTEGERS)
def test_a_label_below_two_to_the_24_comes_back_unchanged(dtype: str, device: str) -> None:
    label = 2**24 - 1 if dtype in ("int32", "int64") else 100
    mask = lucid.full((1, 8, 8), label, dtype=getattr(lucid, dtype), device=device)
    got = T.Resize(4, 4)(T.Mask(mask)).data
    assert got.dtype == mask.dtype
    assert got.unique().tolist() == [label]


def test_a_float_or_bool_mask_is_sampled_as_it_was(device: str) -> None:
    # What ``_sampled`` did before it knew integers: a float is passed through, a bool comes back
    # bool.
    calls: list[lucid.dtype] = []

    def sample(t: lucid.Tensor) -> lucid.Tensor:
        calls.append(t.dtype)
        return t

    floats = lucid.rand(1, 4, 4, device=device)
    assert _sampled(floats, sample) is floats
    flags = lucid.tensor([[True, False]], device=device)
    assert _sampled(flags, sample).dtype == lucid.bool_
    assert calls == [lucid.float32, lucid.float32]
