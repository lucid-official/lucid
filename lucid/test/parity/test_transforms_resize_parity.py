"""Parity: ``lucid.utils.transforms.functional.resize`` vs the reference.

Shrinking antialiases on both sides by default: the same separable filter
(a triangle for bilinear, the a = -0.5 cubic for bicubic) stretched by the
downscale factor.  Covers exact and fractional factors and a resize that
shrinks one side while growing the other.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.utils.transforms.functional as TF
from lucid.test._fixtures.ref_framework import require_ref_vision

_SIZES = [(256, 341), (240, 320), (100, 100), (400, 533), (256, 800)]


@pytest.mark.parity
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("size", _SIZES, ids=[f"{h}x{w}" for h, w in _SIZES])
def test_resize_matches_the_reference(ref: Any, mode: str, size: tuple[int, int]) -> None:
    x = np.random.default_rng(0).random((1, 3, 480, 640), dtype=np.float32)
    got = TF.resize(lucid.from_numpy(x.copy()), size, interpolation=mode).numpy()
    want = ref.nn.functional.interpolate(
        ref.from_numpy(x.copy()),
        size=size,
        mode=mode,
        align_corners=False,
        antialias=True,
    ).numpy()
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-4)


_INT_DTYPES = {"int16": (lucid.int16, np.int16), "int32": (lucid.int32, np.int32)}


@pytest.mark.parity
@pytest.mark.parametrize("dtype", sorted(_INT_DTYPES))
@pytest.mark.parametrize("mode", ["bilinear", "bicubic", "nearest"])
@pytest.mark.parametrize("size", [(20, 25), (80, 100)], ids=["down", "up"])
@pytest.mark.parametrize("antialias", [True, False], ids=["aa", "no-aa"])
def test_an_integer_image_resizes_like_the_reference(
    ref: Any,
    device: str,
    dtype: str,
    mode: str,
    size: tuple[int, int],
    antialias: bool,
) -> None:
    # Filtered in float32, then rounded and cast back to the image's dtype.
    lucid_dtype, np_dtype = _INT_DTYPES[dtype]
    x = (np.random.default_rng(1).random((3, 40, 50)) * 255).astype(np_dtype)
    got = TF.resize(
        lucid.tensor(x, dtype=lucid_dtype, device=device),
        size,
        interpolation=mode,
        antialias=antialias,
    )
    rtf = require_ref_vision().transforms.functional
    want = rtf.resize(
        ref.from_numpy(x.copy()),
        list(size),
        interpolation=getattr(rtf.InterpolationMode, mode.upper()),
        antialias=antialias,
    ).numpy()
    assert got.dtype == lucid_dtype
    got_np = got.to("cpu").numpy().astype(np.int64)
    differ = got_np != want
    if differ.any():
        # Only a value the float filter puts on a rounding tie may land on
        # the other side of it, and only by one.
        unrounded = rtf.resize(
            ref.from_numpy(x.astype(np.float32)),
            list(size),
            interpolation=getattr(rtf.InterpolationMode, mode.upper()),
            antialias=antialias,
        ).numpy()
        assert np.abs(got_np - want).max() == 1
        tie = np.abs(np.abs(unrounded - np.floor(unrounded)) - 0.5) < 1e-3
        assert tie[differ].all()
