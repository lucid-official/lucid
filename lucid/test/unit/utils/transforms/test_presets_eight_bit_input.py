"""An evaluation preset handed 8-bit pixels reproduces PIL's resize.

The published checkpoints were evaluated on PIL images, whose resize runs
in 8 bits — horizontally, then vertically, each pass rounded and clipped to
0..255.  The presets resized in float, keeping fractions PIL rounds away,
and that alone moved ``convnext_xlarge`` from 100% agreement with its
source (same input) to 96.4% on ImageNet-V2 (CHA-13).  Integer input — the
decoded pixels, as ``lucid.tensor(np.asarray(image))`` gives them — now
takes PIL's two passes.  Float input is unchanged.
"""

import numpy as np
import pytest

import lucid
from lucid.utils.transforms import ImageClassification

Image = pytest.importorskip("PIL.Image")


def _photo(w: int, h: int) -> np.ndarray:
    rng = np.random.default_rng(w * h)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            128 + 100 * np.sin(xx / 3.0),
            128 + 100 * np.cos(yy / 2.5),
            128 + 60 * np.sin((xx + yy) / 4.0),
        ],
        -1,
    )
    return np.clip(base + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("size", [(500, 375), (375, 500), (333, 499)])
@pytest.mark.parametrize("interpolation", ["bicubic", "bilinear"])
def test_integer_pixels_resize_as_pil_does(size: tuple[int, int], interpolation: str) -> None:
    w, h = size
    arr = _photo(w, h)
    preset = ImageClassification(crop_size=224, resize_size=256, interpolation=interpolation)
    got = np.asarray(preset(lucid.tensor(arr).permute(2, 0, 1)).numpy())
    # PIL's own resize, then the same crop and normalisation by hand.
    nh, nw = (256, int(256 * w / h)) if h <= w else (int(256 * h / w), 256)
    mode = Image.BICUBIC if interpolation == "bicubic" else Image.BILINEAR
    resized = np.asarray(Image.fromarray(arr).resize((nw, nh), mode), dtype=np.float32) / 255
    top, left = int(round((nh - 224) / 2)), int(round((nw - 224) / 2))
    crop = resized[top : top + 224, left : left + 224].transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    want = (crop - mean) / std
    one_level = 1 / 255 / 0.224
    assert np.abs(got - want).max() <= one_level * 1.01
    # PIL rounds in fixed point: a rare value lands one level the other way.
    assert (np.abs(got - want) > 1e-5).mean() < 5e-3


def test_float_input_is_unchanged() -> None:
    arr = _photo(400, 300).astype(np.float32) / 255
    preset = ImageClassification(crop_size=224, resize_size=256, interpolation="bicubic")
    out = np.asarray(preset(lucid.tensor(arr).permute(2, 0, 1)).numpy())
    assert out.dtype == np.float32 and out.shape == (3, 224, 224)
    # No 8-bit rounding: values fall between the 1/255 steps.
    raw = (
        out * np.array([0.229, 0.224, 0.225])[:, None, None]
        + np.array([0.485, 0.456, 0.406])[:, None, None]
    )
    assert np.abs(raw * 255 - np.round(raw * 255)).max() > 0.1
