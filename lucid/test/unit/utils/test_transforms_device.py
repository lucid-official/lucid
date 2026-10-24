"""Every image transform must run on the input's device and stay there.

Found 2026-07-26: **34 of the image transforms raised ``DeviceMismatch`` on a
Metal tensor** while all 34 worked on the CPU.  The cause was systemic rather
than per-transform — 90 tensor-factory calls across
``lucid/utils/transforms/**.py`` passed no ``device=``, so every derived
helper (noise, blur kernel, dropout mask, warp matrix, sampling grid, LUT) was
born on the CPU and then combined with GPU image data.  Same family as the
``pdist`` bug, at 90 sites instead of one.

The fix threads the input's device through, so helpers are *created* on the
target device rather than transferred to it — no round trip.

Two properties are pinned here:

1. Every transform runs on Metal and returns a Metal tensor.
2. Deterministic transforms are numerically equivalent across devices.
   RNG-driven ones (GaussNoise / ISONoise / PixelDropout / ElasticTransform)
   are compared by statistics instead.  There is one stream, but the two
   devices read it differently — a Metal ``rand(4)`` takes one Philox block
   of four numbers, a CPU one four blocks — so one seed gives each device
   different samples.
"""

import inspect

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T

DEVICES = ["cpu", "metal"]

# Not image-to-image transforms — config/dataclass/base types.
_NOT_A_TRANSFORM = {
    "AutoTransformsPreset",
    "BboxParams",
    "BoundingBoxes",
    "Compose",
    "Detection",
    "GeometricTransform",
    "Image",
    "ImageClassification",
    "ImageClassificationAugment",
    "Interpolation",
}

# Driven by the RNG: identical seeds give different samples per device, because
# the CPU and Metal kernels map the one stream to samples differently.
_STOCHASTIC = {"GaussNoise", "ISONoise", "PixelDropout", "ElasticTransform"}


def _all_transforms():
    out = []
    for name in sorted(dir(T)):
        if name.startswith("_") or not name[0].isupper() or name in _NOT_A_TRANSFORM:
            continue
        cls = getattr(T, name)
        if not inspect.isclass(cls):
            continue
        try:
            params = inspect.signature(cls).parameters
            kwargs = {"p": 1.0} if "p" in params else {}
            cls(**kwargs)
        except TypeError, ValueError:
            continue
        out.append(name)
    return out


TRANSFORMS = _all_transforms()


def _make(name):
    cls = getattr(T, name)
    kwargs = {"p": 1.0} if "p" in inspect.signature(cls).parameters else {}
    return cls(**kwargs)


def _image(device, size=32, seed=0):
    rng = np.random.default_rng(seed)
    return lucid.tensor(rng.random((3, size, size)).astype(np.float32), device=device)


def test_transform_list_is_not_empty():
    """Guards the discovery helper — a silent empty list would pass everything."""
    assert len(TRANSFORMS) > 25, TRANSFORMS


@pytest.mark.parametrize("name", TRANSFORMS)
def test_transform_runs_on_metal_and_stays_there(name):
    """The whole point: no DeviceMismatch, and no silent hop back to the CPU."""
    lucid.manual_seed(0)
    out = _make(name)(_image("metal"))
    assert hasattr(out, "device"), f"{name} did not return a Tensor"
    assert str(out.device) == "device('metal')", f"{name} left the input device"


@pytest.mark.parametrize("name", TRANSFORMS)
def test_transform_matches_across_devices(name):
    outs = {}
    for device in DEVICES:
        lucid.manual_seed(1234)
        outs[device] = _make(name)(_image(device)).numpy()
    cpu, metal = outs["cpu"], outs["metal"]
    assert cpu.shape == metal.shape, name
    if name in _STOCHASTIC:
        # One stream, read differently per device — compare distributions,
        # not samples.
        assert abs(float(cpu.mean()) - float(metal.mean())) < 5e-2, name
        assert abs(float(cpu.std()) - float(metal.std())) < 5e-2, name
    else:
        assert np.abs(cpu - metal).max() < 1e-4, name


@pytest.mark.parametrize("device", DEVICES)
def test_composed_pipeline_stays_on_device(device):
    pipeline = T.Compose(
        [
            T.HorizontalFlip(p=1.0),
            T.RandomCrop(28, 28),
            T.ColorJitter(p=1.0),
            T.GaussNoise(p=1.0),
            T.CoarseDropout(p=1.0),
            T.Rotate(p=1.0),
            T.Sharpen(p=1.0),
            T.Equalize(p=1.0),
        ]
    )
    lucid.manual_seed(0)
    out = pipeline(_image(device))
    assert str(out.device) == f"device('{device}')"
    assert not np.isnan(out.numpy()).any()


# ── data-dependent ops must also ride the input device ───────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_bincount_rides_the_input_device(device):
    """``bincount`` counts on the host but must return on the input's device.

    ``Equalize`` / ``CLAHE`` build their LUT from it, so a CPU return
    device-mismatched the rest of a GPU pipeline.
    """
    values = np.array([0, 1, 1, 2, 3, 3, 3], dtype=np.int64)
    got = lucid.bincount(lucid.tensor(values, device=device), minlength=8)
    assert str(got.device) == f"device('{device}')"
    assert np.array_equal(got.numpy(), np.bincount(values, minlength=8))


@pytest.mark.parametrize("device", DEVICES)
def test_histogram_rides_the_input_device(device):
    rng = np.random.default_rng(0)
    values = rng.random(64).astype(np.float32)
    hist, edges = lucid.histogram(lucid.tensor(values, device=device), bins=4)
    assert str(hist.device) == f"device('{device}')"
    assert str(edges.device) == f"device('{device}')"
    # MLX-Metal has no float64, so the GPU result is emitted at float32.
    if device == "metal":
        assert edges.dtype == lucid.float32
    ref, _ = np.histogram(values, bins=4)
    assert np.array_equal(hist.numpy(), ref)
