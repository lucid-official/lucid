"""``nn.TimestepEmbedding`` hands its MLP the layer's own dtype (LCD-562).

The sinusoid was built in float32 whatever the layer was, so a float16
DDPM, NCSN or Score-SDE ran its whole conditioning path widened to float32
and a float64 one fed float32-rounded angles to float64 weights.  The
reference U-Nets compute the sinusoid in float32 and cast it to the
network's dtype before the time MLP; a float64 layer gets float64 angles.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import metal_available

_STEPS = [0, 1, 250, 777, 999]

_NARROW = [
    ("cpu", lucid.float16),
    ("cpu", lucid.bfloat16),
    ("metal", lucid.float16),
    ("metal", lucid.bfloat16),
]
_CASES = [*_NARROW, ("cpu", lucid.float64)]


def _layer(device: str, dtype: lucid.dtype, flip: bool = True) -> nn.TimestepEmbedding:
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")
    lucid.manual_seed(0)
    return nn.TimestepEmbedding(16, 32, flip_sin_to_cos=flip).to(device).to(dtype)


def _float64_sinusoid(steps: list[int], in_dim: int, flip: bool) -> np.ndarray:
    half = in_dim // 2
    inv = np.array([1.0 / (10_000.0 ** (i / (half - 1))) for i in range(half)])
    ang = np.array(steps, dtype=np.float64)[:, None] * inv[None, :]
    parts = [np.cos(ang), np.sin(ang)] if flip else [np.sin(ang), np.cos(ang)]
    return np.concatenate(parts, axis=-1)


@pytest.mark.parametrize(("device", "dtype"), _CASES)
def test_the_embedding_and_the_output_take_the_layer_dtype(device: str, dtype: lucid.dtype) -> None:
    layer = _layer(device, dtype)
    steps = lucid.tensor(_STEPS, device=device)
    assert layer._sinusoidal(steps).dtype == dtype
    assert layer(steps).dtype == dtype


@pytest.mark.parametrize(("device", "dtype"), _NARROW)
def test_a_narrow_layer_gets_the_float32_sinusoid_rounded_once(
    device: str, dtype: lucid.dtype
) -> None:
    """Angles in float32, then one cast: the reference's convention."""
    narrow = _layer(device, dtype)
    wide = _layer(device, lucid.float32)
    steps = lucid.tensor(_STEPS, device=device)
    got = narrow._sinusoidal(steps).to(lucid.float32).numpy()
    want = wide._sinusoidal(steps).to(dtype).to(lucid.float32).numpy()
    assert np.array_equal(got, want)


@pytest.mark.parametrize("flip", [True, False])
def test_a_float64_layer_computes_float64_angles(flip: bool) -> None:
    layer = _layer("cpu", lucid.float64, flip)
    got = layer._sinusoidal(lucid.tensor(_STEPS)).numpy()
    want = _float64_sinusoid(_STEPS, 16, flip)
    # float32 angles near 999 rad are off by ~3e-5; float64 ones are not.
    assert np.abs(got - want).max() < 1e-12


def test_the_float32_path_is_unchanged() -> None:
    """The arithmetic a float32 layer always did, bit for bit."""
    layer = _layer("cpu", lucid.float32)
    steps = lucid.tensor(_STEPS)
    half = 8
    inv = lucid.tensor([1.0 / (10_000.0 ** (i / (half - 1))) for i in range(half)])
    ang = steps.float().unsqueeze(-1) * inv.unsqueeze(0)
    want = lucid.cat([lucid.cos(ang), lucid.sin(ang)], dim=-1)
    got = layer._sinusoidal(steps)
    assert got.dtype == lucid.float32
    assert np.array_equal(got.numpy(), want.numpy())
    assert math.isclose(float(got[1, 0].item()), math.cos(1.0), rel_tol=1e-6)
