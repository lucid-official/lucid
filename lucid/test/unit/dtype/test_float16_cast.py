"""Casting to/from float16 must not crash the process.

Found 2026-07-26.  ``CpuBackend::astype`` routed any cast touching F16 through
a "bridge" that delegated both legs back into ``astype`` itself, with a comment
claiming the recursion would land in the main cast table.  It never did: every
recursive call still had F16 on one side, so it re-entered the same branch and
the process died of stack exhaustion.

CPU float16 casts therefore **SIGSEGV'd in both directions and had never
worked** — ``.half()``, ``.to(lucid.float16)``, and the way back.  A probe that
merely printed the result looked like it "produced no output"; only the exit
code (139) revealed it.

Metal was unaffected (MLX converts natively).
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import device_dtype_params

DEVICES = ["cpu", "metal"]

# Values chosen to hit the interesting binary16 paths: zero, signed zero,
# normals, the largest finite half, a subnormal-in-half value, and infinities.
_VALUES = np.array(
    [0.0, -0.0, 1.0, -2.5, 65504.0, 1e-8, 3.14159, np.inf, -np.inf], dtype=np.float32
)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_to_float16_matches_numpy(device):
    got = lucid.tensor(_VALUES, device=device).to(lucid.float16)
    assert got.dtype == lucid.float16
    back = got.to(lucid.float32).numpy()
    ref = _VALUES.astype(np.float16).astype(np.float32)
    finite = np.isfinite(ref)
    assert np.abs(back[finite] - ref[finite]).max() == 0.0
    assert np.array_equal(np.isinf(back), np.isinf(ref))
    assert np.array_equal(np.signbit(back), np.signbit(ref))


@pytest.mark.parametrize("device", DEVICES)
def test_float16_to_float32_direction(device):
    half = lucid.tensor(_VALUES.astype(np.float16), device=device, dtype=lucid.float16)
    got = half.to(lucid.float32).numpy()
    ref = _VALUES.astype(np.float16).astype(np.float32)
    finite = np.isfinite(ref)
    assert np.abs(got[finite] - ref[finite]).max() == 0.0


@pytest.mark.parametrize(
    ("device", "other"),
    # Metal holds no float64, so that pair is not generated; its refusal
    # is asserted in lucid/test/unit/device/test_metal_dtype_support.py.
    device_dtype_params([lucid.float64, lucid.int32, lucid.int64], devices=DEVICES),
)
def test_float16_round_trips_through_other_dtypes(device, other):
    """The bridge recurses on the F16-free leg; make sure that leg is real."""
    values = np.array([0.0, 1.0, -3.0, 7.0], dtype=np.float32)
    half = lucid.tensor(values, device=device).to(lucid.float16)
    converted = half.to(other)
    assert converted.dtype == other
    back = converted.to(lucid.float32).numpy()
    assert np.abs(back - values).max() < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_module_half_cast_does_not_crash(device):
    layer = nn.Linear(4, 3).to(device).to(lucid.float16)
    assert layer.weight.dtype == lucid.float16
    assert layer.bias.dtype == lucid.float16
    # And back again.
    layer = layer.to(lucid.float32)
    assert layer.weight.dtype == lucid.float32


@pytest.mark.parametrize("device", DEVICES)
def test_large_buffer_cast(device):
    """Exercises the loop rather than a handful of elements."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal(4096).astype(np.float32)
    back = lucid.tensor(values, device=device).to(lucid.float16).to(lucid.float32)
    ref = values.astype(np.float16).astype(np.float32)
    assert np.abs(back.numpy() - ref).max() == 0.0
