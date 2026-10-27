"""Two cpu-vs-metal divergences found by sweeping the Tensor method surface.

Found 2026-07-26 by calling every zero/one-argument ``Tensor`` method on both
devices and comparing.  Out of ~300 methods, two disagreed for a real reason:

1. ``erfinv`` outside its domain.  ``erf`` maps onto ``(-1, 1)``, so only the
   two endpoints are infinite and anything beyond is undefined.  The CPU kernel
   collapsed ``x >= 1`` to ``+inf``, swallowing the entire out-of-domain half;
   Metal already matched SciPy.  An ``inf`` is the more dangerous answer: it
   quietly becomes finite downstream (``1/inf``, ``exp(-inf)``), so a domain
   error could disappear instead of propagating as NaN.
2. ``gather`` with a float index tensor.  The CPU kernel rejected it with
   ``NotImplementedError`` while the GPU one silently reinterpreted the floats
   and returned plausible-looking garbage.  The dtype check now lives at the op
   layer so both devices behave identically.
"""

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]


# ── erfinv domain ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_erfinv_in_domain_matches_reference(device):
    values = np.array([-0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99], dtype=np.float32)
    got = lucid.tensor(values, device=device).erfinv().numpy()
    # erf(erfinv(x)) == x is the definition; check the round trip instead of
    # depending on SciPy being installed.
    back = lucid.tensor(got, device=device).erf().numpy()
    assert np.abs(back - values).max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [1.5, 2.0, 10.0, -1.5, -2.0, -10.0])
def test_erfinv_outside_domain_is_nan(device, bad):
    got = lucid.tensor(np.array([bad], dtype=np.float32), device=device).erfinv()
    assert np.isnan(got.numpy()).all(), f"{device}: erfinv({bad}) must be NaN"


@pytest.mark.parametrize("device", DEVICES)
def test_erfinv_endpoints_are_infinite(device):
    got = lucid.tensor(np.array([1.0, -1.0], dtype=np.float32), device=device).erfinv()
    arr = got.numpy()
    assert np.isposinf(arr[0])
    assert np.isneginf(arr[1])


@pytest.mark.parametrize("device", DEVICES)
def test_erfinv_propagates_nan(device):
    got = lucid.tensor(np.array([np.nan], dtype=np.float32), device=device).erfinv()
    assert np.isnan(got.numpy()).all()


def test_erfinv_agrees_across_devices():
    values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    cpu = lucid.tensor(values, device="cpu").erfinv().numpy()
    metal = lucid.tensor(values, device="metal").erfinv().numpy()
    assert np.array_equal(np.isnan(cpu), np.isnan(metal))
    assert np.array_equal(np.isinf(cpu), np.isinf(metal))
    finite = np.isfinite(cpu)
    assert np.abs(cpu[finite] - metal[finite]).max() < 1e-5


# ── gather index dtype ───────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_gather_with_integer_indices(device):
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    idx = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]] * 2, dtype=np.int64)
    got = lucid.tensor(values, device=device).gather(1, lucid.tensor(idx, device=device))
    assert np.abs(got.numpy() - np.take_along_axis(values, idx, axis=1)).max() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_gather_rejects_float_indices(device):
    """Metal used to accept these and return plausible-looking garbage."""
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    tensor = lucid.tensor(values, device=device)
    with pytest.raises(Exception, match="integer"):
        tensor.gather(1, lucid.tensor(values, device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [lucid.int32, lucid.int64])
def test_gather_accepts_every_integer_width(device, dtype):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    idx = lucid.tensor(np.zeros((3, 4), dtype=np.int64), device=device).to(dtype)
    got = lucid.tensor(values, device=device).gather(1, idx)
    assert got.shape == (3, 4)
