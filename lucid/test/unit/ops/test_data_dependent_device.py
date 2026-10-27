"""Data-dependent-output ops must return on the input's device.

``nonzero`` / ``unique`` / ``argwhere`` / ``bincount`` / ``histogram`` all have
an output *size* that depends on the values, so the count has to happen on the
host — that host round-trip is the sanctioned carve-out and is not what this
file is about.  What it pins is the **return** device: these ops used to hand
back a CPU tensor for a GPU input (``Sort.cpp`` even documented it as "always
produces output on Device::CPU regardless of the input device"), so every
downstream op device-mismatched and callers had to insert a manual ``.to()`` —
the very transfer the round-trip had already paid for.

Fixed 2026-07-26: the computation still runs on the host, the result rides the
input's device.
"""

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]


def _expected(device):
    return f"device('{device}')"


@pytest.mark.parametrize("device", DEVICES)
def test_nonzero_rides_the_input_device(device):
    values = (np.random.default_rng(0).standard_normal((4, 5)) > 0.3).astype(np.float32)
    got = lucid.nonzero(lucid.tensor(values, device=device))
    assert str(got.device) == _expected(device)
    assert np.array_equal(got.numpy(), np.argwhere(values != 0))


@pytest.mark.parametrize("device", DEVICES)
def test_argwhere_rides_the_input_device(device):
    values = (np.random.default_rng(1).standard_normal((3, 6)) > 0.0).astype(np.float32)
    got = lucid.argwhere(lucid.tensor(values, device=device))
    assert str(got.device) == _expected(device)
    assert np.array_equal(got.numpy(), np.argwhere(values != 0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_unique_rides_the_input_device(device, dtype):
    values = np.array([3, 1, 2, 1, 3, 0, 2], dtype=dtype)
    got = lucid.unique(lucid.tensor(values, device=device))
    assert str(got.device) == _expected(device)
    assert np.array_equal(got.numpy(), np.unique(values))


@pytest.mark.parametrize("device", DEVICES)
def test_empty_result_still_rides_the_device(device):
    """Zero non-zeros is the degenerate allocation path."""
    zeros = np.zeros((3, 4), dtype=np.float32)
    got = lucid.nonzero(lucid.tensor(zeros, device=device))
    assert str(got.device) == _expected(device)
    assert got.shape == (0, 2)


@pytest.mark.parametrize("device", DEVICES)
def test_results_are_identical_across_devices(device):
    values = (np.random.default_rng(2).standard_normal((5, 5)) > 0.1).astype(np.float32)
    ints = np.random.default_rng(3).integers(0, 7, (24,)).astype(np.int64)
    ref_nz = lucid.nonzero(lucid.tensor(values, device="cpu")).numpy()
    ref_u = lucid.unique(lucid.tensor(ints, device="cpu")).numpy()
    assert np.array_equal(lucid.nonzero(lucid.tensor(values, device=device)).numpy(), ref_nz)
    assert np.array_equal(lucid.unique(lucid.tensor(ints, device=device)).numpy(), ref_u)


@pytest.mark.parametrize("device", DEVICES)
def test_downstream_ops_chain_without_a_transfer(device):
    """The point of the fix: the result feeds the next op directly."""
    values = (np.random.default_rng(4).standard_normal((6,)) > 0.0).astype(np.float32)
    src = lucid.tensor(np.arange(6, dtype=np.float32), device=device)
    idx = lucid.nonzero(lucid.tensor(values, device=device))
    picked = src[idx[:, 0]]
    assert str(picked.device) == _expected(device)

    ints = np.array([5, 2, 5, 1], dtype=np.int64)
    shifted = lucid.unique(lucid.tensor(ints, device=device)) + 1
    assert str(shifted.device) == _expected(device)
    assert np.array_equal(shifted.numpy(), np.unique(ints) + 1)


@pytest.mark.parametrize("device", DEVICES)
def test_bincount_and_histogram_ride_the_device(device):
    ints = np.array([0, 1, 1, 2, 3, 3, 3], dtype=np.int64)
    counts = lucid.bincount(lucid.tensor(ints, device=device), minlength=8)
    assert str(counts.device) == _expected(device)
    assert np.array_equal(counts.numpy(), np.bincount(ints, minlength=8))

    floats = np.random.default_rng(5).random(64).astype(np.float32)
    hist, edges = lucid.histogram(lucid.tensor(floats, device=device), bins=4)
    assert str(hist.device) == _expected(device)
    assert str(edges.device) == _expected(device)
    assert np.array_equal(hist.numpy(), np.histogram(floats, bins=4)[0])
