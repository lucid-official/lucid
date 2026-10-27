"""``linalg.vander`` must build its exponent row on the input's device.

Found 2026-07-26 by probing public API names that appear nowhere under
``lucid/test/``.  ``vander`` hardcoded ``_C_engine.CPU`` for the exponent row
and then combined it with ``x`` via ``pow``, so any Metal input raised
``DeviceMismatch``.  Same family as the ``pdist`` / transforms / crossvit bugs.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg as LA

DEVICES = ["cpu", "metal"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("n_cols", [None, 3, 5])
def test_vander_matches_numpy(device, increasing, n_cols):
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    got = LA.vander(lucid.tensor(values, device=device), N=n_cols, increasing=increasing)
    ref = np.vander(values, n_cols if n_cols is not None else len(values), increasing=increasing)
    assert str(got.device) == f"device('{device}')"
    assert got.shape == ref.shape
    assert np.abs(got.numpy() - ref).max() / max(np.abs(ref).max(), 1.0) < 1e-6


def test_vander_agrees_across_devices():
    values = np.array([0.5, -1.5, 2.0], dtype=np.float32)
    cpu = LA.vander(lucid.tensor(values, device="cpu")).numpy()
    metal = LA.vander(lucid.tensor(values, device="metal")).numpy()
    assert np.abs(cpu - metal).max() / max(np.abs(cpu).max(), 1.0) < 1e-6
