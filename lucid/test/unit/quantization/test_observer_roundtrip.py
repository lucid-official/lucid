"""An observer's calibration has to survive a checkpoint.

Found by the audit's module axis: a per-channel observer is seeded with
scalar ``+inf`` / ``-inf`` because the channel count is unknown at
construction, and grows to ``(C,)`` on the first batch.  A fresh observer
is therefore scalar, so loading a calibrated ``state_dict`` into one
failed on a size mismatch — a quantized model's calibration could be
saved and never restored.
"""

import numpy as np
import pytest

import lucid
from lucid.quantization import (
    MinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)

_OBSERVERS = [MinMaxObserver, PerChannelMinMaxObserver, MovingAveragePerChannelMinMaxObserver]


@pytest.mark.parametrize("cls", _OBSERVERS)
def test_calibration_survives_a_state_dict_round_trip(cls) -> None:
    rng = np.random.default_rng(0)
    calibrated = cls()
    calibrated(lucid.tensor(rng.random((2, 3)).astype(np.float32)))
    saved = calibrated.state_dict()

    fresh = cls()
    fresh.load_state_dict(saved)

    for key, value in saved.items():
        assert np.allclose(
            np.asarray(value.numpy()), np.asarray(fresh.state_dict()[key].numpy())
        ), key


@pytest.mark.parametrize("cls", _OBSERVERS)
def test_reloaded_observer_gives_the_same_qparams(cls) -> None:
    """Guard the instrument: the buffers must be the calibration, not its shape."""
    rng = np.random.default_rng(1)
    calibrated = cls()
    calibrated(lucid.tensor(rng.random((4, 3)).astype(np.float32)))
    fresh = cls()
    fresh.load_state_dict(calibrated.state_dict())

    want_scale, want_zp = calibrated.calculate_qparams()
    got_scale, got_zp = fresh.calculate_qparams()
    assert np.allclose(np.asarray(want_scale.numpy()), np.asarray(got_scale.numpy()))
    assert np.allclose(np.asarray(want_zp.numpy()), np.asarray(got_zp.numpy()))
