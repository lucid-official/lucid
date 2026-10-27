"""A module that updates a registered buffer as it runs writes into that buffer.

BatchNorm's cumulative average and its batch counter, InstanceNorm's running
statistics and spectral norm's power-iteration vectors were computed in
Python and then rebound: after each forward the attribute named a new tensor
(LCD-497).  Whatever held the buffer kept the old one and never saw an update.
That includes the base of a view installed as the buffer, a ``state_dict()``
taken earlier, and a hook.  The reference writes all of these in place.

Each module runs once to settle, then runs again.  The first run lets a lazy
module materialise its buffers and lets a per-channel observer grow its
scalar seed to one value per channel; the reference re-registers those too.
After the second run every buffer must be the same tensor, a ``state_dict()``
taken between the runs must hold the new values, and at least one buffer
must have changed, so the case really updates something.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.quantization as Q
from lucid.nn.utils import spectral_norm

_C = 3

# Each module that updates buffers in its forward, made on a device, with an
# input it takes.
_UPDATERS: dict[str, tuple[Callable[[str], nn.Module], tuple[int, ...]]] = {
    "batch_norm_1d": (lambda d: nn.BatchNorm1d(_C, device=d), (6, _C, 4)),
    "batch_norm_2d_cumulative": (
        lambda d: nn.BatchNorm2d(_C, momentum=None, device=d),
        (4, _C, 2, 3),
    ),
    "batch_norm_3d": (lambda d: nn.BatchNorm3d(_C, device=d), (2, _C, 2, 2, 3)),
    "lazy_batch_norm_1d_cumulative": (
        lambda d: nn.LazyBatchNorm1d(momentum=None, device=d),
        (6, _C),
    ),
    "instance_norm_1d": (
        lambda d: nn.InstanceNorm1d(_C, track_running_stats=True, device=d),
        (5, _C, 4),
    ),
    "instance_norm_2d_unbatched": (
        lambda d: nn.InstanceNorm2d(_C, track_running_stats=True, device=d),
        (_C, 3, 4),
    ),
    "instance_norm_3d": (
        lambda d: nn.InstanceNorm3d(_C, affine=True, track_running_stats=True, device=d),
        (2, _C, 2, 2, 3),
    ),
    "lazy_instance_norm_2d": (
        lambda d: nn.LazyInstanceNorm2d(track_running_stats=True, device=d),
        (4, _C, 2, 3),
    ),
    "spectral_norm_linear": (
        lambda d: spectral_norm(nn.Linear(4, _C, device=d)),
        (2, 4),
    ),
    "spectral_norm_conv2d": (
        lambda d: spectral_norm(nn.Conv2d(2, _C, 3, device=d), n_power_iterations=3),
        (1, 2, 5, 5),
    ),
    "min_max_observer": (lambda d: Q.MinMaxObserver().to(d), (4, _C)),
    "moving_average_observer": (
        lambda d: Q.MovingAverageMinMaxObserver().to(d),
        (4, _C),
    ),
    "per_channel_observer": (lambda d: Q.PerChannelMinMaxObserver().to(d), (_C, 4)),
    "histogram_observer": (lambda d: Q.HistogramObserver().to(d), (4, _C)),
    "fake_quantize": (lambda d: Q.FakeQuantize().to(d), (4, _C)),
}


def _input(shape: tuple[int, ...], step: int, device: str) -> lucid.Tensor:
    # A wider spread each step, so a min, a max or a statistic always moves.
    rng = np.random.default_rng(step)
    x = rng.standard_normal(shape).astype(np.float32) * (1.0 + 2.0 * step) + step
    return lucid.tensor(x, device=device)


def _host(t: lucid.Tensor) -> np.ndarray:
    return np.array(t.detach().cpu().numpy(), dtype=np.float64)


@pytest.mark.parametrize("name", list(_UPDATERS))
def test_a_buffer_a_module_updates_is_the_same_tensor_after(name: str, device: str) -> None:
    make, shape = _UPDATERS[name]
    m = make(device)
    m.train()
    m(_input(shape, 0, device))
    before = dict(m.named_buffers())
    values = {k: _host(b) for k, b in before.items()}
    saved = m.state_dict()
    m(_input(shape, 1, device))
    after = dict(m.named_buffers())
    assert after.keys() == before.keys()
    for k, b in before.items():
        assert after[k] is b, f"{k} was rebound to a new tensor"
        if k in saved:
            np.testing.assert_array_equal(_host(saved[k]), _host(b), err_msg=k)
    changed = [k for k, b in after.items() if not np.array_equal(_host(b), values[k])]
    assert changed, "the step updated no buffer, so the case checks nothing"
