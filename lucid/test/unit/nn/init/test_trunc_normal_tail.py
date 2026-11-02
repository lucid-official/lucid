"""``trunc_normal_`` stays inside ``[a, b]`` and keeps its shape when the mean is far outside it.

The inverse-CDF draw puts the bounds' CDF values at 0 or 1 once the interval is
many standard deviations from the mean, which used to pile the samples onto the
near bound (5% of them at ``a`` for ``mean=10``).  Each mean below is checked
against the exact truncated-normal CDF, not only against its bounds.
"""

import math
import warnings

import numpy as np
import pytest

import lucid
import lucid.nn.init as init
from lucid.test._fixtures.devices import metal_available

DEVICES: list[str] = ["cpu", "metal"] if metal_available() else ["cpu"]
MEANS = [0.0, 4.0, 8.0, 10.0, 100.0, -100.0, -4.0, -10.0]
N = 20000
#: Kolmogorov-Smirnov critical value at alpha = 0.001 for one sample of size N.
KS_LIMIT = 1.95 / math.sqrt(N)


def exact_cdf(samples: np.ndarray, mean: float, std: float, a: float, b: float) -> np.ndarray:
    """The truncated-normal CDF at ``samples``, integrated on a fine grid in the tail.

    The density relative to its value at the bound nearest the mean is
    ``exp(-(z**2 - m**2) / 2)``, which stays finite however far out the mean is.
    """
    z_lo, z_hi = (a - mean) / std, (b - mean) / std
    z = (samples.astype(np.float64) - mean) / std
    if z_hi < 0:  # mirror a lower-tail interval onto the upper tail
        z_lo, z_hi, z = -z_hi, -z_lo, -z
        flipped = True
    else:
        flipped = False
    near = max(z_lo, 0.0)
    top = min(z_hi, near + 12.0)
    grid = np.linspace(z_lo, top, 400001)
    density = np.exp(-(grid**2 - near**2) / 2.0)
    cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(grid))])
    cdf /= cdf[-1]
    out = np.interp(z, grid, cdf)
    return 1.0 - out if flipped else out


def ks_statistic(samples: np.ndarray, mean: float, std: float, a: float, b: float) -> float:
    ordered = np.sort(samples)
    cdf = exact_cdf(ordered, mean, std, a, b)
    ranks = np.arange(1, ordered.size + 1)
    return float(max((ranks / ordered.size - cdf).max(), (cdf - (ranks - 1) / ordered.size).max()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mean", MEANS)
def test_far_mean_samples_follow_the_truncated_normal(mean: float, device: str) -> None:
    w = lucid.empty(N, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init.trunc_normal_(w, mean=mean)
    x = w.numpy()
    assert x.min() >= -2.0 and x.max() <= 2.0
    assert max((x == -2.0).mean(), (x == 2.0).mean()) < 1e-3
    assert ks_statistic(x, mean, 1.0, -2.0, 2.0) < KS_LIMIT


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("mean", "std", "a", "b"),
    [
        (1.0, 0.02, -0.04, 0.04),
        (-1.0, 0.02, -0.04, 0.04),
        (0.0, 0.5, 5.0, 9.0),
        (0.0, 1.0, 6.0, 6.5),
    ],
    ids=["scaled_below", "scaled_above", "interval_above_mean", "narrow_interval"],
)
def test_scaled_far_tail_intervals(
    mean: float, std: float, a: float, b: float, device: str
) -> None:
    w = lucid.empty(N, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init.trunc_normal_(w, mean=mean, std=std, a=a, b=b)
    x = w.numpy()
    assert x.min() >= a and x.max() <= b
    assert max((x == a).mean(), (x == b).mean()) < 1e-3
    assert ks_statistic(x, mean, std, a, b) < KS_LIMIT


def test_one_sided_infinite_bound_far_in_the_tail() -> None:
    w = lucid.empty(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init.trunc_normal_(w, mean=0.0, std=1.0, a=8.0, b=math.inf)
    x = w.numpy()
    assert x.min() >= 8.0 and np.isfinite(x).all()
    assert abs(x.mean() - 8.12) < 0.01  # the mean of the tail beyond 8 sigma


@pytest.mark.parametrize("device", DEVICES)
def test_the_far_tail_draw_is_reproducible_and_fills_a_view(device: str) -> None:
    base = lucid.zeros(4, 8, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lucid.manual_seed(5)
        init.trunc_normal_(base[1], mean=10.0)
        first = base.numpy().copy()
        lucid.manual_seed(5)
        init.trunc_normal_(base[1], mean=10.0)
    np.testing.assert_array_equal(base.numpy(), first)
    assert (first[0] == 0).all() and (first[2:] == 0).all() and (first[1] != 0).all()


def test_far_tail_empty_tensor_is_left_alone() -> None:
    w = lucid.empty(0, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert init.trunc_normal_(w, mean=10.0) is w


class TestWarning:
    """The reference's warning, attributed to the caller."""

    @pytest.mark.parametrize("mean", [10.0, -10.0, 100.0])
    def test_far_mean_warns_at_the_call_site(self, mean: float) -> None:
        with pytest.warns(UserWarning, match=r"mean is more than 2 std from \[a, b\]") as caught:
            init.trunc_normal_(lucid.empty(8), mean=mean)
        assert caught[0].filename == __file__

    @pytest.mark.parametrize("mean", [0.0, 1.0, 4.0, -4.0])
    def test_near_mean_does_not_warn(self, mean: float) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            init.trunc_normal_(lucid.empty(8), mean=mean)

    def test_zero_element_warning_is_attributed_to_the_caller(self) -> None:
        with pytest.warns(UserWarning, match="zero-element") as caught:
            init.xavier_uniform_(lucid.empty(0, 3))
        assert caught[0].filename == __file__
