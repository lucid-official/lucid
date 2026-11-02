"""Parity: ``nn.init.trunc_normal_`` draws the reference's distribution at every mean.

Two-sample Kolmogorov-Smirnov against the reference initialiser, on CPU and
Metal, for means from at the interval's centre to a hundred standard deviations
outside it.  The reference also warns on a far mean; both sides must.
"""

import math
import warnings
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.init as init
from lucid.test._fixtures.devices import metal_available

DEVICES: list[str] = ["cpu", "metal"] if metal_available() else ["cpu"]
N = 20000
#: Two-sample Kolmogorov-Smirnov critical value at alpha = 0.001.
KS_LIMIT = 1.95 * math.sqrt(2 / N)


def two_sample_ks(x: np.ndarray, y: np.ndarray) -> float:
    grid = np.sort(np.concatenate([x, y]))
    fx = np.searchsorted(np.sort(x), grid, side="right") / x.size
    fy = np.searchsorted(np.sort(y), grid, side="right") / y.size
    return float(np.abs(fx - fy).max())


@pytest.mark.parity
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mean", [0.0, 4.0, 8.0, 10.0, 100.0, -100.0])
def test_samples_match_the_reference_distribution(ref: Any, mean: float, device: str) -> None:
    mine = lucid.empty(N, device=device)
    theirs = ref.empty(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init.trunc_normal_(mine, mean=mean)
        ref.nn.init.trunc_normal_(theirs, mean=mean)
    got, want = mine.numpy(), theirs.numpy()
    assert got.min() >= -2.0 and got.max() <= 2.0
    assert two_sample_ks(got, want) < KS_LIMIT


@pytest.mark.parity
@pytest.mark.parametrize("mean", [0.0, 4.0, 10.0, -10.0, 100.0])
def test_the_warning_matches_the_reference(ref: Any, mean: float) -> None:
    def messages(call: Any) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        return [str(w.message) for w in caught]

    want = messages(lambda: ref.nn.init.trunc_normal_(ref.empty(8), mean=mean))
    got = messages(lambda: init.trunc_normal_(lucid.empty(8), mean=mean))
    assert got == want
