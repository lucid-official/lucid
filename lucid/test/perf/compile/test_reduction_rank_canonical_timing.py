"""Recorded timing comparisons for reduction rank canonical."""

import pytest
from lucid.test._fixtures.devices import metal_available
from lucid.test.perf.compile._timing import assert_ratio_within
from lucid.test.unit.compile.test_reduction_rank_canonical import _step_of

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


def test_a_rank_5_reduction_is_not_slower_than_its_rank_3_twin() -> None:
    """Coarse on purpose: the unfixed graph was ~30x the twin's time, the fixed one ~1.3x."""
    shape = (96, 64, 32, 32)  # the view is made inside the loss; an input is rank 4 either way
    assert_ratio_within(
        lambda: {
            "rank-5": _step_of(shape, lambda y: y.reshape(6144, 16, 2, 16, 2).sum(dim=2).sum()),
            "rank-3": _step_of(shape, lambda y: y.reshape(6144 * 16, 2, 32).sum(dim=1).sum()),
        },
        "rank-5",
        "rank-3",
        bound=8.0,
        slack_ms=5.0,
    )
