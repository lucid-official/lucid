"""Recorded timing comparisons for rank canon."""

import pytest
from lucid.test.perf.compile._timing import assert_ratio_within
from lucid.test.unit.compile.test_rank_canon import (
    _forward_call,
    needs_metal,
    _default_gate as _default_gate,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_rank_5_multiply_is_not_slower_than_its_rank_4_twin() -> None:
    """Both sides reduce the same 32 elements per output into the same (6144, 16, 2)
    result, so only the rank differs.  The old twin reduced a size-32 axis against the
    rank-5 graph's size-2 axis; that reduction difference, not the rank, was 2.4-2.7x of
    the ratio the hosted runner reported (LCD-935 runner probe, rank5-probe.yml).  With
    matched reductions the fixed graph measures 1.3-1.4x on the M1 Pro and the virtual
    M1 alike; the unfixed graph is about 15-30x."""
    assert_ratio_within(
        lambda: {
            "rank-5": _forward_call((6144, 16, 2, 16, 2), (-2, -1)),
            "rank-4": _forward_call((6144, 16, 2, 32), (-1,)),
        },
        "rank-5",
        "rank-4",
        bound=2.0,
        slack_ms=1.0,
    )
