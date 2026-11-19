"""Recorded timing comparisons for group norm rank canonical."""

import pytest
import lucid
from lucid.test._fixtures.devices import metal_available
from lucid.test.perf.compile._timing import assert_ratio_within
from lucid.test.unit.compile._helpers import step_call, to_metal
from lucid.test.unit.compile.test_group_norm_rank_canonical import _Twin, _input, _loss, _module

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


def test_a_group_norm_is_not_slower_than_its_rank_2_twin() -> None:
    """Coarse on purpose: the grouped rank-5 form ran ~6x the twin at N = 256, the fix ~1.3x."""
    shape = (256, 64, 16, 16)
    x = _input(shape, lucid.float32)
    assert_ratio_within(
        lambda: {
            "group_norm": step_call(_module(shape, 8, True, lucid.float32), _loss, x),
            "rank-2 twin": step_call(to_metal(_Twin(shape, 8)), _loss, x),
        },
        "group_norm",
        "rank-2 twin",
        bound=2.0,
        slack_ms=1.0,
    )
