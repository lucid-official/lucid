"""Recorded timing comparisons for rank canonical feed."""

from collections.abc import Callable
import pytest
from lucid._C import engine as _C_engine
from lucid.test.perf.compile._timing import checked_twin_shape, assert_ratio_within
from lucid.test.unit.compile._helpers import compiled_call, read_first
from lucid.test.unit.compile.test_rank_canonical_feed import (
    _TWINS,
    _filled,
    _on,
    needs_metal,
    _default_gate as _default_gate,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", sorted(_TWINS))
def test_a_rank_5_feed_is_not_slower_than_its_rank_4_twin(name: str) -> None:
    """Held to 1.5x the twin; the views of a rank-5 feed ran 1.9x to 2.6x it, the pass off 4.7x+."""
    fn, wide, twin = _TWINS[name]
    twin = checked_twin_shape(wide, twin)
    xs5 = (_on(_filled(wide, 0.1)), _on(_filled(wide, 0.9)))
    xs4 = (_on(_filled(twin, 0.1)), _on(_filled(twin, 0.9)))

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "rank-5": compiled_call(fn, *xs5, read=read_first),
            "rank-4 twin": compiled_call(fn, *xs4, read=read_first),
        }

    assert_ratio_within(build, "rank-5", "rank-4 twin", bound=1.5, slack_ms=0.1, what=name)
