"""Recorded timing comparisons for wide integer rank canonical."""

from collections.abc import Callable
import pytest
from lucid._C import engine as _C_engine
from lucid.test.perf.compile._timing import assert_ratio_within, compare_ms
from lucid.test.unit.compile.test_rank_canonical_movement import needs_metal
from lucid.test.unit.compile.test_wide_integer_rank_canonical import (
    _BOUND,
    _ROWS,
    _TWINS,
    _UNFIXED_ATTEMPTS,
    _W4,
    _timed,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


@needs_metal
@pytest.mark.parametrize(("name", "where"), _ROWS)
def test_a_rank_4_carried_op_is_not_slower_than_its_rank_3_twin(name: str, where: str) -> None:
    """Held to 1.5x the twin; unfixed the rank-5 carried operand ran 4x to 8x it."""
    twin_shape = _TWINS[where][1]

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "rank-4": _timed(name, where, None, _W4),
            "twin": _timed(name, where, None, twin_shape),
        }

    assert_ratio_within(build, "rank-4", "twin", bound=_BOUND, slack_ms=0.3, what=f"{name} {where}")


@needs_metal
@pytest.mark.parametrize(("name", "where"), _ROWS)
def test_the_carried_op_bound_is_exceeded_by_the_unfixed_graph(name: str, where: str) -> None:
    """The guard above can fail: with the gate off the rank-4 op is over its bound."""
    twin_shape = _TWINS[where][1]

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "gate off": _timed(name, where, "-1", _W4),
            "twin": _timed(name, where, None, twin_shape),
        }

    unfixed = compare_ms(
        build, "gate off", "twin", bound=_BOUND, slack_ms=0.3, attempts=_UNFIXED_ATTEMPTS
    )
    assert not unfixed.within, (
        f"{name} {where}: the unfixed graph meets the guard's bound, so the guard cannot fail: "
        f"{unfixed.report}"
    )
