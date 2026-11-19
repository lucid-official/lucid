"""Recorded timing comparisons for rank canonical widening."""

from collections.abc import Callable
import pytest
import lucid
from lucid._C import engine as _C_engine
from lucid.test.perf.compile._timing import assert_ratio_within, compare_ms
from lucid.test.unit.compile._helpers import compiled_call, with_env
from lucid.test.unit.compile.test_rank_canonical_movement import ENV, Shape, _metal, needs_metal
from lucid.test.unit.compile.test_rank_canonical_widening import (
    _BOUND,
    _REPEAT_TWINS,
    _STEP_BOUND,
    _TILE_TWIN,
    _TILE_WIDE,
    _UNFIXED_ATTEMPTS,
    _W4,
    _repeat_forward,
    _repeat_step,
    _step,
    _tile_twin,
    _tile_wide,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("where", sorted(_REPEAT_TWINS))
def test_a_rank_4_repeat_is_not_slower_than_its_rank_3_twin(where: str) -> None:
    """Held to 1.5x the twin; unfixed the rank-5 intermediate ran 19x to 21x it."""
    dim, twin_shape = _REPEAT_TWINS[where]
    fn = _repeat_forward(dim)

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "rank-4": compiled_call(fn, _metal(_W4)),
            "twin": compiled_call(fn, _metal(twin_shape)),
        }

    assert_ratio_within(build, "rank-4", "twin", bound=_BOUND, slack_ms=0.3, what=where)


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("where", sorted(_REPEAT_TWINS))
def test_the_repeat_bound_is_exceeded_by_the_unfixed_graph(where: str) -> None:
    """The guard above can fail: with the gate off the rank-4 repeat is over its bound."""
    dim, twin_shape = _REPEAT_TWINS[where]

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "gate off": with_env(compiled_call(_repeat_forward(dim), _metal(_W4)), ENV, "-1"),
            "twin": compiled_call(_repeat_forward(dim), _metal(twin_shape)),
        }

    unfixed = compare_ms(
        build, "gate off", "twin", bound=_BOUND, slack_ms=0.3, attempts=_UNFIXED_ATTEMPTS
    )
    assert not unfixed.within, (
        f"{where}: the unfixed graph meets the guard's bound, so the guard cannot fail: "
        f"{unfixed.report}"
    )


@needs_metal
@pytest.mark.parametrize("where", sorted(_REPEAT_TWINS))
def test_a_rank_4_repeat_backward_is_not_slower_than_its_rank_3_twin(where: str) -> None:
    dim, twin_shape = _REPEAT_TWINS[where]

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "rank-4": _step(_repeat_step(dim), _W4, None),
            "twin": _step(_repeat_step(dim), twin_shape, None),
        }

    assert_ratio_within(build, "rank-4", "twin", bound=_STEP_BOUND, slack_ms=0.5, what=where)


@needs_metal
@pytest.mark.parametrize("where", sorted(_REPEAT_TWINS))
def test_the_repeat_backward_bound_is_exceeded_by_the_unfixed_graph(where: str) -> None:
    dim, twin_shape = _REPEAT_TWINS[where]

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "gate off": _step(_repeat_step(dim), _W4, "-1"),
            "twin": _step(_repeat_step(dim), twin_shape, None),
        }

    unfixed = compare_ms(
        build, "gate off", "twin", bound=_STEP_BOUND, slack_ms=0.5, attempts=_UNFIXED_ATTEMPTS
    )
    assert not unfixed.within, (
        f"{where}: the unfixed backward meets the guard's bound, so the guard cannot fail: "
        f"{unfixed.report}"
    )


@needs_metal
def test_a_tile_backward_is_not_slower_than_its_rank_2_twin() -> None:
    """Held to 2x the twin; unfixed the rank-6 interleave ran 22x it."""

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "tile": _step(_tile_wide, _TILE_WIDE, None),
            "twin": _step(_tile_twin, _TILE_TWIN, None),
        }

    assert_ratio_within(build, "tile", "twin", bound=_STEP_BOUND, slack_ms=0.5, what="tile")


@needs_metal
def test_the_tile_backward_bound_is_exceeded_by_the_unfixed_graph() -> None:
    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "gate off": _step(_tile_wide, _TILE_WIDE, "-1"),
            "twin": _step(_tile_twin, _TILE_TWIN, None),
        }

    unfixed = compare_ms(
        build, "gate off", "twin", bound=_STEP_BOUND, slack_ms=0.5, attempts=_UNFIXED_ATTEMPTS
    )
    assert not unfixed.within, (
        f"the unfixed tile backward meets the guard's bound, so the guard cannot fail: "
        f"{unfixed.report}"
    )


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_gqa_sized_repeat_keeps_its_native_speed() -> None:
    """``(8, 8, 512, 64)`` repeated along the heads has a lead of 8, under the gate: the graph is
    the native one (1.3 ms), so the gate-on call is held to the gate-off one."""
    gqa: Shape = (8, 8, 512, 64)

    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return lucid.repeat(t * 1.0, 4, dim=1).reshape(-1).sum()

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "gate default": with_env(compiled_call(fn, _metal(gqa)), ENV, None),
            "gate off": with_env(compiled_call(fn, _metal(gqa)), ENV, "-1"),
        }

    assert_ratio_within(build, "gate default", "gate off", bound=_BOUND, slack_ms=0.3)
