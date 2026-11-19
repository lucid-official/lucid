"""A rank-5+ index op in a compiled graph is not slower than its rank-<=4 twin (LCD-1014).

``RankCanon`` reads an index op's operands and answer in a layout of rank <= 4 (``FlatIndexRule``,
``AlongAxisIndexRule``, ``SearchsortedRule``); unrewritten, MPSGraph runs the op as a loop over the
dims above the last four, 5x to 600x the twin.  The guard that gates a push is on the IR
(``lucid/test/unit/compile/test_rank_canonical_index.py``: the case computes what its twin computes,
at rank <= 4).  This is the clock row for what the IR cannot see, the cost of an emitter at a layout
both sides share or of a view MPSGraph materialises.

It is here and not in the phase gate because a wall-clock ratio of a ~1 ms graph is not something the
hosted runner holds: the int32 embedding read 2.04x against a 1.5x bound (attempt 3 of 3) on a graph
that is its twin's (LCD-1039).  The perf tier runs on every push as a warning and once a night as a
failure, so a regression still shows -- as a number to read, not as a red phase gate.

Run on its own with::

    python3 -m pytest lucid/test/perf/compile -m perf
"""

from collections.abc import Callable

import pytest

import lucid
from lucid.test.unit.compile._layout_twins import LAYOUT_CASES, layout_twin

from lucid._C import engine as _C_engine
from lucid.test._default_device import metal_available
from lucid.test.perf.compile._timing import assert_ratio_within, compare_ms
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, compiled_call, with_env
from lucid.test.unit.compile._index_twins import CASES
from lucid.test.unit.compile._rank_twins import Side
from lucid.test.unit.compile._movement_twins import (
    MOVEMENT_CASES,
    movement_twin,
    space_to_depth_side,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # a fallback to eager would time eager against eager
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]

ENV = "LUCID_COMPILE_RANK_CANON_MIN_LEAD"
_BOUND = 1.5
#: What a sub-millisecond case adds on top of the ratio: the launch and sync jitter of a graph this
#: small, and the ~0.2 ms MPSGraph copies a rank-4 index feed read through the pass's flat view.
#: The unfixed graphs are 13x to 70x their twin.
_SLACK_MS = 0.5
#: The unfixed graph is expected to miss the bound, and ``compare_ms`` only stops early on a hit,
#: so every attempt of an "exceeds the bound" guard runs: one is enough for a miss of 4x or more.
_UNFIXED_ATTEMPTS = 1


def _timed(side: Side, gate: str | None) -> Callable[[], float]:
    return with_env(compiled_call(side.fn, *side.args), ENV, gate)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_rank_5_index_op_is_not_slower_than_its_twin(name: str) -> None:
    """Held to 1.5x the twin; unfixed they ran 5x to 600x it."""
    case = CASES[name](COMPILE_DEVICE)

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {"rank-5": _timed(case.wide, None), "twin": _timed(case.twin, None)}

    assert_ratio_within(build, "rank-5", "twin", bound=_BOUND, slack_ms=_SLACK_MS, what=name)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_index_op_bound_is_exceeded_by_the_unfixed_graph(name: str) -> None:
    """The guard above can fail: with the pass off the same op is over its bound."""
    case = CASES[name](COMPILE_DEVICE)

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {"gate off": _timed(case.wide, "-1"), "twin": _timed(case.twin, None)}

    unfixed = compare_ms(
        build, "gate off", "twin", bound=_BOUND, slack_ms=_SLACK_MS, attempts=_UNFIXED_ATTEMPTS
    )
    assert not unfixed.within, (
        f"{name}: the unfixed graph meets the guard's bound, so the guard cannot fail: "
        f"{unfixed.report}"
    )


@pytest.mark.parametrize("name", sorted(LAYOUT_CASES))
def test_a_rank_5_case_is_not_slower_than_its_rank_4_twin(name: str) -> None:
    """Held to 1.5x the twin; the unfixed graphs ran 2x to 20x it."""
    case = layout_twin(name, COMPILE_DEVICE)

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {"rank-5+": _timed(case.wide, None), "twin": _timed(case.twin, None)}

    with lucid.no_grad():
        assert_ratio_within(build, "rank-5+", "twin", bound=1.5, slack_ms=0.3, what=name)


@pytest.mark.parametrize("name", sorted(MOVEMENT_CASES))
@pytest.mark.parametrize("gate", [None, "-1"], ids=["default", "native"])
def test_movement_time_against_twin(name: str, gate: str | None) -> None:
    case = movement_twin(name, COMPILE_DEVICE)

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {"case": _timed(case.wide, gate), "twin": _timed(case.twin, None)}

    bound = 3.0 if name == "expand" else 1.5
    with lucid.no_grad():
        if gate is None:
            assert_ratio_within(build, "case", "twin", bound=bound, slack_ms=0.3, what=name)
        else:
            timing = compare_ms(build, "case", "twin", bound=bound, slack_ms=0.3, attempts=1)
            assert not timing.within, timing.report


def test_space_to_depth_time_against_always_on() -> None:
    side = space_to_depth_side(COMPILE_DEVICE)

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {"default": _timed(side, None), "always": _timed(side, "0")}

    with lucid.no_grad():
        assert_ratio_within(build, "default", "always", bound=1.5, slack_ms=0.3)
