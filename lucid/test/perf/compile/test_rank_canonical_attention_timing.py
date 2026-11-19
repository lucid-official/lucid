"""Recorded timing comparisons for rank canonical attention."""

from collections.abc import Callable
import pytest
import lucid
from lucid.test.perf.compile._timing import assert_ratio_within
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.compile.test_rank_canonical_layout import _filled, needs_metal
from lucid.test.unit.compile.test_rank_canonical_attention import (
    _DEFAULT_GATE,
    _gated,
    _norm_after_add,
    _scaled_attention,
    attention_state as attention_state,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_rank_5_attention_runs_within_1_5x_of_its_rank_4_twin(attention_state: int) -> None:
    # Lead 1024; the twin merges the two head axes: same elements, rank 4.
    five = tuple(_filled((1024, 2, 8, 49, 32), ph).to(COMPILE_DEVICE) for ph in (0.0, 1.0, 2.0))
    four = tuple(t.reshape(1024, 16, 49, 32) for t in five)
    assert_ratio_within(
        lambda: {
            "rank-5": _gated(_scaled_attention, five, _DEFAULT_GATE),
            "rank-4 twin": _gated(_scaled_attention, four, _DEFAULT_GATE),
        },
        "rank-5",
        "rank-4 twin",
        bound=1.5,
    )


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("case", ["attention_scale", "layer_norm_after_add"])
def test_the_pass_is_no_slower_than_off(case: str) -> None:
    """LCD-925: the scale multiply was wrapped while the exempt softmax read it at rank 5 (the pass
    ran 2.07x off), and an add was wrapped in front of a layer norm that read it at rank 5 (1.32x).

    Held to 1.5x: the hosted runner measured the fixed ``layer_norm_after_add`` at 1.11x off (0.75x
    on an M1 Pro, LCD-935), so no bound under ~1.2x holds there.  A pass that costs 1.5x or more
    fails; a cost between that and the runner's own spread is held by the IR tests above, which
    read the graph (``test_a_producer_read_by_an_op_no_rule_owns_is_left_alone``,
    ``test_a_norm_consumer_reads_the_producers_view_for_free``), not a clock."""
    if case == "attention_scale":
        fn: Callable[..., lucid.Tensor] = _scaled_attention
        args = tuple(_filled((1024, 2, 8, 49, 32), ph).to(COMPILE_DEVICE) for ph in (0.0, 1.0, 2.0))
    else:
        fn = _norm_after_add
        args = (_filled((64, 8, 16, 16, 96)).to(COMPILE_DEVICE),)
    assert_ratio_within(
        lambda: {"off": _gated(fn, args, "-1"), "on": _gated(fn, args, _DEFAULT_GATE)},
        "on",
        "off",
        bound=1.5,
        what=case,
    )
