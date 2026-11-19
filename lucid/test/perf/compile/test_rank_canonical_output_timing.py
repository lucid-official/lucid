"""Recorded timing comparisons for rank canonical output."""

from collections.abc import Callable
import pytest
import lucid
from lucid.test._fixtures.devices import metal_available
from lucid.test.perf.compile._timing import assert_ratio_within
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, compiled_call, read_first
from lucid.test.unit.compile.test_rank_canonical_output import (
    _filled,
    _default_gate as _default_gate,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


@pytest.mark.usefixtures("inference")
def test_a_rank_5_output_is_not_slower_than_its_rank_4_twin() -> None:
    """Unfixed: 26 ms against 2 ms.  Each rank-5 output is held to 1.5x its own rank-4 twin.

    The twin differs from the case in the output alone, so a cost the *input* side pays
    (a rank-5 feed) is in both.
    """
    r5 = (6144, 16, 2, 16, 2)
    r4 = (6144, 16, 2, 32)
    x5, x4 = _filled(r5).to(COMPILE_DEVICE), _filled(r4).to(COMPILE_DEVICE)

    def pair(
        case: tuple[Callable[[lucid.Tensor], lucid.Tensor], lucid.Tensor],
        twin: tuple[Callable[[lucid.Tensor], lucid.Tensor], lucid.Tensor],
    ) -> Callable[[], dict[str, Callable[[], float]]]:
        return lambda: {
            "case": compiled_call(case[0], case[1], read=read_first),
            "twin": compiled_call(twin[0], twin[1], read=read_first),
        }

    # The unfixed output is ~10x its twin, so a noisy attempt cannot hide the defect.
    assert_ratio_within(
        pair((lambda a: a + 1.0, x5), (lambda a: (a + 1.0).reshape(*r4), x5)),
        "case",
        "twin",
        bound=1.5,
        what="rank-5 output",
    )
    # A graph output that is a reshape of a computed value costs a rank-4 reshape too
    # (+25% here, whatever the rank), so the twin is a rank-4 reshape.
    assert_ratio_within(
        pair(
            (lambda a: (a + 1.0).reshape(*r5), x4),
            (lambda a: (a + 1.0).reshape(98304, 2, 16, 2), x4),
        ),
        "case",
        "twin",
        bound=1.5,
        what="reshape to rank 5",
    )
