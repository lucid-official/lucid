"""Recorded timing comparisons for sort rank order."""

import warnings
from collections.abc import Callable
import pytest
import lucid
from lucid._C import engine as _C_engine
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.test._fixtures.devices import metal_available
from lucid.test.perf.compile._timing import assert_ratio_within
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.compile.test_sort_rank_order import _OPS, _fill

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


@pytest.mark.parametrize("name", list(_OPS))
@pytest.mark.parametrize("shape", [(256, 8, 16, 16), (64, 4, 8, 16, 16)])
def test_a_compiled_sort_is_not_slower_than_eager(name: str, shape: tuple[int, ...]) -> None:
    fn = _OPS[name]
    x = _fill(shape).to(COMPILE_DEVICE)

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(fn)
        return {"compiled": lambda: float(compiled(x).item()), "eager": lambda: float(fn(x).item())}

    # The guard is for the per-row argSort cost (~100x, LCD-926).  The ranked path runs
    # 1.7-2.6x eager at rank 5 under load (topk smallest), so 3x flaked.
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        assert_ratio_within(build, "compiled", "eager", bound=5.0, what=f"{name} {shape}")
