"""Recorded timing comparisons for norm pool rank canonical."""

from collections.abc import Callable
import pytest
import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available
from lucid.test.perf.compile._timing import assert_ratio_within
from lucid.test.unit.compile._helpers import with_env
from lucid.test.unit.compile.test_norm_pool_rank_canonical import (
    _ENV,
    _fwd_call,
    _hand_view_norm,
    _input,
    _step_call,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


def _ratio_within(
    build: Callable[[], dict[str, Callable[[], float]]], num: str, den: str, bound: float
) -> None:
    """``num / den <= bound``: a fresh build and measurement per attempt."""

    def fresh() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return build()

    assert_ratio_within(fresh, num, den, bound=bound)


@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("k", [96, 16])
def test_a_rank_5_norm_is_not_slower_than_the_same_elements_at_rank_2(kind: str, k: int) -> None:
    """Producer -> norm -> consumer on a rank-5 value, against the rank-2 view of it (the
    layout the emitter now builds).  The native form ran 2-19x that."""
    shape = (64, 8, 16, 16, k) if k == 96 else (1024, 4, 8, 16, k)
    x = _input(shape, lucid.float32)

    def native(t: lucid.Tensor) -> lucid.Tensor:
        n = F.layer_norm(t * 1.5, (k,)) if kind == "layer_norm" else F.rms_norm(t * 1.5, (k,))
        return n * 2.0

    twin = _hand_view_norm(kind, k)

    def view(t: lucid.Tensor) -> lucid.Tensor:
        return twin(t * 1.5) * 2.0

    _ratio_within(
        lambda: {"native": _fwd_call(native, x), "view": _fwd_call(view, x)}, "native", "view", 1.5
    )


@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
def test_a_rank_5_norm_step_is_not_slower_than_the_same_elements_at_rank_2(kind: str) -> None:
    shape = (1024, 4, 8, 16, 16)
    x = _input(shape, lucid.float32)
    norm = (
        (lambda t: F.layer_norm(t, (16,)))
        if kind == "layer_norm"
        else (lambda t: F.rms_norm(t, (16,)))
    )
    twin = _hand_view_norm(kind, 16)
    _ratio_within(
        lambda: {"native": _step_call(norm, shape, x), "view": _step_call(twin, shape, x)},
        "native",
        "view",
        1.5,
    )


def test_the_canonical_norm_beats_the_native_rank_5_form() -> None:
    """The same graph with the policy off (the old rank-5 spelling) against on."""
    shape = (1024, 4, 8, 16, 16)
    x = _input(shape, lucid.float32)

    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return F.layer_norm(t * 1.5, (16,)) * 2.0

    assert_ratio_within(
        lambda: {
            "off": with_env(_fwd_call(fn, x), _ENV, "-1"),
            "on": with_env(_fwd_call(fn, x), _ENV, None),
        },
        "on",
        "off",
        bound=0.6,
        what="canonical layer_norm is not 1.7x the native form",
    )


def test_a_rank_5_avg_pool3d_is_not_slower_than_its_rank_4_view() -> None:
    """A (1, 2, 2) window is a 2-D pool of ``(N*C, D, H, W)``; the rank-6 lift ran 5x that."""
    shape = (64, 16, 8, 32, 32)
    x = _input(shape, lucid.float32)

    def pool(t: lucid.Tensor) -> lucid.Tensor:
        return F.avg_pool3d(t * 1.5, (1, 2, 2)) * 2.0

    def view(t: lucid.Tensor) -> lucid.Tensor:
        y = F.avg_pool2d((t * 1.5).reshape(-1, *t.shape[2:]), 2)
        return y.reshape(t.shape[0], t.shape[1], *y.shape[1:]) * 2.0

    _ratio_within(
        lambda: {"pool3d": _fwd_call(pool, x), "view": _fwd_call(view, x)}, "pool3d", "view", 1.5
    )


def test_a_rank_5_max_pool3d_step_is_not_slower_than_its_rank_4_view() -> None:
    """The max pool's backward slices its source per window offset: at rank 6 it ran 18x."""
    shape = (64, 16, 8, 32, 32)
    x = _input(shape, lucid.float32)

    def pool(t: lucid.Tensor) -> lucid.Tensor:
        return F.max_pool3d(t, (1, 2, 2))

    def view(t: lucid.Tensor) -> lucid.Tensor:
        y = F.max_pool2d(t.reshape(-1, *t.shape[2:]), 2)
        return y.reshape(t.shape[0], t.shape[1], *y.shape[1:])

    _ratio_within(
        lambda: {"pool3d": _step_call(pool, shape, x), "view": _step_call(view, shape, x)},
        "pool3d",
        "view",
        1.5,
    )
