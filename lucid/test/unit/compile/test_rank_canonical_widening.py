"""A repeat or tile keeps its intermediate at rank <= 4, whatever its operand's rank (LCD-995).

``repeat`` is spelled "insert a size-1 axis, broadcast it, collapse it"; its backward folds the
gradient through the same split, and a tile's through ``(r0, d0, r1, d1, ...)``, rank 2n.  On a
rank-4 operand that is a rank-5 intermediate (a rank-6 one and up for a tile's backward), which
MPSGraph runs as a loop over the dims above the last four, ~4.5 us a slice: ``repeat`` of
``(8192, 8, 16, 16)`` took 51 ms against 3.2 ms for the same elements at rank 3.  ``RankCanon``
keys on a node's own rank and builds nothing for the backward, so the emitters keep the
intermediate at rank <= 4 themselves (``OpEmitters/_AxisSplitLayout.h``): the axes around the one
the op names merge.  The policy is the pass's lead gate, so a small lead, a symbolic dim and
``LUCID_COMPILE_RANK_CANON_MIN_LEAD=-1`` keep the native spelling.

The graph cannot be inspected from Python, so the guards are

* a value table on Metal: the forward and the ``make_step`` gradients of the merged spelling
  (gate ``0``) equal eager's and the native spelling's (gate ``-1``), at float32, float16 and
  bfloat16, over the axes and repeat counts that change where the merge falls;
* a symbolic batch, which keeps the native spelling;
* a coarse timing bound against the rank-3 twin of each case (the same elements, whose
  intermediate is rank 4 natively), the same measurement with the gate off, which must exceed the
  bound (a bound the unfixed graph meets guards nothing), and the GQA-sized repeat, whose lead is
  under the gate and must keep its native speed.
"""

import os
from collections.abc import Callable
from unittest import mock

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_ratio_within,
    compare_ms,
    compiled_call,
    step_call,
    with_env,
)
from lucid.test.unit.compile.test_rank_canonical_movement import (
    _DTYPES,
    ENV,
    Op,
    Shape,
    _check_step,
    _filled,
    _forward,
    _metal,
    _on_metal,
    _same,
    _with_nonfinite,
    needs_metal,
)

pytestmark = pytest.mark.filterwarnings(
    "error::lucid.compile._core.fallback.CompileFallbackWarning"
)

R3: Shape = (4, 3, 5)
R4: Shape = (3, 4, 2, 3)

#: ``(shape, op)``.  A rank-4 operand widens a repeat to rank 5 (over the gate at ``0``); a rank-3
#: operand widens a tile's backward to rank 6.  Both are pure data moves forward, so the forward is
#: bit-identical to eager's; the backward sums and is held to the dtype's rounding.
_CASES: dict[str, tuple[Shape, Op]] = {
    "repeat_last": (R4, lambda t: lucid.repeat(t, 3, dim=-1)),
    "repeat_first": (R4, lambda t: lucid.repeat(t, 2, dim=0)),
    "repeat_mid": (R4, lambda t: lucid.repeat(t, 2, dim=2)),
    "repeat_once": (R4, lambda t: lucid.repeat(t, 1, dim=1)),
    "repeat_rank_2": ((6, 5), lambda t: lucid.repeat(t, 2, dim=1)),
    "tile_mid": (R3, lambda t: lucid.tile(t, (1, 2, 1))),
    "tile_two": (R3, lambda t: lucid.tile(t, (2, 1, 3))),
    "tile_all": (R3, lambda t: lucid.tile(t, (2, 2, 2))),
    "tile_none": (R3, lambda t: lucid.tile(t, (1, 1, 1))),
    "tile_first_r4": (R4, lambda t: lucid.tile(t, (2, 1, 1, 1))),
    "tile_two_r4": (R4, lambda t: lucid.tile(t, (2, 1, 2, 1))),
}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_widening_forward_matches_eager_and_the_native_spelling(
    name: str, dtype: lucid.dtype
) -> None:
    shape, op = _CASES[name]
    x = _on_metal(_with_nonfinite(_filled(shape, 0.3)), dtype)
    fn: Op = lambda t: op(t * 1.0) + 0.0
    want = fn(x)
    _same(_forward(fn, (x,), "0"), want, "merged vs eager")
    _same(_forward(fn, (x,), "-1"), want, "native vs eager")


@needs_metal
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_widening_step_matches_eager_and_the_native_spelling(
    name: str, dtype: lucid.dtype
) -> None:
    shape, op = _CASES[name]
    _check_step(op, True, shape, dtype)


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_symbolic_batch_keeps_the_native_spelling() -> None:
    """A dim the batch makes unknown cannot be merged: the layout steps aside, the result holds."""

    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return lucid.tile(lucid.repeat(t * 1.5, 2, dim=2), (1, 1, 1, 2)) + 1.0

    with mock.patch.dict(os.environ, {ENV: "0"}):
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(fn, dynamic=True)
        for batch in (3, 5, 3):
            x = _on_metal(_filled((batch, 4, 2, 3), 0.2 * batch), lucid.float32)
            _same(compiled(x), fn(x), f"batch {batch}")


# ── time ──────────────────────────────────────────────────────────────────────────────

#: 16.8M elements at rank 4 and the same elements at rank 3 with the axes around the repeated one
#: merged, which is what the layout reads them as.
_W4: Shape = (8192, 8, 16, 16)
_REPEAT_TWINS: dict[str, tuple[int, Shape]] = {
    "last": (-1, (65536, 16, 16)),
    "mid": (1, (8192, 8, 256)),
    "first": (0, (8192, 2048)),
}


class _Stepped(nn.Module):
    """``sum(op(x * s) ** 2)`` over a parameter ``x``: the step runs the op's backward."""

    def __init__(self, op: Op, shape: Shape) -> None:
        super().__init__()
        self.op = op
        self.x = nn.Parameter(_metal(shape, 0.1))

    def forward(self, s: lucid.Tensor) -> lucid.Tensor:
        y = self.op(self.x * s)
        return (y * y).sum()


def _step(op: Op, shape: Shape, env: str | None) -> Callable[[], float]:
    step = step_call(_Stepped(op, shape), lambda y: y, lucid.ones(1).to(COMPILE_DEVICE))
    return with_env(step, ENV, env)


def _repeat_forward(dim: int) -> Op:
    return lambda t: lucid.repeat(t * 1.0, 2, dim=dim).reshape(-1).sum()


def _repeat_step(dim: int) -> Op:
    return lambda t: lucid.repeat(t, 2, dim=dim)


#: The tile's twin at rank 2, where the backward's interleaved ``(r0, d0, r1, d1)`` is rank 4: the
#: same output elements, tiled along the leading axis.
_TILE_WIDE: Shape = (65536, 16, 16)
_TILE_TWIN: Shape = (1048576, 16)

_BOUND = 1.5
#: A step is 10-20 ms and runs the forward and the backward, so it gets more slack than the
#: forward alone (fixed ratios 0.75x-1.4x on a loaded M1 Pro; unfixed 10x to 70x).
_STEP_BOUND = 2.0


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("where", sorted(_REPEAT_TWINS))
def test_a_rank_4_repeat_is_not_slower_than_its_rank_3_twin(where: str) -> None:
    """Held to 1.5x the twin; unfixed the rank-5 intermediate ran 14x to 17x it."""
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

    unfixed = compare_ms(build, "gate off", "twin", bound=_BOUND, slack_ms=0.3, attempts=2)
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

    unfixed = compare_ms(build, "gate off", "twin", bound=_STEP_BOUND, slack_ms=0.5, attempts=2)
    assert not unfixed.within, (
        f"{where}: the unfixed backward meets the guard's bound, so the guard cannot fail: "
        f"{unfixed.report}"
    )


def _tile_wide(t: lucid.Tensor) -> lucid.Tensor:
    return lucid.tile(t, (1, 2, 1))


def _tile_twin(t: lucid.Tensor) -> lucid.Tensor:
    return lucid.tile(t, (2, 1))


@needs_metal
def test_a_tile_backward_is_not_slower_than_its_rank_2_twin() -> None:
    """Held to 2x the twin; unfixed the rank-6 interleave ran 40x it."""

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

    unfixed = compare_ms(build, "gate off", "twin", bound=_STEP_BOUND, slack_ms=0.5, attempts=2)
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
