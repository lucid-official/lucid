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

The guards cover

* a value table on Metal: the forward and the ``make_step`` gradients of the merged spelling
  (gate ``0``) equal eager's and the native spelling's (gate ``-1``), at float32, float16 and
  bfloat16, over the axes and repeat counts that change where the merge falls;
* a symbolic batch, which keeps the native spelling;
* the physical operand shapes of the emitted work against an equal-element twin. The clock
  comparisons run under ``lucid/test/perf/compile/``.
"""

import os
from collections.abc import Callable
from unittest import mock

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, step_call, with_env
from lucid.test.unit.compile._rank_twins import _trace, RankTwin, Side, emitted, operation_shapes
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

#: 1M elements at rank 4 and the same elements at rank 3 with the axes around the repeated one
#: merged, which is what the layout reads them as.  What the unfixed graph pays is the slice loop
#: over the lead (2048 slices), not the elements, so the inner dims are small: the first version of
#: this guard, ``(8192, 8, 16, 16)`` (16.8M elements), ran the file to a 39 GB footprint, and at
#: this size the whole file stays under 1 GB with the unfixed graph still 19x to 23x its twin.
_W4: Shape = (2048, 8, 8, 8)
_REPEAT_TWINS: dict[str, tuple[int, Shape]] = {
    "last": (-1, (131072, 8)),
    "mid": (1, (2048, 8, 64)),
    "first": (0, (2048, 512)),
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
#: same output elements, tiled along the leading axis.  Sized like ``_W4``: 2048 slices of 64.
_TILE_WIDE: Shape = (2048, 8, 8)
_TILE_TWIN: Shape = (16384, 8)

_BOUND = 1.5
#: A step runs the forward and the backward, so it gets more slack than the forward alone (fixed
#: ratios 0.75x-1.4x on a loaded M1 Pro, 0.98x-1.00x at this size; unfixed 21x to 23x).
_STEP_BOUND = 2.0
#: The unfixed graph is expected to miss the bound, and ``compare_ms`` only stops early on a hit,
#: so every attempt of an "exceeds the bound" guard runs: one is enough for a 20x miss.
_UNFIXED_ATTEMPTS = 1


def _tile_wide(t: lucid.Tensor) -> lucid.Tensor:
    return lucid.tile(t, (1, 2, 1))


def _tile_twin(t: lucid.Tensor) -> lucid.Tensor:
    return lucid.tile(t, (2, 1))


@needs_metal
@pytest.mark.parametrize("where", sorted(_REPEAT_TWINS))
@pytest.mark.parametrize("backward", [False, True])
def test_repeat_uses_its_twins_split_axis(where: str, backward: bool) -> None:
    dim, shape = _REPEAT_TWINS[where]

    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return lucid.repeat(t, 2, dim=dim).reshape(-1).sum()

    pair = RankTwin("repeat", Side(fn, (_metal(_W4),)), Side(fn, (_metal(shape),)))
    op, label = (
        ("ReductionSum", "repeat_vjp_sum") if backward else ("Broadcast", "repeat_spread")
    )
    twin = operation_shapes(emitted(pair.twin, backward=backward), op, label=label)
    assert operation_shapes(emitted(pair.wide, backward=backward), op, label=label) == twin
    assert operation_shapes(emitted(pair.wide, "-1", backward), op, label=label) != twin


@needs_metal
def test_tile_gradient_folds_only_the_repeated_axis() -> None:
    x = _metal(_TILE_WIDE)
    wide = Side(lambda t: _tile_wide(t).reshape(-1).sum(), (x,))
    # The incoming gradient has two copies of each eight-by-eight row.
    gradient = _metal((2048, 2, 8, 8))
    assert gradient.numel() == 2 * x.numel()
    twin = Side(lambda g: g.sum(dim=1, keepdim=True), (gradient,))
    expected = operation_shapes(emitted(twin), "ReductionSum", label="sum")
    assert (
        operation_shapes(emitted(wide, backward=True), "ReductionSum", label="tile_vjp_sum")
        == expected
    )
    assert (
        operation_shapes(emitted(wide, "-1", True), "ReductionSum", label="tile_vjp_sum")
        != expected
    )


@needs_metal
def test_repeat_below_threshold_preserves_the_native_graph() -> None:
    side = Side(_repeat_forward(1), (_metal((8, 8, 512, 64)),))
    default = operation_shapes(emitted(side), "Broadcast")
    assert default == operation_shapes(emitted(side, "-1"), "Broadcast")
    assert default != operation_shapes(emitted(side, "0"), "Broadcast")


@needs_metal
def test_inspection_reports_an_invalid_gradient_request() -> None:
    x = _metal((2, 3))
    graph, feeds = _trace(lambda t: t.sum(), x)
    loss = graph.ops[-1].outputs[0].id
    with pytest.raises(RuntimeError, match="param_ids must be non-empty"):
        _C_engine.compile._inspect_emitted_operations(graph, feeds, loss, [])
    assert _C_engine.compile._inspect_emitted_operations(graph, feeds)
