"""A compiled graph keeps rank-5+ data-movement ops out of MPSGraph's slice loop (LCD-928).

MPSGraph runs a kernel that reads or writes a rank >= 5 tensor as a loop over the dims above the
last four.  ``RankCanon`` (``lucid/_C/compile/RankCanon.h``) rewrites the data-movement ops that
name axes through their attribute, merging every axis the op does not name:

* ``masked_fill`` is elementwise (mask, constant, value) and joins ``ElementwiseRule``;
* ``flip`` / ``roll`` / ``cumsum`` family / ``repeat`` / ``concatenate`` re-point ``dims`` /
  ``axes`` / ``axis`` / ``dim`` at the merged axes;
* ``pad`` / ``tile`` re-point their per-axis ``pads`` / ``reps``; ``tril`` / ``triu`` keep their
  last two axes; ``broadcast_to`` coalesces the way a binary op's operand does;
* ``split_at`` / ``split`` (every slice) cut the merged tensor and view each piece back; ``stack``
  reads its inputs as ``(rows, rest)`` and writes ``(rows, count, rest)``.

An elementwise producer takes the layout of the rewritten op that reads it, so the multiply
in ``cat([x * 1.0, x * 2.0])`` runs at the shape the concatenation wants.

Three guards, as for the earlier rules:

* the pass's IR, read through ``canonicalize_rank`` with no GPU: which nodes are rewritten, at
  which shapes, with which attribute, and what is left alone (below the gate, more named axes than
  rank 4 can hold, a symbolic batch on a named axis);
* a value table on Metal: the forward and the ``make_step`` gradients of the rewritten graph equal
  eager's (and the unrewritten graph's, exactly, for ops that only move values) at float32,
  float16 and bfloat16, over rank 5 and rank 6 and with NaN and inf in the data;
* a coarse timing bound against the rank-4 twin of each case, and the permute gate that now reads
  the larger of the input's and the output's lead.
"""

import os
import time
from collections.abc import Callable, Iterator
from unittest import mock

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.compile import _tracing, make_step
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, to_metal

ENV = "LUCID_COMPILE_RANK_CANON_MIN_LEAD"

needs_metal = pytest.mark.skipif(not metal_available(), reason="compile needs Metal")

# A fallback to eager would make every compiled == eager row below vacuous.
pytestmark = pytest.mark.filterwarnings(
    "error::lucid.compile._core.fallback.CompileFallbackWarning"
)

Shape = tuple[int, ...]
Op = Callable[[lucid.Tensor], lucid.Tensor]


@pytest.fixture(autouse=True)
def _default_gate() -> Iterator[None]:
    """Every test starts from the default lead threshold, whatever the shell exports."""
    with mock.patch.dict(os.environ):
        os.environ.pop(ENV, None)
        yield


def _filled(shape: Shape, phase: float = 0.0) -> lucid.Tensor:
    n = 1
    for s in shape:
        n *= s
    return (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 0.37 + phase) + 1.5).reshape(*shape)


def _trace(
    fn: Callable[..., lucid.Tensor], *args: lucid.Tensor
) -> tuple[_C_engine.compile.TraceGraph, dict[int, object]]:
    with lucid.no_grad(), _tracing() as tracer:
        fn(*args)
    return tracer.graph, tracer.external_feeds


def _canon(
    fn: Callable[..., lucid.Tensor],
    *args: lucid.Tensor,
    dynamic: bool = False,
    gate: str | None = "0",
) -> list[_C_engine.compile.OpNode]:
    graph, feeds = _trace(fn, *args)
    env = {} if gate is None else {ENV: gate}
    with mock.patch.dict(os.environ, env):
        return list(_C_engine.compile.canonicalize_rank(graph, feeds, dynamic).ops)


def _named(ops: list[_C_engine.compile.OpNode], name: str) -> list[_C_engine.compile.OpNode]:
    return [n for n in ops if n.name == name]


def _only(ops: list[_C_engine.compile.OpNode], name: str) -> _C_engine.compile.OpNode:
    (node,) = _named(ops, name)
    return node


def _shape(node: _C_engine.compile.OpNode) -> Shape:
    return tuple(node.outputs[0].shape)


def _ints(node: _C_engine.compile.OpNode, key: str) -> list[int]:
    return list(node.attrs[key])


R5: Shape = (3, 4, 2, 3, 2)
R6: Shape = (3, 2, 2, 3, 2, 2)

# ── the IR ────────────────────────────────────────────────────────────────────────────


def test_masked_fill_runs_flat_between_views() -> None:
    x = _filled(R5)
    ops = _canon(lambda t: t.masked_fill(t > 2.0, 0.5), x)
    assert _shape(_only(ops, "masked_fill")) == (144,)
    assert _only(ops, "masked_fill").attrs["fill_value"] == 0.5
    assert _shape(ops[-1]) == R5 and ops[-1].name == "reshape"


def test_flip_merges_axes_flipped_alike_and_keeps_the_others_apart() -> None:
    ops = _canon(lambda t: lucid.flip(t, [1, 2]), _filled(R5))
    node = _only(ops, "flip")
    # axes 1 and 2 are both flipped and merge; 0 and (3, 4) are left in a run each
    assert _shape(node) == (3, 8, 6)
    assert _ints(node, "dims") == [1]
    assert _shape(ops[-1]) == R5


def test_flip_over_negative_dims_names_the_same_runs() -> None:
    ops = _canon(lambda t: lucid.flip(t, [-1, -2]), _filled(R5))
    node = _only(ops, "flip")
    assert _shape(node) == (24, 6)
    assert _ints(node, "dims") == [1]


def test_flip_of_every_other_axis_cannot_reach_rank_4() -> None:
    ops = _canon(lambda t: lucid.flip(t, [0, 2, 4]), _filled(R5))
    assert _shape(_only(ops, "flip")) == R5


def test_roll_keeps_each_rolled_axis_in_a_run_of_its_own() -> None:
    ops = _canon(lambda t: lucid.roll(t, 1, 2), _filled(R5))
    node = _only(ops, "roll")
    assert _shape(node) == (12, 2, 6)
    assert _ints(node, "axes") == [1] and _ints(node, "shifts") == [1]


def test_roll_by_a_whole_turn_moves_nothing_and_is_left_alone() -> None:
    ops = _canon(lambda t: lucid.roll(t, 2, 2), _filled(R5))
    assert _shape(_only(ops, "roll")) == R5


def test_roll_adds_the_shifts_of_a_repeated_axis() -> None:
    ops = _canon(lambda t: lucid.roll(t, [1, 2], [2, 2]), _filled(R5))
    node = _only(ops, "roll")
    # 3 mod 2 == 1
    assert _ints(node, "axes") == [1] and _ints(node, "shifts") == [1]


@pytest.mark.parametrize("name", ["cumsum", "cumprod"])
def test_a_scan_runs_over_the_axis_between_two_merged_runs(name: str) -> None:
    ops = _canon(lambda t: getattr(lucid, name)(t, dim=2), _filled(R5))
    node = _only(ops, name)
    assert _shape(node) == (12, 2, 6)
    assert node.attrs["axis"] == 1


def test_a_scan_over_the_first_axis_has_no_run_before_it() -> None:
    ops = _canon(lambda t: lucid.cumsum(t, dim=0), _filled(R5))
    node = _only(ops, "cumsum")
    assert _shape(node) == (3, 48) and node.attrs["axis"] == 0


def test_repeat_re_points_its_axis() -> None:
    ops = _canon(lambda t: lucid.repeat(t, 2, dim=2), _filled(R5))
    node = _only(ops, "repeat")
    assert _shape(node) == (12, 4, 6)
    assert node.attrs["axis"] == 1 and node.attrs["repeats"] == 2


def test_concatenate_merges_around_dim_in_every_input() -> None:
    a, b = _filled(R5), _filled((3, 4, 5, 3, 2), 1.0)
    ops = _canon(lambda x, y: lucid.cat([x, y], dim=2), a, b)
    node = _only(ops, "concatenate")
    assert _shape(node) == (12, 7, 6) and node.attrs["dim"] == 1
    assert [tuple(m.shape) for m in (node.outputs[0],)] == [(12, 7, 6)]


def test_pad_keeps_the_padded_axes_and_re_points_pads() -> None:
    ops = _canon(lambda t: F.pad(t, (1, 2, 3, 4)), _filled(R5))
    node = _only(ops, "pad")
    assert _shape(node) == (24, 3 + 7, 2 + 3)
    assert _ints(node, "pads") == [0, 0, 3, 4, 1, 2]


def test_pad_of_three_separated_axes_cannot_reach_rank_4() -> None:
    ops = _canon(lambda t: F.pad(t, (1, 1, 0, 0, 2, 2, 0, 0, 1, 1)), _filled(R5))
    assert _shape(_only(ops, "pad")) == (5, 4, 6, 3, 4)


def test_tile_re_points_reps() -> None:
    ops = _canon(lambda t: lucid.tile(t, (1, 1, 2, 1, 1)), _filled(R5))
    node = _only(ops, "tile")
    assert _shape(node) == (12, 4, 6)
    assert _ints(node, "reps") == [1, 2, 1]


@pytest.mark.parametrize("name", ["tril", "triu"])
def test_a_triangle_mask_keeps_the_last_two_axes(name: str) -> None:
    ops = _canon(lambda t: getattr(lucid, name)(t, 1), _filled(R5))
    node = _only(ops, name)
    assert _shape(node) == (24, 3, 2)
    assert node.attrs["k"] == 1


def test_broadcast_to_coalesces_like_a_binary_operand() -> None:
    small = _filled((1, 4, 1, 3, 2))
    ops = _canon(lambda t: lucid.broadcast_to(t, R5), small)
    node = _only(ops, "broadcast_to")
    assert _shape(node) == (3, 4, 2, 6)
    view = _named(ops, "reshape")[0]
    assert tuple(view.outputs[0].shape) == (1, 4, 1, 6)


def test_a_slice_cuts_the_merged_tensor_and_views_each_piece_back() -> None:
    ops = _canon(lambda t: (t * 1.0)[:, :, 1:], _filled(R5))
    node = _only(ops, "split_at")
    # axis 2 is the middle run: (3*4, 2, 3*2); the pieces are (12, 1, 6) and (12, 1, 6)
    assert node.attrs["axis"] == 1
    assert [tuple(o.shape) for o in node.outputs] == [(12, 1, 6), (12, 1, 6)]
    assert [tuple(o.shape) for n in ops if n.name == "reshape" for o in n.outputs][-2:] == [
        (3, 4, 1, 3, 2),
        (3, 4, 1, 3, 2),
    ]


def test_a_split_re_points_its_axis() -> None:
    ops = _canon(lambda t: lucid.split(t * 1.0, 3, dim=-1)[0], _filled((3, 4, 2, 3, 6)))
    node = _only(ops, "split")
    assert node.attrs["axis"] == 1 and tuple(node.outputs[0].shape) == (72, 3)


def test_a_stack_reads_rows_and_the_rest() -> None:
    ops = _canon(lambda t: lucid.stack([t * 1.0, t * 2.0], dim=2), _filled(R5))
    node = _only(ops, "stack")
    # rows = 3 * 4, rest = 2 * 3 * 2 (the axes from ``dim`` on)
    assert _shape(node) == (12, 2, 12) and node.attrs["axis"] == 1


def test_an_elementwise_producer_takes_the_layout_its_reader_asks_for() -> None:
    ops = _canon(lambda t: lucid.cat([t * 1.0, t * 2.0], dim=2), _filled(R5))
    cat = _only(ops, "concatenate")
    assert _shape(cat) == (12, 4, 6)
    # both multiplies run at the (12, 2, 6) the concatenation reads, with no view in between
    assert [_shape(n) for n in _named(ops, "mul")] == [(12, 2, 6), (12, 2, 6)]


def test_every_original_id_keeps_its_shape_behind_the_views() -> None:
    graph, feeds = _trace(lambda t: lucid.flip(t, [1, 2]) + 1.0, _filled(R5))
    before = {m.id: tuple(m.shape) for n in graph.ops for m in n.outputs}
    after_graph = _C_engine.compile.canonicalize_rank(graph, feeds, False)
    after = {m.id: tuple(m.shape) for n in after_graph.ops for m in n.outputs}
    assert all(after[i] == s for i, s in before.items())


def test_a_movement_op_below_the_lead_gate_is_left_alone() -> None:
    # lead 3 < 64 at the default gate
    ops = _canon(lambda t: lucid.flip(t, [1, 2]), _filled(R5), gate=None)
    assert _shape(_only(ops, "flip")) == R5


def test_a_symbolic_batch_axis_cannot_be_the_named_axis() -> None:
    x = _filled(R5)
    scan = _canon(lambda t: lucid.cumsum(t, dim=0), x, dynamic=True)
    assert _shape(_only(scan, "cumsum")) == R5
    flip = _canon(lambda t: lucid.flip(t, [2, 3]), x, dynamic=True)
    node = _only(flip, "flip")
    # axis 0 stays a run of its own
    assert _shape(node) == (3, 4, 6, 2) and _ints(node, "dims") == [2]


def test_a_consumer_that_reads_wide_keeps_its_producer_unrewritten() -> None:
    # ``topk`` has no rule and reads the flipped tensor at rank 5: the guard drops the flip's
    # rewrite, since the view would be materialised for it.
    ops = _canon(lambda t: lucid.topk(lucid.flip(t, [1, 2]), 2, dim=-1)[0], _filled(R5))
    assert _shape(_only(ops, "flip")) == R5


# ── permute: the gate reads the larger of the input's and the output's lead ───────────


def _space_to_depth(x: lucid.Tensor, patch: int = 2) -> lucid.Tensor:
    b, c, h, w = x.shape
    v = x.reshape(b, c, h // patch, patch, w // patch, patch).permute(0, 3, 5, 1, 2, 4)
    return v.reshape(b, patch * patch * c, h // patch, w // patch)


def test_a_space_to_depth_splits_when_only_its_input_view_has_a_large_lead() -> None:
    # Output lead 12 * ... is under the costly gate (256); the input view (6, 16, 4, 2, 4, 2)
    # has 96 slices at gate 64 (costly 256) -> use gate 16 (costly 64): 12 < 64, 96 >= 64.
    x = _filled((6, 16, 8, 8))
    on = _canon(lambda t: _space_to_depth(t) * 1.0, x, gate="16")
    permutes = _named(on, "permute")
    assert len(permutes) > 1 and all(len(_shape(p)) <= 4 for p in permutes)


# ── values ────────────────────────────────────────────────────────────────────────────

_DTYPES = [lucid.float32, lucid.float16, lucid.bfloat16]
_TOL = {lucid.float32: 3e-4, lucid.float16: 4e-3, lucid.bfloat16: 3e-2}


def _values(t: lucid.Tensor) -> list[float]:
    return t.to(lucid.float32).reshape(-1).tolist()


def _alike(a: float, b: float) -> bool:
    return a == b or (a != a and b != b)


def _same(got: lucid.Tensor, want: lucid.Tensor, why: str) -> None:
    g, w = _values(got), _values(want)
    assert len(g) == len(w), why
    bad = [i for i, (a, b) in enumerate(zip(g, w, strict=True)) if not _alike(a, b)]
    assert not bad, f"{why}: {len(bad)} values differ, first at {bad[0]}: {g[bad[0]]} != {w[bad[0]]}"


def _close(got: lucid.Tensor, want: lucid.Tensor, tol: float, why: str) -> None:
    g, w = _values(got), _values(want)
    assert len(g) == len(w), why
    for i, (a, b) in enumerate(zip(g, w, strict=True)):
        assert _alike(a, b) or a == pytest.approx(b, rel=tol, abs=tol), f"{why}[{i}]: {a} != {b}"


def _with_nonfinite(x: lucid.Tensor) -> lucid.Tensor:
    """``x`` with NaN, +inf and -inf spread through it (a row each of a few)."""
    n = 1
    for s in x.shape:
        n *= s
    idx = lucid.arange(n, dtype=lucid.float32).reshape(*x.shape)
    nan = lucid.full_like(x, float("nan"))
    inf = lucid.full_like(x, float("inf"))
    out = lucid.where(idx % 11.0 == 3.0, nan, x)
    out = lucid.where(idx % 13.0 == 5.0, inf, out)
    return lucid.where(idx % 17.0 == 7.0, -inf, out)


def _on_metal(t: lucid.Tensor, dtype: lucid.dtype) -> lucid.Tensor:
    return t.to(COMPILE_DEVICE).to(dtype)


def _forward(fn: Callable[..., lucid.Tensor], args: tuple[lucid.Tensor, ...], gate: str) -> lucid.Tensor:
    with mock.patch.dict(os.environ, {ENV: gate}):
        _C_engine.compile.session_cache_clear()
        return lucid.compile(fn)(*args)


def _mask(t: lucid.Tensor) -> lucid.Tensor:
    return t > 1.6


#: ``(op, exact)``: ``exact`` is a pure data move (bit-identical to eager and to the unrewritten
#: graph); a scan sums, so it is held to the dtype's rounding.
_OPS: dict[str, tuple[Op, bool]] = {
    "masked_fill": (lambda t: t.masked_fill(_mask(t), -2.5), True),
    "masked_fill_nan": (lambda t: t.masked_fill(lucid.isnan(t), 0.0), True),
    "flip_inner": (lambda t: lucid.flip(t, [2, 3]), True),
    "flip_outer": (lambda t: lucid.flip(t, [0, 1]), True),
    "flip_last": (lambda t: lucid.flip(t, [-1]), True),
    "roll_one": (lambda t: lucid.roll(t, 1, 2), True),
    "roll_two": (lambda t: lucid.roll(t, [1, 2], [1, -1]), True),
    "cumsum_mid": (lambda t: lucid.cumsum(t, dim=2), False),
    "cumsum_last": (lambda t: lucid.cumsum(t, dim=-1), False),
    "cumsum_first": (lambda t: lucid.cumsum(t, dim=0), False),
    "repeat_mid": (lambda t: lucid.repeat(t, 2, dim=2), True),
    "repeat_last": (lambda t: lucid.repeat(t, 3, dim=-1), True),
    "cat_mid": (lambda t: lucid.cat([t, t * 2.0], dim=2), True),
    "cat_first": (lambda t: lucid.cat([t, t * 2.0, t], dim=0), True),
    "cat_last": (lambda t: lucid.cat([t, t * 2.0], dim=-1), True),
    "pad_last_two": (lambda t: F.pad(t, (1, 2, 3, 0)), True),
    "pad_value": (lambda t: F.pad(t, (1, 1, 0, 0, 2, 1), value=7.0), True),
    "tile_mid": (lambda t: lucid.tile(t, (1, 1, 2, 1, 1)), True),
    "tile_two": (lambda t: lucid.tile(t, (1, 2, 1, 1, 3)), True),
    "tril": (lambda t: lucid.tril(t), True),
    "triu_k": (lambda t: lucid.triu(t, 1), True),
    "tril_neg": (lambda t: lucid.tril(t, -1), True),
    "expand": (lambda t: t[:, :1].expand(*t.shape) + t, True),
    "slice_mid": (lambda t: t[:, :, 1:3] * 1.0, True),
    "slice_first": (lambda t: t[1:] * 1.0, True),
    "split_last": (lambda t: lucid.split(t, 1, dim=-1)[1] * 1.0, True),
    "stack_mid": (lambda t: lucid.stack([t, t * 2.0, t], dim=2), True),
    "stack_first": (lambda t: lucid.stack([t, t * 2.0], dim=0), True),
    "stack_last": (lambda t: lucid.stack([t, t * 2.0], dim=-1), True),
}

_SHAPES: dict[str, Shape] = {"rank5": R5, "rank6": R6}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("shape_name", sorted(_SHAPES))
@pytest.mark.parametrize("name", sorted(_OPS))
def test_a_movement_op_matches_eager_and_the_unrewritten_graph(
    name: str, shape_name: str, dtype: lucid.dtype
) -> None:
    op, exact = _OPS[name]
    shape = _SHAPES[shape_name]
    base = _filled(shape, 0.3)
    x = _on_metal(base if name == "masked_fill_nan" else _with_nonfinite(base), dtype)
    # An elementwise tail keeps the moved tensor consumed inside the graph.
    fn: Op = lambda t: op(t * 1.0) + 0.0
    want = fn(x)
    on = _forward(fn, (x,), "0")
    if exact:
        _same(on, want, "rewritten vs eager")
        _same(on, _forward(fn, (x,), "-1"), "rewritten vs unrewritten")
    else:
        # A scan over non-finite data carries inf - inf: compare on the finite rows only.
        finite = _on_metal(base, dtype)
        want, on = fn(finite), _forward(fn, (finite,), "0")
        _close(on, want, _TOL[dtype] * 8, "rewritten vs eager")


class _Moved(nn.Module):
    """``loss`` over ``op(x * s) * w``: the gradient of ``x`` runs the op's backward."""

    def __init__(self, op: Op, shape: Shape, dtype: lucid.dtype) -> None:
        super().__init__()
        self.op = op
        self.x = nn.Parameter(_on_metal(_filled(shape, 0.1), dtype))
        out = tuple(op(_filled(shape, 0.1)).shape)
        self.w = nn.Parameter(_on_metal(_filled(out, 0.7), dtype))

    def forward(self, s: lucid.Tensor) -> lucid.Tensor:
        return self.op(self.x * s) * self.w


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    return (y * y).sum()


def _step(
    make: Callable[[], nn.Module], arg: lucid.Tensor, gate: str
) -> tuple[lucid.Tensor, dict[str, lucid.Tensor]]:
    model = to_metal(make())
    with mock.patch.dict(os.environ, {ENV: gate}):
        _C_engine.compile.session_cache_clear()
        loss = make_step(model, _loss)(arg)
        loss.backward()
    return loss, {n: p.grad for n, p in model.named_parameters() if p.grad is not None}


_STEP_OPS = [
    "masked_fill",
    "flip_inner",
    "roll_two",
    "cumsum_mid",
    "cumsum_first",
    "repeat_mid",
    "cat_mid",
    "cat_first",
    "pad_last_two",
    "tile_mid",
    "tril",
    "expand",
    "slice_mid",
    "stack_mid",
]


@needs_metal
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("shape_name", sorted(_SHAPES))
@pytest.mark.parametrize("name", _STEP_OPS)
def test_a_movement_step_matches_eager_and_the_unrewritten_graph(
    name: str, shape_name: str, dtype: lucid.dtype
) -> None:
    op, exact = _OPS[name]
    shape = _SHAPES[shape_name]
    arg = _on_metal(lucid.ones(1), dtype)
    eager = to_metal(_Moved(op, shape, dtype))
    want = _loss(eager(arg))
    want.backward()
    on, on_grads = _step(lambda: _Moved(op, shape, dtype), arg, "0")
    off, off_grads = _step(lambda: _Moved(op, shape, dtype), arg, "-1")
    tol = _TOL[dtype]
    _close(on, want, tol * 4, "loss vs eager")
    if exact:
        _same(on, off, "loss, rewritten vs unrewritten")
    assert on_grads.keys() == off_grads.keys() and on_grads
    for pname, grad in on_grads.items():
        eager_grad = dict(eager.named_parameters())[pname].grad
        assert eager_grad is not None
        scale = max(max(abs(v) for v in _values(eager_grad)), 1.0)
        _close(grad / scale, eager_grad / scale, tol * 8, f"grad {pname} vs eager")
        if exact:
            _close(grad / scale, off_grads[pname] / scale, tol * 8, f"grad {pname} vs off")


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_symbolic_batch_survives_the_movement_rewrites() -> None:
    def fn(t: lucid.Tensor) -> lucid.Tensor:
        y = lucid.cumsum(lucid.flip(t * 1.5, [2, 3]), dim=2)
        y = lucid.cat([y, y * 2.0], dim=2)
        return F.pad(y, (1, 1, 0, 0)) + 1.0

    with mock.patch.dict(os.environ, {ENV: "0"}):
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(fn, dynamic=True)
        for batch in (3, 3, 3):
            x = _on_metal(_filled((batch, 4, 2, 3, 2), 0.2 * batch), lucid.float32)
            _close(compiled(x), fn(x), 3e-4, f"batch {batch}")


# ── time ──────────────────────────────────────────────────────────────────────────────


def _forward_ms(fn: Callable[..., lucid.Tensor], *args: lucid.Tensor) -> float:
    cm = lucid.compile(fn)
    for _ in range(3):
        cm(*args)
    best = float("inf")
    for _ in range(15):
        t0 = time.perf_counter()
        float(cm(*args).sum().item())
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _metal(shape: Shape, phase: float = 0.0) -> lucid.Tensor:
    return _filled(shape, phase).to(COMPILE_DEVICE)


#: The rank-5 case and the rank-4 twin of the same elements and work (the axis the op
#: names moves down by one with the merged leading axes).  16.8M elements.
_S5: Shape = (2048, 4, 8, 16, 16)
_S4: Shape = (8192, 8, 16, 16)


def _tot(y: lucid.Tensor) -> lucid.Tensor:
    return y.reshape(-1).sum()


_TWINS: dict[str, tuple[Op, Op]] = {
    "masked_fill": (
        lambda t: _tot((t * 1.0).masked_fill(t > 1.5, 0.0)),
        lambda t: _tot((t * 1.0).masked_fill(t > 1.5, 0.0)),
    ),
    "flip": (lambda t: _tot(lucid.flip(t * 1.0, (2,))), lambda t: _tot(lucid.flip(t * 1.0, (1,)))),
    "tril": (lambda t: _tot(lucid.tril(t * 1.0)), lambda t: _tot(lucid.tril(t * 1.0))),
    "cat": (
        lambda t: _tot(lucid.cat([t * 1.0, t * 2.0], dim=2)),
        lambda t: _tot(lucid.cat([t * 1.0, t * 2.0], dim=1)),
    ),
    "pad": (
        lambda t: _tot(F.pad(t * 1.0, (1, 1, 1, 1))),
        lambda t: _tot(F.pad(t * 1.0, (1, 1, 1, 1))),
    ),
    "tile": (
        lambda t: _tot(lucid.tile(t * 1.0, (1, 1, 2, 1, 1))),
        lambda t: _tot(lucid.tile(t * 1.0, (1, 2, 1, 1))),
    ),
    "repeat": (
        lambda t: _tot(lucid.repeat(t * 1.0, 2, dim=-1)),
        lambda t: _tot(lucid.repeat(t * 1.0, 2, dim=-1)),
    ),
    "cumsum": (
        lambda t: _tot(lucid.cumsum(t * 1.0, dim=2)),
        lambda t: _tot(lucid.cumsum(t * 1.0, dim=1)),
    ),
    "expand": (
        lambda t: _tot(t + t[:1, :, :1].expand(*_S5)),
        lambda t: _tot(t + t[:1, :1].expand(*_S4)),
    ),
    "slice": (lambda t: _tot((t * 1.0)[:, :, 2:5]), lambda t: _tot((t * 1.0)[:, 2:5])),
    "stack": (
        lambda t: _tot(lucid.stack([t * 1.0, t * 2.0], dim=2)),
        lambda t: _tot(lucid.stack([t * 1.0, t * 2.0], dim=1)),
    ),
}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", sorted(_TWINS))
def test_a_rank_5_movement_op_is_not_slower_than_its_rank_4_twin(name: str) -> None:
    """Held to 1.5x the twin; the unfixed graphs ran 2x to 15x it."""
    wide_fn, twin_fn = _TWINS[name]
    _C_engine.compile.session_cache_clear()
    wide, twin = _forward_ms(wide_fn, _metal(_S5)), _forward_ms(twin_fn, _metal(_S4))
    assert wide <= 1.5 * twin + 0.3, f"{name}: rank-5 {wide:.2f} ms vs twin {twin:.2f} ms"


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_space_to_depth_of_a_feed_is_not_slower_than_its_rank_4_twin() -> None:
    """The permute gate reads the input view's lead too (25.8 ms against 2.35 unfixed)."""

    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return _space_to_depth(t).sum()

    _C_engine.compile.session_cache_clear()
    on = _forward_ms(fn, _metal((96, 64, 32, 64)))
    with mock.patch.dict(os.environ, {ENV: "0"}):
        _C_engine.compile.session_cache_clear()
        always = _forward_ms(fn, _metal((96, 64, 32, 64)))
    assert on <= 1.5 * always + 0.3, f"gate default {on:.2f} ms vs always-on {always:.2f} ms"
