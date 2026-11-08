"""A compiled graph keeps rank-5+ permute, softmax and matmul batch axes out of the slice loop (LCD-920b).

MPSGraph runs a compute op that reads or writes a tensor of rank >= 5 as a loop over the dims
above the last four, one dispatch per slice.  ``RankCanon`` (``lucid/_C/compile/RankCanon.h``)
carries three rules for the ops an elementwise rewrite cannot reach, because they name axes:

* ``PermuteRule``: drop size-1 axes, merge axes that stay adjacent and in order, and when the
  rank is still above 4 split the permute into rank <= 4 block moves (at the larger "costly"
  lead, each step being another pass over the data);
* ``SoftmaxRule``: the axes before ``dim`` merge into one, and so do the axes after it;
* ``MatmulRule``: adjacent batch axes that both operands treat alike merge.

Three guards, as for the elementwise rule:

* the pass's IR, read through ``canonicalize_rank`` with no GPU: which nodes are rewritten, at
  which shapes, with which axis attribute, and what is left alone (below the gate, a symbolic
  batch axis, attention weights a matmul reads);
* a value table on Metal: forward and ``make_step`` gradients of the rewritten graph equal
  eager's, and equal the *unrewritten* compiled graph's to the last bit, at float32, float16 and
  bfloat16;
* a coarse timing bound against the rank-4 twin of each case.
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
    gate: str | None = None,
) -> list[_C_engine.compile.OpNode]:
    graph, feeds = _trace(fn, *args)
    env = {} if gate is None else {ENV: gate}
    with mock.patch.dict(os.environ, env):
        return list(_C_engine.compile.canonicalize_rank(graph, feeds, dynamic).ops)


def _named(ops: list[_C_engine.compile.OpNode], name: str) -> list[_C_engine.compile.OpNode]:
    return [n for n in ops if n.name == name]


def _widest(ops: list[_C_engine.compile.OpNode], name: str) -> int:
    return max(len(n.outputs[0].shape) for n in _named(ops, name))


def _perm(node: _C_engine.compile.OpNode) -> list[int]:
    return list(node.attrs["permutation"])


# ── permute: the IR ───────────────────────────────────────────────────────────────────


def _numel(shape: Shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _permuted(shape: Shape, perm: tuple[int, ...]) -> Callable[[lucid.Tensor], lucid.Tensor]:
    """``t`` (a flat feed) viewed at ``shape`` and permuted: a feed is permuted in place by
    MPSGraph at full speed, so the rule only rewrites the permute of a computed tensor."""
    return lambda t: t.reshape(*shape).permute(*perm)


def _flat(shape: Shape) -> lucid.Tensor:
    return _filled((_numel(shape),))


def _window_partition(x: lucid.Tensor) -> lucid.Tensor:
    b, h, w, c = x.shape
    ws = 7
    t = x.reshape(b, h // ws, ws, w // ws, ws, c)
    return t.permute(0, 1, 3, 2, 4, 5)


def _window_reverse(windows: lucid.Tensor) -> lucid.Tensor:
    # (B * nH * nW, ws, ws, C) -> (B, nH, nW, ws, ws, C) -> (B, nH, ws, nW, ws, C)
    t = windows.reshape(8, 8, 8, 7, 7, 4)
    return t.permute(0, 1, 3, 2, 4, 5)


def test_a_window_partition_permute_merges_to_one_rank_4_permute() -> None:
    ops = _canon(_window_partition, _filled((8, 56, 56, 4)))
    (perm,) = _named(ops, "permute")
    assert tuple(perm.outputs[0].shape) == (64, 8, 7, 28)
    assert _perm(perm) == [0, 2, 1, 3]
    # The original result keeps its id and rank-6 shape behind a view.
    assert tuple(ops[-1].outputs[0].shape) == (8, 8, 8, 7, 7, 4)
    assert ops[-1].name == "reshape"


def test_a_window_reverse_permute_merges_to_one_rank_4_permute() -> None:
    ops = _canon(_window_reverse, _filled((512, 7, 7, 4)))
    (perm,) = _named(ops, "permute")
    assert _widest(ops, "permute") == 4
    assert tuple(perm.outputs[0].shape) == (64, 7, 8, 28)
    assert _perm(perm) == [0, 2, 1, 3]


def test_a_space_to_depth_permute_splits_into_two_rank_4_permutes() -> None:
    # (bc, H, p, W, q) -> (bc, p, q, H, W): no two axes stay adjacent, so one permute stays
    # rank 5.  Two block moves reach it.
    shape = (512, 4, 2, 4, 2)
    ops = _canon(_permuted(shape, (0, 2, 4, 1, 3)), _flat(shape))
    steps = _named(ops, "permute")
    assert len(steps) == 2
    assert _widest(ops, "permute") <= 4
    # The steps are chained: the second reads the first's result, through a view.
    ids = {n.outputs[0].id: n for n in ops}
    assert ids[ops[ops.index(steps[1]) - 1].inputs[0]] is steps[0]
    # ... and the first reads the feed through one view, not a view of the recorded reshape.
    assert ops[0].inputs[0] == ops[ops.index(steps[0]) - 1].inputs[0]
    assert tuple(ops[-1].outputs[0].shape) == (512, 2, 2, 4, 4)


def test_the_costly_lead_scales_with_the_override() -> None:
    # The costly gate is 4x the free one: an override of 25 splits a lead-100 permute, 26 does not.
    shape = (100, 4, 2, 4, 2)
    for gate, splits in [("25", True), ("26", False), ("-1", False)]:
        ops = _canon(_permuted(shape, (0, 2, 4, 1, 3)), _flat(shape), gate=gate)
        assert (len(_named(ops, "permute")) == 2) is splits, gate


def test_a_split_needs_the_costly_lead() -> None:
    # lead 100: above the free gate, below the costly one — the permute is left as recorded.
    shape = (100, 4, 2, 4, 2)
    ops = _canon(_permuted(shape, (0, 2, 4, 1, 3)), _flat(shape))
    assert [n.name for n in ops] == ["reshape", "permute"]
    assert _widest(ops, "permute") == 5
    ops = _canon(_permuted(shape, (0, 2, 4, 1, 3)), _flat(shape), gate="0")
    assert len(_named(ops, "permute")) == 2


def test_a_reshape_reading_a_rewritten_permute_reads_the_canonical_result() -> None:
    shape = (64, 4, 2, 4, 2)
    graph, feeds = _trace(lambda t: _permuted(shape, (0, 1, 2, 4, 3))(t).reshape(-1), _flat(shape))
    ops = list(_C_engine.compile.canonicalize_rank(graph, feeds, False).ops)
    (perm,) = _named(ops, "permute")
    last = ops[-1]
    # The recorded reshape-to-flat reads the rank-4 permute result directly, so MPSGraph sees
    # permute -> reshape, not permute -> reshape(rank 5) -> reshape.
    assert last.name == "reshape" and last.inputs[0] == perm.outputs[0].id


def test_a_permute_of_a_feed_is_left_alone() -> None:
    shape = (512, 4, 2, 4, 2)
    ops = _canon(lambda t: t.permute(0, 2, 4, 1, 3), _filled(shape), gate="0")
    assert [n.name for n in ops] == ["permute"]


def test_a_permute_merging_to_one_permute_does_not_need_the_costly_lead() -> None:
    ops = _canon(_window_partition, _filled((2, 14, 14, 4)), gate="0")
    assert _widest(ops, "permute") == 4
    # lead 4 < 64: untouched by default.
    ops = _canon(_window_partition, _filled((2, 14, 14, 4)))
    assert _widest(ops, "permute") == 6


def test_size_1_axes_do_not_count() -> None:
    # (64, 1, 4, 2, 1, 3) with the unit axes moved around is a rank-4 permute of (64, 4, 2, 3).
    shape = (64, 1, 4, 2, 1, 3)
    ops = _canon(_permuted(shape, (4, 2, 0, 5, 1, 3)), _flat(shape), gate="0")
    (perm,) = _named(ops, "permute")
    assert tuple(perm.outputs[0].shape) == (4, 64, 3, 2)
    assert _perm(perm) == [1, 0, 3, 2]


def test_an_identity_after_merging_is_a_rank_1_permute() -> None:
    shape = (64, 2, 2, 2, 2)
    ops = _canon(_permuted(shape, (0, 1, 2, 3, 4)), _flat(shape))
    (perm,) = _named(ops, "permute")
    assert tuple(perm.outputs[0].shape) == (64 * 16,)
    assert _perm(perm) == [0]


def test_a_permute_out_of_reach_of_three_block_moves_is_left_alone() -> None:
    # Rank 7 with nothing to merge: past the search's cap.
    shape = (2, 2, 2, 2, 2, 2, 3)
    ops = _canon(_permuted(shape, (6, 5, 4, 3, 2, 1, 0)), _flat(shape), gate="0")
    assert [n.name for n in ops] == ["reshape", "permute"]


def test_a_symbolic_batch_stays_the_leading_axis_of_every_permute() -> None:
    shape = (64, 4, 2, 4, 2)
    x = _flat(shape)
    ops = _canon(_permuted(shape, (0, 2, 4, 1, 3)), x, dynamic=True, gate="0")
    for n in _named(ops, "permute"):
        assert n.outputs[0].shape[0] == 64
        assert _perm(n)[0] == 0
    assert _widest(ops, "permute") <= 4
    ops = _canon(_permuted(shape, (1, 0, 2, 3, 4)), x, dynamic=True, gate="0")
    assert [n.name for n in ops] == ["reshape", "permute"]
    # Pinning the batch costs the window partition its one-step form: (B, nH, ws, nW, ws * C)
    # is rank 5, so it takes several rank <= 4 steps (at the costly lead, B * nH = 512).
    window = _canon(
        lambda t: _window_partition(t.reshape(64, 56, 56, 4)), _flat((64, 56, 56, 4)), dynamic=True
    )
    steps = _named(window, "permute")
    assert len(steps) > 1 and _widest(window, "permute") <= 4
    assert all(n.outputs[0].shape[0] == 64 and _perm(n)[0] == 0 for n in steps)


# ── softmax: the IR ───────────────────────────────────────────────────────────────────

_SOFT = (64, 4, 2, 3, 2)


@pytest.mark.parametrize(
    "dim,shape,canonical_dim",
    [
        (-1, (64 * 4 * 2 * 3, 2), 1),
        (0, (64, 4 * 2 * 3 * 2), 0),
        (2, (64 * 4, 2, 6), 1),
        (1, (64, 4, 12), 1),
        (3, (64 * 4 * 2, 3, 2), 1),
    ],
)
@pytest.mark.parametrize("log", [False, True])
def test_a_softmax_merges_the_axes_around_its_dim(
    dim: int, shape: Shape, canonical_dim: int, log: bool
) -> None:
    op = F.log_softmax if log else F.softmax
    ops = _canon(lambda t: op(t, dim=dim), _filled(_SOFT))
    (node,) = _named(ops, "log_softmax" if log else "softmax")
    assert tuple(node.outputs[0].shape) == shape
    assert node.attrs["dim"] == canonical_dim
    assert tuple(ops[-1].outputs[0].shape) == _SOFT


def test_a_softmax_is_left_alone_below_the_gate() -> None:
    ops = _canon(lambda t: F.softmax(t, dim=-1), _filled((4, 4, 2, 3, 2)))
    assert [n.name for n in ops] == ["softmax"]


def test_a_symbolic_batch_is_not_merged_into_the_rows_of_a_softmax() -> None:
    ops = _canon(lambda t: F.softmax(t, dim=3), _filled(_SOFT), dynamic=True)
    (node,) = _named(ops, "softmax")
    assert tuple(node.outputs[0].shape) == (64, 4 * 2, 3, 2)
    assert node.attrs["dim"] == 2


def test_attention_weights_a_matmul_reads_are_left_as_recorded() -> None:
    """``MatmulEmitter`` finds a softmax output by id, so neither op may be wrapped."""

    def attention(q: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        weights = F.softmax(q @ q.permute(0, 1, 2, 4, 3), dim=-1)
        return weights @ v

    q = _filled((64, 4, 2, 8, 16))
    ops = _canon(attention, q, _filled((64, 4, 2, 8, 16), 1.0))
    soft = _named(ops, "softmax")
    assert len(soft) == 1 and len(soft[0].outputs[0].shape) == 5
    last = _named(ops, "matmul")[-1]
    assert len(last.outputs[0].shape) == 5
    # The weights feed the matmul directly, with no view in between.
    assert last.inputs[0] == soft[0].outputs[0].id


def test_a_softmax_that_no_matmul_reads_is_wrapped_even_beside_one() -> None:
    def fn(q: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        return F.softmax(q, dim=-1).sum() + (q @ v).sum()

    q, v = _filled((64, 4, 2, 8, 16)), _filled((64, 4, 2, 16, 8), 1.0)
    ops = _canon(fn, q, v)
    assert _widest(ops, "softmax") <= 3


# ── matmul: the IR ────────────────────────────────────────────────────────────────────


def _matmul_shapes(
    a: Shape, b: Shape, gate: str | None = None
) -> tuple[list[Shape], Shape, list[_C_engine.compile.OpNode]]:
    """The shapes the matmul's operands are read at, its own shape, and the whole graph."""
    graph, feeds = _trace(lambda x, y: x @ y, _filled(a), _filled(b, 1.0))
    shapes = {i: tuple(t.shape) for i, t in feeds.items()}
    env = {} if gate is None else {ENV: gate}
    with mock.patch.dict(os.environ, env):
        ops = list(_C_engine.compile.canonicalize_rank(graph, feeds, False).ops)
    for n in ops:
        shapes.update({m.id: tuple(m.shape) for m in n.outputs})
    (node,) = _named(ops, "matmul")
    return [shapes[i] for i in node.inputs], tuple(node.outputs[0].shape), ops


def test_matmul_batch_axes_both_operands_span_merge() -> None:
    ins, out, _ = _matmul_shapes((64, 4, 2, 6, 5), (64, 4, 2, 5, 3))
    assert ins == [(512, 6, 5), (512, 5, 3)]
    assert out == (512, 6, 3)


def test_a_broadcast_batch_axis_is_kept_apart_from_the_shared_ones() -> None:
    ins, out, _ = _matmul_shapes((64, 4, 2, 6, 5), (2, 5, 3))
    assert ins == [(256, 2, 6, 5), (1, 2, 5, 3)]
    assert out == (256, 2, 6, 3)
    ins, out, _ = _matmul_shapes((64, 4, 2, 6, 5), (5, 3))
    assert ins == [(512, 6, 5), (5, 3)]
    assert out == (512, 6, 3)
    ins, out, _ = _matmul_shapes((6, 5), (64, 4, 2, 5, 3))
    assert ins == [(6, 5), (512, 5, 3)]
    assert out == (512, 6, 3)


def test_a_batch_that_needs_three_axes_is_left_alone() -> None:
    # a spans (64, 4), b spans (16, 4): the three batch axes differ in who spans them.
    _, out, ops = _matmul_shapes((64, 1, 4, 6, 5), (1, 16, 4, 5, 3))
    assert out == (64, 16, 4, 6, 3)
    assert [n.name for n in ops] == ["matmul"]


def test_a_matmul_below_the_gate_is_left_alone() -> None:
    _, out, ops = _matmul_shapes((2, 3, 4, 6, 5), (2, 3, 4, 5, 3))
    assert out == (2, 3, 4, 6, 3) and len(ops) == 1
    ins, out, _ = _matmul_shapes((2, 3, 4, 6, 5), (2, 3, 4, 5, 3), gate="0")
    assert ins == [(24, 6, 5), (24, 5, 3)]


def test_a_vector_operand_reaches_the_rule_as_a_column() -> None:
    # ``x @ v`` records a reshape of ``v`` to a column first, so the matmul it reaches is a
    # matrix product with a rank-2 operand, whose batch is x's alone.
    ins, out, _ = _matmul_shapes((64, 4, 2, 6, 5), (5, 1))
    assert ins == [(512, 6, 5), (5, 1)] and out == (512, 6, 1)


def test_a_symbolic_batch_keeps_axis_0_of_a_matmul() -> None:
    a, b = _filled((64, 4, 2, 6, 5)), _filled((64, 4, 2, 5, 3), 1.0)
    ops = _canon(lambda x, y: x @ y, a, b, dynamic=True)
    (node,) = _named(ops, "matmul")
    assert tuple(node.outputs[0].shape) == (64, 8, 6, 3)
    ops = _canon(lambda x, y: x @ y, a, _filled((4, 2, 5, 3), 1.0), dynamic=True)
    assert [n.name for n in ops] == ["matmul"]


# ── values ────────────────────────────────────────────────────────────────────────────

_DTYPES = [lucid.float32, lucid.float16, lucid.bfloat16]
#: A half-precision result is exact to a few steps of its mantissa.
_TOL = {lucid.float32: 3e-4, lucid.float16: 4e-3, lucid.bfloat16: 3e-2}


def _values(t: lucid.Tensor) -> list[float]:
    return t.to(lucid.float32).reshape(-1).tolist()


def _close(got: lucid.Tensor, want: lucid.Tensor, tol: float, why: str) -> None:
    g, w = _values(got), _values(want)
    assert len(g) == len(w), why
    for i, (a, b) in enumerate(zip(g, w, strict=True)):
        assert a == pytest.approx(b, rel=tol, abs=tol), f"{why}[{i}]: {a} != {b}"


def _same(got: lucid.Tensor, want: lucid.Tensor, why: str) -> None:
    g, w = _values(got), _values(want)
    assert g == w, f"{why}: not bit-identical ({sum(a != b for a, b in zip(g, w, strict=True))})"


def _forward(
    fn: Callable[..., lucid.Tensor], args: tuple[lucid.Tensor, ...], gate: str
) -> lucid.Tensor:
    """``fn`` compiled with the pass at lead gate ``gate`` (a fresh executable each time)."""
    with mock.patch.dict(os.environ, {ENV: gate}):
        _C_engine.compile.session_cache_clear()
        return lucid.compile(fn)(*args)


def _check_forward(
    fn: Callable[..., lucid.Tensor], *args: lucid.Tensor, exact: bool = True
) -> None:
    """The rewritten graph equals eager within the dtype's rounding, and, when ``exact``, the
    unrewritten compiled graph to the bit (an op fused with half-precision neighbours rounds
    by layout, so those rows keep only the eager bound)."""
    want = fn(*args)
    on = _forward(fn, args, "0")
    if exact:
        _same(on, _forward(fn, args, "-1"), "rewritten vs unrewritten")
    _close(on, want, _TOL[args[0].dtype], "vs eager")


def _on_metal(t: lucid.Tensor, dtype: lucid.dtype) -> lucid.Tensor:
    return t.to(COMPILE_DEVICE).to(dtype)


#: ``(shape, perm)`` — window partition and reverse, space-to-depth and back, the qkv split,
#: size-1 axes anywhere, an identity, and one past the search's cap (left as recorded).
_PERMUTES: dict[str, tuple[Shape, tuple[int, ...]]] = {
    "window_partition": ((2, 2, 3, 2, 3, 4), (0, 1, 3, 2, 4, 5)),
    "window_reverse": ((2, 2, 2, 3, 3, 4), (0, 1, 3, 2, 4, 5)),
    "space_to_depth": ((3, 4, 2, 3, 2), (0, 2, 4, 1, 3)),
    "depth_to_space": ((3, 2, 2, 4, 3), (0, 3, 1, 4, 2)),
    "qkv_split": ((2, 5, 3, 2, 4), (2, 0, 3, 1, 4)),
    "unit_axes": ((3, 1, 4, 2, 1, 3), (4, 2, 0, 5, 1, 3)),
    "unit_axes_split": ((3, 1, 2, 3, 2, 2), (0, 2, 4, 1, 3, 5)),
    "identity": ((3, 2, 2, 2, 2), (0, 1, 2, 3, 4)),
    "reversal_rank_6": ((2, 3, 2, 3, 2, 2), (5, 4, 3, 2, 1, 0)),
    "rank_7_out_of_reach": ((2, 2, 2, 2, 2, 2, 3), (6, 5, 4, 3, 2, 1, 0)),
}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("name", sorted(_PERMUTES))
def test_a_permute_matches_eager_and_the_unrewritten_graph(name: str, dtype: lucid.dtype) -> None:
    shape, perm = _PERMUTES[name]
    w = _on_metal(_filled(tuple(shape[p] for p in perm), 0.5), dtype)
    x = _on_metal(_flat(shape) * 0.5, dtype)
    # The permuted tensor is consumed in the graph, where it feeds an elementwise product.
    _check_forward(lambda t, u: (_permuted(shape, perm)(t) * u).sum(dim=-1), x, w)


class _Permuted(nn.Module):
    """``loss`` over ``permute(x * s) * w``: the gradient of ``x`` runs the inverse chain.

    With ``flat`` the permuted tensor is read through a ``reshape`` first, which the pass
    points at the canonical result instead of the recorded rank-5+ one.
    """

    def __init__(
        self, shape: Shape, perm: tuple[int, ...], dtype: lucid.dtype, flat: bool = False
    ) -> None:
        super().__init__()
        self.perm = perm
        self.shape = shape
        self.flat = flat
        self.x = nn.Parameter(_on_metal(_filled((_numel(shape),), 0.1), dtype))
        out = tuple(shape[p] for p in perm)
        self.w = nn.Parameter(_on_metal(_filled(out, 0.7), dtype))

    def forward(self, s: lucid.Tensor) -> lucid.Tensor:
        y = (self.x * s).reshape(*self.shape).permute(*self.perm)
        if self.flat:
            return y.reshape(-1) * self.w.reshape(-1)
        return y * self.w


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


def _check_step(
    make: Callable[[], nn.Module], arg: lucid.Tensor, dtype: lucid.dtype, exact: bool = True
) -> None:
    eager = to_metal(make())
    want = _loss(eager(arg))
    want.backward()
    on, on_grads = _step(make, arg, "0")
    off, off_grads = _step(make, arg, "-1")
    tol = _TOL[dtype]
    if exact:
        _same(on, off, "loss, rewritten vs unrewritten")
    _close(on, want, tol, "loss vs eager")
    assert on_grads.keys() == off_grads.keys() and on_grads
    for name, grad in on_grads.items():
        if exact:
            _same(grad, off_grads[name], f"grad {name}, rewritten vs unrewritten")
        eager_grad = dict(eager.named_parameters())[name].grad
        assert eager_grad is not None
        # A half-precision backward rounds every intermediate: scale by the gradient's size.
        scale = max(max(abs(v) for v in _values(eager_grad)), 1.0)
        _close(grad / scale, eager_grad / scale, tol * 4, f"grad {name} vs eager")


@needs_metal
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("name", sorted(_PERMUTES))
def test_a_permute_step_matches_eager_and_the_unrewritten_graph(
    name: str, dtype: lucid.dtype
) -> None:
    shape, perm = _PERMUTES[name]
    arg = _on_metal(lucid.ones(1), dtype)
    _check_step(lambda: _Permuted(shape, perm, dtype), arg, dtype)


@needs_metal
@pytest.mark.parametrize("dtype", [lucid.float32, lucid.bfloat16], ids=str)
@pytest.mark.parametrize("name", sorted(_PERMUTES))
def test_a_reshape_reading_a_rewritten_permute_keeps_the_gradient(
    name: str, dtype: lucid.dtype
) -> None:
    shape, perm = _PERMUTES[name]
    arg = _on_metal(lucid.ones(1), dtype)
    _check_step(lambda: _Permuted(shape, perm, dtype, flat=True), arg, dtype)


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_symbolic_batch_survives_the_permute_rewrite() -> None:
    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return ((t * 1.5).permute(0, 2, 4, 1, 3) * 2.0).sum(dim=-1)

    with mock.patch.dict(os.environ, {ENV: "0"}):
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(fn, dynamic=True)
        for batch in (3, 3, 3):
            x = _on_metal(_filled((batch, 4, 2, 3, 2), 0.2 * batch), lucid.float32)
            _close(compiled(x), fn(x), 3e-4, f"batch {batch}")


#: softmax / log_softmax over each axis of rank 5 and rank 6 inputs.
_SOFTMAXES = [((3, 4, 2, 3, 2), dim) for dim in (0, 1, 2, 3, 4)] + [
    ((3, 4, 2, 3, 2, 2), dim) for dim in (0, 2, 5)
]


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("log", [False, True], ids=["softmax", "log_softmax"])
@pytest.mark.parametrize("shape,dim", _SOFTMAXES, ids=lambda v: str(v).replace(" ", ""))
def test_a_softmax_matches_eager_and_the_unrewritten_graph(
    shape: Shape, dim: int, log: bool, dtype: lucid.dtype
) -> None:
    op = F.log_softmax if log else F.softmax
    x = _on_metal(_filled(shape, 0.3), dtype)
    w = _on_metal(_filled(shape, 0.9), dtype)
    # Fused with its neighbours at half precision the rounding follows the layout, so only
    # float32 is held to the bit; ``test_a_bare_softmax...`` holds the op itself.
    _check_forward(
        lambda t, u: (op(t * u, dim=dim) * u).sum(dim=-1), x, w, exact=dtype is lucid.float32
    )


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("log", [False, True], ids=["softmax", "log_softmax"])
@pytest.mark.parametrize("shape,dim", _SOFTMAXES, ids=lambda v: str(v).replace(" ", ""))
def test_a_bare_softmax_is_bit_identical_to_the_unrewritten_graph(
    shape: Shape, dim: int, log: bool, dtype: lucid.dtype
) -> None:
    """Exact at float32 for every axis, and at half precision for every axis but 0.

    MPSGraph's half-precision softmax over the leading axis of a rank-5 tensor reduces in
    another order than over the same axis of the merged rank-2 tensor (which equals the
    rank-2 twin to the bit): both are within a few half ulps of the float32 answer.
    """
    op = F.log_softmax if log else F.softmax
    x = _on_metal(_filled(shape, 0.3), dtype)
    _check_forward(lambda t: op(t, dim=dim), x, exact=dtype is lucid.float32 or dim != 0)


class _Softmaxed(nn.Module):
    def __init__(self, shape: Shape, dim: int, log: bool, dtype: lucid.dtype) -> None:
        super().__init__()
        self.dim, self.log = dim, log
        self.x = nn.Parameter(_on_metal(_filled(shape, 0.3), dtype))
        self.w = nn.Parameter(_on_metal(_filled(shape, 0.9), dtype))

    def forward(self, s: lucid.Tensor) -> lucid.Tensor:
        op = F.log_softmax if self.log else F.softmax
        return op(self.x * s, dim=self.dim) * self.w


@needs_metal
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("log", [False, True], ids=["softmax", "log_softmax"])
@pytest.mark.parametrize("dim", [0, 2, 4])
def test_a_softmax_step_matches_eager_and_the_unrewritten_graph(
    dim: int, log: bool, dtype: lucid.dtype
) -> None:
    arg = _on_metal(lucid.ones(1), dtype)
    _check_step(
        lambda: _Softmaxed((3, 4, 2, 3, 2), dim, log, dtype),
        arg,
        dtype,
        exact=dtype is lucid.float32,
    )


#: ``(a, b)`` batch patterns: shared axes, a broadcast one, a leading and a trailing one.
_MATMULS: list[tuple[Shape, Shape]] = [
    ((3, 4, 2, 6, 5), (3, 4, 2, 5, 3)),
    ((3, 4, 2, 6, 5), (2, 5, 3)),
    ((3, 4, 2, 6, 5), (5, 3)),
    ((6, 5), (3, 4, 2, 5, 3)),
    ((3, 4, 2, 6, 5), (4, 2, 5, 3)),
    ((3, 1, 2, 6, 5), (3, 4, 2, 5, 3)),
    ((3, 4, 2, 6, 5), (1, 1, 1, 5, 3)),
    ((2, 3, 4, 2, 6, 5), (2, 3, 4, 2, 5, 3)),
    ((2, 3, 4, 2, 6, 5), (3, 1, 2, 5, 3)),
    ((3, 4, 2, 1, 5), (3, 4, 2, 5, 1)),
]


def _ids(pair: tuple[Shape, Shape]) -> str:
    return "-".join("x".join(map(str, s)) for s in pair)


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("pair", _MATMULS, ids=_ids)
def test_a_matmul_matches_eager_and_the_unrewritten_graph(
    pair: tuple[Shape, Shape], dtype: lucid.dtype
) -> None:
    a = _on_metal(_filled(pair[0], 0.1) - 1.0, dtype)
    b = _on_metal(_filled(pair[1], 0.8) - 1.0, dtype)
    _check_forward(lambda x, y: (x @ y).sum(dim=-1), a, b)


class _Product(nn.Module):
    def __init__(self, pair: tuple[Shape, Shape], dtype: lucid.dtype) -> None:
        super().__init__()
        self.a = nn.Parameter(_on_metal(_filled(pair[0], 0.1) - 1.0, dtype))
        self.b = nn.Parameter(_on_metal(_filled(pair[1], 0.8) - 1.0, dtype))

    def forward(self, s: lucid.Tensor) -> lucid.Tensor:
        return (self.a * s) @ self.b


@needs_metal
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("pair", _MATMULS, ids=_ids)
def test_a_matmul_step_matches_eager_and_the_unrewritten_graph(
    pair: tuple[Shape, Shape], dtype: lucid.dtype
) -> None:
    arg = _on_metal(lucid.ones(1), dtype)
    _check_step(lambda: _Product(pair, dtype), arg, dtype)


# ── time ──────────────────────────────────────────────────────────────────────────────

Case = tuple[Callable[..., lucid.Tensor], tuple[lucid.Tensor, ...]]


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


# Each case reads its rank-5+ result through a reshape, as a model does (Swin merges the
# window axes, attention merges heads): a reduction reads the recorded rank-5+ result by
# itself, which is the reduction rule's to fix (LCD-920e).
def _window_cases() -> tuple[Case, Case]:
    x = _metal((256, 56, 56, 4))
    return (
        (lambda t: _window_partition(t).reshape(256 * 64, 49, 4).sum(dim=1), (x,)),
        (
            lambda t: (
                t.reshape(2048, 7, 8, 28).permute(0, 2, 1, 3).reshape(256 * 64, 49, 4).sum(dim=1)
            ),
            (x,),
        ),
    )


def _narrow_cases() -> tuple[Case, Case]:
    # Swap the last two axes of a rank-5 tensor whose last axis is 2 wide: 250 ms unfixed.
    x = _metal((65536 * 4 * 2 * 8 * 2,))
    return (
        (
            lambda t: (
                t.reshape(65536, 4, 2, 8, 2)
                .permute(0, 1, 2, 4, 3)
                .reshape(65536 * 8 * 2, 8)
                .sum(dim=1)
            ),
            (x,),
        ),
        (
            lambda t: (
                t.reshape(65536 * 8, 8, 2).permute(0, 2, 1).reshape(65536 * 8 * 2, 8).sum(dim=1)
            ),
            (x,),
        ),
    )


def _softmax_cases() -> tuple[Case, Case]:
    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return F.softmax(t, dim=-1).reshape(1024 * 8 * 4 * 16, 16).sum(dim=-1)

    return (fn, (_metal((1024, 8, 4, 16, 16)),)), (fn, (_metal((1024 * 8 * 4, 16, 16)),))


def _matmul_cases() -> tuple[Case, Case]:
    def fn(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        return (a @ b).reshape(1024 * 8 * 16, 16).sum(dim=-1)

    return (
        (fn, (_metal((1024, 4, 2, 16, 32)), _metal((1024, 4, 2, 32, 16), 1.0))),
        (fn, (_metal((1024 * 8, 16, 32)), _metal((1024 * 8, 32, 16), 1.0))),
    )


#: Each row: the rank-5+ graph and the same elements and the same work at rank <= 4.
_TWINS: dict[str, Callable[[], tuple[Case, Case]]] = {
    "window_partition": _window_cases,
    "narrow_last_axis_permute": _narrow_cases,
    "softmax_lead_1024": _softmax_cases,
    "matmul_lead_1024": _matmul_cases,
}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", sorted(_TWINS))
def test_a_rank_5_case_is_not_slower_than_its_rank_4_twin(name: str) -> None:
    """Held to 1.5x the twin; the unfixed graphs ran 2x to 20x it."""
    wide_case, twin_case = _TWINS[name]()
    _C_engine.compile.session_cache_clear()
    wide, twin = _forward_ms(wide_case[0], *wide_case[1]), _forward_ms(twin_case[0], *twin_case[1])
    assert wide <= 1.5 * twin + 0.3, f"{name}: rank-5+ {wide:.2f} ms vs twin {twin:.2f} ms"
