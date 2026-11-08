"""A compiled graph keeps rank-5+ elementwise ops out of MPSGraph's slice loop (LCD-920a).

MPSGraph runs a compute op over a rank >= 5 tensor as a loop over the dims above the last
four.  ``RankCanon`` (``lucid/_C/compile/RankCanon.h``) wraps such an op in reshape views so
it runs at the lowest rank its operands' broadcast pattern allows.  Three guards:

* the pass's IR, read through ``canonicalize_rank`` with no GPU: which ops are wrapped, at
  which canonical shapes, what stays alone (below the lead gate, pass off, symbolic batch
  axis), that every original id survives with its shape, and that a custom-backward region's
  op indices are remapped;
* a value table: loss and gradient of a compiled step equal eager's across broadcast
  patterns, ranks 5..6, a tensor read at two layouts (its gradient accumulates at rank 5+),
  a region between rewritten ops, and a symbolic batch;
* a coarse timing bound: the rank-5 ``x * w`` may not run several times slower than the same
  elements as a rank-4 twin (the unfixed graph was ~14x).
"""

import os
import time
from collections.abc import Callable, Iterator
from unittest import mock

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.autograd import Function
from lucid.compile import _tracing, make_step
from lucid.compile._core.trace_context import recording
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, assert_cache_hit, to_metal

ENV = "LUCID_COMPILE_RANK_CANON_MIN_LEAD"

needs_metal = pytest.mark.skipif(not metal_available(), reason="compile needs Metal")

# A fallback to eager would make every compiled == eager row below vacuous.
pytestmark = pytest.mark.filterwarnings(
    "error::lucid.compile._core.fallback.CompileFallbackWarning"
)


@pytest.fixture(autouse=True)
def _default_gate() -> Iterator[None]:
    """Every test starts from the default lead threshold, whatever the shell exports."""
    with mock.patch.dict(os.environ):
        os.environ.pop(ENV, None)
        yield


def _filled(shape: tuple[int, ...], phase: float = 0.0) -> lucid.Tensor:
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
    fn: Callable[..., lucid.Tensor], *args: lucid.Tensor, dynamic: bool = False
) -> tuple[list[tuple[str, tuple[int, ...]]], _C_engine.compile.TraceGraph]:
    graph, feeds = _trace(fn, *args)
    out = _C_engine.compile.canonicalize_rank(graph, feeds, dynamic)
    return [(n.name, tuple(n.outputs[0].shape)) for n in out.ops], out


# ── policy ────────────────────────────────────────────────────────────────────────────


def test_the_lead_threshold_defaults_to_64_and_reads_the_environment() -> None:
    assert _C_engine.compile.rank_canon_min_lead() == 64
    for raw, want in [("0", 0), ("256", 256), ("-1", -1), ("-7", -1), ("junk", 64), ("", 64)]:
        with mock.patch.dict(os.environ, {ENV: raw}):
            assert _C_engine.compile.rank_canon_min_lead() == want, raw


# ── the pass's IR ─────────────────────────────────────────────────────────────────────

_BIG = (96, 64, 16, 2, 32)


def test_a_same_shape_binary_op_runs_flat_between_views() -> None:
    x, w = _filled(_BIG), _filled(_BIG, 1.0)
    ops, out = _canon(lambda a, b: a * b, x, w)
    n = 96 * 64 * 16 * 2 * 32
    assert ops == [("reshape", (n,)), ("reshape", (n,)), ("mul", (n,)), ("reshape", _BIG)]
    assert out.next_id > max(m.id for node in out.ops for m in node.outputs)


def test_a_broadcast_operand_keeps_the_axes_it_does_not_span() -> None:
    x, bias = _filled(_BIG), _filled((1, 1, 1, 1, 32), 2.0)
    ops, _ = _canon(lambda a, b: a + b, x, bias)
    rows = 96 * 64 * 16 * 2
    assert ops == [
        ("reshape", (rows, 32)),
        ("reshape", (1, 32)),
        ("add", (rows, 32)),
        ("reshape", _BIG),
    ]


def test_a_scalar_operand_is_left_alone() -> None:
    x = _filled(_BIG)
    ops, _ = _canon(lambda a: a * 3.0, x)
    n = 96 * 64 * 16 * 2 * 32
    assert [name for name, _ in ops].count("mul") == 1
    assert ("mul", (n,)) in ops
    assert ("reshape", _BIG) in ops


def test_a_unary_chain_shares_one_view_per_value() -> None:
    x = _filled(_BIG)
    ops, out = _canon(lambda a: lucid.exp(lucid.sin(a)), x)
    out_ops = list(out.ops)
    names = [name for name, _ in ops]
    # sin wraps its input in a view; exp reads sin's flat result directly rather than a view
    # of the view that restores the recorded shape.
    assert names == ["reshape", "sin", "reshape", "exp", "reshape"]
    flat = [n for n in out_ops if n.name in ("sin", "exp")]
    assert flat[1].inputs[0] == flat[0].outputs[0].id


def test_a_value_read_by_two_ops_at_one_layout_is_viewed_once() -> None:
    x, a, b = _filled(_BIG), _filled(_BIG, 1.0), _filled(_BIG, 2.0)
    graph, feeds = _trace(lambda p, q, r: p * q + p * r, x, a, b)
    out = _C_engine.compile.canonicalize_rank(graph, feeds, False)
    views = [n for n in out.ops if n.name == "reshape" and len(n.outputs[0].shape) == 1]
    # x, a and b once each (x is shared by both muls); the two products reach the add as the
    # flat results they are, with no view of a view in between.
    assert len(views) == 3
    assert len({n.inputs[0] for n in views}) == 3


def test_every_original_id_survives_with_its_shape() -> None:
    x, w = _filled(_BIG), _filled(_BIG, 1.0)
    graph, feeds = _trace(lambda a, b: lucid.exp(a * b), x, w)
    out = _C_engine.compile.canonicalize_rank(graph, feeds, False)
    kept = {m.id: tuple(m.shape) for n in out.ops for m in n.outputs}
    for node in graph.ops:
        for m in node.outputs:
            assert kept[m.id] == tuple(m.shape)


@pytest.mark.parametrize("shape", [(4, 16, 2, 16, 2), (1, 2, 3, 4, 5), (96, 64, 32, 32)])
def test_a_trace_below_the_lead_gate_or_rank_is_returned_unchanged(shape: tuple[int, ...]) -> None:
    x = _filled(shape)
    graph, feeds = _trace(lambda a: a * a + a, x)
    out = _C_engine.compile.canonicalize_rank(graph, feeds, False)
    assert len(out) == len(graph)


def test_zero_rewrites_every_rank_5_op_and_a_negative_value_turns_the_pass_off() -> None:
    x = _filled((4, 3, 2, 3, 2))
    graph, feeds = _trace(lambda a: a * a, x)
    with mock.patch.dict(os.environ, {ENV: "0"}):
        wrapped = _C_engine.compile.canonicalize_rank(graph, feeds, False)
    assert [n.name for n in wrapped.ops] == ["reshape", "mul", "reshape"]
    big = _filled(_BIG)
    graph, feeds = _trace(lambda a: a * a, big)
    with mock.patch.dict(os.environ, {ENV: "-1"}):
        assert len(_C_engine.compile.canonicalize_rank(graph, feeds, False)) == len(graph)


def test_a_symbolic_batch_axis_is_never_merged() -> None:
    x, w = _filled((128, 8, 2, 4, 4)), _filled((128, 8, 2, 4, 4), 1.0)
    ops, _ = _canon(lambda a, b: a * b, x, w, dynamic=True)
    assert ("mul", (128, 8 * 2 * 4 * 4)) in ops
    padded = _filled((8, 2, 4, 4), 2.0)
    ops, _ = _canon(lambda a, b: a * b, x, padded, dynamic=True)
    # An operand the broadcast pads on the left has no batch axis to keep: untouched.
    assert [name for name, _ in ops] == ["mul"]


def test_an_op_the_rule_does_not_own_is_left_alone() -> None:
    x = _filled(_BIG)
    ops, _ = _canon(lambda a: a.sum(dim=(0, 1)), x)
    assert [name for name, _ in ops] == ["sum"]


class _Reversal(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx, h):
        ctx.save_for_backward(h)
        return h * 2.0

    @staticmethod
    def backward(ctx, g):
        (h,) = ctx.saved_tensors
        return -g * (h.abs() + 1.5)


@needs_metal
def test_a_custom_backward_region_keeps_its_ops_and_its_indices_follow_the_rewrite() -> None:
    x = _filled(_BIG).to(COMPILE_DEVICE)
    w = _filled(_BIG, 1.0).to(COMPILE_DEVICE)
    x.requires_grad_(True)
    with recording(watch=(x,)) as tracer:
        out = _Reversal.apply(x * w) * w
        (out * 1.5).exp().sum()
    graph, feeds = tracer.graph, tracer.external_feeds
    assert len(graph.vjp_regions) >= 1
    result = _C_engine.compile.canonicalize_rank(graph, feeds, False)
    assert len(result) > len(graph)
    for before, after in zip(graph.vjp_regions, result.vjp_regions, strict=True):
        want = [n.name for n in graph.ops[before.fwd_begin : before.fwd_end]]
        assert [n.name for n in result.ops[after.fwd_begin : after.fwd_end]] == want
        assert len(after.bwd_ops) == len(before.bwd_ops)
    # The ops inside the region were not wrapped; the ones outside it were.
    original = {n.outputs[0].id: tuple(n.outputs[0].shape) for n in graph.ops}
    inside = {
        n.outputs[0].id for r in graph.vjp_regions for n in graph.ops[r.fwd_begin : r.fwd_end]
    }
    for node in result.ops:
        if node.outputs[0].id in inside:
            assert tuple(node.outputs[0].shape) == original[node.outputs[0].id]
    wrapped = [n for n in result.ops if n.name == "mul" and len(n.outputs[0].shape) == 1]
    assert wrapped, "a mul outside the region runs flat"


# ── values ────────────────────────────────────────────────────────────────────────────


class _Pointwise(nn.Module):
    """``act(x * w + b)`` with a full-shape ``w`` and a broadcast ``b``."""

    def __init__(self, shape: tuple[int, ...], bias_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.w = nn.Parameter(_filled(shape, 0.5).to(COMPILE_DEVICE))
        self.b = nn.Parameter((_filled(bias_shape, 1.5) * 0.1).to(COMPILE_DEVICE))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return lucid.tanh(x * self.w + self.b)


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    return (y * y).sum()


def _same(got: lucid.Tensor, want: lucid.Tensor, why: str) -> None:
    g, w = got.reshape(-1).tolist(), want.reshape(-1).tolist()
    assert len(g) == len(w), why
    for i, (a, b) in enumerate(zip(g, w, strict=True)):
        if b != b:
            assert a != a, f"{why}[{i}]: {a} != nan"
        else:
            assert a == pytest.approx(b, rel=3e-4, abs=3e-4), f"{why}[{i}]: {a} != {b}"


def _step_matches_eager(
    make: Callable[[], nn.Module],
    shape: tuple[int, ...],
    loss: Callable[[lucid.Tensor], lucid.Tensor] = _loss,
) -> None:
    x = _filled(shape, 0.25).to(COMPILE_DEVICE)
    eager = to_metal(make())
    want = loss(eager(x))
    want.backward()
    comp = to_metal(make())
    step = make_step(comp, loss)
    got = step(x)
    got.backward()
    _same(got, want, "loss")
    for (name, pe), (_, pc) in zip(eager.named_parameters(), comp.named_parameters(), strict=True):
        assert pe.grad is not None and pc.grad is not None, name
        _same(pc.grad, pe.grad, f"grad {name}")


_VALUE_CASES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((64, 4, 2, 3, 2), (64, 4, 2, 3, 2)),
    ((64, 4, 2, 3, 2), (1, 1, 1, 1, 2)),
    ((64, 4, 2, 3, 2), (4, 1, 1, 2)),
    ((65, 3, 2, 2, 3), (3, 1, 2, 1)),
    ((8, 8, 2, 3, 2, 2), (3, 1, 2)),
    ((8, 8, 2, 3, 2, 2), (8, 2, 3, 2, 2)),
]


@needs_metal
@pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning")
@pytest.mark.parametrize("shape,bias", _VALUE_CASES, ids=lambda v: "x".join(map(str, v)))
def test_a_wrapped_pointwise_step_matches_eager(
    shape: tuple[int, ...], bias: tuple[int, ...]
) -> None:
    _step_matches_eager(lambda: _Pointwise(shape, bias), shape)


class _TwoLayouts(nn.Module):
    """``x`` is read at two layouts, so its gradient is summed from rank-5+ parts."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.x = nn.Parameter(_filled(shape, 0.1).to(COMPILE_DEVICE))
        self.full = nn.Parameter(_filled(shape, 0.7).to(COMPILE_DEVICE))
        self.row = nn.Parameter(_filled((1, 1, 1, 1, shape[-1]), 0.3).to(COMPILE_DEVICE))

    def forward(self, scale: lucid.Tensor) -> lucid.Tensor:
        a = self.x * self.full  # flat layout
        b = self.x * self.row  # (rows, last) layout
        c = self.x.sum(dim=2)  # a reduction reads x as it is
        return a + b + c.unsqueeze(2) * scale


@needs_metal
@pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning")
def test_a_tensor_read_at_two_layouts_sums_its_gradient_correctly() -> None:
    shape = (64, 4, 2, 3, 2)
    _step_matches_eager(lambda: _TwoLayouts(shape), (1,))


class _ThroughRegion(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.w = nn.Parameter(_filled(shape, 0.4).to(COMPILE_DEVICE))
        self.v = nn.Parameter(_filled(shape, 0.9).to(COMPILE_DEVICE))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return _Reversal.apply(x * self.w) * self.v + x


@needs_metal
@pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning")
def test_elementwise_ops_around_a_custom_backward_region_match_eager() -> None:
    shape = (64, 4, 2, 3, 2)
    _step_matches_eager(lambda: _ThroughRegion(shape), shape)


class _Gate(nn.Module):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        mask = x > 1.5
        return lucid.where(mask, x * 2.0, lucid.exp(-x)) + lucid.maximum(x, 1.2)


@needs_metal
@pytest.mark.usefixtures("inference")
def test_where_compare_and_extremum_over_rank_5_match_eager() -> None:
    m = to_metal(_Gate()).eval()
    x = _filled((96, 4, 2, 3, 2), 0.2).to(COMPILE_DEVICE)
    got = lucid.compile(m)(x)
    _same(got, m(x), "gate")


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_symbolic_batch_survives_the_wrapped_ops() -> None:
    m = to_metal(_Pointwise((64, 4, 2, 3, 2), (1, 1, 1, 1, 2))).eval()
    cm = lucid.compile(m, dynamic=True)
    for batch in (64, 64, 64):
        x = _filled((batch, 4, 2, 3, 2), 0.25 * batch).to(COMPILE_DEVICE)
        _same(cm(x), m(x), f"batch {batch}")
    assert_cache_hit(cm, 1)


# ── time ──────────────────────────────────────────────────────────────────────────────


class _Scaled(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.w = nn.Parameter(_filled(shape, 0.5).to(COMPILE_DEVICE))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        # A rank-5 *output* is copied out by a slow kernel whatever ran before it, so the
        # product is consumed inside the graph.
        return (x * self.w).sum(dim=-1)


def _forward_ms(shape: tuple[int, ...]) -> float:
    m = to_metal(_Scaled(shape)).eval()
    x = _filled(shape).to(COMPILE_DEVICE)
    cm = lucid.compile(m)
    for _ in range(3):
        cm(x)
    best = float("inf")
    for _ in range(15):
        t0 = time.perf_counter()
        float(cm(x)[0].sum().item())
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_rank_5_multiply_is_not_slower_than_its_rank_4_twin() -> None:
    """Coarse on purpose: the unfixed graph ran ~14x the twin, the fixed one ~1x."""
    rank5 = _forward_ms((6144, 16, 2, 16, 2))
    rank4 = _forward_ms((6144, 16, 2, 32))
    assert rank5 < 3.0 * rank4 + 2.0, f"rank-5 {rank5:.1f} ms vs rank-4 {rank4:.1f} ms"


# ── every op the rule owns, wrapped on shapes far below the gate ───────────────────────

_UNARY: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "relu": lambda t: lucid.relu(t - 1.5),
    "sigmoid": lucid.sigmoid,
    "tanh": lucid.tanh,
    "exp": lambda t: lucid.exp(t * 0.5),
    "log": lambda t: lucid.log(t + 0.5),
    "sqrt": lucid.sqrt,
    "neg": lambda t: -t,
    "abs": lambda t: lucid.abs(t - 1.5),
    "sin": lucid.sin,
    "cos": lucid.cos,
    "square": lambda t: t * t,
    "reciprocal": lambda t: 1.0 / t,
    "floor": lambda t: lucid.floor(t * 3.0),
    "clip": lambda t: t.clip(1.2, 1.8),
    "pow": lambda t: t**1.5,
    "erf": lucid.erf,
    "sign": lambda t: lucid.sign(t - 1.5),
}

_BINARY: dict[str, Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / (b + 0.5),
    "maximum": lucid.maximum,
    "minimum": lucid.minimum,
    "where": lambda a, b: lucid.where(a > b, a, b * 2.0),
    "gt": lambda a, b: (a > b).to(a.dtype),
    "pow": lambda a, b: a ** (b * 0.5),
}

#: ``(x shape, other shape)``: same shape, a trailing broadcast, a leading one, a middle one.
_PAIRS = [
    ((3, 2, 3, 2, 2), (3, 2, 3, 2, 2)),
    ((3, 2, 3, 2, 2), (1, 1, 1, 1, 2)),
    ((3, 2, 3, 2, 2), (3, 1, 1, 1, 1)),
    ((3, 2, 3, 2, 2), (2, 1, 2, 2)),
    ((3, 2, 3, 2, 2, 2), (3, 1, 3, 1, 2, 1)),
]


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", sorted(_UNARY))
def test_a_wrapped_unary_op_matches_eager(name: str) -> None:
    fn = _UNARY[name]
    x = _filled((3, 2, 3, 2, 2), 0.1).to(COMPILE_DEVICE)
    with mock.patch.dict(os.environ, {ENV: "0"}):
        got = lucid.compile(fn)(x)
    _same(got, fn(x), name)


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", sorted(_BINARY))
@pytest.mark.parametrize("pair", _PAIRS, ids=lambda p: "-".join("x".join(map(str, s)) for s in p))
def test_a_wrapped_binary_op_matches_eager(name: str, pair: tuple[tuple[int, ...], ...]) -> None:
    fn = _BINARY[name]
    a = _filled(pair[0], 0.1).to(COMPILE_DEVICE)
    b = _filled(pair[1], 1.3).to(COMPILE_DEVICE)
    with mock.patch.dict(os.environ, {ENV: "0"}):
        got = lucid.compile(fn)(a, b)
    _same(got, fn(a, b), name)
