"""Attention at rank 5+ runs at rank <= 4, and a rewrite never feeds a rank-5+ consumer (LCD-923, LCD-925).

A window-attention block over ``(B, nW, heads, N, d)`` is ``q @ kᵀ -> softmax -> (dropout) -> @ v``.
``MpsBuilder``'s fused-attention workaround finds the weights a matmul reads by following the
softmax output through ``dropout`` and ``reshape`` on the graph the pass hands it, so the
softmax and both matmuls are rewritten like any other op and the workaround still lands on the
canonical weights.

* the pass's IR: the block has no rank-5+ op but reshapes, in eval and in train;
* the workaround: at the shapes where MPSGraph's fused attention miscompiled, a rank-5 block still
  matches a NumPy reference (it would not if the matmul read the weights unmarked);
* values on Metal: forward and ``make_step`` gradients equal eager, and the unrewritten graph to
  the bit at float32;
* time: against the rank-4 twin;
* LCD-925: a rewritten producer whose consumer still reads rank 5+ (an op no rule owns) is left
  as recorded, so the pass is not slower than off.
"""

import os
from collections.abc import Callable, Iterator
from unittest import mock

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.compile import make_step
from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_ratio_within,
    compiled_call,
    with_env,
)
from lucid.test.unit.compile.test_rank_canonical_layout import (
    _DTYPES,
    ENV,
    Shape,
    _canon,
    _check_forward,
    _check_step,
    _filled,
    _named,
    _on_metal,
    _widest,
    needs_metal,
)

# A fallback to eager would make every compiled == eager row below vacuous.
pytestmark = pytest.mark.filterwarnings(
    "error::lucid.compile._core.fallback.CompileFallbackWarning"
)

_BLOCK: Shape = (4, 3, 2, 8, 16)
_RANK_6: Shape = (2, 3, 2, 3, 8, 16)


def _attention(
    q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor, p: float = 0.0, train: bool = False
) -> lucid.Tensor:
    weights = F.softmax((q @ (k * 1.0).mT) * 0.25, dim=-1)
    return F.dropout(weights, p, training=train) @ v


def _wide(ops: list[_C_engine.compile.OpNode]) -> list[str]:
    return [n.name for n in ops if len(n.outputs[0].shape) >= 5 and n.name != "reshape"]


# ── the IR ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("train", [False, True], ids=["eval_p0", "train_p0.1"])
@pytest.mark.parametrize("shape", [_BLOCK, _RANK_6], ids=["rank5", "rank6"])
def test_a_rank_5_attention_block_has_no_wide_op_but_reshapes(train: bool, shape: Shape) -> None:
    q, k, v = (_filled(shape, ph) for ph in (0.0, 1.0, 2.0))

    def fn(a: lucid.Tensor, b: lucid.Tensor, c: lucid.Tensor) -> lucid.Tensor:
        return _attention(a, b, c, 0.1 if train else 0.0, train)

    ops = _canon(fn, q, k, v, gate="0")
    assert _wide(ops) == []
    assert len(_named(ops, "softmax")) == 1
    assert len(_named(ops, "matmul")) == 2


def test_the_block_stays_as_recorded_below_the_gate() -> None:
    q, k, v = (_filled(_BLOCK, ph) for ph in (0.0, 1.0, 2.0))
    ops = _canon(_attention, q, k, v)
    assert _widest(ops, "softmax") == 5


# ── LCD-925: a rewrite whose consumer reads rank 5+ ───────────────────────────────────


# ``topk`` has no rule and builds no canonical layout of its own: it reads its operand at
# the rank it was recorded at.  (``layer_norm`` and the pool3d ops canonicalise internally
# since LCD-927, so a view in front of them is free: see the last test of this block.)
def _topk(x: lucid.Tensor) -> lucid.Tensor:
    return lucid.topk(x, 2, dim=-1)[0].sum()


def test_a_producer_read_by_an_op_no_rule_owns_is_left_alone() -> None:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return _topk(x + x)

    ops = _canon(fn, _filled((8, 4, 2, 4, 16)), gate="0")
    assert "reshape" not in [n.name for n in ops]
    assert len(_named(ops, "add")[0].outputs[0].shape) == 5


def test_a_chain_ending_at_such_a_consumer_is_left_alone_back_to_its_start() -> None:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return _topk(((x + x) * 2.0).exp())

    ops = _canon(fn, _filled((8, 4, 2, 4, 16)), gate="0")
    assert "reshape" not in [n.name for n in ops]


def test_a_producer_read_through_a_view_by_such_a_consumer_is_left_alone() -> None:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        y = (x + x).reshape(8, 4, 2, 16, 4)
        return _topk(y)

    ops = _canon(fn, _filled((8, 4, 2, 4, 16)), gate="0")
    assert list(_named(ops, "add")[0].outputs[0].shape) == [8, 4, 2, 4, 16]


def test_a_view_into_a_low_rank_shape_frees_the_producer() -> None:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return F.layer_norm((x + x).reshape(256, 16), (16,)).sum()

    ops = _canon(fn, _filled((8, 4, 2, 4, 16)), gate="0")
    assert len(_named(ops, "add")[0].outputs[0].shape) == 1


def test_a_producer_read_by_a_rewritten_op_is_rewritten() -> None:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return (x + x) * 2.0

    ops = _canon(fn, _filled((8, 4, 2, 4, 16)), gate="0")
    assert _wide(ops) == []


# ── the workaround at the shapes MPSGraph's fused attention miscompiled ───────────────


def _np_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = np.einsum("...id,...jd->...ij", q, k) * 0.25
    s = s - s.max(-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(-1, keepdims=True)
    return np.einsum("...ij,...jd->...id", w, v)


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize(
    "shape",
    [(8, 3, 4, 17, 64), (8, 3, 4, 20, 64), (3, 2, 6, 24, 64), (4, 2, 3, 4, 18, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_a_rank_5_attention_in_the_miscompile_window_matches_numpy(shape: Shape) -> None:
    rng = np.random.default_rng(shape[-2] * 100 + shape[0])
    arrs = [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]
    args = tuple(lucid.tensor(a.copy(), device=COMPILE_DEVICE) for a in arrs)
    with mock.patch.dict(os.environ, {ENV: "0"}):
        got = lucid.compile(lambda a, b, c: _attention(a, b, c))(*args).numpy()
    assert float(np.abs(got - _np_attention(*arrs)).max()) < 1e-3


# ── values ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(params=[1, 0], ids=["workaround_applied", "fast_path"])
def attention_state(request: pytest.FixtureRequest) -> Iterator[int]:
    """The fused-attention workaround on (1, as on a GPU the probe finds affected) and off (0).

    On, ``softmax(QK) @ V`` is emitted transposed and runs as plain ops at any rank, so the
    rewritten graph equals the unrewritten one to the bit at float32.  Off, MPSGraph fuses the
    rank-4 chain into its attention kernel but not the rank-5 one: the two differ by an ulp.
    """
    before = _C_engine.compile.attention_workaround_state()
    _C_engine.compile.set_attention_workaround_state(request.param)
    yield int(request.param)
    _C_engine.compile.set_attention_workaround_state(before)


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("shape", [_BLOCK, _RANK_6], ids=["rank5", "rank6"])
def test_a_rank_5_attention_matches_eager_and_the_unrewritten_graph(
    shape: Shape, dtype: lucid.dtype, attention_state: int
) -> None:
    q, k, v = (_on_metal(_filled(shape, ph) - 1.0, dtype) for ph in (0.0, 1.0, 2.0))
    _check_forward(_attention, q, k, v, exact=attention_state == 1 and dtype is lucid.float32)


class _Attention(nn.Module):
    def __init__(
        self, shape: Shape, dtype: lucid.dtype, p: float = 0.0, train: bool = False
    ) -> None:
        super().__init__()
        self.p, self.train_mode = p, train
        self.q = nn.Parameter(_on_metal(_filled(shape, 0.0) - 1.0, dtype))
        self.k = nn.Parameter(_on_metal(_filled(shape, 1.0) - 1.0, dtype))
        self.v = nn.Parameter(_on_metal(_filled(shape, 2.0) - 1.0, dtype))

    def forward(self, s: lucid.Tensor) -> lucid.Tensor:
        return _attention(self.q * s, self.k, self.v, self.p, self.train_mode)


@needs_metal
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("shape", [_BLOCK, _RANK_6], ids=["rank5", "rank6"])
def test_a_rank_5_attention_step_matches_eager_and_the_unrewritten_graph(
    shape: Shape, dtype: lucid.dtype, attention_state: int
) -> None:
    arg = _on_metal(lucid.ones(1), dtype)
    _check_step(
        lambda: _Attention(shape, dtype),
        arg,
        dtype,
        exact=attention_state == 1 and dtype is lucid.float32,
    )


@needs_metal
def test_a_training_dropout_attention_step_runs_with_finite_gradients() -> None:
    """The mask is drawn per run, so the values cannot be held to eager's: only to sanity."""
    model = _Attention(_BLOCK, lucid.float32, 0.1, True).to(COMPILE_DEVICE)
    with mock.patch.dict(os.environ, {ENV: "0"}):
        loss = make_step(model, lambda y: (y * y).sum())(_on_metal(lucid.ones(1), lucid.float32))
        loss.backward()
    assert abs(float(loss.item())) < float("inf")
    for p in model.parameters():
        assert p.grad is not None
        assert all(abs(g) < float("inf") for g in p.grad.reshape(-1).tolist())


# ── time ──────────────────────────────────────────────────────────────────────────────


def _gated(
    fn: Callable[..., lucid.Tensor], args: tuple[lucid.Tensor, ...], gate: str
) -> Callable[[], float]:
    return with_env(compiled_call(fn, *args), ENV, gate)


_DEFAULT_GATE = "64"


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


def _scaled_attention(q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
    weights = F.softmax((q @ k.mT) * 0.17, dim=-1)
    return (weights @ v).reshape(-1).sum()


def _norm_after_add(x: lucid.Tensor) -> lucid.Tensor:
    return (F.layer_norm(x + x, (96,)) * 2.0).reshape(-1).sum()


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


@pytest.mark.parametrize(
    "consumer",
    [
        lambda y: F.layer_norm(y, (16,)),
        lambda y: F.rms_norm(y, (16,)),
        lambda y: F.group_norm(y, 2),
    ],
    ids=["layer_norm", "rms_norm", "group_norm"],
)
def test_a_norm_consumer_reads_the_producers_view_for_free(
    consumer: Callable[[lucid.Tensor], lucid.Tensor],
) -> None:
    """The norm emitters build a rank <= 4 layout themselves (LCD-927), so the producer is
    rewritten as for a reduction: 3.70 ms against 2.15 when the guard dropped it."""

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return consumer(x * 1.5).sum()

    ops = _canon(fn, _filled((8, 4, 2, 8, 16)), gate="0")
    assert len(_named(ops, "mul")[0].outputs[0].shape) < 5
