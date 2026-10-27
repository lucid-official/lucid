"""A node refuses an in-place write into exactly the tensors its backward reads.

Two halves of one rule, both broken before LCD-320 / LCD-306:

* **No more.**  ``validate_versions`` checked every input a node was given.
  ``add``, ``sum``, ``neg`` or a view read none of their input's values, yet
  a write into that input after forward was refused — on the CPU, where a
  column slice is a view sharing its base's version, even
  ``y[:, 1] = y[:, 0] + 5; y.sum().backward()`` raised, where the reference
  and Metal succeed.
* **No fewer.**  A node that keeps a tensor's values must guard its version.
  ``LstmBackward`` recorded versions and never compared them; the
  ``scatter_reduce`` and ``lu_solve`` backwards held operands as tensors and
  read whatever a later write left in them.

The rule is enforced in one place per kind of node — ``AutogradNode`` guards
the inputs it saved by value — and ``tools/check_saved_versions.py`` refuses
a node class that holds a tensor without saying how it is guarded.  Each case
here is checked against the reference: the same graph, the same write, the
same verdict.
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from tools import check_saved_versions


def _const(lib: Any, like: Any, values: Any) -> Any:
    """A constant beside ``like``, built by ``lib``."""
    return lib.tensor(values, device=like.device)


# A backward that reads none of its input's values: ``v`` is that input.
# Each builds the same thing from Lucid or the reference (``lib``); the
# constant factors only make the gradient differ from position to position.
READS_NOTHING: dict[str, Callable[[Any, Any], Any]] = {
    "add": lambda lib, v: v + 1.0,
    "sub": lambda lib, v: 1.0 - v,
    "neg": lambda lib, v: -v,
    "sum": lambda lib, v: v.sum(),
    "mean": lambda lib, v: v.mean(),
    "reshape": lambda lib, v: v.reshape(1, 2) * 3.0,
    "unsqueeze": lambda lib, v: v.unsqueeze(0) * 3.0,
    "flip": lambda lib, v: v.flip(0) * _const(lib, v, [1.0, 2.0]),
    "roll": lambda lib, v: v.roll([1], [0]) * _const(lib, v, [1.0, 2.0]),
    "cat": lambda lib, v: lib.cat([v, v]) * _const(lib, v, [1.0, 2.0, 3.0, 4.0]),
    "stack": lambda lib, v: lib.stack([v, v]).sum(0),
    "repeat": lambda lib, v: v.repeat(2) * _const(lib, v, [1.0, 2.0, 3.0, 4.0]),
    "contiguous": lambda lib, v: v.contiguous() * 2.0,
    "clone": lambda lib, v: v.clone() * 2.0,
    "cumsum": lambda lib, v: v.cumsum(0),
}

# The writes made into ``y`` (whose column 0 is ``v``) after the forward.
WRITES: dict[str, Callable[[Any, Any], object]] = {
    "setitem through the base": lambda y, v: y.__setitem__((slice(None), 1), 7.0),
    "setitem reading the view": lambda y, v: y.__setitem__((slice(None), 1), y[:, 0] + 5.0),
    "in-place on the base": lambda y, v: y.add_(1.0),
    "in-place on the view": lambda y, v: v.mul_(2.0),
}


def _grad_reading_nothing(lib: Any, device: str | None, op: str, write: str) -> np.ndarray:
    kw = {} if device is None else {"device": device}
    w = lib.arange(6.0, **kw).reshape(2, 3).requires_grad_()
    y = w * 1.0
    v = y[:, 0]
    out = READS_NOTHING[op](lib, v)
    WRITES[write](y, v)
    out.sum().backward()
    return np.asarray(w.grad.detach().cpu().numpy())


@pytest.mark.parametrize("write", list(WRITES))
@pytest.mark.parametrize("op", list(READS_NOTHING))
def test_a_write_into_an_input_the_backward_never_reads_is_allowed(
    op: str, write: str, device: str, ref: ModuleType
) -> None:
    got = _grad_reading_nothing(lucid, device, op, write)
    want = _grad_reading_nothing(ref, None, op, write)
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_the_reported_repro(device: str) -> None:
    # LCD-320, verbatim.
    w = lucid.arange(6.0, device=device).reshape(2, 3).requires_grad_()
    y = w * 1
    y[:, 1] = y[:, 0] + 5
    y.sum().backward()
    assert w.grad is not None
    assert w.grad.tolist() == [[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]]


def test_a_copy_from_a_view_of_the_destination(device: str, ref: ModuleType) -> None:
    # ``copy_`` makes its source contiguous; the contiguous node never reads
    # the source's values, and the write it feeds moves the source's (shared)
    # version — which that node refused before (MV-ENG-1).
    def run(lib: Any, kw: dict[str, str]) -> np.ndarray:
        x = lib.tensor([1.0, -2.0, 3.0, 0.5], **kw).requires_grad_()
        y = x * 2
        y.copy_(y.view(2, 2).view(4))
        (y * lib.tensor([1.0, 2.0, 3.0, 4.0], **kw)).sum().backward()
        return np.asarray(x.grad.detach().cpu().numpy())

    np.testing.assert_allclose(run(lucid, {"device": device}), run(ref, {}))


# ── a bias is read by shape alone ─────────────────────────────────────────────


def _bias_written(lib: Any, device: str | None, layer: str) -> list[np.ndarray]:
    kw = {} if device is None else {"device": device}
    rng = np.random.default_rng(0)
    if layer == "linear":
        x = lib.tensor(rng.standard_normal((2, 3)).astype(np.float32), **kw)
        W = lib.tensor(rng.standard_normal((4, 3)).astype(np.float32), **kw)
        b = lib.tensor(rng.standard_normal(4).astype(np.float32), **kw)
        leaves = [x.requires_grad_(), W.requires_grad_(), b.requires_grad_()]
        out = F.linear(x, W, b) if lib is lucid else lib.nn.functional.linear(x, W, b)
    else:
        x = lib.tensor(rng.standard_normal((1, 2, 4, 4)).astype(np.float32), **kw)
        W = lib.tensor(rng.standard_normal((3, 2, 3, 3)).astype(np.float32), **kw)
        b = lib.tensor(rng.standard_normal(3).astype(np.float32), **kw)
        leaves = [x.requires_grad_(), W.requires_grad_(), b.requires_grad_()]
        conv = F.conv2d if lib is lucid else lib.nn.functional.conv2d
        out = conv(x, W, b)
    with lib.no_grad():
        b.add_(1.0)
    (out * out.detach()).sum().backward()
    return [np.asarray(t.grad.detach().cpu().numpy()) for t in leaves]


def _recurrent_bias_written(lib: Any, device: str | None, kind: str) -> list[np.ndarray]:
    """``kind`` run once, its ``bias_ih`` written after forward, then backward.

    The parameters come from one seeded generator in the reference's order
    (``weight_ih, weight_hh, bias_ih, bias_hh``), so both libraries hold the
    same values whatever their parameter names.
    """
    kw = {} if device is None else {"device": device}
    rng = np.random.default_rng(3)
    cell = getattr(lib.nn, kind)(3, 4)
    if lib is lucid:
        cell = cell.to(device)
    params = list(cell.parameters())
    with lib.no_grad():
        for p in params:
            p.copy_(lib.tensor(rng.standard_normal(tuple(p.shape)).astype(np.float32), **kw))
    x = lib.tensor(rng.standard_normal((5, 2, 3)).astype(np.float32), **kw).requires_grad_()
    out, _ = cell(x)
    with lib.no_grad():
        params[2].add_(1.0)
    (out * out.detach()).sum().backward()
    return [np.asarray(t.grad.detach().cpu().numpy()) for t in [x, *params]]


@pytest.mark.parametrize("layer", ["linear", "conv2d", "LSTM", "GRU", "RNN"])
def test_a_write_into_a_bias_after_forward_is_allowed(
    layer: str, device: str, ref: ModuleType
) -> None:
    # A bias is read by shape alone — its gradient is the summed incoming
    # gradient — so no node holds it and a write into it after forward is
    # allowed; the gradients are the ones of the forward that ran.  LSTM's
    # BPTT reads the weight matrices only (LCD-320 FIX round).
    if layer in ("linear", "conv2d"):
        got = _bias_written(lucid, device, layer)
        want = _bias_written(ref, None, layer)
    else:
        got = _recurrent_bias_written(lucid, device, layer)
        want = _recurrent_bias_written(ref, None, layer)
    for g, r in zip(got, want, strict=True):
        np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-5)


def test_a_global_response_norm_beta_is_guarded(device: str) -> None:
    # Unlike a norm's shift, GRN's beta multiplies x (gamma * x * Nx + beta *
    # x), so dx reads it and a write into it after forward is refused.
    x = lucid.randn(1, 2, 3, 3, device=device).requires_grad_()
    gamma = lucid.ones(2, device=device).requires_grad_()
    beta = (lucid.full((2,), 0.5, device=device).requires_grad_()) * 1.0
    out = _wrap(_C_engine.nn.global_response_norm(_unwrap(x), _unwrap(gamma), _unwrap(beta)))
    try:
        beta.mul_(2.0)
    except RuntimeError as exc:
        assert device == "cpu" and "saved for backward" in str(exc)
        return
    with pytest.raises(_C_engine.VersionMismatch, match="global_response_norm input 2"):
        out.sum().backward()


# ── a node that reads a tensor refuses a write into it ────────────────────────
#
# The saved tensor ``x`` is a column slice of a wider base, so on the CPU it is
# a view that is not dense: the node keeps a packed copy, the write into ``x``
# is allowed, and it is the version check that has to refuse.  Where the CPU
# refuses the write itself (a tensor whose very bytes a node holds — the whole
# tensor ``lstm`` takes), that is the same verdict, given earlier.
#
# The other operands require grad: the reference saves ``x`` only for a
# gradient that reads it, and so, now, does the check here.


def _cols(lib: Any, kw: dict[str, str], shape: tuple[int, ...], seed: int) -> Any:
    """A tensor of ``shape`` with one extra trailing column, requiring grad."""
    rng = np.random.default_rng(seed)
    full = rng.standard_normal((*shape[:-1], shape[-1] + 1)).astype(np.float32)
    return lib.tensor(full, **kw).requires_grad_()


def _param(lib: Any, like: Any, *shape: int) -> Any:
    """An operand beside ``like`` that takes a gradient."""
    return lib.full(shape, 0.5, device=like.device).requires_grad_()


def _fn(lib: Any, name: str) -> Callable[..., Any]:
    return getattr(F if lib is lucid else lib.nn.functional, name)


def _lstm(lib: Any, x: Any) -> Any:
    lib.manual_seed(0)
    cell = nn.LSTM(3, 4) if lib is lucid else lib.nn.LSTM(3, 4)
    if lib is lucid:
        cell = cell.to(x.device)
    out, _ = cell(x)
    return out


# name -> (shape of the saved input, the loss built from it)
SAVES: dict[str, tuple[tuple[int, ...], Callable[[Any, Any], Any]]] = {
    "mul": ((2, 3), lambda lib, x: x * _param(lib, x, 2, 3)),
    "div": ((2, 3), lambda lib, x: _param(lib, x, 2, 3) / x),
    "matmul": ((2, 3), lambda lib, x: x @ _param(lib, x, 3, 2)),
    "dot": ((3,), lambda lib, x: x.dot(_param(lib, x, 3))),
    "outer": ((3,), lambda lib, x: x.outer(_param(lib, x, 2))),
    "var": ((2, 3), lambda lib, x: x.var()),
    "cumprod": ((2, 3), lambda lib, x: x.cumprod(1)),
    "pow": ((2, 3), lambda lib, x: x**2),
    "linear input": (
        (2, 3),
        lambda lib, x: _fn(lib, "linear")(x, _param(lib, x, 4, 3)),
    ),
    "conv2d input": (
        (1, 1, 4, 4),
        lambda lib, x: _fn(lib, "conv2d")(x, _param(lib, x, 2, 1, 3, 3)),
    ),
    "lstm input": ((5, 2, 3), _lstm),
}


def _saved_then_written(lib: Any, device: str | None, case: str) -> tuple[Any, Any, Any]:
    """Build ``case`` on a slice ``x``, then double ``x`` in place.

    Returns the loss, whatever the write raised (``None`` if it went in), and
    ``x`` — kept alive, since a node that holds it weakly (``lstm``) skips
    the check for a tensor that is gone, and nothing can then observe it.
    """
    kw = {} if device is None else {"device": device}
    shape, build = SAVES[case]
    if case == "lstm input":
        # LSTM flattens its input, which copies a slice, so ``x`` is a whole
        # tensor here; the CPU then refuses the write into bytes it holds.
        rng = np.random.default_rng(1)
        x = lib.tensor(rng.standard_normal(shape).astype(np.float32), **kw)
        x = x.requires_grad_() * 1.0
    else:
        x = (_cols(lib, kw, shape, seed=1) * 1.0)[..., :-1]
    loss = build(lib, x).sum()
    try:
        x.mul_(2.0)
    except RuntimeError as exc:  # the CPU refuses a write into bytes a node holds
        return loss, exc, x
    return loss, None, x


@pytest.mark.parametrize("case", list(SAVES))
def test_a_write_into_a_tensor_the_backward_reads_is_refused(
    case: str, device: str, ref: ModuleType
) -> None:
    ref_loss, ref_refused, _ = _saved_then_written(ref, None, case)
    assert ref_refused is None
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        ref_loss.backward()

    loss, refused, _x = _saved_then_written(lucid, device, case)
    if refused is not None:
        assert device == "cpu" and "saved for backward" in str(refused)
        return
    with pytest.raises(_C_engine.VersionMismatch):
        loss.backward()


# name -> (shape of ``x``, loss from ``x`` and a constant): no gradient that is
# needed reads ``x``, so neither the reference nor Lucid saves or guards it.
CONSTANT_OPERAND: dict[str, tuple[tuple[int, ...], Callable[[Any, Any], Any]]] = {
    "mul": (
        (2, 3),
        lambda lib, x: x * _const(lib, x, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ),
    "rmul": (
        (2, 3),
        lambda lib, x: _const(lib, x, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * x,
    ),
    "div": (
        (2, 3),
        lambda lib, x: x / _const(lib, x, [[1.0, 2.0, 4.0], [8.0, 2.0, 1.0]]),
    ),
    "fmod": (
        (2, 3),
        lambda lib, x: lib.fmod(x, _const(lib, x, [[1.5, 2.0, 4.0], [8.0, 2.5, 1.5]])),
    ),
    "remainder": (
        (2, 3),
        lambda lib, x: lib.remainder(x, _const(lib, x, [[1.5, 2.0, 4.0], [8.0, 2.5, 1.5]])),
    ),
    "matmul": (
        (2, 3),
        lambda lib, x: x @ _const(lib, x, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    ),
    "rmatmul": (
        (3, 2),
        lambda lib, x: _const(lib, x, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) @ x,
    ),
    "mm": (
        (2, 3),
        lambda lib, x: lib.mm(x, _const(lib, x, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
    ),
    "bmm": (
        (2, 2, 3),
        lambda lib, x: lib.bmm(
            x,
            _const(
                lib,
                x,
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    [[-1.0, 0.5], [2.0, 1.0], [0.5, -2.0]],
                ],
            ),
        ),
    ),
    "mv": (
        (2, 3),
        lambda lib, x: lib.mv(x, _const(lib, x, [1.0, 2.0, 3.0])),
    ),
    "dot": (
        (3,),
        lambda lib, x: x.dot(_const(lib, x, [1.0, 2.0, 3.0])),
    ),
    "outer": (
        (3,),
        lambda lib, x: x.outer(_const(lib, x, [1.0, 2.0])),
    ),
    "linear": (
        (2, 3),
        lambda lib, x: _fn(lib, "linear")(x, _const(lib, x, [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])),
    ),
    "linear weight": (
        (2, 3),
        lambda lib, w: _fn(lib, "linear")(_const(lib, w, [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]), w),
    ),
    "addcmul": (
        (2, 3),
        lambda lib, x: lib.addcmul(
            _const(lib, x, [[0.5, 1.0, -0.5], [2.0, 0.0, 1.0]]),
            x,
            _const(lib, x, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ),
    ),
    "where": (
        (2, 3),
        lambda lib, x: lib.where(
            lib.tensor([[True, False, True], [False, True, False]], device=x.device),
            x,
            _const(lib, x, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ),
    ),
    "ldexp": (
        (2, 3),
        lambda lib, x: lib.ldexp(
            x,
            lib.tensor([[1, 2, 0], [0, 3, 1]], dtype=lib.int32, device=x.device),
        ),
    ),
}

# Lucid's ConvNdBackward / ConvTransposeNdBackward / NextafterBackward also
# declare kGradReads / saves=false so ``dx`` does not save ``x`` and ``dW`` does
# not save ``W`` when the peer is a constant (the reference saves both).
CONV_CONSTANT_OPERAND: dict[str, tuple[tuple[int, ...], Callable[[Any, Any], Any]]] = {
    "nextafter": (
        (2, 3),
        lambda lib, x: lib.nextafter(x, _const(lib, x, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
    ),
    "conv2d": (
        (1, 1, 3, 3),
        lambda lib, x: _fn(lib, "conv2d")(x, _const(lib, x, [[[[1.0, -0.5], [0.25, 2.0]]]])),
    ),
    "conv2d weight": (
        (1, 1, 2, 2),
        lambda lib, w: _fn(lib, "conv2d")(
            _const(lib, w, [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
            w,
        ),
    ),
    "conv_transpose2d": (
        (1, 1, 2, 2),
        lambda lib, x: _fn(lib, "conv_transpose2d")(
            x, _const(lib, x, [[[[1.0, -0.5], [0.25, 2.0]]]])
        ),
    ),
    "conv_transpose2d weight": (
        (1, 1, 2, 2),
        lambda lib, w: _fn(lib, "conv_transpose2d")(
            _const(lib, w, [[[[1.0, 2.0], [3.0, 4.0]]]]), w
        ),
    ),
}


def _constant_operand_grad(
    lib: Any,
    device: str | None,
    case: str,
    *,
    sliced: bool = False,
    create_graph: bool = False,
    mutate: bool = True,
) -> np.ndarray:
    kw = {} if device is None else {"device": device}
    table = CONSTANT_OPERAND if case in CONSTANT_OPERAND else CONV_CONSTANT_OPERAND
    shape, build = table[case]
    if sliced:
        leaf = _cols(lib, kw, shape, seed=2)
        x = (leaf * 1.0)[..., :-1]
    else:
        rng = np.random.default_rng(2)
        leaf = lib.tensor(rng.standard_normal(shape).astype(np.float32), **kw).requires_grad_()
        x = leaf * 1.0
    out = build(lib, x)
    if mutate:
        x.mul_(3.0)
    (out * out.detach()).sum().backward(create_graph=create_graph)
    return np.asarray(leaf.grad.detach().cpu().numpy())


@pytest.mark.parametrize("sliced", [False, True], ids=["whole", "sliced"])
@pytest.mark.parametrize("case", list(CONSTANT_OPERAND))
def test_a_write_into_an_operand_only_an_unneeded_gradient_reads_is_allowed(
    case: str, sliced: bool, device: str, ref: ModuleType
) -> None:
    got = _constant_operand_grad(lucid, device, case, sliced=sliced)
    want = _constant_operand_grad(ref, None, case, sliced=sliced)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("create_graph", [False, True], ids=["eager", "create_graph"])
@pytest.mark.parametrize("sliced", [False, True], ids=["whole", "sliced"])
@pytest.mark.parametrize("case", list(CONV_CONSTANT_OPERAND))
def test_conv_with_constant_peer_allows_inplace_write(
    case: str, sliced: bool, create_graph: bool, device: str, ref: ModuleType
) -> None:
    got = _constant_operand_grad(lucid, device, case, sliced=sliced, create_graph=create_graph)
    want = _constant_operand_grad(
        ref, None, case, sliced=sliced, create_graph=create_graph, mutate=False
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "case",
    [
        "mul",
        "div",
        "fmod",
        "remainder",
        "matmul",
        "dot",
        "outer",
        "linear",
        "linear weight",
        "addcmul",
    ],
)
def test_create_graph_backward_with_constant_operand_after_inplace_write(
    case: str, device: str, ref: ModuleType
) -> None:
    got = _constant_operand_grad(lucid, device, case, create_graph=True)
    want = _constant_operand_grad(ref, None, case, create_graph=True)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("which", ["input", "weight"])
def test_lstm_refuses_a_write_into_what_bptt_reads(which: str, device: str) -> None:
    # LCD-306: the node recorded versions and never compared them.
    lucid.manual_seed(0)
    cell = nn.LSTM(3, 4).to(device)
    x = (lucid.randn(5, 2, 3, device=device).requires_grad_()) * 1.0
    out, _ = cell(x)
    target = x if which == "input" else next(iter(cell.parameters()))
    try:
        with lucid.no_grad():
            target.mul_(2.0)
    except RuntimeError as exc:
        assert device == "cpu" and "saved for backward" in str(exc)
        return
    with pytest.raises(_C_engine.VersionMismatch, match="lstm input"):
        out.sum().backward()


def test_lstm_allows_a_write_into_c0(device: str) -> None:
    # ``c0`` reaches BPTT as the first cell state the forward wrote; the
    # node holds no reference to it, so a write into it changes nothing.
    lucid.manual_seed(0)
    cell = nn.LSTM(3, 4).to(device)
    x = lucid.randn(5, 2, 3, device=device).requires_grad_()
    h0 = lucid.zeros(1, 2, 4, device=device)
    c0 = lucid.randn(1, 2, 4, device=device)

    def x_grad(written: bool) -> np.ndarray:
        x.grad = None
        c = c0.clone()
        out, _ = cell(x, (h0, c))
        if written:
            c.zero_()
        out.sum().backward()
        assert x.grad is not None
        return np.asarray(x.grad.tolist())

    np.testing.assert_allclose(x_grad(True), x_grad(False), rtol=1e-5)


def test_scatter_reduce_refuses_a_write_into_its_operands(device: str, ref: ModuleType) -> None:
    # The node held base and src as tensors and read their live values.
    def run(lib: Any, kw: dict[str, str]) -> Any:
        base = lib.tensor([5.0, 0.0, 5.0, 0.0], **kw).requires_grad_() * 1.0
        src = lib.tensor([1.0, 2.0, 3.0, 4.0], **kw).requires_grad_() * 1.0
        idx = lib.tensor([0, 1, 2, 3], **kw)
        out = lib.scatter_reduce(base, 0, idx, src, "amax")
        src.mul_(10.0)
        return out.sum()

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        run(ref, {}).backward()
    with pytest.raises(_C_engine.VersionMismatch, match="scatter_amax input 2"):
        run(lucid, {"device": device}).backward()


# ── selectors held as tensors ─────────────────────────────────────────────────
#
# scatter_add / scatter / scatter_reduce hold the index tensor and lu_solve
# the pivots, and read them at backward.  The public path keeps an int32
# index (and int32 pivots) as the very tensor it was given — an int64 one is
# converted into a fresh tensor first (LCD-351), which a later write into
# the caller's tensor cannot reach — so these use int32.

_INDEX_WRITERS: dict[str, Callable[[Any, Any, Any], Any]] = {
    "scatter_add": lambda base, idx, src: base.scatter_add(0, idx, src),
    "scatter": lambda base, idx, src: base.scatter(0, idx, src),
    "scatter_reduce amax": lambda base, idx, src: lucid.scatter_reduce(base, 0, idx, src, "amax"),
}
_INDEX_NAMES = {
    "scatter_add": "scatter_add input 1",
    "scatter": "scatter input 2",
    "scatter_reduce amax": "scatter_amax input 1",
}


@pytest.mark.parametrize("op", list(_INDEX_WRITERS))
def test_a_write_into_a_held_index_is_refused(op: str, device: str) -> None:
    base = lucid.zeros(4, device=device).requires_grad_()
    src = lucid.arange(1.0, 5.0, device=device).requires_grad_()
    idx = lucid.tensor([0, 1, 2, 3], dtype=lucid.int32, device=device)
    out = _INDEX_WRITERS[op](base, idx, src)
    idx.copy_(lucid.tensor([3, 2, 1, 0], dtype=lucid.int32, device=device))
    with pytest.raises(_C_engine.VersionMismatch, match=_INDEX_NAMES[op]):
        (out * lucid.arange(1.0, 5.0, device=device)).sum().backward()


def test_a_write_into_lu_solve_pivots_is_refused(device_cpu_only: str) -> None:
    from lucid.linalg import lu_factor, lu_solve

    # Batched, so the pivots already have the shape the solve aligns them to
    # and reach the node as the caller's own tensor.
    A = lucid.tensor([[[0.0, 2.0], [3.0, 1.0]]], device=device_cpu_only)
    LU, piv = lu_factor(A)
    LU = LU.detach().requires_grad_()
    b = lucid.tensor([[[1.0], [2.0]]], device=device_cpu_only).requires_grad_()
    x = lu_solve(LU, piv, b)
    piv.copy_(lucid.tensor([[1, 2]], dtype=lucid.int32, device=device_cpu_only))
    with pytest.raises(_C_engine.VersionMismatch, match="lu_solve input 1"):
        x.sum().backward()


def test_a_write_into_lu_solve_b_is_named_as_argument_2(device_cpu_only: str) -> None:
    from lucid.linalg import lu_factor, lu_solve

    A = lucid.tensor([[[0.0, 2.0], [3.0, 1.0]]], device=device_cpu_only)
    LU, piv = lu_factor(A)
    LU = LU.detach().requires_grad_()
    wide = lucid.tensor([[[1.0, 9.0], [2.0, 9.0]]], device=device_cpu_only).requires_grad_()
    b = (wide * 1.0)[..., :1]
    x = lu_solve(LU, piv, b)
    b.mul_(2.0)
    with pytest.raises(_C_engine.VersionMismatch, match="lu_solve input 2"):
        x.sum().backward()


def _embedding_bag_written(lib: Any, mode: str, kw: dict[str, str]) -> np.ndarray:
    w = lib.tensor(np.arange(12.0, dtype=np.float32).reshape(4, 3), **kw)
    w.requires_grad_()
    idx = lib.tensor([0, 2, 1, 3], **kw)
    off = lib.tensor([0, 2], **kw)
    out = _fn(lib, "embedding_bag")(idx, w, off, mode=mode)
    with lib.no_grad():
        w.add_(1.0)
    (out * lib.tensor([[1.0, 2.0, 3.0]], **kw)).sum().backward()
    return np.asarray(w.grad.detach().cpu().numpy())


@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_embedding_bag_sum_and_mean_allow_a_write_into_the_weight(
    mode: str, device: str, ref: ModuleType
) -> None:
    # Only ``mode="max"`` reads the weight's values (to find each winner), so
    # ``sum`` and ``mean`` do not save the weight or refuse a later write.
    got = _embedding_bag_written(lucid, mode, {"device": device})
    np.testing.assert_allclose(got, _embedding_bag_written(ref, mode, {}))


@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_embedding_bag_sum_and_mean_allow_inplace_on_nonleaf_weight(
    mode: str, device: str, ref: ModuleType
) -> None:
    base = np.arange(15.0, dtype=np.float32).reshape(5, 3)
    x = lucid.tensor(base, device=device).requires_grad_()
    wt = x * 1.0
    y = F.embedding_bag(
        lucid.tensor([0, 1, 2, 3], device=device),
        wt,
        lucid.tensor([0, 2], device=device),
        mode=mode,
    )
    wt.add_(1.0)
    y.sum().backward()

    rx = ref.tensor(base).requires_grad_()
    rwt = rx * 1.0
    ry = ref.nn.functional.embedding_bag(
        ref.tensor([0, 1, 2, 3]), rwt, ref.tensor([0, 2]), mode=mode
    )
    rwt.add_(1.0)
    ry.sum().backward()
    np.testing.assert_allclose(x.grad.detach().cpu().numpy(), rx.grad.numpy())


def test_embedding_bag_max_refuses_a_write_into_the_weight(device: str) -> None:
    w = lucid.tensor(np.arange(12.0, dtype=np.float32).reshape(4, 3), device=device)
    w.requires_grad_()
    out = F.embedding_bag(
        lucid.tensor([0, 2, 1, 3], device=device),
        w,
        lucid.tensor([0, 2], device=device),
        mode="max",
    )
    try:
        with lucid.no_grad():
            w.mul_(-1.0)
    except RuntimeError as exc:
        assert device == "cpu" and "saved for backward" in str(exc)
        return
    with pytest.raises(_C_engine.VersionMismatch):
        out.sum().backward()


def test_every_node_class_says_how_what_it_holds_is_guarded() -> None:
    # The static half of the rule: a node member that holds a tensor is
    # either guarded by the node's validate_versions or listed with the
    # reason no write can reach the gradient through it.
    assert check_saved_versions.check() == []


_UNGUARDED = """
class FooBackward : public FuncOp<FooBackward, 1> {
public:
    Storage saved_x_;  // the input's values, kept outside saved_inputs_
    std::vector<Storage> apply(Storage g) override;
};
struct BarNode : Node {
    TensorImplPtr held_;
    std::array<std::shared_ptr<TensorImpl>, 2> arr_;
    std::vector<Storage> apply(Storage g) override;
};
class BazBackward : public FuncOp<BazBackward, 1> {
    void validate_versions() override {}
};
void wire(std::shared_ptr<BazBackward> bwd, const TensorImplPtr& out) {
    bwd->saved_output_ = out->storage();
}
class QuxBackward : public FuncOp<QuxBackward, 1> {
public:
    static constexpr bool kKeepsOwnValues = true;
    Storage mask_;
    std::vector<Storage> apply(Storage g) override;
};
"""


def test_the_check_refuses_each_way_a_node_can_escape_the_guard(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "Foo.cpp").write_text(_UNGUARDED)
    problems = check_saved_versions.check(
        tmp_path,
        {
            "BarNode::held_": check_saved_versions.GUARDED,
            "QuxBackward::mask_": "computed",
            "Gone::x_": "why",
        },
    )
    text = "\n".join(problems)
    assert "FooBackward::saved_x_ holds tensor data and is not classified" in text
    assert "FooBackward keeps a tensor of its own but does not declare kKeepsOwnValues" in text
    assert (
        "QuxBackward declares kKeepsOwnValues = true but does not define release_own_saved()"
        in text
    )
    assert "BarNode::held_ is listed as guarded but BarNode declares no" in text
    assert "BarNode::arr_ holds tensor data and is not classified" in text
    assert "an empty validate_versions override" in text
    assert "HELD lists Gone::x_" in text
    assert "ops/Foo.cpp:16: writes a saved slot directly" in text
    assert len(problems) == 8
