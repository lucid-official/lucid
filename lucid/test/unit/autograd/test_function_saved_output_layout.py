"""A custom ``Function`` that saves an output keeps its values, whatever its layout.

``save_for_backward`` on one of the node's own outputs stores a detached alias
of it, not the output (the node must not hold what holds the node).  On the
CPU the alias was re-owned onto the buffer ``TensorImpl::saved_output_storage``
split off — which is the buffer only for a dense tensor.  For a transposed,
strided or expanded output that call returns a packed copy in another
allocation, so the alias kept the copy alive and read the original buffer
through the output's strides.  Once the output's Python wrapper died the
buffer was freed, and ``backward`` read whatever was allocated over it: a
silent wrong gradient (``lstsq`` of a wide matrix, LCD-941).

Each layout is run through the same forward-then-drop-then-backward sequence
and the saved values compared with what ``forward`` saw.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
from lucid.autograd import Function, FunctionCtx

_SHAPE = (3, 4)
_LAYOUTS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "dense": lambda t: t.contiguous(),
    "transposed": lambda t: t.mT,
    "permuted": lambda t: t.reshape(3, 2, 2).permute(2, 0, 1),
    "stepped": lambda t: t[:, ::2],
    "expanded": lambda t: t[:1].expand(3, 4),
}


class _SavesTheViewItReturns(Function):
    """``y = layout(2 x)`` built from a temporary, saved as the output."""

    seen: np.ndarray

    @staticmethod
    def forward(
        ctx: FunctionCtx, x: lucid.Tensor, layout: Callable[[lucid.Tensor], lucid.Tensor]
    ) -> lucid.Tensor:
        y = layout(x.detach() * 2.0)
        ctx.save_for_backward(y)
        ctx.x_shape = tuple(x.shape)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, g: lucid.Tensor) -> tuple[lucid.Tensor, None]:
        (y,) = ctx.saved_tensors
        _SavesTheViewItReturns.seen = y.numpy().copy()
        return lucid.ones(ctx.x_shape, dtype=g.dtype, device=g.device), None


def _forward_and_drop(kind: str, device: str) -> tuple[lucid.Tensor, np.ndarray]:
    """The loss of one call, with the Function's output already gone."""
    x_np = np.arange(12.0).reshape(_SHAPE)
    x = lucid.tensor(x_np, dtype=lucid.float32, requires_grad=True, device=device)
    out = _SavesTheViewItReturns.apply(x, _LAYOUTS[kind])
    want = _LAYOUTS[kind](lucid.tensor(x_np * 2.0, dtype=lucid.float32, device=device)).numpy()
    rhs = lucid.ones((int(out.shape[-1]), 2), dtype=out.dtype, device=device)
    # ``matmul`` reads ``out`` into storage of its own, so nothing but the
    # Function's context is left holding the output's buffer.
    return lucid.matmul(out, rhs).sum(), want


@pytest.mark.parametrize("kind", list(_LAYOUTS))
def test_the_saved_output_reads_the_values_forward_saw(kind: str, device: str) -> None:
    loss, want = _forward_and_drop(kind, device)
    # Anything allocated now lands on a buffer the dropped output used to own.
    junk = [
        lucid.full((n,), -7.0, dtype=lucid.float32, device=device)
        for n in range(1, 40)
        for _ in range(2)
    ]
    loss.backward()
    np.testing.assert_array_equal(_SavesTheViewItReturns.seen, want)
    del junk


class _ExpTransposed(Function):
    """``y = exp(x)ᵀ`` from a temporary; its backward reads the saved output."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
        y = lucid.exp(x.detach()).mT
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
        (y,) = ctx.saved_tensors
        return g.mT * y.mT


def test_a_second_derivative_through_a_saved_transposed_output_is_exact(device: str) -> None:
    """``create_graph`` keeps the saved output in the graph of the first gradient."""
    x_np = np.linspace(-0.5, 0.5, 12).reshape(3, 4)
    rhs_np = np.cos(np.arange(6.0)).reshape(3, 2)
    w_np = np.sin(np.arange(8.0)).reshape(4, 2)
    x = lucid.tensor(x_np, dtype=lucid.float32, requires_grad=True, device=device)

    def loss() -> lucid.Tensor:
        out = _ExpTransposed.apply(x)
        rhs = lucid.tensor(rhs_np, dtype=lucid.float32, device=device)
        return (
            lucid.matmul(out, rhs) * lucid.tensor(w_np, dtype=lucid.float32, device=device)
        ).sum()

    (g1,) = lucid.autograd.grad(loss(), [x], create_graph=True)
    junk = [lucid.full((n,), -7.0, dtype=lucid.float32, device=device) for n in range(1, 40)]
    (g2,) = lucid.autograd.grad((g1 * g1).sum(), [x])
    dl_dy = w_np @ rhs_np.T  # (4, 3)
    want = 2.0 * dl_dy.T**2 * np.exp(2.0 * x_np)
    np.testing.assert_allclose(g2.numpy(), want, rtol=1e-4)
    del junk
