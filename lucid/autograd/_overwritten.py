"""The graph node an in-place overwrite leaves behind.

``y.fill_(v)`` and ``y.zero_()`` replace every value of ``y``, so what ``y``
holds afterwards no longer depends on what it held before.  The reference
framework keeps ``y`` in the graph anyway, with that derivative — zero — so
a loss built from it can still be differentiated and sends zeros upstream.
Lucid detached it instead, which made the same loss impossible to
differentiate once ``backward()`` started refusing roots with no graph.

The node is a Python :class:`~lucid.autograd.function.Function` because it
saves nothing: an engine op applied to ``y`` records ``y``'s version, and the
assignment that follows bumps it, so its backward would report an in-place
modification that the node does not care about.

The node lands on ``new`` itself, so a compile recording, which records
nothing for it (:attr:`~lucid.autograd.function.Function._trace_form`),
sees the values the ``full`` it recorded made — what a recording under
``no_grad`` writes — while ``y`` still requires grad as it does eagerly.
A detached alias would be an op of the recording, a gradient sink, and
the node noted as a custom backward the compile cannot run.
"""

from typing import final, override

from lucid._engine_scope import untraced
from lucid._factories.creation import zeros_like
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx


def _written(old: Tensor, new: Tensor) -> Tensor:
    """The node on ``new``, with nothing recorded — an overwrite in a recording."""
    with untraced():
        out = _Overwritten.apply(old, new)
    assert isinstance(out, Tensor)
    return out


@final
class _Overwritten(Function):
    """``new``'s values, with a zero derivative with respect to ``old``."""

    _trace_form = staticmethod(_written)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, old: Tensor, new: Tensor) -> Tensor:
        # As-is, not detached: see the module notes.
        return new

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        return zeros_like(grad), None


def overwritten(old: Tensor, new: Tensor) -> Tensor:
    """``new``'s values, standing in for ``old`` in the graph.

    Parameters
    ----------
    old : Tensor
        The tensor being overwritten; it requires grad and has a graph.
    new : Tensor
        The values written over it, which depend on nothing; the node is
        registered on it, so nothing else may hold it.

    Returns
    -------
    Tensor
        ``new``'s values with a zero derivative with respect to ``old``.
    """
    out = _Overwritten.apply(old, new)
    assert isinstance(out, Tensor)
    return out
