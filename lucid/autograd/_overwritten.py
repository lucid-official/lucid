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

A compile recording cannot run that node, so it takes the plain ops for it
(:attr:`~lucid.autograd.function.Function._trace_form`): a ``where`` that
never picks ``y``, whose derivative with respect to ``y`` is that same zero.
Recording only the values, a compiled backward would find no path to what
computed ``y`` and leave its gradient ``None`` where eager leaves zeros, a
difference a weight decay or a momentum reads.  A multiply by zero would not
do: it turns ``inf`` and ``NaN`` into ``NaN``.
"""

from typing import final, override

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._factories.creation import zeros_like
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx


def _written(old: Tensor, new: Tensor) -> Tensor:
    """An overwrite in a recording: ``new``'s values, with a zero derivative to ``old``.

    Reads a copy of ``old``: the ``where`` saves what it reads, and the write
    that follows bumps ``old``'s version.
    """
    target = _unwrap(new)
    never = _C_engine.zeros(list(target.shape), _C_engine.Dtype.Bool, target.device)
    return _wrap(_C_engine.where(never, _unwrap(old.clone()), target))


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
