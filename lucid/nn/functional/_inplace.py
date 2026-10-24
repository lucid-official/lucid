"""The one route ``inplace=True`` takes through ``nn.functional``.

Every activation and dropout that takes ``inplace`` used to accept the flag
and return a fresh tensor, leaving ``x`` as it was: ``F.relu(x, inplace=True)``
and ``nn.ReLU(inplace=True)`` computed the right values into an object the
caller did not hold.  Each one now computes its out-of-place result and hands
it to :func:`_in_place`, so they all refuse, record and write through views
the same way.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from lucid._ops._autocast import _autocast_suspended
from lucid._ops.composite.indexing import _write_inplace

if TYPE_CHECKING:
    from lucid._C import engine as _C_engine
    from lucid._tensor.tensor import Tensor


def _in_place(x: Tensor, name: str, out_of_place: Callable[[Tensor], Tensor]) -> Tensor:
    """Make ``out_of_place(x)`` the values of ``x``, as an in-place op does, and return ``x``.

    * A leaf that requires grad is refused while autograd records — and so
      is a view of one — before anything is written, as the reference
      refuses it.  Under :func:`lucid.no_grad` it is written and stays a
      trainable leaf.
    * ``x`` takes the result's place in the autograd graph, so the gradient
      that reaches whatever produced ``x`` is the op's own derivative.
    * The values land in ``x``'s buffer when a view shares it, so the view's
      base sees them on both CPU and Metal.

    Parameters
    ----------
    x : Tensor
        The tensor written in place and returned.
    name : str
        The op's name, for errors.
    out_of_place : callable
        The op without ``inplace``, given the tensor to read ``x`` from.

    Returns
    -------
    Tensor
        ``x`` itself, now holding the result.

    Raises
    ------
    RuntimeError
        ``x`` is, or is a view of, a leaf that requires grad while autograd
        records; or ``x``'s elements overlap (an expanded view).
    """
    _write_inplace(x, name, out_of_place)
    return x


def _keep_in_graph(out: Tensor, x: Tensor) -> Tensor:
    """Keep a piecewise composite's ``create_graph`` gradient in the graph
    with derivative zero, as the reference's backward ops do."""
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap

    if not (x.requires_grad and _C_engine.grad_enabled()):
        return out
    xi = _unwrap(x)
    oi = _unwrap(out)
    # Graph plumbing, not arithmetic, so it has no row.  Under a Metal
    # autocast ``sum``'s FP32 row copied the whole zero to float32, and the
    # float32 zero beside a float16 ``out`` was refused by ``add``.
    with _autocast_suspended(oi.device):
        safe = _C_engine.nan_to_num(xi, 0.0, 0.0, 0.0)
        zero = _C_engine.relu(_C_engine.clip(safe, 0.0, 0.0))
        if zero.shape != oi.shape:
            zero = _C_engine.sum(zero, [], False)
        if zero.dtype != oi.dtype:
            zero = _C_engine.astype(zero, oi.dtype)
        return _wrap(_C_engine.add(oi, zero))


def _dead_zone_zero(x: Tensor) -> _C_engine.TensorImpl:
    """Zero inside a shrinkage dead zone, preserving ``NaN`` (detached)."""
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap

    xi = _unwrap(x)
    zeros = _C_engine.zeros(xi.shape, xi.dtype, xi.device)
    return _C_engine.where(_C_engine.isnan(xi), _unwrap(x.detach()), zeros)
