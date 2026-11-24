"""``vecdot``: the batched vector dot product, refusing operands of two dtypes.

Re-exported by :mod:`lucid.linalg`.
"""

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry
from lucid._ops._bool_refusal import refuse, refuse_unlike
from lucid._tensor.tensor import Tensor


def vecdot(x: Tensor, y: Tensor, *, dim: int = -1) -> Tensor:
    r"""Compute a batched vector dot product along an axis.

    Reduces the chosen axis with a sum of element-wise products, the
    first operand conjugated:

    .. math::

        (x \cdot y)_{\ldots} \,=\, \sum_{k} \overline{x_{\ldots, k, \ldots}}\,
                                              y_{\ldots, k, \ldots}.

    Parameters
    ----------
    x : Tensor
        First operand.
    y : Tensor
        Second operand, broadcast-compatible with ``x``.
    dim : int, keyword-only, optional
        Axis to contract.  Default ``-1``.

    Returns
    -------
    Tensor
        Reduced tensor with ``dim`` removed.

    Raises
    ------
    DtypeMismatch
        If ``x`` and ``y`` have different dtypes -- a 0-d operand beside a vector
        and a ``float16`` beside a ``float32`` included.  The reference does not
        promote the pair (a ``RuntimeError`` and a ``TypeError``).

    Notes
    -----
    Equivalent to ``(x.conj() * y).sum(dim=dim)`` — for real inputs
    simply ``(x * y).sum(dim=dim)``.  Useful for computing many
    independent dot products in one shot (e.g., per-row inner products
    of two matrices).

    The conjugate makes ``vecdot(x, x)`` the squared norm
    :math:`\sum_k |x_k|^2` of a complex vector.  Without it, the
    complex result was the bilinear :math:`\sum_k x_k y_k` instead.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import vecdot
    >>> x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> y = lucid.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    >>> vecdot(x, y)
    tensor([-2., 5.])
    >>> vecdot(lucid.tensor([1 + 2j, 3 - 1j]), lucid.tensor([2 - 1j, 1 + 1j]))
    tensor((2.-1.j), dtype=lucid.complex64)
    """
    refuse("vecdot", x)
    refuse("vecdot", y)
    with _autocast_entry("vecdot", _unwrap(x), _unwrap(y)) as (xi, yi):
        refuse_unlike("linalg.vecdot", xi, yi)
        xi = _C_engine.conj(xi) if xi.dtype in (_C_engine.C64, _C_engine.C128) else xi
        return _wrap(_C_engine.sum(_C_engine.mul(xi, yi), [dim], False))
