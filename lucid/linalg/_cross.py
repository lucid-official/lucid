"""``cross``: the 3-D vector cross product, refusing operands of two dtypes.

Re-exported by :mod:`lucid.linalg`.
"""

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._bool_refusal import refuse, refuse_unlike
from lucid._ops._dims import _wrap_dim
from lucid._tensor.tensor import Tensor


def cross(input: Tensor, other: Tensor, *, dim: int = -1) -> Tensor:
    r"""Compute the cross product of two 3-D vectors.

    Returns the standard 3-D vector cross product

    .. math::

        x \times y \,=\, \big(x_2 y_3 - x_3 y_2,\;
                              x_3 y_1 - x_1 y_3,\;
                              x_1 y_2 - x_2 y_1\big).

    Parameters
    ----------
    input : Tensor
        First operand.  Must have size 3 along ``dim``.
    other : Tensor
        Second operand.  Same shape as ``input``.
    dim : int, keyword-only, optional
        Axis along which the 3 components are stored.  Default ``-1``.

    Returns
    -------
    Tensor
        Cross product, same shape as inputs.

    Raises
    ------
    DtypeMismatch
        If ``input`` and ``other`` have different dtypes, as the reference
        does.
    NotImplementedError
        If both are bool tensors: the reference has no bool kernel.  An
        integer pair is answered.

    Notes
    -----
    Implemented via :func:`gather` + element-wise products so that
    autograd flows through naturally.  Recall that
    :math:`x \times y` is orthogonal to both :math:`x` and :math:`y`,
    with magnitude :math:`\|x\|\,\|y\|\,\sin\theta`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import cross
    >>> cross(lucid.tensor([1.0, 0.0, 0.0]), lucid.tensor([0.0, 1.0, 0.0]))
    tensor([0., 0., 1.])
    """
    # The components below are multiplied as 0-d tensors for a single vector,
    # where the engine's half-precision op-math would read a float32 operand as
    # a scalar and answer in float16: the dtypes are checked here, as the
    # reference does, rather than left to that.
    refuse_unlike("linalg.cross", input, other)
    xi, yi = _unwrap(input), _unwrap(other)
    if len(xi.shape) != len(yi.shape):
        raise _C_engine.ShapeMismatch(
            "linalg.cross: inputs must have the same number of dimensions"
        )
    d = _wrap_dim(dim, len(xi.shape), "linalg.cross")
    if not xi.shape:
        raise IndexError(f"linalg.cross: axis out of range: got {dim} for a 0-D tensor")
    if xi.shape[d] != 3 or yi.shape[d] != 3:
        raise _C_engine.ShapeMismatch(
            f"linalg.cross: inputs dimension {d} must have length 3, got {xi.shape[d]} and {yi.shape[d]}"
        )
    refuse("cross", xi)

    def _idx(t: _C_engine.TensorImpl, i: int) -> _C_engine.TensorImpl:
        idx_shape = list(t.shape)
        idx_shape[d] = 1
        idx = _C_engine.full(idx_shape, float(i), _C_engine.I32, t.device)
        return _C_engine.squeeze(_C_engine.gather(t, idx, d), d)

    x0, x1, x2 = _idx(xi, 0), _idx(xi, 1), _idx(xi, 2)
    y0, y1, y2 = _idx(yi, 0), _idx(yi, 1), _idx(yi, 2)

    c0 = _C_engine.sub(_C_engine.mul(x1, y2), _C_engine.mul(x2, y1))
    c1 = _C_engine.sub(_C_engine.mul(x2, y0), _C_engine.mul(x0, y2))
    c2 = _C_engine.sub(_C_engine.mul(x0, y1), _C_engine.mul(x1, y0))
    return _wrap(_C_engine.concatenate([_C_engine.unsqueeze(c, d) for c in (c0, c1, c2)], d))
