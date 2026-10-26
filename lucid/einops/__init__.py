"""lucid.einops — shape manipulation written in named axes.

Four entry points: ``rearrange`` (permute, reshape, split or merge axes),
``reduce`` (the same, carrying a reduction), ``repeat`` (the same, carrying a
broadcast expansion) and ``einsum``.

Patterns are strings of axis names, so the intent is stated in the call
rather than reconstructed from a run of anonymous ``transpose`` / ``reshape``
steps::

    lucid.einops.rearrange(x, "b c h w -> b (h w) c")   # (2,3,4,5) → (2,20,3)

The grammar is the reference einops grammar: names, parenthesised groups
one level deep, ``...`` for the axes the pattern does not name, ``1`` or
``()`` for a unit axis and, in ``reduce`` and ``repeat``, a number for an
unnamed axis of that length.  A keyword length is also a check: an axis
the input already fixes must have the length given for it, and a keyword
naming no axis of the pattern is refused.  A mismatch fails with a message
naming the axis that did not fit, instead of producing a silently wrong
shape that only surfaces layers later.

Reachable only as ``lucid.einops.*``: there is no top-level ``lucid.einsum``
and no ``Tensor.rearrange`` method, so each op has exactly one path.
"""

from collections.abc import Mapping
from typing import SupportsIndex

import lucid
from lucid._C import engine as _C_engine
from lucid._C.engine import einops as _C_einops
from lucid._dispatch import _REAL_FLOATS, _unwrap, _wrap
from lucid._dtype import _ENGINE_TO_DTYPE
from lucid._tensor.tensor import Tensor
from lucid.einops import _einsum, _recipe


def _wrap_own(out: _C_engine.TensorImpl, *inputs: _C_engine.TensorImpl) -> Tensor:
    """``out`` as a new tensor that never shares its input's impl.

    An identity pattern runs no step and would hand the input's own impl
    back under a second Python object, so ``requires_grad_`` on the result
    would flip the input's flag too (LCD-75).
    """
    for inp in inputs:
        if out is inp:
            out = _C_engine.reshape(inp, list(inp.shape))
            break
    return _wrap(out)


_ENGINE_REDUCERS = {
    "sum": _C_engine.sum,
    "mean": _C_engine.mean,
    "max": _C_engine.max,
    "min": _C_engine.min,
    "prod": _C_engine.prod,
}


def _reduce_axes(
    impl: _C_engine.TensorImpl, reduction: str, axes: tuple[int, ...]
) -> _C_engine.TensorImpl:
    # Real floating dtypes only: the reference einops refuses a mean over
    # integer, boolean and complex tensors alike.
    if reduction == "mean" and impl.dtype not in _REAL_FLOATS:
        raise _C_engine.NotImplementedError(
            "einops.reduce: 'mean' needs a floating-point tensor, got "
            f"{_ENGINE_TO_DTYPE[impl.dtype]}"
        )
    if reduction == "any":
        return _unwrap(lucid.any(_wrap(impl), dim=list(axes)))
    if reduction == "all":
        return _unwrap(lucid.all(_wrap(impl), dim=list(axes)))
    return _ENGINE_REDUCERS[reduction](impl, list(axes), False)


def _transform(
    tensor: Tensor,
    pattern: str,
    operation: str,
    axes_lengths: Mapping[str, SupportsIndex],
) -> Tensor:
    """Run the at most six engine steps a pattern lowers to."""
    inp = _unwrap(tensor)
    plan = _recipe.plan(pattern, operation, tuple(inp.shape), axes_lengths)
    out = inp
    if plan.split is not None:
        out = _C_engine.reshape(out, list(plan.split))
    if plan.reduced:
        out = _reduce_axes(out, operation, plan.reduced)
    if plan.permutation is not None:
        out = _C_engine.permute(out, list(plan.permutation))
    for position in plan.inserted:
        out = _C_engine.unsqueeze(out, position)
    if plan.broadcast is not None:
        out = _C_engine.broadcast_to(out, list(plan.broadcast))
    if plan.merge is not None:
        out = _C_engine.reshape(out, list(plan.merge))
    return _wrap_own(out, inp)


def rearrange(tensor: Tensor, pattern: str, **axes_lengths: int) -> Tensor:
    r"""Rearrange axes of ``tensor`` according to an einops pattern.

    Performs arbitrary axis manipulation — transpose, reshape,
    composition, and decomposition — expressed in a single readable
    pattern string. The notation makes the intent explicit and removes
    the chain of ``transpose``/``view`` calls that obscure dimensionality
    transforms in raw tensor code.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to rearrange.
    pattern : str
        Einops pattern of the form ``"<lhs> -> <rhs>"``. Parentheses
        group axes for composition or decomposition, ``...`` stands for
        the axes the pattern does not name, and ``1`` or ``()`` is a unit
        axis.  Every named axis must appear on both sides.
    **axes_lengths : int
        Axis lengths, needed to split a dimension into a group and checked
        against the input wherever the input already fixes the axis.

    Returns
    -------
    Tensor
        The rearranged tensor.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The pattern is malformed, an axis appears on one side only, or a
        keyword names no axis of the pattern.
    lucid._C.engine.ShapeMismatch
        The input's rank or a dimension does not fit the pattern or the
        keyword lengths.

    Notes
    -----
    The transformation is purely a reshape and permutation — no data is
    copied unless required by memory layout. Typical patterns include
    ``"b c h w -> b (h w) c"`` for flattening spatial dims into a token
    sequence, and ``"b (h w) c -> b c h w"`` for the inverse.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(2, 3, 4)
    >>> lucid.einops.rearrange(x, "b h w -> b (h w)").shape
    (2, 12)
    >>> lucid.einops.rearrange(x, "b ... -> b (...)").shape
    (2, 12)
    >>> lucid.einops.rearrange(x, "b (h1 h2) w -> b h1 h2 w 1", h2=3).shape
    (2, 1, 3, 4, 1)
    """
    return _transform(tensor, pattern, "rearrange", dict(axes_lengths))


def reduce(
    tensor: Tensor,
    pattern: str,
    reduction: str,
    **axes_lengths: int,
) -> Tensor:
    r"""Reduce ``tensor`` along axes implied by an einops pattern.

    Generalises standard ``sum`` / ``mean`` / ``max`` / ``min`` /
    ``prod`` reductions to arbitrary axes selected by symbolic pattern.
    Any axis present on the left-hand side of ``->`` but absent on the
    right-hand side is collapsed by the chosen reduction.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to reduce.
    pattern : str
        Einops pattern such as ``"b h w -> b w"``.  A number on the left
        is an unnamed axis of that length, so ``"(h 2) (w 2) -> h w"``
        pools 2x2 windows.
    reduction : str
        Reduction operation; one of ``"sum"``, ``"mean"``, ``"max"``,
        ``"min"``, ``"prod"``, ``"any"``, ``"all"``.
    **axes_lengths : int
        Axis lengths, needed to split a dimension into a group and checked
        against the input wherever the input already fixes the axis.

    Returns
    -------
    Tensor
        The reduced tensor.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The pattern is malformed, the reduction is unknown, an axis
        appears only on the right, or a keyword names no axis.
    lucid._C.engine.ShapeMismatch
        The input's shape does not fit the pattern or the keyword lengths.
    lucid._C.engine.NotImplementedError
        ``reduction="mean"`` over a tensor that is not floating point.

    Notes
    -----
    Conceptually equivalent to a permutation followed by a reduction
    along contiguous trailing axes. For ``reduction="mean"`` the result
    is

    .. math::

        y_{b w} = \frac{1}{H} \sum_{h=1}^{H} x_{b h w}.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(2, 3, 4)
    >>> lucid.einops.reduce(x, "b h w -> b w", "mean").shape
    (2, 4)
    >>> lucid.einops.reduce(lucid.randn(2, 4, 6), "b (h 2) (w 3) -> b h w", "max").shape
    (2, 2, 2)
    """
    return _transform(tensor, pattern, reduction, dict(axes_lengths))


def repeat(tensor: Tensor, pattern: str, **axes_lengths: int) -> Tensor:
    r"""Repeat ``tensor`` along new or existing axes via an einops pattern.

    Provides a single notation for both tiling along existing axes and
    broadcasting into entirely new axes. The pattern's right-hand side
    enumerates the desired output axes; any axis appearing on the right
    but not on the left is introduced and must be given an explicit
    length through ``**axes_lengths``, or written as a number.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to repeat.
    pattern : str
        Einops pattern such as ``"b h w -> b h w c"`` (new axis ``c``) or
        ``"h w -> (h 2) w"`` (each row twice).
    **axes_lengths : int
        Lengths for every new named axis, and checks for axes the input
        already fixes.

    Returns
    -------
    Tensor
        The repeated tensor.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The pattern is malformed, drops an axis, or introduces a named
        axis without a length.
    lucid._C.engine.ShapeMismatch
        The input's shape does not fit the pattern or the keyword lengths.

    Notes
    -----
    For an axis :math:`c` of size :math:`C` introduced via ``repeat``,

    .. math::

        y_{b h w c} = x_{b h w}, \quad c = 1, \ldots, C.

    No data is materialised until reading — internally the new axis is
    expressed as a broadcast / stride manipulation when possible.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(2, 3, 4)
    >>> lucid.einops.repeat(x, "b h w -> b h w c", c=5).shape
    (2, 3, 4, 5)
    >>> lucid.einops.repeat(lucid.tensor([[1.0, 2.0]]), "h w -> (h 2) w")
    tensor([[1., 2.], [1., 2.]])
    """
    return _transform(tensor, pattern, "repeat", dict(axes_lengths))


def einsum(equation: str, *operands: Tensor) -> Tensor:
    r"""Evaluate an Einstein-summation expression over the given tensors.

    Implements the general tensor contraction operator: each input is
    indexed by labels declared in ``equation`` and the output retains
    only the labels appearing on the right-hand side. Repeated labels
    are summed; matching labels broadcast.

    Parameters
    ----------
    equation : str
        Einsum equation of the form ``"<lhs1>,<lhs2>,... -> <rhs>"``.
    *operands : Tensor
        Tensors corresponding to the comma-separated factors on the
        left-hand side.

    Returns
    -------
    Tensor
        Tensor whose axes correspond to the labels on the right-hand
        side of ``equation``.

    Notes
    -----
    Matrix multiplication is the canonical example: ``"ij,jk->ik"``
    realises

    .. math::

        C_{ik} = \sum_{j} A_{ij} \, B_{jk}.

    More generally, einsum expresses traces (``"ii->"``), batched
    contractions (``"bij,bjk->bik"``), and outer products
    (``"i,j->ij"``) within one consistent notation.

    A subscript repeated within one operand selects that operand's
    diagonal over those axes (``"ii->i"``, ``"bii->bi"``), and must name
    axes of equal size.  ``...`` stands for the remaining leading axes;
    they broadcast across operands, right-aligned.  A subscript's size
    may be 1 in one operand and longer in another, which broadcasts.
    Without ``->`` the output is ``...`` followed by the subscripts that
    occur exactly once, in sorted order.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The equation is malformed: a subscript outside ``[a-zA-Z]``, a
        stray ``.``, the wrong number of operands, or an output subscript
        that repeats or appears in no input.
    lucid._C.engine.ShapeMismatch
        An operand's rank does not match its subscripts, or a subscript's
        sizes disagree (within one operand, or across operands without a
        size of 1 to broadcast).

    Examples
    --------
    >>> import lucid
    >>> a = lucid.randn(2, 3)
    >>> b = lucid.randn(3, 4)
    >>> lucid.einops.einsum("ij,jk->ik", a, b).shape
    (2, 4)
    >>> m = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> lucid.einops.einsum("ii->i", m)
    tensor([1., 4.])
    >>> lucid.einops.einsum("ii->", m)
    tensor(5.)
    """
    raw_impls = [_unwrap(t) for t in operands]
    explicit, impl_list = _einsum.normalise(equation, raw_impls)
    return _wrap_own(_C_einops.einsum(explicit, impl_list), *raw_impls)


__all__ = ["rearrange", "reduce", "repeat", "einsum"]
