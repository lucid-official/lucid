"""lucid.einops — shape manipulation written in named axes.

Entry points: ``rearrange`` (permute, reshape, split or merge axes),
``reduce`` (the same, carrying a reduction), ``repeat`` (the same, carrying a
broadcast expansion), ``einsum``, ``pack`` / ``unpack`` (join tensors of
different rank along one axis and split them again) and ``parse_shape``
(read named lengths off a shape).

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

from collections.abc import Callable, Mapping, Sequence
from typing import SupportsIndex

import lucid
from lucid import _contraction
from lucid._C import engine as _C_engine
from lucid._C.engine import einops as _C_einops
from lucid._dispatch import _REAL_FLOATS, _unwrap, _wrap
from lucid._dtype import _ENGINE_TO_DTYPE
from lucid._tensor.tensor import Tensor
from lucid.einops import _einsum, _recipe
from lucid.einops._packing import pack, unpack


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


def _single(tensor: Tensor | list[Tensor]) -> Tensor:
    """The tensor a call acts on; a list of tensors is stacked on a new axis 0."""
    if not isinstance(tensor, list):
        return tensor
    if not tensor:
        raise TypeError("einops cannot be applied to an empty list of tensors")
    for i, item in enumerate(tensor):
        if not isinstance(item, Tensor):
            raise TypeError(f"einops: the list holds a {type(item).__name__} at {i}, not a Tensor")
    return lucid.stack(tensor, dim=0)


def _reduce_with(
    impl: _C_engine.TensorImpl,
    reduction: Callable[[Tensor, tuple[int, ...]], Tensor],
    order: tuple[int, ...] | None,
    axes: tuple[int, ...],
) -> _C_engine.TensorImpl:
    """Hand ``reduction`` the tensor with the reduced axes last, as the reference does."""
    if order is not None:
        impl = _C_engine.permute(impl, list(order))
    reduced = reduction(_wrap(impl), axes)
    if not isinstance(reduced, Tensor):
        raise TypeError(
            f"einops.reduce: the reduction returned a {type(reduced).__name__}, not a Tensor"
        )
    return _unwrap(reduced)


def _transform(
    tensor: Tensor | list[Tensor],
    pattern: str,
    operation: str | Callable[[Tensor, tuple[int, ...]], Tensor],
    axes_lengths: Mapping[str, SupportsIndex],
) -> Tensor:
    """Run the at most six engine steps a pattern lowers to."""
    inp = _unwrap(_single(tensor))
    named = operation if isinstance(operation, str) else None
    plan = _recipe.plan(pattern, named, tuple(inp.shape), axes_lengths)
    out = inp
    if plan.split is not None:
        out = _C_engine.reshape(out, list(plan.split))
    if plan.reduced and not isinstance(operation, str):
        out = _reduce_with(out, operation, plan.reduce_last, plan.reduce_axes)
    else:
        if plan.reduced and named is not None:
            out = _reduce_axes(out, named, plan.reduced)
        if plan.permutation is not None:
            out = _C_engine.permute(out, list(plan.permutation))
    for position in plan.inserted:
        out = _C_engine.unsqueeze(out, position)
    if plan.broadcast is not None:
        out = _C_engine.broadcast_to(out, list(plan.broadcast))
    if plan.merge is not None:
        out = _C_engine.reshape(out, list(plan.merge))
    return _wrap_own(out, inp)


def rearrange(tensor: Tensor | list[Tensor], pattern: str, **axes_lengths: int) -> Tensor:
    r"""Rearrange axes of ``tensor`` according to an einops pattern.

    Performs arbitrary axis manipulation — transpose, reshape,
    composition, and decomposition — expressed in a single readable
    pattern string. The notation makes the intent explicit and removes
    the chain of ``transpose``/``view`` calls that obscure dimensionality
    transforms in raw tensor code.

    Parameters
    ----------
    tensor : Tensor or list of Tensor
        Input tensor to rearrange.  A list of equally shaped tensors is
        stacked along a new leading axis first, so the pattern names it.
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
    tensor: Tensor | list[Tensor],
    pattern: str,
    reduction: str | Callable[[Tensor, tuple[int, ...]], Tensor],
    **axes_lengths: int,
) -> Tensor:
    r"""Reduce ``tensor`` along axes implied by an einops pattern.

    Generalises standard ``sum`` / ``mean`` / ``max`` / ``min`` /
    ``prod`` reductions to arbitrary axes selected by symbolic pattern.
    Any axis present on the left-hand side of ``->`` but absent on the
    right-hand side is collapsed by the chosen reduction.

    Parameters
    ----------
    tensor : Tensor or list of Tensor
        Input tensor to reduce.  A list of equally shaped tensors is
        stacked along a new leading axis first, so the pattern names it.
    pattern : str
        Einops pattern such as ``"b h w -> b w"``.  A number on the left
        is an unnamed axis of that length, so ``"(h 2) (w 2) -> h w"``
        pools 2x2 windows.
    reduction : str or callable
        Reduction operation; one of ``"sum"``, ``"mean"``, ``"max"``,
        ``"min"``, ``"prod"``, ``"any"``, ``"all"``.  A callable
        ``f(tensor, axes)`` is also accepted: it is given the tensor with
        the axes to reduce moved last, and the positions ``axes`` they
        then occupy, and returns the reduced tensor.
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
    >>> total = lambda t, axes: lucid.sum(t, dim=list(axes))
    >>> lucid.einops.reduce(lucid.ones(2, 3), "b w -> b", total)
    tensor([3., 3.])
    """
    return _transform(tensor, pattern, reduction, dict(axes_lengths))


def repeat(tensor: Tensor | list[Tensor], pattern: str, **axes_lengths: int) -> Tensor:
    r"""Repeat ``tensor`` along new or existing axes via an einops pattern.

    Provides a single notation for both tiling along existing axes and
    broadcasting into entirely new axes. The pattern's right-hand side
    enumerates the desired output axes; any axis appearing on the right
    but not on the left is introduced and must be given an explicit
    length through ``**axes_lengths``, or written as a number.

    Parameters
    ----------
    tensor : Tensor or list of Tensor
        Input tensor to repeat.  A list of equally shaped tensors is
        stacked along a new leading axis first, so the pattern names it.
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


def _einsum_operands(operands: tuple[Tensor | Sequence[Tensor], ...]) -> list[Tensor]:
    """The tensors of a call, given one by one or as a single list or tuple."""
    flat = (
        operands[0] if len(operands) == 1 and isinstance(operands[0], (list, tuple)) else operands
    )
    tensors: list[Tensor] = []
    for operand in flat:
        if not isinstance(operand, Tensor):
            raise TypeError(f"einsum(): expected Tensor operands, got {type(operand).__name__}")
        tensors.append(operand)
    return tensors


def einsum(equation: str, *operands: Tensor | Sequence[Tensor]) -> Tensor:
    r"""Evaluate an Einstein-summation expression over the given tensors.

    Implements the general tensor contraction operator: each input is
    indexed by labels declared in ``equation`` and the output retains
    only the labels appearing on the right-hand side. Repeated labels
    are summed; matching labels broadcast.

    Parameters
    ----------
    equation : str
        Einsum equation of the form ``"<lhs1>,<lhs2>,... -> <rhs>"``.
    *operands : Tensor or sequence of Tensor
        Tensors corresponding to the comma-separated factors on the
        left-hand side, given one by one or as a single list or tuple.

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

    An integer contraction keeps its operands' dtype, as ``matmul`` does:
    ``"ij,jk->ik"`` over ``int32`` is ``int32``.  A subscript summed away
    inside one operand (``"ij->j"``, a trace) is a ``sum`` and widens an
    integer or bool operand to ``int64``.  Contracting bool operands is
    refused.

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
    lucid._C.engine.NotImplementedError
        Bool operands share a subscript that the output drops.

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
    raw_impls = [_unwrap(t) for t in _einsum_operands(operands)]
    explicit, impl_list = _einsum.normalise(equation, raw_impls)
    stand_alone, contracted = _contraction.split_sums(explicit)
    dtype = _contraction.operand_dtype(raw_impls)
    if contracted and not stand_alone:
        _contraction.require_summable(dtype, "einsum")
    if stand_alone or contracted:
        impl_list = _contraction.widen(impl_list, dtype)
    out = _C_einops.einsum(explicit, impl_list)
    if not stand_alone:
        out = _contraction.settle(out, dtype)
    return _wrap_own(out, *raw_impls)


def parse_shape(tensor: Tensor, pattern: str) -> dict[str, int]:
    r"""Read named axis lengths off the shape of ``tensor``.

    The result can be passed as the keyword lengths of the other entry
    points, so a length is written once.

    Parameters
    ----------
    tensor : Tensor
        The tensor whose shape is read.
    pattern : str
        One name per dimension, ``_`` for a dimension to skip, ``...`` for
        any run of dimensions, ``1`` or ``()`` for a unit axis and a number
        for an axis of that length.  Parenthesised groups of several axes
        are refused: a length cannot be read through a composition.

    Returns
    -------
    dict of str to int
        Each named axis and its length.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The pattern is malformed or composes axes.
    lucid._C.engine.ShapeMismatch
        The rank of ``tensor`` does not fit the pattern, or a unit or
        numbered axis is not the length the pattern writes.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros(2, 3, 5, 7)
    >>> lucid.einops.parse_shape(x, "batch _ h w")
    {'batch': 2, 'h': 5, 'w': 7}
    >>> y = lucid.zeros(70)
    >>> lucid.einops.rearrange(y, "(c h w) -> c h w", **lucid.einops.parse_shape(x, "_ _ h w")).shape
    (2, 5, 7)
    """
    return _recipe.parse_shape(tuple(tensor.shape), pattern)


__all__ = ["rearrange", "reduce", "repeat", "einsum", "pack", "unpack", "parse_shape"]
