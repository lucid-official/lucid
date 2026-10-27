"""Shape-manipulation composites: axis swaps, stacks, splits, and the
miscellaneous fillers (``rot90``, ``vander``, ``take_along_dim``).
"""

import itertools
import operator
from typing import Sequence, SupportsIndex, TYPE_CHECKING

import lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._globals import get_default_device
from lucid._types import DTypeLike, DeviceLike
from lucid._ops._dims import _dim_size, _require_real_axis, _to_axes, _wrap_dim
from lucid._ops.composite._shared import _swap_dims
from lucid._tensor._indexing import (
    _broadcast_shape,
    broadcast_shapes,
    broadcast_tensors,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Axis swaps ─────────────────────────────────────────────────────────────


def swapaxes(input: Tensor, axis0: int, axis1: int) -> Tensor:
    r"""Return ``input`` with axes ``axis0`` and ``axis1`` exchanged.

    NumPy-style spelling of pairwise axis transposition.  Verbose alias of
    :func:`swapdims`; both names refer to the same composite.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    axis0 : int
        First axis to swap.  Negative values count from the end.
    axis1 : int
        Second axis to swap.  Negative values count from the end.

    Returns
    -------
    Tensor
        Tensor with the same data as ``input`` but with ``axis0`` and
        ``axis1`` exchanged — a view of ``input``'s buffer on the CPU, so a
        write through it reaches ``input``; a copy on metal.

    Notes
    -----
    For a tensor with shape :math:`(s_0, \dots, s_{n-1})`, the output
    shape is

    .. math::

        s'_i =
        \begin{cases}
            s_{a_1}, & i = a_0, \\
            s_{a_0}, & i = a_1, \\
            s_i,     & \text{otherwise.}
        \end{cases}

    As in the reference, the result is a view on the CPU: no data moves,
    only the two strides trade places.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros((2, 3, 4))
    >>> lucid.swapaxes(x, 0, 2).shape
    (4, 3, 2)
    """
    return _swap_dims(input, axis0, axis1)


def swapdims(input: Tensor, dim0: int, dim1: int) -> Tensor:
    r"""Swap two named dimensions of a tensor.

    Returns a tensor whose dimensions ``dim0`` and ``dim1`` have been
    exchanged — a view on the CPU, a copy on metal. All other
    dimensions retain their position.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    dim0 : int
        First dimension to swap. Negative values count from the end.
    dim1 : int
        Second dimension to swap. Negative values count from the end.

    Returns
    -------
    Tensor
        Tensor with the same data as ``input`` but with axes ``dim0`` and
        ``dim1`` exchanged.

    Notes
    -----
    For a tensor with shape :math:`(s_0, \dots, s_{n-1})`, the result
    has shape

    .. math::

        s'_i =
        \begin{cases}
            s_{d_1}, & i = d_0, \\
            s_{d_0}, & i = d_1, \\
            s_i,     & \text{otherwise.}
        \end{cases}

    Equivalent to :func:`swapaxes`; both names are provided for parity
    with NumPy and reference-framework conventions.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros((2, 3, 4))
    >>> lucid.swapdims(x, 0, 2).shape
    (4, 3, 2)
    """
    return _swap_dims(input, dim0, dim1)


def moveaxis(
    input: Tensor,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Tensor:
    r"""Move one or more axes to new positions.

    More flexible than :func:`swapaxes` — instead of pairwise swapping,
    ``moveaxis`` re-positions a whole set of axes while preserving the
    relative order of the remaining axes.  Thin wrapper around
    :func:`lucid.movedim` that accepts either a single ``int`` or a
    sequence on both endpoints.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    source : int | Sequence[int]
        Original position(s) of the axes to move.  Negative values count
        from the end.
    destination : int | Sequence[int]
        Final position(s) for each moved axis.  Must have the same length
        as ``source`` (after normalisation) and be a permutation-compatible
        target set.

    Returns
    -------
    Tensor
        Tensor whose axes have been re-ordered as requested.

    Notes
    -----
    Equivalent to applying a permutation :math:`\pi` to the axes such
    that ``axis at source[i]`` ends up at ``destination[i]`` while the
    other axes slide to fill in the gaps preserving their relative
    order.  Useful for vectorised code that needs to ferry a batch
    dimension across the rank of a tensor.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros((2, 3, 4, 5))
    >>> lucid.moveaxis(x, 0, -1).shape
    (3, 4, 5, 2)
    >>> lucid.moveaxis(x, [0, 1], [-1, -2]).shape
    (4, 5, 3, 2)
    """
    source, destination = _promotion.as_number(source), _promotion.as_number(destination)
    _src = list(source) if not isinstance(source, int) else source
    _dst = list(destination) if not isinstance(destination, int) else destination
    return lucid.movedim(input, _src, _dst)


def adjoint(input: Tensor) -> Tensor:
    r"""Conjugate (Hermitian) transpose of the trailing two dimensions.

    For real-valued tensors this is identical to a plain transpose of the
    last two axes. For complex tensors the entries are also conjugated,
    yielding the Hermitian transpose :math:`A^{*} = \overline{A^{T}}`.

    Parameters
    ----------
    input : Tensor
        Input tensor with at least 2 dimensions.

    Returns
    -------
    Tensor
        Tensor with the last two axes swapped (and complex conjugated
        when applicable).

    Raises
    ------
    ValueError
        If ``x.ndim < 2``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \mathbf{A}^{*} = \overline{\mathbf{A}^{T}}.

    For real inputs the conjugation step is a no-op, so this is simply
    a permutation of the last two axes. Batch dimensions are preserved.

    Examples
    --------
    >>> import lucid
    >>> A = lucid.tensor([[1., 2., 3.], [4., 5., 6.]])
    >>> lucid.adjoint(A)
    tensor([[1., 4.], [2., 5.], [3., 6.]])
    """
    if input.ndim < 2:
        raise ValueError("adjoint requires at least 2 dimensions")
    swapped = _swap_dims(input, input.ndim - 2, input.ndim - 1)
    # The conjugation the definition promises.  It was never applied: a
    # complex transpose used to be refused, so this returned nothing wrong —
    # until transposes read every dtype, and a complex input came back only
    # transposed.
    return lucid.conj(swapped) if lucid.is_complex(input) else swapped


def t(input: Tensor) -> Tensor:
    r"""Short-form transpose for tensors of rank at most 2.

    A convenience analogue of MATLAB's ``A'`` or the reference
    framework's ``Tensor.t()``: returns the transpose of a 2-D matrix,
    or the input unchanged for 0-D and 1-D tensors.

    Parameters
    ----------
    input : Tensor
        Input tensor with at most 2 dimensions.

    Returns
    -------
    Tensor
        For ``ndim <= 1``, returns ``input`` unchanged.
        For ``ndim == 2``, returns the 2-D transpose.

    Raises
    ------
    RuntimeError
        If ``input`` has 3 or more dimensions.

    Notes
    -----
    Defined as

    .. math::

        \text{t}(x) =
        \begin{cases}
            x,            & \text{ndim}(x) \leq 1, \\
            x^{T},        & \text{ndim}(x) = 2.
        \end{cases}

    Use :func:`adjoint` or :func:`lucid.transpose` for batched / higher
    rank inputs.

    Examples
    --------
    >>> import lucid
    >>> A = lucid.tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> lucid.t(A).shape
    (2, 3)
    """
    if input.ndim < 2:
        return input
    if input.ndim != 2:
        raise RuntimeError("t() expects a tensor with <= 2 dimensions")
    return _swap_dims(input, 0, 1)


# ── Stacks ─────────────────────────────────────────────────────────────────


def column_stack(tensors: Sequence[Tensor]) -> Tensor:
    r"""Stack tensors as columns of a 2-D matrix.

    Each 1-D tensor is first promoted to a column vector of shape
    ``(N, 1)``; tensors with rank ``>= 2`` are passed through unchanged.
    The results are then concatenated along axis 1.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Tensors to stack. 1-D tensors are reshaped to column form;
        higher-rank tensors must already have matching first-axis size.

    Returns
    -------
    Tensor
        2-D tensor formed by concatenating the (possibly promoted)
        inputs along axis 1.

    Notes
    -----
    For all-1-D inputs :math:`v^{(1)}, \dots, v^{(k)}` of length ``N``,
    the result has shape ``(N, k)``:

    .. math::

        \text{out}_{ij} = v^{(j)}_i.

    For higher-rank inputs, the first dimension is the "row count" along
    which axis-1 concatenation happens, matching NumPy semantics.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1., 2., 3.])
    >>> b = lucid.tensor([4., 5., 6.])
    >>> lucid.column_stack([a, b])
    tensor([[1., 4.], [2., 5.], [3., 6.]])
    """
    fixed = [t_i.unsqueeze(1) if t_i.ndim == 1 else t_i for t_i in tensors]
    return lucid.cat(fixed, 1)


def row_stack(tensors: Sequence[Tensor]) -> Tensor:
    r"""Stack tensors as rows — alias of :func:`lucid.vstack`.

    Verbose name provided for API parity. Each 1-D input is treated as a
    row and concatenated along axis 0.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Tensors to stack. 1-D tensors are promoted to ``(1, N)`` row
        vectors; higher-rank tensors must have matching column count.

    Returns
    -------
    Tensor
        Tensor formed by concatenating the (possibly promoted) inputs
        along axis 0.

    Notes
    -----
    For all-1-D inputs of length ``N``, the result has shape ``(k, N)``:

    .. math::

        \text{out}_{ij} = v^{(i)}_j.

    The 1-D-to-2-D promotion is the inverse of :func:`column_stack`'s
    promotion.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1., 2., 3.])
    >>> b = lucid.tensor([4., 5., 6.])
    >>> lucid.row_stack([a, b])
    tensor([[1., 2., 3.], [4., 5., 6.]])
    """
    return lucid.vstack(list(tensors))


def dstack(tensors: Sequence[Tensor]) -> Tensor:
    r"""Stack tensors along the third (depth) axis.

    Reshapes lower-rank inputs to give them a depth dimension, then
    concatenates along axis 2. 0-D scalars become ``(1, 1, 1)``, 1-D
    vectors of length ``N`` become ``(1, N, 1)``, and 2-D matrices of
    shape ``(H, W)`` become ``(H, W, 1)``.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Tensors to stack. 0-, 1-, and 2-D inputs are auto-promoted to
        3-D as described above; 3-D inputs pass through unchanged.

    Returns
    -------
    Tensor
        3-D tensor formed by concatenating the (possibly promoted)
        inputs along axis 2.

    Notes
    -----
    The promotion rules ensure the result is always at least 3-D, which
    is useful for stacking colour channels of images: ``dstack([R, G, B])``
    produces an ``(H, W, 3)`` tensor from three ``(H, W)`` planes.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([[1., 2.], [3., 4.]])
    >>> b = lucid.tensor([[5., 6.], [7., 8.]])
    >>> lucid.dstack([a, b]).shape
    (2, 2, 2)
    """
    fixed: list[Tensor] = []
    for t_i in tensors:
        if t_i.ndim == 0:
            t_i = t_i.reshape(1, 1, 1)
        elif t_i.ndim == 1:
            t_i = t_i.reshape(1, -1, 1)
        elif t_i.ndim == 2:
            t_i = t_i.unsqueeze(2)
        fixed.append(t_i)
    return lucid.cat(fixed, 2)


def atleast_1d(*tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
    r"""Promote each input to at least 1 dimension.

    Scalars (0-D tensors) are reshaped to length-1 vectors; tensors that
    already have rank :math:`\geq 1` pass through unchanged.  Commonly
    used as a defensive guard at the top of routines that vectorise over
    a leading axis.

    Parameters
    ----------
    *tensors : Tensor
        One or more input tensors of arbitrary rank.

    Returns
    -------
    Tensor | tuple[Tensor, ...]
        A single tensor when called with one argument, otherwise a tuple
        of the promoted tensors in the same order as the inputs.

    Notes
    -----
    Promotion rule per input ``t`` with original rank ``r``:

    .. math::

        \text{shape}'(t) =
        \begin{cases}
            (1,),               & r = 0, \\
            \text{shape}(t),    & r \geq 1.
        \end{cases}

    Mirrors NumPy's ``np.atleast_1d``.  Higher-rank ``atleast_2d`` and
    ``atleast_3d`` variants follow the same pattern with extra leading /
    trailing unit axes.

    Examples
    --------
    >>> import lucid
    >>> s = lucid.tensor(3.0)
    >>> v = lucid.tensor([1.0, 2.0])
    >>> a, b = lucid.atleast_1d(s, v)
    >>> a.shape, b.shape
    ((1,), (2,))
    """
    out = [t_i.reshape(1) if t_i.ndim == 0 else t_i for t_i in tensors]
    return out[0] if len(out) == 1 else tuple(out)


def atleast_2d(*tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
    r"""Promote each input to at least 2 dimensions.

    Scalars become shape ``(1, 1)``, 1-D tensors gain a leading unit axis
    to become ``(1, N)``; tensors that already have rank :math:`\geq 2`
    pass through unchanged.

    Parameters
    ----------
    *tensors : Tensor
        One or more input tensors of arbitrary rank.

    Returns
    -------
    Tensor | tuple[Tensor, ...]
        A single tensor when called with one argument, otherwise a tuple
        of the promoted tensors in the same order as the inputs.

    Notes
    -----
    Promotion rule per input ``t`` with original shape :math:`\mathbf{s}`:

    .. math::

        \text{shape}'(t) =
        \begin{cases}
            (1, 1),           & \text{ndim}(t) = 0, \\
            (1, N),           & \mathbf{s} = (N,), \\
            \mathbf{s},       & \text{ndim}(t) \geq 2.
        \end{cases}

    The leading axis insertion (rather than trailing) matches NumPy's
    ``np.atleast_2d`` convention.

    Examples
    --------
    >>> import lucid
    >>> v = lucid.tensor([1.0, 2.0, 3.0])
    >>> lucid.atleast_2d(v).shape
    (1, 3)
    """
    out: list[Tensor] = []
    for t_i in tensors:
        if t_i.ndim == 0:
            t_i = t_i.reshape(1, 1)
        elif t_i.ndim == 1:
            t_i = t_i.unsqueeze(0)
        out.append(t_i)
    return out[0] if len(out) == 1 else tuple(out)


def atleast_3d(*tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
    r"""Promote each input to at least 3 dimensions.

    Scalars become shape ``(1, 1, 1)``, 1-D tensors become ``(1, N, 1)``,
    2-D tensors gain a trailing unit axis to become ``(M, N, 1)``;
    tensors with rank :math:`\geq 3` pass through unchanged.

    Parameters
    ----------
    *tensors : Tensor
        One or more input tensors of arbitrary rank.

    Returns
    -------
    Tensor | tuple[Tensor, ...]
        A single tensor when called with one argument, otherwise a tuple
        of the promoted tensors in the same order as the inputs.

    Notes
    -----
    Promotion rule per input ``t``:

    .. math::

        \text{shape}'(t) =
        \begin{cases}
            (1, 1, 1),    & \text{ndim}(t) = 0, \\
            (1, N, 1),    & \text{shape}(t) = (N,), \\
            (M, N, 1),    & \text{shape}(t) = (M, N), \\
            \text{shape}(t), & \text{ndim}(t) \geq 3.
        \end{cases}

    Note the asymmetric padding: 1-D tensors get both a leading and a
    trailing unit axis (so they live in the middle, image-like axis),
    while 2-D tensors gain only a trailing channel axis.  This matches
    NumPy and is convenient for image-processing code that expects
    ``(H, W, C)`` arrays.

    Examples
    --------
    >>> import lucid
    >>> v = lucid.tensor([1.0, 2.0])
    >>> lucid.atleast_3d(v).shape
    (1, 2, 1)
    """
    out: list[Tensor] = []
    for t_i in tensors:
        if t_i.ndim == 0:
            t_i = t_i.reshape(1, 1, 1)
        elif t_i.ndim == 1:
            t_i = t_i.reshape(1, -1, 1)
        elif t_i.ndim == 2:
            t_i = t_i.unsqueeze(2)
        out.append(t_i)
    return out[0] if len(out) == 1 else tuple(out)


# ── Splits ─────────────────────────────────────────────────────────────────


def _section_count(indices_or_sections: object, op: str) -> int | None:
    """``indices_or_sections`` as a section count, or ``None`` for cut indices.

    A count is any integer — a NumPy one or a 0-d integer tensor too — and
    indices are a sequence or a 1-D tensor.  Only a Python ``int`` used to
    count: ``np.int64(3)`` and ``tensor(3)`` were read as the one index 3.

    Raises
    ------
    TypeError
        A ``bool``, or neither an integer nor a sequence of them.
    """
    num = _promotion.as_number(indices_or_sections)
    if isinstance(num, int) and not isinstance(num, bool):
        return num
    if isinstance(indices_or_sections, lucid.Tensor):
        return None if indices_or_sections.ndim else operator.index(indices_or_sections)
    if isinstance(indices_or_sections, Sequence):
        return None
    if isinstance(indices_or_sections, SupportsIndex) and not isinstance(num, bool):
        return operator.index(indices_or_sections)
    raise TypeError(
        f"{op}(): indices_or_sections must be an int or a sequence of ints, "
        f"got {indices_or_sections!r}"
    )


def _split_along(
    x: Tensor,
    indices_or_sections: int | Sequence[int],
    dim: int,
    op: str = "tensor_split",
) -> list[Tensor]:
    """The pieces of ``x`` for a section count or a list of cut indices.

    Piece ``i`` of an index list is ``x[indices[i-1]:indices[i]]`` along
    ``dim``, read as a slice: a negative index counts from the end, one past
    either end is clamped to it, and a cut before the previous one gives an
    empty piece while the next piece still starts at that cut — so the
    pieces can overlap (``[5, 2]`` of 10 is 5, 0 and 8 long).  This used to
    become a size list for :func:`lucid.split`, dropping the negative sizes,
    and relied on ``split`` letting its last piece take up the difference;
    ``split`` refuses sizes that do not sum to the axis now, and the slices
    are cut here instead.

    Raises
    ------
    IndexError
        ``dim`` is past the last axis (a 0-d tensor's notional one included).
    lucid._C.engine.InvalidArgument
        A 0-d ``x``, or a section count that is not positive.
    """
    dim = _wrap_dim(dim, x.ndim, op)
    _require_real_axis(
        x.ndim,
        f"{op} expected at least a 1-dimensional tensor, but got a tensor " "with 0 dims",
    )
    n = _dim_size(x.shape, dim)
    k = _section_count(indices_or_sections, op)
    if k is not None:
        if k <= 0:
            raise _C_engine.InvalidArgument(f"number of sections must be larger than 0, got {k}")
        base, extra = divmod(n, k)
        return lucid.split(x, [base + 1] * extra + [base] * (k - extra), dim)
    if isinstance(indices_or_sections, lucid.Tensor):
        indices_or_sections = indices_or_sections.tolist()
    indices = _to_axes(indices_or_sections, op, "indices_or_sections")
    cuts = [min(max(i + n if i < 0 else i, 0), n) for i in indices]
    spans = list(itertools.pairwise([0, *cuts, n]))
    if all(start <= stop for start, stop in spans):
        return lucid.split(x, [stop - start for start, stop in spans], dim)
    return [lucid.narrow(x, dim, start, max(stop - start, 0)) for start, stop in spans]


def _split_evenly_along(
    op: str,
    input: Tensor,
    indices_or_sections: int | Sequence[int],
    min_ndim: int,
    dim: int,
) -> list[Tensor]:
    """``vsplit`` / ``hsplit`` / ``dsplit``: :func:`_split_along` with their checks.

    A section count must divide the axis, as the reference requires: it
    gave ragged pieces instead (``vsplit(zeros(4, 2), 3)`` was 2, 1, 1),
    which is ``tensor_split``'s reading.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        ``input`` has fewer than ``min_ndim`` dimensions, or a section count
        does not divide its axis.
    """
    if input.ndim < min_ndim:
        raise _C_engine.InvalidArgument(
            f"{op} requires a tensor with at least {min_ndim} dimension"
            f"{'s' if min_ndim > 1 else ''}, but got a tensor with {input.ndim} "
            "dimensions"
        )
    k = _section_count(indices_or_sections, op)
    size = int(input.shape[dim])
    if k is not None and (k == 0 or size % k):
        raise _C_engine.InvalidArgument(
            f"{op} attempted to split along dimension {dim}, but the size of the "
            f"dimension {size} is not divisible by the split_size {k}"
        )
    return _split_along(input, indices_or_sections, dim, op)


def vsplit(input: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    r"""Split a tensor along its first (vertical) axis.

    NumPy-style vertical split: cuts the input into pieces along axis 0.
    With an integer ``k``, the tensor is divided into ``k`` equal pieces;
    with a sequence of indices, the cuts occur at those positions.

    Parameters
    ----------
    input : Tensor
        Input tensor with at least 2 dimensions.
    indices_or_sections : int | Sequence[int]
        - ``int``: number of equal-sized splits; it must divide the axis
          length (:func:`tensor_split` takes one that does not).
        - ``Sequence[int]``: cut indices along axis 0.

    Returns
    -------
    list[Tensor]
        Sub-tensors whose stacking along axis 0 reproduces ``input``.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        If ``input.ndim < 2``, or a section count does not divide axis 0.

    Notes
    -----
    For a tensor of shape :math:`(N, \dots)` split into ``k`` pieces, each
    piece has shape :math:`(N/k, \dots)`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(12).reshape(4, 3)
    >>> [s.shape for s in lucid.vsplit(x, 2)]
    [(2, 3), (2, 3)]
    """
    return _split_evenly_along("vsplit", input, indices_or_sections, 2, 0)


def hsplit(input: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    r"""Split a tensor horizontally (along axis 1 for rank :math:`\geq 2`).

    NumPy-style horizontal split: cuts the input column-wise along axis 1
    for tensors of rank :math:`\geq 2`.  For 1-D inputs the axis collapses
    to 0, since there is only one dimension available.

    Parameters
    ----------
    input : Tensor
        Input tensor.  At least 1-D.
    indices_or_sections : int | Sequence[int]
        - ``int``: number of equal-sized splits; it must divide the axis
          length (:func:`tensor_split` takes one that does not).
        - ``Sequence[int]``: cut indices along the split axis (axis 1
          for rank :math:`\geq 2`, axis 0 for 1-D).

    Returns
    -------
    list[Tensor]
        Sub-tensors whose concatenation along the split axis reproduces
        ``input``.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        If ``input`` is 0-d, or a section count does not divide the axis.

    Notes
    -----
    Companion to :func:`vsplit` (axis 0) and :func:`dsplit` (axis 2).
    The axis-selection rule

    .. math::

        \text{axis} =
        \begin{cases}
            0, & \text{ndim}(x) = 1, \\
            1, & \text{ndim}(x) \geq 2,
        \end{cases}

    matches NumPy's convention so that 1-D inputs behave intuitively
    rather than raising on the missing column axis.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(12).reshape(3, 4)
    >>> [s.shape for s in lucid.hsplit(x, 2)]
    [(3, 2), (3, 2)]
    >>> v = lucid.arange(6)
    >>> [s.shape for s in lucid.hsplit(v, 3)]
    [(2,), (2,), (2,)]
    """
    dim = 0 if input.ndim == 1 else 1
    return _split_evenly_along("hsplit", input, indices_or_sections, 1, dim)


def dsplit(input: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    r"""Split a tensor along its third (depth) axis.

    NumPy-style depth-wise split: cuts the input into pieces along axis 2.
    Mirrors :func:`vsplit` (axis 0) and :func:`hsplit` (axis 1).

    Parameters
    ----------
    input : Tensor
        Input tensor with at least 3 dimensions.
    indices_or_sections : int | Sequence[int]
        - ``int``: number of equal-sized splits along axis 2; it must
          divide the axis length.
        - ``Sequence[int]``: cut indices along axis 2.

    Returns
    -------
    list[Tensor]
        Sub-tensors whose concatenation along axis 2 reproduces ``input``.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        If ``input.ndim < 3``, or a section count does not divide axis 2.

    Notes
    -----
    For a tensor of shape :math:`(\dots, D, \dots)` split into ``k`` pieces
    along axis 2, each piece has shape :math:`(\dots, D/k, \dots)`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(24).reshape(2, 3, 4)
    >>> [s.shape for s in lucid.dsplit(x, 2)]
    [(2, 3, 2), (2, 3, 2)]
    """
    return _split_evenly_along("dsplit", input, indices_or_sections, 3, 2)


def tensor_split(
    input: Tensor,
    indices_or_sections: int | Sequence[int],
    dim: int = 0,
) -> list[Tensor]:
    r"""Split a tensor along ``dim``, permitting unequal final-piece sizes.

    More permissive than :func:`lucid.split`: when ``indices_or_sections``
    is an integer ``k`` and ``x.shape[dim]`` is not divisible by ``k``,
    the first ``x.shape[dim] % k`` pieces receive one extra element each
    while the remaining pieces take the smaller floor size.  This mirrors
    NumPy's ``np.array_split`` semantics, where ``split`` raises on
    non-divisible counts but ``tensor_split`` quietly returns ragged
    pieces.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    indices_or_sections : int | Sequence[int]
        - ``int``: number of (near-)equal-sized splits.  The first
          ``axis_len % k`` pieces receive one extra element.
        - ``Sequence[int]``: cut indices along ``dim``.
    dim : int, optional
        Axis to split along.  Defaults to ``0``.

    Returns
    -------
    list[Tensor]
        Sub-tensors whose concatenation along ``dim`` reproduces ``input``.

    Notes
    -----
    For integer ``k`` with :math:`n = \text{shape}(x)[\text{dim}]` and
    :math:`b = \lfloor n / k \rfloor`, :math:`r = n \bmod k`, the piece
    sizes are

    .. math::

        \underbrace{b + 1, \dots, b + 1}_{r\text{ terms}},\;
        \underbrace{b, \dots, b}_{k - r\text{ terms}}.

    Contrast with :func:`lucid.split`, whose integer argument is a piece
    size and whose size list must sum to ``n``.

    With a list of indices, piece :math:`i` is the slice
    ``x[indices[i-1]:indices[i]]`` along ``dim``: a negative index counts
    from the end, an index past either end is clamped to it, and an index
    below the one before it gives an empty piece, with the next piece
    starting from that index again.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        If ``input`` is 0-d, or ``indices_or_sections`` is an integer that
        is not positive.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(10)
    >>> [s.shape for s in lucid.tensor_split(x, 3)]
    [(4,), (3,), (3,)]
    >>> [s.shape for s in lucid.tensor_split(x, [2, 5])]
    [(2,), (3,), (5,)]
    >>> [s.shape for s in lucid.tensor_split(x, [5, 2, 12])]
    [(5,), (0,), (8,), (0,)]
    """
    return _split_along(input, indices_or_sections, dim)


# ── Misc ───────────────────────────────────────────────────────────────────


def take_along_dim(input: Tensor, indices: Tensor, dim: int | None = None) -> Tensor:
    r"""Gather elements from ``input`` at positions ``indices`` along ``dim``.

    Advanced indexing primitive analogous to ``np.take_along_axis``:
    selects one element from ``input`` for every entry in ``indices``,
    broadcasting the remaining (non-``dim``) axes between the two
    tensors — :func:`lucid.gather` after that broadcast, which ``gather``
    itself does not do.  With ``dim=None`` both are read flattened.

    Parameters
    ----------
    input : Tensor
        Source tensor.
    indices : Tensor
        Integer tensor of positions along ``dim``, with as many dims as
        ``input``.  Its shape must be broadcast-compatible with ``input`` on every
        axis other than ``dim``; the size along ``dim`` controls the size
        of the output along that axis.  A negative position counts from
        the end.
    dim : int, optional
        Axis along which to gather.  Negative values count from the end.
        ``None`` (default) reads ``input`` and ``indices`` flattened.

    Returns
    -------
    Tensor
        Tensor with the broadcast shape of ``input`` and ``indices`` (with
        the size along ``dim`` taken from ``indices``), 1-D when ``dim``
        is ``None``.  Dtype matches ``input``.

    Raises
    ------
    lucid._C.engine.ShapeMismatch
        ``indices`` has another number of dims than ``input``.
    IndexError
        The shapes do not broadcast on an axis other than ``dim``.

    Notes
    -----
    For 1-D inputs the operation reduces to plain integer indexing.  For
    higher-rank inputs, the result satisfies

    .. math::

        \text{out}[i_0, \dots, i_{d-1}, j, i_{d+1}, \dots] =
        x[i_0, \dots, i_{d-1},\;
          \text{indices}[i_0, \dots, i_{d-1}, j, i_{d+1}, \dots],\;
          i_{d+1}, \dots].

    Typical uses include "gather the top-k elements per row" patterns
    after :func:`lucid.argsort` / :func:`lucid.topk`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([[1.0, 2.0, 3.0],
    ...                   [4.0, 5.0, 6.0]])
    >>> idx = lucid.tensor([[2, 0], [1, 2]])
    >>> lucid.take_along_dim(x, idx, dim=1)
    tensor([[3., 1.], [5., 6.]])
    """
    if dim is None:
        return lucid.gather(input.reshape(-1), 0, indices.reshape(-1))
    if indices.ndim != input.ndim:
        raise _C_engine.ShapeMismatch(
            f"take_along_dim: input and indices should have the same number of "
            f"dims, got {input.ndim} and {indices.ndim}"
        )
    dim = _wrap_dim(dim, input.ndim, "take_along_dim")
    if input.ndim == 0:
        raise IndexError("take_along_dim: dimension specified as 0 but tensor has no dimensions")
    x_shape = list(input.shape)
    i_shape = list(indices.shape)
    x_shape[dim] = i_shape[dim] = 1
    shape = _broadcast_shape([x_shape, i_shape])
    x_to = list(shape)
    x_to[dim] = int(input.shape[dim])
    i_to = list(shape)
    i_to[dim] = int(indices.shape[dim])
    return lucid.gather(input.broadcast_to(tuple(x_to)), dim, indices.broadcast_to(tuple(i_to)))


def _triangle_indices(
    row: int,
    col: int | None,
    offset: int,
    *,
    lower: bool,
    dtype: DTypeLike,
    device: DeviceLike,
) -> Tensor:
    """The ``(2, N)`` coordinates where ``j - i <= offset`` (``lower``) or
    ``>= offset`` holds in a ``(row, col)`` matrix, in row-major order.

    Found by one comparison over the matrix and one ``nonzero``, on the CPU
    — a size known before any value is read, so no device round trip.
    """
    if col is None:
        col = row
    rows = lucid.arange(row, dtype=lucid.int64, device="cpu").reshape(row, 1)
    cols = lucid.arange(col, dtype=lucid.int64, device="cpu").reshape(1, col)
    diagonal = cols - rows
    keep = diagonal <= offset if lower else diagonal >= offset
    coords = lucid.nonzero(keep).transpose(0, 1).contiguous()
    out_dtype: DTypeLike = dtype if dtype is not None else lucid.int64
    return coords.to(dtype=out_dtype, device=device if device is not None else get_default_device())


def tril_indices(
    row: int,
    col: int | None = None,
    offset: int = 0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Indices of the lower-triangular part of an ``(row, col)`` matrix.

    Parameters
    ----------
    row : int
        Number of rows in the matrix being indexed.
    col : int, optional
        Number of columns.  ``None`` (default) → ``col = row``
        (square matrix).
    offset : int, optional
        Diagonal offset.  ``0`` (default) = main diagonal,
        positive = include diagonals above it, negative = exclude
        diagonals below it.  Indexing keeps positions where
        ``j - i <= offset``.
    dtype : DTypeLike, optional
        Output dtype (default: int64).
    device : DeviceLike, optional
        Placement of the result.

    Returns
    -------
    Tensor
        Shape ``(2, N)``.  Row ``0`` holds row indices, row ``1`` holds
        column indices, in row-major order.
    """
    return _triangle_indices(row, col, offset, lower=True, dtype=dtype, device=device)


def triu_indices(
    row: int,
    col: int | None = None,
    offset: int = 0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Indices of the upper-triangular part of an ``(row, col)`` matrix.

    Mirrors :func:`tril_indices` with the inequality flipped — keeps
    entries where ``j - i >= offset``.

    Parameters
    ----------
    row : int
        Number of rows in the matrix being indexed.
    col : int, optional
        Number of columns.  ``None`` (default) → ``col = row``.
    offset : int, optional
        Diagonal offset.  ``0`` (default) = main diagonal,
        positive = exclude diagonals close to the main one,
        negative = include diagonals below it.
    dtype : DTypeLike, optional
        Output dtype (default: int64).
    device : DeviceLike, optional
        Placement of the result.

    Returns
    -------
    Tensor
        Shape ``(2, N)``.  Row ``0`` holds row indices, row ``1`` holds
        column indices, in row-major order.
    """
    return _triangle_indices(row, col, offset, lower=False, dtype=dtype, device=device)


def combinations(
    input: Tensor,
    r: int = 2,
    with_replacement: bool = False,
) -> Tensor:
    """All ``r``-length combinations of the elements of a 1-D ``input``.

    The positions of every combination are built on the CPU, then read
    from ``input`` with one ``index_select`` — differentiable through
    ``input``, and on its device.

    Parameters
    ----------
    input : Tensor
        1-D source tensor of length :math:`n`.  Raises on higher rank.
    r : int, optional
        Combination length.  Default ``2``.
    with_replacement : bool, optional
        When ``True`` the output enumerates multisets (an element can
        appear multiple times in a single row), giving
        :math:`\\binom{n + r - 1}{r}` rows.  Default ``False`` (strict
        combinations, :math:`\\binom{n}{r}` rows).

    Returns
    -------
    Tensor
        Shape ``(C, r)`` where :math:`C` is the combination count, in
        lexicographic order of positions; shape ``(0,)`` when ``r`` is 0.
        Dtype and device follow ``input``.

    Raises
    ------
    ValueError
        ``input`` is not 1-D, or ``r`` is negative.

    Examples
    --------
    >>> import lucid
    >>> lucid.combinations(lucid.tensor([1, 2, 3]), r=2)
    tensor([[1, 2], [1, 3], [2, 3]], dtype=lucid.int64)
    """
    if input.ndim != 1:
        raise ValueError(f"combinations: input must be 1-D, got shape {tuple(input.shape)}")
    if r < 0:
        raise ValueError(f"combinations: r must be non-negative, got {r}")
    if r == 0:
        return lucid.empty(0, dtype=input.dtype, device=input.device)
    rows = _combination_indices(int(input.shape[0]), r, with_replacement)
    picked = lucid.index_select(input, 0, rows.reshape(-1).to(input.device))
    return picked.reshape(int(rows.shape[0]), r)


def _combination_indices(n: int, r: int, with_replacement: bool) -> Tensor:
    """The ``(C, r)`` int64 positions of every combination, in lexicographic
    order, built on the CPU one column at a time.

    A row ending in ``last`` has a child for each position after it (from
    it, ``with_replacement``); the children of all rows are laid out in
    one run, and each finds its parent by where it falls in the running
    total of children — a ``searchsorted``, not a Python loop per row.
    """
    combos = lucid.arange(n, dtype=lucid.int64, device="cpu").reshape(n, 1)
    first = 0 if with_replacement else 1
    for _ in range(r - 1):
        last = combos[:, -1]
        children = (n - first) - last
        ends = lucid.cumsum(children, 0)
        total = int(ends[-1].item()) if ends.numel() else 0
        slots = lucid.arange(total, dtype=lucid.int64, device="cpu")
        parent = lucid.searchsorted(ends, slots, right=True)
        starts = ends - children
        position = last[parent] + first + (slots - starts[parent])
        combos = lucid.cat([combos[parent], position.reshape(total, 1)], dim=1)
    return combos


def rot90(input: Tensor, k: int = 1, dims: Sequence[int] = (0, 1)) -> Tensor:
    r"""Rotate a tensor by 90° in a chosen plane.

    Applies ``k`` successive 90° rotations in the plane spanned by the
    two axes ``dims = (d_0, d_1)``. The rotation direction is from
    ``d_0`` toward ``d_1`` (counter-clockwise when those axes are
    displayed as the usual ``(row, column)`` pair).

    Parameters
    ----------
    input : Tensor
        Input tensor (any rank :math:`\geq 2`).
    k : int, optional
        Number of 90° rotations to apply. Negative values rotate in the
        opposite direction. Reduced modulo 4 — values outside
        ``{0, 1, 2, 3}`` are equivalent to one of those four. Defaults
        to ``1``.
    dims : Sequence[int], optional
        Pair of axes defining the rotation plane. Defaults to ``(0, 1)``.

    Returns
    -------
    Tensor
        Tensor of the same rank as ``input``; the two axes in ``dims`` are
        permuted and one of them is flipped (their lengths swap when
        ``k`` is odd).

    Notes
    -----
    Implemented via flip + axis-swap:

    .. math::

        \text{rot90}_k(x) =
        \begin{cases}
            x,                                    & k \bmod 4 = 0, \\
            \operatorname{swap}_{d_0, d_1}(\operatorname{flip}_{d_1}(x)), & k \bmod 4 = 1, \\
            \operatorname{flip}_{d_0, d_1}(x),    & k \bmod 4 = 2, \\
            \operatorname{swap}_{d_0, d_1}(\operatorname{flip}_{d_0}(x)), & k \bmod 4 = 3.
        \end{cases}

    Applying ``rot90`` four times returns the original tensor.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([[1., 2.], [3., 4.]])
    >>> lucid.rot90(x)
    tensor([[2., 4.], [1., 3.]])
    """
    if len(dims) != 2:
        raise _C_engine.InvalidArgument(
            f"expected total rotation dims == 2, but got dims = {len(dims)}"
        )
    if input.ndim < 2:
        raise _C_engine.InvalidArgument(
            f"expected total dims >= 2, but got total dims = {input.ndim}"
        )
    d0 = _wrap_dim(dims[0], input.ndim, "rot90")
    d1 = _wrap_dim(dims[1], input.ndim, "rot90")
    if d0 == d1:
        raise _C_engine.InvalidArgument(
            f"expected rotation dims to be different, but got dim0 = {dims[0]} "
            f"and dim1 = {dims[1]}"
        )
    k = k % 4
    if k == 0:
        return input
    if k == 1:
        return _swap_dims(lucid.flip(input, [d1]), d0, d1)
    if k == 2:
        return lucid.flip(input, [d0, d1])
    return _swap_dims(lucid.flip(input, [d0]), d0, d1)


def diag_embed(
    input: Tensor,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> Tensor:
    """Embed the last dimension of ``input`` as the diagonal of a new matrix.

    Inverse of :func:`lucid.diagonal`.  For a 1-D input of length ``n``
    the result is shape ``(n+|offset|, n+|offset|)``.  For batch
    inputs the last axis is treated as the diagonal vector and
    ``dim1`` / ``dim2`` pick which two axes of the *output* carry the
    matrix.

    Parameters
    ----------
    input : Tensor
        Diagonal values.  Any shape; the last axis becomes the
        diagonal of each output matrix.
    offset : int, optional
        Diagonal offset.  ``0`` (default) = main diagonal,
        positive = above main, negative = below main.  Output size
        grows by ``|offset|`` along each matrix dimension.
    dim1 : int, optional
        First axis of the output matrix dimensions.  Default ``-2``.
    dim2 : int, optional
        Second axis of the output matrix dimensions.  Default ``-1``.

    Returns
    -------
    Tensor
        ``input.ndim + 1`` dims: ``input``'s leading dims in order, and two
        of size ``n + |offset|`` at ``dim1`` (rows) and ``dim2`` (columns).
        Off-diagonal positions are zero.

    Raises
    ------
    IndexError
        ``dim1`` or ``dim2`` outside ``[-(input.ndim + 1), input.ndim]``.
    ValueError
        ``dim1`` and ``dim2`` name the same dim.
    """
    out_ndim = input.ndim + 1
    rows = _wrap_dim(dim1, out_ndim, "diag_embed")
    cols = _wrap_dim(dim2, out_ndim, "diag_embed")
    if rows == cols:
        raise ValueError(f"diag_embed: dim1 and dim2 name the same dim ({dim1}, {dim2})")
    n = int(input.shape[-1])
    size = n + abs(offset)
    batch_shape = list(input.shape[:-1])
    row_off = max(0, -offset)
    col_off = max(0, offset)

    # Build (size, size) diagonal mask by padding eye(n) with zero rows/cols.
    zeros = input.new_zeros
    eye_n = lucid.eye(n, dtype=input.dtype, device=input.device)
    if col_off > 0:
        eye_n = lucid.cat([zeros(n, col_off), eye_n], dim=1)
    right = size - n - col_off
    if right > 0:
        eye_n = lucid.cat([eye_n, zeros(n, right)], dim=1)
    if row_off > 0:
        eye_n = lucid.cat([zeros(row_off, size), eye_n], dim=0)
    bot = size - n - row_off
    if bot > 0:
        eye_n = lucid.cat([eye_n, zeros(bot, size)], dim=0)
    # eye_n is now (size, size) with the diagonal at (row_off+i, col_off+i).

    # Broadcast input (..., n) against the mask (size, size); for a nonzero
    # offset, pad it to (..., size) with zeros first.
    diag_view = eye_n.reshape([1] * len(batch_shape) + [size, size])
    if offset == 0:
        out = input.reshape(batch_shape + [n, 1]) * diag_view
    else:
        pad_before = zeros(*batch_shape, row_off)
        pad_after = zeros(*batch_shape, size - n - row_off)
        x_pad = lucid.cat([pad_before, input, pad_after], dim=-1)  # (..., size)
        out = x_pad.reshape(batch_shape + [size, 1]) * diag_view
    # The matrix sits in the last two dims; ``dim1`` / ``dim2`` place it.
    last = (out_ndim - 2, out_ndim - 1)
    return out if (rows, cols) == last else lucid.movedim(out, last, (rows, cols))


def select(input: Tensor, dim: int, index: int) -> Tensor:
    r"""Slice ``input`` along ``dim`` at ``index``, removing that dimension.

    Parameters
    ----------
    input : Tensor
        Input tensor with at least 1 dimension.
    dim : int
        Dimension to slice.
    index : int
        Index to select along ``dim``. Negative indices count from the end.

    Returns
    -------
    Tensor
        View of ``input`` with dimension ``dim`` removed.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(6).reshape(2, 3)
    >>> lucid.select(x, 1, -1)
    tensor([2, 5], dtype=lucid.int64)
    """
    if input.ndim == 0:
        raise IndexError("select() cannot be applied to a 0-dim tensor.")
    d = _wrap_dim(dim, input.ndim, "select")
    idx = operator.index(index)
    size = int(input.shape[d])
    if idx < -size or idx >= size:
        raise IndexError(
            f"select(): index {idx} out of range for tensor of size "
            f"{list(input.shape)} at dimension {d}"
        )
    return input[(slice(None),) * d + (idx,)]


__all__ = [
    "broadcast_shapes",
    "broadcast_tensors",
    "diag_embed",
    "swapaxes",
    "swapdims",
    "moveaxis",
    "adjoint",
    "t",
    "column_stack",
    "row_stack",
    "dstack",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "vsplit",
    "hsplit",
    "dsplit",
    "tensor_split",
    "take_along_dim",
    "tril_indices",
    "triu_indices",
    "combinations",
    "rot90",
    "select",
]

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call. Imported at the bottom because this
# module is inside the cycle ``Tensor`` closes.
from lucid._tensor.tensor import Tensor
