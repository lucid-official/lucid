"""NaN-safe reductions composed from ``where`` + standard reductions."""

from typing import Sequence, TYPE_CHECKING

import lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._ops._dims import _wrap_dim
from lucid._ops._reduce import _MinMax, _ValuesIndices, _reduction_axes, _run_with_dtype
from lucid._types import DTypeLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def nansum(
    input: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: DTypeLike | None = None,
) -> Tensor:
    r"""Sum the tensor, treating NaN entries as zero.

    A NaN-safe variant of :func:`lucid.sum`. NaN values are replaced by
    ``0`` before the reduction, so they neither contaminate the result
    nor contribute to it.

    Parameters
    ----------
    input : Tensor
        Input tensor (any floating-point dtype).
    dim : int | Sequence[int] | None, optional
        Axis or axes along which to sum. ``None`` (default) reduces over
        the entire tensor.
    keepdim : bool, optional
        If ``True``, retains the reduced dimensions with size 1.
        Defaults to ``False``.
    dtype : DTypeLike | None, optional
        Target dtype for the reduction. When given, the input is cast to
        ``dtype`` before summing.

    Returns
    -------
    Tensor
        Reduced tensor.

    Notes
    -----
    With :math:`\mathcal{S}` the set of indices being reduced and
    :math:`\mathbb{1}_{x_i \text{ is NaN}}` an indicator,

    .. math::

        \text{nansum}(x) =
        \sum_{i \in \mathcal{S}}
        (1 - \mathbb{1}_{x_i \text{ is NaN}}) \cdot x_i.

    The gradient at NaN positions is zero — those entries contribute
    nothing to the forward sum, so they cannot influence it through
    perturbation.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([1.0, math.nan, 3.0])
    >>> lucid.nansum(x)
    tensor(4.)
    """
    axes = _reduction_axes(input._impl, dim, "nansum")
    safe = _C_engine.where(
        _C_engine.isnan(input._impl), _C_engine.zeros_like(input._impl), input._impl
    )
    return _wrap(
        _run_with_dtype(safe, dtype, lambda xi: _C_engine.sum(xi, axes, bool(keepdim)), op="nansum")
    )


def _nanmean_impl(
    x_impl: _C_engine.TensorImpl, axes: list[int], keepdim: bool
) -> _C_engine.TensorImpl:
    mask = _C_engine.isnan(x_impl)
    zero = _C_engine.zeros_like(x_impl)
    one = _C_engine.ones_like(x_impl)
    safe = _C_engine.where(mask, zero, x_impl)
    not_nan = _C_engine.where(mask, zero, one)
    return _C_engine.div(_C_engine.sum(safe, axes, keepdim), _C_engine.sum(not_nan, axes, keepdim))


def nanmean(
    input: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: DTypeLike | None = None,
) -> Tensor:
    r"""Mean of the tensor, ignoring NaN entries.

    A NaN-safe variant of :func:`lucid.mean`. Both the numerator (sum)
    and the denominator (element count) are computed only over the
    non-NaN entries.

    Parameters
    ----------
    input : Tensor
        Input tensor (any floating-point dtype).
    dim : int | Sequence[int] | None, optional
        Axis or axes along which to take the mean. ``None`` (default)
        reduces over the entire tensor.
    keepdim : bool, optional
        If ``True``, retains the reduced dimensions with size 1.
        Defaults to ``False``.
    dtype : DTypeLike | None, optional
        Target floating-point or complex dtype for the reduction. When
        given, ``input`` is cast to ``dtype`` before averaging.

    Returns
    -------
    Tensor
        Reduced tensor.

    Notes
    -----
    With :math:`\mathcal{S}` the set of indices being reduced and
    :math:`\mathbb{1}_i = 1 - \mathbb{1}_{x_i \text{ is NaN}}`,

    .. math::

        \text{nanmean}(x) =
        \frac{\sum_{i \in \mathcal{S}} \mathbb{1}_i \cdot x_i}
             {\sum_{i \in \mathcal{S}} \mathbb{1}_i}.

    A slice that is entirely NaN produces ``0 / 0 = NaN``; this is the
    standard convention. The gradient at NaN positions is zero.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([1.0, math.nan, 3.0])
    >>> lucid.nanmean(x)
    tensor(2.)
    """
    axes = _reduction_axes(input._impl, dim, "nanmean")
    return _wrap(
        _run_with_dtype(
            input._impl,
            dtype,
            lambda xi: _nanmean_impl(xi, axes, bool(keepdim)),
            op="nanmean",
            inexact_only=True,
        )
    )


def median(
    input: Tensor, dim: int | None = None, keepdim: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Median of a tensor along ``dim`` (or over the whole tensor when ``None``).

    Unlike :func:`nanmedian`, a ``NaN`` entry in a reduced slice propagates
    to the slice's median and its index names the first ``NaN`` in input
    order, matching the reference framework.

    Parameters
    ----------
    input : Tensor
        Real-valued input tensor (floating-point or integer).
    dim : int | None, optional
        Axis along which to take the median, or ``None`` for all elements.
    keepdim : bool, optional
        Whether the reduced dimension is kept with size ``1`` (default ``False``).

    Returns
    -------
    Tensor or tuple of Tensor
        With ``dim is None``, the scalar median.  With ``dim`` given,
        a ``(values, indices)`` named tuple.

    Examples
    --------
    >>> import lucid
    >>> lucid.median(lucid.tensor([1.0, 5.0, 3.0, 2.0]))
    tensor(2.)
    """
    return _run_median(input, dim, keepdim, ignore_nan=False, op="median")


def nanmedian(
    input: Tensor, dim: int | None = None, keepdim: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Median of a tensor, ignoring NaN entries.

    Computes the median over the requested axis (or the entire tensor
    when ``dim`` is ``None``), treating NaN entries as missing data
    rather than propagating them.  This is the NaN-safe counterpart of
    :func:`median`.

    Parameters
    ----------
    input : Tensor
        Input tensor of a floating or integer dtype.
    dim : int | None, optional
        Reduction axis.  ``None`` (default) flattens ``input`` first.
    keepdim : bool, optional
        If ``True``, retain the reduced axis as a singleton in the
        output shape.  Ignored when ``dim is None``.  Defaults to
        ``False``.

    Returns
    -------
    Tensor or tuple of Tensor
        With ``dim is None``, the scalar median of the non-NaN entries.
        With ``dim`` given, ``(values, indices)``: the median of every
        slice along ``dim`` and the ``int64`` position of that median in
        ``input``, both the input shape with ``dim`` removed (or kept as
        size 1 if ``keepdim=True``).

    Raises
    ------
    NotImplementedError
        For a ``bool`` or complex input.
    IndexError
        When ``dim`` is out of range, or the reduced axis has length 0.

    Notes
    -----
    Every slice keeps its own count :math:`n` of non-NaN entries.  NaN
    sorts last, so the median is the entry at sorted position
    :math:`\lfloor (n - 1) / 2 \rfloor` — the lower median for an even
    count, matching the reference framework.  A slice that is all NaN
    has no number to take and answers NaN, at the position of its first
    entry.  Integer inputs hold no NaN and take the plain lower median.

    The gradient of a ``dim`` reduction flows to the entry ``indices``
    names.  The whole-tensor median shares its gradient evenly between
    every entry equal to it (every NaN, when the answer is NaN), as the
    reference does.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([1.0, math.nan, 3.0, 5.0])
    >>> lucid.nanmedian(x)
    tensor(3.)
    >>> values, indices = lucid.nanmedian(
    ...     lucid.tensor([[1.0, math.nan, math.nan], [1.0, 2.0, 3.0]]), 1
    ... )
    >>> values
    tensor([1., 2.])
    >>> indices
    tensor([0, 1], dtype=lucid.int64)
    """
    return _run_median(input, dim, keepdim, ignore_nan=True, op="nanmedian")


def _run_median(
    input: Tensor, dim: int | None, keepdim: bool, *, ignore_nan: bool, op: str
) -> Tensor | tuple[Tensor, Tensor]:
    if input.dtype == lucid.bool_ or lucid.is_complex(input):
        raise NotImplementedError(
            f"{op} does not accept {input.dtype} — cast to an integer or floating dtype"
        )
    if dim is None:
        return _median_all(input, ignore_nan=ignore_nan, op=op)
    return _median_along(input, dim, keepdim, ignore_nan=ignore_nan, op=op)


def _pack_values_indices(
    values: Tensor, indices: Tensor, d: int, keepdim: bool
) -> _ValuesIndices[Tensor]:
    if not keepdim:
        return _ValuesIndices((values.squeeze(d), indices.squeeze(d)))
    return _ValuesIndices((values, indices))


def _median_along(
    x: Tensor, dim: int, keepdim: bool, *, ignore_nan: bool, op: str
) -> _ValuesIndices[Tensor]:
    """``(values, indices)`` of the median of every slice along ``dim``."""
    d = _wrap_dim(dim, x.ndim, op)
    if x.ndim == 0:
        return _median_along(x.reshape(1), 0, False, ignore_nan=ignore_nan, op=op)
    n = x.shape[d]
    if n == 0:
        raise IndexError(
            f"{op}: the reduced dim {dim} has length 0 — there is no " "median of nothing"
        )
    # The engine's sort puts NaN after every number, ``+inf`` included,
    # and keeps tied entries in their input order.
    order = lucid.argsort(x, d)
    if lucid.is_floating_point(x):
        count = lucid.sum(~lucid.isnan(x), d, True)
        if ignore_nan:
            pos = lucid.clamp((count - 1) // 2, min=0)
        else:
            pos = lucid.where(count < n, count, lucid.full_like(count, (n - 1) // 2))
    else:
        pos = lucid.full_like(order.narrow(d, 0, 1), (n - 1) // 2)
    indices = lucid.take_along_dim(order, pos, d)
    values = lucid.take_along_dim(x, indices, d)
    return _pack_values_indices(values, indices, d, keepdim)


def _median_all(x: Tensor, *, ignore_nan: bool, op: str) -> Tensor:
    """The whole-tensor median of ``x``."""
    flat = x.reshape(-1)
    if flat.shape[0] == 0:
        # There is no entry, so the answer is NaN.  A fresh constant would
        # cut the autograd edge; the mean of no entries is NaN as well.
        return lucid.mean(flat)
    value, _ = _median_along(flat, 0, False, ignore_nan=ignore_nan, op=op)
    if not x.requires_grad:
        return value
    # The reference hands every entry equal to the median an even share of
    # the gradient, not only the one the sort happened to put in the middle.
    picked = value.detach()
    tie = (flat == picked) | (lucid.isnan(flat) & lucid.isnan(picked))
    share = (1.0 / lucid.sum(tie).to(lucid.float32)).to(flat.dtype)
    ties = lucid.where(tie, flat, lucid.zeros_like(flat))
    # ``ties - ties.detach()`` is zero with a unit derivative, so ``picked``
    # keeps its exact value — except at ``inf`` or NaN, where it is NaN.
    # Those answers are spread instead: a sum of copies of ``inf`` or NaN,
    # scaled, is still exactly that.
    exact = picked + lucid.sum(ties - ties.detach()) * share
    spread = lucid.sum(ties) * share
    return lucid.where(lucid.isfinite(picked), exact, spread)


def mode(input: Tensor, dim: int = -1, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    r"""Return a ``(values, indices)`` pair of the modal value of each slice along ``dim``.

    Parameters
    ----------
    input : Tensor
        Input tensor of a real or boolean dtype.
    dim : int, optional
        Dimension to reduce.  Defaults to ``-1``.
    keepdim : bool, optional
        Whether the output tensors have ``dim`` retained with size 1.
        Defaults to ``False``.

    Returns
    -------
    tuple of Tensor
        ``(values, indices)`` named tuple of modal values and their ``int64``
        indices along ``dim``.

    Examples
    --------
    >>> import lucid
    >>> v, idx = lucid.mode(lucid.tensor([[1, 2, 2], [3, 1, 3]]))
    >>> v
    tensor([2, 3], dtype=lucid.int64)
    """
    if lucid.is_complex(input):
        raise NotImplementedError(f"mode does not accept {input.dtype}")
    d = _wrap_dim(dim, input.ndim, "mode")
    if input.ndim == 0:
        return _ValuesIndices(
            (input.clone(), lucid.zeros((), dtype=lucid.int64, device=input.device))
        )
    n = input.shape[d]
    if n == 0:
        raise IndexError(f"mode(): Expected reduction dim {d} to have non-zero size.")
    if input.numel() == 0:
        values = input.narrow(d, 0, 1)
        indices = lucid.zeros(values.shape, dtype=lucid.int64, device=input.device)
        return _pack_values_indices(values, indices, d, keepdim)
    moved = lucid.movedim(input, d, -1)
    order = lucid.argsort(moved, -1)
    s = lucid.take_along_dim(moved, order, -1)
    eq = s.unsqueeze(-1) == s.unsqueeze(-2)
    if lucid.is_floating_point(input):
        s_nan = lucid.isnan(s)
        eq = eq | (s_nan.unsqueeze(-1) & s_nan.unsqueeze(-2))
    tri = lucid.triu(lucid.ones((n, n), dtype=lucid.bool_, device=input.device))
    counts = lucid.sum(eq & tri, dim=-2)
    best_pos = lucid.argmax(counts, dim=-1, keepdim=True)
    indices = lucid.movedim(lucid.take_along_dim(order, best_pos, -1), -1, d)
    values = lucid.take_along_dim(input, indices, d)
    return _pack_values_indices(values, indices, d, keepdim)


def msort(input: Tensor) -> Tensor:
    r"""Sort the elements of ``input`` along its first dimension in ascending order.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor of the same shape and dtype as ``input`` sorted along ``dim=0``.

    Examples
    --------
    >>> import lucid
    >>> lucid.msort(lucid.tensor([[3.0, 1.0], [2.0, 0.0]]))
    tensor([[2., 0.], [3., 1.]])
    """
    return lucid.sort(input, dim=0).values


def count_nonzero(input: Tensor, dim: int | Sequence[int] | None = None) -> Tensor:
    """Count non-zero elements along ``dim`` (or whole tensor when ``None``).

    Parameters
    ----------
    input : Tensor
        Input.  Non-zero is determined by element-wise comparison to
        zero (NaN counts as non-zero).
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) reduces the entire
        tensor to a 0-D scalar.

    Returns
    -------
    Tensor
        ``int64`` count(s).  Shape mirrors :func:`lucid.sum` over the
        same ``dim`` argument.

    Notes
    -----
    The sum of the bool mask ``x != 0``, which is int64 and exact.  It
    used to build a float32 ``1.0/0.0`` mask with ``where`` and cast the
    float sum back — three full-size temporaries and ten times a plain
    reduction, from a time the CPU could not sum a bool.
    """
    mask = input != 0
    _dim2 = None if dim is None else _reduction_axes(input._impl, dim, "count_nonzero")
    return lucid.sum(mask) if _dim2 is None else lucid.sum(mask, _dim2, False)


def amax(input: Tensor, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
    """Reduce ``input`` to per-slice maxima along ``dim`` — values only.

    Multi-axis sibling of :func:`lucid.max` that returns *only* the
    reduced values (never the indices) and accepts a tuple of axes
    for a single fused reduction.  Use when you don't need the
    ``(values, indices)`` tuple :func:`lucid.max` returns.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) reduces over every
        axis and returns a 0-D scalar tensor.
    keepdim : bool, optional
        Retain the reduced axes as size-1 dimensions.  Default
        ``False``.

    Returns
    -------
    Tensor
        Maxima along ``dim``; shape depends on ``dim`` and ``keepdim``.

    See Also
    --------
    amin : analogous minimum reduction.
    lucid.max : returns ``(values, indices)`` for a single axis.
    """
    return _wrap(
        _C_engine.max(input._impl, _reduction_axes(input._impl, dim, "amax"), bool(keepdim))
    )


def amin(input: Tensor, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
    """Reduce ``input`` to per-slice minima along ``dim`` — values only.

    Mirror image of :func:`amax`.  Accepts a tuple of axes for a fused
    multi-axis reduction; returns just the minimum values (never the
    indices).

    Parameters
    ----------
    input : Tensor
        Input tensor.
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) reduces over every
        axis to a 0-D scalar.
    keepdim : bool, optional
        Retain the reduced axes as size-1 dimensions.  Default
        ``False``.

    Returns
    -------
    Tensor
        Minima along ``dim``.

    See Also
    --------
    amax : analogous maximum reduction.
    lucid.min : returns ``(values, indices)`` for a single axis.
    """
    return _wrap(
        _C_engine.min(input._impl, _reduction_axes(input._impl, dim, "amin"), bool(keepdim))
    )


def aminmax(input: Tensor, *, dim: int | None = None, keepdim: bool = False) -> _MinMax[Tensor]:
    r"""Compute the minimum and maximum of ``input``.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    dim : int or None, optional
        Dimension along which to compute minimum and maximum. When ``None``
        (default), reduces over the entire tensor.
    keepdim : bool, optional
        If ``True``, retains the reduced dimensions with size 1.

    Returns
    -------
    tuple[Tensor, Tensor]
        Named pair ``(min, max)`` with ``.min`` and ``.max`` attributes.

    Examples
    --------
    >>> import lucid
    >>> out = lucid.aminmax(lucid.tensor([3.0, -1.0, 4.0]))
    >>> (out.min.item(), out.max.item())
    (-1.0, 4.0)
    """
    dim_num = _promotion.as_number(dim)
    if dim_num is not None and (isinstance(dim_num, bool) or not isinstance(dim_num, int)):
        raise TypeError(f"aminmax(): argument 'dim' must be int, not {type(dim).__name__}")
    if dim_num is None:
        lo = lucid.min(input)
        hi = lucid.max(input)
        if keepdim and input.ndim > 0:
            ones = (1,) * input.ndim
            lo = lo.reshape(ones)
            hi = hi.reshape(ones)
        return _MinMax((lo, hi))
    axes = _reduction_axes(input._impl, dim_num, "aminmax")
    return _MinMax((lucid.min(input, axes, keepdim), lucid.max(input, axes, keepdim)))


def _logaddexp_pair(a: Tensor, b: Tensor) -> Tensor:
    m_raw = lucid.maximum(a, b)
    m = lucid.where(lucid.isinf(m_raw), lucid.zeros_like(m_raw), m_raw)
    return m + lucid.log(lucid.exp(a - m) + lucid.exp(b - m))


def logcumsumexp(input: Tensor, dim: int) -> Tensor:
    r"""Return the logarithm of the cumulative sum of exponentials of ``input`` along ``dim``.

    Parameters
    ----------
    input : Tensor
        Floating-point input tensor.
    dim : int
        Dimension along which to compute the scan.

    Returns
    -------
    Tensor
        Tensor of the same shape and dtype as ``input``.

    Examples
    --------
    >>> import lucid
    >>> lucid.logcumsumexp(lucid.tensor([0.0, 0.0]), dim=0)
    tensor([0., 0.6931])
    """
    if not lucid.is_floating_point(input):
        raise _C_engine.NotImplementedError(f"logcumsumexp is not implemented for {input.dtype}")
    d = _wrap_dim(dim, input.ndim, "logcumsumexp")
    if input.ndim == 0 or input.shape[d] <= 1:
        return input.clone()
    n = int(input.shape[d])
    work_dt = lucid.float32 if input.dtype in (lucid.float16, lucid.bfloat16) else input.dtype
    acc = input.to(work_dt) if work_dt != input.dtype else input
    step = 1
    while step < n:
        head = acc.narrow(d, 0, step)
        pair = _logaddexp_pair(acc.narrow(d, 0, n - step), acc.narrow(d, step, n - step))
        acc = lucid.cat([head, pair], dim=d)
        step <<= 1
    return acc.to(input.dtype) if work_dt != input.dtype else acc


def renorm(input: Tensor, p: float, dim: int, maxnorm: float) -> Tensor:
    r"""Rescale each sub-tensor along ``dim`` whose ``p``-norm exceeds ``maxnorm``.

    Parameters
    ----------
    input : Tensor
        Floating-point or complex tensor with at least 2 dimensions.
    p : float
        Power for the norm calculation (must be positive).
    dim : int
        Dimension to slice over in order to obtain the sub-tensors.
    maxnorm : float
        Maximum allowed norm for each sub-tensor (must be non-negative).

    Returns
    -------
    Tensor
        Renormalized tensor of the same shape and dtype as ``input``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([[3.0, 4.0], [0.3, 0.4]])
    >>> lucid.renorm(x, p=2.0, dim=0, maxnorm=1.0)
    tensor([[0.6, 0.8], [0.3, 0.4]])
    """
    if p <= 0:
        raise RuntimeError("renorm: non-positive-norm not supported")
    if maxnorm < 0:
        raise RuntimeError(f"renorm: expected maxnorm to be >= 0 but got {maxnorm}")
    if input.ndim < 2:
        raise RuntimeError(
            f"renorm: input needs at least 2 dimensions, got {input.ndim} dimensions"
        )
    d = _wrap_dim(dim, input.ndim, "renorm")
    if input.numel() == 0:
        return input.clone()
    is_half = input.dtype in (lucid.float16, lucid.bfloat16)
    work = input.to(lucid.float32) if is_half else input
    rdims = [i for i in range(input.ndim) if i != d]
    norm = lucid.linalg.vector_norm(work, ord=p, dim=rdims, keepdim=True)
    scale = lucid.where(norm > maxnorm, maxnorm / (norm + 1e-7), lucid.ones_like(norm))
    out = work * scale
    return out.to(input.dtype) if is_half else out


def _align_trapezoid_x(name: str, yw: Tensor, x: Tensor, d: int) -> tuple[Tensor, Tensor]:
    if x.dtype == lucid.bool:
        raise RuntimeError(f"{name}: received a bool input for `x`, but bool is not supported")
    target_dt = lucid.result_type(yw, x)
    yw = yw.to(target_dt) if yw.dtype != target_dt else yw
    xw = x.to(target_dt) if x.dtype != target_dt else x
    if xw.ndim == 1 and xw.shape[0] == yw.shape[d]:
        shape = [int(xw.shape[0]) if i == d else 1 for i in range(yw.ndim)]
        return yw, xw.reshape(shape)
    if xw.ndim == yw.ndim and xw.shape[d] == yw.shape[d]:
        return yw, xw
    raise RuntimeError(f"{name}: There must be one `x` value for each sample point")


def _trapezoid_operands(
    name: str, y: Tensor, x: Tensor | None, dx: float | None, dim: int
) -> tuple[Tensor, Tensor | None, int]:
    if y.dtype == lucid.bool:
        raise RuntimeError(f"{name}: received a bool input for `y`, but bool is not supported")
    if x is not None and dx is not None:
        raise TypeError(
            f"{name}() received an invalid combination of arguments - "
            "cannot specify both `x` and `dx`"
        )
    if y.ndim == 0:
        msg = (
            f"Dimension specified as {dim} but tensor has no dimensions"
            if name == "trapezoid"
            else "slice() cannot be applied to a 0-dim tensor."
        )
        raise IndexError(msg)
    d = _wrap_dim(dim, y.ndim, name)
    is_inexact = lucid.is_floating_point(y) or lucid.is_complex(y)
    yw = y if is_inexact else y.to(lucid.get_default_dtype())
    if x is None:
        return yw, None, d
    yw, xw = _align_trapezoid_x(name, yw, x, d)
    return yw, xw, d


def _trapezoid_areas(
    name: str, y: Tensor, x: Tensor | None, dx: float | None, dim: int
) -> tuple[Tensor, int]:
    yw, xw, d = _trapezoid_operands(name, y, x, dx, dim)
    n = int(yw.shape[d])
    if n < 2:
        return yw.narrow(d, 0, 0), d
    avg = (yw.narrow(d, 0, n - 1) + yw.narrow(d, 1, n - 1)) * 0.5
    if xw is not None:
        return (xw.narrow(d, 1, n - 1) - xw.narrow(d, 0, n - 1)) * avg, d
    return avg * (1.0 if dx is None else dx), d


def trapezoid(
    y: Tensor, x: Tensor | None = None, *, dx: float | None = None, dim: int = -1
) -> Tensor:
    r"""Integrate ``y`` along ``dim`` using the composite trapezoidal rule.

    Parameters
    ----------
    y : Tensor
        Function values sampled along ``dim``.
    x : Tensor or None, optional
        Abscissa coordinates matching ``y`` along ``dim``.
    dx : float or None, optional
        Uniform step width used when ``x`` is ``None`` (defaults to ``1.0``).
    dim : int, optional
        Integration axis (default ``-1``).

    Returns
    -------
    Tensor
        Definite trapezoidal integral with ``dim`` reduced.

    Examples
    --------
    >>> import lucid
    >>> lucid.trapezoid(lucid.tensor([1.0, 2.0, 4.0]))
    tensor(4.5)
    """
    areas, d = _trapezoid_areas("trapezoid", y, x, dx, dim)
    return lucid.sum(areas, dim=d)


def cumulative_trapezoid(
    y: Tensor, x: Tensor | None = None, *, dx: float | None = None, dim: int = -1
) -> Tensor:
    r"""Compute the running trapezoidal integral of ``y`` along ``dim``.

    Parameters
    ----------
    y : Tensor
        Samples to integrate cumulatively.
    x : Tensor or None, optional
        Grid coordinates along ``dim``; mutually exclusive with ``dx``.
    dx : float or None, optional
        Constant grid spacing when ``x`` is omitted.
    dim : int, optional
        Axis of cumulative integration (default ``-1``).

    Returns
    -------
    Tensor
        Running integral whose ``dim`` extent is ``y.shape[dim] - 1``.

    Examples
    --------
    >>> import lucid
    >>> lucid.cumulative_trapezoid(lucid.tensor([1.0, 2.0, 4.0]))
    tensor([1.5, 4.5])
    """
    areas, d = _trapezoid_areas("cumulative_trapezoid", y, x, dx, dim)
    return lucid.cumsum(areas, dim=d)


def _validate_diff_extra(input: Tensor, d: int, extra: Tensor | None) -> None:
    if extra is None:
        return
    if extra.ndim != input.ndim or any(
        extra.shape[i] != input.shape[i] for i in range(input.ndim) if i != d
    ):
        raise _C_engine.InvalidArgument(
            "diff expects prepend and append to match input shape " "on all dimensions except dim"
        )


def _diff(
    input: Tensor,
    n: int = 1,
    dim: int = -1,
    prepend: Tensor | None = None,
    append: Tensor | None = None,
) -> Tensor:
    """Compute the ``n``-th discrete difference along ``dim``."""
    d = _wrap_dim(dim, input.ndim, "diff")
    if input.ndim == 0:
        raise _C_engine.InvalidArgument("diff expects input to be at least one-dimensional")
    if n < 0:
        raise _C_engine.InvalidArgument(f"order must be non-negative but got {n}")
    _validate_diff_extra(input, d, prepend)
    _validate_diff_extra(input, d, append)
    if n == 0:
        return input
    parts = [t for t in (prepend, input, append) if t is not None]
    result = lucid.cat(parts, dim=d) if len(parts) > 1 else input
    for _ in range(n):
        length = int(result.shape[d])
        if length <= 1:
            result = result.narrow(d, 0, 0)
            continue
        a = result.narrow(d, 1, length - 1)
        b = result.narrow(d, 0, length - 1)
        result = lucid.logical_xor(a, b) if result.dtype == lucid.bool else a - b
    return result


__all__ = [
    "nansum",
    "nanmean",
    "median",
    "nanmedian",
    "mode",
    "msort",
    "count_nonzero",
    "amax",
    "amin",
    "aminmax",
    "logcumsumexp",
    "renorm",
    "trapezoid",
    "cumulative_trapezoid",
]

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call. Imported at the bottom because this
# module is inside the cycle ``Tensor`` closes.
from lucid._tensor.tensor import Tensor
