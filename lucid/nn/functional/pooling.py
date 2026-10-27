"""
nn.functional pooling operations.
"""

from typing import Literal, TYPE_CHECKING, overload

from lucid.nn.functional._index_checks import _check_unpool
from lucid.nn.functional._pool_core import (
    _adaptive_max_pool,
    _adaptive_pool,
    _avg_pool,
    _fractional_max_pool,
    _lp_pool,
    _max_pool,
    _scatter_unpool,
)
from lucid.nn.functional._spatial_args import _ntuple, _pool_geometry, _spatial_extent

type _OutSize = int | tuple[int | None, ...]

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _unpool_default_size(
    x: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None,
    padding: int | tuple[int, ...],
    n: int,
) -> tuple[int, ...]:
    """The reference's default ``output_size`` for ``max_unpool``.

    The extent that pooling with these settings would have shrunk to
    ``x``'s: ``(in - 1) * stride - 2 * padding + kernel`` per axis.
    """
    k = _ntuple(kernel_size, n, f"max_unpool{n}d", "kernel_size")
    s = k if stride is None else _ntuple(stride, n, f"max_unpool{n}d", "stride")
    p = _ntuple(padding, n, f"max_unpool{n}d", "padding")
    extent = _spatial_extent(x, n, f"max_unpool{n}d")
    return tuple((i - 1) * sd - 2 * pd + kd for i, kd, sd, pd in zip(extent, k, s, p, strict=True))


@overload
def max_pool1d(
    input: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    ceil_mode: bool = False,
    return_indices: Literal[False] = False,
) -> Tensor: ...


@overload
def max_pool1d(
    input: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    *,
    return_indices: Literal[True],
    ceil_mode: bool = False,
) -> tuple[Tensor, Tensor]: ...


@overload
def max_pool1d(
    input: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]: ...


def max_pool1d(
    input: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""1-D max pooling over a sliding window.

    For each window, takes the maximum value — a translation-invariant
    feature aggregator that also provides a degree of non-linearity.
    Standard ingredient of 1-D temporal CNNs (text, audio, sensor data).

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, L)``, or ``(C, L)`` unbatched.
    kernel_size : int or tuple of int
        Size of the pooling window.
    stride : int or tuple of int, optional
        Window step.  Defaults to ``kernel_size`` (non-overlapping).
    padding : int or tuple of int, optional
        Implicit zero-padding on both sides of the spatial axis.
    dilation : int or tuple of int, optional
        Spacing between window elements.  Default ``1``.
    ceil_mode : bool, optional
        When ``True``, use ceil instead of floor in the output-size
        formula.
    return_indices : bool, optional
        Also return where each maximum came from: ``(output, indices)``,
        ``indices`` holding each window's flat position in the unpadded
        input — the form :func:`max_unpool1d` takes.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, L_out)`` where

        .. math::

            L_{\text{out}} = \left\lfloor \frac{L + 2 p - d(k - 1) - 1}{s} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,l} = \max_{m = 0, \ldots, k-1} x_{i,\,c,\,s l + d m - p}

    Max-pool is sub-differentiable: the gradient flows only through
    the position(s) holding the maximum in each window.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_pool1d
    >>> x = lucid.randn(2, 4, 16)
    >>> y = max_pool1d(x, kernel_size=2)
    >>> y.shape
    (2, 4, 8)
    """
    g = _pool_geometry(kernel_size, stride, padding, dilation, 1, "max_pool1d")
    return _max_pool(input, g, ceil_mode, return_indices)


@overload
def max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
    return_indices: Literal[False] = False,
) -> Tensor: ...


@overload
def max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    *,
    return_indices: Literal[True],
    ceil_mode: bool = False,
) -> tuple[Tensor, Tensor]: ...


@overload
def max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]: ...


def max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""2-D max pooling over a sliding window.

    Aggregates each spatial neighbourhood into its maximum value.
    The canonical downsampling primitive in classical image CNNs —
    provides invariance to small local shifts and reduces both spatial
    extent and compute for subsequent layers.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, H, W)``, or ``(C, H, W)`` unbatched.
    kernel_size : int or (int, int)
        Size of the pooling window.
    stride : int or (int, int), optional
        Window step.  Defaults to ``kernel_size`` (non-overlapping).
    padding : int or (int, int), optional
        Implicit zero-padding on each spatial side.
    dilation : int or (int, int), optional
        Spacing between window elements.  Default ``1``.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.
    return_indices : bool, optional
        Also return ``(output, indices)`` — see :func:`max_pool1d`; the
        indices are what :func:`max_unpool2d` takes.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, H_out, W_out)`` where each dim obeys

        .. math::

            H_{\text{out}} = \left\lfloor \frac{H + 2 p_H - d_H (k_H - 1) - 1}{s_H} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,h,w} = \max_{m,n} x_{i,\,c,\,s_H h + m,\,s_W w + n}

    Pure max is sub-differentiable; gradient routes to the per-window
    argmax position only.  Empirically the strongest pooling operator
    for classification tasks (preserves high-magnitude features).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_pool2d
    >>> x = lucid.randn(1, 16, 32, 32)
    >>> y = max_pool2d(x, kernel_size=2, stride=2)
    >>> y.shape
    (1, 16, 16, 16)
    """
    g = _pool_geometry(kernel_size, stride, padding, dilation, 2, "max_pool2d")
    return _max_pool(input, g, ceil_mode, return_indices)


def avg_pool1d(
    input: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    r"""1-D average pooling over a sliding window.

    Replaces each window with its arithmetic mean.  Smoother than
    max-pool and used wherever maintaining the overall response
    magnitude matters (e.g. before a fully connected classifier head).

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, L)``, or ``(C, L)`` unbatched.
    kernel_size : int or tuple of int
        Size of the pooling window.
    stride : int or tuple of int, optional
        Window step.  Defaults to ``kernel_size`` (non-overlapping).
    padding : int or tuple of int, optional
        Implicit zero-padding on both sides.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.
    count_include_pad : bool, optional
        When ``True``, padding cells contribute to the denominator.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, L_out)`` where

        .. math::

            L_{\text{out}} = \left\lfloor \frac{L + 2 p - k}{s} + 1 \right\rfloor

    Notes
    -----
    Math (averaging window :math:`R` of cardinality :math:`|R|`):

    .. math::

        y_{i,c,l} = \frac{1}{|R|} \sum_{m \in R} x_{i,\,c,\,s l + m - p}

    Unlike max-pool, average-pool is fully differentiable everywhere,
    so the gradient is uniformly distributed back across all window
    positions.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import avg_pool1d
    >>> x = lucid.randn(2, 3, 20)
    >>> y = avg_pool1d(x, kernel_size=4, stride=2)
    >>> y.shape
    (2, 3, 9)
    """
    g = _pool_geometry(kernel_size, stride, padding, 1, 1, "avg_pool1d")
    return _avg_pool(input, g, ceil_mode, count_include_pad, None)


def avg_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    r"""2-D average pooling over a sliding window.

    Replaces each spatial window with its arithmetic mean.  Common as
    the last spatial reduction before a fully connected classifier
    (modern nets often use :func:`adaptive_avg_pool2d` for the final
    pool because it adapts to arbitrary input sizes).

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, H, W)``, or ``(C, H, W)`` unbatched.
    kernel_size : int or (int, int)
        Size of the pooling window.
    stride : int or (int, int), optional
        Window step.  Defaults to ``kernel_size``.
    padding : int or (int, int), optional
        Implicit zero-padding on each spatial side.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.
    count_include_pad : bool, optional
        When ``True``, padding cells contribute to the denominator.
    divisor_override : int, optional
        Divide each window's sum by this instead of ``|R|`` (rare; useful
        for symmetric / weight-style averages).  Must not be ``0``.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, H_out, W_out)`` where each dim obeys

        .. math::

            H_{\text{out}} = \left\lfloor \frac{H + 2 p_H - k_H}{s_H} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,h,w} = \frac{1}{|R|} \sum_{(m, n) \in R} x_{i,\,c,\,s_H h + m,\,s_W w + n}

    Average-pool's gradient is the uniform distribution :math:`1/|R|`
    over the window, which produces smoother backward signals than
    max-pool.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import avg_pool2d
    >>> x = lucid.randn(1, 8, 28, 28)
    >>> y = avg_pool2d(x, kernel_size=2)
    >>> y.shape
    (1, 8, 14, 14)
    """
    g = _pool_geometry(kernel_size, stride, padding, 1, 2, "avg_pool2d")
    return _avg_pool(input, g, ceil_mode, count_include_pad, divisor_override)


def adaptive_avg_pool1d(input: Tensor, output_size: _OutSize) -> Tensor:
    r"""1-D adaptive average pooling — produces a fixed output length.

    Computes kernel / stride dynamically so that the output's spatial
    length equals ``output_size`` regardless of input length.  Handy
    when feeding variable-length sequences into a head expecting a
    fixed embedding size.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, L)``, or ``(C, L)`` unbatched.
    output_size : int or tuple of (int or None)
        Desired length of the output spatial axis.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, output_size)``.

    Notes
    -----
    For each output index :math:`i`, the corresponding input window is

    .. math::

        \big[\lfloor i \cdot L / L_{\text{out}} \rfloor,\;
              \lceil (i+1) \cdot L / L_{\text{out}} \rceil\big)

    and the cell is the mean over that window.  When ``L`` divides
    evenly by ``L_out`` the operation reduces to plain
    :func:`avg_pool1d` with kernel == stride == ``L / L_out``;
    otherwise Lucid falls back to a per-slot Python-level computation
    that still keeps all data on the original device.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_avg_pool1d
    >>> x = lucid.randn(2, 32, 13)
    >>> y = adaptive_avg_pool1d(x, output_size=5)
    >>> y.shape
    (2, 32, 5)
    """
    return _adaptive_pool(input, output_size, 1, "mean")


def adaptive_avg_pool2d(input: Tensor, output_size: _OutSize) -> Tensor:
    r"""2-D adaptive average pooling — produces a fixed ``(H, W)``.

    Computes per-axis kernel / stride so the output shape exactly
    matches ``output_size`` regardless of the input spatial size.  The
    workhorse "global pool to fixed grid" used by virtually every
    modern classification network (``output_size=(1, 1)`` collapses
    the spatial axes entirely).

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, H, W)``, or ``(C, H, W)`` unbatched.
    output_size : int or (int or None, int or None)
        Desired spatial shape of the output.  Scalar duplicates per
        axis; ``(1, 1)`` performs global average pooling.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, oH, oW)``.

    Notes
    -----
    For each output cell :math:`(i, j)` the corresponding input
    window is

    .. math::

        \big[\lfloor i H / H_{\text{out}} \rfloor,
              \lceil (i + 1) H / H_{\text{out}} \rceil\big) \times
        \big[\lfloor j W / W_{\text{out}} \rfloor,
              \lceil (j + 1) W / W_{\text{out}} \rceil\big)

    Foundational to Spatial Pyramid Pooling and the "global average
    pool + linear classifier" pattern introduced by NIN /
    ResNet-style architectures.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_avg_pool2d
    >>> x = lucid.randn(1, 512, 7, 9)
    >>> y = adaptive_avg_pool2d(x, output_size=(1, 1))
    >>> y.shape
    (1, 512, 1, 1)
    """
    return _adaptive_pool(input, output_size, 2, "mean")


@overload
def adaptive_max_pool2d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: Literal[False] = False
) -> Tensor: ...


@overload
def adaptive_max_pool2d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: Literal[True]
) -> tuple[Tensor, Tensor]: ...


@overload
def adaptive_max_pool2d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]: ...


def adaptive_max_pool2d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    r"""2-D adaptive max pooling — fixed-shape ``(H, W)`` via per-cell max.

    Like :func:`adaptive_avg_pool2d` but takes the maximum over each
    dynamically computed window instead of the mean.  Useful when
    salient peaks should drive the downstream representation
    (e.g. object detectors that rely on strong activations).

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, H, W)``, or ``(C, H, W)`` unbatched.
    output_size : int or (int or None, int or None)
        Desired spatial output shape.
    return_indices : bool, optional
        Also return ``(output, indices)``: each output cell's maximum as a
        flat position in the input's spatial extent.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, oH, oW)``.

    Notes
    -----
    For each output cell, the window definition is the same as in
    :func:`adaptive_avg_pool2d`; the cell value is

    .. math::

        y_{i,c,h,w} = \max_{(m, n) \in R(h, w)} x_{i,\,c,\,m,\,n}

    Gradient flows only through the per-window argmax position.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_max_pool2d
    >>> x = lucid.randn(1, 64, 11, 13)
    >>> y = adaptive_max_pool2d(x, output_size=(3, 3))
    >>> y.shape
    (1, 64, 3, 3)
    """
    return _adaptive_max_pool(input, output_size, 2, return_indices)


@overload
def adaptive_max_pool1d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: Literal[False] = False
) -> Tensor: ...


@overload
def adaptive_max_pool1d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: Literal[True]
) -> tuple[Tensor, Tensor]: ...


@overload
def adaptive_max_pool1d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]: ...


def adaptive_max_pool1d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    r"""1-D adaptive max pooling — produces a fixed output length.

    Computes kernel / stride dynamically so the output length equals
    ``output_size`` regardless of input length, then takes the
    per-window maximum.  Common in NLP / audio CNNs that feed
    fixed-size embeddings into a head.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, L)``, or ``(C, L)`` unbatched.
    output_size : int or tuple of (int or None)
        Desired output length.
    return_indices : bool, optional
        Also return ``(output, indices)``: each output cell's maximum as a
        flat position in the input's spatial extent.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, output_size)``.

    Notes
    -----
    The window for output index :math:`i` is

    .. math::

        \big[\lfloor i L / L_{\text{out}} \rfloor,\;
              \lceil (i+1) L / L_{\text{out}} \rceil\big)

    and the cell takes the max over that window.  Equivalent to
    :func:`max_pool1d` with kernel == stride == ``L / L_out`` when the
    division is exact.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_max_pool1d
    >>> x = lucid.randn(2, 8, 21)
    >>> y = adaptive_max_pool1d(x, output_size=4)
    >>> y.shape
    (2, 8, 4)
    """
    return _adaptive_max_pool(input, output_size, 1, return_indices)


@overload
def adaptive_max_pool3d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: Literal[False] = False
) -> Tensor: ...


@overload
def adaptive_max_pool3d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: Literal[True]
) -> tuple[Tensor, Tensor]: ...


@overload
def adaptive_max_pool3d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]: ...


def adaptive_max_pool3d(
    input: Tensor, output_size: int | tuple[int | None, ...], return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    r"""3-D adaptive max pooling — produces a fixed ``(D, H, W)``.

    Volumetric analogue of :func:`adaptive_max_pool2d`.  Computes
    per-axis kernel and stride so the output spatial shape exactly
    matches ``output_size`` regardless of input shape.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, D, H, W)``, or ``(C, D, H, W)`` unbatched.
    output_size : int or (int or None, int or None, int or None)
        Desired output spatial shape.
    return_indices : bool, optional
        Also return ``(output, indices)``: each output cell's maximum as a
        flat position in the input's spatial extent.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, oD, oH, oW)``.

    Notes
    -----
    The window for output cell :math:`(i, j, k)` is the 3-way product
    of axis-wise floor / ceil intervals (see
    :func:`adaptive_avg_pool2d` for the 2-D version).  The cell takes
    the maximum over that volume:

    .. math::

        y_{n,c,d,h,w} = \max_{(p, q, r) \in R(d, h, w)} x_{n,\,c,\,p,\,q,\,r}

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_max_pool3d
    >>> x = lucid.randn(1, 8, 5, 7, 9)
    >>> y = adaptive_max_pool3d(x, output_size=(1, 1, 1))
    >>> y.shape
    (1, 8, 1, 1, 1)
    """
    return _adaptive_max_pool(input, output_size, 3, return_indices)


def adaptive_avg_pool3d(input: Tensor, output_size: int | tuple[int | None, ...]) -> Tensor:
    r"""3-D adaptive average pooling — produces a fixed ``(D, H, W)``.

    Volumetric counterpart of :func:`adaptive_avg_pool2d`.  Used by
    3-D classification heads to collapse a volumetric feature map to
    a fixed embedding shape regardless of input size.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, D, H, W)``, or ``(C, D, H, W)`` unbatched.
    output_size : int or (int or None, int or None, int or None)
        Desired ``(oD, oH, oW)``; a single int applies to all three axes.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, oD, oH, oW)``.

    Notes
    -----
    Per-cell window definitions follow the same floor / ceil
    construction used in the 2-D variant.  The cell value is the mean
    over the volumetric window:

    .. math::

        y_{n,c,d,h,w} = \frac{1}{|R|}
            \sum_{(p, q, r) \in R(d, h, w)} x_{n,\,c,\,p,\,q,\,r}

    When the input dims divide ``output_size`` evenly the engine path
    is taken; otherwise a per-slot Python fallback runs (all data
    stays on-device).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import adaptive_avg_pool3d
    >>> x = lucid.randn(1, 16, 8, 10, 10)
    >>> y = adaptive_avg_pool3d(x, output_size=(2, 2, 2))
    >>> y.shape
    (1, 16, 2, 2, 2)
    """
    return _adaptive_pool(input, output_size, 3, "mean")


@overload
def max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    ceil_mode: bool = False,
    return_indices: Literal[False] = False,
) -> Tensor: ...


@overload
def max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    *,
    return_indices: Literal[True],
    ceil_mode: bool = False,
) -> tuple[Tensor, Tensor]: ...


@overload
def max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]: ...


def max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""3-D max pooling over a sliding window.

    Volumetric downsampling primitive — extends :func:`max_pool2d` by
    one depth dimension.  Used in 3-D CNNs for medical imaging and
    video understanding.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, D, H, W)``, or ``(C, D, H, W)`` unbatched.
    kernel_size : int or (int, int, int)
        Size of the pooling window per axis.
    stride : int or (int, int, int), optional
        Window step.  Defaults to ``kernel_size``.
    padding : int or (int, int, int), optional
        Implicit zero-padding on each spatial side.
    dilation : int or (int, int, int), optional
        Spacing between window elements.  Default ``1``.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.
    return_indices : bool, optional
        Also return ``(output, indices)`` — see :func:`max_pool1d`; the
        indices are what :func:`max_unpool3d` takes.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, D_out, H_out, W_out)`` where each
        spatial dim obeys

        .. math::

            D_{\text{out}} = \left\lfloor \frac{D + 2 p_D - d_D (k_D - 1) - 1}{s_D} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,d,h,w} = \max_{l,m,n} x_{i,\,c,\,s_D d + l,\,s_H h + m,\,s_W w + n}

    Memory cost can become significant for large volumes — when the
    network only needs a global summary, prefer
    :func:`adaptive_max_pool3d` with ``output_size=(1, 1, 1)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_pool3d
    >>> x = lucid.randn(1, 4, 8, 16, 16)
    >>> y = max_pool3d(x, kernel_size=2)
    >>> y.shape
    (1, 4, 4, 8, 8)
    """
    g = _pool_geometry(kernel_size, stride, padding, dilation, 3, "max_pool3d")
    return _max_pool(input, g, ceil_mode, return_indices)


def avg_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    r"""3-D average pooling over a sliding window.

    Volumetric counterpart of :func:`avg_pool2d` — replaces each 3-D
    window with its arithmetic mean.  Useful in 3-D CNNs as a smoother
    alternative to :func:`max_pool3d`, and as the final "global"
    reduction before a classifier head.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, D, H, W)``, or ``(C, D, H, W)`` unbatched.
    kernel_size : int or (int, int, int)
        Size of the pooling window per axis.
    stride : int or (int, int, int), optional
        Window step.  Defaults to ``kernel_size``.
    padding : int or (int, int, int), optional
        Implicit zero-padding on each spatial side.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.
    count_include_pad : bool, optional
        Whether padding cells contribute to the denominator.
    divisor_override : int, optional
        Divide each window's sum by this instead of ``|R|`` (never ``0``).

    Returns
    -------
    Tensor
        Output of shape ``(N, C, D_out, H_out, W_out)`` where each
        spatial dim obeys

        .. math::

            D_{\text{out}} = \left\lfloor \frac{D + 2 p_D - k_D}{s_D} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,c,d,h,w} = \frac{1}{|R|}
            \sum_{(l, m, n) \in R} x_{i,\,c,\,s_D d + l,\,s_H h + m,\,s_W w + n}

    Average-pool's gradient is uniformly distributed across the
    window, so it produces smoother backward signals than its
    max-based counterpart.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import avg_pool3d
    >>> x = lucid.randn(1, 4, 8, 16, 16)
    >>> y = avg_pool3d(x, kernel_size=2)
    >>> y.shape
    (1, 4, 4, 8, 8)
    """
    g = _pool_geometry(kernel_size, stride, padding, 1, 3, "avg_pool3d")
    return _avg_pool(input, g, ceil_mode, count_include_pad, divisor_override)


# ── P3 fills: lp_pool / max_unpool ──────────────────────────────────────────


def lp_pool1d(
    input: Tensor,
    norm_type: float,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""1-D power-average pooling — :math:`\big(\sum x^p\big)^{1/p}`.

    Generalises average and max pooling under a single parameter
    :math:`p = \text{norm\_type}`.  :math:`p = 1` recovers
    sum-pooling (= ``|R|`` × :func:`avg_pool1d`); :math:`p = 2` is
    energy / RMS-style pooling; :math:`p = \infty` is
    :func:`max_pool1d` of :math:`|x|`.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, L)``, or ``(C, L)`` unbatched.
    norm_type : float
        Exponent :math:`p > 0`, or ``inf``.
    kernel_size : int or tuple of int
        Size of the pooling window.
    stride : int or tuple of int, optional
        Window step.  Defaults to ``kernel_size``.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, L_out)``.

    Notes
    -----
    Math (per window :math:`R`):

    .. math::

        y_{i,c,l} = \left( \sum_{m \in R} x_{i,c,m}^p \right)^{1/p}

    The power keeps the sign of :math:`x`, as the reference's does: it is
    the :math:`\ell^p` norm of the window only for a non-negative input or
    an even :math:`p`.  A window whose sum is negative, or a negative
    element under a fractional :math:`p`, gives ``nan``; pool
    ``lucid.abs(x)`` for the norm.  A window whose sum is zero has a zero
    gradient.  Computed as ``(avg_pool1d(x^p) · |R|)^(1/p)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import lp_pool1d
    >>> x = lucid.randn(2, 4, 12)
    >>> y = lp_pool1d(x, norm_type=2.0, kernel_size=2)
    >>> y.shape
    (2, 4, 6)
    """
    g = _pool_geometry(kernel_size, stride, 0, 1, 1, "lp_pool1d")
    return _lp_pool(input, norm_type, g, ceil_mode)


def lp_pool2d(
    input: Tensor,
    norm_type: float,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""2-D power-average pooling — :math:`\big(\sum x^p\big)^{1/p}`.

    Two-dimensional version of :func:`lp_pool1d`.  The exponent
    :math:`p` interpolates between sum / average pooling (small
    :math:`p`) and max pooling (large :math:`p`), giving the network
    a tunable "softness" parameter.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, H, W)``, or ``(C, H, W)`` unbatched.
    norm_type : float
        Exponent :math:`p > 0`, or ``inf``.
    kernel_size : int or (int, int)
        Size of the pooling window.
    stride : int or (int, int), optional
        Window step.  Defaults to ``kernel_size``.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, H_out, W_out)``.

    Notes
    -----
    Math (per window :math:`R`):

    .. math::

        y_{i,c,h,w} = \left( \sum_{(m, n) \in R} x_{i,c,m,n}^p \right)^{1/p}

    The power keeps the sign of :math:`x`; see :func:`lp_pool1d`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import lp_pool2d
    >>> x = lucid.randn(1, 8, 16, 16)
    >>> y = lp_pool2d(x, norm_type=3.0, kernel_size=2)
    >>> y.shape
    (1, 8, 8, 8)
    """
    g = _pool_geometry(kernel_size, stride, 0, 1, 2, "lp_pool2d")
    return _lp_pool(input, norm_type, g, ceil_mode)


def lp_pool3d(
    input: Tensor,
    norm_type: float,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""3-D power-average pooling — :math:`\big(\sum x^p\big)^{1/p}`.

    Volumetric version of :func:`lp_pool2d`.  Interpolates between
    sum / average pooling (small :math:`p`) and max pooling (large
    :math:`p`) over a 3-D window.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, D, H, W)``, or ``(C, D, H, W)`` unbatched.
    norm_type : float
        Exponent :math:`p > 0`, or ``inf``.
    kernel_size : int or (int, int, int)
        Size of the pooling window per axis.
    stride : int or (int, int, int), optional
        Window step.  Defaults to ``kernel_size``.
    ceil_mode : bool, optional
        Use ceil instead of floor in the output-size formula.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, D_out, H_out, W_out)``.

    Notes
    -----
    Math (per window :math:`R`):

    .. math::

        y_{i,c,d,h,w} = \left( \sum_{(l, m, n) \in R} x_{i,c,l,m,n}^p \right)^{1/p}

    The power keeps the sign of :math:`x`; see :func:`lp_pool1d`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import lp_pool3d
    >>> x = lucid.randn(1, 2, 8, 8, 8)
    >>> y = lp_pool3d(x, norm_type=2.0, kernel_size=2)
    >>> y.shape
    (1, 2, 4, 4, 4)
    """
    g = _pool_geometry(kernel_size, stride, 0, 1, 3, "lp_pool3d")
    return _lp_pool(input, norm_type, g, ceil_mode)


def max_unpool1d(
    input: Tensor,
    indices: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    padding: int | tuple[int] = 0,
    output_size: tuple[int, ...] | None = None,
) -> Tensor:
    r"""Inverse of :func:`max_pool1d` via scatter at saved argmax indices.

    Restores a "sparse" higher-resolution feature map by placing each
    value of ``input`` at the spatial position recorded in ``indices``
    (i.e. the argmax positions captured by the corresponding max-pool
    forward pass).  Non-selected positions remain zero.  Standard
    component of encoder-decoder networks such as SegNet that
    propagate pooling indices through skip connections.

    Parameters
    ----------
    input : Tensor
        Pooled values, shape ``(N, C, L_in)``.
    indices : Tensor
        Per-window argmax positions (flattened spatial coordinate),
        same shape as ``input``.
    kernel_size : int or tuple of int
        Original pooling window size (recorded for symmetry — not used
        directly in the scatter math).
    stride : int or tuple of int, optional
        Original stride.  Recorded for symmetry.
    padding : int or tuple of int, optional
        Original padding.  Recorded for symmetry.
    output_size : tuple of int, optional
        Target length — typically that of the tensor fed to
        :func:`max_pool1d`.  Defaults, as the reference's does, to the
        length pooling with these settings shrinks to ``input``'s:
        ``(L - 1) * stride - 2 * padding + kernel_size``.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, L_out)`` containing ``input`` scattered
        into a zero buffer at the positions recorded in ``indices``.

    Notes
    -----
    Math (let :math:`\phi` be the bijection from pooled-cell index to
    saved flat input position):

    .. math::

        y_{n,c,\phi(n,c,l)} = x_{n,c,l},\qquad y_{n,c,j} = 0 \text{ otherwise}

    Implemented as a differentiable :func:`scatter_add` over a
    flattened spatial axis — gradients flow back to ``input``.
    ``output_size`` is required because Lucid does not yet emit
    argmax indices from the forward pool, so we cannot infer the
    original spatial extent.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_unpool1d
    >>> x = lucid.randn(1, 2, 4)
    >>> idx = lucid.tensor([[[0, 2, 4, 6], [1, 3, 5, 7]]], dtype=lucid.int64)
    >>> y = max_unpool1d(x, idx, kernel_size=2, output_size=(8,))
    >>> y.shape
    (1, 2, 8)
    """
    if output_size is None:
        output_size = _unpool_default_size(input, kernel_size, stride, padding, 1)
    spatial = tuple(int(s) for s in output_size[-1:])
    return _scatter_unpool(
        input, _check_unpool(input, indices, spatial, "max_unpool1d"), spatial, 1
    )


def max_unpool2d(
    input: Tensor,
    indices: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    output_size: tuple[int, ...] | None = None,
) -> Tensor:
    r"""Inverse of :func:`max_pool2d` via scatter at saved argmax indices.

    Two-dimensional version of :func:`max_unpool1d`.  Scatters the
    pooled values back to their original ``(H, W)`` positions
    (recorded as flat ``H × W`` indices) on a zero canvas.  Standard
    in SegNet-style decoders where decoder layers reuse pooling
    indices from the encoder for shape-preserving up-sampling.

    Parameters
    ----------
    input : Tensor
        Pooled values, shape ``(N, C, H_in, W_in)``.
    indices : Tensor
        Flattened argmax positions (``row * W + col``), same shape as
        ``input``.
    kernel_size : int or (int, int)
        Original pooling window size.
    stride : int or (int, int), optional
        Original stride.
    padding : int or (int, int), optional
        Original padding.
    output_size : tuple of int, optional
        Target spatial shape ``(..., H_out, W_out)`` — the original input
        shape to the matching max-pool.  Defaults, as the reference's
        does, to the shape pooling with these settings shrinks to ``input``'s,
        ``(in - 1) * stride - 2 * padding + kernel_size`` per axis.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, H_out, W_out)``.

    Notes
    -----
    Implemented via :func:`scatter_add` over a flattened spatial axis,
    fully differentiable.  See :func:`max_unpool1d` for the math.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_unpool2d, max_pool2d
    >>> x = lucid.randn(1, 1, 4, 4)
    >>> pooled, idx = max_pool2d(x, kernel_size=2, return_indices=True)
    >>> y = max_unpool2d(pooled, idx, kernel_size=2, output_size=(4, 4))
    >>> y.shape
    (1, 1, 4, 4)
    """
    if output_size is None:
        output_size = _unpool_default_size(input, kernel_size, stride, padding, 2)
    spatial = tuple(int(s) for s in output_size[-2:])
    return _scatter_unpool(
        input, _check_unpool(input, indices, spatial, "max_unpool2d"), spatial, 2
    )


def max_unpool3d(
    input: Tensor,
    indices: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    output_size: tuple[int, ...] | None = None,
) -> Tensor:
    r"""Inverse of :func:`max_pool3d` via scatter at saved argmax indices.

    Volumetric version of :func:`max_unpool2d`.  Scatters the pooled
    values back to their saved ``(D, H, W)`` positions (encoded as
    flat ``D × H × W`` indices) on a zero canvas.

    Parameters
    ----------
    input : Tensor
        Pooled values, shape ``(N, C, D_in, H_in, W_in)``.
    indices : Tensor
        Flattened argmax positions, same shape as ``input``.
    kernel_size : int or (int, int, int)
        Original pooling window size.
    stride : int or (int, int, int), optional
        Original stride.
    padding : int or (int, int, int), optional
        Original padding.
    output_size : tuple of int, optional
        Target spatial shape ``(..., D_out, H_out, W_out)``.  Defaults, as
        the reference's does, to the shape pooling with these settings
        shrinks to ``input``'s, ``(in - 1) * stride - 2 * padding +
        kernel_size`` per axis.

    Returns
    -------
    Tensor
        Output of shape ``(N, C, D_out, H_out, W_out)``.

    Notes
    -----
    Implemented via :func:`scatter_add` over a flattened spatial axis
    (so the gradient flows back to ``input`` and ``indices`` is treated
    as a non-differentiable lookup).  See :func:`max_unpool1d` for
    the math.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import max_pool3d, max_unpool3d
    >>> x = lucid.randn(1, 1, 4, 4, 4)
    >>> pooled, idx = max_pool3d(x, kernel_size=2, return_indices=True)
    >>> y = max_unpool3d(pooled, idx, kernel_size=2, output_size=(4, 4, 4))
    >>> y.shape
    (1, 1, 4, 4, 4)
    """
    if output_size is None:
        output_size = _unpool_default_size(input, kernel_size, stride, padding, 3)
    spatial = tuple(int(s) for s in output_size[-3:])
    return _scatter_unpool(
        input, _check_unpool(input, indices, spatial, "max_unpool3d"), spatial, 3
    )


@overload
def fractional_max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    output_size: int | tuple[int, int] | None = None,
    output_ratio: float | tuple[float, float] | None = None,
    return_indices: Literal[False] = False,
    _random_samples: Tensor | None = None,
) -> Tensor: ...


@overload
def fractional_max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    output_size: int | tuple[int, int] | None = None,
    output_ratio: float | tuple[float, float] | None = None,
    *,
    return_indices: Literal[True],
    _random_samples: Tensor | None = None,
) -> tuple[Tensor, Tensor]: ...


@overload
def fractional_max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    output_size: int | tuple[int, int] | None = None,
    output_ratio: float | tuple[float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]: ...


def fractional_max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    output_size: int | tuple[int, int] | None = None,
    output_ratio: float | tuple[float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Fractional max-pooling over a 2-D input (Graham, 2014).

    Performs max-pooling with a **non-integer** effective stride.
    Instead of a uniform window grid, random per-axis offsets pick
    "slightly irregular" window boundaries, yielding a smoother
    downsampling than the abrupt 2× reductions of standard max-pool.
    Acts as a structural regulariser for image classifiers.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, H, W)``, or ``(C, H, W)`` unbatched.
    kernel_size : int or (int, int)
        Size of the pooling window.
    output_size : int or (int, int), optional
        Exact desired output spatial shape.  Mutually exclusive with
        ``output_ratio``.
    output_ratio : float or (float, float), optional
        Multiplicative shrinkage factor per axis (e.g. ``0.5`` halves
        each dim).  Mutually exclusive with ``output_size``.
    return_indices : bool, optional
        When ``True``, also return per-output-cell argmax positions
        (flattened ``H × W`` index) as a second tensor.
    _random_samples : Tensor, optional
        Per-instance uniform samples of shape ``(N, C, 2)`` driving
        the window-boundary draws, ordered ``(width, height)`` as the
        reference orders them (``N = 1`` for an unbatched input).
        Defaults to fresh :func:`lucid.rand`.

    Returns
    -------
    Tensor or (Tensor, Tensor)
        Output of shape ``(N, C, oH, oW)``; when ``return_indices``,
        also the argmax indices of matching shape.

    Notes
    -----
    Window starts are drawn from Graham's formula (§3 of the paper)

    .. math::

        \alpha = \frac{H - k}{H_{\text{out}} - 1},\quad
        \text{start}_i = \lfloor (i + u) \alpha \rfloor - \lfloor u \alpha \rfloor

    where :math:`u` is a per-(sample, channel) uniform sample.  The
    final start clamps to ``H - k`` so all windows lie inside the
    input.  Implementation is a pure-Python composite — gradients
    flow through :meth:`Tensor.max`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import fractional_max_pool2d
    >>> x = lucid.randn(1, 4, 14, 14)
    >>> y = fractional_max_pool2d(x, kernel_size=2, output_ratio=0.5)
    >>> y.shape
    (1, 4, 7, 7)
    """
    args = (kernel_size, output_size, output_ratio, return_indices, _random_samples)
    return _fractional_max_pool(input, *args, 2)


@overload
def fractional_max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    output_size: int | tuple[int, int, int] | None = None,
    output_ratio: float | tuple[float, float, float] | None = None,
    return_indices: Literal[False] = False,
    _random_samples: Tensor | None = None,
) -> Tensor: ...


@overload
def fractional_max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    output_size: int | tuple[int, int, int] | None = None,
    output_ratio: float | tuple[float, float, float] | None = None,
    *,
    return_indices: Literal[True],
    _random_samples: Tensor | None = None,
) -> tuple[Tensor, Tensor]: ...


@overload
def fractional_max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    output_size: int | tuple[int, int, int] | None = None,
    output_ratio: float | tuple[float, float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]: ...


def fractional_max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    output_size: int | tuple[int, int, int] | None = None,
    output_ratio: float | tuple[float, float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Fractional max-pooling over a 3-D input (Graham, 2014).

    Volumetric extension of :func:`fractional_max_pool2d`.  Provides
    non-integer effective stride along all three spatial axes — a
    structural regulariser for 3-D CNNs.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, D, H, W)``, or ``(C, D, H, W)`` unbatched.
    kernel_size : int or (int, int, int)
        Size of the pooling window per axis.
    output_size : int or (int, int, int), optional
        Exact desired output spatial shape.  Mutually exclusive with
        ``output_ratio``.
    output_ratio : float or (float, float, float), optional
        Multiplicative shrinkage per axis.  Mutually exclusive with
        ``output_size``.
    return_indices : bool, optional
        When ``True``, also return per-output-cell argmax positions
        (flattened ``D × H × W`` index).
    _random_samples : Tensor, optional
        Per-instance uniform samples of shape ``(N, C, 3)``.  Defaults
        to fresh :func:`lucid.rand`.

    Returns
    -------
    Tensor or (Tensor, Tensor)
        Output of shape ``(N, C, oD, oH, oW)``; when
        ``return_indices``, also the argmax indices.

    Notes
    -----
    Each axis runs Graham's start-position formula independently
    (see :func:`fractional_max_pool2d` Notes).  Implementation is a
    pure-Python composite; gradients flow through :meth:`Tensor.max`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import fractional_max_pool3d
    >>> x = lucid.randn(1, 2, 8, 8, 8)
    >>> y = fractional_max_pool3d(x, kernel_size=2, output_ratio=0.5)
    >>> y.shape
    (1, 2, 4, 4, 4)
    """
    args = (kernel_size, output_size, output_ratio, return_indices, _random_samples)
    return _fractional_max_pool(input, *args, 3)
