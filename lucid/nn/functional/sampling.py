"""
nn.functional sampling / interpolation / padding operations.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry
from lucid._ops._reduce import _require_floating
from lucid.nn.functional._inplace import _keep_in_graph
from lucid.nn.functional._grid_sampling import (
    _as_real,
    affine_grid_3d as _affine_grid_3d,
    check_options as _check_options,
    composite_grid_sample as _composite_grid_sample,
    opmath as _opmath,
    reflect_grid as _reflect_grid,
)
from lucid.nn.functional._interpolate import resample as _resample
from lucid.nn.functional._spatial_args import _add_batch_axis, _drop_batch_axis, _ntuple
from lucid.nn.functional._index_checks import _check_offsets, _check_table, _padding_index
from lucid.nn.functional.sparse import _bag_by_lookup, _bag_of_each_index, _bag_mode, _renorm_rows

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# What the engine kernel implements, in its encoding.  Anything else is
# sampled by the composite, or rewritten (reflection) before it gets there.
_KERNEL_MODES = {"bilinear": 0, "nearest": 1}
_KERNEL_PADDINGS = {"zeros": 0, "border": 1}


def interpolate(
    input: Tensor,
    size: int | tuple[int, ...] | None = None,
    scale_factor: float | tuple[float, ...] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
    antialias: bool = False,
) -> Tensor:
    r"""Resample an N-D tensor to a target spatial size or scale factor.

    Supports the standard family of image / volume resampling kernels.
    The leading two axes (``N`` batch and ``C`` channels) are preserved;
    only the trailing spatial axes are resampled.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, *spatial)``.  ``"nearest"``,
        ``"nearest-exact"`` and ``"area"`` take 1, 2 or 3 spatial axes;
        ``"linear"`` takes 1, ``"bilinear"`` / ``"bicubic"`` / ``"lanczos"``
        2 and ``"trilinear"`` 3.
    size : int or tuple of int, optional
        Target spatial size.  Mutually exclusive with ``scale_factor``.
    scale_factor : float or tuple of float, optional
        Multiplicative factor applied to each spatial axis.  Output size
        is ``floor(in_size * scale_factor)``.
    mode : str, optional
        One of ``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``,
        ``"bicubic"``, ``"lanczos"``, ``"trilinear"``, ``"area"``.
        Default ``"nearest"``.
    align_corners : bool, optional
        Only accepted by ``"linear"`` / ``"bilinear"`` / ``"bicubic"`` /
        ``"trilinear"`` (and ``False`` for ``"lanczos"``).  If ``True``, the
        corner pixels of the input and output are aligned and their values
        are preserved.  If ``False`` (default), pixels are treated as
        ``1×1`` squares and the sampling grid is centred.
    recompute_scale_factor : bool, optional
        If ``True``, the sampling step is ``in_size / out_size`` from the
        resolved output size rather than ``1 / scale_factor``.  Not
        allowed together with ``size``.
    antialias : bool, optional
        For ``"bilinear"``, ``"bicubic"`` and ``"lanczos"`` (required for
        ``"lanczos"``): when downsampling, widen the filter over every input
        sample an output sample covers, so detail finer than the output
        grid is averaged rather than aliased.  Default ``False``.

    Returns
    -------
    Tensor
        Resampled tensor of shape ``(N, C, *out_spatial)``.

    Raises
    ------
    ValueError
        For an unknown mode, a mode given the wrong number of spatial
        axes, both or neither of ``size`` and ``scale_factor``,
        ``align_corners`` or ``antialias`` with a mode that has no use
        for it, or a non-positive size.
    TypeError
        For an integer input with any mode but the two nearest ones.

    Notes
    -----
    Output sample :math:`d` reads the input around

    .. math::

        x_{\mathrm{src}} = \begin{cases}
            d \cdot \frac{S_{\mathrm{in}} - 1}{S_{\mathrm{out}} - 1}
                & \text{align\_corners} \\
            (d + 0.5) \cdot r - 0.5 & \text{otherwise}
        \end{cases}

    where :math:`r = 1 / \text{scale\_factor}` when a scale factor is
    given and ``recompute_scale_factor`` is not ``True``, and
    :math:`r = S_{\mathrm{in}} / S_{\mathrm{out}}` otherwise.  The two
    differ whenever :math:`S_{\mathrm{in}} \cdot \text{scale\_factor}`
    is not a whole number.  ``"nearest"`` copies sample
    :math:`\lfloor d \cdot r \rfloor` and ``"nearest-exact"`` sample
    :math:`\lfloor (d + 0.5) \cdot r \rfloor`; ``"area"`` is adaptive
    average pooling to the output size.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import interpolate
    >>> x = lucid.randn(1, 3, 32, 32)
    >>> y = interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    >>> y.shape
    (1, 3, 64, 64)
    >>> interpolate(x, scale_factor=0.25, mode="bicubic", antialias=True).shape
    (1, 3, 8, 8)
    """
    return _resample(
        input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias
    )


def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool | None = None,
) -> Tensor:
    r"""Sample an input feature map at flow-field coordinates.

    For each output location, looks up a value in ``input`` at the
    (sub-pixel) location specified by ``grid``, using bilinear,
    nearest or bicubic interpolation.  Forms the second half of a spatial
    transformer network (STN) when paired with :func:`affine_grid`.

    The grid uses normalised coordinates in :math:`[-1, 1]`:
    :math:`(-1, -1)` denotes the top-left corner of ``input`` and
    :math:`(+1, +1)` the bottom-right corner (with the
    ``align_corners=False`` convention placing corners just outside the
    edge pixels).

    Parameters
    ----------
    input : Tensor
        Source feature map of shape ``(N, C, H_in, W_in)`` (4-D) or
        ``(N, C, D_in, H_in, W_in)`` (5-D for volumetric sampling).
    grid : Tensor
        Sampling grid of shape ``(N, H_out, W_out, 2)`` for 2-D input
        or ``(N, D_out, H_out, W_out, 3)`` for 3-D input.  Last axis
        stores ``(x, y)`` (or ``(x, y, z)``) in :math:`[-1, 1]`.
    mode : str, optional
        Interpolation mode: ``"bilinear"`` (default; trilinear for 5-D
        input), ``"nearest"``, or ``"bicubic"`` (4-D input only).
    padding_mode : str, optional
        Behaviour for grid locations outside :math:`[-1, 1]`:
        ``"zeros"`` (default), ``"border"`` (clamp to edge), or
        ``"reflection"`` (mirror at the image's edges, then clamp).
    align_corners : bool, optional
        Controls whether ``±1`` denotes the centre of the corner pixels
        (``True``) or the outer edge of the image (``False``, default).

    Returns
    -------
    Tensor
        Sampled output of shape ``(N, C, H_out, W_out)`` (or 5-D
        analogue), with the same dtype as ``input``.

    Notes
    -----
    ``grid_sample`` is differentiable w.r.t. both ``input`` and ``grid``.
    The gradient w.r.t. ``grid`` is what makes STN-style learned warps
    trainable end-to-end (Jaderberg et al., 2015).  Under
    ``"nearest"`` the grid's gradient is zero: the sample does not move
    with the grid.

    4-D bilinear and nearest sampling run on one engine kernel;
    reflection padding mirrors the grid in pixel space first and then
    samples with border padding.  Bicubic and 5-D sampling are composed
    from ``gather`` and elementwise ops, so they cost more memory per
    output point than the kernel.  Coordinates far outside the image, and
    NaN or infinite ones, land where the reference framework's do; the
    few 5-D cases where its forward and backward disagree with each other
    are listed in ``lucid/nn/functional/_grid_sampling.py``.

    Raises
    ------
    ValueError
        Unknown ``mode`` / ``padding_mode``, an input that is not 4-D or
        5-D, ``"bicubic"`` with a 5-D input, or a grid whose shape does
        not fit the input.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import affine_grid, grid_sample
    >>> x = lucid.randn(1, 3, 32, 32)
    >>> theta = lucid.eye(2, 3).unsqueeze(0)        # identity
    >>> g = affine_grid(theta, (1, 3, 32, 32), align_corners=False)
    >>> y = grid_sample(x, g, align_corners=False)
    >>> y.shape
    (1, 3, 32, 32)
    """
    _check_options(input, mode, padding_mode)
    ac = align_corners if align_corners is not None else False
    # The row applies once, here, for every path: the reflection that runs
    # before the kernel computed its coordinates at the grid's half
    # precision, and the composite applied no row at all.  An integer
    # operand is promoted first, as the kernel's schema does before its row:
    # the row skips integers, and a float32 one beside a half one would
    # otherwise meet the body with autocast already off.
    operands = [_unwrap(_as_real(t)) for t in (input, grid)]
    with _autocast_entry("grid_sample", *operands) as (xi, gi):
        return _grid_sample(_wrap(xi), _wrap(gi), mode, padding_mode, ac)


def _grid_sample(
    x: Tensor, grid: Tensor, mode: str, padding_mode: str, align_corners: bool
) -> Tensor:
    """:func:`grid_sample` on operands its row has cast, with autocast off."""
    if x.ndim == 5 or mode == "bicubic":
        return _composite_grid_sample(x, grid, mode, padding_mode, align_corners)
    if padding_mode == "reflection":
        # The kernel has no reflection: the grid is reflected in pixel
        # space first and then read with border padding — for a half pair,
        # all of it in float32, rounded once, as the composites sample.
        dtype = x.dtype
        x, grid = _opmath(x, grid)
        grid = _reflect_grid(x, grid, align_corners, mode == "nearest")
        return _grid_sample(x, grid, mode, "border", align_corners).to(dtype)
    return _wrap(
        _C_engine.nn.grid_sample(
            _unwrap(x),
            _unwrap(grid),
            _KERNEL_MODES[mode],
            _KERNEL_PADDINGS[padding_mode],
            align_corners,
        )
    )


def affine_grid(
    theta: Tensor, size: list[int] | tuple[int, ...], align_corners: bool | None = None
) -> Tensor:
    r"""Generate a sampling grid from a batch of affine transform matrices.

    Builds the flow field needed to apply an affine transform via
    :func:`grid_sample`.  Together they form a Spatial Transformer
    Network (STN, Jaderberg et al. 2015), where ``theta`` is typically
    the output of a localisation network.

    For each output pixel :math:`(x_{\mathrm{out}}, y_{\mathrm{out}})`
    the corresponding source coordinate is computed as:

    .. math::

        \begin{pmatrix} x_{\mathrm{src}} \\ y_{\mathrm{src}} \end{pmatrix}
        = \theta \,
          \begin{pmatrix} x_{\mathrm{out}} \\ y_{\mathrm{out}} \\ 1 \end{pmatrix}

    Parameters
    ----------
    theta : Tensor
        Batch of affine matrices of shape ``(N, 2, 3)`` (2-D) or
        ``(N, 3, 4)`` (3-D).  Each row encodes one element of the affine
        transform applied to homogeneous output coordinates.
    size : sequence of int
        Target output shape passed to ``grid_sample`` afterwards —
        ``(N, C, H, W)`` for 2-D, ``(N, C, D, H, W)`` for 3-D.
    align_corners : bool, optional
        Must match the ``align_corners`` argument later passed to
        :func:`grid_sample`.  Controls whether ``-1``/``+1`` refer to
        the centre of the corner pixels (``True``) or to the outer edge
        (``False``, default).

    Returns
    -------
    Tensor
        Sampling grid of shape ``(N, H, W, 2)`` (or
        ``(N, D, H, W, 3)`` for 3-D) in normalised :math:`[-1, 1]`
        coordinates, ready to be fed to :func:`grid_sample`.

    Raises
    ------
    ValueError
        ``size`` has neither 4 nor 5 entries, or ``theta`` is not
        ``(N, 2, 3)`` / ``(N, 3, 4)`` to match it.

    Notes
    -----
    ``affine_grid`` does not look at the source image — only at the
    requested output size — so the resulting grid is reusable across any
    input of matching spatial dims and any channel count.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import affine_grid
    >>> theta = lucid.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # identity
    >>> g = affine_grid(theta, (1, 3, 4, 4), align_corners=False)
    >>> g.shape
    (1, 4, 4, 2)
    """
    ac = align_corners if align_corners is not None else False
    sz = [int(s) for s in size]
    if len(sz) not in (4, 5):
        raise _C_engine.InvalidArgument(
            "affine_grid: size must be (N, C, H, W) or (N, C, D, H, W), " f"got {tuple(sz)}"
        )
    if min(sz) <= 0:
        raise _C_engine.InvalidArgument(
            f"affine_grid: every entry of size must be positive, got {tuple(sz)}"
        )
    if len(sz) == 5:
        return _affine_grid_3d(theta, sz, ac)
    # Engine signature: (theta, N, H, W, align_corners) — no C dimension.
    N, H, W = sz[0], sz[2], sz[3]
    return _wrap(_C_engine.nn.affine_grid(_unwrap(theta), N, H, W, ac))


def unfold(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] = 1,
) -> Tensor:
    r"""Extract sliding local blocks (im2col) from a 4-D or unbatched 3-D tensor.

    Slides a ``(kH, kW)`` window across the spatial extent of ``input`` (with
    given stride, padding, and dilation) and flattens each window into a
    column of the output.  This is the classical *im2col* operation —
    convolution can be written as ``unfold`` + matmul + ``fold``.

    The number of output locations is

    .. math::

        L = \prod_d
            \left\lfloor
                \frac{S_{\mathrm{in},d} + 2 p_d - d_d (k_d - 1) - 1}{s_d}
            \right\rfloor + 1

    where :math:`d` indexes spatial dims and :math:`s, p, k, d` are
    stride / padding / kernel / dilation respectively.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, H, W)``, or ``(C, H, W)`` unbatched.
    kernel_size : int or (int, int)
        Spatial size of the sliding window.
    dilation : int or (int, int), optional
        Spacing between elements within a window.  Default ``1``.
    padding : int or (int, int), optional
        Implicit zero padding applied symmetrically on both sides.
    stride : int or (int, int), optional
        Step between successive windows.  Default ``1``.

    Returns
    -------
    Tensor
        Tensor of shape ``(N, C·kH·kW, L)``, or ``(C·kH·kW, L)`` for an
        unbatched input.

    Notes
    -----
    The inverse, :func:`fold`, sums overlapping blocks back into an
    image.  ``fold(unfold(x)) == x`` only when stride equals kernel
    size and padding is zero — otherwise overlapping windows produce
    higher counts at interior pixels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import unfold
    >>> x = lucid.randn(1, 3, 8, 8)
    >>> u = unfold(x, kernel_size=3, stride=1, padding=1)
    >>> u.shape                        # (N, C·kH·kW, L) = (1, 27, 64)
    (1, 27, 64)
    """

    xb, unbatched = _add_batch_axis(input, 2)
    kernel, dilations, pads, strides = (
        list(_ntuple(v, 2, "unfold", arg))
        for v, arg in (
            (kernel_size, "kernel_size"),
            (dilation, "dilation"),
            (padding, "padding"),
            (stride, "stride"),
        )
    )
    out = _C_engine.nn.unfold(_unwrap(xb), kernel, strides, pads, dilations)
    return _drop_batch_axis(_wrap(out), unbatched)


def fold(
    input: Tensor,
    output_size: tuple[int, int],
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] = 1,
) -> Tensor:
    r"""Combine an array of sliding local blocks back into an image (col2im).

    Inverse of :func:`unfold`.  Given an ``(N, C·kH·kW, L)`` tensor of
    column-vectorised blocks, scatter-adds each block into its place in
    a fresh ``(N, C, outH, outW)`` canvas.  Overlapping positions are
    summed — which is precisely what is needed for the gradient of
    convolution and for transposed-convolution-style upsampling.

    Parameters
    ----------
    input : Tensor
        Block tensor of shape ``(N, C·kH·kW, L)``, or ``(C·kH·kW, L)``
        unbatched.
    output_size : (int, int)
        Spatial size ``(outH, outW)`` of the destination canvas.
    kernel_size : int or (int, int)
        Spatial size of each block.  Must match the ``kernel_size`` used
        to produce ``input``.
    dilation : int or (int, int), optional
        Spacing between elements within a block.
    padding : int or (int, int), optional
        Implicit zero padding to subtract from the destination canvas
        (mirrors the ``padding`` argument of :func:`unfold`).
    stride : int or (int, int), optional
        Step between block positions on the destination canvas.

    Returns
    -------
    Tensor
        Reconstructed tensor of shape ``(N, C, outH, outW)``, or
        ``(C, outH, outW)`` for an unbatched input.

    Notes
    -----
    CPU path uses a scatter-add loop; the GPU path emits a single
    ``scatter_add_axis`` over precomputed flat destination indices, with
    no host round-trip.  In conjunction with :func:`unfold`,
    ``fold`` lets you implement arbitrary local linear operators as
    plain matrix multiplications.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import unfold, fold
    >>> x = lucid.randn(1, 3, 8, 8)
    >>> u = unfold(x, kernel_size=3, stride=3)
    >>> y = fold(u, output_size=(8, 8), kernel_size=3, stride=3)
    >>> y.shape
    (1, 3, 8, 8)
    """

    # The unbatched block tensor is ``(C·kH·kW, L)``: one axis short of
    # the batched rank, as a 1-spatial-axis input is.
    xb, unbatched = _add_batch_axis(input, 1)
    size, kernel, dilations, pads, strides = (
        list(_ntuple(v, 2, "fold", arg))
        for v, arg in (
            (output_size, "output_size"),
            (kernel_size, "kernel_size"),
            (dilation, "dilation"),
            (padding, "padding"),
            (stride, "stride"),
        )
    )
    out = _C_engine.nn.fold(_unwrap(xb), size, kernel, strides, pads, dilations)
    return _drop_batch_axis(_wrap(out), unbatched)


def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Tensor | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Tensor | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> Tensor:
    r"""Aggregate embeddings into per-bag pooled vectors.

    Conceptually equivalent to looking up each index with
    :func:`embedding` and then reducing across the bag axis, but fused
    into a single op that avoids materialising the per-token embedding
    matrix — essential for very large vocabularies in recommendation
    and NLP models.

    Given a flat index sequence ``input`` partitioned into bags by
    ``offsets``, the ``i``-th output row is

    .. math::

        \mathrm{out}[i] = \mathrm{reduce}_{j \in \mathrm{bag}_i} W[\, x[j]\,]

    where ``reduce`` is one of ``sum``, ``mean``, or ``max``.

    Parameters
    ----------
    input : Tensor
        Either a 1-D index tensor (use with ``offsets``) or a 2-D
        ``(num_bags, seq_len)`` tensor of indices where each row is a
        bag of equal length.
    weight : Tensor
        Embedding table of shape ``(num_embeddings, embedding_dim)``.
    offsets : Tensor, optional
        Required when ``input`` is 1-D.  Integer tensor whose ``i``-th
        element is the starting index of bag ``i`` within ``input``.  It must
        start at 0 and never decrease (else ``ValueError``), and each
        entry must lie in ``[0, len(input)]`` (else ``IndexError``).
    max_norm : float, optional
        Rescale, in place and outside autograd, every row ``input`` looks up
        whose :math:`L_p` norm exceeds ``max_norm`` to that norm before the
        lookup, as :func:`~lucid.nn.functional.embedding` does.
    norm_type : float, optional
        :math:`p` exponent for ``max_norm``.  Default ``2.0``.
    scale_grad_by_freq : bool, optional
        Divide the gradient reaching each row by how many times its index
        occurs in ``input`` (all bags together).  ``mode="max"`` raises
        ``ValueError``.
    mode : str, optional
        Bag reduction: ``"sum"``, ``"mean"`` (default), or ``"max"``.
    sparse : bool, optional
        Request a sparse gradient.  Gradients are always dense, so ``True``
        raises ``NotImplementedError``.
    per_sample_weights : Tensor, optional
        One weight per index, the shape of ``input``; each looked-up row is
        scaled by its weight before the bag is summed.  Only
        ``mode="sum"`` takes weights — any other mode raises
        ``NotImplementedError``, as the reference does.
    include_last_offset : bool, optional
        If ``True``, ``offsets`` has length ``num_bags + 1`` and its
        last entry is the total number of indices in ``input``.
    padding_idx : int, optional
        Embedding row to mask out (its lookup result contributes zero), in
        ``[-num_embeddings, num_embeddings)``, else ``ValueError``.

    Returns
    -------
    Tensor
        Pooled output of shape ``(num_bags, embedding_dim)``.

    Notes
    -----
    Compared with ``embedding`` + manual reduction, ``embedding_bag``
    saves a full materialisation of the per-token table and fuses the
    reduction into a single scatter-add (or scatter-max) pass.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import embedding_bag
    >>> w = lucid.randn(10, 4)
    >>> ids = lucid.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=lucid.int64)
    >>> off = lucid.tensor([0, 4], dtype=lucid.int64)
    >>> out = embedding_bag(ids, w, offsets=off, mode="mean")
    >>> out.shape
    (2, 4)
    """
    mode_int = _bag_mode(mode, sparse, scale_grad_by_freq)

    # Same index contract as ``embedding``, checked before the gather:
    # the indices, ``padding_idx``, and the bag boundaries in ``offsets``.
    op = "embedding_bag"
    padding_idx = _padding_index(padding_idx, weight, op)
    pad_idx = -1 if padding_idx is None else padding_idx
    _check_table(input, weight, op)
    if input.ndim == 1:
        if offsets is None:
            raise _C_engine.InvalidArgument("embedding_bag: offsets required for 1-D input")
        _check_offsets(offsets, int(input.shape[0]), op)
    if max_norm is not None:
        _renorm_rows(weight, input, max_norm, norm_type)

    # The engine op pools rows it looked up itself, so neither a weight per
    # index nor the lookup's own gradient scaling can reach it.
    scaled = scale_grad_by_freq and weight.requires_grad and _C_engine.grad_enabled()
    if per_sample_weights is not None or scaled:
        return _bag_by_lookup(
            input,
            weight,
            _bag_of_each_index(input, offsets, include_last_offset),
            per_sample_weights,
            mode,
            padding_idx,
            scale_grad_by_freq,
        )

    x_impl = _unwrap(input)
    w_impl = _unwrap(weight)
    # The table holds embeddings, which are real numbers — pooling them
    # sums or averages, and the CPU kernel is written for floats only.
    # Metal took an integer table and the CPU refused one, so the same
    # call worked on one device and not the other.  Promote, as the
    # engine's real-valued ops do; the indices and offsets beside it stay
    # integral, because they are positions rather than values.
    if w_impl.dtype in (
        _C_engine.Dtype.Bool,
        _C_engine.Dtype.I8,
        _C_engine.Dtype.I16,
        _C_engine.Dtype.I32,
        _C_engine.Dtype.I64,
    ):
        w_impl = _C_engine.astype(w_impl, _C_engine.Dtype.F32)
    x_ndim = len(x_impl.shape)

    if x_ndim == 2:
        # 2-D: each row is a bag — build synthetic 1-D x and offsets
        n_rows, seq_len = int(x_impl.shape[0]), int(x_impl.shape[1])
        # Flatten indices to 1-D
        flat_x = _C_engine.reshape(x_impl, [n_rows * seq_len])
        # One bag per row → offsets [0, seq_len, 2*seq_len, …].
        off_impl = _C_engine.arange(0, n_rows * seq_len, seq_len, _C_engine.I32, x_impl.device)
        impl = _C_engine.nn.embedding_bag(w_impl, flat_x, off_impl, mode_int, pad_idx, False)
    else:
        # 1-D x with explicit offsets, checked above.
        assert offsets is not None
        off_impl = _unwrap(offsets)
        if off_impl.dtype != _C_engine.I32:
            off_impl = _C_engine.astype(off_impl, _C_engine.I32)
        impl = _C_engine.nn.embedding_bag(
            w_impl, x_impl, off_impl, mode_int, pad_idx, include_last_offset
        )

    return _wrap(impl)


_PAD_MODES: frozenset[str] = frozenset({"constant", "reflect", "replicate", "circular"})


def _flat_to_per_dim_pairs(padding: Sequence[int], ndim: int) -> list[tuple[int, int]]:
    """Per-dim (first→last) pairs from a flat (last→first) padding spec.

    The spec is checked as the reference checks it: an odd one dropped its
    last entry, and one longer than two per dimension wrapped round onto
    the leading dimensions, padding them instead of being refused.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        An odd-length spec, or one longer than ``2 * ndim``.
    """
    n = len(padding)
    if n % 2:
        raise _C_engine.InvalidArgument(f"pad: padding length must be divisible by 2, got {n}")
    if n > 2 * ndim:
        raise _C_engine.InvalidArgument(
            f"pad: padding length should be at most two times the input "
            f"dimension, got padding length {n} for a {ndim}-D input"
        )
    pad_pairs: list[tuple[int, int]] = [(0, 0)] * ndim
    for i in range(n // 2):
        pad_pairs[ndim - 1 - i] = (int(padding[2 * i]), int(padding[2 * i + 1]))
    return pad_pairs


def _crop(x: Tensor, dim: int, lo: int, hi: int) -> Tensor:
    """``x`` without the ``-lo`` / ``-hi`` entries a negative pad removes.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The negative pads remove more than ``dim`` holds.
    """
    size = int(x.shape[dim])
    start, stop = max(-lo, 0), size - max(-hi, 0)
    if stop < start:
        raise _C_engine.InvalidArgument(
            f"pad: the input size {size} at dimension {dim}, plus negative "
            f"padding {lo} and {hi}, gives a negative output size"
        )
    if start == 0 and stop == size:
        return x
    return x.narrow(dim, start, stop - start)


def _gather_along(x: Tensor, dim: int, indices_1d: list[int]) -> Tensor:
    """Gather x along `dim` with a 1-D index list, broadcasting to x.shape.

    ``indices_1d`` is a Python list of ints of length ``k``.  Returns a
    tensor of the same shape as ``x`` except ``dim`` has size ``k``.  Uses
    ``lucid.gather`` so backward accumulates gradients into the correct
    source positions — this is what makes reflect/circular padding
    correctly differentiable.
    """
    k: int = len(indices_1d)
    target_shape: list[int] = [1] * x.ndim
    target_shape[dim] = k
    bcast_shape: list[int] = [k if i == dim else int(x.shape[i]) for i in range(x.ndim)]
    idx_flat: Tensor = _lucid.tensor(indices_1d, dtype=_lucid.int32, device=x.device).reshape(
        target_shape
    )
    idx: Tensor = idx_flat.broadcast_to(bcast_shape).contiguous()
    return _lucid.gather(x, dim, idx)


def _edge_indices(mode: str, size: int, lo: int, hi: int) -> tuple[list[int], list[int]]:
    """The source indices of the ``lo`` entries before and ``hi`` after an axis.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        A reflection as wide as the axis, a wrap round it more than once, or
        an edge to replicate on an empty axis — the reference's refusals.
    """
    if mode == "reflect":
        if max(lo, hi) >= size:
            raise _C_engine.InvalidArgument(
                f"pad: mode='reflect' needs padding less than the input size "
                f"{size}, got ({lo}, {hi})"
            )
        # The boundary itself is not repeated: [lo, ..., 1] and [size-2, ...].
        return list(range(lo, 0, -1)), list(range(size - 2, size - 2 - hi, -1))
    if mode == "circular":
        if max(lo, hi) > size:
            raise _C_engine.InvalidArgument(
                f"pad: mode='circular' padding ({lo}, {hi}) wraps around an "
                f"input of size {size} more than once"
            )
        return list(range(size - lo, size)), list(range(hi))
    if size == 0 and (lo or hi):
        raise _C_engine.InvalidArgument(
            "pad: mode='replicate' has no edge to repeat on an empty dimension"
        )
    return [0] * lo, [size - 1] * hi


def _pad_one_dim(x: Tensor, dim: int, lo: int, hi: int, mode: str) -> Tensor:
    """Apply non-constant padding along a single dimension.

    `mode` must be one of {"reflect", "replicate", "circular"}.  Constant
    padding is handled separately by the engine pad op.

    Implementation: build the lo-side and hi-side slabs via `gather` with
    explicit index lists, then `cat` everything along the target dim.
    Gather is differentiable via scatter-add, which gives the correct
    gradient accumulation for reflect (where the same source element
    contributes to both the centre and the reflected copy).

    A negative pad removes entries, where the reference removes them: a
    circular pad wraps what is left, a reflected or replicated one reads
    the whole axis and is cropped after.
    """
    if lo == 0 and hi == 0:
        return x
    if mode == "circular":
        x = _crop(x, dim, lo, hi)
    before, after = _edge_indices(mode, int(x.shape[dim]), max(lo, 0), max(hi, 0))
    parts = [x]
    if before:
        parts.insert(0, _gather_along(x, dim, before))
    if after:
        parts.append(_gather_along(x, dim, after))
    out = parts[0] if len(parts) == 1 else _lucid.cat(parts, dim)
    return out if mode == "circular" else _crop(out, dim, lo, hi)


def pad(
    input: Tensor, pad: Sequence[int], mode: str = "constant", value: float | None = None
) -> Tensor:
    r"""Pad an N-D tensor along an arbitrary set of trailing dimensions.

    Generalises ``np.pad`` with several boundary modes useful in deep
    learning: zero/constant fill, mirror reflection, edge replication,
    and circular wrap-around.  This is Lucid's one ``pad``: there is no
    ``lucid.pad`` or ``Tensor.pad``.

    Parameters
    ----------
    input : Tensor
        Input tensor of any rank ``N``.
    pad : Sequence of int
        Flat padding spec **starting from the last dimension**, in
        ``(left, right)`` pairs.  For example:

        - ``(l, r)`` pads only the last dim,
        - ``(l, r, t, b)`` pads the last two dims (last with ``(l, r)``,
          second-to-last with ``(t, b)``),
        - ``(l, r, t, b, f, k)`` extends the pattern to three dims.

        Length must be even and at most ``2·N``.  A negative entry removes
        that many entries from its side instead.
    mode : str, optional
        One of:

        - ``"constant"`` (default): fill with ``value``.
        - ``"reflect"``: mirror around the boundary, excluding the
          boundary itself (so size-1 along a padded dim is illegal).
        - ``"replicate"``: repeat the edge element.
        - ``"circular"``: wrap around (toroidal boundary conditions).
    value : float, optional
        Fill value used only when ``mode == "constant"``.

    Returns
    -------
    Tensor
        Padded tensor with each padded dim ``d`` enlarged by
        ``pad[2·k] + pad[2·k+1]`` where ``k`` is its offset from
        the last dim.

    Raises
    ------
    lucid._C.engine.NotImplementedError
        An unknown ``mode``.
    lucid._C.engine.InvalidArgument
        A ``pad`` of odd length or longer than ``2·N``, a reflection
        as wide as its dim, a circular pad wider than it, a replicated
        empty dim, or negative pads that remove more than a dim holds.
        Both a ``RuntimeError``, as the reference raises, and a
        ``ValueError``.

    Notes
    -----
    Non-constant modes are decomposed into gather + concat in Python,
    relying on existing autograd machinery (gather's scatter-add
    backward correctly accumulates gradients for reflected /
    replicated entries).  ``"reflect"`` requires
    ``pad_amount <= size - 1`` on each side; ``"circular"`` requires
    ``pad_amount <= size``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pad
    >>> x = lucid.arange(6, dtype=lucid.float32).reshape(1, 1, 2, 3)
    >>> y = pad(x, (1, 1, 0, 0), mode="reflect")
    >>> y.shape                    # last dim grows by 2
    (1, 1, 2, 5)
    >>> pad(lucid.arange(4.0), (-1, 2)).tolist()
    [1.0, 2.0, 3.0, 0.0, 0.0]
    """
    if mode not in _PAD_MODES:
        raise _C_engine.NotImplementedError(
            f"pad: unrecognised padding mode {mode!r}; expected one of {sorted(_PAD_MODES)}"
        )
    pad_pairs = _flat_to_per_dim_pairs(pad, input.ndim)
    if mode != "constant":
        if value is not None and value != 0:
            raise _C_engine.InvalidArgument(
                f'Padding mode "{mode}" doesn\'t take in value argument'
            )
        result = input
        for d, (lo, hi) in enumerate(pad_pairs):
            result = _pad_one_dim(result, d, lo, hi, mode)
        return result if mode == "circular" else _keep_in_graph(result, input)
    # The engine pads by non-negative widths: a negative one is a crop.
    x = input
    for d, (lo, hi) in enumerate(pad_pairs):
        x = _crop(x, d, lo, hi)
    widths = [(max(lo, 0), max(hi, 0)) for lo, hi in pad_pairs]
    fill = 0.0 if value is None else float(value)
    return _wrap(_C_engine.pad(_unwrap(x), widths, fill))


def pixel_shuffle(input: Tensor, upscale_factor: int) -> Tensor:
    r"""Sub-pixel upsampling: rearrange channels into spatial resolution.

    Reshapes a low-resolution multi-channel feature map into a
    higher-resolution map with fewer channels, *without* introducing any
    interpolation.  Originally proposed by Shi et al., 2016 ("Real-Time
    Single Image and Video Super-Resolution"), this is the canonical
    decoder layer in single-image super-resolution and many GAN /
    diffusion upsamplers.

    Concretely, with upscale factor :math:`r`:

    .. math::

        (N, C \cdot r^2, H, W) \;\longrightarrow\; (N, C, H \cdot r, W \cdot r)

    Each :math:`r \times r` patch in the output is taken from a
    block of :math:`r^2` channels in the input.

    Parameters
    ----------
    input : Tensor
        4-D input of shape ``(N, C·r², H, W)``.  The channel dimension
        must be divisible by ``upscale_factor²``.
    upscale_factor : int
        Spatial upscaling factor :math:`r`.

    Returns
    -------
    Tensor
        Upsampled tensor of shape ``(N, C, H·r, W·r)``.

    Notes
    -----
    The transformation is a pure reshape + permute + reshape — autograd
    flows through ``ReshapeBackward`` and ``PermuteBackward`` with no
    custom gradient.  Because no convolution is involved, the operator
    is essentially free; it is most often preceded by a convolution
    that produces the required ``C·r²`` channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pixel_shuffle
    >>> x = lucid.randn(1, 12, 8, 8)      # C=3, r=2 → C·r²=12
    >>> y = pixel_shuffle(x, upscale_factor=2)
    >>> y.shape
    (1, 3, 16, 16)
    """
    r: int = int(upscale_factor)
    if r <= 0:
        raise _C_engine.InvalidArgument(
            f"pixel_shuffle: upscale_factor must be a positive integer, got {r}"
        )
    if len(input.shape) != 4:
        raise _C_engine.InvalidArgument(
            f"pixel_shuffle: expected 4-D input, got shape {tuple(input.shape)}"
        )
    n, c_r2, h, w = input.shape
    if c_r2 % (r * r) != 0:
        raise _C_engine.InvalidArgument(
            f"pixel_shuffle: channels {c_r2} not divisible by upscale_factor² ({r * r})"
        )
    c: int = c_r2 // (r * r)
    impl = _unwrap(input)
    t = _C_engine.reshape(impl, [n, c, r, r, h, w])
    t = _C_engine.permute(t, [0, 1, 4, 2, 5, 3])
    return _wrap(_C_engine.reshape(t, [n, c, h * r, w * r]))


def pixel_unshuffle(input: Tensor, downscale_factor: int) -> Tensor:
    r"""Inverse of :func:`pixel_shuffle`: pack spatial blocks into channels.

    Folds each :math:`r \times r` spatial block into ``r²`` extra
    channels, shrinking the spatial extent by :math:`r` along each axis:

    .. math::

        (N, C, H \cdot r, W \cdot r) \;\longrightarrow\; (N, C \cdot r^2, H, W)

    Useful as a strided "downsampling without information loss" stage
    in invertible architectures (e.g. RealNVP, normalising flows) and
    as the analyzer step in sub-pixel encoders.

    Parameters
    ----------
    input : Tensor
        4-D input of shape ``(N, C, H·r, W·r)``.  Both spatial dims
        must be divisible by ``downscale_factor``.
    downscale_factor : int
        Spatial downscaling factor :math:`r`.

    Returns
    -------
    Tensor
        Downsampled tensor of shape ``(N, C·r², H, W)``.

    Notes
    -----
    Like :func:`pixel_shuffle`, this is a reshape + permute + reshape
    chain — completely lossless and free of multiply-add cost.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pixel_unshuffle
    >>> x = lucid.randn(1, 3, 16, 16)
    >>> y = pixel_unshuffle(x, downscale_factor=2)
    >>> y.shape
    (1, 12, 8, 8)
    """
    r: int = int(downscale_factor)
    if r <= 0:
        raise _C_engine.InvalidArgument(
            f"pixel_unshuffle: downscale_factor must be a positive integer, got {r}"
        )
    if len(input.shape) != 4:
        raise _C_engine.InvalidArgument(
            f"pixel_unshuffle: expected 4-D input, got shape {tuple(input.shape)}"
        )
    n, c, h_r, w_r = input.shape
    if h_r % r != 0 or w_r % r != 0:
        raise _C_engine.InvalidArgument(
            f"pixel_unshuffle: spatial dims ({h_r}, {w_r}) not divisible by {r}"
        )
    h, w = h_r // r, w_r // r
    impl = _unwrap(input)
    t = _C_engine.reshape(impl, [n, c, h, r, w, r])
    t = _C_engine.permute(t, [0, 1, 3, 5, 2, 4])
    return _wrap(_C_engine.reshape(t, [n, c * r * r, h, w]))


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor | None = None,
    in_proj_bias: Tensor | None = None,
    bias_k: Tensor | None = None,
    bias_v: Tensor | None = None,
    add_zero_attn: bool = False,
    dropout_p: float = 0.0,
    out_proj_weight: Tensor | None = None,
    out_proj_bias: Tensor | None = None,
    training: bool = True,
    key_padding_mask: Tensor | None = None,
    need_weights: bool = True,
    attn_mask: Tensor | None = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Tensor | None = None,
    k_proj_weight: Tensor | None = None,
    v_proj_weight: Tensor | None = None,
    static_k: Tensor | None = None,
    static_v: Tensor | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> tuple[Tensor, Tensor | None]:
    r"""Stateless functional multi-head attention forward pass.

    Performs the full Vaswani et al. (2017) multi-head attention
    computation as a single pure function — useful for porting code
    that builds attention layers without instantiating a stateful
    module.

    The computation has four stages:

    1. **Input projection.**  Linear projections produce
       :math:`Q = X_q W_Q`, :math:`K = X_k W_K`, :math:`V = X_v W_V`
       (when ``use_separate_proj_weight=False``, these come from a
       single fused ``in_proj_weight``).
    2. **Head split.**  Reshape each of ``Q``, ``K``, ``V`` from
       ``(L, B, d_model)`` to ``(L, B·H, d_model/H)`` so each of the
       :math:`H` heads attends independently.
    3. **Scaled dot-product attention.**

       .. math::

           \mathrm{Attn} = \mathrm{softmax}\!\left(
               \frac{Q K^\top}{\sqrt{d_k}} + M\right) V

       where ``M`` is the union of the optional ``attn_mask``,
       ``key_padding_mask``, and the implicit causal mask when
       ``is_causal=True``.

    4. **Output projection.**  Concatenate heads and apply
       :math:`\mathrm{out} = \mathrm{Attn}_\text{concat} W_O`.

    Parameters
    ----------
    query, key, value : Tensor
        Inputs of shape ``(L, N, E)`` (target / source / source length
        × batch × ``embed_dim``).  Self-attention uses the same tensor
        for all three; cross-attention takes a separate ``query``.
    embed_dim_to_check : int
        Must equal ``query.shape[-1]``; sanity check.
    num_heads : int
        Number of attention heads :math:`H`.  Must divide ``embed_dim``.
    in_proj_weight, in_proj_bias : Tensor, optional
        Fused QKV projection.  ``in_proj_weight`` has shape
        ``(3·embed_dim, embed_dim)``.
    bias_k, bias_v : Tensor, optional
        Optional learned bias vectors appended to ``K`` / ``V`` along
        the sequence axis.
    add_zero_attn : bool, optional
        If ``True``, append a row of zeros to ``K`` / ``V`` after any
        ``bias_k`` / ``bias_v`` row, widening the masks to match.
    dropout_p : float, optional
        Dropout probability applied to attention weights during
        training.
    out_proj_weight, out_proj_bias : Tensor, optional
        Output linear projection :math:`W_O`.
    training : bool, optional
        If ``True``, dropout is active.
    key_padding_mask : Tensor, optional
        Boolean mask of shape ``(N, S)`` with ``True`` at padded
        positions to be ignored.
    need_weights : bool, optional
        If ``True``, also return the attention weight matrix.
    attn_mask : Tensor, optional
        Additional additive mask, e.g. a causal triangular mask.
    use_separate_proj_weight : bool, optional
        Use ``q_proj_weight`` / ``k_proj_weight`` / ``v_proj_weight``
        instead of a fused ``in_proj_weight`` — how keys and values of
        their own widths (``kdim`` / ``vdim``) are projected.
    q_proj_weight, k_proj_weight, v_proj_weight : Tensor, optional
        Separate Q / K / V projection weights, ``(E, E)``, ``(E, kdim)``
        and ``(E, vdim)``; all three are required with
        ``use_separate_proj_weight``.
    static_k, static_v : Tensor, optional
        Keys and values already projected and split per head.  Not
        supported: passing either raises ``NotImplementedError``.
    average_attn_weights : bool, optional
        If ``True``, the returned weight tensor is averaged across
        heads; otherwise per-head weights are returned.
    is_causal : bool, optional
        Apply an upper-triangular causal mask.  Mutually exclusive
        with a user-supplied ``attn_mask``.

    Returns
    -------
    (Tensor, Tensor or None)
        - The attention output of shape ``(L, N, E)``.
        - The attention weights of shape ``(N, L, S)`` (or
          ``(N, H, L, S)`` if ``average_attn_weights=False``),
          or ``None`` if ``need_weights=False``.

    Notes
    -----
    This functional is implemented by binding the supplied weight
    tensors onto a transient ``MultiheadAttention`` module so the
    forward behaviour is bit-identical to the module path.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import multi_head_attention_forward
    >>> L, N, E, H = 16, 2, 64, 8
    >>> q = k = v = lucid.randn(L, N, E)
    >>> Wi = lucid.randn(3 * E, E)
    >>> Wo = lucid.randn(E, E)
    >>> out, attn = multi_head_attention_forward(
    ...     q, k, v, embed_dim_to_check=E, num_heads=H,
    ...     in_proj_weight=Wi, out_proj_weight=Wo,
    ...     need_weights=False,
    ... )
    >>> out.shape
    (16, 2, 64)
    """
    from lucid.nn.modules.attention import MultiheadAttention

    # Keys and values already projected and split per head replace the K/V
    # projection inside the attention itself; the module has no seam for it.
    if static_k is not None or static_v is not None:
        raise NotImplementedError(
            "multi_head_attention_forward: static_k/static_v (keys and values "
            "already projected and split per head) are not supported"
        )

    # Separate Q/K/V weights are how keys and values of their own widths
    # reach the attention; the module takes those widths as kdim / vdim.
    kdim = vdim = int(embed_dim_to_check)
    if use_separate_proj_weight:
        if q_proj_weight is None or k_proj_weight is None or v_proj_weight is None:
            raise _C_engine.InvalidArgument(
                "multi_head_attention_forward: use_separate_proj_weight=True "
                "needs q_proj_weight, k_proj_weight and v_proj_weight"
            )
        kdim = int(k_proj_weight.shape[-1])
        vdim = int(v_proj_weight.shape[-1])

    # Build a temporary module on ``query``'s device and dtype so any unbound
    # fallback parameters match the caller's inputs without a round-trip
    # ``mha.to(...)`` copy.
    mha: MultiheadAttention = MultiheadAttention(
        embed_dim=int(embed_dim_to_check),
        num_heads=int(num_heads),
        dropout=float(dropout_p),
        bias=in_proj_bias is not None or out_proj_bias is not None,
        add_bias_kv=bias_k is not None,
        add_zero_attn=add_zero_attn,
        kdim=kdim,
        vdim=vdim,
        batch_first=False,
        device=query.device,
        dtype=query.dtype,
    )
    if use_separate_proj_weight:
        assert q_proj_weight is not None
        assert k_proj_weight is not None and v_proj_weight is not None
        if mha.in_proj_weight is not None:
            # Every width equals embed_dim, so the module fused the three
            # projections; the separate weights stack into that one matrix.
            stacked = _lucid.cat([q_proj_weight, k_proj_weight, v_proj_weight], 0)
            mha.in_proj_weight._impl = _unwrap(stacked)
        else:
            assert mha.q_proj_weight is not None
            assert mha.k_proj_weight is not None and mha.v_proj_weight is not None
            mha.q_proj_weight._impl = _unwrap(q_proj_weight)
            mha.k_proj_weight._impl = _unwrap(k_proj_weight)
            mha.v_proj_weight._impl = _unwrap(v_proj_weight)
    elif in_proj_weight is not None and mha.in_proj_weight is not None:
        mha.in_proj_weight._impl = _unwrap(in_proj_weight)
    if in_proj_bias is not None and mha.in_proj_bias is not None:
        mha.in_proj_bias._impl = _unwrap(in_proj_bias)
    if out_proj_weight is not None:
        mha.out_proj_weight._impl = _unwrap(out_proj_weight)
    if out_proj_bias is not None and mha.out_proj_bias is not None:
        mha.out_proj_bias._impl = _unwrap(out_proj_bias)
    if bias_k is not None and mha.bias_k is not None:
        mha.bias_k._impl = _unwrap(bias_k)
    if bias_v is not None and mha.bias_v is not None:
        mha.bias_v._impl = _unwrap(bias_v)

    if training:
        mha.train()
    else:
        mha.eval()
    return mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )


# ── P3 fills: channel_shuffle / pdist ──────────────────────────────────────


def channel_shuffle(input: Tensor, groups: int) -> Tensor:
    r"""Group-then-transpose channel rearrangement (ShuffleNet).

    Splits the channel axis into ``groups`` equal-sized groups,
    transposes the group and intra-group axes, and flattens back.
    This re-orders channels so that, after a subsequent grouped
    convolution, information from each input group reaches every
    output group — enabling cheap cross-group information flow that
    grouped convolutions alone cannot provide.

    Conceptually:

    .. math::

        (N, g \cdot c, H, W)
            \xrightarrow{\mathrm{reshape}} (N, g, c, H, W)
            \xrightarrow{\mathrm{transpose}} (N, c, g, H, W)
            \xrightarrow{\mathrm{reshape}} (N, g \cdot c, H, W)

    where ``g = groups`` and ``c = C / groups``.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape ``(N, C, *spatial)`` (at least 2-D).
        ``C`` must be divisible by ``groups``.
    groups : int
        Number of channel groups :math:`g`.

    Returns
    -------
    Tensor
        Channel-shuffled tensor of the same shape as ``input``.

    Notes
    -----
    Introduced in Zhang et al., 2018 — "ShuffleNet: An Extremely
    Efficient Convolutional Neural Network for Mobile Devices".  The
    operation is a pure reshape + permute, so it adds no FLOPs and
    very little memory traffic.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import channel_shuffle
    >>> x = lucid.arange(24, dtype=lucid.float32).reshape(1, 6, 2, 2)
    >>> y = channel_shuffle(x, groups=3)
    >>> y.shape
    (1, 6, 2, 2)
    >>> y[0, :, 0, 0]              # channels 0..5 reordered as 0, 2, 4, 1, 3, 5
    tensor([0., 8., 16., 4., 12., 20.])
    """
    if input.ndim < 2:
        raise _C_engine.InvalidArgument(
            f"channel_shuffle: expected at least 2-D input (N, C, *), got "
            f"shape {tuple(input.shape)}"
        )
    if int(groups) <= 0:
        raise _C_engine.InvalidArgument(
            f"channel_shuffle: groups must be a positive integer, got {groups}"
        )
    n = int(input.shape[0])
    c = int(input.shape[1])
    if c % int(groups) != 0:
        raise _C_engine.InvalidArgument(
            f"channel_shuffle: channel count {c} not divisible by groups " f"{groups}"
        )
    ch_per_group = c // int(groups)
    spatial = list(input.shape[2:])

    # (N, groups, C/groups, *spatial) → (N, C/groups, groups, *spatial) →
    # flatten the two leading channel-related axes back into a single ``C``.
    reshaped = input.reshape(n, int(groups), ch_per_group, *spatial)
    perm = [0, 2, 1] + list(range(3, 3 + len(spatial)))
    transposed = reshaped.permute(*perm)
    return transposed.reshape(n, c, *spatial)


def pdist(input: Tensor, p: float = 2.0) -> Tensor:
    r"""Pairwise :math:`L_p` distances between rows of a 2-D tensor.

    For an input of shape ``(N, M)``, returns the
    :math:`N \cdot (N - 1) / 2` strictly-upper-triangular pairwise
    distances in row-major order:

    .. math::

        \mathrm{out}[k(i, j)] =
            \left( \sum_{m=1}^{M} | x_{i, m} - x_{j, m} |^p \right)^{1/p}
        \quad \text{for } 0 \le i < j < N

    where ``k(i, j)`` enumerates pairs in row-major order.

    Parameters
    ----------
    input : Tensor
        2-D input of shape ``(N, M)``.
    p : float, optional
        Exponent of the :math:`L_p` norm.  Common values: ``2`` for
        Euclidean (default), ``1`` for Manhattan, ``float("inf")`` for
        Chebyshev.

    Returns
    -------
    Tensor
        1-D tensor of length :math:`N (N - 1) / 2` (empty when
        :math:`N < 2`).

    Notes
    -----
    Internally computed by extracting the strict upper triangle of the
    full :func:`lucid.cdist` matrix, which keeps the implementation
    simple and fully differentiable but is :math:`O(N^2)` in both
    compute and intermediate memory — for very large ``N``, prefer a
    blocked / streaming pairwise routine.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pdist
    >>> x = lucid.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> pdist(x, p=2)
    tensor([1., 1., 1.414])
    """
    if input.ndim != 2:
        raise _C_engine.InvalidArgument(
            f"pdist: expected a 2-D input, got shape {tuple(input.shape)}"
        )
    if p < 0:
        raise _C_engine.InvalidArgument(f"pdist: p must be non-negative, got {p}")
    _require_floating(_unwrap(input), "pdist")
    n = int(input.shape[0])
    if n < 2:
        return _lucid.zeros(0, dtype=input.dtype, device=input.device)
    full = _lucid.cdist(input, input, p=p)  # (N, N)
    # Pull out the strict upper triangle (i < j) in row-major order via the
    # index pair generator from ``triu_indices``.
    # ``device=`` matters: a CPU-default index tensor fed to ``gather`` on a
    # Metal tensor raises bad_variant_access.
    pairs = _lucid.triu_indices(n, n, offset=1, device=input.device)
    flat = full.reshape(n * n)
    flat_idx = pairs[0] * n + pairs[1]
    return _lucid.gather(flat, 0, flat_idx)
