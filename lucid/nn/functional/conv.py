"""
nn.functional convolution operations.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional._spatial_args import (
    _add_batch_axis,
    _conv_padding,
    _drop_batch_axis,
    _ntuple,
    _require_non_negative,
    _require_positive,
    _weight_kernel,
)
from lucid.nn.functional.sampling import pad as _pad

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# A convolution's ``padding_mode`` as the mode :func:`pad` calls it.
_PAD_MODE: dict[str, str] = {
    "zeros": "constant",
    "reflect": "reflect",
    "replicate": "replicate",
    "circular": "circular",
}

_CONV: dict[int, Callable[..., _C_engine.TensorImpl]] = {
    1: _C_engine.nn.conv1d,
    2: _C_engine.nn.conv2d,
    3: _C_engine.nn.conv3d,
}
_CONV_TRANSPOSE: dict[int, Callable[..., _C_engine.TensorImpl]] = {
    1: _C_engine.nn.conv_transpose1d,
    2: _C_engine.nn.conv_transpose2d,
    3: _C_engine.nn.conv_transpose3d,
}


class _ConvArgs(NamedTuple):
    """A convolution's geometry, as its caller spelled it."""

    stride: int | Sequence[int]
    padding: str | int | Sequence[int]
    dilation: int | Sequence[int]
    groups: int


def _convnd(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    args: _ConvArgs,
    n: int,
    padding_mode: str = "zeros",
) -> Tensor:
    """The one body of ``conv{1,2,3}d`` and of the ``Conv*d`` modules.

    Reads the arguments through ``_spatial_args``, runs an unbatched
    input as a batch of one, and pre-pads with :func:`pad` whenever the
    engine's symmetric zero padding cannot express the request — an odd
    ``"same"`` total, or a ``padding_mode`` other than ``"zeros"``.
    """
    op = f"conv{n}d"
    s = _ntuple(args.stride, n, op, "stride")
    _require_positive(s, op, "stride")
    d = _ntuple(args.dilation, n, op, "dilation")
    _require_positive(d, op, "dilation")
    kernel = _weight_kernel(weight, n, op)
    before, after = _conv_padding(args.padding, kernel, s, d, op)
    xb, unbatched = _add_batch_axis(x, n)
    if padding_mode != "zeros" or before != after:
        flat: list[int] = []
        for axis in reversed(range(n)):
            flat += [before[axis], after[axis]]
        xb = _pad(xb, tuple(flat), mode=_PAD_MODE[padding_mode])
        before = (0,) * n
    b = _unwrap(bias) if bias is not None else None
    out = _wrap(_CONV[n](_unwrap(xb), _unwrap(weight), b, *s, *before, *d, args.groups))
    return _drop_batch_axis(out, unbatched)


def _conv_transposend(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    args: _ConvArgs,
    output_padding: int | Sequence[int],
    n: int,
) -> Tensor:
    """The one body of ``conv_transpose{1,2,3}d`` and of their modules."""
    op = f"conv_transpose{n}d"
    if isinstance(args.padding, str):
        raise TypeError(f"{op}: padding must be an integer or {n} integers, got {args.padding!r}")
    s = _ntuple(args.stride, n, op, "stride")
    _require_positive(s, op, "stride")
    d = _ntuple(args.dilation, n, op, "dilation")
    _require_positive(d, op, "dilation")
    p = _ntuple(args.padding, n, op, "padding")
    _require_non_negative(p, op, "padding")
    out_pad = _ntuple(output_padding, n, op, "output_padding")
    _check_output_padding(op, out_pad, s, d)
    _weight_kernel(weight, n, op)  # only to refuse an empty kernel
    xb, unbatched = _add_batch_axis(x, n)
    wi = _unwrap(weight)
    # The engine requires an explicit bias.  The transposed weight's second
    # axis is C_out // groups, so sizing the bias from that axis alone is
    # short by exactly ``groups``.
    if bias is not None:
        b = _unwrap(bias)
    else:
        b = _C_engine.zeros([int(wi.shape[1]) * args.groups], wi.dtype, wi.device)
    out = _wrap(_CONV_TRANSPOSE[n](_unwrap(xb), wi, b, *s, *p, *out_pad, *d, args.groups))
    return _drop_batch_axis(out, unbatched)


def _output_padding_for_size(
    x: Tensor,
    output_size: Sequence[int],
    args: _ConvArgs,
    kernel_size: int | Sequence[int],
    n: int,
    label: str,
) -> tuple[int, ...]:
    """The ``output_padding`` that makes a transposed conv give ``output_size``.

    A stride ``s`` maps ``s`` input sizes onto one convolution output, so
    its transpose has to be told which to give back.  ``output_size`` names
    it, with or without the leading batch and channel entries; each axis
    has to lie within ``s - 1`` of the smallest size it can produce.
    """
    sizes = tuple(int(v) for v in output_size)
    if len(sizes) in (n + 1, n + 2):
        sizes = sizes[len(sizes) - n :]
    if len(sizes) != n:
        raise _C_engine.InvalidArgument(
            f"{label}: output_size must have {n} or {n + 2} elements, got {sizes}"
        )
    if isinstance(args.padding, str):
        raise TypeError(f"{label}: padding must be integers, got {args.padding!r}")
    stride = _ntuple(args.stride, n, label, "stride")
    padding = _ntuple(args.padding, n, label, "padding")
    dilation = _ntuple(args.dilation, n, label, "dilation")
    kernel = _ntuple(kernel_size, n, label, "kernel_size")
    spatial = tuple(int(v) for v in x.shape[x.ndim - n :])
    out: list[int] = []
    for axis in range(n):
        smallest = (spatial[axis] - 1) * stride[axis] - 2 * padding[axis]
        smallest += dilation[axis] * (kernel[axis] - 1) + 1
        if not smallest <= sizes[axis] < smallest + stride[axis]:
            raise _C_engine.InvalidArgument(
                f"{label}: requested an output size of {sizes}, but valid sizes "
                f"on axis {axis} range from {smallest} to "
                f"{smallest + stride[axis] - 1} (for an input of {spatial})"
            )
        out.append(sizes[axis] - smallest)
    return tuple(out)


def _check_output_padding(
    name: str, output_padding: tuple[int, ...], stride: tuple[int, ...], dilation: tuple[int, ...]
) -> None:
    """Refuse an ``output_padding`` the transposed convolution cannot mean.

    ``output_padding`` picks one of the ``stride`` input sizes a strided
    convolution maps to the same output, so it has to be smaller than the
    stride — or than the dilation, which spaces the kernel taps the same
    way.  Anything larger reaches past every input position: the engine used
    to answer anyway, with rows that hold the bias alone.
    """
    for axis, (op, s, d) in enumerate(zip(output_padding, stride, dilation, strict=True)):
        if op < 0 or op >= max(s, d):
            raise _C_engine.InvalidArgument(
                f"{name}: output_padding {tuple(output_padding)} must be "
                f"non-negative and smaller than stride {tuple(stride)} or "
                f"dilation {tuple(dilation)} on every axis (axis {axis}: {op})"
            )


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: str | int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    r"""1-D cross-correlation over a 3-D (or unbatched 2-D) input.

    Slides a learned filter along a single spatial axis.  As with every
    major deep-learning framework, the operation is technically
    cross-correlation rather than strict mathematical convolution (no
    kernel flip), but the "conv" name is kept for familiarity.  Channel
    mixing happens through the ``in_channels`` dimension of ``weight``.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C_in, L)``, or ``(C_in, L)`` unbatched.
    weight : Tensor
        Filters of shape ``(C_out, C_in/groups, kL)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.
    stride : int or tuple of int, optional
        Step between adjacent kernel positions (default ``1``).
    padding : int, tuple of int, or {'valid', 'same'}, optional
        Zero padding applied to both sides of the spatial axis.
        ``"valid"`` is no padding; ``"same"`` pads so the output is as long
        as the input, which needs ``stride=1`` — an odd total puts the
        extra element at the end.
    dilation : int or tuple of int, optional
        Spacing between kernel taps (atrous convolution).  Default ``1``.
    groups : int, optional
        Split the channel dimension into ``groups`` independent groups.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, L_out)`` where

        .. math::

            L_{\text{out}} = \left\lfloor \frac{L + 2p - d(k - 1) - 1}{s} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,\,c_o,\,l} = b_{c_o} + \sum_{c_i, m} w_{c_o,\,c_i,\,m} \cdot x_{i,\,c_i,\,s l + d m}

    Backward is registered automatically; gradients flow to ``input``,
    ``weight``, and ``bias``.  Set ``groups == C_in`` and ``C_out == C_in``
    for a depthwise convolution.  Using ``dilation > 1`` enlarges the
    receptive field without increasing parameter count — common in
    speech / audio models (WaveNet) and 1-D temporal CNNs.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv1d
    >>> x = lucid.randn(2, 4, 50)
    >>> w = lucid.randn(8, 4, 3)
    >>> y = conv1d(x, w, padding=1)
    >>> y.shape
    (2, 8, 50)
    """
    return _convnd(input, weight, bias, _ConvArgs(stride, padding, dilation, groups), 1)


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: str | int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    r"""2-D cross-correlation over a 4-D (or unbatched 3-D) input.

    Despite the name, computes **cross-correlation** rather than strict
    mathematical convolution (no kernel flip).  The kernel slides over
    the input applying a learned linear combination at each spatial
    position; channel mixing happens through the ``in_channels``
    dimension of ``weight``.  This is the workhorse op of modern image
    CNNs (ResNet, ConvNeXt, EfficientNet, ...).

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C_in, H, W)``, or ``(C_in, H, W)`` unbatched.
    weight : Tensor
        Filters of shape ``(C_out, C_in/groups, kH, kW)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.
    stride : int or (int, int), optional
        Step between adjacent kernel positions (default ``1``).
    padding : int, (int, int), or {'valid', 'same'}, optional
        Zero padding on each spatial side.  ``"valid"`` is no padding;
        ``"same"`` keeps the spatial size, needs ``stride=1``, and puts
        the extra element of an odd total after.
    dilation : int or (int, int), optional
        Spacing between kernel taps (atrous convolution).  Default ``1``.
    groups : int, optional
        Split the channels into ``groups`` independent groups.  Setting
        ``groups == C_in`` gives a depthwise convolution.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, H_out, W_out)`` where

        .. math::

            H_{\text{out}} &= \left\lfloor \frac{H + 2 p_H - d_H (k_H - 1) - 1}{s_H} + 1 \right\rfloor \\
            W_{\text{out}} &= \left\lfloor \frac{W + 2 p_W - d_W (k_W - 1) - 1}{s_W} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,\,c_o,\,h,\,w} = b_{c_o} + \sum_{c_i, m, n} w_{c_o,\,c_i,\,m,\,n} \cdot x_{i,\,c_i,\,s_H h + d_H m,\,s_W w + d_W n}

    Backward is well-known; gradients w.r.t. ``input``, ``weight``, and
    ``bias`` flow through automatically.  ``groups > 1`` yields grouped
    convolution (channel-blocks computed independently); ``groups ==
    C_in`` plus ``C_out == C_in`` is depthwise convolution.  Dilation
    enlarges the receptive field without inflating parameter count.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv2d
    >>> x = lucid.randn(1, 3, 32, 32)
    >>> w = lucid.randn(8, 3, 3, 3)
    >>> y = conv2d(x, w, stride=1, padding=1)
    >>> y.shape
    (1, 8, 32, 32)
    """
    return _convnd(input, weight, bias, _ConvArgs(stride, padding, dilation, groups), 2)


def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: str | int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
) -> Tensor:
    r"""3-D cross-correlation over a 5-D (or unbatched 4-D) input.

    Extends :func:`conv2d` to volumetric data (depth × height × width).
    Standard in medical imaging (CT, MRI), video understanding (I3D,
    3D-ResNet, SlowFast) and any setting where the input has three
    spatial axes.  As with the 1-D and 2-D variants, this is technically
    cross-correlation rather than strict convolution.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C_in, D, H, W)``, or ``(C_in, D, H, W)``
        unbatched.
    weight : Tensor
        Filters of shape ``(C_out, C_in/groups, kD, kH, kW)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.
    stride : int or (int, int, int), optional
        Step between adjacent kernel positions per axis (default ``1``).
    padding : int, (int, int, int), or {'valid', 'same'}, optional
        Zero padding on each spatial side; the strings read as in
        :func:`conv2d`.
    dilation : int or (int, int, int), optional
        Spacing between kernel taps (atrous convolution).  Default ``1``.
    groups : int, optional
        Split channels into ``groups`` independent groups.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, D_out, H_out, W_out)`` where each
        spatial size obeys

        .. math::

            D_{\text{out}} = \left\lfloor \frac{D + 2 p_D - d_D (k_D - 1) - 1}{s_D} + 1 \right\rfloor

        (analogously for ``H`` and ``W``).

    Notes
    -----
    Math:

    .. math::

        y_{i,\,c_o,\,d,\,h,\,w} = b_{c_o} + \sum_{c_i,\,m,\,n,\,p} w_{c_o,\,c_i,\,m,\,n,\,p} \cdot x_{i,\,c_i,\,s_D d + d_D m,\,s_H h + d_H n,\,s_W w + d_W p}

    3-D convolution has cubic kernel cost in ``k`` — for large kernels,
    consider factorised variants (e.g. ``(1, k, k)`` followed by
    ``(k, 1, 1)``) which trade expressivity for compute.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv3d
    >>> x = lucid.randn(1, 1, 16, 32, 32)
    >>> w = lucid.randn(4, 1, 3, 3, 3)
    >>> y = conv3d(x, w, padding=1)
    >>> y.shape
    (1, 4, 16, 32, 32)
    """
    return _convnd(input, weight, bias, _ConvArgs(stride, padding, dilation, groups), 3)


def conv_transpose1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
    dilation: int | tuple[int, ...] = 1,
) -> Tensor:
    r"""Transposed 1-D convolution (a.k.a. "fractionally-strided" conv).

    Often called "deconvolution" in the literature, but this is **not**
    the mathematical inverse of :func:`conv1d` — it is the gradient
    operator of a forward convolution, used to map a low-resolution
    feature map back to a higher-resolution one.  The standard
    upsampling primitive in 1-D decoder networks (autoencoders, TTS
    vocoders, etc.).

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C_in, L_in)``, or ``(C_in, L_in)`` unbatched.
    weight : Tensor
        Filters of shape ``(C_in, C_out/groups, kL)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.  When ``None``,
        a zero bias is materialised internally (the engine op requires
        an explicit bias tensor).
    stride : int or tuple of int, optional
        Upsampling factor between input cells (default ``1``).
    padding : int or tuple of int, optional
        Symmetric zero-padding shaved from each side of the output.
    output_padding : int or tuple of int, optional
        Extra one-sided trailing padding added to the output.  Used to
        disambiguate the output size when ``stride > 1``.
    groups : int, optional
        Channel grouping.  ``C_in`` and ``C_out`` must both be
        divisible by it; ``groups == C_in`` is the depthwise case.
    dilation : int or tuple of int, optional
        Spacing between kernel taps.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, L_out)`` where

        .. math::

            L_{\text{out}} = (L_{\text{in}} - 1) \cdot s - 2 p + d (k - 1) + \text{op} + 1

    Notes
    -----
    Conceptually, transposed convolution inserts ``stride - 1`` zeros
    between consecutive input samples and then runs a normal
    convolution — hence the "fractional stride" name.  In practice the
    engine implements it as the matrix-transpose of the corresponding
    forward conv, which avoids materialising the zero-stuffed buffer.

    Checkerboard artifacts (regular high-frequency patterns) commonly
    arise when ``kernel_size`` is not divisible by ``stride``; an
    alternative is :func:`conv1d` after upsampling.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv_transpose1d
    >>> x = lucid.randn(1, 4, 10)
    >>> w = lucid.randn(4, 2, 3)     # (C_in, C_out, kL)
    >>> y = conv_transpose1d(x, w, stride=2)
    >>> y.shape
    (1, 2, 21)
    """
    args = _ConvArgs(stride, padding, dilation, groups)
    return _conv_transposend(input, weight, bias, args, output_padding, 1)


def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    groups: int = 1,
    dilation: int | tuple[int, int] = 1,
) -> Tensor:
    r"""Transposed 2-D convolution — the standard upsampling primitive.

    Also referred to as "deconvolution", though this is not the
    mathematical inverse of :func:`conv2d`.  Performs the transpose of
    a 2-D forward convolution: each input cell scatters a weighted
    copy of the kernel into the (larger) output canvas.  Used in
    image-segmentation decoders, GAN generators, super-resolution
    networks, and U-Net style architectures.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C_in, H_in, W_in)``, or unbatched
        ``(C_in, H_in, W_in)``.
    weight : Tensor
        Filters of shape ``(C_in, C_out/groups, kH, kW)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.  Materialised as
        zeros when ``None``.
    stride : int or (int, int), optional
        Upsampling factor between input cells (default ``1``).
    padding : int or (int, int), optional
        Implicit zero-padding subtracted from each side of the output.
    output_padding : int or (int, int), optional
        Additional one-sided trailing padding on the output, used to
        resolve the size ambiguity when ``stride > 1``.
    groups : int, optional
        Channel grouping.  ``C_in`` and ``C_out`` must both be
        divisible by it; ``groups == C_in`` is the depthwise case.
    dilation : int or (int, int), optional
        Spacing between kernel taps.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, H_out, W_out)`` where each spatial
        dimension satisfies

        .. math::

            H_{\text{out}} = (H_{\text{in}} - 1) \cdot s_H - 2 p_H + d_H (k_H - 1) + \text{op}_H + 1

    Notes
    -----
    Checkerboard artifacts are a well-known failure mode when
    ``kernel_size`` is not a multiple of ``stride``.  Alternatives are
    nearest/bilinear upsample + :func:`conv2d`, or pixel-shuffle
    (sub-pixel convolution).

    The forward of this op equals the **backward of conv2d w.r.t.
    input**; that duality is what makes it useful as a learnable
    upsampler — its gradients give the structure of an ordinary conv.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv_transpose2d
    >>> x = lucid.randn(1, 16, 7, 7)
    >>> w = lucid.randn(16, 8, 4, 4)
    >>> y = conv_transpose2d(x, w, stride=2, padding=1)
    >>> y.shape
    (1, 8, 14, 14)
    """
    args = _ConvArgs(stride, padding, dilation, groups)
    return _conv_transposend(input, weight, bias, args, output_padding, 2)


def conv_transpose3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    output_padding: int | tuple[int, int, int] = 0,
    groups: int = 1,
    dilation: int | tuple[int, int, int] = 1,
) -> Tensor:
    r"""Transposed 3-D convolution — volumetric upsampling.

    Extends :func:`conv_transpose2d` to three spatial dimensions.
    Standard in 3-D segmentation decoders (V-Net, 3D-UNet), volumetric
    GANs, and video generation.  Like the 1-D / 2-D variants, this is
    the matrix-transpose of an ordinary 3-D convolution rather than a
    true mathematical inverse.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C_in, D_in, H_in, W_in)``, or unbatched
        without ``N``.
    weight : Tensor
        Filters of shape ``(C_in, C_out/groups, kD, kH, kW)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.  Materialised as
        zeros when ``None``.
    stride : int or (int, int, int), optional
        Upsampling factor per spatial axis.
    padding : int or (int, int, int), optional
        Implicit zero-padding subtracted from each side of the output.
    output_padding : int or (int, int, int), optional
        Trailing one-sided output padding.
    groups : int, optional
        Channel grouping.  ``C_in`` and ``C_out`` must both be
        divisible by it; ``groups == C_in`` is the depthwise case.
    dilation : int or (int, int, int), optional
        Spacing between kernel taps.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, D_out, H_out, W_out)`` where each
        spatial size obeys

        .. math::

            D_{\text{out}} = (D_{\text{in}} - 1) \cdot s_D - 2 p_D + d_D (k_D - 1) + \text{op}_D + 1

        (analogously for ``H`` and ``W``).

    Notes
    -----
    Memory cost scales cubically in the spatial dimensions; for large
    volumes a typical pattern is to interleave trilinear upsample with
    :func:`conv3d` instead.  Checkerboard artifacts also extend into
    three dimensions when ``kernel_size`` is not a multiple of
    ``stride``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv_transpose3d
    >>> x = lucid.randn(1, 8, 4, 8, 8)
    >>> w = lucid.randn(8, 4, 2, 2, 2)
    >>> y = conv_transpose3d(x, w, stride=2)
    >>> y.shape
    (1, 4, 8, 16, 16)
    """
    args = _ConvArgs(stride, padding, dilation, groups)
    return _conv_transposend(input, weight, bias, args, output_padding, 3)
