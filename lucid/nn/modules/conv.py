"""
Convolution modules.

The transposed convolutions live in :mod:`lucid.nn.modules.conv_transpose`
and are importable from here too.
"""

from typing import ClassVar, cast, override

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike, _Size2d, _Size3d, PaddingMode
from lucid.nn.module import Module
from lucid.nn.modules._conv_common import (
    _check_groups,
    _single,
    _ConvParameters,
    _LazyConvMixin,
    _pair,
    _triple,
)
from lucid.nn.modules.conv_transpose import (  # importable from here, as before
    ConvTranspose1d as ConvTranspose1d,
    ConvTranspose2d as ConvTranspose2d,
    ConvTranspose3d as ConvTranspose3d,
    LazyConvTranspose1d as LazyConvTranspose1d,
    LazyConvTranspose2d as LazyConvTranspose2d,
    LazyConvTranspose3d as LazyConvTranspose3d,
)
from lucid.nn.functional._spatial_args import _check_string_padding
from lucid.nn.functional.conv import _ConvArgs, _convnd

_VALID_PADDING_MODES = frozenset({"zeros", "reflect", "replicate", "circular"})


def _validate_padding_mode(mode: str) -> PaddingMode:
    if mode not in _VALID_PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {sorted(_VALID_PADDING_MODES)}, got {mode!r}"
        )
    return cast(PaddingMode, mode)


def _string_padding(padding: object, stride: int | tuple[int, ...], label: str) -> str | None:
    """A ``"same"`` / ``"valid"`` padding, checked when the module is built.

    The forward reads the string again through the functional's own rule,
    so the two can only disagree on *when* a bad string is refused.
    """
    if not isinstance(padding, str):
        return None
    strides = (stride,) if isinstance(stride, int) else stride
    _check_string_padding(padding, strides, label)
    return padding


class _ConvNd(_ConvParameters):
    """What ``Conv{1,2,3}d`` share: the forward, ``extra_repr`` and weight init.

    Each subclass sets ``_rank`` and its attributes in ``__init__``.
    """

    _rank: ClassVar[int]
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    groups: int
    padding_mode: PaddingMode
    _padding_str: str | None

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the convolution to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C_{\text{in}}, *)`.

        Returns
        -------
        Tensor
            Output tensor of shape :math:`(N, C_{\text{out}}, *)` with
            spatial dimensions determined by stride, padding, dilation, and
            kernel size.
        """
        args = _ConvArgs(self.stride, self._padding_str or self.padding, self.dilation, self.groups)
        return _convnd(input, self.weight, self.bias, args, self._rank, self.padding_mode)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class Conv1d(_ConvNd):
    r"""Applies a 1D convolution over a sequence of input signals.

    A 1D convolution slides a learned filter (kernel) across a 1D sequence
    and computes a dot product at each position.  Strictly speaking the
    operation is *cross-correlation*:

    .. math::

        y[n, c_{\text{out}}, l] = \sum_{c_{\text{in}}=0}^{C_{\text{in}}/g - 1}
            \sum_{k=0}^{K-1}
            x\!\left[n,\, c_{\text{in}},\, l \cdot s + k \cdot d\right]
            \cdot W\!\left[c_{\text{out}},\, c_{\text{in}},\, k\right]
            + b\!\left[c_{\text{out}}\right]

    where :math:`s` is the stride, :math:`d` is the dilation factor,
    :math:`g` is the number of groups, and :math:`K` is the kernel size.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input signal.
    out_channels : int
        Number of channels produced by the convolution (i.e. number of
        independent filters).
    kernel_size : int
        Length of the 1D convolving kernel.
    stride : int, optional
        Step size between consecutive kernel placements along the length
        dimension. Default: ``1``.
    padding : int or str, optional
        Zero-padding added to both sides of the input before the
        convolution.  Can also be ``"same"`` (output length equals
        ``ceil(L_in / stride)``, requires ``stride=1``) or ``"valid"``
        (no padding; identical to ``padding=0``). Default: ``0``.
    dilation : int, optional
        Spacing between kernel elements.  A dilation of ``d`` inserts
        ``d - 1`` zeros between consecutive kernel weights, expanding the
        effective receptive field without increasing parameter count
        (also called *atrous* convolution). Default: ``1``.
    groups : int, optional
        Splits the input and output channels into ``groups`` independent
        paths.  ``in_channels`` and ``out_channels`` must both be
        divisible by ``groups``.  Setting ``groups = in_channels``
        gives *depthwise* convolution. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias of shape ``(out_channels,)`` is
        added to the output. Default: ``True``.
    padding_mode : str, optional
        Padding strategy when ``padding`` is an integer.  One of
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable filter tensor of shape
        ``(out_channels, in_channels // groups, kernel_size)``.
        Initialized with Kaiming uniform:

        .. math::

            \text{fan\_in} = \frac{C_{\text{in}}}{g} \cdot K, \quad
            W \sim \mathcal{U}\!\left[
                -\sqrt{\tfrac{6}{\text{fan\_in}}},\;
                \sqrt{\tfrac{6}{\text{fan\_in}}}
            \right]

        (using :math:`a = \sqrt{5}` in the Kaiming formula).
    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None`` if
        ``bias=False``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, L)`
    Output:
        :math:`(N, C_{\text{out}}, L_{\text{out}})` where

        .. math::

            L_{\text{out}} = \left\lfloor
                \frac{L + 2p - d(K - 1) - 1}{s} + 1
            \right\rfloor

        and :math:`p` is the (symmetric) padding, :math:`d` is dilation,
        :math:`K` is kernel size, :math:`s` is stride.

    Notes
    -----
    **Groups and depthwise convolution.**  When ``groups = g > 1``, the
    input channels are split into ``g`` groups of size
    ``in_channels // g``, each group is convolved independently, and the
    results are concatenated.  Depthwise convolution (``groups =
    in_channels``) applies one filter per input channel, dramatically
    reducing the parameter count compared to a full convolution.

    **Dilated (atrous) convolution.**  Setting ``dilation > 1`` inserts
    gaps between kernel taps, enlarging the receptive field without
    additional parameters or pooling.  This is commonly used in semantic
    segmentation and WaveNet-style audio models.

    **Padding modes.**  ``"reflect"`` and ``"circular"`` padding
    avoid boundary artefacts that zero-padding can introduce;
    ``"replicate"`` repeats edge values.  These modes are applied via
    a pre-padding step followed by a ``padding=0`` convolution.

    Examples
    --------
    Basic 1D convolution:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    >>> x = lucid.zeros(4, 1, 16)   # (N, C_in, L)
    >>> y = conv(x)
    >>> y.shape
    (4, 8, 16)

    Depthwise 1D convolution (one filter per channel):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> depthwise = nn.Conv1d(
    ...     in_channels=32, out_channels=32,
    ...     kernel_size=5, padding=2, groups=32
    ... )
    >>> x = lucid.zeros(2, 32, 64)
    >>> y = depthwise(x)
    >>> y.shape
    (2, 32, 64)
    """

    _rank = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the Conv1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: int = _single(kernel_size, "kernel_size")
        self.stride: int = _single(stride, "stride")
        self.dilation: int = _single(dilation, "dilation")
        _check_groups(in_channels, out_channels, groups)
        self.groups = groups
        self.padding_mode = _validate_padding_mode(padding_mode)
        self._padding_str: str | None = _string_padding(padding, self.stride, type(self).__name__)
        self.padding: int = 0 if isinstance(padding, str) else _single(padding, "padding")
        self._make_parameters(
            (out_channels, in_channels // groups, self.kernel_size), bias, device, dtype
        )


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over a batch of images or feature maps.

    This module computes the 2D *cross-correlation* between the input
    and a set of learnable filters.  For a single output channel and a
    single input channel the operation is:

    .. math::

        y[n, c_{\text{out}}, h, w] =
        \sum_{c_{\text{in}}=0}^{C_{\text{in}}/g - 1}
        \sum_{k_h=0}^{K_H-1} \sum_{k_w=0}^{K_W-1}
        x\!\left[n,\; c_{\text{in}},\;
                  h \cdot s_h + k_h \cdot d_h,\;
                  w \cdot s_w + k_w \cdot d_w\right]
        \cdot W\!\left[c_{\text{out}},\; c_{\text{in}},\; k_h,\; k_w\right]
        + b\!\left[c_{\text{out}}\right]

    where :math:`(s_h, s_w)` is the stride, :math:`(d_h, d_w)` is the
    dilation factor, and :math:`g` is the number of groups.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple[int, int]
        Size of the convolving kernel.  A single ``int`` is broadcast
        to ``(kernel_size, kernel_size)``.
    stride : int or tuple[int, int], optional
        Stride of the convolution. Default: ``1``.
    padding : int, tuple[int, int], or str, optional
        Padding added to all four sides of the input.  ``"same"`` pads
        so the output spatial size equals ``ceil(H_in / s)`` (requires
        ``stride=1``); ``"valid"`` means no padding. Default: ``0``.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements (atrous / dilated convolution).
        Default: ``1``.
    groups : int, optional
        Number of blocked connections from input channels to output
        channels.  Both ``in_channels`` and ``out_channels`` must be
        divisible by ``groups``.  ``groups = in_channels`` gives
        *depthwise* convolution. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias to the output. Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable filter tensor of shape
        ``(out_channels, in_channels // groups, K_H, K_W)``.
        Initialized with Kaiming uniform using :math:`a = \sqrt{5}`:

        .. math::

            \text{fan\_in} = \frac{C_{\text{in}}}{g} \cdot K_H \cdot K_W, \quad
            W \sim \mathcal{U}\!\left[
                -\sqrt{\tfrac{6}{\text{fan\_in}}},\;
                \sqrt{\tfrac{6}{\text{fan\_in}}}
            \right]

    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, H, W)`
    Output:
        :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` where

        .. math::

            H_{\text{out}} = \left\lfloor
                \frac{H + 2p_h - d_h(K_H - 1) - 1}{s_h} + 1
            \right\rfloor, \quad
            W_{\text{out}} = \left\lfloor
                \frac{W + 2p_w - d_w(K_W - 1) - 1}{s_w} + 1
            \right\rfloor

    Notes
    -----
    **Groups and depthwise convolution.**  When ``groups = in_channels``
    each input channel is convolved with its own filter, yielding
    *depthwise* convolution.  This is the building block of
    MobileNet-style architectures.  A subsequent ``groups=1`` Conv2d
    with kernel size 1 (pointwise convolution) forms a
    *depthwise-separable* block.

    **Dilated (atrous) convolution.**  ``dilation > 1`` enlarges the
    receptive field of each kernel tap without increasing the number of
    parameters or reducing the spatial resolution.  Widely used in
    semantic segmentation (DeepLab) and generative models.

    **padding="same".**  Mimics the ``SAME`` padding convention: output
    is spatially identical in size to the input.  When the required
    total padding is odd, the extra pixel is added on the bottom/right
    side (low side gets ``pad_total // 2``).  Requires ``stride=1``.

    Examples
    --------
    Basic image convolution:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
    >>> x = lucid.zeros(8, 3, 32, 32)   # (N, C_in, H, W)
    >>> y = conv(x)
    >>> y.shape
    (8, 64, 32, 32)

    Depthwise separable convolution block:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> depthwise  = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
    >>> pointwise  = nn.Conv2d(32, 64, kernel_size=1)
    >>> x = lucid.zeros(4, 32, 16, 16)
    >>> y = pointwise(depthwise(x))
    >>> y.shape
    (4, 64, 16, 16)

    Dilated convolution (receptive field 9×9 with only 3×3 parameters):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> dilated = nn.Conv2d(1, 1, kernel_size=3, padding=4, dilation=4)
    >>> x = lucid.zeros(1, 1, 16, 16)
    >>> y = dilated(x)
    >>> y.shape
    (1, 1, 16, 16)

    Convolution without bias:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv_no_bias = nn.Conv2d(16, 32, kernel_size=1, bias=False)
    >>> x = lucid.zeros(2, 16, 8, 8)
    >>> y = conv_no_bias(x)
    >>> y.shape
    (2, 32, 8, 8)
    """

    _rank = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d | str = 0,
        dilation: _Size2d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the Conv2d module. See the class docstring for parameter semantics."""
        super().__init__()
        kh, kw = _pair(kernel_size, "kernel_size")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: tuple[int, int] = _pair(kernel_size, "kernel_size")
        self.stride: tuple[int, int] = _pair(stride, "stride")
        self.dilation: tuple[int, int] = _pair(dilation, "dilation")
        _check_groups(in_channels, out_channels, groups)
        self.groups = groups
        self.padding_mode = _validate_padding_mode(padding_mode)
        self._padding_str: str | None = _string_padding(padding, self.stride, type(self).__name__)
        self.padding: tuple[int, int] = (
            (0, 0) if isinstance(padding, str) else _pair(padding, "padding")
        )
        self._make_parameters((out_channels, in_channels // groups, kh, kw), bias, device, dtype)


class Conv3d(_ConvNd):
    r"""Applies a 3D convolution over volumetric data (e.g. video or medical scans).

    Computes the 3D *cross-correlation* of the input with a bank of
    learnable 3D filters:

    .. math::

        y[n, c_{\text{out}}, d, h, w] =
        \sum_{c_{\text{in}}=0}^{C_{\text{in}}/g - 1}
        \sum_{k_d, k_h, k_w}
        x\!\left[n,\; c_{\text{in}},\;
                  d \cdot s_d + k_d \cdot d_d,\;
                  h \cdot s_h + k_h \cdot d_h,\;
                  w \cdot s_w + k_w \cdot d_w\right]
        \cdot W\!\left[c_{\text{out}},\; c_{\text{in}},\; k_d,\; k_h,\; k_w\right]
        + b\!\left[c_{\text{out}}\right]

    Parameters
    ----------
    in_channels : int
        Number of channels in the input volume.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple[int, int, int]
        Size of the 3D convolving kernel ``(K_D, K_H, K_W)``.
        A single ``int`` is broadcast to all three dimensions.
    stride : int or tuple[int, int, int], optional
        Stride along each spatial dimension. Default: ``1``.
    padding : int, tuple[int, int, int], or str, optional
        Zero-padding added on both sides along each spatial dimension.
        Accepts ``"same"`` (requires ``stride=1``) or ``"valid"``.
        Default: ``0``.
    dilation : int or tuple[int, int, int], optional
        Spacing between kernel elements. Default: ``1``.
    groups : int, optional
        Number of blocked connections.  ``groups = in_channels`` gives
        depthwise 3D convolution. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias. Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable filter tensor of shape
        ``(out_channels, in_channels // groups, K_D, K_H, K_W)``.
        Initialized with Kaiming uniform:

        .. math::

            \text{fan\_in} = \frac{C_{\text{in}}}{g}
                             \cdot K_D \cdot K_H \cdot K_W, \quad
            W \sim \mathcal{U}\!\left[
                -\sqrt{\tfrac{6}{\text{fan\_in}}},\;
                \sqrt{\tfrac{6}{\text{fan\_in}}}
            \right]

    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, D, H, W)`
    Output:
        :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})` where

        .. math::

            X_{\text{out}} = \left\lfloor
                \frac{X + 2p_x - d_x(K_X - 1) - 1}{s_x} + 1
            \right\rfloor \quad \text{for } X \in \{D, H, W\}

    Notes
    -----
    **Typical use cases.**  Conv3d is the standard building block for
    video understanding (3D ResNets, SlowFast), medical image analysis
    (CT/MRI volumetric segmentation), and point-cloud processing.  It is
    computationally heavier than Conv2d by a factor of roughly
    :math:`K_D` per layer; factorised (2+1)D convolutions are a common
    approximation.

    **Memory.**  A single 3D feature map can be large; consider
    ``groups > 1`` or smaller ``kernel_size`` when memory is a concern.

    Examples
    --------
    Basic volumetric convolution:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv3 = nn.Conv3d(in_channels=1, out_channels=16,
    ...                   kernel_size=3, padding=1)
    >>> x = lucid.zeros(2, 1, 16, 32, 32)   # (N, C, D, H, W)
    >>> y = conv3(x)
    >>> y.shape
    (2, 16, 16, 32, 32)

    Strided 3D convolution for spatial downsampling:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv3_stride = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
    >>> x = lucid.zeros(2, 16, 16, 32, 32)
    >>> y = conv3_stride(x)
    >>> y.shape
    (2, 32, 8, 16, 16)
    """

    _rank = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d | str = 0,
        dilation: _Size3d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the Conv3d module. See the class docstring for parameter semantics."""
        super().__init__()
        kd, kh, kw = _triple(kernel_size, "kernel_size")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: tuple[int, int, int] = _triple(kernel_size, "kernel_size")
        self.stride: tuple[int, int, int] = _triple(stride, "stride")
        self.dilation: tuple[int, int, int] = _triple(dilation, "dilation")
        _check_groups(in_channels, out_channels, groups)
        self.groups = groups
        self.padding_mode = _validate_padding_mode(padding_mode)
        self._padding_str: str | None = _string_padding(padding, self.stride, type(self).__name__)
        self.padding: tuple[int, int, int] = (
            (0, 0, 0) if isinstance(padding, str) else _triple(padding, "padding")
        )
        self._make_parameters(
            (out_channels, in_channels // groups, kd, kh, kw), bias, device, dtype
        )


# ── Lazy convolutions ─────────────────────────────────────────────────────────
#
# Each lazy variant inherits from its eager counterpart so the forward path
# (``_convnd`` / ``_conv_transposend``) is reused unchanged.  Parent
# `__init__` is intentionally skipped via `Module.__init__(self)` because it
# requires `in_channels`, which is what we are deferring.  The first forward
# call (or `_load_from_state_dict`) materialises the real `weight` / `bias`
# Parameter objects.


class LazyConv1d(_LazyConvMixin, Conv1d):
    r"""A :class:`Conv1d` that infers ``in_channels`` from the first input.

    Lazy modules defer the allocation and initialization of ``weight``
    and ``bias`` until the first call to :meth:`forward` (or until a
    compatible ``state_dict`` is loaded).  This removes the need to know
    ``in_channels`` at construction time, simplifying sequential model
    building and automatic architecture search.

    Materialization happens exactly once: on the first :meth:`forward`
    call the channel count is read from ``x.shape[1]``, weights are
    allocated and Kaiming-uniform initialized, and subsequent calls
    behave identically to a fully-initialized :class:`Conv1d`.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int
        Length of the 1D convolving kernel.
    stride : int, optional
        Stride of the convolution. Default: ``1``.
    padding : int or str, optional
        Zero-padding or ``"same"`` / ``"valid"`` string specifier.
        Default: ``0``.
    dilation : int, optional
        Spacing between kernel elements. Default: ``1``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device used when allocating weights at materialization time.
        Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` until first :meth:`forward`; afterwards shape
        ``(out_channels, in_channels // groups, kernel_size)``.
    bias : Parameter or None
        ``None`` until first :meth:`forward`; afterwards shape
        ``(out_channels,)`` if ``bias=True``, else remains ``None``.
    in_channels : int or None
        ``None`` before materialization; set to the inferred value on
        the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, L)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, L_{\text{out}})` — same formula as
        :class:`Conv1d`.

    Notes
    -----
    **State-dict loading.**  Loading a ``state_dict`` that contains a
    ``weight`` key of shape
    ``(out_channels, in_channels // groups, kernel_size)`` also triggers
    materialization, so a lazy module can be restored from a checkpoint
    without ever calling :meth:`forward`.

    **groups constraint.**  ``in_channels`` (inferred at runtime) must
    be divisible by ``groups``; this is checked inside ``_initialize``.

    Examples
    --------
    Lazy conv in a sequential model:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Sequential(
    ...     nn.LazyConv1d(out_channels=32, kernel_size=3, padding=1),
    ...     nn.ReLU(),
    ... )
    >>> x = lucid.zeros(4, 16, 64)   # in_channels=16 inferred here
    >>> y = model(x)
    >>> y.shape
    (4, 32, 64)

    Check that in_channels was inferred:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConv1d(out_channels=8, kernel_size=5, padding=2)
    >>> print(lazy.in_channels)
    None
    >>> _ = lazy(lucid.zeros(1, 3, 20))
    >>> print(lazy.in_channels)
    3
    """

    _lazy_rank = 1

    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LazyConv1d module. See the class docstring for parameter semantics."""
        Module.__init__(self)
        self.out_channels: int = out_channels
        self.kernel_size: int = _single(kernel_size, "kernel_size")
        self.stride: int = _single(stride, "stride")
        self.dilation: int = _single(dilation, "dilation")
        self.groups: int = groups
        self.padding_mode: PaddingMode = _validate_padding_mode(padding_mode)
        self._padding_str: str | None
        self._padding_str = _string_padding(padding, self.stride, type(self).__name__)
        self.padding: int = 0 if isinstance(padding, str) else _single(padding, "padding")
        self._defer(bias, device, dtype)

    @override
    def _kernel(self) -> tuple[int, ...]:
        """The kernel's extent along each spatial axis."""
        return (self.kernel_size,)

    @override
    def forward(self, input: Tensor) -> Tensor:
        """The eager layer's forward; the first call sizes ``in_channels`` from ``input``."""
        self._materialize_from(input)
        return Conv1d.forward(self, input)


class LazyConv2d(_LazyConvMixin, Conv2d):
    r"""A :class:`Conv2d` that infers ``in_channels`` from the first input.

    Lazy initialization allows building 2D convolutional networks without
    knowing the input channel count in advance.  The module registers
    ``weight`` and ``bias`` as ``None`` parameters and allocates them on
    the first :meth:`forward` call, reading ``C_in`` from
    ``x.shape[1]``.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int]
        Size of the 2D convolving kernel.
    stride : int or tuple[int, int], optional
        Stride of the convolution. Default: ``1``.
    padding : int, tuple[int, int], or str, optional
        Zero-padding or ``"same"`` / ``"valid"`` string specifier.
        Default: ``0``.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements. Default: ``1``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(out_channels, in_channels // groups, K_H, K_W)`` afterwards.
    bias : Parameter or None
        ``None`` before materialization; shape ``(out_channels,)`` if
        ``bias=True``, else remains ``None``.
    in_channels : int or None
        ``None`` before the first forward pass; inferred channel count
        afterwards.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, H, W)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` —
        same formula as :class:`Conv2d`.

    Notes
    -----
    **State-dict materialization.**  Loading a ``state_dict`` with a
    4-D ``weight`` tensor triggers materialization before parameter
    copying, enabling round-trip checkpoint compatibility.

    Examples
    --------
    Dynamic channel inference in a feature extractor:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Sequential(
    ...     nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1),
    ...     nn.ReLU(),
    ...     nn.LazyConv2d(out_channels=128, kernel_size=3, padding=1),
    ... )
    >>> x = lucid.zeros(2, 3, 32, 32)   # in_channels=3 inferred here
    >>> y = model(x)
    >>> y.shape
    (2, 128, 32, 32)

    Verify inferred in_channels after forward:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConv2d(out_channels=16, kernel_size=3, padding=1)
    >>> _ = lazy(lucid.zeros(1, 8, 16, 16))
    >>> print(lazy.in_channels)
    8
    """

    _lazy_rank = 2

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d | str = 0,
        dilation: _Size2d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LazyConv2d module. See the class docstring for parameter semantics."""
        Module.__init__(self)
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = _pair(kernel_size, "kernel_size")
        self.stride: tuple[int, int] = _pair(stride, "stride")
        self.dilation: tuple[int, int] = _pair(dilation, "dilation")
        self.groups: int = groups
        self.padding_mode: PaddingMode = _validate_padding_mode(padding_mode)
        self._padding_str: str | None
        self._padding_str = _string_padding(padding, self.stride, type(self).__name__)
        self.padding: tuple[int, int] = (
            (0, 0) if isinstance(padding, str) else _pair(padding, "padding")
        )
        self._defer(bias, device, dtype)

    @override
    def _kernel(self) -> tuple[int, ...]:
        """The kernel's extent along each spatial axis."""
        return self.kernel_size

    @override
    def forward(self, input: Tensor) -> Tensor:
        """The eager layer's forward; the first call sizes ``in_channels`` from ``input``."""
        self._materialize_from(input)
        return Conv2d.forward(self, input)


class LazyConv3d(_LazyConvMixin, Conv3d):
    r"""A :class:`Conv3d` that infers ``in_channels`` from the first input.

    Lazy initialization for 3D convolution: ``in_channels`` need not be
    specified at construction time.  Weight allocation and Kaiming
    uniform initialization are deferred until the first :meth:`forward`
    call (or ``state_dict`` load), at which point ``C_in`` is read from
    ``x.shape[1]``.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int, int]
        Size of the 3D convolving kernel.
    stride : int or tuple[int, int, int], optional
        Stride of the convolution. Default: ``1``.
    padding : int, tuple[int, int, int], or str, optional
        Zero-padding or ``"same"`` / ``"valid"`` string specifier.
        Default: ``0``.
    dilation : int or tuple[int, int, int], optional
        Spacing between kernel elements. Default: ``1``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(out_channels, in_channels // groups, K_D, K_H, K_W)`` afterwards.
    bias : Parameter or None
        ``None`` before materialization; ``(out_channels,)`` if
        ``bias=True``, else ``None``.
    in_channels : int or None
        ``None`` before the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, D, H, W)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})` —
        same formula as :class:`Conv3d`.

    Notes
    -----
    **State-dict materialization.**  A 5-D ``weight`` tensor in the
    ``state_dict`` triggers materialization before parameter copying.

    Examples
    --------
    Lazy 3D conv in a volumetric network:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy3d = nn.LazyConv3d(out_channels=32, kernel_size=3, padding=1)
    >>> x = lucid.zeros(2, 4, 16, 32, 32)   # in_channels=4 inferred
    >>> y = lazy3d(x)
    >>> y.shape
    (2, 32, 16, 32, 32)

    Verify materialization:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy3d = nn.LazyConv3d(out_channels=16, kernel_size=3, padding=1)
    >>> print(lazy3d.in_channels)
    None
    >>> _ = lazy3d(lucid.zeros(1, 8, 4, 4, 4))
    >>> print(lazy3d.in_channels)
    8
    """

    _lazy_rank = 3

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d | str = 0,
        dilation: _Size3d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LazyConv3d module. See the class docstring for parameter semantics."""
        Module.__init__(self)
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int, int] = _triple(kernel_size, "kernel_size")
        self.stride: tuple[int, int, int] = _triple(stride, "stride")
        self.dilation: tuple[int, int, int] = _triple(dilation, "dilation")
        self.groups: int = groups
        self.padding_mode: PaddingMode = _validate_padding_mode(padding_mode)
        self._padding_str: str | None
        self._padding_str = _string_padding(padding, self.stride, type(self).__name__)
        self.padding: tuple[int, int, int] = (
            (0, 0, 0) if isinstance(padding, str) else _triple(padding, "padding")
        )
        self._defer(bias, device, dtype)

    @override
    def _kernel(self) -> tuple[int, ...]:
        """The kernel's extent along each spatial axis."""
        return self.kernel_size

    @override
    def forward(self, input: Tensor) -> Tensor:
        """The eager layer's forward; the first call sizes ``in_channels`` from ``input``."""
        self._materialize_from(input)
        return Conv3d.forward(self, input)
