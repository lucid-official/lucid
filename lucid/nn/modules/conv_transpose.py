"""Transposed convolution modules, eager and lazy."""

from collections.abc import Sequence
from typing import ClassVar, override

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike, _Size2d, _Size3d
from lucid.nn.module import Module
from lucid.nn.modules._conv_common import (
    _check_groups,
    _single,
    _ConvParameters,
    _LazyConvMixin,
    _pair,
    _triple,
)
from lucid.nn.functional.conv import _ConvArgs, _conv_transposend, _output_padding_for_size


def _transpose_padding(module: Module, padding: object, mode: str) -> str:
    """Refuse string ``padding`` and any ``mode`` but ``"zeros"``; return ``mode``.

    A transposed convolution takes neither, as the reference's does not.
    """
    label = type(module).__name__
    if isinstance(padding, str):
        raise TypeError(
            f"{label}: string padding ({padding!r}) is not supported; "
            "use an int or tuple of ints"
        )
    if mode != "zeros":
        raise ValueError(f'Only "zeros" padding mode is supported for {label}')
    return mode


class _ConvTransposeNd(_ConvParameters):
    """What ``ConvTranspose{1,2,3}d`` share: initialisation and the forward.

    Each subclass sets ``_rank`` and builds its weight in ``__init__``.
    """

    _rank: ClassVar[int]
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    output_padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    groups: int
    padding_mode: str

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )

    def _transpose(self, x: Tensor, output_size: Sequence[int] | None) -> Tensor:
        """The forward of every rank: ``output_size``, when given, picks the
        ``output_padding`` that produces it."""
        args = _ConvArgs(self.stride, self.padding, self.dilation, self.groups)
        out_pad = self.output_padding
        if output_size is not None:
            out_pad = _output_padding_for_size(
                x, output_size, args, self.kernel_size, self._rank, type(self).__name__
            )
        return _conv_transposend(x, self.weight, self.bias, args, out_pad, self._rank)


class ConvTranspose1d(_ConvTransposeNd):
    r"""Applies a 1D transposed convolution (fractionally-strided convolution).

    A transposed convolution — sometimes loosely called a *deconvolution*,
    though it is not a true mathematical inverse — is the gradient of a
    standard convolution with respect to its input.  It upsamples the
    spatial dimension by inserting implicit zeros between input elements
    before applying a convolution, and is the go-to building block for
    decoders, generators, and any network that must increase sequence
    length.

    Formally, for a stride :math:`s` and kernel size :math:`K`:

    .. math::

        L_{\text{out}} = (L_{\text{in}} - 1) \cdot s
                         - 2p + d(K - 1) + p_{\text{out}} + 1

    where :math:`p` is ``padding``, :math:`d` is ``dilation``, and
    :math:`p_{\text{out}}` is ``output_padding``.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input.
    out_channels : int
        Number of channels produced by the transposed convolution.
    kernel_size : int
        Size of the convolving kernel.
    stride : int, optional
        Stride of the convolution.  Values ``> 1`` upsample the input.
        Default: ``1``.
    padding : int, optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding is added
        to both sides of each dimension in the input.  Default: ``0``.
    output_padding : int, optional
        Additional size added to one side of the output shape.  Must
        satisfy ``0 <= output_padding < max(stride, dilation)``.  Used
        to disambiguate the output size when the formula
        :math:`(L_{\text{in}} - 1) \cdot s - 2p + d(K-1) + 1` is
        compatible with multiple :math:`L_{\text{in}}`. Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias. Default: ``True``.
    dilation : int, optional
        Spacing between kernel elements. Default: ``1``.
    padding_mode : str, optional
        ``"zeros"``, the only mode; another raises ``ValueError``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable kernel of shape
        ``(in_channels, out_channels // groups, kernel_size)``.
        Note the channel axis ordering is **transposed** relative to
        :class:`Conv1d`: the leading dimension corresponds to
        ``in_channels``, not ``out_channels``.
        Initialized with Kaiming uniform (:math:`a = \sqrt{5}`).
    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, L)`
    Output:
        :math:`(N, C_{\text{out}}, L_{\text{out}})` where

        .. math::

            L_{\text{out}} = (L - 1) \cdot s - 2p + d(K - 1)
                             + p_{\text{out}} + 1

    Notes
    -----
    **Not a true inverse.**  :class:`ConvTranspose1d` is the transpose
    (adjoint) of :class:`Conv1d` in the linear-algebra sense: if
    ``conv_forward`` maps :math:`\mathbb{R}^{C_{\text{in}} \times L}
    \to \mathbb{R}^{C_{\text{out}} \times L'}`, then
    ``ConvTranspose1d`` with the same weights maps back
    :math:`\mathbb{R}^{C_{\text{out}} \times L'} \to
    \mathbb{R}^{C_{\text{in}} \times L}`.  When used in autoencoders or
    flow models the weights are learned independently and no exact
    inversion is implied.

    **output_padding.**  The transposed convolution output size formula
    can map multiple ``L_in`` values to the same ``L_out``.
    ``output_padding`` breaks this ambiguity by adding a single extra row
    on the output; it does *not* add actual padding to the input.

    Examples
    --------
    Upsample a sequence by factor 2:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> upsample = nn.ConvTranspose1d(in_channels=16, out_channels=16,
    ...                               kernel_size=4, stride=2, padding=1)
    >>> x = lucid.zeros(2, 16, 32)
    >>> y = upsample(x)
    >>> y.shape
    (2, 16, 64)

    Simple 1D decoder layer:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1)
    >>> x = lucid.zeros(4, 64, 20)
    >>> y = decoder(x)
    >>> y.shape
    (4, 32, 20)
    """

    _rank = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int] = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the ConvTranspose1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding_mode = _transpose_padding(self, padding, padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: int = _single(kernel_size, "kernel_size")
        self.stride: int = _single(stride, "stride")
        self.padding: int = _single(padding, "padding")
        self.output_padding: int = _single(output_padding, "output_padding")
        _check_groups(in_channels, out_channels, groups)
        self.groups = groups
        self.dilation: int = _single(dilation, "dilation")
        self._make_parameters(
            (in_channels, out_channels // groups, self.kernel_size), bias, device, dtype
        )

    @override
    def forward(self, input: Tensor, output_size: Sequence[int] | None = None) -> Tensor:
        """Upsample ``input`` of shape ``(N, C_in, L)``, or unbatched.

        ``output_size`` picks one of the ``stride`` sizes the layer can
        produce, in place of ``output_padding``: the output's length,
        optionally after its batch and channel sizes.
        """
        return self._transpose(input, output_size)


class ConvTranspose2d(_ConvTransposeNd):
    r"""Applies a 2D transposed convolution (fractionally-strided convolution).

    Also known as a *fractionally-strided convolution*, this module is
    commonly used as the spatial upsampling primitive in generative
    models (VAEs, GANs), dense prediction decoders (U-Net), and
    super-resolution networks.  It is the transpose (adjoint) of
    :class:`Conv2d`.

    The output spatial dimensions satisfy:

    .. math::

        H_{\text{out}} = (H_{\text{in}} - 1) \cdot s_h - 2p_h
                         + d_h(K_H - 1) + p^{\text{out}}_h + 1

        W_{\text{out}} = (W_{\text{in}} - 1) \cdot s_w - 2p_w
                         + d_w(K_W - 1) + p^{\text{out}}_w + 1

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map.
    out_channels : int
        Number of channels produced by the transposed convolution.
    kernel_size : int or tuple[int, int]
        Size of the convolving kernel.
    stride : int or tuple[int, int], optional
        Stride.  Values ``> 1`` upsample the spatial dimensions.
        Default: ``1``.
    padding : int or tuple[int, int], optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding is added
        to both sides of each spatial dimension.  Default: ``0``.
    output_padding : int or tuple[int, int], optional
        Additional size added to one side of each spatial dimension of
        the output.  Must satisfy
        ``0 <= output_padding < max(stride, dilation)`` along each axis.
        Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias. Default: ``True``.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements. Default: ``1``.
    padding_mode : str, optional
        ``"zeros"``, the only mode; another raises ``ValueError``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable kernel of shape
        ``(in_channels, out_channels // groups, K_H, K_W)``.
        The leading axis is ``in_channels`` — the reverse of
        :class:`Conv2d`.
        Initialized with Kaiming uniform (:math:`a = \sqrt{5}`).
    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, H, W)`
    Output:
        :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
        as given by the formulas above.

    Notes
    -----
    **Checkerboard artefacts.**  Transposed convolutions with
    ``stride > 1`` can produce characteristic checkerboard patterns in
    the output when kernel size is not divisible by stride.  A common
    mitigation is to use ``kernel_size = stride * n`` for some integer
    ``n``, or to replace the transposed conv with bilinear upsampling
    followed by a regular convolution.

    **output_padding.**  When ``stride > 1`` the output size formula is
    not injective: multiple input sizes map to the same output size.
    ``output_padding`` resolves this ambiguity and must be set
    consistently with the encoder stride to reconstruct the exact spatial
    dimensions.

    Examples
    --------
    VAE decoder: upsample 4×4 latent to 8×8:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder = nn.ConvTranspose2d(
    ...     in_channels=128, out_channels=64,
    ...     kernel_size=4, stride=2, padding=1
    ... )
    >>> z = lucid.zeros(8, 128, 4, 4)
    >>> y = decoder(z)
    >>> y.shape
    (8, 64, 8, 8)

    U-Net upsampling block:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    >>> x = lucid.zeros(4, 256, 16, 16)
    >>> y = up(x)
    >>> y.shape
    (4, 128, 32, 32)
    """

    _rank = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d = 0,
        output_padding: _Size2d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size2d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the ConvTranspose2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding_mode = _transpose_padding(self, padding, padding_mode)
        kh, kw = _pair(kernel_size, "kernel_size")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding")
        self.output_padding = _pair(output_padding, "output_padding")
        _check_groups(in_channels, out_channels, groups)
        self.groups = groups
        self.dilation = _pair(dilation, "dilation")
        self._make_parameters((in_channels, out_channels // groups, kh, kw), bias, device, dtype)

    @override
    def forward(self, input: Tensor, output_size: Sequence[int] | None = None) -> Tensor:
        """Upsample ``input`` of shape ``(N, C_in, H, W)``, or unbatched.

        ``output_size`` picks one of the ``stride`` sizes the layer can
        produce, in place of ``output_padding``: the output's height and width,
        optionally after its batch and channel sizes.
        """
        return self._transpose(input, output_size)


class ConvTranspose3d(_ConvTransposeNd):
    r"""Applies a 3D transposed convolution (fractionally-strided convolution).

    The 3D extension of :class:`ConvTranspose2d`.  It upsamples all
    three spatial dimensions simultaneously and is used in volumetric
    decoders such as 3D autoencoders, video generators, and medical
    image synthesis.

    Output size along each spatial axis :math:`X \in \{D, H, W\}`:

    .. math::

        X_{\text{out}} = (X_{\text{in}} - 1) \cdot s_x - 2p_x
                         + d_x(K_X - 1) + p^{\text{out}}_x + 1

    Parameters
    ----------
    in_channels : int
        Number of channels in the input volume.
    out_channels : int
        Number of channels produced by the transposed convolution.
    kernel_size : int or tuple[int, int, int]
        Size of the 3D convolving kernel.
    stride : int or tuple[int, int, int], optional
        Stride.  Values ``> 1`` upsample the spatial dimensions.
        Default: ``1``.
    padding : int or tuple[int, int, int], optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding applied
        on both sides of each axis.  Default: ``0``.
    output_padding : int or tuple[int, int, int], optional
        Additional size added to one side of each spatial dimension.
        Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias. Default: ``True``.
    dilation : int or tuple[int, int, int], optional
        Spacing between kernel elements. Default: ``1``.
    padding_mode : str, optional
        ``"zeros"``, the only mode; another raises ``ValueError``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable kernel of shape
        ``(in_channels, out_channels // groups, K_D, K_H, K_W)``.
        Leading axis is ``in_channels`` (same convention as
        :class:`ConvTranspose2d`).
        Initialized with Kaiming uniform (:math:`a = \sqrt{5}`).
    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, D, H, W)`
    Output:
        :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
        as given by the formula above.

    Notes
    -----
    **Memory.**  3D transposed convolutions produce large feature maps
    at decoder stages.  Gradient checkpointing or smaller
    ``out_channels`` values are often necessary when operating on
    high-resolution volumes.

    **Symmetric decoder design.**  Pair each :class:`Conv3d` in the
    encoder with a :class:`ConvTranspose3d` having identical
    ``kernel_size``, ``stride``, and ``padding`` in the decoder to
    guarantee exact shape reconstruction.

    Examples
    --------
    Volumetric upsampling (2× along all spatial axes):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> up3d = nn.ConvTranspose3d(
    ...     in_channels=64, out_channels=32,
    ...     kernel_size=4, stride=2, padding=1
    ... )
    >>> x = lucid.zeros(2, 64, 4, 8, 8)
    >>> y = up3d(x)
    >>> y.shape
    (2, 32, 8, 16, 16)

    3D autoencoder decoder block:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder3d = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1, padding=1)
    >>> x = lucid.zeros(1, 128, 8, 8, 8)
    >>> y = decoder3d(x)
    >>> y.shape
    (1, 64, 8, 8, 8)
    """

    _rank = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d = 0,
        output_padding: _Size3d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size3d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the ConvTranspose3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding_mode = _transpose_padding(self, padding, padding_mode)
        kd, kh, kw = _triple(kernel_size, "kernel_size")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size, "kernel_size")
        self.stride = _triple(stride, "stride")
        self.padding = _triple(padding, "padding")
        self.output_padding = _triple(output_padding, "output_padding")
        _check_groups(in_channels, out_channels, groups)
        self.groups = groups
        self.dilation = _triple(dilation, "dilation")
        self._make_parameters(
            (in_channels, out_channels // groups, kd, kh, kw), bias, device, dtype
        )

    @override
    def forward(self, input: Tensor, output_size: Sequence[int] | None = None) -> Tensor:
        """Upsample ``input`` of shape ``(N, C_in, D, H, W)``, or unbatched.

        ``output_size`` picks one of the ``stride`` sizes the layer can
        produce, in place of ``output_padding``: the output's depth, height and width,
        optionally after its batch and channel sizes.
        """
        return self._transpose(input, output_size)


# ── Lazy ConvTranspose ────────────────────────────────────────────────────────
# Weight layout is (in_channels, out_channels // groups, *K).  The lazy
# dimension is ``in_channels`` — the leading axis of the saved weight — so
# materialisation reads ``weight.shape[0]``.


class LazyConvTranspose1d(_LazyConvMixin, ConvTranspose1d):
    r"""A :class:`ConvTranspose1d` that infers ``in_channels`` from the first input.

    Combines the fractionally-strided upsampling of
    :class:`ConvTranspose1d` with lazy weight materialization.
    ``in_channels`` need not be known at construction time; it is read
    from ``x.shape[1]`` on the first :meth:`forward` call.

    Parameters
    ----------
    out_channels : int
        Number of output channels after the transposed convolution.
    kernel_size : int
        Length of the 1D convolving kernel.
    stride : int, optional
        Stride (upsampling factor). Default: ``1``.
    padding : int, optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding on each
        side.  String padding is not supported. Default: ``0``.
    output_padding : int, optional
        Additional size added to one side of the output. Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    dilation : int, optional
        Spacing between kernel elements. Default: ``1``.
    padding_mode : str, optional
        ``"zeros"``, the only mode; another raises ``ValueError``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(in_channels, out_channels // groups, kernel_size)`` afterwards.
        Note: leading axis is ``in_channels``, not ``out_channels``.
    bias : Parameter or None
        ``None`` before materialization; ``(out_channels,)`` if
        ``bias=True``.
    in_channels : int or None
        ``None`` before the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, L)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, L_{\text{out}})` — same formula as
        :class:`ConvTranspose1d`.

    Notes
    -----
    **State-dict loading.**  A 3-D ``weight`` tensor in the
    ``state_dict`` whose leading axis equals the inferred ``in_channels``
    triggers materialization before parameter copying.

    Examples
    --------
    Lazy transposed conv in a sequence decoder:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder = nn.LazyConvTranspose1d(
    ...     out_channels=16, kernel_size=4, stride=2, padding=1
    ... )
    >>> x = lucid.zeros(2, 32, 10)   # in_channels=32 inferred
    >>> y = decoder(x)
    >>> y.shape
    (2, 16, 20)

    Inspect pre- and post-materialization state:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConvTranspose1d(out_channels=8, kernel_size=3, padding=1)
    >>> print(lazy.in_channels)
    None
    >>> _ = lazy(lucid.zeros(1, 4, 16))
    >>> print(lazy.in_channels)
    4
    """

    _lazy_rank = 1
    _lazy_transposed = True

    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int] = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LazyConvTranspose1d module. See the class docstring for parameter semantics."""
        Module.__init__(self)
        self.padding_mode = _transpose_padding(self, padding, padding_mode)
        self.out_channels: int = out_channels
        self.kernel_size: int = _single(kernel_size, "kernel_size")
        self.stride: int = _single(stride, "stride")
        self.padding: int = _single(padding, "padding")
        self.output_padding: int = _single(output_padding, "output_padding")
        self.groups: int = groups
        self.dilation: int = _single(dilation, "dilation")
        self._defer(bias, device, dtype)

    @override
    def _kernel(self) -> tuple[int, ...]:
        """The kernel's extent along each spatial axis."""
        return (self.kernel_size,)

    @override
    def forward(self, input: Tensor, output_size: Sequence[int] | None = None) -> Tensor:
        """The eager layer's forward; the first call sizes ``in_channels`` from ``input``."""
        self._materialize_from(input)
        return ConvTranspose1d.forward(self, input, output_size)


class LazyConvTranspose2d(_LazyConvMixin, ConvTranspose2d):
    r"""A :class:`ConvTranspose2d` that infers ``in_channels`` from the first input.

    Lazy version of the 2D transposed convolution.  Weight allocation is
    deferred until the first :meth:`forward` call, making it easy to
    compose upsampling decoder networks without manually tracking the
    channel dimension through each block.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int]
        Size of the 2D convolving kernel.
    stride : int or tuple[int, int], optional
        Stride (upsampling factor). Default: ``1``.
    padding : int or tuple[int, int], optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding applied
        on each side.  String padding is not supported. Default: ``0``.
    output_padding : int or tuple[int, int], optional
        Additional size added to one side of each output dimension.
        Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements. Default: ``1``.
    padding_mode : str, optional
        ``"zeros"``, the only mode; another raises ``ValueError``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(in_channels, out_channels // groups, K_H, K_W)`` afterwards.
    bias : Parameter or None
        ``None`` before materialization; ``(out_channels,)`` if
        ``bias=True``.
    in_channels : int or None
        ``None`` before the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, H, W)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` —
        same formula as :class:`ConvTranspose2d`.

    Notes
    -----
    **State-dict loading.**  A 4-D ``weight`` tensor (shape
    ``(in_channels, out_channels // groups, K_H, K_W)``) triggers
    materialization before parameter copying.

    **Symmetric encoder–decoder pairs.**  Pair each :class:`LazyConv2d`
    encoder stride with a :class:`LazyConvTranspose2d` of matching
    ``kernel_size``, ``stride``, and ``padding`` to reconstruct exact
    spatial dimensions.

    Examples
    --------
    Lazy VAE decoder:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder = nn.LazyConvTranspose2d(
    ...     out_channels=64, kernel_size=4, stride=2, padding=1
    ... )
    >>> z = lucid.zeros(4, 128, 8, 8)   # in_channels=128 inferred
    >>> y = decoder(z)
    >>> y.shape
    (4, 64, 16, 16)

    Verify inference:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConvTranspose2d(out_channels=32, kernel_size=3)
    >>> _ = lazy(lucid.zeros(1, 64, 4, 4))
    >>> print(lazy.in_channels)
    64
    """

    _lazy_rank = 2
    _lazy_transposed = True

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d = 0,
        output_padding: _Size2d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size2d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LazyConvTranspose2d module. See the class docstring for parameter semantics."""
        Module.__init__(self)
        self.padding_mode = _transpose_padding(self, padding, padding_mode)
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = _pair(kernel_size, "kernel_size")
        self.stride: tuple[int, int] = _pair(stride, "stride")
        self.padding: tuple[int, int] = _pair(padding, "padding")
        self.output_padding: tuple[int, int] = _pair(output_padding, "output_padding")
        self.groups: int = groups
        self.dilation: tuple[int, int] = _pair(dilation, "dilation")
        self._defer(bias, device, dtype)

    @override
    def _kernel(self) -> tuple[int, ...]:
        """The kernel's extent along each spatial axis."""
        return self.kernel_size

    @override
    def forward(self, input: Tensor, output_size: Sequence[int] | None = None) -> Tensor:
        """The eager layer's forward; the first call sizes ``in_channels`` from ``input``."""
        self._materialize_from(input)
        return ConvTranspose2d.forward(self, input, output_size)


class LazyConvTranspose3d(_LazyConvMixin, ConvTranspose3d):
    r"""A :class:`ConvTranspose3d` that infers ``in_channels`` from the first input.

    Lazy version of the 3D transposed convolution.  Combines volumetric
    upsampling with deferred weight allocation, simplifying the
    construction of 3D generative decoders and medical image synthesis
    networks where channel dimensions are computed dynamically.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int, int]
        Size of the 3D convolving kernel.
    stride : int or tuple[int, int, int], optional
        Stride (upsampling factor). Default: ``1``.
    padding : int or tuple[int, int, int], optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding applied
        on each side of each spatial axis.  String padding is not
        supported. Default: ``0``.
    output_padding : int or tuple[int, int, int], optional
        Additional size added to one side of each spatial output
        dimension. Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    dilation : int or tuple[int, int, int], optional
        Spacing between kernel elements. Default: ``1``.
    padding_mode : str, optional
        ``"zeros"``, the only mode; another raises ``ValueError``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(in_channels, out_channels // groups, K_D, K_H, K_W)``
        afterwards.
    bias : Parameter or None
        ``None`` before materialization; ``(out_channels,)`` if
        ``bias=True``.
    in_channels : int or None
        ``None`` before the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, D, H, W)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})` —
        same formula as :class:`ConvTranspose3d`.

    Notes
    -----
    **State-dict loading.**  A 5-D ``weight`` tensor triggers
    materialization before parameter copying.

    **Symmetric 3D encoder–decoder.**  Pair each :class:`LazyConv3d`
    with a :class:`LazyConvTranspose3d` using matching ``kernel_size``,
    ``stride``, and ``padding`` for lossless spatial reconstruction.

    Examples
    --------
    Lazy 3D decoder block:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> up3d = nn.LazyConvTranspose3d(
    ...     out_channels=32, kernel_size=4, stride=2, padding=1
    ... )
    >>> x = lucid.zeros(2, 64, 4, 8, 8)   # in_channels=64 inferred
    >>> y = up3d(x)
    >>> y.shape
    (2, 32, 8, 16, 16)

    Verify materialization:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConvTranspose3d(out_channels=16, kernel_size=3, padding=1)
    >>> print(lazy.in_channels)
    None
    >>> _ = lazy(lucid.zeros(1, 8, 4, 4, 4))
    >>> print(lazy.in_channels)
    8
    """

    _lazy_rank = 3
    _lazy_transposed = True

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d = 0,
        output_padding: _Size3d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size3d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LazyConvTranspose3d module. See the class docstring for parameter semantics."""
        Module.__init__(self)
        self.padding_mode = _transpose_padding(self, padding, padding_mode)
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int, int] = _triple(kernel_size, "kernel_size")
        self.stride: tuple[int, int, int] = _triple(stride, "stride")
        self.padding: tuple[int, int, int] = _triple(padding, "padding")
        self.output_padding: tuple[int, int, int] = _triple(output_padding, "output_padding")
        self.groups: int = groups
        self.dilation: tuple[int, int, int] = _triple(dilation, "dilation")
        self._defer(bias, device, dtype)

    @override
    def _kernel(self) -> tuple[int, ...]:
        """The kernel's extent along each spatial axis."""
        return self.kernel_size

    @override
    def forward(self, input: Tensor, output_size: Sequence[int] | None = None) -> Tensor:
        """The eager layer's forward; the first call sizes ``in_channels`` from ``input``."""
        self._materialize_from(input)
        return ConvTranspose3d.forward(self, input, output_size)
