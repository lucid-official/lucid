"""Quantized ``ConvTranspose{1,2,3}d`` — int8 weight, dequant-to-float forward.

Same sidecar design (B) as the quantized :class:`~lucid.nn.quantized.Conv2d`
family: the transposed-conv kernel is stored int8 (**per-channel** symmetric on
the output-channel axis — the transposed weight layout ``(in, out/groups, *k)``
puts the output channels on axis 1, so the weight is quantized per-channel there,
matching the reference default), the forward dequantizes it, runs the ordinary
transposed convolution, then fake-quantizes the output to the calibrated
activation grid.
"""

from typing import TYPE_CHECKING, Protocol, Self, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import activation_qparams
from lucid.quantization._functional import dequantize, fake_quantize, quantize
from lucid.quantization._qscheme import QDtype, per_channel_symmetric, qint8, quint8

# An alias of builtins, so it costs nothing at runtime and does not need to
# hide inside TYPE_CHECKING — which is what forced the annotations below to be
# strings, against H7.
_IntTuple = int | tuple[int, ...]

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    class _FloatConvT(Protocol):
        """Structural view of a calibrated float transposed conv."""

        in_channels: int
        out_channels: int
        kernel_size: _IntTuple
        stride: _IntTuple
        padding: _IntTuple | str
        output_padding: _IntTuple
        dilation: _IntTuple
        groups: int
        weight: Tensor
        bias: Tensor | None
        qconfig: object


class _QuantizedConvTransposeNd(nn.Module):
    """Shared implementation for the quantized transposed-conv family."""

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor
    scale: Tensor
    zero_point: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _IntTuple,
        stride: _IntTuple,
        padding: _IntTuple,
        output_padding: _IntTuple,
        dilation: _IntTuple,
        groups: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.out_qdtype: QDtype = quint8
        # Per-output-channel weight quant on the transposed-weight axis 1
        # (``(in, out/groups, *k)``) — matches the reference default and is much
        # tighter than per-tensor for wide-range (esp. 3d) kernels.
        self.weight_ch_axis = 1
        n_out = out_channels // groups
        ks = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        weight_shape = (in_channels, n_out, *ks)
        self.register_buffer("weight_int8", lucid.zeros(weight_shape, dtype=lucid.int8))
        self.register_buffer("weight_scale", lucid.ones(n_out))
        self.register_buffer("weight_zero_point", lucid.zeros(n_out))
        if bias:
            self.register_buffer("bias", lucid.zeros(out_channels))
        else:
            self.bias = None
        self.register_buffer("scale", lucid.tensor(1.0))
        self.register_buffer("zero_point", lucid.tensor(0.0))

    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        """Run the rank-specific transposed convolution — overridden per subclass."""
        raise NotImplementedError

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Dequantize the kernel, transposed-convolve, fake-quantize the output."""
        weight = dequantize(
            self.weight_int8,
            self.weight_scale,
            self.weight_zero_point,
            ch_axis=self.weight_ch_axis,
        )
        y = self._conv_forward(input, weight)
        return fake_quantize(
            y,
            self.scale,
            self.zero_point,
            self.out_qdtype.quant_min,
            self.out_qdtype.quant_max,
        )

    @classmethod
    def from_float(cls, mod: nn.Module) -> Self:
        """Quantize a calibrated float transposed conv (per-channel int8 weight)."""
        from lucid.quantization.observer import PerChannelMinMaxObserver

        f = cast("_FloatConvT", mod)
        if isinstance(f.padding, str):
            raise NotImplementedError(
                "quantized conv_transpose: string padding is not supported yet"
            )
        qmod = cls(
            f.in_channels,
            f.out_channels,
            f.kernel_size,
            f.stride,
            f.padding,
            f.output_padding,
            f.dilation,
            f.groups,
            bias=f.bias is not None,
        )
        wobs = PerChannelMinMaxObserver(ch_axis=1, qscheme=per_channel_symmetric, qdtype=qint8)
        wobs(f.weight)
        w_scale, w_zp = wobs.calculate_qparams()
        qmod.register_buffer("weight_int8", quantize(f.weight, w_scale, w_zp, qint8, ch_axis=1))
        qmod.register_buffer("weight_scale", w_scale)
        qmod.register_buffer("weight_zero_point", w_zp)
        if f.bias is not None:
            qmod.register_buffer("bias", f.bias.detach())
        a_scale, a_zp, a_qdtype = activation_qparams(mod)
        qmod.register_buffer("scale", a_scale)
        qmod.register_buffer("zero_point", a_zp)
        qmod.out_qdtype = a_qdtype
        return qmod

    @override
    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, qdtype={self.out_qdtype.name}"
        )


class ConvTranspose1d(_QuantizedConvTransposeNd):
    r"""Quantized 1-D transposed (fractionally-strided) convolution.

    The int8-weight, dequantize-to-float replacement that
    :func:`lucid.quantization.convert` installs for a calibrated float
    :class:`~lucid.nn.ConvTranspose1d`. Transposed convolutions *upsample* — they
    are the decoder / generator counterpart of :class:`~lucid.nn.quantized.Conv1d`
    in 1-D signal models — and share the exact sidecar recipe of the quantized
    :class:`~lucid.nn.quantized.Conv2d` family.

    **Representation (sidecar design B).** The learned float kernel is quantized
    once, at :meth:`from_float` time, into an ``int8`` code tensor plus a
    per-output-channel ``scale`` / ``zero_point``; the float kernel is then
    dropped. Only the int8 codes, the qparams, and the (still-float) bias live in
    the module. Each forward *dequantizes* the kernel back to float, runs the
    ordinary ``F.conv_transpose1d``, and finally *fake-quantizes* the output onto
    the activation grid that calibration observed. Keeping the transposed
    convolution itself in float means the result matches a genuine int8 kernel to
    within a rounding step **and** the layer runs unchanged on any device.

    **Per-channel weights (axis 1).** The transposed-weight layout ``(in_channels,
    out_channels/groups, k)`` places the *output* channels on **axis 1**, not axis
    0, so the kernel is quantized per-channel along axis 1 — each output channel
    gets its own scale, far tighter than one per-tensor scale for wide kernels.
    The scheme is **symmetric** (``qint8``, ``zero_point = 0``), matching the
    reference default for transposed weights; the bias stays float.

    Encoding happens once (at :meth:`from_float`); decode + transposed-convolve run
    on every forward:

    .. math::

        w^{q}_{c,o,k} = \operatorname{clamp}\!\bigl(
            \operatorname{round}(w_{c,o,k}/s_o),\ -128,\ 127\bigr),
        \qquad
        \hat{w}_{c,o,k} = w^{q}_{c,o,k}\, s_o

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{convT1d}(x, \hat{w}) + b\bigr),
        \qquad
        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    where :math:`s_o` is the per-output-channel (axis-1) weight scale — with a zero
    zero-point because the weight scheme is symmetric — and :math:`S, Z` the scalar
    calibrated output ``(scale, zero_point)`` with grid bounds
    :math:`q_{\min}, q_{\max}` (``0, 255`` for the default ``quint8``).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input signal.
    out_channels : int
        Number of channels produced by the transposed convolution — quantized
        per-channel (on axis 1 of the transposed weight).
    kernel_size : int or tuple of int
        Size of the convolving kernel.
    stride : int or tuple of int
        Stride of the transposed convolution (the upsampling factor).
    padding : int or tuple of int
        Implicit zero-padding subtracted from both sides of the spatial axis.
    output_padding : int or tuple of int
        Extra size added to one side of the output to disambiguate its shape.
    dilation : int or tuple of int
        Spacing between kernel elements.
    groups : int
        Number of blocked connections from input to output channels.
    bias : bool
        Whether a learned **float** bias term is added after the convolution.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` kernel codes, shape ``(in_channels, out_channels/groups, k)``.
    weight_scale : Tensor
        Per-output-channel weight scale, shape ``(out_channels/groups,)``.
    weight_zero_point : Tensor
        Per-output-channel weight zero-point, shape ``(out_channels/groups,)`` —
        all zero under the symmetric scheme.
    scale : Tensor
        Scalar output-activation scale, set from the calibrated observer.
    zero_point : Tensor
        Scalar output-activation zero-point, set from the calibrated observer.
    bias : Tensor or None
        The float bias of shape ``(out_channels,)``, or ``None``.

    Notes
    -----
    - Instances are normally produced by :func:`lucid.quantization.convert` (or
      :meth:`from_float`), **not** constructed directly: a freshly constructed
      layer has identity qparams (``scale = 1``, ``zero_point = 0``) and zeroed
      kernel codes, and only becomes meaningful once ``from_float`` /
      ``load_state_dict`` fills the buffers.
    - Only int8 codes + qparams + float bias enter the ``state_dict``; the float
      kernel never does — the weight payload shrinks ~``3.55x`` (int8), and a
      whole conv-heavy checkpoint ~``3.97x`` (measured on ``resnet_18``).
    - The kernel is quantized **per-channel on axis 1** (the output-channel axis
      of the ``(in, out/groups, k)`` layout) with a **symmetric** ``qint8`` scheme
      (``from_float`` runs a fresh ``PerChannelMinMaxObserver`` over the weight);
      the bias stays float.
    - This layer wins on **memory**, not compute — the transposed convolution runs
      in float (CPU stream = Accelerate, GPU stream = MLX), so its speed tracks the
      float layer's; the payoff is the smaller checkpoint and working set.
    - Device-transparent: because compute stays in float, the layer runs unchanged
      on CPU or Metal and follows the input tensor's device.
    - Calibration is required — the output ``(scale, zero_point)`` come from an
      activation observer that must see representative data between ``prepare`` and
      ``convert``; an uncalibrated observer keeps its ``±inf`` seed, collapsing the
      output toward zero (``from_float`` warns).
    - Only integer / tuple padding is supported; string padding raises
      ``NotImplementedError`` at ``from_float``.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.ConvTranspose1d(8, 4, 3))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(model)             # insert activation/weight observers
    >>> _ = prepared(lucid.randn(2, 8, 16))     # calibrate on representative data
    >>> qmodel = Q.convert(prepared)            # float -> quantized ConvTranspose1d
    >>> type(qmodel[0]).__name__
    'ConvTranspose1d'
    >>> qmodel(lucid.randn(2, 8, 16)).shape     # int8-weight upsampling
    (2, 4, 18)

    Constructing the layer directly is **not** a substitute for that workflow — a
    bare instance carries zeroed codes and identity qparams, so it returns garbage
    until ``from_float`` / ``load_state_dict`` populates the buffers:

    >>> broken = nn.quantized.ConvTranspose1d(
    ...     8, 4, (3,), (1,), (0,), (0,), (1,), 1, True
    ... )
    >>> bool((broken.weight_int8 == 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.ConvTranspose2d : The 2-D upsampling sibling.
    lucid.nn.quantized.Conv1d : The (down-sampling) 1-D convolution counterpart.
    lucid.quantization.convert : Installs this layer from a calibrated float model.
    """

    @override
    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.conv_transpose1d(
            x,
            weight,
            self.bias,
            cast("tuple[int]", self.stride),
            cast("tuple[int]", self.padding),
            cast("tuple[int]", self.output_padding),
            self.groups,
            cast("tuple[int]", self.dilation),
        )


class ConvTranspose2d(_QuantizedConvTransposeNd):
    r"""Quantized 2-D transposed (fractionally-strided) convolution.

    The upsampling counterpart of :class:`~lucid.nn.quantized.Conv2d`, and the
    int8-weight, dequantize-to-float replacement that
    :func:`lucid.quantization.convert` installs for a calibrated float
    :class:`~lucid.nn.ConvTranspose2d`. It is the standard decoder / up-projection
    layer of segmentation heads and generative up-samplers, and shares the sidecar
    recipe of the quantized :class:`~lucid.nn.quantized.Conv2d`.

    **Representation (sidecar design B).** The learned float kernel is quantized
    once, at :meth:`from_float` time, into an ``int8`` code tensor plus a
    per-output-channel ``scale`` / ``zero_point``; the float kernel is then
    dropped. Only the int8 codes, the qparams, and the (still-float) bias live in
    the module. Each forward *dequantizes* the kernel back to float, runs the
    ordinary ``F.conv_transpose2d``, and finally *fake-quantizes* the output onto
    the activation grid that calibration observed. Keeping the transposed
    convolution itself in float means the result matches a genuine int8 kernel to
    within a rounding step **and** the layer runs unchanged on any device.

    **Per-channel weights (axis 1).** The transposed-weight layout ``(in_channels,
    out_channels/groups, kh, kw)`` places the *output* channels on **axis 1**, not
    axis 0, so the kernel is quantized per-channel along axis 1 — each output
    channel gets its own scale, far tighter than one per-tensor scale for wide
    kernels. The scheme is **symmetric** (``qint8``, ``zero_point = 0``), matching
    the reference default for transposed weights; the bias stays float.

    Encoding happens once (at :meth:`from_float`); decode + transposed-convolve run
    on every forward:

    .. math::

        w^{q}_{c,o,i,j} = \operatorname{clamp}\!\bigl(
            \operatorname{round}(w_{c,o,i,j}/s_o),\ -128,\ 127\bigr),
        \qquad
        \hat{w}_{c,o,i,j} = w^{q}_{c,o,i,j}\, s_o

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{convT2d}(x, \hat{w}) + b\bigr),
        \qquad
        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    where :math:`s_o` is the per-output-channel (axis-1) weight scale — with a zero
    zero-point because the weight scheme is symmetric — and :math:`S, Z` the scalar
    calibrated output ``(scale, zero_point)`` with grid bounds
    :math:`q_{\min}, q_{\max}` (``0, 255`` for the default ``quint8``).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the transposed convolution — quantized
        per-channel (on axis 1 of the transposed weight).
    kernel_size : int or tuple of int
        Size of the convolving kernel.
    stride : int or tuple of int
        Stride of the transposed convolution (the upsampling factor).
    padding : int or tuple of int
        Implicit zero-padding subtracted from both sides of each spatial dim.
    output_padding : int or tuple of int
        Extra size added to one side of each output spatial dim.
    dilation : int or tuple of int
        Spacing between kernel elements.
    groups : int
        Number of blocked connections from input to output channels.
    bias : bool
        Whether a learned **float** bias term is added after the convolution.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` kernel codes, shape
        ``(in_channels, out_channels/groups, kh, kw)``.
    weight_scale : Tensor
        Per-output-channel weight scale, shape ``(out_channels/groups,)``.
    weight_zero_point : Tensor
        Per-output-channel weight zero-point, shape ``(out_channels/groups,)`` —
        all zero under the symmetric scheme.
    scale : Tensor
        Scalar output-activation scale, set from the calibrated observer.
    zero_point : Tensor
        Scalar output-activation zero-point, set from the calibrated observer.
    bias : Tensor or None
        The float bias of shape ``(out_channels,)``, or ``None``.

    Notes
    -----
    - Instances are normally produced by :func:`lucid.quantization.convert` (or
      :meth:`from_float`), **not** constructed directly: a freshly constructed
      layer has identity qparams (``scale = 1``, ``zero_point = 0``) and zeroed
      kernel codes, and only becomes meaningful once ``from_float`` /
      ``load_state_dict`` fills the buffers.
    - Only int8 codes + qparams + float bias enter the ``state_dict``; the float
      kernel never does — the weight payload shrinks ~``3.55x`` (int8), and a
      whole conv-heavy checkpoint ~``3.97x`` (measured on ``resnet_18``).
    - The kernel is quantized **per-channel on axis 1** (the output-channel axis
      of the ``(in, out/groups, kh, kw)`` layout) with a **symmetric** ``qint8``
      scheme (``from_float`` runs a fresh ``PerChannelMinMaxObserver`` over the
      weight); the bias stays float.
    - This layer wins on **memory**, not compute — the transposed convolution runs
      in float (CPU stream = Accelerate, GPU stream = MLX), so its speed tracks the
      float layer's; the payoff is the smaller checkpoint and working set.
    - Device-transparent: because compute stays in float, the layer runs unchanged
      on CPU or Metal and follows the input tensor's device.
    - Calibration is required — the output ``(scale, zero_point)`` come from an
      activation observer that must see representative data between ``prepare`` and
      ``convert``; an uncalibrated observer keeps its ``±inf`` seed, collapsing the
      output toward zero (``from_float`` warns).
    - Only integer / tuple padding is supported; string padding raises
      ``NotImplementedError`` at ``from_float``.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.ConvTranspose2d(16, 8, 2, stride=2))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(model)                # insert observers
    >>> _ = prepared(lucid.randn(1, 16, 8, 8))     # calibrate on representative data
    >>> qmodel = Q.convert(prepared)               # float -> quantized ConvTranspose2d
    >>> type(qmodel[0]).__name__
    'ConvTranspose2d'
    >>> qmodel(lucid.randn(1, 16, 8, 8)).shape     # int8-weight 2x upsampling
    (1, 8, 16, 16)

    Constructing the layer directly is **not** a substitute for that workflow — a
    bare instance carries zeroed codes and identity qparams, so it returns garbage
    until ``from_float`` / ``load_state_dict`` populates the buffers:

    >>> broken = nn.quantized.ConvTranspose2d(
    ...     16, 8, (2, 2), (2, 2), (0, 0), (0, 0), (1, 1), 1, True
    ... )
    >>> bool((broken.weight_int8 == 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.ConvTranspose1d : The 1-D upsampling sibling.
    lucid.nn.quantized.ConvTranspose3d : The volumetric upsampling sibling.
    lucid.nn.quantized.Conv2d : The (down-sampling) 2-D convolution counterpart.
    lucid.quantization.convert : Installs this layer from a calibrated float model.
    """

    @override
    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.conv_transpose2d(
            x,
            weight,
            self.bias,
            cast("tuple[int, int]", self.stride),
            cast("tuple[int, int]", self.padding),
            cast("tuple[int, int]", self.output_padding),
            self.groups,
            cast("tuple[int, int]", self.dilation),
        )


class ConvTranspose3d(_QuantizedConvTransposeNd):
    r"""Quantized 3-D transposed (fractionally-strided) convolution.

    The volumetric upsampling layer (video / 3-D medical decoders), and the
    int8-weight, dequantize-to-float replacement that
    :func:`lucid.quantization.convert` installs for a calibrated float
    :class:`~lucid.nn.ConvTranspose3d`. It is the up-projection counterpart of
    :class:`~lucid.nn.quantized.Conv3d`, and shares the sidecar recipe of the
    quantized :class:`~lucid.nn.quantized.Conv2d` family.

    **Representation (sidecar design B).** The learned float kernel is quantized
    once, at :meth:`from_float` time, into an ``int8`` code tensor plus a
    per-output-channel ``scale`` / ``zero_point``; the float kernel is then
    dropped. Only the int8 codes, the qparams, and the (still-float) bias live in
    the module. Each forward *dequantizes* the kernel back to float, runs the
    ordinary ``F.conv_transpose3d``, and finally *fake-quantizes* the output onto
    the activation grid that calibration observed. Keeping the transposed
    convolution itself in float means the result matches a genuine int8 kernel to
    within a rounding step **and** the layer runs unchanged on any device.

    **Per-channel weights (axis 1).** The transposed-weight layout ``(in_channels,
    out_channels/groups, kd, kh, kw)`` places the *output* channels on **axis 1**,
    not axis 0, so the kernel is quantized per-channel along axis 1 — each output
    channel gets its own scale, far tighter than one per-tensor scale for the wide
    dynamic range of volumetric kernels. The scheme is **symmetric** (``qint8``,
    ``zero_point = 0``), matching the reference default; the bias stays float.

    Encoding happens once (at :meth:`from_float`); decode + transposed-convolve run
    on every forward:

    .. math::

        w^{q}_{c,o,d,i,j} = \operatorname{clamp}\!\bigl(
            \operatorname{round}(w_{c,o,d,i,j}/s_o),\ -128,\ 127\bigr),
        \qquad
        \hat{w}_{c,o,d,i,j} = w^{q}_{c,o,d,i,j}\, s_o

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{convT3d}(x, \hat{w}) + b\bigr),
        \qquad
        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    where :math:`s_o` is the per-output-channel (axis-1) weight scale — with a zero
    zero-point because the weight scheme is symmetric — and :math:`S, Z` the scalar
    calibrated output ``(scale, zero_point)`` with grid bounds
    :math:`q_{\min}, q_{\max}` (``0, 255`` for the default ``quint8``).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input volume.
    out_channels : int
        Number of channels produced by the transposed convolution — quantized
        per-channel (on axis 1 of the transposed weight).
    kernel_size : int or tuple of int
        Size of the convolving kernel.
    stride : int or tuple of int
        Stride of the transposed convolution (the upsampling factor).
    padding : int or tuple of int
        Implicit zero-padding subtracted from both sides of each spatial dim.
    output_padding : int or tuple of int
        Extra size added to one side of each output spatial dim.
    dilation : int or tuple of int
        Spacing between kernel elements.
    groups : int
        Number of blocked connections from input to output channels.
    bias : bool
        Whether a learned **float** bias term is added after the convolution.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` kernel codes, shape
        ``(in_channels, out_channels/groups, kd, kh, kw)``.
    weight_scale : Tensor
        Per-output-channel weight scale, shape ``(out_channels/groups,)``.
    weight_zero_point : Tensor
        Per-output-channel weight zero-point, shape ``(out_channels/groups,)`` —
        all zero under the symmetric scheme.
    scale : Tensor
        Scalar output-activation scale, set from the calibrated observer.
    zero_point : Tensor
        Scalar output-activation zero-point, set from the calibrated observer.
    bias : Tensor or None
        The float bias of shape ``(out_channels,)``, or ``None``.

    Notes
    -----
    - Instances are normally produced by :func:`lucid.quantization.convert` (or
      :meth:`from_float`), **not** constructed directly: a freshly constructed
      layer has identity qparams (``scale = 1``, ``zero_point = 0``) and zeroed
      kernel codes, and only becomes meaningful once ``from_float`` /
      ``load_state_dict`` fills the buffers.
    - Only int8 codes + qparams + float bias enter the ``state_dict``; the float
      kernel never does — the weight payload shrinks ~``3.55x`` (int8), and a
      whole conv-heavy checkpoint ~``3.97x`` (measured on ``resnet_18``).
    - The kernel is quantized **per-channel on axis 1** (the output-channel axis
      of the ``(in, out/groups, kd, kh, kw)`` layout) with a **symmetric**
      ``qint8`` scheme (``from_float`` runs a fresh ``PerChannelMinMaxObserver``
      over the weight); the bias stays float.
    - This layer wins on **memory**, not compute — the transposed convolution runs
      in float (CPU stream = Accelerate, GPU stream = MLX), so its speed tracks the
      float layer's; the payoff is the smaller checkpoint and working set.
    - Device-transparent: because compute stays in float, the layer runs unchanged
      on CPU or Metal and follows the input tensor's device.
    - Calibration is required — the output ``(scale, zero_point)`` come from an
      activation observer that must see representative data between ``prepare`` and
      ``convert``; an uncalibrated observer keeps its ``±inf`` seed, collapsing the
      output toward zero (``from_float`` warns).
    - Only integer / tuple padding is supported; string padding raises
      ``NotImplementedError`` at ``from_float``.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.ConvTranspose3d(8, 4, 2, stride=2))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(model)                  # insert observers
    >>> _ = prepared(lucid.randn(1, 8, 4, 8, 8))     # calibrate on representative data
    >>> qmodel = Q.convert(prepared)                 # float -> quantized ConvTranspose3d
    >>> type(qmodel[0]).__name__
    'ConvTranspose3d'
    >>> qmodel(lucid.randn(1, 8, 4, 8, 8)).shape     # int8-weight 2x upsampling
    (1, 4, 8, 16, 16)

    Constructing the layer directly is **not** a substitute for that workflow — a
    bare instance carries zeroed codes and identity qparams, so it returns garbage
    until ``from_float`` / ``load_state_dict`` populates the buffers:

    >>> broken = nn.quantized.ConvTranspose3d(
    ...     8, 4, (2, 2, 2), (2, 2, 2), (0, 0, 0), (0, 0, 0), (1, 1, 1), 1, True
    ... )
    >>> bool((broken.weight_int8 == 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.ConvTranspose2d : The 2-D upsampling sibling.
    lucid.nn.quantized.Conv3d : The (down-sampling) 3-D convolution counterpart.
    lucid.quantization.convert : Installs this layer from a calibrated float model.
    """

    @override
    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.conv_transpose3d(
            x,
            weight,
            self.bias,
            cast("tuple[int, int, int]", self.stride),
            cast("tuple[int, int, int]", self.padding),
            cast("tuple[int, int, int]", self.output_padding),
            self.groups,
            cast("tuple[int, int, int]", self.dilation),
        )
