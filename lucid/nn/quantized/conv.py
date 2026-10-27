"""Quantized ``Conv1d`` / ``Conv2d`` / ``Conv3d`` — int8 weights, float compute.

Same sidecar recipe as the quantized :class:`~lucid.nn.quantized.Linear`:
the kernel is stored as int8 codes with per-output-channel ``scale`` /
``zero_point``; the forward dequantizes it, runs the ordinary convolution,
and fake-quantizes the output to the calibrated activation grid.  Padding
follows the float module it came from: integer / tuple, the ``"same"`` /
``"valid"`` strings, and every ``padding_mode`` — all of which the float
conv resolves per input, so the quantized one carries the spec rather than
a baked-in number.
"""

from typing import TYPE_CHECKING, ClassVar, Protocol, cast, override

import lucid
import lucid.nn as nn
from lucid.nn.functional.conv import _ConvArgs, _convnd
from lucid._types import PaddingMode
from lucid.nn.quantized._utils import activation_qparams, quantize_weight
from lucid.quantization._functional import dequantize, fake_quantize
from lucid.quantization._qscheme import QDtype, quint8

# Outside ``TYPE_CHECKING``: the constructors' signatures name it.
type _IntTuple = tuple[int, ...]

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    class _FloatConv(Protocol):
        """Structural view of a calibrated float conv module."""

        in_channels: int
        out_channels: int
        kernel_size: _IntTuple
        stride: _IntTuple
        # Never a string.  ``Conv*d.__init__`` splits a string spec off into
        # ``_padding_str`` and leaves this at 0 — declaring ``| str`` here is
        # what let a dead ``isinstance(f.padding, str)`` guard look correct.
        padding: _IntTuple
        _padding_str: str | None
        padding_mode: PaddingMode
        dilation: _IntTuple
        groups: int
        weight: Tensor
        bias: Tensor | None
        qconfig: object


def _as_tuple(value: object, rank: int) -> tuple[int, ...]:
    """Normalise an int-or-tuple conv argument to a ``rank``-long tuple.

    ``Conv1d`` stores these as bare ints while 2d/3d store tuples, so the
    shared padding path has to accept either.
    """
    if isinstance(value, int):
        return (value,) * rank
    out = tuple(int(v) for v in cast("tuple[int, ...]", value))
    return out if len(out) == rank else (out[0],) * rank


class _QuantizedConvNd(nn.Module):
    """Shared implementation for the quantized convolution family."""

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor
    scale: Tensor
    zero_point: Tensor

    #: Spatial rank of the convolution (1, 2 or 3), set by each subclass.
    _rank: ClassVar[int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _IntTuple,
        stride: _IntTuple,
        padding: _IntTuple,
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
        self.dilation = dilation
        self.groups = groups
        # ``from_float`` fills these in from the float module.  They default
        # to the plain case so a directly-constructed quantized conv behaves
        # as it always did.
        self.padding_str: str | None = None
        self.padding_mode: PaddingMode = "zeros"
        self.weight_ch_axis = 0
        self.out_qdtype: QDtype = quint8
        # ``nn.Conv1d`` stores ``kernel_size`` as a bare int; the 2d/3d convs
        # store tuples, and a direct constructor call may pass either.  Expand
        # to this conv's rank so the weight buffer has every spatial axis.
        ks = _as_tuple(kernel_size, self._rank)
        weight_shape = (out_channels, in_channels // groups, *ks)
        self.register_buffer("weight_int8", lucid.zeros(weight_shape, dtype=lucid.int8))
        self.register_buffer("weight_scale", lucid.ones(out_channels))
        self.register_buffer("weight_zero_point", lucid.zeros(out_channels))
        if bias:
            self.register_buffer("bias", lucid.zeros(out_channels))
        else:
            self.bias = None
        self.register_buffer("scale", lucid.tensor(1.0))
        self.register_buffer("zero_point", lucid.tensor(0.0))

    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        """Convolve as the float module would, ``"same"`` and ``padding_mode``
        included — ``"same"`` depends on the input's spatial size, which is
        why the module carries the string rather than a baked-in number."""
        args = _ConvArgs(self.stride, self.padding_str or self.padding, self.dilation, self.groups)
        return _convnd(x, weight, self.bias, args, self._rank, self.padding_mode)

    def _activation(self, y: Tensor) -> Tensor:
        """Post-conv activation hook (identity; ReLU in the fused variant)."""
        return y

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Dequantize the kernel, convolve, fake-quantize the output."""
        weight = dequantize(
            self.weight_int8, self.weight_scale, self.weight_zero_point, ch_axis=self.weight_ch_axis
        )
        y = self._activation(self._conv_forward(input, weight))
        return fake_quantize(
            y, self.scale, self.zero_point, self.out_qdtype.quant_min, self.out_qdtype.quant_max
        )

    @classmethod
    def from_float(cls, mod: nn.Module) -> _QuantizedConvNd:
        """Quantize a calibrated float convolution module."""
        f = cast("_FloatConv", mod)
        has_bias = f.bias is not None
        qmod = cls(
            f.in_channels,
            f.out_channels,
            f.kernel_size,
            f.stride,
            f.padding,
            f.dilation,
            f.groups,
            bias=has_bias,
        )
        codes, w_scale, w_zp, ch_axis = quantize_weight(mod)
        qmod.register_buffer("weight_int8", codes)
        qmod.register_buffer("weight_scale", w_scale)
        qmod.register_buffer("weight_zero_point", w_zp)
        qmod.weight_ch_axis = ch_axis
        # ``Conv*d`` moves a string padding into ``_padding_str`` and leaves
        # ``padding`` at 0, so the old ``isinstance(f.padding, str)`` guard
        # could never fire: a ``padding="same"`` conv was quantized as
        # ``"valid"`` and simply came out a different shape.  ``padding_mode``
        # was never looked at at all, which was quieter still — a reflect-
        # padded conv quantized to a zeros-padded one with no shape change to
        # give it away.  Carry both.
        qmod.padding_str = getattr(f, "_padding_str", None)
        qmod.padding_mode = getattr(f, "padding_mode", "zeros")
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


class Conv1d(_QuantizedConvNd):
    r"""Quantized 1-D convolution — int8 kernel, dequantize-to-float compute.

    The inference-time replacement that :func:`lucid.quantization.convert`
    installs in place of a calibrated float :class:`~lucid.nn.Conv1d`. It is the
    quantized member of the 1-D conv family — audio / time-series front ends,
    1-D residual stacks, token-mixing convolutions — and shares the exact sidecar
    recipe of the quantized :class:`~lucid.nn.quantized.Linear`.

    **Representation (sidecar design B).** The learned float kernel is quantized
    once, at :meth:`from_float` time, into an ``int8`` code tensor plus a
    per-output-channel ``scale`` / ``zero_point``; the float kernel is then
    dropped. Only the int8 codes, the qparams, and the (still-float) bias live in
    the module, so the checkpoint is far smaller than the float layer's. Each
    forward *dequantizes* the kernel back to float, runs the ordinary
    ``F.conv1d``, and finally *fake-quantizes* the output onto the activation grid
    that calibration observed. Keeping the convolution itself in float means the
    result matches a genuine int8 kernel to within a rounding step **and** the
    layer runs unchanged on any device, with no int8 conv kernel required.

    **Per-channel weights.** The kernel ``(out_channels, in_channels/groups, k)``
    is quantized independently for every output channel (axis 0): each filter
    carries its own scale and zero-point. Because different output filters often
    span very different dynamic ranges, per-channel quantization tracks them far
    more tightly than a single per-tensor scale, and is the standard
    accuracy-preserving choice for convolution kernels. The bias is small and
    precision-sensitive, so it is left in float.

    Encoding happens once (at :meth:`from_float`); decode + convolve run on every
    forward:

    .. math::

        w^{q}_{o,c,k} = \operatorname{clamp}\!\bigl(
            \operatorname{round}(w_{o,c,k}/s_o) + z_o,\ -128,\ 127\bigr),
        \qquad
        \hat{w}_{o,c,k} = (w^{q}_{o,c,k} - z_o)\, s_o

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{conv1d}(x, \hat{w}) + b\bigr),
        \qquad
        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    where :math:`s_o, z_o` are the per-output-channel weight scale / zero-point
    (index :math:`o` runs over ``out_channels``) and :math:`S, Z` the scalar
    calibrated output ``(scale, zero_point)`` with grid bounds
    :math:`q_{\min}, q_{\max}` (``0, 255`` for the default ``quint8``). The
    decode → float-conv → re-quantize round-trip is what makes the output
    *bit-equivalent* to a true int8 kernel without needing one.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input signal.
    out_channels : int
        Number of channels produced by the convolution — quantized per-channel.
    kernel_size : int or tuple of int
        Size of the convolving kernel.
    stride : int or tuple of int
        Stride of the convolution.
    padding : int or tuple of int
        Implicit zero-padding added to both sides of the spatial axis.
    dilation : int or tuple of int
        Spacing between kernel elements.
    groups : int
        Number of blocked connections from input to output channels.
    bias : bool
        Whether a learned **float** bias term is added after the convolution.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` kernel codes, shape ``(out_channels, in_channels/groups, k)``.
    weight_scale : Tensor
        Per-output-channel weight scale, shape ``(out_channels,)``.
    weight_zero_point : Tensor
        Per-output-channel weight zero-point, shape ``(out_channels,)``.
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
      kernel never does. The weight payload shrinks ~``3.55x`` (int8) and a whole
      model checkpoint ~``3.97x`` (measured on ``resnet_18``, a conv-heavy net).
    - The kernel is quantized **per-output-channel on axis 0** — each filter gets
      its own scale / zero-point, far tighter than one per-tensor scale for wide
      kernels; the bias stays float.
    - This layer wins on **memory**, not compute — the convolution runs in float
      (CPU stream = Accelerate, GPU stream = MLX), so its speed tracks the float
      layer's. There is no weight-only int8 *conv* GEMM analogue of
      :class:`~lucid.nn.quantized.QuantizedLinearMLX`; the payoff is the smaller
      checkpoint and working set.
    - Device-transparent: because compute stays in float, the layer runs
      unchanged on CPU or Metal and follows the input tensor's device.
    - Calibration is required — the output ``(scale, zero_point)`` come from an
      activation observer that must see representative data between ``prepare``
      and ``convert``; an uncalibrated observer keeps its ``±inf`` seed,
      collapsing ``scale`` toward ``eps`` and the output toward zero
      (``from_float`` warns loudly in that case).
    - Only integer / tuple padding with ``padding_mode="zeros"`` is supported;
      string ``"same"`` / ``"valid"`` padding raises ``NotImplementedError`` at
      ``from_float``.
    - ``_activation`` is the identity here; the fused
      :class:`~lucid.nn.quantized.ConvReLU1d` overrides it to apply ReLU *before*
      the output fake-quant so calibration and inference see the same post-ReLU
      range.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Conv1d(8, 16, 3))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(model)             # insert activation/weight observers
    >>> _ = prepared(lucid.randn(4, 8, 32))     # calibrate on representative data
    >>> qmodel = Q.convert(prepared)            # float Conv1d -> quantized Conv1d
    >>> type(qmodel[0]).__name__
    'Conv1d'
    >>> qmodel(lucid.randn(4, 8, 32)).shape     # int8-weight inference
    (4, 16, 30)

    Constructing the layer directly is **not** a substitute for that workflow — a
    bare instance carries zeroed codes and identity qparams, so it returns garbage
    until ``from_float`` / ``load_state_dict`` populates the buffers:

    >>> broken = nn.quantized.Conv1d(8, 16, (3,), (1,), (0,), (1,), 1, True)
    >>> bool((broken.weight_int8 == 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.Conv2d : The 2-D workhorse (per-channel int8 kernel).
    lucid.nn.quantized.ConvReLU1d : Quantized 1-D conv with a fused ReLU.
    lucid.nn.quantized.Linear : The fully-connected analogue (per-channel int8).
    lucid.quantization.convert : Installs this layer from a calibrated float model.
    """

    _rank = 1


class Conv2d(_QuantizedConvNd):
    r"""Quantized 2-D convolution — int8 kernel, dequantize-to-float compute.

    The workhorse of the quantized vision zoo — the inference-time replacement
    that :func:`lucid.quantization.convert` installs for a calibrated float
    :class:`~lucid.nn.Conv2d`. Backbones such as ``resnet_18`` are almost
    entirely 2-D convolutions, so this layer is where the checkpoint / working-set
    savings of int8 quantization are actually realized.

    **Representation (sidecar design B).** The learned float kernel is quantized
    once, at :meth:`from_float` time, into an ``int8`` code tensor plus a
    per-output-channel ``scale`` / ``zero_point``; the float kernel is then
    dropped. Only the int8 codes, the qparams, and the (still-float) bias live in
    the module, so the checkpoint is far smaller than the float layer's. Each
    forward *dequantizes* the kernel back to float, runs the ordinary
    ``F.conv2d``, and finally *fake-quantizes* the output onto the activation grid
    that calibration observed. Keeping the convolution itself in float means the
    result matches a genuine int8 kernel to within a rounding step **and** the
    layer runs unchanged on any device, with no int8 conv kernel required.

    **Per-channel weights.** The kernel ``(out_channels, in_channels/groups, kh,
    kw)`` is quantized independently for every output channel (axis 0): each
    filter carries its own scale and zero-point. Because different output filters
    often span very different dynamic ranges, per-channel quantization tracks them
    far more tightly than a single per-tensor scale, and is the standard
    accuracy-preserving choice for convolution kernels. The bias is small and
    precision-sensitive, so it is left in float.

    Encoding happens once (at :meth:`from_float`); decode + convolve run on every
    forward:

    .. math::

        w^{q}_{o,c,i,j} = \operatorname{clamp}\!\bigl(
            \operatorname{round}(w_{o,c,i,j}/s_o) + z_o,\ -128,\ 127\bigr),
        \qquad
        \hat{w}_{o,c,i,j} = (w^{q}_{o,c,i,j} - z_o)\, s_o

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{conv2d}(x, \hat{w}) + b\bigr),
        \qquad
        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    where :math:`s_o, z_o` are the per-output-channel weight scale / zero-point
    (index :math:`o` runs over ``out_channels``) and :math:`S, Z` the scalar
    calibrated output ``(scale, zero_point)`` with grid bounds
    :math:`q_{\min}, q_{\max}` (``0, 255`` for the default ``quint8``). The
    decode → float-conv → re-quantize round-trip is what makes the output
    *bit-equivalent* to a true int8 kernel without needing one.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution — quantized per-channel.
    kernel_size : int or tuple of int
        Size of the convolving kernel.
    stride : int or tuple of int
        Stride of the convolution.
    padding : int or tuple of int
        Implicit zero-padding added to both sides of each spatial dim.
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
        ``(out_channels, in_channels/groups, kh, kw)``.
    weight_scale : Tensor
        Per-output-channel weight scale, shape ``(out_channels,)``.
    weight_zero_point : Tensor
        Per-output-channel weight zero-point, shape ``(out_channels,)``.
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
      kernel never does. The weight payload shrinks ~``3.55x`` (int8) and a whole
      model checkpoint ~``3.97x`` (measured on ``resnet_18``, a conv-heavy net).
    - The kernel is quantized **per-output-channel on axis 0** — each filter gets
      its own scale / zero-point, far tighter than one per-tensor scale for wide
      kernels; the bias stays float.
    - This layer wins on **memory**, not compute — the convolution runs in float
      (CPU stream = Accelerate, GPU stream = MLX), so its speed tracks the float
      layer's. There is no weight-only int8 *conv* GEMM analogue of
      :class:`~lucid.nn.quantized.QuantizedLinearMLX`; the payoff is the smaller
      checkpoint and working set.
    - Device-transparent: because compute stays in float, the layer runs
      unchanged on CPU or Metal and follows the input tensor's device.
    - Calibration is required — the output ``(scale, zero_point)`` come from an
      activation observer that must see representative data between ``prepare``
      and ``convert``; an uncalibrated observer keeps its ``±inf`` seed,
      collapsing ``scale`` toward ``eps`` and the output toward zero
      (``from_float`` warns loudly in that case).
    - Only integer / tuple padding with ``padding_mode="zeros"`` is supported;
      string ``"same"`` / ``"valid"`` padding raises ``NotImplementedError`` at
      ``from_float``.
    - ``_activation`` is the identity here; the fused
      :class:`~lucid.nn.quantized.ConvReLU2d` overrides it to apply ReLU *before*
      the output fake-quant so calibration and inference see the same post-ReLU
      range.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(model)               # insert activation/weight observers
    >>> _ = prepared(lucid.randn(8, 3, 32, 32))   # calibrate on representative data
    >>> qmodel = Q.convert(prepared)              # float Conv2d -> quantized Conv2d
    >>> type(qmodel[0]).__name__
    'Conv2d'
    >>> qmodel(lucid.randn(1, 3, 32, 32)).shape   # int8-weight inference
    (1, 16, 32, 32)

    Constructing the layer directly is **not** a substitute for that workflow — a
    bare instance carries zeroed codes and identity qparams, so it returns garbage
    until ``from_float`` / ``load_state_dict`` populates the buffers:

    >>> broken = nn.quantized.Conv2d(3, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True)
    >>> bool((broken.weight_int8 == 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.Conv1d : The 1-D sibling (per-channel int8 kernel).
    lucid.nn.quantized.Conv3d : The volumetric sibling (per-channel int8 kernel).
    lucid.nn.quantized.ConvReLU2d : Quantized 2-D conv with a fused ReLU.
    lucid.quantization.convert : Installs this layer from a calibrated float model.
    """

    _rank = 2


class Conv3d(_QuantizedConvNd):
    r"""Quantized 3-D convolution — int8 kernel, dequantize-to-float compute.

    The volumetric member of the quantized conv family (video, 3-D medical
    imaging), and the inference-time replacement that
    :func:`lucid.quantization.convert` installs for a calibrated float
    :class:`~lucid.nn.Conv3d`. Volumetric kernels are the largest weights in a
    3-D net, so the int8 checkpoint / working-set savings matter most here.

    **Representation (sidecar design B).** The learned float kernel is quantized
    once, at :meth:`from_float` time, into an ``int8`` code tensor plus a
    per-output-channel ``scale`` / ``zero_point``; the float kernel is then
    dropped. Only the int8 codes, the qparams, and the (still-float) bias live in
    the module, so the checkpoint is far smaller than the float layer's. Each
    forward *dequantizes* the kernel back to float, runs the ordinary
    ``F.conv3d``, and finally *fake-quantizes* the output onto the activation grid
    that calibration observed. Keeping the convolution itself in float means the
    result matches a genuine int8 kernel to within a rounding step **and** the
    layer runs unchanged on any device, with no int8 conv kernel required.

    **Per-channel weights.** The kernel ``(out_channels, in_channels/groups, kd,
    kh, kw)`` is quantized independently for every output channel (axis 0): each
    filter carries its own scale and zero-point. Because different output filters
    often span very different dynamic ranges, per-channel quantization tracks them
    far more tightly than a single per-tensor scale, and is the standard
    accuracy-preserving choice for convolution kernels. The bias is small and
    precision-sensitive, so it is left in float.

    Encoding happens once (at :meth:`from_float`); decode + convolve run on every
    forward:

    .. math::

        w^{q}_{o,c,d,i,j} = \operatorname{clamp}\!\bigl(
            \operatorname{round}(w_{o,c,d,i,j}/s_o) + z_o,\ -128,\ 127\bigr),
        \qquad
        \hat{w}_{o,c,d,i,j} = (w^{q}_{o,c,d,i,j} - z_o)\, s_o

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{conv3d}(x, \hat{w}) + b\bigr),
        \qquad
        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    where :math:`s_o, z_o` are the per-output-channel weight scale / zero-point
    (index :math:`o` runs over ``out_channels``) and :math:`S, Z` the scalar
    calibrated output ``(scale, zero_point)`` with grid bounds
    :math:`q_{\min}, q_{\max}` (``0, 255`` for the default ``quint8``). The
    decode → float-conv → re-quantize round-trip is what makes the output
    *bit-equivalent* to a true int8 kernel without needing one.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input volume.
    out_channels : int
        Number of channels produced by the convolution — quantized per-channel.
    kernel_size : int or tuple of int
        Size of the convolving kernel.
    stride : int or tuple of int
        Stride of the convolution.
    padding : int or tuple of int
        Implicit zero-padding added to both sides of each spatial dim.
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
        ``(out_channels, in_channels/groups, kd, kh, kw)``.
    weight_scale : Tensor
        Per-output-channel weight scale, shape ``(out_channels,)``.
    weight_zero_point : Tensor
        Per-output-channel weight zero-point, shape ``(out_channels,)``.
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
      kernel never does. The weight payload shrinks ~``3.55x`` (int8) and a whole
      model checkpoint ~``3.97x`` (measured on ``resnet_18``, a conv-heavy net).
    - The kernel is quantized **per-output-channel on axis 0** — each filter gets
      its own scale / zero-point, far tighter than one per-tensor scale for wide
      kernels; the bias stays float.
    - This layer wins on **memory**, not compute — the convolution runs in float
      (CPU stream = Accelerate, GPU stream = MLX), so its speed tracks the float
      layer's. There is no weight-only int8 *conv* GEMM analogue of
      :class:`~lucid.nn.quantized.QuantizedLinearMLX`; the payoff is the smaller
      checkpoint and working set.
    - Device-transparent: because compute stays in float, the layer runs
      unchanged on CPU or Metal and follows the input tensor's device.
    - Calibration is required — the output ``(scale, zero_point)`` come from an
      activation observer that must see representative data between ``prepare``
      and ``convert``; an uncalibrated observer keeps its ``±inf`` seed,
      collapsing ``scale`` toward ``eps`` and the output toward zero
      (``from_float`` warns loudly in that case).
    - Only integer / tuple padding with ``padding_mode="zeros"`` is supported;
      string ``"same"`` / ``"valid"`` padding raises ``NotImplementedError`` at
      ``from_float``.
    - ``_activation`` is the identity here; the fused
      :class:`~lucid.nn.quantized.ConvReLU3d` overrides it to apply ReLU *before*
      the output fake-quant so calibration and inference see the same post-ReLU
      range.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Conv3d(3, 8, 3))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(model)                  # insert observers
    >>> _ = prepared(lucid.randn(1, 3, 8, 16, 16))   # calibrate on representative data
    >>> qmodel = Q.convert(prepared)                 # float Conv3d -> quantized Conv3d
    >>> type(qmodel[0]).__name__
    'Conv3d'
    >>> qmodel(lucid.randn(1, 3, 8, 16, 16)).shape   # int8-weight inference
    (1, 8, 6, 14, 14)

    Constructing the layer directly is **not** a substitute for that workflow — a
    bare instance carries zeroed codes and identity qparams, so it returns garbage
    until ``from_float`` / ``load_state_dict`` populates the buffers:

    >>> broken = nn.quantized.Conv3d(3, 8, (3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, True)
    >>> bool((broken.weight_int8 == 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.Conv2d : The 2-D workhorse (per-channel int8 kernel).
    lucid.nn.quantized.Conv1d : The 1-D sibling (per-channel int8 kernel).
    lucid.nn.quantized.ConvReLU3d : Quantized 3-D conv with a fused ReLU.
    lucid.quantization.convert : Installs this layer from a calibrated float model.
    """

    _rank = 3
