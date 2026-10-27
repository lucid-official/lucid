"""Quantized ``Linear`` — int8 weight storage, float-carried activations.

Under the sidecar representation (design B) the weight is stored as int8
codes plus per-channel ``scale`` / ``zero_point`` buffers; the forward
dequantizes the weight to ``float32``, runs the ordinary linear op, then
fake-quantizes the output to the calibrated activation grid.  This yields
the *numerics* of int8 inference (so accuracy matches a real int8 kernel)
while the actual GEMM stays in float.  The real low-precision kernel
lives beside this rather than underneath it, in
:class:`QuantizedLinearMLX`, and the two agree to 5e-07 — near enough
that the float path is the reference the Core ML export reads a packed
weight through.
"""

from typing import TYPE_CHECKING, Protocol, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import activation_qparams, quantize_weight
from lucid.quantization._functional import dequantize, fake_quantize
from lucid.quantization._qscheme import QDtype, quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    class _FloatLinear(Protocol):
        """Structural view of a calibrated float linear module."""

        in_features: int
        out_features: int
        weight: Tensor
        bias: Tensor | None
        qconfig: object


class Linear(nn.Module):
    r"""Quantized linear (fully-connected) layer — int8 weight, float compute.

    The inference-time replacement that :func:`lucid.quantization.convert` installs
    in place of a calibrated float :class:`~lucid.nn.Linear`. It is the quantized
    workhorse of every fully-connected stack — classifier heads, Transformer MLP
    blocks, projection layers — and the layer whose weight memory dominates most
    language models.

    **Representation (sidecar design B).** The learned float weight is quantized
    once, at :meth:`from_float` time, into an ``int8`` code tensor plus a
    per-output-channel ``scale`` / ``zero_point``; the float weight is then dropped.
    Only the int8 codes, the qparams, and the (still-float) bias live in the module,
    so the checkpoint is roughly ``4x`` smaller than the float layer's. Each forward
    *dequantizes* the weight back to float, runs the ordinary ``F.linear``, and
    finally *fake-quantizes* the output onto the activation grid that calibration
    observed. Keeping the matmul itself in float means the result matches a genuine
    int8 kernel to within a rounding step **and** the layer runs unchanged on any
    device, with no int8 GEMM kernel required.

    **Per-channel weights.** The weight ``(out_features, in_features)`` is quantized
    independently for every output channel (axis 0): each row carries its own scale
    and zero-point. Because different output neurons often span very different
    dynamic ranges, per-channel quantization tracks them far more tightly than a
    single per-tensor scale, and is the standard accuracy-preserving choice for
    linear and convolution kernels. The bias is small and precision-sensitive, so it
    is left in float.

    Encoding happens once (at :meth:`from_float`); decode + matmul run on every
    forward:

    .. math::

        w^{q}_{ij} = \operatorname{clamp}\!\bigl(
            \operatorname{round}(w_{ij}/s_i) + z_i,\ -128,\ 127\bigr),
        \qquad
        \hat{w}_{ij} = (w^{q}_{ij} - z_i)\, s_i

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(\hat{w}\, x^{\top} + b\bigr),
        \qquad
        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t/S) + Z,\ q_{\min},\ q_{\max})
            - Z\bigr)\, S

    where :math:`s_i, z_i` are the per-output-channel weight scale / zero-point and
    :math:`S, Z` the scalar calibrated output ``(scale, zero_point)`` with grid
    bounds :math:`q_{\min}, q_{\max}` (``0, 255`` for the default ``quint8``). The
    decode → float-matmul → re-quantize round-trip is what makes the output *bit-
    equivalent* to a true int8 kernel without needing one.

    Parameters
    ----------
    in_features : int
        Size of each input sample — the ``in`` dimension of the weight.
    out_features : int
        Size of each output sample — the ``out`` dimension, quantized per-channel.
    bias : bool, default True
        If ``True`` the layer keeps a learned **float** bias added after the matmul;
        if ``False`` no bias term is stored.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` weight codes, shape ``(out_features, in_features)``.
    weight_scale : Tensor
        Per-output-channel weight scale, shape ``(out_features,)``.
    weight_zero_point : Tensor
        Per-output-channel weight zero-point, shape ``(out_features,)``.
    scale : Tensor
        Scalar output-activation scale, set from the calibrated observer.
    zero_point : Tensor
        Scalar output-activation zero-point, set from the calibrated observer.
    bias : Tensor or None
        The float bias of shape ``(out_features,)``, or ``None``.

    Notes
    -----
    - Instances are normally produced by :func:`lucid.quantization.convert` (or
      :meth:`from_float`), **not** constructed directly: a freshly constructed layer
      has identity qparams (``scale = 1``, ``zero_point = 0``) and zeroed weights,
      and only becomes meaningful once ``from_float`` / ``load_state_dict`` fills the
      buffers.
    - Only int8 codes + qparams + float bias enter the ``state_dict``; the float
      weight never does. In practice the weight payload shrinks ~``3.55x`` (int8)
      and a whole model checkpoint ~``3.97x`` (measured on ``resnet_18``).
    - This layer wins on **memory**, not compute — the GEMM runs in float, so on the
      CPU stream it is roughly float-speed. For the genuine low-precision *compute*
      speed-up on Metal, convert with the MLX backend active so the Linear family is
      routed to :class:`~lucid.nn.quantized.QuantizedLinearMLX`, which runs a real
      weight-only int4/int8 GEMM (skipping the dequantize hop) — measured ~``3.15x``
      faster at the memory-bound decode shape ``M = 1`` on a fully-Metal model. That
      win is bandwidth-bound: it lands in the inference / generation regime and
      fades toward parity (~``0.9-1x``) in compute-bound training GEMMs.
    - ``_activation`` is the identity here; the fused
      :class:`~lucid.nn.quantized.LinearReLU` overrides it to apply ReLU *before* the
      output fake-quant so calibration and inference see the same post-ReLU range.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Linear(128, 64))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(model)          # insert activation/weight observers
    >>> _ = prepared(lucid.randn(32, 128))   # calibrate on representative data
    >>> qmodel = Q.convert(prepared)         # float Linear -> quantized layer
    >>> type(qmodel[0]).__name__             # MLX kernels built in: the int8 GEMM
    'QuantizedLinearMLX'
    >>> qlin = nn.quantized.Linear.from_float(prepared[0])   # this layer, explicitly
    >>> type(qlin).__name__
    'Linear'
    >>> y = qlin(lucid.randn(1, 128))        # int8-weight inference
    >>> y.shape
    (1, 64)

    Constructing the layer directly is **not** a substitute for that workflow — a
    bare instance carries zeroed weights and identity qparams, so it returns garbage
    until ``from_float`` / ``load_state_dict`` populates the buffers:

    >>> broken = nn.quantized.Linear(128, 64)     # zeroed weight, scale=1, zp=0
    >>> bool((broken.weight_int8 == 0).all().item())
    True

    For the *compute* speed-up on Metal, route through the real low-precision GEMM:

    >>> src = nn.Linear(512, 512)
    >>> qlin = nn.quantized.QuantizedLinearMLX.from_float(src, bits=8)
    >>> qlin(lucid.randn(1, 512).to("metal")).shape
    (1, 512)

    See Also
    --------
    lucid.nn.quantized.QuantizedLinearMLX : Real Metal int4/int8 GEMM (compute win).
    lucid.nn.quantized.LinearReLU : Quantized Linear with a fused ReLU.
    lucid.nn.quantized.Conv2d : The convolutional analogue (per-channel int8 weight).
    lucid.quantization.convert : Installs this layer from a calibrated float model.
    """

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor
    scale: Tensor
    zero_point: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_ch_axis = 0
        self.out_qdtype: QDtype = quint8
        self.register_buffer(
            "weight_int8", lucid.zeros((out_features, in_features), dtype=lucid.int8)
        )
        self.register_buffer("weight_scale", lucid.ones(out_features))
        self.register_buffer("weight_zero_point", lucid.zeros(out_features))
        if bias:
            self.register_buffer("bias", lucid.zeros(out_features))
        else:
            self.bias = None
        self.register_buffer("scale", lucid.tensor(1.0))
        self.register_buffer("zero_point", lucid.tensor(0.0))

    def _activation(self, y: Tensor) -> Tensor:
        """Post-linear activation hook (identity; ReLU in the fused variant)."""
        return y

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Dequantize the weight, run linear, fake-quantize the output."""
        weight = dequantize(
            self.weight_int8, self.weight_scale, self.weight_zero_point, ch_axis=self.weight_ch_axis
        )
        y = self._activation(F.linear(x, weight, self.bias))
        return fake_quantize(
            y, self.scale, self.zero_point, self.out_qdtype.quant_min, self.out_qdtype.quant_max
        )

    @classmethod
    def from_float(cls, mod: nn.Module) -> Linear:
        """Quantize a calibrated float :class:`~lucid.nn.Linear`."""
        f = cast("_FloatLinear", mod)
        has_bias = f.bias is not None
        qmod = cls(f.in_features, f.out_features, bias=has_bias)

        codes, w_scale, w_zp, ch_axis = quantize_weight(mod)
        qmod.register_buffer("weight_int8", codes)
        qmod.register_buffer("weight_scale", w_scale)
        qmod.register_buffer("weight_zero_point", w_zp)
        qmod.weight_ch_axis = ch_axis
        if f.bias is not None:
            qmod.register_buffer("bias", f.bias.detach())  # bias stays float

        a_scale, a_zp, a_qdtype = activation_qparams(mod)
        qmod.register_buffer("scale", a_scale)
        qmod.register_buffer("zero_point", a_zp)
        qmod.out_qdtype = a_qdtype
        return qmod

    @override
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"qdtype={self.out_qdtype.name}"
        )
