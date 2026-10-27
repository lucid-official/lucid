"""Dynamically quantized ``Linear``.

The weight is stored as int8 (per-output-channel symmetric); on every
forward the input's range is measured and the input is fake-quantized to a
per-tensor affine grid, then the ordinary linear op runs in float and the
output is left in ``float32``.  No calibration pass is required — the
activation grid is derived from the live input, which is what makes dynamic
quantization attractive for Transformer / Linear-heavy inference.
"""

from typing import TYPE_CHECKING, Protocol, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.quantization._functional import dequantize, fake_quantize, quantize
from lucid.quantization._qparams import calculate_qparams
from lucid.quantization._qscheme import (
    QDtype,
    per_channel_symmetric,
    per_tensor_affine,
    qint8,
    quint8,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    class _FloatLinear(Protocol):
        in_features: int
        out_features: int
        weight: Tensor
        bias: Tensor | None


class Linear(nn.Module):
    """int8-weight linear with per-forward dynamic activation quantization.

    Dynamic quantization keeps the weight permanently in int8 (per-output-channel
    symmetric) while the activation grid is derived from the *live* input on every
    forward — the input's min/max are measured, it is fake-quantized to a per-tensor
    affine grid, and the ordinary linear op then runs in float with the dequantized
    weight.  Because no representative data is needed to fix the activation range,
    this layer needs no calibration pass, which makes it the standard choice for
    Transformer / Linear-heavy inference (BERT and friends).  Instances are produced
    by :func:`lucid.quantization.quantize_dynamic`, not constructed directly.

    Parameters
    ----------
    in_features : int
        Size of each input sample — the weight's input dimension.
    out_features : int
        Size of each output sample — the weight's output dimension.
    bias : bool, default True
        If ``True``, keep a learned float bias buffer; otherwise no bias is added.

    Notes
    -----
    The weight lives in the ``weight_int8`` buffer alongside per-channel
    ``weight_scale`` / ``weight_zero_point``; only the int8 form is the persistent,
    device-tracked state, and the float weight is reconstructed on demand.
    """

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_ch_axis = 0
        self.act_qdtype: QDtype = quint8
        self.register_buffer(
            "weight_int8", lucid.zeros((out_features, in_features), dtype=lucid.int8)
        )
        self.register_buffer("weight_scale", lucid.ones(out_features))
        self.register_buffer("weight_zero_point", lucid.zeros(out_features))
        if bias:
            self.register_buffer("bias", lucid.zeros(out_features))
        else:
            self.bias = None

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Dynamically quantize the input, then run linear with the int8 weight."""
        x_scale, x_zp = calculate_qparams(x.min(), x.max(), per_tensor_affine, self.act_qdtype)
        x_q = fake_quantize(x, x_scale, x_zp, self.act_qdtype.quant_min, self.act_qdtype.quant_max)
        weight = dequantize(
            self.weight_int8,
            self.weight_scale,
            self.weight_zero_point,
            ch_axis=self.weight_ch_axis,
        )
        return F.linear(x_q, weight, self.bias)

    @classmethod
    def from_float(cls, mod: nn.Module, dtype: QDtype = qint8) -> Linear:
        """Quantize a float :class:`~lucid.nn.Linear`'s weight (no calibration)."""
        from lucid.quantization.observer import PerChannelMinMaxObserver

        f = cast("_FloatLinear", mod)
        qmod = cls(f.in_features, f.out_features, bias=f.bias is not None)
        obs = PerChannelMinMaxObserver(ch_axis=0, qscheme=per_channel_symmetric, qdtype=dtype)
        obs(f.weight)
        scale, zero_point = obs.calculate_qparams()
        qmod.register_buffer("weight_int8", quantize(f.weight, scale, zero_point, dtype, ch_axis=0))
        qmod.register_buffer("weight_scale", scale)
        qmod.register_buffer("weight_zero_point", zero_point)
        if f.bias is not None:
            qmod.register_buffer("bias", f.bias.detach())
        return qmod

    @override
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, dynamic=True"
