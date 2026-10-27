"""QAT ``Linear`` — trains in float with weight + activation fake-quant.

During quantization-aware training the layer keeps a **trainable float
weight** but applies :class:`~lucid.quantization.FakeQuantize` to both the
weight and the output, so the network experiences quantization rounding
(via the straight-through estimator) while learning to compensate for it.
``convert`` later reads the trained weight + the fake-quant observers' final
qparams to build the quantized inference :class:`~lucid.nn.quantized.Linear`.
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig


class Linear(nn.Linear):
    r"""Quantization-aware ``Linear`` — trainable float weight, fake-quant every forward.

    The training-time stand-in that :func:`lucid.quantization.prepare_qat` installs in
    place of a float :class:`~lucid.nn.Linear`. It keeps a **trainable float weight** but
    routes both the weight and the layer output through a
    :class:`~lucid.quantization.FakeQuantize` on every forward, so the network *feels* the
    int8 rounding error while it is still learning and can adapt its weights to
    compensate. This is what closes the train / inference accuracy gap: plain
    post-training quantization perturbs a network that never saw rounding, whereas a
    QAT-trained network has already minimised its loss *with* the rounding baked in, so its
    accuracy barely moves when the weights are finally frozen to int8.

    **Straight-through estimator (STE).** Rounding is a step function; its derivative is
    zero almost everywhere, so a literal ``round`` would block all gradient flow. The
    fake-quant therefore rounds in the forward pass but, in the backward pass, pretends it
    was the identity — the incoming gradient passes *straight through* to the underlying
    float weight (clipped to the quantization range). The float weight thus stays fully
    trainable while every forward value it produces is the *dequantized* number an int8
    kernel would have computed.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` deep-copies the float model
    and swaps each :class:`~lucid.nn.Linear` for this class, attaching the weight and
    activation ``FakeQuantize`` modules described by the layer's ``qconfig``. After
    training, :func:`lucid.quantization.convert` reads the trained weight together with the
    activation observer's final ``(scale, zero_point)`` and bakes them into an inference
    :class:`~lucid.nn.quantized.Linear` whose weight is stored as true int8 codes.

    On each forward the weight is fake-quantized, the ordinary ``F.linear`` runs, and the
    result is fake-quantized onto the observed activation grid:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(W)\, x^{\top} + b\bigr)

    where :math:`s, z` are the scale / zero-point the enclosing ``FakeQuantize`` derives
    from its observer and :math:`q_{\min}, q_{\max}` are the grid bounds (``0, 255`` for
    the default ``quint8`` activation, ``-128, 127`` for the ``qint8`` weight). The
    straight-through unit derivative is what keeps the layer differentiable despite the
    non-differentiable ``round``.

    Parameters
    ----------
    in_features : int
        Size of each input sample — forwarded verbatim to the float
        :class:`~lucid.nn.Linear` parent.
    out_features : int
        Size of each output sample — forwarded verbatim to the float parent.
    bias : bool, default=True
        Whether to add a learnable (float) bias term; forwarded to the float parent. The
        bias is left in float and never fake-quantized.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant built from ``qconfig.weight()``; rounds the float
        weight each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant built from ``qconfig.activation()``; calibrates
        the activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **STE differentiability.** ``round`` is applied in the forward pass but the gradient
      is passed through as if it were the identity, so the float weight trains normally —
      no gradient is lost to the non-differentiable rounding step.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` swaps
      the float layer *in*; :func:`lucid.quantization.convert` folds this layer *out* into
      the matching :class:`~lucid.nn.quantized.Linear`. Manual construction is rarely
      needed and requires an explicit ``qconfig``.
    - **Training is slower, not faster.** Only the *numerics* of int8 are simulated; the
      matmul itself still runs in float and each forward carries two extra fake-quant ops,
      so a QAT model runs *slower* than the float baseline. The speed / memory win arrives
      only after ``convert`` replaces this layer with the int8 inference module.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Linear(16, 4))
    >>> qat = Q.prepare_qat(model)              # nn.Linear -> qat.Linear
    >>> type(qat[0]).__name__
    'Linear'
    >>> loss = (qat(lucid.randn(8, 16)) ** 2).mean()
    >>> loss.backward()                         # STE routes grads to the float weight
    >>> _ = qat.eval()
    >>> qmodel = Q.convert(qat)                 # qat.Linear -> quantized.Linear (int8)
    >>> type(qmodel[0]).__name__
    'Linear'

    A ``qconfig`` is mandatory — constructing the layer directly without one raises:

    >>> import lucid.nn.qat as nnqat
    >>> nnqat.Linear(8, 8)
    Traceback (most recent call last):
        ...
    ValueError: qat.Linear requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Linear : The int8 inference layer ``convert`` bakes this into.
    lucid.nn.qat.LinearReLU : QAT ``Linear`` with a fused ReLU before the output observer.
    lucid.quantization.prepare_qat : Swaps the float ``Linear`` for this QAT layer.
    lucid.quantization.convert : Bakes the trained QAT layer into the quantized layer.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(in_features, out_features, bias)
        if qconfig is None:
            raise ValueError("qat.Linear requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Fake-quantize the weight, run linear, fake-quantize the output."""
        w_q = self.weight_fake_quant(self.weight)
        y = F.linear(x, w_q, self.bias)
        return self.activation_post_process(y)

    @classmethod
    def from_float(cls, mod: nn.Module) -> Linear:
        """Build a QAT ``Linear`` from a float one (shares the trained weight)."""
        lin = cast("nn.Linear", mod)
        qat = cls(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            qconfig=cast("QConfig", mod.qconfig),  # set by prepare_qat
        )
        # Adopt the trained float weight/bias directly — the prepare_qat
        # deep-copy already gave this module tree independent Parameters.
        qat.weight = lin.weight
        if lin.bias is not None:
            qat.bias = lin.bias
        return qat


class LinearReLU(Linear):
    r"""Quantization-aware fused ``Linear`` + ``ReLU`` — trainable, fake-quant every forward.

    A :class:`Linear` whose forward applies ReLU *before* the output fake-quant, so the
    activation observer sees the true post-ReLU (non-negative) range and calibrates a grid
    that spends all of its codes on values the layer can actually emit. Fusing the two ops
    also lets a single activation observer stand in for both, which is exactly the shape
    :func:`lucid.quantization.convert` needs to collapse them into one kernel.

    Like the base :class:`Linear`, the weight and the (post-ReLU) output are fake-quantized
    every forward under a straight-through estimator (STE): rounding happens in the forward
    pass while gradients pass straight through to the float weight, so it stays fully
    trainable and learns to compensate for the eventual int8 rounding. Built from a fused
    float :class:`~lucid.nn.intrinsic.LinearReLU` by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` folds it into
    a single quantized :class:`~lucid.nn.quantized.LinearReLU`.

    On each forward the weight is fake-quantized, ``F.linear`` runs, ReLU clips, and the
    result is fake-quantized onto the observed post-ReLU grid:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\Bigl(
            \operatorname{ReLU}\bigl(
                \operatorname{fake\_quant}_{w}(W)\, x^{\top} + b\bigr)\Bigr)

    where :math:`s, z` are the observer's scale / zero-point and
    :math:`q_{\min}, q_{\max}` the grid bounds. Because the activation observer sits
    *outside* the ReLU, the calibrated :math:`(s, z)` describe the non-negative output.

    Parameters
    ----------
    in_features : int
        Size of each input sample — forwarded verbatim to the float
        :class:`~lucid.nn.Linear` parent.
    out_features : int
        Size of each output sample — forwarded verbatim to the float parent.
    bias : bool, default=True
        Whether to add a learnable (float) bias term; forwarded to the float parent.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant built from ``qconfig.weight()``; rounds the float
        weight each forward and tracks its range for ``convert``.
    activation_post_process : FakeQuantize
        The output observer + fake-quant built from ``qconfig.activation()``; here it
        observes the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **STE differentiability.** Rounding is applied forward but the gradient passes
      through as the identity, so both the weight and the pre-ReLU activations keep clean
      gradients and the float weight trains normally.
    - **Post-ReLU calibration.** Observing after the ReLU means the calibrated
      ``(scale, zero_point)`` cover only :math:`[0, \max]`, doubling the effective
      resolution versus observing a symmetric pre-ReLU range.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this from a fused float :class:`~lucid.nn.intrinsic.LinearReLU`;
      :func:`lucid.quantization.convert` folds it into the quantized
      :class:`~lucid.nn.quantized.LinearReLU`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.qat as nnqat
    >>> m = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
    >>> fused = Q.fuse_modules(m, [["0", "1"]])         # -> nni.LinearReLU marker
    >>> qat = Q.prepare_qat(fused, Q.get_default_qat_qconfig_mapping())
    >>> isinstance(qat[0], nnqat.LinearReLU)
    True
    >>> loss = (qat(lucid.randn(8, 32)) ** 2).mean()
    >>> loss.backward()                                 # STE trains the fused float weight
    >>> _ = qat.eval()
    >>> qc = Q.convert(qat)                             # -> quantized.LinearReLU
    >>> type(qc[0]).__name__
    'LinearReLU'

    See Also
    --------
    lucid.nn.quantized.LinearReLU : The fused int8 inference layer ``convert`` bakes into.
    lucid.nn.intrinsic.LinearReLU : The plain-float fused marker this is built from.
    lucid.nn.qat.Linear : The un-fused QAT base class.
    lucid.quantization.fuse_modules_qat : Fuses ``Linear`` + ``ReLU`` straight to QAT.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Fake-quantize the weight, run linear, ReLU, fake-quantize the output."""
        w_q = self.weight_fake_quant(self.weight)
        y = F.relu(F.linear(x, w_q, self.bias))
        return self.activation_post_process(y)

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> LinearReLU:
        """Build from a fused float ``nni.LinearReLU`` (its inner linear)."""
        inner = cast("nn.Sequential", mod)[0]
        inner.qconfig = mod.qconfig
        return cast("LinearReLU", super().from_float(inner))
