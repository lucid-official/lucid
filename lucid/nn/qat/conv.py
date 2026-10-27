"""QAT ``Conv1d`` / ``Conv2d`` / ``Conv3d`` — weight + output fake-quant.

Same idea as the QAT :class:`~lucid.nn.qat.Linear`: a trainable float kernel
with fake-quant on the weight and the output, so gradients (via STE) adapt
the weights to the eventual int8 grid.  Integer / tuple padding is supported
(string ``"same"`` / ``"valid"`` is deferred, as in the quantized conv).
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig


def _conv_from_float(cls: type, mod: nn.Module) -> nn.Module:
    """Build a QAT conv from a float conv, sharing its trained kernel."""
    c = cast("nn.Conv2d", mod)  # structural: all conv ranks share these attrs
    qat = cast(
        "nn.Conv2d",
        cls(
            c.in_channels,
            c.out_channels,
            c.kernel_size,
            stride=c.stride,
            padding=c.padding,
            dilation=c.dilation,
            groups=c.groups,
            bias=c.bias is not None,
            qconfig=cast("QConfig", mod.qconfig),  # set by prepare_qat
        ),
    )
    # Adopt the trained kernel/bias directly — the prepare_qat deep-copy
    # already produced independent Parameters for this module tree.
    qat.weight = c.weight
    if c.bias is not None:
        qat.bias = c.bias
    return qat


class Conv1d(nn.Conv1d):
    r"""Quantization-aware 1-D convolution — trainable float kernel, fake-quant per forward.

    The training-time stand-in that :func:`lucid.quantization.prepare_qat` installs in
    place of a float :class:`~lucid.nn.Conv1d`. It keeps a **trainable float kernel** but
    routes both the weight and the layer output through a
    :class:`~lucid.quantization.FakeQuantize` on every forward, so the network *feels* the
    int8 rounding error as it learns and adapts its kernel to the eventual int8 grid. This
    closes the train / inference accuracy gap: a network trained *with* the rounding baked
    in barely moves when its weights are finally frozen to int8, whereas plain
    post-training quantization perturbs a kernel that never saw rounding.

    **Straight-through estimator (STE).** Rounding is a step function whose derivative is
    zero almost everywhere, so a literal ``round`` would block all gradient flow. The
    fake-quant rounds in the forward pass but, in the backward pass, pretends it was the
    identity — the incoming gradient passes *straight through* to the float kernel (clipped
    to the quantization range). The kernel thus stays fully trainable while every forward
    value it produces is the *dequantized* number an int8 conv kernel would have computed.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` deep-copies the float model
    and swaps each :class:`~lucid.nn.Conv1d` for this class, attaching the weight and
    activation ``FakeQuantize`` modules from the layer's ``qconfig``. After training,
    :func:`lucid.quantization.convert` reads the trained kernel together with the
    activation observer's final ``(scale, zero_point)`` and bakes them into an inference
    :class:`~lucid.nn.quantized.Conv1d` whose weight is stored as (typically
    per-output-channel) int8 codes.

    On each forward the kernel is fake-quantized, the ordinary ``F.conv1d`` runs, and the
    result is fake-quantized onto the observed activation grid:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(W) \star x + b\bigr)

    where :math:`\star` is the 1-D cross-correlation, :math:`s, z` are the scale /
    zero-point the enclosing ``FakeQuantize`` derives from its observer (per-output-channel
    for the weight), and :math:`q_{\min}, q_{\max}` are the grid bounds. The
    straight-through unit derivative is what keeps the layer differentiable despite the
    non-differentiable ``round``.

    Parameters
    ----------
    qconfig : QConfig, keyword-only
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.
    *args, **kwargs
        All remaining arguments are forwarded verbatim to the float
        :class:`~lucid.nn.Conv1d` constructor (``in_channels``, ``out_channels``,
        ``kernel_size``, ``stride``, ``padding``, ``dilation``, ``groups``, ``bias``),
        which owns the actual trainable kernel and bias.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant built from ``qconfig.weight()``; rounds the float
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant built from ``qconfig.activation()``; calibrates
        the activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **STE differentiability.** ``round`` is applied forward but the gradient is passed
      through as the identity, so the float kernel trains normally.
    - **Deferred string padding.** Integer / tuple ``padding`` is supported; string padding
      (``"same"`` / ``"valid"``) is deferred, matching the quantized conv.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` swaps
      the float conv *in*; :func:`lucid.quantization.convert` folds this layer *out* into
      the matching :class:`~lucid.nn.quantized.Conv1d`. Manual construction is rarely
      needed and requires an explicit ``qconfig``.
    - **Training is slower, not faster.** Only the int8 *numerics* are simulated; the
      convolution itself still runs in float with two extra fake-quant ops, so the speed /
      memory win arrives only after ``convert``.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.qat as nnqat
    >>> model = nn.Sequential(nn.Conv1d(3, 8, 3, padding=1))
    >>> qat = Q.prepare_qat(model)              # nn.Conv1d -> qat.Conv1d
    >>> isinstance(qat[0], nnqat.Conv1d)
    True
    >>> loss = (qat(lucid.randn(2, 3, 8)) ** 2).mean()
    >>> loss.backward()                         # STE routes grads to the float kernel
    >>> _ = qat.eval()
    >>> qc = Q.convert(qat)                     # -> quantized.Conv1d (int8 weight)
    >>> type(qc[0]).__name__
    'Conv1d'

    A ``qconfig`` is mandatory — constructing the layer directly without one raises:

    >>> nnqat.Conv1d(3, 8, 3)
    Traceback (most recent call last):
        ...
    ValueError: qat conv requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Conv1d : The int8 inference conv ``convert`` bakes this into.
    lucid.nn.qat.ConvReLU1d : QAT conv with a fused ReLU before the output observer.
    lucid.quantization.prepare_qat : Swaps the float ``Conv1d`` for this QAT layer.
    lucid.quantization.convert : Bakes the trained QAT layer into the quantized conv.
    """

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(self, *args: object, qconfig: QConfig | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # forwarded to nn.Conv1d
        if qconfig is None:
            raise ValueError("qat conv requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:
        w_q = self.weight_fake_quant(self.weight)
        y = F.conv1d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.activation_post_process(y)

    @classmethod
    def from_float(cls, mod: nn.Module) -> Conv1d:
        return cast("Conv1d", _conv_from_float(cls, mod))


class Conv2d(nn.Conv2d):
    r"""Quantization-aware 2-D convolution — trainable float kernel, fake-quant per forward.

    The training-time stand-in that :func:`lucid.quantization.prepare_qat` installs in
    place of a float :class:`~lucid.nn.Conv2d` — the workhorse of every quantized vision
    backbone. It keeps a **trainable float kernel** but routes both the weight and the
    layer output through a :class:`~lucid.quantization.FakeQuantize` on every forward, so
    the network *feels* the int8 rounding error as it learns and adapts its kernel to the
    eventual int8 grid. This closes the train / inference accuracy gap: a network trained
    *with* the rounding baked in barely moves when its weights are finally frozen to int8,
    whereas plain post-training quantization perturbs a kernel that never saw rounding.

    **Straight-through estimator (STE).** Rounding is a step function whose derivative is
    zero almost everywhere, so a literal ``round`` would block all gradient flow. The
    fake-quant rounds in the forward pass but, in the backward pass, pretends it was the
    identity — the incoming gradient passes *straight through* to the float kernel (clipped
    to the quantization range). The kernel thus stays fully trainable while every forward
    value it produces is the *dequantized* number an int8 conv kernel would have computed.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` deep-copies the float model
    and swaps each :class:`~lucid.nn.Conv2d` for this class, attaching the weight and
    activation ``FakeQuantize`` modules from the layer's ``qconfig``. After training,
    :func:`lucid.quantization.convert` reads the trained kernel together with the
    activation observer's final ``(scale, zero_point)`` and bakes them into an inference
    :class:`~lucid.nn.quantized.Conv2d` whose weight is stored as (typically
    per-output-channel) int8 codes.

    On each forward the kernel is fake-quantized, the ordinary ``F.conv2d`` runs, and the
    result is fake-quantized onto the observed activation grid:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(W) \star x + b\bigr)

    where :math:`\star` is the 2-D cross-correlation, :math:`s, z` are the scale /
    zero-point the enclosing ``FakeQuantize`` derives from its observer (per-output-channel
    for the weight), and :math:`q_{\min}, q_{\max}` are the grid bounds. The
    straight-through unit derivative is what keeps the layer differentiable despite the
    non-differentiable ``round``.

    Parameters
    ----------
    qconfig : QConfig, keyword-only
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.
    *args, **kwargs
        All remaining arguments are forwarded verbatim to the float
        :class:`~lucid.nn.Conv2d` constructor (``in_channels``, ``out_channels``,
        ``kernel_size``, ``stride``, ``padding``, ``dilation``, ``groups``, ``bias``),
        which owns the actual trainable kernel and bias.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant built from ``qconfig.weight()``; rounds the float
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant built from ``qconfig.activation()``; calibrates
        the activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **STE differentiability.** ``round`` is applied forward but the gradient is passed
      through as the identity, so the float kernel trains normally.
    - **Deferred string padding.** Integer / tuple ``padding`` is supported; string padding
      (``"same"`` / ``"valid"``) is deferred, matching the quantized conv.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` swaps
      the float conv *in*; :func:`lucid.quantization.convert` folds this layer *out* into
      the matching :class:`~lucid.nn.quantized.Conv2d`. Manual construction is rarely
      needed and requires an explicit ``qconfig``.
    - **Training is slower, not faster.** Only the int8 *numerics* are simulated; the
      convolution itself still runs in float with two extra fake-quant ops, so the speed /
      memory win arrives only after ``convert``.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.qat as nnqat
    >>> model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1))
    >>> qat = Q.prepare_qat(model)              # nn.Conv2d -> qat.Conv2d
    >>> isinstance(qat[0], nnqat.Conv2d)
    True
    >>> loss = (qat(lucid.randn(2, 3, 8, 8)) ** 2).mean()
    >>> loss.backward()                         # STE routes grads to the float kernel
    >>> _ = qat.eval()
    >>> qc = Q.convert(qat)                     # -> quantized.Conv2d (int8 weight)
    >>> type(qc[0]).__name__
    'Conv2d'

    A ``qconfig`` is mandatory — constructing the layer directly without one raises:

    >>> nnqat.Conv2d(3, 8, 3)
    Traceback (most recent call last):
        ...
    ValueError: qat conv requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Conv2d : The int8 inference conv ``convert`` bakes this into.
    lucid.nn.qat.ConvReLU2d : QAT conv with a fused ReLU before the output observer.
    lucid.quantization.prepare_qat : Swaps the float ``Conv2d`` for this QAT layer.
    lucid.quantization.convert : Bakes the trained QAT layer into the quantized conv.
    """

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(self, *args: object, qconfig: QConfig | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # forwarded to nn.Conv2d
        if qconfig is None:
            raise ValueError("qat conv requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:
        w_q = self.weight_fake_quant(self.weight)
        y = F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.activation_post_process(y)

    @classmethod
    def from_float(cls, mod: nn.Module) -> Conv2d:
        return cast("Conv2d", _conv_from_float(cls, mod))


class Conv3d(nn.Conv3d):
    r"""Quantization-aware 3-D convolution — trainable float kernel, fake-quant per forward.

    The training-time stand-in that :func:`lucid.quantization.prepare_qat` installs in
    place of a float :class:`~lucid.nn.Conv3d` (volumetric / video backbones). It keeps a
    **trainable float kernel** but routes both the weight and the layer output through a
    :class:`~lucid.quantization.FakeQuantize` on every forward, so the network *feels* the
    int8 rounding error as it learns and adapts its kernel to the eventual int8 grid. This
    closes the train / inference accuracy gap: a network trained *with* the rounding baked
    in barely moves when its weights are finally frozen to int8, whereas plain
    post-training quantization perturbs a kernel that never saw rounding.

    **Straight-through estimator (STE).** Rounding is a step function whose derivative is
    zero almost everywhere, so a literal ``round`` would block all gradient flow. The
    fake-quant rounds in the forward pass but, in the backward pass, pretends it was the
    identity — the incoming gradient passes *straight through* to the float kernel (clipped
    to the quantization range). The kernel thus stays fully trainable while every forward
    value it produces is the *dequantized* number an int8 conv kernel would have computed.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` deep-copies the float model
    and swaps each :class:`~lucid.nn.Conv3d` for this class, attaching the weight and
    activation ``FakeQuantize`` modules from the layer's ``qconfig``. After training,
    :func:`lucid.quantization.convert` reads the trained kernel together with the
    activation observer's final ``(scale, zero_point)`` and bakes them into an inference
    :class:`~lucid.nn.quantized.Conv3d` whose weight is stored as (typically
    per-output-channel) int8 codes.

    On each forward the kernel is fake-quantized, the ordinary ``F.conv3d`` runs, and the
    result is fake-quantized onto the observed activation grid:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(W) \star x + b\bigr)

    where :math:`\star` is the 3-D cross-correlation, :math:`s, z` are the scale /
    zero-point the enclosing ``FakeQuantize`` derives from its observer (per-output-channel
    for the weight), and :math:`q_{\min}, q_{\max}` are the grid bounds. The
    straight-through unit derivative is what keeps the layer differentiable despite the
    non-differentiable ``round``.

    Parameters
    ----------
    qconfig : QConfig, keyword-only
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.
    *args, **kwargs
        All remaining arguments are forwarded verbatim to the float
        :class:`~lucid.nn.Conv3d` constructor (``in_channels``, ``out_channels``,
        ``kernel_size``, ``stride``, ``padding``, ``dilation``, ``groups``, ``bias``),
        which owns the actual trainable kernel and bias.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant built from ``qconfig.weight()``; rounds the float
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant built from ``qconfig.activation()``; calibrates
        the activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **STE differentiability.** ``round`` is applied forward but the gradient is passed
      through as the identity, so the float kernel trains normally.
    - **Deferred string padding.** Integer / tuple ``padding`` is supported; string padding
      (``"same"`` / ``"valid"``) is deferred, matching the quantized conv.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` swaps
      the float conv *in*; :func:`lucid.quantization.convert` folds this layer *out* into
      the matching :class:`~lucid.nn.quantized.Conv3d`. Manual construction is rarely
      needed and requires an explicit ``qconfig``.
    - **Training is slower, not faster.** Only the int8 *numerics* are simulated; the
      convolution itself still runs in float with two extra fake-quant ops, so the speed /
      memory win arrives only after ``convert``.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.qat as nnqat
    >>> model = nn.Sequential(nn.Conv3d(3, 8, 3, padding=1))
    >>> qat = Q.prepare_qat(model)              # nn.Conv3d -> qat.Conv3d
    >>> isinstance(qat[0], nnqat.Conv3d)
    True
    >>> loss = (qat(lucid.randn(2, 3, 4, 4, 4)) ** 2).mean()
    >>> loss.backward()                         # STE routes grads to the float kernel
    >>> _ = qat.eval()
    >>> qc = Q.convert(qat)                     # -> quantized.Conv3d (int8 weight)
    >>> type(qc[0]).__name__
    'Conv3d'

    A ``qconfig`` is mandatory — constructing the layer directly without one raises:

    >>> nnqat.Conv3d(3, 8, 3)
    Traceback (most recent call last):
        ...
    ValueError: qat conv requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Conv3d : The int8 inference conv ``convert`` bakes this into.
    lucid.nn.qat.ConvReLU3d : QAT conv with a fused ReLU before the output observer.
    lucid.quantization.prepare_qat : Swaps the float ``Conv3d`` for this QAT layer.
    lucid.quantization.convert : Bakes the trained QAT layer into the quantized conv.
    """

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(self, *args: object, qconfig: QConfig | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # forwarded to nn.Conv3d
        if qconfig is None:
            raise ValueError("qat conv requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:
        w_q = self.weight_fake_quant(self.weight)
        y = F.conv3d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.activation_post_process(y)

    @classmethod
    def from_float(cls, mod: nn.Module) -> Conv3d:
        return cast("Conv3d", _conv_from_float(cls, mod))


def _fused_conv_from_float(cls: type, mod: nn.Module) -> nn.Module:
    """Build a QAT conv-relu from a fused float ``nni.ConvReLU`` (its inner conv)."""
    inner = cast("nn.Sequential", mod)[0]
    inner.qconfig = mod.qconfig
    return _conv_from_float(cls, inner)


class ConvReLU1d(Conv1d):
    r"""Quantization-aware fused 1-D conv + ReLU — trainable, fake-quant per forward.

    A :class:`Conv1d` whose forward applies ReLU *before* the output fake-quant, so the
    activation observer sees the true post-ReLU (non-negative) range and calibrates a grid
    that spends all of its codes on values the layer can actually emit. Fusing conv and
    ReLU also lets a single activation observer stand in for both, which is exactly the
    shape :func:`lucid.quantization.convert` needs to collapse them into one kernel.

    Like the base :class:`Conv1d`, the weight and the (post-ReLU) output are fake-quantized
    every forward under a straight-through estimator (STE): rounding happens in the forward
    pass while gradients pass straight through to the float kernel, so it stays fully
    trainable and learns to compensate for the eventual int8 rounding. Built from a fused
    float :class:`~lucid.nn.intrinsic.ConvReLU1d` by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` folds it into
    a single quantized :class:`~lucid.nn.quantized.ConvReLU1d`.

    On each forward the kernel is fake-quantized, ``F.conv1d`` runs, ReLU clips, and the
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
                \operatorname{fake\_quant}_{w}(W) \star x + b\bigr)\Bigr)

    where :math:`\star` is the 1-D cross-correlation, :math:`s, z` are the observer's scale
    / zero-point and :math:`q_{\min}, q_{\max}` the grid bounds. Because the activation
    observer sits *outside* the ReLU, the calibrated :math:`(s, z)` describe the
    non-negative output.

    Parameters
    ----------
    qconfig : QConfig, keyword-only
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.
    *args, **kwargs
        All remaining arguments are forwarded to the float :class:`~lucid.nn.Conv1d`
        constructor, exactly as for :class:`Conv1d`.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant built from ``qconfig.weight()``; rounds the float
        kernel each forward and tracks its range for ``convert``.
    activation_post_process : FakeQuantize
        The output observer + fake-quant built from ``qconfig.activation()``; here it
        observes the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **STE differentiability.** Rounding is applied forward but the gradient passes
      through as the identity, so the float kernel trains normally.
    - **Post-ReLU calibration.** Observing after the ReLU means the calibrated grid covers
      only :math:`[0, \max]`, roughly doubling resolution versus a symmetric pre-ReLU range.
    - **Deferred string padding.** Integer / tuple ``padding`` is supported; string padding
      (``"same"`` / ``"valid"``) is deferred, matching the quantized conv.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this from a fused float :class:`~lucid.nn.intrinsic.ConvReLU1d`;
      :func:`lucid.quantization.convert` folds it into the quantized
      :class:`~lucid.nn.quantized.ConvReLU1d`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.qat as nnqat
    >>> m = nn.Sequential(nn.Conv1d(3, 8, 3, padding=1), nn.ReLU())
    >>> fused = Q.fuse_modules(m, [["0", "1"]])         # -> nni.ConvReLU1d marker
    >>> qat = Q.prepare_qat(fused, Q.get_default_qat_qconfig_mapping())
    >>> isinstance(qat[0], nnqat.ConvReLU1d)
    True
    >>> loss = (qat(lucid.randn(2, 3, 8)) ** 2).mean()
    >>> loss.backward()                                 # STE trains the fused float kernel
    >>> _ = qat.eval()
    >>> qc = Q.convert(qat)                             # -> quantized.ConvReLU1d
    >>> type(qc[0]).__name__
    'ConvReLU1d'

    See Also
    --------
    lucid.nn.quantized.ConvReLU1d : The fused int8 inference conv ``convert`` bakes into.
    lucid.nn.intrinsic.ConvReLU1d : The plain-float fused marker this is built from.
    lucid.nn.qat.Conv1d : The un-fused QAT base class.
    lucid.quantization.fuse_modules_qat : Fuses ``Conv1d`` + ``ReLU`` straight to QAT.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        w_q = self.weight_fake_quant(self.weight)
        y = F.conv1d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.activation_post_process(F.relu(y))

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> ConvReLU1d:
        return cast("ConvReLU1d", _fused_conv_from_float(cls, mod))


class ConvReLU2d(Conv2d):
    r"""Quantization-aware fused 2-D conv + ReLU — trainable, fake-quant per forward.

    A :class:`Conv2d` whose forward applies ReLU *before* the output fake-quant, so the
    activation observer sees the true post-ReLU (non-negative) range and calibrates a grid
    that spends all of its codes on values the layer can actually emit. This is the
    dominant fused pattern in quantized vision backbones. Fusing conv and ReLU also lets a
    single activation observer stand in for both, which is exactly the shape
    :func:`lucid.quantization.convert` needs to collapse them into one kernel.

    Like the base :class:`Conv2d`, the weight and the (post-ReLU) output are fake-quantized
    every forward under a straight-through estimator (STE): rounding happens in the forward
    pass while gradients pass straight through to the float kernel, so it stays fully
    trainable and learns to compensate for the eventual int8 rounding. Built from a fused
    float :class:`~lucid.nn.intrinsic.ConvReLU2d` by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` folds it into
    a single quantized :class:`~lucid.nn.quantized.ConvReLU2d`.

    On each forward the kernel is fake-quantized, ``F.conv2d`` runs, ReLU clips, and the
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
                \operatorname{fake\_quant}_{w}(W) \star x + b\bigr)\Bigr)

    where :math:`\star` is the 2-D cross-correlation, :math:`s, z` are the observer's scale
    / zero-point and :math:`q_{\min}, q_{\max}` the grid bounds. Because the activation
    observer sits *outside* the ReLU, the calibrated :math:`(s, z)` describe the
    non-negative output.

    Parameters
    ----------
    qconfig : QConfig, keyword-only
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.
    *args, **kwargs
        All remaining arguments are forwarded to the float :class:`~lucid.nn.Conv2d`
        constructor, exactly as for :class:`Conv2d`.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant built from ``qconfig.weight()``; rounds the float
        kernel each forward and tracks its range for ``convert``.
    activation_post_process : FakeQuantize
        The output observer + fake-quant built from ``qconfig.activation()``; here it
        observes the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **STE differentiability.** Rounding is applied forward but the gradient passes
      through as the identity, so the float kernel trains normally.
    - **Post-ReLU calibration.** Observing after the ReLU means the calibrated grid covers
      only :math:`[0, \max]`, roughly doubling resolution versus a symmetric pre-ReLU range.
    - **Deferred string padding.** Integer / tuple ``padding`` is supported; string padding
      (``"same"`` / ``"valid"``) is deferred, matching the quantized conv.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this from a fused float :class:`~lucid.nn.intrinsic.ConvReLU2d`;
      :func:`lucid.quantization.convert` folds it into the quantized
      :class:`~lucid.nn.quantized.ConvReLU2d`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.qat as nnqat
    >>> m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
    >>> fused = Q.fuse_modules(m, [["0", "1"]])         # -> nni.ConvReLU2d marker
    >>> qat = Q.prepare_qat(fused, Q.get_default_qat_qconfig_mapping())
    >>> isinstance(qat[0], nnqat.ConvReLU2d)
    True
    >>> loss = (qat(lucid.randn(2, 3, 8, 8)) ** 2).mean()
    >>> loss.backward()                                 # STE trains the fused float kernel
    >>> _ = qat.eval()
    >>> qc = Q.convert(qat)                             # -> quantized.ConvReLU2d
    >>> type(qc[0]).__name__
    'ConvReLU2d'

    See Also
    --------
    lucid.nn.quantized.ConvReLU2d : The fused int8 inference conv ``convert`` bakes into.
    lucid.nn.intrinsic.ConvReLU2d : The plain-float fused marker this is built from.
    lucid.nn.qat.Conv2d : The un-fused QAT base class.
    lucid.quantization.fuse_modules_qat : Fuses ``Conv2d`` + ``ReLU`` straight to QAT.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        w_q = self.weight_fake_quant(self.weight)
        y = F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.activation_post_process(F.relu(y))

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> ConvReLU2d:
        return cast("ConvReLU2d", _fused_conv_from_float(cls, mod))


class ConvReLU3d(Conv3d):
    r"""Quantization-aware fused 3-D conv + ReLU — trainable, fake-quant per forward.

    A :class:`Conv3d` whose forward applies ReLU *before* the output fake-quant, so the
    activation observer sees the true post-ReLU (non-negative) range and calibrates a grid
    that spends all of its codes on values the layer can actually emit. Fusing conv and
    ReLU also lets a single activation observer stand in for both, which is exactly the
    shape :func:`lucid.quantization.convert` needs to collapse them into one kernel.

    Like the base :class:`Conv3d`, the weight and the (post-ReLU) output are fake-quantized
    every forward under a straight-through estimator (STE): rounding happens in the forward
    pass while gradients pass straight through to the float kernel, so it stays fully
    trainable and learns to compensate for the eventual int8 rounding. Built from a fused
    float :class:`~lucid.nn.intrinsic.ConvReLU3d` by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` folds it into
    a single quantized :class:`~lucid.nn.quantized.ConvReLU3d`.

    On each forward the kernel is fake-quantized, ``F.conv3d`` runs, ReLU clips, and the
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
                \operatorname{fake\_quant}_{w}(W) \star x + b\bigr)\Bigr)

    where :math:`\star` is the 3-D cross-correlation, :math:`s, z` are the observer's scale
    / zero-point and :math:`q_{\min}, q_{\max}` the grid bounds. Because the activation
    observer sits *outside* the ReLU, the calibrated :math:`(s, z)` describe the
    non-negative output.

    Parameters
    ----------
    qconfig : QConfig, keyword-only
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.
    *args, **kwargs
        All remaining arguments are forwarded to the float :class:`~lucid.nn.Conv3d`
        constructor, exactly as for :class:`Conv3d`.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant built from ``qconfig.weight()``; rounds the float
        kernel each forward and tracks its range for ``convert``.
    activation_post_process : FakeQuantize
        The output observer + fake-quant built from ``qconfig.activation()``; here it
        observes the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **STE differentiability.** Rounding is applied forward but the gradient passes
      through as the identity, so the float kernel trains normally.
    - **Post-ReLU calibration.** Observing after the ReLU means the calibrated grid covers
      only :math:`[0, \max]`, roughly doubling resolution versus a symmetric pre-ReLU range.
    - **Deferred string padding.** Integer / tuple ``padding`` is supported; string padding
      (``"same"`` / ``"valid"``) is deferred, matching the quantized conv.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this from a fused float :class:`~lucid.nn.intrinsic.ConvReLU3d`;
      :func:`lucid.quantization.convert` folds it into the quantized
      :class:`~lucid.nn.quantized.ConvReLU3d`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.qat as nnqat
    >>> m = nn.Sequential(nn.Conv3d(3, 8, 3, padding=1), nn.ReLU())
    >>> fused = Q.fuse_modules(m, [["0", "1"]])         # -> nni.ConvReLU3d marker
    >>> qat = Q.prepare_qat(fused, Q.get_default_qat_qconfig_mapping())
    >>> isinstance(qat[0], nnqat.ConvReLU3d)
    True
    >>> loss = (qat(lucid.randn(2, 3, 4, 4, 4)) ** 2).mean()
    >>> loss.backward()                                 # STE trains the fused float kernel
    >>> _ = qat.eval()
    >>> qc = Q.convert(qat)                             # -> quantized.ConvReLU3d
    >>> type(qc[0]).__name__
    'ConvReLU3d'

    See Also
    --------
    lucid.nn.quantized.ConvReLU3d : The fused int8 inference conv ``convert`` bakes into.
    lucid.nn.intrinsic.ConvReLU3d : The plain-float fused marker this is built from.
    lucid.nn.qat.Conv3d : The un-fused QAT base class.
    lucid.quantization.fuse_modules_qat : Fuses ``Conv3d`` + ``ReLU`` straight to QAT.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        w_q = self.weight_fake_quant(self.weight)
        y = F.conv3d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.activation_post_process(F.relu(y))

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> ConvReLU3d:
        return cast("ConvReLU3d", _fused_conv_from_float(cls, mod))
