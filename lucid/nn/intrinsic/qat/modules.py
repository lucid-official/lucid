"""QAT fused Conv+BN(+ReLU) — BatchNorm trained inside a fake-quantized conv.

The delicate QAT primitive: a trainable conv + BatchNorm whose conv weight is
scaled by BN's running ``gamma / std`` and fake-quantized on every forward, so
the network trains against the folded-and-quantized weight inference will use.
In training mode the conv output is scaled back and run through the wrapped
BatchNorm, which normalises with the **batch** statistics and updates its
running statistics exactly as an unfused Conv → BN would; in eval mode (or
after :meth:`_ConvBnNd.freeze_bn_stats`) BN uses its running statistics and
the result is the folded conv.  Gradients flow (via STE) to the conv weight
and BN parameters.  :func:`convert` bakes the folded int8 weight into a
quantized conv.

The module is rank-generic (1d / 2d / 3d): the only rank-specific piece is the
per-channel reshape and the ``F.convNd`` call, both derived from the conv
weight's rank.
"""

from typing import TYPE_CHECKING, Self, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.utils.fusion import fuse_conv_bn_weights

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig

_CONV_FNS = (F.conv1d, F.conv2d, F.conv3d)


class _ConvBnNd(nn.Module):
    """Rank-generic fused Conv+BN(+ReLU) that trains BN under a fake-quant conv."""

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(self, conv: nn.Module, bn: nn.Module, relu: bool, qconfig: QConfig | None) -> None:
        super().__init__()
        if qconfig is None:
            raise ValueError(f"{type(self).__name__} requires a qconfig")
        if not getattr(bn, "track_running_stats", False):
            # The weight is scaled by the running std every forward, and
            # ``convert`` folds the running statistics into the int8 conv.
            raise ValueError(
                f"{type(self).__name__} requires a BatchNorm that tracks running "
                "statistics (track_running_stats=True)"
            )
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())
        self.freeze_bn = False
        self.update_bn_stats()

    def _bn(self) -> nn.BatchNorm2d:
        return cast("nn.BatchNorm2d", self.bn)  # structural: all BN ranks share these

    def _conv_module(self) -> nn.Conv2d:
        return cast("nn.Conv2d", self.conv)  # structural: all conv ranks share these

    def _fold(self) -> tuple[Tensor, Tensor]:
        """Return the BN-folded ``(weight, bias)`` for the conv (any rank)."""
        conv, bn = self._conv_module(), self._bn()
        return fuse_conv_bn_weights(
            conv.weight,
            conv.bias,
            cast("Tensor", bn.running_mean),
            cast("Tensor", bn.running_var),
            bn.eps,
            bn.weight if bn.affine else None,
            bn.bias if bn.affine else None,
        )

    def _conv(self, x: Tensor, w: Tensor) -> Tensor:
        """Bias-free ``F.conv{1,2,3}d``, dispatched by the conv weight's rank."""
        conv = self._conv_module()
        fn = _CONV_FNS[len(conv.weight.shape) - 3]
        # ``conv`` is cast to Conv2d structurally; at runtime its rank matches the
        # selected ``fn``, so the (int|tuple) stride/padding are valid for it.
        return fn(x, w, None, conv.stride, conv.padding, conv.dilation, conv.groups)  # type: ignore[arg-type]

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Scale and fake-quant the weight → conv → unscale → BN → (ReLU) → fake-quant."""
        conv, bn = self._conv_module(), self._bn()
        running_std = (cast("Tensor", bn.running_var) + bn.eps).sqrt()
        gamma = bn.weight if bn.affine else lucid.ones_like(running_std)
        scale = gamma / running_std
        out_channels = conv.weight.shape[0]
        ndim = len(conv.weight.shape)
        w_q = self.weight_fake_quant(
            conv.weight * scale.reshape((out_channels,) + (1,) * (ndim - 1))
        )
        # Dividing the scale back out leaves the unfused conv's output (up to the
        # weight's quantization error), so ``bn`` sees what it would in the float
        # model: batch statistics and the running-stat update in training, the
        # running statistics — hence exactly the folded conv — in eval.
        per_channel = (1, out_channels) + (1,) * (ndim - 2)
        y = self._conv(x, w_q) / scale.reshape(per_channel)
        if conv.bias is not None:
            y = y + conv.bias.reshape(per_channel)
        y = self.bn(y)
        if self.relu:
            y = F.relu(y)
        return self.activation_post_process(y)

    def update_bn_stats(self) -> Self:
        """Let ``bn`` normalise with batch statistics and update its running ones."""
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self) -> Self:
        """Stop the running-stat updates; ``bn`` normalises with its running stats."""
        self.freeze_bn = True
        self.bn.training = False
        return self

    @override
    def train(self, mode: bool = True) -> Self:
        """Set the mode recursively, except that a frozen ``bn`` stays frozen."""
        super().train(mode)
        if self.freeze_bn:
            self.bn.training = False
        return self


class ConvBn1d(_ConvBnNd):
    r"""QAT fused ``Conv1d`` + ``BatchNorm1d`` — BN folded into the weight every forward.

    The training-time stand-in for a ``Conv1d`` immediately followed by a ``BatchNorm1d``.
    At inference the BN scale-and-shift can be absorbed into the conv weight and bias so the
    pair runs as a *single* convolution; a naive QAT layer that folded BN once up front,
    however, would freeze BN's running statistics and stop it training. This class instead
    **re-folds BN into the conv weight on every forward**, fake-quantizes the *folded*
    weight, convolves, and fake-quantizes the output — all differentiably — so BN keeps
    learning while the network already experiences the folded-and-quantized numerics it will
    see at inference.

    **Straight-through estimator (STE).** The weight fake-quant rounds the folded kernel in
    the forward pass but passes gradients *straight through* in the backward pass, so both
    the conv weight and the BN affine parameters receive clean gradients despite the
    non-differentiable ``round``. The running statistics are not learned — the wrapped
    ``bn`` updates them from each training batch, exactly as in the unfused model.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` builds this from a fused
    conv+BN pattern. After training, :func:`lucid.quantization.convert` folds BN one final
    time, quantizes the folded weight to int8, and bakes it — together with the calibrated
    activation qparams — into an inference :class:`~lucid.nn.quantized.Conv1d`. No separate
    BN layer survives conversion; it lives entirely inside the conv weight and bias.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)`. When the conv has no bias (the usual fused case, :math:`b_c = 0`)
    the shift reduces to :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}`.
    In eval mode (or once the BN statistics are frozen) the layer is exactly that folded
    conv, with the folded weight fake-quantized under the STE:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 1-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv1d
        The float 1-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm1d
        The batch-norm layer folded into ``conv``. It must track running statistics; its
        affine parameters train under the STE and its running statistics update from each
        training batch (``ValueError`` if ``track_running_stats=False``).
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; calibrates the
        activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **Batch statistics in training.** ``bn`` normalises with them and updates its running
      ones; :meth:`freeze_bn_stats` / :meth:`update_bn_stats` stop and resume that.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.Conv1d` (BN absorbed, no BN layer left).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbn = nniqat.ConvBn1d(
    ...     nn.Conv1d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm1d(8),
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> w, b = cbn._fold()                 # BN folded into (weight, bias)
    >>> w.shape, b.shape
    ((8, 3, 3), (8,))
    >>> loss = (cbn(lucid.randn(2, 3, 8)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params

    A ``qconfig`` is mandatory — constructing the layer without one raises:

    >>> nniqat.ConvBn1d(nn.Conv1d(3, 8, 3), nn.BatchNorm1d(8))
    Traceback (most recent call last):
        ...
    ValueError: ConvBn1d requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Conv1d : The int8 inference conv ``convert`` bakes this into.
    lucid.nn.intrinsic.qat.ConvBnReLU1d : Same fold, with a fused ReLU on the output.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN.
    lucid.quantization.convert : Folds BN a final time and bakes the int8 conv.
    """

    def __init__(self, conv: nn.Module, bn: nn.Module, qconfig: QConfig | None = None) -> None:
        super().__init__(conv, bn, False, qconfig)


class ConvBn2d(_ConvBnNd):
    r"""QAT fused ``Conv2d`` + ``BatchNorm2d`` — BN folded into the weight every forward.

    The training-time stand-in for a ``Conv2d`` immediately followed by a ``BatchNorm2d``
    — the single most common fused block in quantized vision backbones. At inference the
    BN scale-and-shift can be absorbed into the conv weight and bias so the pair runs as a
    *single* convolution; a naive QAT layer that folded BN once up front, however, would
    freeze BN's running statistics and stop it training. This class instead **re-folds BN
    into the conv weight on every forward**, fake-quantizes the *folded* weight, convolves,
    and fake-quantizes the output — all differentiably — so BN keeps learning while the
    network already experiences the folded-and-quantized numerics it will see at inference.

    **Straight-through estimator (STE).** The weight fake-quant rounds the folded kernel in
    the forward pass but passes gradients *straight through* in the backward pass, so both
    the conv weight and the BN affine parameters receive clean gradients despite the
    non-differentiable ``round``. The running statistics are not learned — the wrapped
    ``bn`` updates them from each training batch, exactly as in the unfused model.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` builds this from a fused
    conv+BN pattern. After training, :func:`lucid.quantization.convert` folds BN one final
    time, quantizes the folded weight to int8, and bakes it — together with the calibrated
    activation qparams — into an inference :class:`~lucid.nn.quantized.Conv2d`. No separate
    BN layer survives conversion; it lives entirely inside the conv weight and bias.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)`. When the conv has no bias (the usual fused case, :math:`b_c = 0`)
    the shift reduces to :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}`.
    In eval mode (or once the BN statistics are frozen) the layer is exactly that folded
    conv, with the folded weight fake-quantized under the STE:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 2-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    In training mode the convolution instead sees the **batch** statistics, as the unfused
    conv → BN would: the conv output is divided back by the per-channel scale
    :math:`\gamma_c/\sqrt{\sigma_c^2+\epsilon}`, the conv bias is added, and the result runs
    through ``bn``, which normalises with the batch mean :math:`\mu_B` and variance
    :math:`\sigma_B^2` and updates its running statistics:

    .. math::

        u = \frac{\operatorname{fake\_quant}_{w}(w') \star x}{\gamma/\sqrt{\sigma^2+\epsilon}}
            + b,
        \qquad
        \operatorname{BN}(u) = \gamma\,\frac{u - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta

    Parameters
    ----------
    conv : nn.Conv2d
        The float 2-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm2d
        The batch-norm layer folded into ``conv``. It must track running statistics; its
        affine parameters train under the STE and its running statistics update from each
        training batch (``ValueError`` if ``track_running_stats=False``).
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; calibrates the
        activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **Batch statistics in training.** ``bn`` normalises with them and updates its running
      ones; :meth:`freeze_bn_stats` / :meth:`update_bn_stats` stop and resume that.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.Conv2d` (BN absorbed, no BN layer left).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbn = nniqat.ConvBn2d(
    ...     nn.Conv2d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm2d(8),
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> w, b = cbn._fold()                 # BN folded into (weight, bias)
    >>> w.shape, b.shape
    ((8, 3, 3, 3), (8,))
    >>> loss = (cbn(lucid.randn(2, 3, 8, 8)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params
    >>> int(cbn.bn.num_batches_tracked.item())   # bn saw the batch statistics
    1
    >>> _ = cbn.freeze_bn_stats()          # running statistics stop updating
    >>> _ = cbn(lucid.randn(2, 3, 8, 8))
    >>> int(cbn.bn.num_batches_tracked.item())
    1

    A ``qconfig`` is mandatory — constructing the layer without one raises:

    >>> nniqat.ConvBn2d(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))
    Traceback (most recent call last):
        ...
    ValueError: ConvBn2d requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Conv2d : The int8 inference conv ``convert`` bakes this into.
    lucid.nn.intrinsic.qat.ConvBnReLU2d : Same fold, with a fused ReLU on the output.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN.
    lucid.quantization.convert : Folds BN a final time and bakes the int8 conv.
    """

    def __init__(self, conv: nn.Module, bn: nn.Module, qconfig: QConfig | None = None) -> None:
        super().__init__(conv, bn, False, qconfig)


class ConvBn3d(_ConvBnNd):
    r"""QAT fused ``Conv3d`` + ``BatchNorm3d`` — BN folded into the weight every forward.

    The training-time stand-in for a ``Conv3d`` immediately followed by a ``BatchNorm3d``
    (volumetric / video backbones). At inference the BN scale-and-shift can be absorbed
    into the conv weight and bias so the pair runs as a *single* convolution; a naive QAT
    layer that folded BN once up front, however, would freeze BN's running statistics and
    stop it training. This class instead **re-folds BN into the conv weight on every
    forward**, fake-quantizes the *folded* weight, convolves, and fake-quantizes the output
    — all differentiably — so BN keeps learning while the network already experiences the
    folded-and-quantized numerics it will see at inference.

    **Straight-through estimator (STE).** The weight fake-quant rounds the folded kernel in
    the forward pass but passes gradients *straight through* in the backward pass, so both
    the conv weight and the BN affine parameters receive clean gradients despite the
    non-differentiable ``round``. The running statistics are not learned — the wrapped
    ``bn`` updates them from each training batch, exactly as in the unfused model.

    **Where it sits.** :func:`lucid.quantization.prepare_qat` builds this from a fused
    conv+BN pattern. After training, :func:`lucid.quantization.convert` folds BN one final
    time, quantizes the folded weight to int8, and bakes it — together with the calibrated
    activation qparams — into an inference :class:`~lucid.nn.quantized.Conv3d`. No separate
    BN layer survives conversion; it lives entirely inside the conv weight and bias.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)`. When the conv has no bias (the usual fused case, :math:`b_c = 0`)
    the shift reduces to :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}`.
    In eval mode (or once the BN statistics are frozen) the layer is exactly that folded
    conv, with the folded weight fake-quantized under the STE:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\bigl(
            \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 3-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv3d
        The float 3-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm3d
        The batch-norm layer folded into ``conv``. It must track running statistics; its
        affine parameters train under the STE and its running statistics update from each
        training batch (``ValueError`` if ``track_running_stats=False``).
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; calibrates the
        activation grid whose ``(scale, zero_point)`` ``convert`` later bakes in.

    Notes
    -----
    - **Batch statistics in training.** ``bn`` normalises with them and updates its running
      ones; :meth:`freeze_bn_stats` / :meth:`update_bn_stats` stop and resume that.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.Conv3d` (BN absorbed, no BN layer left).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbn = nniqat.ConvBn3d(
    ...     nn.Conv3d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm3d(8),
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> w, b = cbn._fold()                 # BN folded into (weight, bias)
    >>> w.shape, b.shape
    ((8, 3, 3, 3, 3), (8,))
    >>> loss = (cbn(lucid.randn(2, 3, 4, 4, 4)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params

    A ``qconfig`` is mandatory — constructing the layer without one raises:

    >>> nniqat.ConvBn3d(nn.Conv3d(3, 8, 3), nn.BatchNorm3d(8))
    Traceback (most recent call last):
        ...
    ValueError: ConvBn3d requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Conv3d : The int8 inference conv ``convert`` bakes this into.
    lucid.nn.intrinsic.qat.ConvBnReLU3d : Same fold, with a fused ReLU on the output.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN.
    lucid.quantization.convert : Folds BN a final time and bakes the int8 conv.
    """

    def __init__(self, conv: nn.Module, bn: nn.Module, qconfig: QConfig | None = None) -> None:
        super().__init__(conv, bn, False, qconfig)


class ConvBnReLU1d(_ConvBnNd):
    r"""QAT fused ``Conv1d`` + ``BatchNorm1d`` + ``ReLU`` — BN folded every forward.

    Like :class:`ConvBn1d`, but a ReLU follows the (BN-folded) convolution and the *output*
    fake-quant observes the **post**-ReLU range, so the calibrated activation grid covers
    only the non-negative values inference can actually produce. This is the fully-fused
    conv→BN→ReLU block that becomes a single quantized kernel after conversion.

    As in :class:`ConvBn1d`, BN is **re-folded into the conv weight on every forward** and
    the folded weight is fake-quantized under a straight-through estimator (STE): rounding
    happens forward, gradients pass straight through, so the conv weight *and* the BN affine
    keep training while ``bn`` updates its running statistics from each training batch. Built by :func:`lucid.quantization.prepare_qat`;
    :func:`lucid.quantization.convert` folds BN a final time, quantizes the folded weight to
    int8, and bakes it into a fused inference :class:`~lucid.nn.quantized.ConvReLU1d`.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)` (reducing to
    :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}` for a bias-free
    conv). In eval mode (or once the BN statistics are frozen) the layer is that folded conv:
    the folded weight is fake-quantized, convolved, passed through ReLU, then the output is
    fake-quantized on the observed post-ReLU grid:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\Bigl(
            \operatorname{ReLU}\bigl(
                \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr)\Bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 1-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv1d
        The float 1-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm1d
        The batch-norm layer folded into ``conv``. It must track running statistics; its
        affine parameters train under the STE and its running statistics update from each
        training batch (``ValueError`` if ``track_running_stats=False``).
    relu : bool, default=True
        Whether the fused ReLU is applied before the output fake-quant.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; here it observes
        the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **Batch statistics in training.** ``bn`` normalises with them and updates its running
      ones; :meth:`freeze_bn_stats` / :meth:`update_bn_stats` stop and resume that.
    - **Post-ReLU calibration.** The activation observer sits after the ReLU, so its grid
      covers only :math:`[0, \max]`, roughly doubling resolution versus a symmetric range.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.ConvReLU1d`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbr = nniqat.ConvBnReLU1d(
    ...     nn.Conv1d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm1d(8),
    ...     relu=True,
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> loss = (cbr(lucid.randn(2, 3, 8)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params
    >>> _ = cbr.eval()
    >>> from lucid.nn.intrinsic.qat.modules import convbnrelu2d_to_quantized
    >>> qc = convbnrelu2d_to_quantized(cbr)     # rank-generic bake
    >>> type(qc).__name__
    'ConvReLU1d'

    See Also
    --------
    lucid.nn.quantized.ConvReLU1d : The fused int8 inference conv ``convert`` bakes into.
    lucid.nn.intrinsic.qat.ConvBn1d : Same fold, without the fused ReLU.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN + ReLU.
    lucid.quantization.convert : Folds BN a final time and bakes the fused int8 conv.
    """

    def __init__(
        self, conv: nn.Module, bn: nn.Module, relu: bool = True, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(conv, bn, relu, qconfig)


class ConvBnReLU2d(_ConvBnNd):
    r"""QAT fused ``Conv2d`` + ``BatchNorm2d`` + ``ReLU`` — BN folded every forward.

    Like :class:`ConvBn2d`, but a ReLU follows the (BN-folded) convolution and the *output*
    fake-quant observes the **post**-ReLU range, so the calibrated activation grid covers
    only the non-negative values inference can actually produce. This conv→BN→ReLU triple
    is *the* canonical fused block of quantized vision backbones and becomes a single
    quantized kernel after conversion.

    As in :class:`ConvBn2d`, BN is **re-folded into the conv weight on every forward** and
    the folded weight is fake-quantized under a straight-through estimator (STE): rounding
    happens forward, gradients pass straight through, so the conv weight *and* the BN affine
    keep training while ``bn`` updates its running statistics from each training batch. Built by :func:`lucid.quantization.prepare_qat`;
    :func:`lucid.quantization.convert` folds BN a final time, quantizes the folded weight to
    int8, and bakes it into a fused inference :class:`~lucid.nn.quantized.ConvReLU2d`.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)` (reducing to
    :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}` for a bias-free
    conv). In eval mode (or once the BN statistics are frozen) the layer is that folded conv:
    the folded weight is fake-quantized, convolved, passed through ReLU, then the output is
    fake-quantized on the observed post-ReLU grid:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\Bigl(
            \operatorname{ReLU}\bigl(
                \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr)\Bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 2-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv2d
        The float 2-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm2d
        The batch-norm layer folded into ``conv``. It must track running statistics; its
        affine parameters train under the STE and its running statistics update from each
        training batch (``ValueError`` if ``track_running_stats=False``).
    relu : bool, default=True
        Whether the fused ReLU is applied before the output fake-quant.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; here it observes
        the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **Batch statistics in training.** ``bn`` normalises with them and updates its running
      ones; :meth:`freeze_bn_stats` / :meth:`update_bn_stats` stop and resume that.
    - **Post-ReLU calibration.** The activation observer sits after the ReLU, so its grid
      covers only :math:`[0, \max]`, roughly doubling resolution versus a symmetric range.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.ConvReLU2d`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbr = nniqat.ConvBnReLU2d(
    ...     nn.Conv2d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm2d(8),
    ...     relu=True,
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> loss = (cbr(lucid.randn(2, 3, 8, 8)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params
    >>> _ = cbr.eval()
    >>> from lucid.nn.intrinsic.qat.modules import convbnrelu2d_to_quantized
    >>> qc = convbnrelu2d_to_quantized(cbr)     # BN folded + baked to int8
    >>> type(qc).__name__
    'ConvReLU2d'

    See Also
    --------
    lucid.nn.quantized.ConvReLU2d : The fused int8 inference conv ``convert`` bakes into.
    lucid.nn.intrinsic.qat.ConvBn2d : Same fold, without the fused ReLU.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN + ReLU.
    lucid.quantization.convert : Folds BN a final time and bakes the fused int8 conv.
    """

    def __init__(
        self, conv: nn.Module, bn: nn.Module, relu: bool = True, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(conv, bn, relu, qconfig)


class ConvBnReLU3d(_ConvBnNd):
    r"""QAT fused ``Conv3d`` + ``BatchNorm3d`` + ``ReLU`` — BN folded every forward.

    Like :class:`ConvBn3d`, but a ReLU follows the (BN-folded) convolution and the *output*
    fake-quant observes the **post**-ReLU range, so the calibrated activation grid covers
    only the non-negative values inference can actually produce. This is the fully-fused
    conv→BN→ReLU block for volumetric / video backbones and becomes a single quantized
    kernel after conversion.

    As in :class:`ConvBn3d`, BN is **re-folded into the conv weight on every forward** and
    the folded weight is fake-quantized under a straight-through estimator (STE): rounding
    happens forward, gradients pass straight through, so the conv weight *and* the BN affine
    keep training while ``bn`` updates its running statistics from each training batch. Built by :func:`lucid.quantization.prepare_qat`;
    :func:`lucid.quantization.convert` folds BN a final time, quantizes the folded weight to
    int8, and bakes it into a fused inference :class:`~lucid.nn.quantized.ConvReLU3d`.

    The per-output-channel BN fold (:math:`c` indexes output channels) is

    .. math::

        w'_{c} = \gamma_c\,\frac{w_c}{\sqrt{\sigma_c^2 + \epsilon}},
        \qquad
        b'_{c} = \beta_c + \gamma_c\,\frac{b_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}

    with BN affine :math:`(\gamma, \beta)`, running statistics
    :math:`(\mu, \sigma^2)`, numerical floor :math:`\epsilon`, and conv weight / bias
    :math:`(w_c, b_c)` (reducing to
    :math:`b'_c = \beta_c - \gamma_c\,\mu_c/\sqrt{\sigma_c^2+\epsilon}` for a bias-free
    conv). In eval mode (or once the BN statistics are frozen) the layer is that folded conv:
    the folded weight is fake-quantized, convolved, passed through ReLU, then the output is
    fake-quantized on the observed post-ReLU grid:

    .. math::

        y = \operatorname{fake\_quant}_{a}\!\Bigl(
            \operatorname{ReLU}\bigl(
                \operatorname{fake\_quant}_{w}(w') \star x + b'\bigr)\Bigr),
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    where :math:`\star` is the 3-D cross-correlation and
    :math:`\operatorname{fake\_quant}(t) = (\operatorname{clamp}(\operatorname{round}(t/s)
    + z,\ q_{\min}, q_{\max}) - z)\, s`.

    Parameters
    ----------
    conv : nn.Conv3d
        The float 3-D convolution whose weight receives the folded BN affine.
    bn : nn.BatchNorm3d
        The batch-norm layer folded into ``conv``. It must track running statistics; its
        affine parameters train under the STE and its running statistics update from each
        training batch (``ValueError`` if ``track_running_stats=False``).
    relu : bool, default=True
        Whether the fused ReLU is applied before the output fake-quant.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
        Required — constructing the layer without one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The weight observer + fake-quant from ``qconfig.weight()``; rounds the *folded*
        kernel each forward and tracks its range so ``convert`` can pick the int8 qparams.
    activation_post_process : FakeQuantize
        The output observer + fake-quant from ``qconfig.activation()``; here it observes
        the **post-ReLU** range so the baked-in grid matches inference.

    Notes
    -----
    - **Batch statistics in training.** ``bn`` normalises with them and updates its running
      ones; :meth:`freeze_bn_stats` / :meth:`update_bn_stats` stop and resume that.
    - **Post-ReLU calibration.** The activation observer sits after the ReLU, so its grid
      covers only :math:`[0, \max]`, roughly doubling resolution versus a symmetric range.
    - **STE differentiability.** The folded weight is rounded forward but the gradient is
      passed through as the identity, so the conv weight *and* the BN parameters train.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` builds
      this fused layer *in*; :func:`lucid.quantization.convert` folds it *out* into the
      matching :class:`~lucid.nn.quantized.ConvReLU3d`.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.intrinsic.qat as nniqat
    >>> cbr = nniqat.ConvBnReLU3d(
    ...     nn.Conv3d(3, 8, 3, padding=1, bias=False),
    ...     nn.BatchNorm3d(8),
    ...     relu=True,
    ...     qconfig=Q.get_default_qat_qconfig(),
    ... )
    >>> loss = (cbr(lucid.randn(2, 3, 4, 4, 4)) ** 2).mean()
    >>> loss.backward()                    # STE trains conv weight AND BN params
    >>> _ = cbr.eval()
    >>> from lucid.nn.intrinsic.qat.modules import convbnrelu2d_to_quantized
    >>> qc = convbnrelu2d_to_quantized(cbr)     # rank-generic bake
    >>> type(qc).__name__
    'ConvReLU3d'

    See Also
    --------
    lucid.nn.quantized.ConvReLU3d : The fused int8 inference conv ``convert`` bakes into.
    lucid.nn.intrinsic.qat.ConvBn3d : Same fold, without the fused ReLU.
    lucid.quantization.prepare_qat : Builds this fused QAT layer from conv + BN + ReLU.
    lucid.quantization.convert : Folds BN a final time and bakes the fused int8 conv.
    """

    def __init__(
        self, conv: nn.Module, bn: nn.Module, relu: bool = True, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(conv, bn, relu, qconfig)


def _convbn_to_quantized(mod: nn.Module) -> nn.Module:
    """Bake a trained fused Conv+BN(+ReLU) into a quantized inference conv (any rank)."""
    import lucid.nn.quantized as nnq
    from lucid.quantization._functional import quantize

    cbr = cast("_ConvBnNd", mod)
    weight, bias = cbr._fold()
    wfq = cbr.weight_fake_quant
    wfq(weight)  # observe the folded weight
    w_scale, w_zp = wfq.calculate_qparams()
    ch_axis = wfq.ch_axis if wfq.ch_axis is not None else 0
    codes = quantize(weight, w_scale, w_zp, wfq.qdtype, ch_axis=wfq.ch_axis)

    conv = cast("nn.Conv2d", cbr.conv)
    rank = len(conv.weight.shape) - 2  # 1 / 2 / 3
    plain = (nnq.Conv1d, nnq.Conv2d, nnq.Conv3d)[rank - 1]
    fused = (nnq.ConvReLU1d, nnq.ConvReLU2d, nnq.ConvReLU3d)[rank - 1]
    q_cls = fused if cbr.relu else plain
    q = q_cls(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
    )
    q.register_buffer("weight_int8", codes)
    q.register_buffer("weight_scale", w_scale)
    q.register_buffer("weight_zero_point", w_zp)
    q.weight_ch_axis = ch_axis
    q.register_buffer("bias", bias)
    a_scale, a_zp = cbr.activation_post_process.calculate_qparams()
    q.register_buffer("scale", a_scale)
    q.register_buffer("zero_point", a_zp)
    q.out_qdtype = cbr.activation_post_process.qdtype
    return q


# Back-compat alias — the original 2d-only name (still imported by convert).
convbnrelu2d_to_quantized = _convbn_to_quantized
