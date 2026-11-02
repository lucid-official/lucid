"""
lucid.nn.utils.fusion — inference-time layer fusion utilities.

The canonical entry point is ``fuse_conv_bn_eval(conv, bn)`` which
returns a new conv module with the BatchNorm's running statistics +
affine parameters absorbed into the conv's weight and bias.  The result
computes exactly the same thing as ``bn(conv(x))`` *at eval time* but
with one fewer kernel launch.

Only valid in evaluation mode (when BN uses its frozen running
statistics rather than batch statistics), so the module-level entry
points refuse a module in training mode, as the reference does.
"""

import copy as _copy_module
from dataclasses import dataclass
from typing import Self

import lucid as _lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.modules.conv import _ConvNd
from lucid.nn.modules.conv_transpose import _ConvTransposeNd
from lucid.nn.modules.linear import Linear
from lucid.nn.modules.normalization import _BatchNormBase
from lucid.nn.parameter import Parameter


@dataclass(frozen=True)
class _BatchNormState:
    """The BatchNorm tensors a fold reads, as constants (no graph, no grad)."""

    mean: Tensor
    var: Tensor
    eps: float
    weight: Tensor | None
    bias: Tensor | None

    @classmethod
    def of(cls, bn: _BatchNormBase, *modules: _ConvNd | _ConvTransposeNd | Linear) -> Self:
        """Read ``bn``, refusing training mode and a BatchNorm with no running statistics."""
        if bn.training or any(module.training for module in modules):
            raise AssertionError("Fusion only for eval!")
        mean, var = bn.running_mean, bn.running_var
        if isinstance(mean, Tensor) and isinstance(var, Tensor):
            return cls(mean, var, bn.eps, bn.weight, bn.bias)
        raise AssertionError("bn.running_mean and bn.running_var must not be None")

    @property
    def features(self) -> int:
        return int(self.mean.shape[0])

    def scale(self) -> Tensor:
        """``gamma / sqrt(var + eps)``, with ``gamma = 1`` when there is no affine."""
        inv_std = _lucid.rsqrt(self.var.detach() + self.eps)
        return inv_std if self.weight is None else inv_std * self.weight.detach()

    def fold_bias(self, bias: Tensor | None, out_features: int) -> Tensor:
        """``(bias - mean) * scale + beta``, a missing ``bias`` being zero."""
        if bias is None:
            bias = _lucid.zeros(out_features, dtype=self.mean.dtype, device=self.mean.device)
        folded = (bias.detach() - self.mean.detach()) * self.scale()
        return folded if self.bias is None else folded + self.bias.detach()


def _as_parameters(
    weight: Tensor, bias: Tensor, source_w: Tensor, source_b: Tensor | None
) -> tuple[Parameter, Parameter]:
    """Wrap the fused tensors as parameters of the source's dtype and ``requires_grad``.

    A missing source bias leaves a fresh bias that does not require grad —
    what the reference gives.
    """
    bias_dtype = source_b.dtype if source_b is not None else source_w.dtype
    return (
        Parameter(weight.to(dtype=source_w.dtype), requires_grad=source_w.requires_grad),
        Parameter(
            bias.to(dtype=bias_dtype),
            requires_grad=source_b.requires_grad if source_b is not None else False,
        ),
    )


def _scaled_weight(weight: Tensor, scale: Tensor, transpose: bool, groups: int) -> Tensor:
    """Scale ``weight`` per output channel — dim 0, or dim 1 (per group) when transposed."""
    tail = (1,) * (weight.ndim - 2)
    if not transpose:
        return weight * scale.reshape((-1, 1, *tail))
    if groups == 1:
        return weight * scale.reshape((1, -1, *tail))
    # A grouped transposed weight is (in, out / groups, *k); the groups sit in dim 0.
    in_channels, per_group = int(weight.shape[0]), int(weight.shape[1])
    grouped = weight.reshape((groups, in_channels // groups, per_group, *weight.shape[2:]))
    return (grouped * scale.reshape((groups, 1, per_group, *tail))).reshape(tuple(weight.shape))


def _fuse_conv_tensors(
    conv_w: Tensor, conv_b: Tensor | None, bn: _BatchNormState, transpose: bool, groups: int
) -> tuple[Parameter, Parameter]:
    channels = int(conv_w.shape[1]) * groups if transpose else int(conv_w.shape[0])
    if bn.features != channels:
        raise ValueError(
            f"fuse_conv_bn_weights: the BatchNorm has {bn.features} features but the "
            f"convolution has {channels} output channels"
        )
    weight = _scaled_weight(conv_w.detach(), bn.scale(), transpose, groups)
    return _as_parameters(weight, bn.fold_bias(conv_b, channels), conv_w, conv_b)


def _fuse_linear_tensors(
    linear_w: Tensor, linear_b: Tensor | None, bn: _BatchNormState
) -> tuple[Parameter, Parameter]:
    out_features = int(linear_w.shape[0])
    if bn.features not in (out_features, 1):
        raise ValueError(
            f"fuse_linear_bn_weights: the BatchNorm has {bn.features} features but the "
            f"linear layer has {out_features} outputs"
        )
    weight = linear_w.detach() * bn.scale().reshape((-1, 1))
    return _as_parameters(weight, bn.fold_bias(linear_b, out_features), linear_w, linear_b)


def fuse_conv_bn_eval[ConvT: (_ConvNd | _ConvTransposeNd)](
    conv: ConvT, bn: _BatchNormBase, transpose: bool = False
) -> ConvT:
    r"""Fold a BatchNorm layer into the preceding Conv weights (inference-only).

    Because BN at eval time is an affine map with frozen statistics, the
    composition ``BN(Conv(x))`` is itself a single convolution — the BN
    can be analytically absorbed into the conv's weight and bias.  The
    fused module computes exactly the same output but with one fewer
    kernel launch and one fewer allocation per call.  Standard step in
    deployment / quantisation pipelines.

    Parameters
    ----------
    conv : Conv1d | Conv2d | Conv3d | ConvTranspose1d | ConvTranspose2d | ConvTranspose3d
        Convolution module whose weight / bias will absorb the BN.
    bn : BatchNorm1d | BatchNorm2d | BatchNorm3d
        BatchNorm whose rank matches ``conv`` and whose feature count is
        the convolution's output channel count.
    transpose : bool, optional
        ``True`` for a transposed convolution, whose weight keeps the
        output channels in dim 1 (``(in, out / groups, *kernel)``).  Must
        agree with the type of ``conv``.  Default ``False``.

    Returns
    -------
    Module
        Deep copy of ``conv`` with fused parameters.  Originals are not
        mutated.

    Raises
    ------
    TypeError
        If ``conv`` or ``bn`` is not one of the supported types.
    AssertionError
        If ``conv`` or ``bn`` is in training mode, or ``bn`` keeps no
        running statistics (``track_running_stats=False``).
    ValueError
        If ``transpose`` disagrees with the type of ``conv``, or the
        BatchNorm's feature count is not the convolution's output channels.

    Notes
    -----
    Let :math:`\mu`, :math:`\sigma^2`, :math:`\epsilon` be the BN running
    statistics and :math:`\gamma`, :math:`\beta` its affine parameters
    (treated as :math:`1` and :math:`0` if ``affine=False``).  The
    fused weight and bias are

    .. math::

        \mathbf{W}_{\text{fused}} \;=\;
            \mathbf{W} \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}},

    .. math::

        b_{\text{fused}} \;=\;
            \frac{\gamma\,(b - \mu)}{\sqrt{\sigma^2 + \epsilon}} + \beta,

    where the scale broadcasts along the output-channel axis.  When the
    original conv has no bias, :math:`b` is taken as :math:`0` and the
    fused module gains one.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import fuse_conv_bn_eval
    >>> conv = nn.Conv2d(3, 16, 3); bn = nn.BatchNorm2d(16)
    >>> _ = conv.eval(); _ = bn.eval()
    >>> fused = fuse_conv_bn_eval(conv, bn)
    >>> x = lucid.randn(1, 3, 8, 8)
    >>> (fused(x) - bn(conv(x))).abs().max().item() < 1e-4
    True
    """
    if not isinstance(conv, (_ConvNd, _ConvTransposeNd)):
        raise TypeError(
            f"fuse_conv_bn_eval: conv must be a Conv or ConvTranspose module, "
            f"got {type(conv).__name__}"
        )
    if not isinstance(bn, _BatchNormBase):
        raise TypeError(f"fuse_conv_bn_eval: bn must be BatchNorm1d/2d/3d, got {type(bn).__name__}")
    if transpose != isinstance(conv, _ConvTransposeNd):
        raise ValueError(
            f"fuse_conv_bn_eval: transpose={transpose} does not match {type(conv).__name__}"
        )
    state = _BatchNormState.of(bn, conv)

    fused = _copy_module.deepcopy(conv)
    fused.weight, fused.bias = _fuse_conv_tensors(
        fused.weight, fused.bias, state, transpose, fused.groups
    )
    return fused


def fuse_conv_bn_weights(
    conv_w: Tensor,
    conv_b: Tensor | None,
    bn_rm: Tensor,
    bn_rv: Tensor,
    bn_eps: float,
    bn_w: Tensor | None,
    bn_b: Tensor | None,
    transpose: bool = False,
) -> tuple[Parameter, Parameter]:
    r"""Low-level form of :func:`fuse_conv_bn_eval` operating on raw tensors.

    Takes the relevant Conv and BN tensors as plain arguments and
    returns the fused ``(weight, bias)`` pair.  Useful for build tools
    that walk a serialised graph (ONNX, ahead-of-time compilation) and
    need to perform the fusion without instantiating Module objects.

    Parameters
    ----------
    conv_w : Tensor
        Convolution weight, shape ``(out_channels, in_channels, *kernel)``.
    conv_b : Tensor or None
        Convolution bias, shape ``(out_channels,)``, or ``None`` if the
        conv has no bias.
    bn_rm : Tensor
        BatchNorm running mean, shape ``(out_channels,)``.
    bn_rv : Tensor
        BatchNorm running variance, shape ``(out_channels,)``.
    bn_eps : float
        BatchNorm numerical-stability epsilon.
    bn_w : Tensor or None
        BatchNorm affine weight :math:`\gamma`, or ``None`` if
        ``affine=False``.
    bn_b : Tensor or None
        BatchNorm affine bias :math:`\beta`, or ``None``.
    transpose : bool, optional
        ``True`` if ``conv_w`` is a transposed-convolution weight
        ``(in_channels, out_channels, *kernel)``, whose output channels are
        in dim 1.  Default ``False``.

    Returns
    -------
    (Parameter, Parameter)
        Fused ``(weight, bias)`` on the dtype / device of ``conv_w``; the
        weight keeps ``conv_w.requires_grad``.

    Raises
    ------
    ValueError
        If the BatchNorm's feature count is not the convolution's output
        channel count.

    Notes
    -----
    Identical math to :func:`fuse_conv_bn_eval`:

    .. math::

        \mathbf{W}_{\text{fused}} = \mathbf{W}
            \cdot \gamma / \sqrt{\sigma^2 + \epsilon}, \qquad
        b_{\text{fused}} = \gamma (b - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import fuse_conv_bn_weights
    >>> conv = nn.Conv2d(3, 16, 3); bn = nn.BatchNorm2d(16)
    >>> W, b = fuse_conv_bn_weights(
    ...     conv.weight, conv.bias,
    ...     bn.running_mean, bn.running_var, bn.eps,
    ...     bn.weight, bn.bias,
    ... )
    >>> W.shape, b.shape
    ((16, 3, 3, 3), (16,))
    """
    state = _BatchNormState(bn_rm, bn_rv, bn_eps, bn_w, bn_b)
    return _fuse_conv_tensors(conv_w, conv_b, state, transpose, 1)


def fuse_linear_bn_eval[LinearT: Linear](linear: LinearT, bn: _BatchNormBase) -> LinearT:
    r"""Fold a BatchNorm1d into the preceding Linear weights (inference-only).

    The 1-D analogue of :func:`fuse_conv_bn_eval` — absorbs the BN's
    eval-time affine transform into a Linear layer's weight and bias.
    The fused linear computes ``BN(Linear(x))`` exactly while using one
    fewer kernel.

    Parameters
    ----------
    linear : Linear
        Linear layer to absorb the BN into.
    bn : BatchNorm1d
        BatchNorm whose feature dimension matches ``linear.out_features``
        (or is ``1``, which broadcasts).  Must be in eval mode and keep
        running statistics.

    Returns
    -------
    Module
        Deep copy of ``linear`` with fused parameters.  Originals are
        not mutated.

    Raises
    ------
    TypeError
        If ``linear`` is not :class:`~lucid.nn.modules.linear.Linear` or
        ``bn`` is not a BatchNorm.
    AssertionError
        If ``linear`` or ``bn`` is in training mode, ``bn`` keeps no running
        statistics, or its feature count is neither ``linear.out_features``
        nor ``1``.

    Notes
    -----
    Math is identical to the conv case — the Linear's weight matrix is
    scaled row-wise by :math:`\gamma / \sqrt{\sigma^2 + \epsilon}` and
    the bias absorbs the mean shift and BN bias.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import fuse_linear_bn_eval
    >>> linear = nn.Linear(128, 64); bn = nn.BatchNorm1d(64)
    >>> _ = linear.eval(); _ = bn.eval()
    >>> fused = fuse_linear_bn_eval(linear, bn)
    >>> x = lucid.randn(4, 128)
    >>> (fused(x) - bn(linear(x))).abs().max().item() < 1e-4
    True
    """
    if not isinstance(linear, Linear):
        raise TypeError(f"fuse_linear_bn_eval: linear must be Linear, got {type(linear).__name__}")
    if not isinstance(bn, _BatchNormBase):
        raise TypeError(f"fuse_linear_bn_eval: bn must be a BatchNorm, got {type(bn).__name__}")
    state = _BatchNormState.of(bn, linear)
    if linear.out_features != bn.num_features and bn.num_features != 1:
        raise AssertionError(
            f"To fuse, linear.out_features == bn.num_features or bn.num_features == 1, "
            f"got linear.out_features={linear.out_features} and bn.num_features={bn.num_features}"
        )

    fused = _copy_module.deepcopy(linear)
    fused.weight, fused.bias = _fuse_linear_tensors(fused.weight, fused.bias, state)
    return fused


def fuse_linear_bn_weights(
    linear_w: Tensor,
    linear_b: Tensor | None,
    bn_rm: Tensor,
    bn_rv: Tensor,
    bn_eps: float,
    bn_w: Tensor | None,
    bn_b: Tensor | None,
) -> tuple[Parameter, Parameter]:
    r"""Low-level form of :func:`fuse_linear_bn_eval` operating on raw tensors.

    Internally identical to :func:`fuse_conv_bn_weights` — a Linear's
    weight is just a 2-D conv weight from the BN's perspective.
    Exposed under its own name to keep call sites self-documenting in
    graph-rewrite passes.

    Parameters
    ----------
    linear_w : Tensor
        Linear weight matrix, shape ``(out_features, in_features)``.
    linear_b : Tensor or None
        Linear bias, shape ``(out_features,)``, or ``None``.
    bn_rm : Tensor
        BatchNorm1d running mean.
    bn_rv : Tensor
        BatchNorm1d running variance.
    bn_eps : float
        BatchNorm numerical-stability epsilon.
    bn_w : Tensor or None
        BatchNorm affine :math:`\gamma`.
    bn_b : Tensor or None
        BatchNorm affine :math:`\beta`.

    Returns
    -------
    (Parameter, Parameter)
        Fused ``(weight, bias)`` on the dtype / device of ``linear_w``.

    Raises
    ------
    ValueError
        If the BatchNorm's feature count is neither ``linear_w.shape[0]``
        nor ``1``.

    Notes
    -----
    Math:

    .. math::

        \mathbf{W}_{\text{fused}} = \mathbf{W}
            \cdot \gamma / \sqrt{\sigma^2 + \epsilon}, \qquad
        b_{\text{fused}} = \gamma (b - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import fuse_linear_bn_weights
    >>> linear = nn.Linear(128, 64); bn = nn.BatchNorm1d(64)
    >>> W, b = fuse_linear_bn_weights(
    ...     linear.weight, linear.bias,
    ...     bn.running_mean, bn.running_var, bn.eps,
    ...     bn.weight, bn.bias,
    ... )
    >>> W.shape, b.shape
    ((64, 128), (64,))
    """
    return _fuse_linear_tensors(
        linear_w, linear_b, _BatchNormState(bn_rm, bn_rv, bn_eps, bn_w, bn_b)
    )
