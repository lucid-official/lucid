"""
lucid.nn.utils.fusion — inference-time layer fusion utilities.

The canonical entry point is ``fuse_conv_bn_eval(conv, bn)`` which
returns a new Conv2d-like module with the BatchNorm's running
statistics + affine parameters absorbed into the conv's weight and
bias.  The result computes exactly the same thing as ``bn(conv(x))``
*at eval time* but with one fewer kernel launch.

Only valid in evaluation mode (when BN uses its frozen running
statistics rather than batch statistics).  Calling this on a training
graph silently produces incorrect results — guard with ``module.eval()``.
"""

import copy as _copy_module
from typing import TYPE_CHECKING

import lucid as _lucid
from lucid.nn.modules.conv import Conv1d, Conv2d, Conv3d
from lucid.nn.modules.linear import Linear
from lucid.nn.modules.normalization import BatchNorm1d, BatchNorm2d, BatchNorm3d

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


_CONV_TYPES = (Conv1d, Conv2d, Conv3d)
_BN_TYPES = (BatchNorm1d, BatchNorm2d, BatchNorm3d)


def fuse_conv_bn_eval(conv: object, bn: object) -> object:
    r"""Fold a BatchNorm layer into the preceding Conv weights (inference-only).

    Because BN at eval time is an affine map with frozen statistics, the
    composition ``BN(Conv(x))`` is itself a single convolution — the BN
    can be analytically absorbed into the conv's weight and bias.  The
    fused module computes exactly the same output but with one fewer
    kernel launch and one fewer allocation per call.  Standard step in
    deployment / quantisation pipelines.

    Parameters
    ----------
    conv : Conv1d | Conv2d | Conv3d
        Convolution module whose weight / bias will absorb the BN.
    bn : BatchNorm1d | BatchNorm2d | BatchNorm3d
        BatchNorm whose rank matches ``conv``.  Must be in eval mode (or
        otherwise be using its frozen ``running_mean`` /
        ``running_var``); calling on a training-mode graph silently
        yields wrong outputs.

    Returns
    -------
    Module
        Deep copy of ``conv`` with fused parameters.  Originals are not
        mutated.

    Raises
    ------
    TypeError
        If ``conv`` or ``bn`` is not one of the supported types.

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
    if not isinstance(conv, _CONV_TYPES):
        raise TypeError(f"fuse_conv_bn_eval: conv must be Conv1d/2d/3d, got {type(conv).__name__}")
    if not isinstance(bn, _BN_TYPES):
        raise TypeError(f"fuse_conv_bn_eval: bn must be BatchNorm1d/2d/3d, got {type(bn).__name__}")

    fused = _copy_module.deepcopy(conv)

    # Pull out the relevant tensors as detached leaves on the conv's device.
    eps = float(bn.eps)
    running_mean = bn.running_mean.detach()
    running_var = bn.running_var.detach()
    gamma = bn.weight.detach() if bn.affine and bn.weight is not None else None
    beta = bn.bias.detach() if bn.affine and bn.bias is not None else None

    # scale = γ / sqrt(σ² + ε)  (γ defaults to 1 if affine is off)
    inv_std = (running_var + eps).rsqrt()
    scale = inv_std if gamma is None else gamma * inv_std

    # Broadcast scale to weight shape: (out_channels, 1, 1, ...).
    weight = fused.weight.detach()
    out_channels = int(weight.shape[0])
    scale_view = scale.reshape(out_channels, *([1] * (weight.ndim - 1)))
    new_weight = weight * scale_view

    # bias: start from (-running_mean) if conv had no bias.
    old_bias: Tensor
    if fused.bias is not None:
        old_bias = fused.bias.detach()
    else:
        old_bias = _lucid.zeros(out_channels, dtype=weight.dtype, device=weight.device)
    new_bias = (old_bias - running_mean) * scale
    if beta is not None:
        new_bias = new_bias + beta

    # Write back as parameters on the fused module.
    from lucid.nn.parameter import Parameter

    fused.weight = Parameter(new_weight)
    if fused.bias is None:
        fused.bias = Parameter(new_bias)
    else:
        fused.bias = Parameter(new_bias)
    return fused


def fuse_conv_bn_weights(
    conv_w: Tensor,
    conv_b: Tensor | None,
    bn_rm: Tensor,
    bn_rv: Tensor,
    bn_eps: float,
    bn_w: Tensor | None,
    bn_b: Tensor | None,
) -> tuple[Tensor, Tensor]:
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

    Returns
    -------
    (Tensor, Tensor)
        Fused ``(weight, bias)`` on the same dtype / device as ``conv_w``.

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
    out_channels = int(conv_w.shape[0])
    inv_std = (bn_rv + bn_eps).rsqrt()
    scale = inv_std if bn_w is None else bn_w * inv_std
    scale_view = scale.reshape(out_channels, *([1] * (conv_w.ndim - 1)))
    new_weight = conv_w * scale_view

    bias = (
        conv_b
        if conv_b is not None
        else _lucid.zeros(out_channels, dtype=conv_w.dtype, device=conv_w.device)
    )
    new_bias = (bias - bn_rm) * scale
    if bn_b is not None:
        new_bias = new_bias + bn_b
    return new_weight, new_bias


def fuse_linear_bn_eval(linear: object, bn: object) -> object:
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
        BatchNorm whose feature dimension matches ``linear.out_features``.
        Must be in eval mode (or otherwise be using its frozen
        statistics).

    Returns
    -------
    Module
        Deep copy of ``linear`` with fused parameters.  Originals are
        not mutated.

    Raises
    ------
    TypeError
        If ``linear`` is not :class:`~lucid.nn.modules.linear.Linear` or
        ``bn`` is not :class:`~lucid.nn.modules.normalization.BatchNorm1d`.

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
    if not isinstance(bn, BatchNorm1d):
        raise TypeError(f"fuse_linear_bn_eval: bn must be BatchNorm1d, got {type(bn).__name__}")

    from lucid.nn.parameter import Parameter

    fused = _copy_module.deepcopy(linear)
    eps = float(bn.eps)
    rm = bn.running_mean.detach()
    rv = bn.running_var.detach()
    gamma = bn.weight.detach() if bn.affine and bn.weight is not None else None
    beta = bn.bias.detach() if bn.affine and bn.bias is not None else None

    conv_b = fused.bias.detach() if fused.bias is not None else None
    new_w, new_b = fuse_conv_bn_weights(fused.weight.detach(), conv_b, rm, rv, eps, gamma, beta)
    fused.weight = Parameter(new_w)
    fused.bias = Parameter(new_b)
    return fused


def fuse_linear_bn_weights(
    linear_w: Tensor,
    linear_b: Tensor | None,
    bn_rm: Tensor,
    bn_rv: Tensor,
    bn_eps: float,
    bn_w: Tensor | None,
    bn_b: Tensor | None,
) -> tuple[Tensor, Tensor]:
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
    (Tensor, Tensor)
        Fused ``(weight, bias)`` on the same dtype / device as
        ``linear_w``.

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
    return fuse_conv_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b)
