"""
nn.functional activation functions.

The softmax family, the distance functions and the shrinkage operators are
written in their own private modules and re-exported from here.
"""

from typing import TYPE_CHECKING

import lucid as _l
from lucid._C import engine as _C_engine
from lucid._dispatch import _float_operand, _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry, _autocast_suspended
from lucid._ops._dims import _dim_size, _wrap_dim
from lucid._types import GeluApproximate
from lucid.nn.functional._distance import (
    cosine_similarity as cosine_similarity,
    pairwise_distance as pairwise_distance,
)
from lucid.nn.functional._inplace import _in_place, _keep_in_graph
from lucid.nn.functional._shrinkage import (
    hardshrink as hardshrink,
    softshrink as softshrink,
    tanhshrink as tanhshrink,
)
from lucid.nn.functional._softmax import (
    gumbel_softmax as gumbel_softmax,
    log_softmax as log_softmax,
    softmax as softmax,
    softmin as softmin,
    straight_through as straight_through,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the rectified linear unit function element-wise.

    The simplest and most widely-used activation function in modern deep
    learning.  Zeros out negative inputs and passes positive inputs
    through unchanged — a piecewise-linear sparsifier that gives neural
    networks their nonlinearity without saturating gradients for large
    positive activations.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Element-wise ReLU output, same shape and dtype as ``input``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{ReLU}(x) = \max(0,\, x)

    Gradient is the indicator :math:`\partial \text{ReLU}/\partial x =
    \mathbb{1}_{x > 0}` — zero for negative inputs, one for positive,
    undefined at exactly zero (convention: 0).  This causes the **dying
    ReLU** problem: a neuron that always produces negative pre-activations
    receives no gradient and stops learning.  Variants like
    :func:`leaky_relu`, :func:`elu`, :func:`gelu`, and :func:`silu`
    address this by allowing a small (or smooth) gradient for negative
    inputs.

    Despite its simplicity, ReLU remains a strong default for CNN
    backbones; transformers commonly prefer :func:`gelu` for its smooth
    non-monotonic shape.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import relu
    >>> relu(lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    tensor([0., 0., 0., 1., 2.])
    """
    if not inplace:
        return _wrap(_C_engine.relu(_unwrap(input)))
    if input.requires_grad and _C_engine.grad_enabled():
        return _in_place(input, "relu_", relu)
    # No graph to extend: the engine's fused kernel allocates nothing (0.44 ms
    # against 1.51 ms through the graph-preserving path on 64x3x128x128).
    _C_engine.relu_(_unwrap(input))
    return input


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    r"""Leaky rectified linear unit activation.

    A simple modification of :func:`relu` that lets a small, non-zero gradient
    pass through for negative inputs.  This avoids the *dying ReLU* problem
    where neurons stuck on the negative side stop updating because their
    gradient is exactly zero.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is applied element-wise.
    negative_slope : float, optional
        Slope :math:`\alpha` of the linear branch for negative inputs.
        Default ``0.01``.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{LeakyReLU}(x) = \max(0, x) + \alpha \min(0, x)

    The derivative is :math:`1` for :math:`x > 0` and :math:`\alpha` for
    :math:`x \le 0`.  Unlike :func:`relu`, the derivative is everywhere
    non-zero (assuming :math:`\alpha \ne 0`), so gradient signal continues
    to flow through inactive units.  Use Leaky ReLU when ReLU networks are
    showing many permanently dead neurons; for a learnable slope, see
    :func:`prelu`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import leaky_relu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0])
    >>> leaky_relu(x, negative_slope=0.1)
    tensor([-0.2, -0.1, 0., 1.])
    """
    if inplace:
        return _in_place(input, "leaky_relu_", lambda t: leaky_relu(t, negative_slope))
    return _wrap(_C_engine.leaky_relu(_unwrap(input), negative_slope))


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""Exponential linear unit activation.

    A smooth alternative to :func:`relu` that saturates to :math:`-\alpha` on
    the negative side instead of clipping at zero.  The exponential branch
    pushes the mean activation closer to zero, which speeds up training by
    reducing internal covariate shift (Clevert et al. 2015).

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    alpha : float, optional
        Saturation value :math:`\alpha > 0` for negative inputs.
        Default ``1.0``.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{ELU}(x) = \begin{cases}
            x & x > 0 \\
            \alpha (e^x - 1) & x \le 0
        \end{cases}

    Continuous and once-differentiable at the origin (derivative is
    :math:`1` from above and :math:`\alpha` from below — equal when
    :math:`\alpha = 1`).  Unlike ReLU it produces negative outputs, which
    helps push activation means toward zero; unlike Leaky ReLU it bounds
    the negative tail, providing implicit noise robustness.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import elu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0])
    >>> elu(x)
    tensor([-0.8647, -0.6321, 0., 1.])
    """
    return (
        _in_place(input, "elu_", lambda t: elu(t, alpha))
        if inplace
        else _wrap(_C_engine.elu(_unwrap(input), alpha))
    )


def selu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Scaled exponential linear unit activation.

    Self-normalising activation from Klambauer et al. (2017): when used in
    a fully-connected network with LeCun-normal initialised weights, the
    fixed scale :math:`\lambda` and slope :math:`\alpha` drive the
    activations toward zero-mean, unit-variance fixed points without the
    need for explicit batch / layer normalisation.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{SELU}(x) = \lambda \, \text{ELU}(x, \alpha) =
        \lambda \begin{cases}
            x & x > 0 \\
            \alpha (e^x - 1) & x \le 0
        \end{cases}

    with the (non-tunable) constants

    .. math::

        \lambda \approx 1.0507, \qquad \alpha \approx 1.6733,

    chosen so that the activation has a zero-mean unit-variance attractor
    fixed point.  Pair with ``alpha_dropout`` and LeCun-normal weight init
    to retain the self-normalising property.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import selu
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> selu(x)
    tensor([-1.111, 0., 1.051])
    """
    return _in_place(input, "selu_", selu) if inplace else _wrap(_C_engine.selu(_unwrap(input)))


def gelu(input: Tensor, *, approximate: GeluApproximate = "none") -> Tensor:
    r"""Gaussian Error Linear Unit activation.

    Smooth, non-monotonic activation that has largely replaced ReLU in
    transformer architectures.  Approximates :math:`x \cdot \mathbb{1}_{x>0}`
    but is differentiable everywhere and lets a small negative signal pass
    through, which improves gradient flow in deep networks.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    approximate : str, optional
        Either ``"none"`` (default, exact formula via ``erf``) or
        ``"tanh"`` (faster polynomial approximation used in BERT /
        Hendrycks 2016).

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    Exact form:

    .. math::

        \text{GELU}(x) = x \, \Phi(x)
        = \frac{x}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]

    where :math:`\Phi` is the standard normal CDF.  The ``"tanh"``
    approximation is

    .. math::

        \text{GELU}(x) \approx \frac{x}{2}\left[1 +
        \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\, x^3)\right)\right].

    Unlike ReLU, GELU has a non-zero gradient everywhere — useful for
    training very deep transformer stacks.  Introduced by
    Hendrycks & Gimpel (2016) and adopted widely after BERT.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import gelu
    >>> x = lucid.tensor([-1.0, 0.0, 1.0, 2.0])
    >>> gelu(x)
    tensor([-0.1587, 0., 0.8413, 1.954])
    """
    xi = _unwrap(input)
    return _wrap(_C_engine.gelu(xi) if approximate == "tanh" else _C_engine.gelu_exact(xi))


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Sigmoid Linear Unit (a.k.a. Swish-1) activation.

    Smooth, non-monotonic activation popularised by Ramachandran et al.
    (2017) as the result of a neural architecture search over activation
    functions.  Equivalent to Swish with :math:`\beta = 1`.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{SiLU}(x) = x \, \sigma(x) = \frac{x}{1 + e^{-x}}

    Has a small negative bump near :math:`x \approx -1.28` (minimum value
    :math:`\approx -0.278`); approaches 0 from below for large negative
    input and approaches the identity for large positive input.  The
    derivative is

    .. math::

        \text{SiLU}'(x) = \sigma(x) + x\sigma(x)(1 - \sigma(x)),

    which is smooth and unbounded above.  Often outperforms ReLU in deep
    CNNs and is the default in EfficientNet and many MoE blocks.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import silu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> silu(x)
    tensor([-0.2384, -0.2689, 0., 0.7311, 1.762])
    """
    return _in_place(input, "silu_", silu) if inplace else _wrap(_C_engine.silu(_unwrap(input)))


def mish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Mish activation function.

    Self-regularising non-monotonic activation introduced by Misra (2019).
    Like :func:`silu` it has a small negative dip and approaches the
    identity for large positive inputs, but its tanh-of-softplus form gives
    slightly smoother gradients which tend to help convergence in CV tasks
    (YOLOv4 uses it throughout).

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{Mish}(x) = x \, \tanh\!\big(\text{softplus}(x)\big)
                      = x \, \tanh\!\big(\ln(1 + e^x)\big)

    :math:`C^\infty`-smooth, non-monotonic, unbounded above and bounded
    below (minimum :math:`\approx -0.3088`).  Empirically outperforms
    ReLU/Swish on some object-detection benchmarks at modest extra cost.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import mish
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> mish(x)
    tensor([-0.2525, -0.3034, 0., 0.8651, 1.944])
    """
    return _in_place(input, "mish_", mish) if inplace else _wrap(_C_engine.mish(_unwrap(input)))


def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Hard Swish activation — piecewise linear approximation of :func:`silu`.

    Introduced in MobileNetV3 (Howard et al. 2019) as a compute-friendly
    replacement for the sigmoid in Swish: the exponential is approximated
    by a clipped linear function, which is far cheaper on mobile / quantised
    hardware while preserving the shape of the activation.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{HardSwish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}
        = \begin{cases}
            0 & x \le -3 \\
            x(x+3)/6 & -3 < x < 3 \\
            x & x \ge 3
        \end{cases}

    Continuous but only piecewise-:math:`C^1` (the second derivative jumps
    at :math:`x = \pm 3`).  Cheap to evaluate and to quantise — preferred
    over :func:`silu` in mobile architectures.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hardswish
    >>> x = lucid.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
    >>> hardswish(x)
    tensor([-0., -0.3333, 0., 0.6667, 4.])
    """
    return (
        _in_place(input, "hardswish_", hardswish)
        if inplace
        else _wrap(_C_engine.hard_swish(_unwrap(input)))
    )


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Hard sigmoid — piecewise linear approximation of :func:`sigmoid`.

    Avoids the cost of an exponential by clipping a scaled-and-shifted
    identity to :math:`[0, 1]`.  Widely used as the gating function inside
    quantised mobile networks (MobileNetV3 Squeeze-and-Excitation blocks).

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``, values in
        :math:`[0, 1]`.

    Notes
    -----
    .. math::

        \text{HardSigmoid}(x) = \begin{cases}
            0 & x \le -3 \\
            x/6 + 1/2 & -3 < x < 3 \\
            1 & x \ge 3
        \end{cases}

    Matches :func:`sigmoid` exactly at :math:`x = 0` (both return ``0.5``)
    but is only piecewise-linear.  Derivative is :math:`1/6` in the linear
    region and :math:`0` in the saturated regions.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hardsigmoid
    >>> x = lucid.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
    >>> hardsigmoid(x)
    tensor([0., 0.3333, 0.5, 0.6667, 1.])
    """
    return (
        _in_place(input, "hardsigmoid_", hardsigmoid)
        if inplace
        else _wrap(_C_engine.hard_sigmoid(_unwrap(input)))
    )


def sigmoid(input: Tensor) -> Tensor:
    r"""Logistic sigmoid activation.

    Squashes any real-valued input into the open interval :math:`(0, 1)`.
    Historically the canonical activation for binary classification heads
    and the gating function in LSTM / GRU cells.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``, values in
        :math:`(0, 1)`.

    Notes
    -----
    .. math::

        \sigma(x) = \frac{1}{1 + e^{-x}}

    Derivative :math:`\sigma'(x) = \sigma(x)(1 - \sigma(x)) \le 0.25`.
    The bounded derivative is the classical cause of the vanishing
    gradient problem in deep feed-forward stacks, so for hidden layers
    prefer ReLU / GELU / SiLU.  For binary classification logits, use
    :func:`logsigmoid` to compute :math:`\log\sigma(x)` in a numerically
    stable way.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import sigmoid
    >>> x = lucid.tensor([-2.0, 0.0, 2.0])
    >>> sigmoid(x)
    tensor([0.1192, 0.5, 0.8808])
    """
    return _wrap(_C_engine.sigmoid(_float_operand(_unwrap(input))))


def tanh(input: Tensor) -> Tensor:
    r"""Hyperbolic tangent activation.

    A rescaled, recentred sigmoid that maps the real line to
    :math:`(-1, 1)`.  Because outputs are zero-centred, tanh is generally
    preferred over :func:`sigmoid` for hidden layers in shallow networks
    and is the canonical squashing function in RNN cells.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``, values in
        :math:`(-1, 1)`.

    Notes
    -----
    .. math::

        \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
                 = 2\sigma(2x) - 1

    Derivative :math:`1 - \tanh^2(x)`, maximum value :math:`1` at the
    origin.  Saturates for :math:`|x| \gtrsim 3`; in deep networks this
    leads to vanishing gradients, so modern feed-forward stacks prefer
    ReLU-family activations.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import tanh
    >>> x = lucid.tensor([-2.0, 0.0, 2.0])
    >>> tanh(x)
    tensor([-0.964, 0., 0.964])
    """
    return _wrap(_C_engine.tanh(_float_operand(_unwrap(input))))


def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    r"""Smooth approximation of :func:`relu`.

    Softplus is a strictly positive, :math:`C^\infty` smooth function that
    behaves like :math:`x` for large positive inputs and like :math:`e^x`
    for large negative inputs.  Its derivative is :func:`sigmoid`, which is
    why softplus appears naturally as the log-partition of a Bernoulli and
    as the analytic primitive of the logistic function.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    beta : float, optional
        Sharpness parameter :math:`\beta > 0`.  Larger ``beta`` makes the
        curve approach a hard ReLU.  Default ``1.0``.
    threshold : float, optional
        For numerical stability the formula collapses to the identity
        :math:`x` whenever :math:`\beta x > \text{threshold}` (the
        :math:`e^{\beta x}` term would otherwise overflow).  Default
        ``20.0``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``, strictly positive.

    Notes
    -----
    .. math::

        \text{softplus}_\beta(x) = \frac{1}{\beta}\log\!\big(1 + e^{\beta x}\big)

    Derivative is :math:`\sigma(\beta x)`.  The overflow guard at
    ``threshold`` exploits that for large :math:`\beta x` the function is
    indistinguishable from :math:`x` in finite precision.  The engine's
    bare ``softplus`` kernel implements the special case
    :math:`\beta = 1`, :math:`\text{threshold} \to \infty`; when either
    argument deviates the full formula is composed so autograd still
    flows through the existing backward nodes.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softplus
    >>> x = lucid.tensor([-2.0, 0.0, 2.0])
    >>> softplus(x)
    tensor([0.1269, 0.6931, 2.127])
    """
    if beta == 1.0 and threshold >= 50.0:
        # Hot-path: the bare engine kernel is already correct here.
        return _wrap(_C_engine.softplus(_unwrap(input)))
    with _autocast_entry("softplus", _unwrap(input)) as (xi,):
        beta_t = _C_engine.full(list(xi.shape), float(beta), xi.dtype, xi.device)
        threshold_t = _C_engine.full(list(xi.shape), float(threshold), xi.dtype, xi.device)
        bx = _C_engine.mul(xi, beta_t)
        # log1p(exp(bx)) / beta — composed because the engine's ``softplus`` is beta-agnostic.
        inv_beta = _C_engine.full(list(xi.shape), 1.0 / float(beta), xi.dtype, xi.device)
        softplus_full = _C_engine.mul(_C_engine.softplus(bx), inv_beta)
        return _wrap(_C_engine.where(_C_engine.greater(bx, threshold_t), xi, softplus_full))


def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Rectified linear unit clipped at six.

    Introduced for quantisation-friendly mobile architectures
    (MobileNetV1 / V2): bounding ReLU above keeps activations inside a
    small fixed range so that low-precision integer arithmetic does not
    overflow.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``, values in
        :math:`[0, 6]`.

    Notes
    -----
    .. math::

        \text{ReLU6}(x) = \min(\max(0, x), 6)

    Derivative is :math:`1` on :math:`(0, 6)` and :math:`0` outside —
    so beyond suffering from "dead neurons" on the negative side, ReLU6
    also kills gradients above :math:`6`.  The trade-off is worthwhile
    for int8 quantisation where the symmetric :math:`[0, 6]` range maps
    cleanly to a small dynamic-range integer codebook.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import relu6
    >>> x = lucid.tensor([-1.0, 0.0, 3.0, 7.0])
    >>> relu6(x)
    tensor([0., 0., 3., 6.])
    """
    return _in_place(input, "relu6_", relu6) if inplace else _wrap(_C_engine.relu6(_unwrap(input)))


def glu(input: Tensor, dim: int = -1) -> Tensor:
    r"""Gated Linear Unit (Dauphin et al. 2017).

    Splits the input in half along ``dim`` and multiplies the first half
    element-wise by the sigmoid of the second.  Acts as a learnable gate
    that lets the network modulate information flow per channel; widely
    used in sequence models and Conformer-style speech architectures.

    Parameters
    ----------
    input : Tensor
        Input tensor.  The size along ``dim`` must be even — it is split
        into two halves :math:`a` and :math:`b` of equal size.
    dim : int, optional
        Dimension along which to split.  Default ``-1``.

    Returns
    -------
    Tensor
        Output tensor whose shape matches ``input`` except along ``dim``,
        where the size is halved.

    Notes
    -----
    .. math::

        \text{GLU}(x) = a \odot \sigma(b),
        \qquad x = [a; b] \text{ split along } \text{dim}

    The sigmoid gate :math:`\sigma(b)` selects how much of :math:`a` to
    pass through.  Because the gate is multiplicative the derivative is
    well-behaved (no saturation in :math:`a`'s direction), which helps
    train deeper stacks than a plain feed-forward layer would allow.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import glu
    >>> x = lucid.tensor([[1.0, 2.0, 0.0, 1.0]])  # split → a=[1,2], b=[0,1]
    >>> glu(x, dim=-1)
    tensor([[0.5, 1.462]])
    """
    impl = _unwrap(input)
    dim = _wrap_dim(dim, len(impl.shape), "glu")
    size = _dim_size(impl.shape, dim)
    if size % 2:  # unequal halves broadcast into a silently wrong product
        raise ValueError(f"glu: dim {dim} must have an even size, got {size}")
    first, second = _C_engine.split_at(impl, [size // 2], dim)
    return _wrap(_C_engine.mul(first, _C_engine.sigmoid(second)))


def prelu(input: Tensor, weight: Tensor) -> Tensor:
    r"""Parametric Rectified Linear Unit (He et al. 2015).

    Variant of :func:`leaky_relu` whose negative slope is a **learnable**
    parameter rather than a fixed hyperparameter.  In practice ``weight``
    may be a scalar (shared across channels) or a per-channel vector,
    matching the convention of the reference framework's ``nn.PReLU``.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    weight : Tensor
        Learnable slope :math:`\alpha`.  Either a scalar tensor (slope
        shared across all elements) or a 1-D tensor broadcastable against
        ``input`` along the channel dimension.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{PReLU}(x) = \max(0, x) + \alpha \min(0, x)

    Allows the optimiser to discover the *optimal* negative-side slope
    per channel; the original paper showed this reaches super-human
    ImageNet accuracy.  When :math:`\alpha = 0` recovers :func:`relu`; for
    a fixed non-zero :math:`\alpha` see :func:`leaky_relu`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import prelu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0])
    >>> w = lucid.tensor(0.25)
    >>> prelu(x, w)
    tensor([-0.5, -0.25, 0., 1.])
    """
    with _autocast_entry("prelu", _unwrap(input), _unwrap(weight)) as (xi, wi):
        # A per-channel ``(C,)`` slope belongs to dim 1 of an ``(N, C, *)`` input.
        # Left as a bare vector it would broadcast against the *last* dim instead,
        # which fails on a feature map (or mixes up channels when that dim is C).
        if len(wi.shape) == 1 and wi.shape[0] > 1 and len(xi.shape) > 2:
            wi = _C_engine.reshape(wi, [wi.shape[0]] + [1] * (len(xi.shape) - 2))
        pos = _C_engine.relu(xi)
        neg_part = _C_engine.mul(
            wi, _C_engine.minimum(_C_engine.zeros(xi.shape, xi.dtype, xi.device), xi)
        )
        return _wrap(_C_engine.add(pos, neg_part))


def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""Continuously-differentiable Exponential Linear Unit (Barron 2017).

    A reparameterisation of :func:`elu` whose first derivative is
    continuous at the origin for *every* :math:`\alpha > 0`, not only the
    specific value :math:`\alpha = 1`.  This makes CELU friendlier to
    optimisers that exploit smooth gradients.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.
    alpha : float, optional
        Saturation value :math:`\alpha > 0` for negative inputs.
        Default ``1.0``.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{CELU}(x) = \max(0, x) + \min\!\big(0, \alpha (e^{x/\alpha} - 1)\big)

    At the origin both branches have derivative :math:`1` regardless of
    :math:`\alpha`, removing the kink ELU has when :math:`\alpha \ne 1`.
    Reduces to ELU when :math:`\alpha = 1`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import celu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0])
    >>> celu(x, alpha=1.0)
    tensor([-0.8647, -0.6321, 0., 1.])
    """
    if alpha == 0.0:
        raise _C_engine.InvalidArgument("ZeroDivisionError: alpha cannot be 0 for CELU")
    if inplace:
        return _in_place(input, "celu_", lambda t: celu(t, alpha))
    xi = _unwrap(input)
    if alpha == 1.0:
        return _wrap(_C_engine.elu(xi, 1.0))
    a_t = _C_engine.full(xi.shape, alpha, xi.dtype, xi.device)
    return _wrap(_C_engine.mul(_C_engine.elu(_C_engine.div(xi, a_t), 1.0), a_t))


# ── P3 fills: missing activations ──────────────────────────────────────────


def hardtanh(
    input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False
) -> Tensor:
    r"""Hardtanh — element-wise clamp to :math:`[\text{min\_val}, \text{max\_val}]`.

    A cheap piecewise-linear surrogate for :func:`tanh` whose forward and
    backward passes require no transcendental functions.  Standard
    activation in compact / quantised RNN cells.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape.
    min_val : float, optional
        Lower clamp.  Default ``-1.0``.
    max_val : float, optional
        Upper clamp.  Default ``1.0``.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Clamped tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & x > \text{max\_val} \\
            x & \text{min\_val} \le x \le \text{max\_val} \\
            \text{min\_val} & x < \text{min\_val}
        \end{cases}

    Derivative is :math:`1` inside the linear region and :math:`0` in the
    saturated regions — gradients stop flowing through saturated units, a
    drawback compared to smooth tanh but acceptable when the linear region
    covers the bulk of activations.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hardtanh
    >>> x = lucid.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    >>> hardtanh(x)
    tensor([-1., -0.5, 0., 0.5, 1.])
    """
    if min_val > max_val:
        raise ValueError("min_val cannot be greater than max_val")
    if inplace:
        return _in_place(input, "hardtanh_", lambda t: hardtanh(t, min_val, max_val))
    inside = _unwrap((input > min_val) & (input < max_val))
    clamped = _C_engine.clip(_unwrap(input.detach()), min_val, max_val)
    return _keep_in_graph(_wrap(_C_engine.where(inside, _unwrap(input), clamped)), input)


def logsigmoid(input: Tensor) -> Tensor:
    r"""Numerically stable :math:`\log \sigma(x)`.

    Direct evaluation of :math:`\log(1 / (1 + e^{-x}))` underflows once
    :math:`\sigma(x)` is below machine epsilon (around :math:`x \lesssim -37`
    for float32).  Rewriting as :math:`-\operatorname{softplus}(-x)`
    matches the same value while staying finite for arbitrarily negative
    inputs.

    Parameters
    ----------
    input : Tensor
        Input logits.

    Returns
    -------
    Tensor
        :math:`\log \sigma(x)` with the same shape as ``input``, values in
        :math:`(-\infty, 0]`.

    Notes
    -----
    .. math::

        \log\sigma(x) = -\log(1 + e^{-x}) = -\operatorname{softplus}(-x)

    Frequently the building block of binary cross-entropy with logits
    (BCE-with-logits) and of importance-weighting computations in RL.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import logsigmoid
    >>> x = lucid.tensor([-10.0, 0.0, 10.0])
    >>> logsigmoid(x)
    tensor([-10., -0.6931, -4.54e-05])
    """
    # No row, as the reference's log_sigmoid has none; softplus's FP32
    # Metal row must not apply inside it.
    with _autocast_suspended(_unwrap(input).device):
        return -softplus(-input)


def softsign(input: Tensor) -> Tensor:
    r"""Softsign activation.

    A bounded :math:`(-1, 1)` activation that saturates **polynomially**
    rather than exponentially — its tails are much heavier than
    :func:`tanh`'s, so gradients vanish more slowly in deep networks.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``, values in
        :math:`(-1, 1)`.

    Notes
    -----
    .. math::

        \text{softsign}(x) = \frac{x}{1 + |x|}, \qquad
        \text{softsign}'(x) = \frac{1}{(1 + |x|)^2}

    Derivative decays like :math:`1/x^2` rather than :math:`\tanh`'s
    :math:`1/\cosh^2`, so saturation is gentler.  Lacks the bi-Lipschitz
    smoothness of tanh near zero (kink at :math:`x = 0` from the absolute
    value) but is monotone and zero-centred.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softsign
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> softsign(x)
    tensor([-0.6667, -0.5, 0., 0.5, 0.6667])
    """
    return input / (1.0 + _l.abs(input))


def threshold(input: Tensor, threshold: float, value: float, inplace: bool = False) -> Tensor:
    r"""Threshold activation — gate elements by a scalar cutoff.

    Replaces every element of ``input`` that is **not** strictly greater than
    ``threshold`` with the constant ``value``.  A generalisation of
    :func:`relu` (recovered by ``threshold=0, value=0``) and a common
    component of dead-zone non-linearities.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    threshold : float
        Cutoff :math:`t`.  Elements satisfying :math:`x > t` are kept.
    value : float
        Replacement constant for elements not above the threshold.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.

    Returns
    -------
    Tensor
        Thresholded tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{threshold}(x; t, v) = \begin{cases}
            x & x > t \\
            v & \text{otherwise}
        \end{cases}

    Discontinuous at :math:`x = t` when :math:`v \ne t`; the derivative
    is :math:`1` above the threshold and :math:`0` below.  Useful as a
    deterministic alternative to :func:`relu` when a non-zero "off"
    value is needed (e.g. setting masked positions to a sentinel).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import threshold
    >>> x = lucid.tensor([-1.0, 0.0, 0.5, 1.0])
    >>> threshold(x, threshold=0.5, value=-99.0)
    tensor([-99., -99., -99., 1.])
    """
    return (
        _in_place(input, "threshold_", lambda t: _threshold(t, threshold, value))
        if inplace
        else _threshold(input, threshold, value)
    )


def _threshold(x: Tensor, threshold: float, value: float) -> Tensor:
    # ``x <= t ? v : x``, not ``x > t ? x : v``.  The two are the same
    # function everywhere except at NaN, where *both* comparisons are
    # false — so the first keeps the NaN and the second replaces it with
    # ``value``.  Written the second way, a NaN entering a network became
    # a 0 at the first threshold and the loss went finite with nothing to
    # show for it.  The reference is the first.  (The public function's
    # ``threshold`` argument shadows its own name, hence this helper.)
    replaced = x <= threshold
    replacement = _l.full_like(x, float(value))
    return _keep_in_graph(_l.where(replaced, replacement, x), x)


def rrelu(
    input: Tensor,
    lower: float = 1.0 / 8.0,
    upper: float = 1.0 / 3.0,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    r"""Randomised leaky ReLU (Xu et al. 2015).

    A regularised variant of :func:`leaky_relu`: during **training** every
    negative element is scaled by an i.i.d. uniform slope, injecting
    activation noise on the negative side.  At **evaluation** time the
    expectation of that uniform draw is used, so the function becomes
    deterministic.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    lower : float, optional
        Lower bound of the uniform slope distribution.  Default ``1/8``.
    upper : float, optional
        Upper bound of the uniform slope distribution.  Default ``1/3``.
    training : bool, optional
        If ``True`` sample a fresh slope per element; if ``False`` use the
        midpoint.  Default ``False``.
    inplace : bool, optional
        If ``True``, ``input`` itself receives the result and is returned.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    During training, with :math:`a \sim \mathrm{Uniform}(\text{lower},
    \text{upper})` sampled element-wise:

    .. math::

        \text{RReLU}(x) = \begin{cases}
            x & x \ge 0 \\
            a x & x < 0
        \end{cases}

    During evaluation, :math:`a` is replaced by the expectation
    :math:`(\text{lower} + \text{upper}) / 2`.  The slope randomisation
    acts as a mild noise regulariser — comparable in spirit to dropout
    but applied to activation slope rather than activation magnitude.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import rrelu
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> rrelu(x, training=False)
    tensor([-0.2292, 0., 1.])
    """
    if inplace:
        return _in_place(input, "rrelu_", lambda t: rrelu(t, lower, upper, training))
    if not training:
        return leaky_relu(input, 0.5 * (lower + upper), inplace=False)
    slope: Tensor = _l.rand_like(input) * (upper - lower) + lower
    return _keep_in_graph(_l.where(input > 0, input, slope * input), input)


# ── In-place variants ─────────────────────────────────────────────────────────
# Each is its functional with ``inplace`` passed on (default ``True``), so
# ``F.elu_(x)`` and ``F.elu(x, inplace=True)`` are one call: ``_in_place``.


def relu_(input: Tensor, inplace: bool = True) -> Tensor:
    """In-place ReLU — ``input`` is overwritten with ``max(0, input)``.

    Backed by the engine's native in-place ReLU kernel when no autograd
    graph has to be extended (no temporary buffer).

    Parameters
    ----------
    input : Tensor
        Input; modified in place and also returned.
    inplace : bool, optional
        ``False`` returns :func:`relu`'s fresh result instead.  Default ``True``.

    Returns
    -------
    Tensor
        The same ``input`` tensor, now holding ``max(0, input)``.
    """
    return relu(input, inplace)


def elu_(input: Tensor, alpha: float = 1.0, scale: float = 1.0, input_scale: float = 1.0) -> Tensor:
    r"""In-place ELU — overwrites ``input`` with :math:`\text{ELU}(\text{input}, \alpha)`.

    Parameters
    ----------
    input : Tensor
        Input; modified in place.
    alpha : float, optional
        Negative-side scale parameter :math:`\alpha`.  Default ``1.0``.
    scale : float, optional
        Output scale factor.  Default ``1.0``.
    input_scale : float, optional
        Negative-side exponent scale factor.  Default ``1.0``.

    Returns
    -------
    Tensor
        The same ``input`` tensor, now holding ``elu(input, alpha)``.
    """
    if scale == 1.0 and input_scale == 1.0:
        return elu(input, alpha, True)
    return _in_place(
        input,
        "elu_",
        lambda t: _l.where(t > 0, t * scale, (alpha * scale) * (_l.exp(t * input_scale) - 1.0)),
    )


def selu_(input: Tensor, inplace: bool = True) -> Tensor:
    """In-place SELU — overwrites ``input`` with ``selu(input)``.

    Uses the canonical SELU coefficients (Klambauer 2017); intended
    for use only inside Self-Normalising Networks where the input was
    initialised with LeCun-normal and downstream blocks expect the
    SELU fixed-point statistics.

    Parameters
    ----------
    input : Tensor
        Input; modified in place.
    inplace : bool, optional
        ``False`` returns :func:`selu`'s fresh result instead.  Default ``True``.

    Returns
    -------
    Tensor
        The same ``input`` tensor, now holding ``selu(input)``.
    """
    return selu(input, inplace)


def leaky_relu_(input: Tensor, negative_slope: float = 0.01, inplace: bool = True) -> Tensor:
    r"""In-place Leaky ReLU — overwrites ``input`` with the leaky-rectified result.

    .. math::

        y_i = \begin{cases} x_i & x_i \ge 0 \\ \text{negative\_slope} \cdot x_i & x_i < 0 \end{cases}

    Parameters
    ----------
    input : Tensor
        Input; modified in place.
    negative_slope : float, optional
        Slope applied to the negative side.  Default ``0.01``.
    inplace : bool, optional
        ``False`` returns :func:`leaky_relu`'s fresh result instead.
        Default ``True``.

    Returns
    -------
    Tensor
        The same ``input`` tensor, now holding ``leaky_relu(input, negative_slope)``.
    """
    return leaky_relu(input, negative_slope, inplace)


def hardtanh_(
    input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = True
) -> Tensor:
    """In-place hardtanh — clamps ``input`` to ``[min_val, max_val]`` in place.

    Parameters
    ----------
    input : Tensor
        Input; modified in place.
    min_val : float, optional
        Lower clamp bound.  Default ``-1.0``.
    max_val : float, optional
        Upper clamp bound.  Default ``1.0``.
    inplace : bool, optional
        ``False`` returns :func:`hardtanh`'s fresh result instead.
        Default ``True``.

    Returns
    -------
    Tensor
        The same ``input`` tensor, now clamped.
    """
    return hardtanh(input, min_val, max_val, inplace)


def threshold_(input: Tensor, threshold: float, value: float, inplace: bool = True) -> Tensor:
    """In-place threshold — replace entries ``<= threshold`` with ``value``.

    Parameters
    ----------
    input : Tensor
        Input; modified in place.
    threshold : float
        Cutoff; positions where ``input <= threshold`` are reset.
    value : float
        Replacement value for clipped positions.
    inplace : bool, optional
        ``False`` returns :func:`threshold`'s fresh result instead.
        Default ``True``.

    Returns
    -------
    Tensor
        The same ``input`` tensor, with sub-threshold positions set to
        ``value`` and the rest unchanged.
    """
    return (
        _in_place(input, "threshold_", lambda t: _threshold(t, threshold, value))
        if inplace
        else _threshold(input, threshold, value)
    )


def rrelu_(
    input: Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
    inplace: bool = True,
) -> Tensor:
    """In-place RReLU — randomised leaky ReLU written back into ``input``.

    During training the negative-slope is sampled per-element from a
    uniform :math:`U(\\text{lower}, \\text{upper})`; at inference the
    midpoint :math:`(\\text{lower} + \\text{upper}) / 2` is used.

    Parameters
    ----------
    input : Tensor
        Input; modified in place.
    lower : float, optional
        Lower bound of the uniform slope sampler.  Default ``1/8``.
    upper : float, optional
        Upper bound of the uniform slope sampler.  Default ``1/3``.
    training : bool, optional
        When ``False`` (default) uses the deterministic midpoint slope;
        when ``True`` samples a fresh slope per element.
    inplace : bool, optional
        ``False`` returns :func:`rrelu`'s fresh result instead.
        Default ``True``.

    Returns
    -------
    Tensor
        The same ``input`` tensor, now holding ``rrelu(input, lower, upper, training)``.
    """
    return rrelu(input, lower, upper, training, inplace)


def symlog(x: Tensor) -> Tensor:
    r"""Symmetric logarithm — compress magnitude, keep sign and zero.

    .. math::

        \mathrm{symlog}(x) = \mathrm{sign}(x)\,\ln(|x| + 1)

    Parameters
    ----------
    x : Tensor
        Any shape.

    Returns
    -------
    Tensor
        Same shape.

    Notes
    -----
    Behaves like the identity near zero and like a logarithm far from it,
    so a single network can be trained on targets whose scale is not known
    in advance — rewards that are ``0.01`` in one environment and
    ``10000`` in another land in the same range without per-task
    normalisation.  That is what it is for in DreamerV3.

    The inverse is :func:`symexp`, and the pair is exact: ``symexp`` of
    ``symlog`` returns the input.  Implemented through ``log1p`` rather
    than ``log(1 + ·)`` so small inputs keep their precision.

    See Also
    --------
    symexp : The inverse transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> F.symlog(lucid.tensor([-100.0, 0.0, 100.0]))
    tensor([-4.615, 0., 4.615])
    """
    return _l.sign(x) * _l.log1p(_l.abs(x))


def symexp(x: Tensor) -> Tensor:
    r"""Inverse of :func:`symlog`.

    .. math::

        \mathrm{symexp}(x) = \mathrm{sign}(x)\big(\exp(|x|) - 1\big)

    Parameters
    ----------
    x : Tensor
        Any shape.

    Returns
    -------
    Tensor
        Same shape.

    Notes
    -----
    Used to read a prediction back out into the original units — a value
    head trained against ``symlog`` targets predicts in the compressed
    space, and this is what turns that back into a return.

    Uses ``expm1`` rather than ``exp(·) - 1``, which loses precision for
    small arguments exactly where ``symlog`` sends most of its inputs.

    See Also
    --------
    symlog : The forward transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> round(float(F.symexp(F.symlog(lucid.tensor([1234.0]))).item()), 2)
    1234.0
    """
    return _l.sign(x) * _l.expm1(_l.abs(x))
