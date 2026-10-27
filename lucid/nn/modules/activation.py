r"""
Activation function modules for ``lucid.nn``.

Each class wraps a functional counterpart from
:mod:`lucid.nn.functional` as a stateful :class:`~lucid.nn.Module`,
making it composable with :class:`~lucid.nn.Sequential` and other
container modules.  Stateless activations (ReLU, Sigmoid, Tanh, …)
carry no learnable parameters; parametric ones (PReLU, RReLU) register
their slopes as :class:`~lucid.nn.Parameter` instances and are updated
by any standard optimiser.
"""

from typing import override

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike, GeluApproximate
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import full
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.activations import (
    relu,
    leaky_relu,
    elu,
    celu,
    selu,
    gelu,
    silu,
    mish,
    hardswish,
    hardsigmoid,
    sigmoid,
    tanh,
    softmax,
    log_softmax,
    relu6,
    prelu,
    softmin,
    glu,
    hardshrink,
    tanhshrink,
    softshrink,
    softplus,
    rrelu,
    hardtanh,
    threshold,
    cosine_similarity,
    pairwise_distance,
)


class _InplaceActivation(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace


class ReLU(_InplaceActivation):
    r"""Rectified Linear Unit activation function.

    Applies element-wise:

    .. math::

        \text{ReLU}(x) = \max(0,\, x)

    The simplest non-linear activation — sets all negative values to zero.
    Commonly used in hidden layers of deep networks because it avoids the
    vanishing-gradient problem that affects sigmoid and tanh for large inputs,
    and is inexpensive to compute.

    Parameters
    ----------
    inplace : bool, optional
        If ``True``, the input tensor itself receives the result.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.ReLU()
    >>> x = lucid.tensor([-1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([0., 0., 1., 2.])

    >>> # In place: the input itself holds the result
    >>> x = lucid.tensor([-1.0, 2.0])
    >>> nn.ReLU(inplace=True)(x) is x, x
    (True, tensor([0., 2.]))
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return relu(input, self.inplace)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"inplace={self.inplace}" if self.inplace else ""


class LeakyReLU(Module):
    r"""Leaky Rectified Linear Unit activation function.

    Applies element-wise:

    .. math::

        \text{LeakyReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            \alpha \cdot x & \text{otherwise}
        \end{cases}

    Unlike standard ReLU, neurons with negative pre-activations receive a
    small gradient :math:`\alpha` during back-propagation, avoiding the
    "dying ReLU" phenomenon where units become permanently inactive.

    Parameters
    ----------
    negative_slope : float, optional
        Slope :math:`\alpha` applied to negative inputs.  Default: ``0.01``.
    inplace : bool, optional
        If ``True``, modifies the input tensor in-place.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.LeakyReLU(negative_slope=0.1)
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-0.2, -0.1, 0., 1., 2.])

    >>> # Default slope 0.01 — very small leak
    >>> m = nn.LeakyReLU()
    >>> x = lucid.randn(3, 32)
    >>> out = m(x)
    >>> out.shape
    (3, 32)
    """

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        """Initialise the LeakyReLU module. See the class docstring for parameter semantics."""
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return leaky_relu(input, self.negative_slope, self.inplace)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"negative_slope={self.negative_slope}"


class ELU(Module):
    r"""Exponential Linear Unit activation function.

    Applies element-wise:

    .. math::

        \text{ELU}(x) =
        \begin{cases}
            x & \text{if } x > 0 \\
            \alpha \left(e^{x} - 1\right) & \text{otherwise}
        \end{cases}

    ELU smoothly saturates to :math:`-\alpha` for large negative inputs,
    producing mean activations closer to zero than ReLU.  This self-normalising
    tendency can accelerate convergence in deep networks.

    Parameters
    ----------
    alpha : float, optional
        Scale :math:`\alpha` for the negative exponential branch.
        Default: ``1.0``.
    inplace : bool, optional
        If ``True``, modifies the input tensor in-place.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.ELU(alpha=1.0)
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-0.8647, -0.6321, 0., 1., 2.])

    >>> # Custom alpha shifts the negative saturation floor
    >>> m = nn.ELU(alpha=0.5)
    >>> x = lucid.randn(2, 16)
    >>> out = m(x)
    >>> out.shape
    (2, 16)
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        """Initialise the ELU module. See the class docstring for parameter semantics."""
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return elu(input, self.alpha, self.inplace)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"alpha={self.alpha}"


class SELU(_InplaceActivation):
    r"""Scaled Exponential Linear Unit activation function.

    Applies element-wise:

    .. math::

        \text{SELU}(x) =
        \lambda
        \begin{cases}
            x & \text{if } x > 0 \\
            \alpha \left(e^{x} - 1\right) & \text{otherwise}
        \end{cases}

    where :math:`\lambda \approx 1.0507` and :math:`\alpha \approx 1.6733`
    are fixed constants derived from the self-normalising property.  When
    weights are initialised with ``lecun_normal`` and the network uses only
    SELU activations, the mean and variance of each layer's output converge
    to 0 and 1, enabling stable training of very deep fully-connected networks
    without batch normalisation.

    Parameters
    ----------
    inplace : bool, optional
        If ``True``, modifies the input tensor in-place.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Notes
    -----
    The self-normalising guarantee holds strictly only for networks with
    no skip connections, no convolutional layers, and Lecun-normal weight
    initialisation.  In practice SELU is used more broadly as a drop-in
    for ELU.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.SELU()
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-1.52, -1.111, 0., 1.051, 2.101])

    >>> # Suitable for deep fully-connected architectures
    >>> layers = nn.Sequential(nn.Linear(128, 64), nn.SELU(), nn.Linear(64, 10))
    >>> x = lucid.randn(8, 128)
    >>> out = layers(x)
    >>> out.shape
    (8, 10)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return selu(input, self.inplace)


class GELU(Module):
    r"""Gaussian Error Linear Unit activation function.

    Applies element-wise:

    .. math::

        \text{GELU}(x) = x \cdot \Phi(x)

    where :math:`\Phi(x)` is the cumulative distribution function of the
    standard normal distribution.  Intuitively, GELU weights each input by
    the probability that a standard Gaussian random variable is smaller than
    it — inputs far into the positive tail pass through nearly unchanged,
    while those deep in the negative tail are suppressed.

    When ``approximate="tanh"`` the following closed-form approximation is
    used instead:

    .. math::

        \text{GELU}(x) \approx
        x \cdot \frac{1}{2}
        \left[
            1 + \tanh\!\left(
                \sqrt{\tfrac{2}{\pi}}
                \left(x + 0.044715\, x^3\right)
            \right)
        \right]

    Parameters
    ----------
    approximate : str, optional
        Approximation method.  ``"none"`` uses the exact erf-based formula;
        ``"tanh"`` uses the faster tanh approximation.  Default: ``"none"``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Notes
    -----
    GELU is the default activation in transformer architectures (BERT, GPT)
    because its smooth non-linearity and non-zero gradient for all inputs
    improve training stability over ReLU for attention-based models.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.GELU()
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-0.0455, -0.1587, 0., 0.8413, 1.954])

    >>> # Fast tanh approximation — nearly identical for most inputs
    >>> m_approx = nn.GELU(approximate="tanh")
    >>> x = lucid.randn(4, 512)
    >>> out = m_approx(x)
    >>> out.shape
    (4, 512)
    """

    def __init__(self, approximate: GeluApproximate = "none") -> None:
        """Initialise the GELU module. See the class docstring for parameter semantics."""
        super().__init__()
        self.approximate = approximate

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return gelu(input, approximate=self.approximate)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"approximate={self.approximate!r}" if self.approximate != "none" else ""


class SiLU(_InplaceActivation):
    r"""Sigmoid Linear Unit (Swish) activation function.

    Applies element-wise:

    .. math::

        \text{SiLU}(x) = x \cdot \sigma(x)
        = \frac{x}{1 + e^{-x}}

    where :math:`\sigma` is the logistic sigmoid function.  The gate
    :math:`\sigma(x)` smoothly interpolates between 0 (deep negative) and
    the identity (large positive), producing a non-monotone shape with a
    small dip below zero near :math:`x \approx -1.3`.

    Parameters
    ----------
    inplace : bool, optional
        If ``True``, modifies the input tensor in-place.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Notes
    -----
    SiLU / Swish is used in EfficientNet, MobileNetV3, and many modern
    vision backbones as a smooth, self-gated alternative to ReLU.  It is
    differentiable everywhere, including at zero.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.SiLU()
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-0.2384, -0.2689, 0., 0.7311, 1.762])

    >>> # Common backbone activation — shape-preserving
    >>> x = lucid.randn(8, 256)
    >>> out = nn.SiLU()(x)
    >>> out.shape
    (8, 256)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return silu(input, self.inplace)


class Mish(_InplaceActivation):
    r"""Mish activation function.

    Applies element-wise:

    .. math::

        \text{Mish}(x) = x \cdot \tanh\!\bigl(\text{Softplus}(x)\bigr)
        = x \cdot \tanh\!\bigl(\ln(1 + e^x)\bigr)

    Mish is smooth, non-monotone, and unbounded above while being bounded
    below (approaching zero for large negative inputs).  It preserves small
    negative values — unlike ReLU — and empirically outperforms Swish/SiLU on
    several object detection benchmarks (YOLOv4, YOLOv5).

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Notes
    -----
    Mish requires computing both a softplus and a tanh, making it slightly
    more expensive than ReLU or SiLU.  The smooth gradient landscape can
    aid optimisation in very deep networks.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Mish()
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-0.2525, -0.3034, 0., 0.8651, 1.944])

    >>> # Drop-in for SiLU in detection backbones
    >>> x = lucid.randn(4, 128, 7, 7)
    >>> out = m(x)
    >>> out.shape
    (4, 128, 7, 7)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return mish(input, inplace=self.inplace)


class Softplus(Module):
    r"""Softplus activation function.

    Applies element-wise:

    .. math::

        \text{Softplus}(x) = \frac{1}{\beta} \ln\!\bigl(1 + e^{\beta x}\bigr)

    For numerical stability the output falls back to the identity when
    :math:`\beta x > \text{threshold}`, avoiding overflow in the exponential.
    Softplus is a smooth, everywhere-differentiable approximation of ReLU,
    and is useful as the positivity-enforcing transformation in probabilistic
    models (e.g. predicting variances or scale parameters).

    Parameters
    ----------
    beta : float, optional
        Sharpness parameter :math:`\beta` controlling how closely the curve
        approximates a hard hinge.  Larger values approach ReLU.
        Default: ``1.0``.
    threshold : float, optional
        Above this value of :math:`\beta x` the function falls back to the
        identity to prevent overflow.  Default: ``20.0``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Notes
    -----
    Setting ``beta=1`` recovers the standard log-sum-exp formulation.
    For very large ``beta`` the function becomes numerically equivalent to
    ReLU almost everywhere.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Softplus()
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([0.1269, 0.3133, 0.6931, 1.313, 2.127])

    >>> # Sharper approximation with beta=5
    >>> m_sharp = nn.Softplus(beta=5.0)
    >>> x = lucid.randn(3, 64)
    >>> out = m_sharp(x)
    >>> out.shape
    (3, 64)
    """

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        """Initialise the Softplus module. See the class docstring for parameter semantics."""
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return softplus(input, self.beta, self.threshold)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        if self.beta == 1.0 and self.threshold == 20.0:
            return ""
        return f"beta={self.beta}, threshold={self.threshold}"


class Hardswish(_InplaceActivation):
    r"""Hard Swish activation function.

    Applies element-wise:

    .. math::

        \text{Hardswish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}

    which is equivalent to:

    .. math::

        \text{Hardswish}(x) =
        \begin{cases}
            0           & \text{if } x \leq -3 \\
            x(x+3)/6    & \text{if } -3 < x < 3 \\
            x           & \text{if } x \geq 3
        \end{cases}

    Hard Swish approximates the SiLU (Swish) activation using only integer
    arithmetic-friendly piecewise-linear operations, making it particularly
    efficient on hardware without native sigmoid support (e.g. mobile NPUs).
    It was introduced in MobileNetV3.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Hardswish()
    >>> x = lucid.tensor([-4.0, -1.5, 0.0, 1.5, 4.0])
    >>> m(x)
    tensor([-0., -0.375, 0., 1.125, 4.])

    >>> # Efficient mobile backbone activation
    >>> x = lucid.randn(1, 96, 28, 28)
    >>> out = m(x)
    >>> out.shape
    (1, 96, 28, 28)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return hardswish(input, self.inplace)


class Hardsigmoid(_InplaceActivation):
    r"""Hard Sigmoid activation function.

    Applies element-wise:

    .. math::

        \text{Hardsigmoid}(x) = \max\!\left(0,\, \min\!\left(1,\, \frac{x}{6} + \frac{1}{2}\right)\right)

    This is a piecewise-linear approximation of the logistic sigmoid that
    saturates to 0 for :math:`x \leq -3` and to 1 for :math:`x \geq 3`,
    with a linear ramp in between.  It avoids the exponential required by the
    exact sigmoid, making it suitable for resource-constrained deployments.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input, values in :math:`[0, 1]`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Hardsigmoid()
    >>> x = lucid.tensor([-4.0, -3.0, 0.0, 3.0, 4.0])
    >>> m(x)
    tensor([0., 0., 0.5, 1., 1.])

    >>> # Lightweight gating in mobile attention heads
    >>> x = lucid.randn(2, 32)
    >>> out = m(x)
    >>> out.shape
    (2, 32)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return hardsigmoid(input, self.inplace)


class Sigmoid(Module):
    r"""Sigmoid (logistic) activation function.

    Applies element-wise:

    .. math::

        \sigma(x) = \frac{1}{1 + e^{-x}}

    Maps all real inputs to the open interval :math:`(0, 1)`.  Commonly used
    in the output layer of binary classifiers and as a gating function in
    recurrent architectures (LSTM, GRU).

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input, values in :math:`(0, 1)`.

    Notes
    -----
    Sigmoid saturates for large absolute inputs, causing vanishing gradients
    in deep networks.  For hidden layers, ReLU or GELU are generally preferred.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Sigmoid()
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([0.1192, 0.2689, 0.5, 0.7311, 0.8808])

    >>> # Binary classification output layer
    >>> x = lucid.randn(16, 1)
    >>> probs = m(x)   # probabilities in (0, 1)
    >>> probs.shape
    (16, 1)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return sigmoid(input)


class Tanh(Module):
    r"""Hyperbolic tangent activation function.

    Applies element-wise:

    .. math::

        \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

    Maps all real inputs to the open interval :math:`(-1, 1)`.  Zero-centred,
    unlike sigmoid, which makes it the preferred gate/output activation in
    many recurrent architectures (LSTM, GRU).

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input, values in :math:`(-1, 1)`.

    Notes
    -----
    Like sigmoid, tanh saturates for large inputs.  Its zero-centred output
    reduces the bias shift problem in successive layers, but the vanishing
    gradient issue still applies for very deep networks.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Tanh()
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-0.964, -0.7616, 0., 0.7616, 0.964])

    >>> # Hidden state output in a simple recurrent cell
    >>> h = lucid.randn(32, 128)
    >>> h_next = m(h)
    >>> h_next.shape
    (32, 128)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return tanh(input)


class Softmax(Module):
    r"""Softmax activation function.

    Applies to each slice along ``dim``:

    .. math::

        \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}

    Normalises the input to a proper probability distribution: all outputs
    are non-negative and sum to 1 along the specified dimension.  Used as
    the final layer of multi-class classifiers and in attention mechanisms.

    Parameters
    ----------
    dim : int or None, optional
        The dimension along which softmax is computed.  Must be specified
        explicitly for most use cases; ``None`` is retained for compatibility
        but raises a warning at runtime.  Default: ``None``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input; values along ``dim`` sum
      to 1.

    Notes
    -----
    For numerical stability the implementation subtracts the maximum value
    along ``dim`` before exponentiation (log-sum-exp trick), preventing
    overflow without changing the result.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Softmax(dim=-1)
    >>> x = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> m(x)
    tensor([[0.09003, 0.2447, 0.6652]])

    >>> # Attention weight normalisation over sequence length
    >>> scores = lucid.randn(4, 8, 64)   # (batch, heads, seq_len)
    >>> weights = nn.Softmax(dim=-1)(scores)
    >>> weights.shape
    (4, 8, 64)
    """

    def __init__(self, dim: int | None = None) -> None:
        """Initialise the Softmax module. See the class docstring for parameter semantics."""
        super().__init__()
        self.dim = dim

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return softmax(input, self.dim, _stacklevel=5)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"dim={self.dim}"


class LogSoftmax(Module):
    r"""Log-Softmax activation function.

    Applies element-wise:

    .. math::

        \text{LogSoftmax}(x_i)
        = \ln\!\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right)
        = x_i - \ln\!\sum_j e^{x_j}

    Combines the softmax normalisation with a logarithm in a single,
    numerically stable pass.  The output lives in :math:`(-\infty, 0]` and
    is intended to be paired with :class:`~lucid.nn.NLLLoss` for
    multi-class classification, or equivalently used directly with
    :func:`~lucid.nn.functional.cross_entropy`.

    Parameters
    ----------
    dim : int or None, optional
        The dimension along which log-softmax is computed.  Must be specified
        explicitly for most use cases.  Default: ``None``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input; values along ``dim`` are
      non-positive and represent log-probabilities.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.LogSoftmax(dim=-1)
    >>> x = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> m(x)
    tensor([[-2.408, -1.408, -0.4076]])

    >>> # Classifier output layer — pair with NLLLoss
    >>> logits = lucid.randn(32, 10)
    >>> log_probs = nn.LogSoftmax(dim=-1)(logits)
    >>> log_probs.shape
    (32, 10)
    """

    def __init__(self, dim: int | None = None) -> None:
        """Initialise the LogSoftmax module. See the class docstring for parameter semantics."""
        super().__init__()
        self.dim = dim

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return log_softmax(input, self.dim, _stacklevel=5)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"dim={self.dim}"


class Softmax2d(Module):
    r"""Softmax activation applied over the channel dimension of a 4-D tensor.

    Given a spatial feature map of shape :math:`(N, C, H, W)`, applies
    softmax along the channel axis :math:`C`:

    .. math::

        \text{Softmax2d}(x)_{n,c,h,w}
        = \frac{e^{x_{n,c,h,w}}}{\sum_{c'} e^{x_{n,c',h,w}}}

    Equivalent to ``Softmax(dim=-3)`` — channels are the third-from-last
    axis.  Used in dense prediction heads (semantic segmentation, saliency
    maps) where each spatial location's channel scores form a categorical
    distribution.

    Shape
    -----
    - Input: :math:`(N, C, H, W)` — must be exactly 4-D.
    - Output: :math:`(N, C, H, W)` — values along the channel dimension sum
      to 1 at every spatial location.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Softmax2d()
    >>> x = lucid.randn(1, 5, 4, 4)
    >>> out = m(x)
    >>> out.shape
    (1, 5, 4, 4)

    >>> # Verify that channel probabilities sum to 1 at each pixel
    >>> import lucid
    >>> x = lucid.randn(2, 10, 8, 8)
    >>> out = nn.Softmax2d()(x)
    >>> # out.sum(dim=1) should be all-ones spatially
    >>> out.sum(dim=1).shape
    (2, 8, 8)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        if input.ndim != 4:
            raise ValueError(f"Softmax2d: expected 4-D input (N, C, H, W), got ndim={input.ndim}")
        return softmax(input, -3, _stacklevel=5)


class RReLU(Module):
    r"""Randomized Leaky ReLU activation function.

    During **training**, the negative slope for each element is drawn
    independently and uniformly from :math:`[\text{lower}, \text{upper}]`:

    .. math::

        \text{RReLU}(x) =
        \begin{cases}
            x          & \text{if } x \geq 0 \\
            a \cdot x  & \text{otherwise},\quad
                          a \sim \mathcal{U}(\text{lower},\, \text{upper})
        \end{cases}

    During **evaluation**, the slope is fixed at the deterministic midpoint
    :math:`a = (\text{lower} + \text{upper}) / 2`.

    The stochastic negative slope acts as a form of noise-based
    regularisation, similar in spirit to Dropout, and was found to improve
    generalisation in image classification tasks.

    Parameters
    ----------
    lower : float, optional
        Lower bound of the uniform slope distribution.  Default: ``1/8``.
    upper : float, optional
        Upper bound of the uniform slope distribution.  Default: ``1/3``.
    inplace : bool, optional
        If ``True``, modifies the input tensor in-place.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Notes
    -----
    Call ``m.train()`` / ``m.eval()`` to switch between stochastic and
    deterministic modes respectively.  The module inherits training-mode
    tracking from :class:`~lucid.nn.Module`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.RReLU(lower=0.1, upper=0.3)
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> out_train = m(x)          # slope sampled from U(0.1, 0.3)

    >>> _ = m.eval()
    >>> out_eval = m(x)           # slope fixed at 0.2
    >>> out_eval.shape
    (5,)
    """

    def __init__(
        self, lower: float = 1.0 / 8.0, upper: float = 1.0 / 3.0, inplace: bool = False
    ) -> None:
        """Initialise the RReLU module. See the class docstring for parameter semantics."""
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return rrelu(input, self.lower, self.upper, training=self.training, inplace=self.inplace)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"lower={self.lower}, upper={self.upper}"


class ReLU6(_InplaceActivation):
    r"""ReLU6 activation function.

    Applies element-wise:

    .. math::

        \text{ReLU6}(x) = \min\!\bigl(\max(0,\, x),\, 6\bigr)

    A capped variant of ReLU that clamps activations to the range
    :math:`[0, 6]`.  The hard cap prevents activations from growing
    unboundedly, which can improve robustness of fixed-point quantisation
    (8-bit or lower) by keeping the dynamic range bounded.  Widely used in
    mobile architectures such as MobileNetV1 and MobileNetV2.

    Parameters
    ----------
    inplace : bool, optional
        If ``True``, modifies the input tensor in-place.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input, values in :math:`[0, 6]`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.ReLU6()
    >>> x = lucid.tensor([-1.0, 0.0, 3.0, 6.0, 10.0])
    >>> m(x)
    tensor([0., 0., 3., 6., 6.])

    >>> # Quantisation-friendly activation in depthwise-separable convolutions
    >>> x = lucid.randn(4, 32, 14, 14)
    >>> out = m(x)
    >>> out.shape
    (4, 32, 14, 14)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return relu6(input, self.inplace)


class PReLU(Module):
    r"""Parametric Rectified Linear Unit activation function.

    Applies element-wise:

    .. math::

        \text{PReLU}(x) =
        \begin{cases}
            x            & \text{if } x \geq 0 \\
            \alpha \cdot x & \text{otherwise}
        \end{cases}

    Unlike LeakyReLU, the slope :math:`\alpha` is a **learnable parameter**
    updated during back-propagation.  A single shared slope can be used for
    all channels (``num_parameters=1``) or each channel can have its own
    slope (``num_parameters=num_channels``).

    Parameters
    ----------
    num_parameters : int, optional
        Number of learnable slopes.  Use ``1`` for a single shared slope or
        set to the number of input channels for per-channel slopes.
        Default: ``1``.
    init : float, optional
        Initial value for all slope parameters.  Default: ``0.25``.
    device : DeviceLike, optional
        Device on which the parameter tensor is allocated.  Default: ``None``
        (uses the default device).
    dtype : DTypeLike, optional
        Data type of the parameter tensor.  Default: ``None`` (uses the
        default floating-point type).

    Attributes
    ----------
    weight : Parameter of shape ``(num_parameters,)``
        Learnable negative slopes :math:`\alpha`.  Updated by the optimiser
        during training.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Notes
    -----
    When ``num_parameters > 1``, the input is expected to have the channel
    dimension second (i.e. shape :math:`(N, C, *)`), and ``num_parameters``
    must equal :math:`C`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.PReLU(num_parameters=1, init=0.25)
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-0.5, -0.25, 0., 1., 2.], requires_grad=True)

    >>> # Per-channel slopes for a feature map with 64 channels
    >>> m = nn.PReLU(num_parameters=64)
    >>> x = lucid.randn(8, 64, 16, 16)
    >>> out = m(x)
    >>> out.shape
    (8, 64, 16, 16)
    """

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the PReLU module. See the class docstring for parameter semantics."""
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(full((num_parameters,), init, dtype=dtype, device=device))

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return prelu(input, self.weight)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"num_parameters={self.num_parameters}"


class Threshold(Module):
    r"""Threshold activation function.

    Applies element-wise:

    .. math::

        \text{Threshold}(x) =
        \begin{cases}
            x               & \text{if } x > \text{threshold} \\
            \text{value}    & \text{otherwise}
        \end{cases}

    Values that fall at or below the threshold are replaced with a constant
    ``value``.  This generalises the standard ReLU (threshold=0, value=0) to
    arbitrary cut-off points and fill values.

    Parameters
    ----------
    threshold : float
        The cut-off value.  Elements strictly greater than this pass through
        unchanged.
    value : float
        The replacement constant for elements that do not exceed the threshold.
    inplace : bool, optional
        If ``True``, modifies the input tensor in-place.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Threshold(threshold=1.0, value=-1.0)
    >>> x = lucid.tensor([0.5, 1.0, 1.5, 2.0])
    >>> m(x)
    tensor([-1., -1., 1.5, 2.])

    >>> # Use as a generalised ReLU with a non-zero floor
    >>> m = nn.Threshold(threshold=0.0, value=0.0)
    >>> x = lucid.randn(8, 32)
    >>> out = m(x)
    >>> out.shape
    (8, 32)
    """

    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        """Initialise the Threshold module. See the class docstring for parameter semantics."""
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return threshold(input, self.threshold, self.value, self.inplace)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"threshold={self.threshold}, value={self.value}"


class Hardtanh(Module):
    r"""Hardtanh activation function.

    Applies element-wise:

    .. math::

        \text{Hardtanh}(x) = \text{clamp}(x,\; \text{min\_val},\; \text{max\_val})
        =
        \begin{cases}
            \text{min\_val} & \text{if } x < \text{min\_val} \\
            x               & \text{if } \text{min\_val} \leq x \leq \text{max\_val} \\
            \text{max\_val} & \text{if } x > \text{max\_val}
        \end{cases}

    A piecewise-linear activation that clips the input to a fixed interval.
    With the default ``[-1, 1]`` range it approximates the tanh function using
    only comparisons and clamps, making it suitable for quantised models.

    Parameters
    ----------
    min_val : float, optional
        Lower bound of the clamping range.  Default: ``-1.0``.
    max_val : float, optional
        Upper bound of the clamping range.  Default: ``1.0``.
    inplace : bool, optional
        If ``True``, modifies the input tensor in-place.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input, values in
      :math:`[\text{min\_val},\; \text{max\_val}]`.

    Notes
    -----
    :class:`ReLU6` is a special case of Hardtanh with ``min_val=0`` and
    ``max_val=6``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Hardtanh(min_val=-1.0, max_val=1.0)
    >>> x = lucid.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    >>> m(x)
    tensor([-1., -0.5, 0., 0.5, 1.])

    >>> # Custom range for output normalisation
    >>> m = nn.Hardtanh(min_val=0.0, max_val=6.0)
    >>> x = lucid.randn(4, 64)
    >>> out = m(x)
    >>> out.shape
    (4, 64)
    """

    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """Initialise the Hardtanh module. See the class docstring for parameter semantics."""
        super().__init__()
        self.min_val = min_val if min_value is None else min_value
        self.max_val = max_val if max_value is None else max_value
        self.inplace = inplace
        if self.max_val <= self.min_val:
            raise AssertionError(
                f"max_val ({self.max_val}) must be greater than min_val ({self.min_val})"
            )

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return hardtanh(input, self.min_val, self.max_val, self.inplace)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"min_val={self.min_val}, max_val={self.max_val}"


class LogSigmoid(Module):
    r"""Log-Sigmoid activation function.

    Applies element-wise:

    .. math::

        \text{LogSigmoid}(x)
        = \ln\!\left(\sigma(x)\right)
        = \ln\!\left(\frac{1}{1 + e^{-x}}\right)
        = -\ln\!\bigl(1 + e^{-x}\bigr)

    The output is always :math:`\leq 0`.  Useful as a numerically stable
    log-probability output for binary models, or when computing negative
    log-likelihood loss without a separate sigmoid step.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input, values in :math:`(-\infty, 0]`.

    Notes
    -----
    The implementation fuses the log and sigmoid into a single engine call
    to avoid materialising the intermediate sigmoid tensor and to ensure
    numerical accuracy for large negative inputs.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.LogSigmoid()
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-2.127, -1.313, -0.6931, -0.3133, -0.1269])

    >>> # Log-probability output for binary classification
    >>> x = lucid.randn(32, 1)
    >>> log_probs = m(x)   # values in (-inf, 0]
    >>> log_probs.shape
    (32, 1)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return _wrap(_C_engine.log(_C_engine.sigmoid(_unwrap(input))))


class Softsign(Module):
    r"""Softsign activation function.

    Applies element-wise:

    .. math::

        \text{Softsign}(x) = \frac{x}{1 + |x|}

    Maps all real inputs to the open interval :math:`(-1, 1)`, similar to
    tanh but with polynomial rather than exponential saturation — meaning the
    function approaches its limits more slowly and gradients remain larger for
    moderately large inputs.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input, values in :math:`(-1, 1)`.

    Notes
    -----
    Softsign saturates more slowly than tanh, so it is less prone to
    vanishing gradients for large activations.  It is sometimes used as a
    tanh substitute in recurrent networks.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Softsign()
    >>> x = lucid.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
    >>> m(x)
    tensor([-0.8, -0.5, 0., 0.5, 0.8])

    >>> # Slower saturation compared to tanh
    >>> x = lucid.randn(8, 64)
    >>> out = m(x)
    >>> out.shape
    (8, 64)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        impl = _unwrap(input)
        denom = _C_engine.add(
            _C_engine.full(impl.shape, 1.0, impl.dtype, impl.device), _C_engine.abs(impl)
        )
        return _wrap(_C_engine.div(impl, denom))


class Softmin(Module):
    r"""Softmin activation function.

    Applies softmax to the negated input along a specified dimension:

    .. math::

        \text{Softmin}(x_i) = \frac{e^{-x_i}}{\sum_j e^{-x_j}}

    Equivalent to ``Softmax(-x)``.  The result is a valid probability
    distribution where **lower** values receive **higher** weight — the
    inverse of softmax.  Useful when scores represent costs or distances
    rather than affinities.

    Parameters
    ----------
    dim : int or None, optional
        The dimension along which softmin is computed.  Default: ``None``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input; values along ``dim`` are
      non-negative and sum to 1.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Softmin(dim=-1)
    >>> x = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> m(x)
    tensor([[0.6652, 0.2447, 0.09003]])

    >>> # Lowest-cost option gets the highest weight
    >>> costs = lucid.tensor([[0.1, 0.5, 0.9]])
    >>> weights = nn.Softmin(dim=-1)(costs)
    >>> weights.shape
    (1, 3)
    """

    def __init__(self, dim: int | None = None) -> None:
        """Initialise the Softmin module. See the class docstring for parameter semantics."""
        super().__init__()
        self.dim = dim

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return softmin(input, self.dim, _stacklevel=5)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"dim={self.dim}"


class GLU(Module):
    r"""Gated Linear Unit activation function.

    Splits the input tensor into two equal halves along ``dim``, then
    applies an element-wise gate:

    .. math::

        \text{GLU}(x) = x_1 \otimes \sigma(x_2)

    where :math:`x_1` and :math:`x_2` are the two halves of :math:`x` along
    ``dim``, :math:`\otimes` is element-wise multiplication, and
    :math:`\sigma` is the logistic sigmoid.  The sigmoid gate controls how
    much information from the first half flows through, enabling the network
    to learn a soft feature-selection mechanism.

    Parameters
    ----------
    dim : int, optional
        Dimension along which the input is split.  The size along this
        dimension must be even.  Default: ``-1``.

    Shape
    -----
    - Input: :math:`(\ldots,\; 2N,\; \ldots)` — size along ``dim`` must be
      even.
    - Output: :math:`(\ldots,\; N,\; \ldots)` — output is half the size of
      the input along ``dim``.

    Notes
    -----
    GLU was introduced for language modelling with convolutional sequence
    models and is also used in transformer feed-forward blocks as an
    alternative to ReLU/GELU projection layers.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.GLU(dim=-1)
    >>> x = lucid.tensor([[1.0, 2.0, 3.0, 4.0]])   # split into [1,2] and [3,4]
    >>> m(x)
    tensor([[0.9526, 1.964]])

    >>> # Feed-forward block with GLU gating
    >>> ff = nn.Sequential(nn.Linear(256, 512), nn.GLU(dim=-1))
    >>> x = lucid.randn(4, 64, 256)
    >>> out = ff(x)
    >>> out.shape
    (4, 64, 256)
    """

    def __init__(self, dim: int = -1) -> None:
        """Initialise the GLU module. See the class docstring for parameter semantics."""
        super().__init__()
        self.dim = dim

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return glu(input, self.dim)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"dim={self.dim}"


class CELU(Module):
    r"""Continuously Differentiable Exponential Linear Unit activation function.

    Applies element-wise:

    .. math::

        \text{CELU}(x) =
        \max(0,\, x)
        + \min\!\left(0,\; \alpha\!\left(e^{x/\alpha} - 1\right)\right)

    CELU differs from ELU in that it is continuously differentiable
    everywhere, including at zero, by ensuring the left and right derivatives
    match at :math:`x = 0`.  Both branches coincide with an exponential
    scaled by :math:`\alpha`, and the output transitions smoothly from the
    negative saturating region to the linear positive branch.

    Parameters
    ----------
    alpha : float, optional
        Scale :math:`\alpha > 0` controlling the slope of the negative branch
        and the value to which it saturates.  Default: ``1.0``.
    inplace : bool, optional
        If ``True``, modifies the input tensor in-place.  Default: ``False``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.CELU(alpha=1.0)
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-0.8647, -0.6321, 0., 1., 2.])

    >>> # Smoother negative branch than ELU for gradient-sensitive architectures
    >>> m = nn.CELU(alpha=0.5)
    >>> x = lucid.randn(4, 128)
    >>> out = m(x)
    >>> out.shape
    (4, 128)
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        """Initialise the CELU module. See the class docstring for parameter semantics."""
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return celu(input, self.alpha, self.inplace)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"alpha={self.alpha}"


class Hardshrink(Module):
    r"""Hard Shrinkage activation function.

    Applies element-wise:

    .. math::

        \text{Hardshrink}(x) =
        \begin{cases}
            x   & \text{if } |x| > \lambda \\
            0   & \text{otherwise}
        \end{cases}

    Values within the band :math:`[-\lambda, \lambda]` are zeroed out
    ("shrunk" to zero), while large values pass through unchanged.
    Hard shrinkage encourages sparse representations and is commonly used
    in sparse coding and wavelet-based signal processing.

    Parameters
    ----------
    lambd : float, optional
        Threshold :math:`\lambda \geq 0`.  Default: ``0.5``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Hardshrink(lambd=0.5)
    >>> x = lucid.tensor([-1.0, -0.4, 0.0, 0.4, 1.0])
    >>> m(x)
    tensor([-1.,  0.,  0.,  0.,  1.])

    >>> # Sparsifying activation for dictionary learning
    >>> m = nn.Hardshrink(lambd=1.0)
    >>> x = lucid.randn(8, 64)
    >>> out = m(x)
    >>> out.shape
    (8, 64)
    """

    def __init__(self, lambd: float = 0.5) -> None:
        """Initialise the Hardshrink module. See the class docstring for parameter semantics."""
        super().__init__()
        self.lambd = lambd

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return hardshrink(input, self.lambd)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"lambd={self.lambd}"


class Tanhshrink(Module):
    r"""Tanhshrink activation function.

    Applies element-wise:

    .. math::

        \text{Tanhshrink}(x) = x - \tanh(x)

    Subtracts the tanh of the input from the input itself.  Near zero the
    output is approximately :math:`x^3/3` (the tanh Taylor residual), so the
    function is smooth and cubic near the origin.  For large inputs,
    :math:`\tanh(x) \to \pm 1`, so the output approaches :math:`x \mp 1`,
    behaving asymptotically like the identity.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Notes
    -----
    Tanhshrink is used as a soft thresholding operator in some sparse
    representation and compressed-sensing networks.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Tanhshrink()
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> m(x)
    tensor([-1.036, -0.2384, 0., 0.2384, 1.036])

    >>> # Smooth sparse regularisation in an encoder
    >>> x = lucid.randn(4, 256)
    >>> out = m(x)
    >>> out.shape
    (4, 256)
    """

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return tanhshrink(input)


class Softshrink(Module):
    r"""Soft Shrinkage activation function.

    Applies element-wise:

    .. math::

        \text{Softshrink}(x) =
        \begin{cases}
            x - \lambda & \text{if } x > \lambda \\
            x + \lambda & \text{if } x < -\lambda \\
            0           & \text{otherwise}
        \end{cases}

    Shrinks all values towards zero by :math:`\lambda`: values outside the
    dead band are shifted, while values inside are zeroed.  This is the
    proximal operator of the :math:`\ell_1` norm and appears in LASSO and
    iterative soft-thresholding algorithms (ISTA / FISTA).

    Parameters
    ----------
    lambd : float, optional
        Threshold :math:`\lambda \geq 0`.  Default: ``0.5``.

    Shape
    -----
    - Input: :math:`(*)` — any shape.
    - Output: :math:`(*)` — same shape as input.

    Notes
    -----
    Unlike :class:`Hardshrink`, Softshrink *shifts* surviving values toward
    zero rather than preserving them exactly.  This makes it a continuous
    function and gives it a well-defined subgradient everywhere.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Softshrink(lambd=0.5)
    >>> x = lucid.tensor([-1.5, -0.3, 0.0, 0.3, 1.5])
    >>> m(x)
    tensor([-1., 0., 0., 0., 1.])

    >>> # L1-proximal layer in an unrolled ISTA network
    >>> m = nn.Softshrink(lambd=0.1)
    >>> x = lucid.randn(8, 128)
    >>> out = m(x)
    >>> out.shape
    (8, 128)
    """

    def __init__(self, lambd: float = 0.5) -> None:
        """Initialise the Softshrink module. See the class docstring for parameter semantics."""
        super().__init__()
        self.lambd = lambd

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return softshrink(input, self.lambd)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"lambd={self.lambd}"


# ── distance / similarity wrappers ─────────────────────────────────────────


class CosineSimilarity(Module):
    r"""Cosine similarity between two tensors along a specified dimension.

    Computes element-wise cosine similarity:

    .. math::

        \text{CosineSimilarity}(x_1, x_2)
        = \frac{x_1 \cdot x_2}{\max\!\bigl(\|x_1\|_2 \cdot \|x_2\|_2,\; \varepsilon\bigr)}

    The result is a scalar (or tensor with ``dim`` reduced) in the range
    :math:`[-1, 1]`, where :math:`1` indicates identical direction, :math:`0`
    orthogonality, and :math:`-1` opposite direction.  The ``eps`` floor
    on the denominator prevents division by zero for zero-norm vectors.

    Parameters
    ----------
    dim : int, optional
        Dimension along which cosine similarity is computed.  Default: ``1``.
    eps : float, optional
        Small value :math:`\varepsilon` added to the denominator for
        numerical stability.  Default: ``1e-8``.

    Shape
    -----
    - Input ``x1``: :math:`(N, D)` or any shape broadcastable to ``x2``.
    - Input ``x2``: :math:`(N, D)` or any shape broadcastable to ``x1``.
    - Output: :math:`(N,)` — ``dim`` is reduced; same batch dimensions as
      input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.CosineSimilarity(dim=1)
    >>> x1 = lucid.tensor([[1.0, 0.0], [0.0, 1.0]])
    >>> x2 = lucid.tensor([[1.0, 0.0], [1.0, 0.0]])
    >>> m(x1, x2)
    tensor([1., 0.])

    >>> # Comparing sentence embeddings along the feature dimension
    >>> emb1 = lucid.randn(32, 768)
    >>> emb2 = lucid.randn(32, 768)
    >>> sim = nn.CosineSimilarity(dim=1)(emb1, emb2)
    >>> sim.shape
    (32,)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """Initialise the CosineSimilarity module. See the class docstring for parameter semantics."""
        super().__init__()
        self.dim = dim
        self.eps = eps

    @override
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"dim={self.dim}, eps={self.eps}"


class PairwiseDistance(Module):
    r"""Pairwise :math:`\ell_p` distance between corresponding rows of two tensors.

    Computes the :math:`\ell_p` norm of the difference vector for each pair
    of rows:

    .. math::

        d_p(x_1, x_2)
        = \left\| x_1 - x_2 + \varepsilon \right\|_p
        = \left( \sum_i \left| x_{1,i} - x_{2,i} + \varepsilon \right|^p \right)^{1/p}

    A small :math:`\varepsilon` is added before the norm computation for
    numerical stability.  With ``p=2`` this yields the standard Euclidean
    distance; with ``p=1`` the Manhattan distance.

    Parameters
    ----------
    p : float, optional
        The exponent :math:`p` of the :math:`\ell_p` norm.  Default: ``2.0``.
    eps : float, optional
        Small :math:`\varepsilon` added to the difference for numerical
        stability.  Default: ``1e-6``.
    keepdim : bool, optional
        If ``True``, the reduced dimension is retained as a size-1 dimension
        in the output.  Default: ``False``.

    Shape
    -----
    - Input ``x1``: :math:`(N, D)`.
    - Input ``x2``: :math:`(N, D)`.
    - Output: :math:`(N,)` if ``keepdim=False``, else :math:`(N, 1)`.

    Notes
    -----
    Commonly used in Siamese networks and metric-learning objectives such as
    contrastive loss and triplet loss, where the distance between paired
    embeddings must be differentiable.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.PairwiseDistance(p=2.0)
    >>> x1 = lucid.tensor([[1.0, 0.0], [0.0, 1.0]])
    >>> x2 = lucid.tensor([[0.0, 0.0], [0.0, 0.0]])
    >>> m(x1, x2)
    tensor([1., 1.])

    >>> # Euclidean distance for embedding comparison in a Siamese network
    >>> emb1 = lucid.randn(64, 128)
    >>> emb2 = lucid.randn(64, 128)
    >>> dist = nn.PairwiseDistance(p=2.0)(emb1, emb2)
    >>> dist.shape
    (64,)
    """

    def __init__(self, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> None:
        """Initialise the PairwiseDistance module. See the class docstring for parameter semantics."""
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    @override
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        r"""Apply the activation function element-wise.

        Parameters
        ----------
        input : Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Output tensor of the same shape as ``input``.
        """
        return pairwise_distance(x1, x2, p=self.p, eps=self.eps, keepdim=self.keepdim)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"p={self.p}, eps={self.eps}, keepdim={self.keepdim}"
