"""Bijective ``Transform`` system for ``lucid.distributions``.

A :class:`Transform` is an invertible map ``T : X → Y`` with a known
log-absolute-determinant Jacobian, used to push a base distribution
through a measurable bijection (e.g. ``ExpTransform`` turns a Normal
into a LogNormal).

Composition + inversion are handled by :class:`ComposeTransform` and
:class:`Transform.__invert__` so users can build Lucid-side bijectors
without worrying about Jacobian bookkeeping.
"""

import math
from typing import override

import lucid
from lucid._dtype import finfo
from lucid._tensor.tensor import Tensor
from lucid._types import DTypeLike, DeviceLike
from lucid.distributions import constraints
from lucid.distributions._util import _align_device, _as_tensor, _sum_rightmost
from lucid.distributions.constraints import Constraint
from lucid.distributions.distribution import Distribution
from lucid.distributions.transformed_distribution import (
    ComposeTransform as ComposeTransform,
    Transform as Transform,
    TransformedDistribution as TransformedDistribution,
    _InverseTransform as _InverseTransform,
)


class ExpTransform(Transform):
    r"""Element-wise exponential bijection :math:`y = e^x`.

    Maps :math:`\mathbb{R} \to (0, \infty)` element-wise; composed with a
    Normal base distribution it yields a LogNormal.  Monotone increasing,
    bijective, ``event_dim = 0``.

    Notes
    -----
    Forward:  :math:`y = e^x`.

    Inverse:  :math:`x = \log y` (defined for :math:`y > 0`).

    Log Jacobian determinant (element-wise):

    .. math::

        \log\!\left|\frac{\partial y}{\partial x}\right| = x

    since :math:`|dy/dx| = e^x`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import ExpTransform
    >>> T = ExpTransform()
    >>> x = lucid.tensor(0.0)
    >>> y = T(x)
    >>> T.log_abs_det_jacobian(x, y)
    tensor(0.)
    """

    domain: Constraint = constraints.real
    codomain: Constraint = constraints.positive

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = e^x`."""
        return x.exp()

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = \\log(y)`."""
        return y.log()

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r""":math:`\log|\partial y/\partial x| = x` since :math:`|dy/dx| = e^x`.

        Parameters
        ----------
        x : Tensor
            Pre-transform input.
        y : Tensor
            Post-transform output (``e^x``); ignored — the Jacobian is
            cheaper to express via ``x``.

        Returns
        -------
        Tensor
            ``x`` itself.
        """
        return x  # |dy/dx| = exp(x), so log|dy/dx| = x.


class SigmoidTransform(Transform):
    r"""Element-wise sigmoid bijection :math:`y = \sigma(x) = 1/(1 + e^{-x})`.

    Maps :math:`\mathbb{R} \to (0, 1)` element-wise.  The natural
    unconstrained → probability transform — composed with a base
    distribution on :math:`\mathbb{R}`, the pushforward lives on the
    unit interval.  Monotone increasing, bijective, ``event_dim = 0``.

    Notes
    -----
    Forward:  :math:`y = \sigma(x) = 1/(1 + e^{-x})`.

    Inverse (logit):
    :math:`x = \log\!\bigl(y/(1-y)\bigr)`.

    Log Jacobian determinant:

    .. math::

        \log\!\left|\frac{\partial y}{\partial x}\right|
        = \log\bigl(\sigma(x)(1 - \sigma(x))\bigr)
        = \log y + \log(1 - y)

    The transform is closely related to the :class:`StickBreakingTransform`
    (which generalises to the simplex) and to :class:`TanhTransform`
    (which targets :math:`(-1, 1)`).

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import SigmoidTransform
    >>> T = SigmoidTransform()
    >>> T(lucid.tensor(0.0))  # σ(0) = 0.5
    tensor(0.5)
    """

    domain: Constraint = constraints.real
    codomain: Constraint = constraints.unit_interval

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = \\sigma(x)`."""
        return x.sigmoid()

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = \\log(y/(1-y))`, clamped to ``[tiny, 1 - eps]``."""
        fi = finfo(y.dtype)
        y = lucid.clip(y, min=float(fi.tiny), max=1.0 - float(fi.eps))
        return y.log() - (-y).log1p()

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Log-Jacobian :math:`\log\bigl(y\,(1-y)\bigr)` of the sigmoid map.

        Parameters
        ----------
        x : Tensor
            Pre-transform input (sigmoid logit); ignored — the
            log-Jacobian is cheaper to express via ``y``.
        y : Tensor
            Post-transform output in :math:`(0, 1)`.

        Returns
        -------
        Tensor
            :math:`\log y + \log (1 - y)`.
        """
        # log|y · (1 − y)|.
        return y.log() + (1.0 - y).log()


class TanhTransform(Transform):
    r"""Element-wise hyperbolic tangent bijection :math:`y = \tanh(x)`.

    Maps :math:`\mathbb{R} \to (-1, 1)` element-wise.  Commonly used in
    reinforcement learning to push a Normal policy through ``tanh`` so
    actions live in a bounded box (SAC and friends).  Monotone
    increasing, bijective, ``event_dim = 0``.

    Notes
    -----
    Forward:  :math:`y = \tanh(x)`.

    Inverse:
    :math:`x = \tfrac{1}{2}\log\!\bigl((1+y)/(1-y)\bigr)`
    (the inverse hyperbolic tangent).

    Log Jacobian determinant (numerically stable form):

    .. math::

        \log\!\left|\frac{\partial y}{\partial x}\right|
        = \log\bigl(1 - \tanh^2(x)\bigr)
        = 2 \bigl(\log 2 - x - \operatorname{softplus}(-2x)\bigr)

    This formulation avoids overflow / underflow when :math:`|x|` is
    large, which the naive form :math:`\log(1 - \tanh^2 x)` would suffer
    (since :math:`\tanh(x) \to \pm 1`).

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import TanhTransform
    >>> T = TanhTransform()
    >>> T(lucid.tensor(0.0))  # tanh(0) = 0
    tensor(0.)
    """

    domain: Constraint = constraints.real
    codomain: Constraint = constraints.interval(-1.0, 1.0)

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = \\tanh(x)`."""
        return x.tanh()

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        r"""Inverse: :math:`x = \tfrac{1}{2}\log\!\bigl((1+y)/(1-y)\bigr)`."""
        return 0.5 * ((1.0 + y) / (1.0 - y)).log()

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Numerically-stable :math:`\log(1 - \tanh^2(x))`.

        Computed as :math:`2\bigl(\log 2 - x - \operatorname{softplus}(-2x)\bigr)`
        — the stable form used by the reference framework that avoids
        the catastrophic ``log(1 - 1)`` cancellation when ``|x|`` is
        large.

        Parameters
        ----------
        x : Tensor
            Pre-transform input.
        y : Tensor
            Post-transform output (``tanh(x)``); ignored — the
            log-Jacobian is expressed in ``x``.

        Returns
        -------
        Tensor
            Same shape as ``x``.
        """
        # log(1 − tanh²(x)) = 2·(log 2 − x − softplus(−2x)) — the stable
        # form used by the reference framework.
        return 2.0 * (math.log(2.0) - x - (-2.0 * x).exp().log1p())


class AffineTransform(Transform):
    r"""Element-wise affine bijection :math:`y = \mathrm{loc} + \mathrm{scale}\cdot x`.

    The fundamental location-scale transform.  Composing with a
    standard Normal yields :math:`\mathcal{N}(\mathrm{loc},
    \mathrm{scale}^2)`; it underpins virtually every reparameterised
    sampler in the codebase.  Bijective with sign matching
    ``sign(scale)``; the implementation fixes ``self.sign = +1``, so the
    caller is responsible for ensuring ``scale > 0``.

    Parameters
    ----------
    loc : Tensor or float
        Additive offset :math:`b`.
    scale : Tensor or float
        Multiplicative slope :math:`a` (assumed positive).
    event_dim : int, optional
        Number of trailing dimensions treated as a single event. Default ``0``.

    Notes
    -----
    Forward:  :math:`y = b + a\,x`.

    Inverse:  :math:`x = (y - b) / a`.

    Log Jacobian determinant (broadcast to ``x.shape``):

    .. math::

        \log\!\left|\frac{\partial y}{\partial x}\right|
        = \log|a|

    For element-wise application (``event_dim = 0``) this is summed by
    :class:`IndependentTransform` if the caller reinterprets trailing
    dims as event dims.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import AffineTransform
    >>> T = AffineTransform(loc=1.0, scale=2.0)
    >>> T(lucid.tensor(3.0))
    tensor(7.)
    """

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        event_dim: int = 0,
    ) -> None:
        """Store the affine offset ``loc``, multiplier ``scale``, and ``event_dim``.

        The caller is responsible for ensuring ``scale > 0``; ``sign`` is
        fixed at ``+1`` here so the transform is treated as increasing.
        """
        super().__init__()
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.event_dim = event_dim
        self.sign = 1  # Caller is responsible for ``scale > 0``.

    @override
    @property
    def domain(self) -> Constraint:
        """``real`` for scalar events, or ``independent(real, event_dim)``."""
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @override
    @property
    def codomain(self) -> Constraint:
        """``real`` for scalar events, or ``independent(real, event_dim)``."""
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = \\text{loc} + \\text{scale} \\cdot x`."""
        # A ``loc`` / ``scale`` given as a number is a 0-dim host tensor; it
        # follows ``x`` to its device.
        return _align_device(self.loc, x) + _align_device(self.scale, x) * x

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = (y - \\text{loc}) / \\text{scale}`."""
        return (y - _align_device(self.loc, y)) / _align_device(self.scale, y)

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Constant log-Jacobian :math:`\log|\text{scale}|`.

        The Jacobian of :math:`y = \text{scale} \cdot x + \text{loc}`
        is just ``scale``, so the log-determinant is constant in ``x``.
        We add ``zeros_like(x)`` so the result broadcasts to ``x``'s
        full shape (callers expect element-wise log-Jacobians).

        Parameters
        ----------
        x : Tensor
            Pre-transform input; used only for shape.
        y : Tensor
            Post-transform output; ignored.

        Returns
        -------
        Tensor
            ``log|scale|`` broadcast to ``x.shape`` and summed over ``event_dim``.
        """
        ladj = _align_device(self.scale, x).abs().log() + lucid.zeros(
            tuple(x.shape), dtype=x.dtype, device=x.device
        )
        return _sum_rightmost(ladj, self.event_dim)


class PowerTransform(Transform):
    r"""Element-wise power bijection :math:`y = x^{\mathrm{exponent}}`.

    Bijection on :math:`(0, \infty) \to (0, \infty)` (the input must be
    positive; the implementation does not check this).  Useful for
    Box–Cox-style reparameterisations and for mapping between Gamma
    families with different shape parameters.

    Parameters
    ----------
    exponent : Tensor or float
        Power :math:`p` applied element-wise.  Sign-aware: negative
        exponents are admissible but flip orientation, and zero
        exponents are not invertible (excluded).

    Notes
    -----
    Forward:  :math:`y = x^{p}`.

    Inverse:  :math:`x = y^{1/p}`.

    Log Jacobian determinant:

    .. math::

        \log\!\left|\frac{\partial y}{\partial x}\right|
        = \log|p| + (p - 1)\log x

    Special cases:

    * :math:`p = 1` → identity.
    * :math:`p = -1` → reciprocal :math:`y = 1/x`.
    * :math:`p = 2` → squaring on the positive half-line.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import PowerTransform
    >>> T = PowerTransform(exponent=2.0)
    >>> T(lucid.tensor(3.0))  # 3² = 9
    tensor(9.)
    """

    domain: Constraint = constraints.positive
    codomain: Constraint = constraints.positive

    def __init__(self, exponent: Tensor | float) -> None:
        """Store the (element-wise) power ``exponent``."""
        super().__init__()
        self.exponent = _as_tensor(exponent)

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = x^{\\text{exponent}}`."""
        return x ** _align_device(self.exponent, x)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = y^{1/\\text{exponent}}`."""
        return y ** (1.0 / _align_device(self.exponent, y))

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r""":math:`\log|\partial y/\partial x| = \log|\text{exponent}| + (\text{exponent} - 1)\log x`.

        Derivative of :math:`y = x^{\text{exponent}}`.

        Parameters
        ----------
        x : Tensor
            Pre-transform input.  Must be strictly positive for the
            ``log x`` term to be real.
        y : Tensor
            Post-transform output (``x^exponent``); ignored.

        Returns
        -------
        Tensor
            Element-wise log-Jacobian, same shape as ``x``.
        """
        # |dy/dx| = exponent · x^(exponent − 1) ⇒ log|dy/dx| =
        #   log|exponent| + (exponent − 1)·log(x).
        exponent = _align_device(self.exponent, x)
        return exponent.abs().log() + (exponent - 1.0) * x.log()


class SoftmaxTransform(Transform):
    r"""Softmax transform mapping :math:`\mathbb{R}^K \to \Delta^{K-1}`.

    Pushes an unconstrained vector :math:`\mathbf{x} \in \mathbb{R}^K`
    onto the open :math:`K`-simplex via
    :math:`\mathbf{y} = \mathrm{softmax}(\mathbf{x})`.  Operates on the
    *last* axis; ``event_dim = 1``.

    The transform is over-parameterised: any constant shift along the
    softmax axis (:math:`\mathbf{x} \to \mathbf{x} + c\mathbf{1}`)
    yields the same :math:`\mathbf{y}`, so it is not a true bijection.
    The standard convention used here is to anchor the inverse at
    :math:`\mathbf{x} = \log \mathbf{y}` (un-normalised
    log-probabilities), which is one canonical preimage.

    Notes
    -----
    Forward:

    .. math::

        y_k = \frac{e^{x_k}}{\sum_{j=1}^{K} e^{x_j}}

    Inverse (canonical anchor):

    .. math::

        x_k = \log y_k

    Pseudo-Jacobian used in change-of-variable bookkeeping:

    .. math::

        \log|\det J| = \sum_{k=1}^{K} \log y_k

    This is the convention that keeps the simplex-valued pushforward
    consistent in flow stacks; for a *true* bijection between
    :math:`\mathbb{R}^{K-1}` and the simplex use
    :class:`StickBreakingTransform` instead.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import SoftmaxTransform
    >>> T = SoftmaxTransform()
    >>> T(lucid.tensor([0.0, 1.0, 2.0]))
    tensor([0.09003, 0.2447, 0.6652])
    """

    event_dim: int = 1
    domain: Constraint = constraints.real_vector
    codomain: Constraint = constraints.simplex

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: ``softmax(x, dim=-1)``."""
        from lucid.nn.functional.activations import softmax

        return softmax(x, dim=-1)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: ``log(y)``, anchored at the un-normalised log-probabilities.

        Any constant shift along the last axis yields the same softmax, so
        this is one canonical preimage rather than the unique inverse.
        """
        # Standard convention: take the un-normalised log-probabilities.
        # Any constant shift gives the same softmax — we anchor at log y.
        return y.log()

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Pseudo-Jacobian :math:`\sum_k \log y_k` for the over-parameterised softmax.

        Parameters
        ----------
        x : Tensor
            Pre-softmax logits.  Used only for shape; the formula
            collapses through ``y``.
        y : Tensor
            Post-softmax probabilities in the simplex.

        Returns
        -------
        Tensor
            :math:`\sum_k \log y_k`, reduced over the last axis.
        """
        # log |det J| = sum_k log y_k — pseudo-Jacobian for the
        # over-parameterised softmax, matching the reference framework.
        return y.log().sum(dim=-1)


class StickBreakingTransform(Transform):
    r"""Logistic stick-breaking bijection :math:`\mathbb{R}^{K-1} \to \Delta^{K-1}`.

    A **true** bijection between unconstrained :math:`(K-1)`-vectors and
    the :math:`K`-simplex (i.e. one extra dimension is broken off as the
    residual stick).  Unlike :class:`SoftmaxTransform` it has the
    correct dimensionality and a tractable log-Jacobian determinant,
    making it the preferred choice for normalising flows over
    probability vectors and for unconstrained reparameterisations of
    :class:`~lucid.distributions.Dirichlet` priors.  ``event_dim = 1``.

    Notes
    -----
    Forward "stick breaking" (with :math:`z_k = \sigma(x_k - \log(K-k))`):

    .. math::

        y_k = z_k \prod_{j < k}(1 - z_j),
        \quad
        y_{K-1} = \prod_{j=0}^{K-2}(1 - z_j)

    The last component is the residual stick remaining after the first
    :math:`K - 1` breaks.  Each :math:`y_k > 0` and :math:`\sum_k y_k = 1`
    by construction.

    Inverse (back-solve the stick lengths):

    .. math::

        z_k = \frac{y_k}{1 - \sum_{j < k} y_j},
        \qquad
        x_k = \mathrm{logit}(z_k) + \log(K - k)

    Log Jacobian determinant:

    .. math::

        \log|\det J| =
        \sum_{k=0}^{K-2}
        \bigl[\log y_k + \log(1 - \textstyle\sum_{j < k} y_j) + \log(1 - z_k)\bigr]

    where the :math:`-\log(K-k)` shifts ensure the uniform Dirichlet
    corresponds to :math:`\mathbf{x} = \mathbf{0}`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import StickBreakingTransform
    >>> T = StickBreakingTransform()
    >>> y = T(lucid.tensor([0.0, 0.0]))  # maps to a Dirichlet(1,1,1) sample
    >>> y.sum()
    tensor(1.)
    """

    event_dim: int = 1
    domain: Constraint = constraints.real_vector
    codomain: Constraint = constraints.simplex

    @override
    def forward_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Extend the trailing axis by 1 for the residual stick."""
        return shape[:-1] + (shape[-1] + 1,) if shape else shape

    @override
    def inverse_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Drop the residual stick axis by 1."""
        return shape[:-1] + (shape[-1] - 1,) if shape else shape

    @override
    def _call(self, x: Tensor) -> Tensor:
        r"""Forward stick-breaking: ``x ∈ ℝ^{K-1}`` → simplex ``y ∈ Δ^K``.

        Each :math:`y_k = \sigma(x_k - \log(K-k)) \cdot \prod_{j<k}(1 - y_j)`,
        with :math:`y_{K-1}` taking the residual stick.
        """
        # ``x`` shape (..., K-1).  We extend with a final 0 so the
        # cumulative-product step yields the residual stick automatically.
        K_minus_1: int = int(x.shape[-1])
        offsets: Tensor = lucid.arange(K_minus_1, 0, -1, dtype=x.dtype, device=x.device)
        # σ(x_k − log(K−k)) for k = 0..K-2.
        z: Tensor = (x - offsets.log()).sigmoid()
        # Stick-breaking: y_k = z_k · ∏_{j<k} (1 − z_j).
        # Compute via cumulative product of (1 − z) and pad with 1 in front.
        one_minus_z: Tensor = 1.0 - z
        # Manual prefix-product since lucid has no cumprod-from-1 op handy:
        # build [1, 1−z₀, (1−z₀)(1−z₁), …].
        prods: list[Tensor] = []
        running: Tensor = lucid.ones_like(z.narrow(-1, 0, 1)).squeeze(-1)
        prods.append(running)
        for k in range(K_minus_1 - 1):
            running = running * one_minus_z.narrow(-1, k, 1).squeeze(-1)
            prods.append(running)
        prefix: Tensor = lucid.stack(prods, dim=-1)  # shape (..., K-1)
        head: Tensor = z * prefix  # y_0 .. y_{K-2}
        # Last component is the residual: y_{K-1} = ∏_{j<K-1} (1 − z_j).
        last: Tensor = prefix.narrow(-1, K_minus_1 - 1, 1) * one_minus_z.narrow(
            -1, K_minus_1 - 1, 1
        )
        return lucid.cat([head, last], dim=-1)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        r"""Inverse stick-breaking: simplex ``y ∈ Δ^K`` → ``x ∈ ℝ^{K-1}``.

        Recovers :math:`z_k = y_k / (1 - \sum_{j<k} y_j)` and then
        :math:`x_k = \operatorname{logit}(z_k) + \log(K - k)`.
        """
        K_minus_1: int = int(y.shape[-1]) - 1
        y_crop: Tensor = y.narrow(-1, 0, K_minus_1)
        sf: Tensor = (1.0 - y_crop.cumsum(dim=-1)).clamp(min=lucid.finfo(y.dtype).tiny)
        offsets: Tensor = lucid.arange(K_minus_1, 0, -1, dtype=y.dtype, device=y.device)
        return y_crop.log() - sf.log() + offsets.log()

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Standard simplex-to-:math:`\mathbb{R}^{K-1}` log-Jacobian.

        :math:`\log|\det J| = \sum_k \bigl[\log y_k + \log\!\bigl(1 - z_k\bigr)\bigr]`.

        Parameters
        ----------
        x : Tensor
            Unconstrained :math:`(K-1)`-vectors.
        y : Tensor
            Corresponding :math:`K`-simplex points (output of the
            forward map).

        Returns
        -------
        Tensor
            Per-sample log-Jacobian.
        """
        # ``dy_k/dz_k = rem_k`` and ``dz_k/dx_k = z_k (1 - z_k)``, and the
        # map is triangular in ``k``, so
        #
        #     log|det J| = Σ_k [ log(rem_k · z_k) + log(1 - z_k) ]
        #                = Σ_k [ log(y_k)         + log(1 - z_k) ]
        #
        # since ``y_k = rem_k · z_k``.  Using ``1 - sigmoid(u) = exp(-u) * sigmoid(u)``
        # with ``u = x_k - log(K - k)`` avoids ``0 / 0`` when a prefix reaches 1.
        from lucid.nn.functional.activations import logsigmoid

        K_minus_1: int = int(x.shape[-1])
        offsets: Tensor = lucid.arange(K_minus_1, 0, -1, dtype=x.dtype, device=x.device)
        x_shifted: Tensor = x - offsets.log()
        return (
            -x_shifted + logsigmoid(x_shifted) + y.narrow(-1, 0, K_minus_1).log()
        ).sum(dim=-1)


class LowerCholeskyTransform(Transform):
    r"""Bijection mapping an unconstrained matrix to a positive-diagonal Cholesky factor.

    The standard reparameterisation used to learn covariance / scale
    matrices: a free :math:`D \times D` matrix is mapped to a lower
    triangular matrix with strictly positive diagonal by zeroing the
    strict upper triangle and applying ``softplus`` to the diagonal.
    Composing with a base on :math:`\mathbb{R}^{D \times D}` produces a
    pushforward over the cone of valid Cholesky factors.
    ``event_dim = 2``.

    Notes
    -----
    Forward (element-wise on a :math:`D \times D` input :math:`X`):

    .. math::

        L_{ij} =
        \begin{cases}
            \operatorname{softplus}(X_{ii}) & i = j \\
            X_{ij} & i > j \\
            0 & i < j
        \end{cases}

    Inverse:

    .. math::

        X_{ii} = \operatorname{softplus}^{-1}(L_{ii})
               = \log(e^{L_{ii}} - 1),
        \qquad
        X_{ij} = L_{ij}\;\;(i > j)

    Log Jacobian determinant (summed over the matrix event dims):

    .. math::

        \log|\det J| = \sum_{i=1}^{D} \log \sigma(X_{ii})
                     = -\sum_{i=1}^{D} \operatorname{softplus}(-X_{ii})

    Off-diagonal entries contribute unit Jacobian (identity map); only
    the softplus applied to the diagonal carries a non-trivial factor.

    For correlation-matrix factors (unit diagonal) use
    :class:`CorrCholeskyTransform` instead.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import LowerCholeskyTransform
    >>> T = LowerCholeskyTransform()
    >>> X = lucid.tensor([[0.0, 0.0], [0.5, 0.0]])
    >>> L = T(X)  # softplus on diagonal, raw on strict lower
    """

    event_dim: int = 2
    domain: Constraint = constraints.independent(constraints.real, 2)
    codomain: Constraint = constraints.lower_cholesky

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: lower-triangular extraction with softplus on the diagonal.

        Off-diagonal entries pass through unchanged; diagonal entries are
        mapped via ``softplus`` to keep them strictly positive; the upper
        triangle is masked to zero.
        """
        from lucid.nn.functional.activations import softplus

        # ``x`` is already a ``(*batch, D, D)`` matrix; we mask out the
        # upper triangle (lucid has no native vec→tril helper, so the
        # caller pre-shapes the input).
        D: int = int(x.shape[-1])
        # Build a lower-triangular mask.
        mask: Tensor = _tril_mask(D, x.dtype, x.device)
        diag_mask: Tensor = _eye_mask(D, x.dtype, x.device)
        off_mask: Tensor = mask - diag_mask
        # Diagonal: softplus; off-diagonal: identity; rest: zero.
        diag: Tensor = softplus(x * diag_mask) * diag_mask
        off: Tensor = x * off_mask
        return diag + off

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: invert softplus on the diagonal, leave the rest unchanged.

        Uses ``softplus^{-1}(z) = log(exp(z) - 1)`` on the diagonal entries
        of ``y`` and copies the strict lower triangle directly.
        """
        # Inverse: invert softplus on the diagonal, off-diagonal as-is,
        # zero out the upper triangle.
        D: int = int(y.shape[-1])
        diag_mask: Tensor = _eye_mask(D, y.dtype, y.device)
        tril_mask: Tensor = _tril_mask(D, y.dtype, y.device)
        off_mask: Tensor = tril_mask - diag_mask
        # ``softplus^{-1}(z) = log(expm1(z))``.
        #
        # Two things have to be right here and neither was.  Masking to the
        # diagonal *before* the inverse leaves zeros off it, and
        # ``log(exp(0) - 1) = log(0) = -inf``; multiplying that back by a
        # zero mask gives NaN, so the entire strict lower triangle came back
        # NaN and the transform had no working inverse at all — nor, in
        # consequence, did any ``TransformedDistribution`` built on it, since
        # ``log_prob`` goes through the inverse.  Substituting 1.0 off the
        # diagonal keeps the masked-out lanes finite.
        #
        # ``exp(z) - 1`` also loses every significant digit as ``z`` grows;
        # ``expm1`` is the form that survives a large diagonal.
        safe: Tensor = y * diag_mask + (1.0 - diag_mask)
        diag_in: Tensor = lucid.log(lucid.expm1(safe)) * diag_mask
        off_in: Tensor = y * off_mask
        return diag_in + off_in

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Sum of :math:`\log \sigma(x_{ii})` over the diagonal.

        Off-diagonal entries contribute unit Jacobian; only the softplus
        applied to the diagonal contributes the non-trivial factor
        :math:`\sigma'(x) = \sigma(x)`, computed via the stable identity
        :math:`\log \sigma(x) = -\operatorname{softplus}(-x)`.

        Parameters
        ----------
        x : Tensor
            Pre-transform input matrix (last two dims square).
        y : Tensor
            Post-transform output; ignored — the log-Jacobian uses
            ``x``'s diagonal only.

        Returns
        -------
        Tensor
            Per-sample log-Jacobian (reduces the last two matrix axes).
        """
        D: int = int(x.shape[-1])
        diag_mask: Tensor = _eye_mask(D, x.dtype, x.device)
        # σ(x) = 1/(1+exp(-x)) ⇒ log σ(x) = −softplus(−x).
        from lucid.nn.functional.activations import softplus

        log_sig: Tensor = -softplus(-(x * diag_mask))
        return (log_sig * diag_mask).sum(dim=(-2, -1))


def _eye_mask(D: int, dtype: DTypeLike, device: DeviceLike) -> Tensor:
    """``D×D`` identity mask as a Lucid tensor."""
    return lucid.eye(D, dtype=dtype, device=device)


def _tril_mask(D: int, dtype: DTypeLike, device: DeviceLike) -> Tensor:
    """``D×D`` lower-triangular indicator (1 on/below diag, 0 above)."""
    ones: Tensor = lucid.ones(D, D, dtype=dtype, device=device)
    return lucid.tril(ones)


class AbsTransform(Transform):
    r"""Element-wise absolute value :math:`y = |x|` (folded, **not** bijective).

    Maps :math:`\mathbb{R} \to [0, \infty)` element-wise by folding the
    sign.  Useful when composing with symmetric base distributions
    (e.g. a Normal) to produce a half-Normal or folded-Normal
    pushforward.  ``bijective = False`` and ``sign = +1``.

    Because both :math:`x` and :math:`-x` map to the same :math:`y`,
    the transform is non-invertible.  The pseudo-inverse used here is
    the identity, under the convention that the preimage is assumed to
    lie on the non-negative half-line.

    Notes
    -----
    Forward:  :math:`y = |x|`.

    Pseudo-inverse:  :math:`x = y` (assumes :math:`x \geq 0`).

    Log Jacobian determinant:

    .. math::

        \log\!\left|\frac{\partial y}{\partial x}\right| = 0

    everywhere off the measure-zero set :math:`\{x = 0\}` where the map
    is non-differentiable.

    Folding the sign halves the density: composing with a Normal(0,1)
    base gives the half-Normal density :math:`p(y) = 2 \phi(y)` for
    :math:`y \geq 0`, which requires special handling outside the
    standard change-of-variable formula.  Lucid's
    :class:`TransformedDistribution` does not currently correct for
    this folding factor — apply the :math:`+\log 2` constant manually
    when needed.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import AbsTransform
    >>> T = AbsTransform()
    >>> T(lucid.tensor(-3.0))
    tensor(3.)
    """

    bijective: bool = False
    sign: int = 1
    domain: Constraint = constraints.real
    codomain: Constraint = constraints.positive

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = |x|`."""
        return x.abs()

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Pseudo-inverse: returns ``y`` unchanged (assumes non-negative input)."""
        return y  # convention: non-negative pre-image

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Zero everywhere — :math:`|dy/dx| = 1` outside the kink at zero."""
        return lucid.zeros(tuple(x.shape), dtype=x.dtype, device=x.device)


class IndependentTransform(Transform):
    r"""Reinterpret ``n`` trailing batch dimensions of an inner transform as event dims.

    A thin wrapper that does not modify the forward / inverse maps but
    changes how the log-Jacobian-determinant is *aggregated*: the inner
    transform's per-element :math:`\log|\det J|` is summed over the
    reinterpreted trailing axes, producing a joint event Jacobian
    instead of an element-wise one.  This is the bijection analogue of
    :class:`~lucid.distributions.Independent` for distributions.

    Parameters
    ----------
    transform : Transform
        Inner bijection applied to each element.
    reinterpreted_batch_ndims : int
        Number of trailing batch dimensions to promote to event
        dimensions.

    Notes
    -----
    Forward / inverse: unchanged from the inner transform.

    Log Jacobian determinant (with :math:`n` = ``reinterpreted_batch_ndims``):

    .. math::

        \log|\det J|_{\text{outer}}
        = \sum_{\text{last } n \text{ axes}}
          \log|\det J|_{\text{inner}}

    Use this whenever an element-wise transform (e.g. :class:`ExpTransform`)
    is applied to a vector that you want treated as a single event for the
    purposes of computing :math:`\log p(\mathbf{y})` under the pushforward
    distribution.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import ExpTransform, IndependentTransform
    >>> T = IndependentTransform(ExpTransform(), reinterpreted_batch_ndims=1)
    >>> T(lucid.tensor([0.0, 1.0, 2.0]))
    tensor([1., 2.718, 7.389])
    """

    def __init__(
        self,
        transform: Transform,
        reinterpreted_batch_ndims: int,
    ) -> None:
        """Wrap ``transform`` and promote ``reinterpreted_batch_ndims`` batch dims to event dims."""
        super().__init__()
        self.transform = transform
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self.bijective = transform.bijective
        self.sign = transform.sign
        self.event_dim = transform.event_dim + reinterpreted_batch_ndims

    @override
    @property
    def domain(self) -> Constraint:
        """``independent(transform.domain, reinterpreted_batch_ndims)``."""
        return constraints.independent(
            self.transform.domain, self.reinterpreted_batch_ndims
        )

    @override
    @property
    def codomain(self) -> Constraint:
        """``independent(transform.codomain, reinterpreted_batch_ndims)``."""
        return constraints.independent(
            self.transform.codomain, self.reinterpreted_batch_ndims
        )

    @override
    def forward_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Delegate forward shape inference to the inner transform."""
        return self.transform.forward_shape(shape)

    @override
    def inverse_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Delegate inverse shape inference to the inner transform."""
        return self.transform.inverse_shape(shape)

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Apply the wrapped transform element-wise."""
        return self.transform(x)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Apply the wrapped transform's inverse element-wise."""
        return self.transform._inverse(y)

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Sum the wrapped Jacobian over the reinterpreted trailing dims."""
        lp = self.transform.log_abs_det_jacobian(x, y)
        return _sum_rightmost(lp, self.reinterpreted_batch_ndims)


class ReshapeTransform(Transform):
    r"""Pure-shape reinterpretation of the event shape :math:`y = \mathrm{reshape}(x)`.

    Maps the trailing event tail of a tensor from ``in_shape`` to
    ``out_shape``.  This is a *volume-preserving* bijection — the
    underlying storage is unchanged, only the interpretation of which
    axes constitute the event differs.  The product of dimensions must
    match.

    Parameters
    ----------
    in_shape : tuple[int, ...]
        Event shape of the input tail.
    out_shape : tuple[int, ...]
        Event shape of the output tail.  Must satisfy
        :math:`\prod_i \mathrm{in\_shape}_i = \prod_i \mathrm{out\_shape}_i`.

    Raises
    ------
    ValueError
        If ``prod(in_shape) != prod(out_shape)``.

    Notes
    -----
    Forward:  :math:`\mathbf{y} = \mathrm{reshape}(\mathbf{x},
    \mathrm{out\_shape})`.

    Inverse:  :math:`\mathbf{x} = \mathrm{reshape}(\mathbf{y},
    \mathrm{in\_shape})`.

    Log Jacobian determinant:

    .. math::

        \log|\det J| = 0

    everywhere — a reshape preserves Lebesgue measure exactly.  Useful
    for stitching together transforms that expect different event
    layouts (e.g. flatten a matrix event for a Cholesky-style
    transform, then reshape back).

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import ReshapeTransform
    >>> T = ReshapeTransform(in_shape=(2, 3), out_shape=(6,))
    >>> T(lucid.zeros(2, 3)).shape
    (6,)
    """

    def __init__(
        self,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
    ) -> None:
        """Validate that ``in_shape`` and ``out_shape`` have the same total size and store them.

        Raises
        ------
        ValueError
            If ``prod(in_shape) != prod(out_shape)``.
        """
        super().__init__()
        import math as _math

        in_n = _math.prod(in_shape) if in_shape else 1
        out_n = _math.prod(out_shape) if out_shape else 1
        if in_n != out_n:
            raise ValueError(
                f"ReshapeTransform: in_shape {in_shape} ({in_n} elements) "
                f"!= out_shape {out_shape} ({out_n} elements)."
            )
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.event_dim = max(len(in_shape), len(out_shape))

    @override
    @property
    def domain(self) -> Constraint:
        """``independent(real, len(in_shape))``."""
        return constraints.independent(constraints.real, len(self.in_shape))

    @override
    @property
    def codomain(self) -> Constraint:
        """``independent(real, len(out_shape))``."""
        return constraints.independent(constraints.real, len(self.out_shape))

    @override
    def forward_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Replace the trailing ``in_shape`` dimensions with ``out_shape``."""
        if len(shape) < len(self.in_shape):
            return tuple(shape)
        cut = len(shape) - len(self.in_shape)
        return tuple(shape[:cut]) + tuple(self.out_shape)

    @override
    def inverse_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Replace the trailing ``out_shape`` dimensions with ``in_shape``."""
        if len(shape) < len(self.out_shape):
            return tuple(shape)
        cut = len(shape) - len(self.out_shape)
        return tuple(shape[:cut]) + tuple(self.in_shape)

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Reshape the event tail of ``x`` from ``in_shape`` to ``out_shape``."""
        batch = x.shape[: x.dim() - len(self.in_shape)]
        return x.reshape(*batch, *self.out_shape)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Reshape the event tail of ``y`` from ``out_shape`` back to ``in_shape``."""
        batch = y.shape[: y.dim() - len(self.out_shape)]
        return y.reshape(*batch, *self.in_shape)

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Zero Jacobian — pure reshape is volume-preserving."""
        batch = x.shape[: x.dim() - len(self.in_shape)]
        return lucid.zeros(tuple(batch), dtype=x.dtype, device=x.device)


class CorrCholeskyTransform(Transform):
    r"""Bijection from :math:`\mathbb{R}^{d(d-1)/2}` to Cholesky factors of correlation matrices.

    Maps an unconstrained vector of length :math:`d(d-1)/2` to the
    lower-triangular Cholesky factor :math:`L` of a :math:`d \times d`
    correlation matrix (so :math:`L L^\top` is positive-definite with
    **unit diagonal**).  This is the standard reparameterisation used by
    Stan, by Lewandowski–Kurowicka–Joe (LKJ) priors, and by the reference
    framework's correlation-matrix flow.  ``event_dim = 2``.

    The construction applies ``tanh`` to the free parameters, treats each
    resulting value as a *partial correlation*, then normalises each row
    of the lower triangle so the rows of :math:`L` have unit 2-norm
    (which forces :math:`L L^\top` to have unit diagonal).

    Parameters
    ----------
    dim : int or None, optional
        Size :math:`d \geq 2` of the square correlation matrix.  If ``None``,
        inferred from the trailing input dimension :math:`d(d-1)/2` or matrix
        shape :math:`(d, d)`.

    Notes
    -----
    Forward (row-by-row stick-breaking on the unit sphere):

    .. math::

        z_{ij} = \tanh(x_{ij}), \qquad
        L_{ij} = z_{ij}\sqrt{1 - \sum_{k < j} L_{ik}^2}\;\;(i > j),
        \qquad
        L_{ii} = \sqrt{1 - \sum_{k < i} L_{ik}^2}

    Inverse:

    .. math::

        z_{ij} = \frac{L_{ij}}{\sqrt{1 - \sum_{k < j} L_{ik}^2}},
        \qquad
        x_{ij} = \mathrm{arctanh}(z_{ij})

    Log Jacobian determinant (closed form from Lewandowski et al., 2009):

    .. math::

        \log|\det J|
        = \sum_{i>j} \log(1 - \tanh^2(x_{ij}))
        + \sum_{i=1}^{d-1} (d - i)\log L_{ii}

    Combine with the :class:`~lucid.distributions.LKJCholesky`
    distribution (or equivalent) to learn unconstrained correlation
    matrices end-to-end.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import CorrCholeskyTransform
    >>> T = CorrCholeskyTransform(dim=3)
    >>> x = lucid.zeros(3)  # d*(d-1)/2 = 3 free params
    >>> L = T(x)  # 3x3 lower-triangular Cholesky of identity
    """

    domain: Constraint = constraints.real_vector
    codomain: Constraint = constraints.corr_cholesky
    event_dim: int = 2

    def __init__(self, dim: int | None = None) -> None:
        """Store the correlation-matrix dimension ``dim`` (or ``None`` to infer from inputs)."""
        super().__init__()
        if dim is not None and dim < 2:
            raise ValueError(f"CorrCholeskyTransform: dim must be ≥ 2, got {dim}.")
        self.dim = dim

    def _resolve_dim_from_free(self, free_n: int) -> int:
        if self.dim is not None:
            if free_n != self._free_ndim:
                raise ValueError(
                    f"CorrCholeskyTransform: expected {self._free_ndim} free parameters "
                    f"for dim={self.dim}, got {free_n}."
                )
            return self.dim
        d = round((math.sqrt(1 + 8 * free_n) + 1) / 2)
        if d * (d - 1) // 2 != free_n or d < 2:
            raise ValueError(
                f"CorrCholeskyTransform: invalid unconstrained dimension {free_n}"
            )
        return d

    def _resolve_dim_from_matrix(self, mat_d: int) -> int:
        if self.dim is not None:
            return self.dim
        if mat_d < 2:
            raise ValueError(f"CorrCholeskyTransform: dim must be ≥ 2, got {mat_d}.")
        return mat_d

    # Dimension of the free vector
    @property
    def _free_ndim(self) -> int:
        """Number of free parameters in the unconstrained representation: :math:`d(d-1)/2`."""
        if self.dim is None:
            raise ValueError("CorrCholeskyTransform: dim is not fixed on this instance")
        d = self.dim
        return d * (d - 1) // 2

    @override
    def forward_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Map ``(..., d*(d-1)/2)`` to ``(..., d, d)``."""
        if len(shape) < 1:
            raise ValueError(f"Event shape {shape} must have at least 1 dimension")
        d = self._resolve_dim_from_free(shape[-1])
        return shape[:-1] + (d, d)

    @override
    def inverse_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Map ``(..., d, d)`` to ``(..., d*(d-1)/2)``."""
        if len(shape) < 2:
            raise ValueError(f"Event shape {shape} must have at least 2 dimensions")
        d = self._resolve_dim_from_matrix(shape[-1])
        return shape[:-2] + (d * (d - 1) // 2,)

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Map unconstrained ``(..., d*(d-1)/2)`` vector to ``(..., d, d)`` Chol."""
        *batch, free_n = x.shape
        d = self._resolve_dim_from_free(int(free_n))
        z = x.tanh()  # (..., free_ndim)
        return self._build_chol(z, batch, d, x.dtype, x.device)

    def _build_chol(
        self,
        z: Tensor,
        batch: list[int],
        d: int,
        dtype: DTypeLike,
        device: DeviceLike,
    ) -> Tensor:
        """Build the Cholesky factor row by row from ``z`` (tanh-ed free params)."""
        row_tensors: list[Tensor] = []
        idx = 0
        cumsum_sq: list[Tensor] = []  # cumulative row norms squared

        for row in range(d):
            if row == 0:
                # L[0, 0] = 1, rest of row = 0
                r = lucid.zeros(*batch, d, dtype=dtype, device=device)
                # set r[..., 0] = 1
                ones_val = lucid.ones(*batch, 1, dtype=dtype, device=device)
                r = lucid.cat(
                    [ones_val, lucid.zeros(*batch, d - 1, dtype=dtype, device=device)],
                    dim=-1,
                )
                row_tensors.append(r.unsqueeze(-2))
                cumsum_sq.append(lucid.ones(*batch, 1, dtype=dtype, device=device))
            else:
                # off-diagonal: z[..., idx:idx+row]
                n_free = row  # number of free params in this row
                zr = z[..., idx : idx + n_free]  # (..., row)
                idx += n_free

                # Scale: z_{row, col} goes into L_{row, col}; then normalise.
                # Build row with tanh values in positions 0..row-1, 0 elsewhere.
                if n_free > 0:
                    # Normalise: L[row, :row] = z_row * sqrt(cumprod(1 - z²))
                    # Actually use the standard normalisation:
                    # L[row,0] := z[0]
                    # L[row,1] = z[1]*sqrt(1-z[0]²)
                    # L[row,2] = z[2]*sqrt(1-z[0]²-z[1]²*(...))
                    # i.e. L[row, j] = z[j] * sqrt(1 - Σ_{k<j} L[row,k]²)
                    # We compute this step-by-step.
                    elems: list[Tensor] = []
                    cum_sq = lucid.zeros(*batch, dtype=dtype, device=device)
                    for j in range(n_free):
                        scale = (1.0 - cum_sq).sqrt()
                        # Clamp to avoid sqrt of negative from fp noise.
                        scale = lucid.where(
                            scale < 0.0,
                            lucid.zeros_like(scale),
                            scale,
                        )
                        elem = zr[..., j] * scale
                        elems.append(elem.unsqueeze(-1))
                        cum_sq = cum_sq + elem * elem

                    off = lucid.cat(elems, dim=-1)  # (..., row)
                    # diagonal element: sqrt(1 - ||off||²)
                    raw_diag_sq = 1.0 - (off * off).sum(dim=-1)
                    diag_sq = lucid.where(
                        raw_diag_sq < 0.0,
                        lucid.zeros_like(raw_diag_sq),
                        raw_diag_sq,
                    )
                    diag = diag_sq.sqrt()  # (...,)
                    # pad to length d: [off | diag | zeros]
                    zero_pad = lucid.zeros(
                        *batch, d - row - 1, dtype=dtype, device=device
                    )
                    full_row = lucid.cat([off, diag.unsqueeze(-1), zero_pad], dim=-1)
                else:
                    full_row = lucid.zeros(*batch, d, dtype=dtype, device=device)

                row_tensors.append(full_row.unsqueeze(-2))

        # Stack rows → (..., d, d)
        return lucid.cat(row_tensors, dim=-2)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Extract the free parameters ``x`` from a Cholesky factor ``L``."""
        *batch, _, mat_d = y.shape
        d = self._resolve_dim_from_matrix(int(mat_d))
        # Each row i of L: first i elements are off-diagonal, element i is diag.
        # Back-solve for z from L[row, :row].
        free: list[Tensor] = []
        for row in range(1, d):
            off = y[..., row, :row]  # (..., row)
            # Recover z from the cumulative parameterisation.
            elems: list[Tensor] = []
            cum_sq = lucid.zeros(*batch, dtype=y.dtype, device=y.device)
            for j in range(row):
                raw_scale = (1.0 - cum_sq).sqrt()
                scale = lucid.where(
                    raw_scale < 1e-8,
                    lucid.full_like(raw_scale, 1e-8),
                    raw_scale,
                )
                z_raw = off[..., j] / scale
                z_j = lucid.where(
                    z_raw < -1.0 + 1e-6,
                    lucid.full_like(z_raw, -1.0 + 1e-6),
                    lucid.where(
                        z_raw > 1.0 - 1e-6,
                        lucid.full_like(z_raw, 1.0 - 1e-6),
                        z_raw,
                    ),
                )
                elems.append(z_j.unsqueeze(-1))
                cum_sq = cum_sq + (off[..., j]) ** 2

            free.append(lucid.cat(elems, dim=-1))  # (..., row)

        z = lucid.cat(free, dim=-1)  # (..., d*(d-1)/2)
        # inverse tanh: arctanh(z) = 0.5 * log((1+z)/(1-z))
        denom = lucid.where(
            (1.0 - z) < 1e-8,
            lucid.full_like(z, 1e-8),
            1.0 - z,
        )
        return 0.5 * ((1.0 + z) / denom).log()

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Log-Jacobian of the unconstrained-to-Cholesky correlation map.

        Combines the tanh contribution
        :math:`\sum_{i>j} \log(1 - \tanh^2(x_{ij}))` with the
        closed-form Lewandowski normalisation terms expressed directly
        from the output diagonal entries of ``y``.

        Parameters
        ----------
        x : Tensor
            Unconstrained parameters (strictly-lower-triangular real
            entries).
        y : Tensor
            Output Cholesky correlation matrix.

        Returns
        -------
        Tensor
            Per-sample log-Jacobian.
        """
        # The domain and codomain have different dimensions, so this is the
        # log-Jacobian of ``x`` against the *free* (strictly lower) entries
        # of ``y``, which is the only determinant that is defined here.
        #
        # Two factors compose.  ``tanh`` contributes ``Σ log(1 - tanh²(x))``
        # over the free vector.  The stick-breaking that follows scales the
        # entry at ``(i, j)`` by ``sqrt(∏_{k<j} (1 - r_{ik}²))``, and that
        # product is exactly ``1 - Σ_{k≤j-1} y_{ik}²`` — so the whole
        # contribution is half the log of ``1 - cumsum(y², -1)`` read at the
        # positions two or more columns below the diagonal.
        #
        # What stood here was a guess: the comment above it sketched three
        # different formulas and settled on weighting ``log L_{rr}`` by
        # ``d - row``.  It is off by a sample-dependent amount — ``-0.5973``
        # against ``-0.3885`` at ``d = 3`` — which is a wrong density rather
        # than a wrong constant, so nothing downstream normalises it away.
        d = self._resolve_dim_from_matrix(int(y.shape[-1]))
        if d < 2:
            return lucid.zeros(tuple(x.shape[:-1]), dtype=x.dtype, device=x.device)

        z = x.tanh()
        one_minus_z2 = lucid.clip(1.0 - z * z, min=1e-12)
        tanh_logdet = one_minus_z2.log().sum(dim=-1)

        # ``1 - cumsum(y², -1)`` at the entries with ``col <= row - 2``.
        # Masked-out lanes are set to 1.0 *before* the log: a zero there
        # would make ``log(0) * 0`` a NaN and poison the whole sum.
        remaining = 1.0 - (y * y).cumsum(dim=-1)
        mask = lucid.tril(lucid.ones(d, d, dtype=x.dtype, device=x.device), diagonal=-2)
        safe = lucid.clip(remaining, min=1e-12) * mask + (1.0 - mask)
        stick_logdet = 0.5 * (safe.log() * mask).sum(dim=-1).sum(dim=-1)

        return tanh_logdet + stick_logdet


class CumulativeDistributionTransform(Transform):
    r"""Probability-integral transform :math:`y = F(x)` via a base distribution's CDF.

    Pushes any continuous random variable through its own CDF to obtain
    a :math:`\mathrm{Uniform}(0, 1)` random variable — the classical
    *probability integral transform* of Smirnov.  Useful as a copula
    building block, for goodness-of-fit testing, and for stitching
    together heterogeneous flows.  ``event_dim = 0``.

    Parameters
    ----------
    distribution : Distribution
        Continuous distribution whose :meth:`~Distribution.cdf` and
        :meth:`~Distribution.icdf` define the transform.  Both methods
        must be implemented and the distribution must be continuous
        (i.e., have a density) for the transform to be a true bijection.

    Notes
    -----
    Forward (CDF):

    .. math::

        y = F(x)

    Inverse (quantile / ICDF):

    .. math::

        x = F^{-1}(y)

    Log Jacobian determinant equals the log-density of the base
    distribution at :math:`x`:

    .. math::

        \log\!\left|\frac{\partial F(x)}{\partial x}\right| = \log p(x)

    By the probability-integral transform, if :math:`X \sim p` then
    :math:`F(X) \sim \mathrm{Uniform}(0, 1)`.  Conversely, if
    :math:`U \sim \mathrm{Uniform}(0, 1)` then :math:`F^{-1}(U) \sim p`,
    which is the inverse-CDF sampling identity used throughout
    Lucid's :meth:`~Distribution.rsample` implementations.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Normal
    >>> from lucid.distributions.transforms import CumulativeDistributionTransform
    >>> T = CumulativeDistributionTransform(Normal(loc=0.0, scale=1.0))
    >>> T(lucid.tensor(0.0))  # F(0) = 0.5
    tensor(0.5)
    """

    codomain: Constraint = constraints.unit_interval
    event_dim: int = 0

    def __init__(self, distribution: Distribution) -> None:
        """Store the distribution whose CDF/ICDF pair defines the transform."""
        super().__init__()
        self.distribution = distribution

    @override
    @property
    def domain(self) -> Constraint:
        """Input domain inherited from ``self.distribution.support``."""
        support = self.distribution.support
        if support is None:
            return constraints.real
        return support

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = F(x)` via ``distribution.cdf``."""
        return self.distribution.cdf(x)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = F^{-1}(y)` via ``distribution.icdf``."""
        return self.distribution.icdf(y)

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r""":math:`\log|F'(x)| = \log p(x)` — the base distribution's log-density at ``x``."""
        return self.distribution.log_prob(x)


from lucid.distributions.constraint_registry import (
    CatTransform as CatTransform,
    ConstraintRegistry as ConstraintRegistry,
    PositiveDefiniteTransform as PositiveDefiniteTransform,
    SoftplusTransform as SoftplusTransform,
    StackTransform as StackTransform,
    biject_to as biject_to,
    transform_to as transform_to,
)
