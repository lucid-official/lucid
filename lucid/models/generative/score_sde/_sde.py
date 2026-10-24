r"""The three stochastic differential equations of Song et al. (2021).

Score-based generative modelling and denoising diffusion turn out to be
two discretisations of the same object.  This module is that object: a
forward SDE that carries data to noise, the reverse-time SDE that carries
it back, and the deterministic ODE which shares the reverse SDE's
marginals at every instant.

    dx = f(x, t) dt + g(t) dw                                    forward
    dx = [f(x, t) - g(t)^2 * score] dt + g(t) dw̄            reverse (6)
    dx = [f(x, t) - g(t)^2 * score / 2] dt          probability flow (13)

The point of writing it once is that ``VE`` and ``VP`` are not new models
— they are the continuous limits of NCSN and DDPM respectively, both of
which this zoo already ships.  Everything downstream (the samplers, the
loss, the likelihood) is written against the interface and does not know
which of the three it has.
"""

import math
from abc import ABC, abstractmethod
from typing import override

import lucid
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor

__all__ = ["SDE", "VESDE", "VPSDE", "SubVPSDE", "make_sde"]


def _expand(scalar: Tensor, like: Tensor) -> Tensor:
    """Give a per-sample scalar the trailing axes and the dtype of ``like``."""
    shaped = scalar.reshape(int(scalar.shape[0]), *([1] * (like.ndim - 1)))
    return shaped.to(like.dtype)


def _wide(t: Tensor) -> Tensor:
    """``t`` at float32 or wider, for a coefficient of the form ``1 - e^{-z}``.

    Near ``t_min`` that ``z`` is ~1e-4, and half precision rounds ``e^{-z}``
    to 1: the standard deviation came out 0 and the score infinite.  Such
    coefficients are worked out here and cast back to ``t``'s dtype once.
    """
    return t.to(_float32_or_wider(t.dtype))


class SDE(ABC):
    """What the samplers and the loss need an SDE to provide.

    Notes
    -----
    Four things, and no more: the drift and diffusion of the forward
    process, the closed-form perturbation kernel that makes training a
    single network evaluation, and the prior the reverse process starts
    from.  Everything else — reverse SDE, probability-flow ODE,
    Predictor-Corrector — is derived from these and lives with the model.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.score_sde._sde import make_sde
    >>> sde = make_sde("vp")
    >>> x = lucid.zeros(2, 3, 8, 8)
    >>> t = lucid.zeros(2) + 0.5
    >>> sde.drift(x, t).shape
    (2, 3, 8, 8)

    The drift follows the state and the diffusion does not: it is a
    scalar per sample, which is what makes the noise isotropic.

    >>> sde.diffusion(t).shape
    (2,)

    ``t_min`` is not zero. The reverse process is integrated backwards
    and the score blows up at the origin, so sampling stops just short
    of it.

    >>> sde.t_min
    0.001
    """

    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """:math:`f(x, t)`, the deterministic part of the forward SDE."""

    @abstractmethod
    def diffusion(self, t: Tensor) -> Tensor:
        """:math:`g(t)`, per sample."""

    @abstractmethod
    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        r"""The perturbation kernel :math:`p_{0t}(x(t) \mid x(0))`.

        Parameters
        ----------
        x : Tensor
            The clean sample, ``(N, ...)``.
        t : Tensor
            Times, ``(N,)``.

        Returns
        -------
        mean, std : Tensor
            Mean broadcast to ``x``'s shape, and the per-sample standard
            deviation.  Having this in closed form is what lets the loss
            perturb a sample in one step rather than integrating.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.score_sde._sde import make_sde
        >>> x = lucid.ones((2, 3, 4, 4))
        >>> t = lucid.tensor([0.1, 0.9])
        >>> vp = make_sde("vp")
        >>> mean, std = vp.marginal_prob(x, t)
        >>> mean.shape, std.shape
        ((2, 3, 4, 4), (2,))
        >>> scale = mean[:, 0, 0, 0]  # x is all ones, so this is the mean's factor
        >>> bool(((scale**2 + std**2 - 1.0).abs() < 1e-5).all().item())  # VP
        True
        >>> ve = make_sde("ve")
        >>> mean, std = ve.marginal_prob(x, t)
        >>> mean is x  # VE never moves the mean
        True
        >>> bool(lucid.allclose(std, ve.sigma(t)))
        True
        """

    @abstractmethod
    def prior_sampling(
        self, shape: tuple[int, ...], device: str, *, dtype: lucid.dtype | None = None
    ) -> Tensor:
        """Draw from the distribution the reverse process starts at.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the draw, ``(N, C, H, W)``.
        device : str
            Where to draw.
        dtype : lucid.dtype or None, optional, keyword-only
            The sampling model's, so a float16 model starts from a float16
            draw; ``None`` is the default dtype.

        Returns
        -------
        Tensor
            ``shape``-shaped prior sample.
        """

    @property
    @abstractmethod
    def t_min(self) -> float:
        """The earliest time the SDE is defined at."""

    @property
    @abstractmethod
    def prior_variance(self) -> float:
        r"""Variance of the distribution :meth:`prior_sampling` draws from.

        Needed by the exact-likelihood computation, which has to evaluate
        :math:`\log p_T` rather than merely sample it — and VE's prior is
        as wide as ``sigma_max`` while VP's is unit, so assuming either
        would be wrong for the other.
        """


class VESDE(SDE):
    r"""Variance Exploding — the continuous limit of NCSN.

    Parameters
    ----------
    sigma_min, sigma_max : float
        The geometric noise range.

    Notes
    -----
    Song et al. (2021), equation 30:

    .. math::

        dx = \sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t
             \sqrt{2\log\frac{\sigma_{\max}}{\sigma_{\min}}}\; dw,
        \qquad t \in (0, 1],

    with the perturbation kernel of equation 31,
    :math:`\mathcal{N}(x(t); x(0), \sigma(t)^2 I)` — the mean does not
    move, which is what "variance exploding" means.

    Defined on the **open** interval.  The paper is explicit that
    :math:`\sigma` is not differentiable at zero because
    :math:`\sigma(0) = 0` by the discrete convention while
    :math:`\sigma(0^+) = \sigma_{\min}`, "causing the VE SDE ... undefined
    for t = 0".  Sampling therefore stops at :attr:`t_min` rather than
    running to zero.

    Examples
    --------
    >>> from lucid.models.generative.score_sde._sde import VESDE
    >>> sde = VESDE(sigma_min=0.01, sigma_max=50.0)
    >>> sde.prior_variance
    2500.0

    Variance exploding: the prior is as wide as sigma_max squared, so
    the terminal distribution is nothing like the data and the model
    learns the whole way down.
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0) -> None:
        """Initialise the SDE. See the class docstring for parameters."""
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError(
                f"need 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}"
            )
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._ratio = sigma_max / sigma_min
        self._log_ratio = math.log(self._ratio)

    @override
    @property
    def t_min(self) -> float:
        """Sampling stops here — see the class docstring on ``t = 0``."""
        return 1e-5

    def sigma(self, t: Tensor) -> Tensor:
        r""":math:`\sigma(t) = \sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t`."""
        return self.sigma_min * (self._ratio**t)

    @property
    @override
    def prior_variance(self) -> float:
        r""":math:`\sigma_{\max}^2`."""
        return self.sigma_max**2

    @override
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Zero — the VE process only adds variance."""
        return lucid.zeros_like(x)

    @override
    def diffusion(self, t: Tensor) -> Tensor:
        """Equation 30's coefficient."""
        return self.sigma(t) * math.sqrt(2.0 * self._log_ratio)

    @override
    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Equation 31 — the mean is the sample itself."""
        return x, self.sigma(t)

    @override
    def prior_sampling(
        self, shape: tuple[int, ...], device: str, *, dtype: lucid.dtype | None = None
    ) -> Tensor:
        """:math:`\\mathcal{N}(0, \\sigma_{\\max}^2 I)`."""
        return lucid.randn(shape, dtype=dtype, device=device) * self.sigma_max


class VPSDE(SDE):
    r"""Variance Preserving — the continuous limit of DDPM.

    Parameters
    ----------
    beta_min, beta_max : float
        The linear rate range.  The paper's values, "to match the
        settings in Ho et al. (2020)".

    Notes
    -----
    Song et al. (2021), equation 32:

    .. math::

        dx = -\tfrac{1}{2}\big(\bar\beta_{\min}
             + t(\bar\beta_{\max} - \bar\beta_{\min})\big) x\, dt
             + \sqrt{\bar\beta_{\min}
             + t(\bar\beta_{\max} - \bar\beta_{\min})}\; dw,

    with the perturbation kernel of equation 33.  Starting from unit
    variance the process keeps it, which is the difference from VE and
    the reason the prior is a standard normal rather than a wide one.

    Examples
    --------
    >>> from lucid.models.generative.score_sde._sde import VPSDE
    >>> sde = VPSDE(beta_min=0.1, beta_max=20.0)
    >>> sde.prior_variance
    1.0

    Variance preserving: the process is scaled so the terminal
    distribution is the unit Gaussian, which is the same place DDPM
    ends up and why the two are usually interchangeable.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        """Initialise the SDE. See the class docstring for parameters."""
        if not 0.0 < beta_min < beta_max:
            raise ValueError(
                f"need 0 < beta_min < beta_max, got {beta_min}, {beta_max}"
            )
        self.beta_min = beta_min
        self.beta_max = beta_max

    @override
    @property
    def t_min(self) -> float:
        """The paper's sampling epsilon, chosen so the variance matches DDPM's."""
        return 1e-3

    @property
    @override
    def prior_variance(self) -> float:
        """One — the process is variance preserving."""
        return 1.0

    def beta(self, t: Tensor) -> Tensor:
        r""":math:`\beta(t) = \bar\beta_{\min} + t(\bar\beta_{\max} - \bar\beta_{\min})`."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _log_mean_coeff(self, t: Tensor) -> Tensor:
        """The exponent of equation 33's mean scale."""
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    @override
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """:math:`-\\tfrac{1}{2}\\beta(t)x`."""
        return -0.5 * _expand(self.beta(t), x) * x

    @override
    def diffusion(self, t: Tensor) -> Tensor:
        """:math:`\\sqrt{\\beta(t)}`."""
        return self.beta(t) ** 0.5

    @override
    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Equation 33."""
        log_coeff = self._log_mean_coeff(_wide(t))
        mean = _expand(log_coeff.exp(), x) * x
        std = (1.0 - (2.0 * log_coeff).exp()) ** 0.5
        return mean, std.to(t.dtype)

    @override
    def prior_sampling(
        self, shape: tuple[int, ...], device: str, *, dtype: lucid.dtype | None = None
    ) -> Tensor:
        """:math:`\\mathcal{N}(0, I)`."""
        return lucid.randn(shape, dtype=dtype, device=device)


class SubVPSDE(VPSDE):
    r"""Sub-VP — the SDE this paper introduces, and its best likelihoods.

    Notes
    -----
    Song et al. (2021), equation 12:

    .. math::

        dx = -\tfrac{1}{2}\beta(t)x\,dt
             + \sqrt{\beta(t)\big(1 - e^{-2\int_0^t\beta(s)ds}\big)}\; dw.

    Same drift as VP, a diffusion damped by how far the process has
    already travelled.  "The variance of the stochastic process induced
    by Eq. (12) is always bounded by the VP SDE at every intermediate
    time step", which is why the name and why the paper reports its best
    bits-per-dimension here.

    The perturbation kernel differs from VP's in exactly one place —
    equation 29 gives ``[1 - e^{-∫β}]^2`` where VP has ``1 - e^{-∫β}`` —
    so the standard deviation is that quantity rather than its square
    root.  Getting that wrong leaves a model that still trains.

    Examples
    --------
    >>> from lucid.models.generative.score_sde._sde import SubVPSDE
    >>> sde = SubVPSDE(beta_min=0.1, beta_max=20.0)
    >>> sde.prior_variance
    1.0

    The same terminal variance as VP with a smaller one along the way,
    which is what the "sub" refers to — it reports better likelihoods at
    the same endpoints.
    """

    def _integral_beta(self, t: Tensor) -> Tensor:
        r""":math:`\int_0^t \beta(s)\,ds`, in closed form."""
        return 0.5 * t**2 * (self.beta_max - self.beta_min) + t * self.beta_min

    @override
    def diffusion(self, t: Tensor) -> Tensor:
        """Equation 12's damped coefficient."""
        wide = _wide(t)
        discount = 1.0 - (-2.0 * self._integral_beta(wide)).exp()
        return ((self.beta(wide) * discount) ** 0.5).to(t.dtype)

    @override
    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Equation 29's sub-VP row — note the square, not a square root."""
        log_coeff = self._log_mean_coeff(_wide(t))
        mean = _expand(log_coeff.exp(), x) * x
        std = 1.0 - (2.0 * log_coeff).exp()
        return mean, std.to(t.dtype)


def make_sde(
    kind: str,
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
) -> SDE:
    """Build one of the three by name.

    Parameters
    ----------
    kind : {"ve", "vp", "subvp"}
        Which SDE.
    sigma_min, sigma_max : float
        VE only.
    beta_min, beta_max : float
        VP and sub-VP.

    Returns
    -------
    SDE
        The requested process.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three.

    Examples
    --------
    >>> from lucid.models.generative.score_sde._sde import make_sde
    >>> [type(make_sde(k)).__name__ for k in ("ve", "vp", "subvp")]
    ['VESDE', 'VPSDE', 'SubVPSDE']

    The three the literature settled on, by name rather than by import,
    so a config can carry the choice as a string.
    """
    if kind == "ve":
        return VESDE(sigma_min, sigma_max)
    if kind == "vp":
        return VPSDE(beta_min, beta_max)
    if kind == "subvp":
        return SubVPSDE(beta_min, beta_max)
    raise ValueError(f"sde_type must be 've', 'vp' or 'subvp', got {kind!r}")
