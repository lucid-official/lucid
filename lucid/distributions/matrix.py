"""Matrix-valued distributions: ``Wishart``, ``LKJCholesky``.

Both distributions operate over symmetric positive-definite matrices or
their Cholesky factors, making them useful for learning covariance and
correlation structure in Bayesian models.

All computation is done in pure Lucid — no external libraries.
"""

import math
from collections.abc import Sequence
from typing import Self, override

import lucid
import lucid.linalg as _la
import lucid.special as _sp
from lucid._engine_scope import untraced
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import _align_device, _as_tensor, _lazy_param
from lucid.distributions.constraints import (
    Constraint,
    corr_cholesky,
    lower_cholesky,
    positive,
    positive_definite,
)
from lucid.distributions.distribution import Distribution
from lucid.distributions.gamma import Beta, _sample_standard_gamma, _standard_gamma
from lucid.distributions.multivariate import _precision_of, _scale_tril_of

# ── Wishart ───────────────────────────────────────────────────────────────────


class Wishart(Distribution):
    r"""Wishart distribution over symmetric positive-definite matrices.

    Matrix-variate generalisation of the
    :class:`~lucid.distributions.Chi2` / Gamma distribution: the
    distribution of the (unnormalised) sample covariance matrix
    :math:`S = \sum_{i=1}^{\nu} \mathbf{z}_i \mathbf{z}_i^\top` formed
    from :math:`\nu` independent Multivariate Normal draws
    :math:`\mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \Sigma)`.  It is the
    **conjugate prior** of the precision matrix of a Multivariate Normal
    with known mean.

    Specify exactly one of ``covariance_matrix``, ``precision_matrix``, or
    ``scale_tril`` — the other parameterisations are derived internally
    via Cholesky factorisation.

    Parameters
    ----------
    df : Tensor or float
        Degrees of freedom :math:`\nu > D - 1` where :math:`D` is the
        matrix dimension.  Values :math:`\nu \leq D - 1` violate the
        positive-definiteness guarantee.
    covariance_matrix : Tensor, optional
        Positive-definite scale matrix :math:`\Sigma` of shape
        ``(..., D, D)``.
    precision_matrix : Tensor, optional
        Positive-definite precision matrix :math:`\Sigma^{-1}` of shape
        ``(..., D, D)``.
    scale_tril : Tensor, optional
        Lower-triangular Cholesky factor :math:`L` of :math:`\Sigma`
        (so :math:`\Sigma = L L^\top`), shape ``(..., D, D)`` with
        positive diagonal.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on the cone of symmetric positive-definite
    :math:`D \times D` matrices:

    .. math::

        p(\mathbf{X}; \nu, \Sigma) =
            \frac{|\mathbf{X}|^{(\nu - D - 1)/2}
                  \exp\!\bigl(-\tfrac{1}{2}\,\mathrm{tr}(\Sigma^{-1}\mathbf{X})\bigr)}
                 {2^{\nu D / 2} |\Sigma|^{\nu/2} \Gamma_D(\nu/2)}

    where :math:`\Gamma_D` is the multivariate gamma function:

    .. math::

        \Gamma_D(a) = \pi^{D(D-1)/4} \prod_{i=1}^{D} \Gamma\!\left(a + \tfrac{1 - i}{2}\right)

    Moments:

    .. math::

        \mathbb{E}[\mathbf{X}] = \nu \Sigma, \qquad
        \mathrm{Var}[X_{ij}] =
            \nu (\Sigma_{ij}^2 + \Sigma_{ii} \Sigma_{jj})

    Special cases / relations:

    * :math:`D = 1` → :math:`\mathrm{Wishart}(\nu, \sigma^2) =
      \sigma^2 \chi^2(\nu)`.
    * Inverse: if :math:`\mathbf{X} \sim \mathrm{Wishart}(\nu, \Sigma)`
      then :math:`\mathbf{X}^{-1}` follows the inverse-Wishart
      distribution, the conjugate prior of the **covariance** (not
      precision) matrix.
    * Bartlett decomposition: :math:`\mathbf{X} = L A A^\top L^\top`
      with :math:`A` lower-triangular containing :math:`\chi^2` draws on
      the diagonal and standard Normals below — used for sampling.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Wishart
    >>> Sigma = lucid.tensor([[2.0, 0.5], [0.5, 1.0]])
    >>> d = Wishart(df=5.0, covariance_matrix=Sigma)
    >>> d.sample((4,)).shape
    (4, 2, 2)
    """

    # ``df`` must also exceed ``D - 1``; the constructor refuses that whether
    # or not it validates, as the reference framework's does, so the table
    # only needs it positive.
    arg_constraints = {
        "covariance_matrix": positive_definite,
        "precision_matrix": positive_definite,
        "scale_tril": lower_cholesky,
        "df": positive,
    }
    has_rsample: bool = True

    def __init__(
        self,
        df: Tensor | float,
        covariance_matrix: Tensor | None = None,
        *,
        precision_matrix: Tensor | None = None,
        scale_tril: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Wishart distribution.

        Converts ``covariance_matrix`` or ``precision_matrix`` to a Cholesky
        factor internally.  Exactly one scale specification must be provided.

        Parameters
        ----------
        df : Tensor | float
            Degrees of freedom :math:`\nu > D - 1` where :math:`D` is the
            matrix dimension.  Values :math:`\nu \leq D - 1` violate the
            positive-definiteness guarantee.
        covariance_matrix : Tensor | None, optional
            Positive-definite scale matrix :math:`\Sigma` of shape
            ``(..., D, D)``.
        precision_matrix : Tensor | None, optional
            Positive-definite precision matrix :math:`\Sigma^{-1}` of shape
            ``(..., D, D)``.  Converted via :math:`\Sigma = (\Sigma^{-1})^{-1}`.
        scale_tril : Tensor | None, optional
            Lower-triangular Cholesky factor :math:`L` of :math:`\Sigma`,
            shape ``(..., D, D)`` with positive diagonal.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If not exactly one scale specification is given, if
            :math:`\nu \leq D - 1`, or — under validation — if the matrix
            given is not positive-definite (a ``scale_tril``: not
            lower-triangular with a positive diagonal).
        lucid.linalg.LinAlgError
            With validation off, if a covariance or precision matrix has no
            Cholesky factorisation.
        """
        n_spec = sum(x is not None for x in (covariance_matrix, precision_matrix, scale_tril))
        if n_spec != 1:
            raise ValueError(
                "Wishart: pass exactly one of covariance_matrix, "
                "precision_matrix, or scale_tril."
            )

        # The matrix given is stored under its own name, so validation checks
        # it before anything factorises it.
        given = _as_tensor(
            next(m for m in (covariance_matrix, precision_matrix, scale_tril) if m is not None)
        )
        if scale_tril is not None:
            self.scale_tril = given
        elif covariance_matrix is not None:
            self.covariance_matrix = given
        else:
            self.precision_matrix = given

        self.df = _as_tensor(df)
        dim: int = int(given.shape[-1])
        self._dim = dim

        # ``support`` promises positive-definite draws, and the Bartlett
        # diagonal is χ²(df − i) for i = 0, …, d − 1.  At df ≤ d − 1 the
        # trailing concentrations reach zero and go negative, which is not a
        # Gamma at all: the sampler returns a rank-deficient matrix outside
        # the distribution's own declared support, with nothing to say so.
        with untraced():
            min_df = float(self.df.min().item())
        if min_df <= dim - 1:
            raise ValueError(
                f"Wishart: df must exceed dim - 1 = {dim - 1} for a "
                f"non-singular distribution, got {min_df}."
            )

        batch_shape: tuple[int, ...] = tuple(given.shape[:-2])
        event_shape: tuple[int, ...] = (dim, dim)
        super().__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args
        )
        if scale_tril is None:
            self.scale_tril = _scale_tril_of(covariance_matrix, precision_matrix)
        # Pre-compute covariance for mean/variance.
        self._cov = self.scale_tril @ self.scale_tril.mT

    @_lazy_param
    def covariance_matrix(self) -> Tensor:
        r"""Scale matrix :math:`\Sigma = L L^\top` — as given, or derived
        from ``scale_tril`` on access."""
        return self.scale_tril @ self.scale_tril.mT

    @_lazy_param
    def precision_matrix(self) -> Tensor:
        r""":math:`\Sigma^{-1}` — as given, or derived from ``scale_tril`` on
        access."""
        return _precision_of(self.scale_tril)

    @override
    @property
    def support(self) -> Constraint:
        """Constraint: positive-definite symmetric matrices."""
        return positive_definite

    @override
    @property
    def mean(self) -> Tensor:
        """``df · Σ``."""
        return self.df.unsqueeze(-1).unsqueeze(-1) * self._cov

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Wishart distribution: :math:`(\nu - D - 1)\, \Sigma`.

        ``nan`` where :math:`\nu \leq D + 1`, as the reference framework
        answers.
        """
        factor = self.df - self._dim - 1.0
        factor = lucid.where(factor > 0.0, factor, float("nan"))
        return factor.unsqueeze(-1).unsqueeze(-1) * self._cov

    @override
    @property
    def variance(self) -> Tensor:
        """``Var[W_{ij}] = df · (Σ_{ij}² + Σ_{ii} Σ_{jj})``."""
        S = self._cov
        diag = S.diagonal(dim1=-2, dim2=-1)  # (..., d)
        diag_outer = diag.unsqueeze(-1) * diag.unsqueeze(-2)  # (..., d, d)
        return self.df.unsqueeze(-1).unsqueeze(-1) * (S * S + diag_outer)

    # -- sampling (Bartlett decomposition) ------------------------------------

    def _bartlett_sample(self, sample_shape: tuple[int, ...], *, reparameterised: bool) -> Tensor:
        d: int = self._dim
        L: Tensor = self.scale_tril  # (..., d, d)
        dt = L.dtype
        dev = L.device

        # ── diagonal: chi-squared samples ──────────────────────────────────
        # chi²(k) = 2 · Gamma(k/2, rate=1)
        # concentrations for each diagonal position: (df−0)/2, (df−1)/2, …
        idx = lucid.arange(0, d, 1, dtype=dt, device=dev)
        chi2_conc = (
            self.df.broadcast_to(self._batch_shape).unsqueeze(-1) - idx
        ) / 2.0  # (*batch_shape, d)
        if reparameterised:
            conc_full = chi2_conc.expand(tuple(sample_shape) + tuple(chi2_conc.shape))
            std_g = _standard_gamma(conc_full)
        else:
            std_g = _sample_standard_gamma(chi2_conc, sample_shape)
        sqrt_chi = (2.0 * std_g).sqrt()  # (*sample_shape, *batch_shape, d)

        # ── off-diagonal: standard normals ──────────────────────────────────
        full_shape = self._extended_shape(sample_shape)
        Z = lucid.randn(*full_shape, dtype=dt, device=dev)
        lower_mask = lucid.tril(lucid.ones(d, d, dtype=dt, device=dev), diagonal=-1)
        Z_lower = Z * lower_mask  # strictly lower triangular

        # ── assemble A ──────────────────────────────────────────────────────
        A = Z_lower + lucid.diag_embed(sqrt_chi)  # (*sample_shape, *batch_shape, d, d)

        # ── W = L A Aᵀ Lᵀ ──────────────────────────────────────────────────
        return L @ A @ A.mT @ L.mT

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw a reparameterised sample via the Bartlett decomposition."""
        return self._bartlett_sample(sample_shape, reparameterised=True)

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw a sample via the Bartlett decomposition.

        Each sample is ``L A Aᵀ Lᵀ`` where ``L = scale_tril`` and ``A``
        is a random lower-triangular matrix with:

        * diagonal ``A_{ii} ~ sqrt(χ²_{df − i})``  for ``i = 0, …, d−1``,
        * off-diagonal ``A_{ij} ~ N(0, 1)``  for ``i > j``.
        """
        with lucid.no_grad():
            return self._bartlett_sample(sample_shape, reparameterised=False).detach()

    # -- log probability ──────────────────────────────────────────────────────

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        """Log-density of the Wishart distribution.

        .. code::

            log p(X) = (df − d − 1)/2 · log|X|
                     − tr(Σ⁻¹ X) / 2
                     − df · d / 2 · log 2
                     − df/2 · log|Σ|
                     − log Γ_d(df/2)

        where ``Γ_d`` is the multivariate gamma function.
        """
        d: int = self._dim
        df: Tensor = self.df
        L: Tensor = self.scale_tril

        # log |X| via Cholesky (numerically stable)
        log_det_X = lucid.logdet(value)  # scalar or (...)
        t1 = (df - d - 1.0) / 2.0 * log_det_X

        # tr(Σ⁻¹ X) using Cholesky solve: Σ⁻¹ X = L⁻ᵀ (L⁻¹ X)
        # tr(AB) = (A ∘ Bᵀ).sum()  — but we compute via solve_triangular.
        # cholesky_solve(B, L) returns Σ⁻¹ B.
        # We need tr(Σ⁻¹ X) = sum of diagonal of Σ⁻¹ X.
        # Implemented as: Σ⁻¹ X = inv(Σ) @ X then trace.
        sigma_inv = _la.inv(self._cov)
        t2 = -0.5 * (sigma_inv @ value).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # log |Σ| = 2 · sum(log(diag(L)))
        log_det_sigma = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        t3 = -df / 2.0 * log_det_sigma

        # constant terms
        t4 = -df * d / 2.0 * math.log(2.0)
        t5 = -_sp.multigammaln(df / 2.0, d)

        return t1 + t2 + t3 + t4 + t5

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Wishart` instance with parameters expanded to ``batch_shape``."""
        new = self._get_checked_instance(Wishart, _instance)
        batch_shape = tuple(batch_shape)
        cov_shape = batch_shape + self.event_shape
        new.df = self.df.expand(batch_shape)
        new._dim = self._dim
        new.scale_tril = self.scale_tril.expand(cov_shape)
        new._cov = self._cov.expand(cov_shape)
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(Wishart, new).__init__(
            batch_shape=batch_shape, event_shape=self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


# ── LKJCholesky ───────────────────────────────────────────────────────────────


class LKJCholesky(Distribution):
    r"""LKJ distribution over Cholesky factors of correlation matrices.

    Generative prior for the lower-triangular Cholesky factor :math:`L`
    of a correlation matrix :math:`R = L L^\top` of size :math:`D \times D`.
    Density is proportional to :math:`\det(R)^{\eta - 1}` so the
    :math:`\eta` parameter ("concentration") controls how concentrated
    the prior is around the identity (uncorrelated).  Widely used as a
    weakly-informative prior on correlation structure in Bayesian
    hierarchical models — replacing an inverse-Wishart prior (which
    couples scale and correlation) with a clean separation.

    Parameters
    ----------
    dim : int
        Size :math:`D` of the correlation matrix (must be :math:`\geq 2`).
    concentration : Tensor or float, optional
        Shape parameter :math:`\eta > 0`.  Default ``1.0`` (uniform over
        the space of correlation matrices).  :math:`\eta > 1` concentrates
        mass near the identity (low correlation), :math:`\eta < 1`
        favours high-correlation factors.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction.

    Notes
    -----
    Probability density on the manifold of valid Cholesky factors
    :math:`L`:

    .. math::

        p(L; \eta) \propto |\det R(L)|^{\eta - 1}
        = \prod_{i=1}^{D-1} L_{ii}^{D - i + 2(\eta - 1)}

    where the second equality follows from the Jacobian of the
    transformation between :math:`R` and its Cholesky factor.

    **Sampling** uses the vectorised **Onion method** (Lewandowski,
    Kurowicka & Joe, 2009, §3) — row :math:`i` is built incrementally so
    that its squared norm equals one, with off-diagonal magnitude drawn
    from a Beta and angular components from a uniform on the sphere.
    The full sampler runs in :math:`\mathcal{O}(D^2)` per draw with no
    rejections.

    For :math:`\eta = 1` the distribution is uniform over valid
    correlation matrices (specifically, over the set of valid Cholesky
    factors).

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import LKJCholesky
    >>> dist = LKJCholesky(dim=4, concentration=2.0)
    >>> L = dist.sample()           # (4, 4) lower-triangular Cholesky factor
    >>> R = L @ L.T                 # implied correlation matrix
    >>> R.diagonal()                # diagonal of R is all 1's
    tensor([1., 1., 1., 1.])

    Use as a Bayesian prior over correlation structure:

    >>> prior = LKJCholesky(dim=8, concentration=1.5)
    >>> # ... pair with marginal scales / standard deviations to form a
    >>> #     covariance prior: Σ = diag(σ) · R · diag(σ)
    """

    arg_constraints = {"concentration": positive}
    has_rsample: bool = False

    def __init__(
        self, dim: int, concentration: Tensor | float = 1.0, validate_args: bool | None = None
    ) -> None:
        r"""Initialise an LKJ distribution over Cholesky factors.

        Pre-computes the Beta distribution parameters used by the vectorised
        Onion sampler (Lewandowski et al. 2009, §3).

        Parameters
        ----------
        dim : int
            Dimension :math:`D \geq 2` of the correlation matrix.
        concentration : Tensor | float, optional
            Shape parameter :math:`\eta > 0`.  Default is ``1.0`` (uniform
            distribution over correlation matrices).  Larger values concentrate
            mass near the identity (near-zero correlations).
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        if dim < 2:
            raise ValueError(f"LKJCholesky: dim must be ≥ 2, got {dim}.")
        self.dim = dim
        self.concentration = _as_tensor(concentration)

        # Pre-compute Beta parameters for the Onion sampler (vectorised).
        # marginal_conc = η + (d − 2) / 2
        d: int = dim
        eta: Tensor = self.concentration
        marginal_conc = eta + 0.5 * (d - 2)

        # Row ``i`` of the factor has ``i`` entries below the diagonal, so its
        # squared norm is ``y_i ~ Beta(i/2, η + (d − 1 − i)/2)``.  Row 0 has
        # none — its draw is discarded — but ``Beta`` needs a positive pair, so
        # it borrows row 1's, which is what the leading zero in ``offset`` is.
        offset = lucid.arange(0, d, 1, dtype=eta.dtype, device=eta.device)  # [0, 1, ..., d-1]
        offset = lucid.clip(offset - 1.0, min=0.0)  # [0, 0, 1, ..., d-2]
        beta_conc1 = 0.5 * offset + 0.5  # (d,)
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset  # (*batch, d)

        self._beta = Beta(beta_conc1, beta_conc0)

        batch_shape: tuple[int, ...] = tuple(self.concentration.shape)
        event_shape: tuple[int, ...] = (dim, dim)
        super().__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args
        )

    @override
    @property
    def support(self) -> Constraint:
        """Constraint: Cholesky factors of correlation matrices (``corr_cholesky``)."""
        return corr_cholesky

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw a sample via the vectorised Onion method (Lewandowski 2009 §3).

        Returns the lower Cholesky factor ``L`` of a random correlation matrix.
        """
        d: int = self.dim
        dtype = self.concentration.dtype
        dev = self.concentration.device

        y = self._beta.sample(sample_shape).unsqueeze(-1)  # (*s, *b, d, 1)

        # u_normal: (*s, *b, d, d) strictly lower triangular standard normal
        full_shape = self._extended_shape(sample_shape)  # (*s, *b, d, d)
        u_normal = lucid.randn(*full_shape, dtype=dtype, device=dev)
        # Zero strictly upper triangle (keep only lower triangle with k=-1)
        lower_mask = lucid.tril(lucid.ones(d, d, dtype=dtype, device=dev), diagonal=-1)
        u_normal = u_normal * lower_mask

        # Normalise rows to lie on the unit hypersphere (ignoring row 0 which
        # is all zeros in the lower triangle).
        row_norms = (u_normal * u_normal).sum(dim=-1, keepdim=True).sqrt()
        # Avoid division by zero on rows that are entirely zero.
        # Avoid division by zero: replace zero norms with 1.0 (those rows
        # are all zeros anyway, so the division result is still zero).
        row_norms = lucid.where(row_norms < 1e-10, lucid.ones_like(row_norms), row_norms)
        u_hypersphere = u_normal / row_norms

        # Scale — w := sqrt(y) * u_hypersphere
        w = y.sqrt() * u_hypersphere

        # Diagonal: d_i = sqrt(1 − ‖w_i‖²), clamped for numerical stability.
        w_sq_sum = (w * w).sum(dim=-1)  # (*s, *b, d)
        eps = 1e-6
        diag_elems = lucid.where(
            1.0 - w_sq_sum < eps, lucid.full_like(w_sq_sum, eps), 1.0 - w_sq_sum
        ).sqrt()

        # Assemble L = strictly-lower part + diagonal
        L = w + lucid.diag_embed(diag_elems)
        return L.detach()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        """Log-density of the LKJ distribution on a Cholesky factor.

        The density is proportional to
        ``∏_{i=2}^{d} L_{ii}^{order_i}``
        where ``order_i = 2(η − 1) + (d − i)``.

        The normaliser follows Eq. (15) of Lewandowski et al. (2009).
        """
        d: int = self.dim
        eta: Tensor = _align_device(self.concentration, value)

        # Extract diagonal elements (exclude L_{11} = 1).
        diag = value.diagonal(dim1=-2, dim2=-1)[..., 1:]  # (..., d-1)

        # order_i for i = 2, ..., d (1-indexed)
        order = lucid.arange(2, d + 1, 1, dtype=eta.dtype, device=eta.device)
        order_coeff = 2.0 * (eta - 1.0).unsqueeze(-1) + d - order  # (*batch, d-1)
        unnorm_lp = (order_coeff * diag.log()).sum(dim=-1)

        return unnorm_lp - self._log_normalizer()

    def _log_normalizer(self) -> Tensor:
        """Log of the LKJ normalisation constant.

        From the Stan functions reference (based on Lewandowski 2009, page 1999):

        .. code::

            log Z(η, d) = (d-1)*(d-2)/4 * log(π)
                        + Σ_{j=1}^{d-1} [lmgamma(η + (d-j-1)/2, 1)
                                         − lgamma(η + (d-j-1)/2)]
        """
        d: int = self.dim
        dm1 = d - 1
        eta: Tensor = self.concentration

        # alpha = η + (d - 1) / 2 (scalar part of the beta parameterisation)
        alpha = eta + 0.5 * dm1

        # numerator — mvlgamma(alpha - 0.5, dm1)
        # denominator — lgamma(alpha) * dm1
        numerator = _sp.multigammaln(alpha - 0.5, dm1)
        denominator = lucid.lgamma(alpha) * dm1

        # pi-constant adjustment
        pi_constant = 0.5 * dm1 * math.log(math.pi)

        return pi_constant + numerator - denominator

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`LKJCholesky` instance with parameters expanded to ``batch_shape``."""
        new = self._get_checked_instance(LKJCholesky, _instance)
        batch_shape = tuple(batch_shape)
        new.dim = self.dim
        new.concentration = self.concentration.expand(batch_shape)
        new._beta = self._beta.expand(batch_shape + (self.dim,))
        super(LKJCholesky, new).__init__(
            batch_shape=batch_shape, event_shape=self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new
