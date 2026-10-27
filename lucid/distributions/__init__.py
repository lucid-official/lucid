"""``lucid.distributions`` — probability distributions with the
reference-framework's interface (sample / rsample / log_prob / entropy / …).

Mirrors the reference framework's distributions surface for the most-used univariate and
discrete families plus ``MultivariateNormal``, ``Beta``, ``Gamma``, and
``Dirichlet`` from the gamma family.  See [[api-python-distributions]]
for the full surface.
"""

from lucid.distributions import constraints, transforms
from lucid.distributions.bernoulli import Bernoulli, Geometric
from lucid.distributions.categorical import (
    Categorical,
    OneHotCategorical,
    OneHotCategoricalStraightThrough,
)
from lucid.distributions.constraint_registry import (
    ConstraintRegistry,
    PositiveDefiniteTransform,
    SoftplusTransform,
    biject_to,
    transform_to,
)
from lucid.distributions.continuous_extra import (
    FisherSnedecor,
    HalfCauchy,
    HalfNormal,
    Pareto,
    Weibull,
)
from lucid.distributions.discrete import Binomial, NegativeBinomial, Poisson
from lucid.distributions.distribution import (
    Distribution,
    ExponentialFamily,
    set_default_validate_args,
)
from lucid.distributions.exponential import Cauchy, Exponential, Laplace
from lucid.distributions.extra import (
    ContinuousBernoulli,
    Gumbel,
    InverseGamma,
    Kumaraswamy,
    Multinomial,
)
from lucid.distributions.gamma import Beta, Chi2, Dirichlet, Gamma
from lucid.distributions.independent import Independent
from lucid.distributions.kl import kl_divergence, register_kl
from lucid.distributions.matrix import LKJCholesky, Wishart
from lucid.distributions.mixture import MixtureSameFamily
from lucid.distributions.multivariate import LowRankMultivariateNormal, MultivariateNormal
from lucid.distributions.normal import LogNormal, Normal
from lucid.distributions.relaxed import RelaxedBernoulli, RelaxedOneHotCategorical
from lucid.distributions.student import StudentT
from lucid.distributions.transformed_distribution import TransformedDistribution
from lucid.distributions.transforms import (
    AbsTransform,
    AffineTransform,
    CatTransform,
    ComposeTransform,
    CorrCholeskyTransform,
    CumulativeDistributionTransform,
    ExpTransform,
    IndependentTransform,
    LowerCholeskyTransform,
    PowerTransform,
    ReshapeTransform,
    SigmoidTransform,
    SoftmaxTransform,
    StackTransform,
    StickBreakingTransform,
    TanhTransform,
    Transform,
)
from lucid.distributions.uniform import Uniform
from lucid.distributions.von_mises import GeneralizedPareto, LogisticNormal, VonMises

__all__ = [
    # base
    "Distribution",
    "ExponentialFamily",
    "set_default_validate_args",
    # submodules
    "constraints",
    "transforms",
    # univariate continuous
    "Normal",
    "LogNormal",
    "Uniform",
    "Exponential",
    "Laplace",
    "Cauchy",
    "StudentT",
    "Pareto",
    "GeneralizedPareto",
    "Weibull",
    "HalfNormal",
    "HalfCauchy",
    "FisherSnedecor",
    "VonMises",
    # gamma family
    "Gamma",
    "Chi2",
    "Beta",
    "Dirichlet",
    # discrete
    "Bernoulli",
    "Geometric",
    "Categorical",
    "OneHotCategorical",
    "OneHotCategoricalStraightThrough",
    "Poisson",
    "Binomial",
    "NegativeBinomial",
    # extra univariate continuous
    "Gumbel",
    "InverseGamma",
    "Kumaraswamy",
    "LogisticNormal",
    # -- discrete (extra) --
    "Multinomial",
    # -- continuous [0,1] --
    "ContinuousBernoulli",
    # relaxed (Concrete) — differentiable approximations
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
    # multivariate
    "MultivariateNormal",
    "LowRankMultivariateNormal",
    # matrix-valued
    "Wishart",
    "LKJCholesky",
    # wrappers
    "Independent",
    "TransformedDistribution",
    "MixtureSameFamily",
    # transforms & constraint registry
    "Transform",
    "AbsTransform",
    "AffineTransform",
    "CatTransform",
    "ComposeTransform",
    "ConstraintRegistry",
    "CorrCholeskyTransform",
    "CumulativeDistributionTransform",
    "ExpTransform",
    "IndependentTransform",
    "LowerCholeskyTransform",
    "PositiveDefiniteTransform",
    "PowerTransform",
    "ReshapeTransform",
    "SigmoidTransform",
    "SoftmaxTransform",
    "SoftplusTransform",
    "StackTransform",
    "StickBreakingTransform",
    "TanhTransform",
    "biject_to",
    "transform_to",
    # kl
    "kl_divergence",
    "register_kl",
]
