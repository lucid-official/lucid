"""Registry factories for the Neural ODE flow (Chen et al., 2018).

The paper describes one continuous normalizing flow, not a family of
sized variants, so there is one configuration here rather than a table of
them:

    * ``neural_ode``     — the bare flow (``encode`` / ``decode`` /
      ``log_prob`` / ``bits_per_dim``)
    * ``neural_ode_gen`` — the same flow with the bits/dim objective and
      ``generate()``

``hidden_dim`` is the paper's :math:`M = 64`.  The sample shape is the
zoo's usual ``32 x 32`` RGB rather than the paper's own experiments,
which are two-dimensional density matching — the flow is defined on a
flat vector and does not care, but the exact trace it uses does, so at
image width the default estimator switches to Hutchinson.  To run the
paper's setting exactly::

    create_model("neural_ode", sample_size=(1, 2), in_channels=1,
                 out_channels=1)

which is two-dimensional, and therefore keeps the exact trace.
"""

from lucid.models._registry import register_model
from lucid.models.generative.neural_ode._config import NeuralODEConfig
from lucid.models.generative.neural_ode._model import (
    NeuralODEForImageGeneration,
    NeuralODEModel,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# Paper §4: the vector field is a gated sum of M = 64 functions.  Two
# blocks is the shallowest stack that lets the gates act on a hidden
# representation rather than directly on the data.
_CFG_DEFAULT = NeuralODEConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    hidden_dim=64,
    num_blocks=2,
)


# ── Bare flow ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="neural_ode",
    model_type="neural_ode",
    model_class=NeuralODEModel,
    default_config=_CFG_DEFAULT,
)
def neural_ode(pretrained: bool = False, **overrides: object) -> NeuralODEModel:
    r"""Construct the Neural ODE continuous normalizing flow.

    A single vector field integrated from ``t = 0`` to ``t = 1``, with the
    log-density carried alongside it by the instantaneous change of
    variables.  Depth is not a hyper-parameter: :mod:`lucid.diffeq` spends
    as many evaluations as the requested tolerance needs, and ``nfe``
    reports how many that was.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``neural_ode``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NeuralODEConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    NeuralODEModel
        Bare flow configured with the paper's :math:`M = 64` field and any
        overrides.

    Notes
    -----
    Reference: Chen, Rubanova, Bettencourt, and Duvenaud, *"Neural
    Ordinary Differential Equations"*, NeurIPS, 2018 (arXiv:1806.07366),
    §4.  The paper's own continuous-flow experiments are two-dimensional
    density matching, where the trace of the Jacobian is summed exactly;
    at the image shape this factory defaults to, the same flow switches to
    the Hutchinson estimator of Grathwohl et al. (*FFJORD*, ICLR 2019),
    which is what makes the cost independent of the dimension.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.neural_ode import neural_ode
    >>> model = neural_ode(sample_size=(1, 2), in_channels=1,
    ...                    out_channels=1).eval()
    >>> z, log_det = model.encode(lucid.rand((2, 1, 1, 2)))
    >>> z.shape, log_det.shape
    ((2, 2), (2,))
    >>> model.trace_method            # two-dimensional, so exact
    'exact'
    """
    if pretrained:
        reject_unavailable_pretrained("neural_ode")
    return NeuralODEModel(apply_overrides(_CFG_DEFAULT, overrides))


# ── Generator ─────────────────────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="neural_ode",
    model_type="neural_ode",
    model_class=NeuralODEForImageGeneration,
    default_config=_CFG_DEFAULT,
)
def neural_ode_gen(pretrained: bool = False, **overrides: object) -> NeuralODEForImageGeneration:
    r"""Neural ODE flow with the bits/dim objective and a sampler.

    Same field as :func:`neural_ode`, wrapped so ``forward`` reports a
    loss and ``generate`` integrates a prior draw back to ``t = 0``.  The
    reverse pass carries no density term, which makes sampling the cheaper
    of the two directions — the opposite of the usual situation for a
    flow, where the inverse is the awkward one to write.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``neural_ode_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NeuralODEConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    NeuralODEForImageGeneration
        Generator configured with the paper's :math:`M = 64` field.

    Notes
    -----
    Reference: Chen, Rubanova, Bettencourt, and Duvenaud, *"Neural
    Ordinary Differential Equations"*, NeurIPS, 2018 (arXiv:1806.07366).
    The paper reports no bits/dim figure for the continuous flow — its
    density experiments are two-dimensional and reported as KL — so there
    is no published number for this configuration to be compared against.

    Examples
    --------
    >>> from lucid.models.generative.neural_ode import neural_ode_gen
    >>> model = neural_ode_gen(sample_size=(1, 2), in_channels=1,
    ...                        out_channels=1).eval()
    >>> model.generate(n_samples=2).samples.shape
    (2, 1, 1, 2)
    """
    if pretrained:
        reject_unavailable_pretrained("neural_ode_gen")
    return NeuralODEForImageGeneration(apply_overrides(_CFG_DEFAULT, overrides))


__all__ = ["neural_ode", "neural_ode_gen"]
