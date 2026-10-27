"""Image-generative model families — Phase 5 of the model zoo.

Concrete families live in sub-packages: latent-variable (``vae``),
diffusion / score-based (``ddpm``, ``ncsn``), and exact-likelihood flows
(``nice``).  The infrastructure exported here — base configs (one tier per
model class: generative → diffusion / normalizing-flow), output
dataclasses, noise schedulers — is what a new family builds on, so each
one only needs its own ``_config.py``, ``_model.py``, and
``_pretrained.py``.

Positional / timestep encoding primitives (``SinusoidalEmbedding``,
``TimestepEmbedding``) live in :mod:`lucid.nn`; family code imports them
from there rather than redefining locally.
"""

from lucid.models.generative._common._config import (
    BetaSchedule,
    DiffusionModelConfig,
    FlowPrior,
    GenerativeActivation,
    GenerativeModelConfig,
    NormalizingFlowConfig,
)
from lucid.models.generative._common._rssm import RSSM, RSSMState
from lucid.models.generative._common._schedulers import DDPMScheduler, DiffusionScheduler

__all__ = [
    "BetaSchedule",
    "DDPMScheduler",
    "RSSM",
    "RSSMState",
    "DiffusionModelConfig",
    "FlowPrior",
    "GenerativeActivation",
    "GenerativeModelConfig",
    "NormalizingFlowConfig",
    "DiffusionScheduler",
]
