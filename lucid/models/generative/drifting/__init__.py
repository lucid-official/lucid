"""Drifting Models — one-step generation by evolving the pushforward in training.

Deng, Li, Li, Du and He, arXiv:2602.04770, 2026.  A generator is trained
toward a frozen, drifted copy of its own samples: attracted by the data,
repelled by the other samples, through an anti-symmetric field that
vanishes exactly when the two distributions agree.  Inference is one
network evaluation, with guidance trained in rather than applied.
"""

from lucid.models.generative.drifting._config import DriftingConfig
from lucid.models.generative.drifting._model import (
    DriftingForImageGeneration,
    DriftingModel,
    DriftingOutput,
)
from lucid.models.generative.drifting._pretrained import (
    drifting_base_2,
    drifting_base_2_gen,
    drifting_base_16,
    drifting_base_16_gen,
    drifting_large_2,
    drifting_large_2_gen,
    drifting_large_16,
    drifting_large_16_gen,
)

__all__ = [
    "DriftingConfig",
    "DriftingModel",
    "DriftingForImageGeneration",
    "DriftingOutput",
    "drifting_base_2",
    "drifting_large_2",
    "drifting_base_16",
    "drifting_large_16",
    "drifting_base_2_gen",
    "drifting_large_2_gen",
    "drifting_base_16_gen",
    "drifting_large_16_gen",
]
