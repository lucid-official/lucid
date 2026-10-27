"""DIAMOND — the world model that predicts frames, not latents.

Alonso et al., NeurIPS 2024 (arXiv:2405.12399).  A conditional diffusion
model over pixels, an EDM parameterisation chosen so three denoising
steps suffice, and an agent trained entirely on what it imagines.
"""

from lucid.models.generative.diamond._config import DIAMONDConfig
from lucid.models.generative.diamond._model import (
    DIAMONDBehaviorOutput,
    DIAMONDForWorldModeling,
    DIAMONDModel,
    DIAMONDOutput,
)
from lucid.models.generative.diamond._pretrained import diamond, diamond_csgo, diamond_world_model
from lucid.models.generative.diamond._weights import DIAMONDWeights

__all__ = [
    "DIAMONDConfig",
    "DIAMONDWeights",
    "DIAMONDModel",
    "DIAMONDForWorldModeling",
    "DIAMONDOutput",
    "DIAMONDBehaviorOutput",
    "diamond",
    "diamond_csgo",
    "diamond_world_model",
]
