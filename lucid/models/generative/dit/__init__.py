"""DiT — the diffusion model that replaced the U-Net with a transformer.

Peebles and Xie, ICCV 2023 (arXiv:2212.09748).  Latent patches through a
plain Vision Transformer, conditioned by adaLN-Zero, and the finding that
FID tracks the backbone's Gflops rather than its parameter count.
"""

from lucid.models.generative.dit._config import DiTConfig, DiTConditioning
from lucid.models.generative.dit._model import DiTForImageGeneration, DiTModel, DiTOutput
from lucid.models.generative.dit._weights import DiTXLarge2Weights
from lucid.models.generative.dit._pretrained import (
    dit_base_2,
    dit_base_2_gen,
    dit_base_4,
    dit_base_4_gen,
    dit_base_8,
    dit_base_8_gen,
    dit_large_2,
    dit_large_2_gen,
    dit_large_4,
    dit_large_4_gen,
    dit_large_8,
    dit_large_8_gen,
    dit_small_2,
    dit_small_2_gen,
    dit_small_4,
    dit_small_4_gen,
    dit_small_8,
    dit_small_8_gen,
    dit_xlarge_2,
    dit_xlarge_2_gen,
    dit_xlarge_4,
    dit_xlarge_4_gen,
    dit_xlarge_8,
    dit_xlarge_8_gen,
)

__all__ = [
    "DiTConfig",
    "DiTXLarge2Weights",
    "DiTConditioning",
    "DiTModel",
    "DiTForImageGeneration",
    "DiTOutput",
    "dit_small_2",
    "dit_small_4",
    "dit_small_8",
    "dit_base_2",
    "dit_base_4",
    "dit_base_8",
    "dit_large_2",
    "dit_large_4",
    "dit_large_8",
    "dit_xlarge_2",
    "dit_xlarge_4",
    "dit_xlarge_8",
    "dit_small_2_gen",
    "dit_small_4_gen",
    "dit_small_8_gen",
    "dit_base_2_gen",
    "dit_base_4_gen",
    "dit_base_8_gen",
    "dit_large_2_gen",
    "dit_large_4_gen",
    "dit_large_8_gen",
    "dit_xlarge_2_gen",
    "dit_xlarge_4_gen",
    "dit_xlarge_8_gen",
]
