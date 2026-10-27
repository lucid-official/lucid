"""Terminal Velocity Matching — one- and few-step generation from scratch.

Zhou, Parger, Haque and Song, ICLR 2026 (arXiv:2511.19797).  MeanFlow's
dual: where MeanFlow differentiates the learned displacement at the
interval's start, TVM differentiates it at the terminal time and matches
it to the velocity where the jump lands — an objective that bounds the
2-Wasserstein distance once the network is made Lipschitz, which is what
its RMS-normalised DiT is for.
"""

from lucid.models.generative.tvm._config import TimeEmbeddingInit, TimeSampler, TVMConfig
from lucid.models.generative.tvm._model import TVMForImageGeneration, TVMModel, TVMOutput
from lucid.models.generative.tvm._pretrained import (
    tvm_imagenet256,
    tvm_imagenet256_gen,
    tvm_imagenet512,
    tvm_imagenet512_gen,
)

__all__ = [
    "TVMConfig",
    "TimeSampler",
    "TimeEmbeddingInit",
    "TVMModel",
    "TVMForImageGeneration",
    "TVMOutput",
    "tvm_imagenet256",
    "tvm_imagenet512",
    "tvm_imagenet256_gen",
    "tvm_imagenet512_gen",
]
