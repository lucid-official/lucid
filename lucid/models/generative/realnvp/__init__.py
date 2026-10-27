"""RealNVP family — Dinh, Sohl-Dickstein & Bengio, 2016 (affine-coupling flow)."""

from lucid.models.generative.realnvp._config import RealNVPConfig
from lucid.models.generative.realnvp._model import RealNVPForImageGeneration, RealNVPModel
from lucid.models.generative.realnvp._pretrained import (
    realnvp_celeba,
    realnvp_celeba_gen,
    realnvp_cifar,
    realnvp_cifar_gen,
    realnvp_imagenet32,
    realnvp_imagenet32_gen,
    realnvp_imagenet64,
    realnvp_imagenet64_gen,
    realnvp_lsun,
    realnvp_lsun_gen,
)

__all__ = [
    "RealNVPConfig",
    "RealNVPModel",
    "RealNVPForImageGeneration",
    "realnvp_cifar",
    "realnvp_imagenet32",
    "realnvp_imagenet64",
    "realnvp_lsun",
    "realnvp_celeba",
    "realnvp_cifar_gen",
    "realnvp_imagenet32_gen",
    "realnvp_imagenet64_gen",
    "realnvp_lsun_gen",
    "realnvp_celeba_gen",
]
