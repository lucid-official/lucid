"""MeanFlow — one-step generation by modelling the average velocity.

Geng, Deng, Bai, Kolter and He, arXiv:2505.13447, 2025.  The fifth step
of the flow lineage and the one that removes the integral: where Flow
Matching learns the instantaneous velocity and integrates it at sampling
time, MeanFlow learns the average velocity over an interval, which *is*
that integral, and reads a sample off a single evaluation.
"""

from lucid.models.generative.mean_flow._config import MeanFlowConfig, TimeConditioning, TimeSampler
from lucid.models.generative.mean_flow._model import (
    MeanFlowForImageGeneration,
    MeanFlowModel,
    MeanFlowOutput,
)
from lucid.models.generative.mean_flow._pretrained import (
    mean_flow_base_2,
    mean_flow_base_2_gen,
    mean_flow_base_4,
    mean_flow_base_4_gen,
    mean_flow_large_2,
    mean_flow_large_2_gen,
    mean_flow_medium_2,
    mean_flow_medium_2_gen,
    mean_flow_xlarge_2,
    mean_flow_xlarge_2_gen,
)

__all__ = [
    "MeanFlowConfig",
    "TimeConditioning",
    "TimeSampler",
    "MeanFlowModel",
    "MeanFlowForImageGeneration",
    "MeanFlowOutput",
    "mean_flow_base_4",
    "mean_flow_base_2",
    "mean_flow_medium_2",
    "mean_flow_large_2",
    "mean_flow_xlarge_2",
    "mean_flow_base_4_gen",
    "mean_flow_base_2_gen",
    "mean_flow_medium_2_gen",
    "mean_flow_large_2_gen",
    "mean_flow_xlarge_2_gen",
]
