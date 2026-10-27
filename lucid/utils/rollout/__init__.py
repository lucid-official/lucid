"""
lucid.utils.rollout: environment contract, episode replay, and the loop
that connects a policy to an environment.

Lucid ships no environments — every compute path here is closed to
outside packages. What it ships is the shape one has to have, storage
that hands a recurrent model contiguous sequences rather than loose
transitions, and the two collection hyperparameters the world-model
papers state: action repeat and exploration noise.
"""

from lucid.utils.rollout._driver import LatentPolicy, Policy, RandomPolicy, rollout
from lucid.utils.rollout._env import Environment, Episode, StepResult
from lucid.utils.rollout._replay import (
    PrioritizedBatch,
    PrioritizedSequenceReplay,
    SequenceReplay,
    SumTree,
)

__all__ = [
    "Environment",
    "Episode",
    "StepResult",
    "SequenceReplay",
    "SumTree",
    "PrioritizedBatch",
    "PrioritizedSequenceReplay",
    "Policy",
    "RandomPolicy",
    "LatentPolicy",
    "rollout",
]
