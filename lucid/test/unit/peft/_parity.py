"""Helpers for the tests that compare a tuner with the reference adapter library, tensor by tensor.

The reference library is optional (it is not a Lucid dependency): a test asks for
it through :func:`adapter_library` and is skipped where it is missing, as it is
when the reference framework (the ``ref`` fixture) is.  Weights are copied across
as nested lists, so neither side imports the other's tensors.
"""

from types import ModuleType
from typing import Any

import pytest

import lucid


def adapter_library() -> ModuleType:
    """The reference adapter library, or skip the test."""
    return pytest.importorskip("peft", reason="the reference adapter library is not installed")


def across(ref: ModuleType, tensor: lucid.Tensor) -> Any:
    """``tensor`` as a float32 tensor of the reference framework."""
    return ref.tensor(tensor.tolist(), dtype=ref.float32)


def gap(ref: ModuleType, mine: lucid.Tensor, theirs: Any) -> float:
    """The largest absolute difference between a Lucid tensor and the reference framework's."""
    return float(ref.sub(across(ref, mine), theirs).detach().abs().max())
