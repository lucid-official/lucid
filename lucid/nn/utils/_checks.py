"""Shared boundary validators for :mod:`lucid.nn.utils` reparametrisation and pruning."""

from lucid.nn.module import Module
from lucid.nn.parameter import Parameter


def _require_module(module: object, caller: str) -> Module:
    """Validate that ``module`` is an :class:`~lucid.nn.Module`."""
    if not isinstance(module, Module):
        raise TypeError(f"{caller} requires a Module, got {type(module).__name__}")
    return module


def _require_parameter(module: Module, name: str, action: str) -> Parameter:
    """Return ``module.<name>``, refusing missing or non-``Parameter`` attributes."""
    if not hasattr(module, name):
        raise AttributeError(f"module has no parameter '{name}'")
    weight = getattr(module, name)
    if not isinstance(weight, Parameter):
        raise TypeError(f"'{name}' must be a Parameter to {action}, got {type(weight).__name__}")
    return weight
