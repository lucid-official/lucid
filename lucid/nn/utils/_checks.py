"""Shared boundary validators for :mod:`lucid.nn.utils` reparametrisation and pruning."""

from collections.abc import Iterable

from lucid._tensor.tensor import Tensor
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


def _as_tensor_list(
    tensors: Tensor | Iterable[Tensor | None], caller: str, *, skip_none: bool = False
) -> list[Tensor]:
    """The tensors named by ``tensors``: one ``Tensor`` is a one-element list.

    The single owner of the "a tensor or an iterable of them" rule of every
    ``nn.utils`` function that takes parameters or gradients.  Listing a lone
    ``Tensor`` would iterate its rows, which are views — clipping or writing
    them reaches nothing the caller holds.

    ``None`` entries are refused unless ``skip_none`` (a gradient list is
    allowed to have holes); any other non-tensor is a ``TypeError`` naming
    ``caller``.
    """
    items: list[Tensor | None] = [tensors] if isinstance(tensors, Tensor) else list(tensors)
    out: list[Tensor] = []
    for t in items:
        if t is None and skip_none:
            continue
        if not isinstance(t, Tensor):
            raise TypeError(f"{caller}: expected a Tensor, got {type(t).__name__}")
        out.append(t)
    return out
