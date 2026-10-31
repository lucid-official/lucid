"""A module seen as a function of its parameters: ``functional_call`` and
``stack_module_state``.

``functional_call`` runs a module with other tensors in its parameter and
buffer slots (:class:`~lucid.nn._swap.TensorSwap`), so the parameters become
arguments a transform can differentiate or batch.  ``stack_module_state``
is the inverse bookkeeping: the state of ``N`` models of one class stacked
along a new leading axis — the argument ``vmap(functional_call)`` maps over
to run an ensemble.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn._swap import TensorSwap

if TYPE_CHECKING:
    from lucid.nn.module import Module

__all__ = ["functional_call_at", "stack_state"]


def _merged(dicts: Mapping[str, Tensor] | Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    """The one mapping the dictionaries make; a name in two is ambiguous."""
    parts = [dicts] if isinstance(dicts, Mapping) else list(dicts)
    merged: dict[str, Tensor] = {}
    repeated: list[str] = []
    for part in parts:
        for name, tensor in part.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{tensor!r} is not an instance of Tensor (key '{name}')")
            if name in merged:
                repeated.append(name)
            merged[name] = tensor
    if repeated:
        raise ValueError(
            f"{sorted(set(repeated))} appeared in multiple dictionaries; "
            "behavior of functional call is ambiguous"
        )
    return merged


def functional_call_at(
    module: Module,
    parameter_and_buffer_dicts: Mapping[str, Tensor] | Sequence[Mapping[str, Tensor]],
    args: object,
    kwargs: Mapping[str, object] | None,
    *,
    tie_weights: bool,
    strict: bool,
) -> object:
    """``module(*args, **kwargs)`` with the given tensors in its slots.

    Parameters
    ----------
    module : Module
        The module to call.
    parameter_and_buffer_dicts : dict or sequence of dict
        Dotted name to tensor; merged, and a name in two is an error.
    args : object
        A tuple of positional arguments, a single argument, or ``None``.
    kwargs : mapping or None
        The keyword arguments.
    tie_weights : bool
        Whether a tensor for one name also stands for the names tied to it.
    strict : bool
        Whether the names must be exactly the module's.

    Returns
    -------
    object
        What the module returned; the module is as it was afterwards.
    """
    tensors = _merged(parameter_and_buffer_dicts)
    positional = args if isinstance(args, tuple) else () if args is None else (args,)
    with TensorSwap(module, tensors, tie_weights=tie_weights, strict=strict):
        return module(*positional, **(kwargs or {}))


def _stacked(tensors: Sequence[Tensor], name: str) -> Tensor:
    """``tensors`` — one per model — along a new first axis; it requires
    grad when theirs do, and they must all agree on that."""
    required = [t.requires_grad for t in tensors]
    if any(required) and not all(required):
        raise RuntimeError(f"Expected {name} from each model to have the same .requires_grad")
    stacked = lucid.stack(list(tensors))
    return stacked.detach().requires_grad_() if all(required) else stacked


def _stacked_by_name(per_model: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    """The tensors of the same name in every model, stacked."""
    return {name: _stacked([state[name] for state in per_model], name) for name in per_model[0]}


def stack_state(models: Sequence[Module]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """``(params, buffers)`` of ``models`` stacked along a new first axis.

    Parameters
    ----------
    models : sequence of Module
        At least one model; all of one class and mode.

    Returns
    -------
    tuple of dict
        Parameter name to stacked tensor, and buffer name to stacked tensor.
    """
    if len(models) == 0:
        raise RuntimeError("stack_module_state: Expected at least one model, got 0.")
    if len({m.training for m in models}) > 1:
        raise RuntimeError(
            "stack_module_state: Expected all models to have the same training/eval mode."
        )
    if len({type(m) for m in models}) > 1:
        raise RuntimeError("stack_module_state: Expected all models to be of the same class.")
    return (
        _stacked_by_name([dict(m.named_parameters()) for m in models]),
        _stacked_by_name([dict(m.named_buffers()) for m in models]),
    )
