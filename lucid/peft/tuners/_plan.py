"""Planning an adapter before the model is touched: what it targets, what it stores, what it saves.

Adding an adapter is planned in full first — every target found and typed,
every ``modules_to_save`` entry found, the config to keep spelled out — so
a config that cannot be applied raises with the model as it was.  Nothing
here changes a model except ``_put_back``, which undoes the one write an
initialisation may make before the tree is touched.

A method adapts a module (its layer found by the module's type) or one raw
``Parameter`` of a module (its layer found by the parameter's name): a
``_Target`` that carries a ``parameter`` is the second kind.  The layers of
several parameters of one module nest, and the outermost stands in the tree.
"""

import copy
import warnings
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Final, NamedTuple, Protocol, runtime_checkable

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import PromptLearningConfig, _TargetedConfig
from lucid.peft._errors import NoMatchingPeftModuleError
from lucid.peft._slots import _outside_slots
from lucid.peft._target_aliases import _zoo_target_names
from lucid.peft._target_defaults import (
    _default_skipped_subtrees,
    _default_target_modules,
    _default_target_parameters,
)
from lucid.peft._targeting import (
    _module_names_matching,
    _parameter_paths,
    _resolve_all_linear,
    _selects,
    _shared_outside,
    _with_subtrees_excluded,
)
from lucid.peft.tuners._layer import _BaseTunerLayer, _dotted


#: For a method that adapts modules only: no model type has raw parameters as its default.
_NO_PARAMETERS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({})


@runtime_checkable
class _NamesParameters(Protocol):
    """A config that can name raw parameters (``target_parameters``) beside modules."""

    target_parameters: list[str] | None


def _named[T: nn.Module](model: nn.Module, kind: type[T]) -> list[tuple[str, T]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, kind)]


def _set_submodule(model: nn.Module, key: str, module: nn.Module) -> None:
    parent, _, child = key.rpartition(".")
    setattr(model.get_submodule(parent), child, module)


class _Target(NamedTuple):
    key: str
    module: nn.Module
    #: The layer class that wraps ``module``, or ``None`` when ``module`` is
    #: already such a layer and takes the new adapter beside its others.
    layer: type[_BaseTunerLayer] | None
    #: The name of the one raw parameter of ``module`` this target adapts;
    #: ``None`` when it is the whole module.
    parameter: str | None = None

    @property
    def full_name(self) -> str:
        """The module's name, with the parameter's after it when the target is one."""
        return _dotted(self.key, self.parameter or "")


class _Placement(NamedTuple):
    """A slot new to the tree: where it goes, and the module it replaces."""

    key: str
    slot: nn.Module
    original: nn.Module


class _Injected(NamedTuple):
    """What one injection put in the tree, and what it overwrote, so it can be taken back."""

    placements: list[_Placement]
    #: The base parameters an initialisation rewrote in place, each with its value before.
    overwritten: list[tuple[Parameter, Tensor]]


def _values_of(modules: Iterable[nn.Module]) -> list[tuple[Parameter, Tensor]]:
    """Each distinct parameter of ``modules`` with a copy of its value."""
    held: dict[int, Parameter] = {}
    for module in modules:
        held.update((id(param), param) for param in module.parameters())
    with lucid.no_grad():
        return [(param, param.clone()) for param in held.values()]


def _put_back(values: Iterable[tuple[Parameter, Tensor]]) -> None:
    with lucid.no_grad():
        for param, value in values:
            param.copy_(value)


def _not_found(config: _TargetedConfig) -> NoMatchingPeftModuleError:
    targets = config.target_modules
    shown = sorted(targets) if isinstance(targets, set) else targets
    message = (
        f"Target modules {shown} not found in the base model. "
        "Please check the target modules and try again."
    )
    if config.layers_to_transform is not None:
        message += f" Note: You specified 'layers_to_transform': {config.layers_to_transform}."
    if config.layers_pattern is not None:
        message += f" You also specified 'layers_pattern': {config.layers_pattern}."
    return NoMatchingPeftModuleError(message)


def _plan_modules_to_save(
    model: nn.Module, config: _TargetedConfig | PromptLearningConfig
) -> list[str]:
    names = config.modules_to_save
    if not names:
        return []
    keys = _module_names_matching(model, names)
    if not keys:
        raise NoMatchingPeftModuleError(f"modules_to_save {names} matched no module of the model")
    paths = _parameter_paths(model)
    for key in keys:
        tied = _shared_outside(paths, key, model.get_submodule(key))
        if tied:
            # The reference allows it and warns of the same trouble at merge.
            warnings.warn(
                f"modules_to_save {key!r} shares its weight with {', '.join(tied)}; the copy "
                "that trains will not.  The adapter saves and loads as usual, but "
                "merge_and_unload() will refuse: the merged model would hold two tensors "
                "where the base ties one.",
                stacklevel=5,
            )
    return keys


def _stored_config[C: _TargetedConfig](
    config: C,
    model: nn.Module,
    *,
    default_targets: Mapping[str, frozenset[str]],
    default_parameters: Mapping[str, frozenset[str]],
    adapts_parameters: Callable[[C], bool],
    adapted_linear: Callable[[nn.Module], nn.Linear | None],
) -> C:
    """The config to store: a copy, its default targets and ``"all-linear"`` spelled out.

    A config that names no ``target_modules`` takes the model type's defaults
    (``default_targets``, and ``default_parameters`` for a type whose weights
    are not layers') unless the method adapts raw parameters
    (``adapts_parameters``); ``adapted_linear`` tells which modules
    ``"all-linear"`` may take as already adapted.
    """
    stored = copy.deepcopy(config)
    if stored.target_modules is None and not adapts_parameters(stored):
        parameters = _default_target_parameters(model, default_parameters)
        stored.target_modules = set(_default_target_modules(model, default_targets, parameters))
        skipped = _default_skipped_subtrees(model)
        if skipped:
            stored.exclude_modules = _with_subtrees_excluded(stored.exclude_modules, skipped)
        if parameters and isinstance(stored, _NamesParameters):
            stored.target_parameters = sorted(parameters)
    elif stored.target_modules is not None and not isinstance(stored.target_modules, str):
        stored.target_modules = _zoo_target_names(model, stored.target_modules)
    resolved = _resolve_all_linear(stored, model, adapted_linear)
    if resolved is not None:
        stored.target_modules = resolved
    return stored


def _selected_modules(model: nn.Module, config: _TargetedConfig) -> list[tuple[str, nn.Module]]:
    """The modules ``config`` selects by name, outside the slots an earlier adapter put in."""
    return [(key, module) for key, module in _outside_slots(model) if key and _selects(config, key)]
