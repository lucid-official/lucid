"""What every tuner shares: the layer base, the model base, and their registries.

A method is three classes, each registered as it is defined:

* its config (``PeftConfig`` subclass, ``peft_type=``) — in ``lucid.peft._config``;
* its tuner (``_BaseTuner`` subclass, ``peft_type=``), which puts layers
  into a model and takes them out — the generic half is here, the
  method's half is the hooks it implements;
* its layers (``_BaseTunerLayer`` subclasses, ``base_type=``), one per
  kind of module it can wrap, found by the wrapped module's type.

``get_peft_model``, ``PeftModel.from_pretrained`` and the state-dict
functions dispatch through these tables, so a new method is subclasses
only.  Everything that stands in for a base module — a tuner layer, a
``modules_to_save`` copy — answers the ``_AdapterSlot`` protocol
(``lucid.peft._slots``), and the generic code walks slots rather than
naming their classes.

A model holds any number of adapters.  Each slot keeps one set of
parameters per adapter name, the tuner keeps their configs, and the
adapters named in ``active_adapter`` are the ones the forward pass applies.
Adding an adapter is planned in full before the model is touched — every
target found, typed and given its layer, every ``modules_to_save`` entry
found — so a config that cannot be applied raises with the model as it was.
The layers then go in by ``setattr`` on their parents, which keeps each
child's slot in place (a ``Sequential`` still runs it in the same turn); a
module already holding an adapter of the method takes the new one beside it.

A method adapts a module (its layer found by the module's type) or one raw
``Parameter`` of a module (its layer found by the parameter's name): a
``_Target`` that carries a ``parameter`` is the second kind.  The layers of
several parameters of one module nest, and the outermost stands in the tree.
"""

import copy
import functools
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from types import MappingProxyType
from typing import ClassVar, Final, NamedTuple, Protocol, override, runtime_checkable

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._adapter_file import _AdapterFile
from lucid.peft._config import PeftConfig, PeftType, PromptLearningConfig, _TargetedConfig
from lucid.peft._forwarding import _as_module, _ForwardsAttributes
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._slots import (
    _adapter_slots,
    _as_names,
    _outermost_slots,
    _outside_slots,
    _SlotState,
)
from lucid.peft._target_defaults import (
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_LORA_TARGET_PARAMETERS,
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

#: Where a model adapted in place keeps the ``requires_grad`` it had before its first adapter.
_FLAGS_KEY: Final = "_peft_requires_grad_before"

#: For a method that adapts modules only: no model type has raw parameters as its default.
_NO_PARAMETERS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({})


@runtime_checkable
class _NamesParameters(Protocol):
    """A config that can name raw parameters (``target_parameters``) beside modules."""

    target_parameters: list[str] | None


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _named[T: nn.Module](model: nn.Module, kind: type[T]) -> list[tuple[str, T]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, kind)]


def _set_submodule(model: nn.Module, key: str, module: nn.Module) -> None:
    parent, _, child = key.rpartition(".")
    setattr(model.get_submodule(parent), child, module)


@functools.cache
def _module_dict_attributes() -> frozenset[str]:
    return frozenset(dir(nn.ModuleDict()))


def _check_adapter_name(adapter_name: object) -> str:
    """``adapter_name`` if it can be the key of an adapter in a layer's ``ModuleDict``\\ s."""
    if not isinstance(adapter_name, str):
        raise TypeError(f"adapter_name must be a str, got {adapter_name!r}")
    if not adapter_name or "." in adapter_name:
        raise ValueError(f"adapter_name must be a non-empty name without '.', got {adapter_name!r}")
    if adapter_name in _module_dict_attributes():
        raise ValueError(
            f"adapter_name {adapter_name!r} is an attribute of nn.ModuleDict, which holds the "
            "adapters of a layer; choose another name"
        )
    return adapter_name


def _dotted(*parts: str) -> str:
    """The non-empty ``parts`` joined with dots."""
    return ".".join(part for part in parts if part)


def _entry_parameters(entry: nn.Module | Parameter | None) -> list[Parameter]:
    """The parameters of one table entry: a module's own, or the bare parameter."""
    if isinstance(entry, nn.Module):
        return list(entry.parameters())
    return [] if entry is None else [entry]


def _parameter_at(table: nn.ParameterDict, name: str) -> Parameter:
    """``table[name]``, which a ``ParameterDict`` types as optional: it must hold that parameter."""
    entry = table[name]
    if entry is None:
        raise KeyError(f"{name!r} holds no parameter")
    return entry


def _entry_leaves(entry: nn.Module | Parameter | None) -> list[str]:
    """The state-dict suffixes under one table entry: a module's keys, or ``""`` for a bare parameter."""
    return list(entry.state_dict()) if isinstance(entry, nn.Module) else [""]


def _tuner_layers(model: nn.Module) -> list[tuple[str, _BaseTunerLayer]]:
    # Not ``_named``: a ``type[T]`` parameter takes concrete classes only.
    return [(name, m) for name, m in model.named_modules() if isinstance(m, _BaseTunerLayer)]


class _BaseTunerLayer(nn.Module, ABC):
    """The base of a tuner's layers: the wrapped module and the state forward branches on.

    A tuner family's root class — the direct subclass of this one — starts
    a table of layer classes; each concrete layer joins it with
    ``base_type=`` — the kind of module it wraps — and the tuner finds a
    target's layer class there.  A layer names its per-adapter tables in
    ``adapter_layer_names``: ``ModuleDict``\\ s, or ``ParameterDict``\\ s
    where the factors are bare parameters.  It holds any number of
    adapters, one entry per name in each of those tables.

    What the forward pass branches on — ``active_adapter``,
    ``disable_adapters``, ``merged`` — are plain attributes in the
    instance ``__dict__``, not properties or descriptors as in the
    reference: ``lucid.compile`` keys an executable on each module's
    public scalar ``__dict__`` attributes, and state kept anywhere else
    would let a compiled model replay a branch it has left (LCD-703).
    ``active_adapter`` is one name or a list of names; a layer that holds
    none of them runs its base alone.  They change only through
    ``set_adapter``, ``delete_adapter``, ``enable_adapters``, ``merge``
    and ``unmerge``.
    """

    #: The per-adapter table attributes (``ModuleDict`` / ``ParameterDict``).
    adapter_layer_names: ClassVar[tuple[str, ...]] = ()
    #: Wrapped module type → layer class, one table per family.
    _layer_types: ClassVar[dict[type[nn.Module], type[_BaseTunerLayer]]] = {}
    #: The module type this layer wraps, from ``base_type=``.
    _base_type: ClassVar[type[nn.Module]] = nn.Module

    @override
    def __init_subclass__(
        cls, *, base_type: type[nn.Module] | None = None, family: bool = False, **kwargs: object
    ) -> None:
        super().__init_subclass__(**kwargs)
        # A direct subclass starts a table; ``family=True`` starts one for a
        # class that shares a base with other methods (the LyCORIS layers).
        if family or _BaseTunerLayer in cls.__bases__:
            cls._layer_types = {}
        if base_type is not None:
            cls._base_type = base_type
            cls._layer_types[base_type] = cls

    def __init__(self, base_layer: nn.Module) -> None:
        kind = type(self)._base_type
        if not isinstance(base_layer, kind):
            raise TypeError(
                f"{type(self).__name__} adapts nn.{kind.__name__} in lucid.peft, "
                f"got {type(base_layer).__name__}"
            )
        if base_layer.has_uninitialized_parameters():
            raise ValueError(
                f"{type(base_layer).__name__} builds its parameters on its first input; "
                "run it once before adapting it"
            )
        super().__init__()
        self.base_layer = base_layer
        self.merged_adapters: list[str] = []
        #: Name (or names) of the adapter the forward pass applies.
        self.active_adapter: str | list[str] = ""
        #: Whether the forward pass runs the base layer alone.
        self.disable_adapters = False
        #: Whether an adapter is folded into the base weight (``merged_adapters`` non-empty).
        self.merged = False

    @property
    def active_adapters(self) -> list[str]:
        """The adapters the forward pass applies, as a list."""
        return [name for name in _as_names(self.active_adapter) if name]

    def get_base_layer(self) -> nn.Module:
        """Return the wrapped module, whose parameters are the base model's own."""
        return self.base_layer

    @property
    def weight(self) -> Parameter:
        """The base layer's weight (unchanged unless an adapter is merged into it)."""
        weight = self.get_base_layer().weight
        if isinstance(weight, Parameter):
            return weight
        raise AttributeError(f"{type(self.get_base_layer()).__name__} has no weight parameter")

    @property
    def bias(self) -> Parameter | None:
        """The base layer's bias, or ``None``."""
        bias = getattr(self.get_base_layer(), "bias", None)
        return bias if isinstance(bias, Parameter) else None

    def enable_adapters(self, enabled: bool) -> None:
        """Apply the adapter in the forward pass (``True``) or run the base alone (``False``).

        Disabling a merged layer unmerges it first, here rather than on
        the next call, so the forward pass never writes a weight — a
        compiled forward must not replay that write.

        Parameters
        ----------
        enabled : bool
            Whether the adapter takes part.  Gradient flags are left as
            they are: a frozen adapter stays frozen when it is re-enabled.
        """
        if not enabled and self.merged:
            self.unmerge()
        self.disable_adapters = not enabled

    def _activate(self, adapter_names: str | list[str]) -> None:
        self.active_adapter = adapter_names if isinstance(adapter_names, str) else [*adapter_names]

    def _check_activation(self, adapter_names: list[str]) -> None:
        del adapter_names  # any set of adapters may apply together in a layer

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        """Apply the named adapters, and train exactly those unless ``inference_mode``.

        The names this layer holds nothing for are skipped by the forward
        pass.  A merged layer is unmerged first, since the active adapters
        are no longer the merged ones.

        Parameters
        ----------
        adapter_names : str or list of str
            The adapter, or adapters, to apply.
        inference_mode : bool, default=False
            Freeze the activated adapters as well.
        """
        if self.merged:
            warnings.warn(
                "Adapter cannot be set when the model is merged. Unmerging the model first.",
                stacklevel=2,
            )
            self.unmerge()
        wanted = _as_names(adapter_names)
        for held in self._adapter_names():
            self.set_requires_grad(held, held in wanted and not inference_mode)
        self._activate(adapter_names)

    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Set ``requires_grad`` on the parameters of the named adapters this layer holds."""
        wanted = _as_names(adapter_names)
        for table in self._tables():
            for name, module in table.items():
                if name in wanted:
                    module.requires_grad_(requires_grad)

    def delete_adapter(self, adapter_name: str) -> None:
        """Drop ``adapter_name`` from this layer; nothing happens if it holds no such adapter.

        An active adapter that is deleted leaves the active set; the tuner
        chooses what takes its place, the same on every layer.
        """
        for table in self._tables():
            if adapter_name in table:
                del table[adapter_name]
        active = self.active_adapters
        if adapter_name in active:
            self._activate([name for name in active if name != adapter_name])

    def _pending_merges(self, adapter_names: list[str] | None) -> list[str]:
        """The requested adapters (the active ones by default) not merged yet.

        A disabled layer runs its base alone, so merging into it would make
        it run the adapter while it reads as disabled; it is refused.  With
        no names given, the active adapters this layer holds are taken.
        """
        if self.disable_adapters:
            raise RuntimeError(
                "the adapter is disabled; enable it (leave disable_adapter()) before merging"
            )
        held = self._adapter_names()
        if adapter_names is None:
            names = [name for name in self.active_adapters if name in held]
        else:
            names = list(adapter_names)
            unknown = [name for name in names if name not in held]
            if unknown:
                raise ValueError(f"unknown adapter(s) {unknown}; this layer holds {held}")
        return [name for name in names if name not in self.merged_adapters]

    def _adapter_names(self) -> list[str]:
        """The adapter names this layer holds."""
        tables = self._tables()
        return list(tables[0]) if tables else []

    def _state(self) -> _SlotState:
        return _SlotState(not self.disable_adapters, self.active_adapters, [*self.merged_adapters])

    def _tables(self) -> list[nn.ModuleDict[nn.Module] | nn.ParameterDict]:
        """The per-adapter tables ``adapter_layer_names`` names."""
        return [
            table
            for name in self.adapter_layer_names
            if isinstance(table := getattr(self, name), (nn.ModuleDict, nn.ParameterDict))
        ]

    def _adapter_parameters(self, adapter_name: str) -> list[Parameter]:
        return [
            p
            for table in self._tables()
            if adapter_name in table
            for p in _entry_parameters(table[adapter_name])
        ]

    def _disk_table(self, table_name: str) -> str:
        """The name an adapter file gives the table ``table_name``; the reference's layout."""
        return table_name

    def _adapter_keys(self, adapter_name: str) -> dict[str, str]:
        return {
            _dotted(name, adapter_name, leaf): _dotted(self._disk_table(name), leaf)
            for name, table in zip(self.adapter_layer_names, self._tables(), strict=True)
            if adapter_name in table
            for leaf in _entry_leaves(table[adapter_name])
        }

    def _unwrap(self, keep_trained: bool) -> nn.Module:
        if self.merged and not keep_trained:
            self.unmerge()
        return self.base_layer

    # ``update_layer`` — creating an adapter — is each family's own, not an
    # abstract method here: its parameters are the method's hyper-parameters,
    # and one signature for all would be either ``**kwargs`` or wrong.

    @abstractmethod
    def get_delta_weight(self, adapter: str) -> Tensor:
        """The change ``adapter`` makes to the base weight, shaped like it."""

    @abstractmethod
    def _merge_targets(self) -> dict[str, Parameter]:
        """The base parameters ``merge`` writes, by their name under this layer."""

    @abstractmethod
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        """Each merge target as it would read with ``adapter`` folded in now, in the target's dtype."""

    @abstractmethod
    def _fold(self, adapter: str) -> None:
        """Write ``adapter`` into the merge targets, in place."""

    @abstractmethod
    def _unfold(self) -> None:
        """Take every merged adapter back out of the merge targets, in place."""

    def _check_finite_merge(self, adapter: str) -> None:
        """Raise ``ValueError`` if merging ``adapter`` would write NaN or inf."""
        with lucid.no_grad():
            finite = all(
                bool(lucid.isfinite(value).all().item())
                for value in self._merged_values(adapter).values()
            )
        if not finite:
            raise ValueError(
                f"NaNs detected in the merged weights. The adapter {adapter} seems to be broken"
            )

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """Fold the update into the base weights, in place.

        The weights keep their identity — they are written, not replaced —
        so whatever holds them (an optimizer, a tied embedding) sees the
        merged values.  That is also why a weight shared with another
        module must not be merged into; the tuner's ``merge_adapter`` and
        ``merge_and_unload`` refuse it, as this layer alone cannot see the
        rest of the model.

        Parameters
        ----------
        safe_merge : bool, default=False
            Check the merged weights for NaN / inf first and leave the base
            untouched if there are any.
        adapter_names : list of str, optional
            Adapters to merge; the active ones this layer holds by default.

        Raises
        ------
        ValueError
            An adapter name this layer does not hold, or ``safe_merge``
            found a non-finite merged weight.
        RuntimeError
            The adapter is disabled, or the layer's own rules refuse the merge
            (``lora_bias`` on a base with no bias).
        NotImplementedError
            This layer's update cannot be written into its base weights.
        """
        pending = self._pending_merges(adapter_names)
        if not pending:
            warnings.warn(f"Already merged: {self.merged_adapters}. Nothing to do.", stacklevel=2)
        if safe_merge:
            for adapter in pending:
                self._check_finite_merge(adapter)
        with lucid.no_grad():
            for adapter in pending:
                self._fold(adapter)
                self.merged_adapters.append(adapter)
                self.merged = True

    def unmerge(self) -> None:
        """Take every merged update back out of the base weights, in place.

        The result equals the original weights up to the rounding of the
        round trip.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.", stacklevel=2)
            return
        with lucid.no_grad():
            self._unfold()
        self.merged_adapters.clear()
        self.merged = False


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


#: ``PeftType`` → the tuner that applies it, filled as each is defined.
_TUNERS: dict[PeftType, type[_AdapterHost]] = {}


def _tuner_for(config: PeftConfig) -> type[_AdapterHost]:
    """The tuner registered for ``config``'s method."""
    for peft_type, tuner in _TUNERS.items():
        if config.peft_type == peft_type:
            return tuner
    raise NotImplementedError(f"lucid.peft has no tuner for {config.peft_type} adapters")


class _AdapterHost(_ForwardsAttributes, ABC):
    """What ``PeftModel`` asks of the object that holds a model's adapters.

    A method registers its host by ``peft_type`` as it is defined.  Most
    methods put layers into the model (``_BaseTuner``); a method that
    learns an input instead (prompt, prefix and p-tuning) holds its
    adapters beside the model, and answers the same questions.
    """

    #: The model the adapters are for, and its adapters' configs by name.
    model: nn.Module
    peft_config: dict[str, PeftConfig]
    active_adapter: str | list[str]
    #: Where the user's model sits inside this host, in the host's state-dict keys.
    _model_path: ClassVar[str] = "model."
    #: The config class this method takes.
    _config_class: ClassVar[type[PeftConfig]]

    def __init__(self, model: nn.Module, config: PeftConfig, adapter_name: str = "default") -> None:
        """Make the host; a method's host builds its adapter from ``model`` and ``config`` after this."""
        del model, config, adapter_name  # the subclass keeps what it needs
        super().__init__()

    @override
    def __init_subclass__(cls, *, peft_type: PeftType | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if peft_type is not None:
            _TUNERS[peft_type] = cls

    @property
    @abstractmethod
    def active_adapters(self) -> list[str]:
        """The adapters the forward pass applies, as a list."""

    @override
    def _wrapped(self) -> nn.Module | None:
        modules = self.__dict__.get("_modules")
        return _as_module(modules.get("model")) if modules else None

    def _task_scope(self, task_ids: Tensor | None) -> AbstractContextManager[None]:
        """The context a call runs in when it names a task per batch row.

        An adapter whose prompt is not per task has nothing to pick, so it takes none.

        Raises
        ------
        ValueError
            ``task_ids`` is given.
        """
        if task_ids is not None:
            raise ValueError("task_ids given to an adapter whose prompt is not per-task")
        return nullcontext()

    def _admit(self, adapter_name: str, peft_config: PeftConfig) -> str:
        """``adapter_name`` once it and ``peft_config`` are fit to be added beside the adapters held.

        Raises
        ------
        ValueError
            The name is malformed or taken, or the config is another method's.
        TypeError
            The config is not this method's config class.
        """
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        kind = type(self)._config_class
        if peft_config.peft_type != kind._peft_type:
            raise ValueError(
                "Cannot combine adapters with different peft types. "
                f"Found {kind._peft_type} and {peft_config.peft_type}."
            )
        if not isinstance(peft_config, kind):
            raise TypeError(f"config must be a {kind.__name__}, got {type(peft_config).__name__}")
        return adapter_name

    @classmethod
    @abstractmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        """Parameters beyond the adapters' own that train, by name under ``model``."""

    @classmethod
    def _adapt_in_place(cls, model: nn.Module, config: PeftConfig, adapter_name: str) -> None:
        """Adapt a bare ``model`` without a wrapper (``inject_adapter_in_model``)."""
        cls(model, config, adapter_name)

    @classmethod
    def _require_adapter_state(cls, model: nn.Module, adapter_name: str) -> None:
        """Raise ``RuntimeError`` if ``model`` no longer holds what ``adapter_name`` saves."""

    @classmethod
    def _check_loadable(cls, config: PeftConfig, is_trainable: bool) -> None:
        """Raise if a saved adapter of this method cannot be loaded as ``is_trainable``."""

    @classmethod
    def _converted_to_plain(
        cls, trained: _AdapterFile[PeftConfig], initial: _AdapterFile[PeftConfig]
    ) -> _AdapterFile[PeftConfig]:
        """``trained`` as an adapter that fits the pristine base, given the adapter as it was at initialisation.

        Only a method whose initialisation rewrites the base weight has anything to
        convert (LoRA's PiSSA / OLoRA); every other method refuses.

        Raises
        ------
        ValueError
            This method has no such conversion.
        """
        del trained, initial  # the default has no conversion
        raise ValueError(
            "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not "
            f"supported for {cls.__name__}."
        )

    @abstractmethod
    def add_adapter(
        self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False
    ) -> None:
        """Add an adapter beside the ones held."""

    @abstractmethod
    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Apply the named adapter, and train it unless ``inference_mode``."""

    @abstractmethod
    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Enable or disable gradients on the named adapters."""

    @abstractmethod
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove an adapter."""

    @abstractmethod
    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        """Fold the adapters into the base model and return it."""

    @abstractmethod
    def unload(self) -> nn.Module:
        """Remove every adapter without merging and return the original model."""

    @abstractmethod
    def enable_adapter_layers(self) -> None:
        """Apply the adapter again after ``disable_adapter_layers``."""

    @abstractmethod
    def disable_adapter_layers(self) -> None:
        """Run the base model alone."""

    @abstractmethod
    def _mixed_batch(
        self, adapter_names: Sequence[str], num_beams: int = 1
    ) -> AbstractContextManager[None]:
        """A context in which each batch row goes through the adapter named for it."""


class _BaseTuner(_AdapterHost, ABC):
    """Put a method's layers into a model, freeze the rest, take them out again.

    The generic half of a tuner.  Construction checks the first adapter,
    plans its injection, applies it and freezes; ``add_adapter`` does the
    same for each further one, which starts frozen and inactive until
    ``set_adapter`` names it.  A method supplies its layer family
    (``_layer_types``), how to build a layer for a target
    (``_create_new_module``) or add an adapter to one it already built
    (``_add_to_layer``), and may find raw parameters to adapt
    (``_plan_parameter_targets``), adjust the stored config
    (``_prepare_config``), check a new adapter against the others
    (``_check_new_adapter_config``) or add parameters that train
    (``_extra_trainable_names``).
    """

    #: The method's layer classes by the module type each wraps — its
    #: family root's table (``LoraLayer._layer_types``).
    _layer_types: ClassVar[dict[type[nn.Module], type[_BaseTunerLayer]]]
    #: The config class this tuner takes, e.g. ``LoraConfig``.
    _config_class: ClassVar[type[_TargetedConfig]]
    #: What a config that names no ``target_modules`` adapts, by the model's ``model_type``.
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = DEFAULT_LORA_TARGET_MODULES
    #: The raw parameters it adapts instead, for a model type whose weights are not layers'.
    _default_parameters: ClassVar[Mapping[str, frozenset[str]]] = DEFAULT_LORA_TARGET_PARAMETERS

    def __init__(self, model: nn.Module, config: PeftConfig, adapter_name: str = "default") -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        kind = type(self)._config_class
        if not isinstance(config, kind):
            raise TypeError(f"config must be a {kind.__name__}, got {type(config).__name__}")
        adapter_name = _check_adapter_name(adapter_name)
        earlier = self._adopted_configs(model, adapter_name)
        # By name, which a deep copy or a pickle of the model keeps; a model
        # adapted before keeps what it had before the first adapter.
        trainable_now = {
            name: param.requires_grad
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        flags = model.__dict__.get(_FLAGS_KEY) if earlier is not None else None
        if flags is None:
            flags = trainable_now
        super().__init__(model, config, adapter_name)
        self.model = model
        self.peft_config: dict[str, PeftConfig] = {} if earlier is None else earlier
        self.active_adapter: str | list[str] = adapter_name
        self.targeted_module_names: list[str] = []
        self.targeted_parameter_names: list[str] = []
        self._requires_grad_before = flags
        injected = self._inject(adapter_name, config)
        try:
            if earlier is not None:
                # The newest adapter is the one in use; a merged earlier one would hide it.
                self.unmerge_adapter()
            # Where ``get_peft_model_state_dict`` finds the config of a model
            # adapted in place, as the reference's does.
            model.peft_config = self.peft_config
            model.__dict__[_FLAGS_KEY] = flags
            self._mark_only_adapters_as_trainable()
        except BaseException:
            # A constructor that raises leaves the model it was given as it found it.
            self._rollback(adapter_name, injected)
            for name, param in model.named_parameters(remove_duplicate=False):
                param.requires_grad_(trainable_now[name])
            if earlier is None:
                model.__dict__.pop("peft_config", None)
                model.__dict__.pop(_FLAGS_KEY, None)
            raise

    @property
    @override
    def active_adapters(self) -> list[str]:
        """The adapters the forward pass applies, as a list."""
        return _as_names(self.active_adapter)

    @classmethod
    def _adopted_configs(cls, model: nn.Module, adapter_name: str) -> dict[str, PeftConfig] | None:
        """The configs of the adapters ``model`` holds, if it already holds some of this method; else ``None``.

        A model adapted before takes the new adapter beside the ones it
        has, as the reference does when ``get_peft_model`` is called on it
        again, with the same warning; the new tuner then answers for all of
        them.

        Raises
        ------
        ValueError
            The model holds another method's adapters, or one named
            ``adapter_name``.
        """
        slots = _adapter_slots(model)
        if not slots:
            return None
        for key, slot in slots:
            # A family's layers share one table, which tells a layer's method.
            if isinstance(slot, _BaseTunerLayer) and slot._layer_types is not cls._layer_types:
                raise ValueError(
                    "Cannot combine adapters with different peft types: the model already holds "
                    f"{type(slot).__name__} (at {key!r}), which is not a layer of "
                    f"{cls.__name__}."
                )
        held = {name for _, slot in slots for name in slot._adapter_names()}
        known: dict[str, PeftConfig] = dict(model.__dict__.get("peft_config", {}))
        if adapter_name in held | set(known):
            raise ValueError(
                f"Adapter with name '{adapter_name}' already exists; the model holds "
                f"{sorted(held | set(known))}.  Pass another adapter_name to add a new one."
            )
        warnings.warn(
            "You are trying to modify a model with PEFT for a second time. If you want to reload "
            "the model with a different config, make sure to call `.unload()` before.",
            stacklevel=6,
        )
        return known

    # ── a method's hooks ──────────────────────────────────────────────────

    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        """The config to store: a copy, its default targets and ``"all-linear"`` spelled out."""
        stored = copy.deepcopy(config)
        if stored.target_modules is None and not cls._adapts_parameters(stored):
            parameters = _default_target_parameters(model, cls._default_parameters)
            stored.target_modules = set(
                _default_target_modules(model, cls._default_targets, parameters)
            )
            skipped = _default_skipped_subtrees(model)
            if skipped:
                stored.exclude_modules = _with_subtrees_excluded(stored.exclude_modules, skipped)
            if parameters and isinstance(stored, _NamesParameters):
                stored.target_parameters = sorted(parameters)
        resolved = _resolve_all_linear(stored, model, cls._adapted_linear)
        if resolved is not None:
            stored.target_modules = resolved
        return stored

    @classmethod
    def _adapted_linear(cls, module: nn.Module) -> nn.Linear | None:
        """The ``nn.Linear`` ``module`` wraps, if it is one of this method's layers around one."""
        if isinstance(module, tuple(cls._layer_types.values())):
            base = module.get_base_layer()
            return base if isinstance(base, nn.Linear) else None
        return None

    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        """Raise ``ValueError`` if ``config`` cannot be added beside the adapters already held."""

    @classmethod
    @abstractmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        """The layer that takes ``target``'s place."""

    @classmethod
    @abstractmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        """Give the layer ``target`` already is one more adapter."""

    @classmethod
    @abstractmethod
    @override
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        """Parameters beyond the adapters' that train, by name under ``model`` (LoRA's biases)."""

    @classmethod
    @override
    def _require_adapter_state(cls, model: nn.Module, adapter_name: str) -> None:
        if not _tuner_layers(model):
            raise RuntimeError(
                f"adapter {adapter_name!r} has been merged or unloaded: the model holds no "
                f"{cls._config_class._peft_type} layer left to save or load"
            )

    @classmethod
    def _adapts_parameters(cls, config: _TargetedConfig) -> bool:
        """Whether ``config`` names raw parameters to adapt.

        Then a config naming no ``target_modules`` adapts no module, rather
        than the model type's default ones.
        """
        del config  # the default adapts modules only
        return False

    @classmethod
    def _plan_parameter_targets(
        cls, model: nn.Module, config: _TargetedConfig, claimed: set[str]
    ) -> list[_Target]:
        """The raw parameters ``config`` adapts, as targets that carry a ``parameter``.

        None by default: a method that adapts parameters finds them here.
        ``claimed`` holds the names of the modules already planned as
        modules — a module cannot be both.
        """
        del model, config, claimed  # the default has no parameter targets
        return []

    # ── injecting an adapter ──────────────────────────────────────────────

    @classmethod
    def _rewrites_base(cls, config: _TargetedConfig) -> bool:
        """Whether building an adapter from ``config`` writes into the base weights (LoRA's PiSSA, OLoRA).

        Those writes happen as the layers are built, ahead of the tree being touched, so
        an injection that fails afterwards has to put the values back (``_Injected``).
        """
        del config  # the default leaves the base as it is
        return False

    def _inject(self, adapter_name: str, config: _TargetedConfig) -> _Injected:
        """Plan, then apply, ``config`` as adapter ``adapter_name``; the model is as it was on failure."""
        model = self.model
        stored = self._prepare_config(config, model)
        self._check_new_adapter_config(stored)
        targets = self._plan_targets(model, stored)
        saved = _plan_modules_to_save(model, stored)
        # Everything new to the tree is built before the first change to it; an
        # initialisation that rewrites the base weights is the one thing built
        # in place, so its targets' values are kept.
        kept = _values_of(t.module for t in targets) if self._rewrites_base(stored) else []
        try:
            placements = self._new_slots(stored, adapter_name, targets, saved)
        except BaseException:
            _put_back(kept)
            raise
        injected = _Injected(placements, kept)
        self.peft_config[adapter_name] = stored
        try:
            self._attach_adapter(stored, adapter_name, targets, saved, placements)
            # A slot new to the tree began with its one adapter active.
            for _, slot in _adapter_slots(model):
                slot._activate(self.active_adapter)
        except BaseException:
            self._rollback(adapter_name, injected)
            raise
        for target in targets:
            names = (
                self.targeted_module_names
                if target.parameter is None
                else self.targeted_parameter_names
            )
            if target.full_name not in names:
                names.append(target.full_name)
        return injected

    def _new_slots(
        self, config: _TargetedConfig, adapter_name: str, targets: list[_Target], saved: list[str]
    ) -> list[_Placement]:
        """The slots the adapter adds to the tree, built and not yet placed.

        One per module: where several parameters of a module are adapted
        the later layers wrap the earlier ones, and the outermost is placed.
        """
        outermost: dict[str, _Placement] = {}
        for target in targets:
            if target.layer is None:
                continue
            earlier = outermost.get(target.key)
            below = target if earlier is None else target._replace(module=earlier.slot)
            original = target.module if earlier is None else earlier.original
            slot = self._create_new_module(config, adapter_name, below)
            outermost[target.key] = _Placement(target.key, slot, original)
        placements = list(outermost.values())
        for key in saved:
            held = self.model.get_submodule(key)
            if not isinstance(held, ModulesToSaveWrapper):
                placements.append(_Placement(key, ModulesToSaveWrapper(held, adapter_name), held))
        return placements

    def _attach_adapter(
        self,
        config: _TargetedConfig,
        adapter_name: str,
        targets: list[_Target],
        saved: list[str],
        placements: list[_Placement],
    ) -> None:
        """Give the adapter to the slots that exist, then put the new slots in the tree."""
        for target in targets:
            if target.layer is None:
                self._add_to_layer(config, adapter_name, target)
        for key in saved:
            wrapper = self.model.get_submodule(key)
            if isinstance(wrapper, ModulesToSaveWrapper):
                wrapper.update(adapter_name)
        for placement in placements:
            _set_submodule(self.model, placement.key, placement.slot)

    def _rollback(self, adapter_name: str, injected: _Injected) -> None:
        """Take ``injected`` back out: the adapter out of the tree, the base weights as they were."""
        self._withdraw(adapter_name, injected.placements)
        _put_back(injected.overwritten)

    def _withdraw(self, adapter_name: str, placements: list[_Placement]) -> None:
        """Undo a failed ``_inject``: the adapter out of every slot, the new slots out of the tree."""
        model = self.model
        for _, slot in _adapter_slots(model):
            slot.delete_adapter(adapter_name)
        for placement in placements:
            if model.get_submodule(placement.key) is placement.slot:
                _set_submodule(model, placement.key, placement.original)
        self.peft_config.pop(adapter_name, None)

    @classmethod
    def _plan_targets(cls, model: nn.Module, config: _TargetedConfig) -> list[_Target]:
        targets = []
        for key, module in _outside_slots(model):
            if key and _selects(config, key):
                targets.append(cls._target_for(key, module))
        targets += cls._plan_parameter_targets(model, config, {t.key for t in targets})
        if not targets:
            raise _not_found(config)
        return targets

    @classmethod
    def _target_for(cls, key: str, module: nn.Module) -> _Target:
        if isinstance(module, tuple(cls._layer_types.values())):
            return _Target(key, module, None)
        return _Target(key, module, cls._target_layer(key, module))

    @classmethod
    def _target_layer(cls, key: str, module: nn.Module) -> type[_BaseTunerLayer]:
        for kind in type(module).__mro__:
            layer = cls._layer_types.get(kind)
            if layer is not None:
                return layer
        supported = ", ".join(f"nn.{k.__name__}" for k in cls._layer_types)
        raise ValueError(
            f"Target module {key} ({type(module).__name__}) is not supported. "
            f"Currently, only {supported} is supported."
        )

    @override
    def add_adapter(
        self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False
    ) -> None:
        """Add an adapter beside the ones the model holds.

        The new adapter starts frozen and inactive: the forward pass keeps
        applying the adapters it applied, and every parameter keeps the
        ``requires_grad`` it had.  ``set_adapter`` switches to it.  A
        module an earlier adapter already wraps takes the new one beside
        it; any other target is wrapped.  The model is left as it was if
        the adapter cannot be added.

        Parameters
        ----------
        adapter_name : str
            Name of the new adapter; unique among the model's adapters.
        peft_config : PeftConfig
            What to adapt and how — the config class of this tuner's method.
        low_cpu_mem_usage : bool, default=False
            Accepted for the reference's signature.  There is no meta
            device to build the adapter on: it is allocated either way.

        Raises
        ------
        ValueError
            The name is taken or malformed, the config is another
            method's, no module matches it, or it cannot sit beside the
            adapters held (two adapters training biases).
        TypeError
            ``peft_config`` is not an adapter config.
        """
        del low_cpu_mem_usage  # a build-time memory hint; the adapter is the same either way
        adapter_name = self._admit(adapter_name, peft_config)
        assert isinstance(peft_config, _TargetedConfig)  # _admit checked the class
        self._inject(adapter_name, peft_config)
        self.set_requires_grad(adapter_name, False)

    # ── switching, freezing, deleting ─────────────────────────────────────

    def _check_weighted_sources(
        self, adapters: list[str], weights: list[float], adapter_name: str
    ) -> str:
        """The name of an adapter to be combined from ``adapters``, once the sources are checked.

        The rules every ``add_weighted_adapter`` shares: a free, well-formed name; sources that
        exist, one weight each; and no ``modules_to_save`` module trained by two sources (two
        copies of one module cannot be averaged).
        """
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        if not adapters:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        for name in adapters:
            if name not in self.peft_config:
                raise ValueError(f"Adapter {name} does not exist")
        if len(weights) != len(adapters):
            raise ValueError(
                f"add_weighted_adapter needs one weight per adapter, got {len(weights)} "
                f"weights for {len(adapters)} adapters"
            )
        for key, wrapper in _named(self.model, ModulesToSaveWrapper):
            holders = [name for name in adapters if name in wrapper.modules_to_save]
            if len(holders) > 1:
                raise ValueError(
                    "Cannot add weighted adapters if they target the same module with "
                    f"modules_to_save, but {key!r} is trained by {holders}."
                )
        return adapter_name

    def _add_combined_adapter(
        self, adapter_name: str, config: PeftConfig, fill: Callable[[], None]
    ) -> None:
        """Add ``adapter_name``, then ``fill`` it from other adapters; withdrawn if ``fill`` fails.

        A half-filled adapter must not stay behind under a name that looks complete.
        """
        self.add_adapter(adapter_name, config)
        try:
            fill()
        except BaseException:
            self.delete_adapter(adapter_name)
            raise

    def _require_known(self, adapter_names: Sequence[str]) -> None:
        for name in adapter_names:
            if name not in self.peft_config:
                raise ValueError(f"Adapter {name} not found.")

    @override
    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Apply the named adapter(s) in the forward pass, and train exactly those.

        Every other adapter is frozen; the activated ones train unless
        ``inference_mode``.  A merged layer is unmerged first.  Layers that
        hold none of the named adapters run their base alone.  The
        parameters ``bias`` makes trainable follow the adapter that asked
        for them.

        Parameters
        ----------
        adapter_name : str or list of str
            The adapter, or adapters, to apply.
        inference_mode : bool, default=False
            Freeze the activated adapters too.

        Raises
        ------
        ValueError
            A name is not one of the model's adapters, or two named
            adapters train the same ``modules_to_save`` module.
        """
        names = _as_names(adapter_name)
        self._require_known(names)
        slots = _adapter_slots(self.model)
        for _, slot in slots:
            slot._check_activation(names)
        for _, slot in slots:
            slot.set_adapter(adapter_name, inference_mode)
        self.active_adapter = adapter_name if isinstance(adapter_name, str) else [*adapter_name]
        params = dict(self.model.named_parameters())
        for held, config in self.peft_config.items():
            on = held in names and not inference_mode
            for extra in self._extra_trainable_names(self.model, config):
                params[extra].requires_grad_(on)

    @override
    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Enable or disable gradients on the parameters of the named adapter(s).

        Raises
        ------
        ValueError
            A name is not one of the model's adapters.
        """
        names = _as_names(adapter_names)
        self._require_known(names)
        for _, slot in _adapter_slots(self.model):
            slot.set_requires_grad(names, requires_grad)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove an adapter: its parameters, its config, its ``modules_to_save`` copies.

        If it was active, the adapters still active stay so; when none
        remain the first of the others (by name) takes over, with a
        warning, and trains.  Deleting the last adapter leaves the layers
        in the model, applying nothing; ``unload()`` removes them.

        Raises
        ------
        ValueError
            The adapter does not exist, or it is merged — unmerge first.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} does not exist")
        merged = {name for _, layer in _tuner_layers(self.model) for name in layer.merged_adapters}
        if adapter_name in merged:
            raise ValueError(
                f"Cannot delete adapter(s) [{adapter_name!r}] while they are merged. "
                "Please unmerge them first."
            )
        del self.peft_config[adapter_name]
        for _, slot in _adapter_slots(self.model):
            slot.delete_adapter(adapter_name)
        active = self.active_adapters
        if adapter_name not in active:
            return
        remaining = [name for name in active if name != adapter_name]
        if not remaining and self.peft_config:
            successor = sorted(self.peft_config)[0]
            warnings.warn(
                f"Adapter {adapter_name} was active which is now deleted. "
                f"Setting active adapter to {successor}.",
                stacklevel=3,
            )
            remaining = [successor]
        self.set_adapter(remaining[0] if len(remaining) == 1 else remaining)

    def _mark_only_adapters_as_trainable(self) -> None:
        model = self.model
        trainable: set[int] = set()
        params = dict(model.named_parameters())
        for name in self.active_adapters:
            config = self.peft_config[name]
            if config.inference_mode:
                continue
            for _, slot in _adapter_slots(model):
                trainable.update(id(p) for p in slot._adapter_parameters(name))
            trainable.update(id(params[n]) for n in self._extra_trainable_names(model, config))
        for param in model.parameters():
            param.requires_grad_(id(param) in trainable)

    @override
    def _mixed_batch(
        self, adapter_names: Sequence[str], num_beams: int = 1
    ) -> AbstractContextManager[None]:
        """A context in which each batch row goes through the adapter ``adapter_names`` names for it.

        Only a method that can answer it implements this (LoRA).

        Raises
        ------
        NotImplementedError
            This method cannot route the rows of a batch to different adapters.
        """
        del adapter_names, num_beams
        raise NotImplementedError(
            f"{type(self).__name__} cannot send the rows of a batch through different adapters"
        )

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        return self.model(*args, **kwargs)

    # ── merging and unloading ─────────────────────────────────────────────

    def _mergeable_layers(self) -> list[tuple[str, _BaseTunerLayer]]:
        """The tuner layers, after refusing any whose base weight another module shares."""
        layers = _tuner_layers(self.model)
        paths = _parameter_paths(self.model)
        for key, layer in layers:
            for leaf, param in layer._merge_targets().items():
                own = _join(key, leaf)
                others = [path for path in paths[id(param)] if path != own]
                if others:
                    raise RuntimeError(
                        f"cannot merge into {own}: the same Parameter is also "
                        f"{', '.join(others)}, which merging would change too — keep this "
                        "adapter unmerged, or untie the weight first"
                    )
        return layers

    def merge_adapter(
        self, adapter_names: list[str] | None = None, safe_merge: bool = False
    ) -> None:
        r"""Fold adapter updates into their base weights, keeping the adapter layers.

        A merged model runs at the base model's cost; ``unmerge_adapter``
        takes the updates out again.  Every refusal is made before any
        weight is written: nothing is merged if a layer's base weight is
        shared with another module (a tied LM head — merging would change
        that module too) or if the adapter is disabled.

        Parameters
        ----------
        adapter_names : list of str, optional
            Adapters to merge; the active ones by default.  A layer holding
            only some of them merges those.
        safe_merge : bool, default=False
            Check every merged weight for NaN / inf before writing any.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module, or the adapter is
            disabled (inside ``disable_adapter()``).
        ValueError
            A name is not one of the model's adapters, or ``safe_merge``
            found a non-finite merged weight.
        """
        names = self.active_adapters if adapter_names is None else list(adapter_names)
        self._require_known(names)
        pending = []
        for _, layer in self._mergeable_layers():
            held = [name for name in names if name in layer._adapter_names()]
            pending.append((layer, held, layer._pending_merges(held)))
        if safe_merge:
            for layer, _, adapters in pending:
                for adapter in adapters:
                    layer._check_finite_merge(adapter)
        for layer, held, _ in pending:
            if held:
                layer.merge(adapter_names=held)

    def unmerge_adapter(self) -> None:
        """Take every merged adapter update back out of its base weight."""
        for _, layer in _tuner_layers(self.model):
            if layer.merged:
                layer.unmerge()

    def _unload(self, keep_trained: bool) -> nn.Module:
        model = self.model
        for key, slot in _outermost_slots(model):
            _set_submodule(model, key, slot._unwrap(keep_trained))
        for name, param in model.named_parameters(remove_duplicate=False):
            before = self._requires_grad_before.get(name)
            if before is not None:
                param.requires_grad_(before)
        if "peft_config" in model.__dict__:
            del model.peft_config
        model.__dict__.pop(_FLAGS_KEY, None)
        return model

    @override
    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        r"""Merge the adapters into the base weights and return the plain model.

        Each adapter layer is replaced by its merged base module — the base
        model's own module, holding its own ``Parameter`` objects — and each
        ``modules_to_save`` entry by the active adapter's trained copy (the
        original when that adapter has none).  Adapters not merged are
        dropped with their layers.  Every base parameter gets back the
        ``requires_grad`` it had before injection.

        Parameters
        ----------
        progressbar : bool, default=False
            Accepted for the reference's signature; there is no progress bar.
        safe_merge : bool, default=False
            Check every merged weight for NaN / inf before writing any.
        adapter_names : list of str, optional
            Adapters to merge; the active ones by default.

        Returns
        -------
        nn.Module
            The model, merged, with no adapter layer left in it.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module, or a
            ``modules_to_save`` module shared one (its trained copy would
            break the tie).
        """
        del progressbar  # the reference's signature; nothing to report progress on
        self._refuse_untying()
        self.merge_adapter(adapter_names=adapter_names, safe_merge=safe_merge)
        return self._unload(keep_trained=True)

    def _refuse_untying(self) -> None:
        """Refuse to swap in a trained copy of a module whose weight another module shares.

        The copy trained apart from the tie, so the merged model would hold
        two different tensors under names the base model ties into one —
        and a loader that re-ties them (the zoo's ``from_pretrained``)
        keeps whichever it reads last.
        """
        paths = _parameter_paths(self.model)
        for key, wrapper in _named(self.model, ModulesToSaveWrapper):
            tied = _shared_outside(paths, key, wrapper.original_module)
            if tied:
                raise RuntimeError(
                    f"cannot merge_and_unload: modules_to_save {key!r} was trained apart from "
                    f"{', '.join(tied)}, which it shares a weight with; the merged model would "
                    "hold two tensors where the base ties one.  Keep the adapter "
                    "(save_pretrained / PeftModel.from_pretrained), or untie the weight before "
                    "injecting"
                )

    @override
    def unload(self) -> nn.Module:
        """Remove every adapter without merging and return the original model.

        Every adapter layer is replaced by the module it wrapped (a merged
        one is unmerged first), every ``modules_to_save`` entry by its
        original module, and every base parameter gets back the
        ``requires_grad`` it had before injection.
        """
        return self._unload(keep_trained=False)

    @override
    def enable_adapter_layers(self) -> None:
        """Apply the adapter again after ``disable_adapter_layers``."""
        for _, slot in _adapter_slots(self.model):
            slot.enable_adapters(True)

    @override
    def disable_adapter_layers(self) -> None:
        """Run the base model alone: adapter updates off, original ``modules_to_save``."""
        for _, slot in _adapter_slots(self.model):
            slot.enable_adapters(False)


def _not_found(config: _TargetedConfig) -> ValueError:
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
    return ValueError(message)


def _plan_modules_to_save(
    model: nn.Module, config: _TargetedConfig | PromptLearningConfig
) -> list[str]:
    names = config.modules_to_save
    if not names:
        return []
    keys = _module_names_matching(model, names)
    if not keys:
        raise ValueError(f"modules_to_save {names} matched no module of the model")
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
