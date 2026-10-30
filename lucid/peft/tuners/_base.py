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
``modules_to_save`` copy — answers the ``_AdapterSlot`` protocol, and the
generic code walks slots rather than naming their classes.

Injection is planned in full before the model is touched — every target
found, typed and given its layer, every ``modules_to_save`` entry found —
so a config that cannot be applied raises with the model as it was.  The
layers then go in by ``setattr`` on their parents, which keeps each
child's slot in place (a ``Sequential`` still runs it in the same turn).
"""

import copy
import warnings
from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple, Protocol, override, runtime_checkable

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft._forwarding import _as_module, _ForwardsAttributes
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._targeting import (
    _module_names_matching,
    _parameter_paths,
    _resolve_all_linear,
    _selects,
    _shared_outside,
)


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _named[T: nn.Module](model: nn.Module, kind: type[T]) -> list[tuple[str, T]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, kind)]


def _set_submodule(model: nn.Module, key: str, module: nn.Module) -> None:
    parent, _, child = key.rpartition(".")
    setattr(model.get_submodule(parent), child, module)


def _check_adapter_name(adapter_name: object) -> str:
    if not isinstance(adapter_name, str):
        raise TypeError(f"adapter_name must be a str, got {adapter_name!r}")
    if not adapter_name or "." in adapter_name:
        raise ValueError(f"adapter_name must be a non-empty name without '.', got {adapter_name!r}")
    return adapter_name


@runtime_checkable
class _AdapterSlot(Protocol):
    """A module standing in for a base module: a tuner layer, a ``modules_to_save`` copy.

    What the tuner needs of each — switch it, list what trains, map its
    keys to the file's, and put the base back — so it walks slots without
    knowing their classes.
    """

    def get_base_layer(self) -> nn.Module:
        """The module this one stands in for."""
        ...

    def enable_adapters(self, enabled: bool) -> None:
        """Take part (``True``) or let the base module run alone (``False``)."""
        ...

    def _unwrap(self, keep_trained: bool) -> nn.Module:
        """What goes back in this slot's place: the trained result, or the base as it was."""
        ...

    def _adapter_parameters(self, adapter_name: str) -> list[Parameter]:
        """The parameters that train for ``adapter_name``."""
        ...

    def _adapter_keys(self, adapter_name: str) -> dict[str, str]:
        """Own state-dict key → the adapter file's key, for ``adapter_name``'s tensors."""
        ...


def _adapter_slots(model: nn.Module) -> list[tuple[str, _AdapterSlot]]:
    return [(name, m) for name, m in model.named_modules() if isinstance(m, _AdapterSlot)]


def _tuner_layers(model: nn.Module) -> list[tuple[str, _BaseTunerLayer]]:
    # Not ``_named``: a ``type[T]`` parameter takes concrete classes only.
    return [(name, m) for name, m in model.named_modules() if isinstance(m, _BaseTunerLayer)]


class _BaseTunerLayer(nn.Module, ABC):
    """The base of a tuner's layers: the wrapped module and the state forward branches on.

    A tuner family's root class names its per-adapter ``ModuleDict``\\ s in
    ``adapter_layer_names`` and starts a table of layer classes; each
    concrete layer joins it with ``base_type=`` — the kind of module it
    wraps — and the tuner finds a target's layer class there.

    What the forward pass branches on — ``active_adapter``,
    ``disable_adapters``, ``merged`` — are plain attributes in the
    instance ``__dict__``, not properties or descriptors as in the
    reference: ``lucid.compile`` keys an executable on each module's
    public scalar ``__dict__`` attributes, and state kept anywhere else
    would let a compiled model replay a branch it has left (LCD-703).
    They change only through ``update_layer``, ``enable_adapters``,
    ``merge`` and ``unmerge``.
    """

    #: The per-adapter ``ModuleDict`` attributes; a family's root sets it.
    adapter_layer_names: ClassVar[tuple[str, ...]] = ()
    #: Wrapped module type → layer class, one table per family.
    _layer_types: ClassVar[dict[type[nn.Module], type[_BaseTunerLayer]]] = {}
    #: The module type this layer wraps, from ``base_type=``.
    _base_type: ClassVar[type[nn.Module]] = nn.Module

    @override
    def __init_subclass__(
        cls, *, base_type: type[nn.Module] | None = None, **kwargs: object
    ) -> None:
        super().__init_subclass__(**kwargs)
        if "adapter_layer_names" in cls.__dict__:
            cls._layer_types = {}
        if base_type is not None:
            cls._base_type = base_type
            cls._layer_types[base_type] = cls

    def __init__(self, base_layer: nn.Module) -> None:
        kind = type(self)._base_type
        if not isinstance(base_layer, kind):
            raise TypeError(
                f"{type(self).__name__} adapts nn.{kind.__name__} in lucid.peft v1, "
                f"got {type(base_layer).__name__}"
            )
        super().__init__()
        self.base_layer = base_layer
        self.merged_adapters: list[str] = []
        #: Name of the adapter the forward pass applies.
        self.active_adapter = ""
        #: Whether the forward pass runs the base layer alone.
        self.disable_adapters = False
        #: Whether an adapter is folded into the base weight (``merged_adapters`` non-empty).
        self.merged = False

    @abstractmethod
    def get_base_layer(self) -> nn.Module:
        """Return the wrapped module, whose parameters are the base model's own.

        Each family returns its own base type (``LoraLayer``: ``nn.Linear``).
        """

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

    def _pending_merges(self, adapter_names: list[str] | None) -> list[str]:
        """The requested adapters (the active one by default) not merged yet.

        A disabled layer runs its base alone, so merging into it would make
        it run the adapter while it reads as disabled; it is refused.
        """
        if self.disable_adapters:
            raise RuntimeError(
                "the adapter is disabled; enable it (leave disable_adapter()) before merging"
            )
        names = [self.active_adapter] if adapter_names is None else list(adapter_names)
        unknown = [name for name in names if name not in self._adapters()]
        if unknown:
            raise ValueError(f"unknown adapter(s) {unknown}; this layer holds {self._adapters()}")
        return [name for name in names if name not in self.merged_adapters]

    def _adapters(self) -> list[str]:
        """The adapter names this layer holds."""
        tables = self._tables()
        return list(tables[0]) if tables else []

    def _tables(self) -> list[nn.ModuleDict[nn.Module]]:
        """The per-adapter ``ModuleDict``\\ s ``adapter_layer_names`` names."""
        return [
            table
            for name in self.adapter_layer_names
            if isinstance(table := getattr(self, name), nn.ModuleDict)
        ]

    def _adapter_parameters(self, adapter_name: str) -> list[Parameter]:
        return [p for table in self._tables() for p in table[adapter_name].parameters()]

    def _adapter_keys(self, adapter_name: str) -> dict[str, str]:
        return {
            f"{name}.{adapter_name}.{leaf}": f"{name}.{leaf}"
            for name, table in zip(self.adapter_layer_names, self._tables(), strict=True)
            for leaf in table[adapter_name].state_dict()
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
    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """Fold the adapter into the base weight, in place."""

    @abstractmethod
    def unmerge(self) -> None:
        """Take every merged adapter back out of the base weight, in place."""

    @abstractmethod
    def _merge_targets(self) -> dict[str, Parameter]:
        """The base parameters ``merge`` writes, by their name under this layer."""

    @abstractmethod
    def _check_finite_merge(self, adapter: str) -> None:
        """Raise ``ValueError`` if merging ``adapter`` would write NaN or inf."""


class _Target(NamedTuple):
    key: str
    module: nn.Module
    layer: type[_BaseTunerLayer]


#: ``PeftType`` → the tuner that applies it, filled as each is defined.
_TUNERS: dict[PeftType, type[_BaseTuner]] = {}


def _tuner_for(config: PeftConfig) -> type[_BaseTuner]:
    """The tuner registered for ``config``'s method."""
    for peft_type, tuner in _TUNERS.items():
        if config.peft_type == peft_type:
            return tuner
    raise NotImplementedError(f"lucid.peft has no tuner for {config.peft_type} adapters")


class _BaseTuner(_ForwardsAttributes, ABC):
    """Put a method's layers into a model, freeze the rest, take them out again.

    The generic half of a tuner.  Construction checks, plans the whole
    injection, applies it and freezes; a method supplies its layer family
    (``_layer_types``), the module types it refuses as not-yet-supported,
    how to build a layer for a target (``_create_new_module``), and may
    adjust the stored config (``_prepare_config``) or add parameters that
    train (``_extra_trainable_names``).
    """

    #: The method's layer classes by the module type each wraps — its
    #: family root's table (``LoraLayer._layer_types``).
    _layer_types: ClassVar[dict[type[nn.Module], type[_BaseTunerLayer]]]
    #: The config class this tuner takes, e.g. ``LoraConfig``.
    _config_class: ClassVar[type[_TargetedConfig]]
    #: Module types the reference adapts with this method and v1 does not yet.
    _later_types: ClassVar[tuple[type[nn.Module], ...]] = ()

    @override
    def __init_subclass__(cls, *, peft_type: PeftType | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if peft_type is not None:
            _TUNERS[peft_type] = cls

    def __init__(self, model: nn.Module, config: PeftConfig, adapter_name: str = "default") -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        kind = type(self)._config_class
        if not isinstance(config, kind):
            raise TypeError(f"config must be a {kind.__name__}, got {type(config).__name__}")
        adapter_name = _check_adapter_name(adapter_name)
        _refuse_existing_adapters(model)
        stored = self._prepare_config(config, model)
        targets = self._plan_targets(model, stored)
        saved = _plan_modules_to_save(model, stored)
        # By name, which a deep copy or a pickle of the model keeps.
        flags = {
            name: param.requires_grad
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        layers: list[tuple[str, nn.Module]] = [
            (t.key, self._create_new_module(stored, adapter_name, t)) for t in targets
        ]
        layers += [
            (key, ModulesToSaveWrapper(model.get_submodule(key), adapter_name)) for key in saved
        ]
        super().__init__()
        self.model = model
        self.peft_config: dict[str, PeftConfig] = {adapter_name: stored}
        self.active_adapter: str = adapter_name
        self.targeted_module_names: list[str] = [target.key for target in targets]
        self._requires_grad_before = flags
        for key, layer in layers:
            _set_submodule(model, key, layer)
        # Where ``get_peft_model_state_dict`` finds the config of a model
        # adapted in place, as the reference's does.
        model.peft_config = self.peft_config
        self._mark_only_adapters_as_trainable()

    # ── a method's hooks ──────────────────────────────────────────────────

    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        """The config to store: a copy of ``config``, ``"all-linear"`` spelled out."""
        stored = copy.deepcopy(config)
        resolved = _resolve_all_linear(stored, model)
        if resolved is not None:
            stored.target_modules = resolved
        return stored

    @classmethod
    @abstractmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        """The layer that takes ``target``'s place."""

    @classmethod
    @abstractmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        """Parameters beyond the adapters' that train, by name under ``model`` (LoRA's biases)."""

    # ── generic steps ─────────────────────────────────────────────────────

    @classmethod
    def _plan_targets(cls, model: nn.Module, config: _TargetedConfig) -> list[_Target]:
        targets = []
        for key, module in model.named_modules():
            if key and _selects(config, key):
                targets.append(_Target(key, module, cls._target_layer(key, module)))
        if not targets:
            raise _not_found(config)
        return targets

    @classmethod
    def _target_layer(cls, key: str, module: nn.Module) -> type[_BaseTunerLayer]:
        for kind in type(module).__mro__:
            layer = cls._layer_types.get(kind)
            if layer is not None:
                return layer
        supported = ", ".join(f"nn.{k.__name__}" for k in cls._layer_types)
        if isinstance(module, cls._later_types):
            raise NotImplementedError(
                f"Target module {key} ({type(module).__name__}) is not supported in "
                f"lucid.peft v1, which adapts {supported} only"
            )
        raise ValueError(
            f"Target module {key} ({type(module).__name__}) is not supported. "
            f"Currently, only {supported} is supported."
        )

    @override
    def _wrapped(self) -> nn.Module | None:
        modules = self.__dict__.get("_modules")
        return _as_module(modules.get("model")) if modules else None

    def _mark_only_adapters_as_trainable(self) -> None:
        config = self.peft_config[self.active_adapter]
        model = self.model
        trainable: set[int] = set()
        if not config.inference_mode:
            for _, slot in _adapter_slots(model):
                trainable.update(id(p) for p in slot._adapter_parameters(self.active_adapter))
            params = dict(model.named_parameters())
            extra = self._extra_trainable_names(model, config)
            trainable.update(id(params[name]) for name in extra)
        for _, param in model.named_parameters():
            param.requires_grad_(id(param) in trainable)

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        return self.model(*args, **kwargs)

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
        r"""Fold every adapter update into its base weight, keeping the adapter layers.

        A merged model runs at the base model's cost; ``unmerge_adapter``
        takes the updates out again.  Every refusal is made before any
        weight is written: nothing is merged if a layer's base weight is
        shared with another module (a tied LM head — merging would change
        that module too) or if the adapter is disabled.

        Parameters
        ----------
        adapter_names : list of str, optional
            Adapters to merge; the active one by default.
        safe_merge : bool, default=False
            Check every merged weight for NaN / inf before writing any.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module, or the adapter is
            disabled (inside ``disable_adapter()``).
        ValueError
            ``safe_merge`` found a non-finite merged weight.
        """
        layers = self._mergeable_layers()
        pending = [(layer, layer._pending_merges(adapter_names)) for _, layer in layers]
        if safe_merge:
            for layer, adapters in pending:
                for adapter in adapters:
                    layer._check_finite_merge(adapter)
        for layer, _ in pending:
            layer.merge(adapter_names=adapter_names)

    def unmerge_adapter(self) -> None:
        """Take every merged adapter update back out of its base weight."""
        for _, layer in _tuner_layers(self.model):
            if layer.merged:
                layer.unmerge()

    def _unload(self, keep_trained: bool) -> nn.Module:
        model = self.model
        for key, slot in _adapter_slots(model):
            _set_submodule(model, key, slot._unwrap(keep_trained))
        for name, param in model.named_parameters(remove_duplicate=False):
            before = self._requires_grad_before.get(name)
            if before is not None:
                param.requires_grad_(before)
        if "peft_config" in model.__dict__:
            del model.peft_config
        return model

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        r"""Merge the adapter into the base weights and return the plain model.

        Each adapter layer is replaced by its merged base module — the base
        model's own module, holding its own ``Parameter`` objects — and each
        ``modules_to_save`` entry by its trained copy.  Every base parameter
        gets back the ``requires_grad`` it had before injection.

        Parameters
        ----------
        progressbar : bool, default=False
            Accepted for the reference's signature; there is no progress bar.
        safe_merge : bool, default=False
            Check every merged weight for NaN / inf before writing any.
        adapter_names : list of str, optional
            Adapters to merge; the active one by default.

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

    def unload(self) -> nn.Module:
        """Remove the adapter without merging it and return the original model.

        Every adapter layer is replaced by the module it wrapped (a merged
        one is unmerged first), every ``modules_to_save`` entry by its
        original module, and every base parameter gets back the
        ``requires_grad`` it had before injection.
        """
        return self._unload(keep_trained=False)

    def enable_adapter_layers(self) -> None:
        """Apply the adapter again after ``disable_adapter_layers``."""
        for _, slot in _adapter_slots(self.model):
            slot.enable_adapters(True)

    def disable_adapter_layers(self) -> None:
        """Run the base model alone: adapter updates off, original ``modules_to_save``."""
        for _, slot in _adapter_slots(self.model):
            slot.enable_adapters(False)


def _not_found(config: _TargetedConfig) -> ValueError:
    targets = config.target_modules
    if targets is None:
        # The reference falls back to a table of per-architecture defaults;
        # Lucid's zoo has none yet, so the choice is the caller's.
        return ValueError(
            "Please specify `target_modules` in `peft_config`: a list of module names, "
            "a regex, or 'all-linear'"
        )
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


def _plan_modules_to_save(model: nn.Module, config: _TargetedConfig) -> list[str]:
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
                stacklevel=4,
            )
    return keys


def _refuse_existing_adapters(model: nn.Module) -> None:
    held = [name for name, _ in _adapter_slots(model)]
    if held:
        raise NotImplementedError(
            f"the model already holds a LoRA adapter (at {held[0]}); lucid.peft v1 keeps one "
            "adapter per model — call unload() or merge_and_unload() first"
        )
