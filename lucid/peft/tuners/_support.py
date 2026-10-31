"""Pieces the layer and tuner families share: spec views, dtype, additive weight updates, ``fan_in_fan_out``."""

import warnings
from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from types import MappingProxyType
from typing import ClassVar, Final, Protocol, Self, overload, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid._types import DTypeLike
from lucid.nn.parameter import Parameter
from lucid.peft._config import PeftConfig, _TargetedConfig
from lucid.peft.tuners._base import (
    _NO_PARAMETERS,
    _BaseTuner,
    _BaseTunerLayer,
    _check_adapter_name,
    _Target,
)

_HALF: Final = (lucid.float16, lucid.bfloat16)
_WEIGHT: Final = "base_layer.weight"


def _adapter_dtype(anchor: Tensor) -> DTypeLike:
    """The dtype an adapter's tensors take beside ``anchor``: its own, widened to float32 if half."""
    return lucid.float32 if anchor.dtype in _HALF else anchor.dtype


class _Specced(Protocol):
    """A layer that keeps a ``NamedTuple`` of scalars per adapter in ``_specs``."""

    @property
    def _specs(self) -> Mapping[str, tuple[object, ...]]: ...


class _SpecField[T]:
    """A read-only view of one field of every adapter's spec: ``{adapter name: value}``.

    The field is the attribute's own name, so ``scaling = _SpecField[float](doc)`` reads
    ``spec.scaling`` of each adapter in ``layer._specs``.
    """

    def __init__(self, doc: str) -> None:
        self.__doc__ = doc
        self._field = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._field = name

    @overload
    def __get__(self, layer: None, owner: type) -> Self: ...

    @overload
    def __get__(self, layer: _Specced, owner: type) -> Mapping[str, T]: ...

    def __get__(self, layer: _Specced | None, owner: type) -> Self | Mapping[str, T]:
        if layer is None:
            return self
        return MappingProxyType({n: getattr(spec, self._field) for n, spec in layer._specs.items()})


class _WeightRewrite(_BaseTunerLayer):
    """A layer whose every adapter rewrites the one base weight.

    A subclass says how a weight becomes the adapted one (``_rewritten``) and
    how the adapted one becomes the weight again (``_reverted``).  Merging
    writes the first in place and unmerging applies the second for the merged
    adapters, newest first.
    """

    @abstractmethod
    def _rewritten(self, weight: Tensor, adapter: str) -> Tensor:
        """``weight`` with ``adapter`` applied."""

    @abstractmethod
    def _reverted(self, weight: Tensor, adapter: str) -> Tensor:
        """``weight`` with ``adapter`` taken back out."""

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        return {_WEIGHT: self.weight}

    @override
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        weight = self.weight
        return {_WEIGHT: self._rewritten(weight, adapter).to(weight.dtype)}

    @override
    def _fold(self, adapter: str) -> None:
        self.weight.copy_(self._merged_values(adapter)[_WEIGHT])

    @override
    def _unfold(self) -> None:
        weight = self.weight
        for adapter in reversed(self.merged_adapters):
            weight.copy_(self._reverted(weight, adapter).to(weight.dtype))


class _AdditiveWeight(_WeightRewrite):
    """A layer whose update is a weight-shaped change added to the base weight.

    A subclass gives ``get_delta_weight``, the change.  Merging adds it to the
    base weight in place and unmerging subtracts it.
    """

    @override
    def _rewritten(self, weight: Tensor, adapter: str) -> Tensor:
        return weight + self.get_delta_weight(adapter)

    @override
    def _reverted(self, weight: Tensor, adapter: str) -> Tensor:
        return weight - self.get_delta_weight(adapter)


class _ResidualUpdate(_AdditiveWeight):
    """An additive layer whose forward pass is the base output plus each active adapter's contribution.

    A subclass gives ``_contribution``, the update applied to an input.  An
    adapter the layer holds nothing for (one aimed at other layers) is not applied.
    """

    @abstractmethod
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        """The adapter's addition to the base output for the input ``x``."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.base_layer(x)
        if self.disable_adapters or self.merged:
            return result
        out = result
        held = self._adapter_names()
        for adapter in self.active_adapters:
            if adapter in held:  # an adapter aimed at other layers: this one runs its base
                out = out + self._contribution(adapter, x)
        return out if out.dtype == result.dtype else out.to(result.dtype)


class _HasFanInFanOut(Protocol):
    fan_in_fan_out: bool


def _reset_fan_in_fan_out(config: _HasFanInFanOut) -> None:
    """Warn and clear ``config.fan_in_fan_out``: ``nn.Linear`` stores its weight ``(out, in)``, so the flag has no meaning here."""
    if config.fan_in_fan_out:
        warnings.warn(
            "fan_in_fan_out is set to True but the target module is `nn.Linear`. "
            "Setting fan_in_fan_out to False.",
            stacklevel=6,
        )
        config.fan_in_fan_out = False


def _config_as[C: _TargetedConfig](kind: type[C], config: _TargetedConfig) -> C:
    """``config`` as the config class ``kind``: the tuner has checked it, a layer used on its own may not have."""
    if not isinstance(config, kind):
        raise TypeError(f"config must be a {kind.__name__}, got {type(config).__name__}")
    return config


class _ConfigLayer(_BaseTunerLayer):
    """A layer built from the module, an adapter name and the method's config alone.

    The shape OFT, BOFT, HRA and LN tuning share: a family root names its
    per-adapter tables (``adapter_layer_names``, all ``ParameterDict`` or all
    ``ModuleDict`` by ``_table``), creates an adapter in ``update_layer`` and
    keeps its scalar state in ``_specs`` — a tuple per adapter, so the scalars
    ``lucid.compile`` keys a trace on include them.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : _TargetedConfig
        The method's config, read by ``update_layer``.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts, or ``config`` is another method's.
    ValueError
        ``base_layer`` builds parameters on its first input, or the config does not fit it.
    """

    _specs: MutableMapping[str, tuple[object, ...]]
    #: The prefix of the layer's ``repr`` (``"oft."``), as the reference prints it.
    _repr_prefix: ClassVar[str] = ""

    def __init__(self, base_layer: nn.Module, adapter_name: str, config: _TargetedConfig) -> None:
        super().__init__(base_layer)
        self._specs = {}
        for table in self.adapter_layer_names:
            setattr(self, table, self._new_table())
        self.update_layer(adapter_name, config)

    def _new_table(self) -> nn.ParameterDict | nn.ModuleDict[nn.Module]:
        """An empty per-adapter table: bare parameters by default, modules where an adapter is one (LN tuning)."""
        return nn.ParameterDict()

    def update_layer(self, adapter_name: str, config: _TargetedConfig) -> None:
        """Create adapter ``adapter_name`` from ``config``.

        The adapter is built whole before the layer holds any of it.  A layer
        that holds an adapter already keeps applying the adapters it applied:
        the new one is inactive until ``set_adapter`` names it; a layer with
        none starts applying this one.

        Parameters
        ----------
        adapter_name : str
            Key of the adapter in each of the layer's tables.
        config : _TargetedConfig
            The method's config.

        Raises
        ------
        ValueError
            The layer already holds an adapter of this name, or ``config`` does not fit the layer.
        """
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self._adapter_names():
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        self._enter(adapter_name, config)
        if not self.active_adapters:
            self.active_adapter = adapter_name

    @abstractmethod
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        """Allocate the adapter's tensors, then put them (and its spec) in the tables; the name is free."""

    @override
    def __repr__(self) -> str:
        return type(self)._repr_prefix + super().__repr__()

    @override
    def extra_repr(self) -> str:
        return (
            f"adapter={self.active_adapter!r}, merged={self.merged}, "
            f"disabled={self.disable_adapters}"
        )

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        self._specs.pop(adapter_name, None)


class _ConfigTuner(_BaseTuner):
    """A tuner whose layers are built by ``layer(module, adapter, config)`` and extended by ``layer.update_layer(adapter, config)``."""

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        layer = target.layer
        assert layer is not None and issubclass(layer, _ConfigLayer)
        return layer(target.module, adapter_name, config)

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        layer = target.module
        assert isinstance(layer, _ConfigLayer)
        layer.update_layer(adapter_name, config)

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        del model, config  # these methods train their own tensors only
        return []

    _default_parameters: ClassVar[Mapping[str, frozenset[str]]] = _NO_PARAMETERS
