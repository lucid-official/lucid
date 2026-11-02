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
from lucid.peft.tuners._layer import _BaseTunerLayer, _check_adapter_name
from lucid.peft.tuners._plan import _NO_PARAMETERS, _Target
from lucid.peft.tuners._tuner import _BaseTuner

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
        """The base layer's output plus every active adapter's contribution to it.

        The base output alone when adapters are disabled or merged; the sum is cast back to the
        base output's dtype.
        """
        result: Tensor = self.base_layer(x)
        if self.disable_adapters or self.merged:
            return result
        out = result
        held = self._adapter_names()
        for adapter in self.active_adapters:
            if adapter in held:  # an adapter aimed at other layers: this one runs its base
                out = out + self._contribution(adapter, x)
        return out if out.dtype == result.dtype else out.to(result.dtype)


class _Unmergeable(_BaseTunerLayer):
    """A layer whose adapter computes something no change of the base weights can hold.

    An adaption prompt attends to learned keys, a skill router mixes factors per input: folding
    either into the weight would be a different function.  The layer owns no merge target and
    ``merge`` says why it cannot.
    """

    #: What the adapter is, for the refusal's message ("an adaption prompt").
    _unmergeable_because: ClassVar[str] = "this adapter"

    def _refusal(self) -> NotImplementedError:
        return NotImplementedError(
            f"{self._unmergeable_because} cannot be folded into the base weights; "
            "keep the adapter, or unload() it"
        )

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        """Refused: there is no weight-shaped change to report."""
        del adapter
        raise self._refusal()

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        return {}

    @override
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        del adapter
        raise self._refusal()

    @override
    def _fold(self, adapter: str) -> None:
        del adapter
        raise self._refusal()

    @override
    def _unfold(self) -> None:
        raise self._refusal()

    @override
    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """Refused with ``NotImplementedError``: the adapter has no weight form."""
        del safe_merge, adapter_names
        raise self._refusal()


class _LinearSized(_BaseTunerLayer):
    """A layer around an ``nn.Linear``-shaped weight, which reads the base's sizes off it."""

    @property
    def in_features(self) -> int:
        """Input features of the base layer."""
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        """Output features of the base layer."""
        return int(self.weight.shape[0])


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


class _ConfigLayer[C: _TargetedConfig = _TargetedConfig](_BaseTunerLayer):
    """A layer built from the module, an adapter name and the method's config alone.

    The shape OFT, BOFT, HRA and LN tuning share: a family root names its
    per-adapter tables (``adapter_layer_names``, all ``ParameterDict`` or all
    ``ModuleDict`` by ``_table``), creates an adapter in ``update_layer`` and
    keeps its scalar state in ``_specs`` — a tuple per adapter, so the scalars
    ``lucid.compile`` keys a trace on include them.

    ``C`` is the config class of the method: a layer that parametrises the base
    (``_ConfigLayer[OftConfig]``) receives its config typed, with no cast.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : C
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

    def __init__(self, base_layer: nn.Module, adapter_name: str, config: C) -> None:
        """Wrap ``base_layer`` and build the adapter ``adapter_name`` on it as ``config`` says."""
        super().__init__(base_layer)
        self._specs = {}
        for table in self.adapter_layer_names:
            setattr(self, table, self._new_table())
        self._prepare(config)
        self.update_layer(adapter_name, config)

    def _prepare(self, config: C) -> None:
        """Make the state a layer keeps beside its per-adapter tables, before the first adapter is entered.

        Dropout tables, references to tables the tuner shares: whatever ``_enter`` reads that is
        not a ``ParameterDict`` of ``adapter_layer_names``.
        """
        del config  # most layers have only their tables

    def _new_table(self) -> nn.ParameterDict | nn.ModuleDict[nn.Module]:
        """An empty per-adapter table: bare parameters by default, modules where an adapter is one (LN tuning)."""
        return nn.ParameterDict()

    def update_layer(self, adapter_name: str, config: C) -> None:
        """Create adapter ``adapter_name`` from ``config``.

        The adapter is built whole before the layer holds any of it.  A layer
        that holds an adapter already keeps applying the adapters it applied:
        the new one is inactive until ``set_adapter`` names it; a layer with
        none starts applying this one.

        Parameters
        ----------
        adapter_name : str
            Key of the adapter in each of the layer's tables.
        config : C
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
    def _enter(self, adapter_name: str, config: C) -> None:
        """Allocate the adapter's tensors, then put them (and its spec) in the tables; the name is free."""

    @override
    def __repr__(self) -> str:
        """The module's repr, led by the method's prefix so a printed model names it."""
        return type(self)._repr_prefix + super().__repr__()

    @override
    def extra_repr(self) -> str:
        """The active adapter, whether it is merged and whether adapters are disabled."""
        return (
            f"adapter={self.active_adapter!r}, merged={self.merged}, "
            f"disabled={self.disable_adapters}"
        )

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its tables, through the base, and its recorded settings."""
        super().delete_adapter(adapter_name)
        self._specs.pop(adapter_name, None)


class _ConfigTuner[C: _TargetedConfig = _TargetedConfig](_BaseTuner[C]):
    """A tuner whose layers are built by ``layer(module, adapter, config)`` and extended by ``layer.update_layer(adapter, config)``."""

    @override
    @classmethod
    def _create_new_module(cls, config: C, adapter_name: str, target: _Target) -> nn.Module:
        layer = target.layer
        assert layer is not None and issubclass(layer, _ConfigLayer)
        return layer(target.module, adapter_name, config)

    @override
    @classmethod
    def _add_to_layer(cls, config: C, adapter_name: str, target: _Target) -> None:
        layer = target.module
        assert isinstance(layer, _ConfigLayer)
        layer.update_layer(adapter_name, config)

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        del model, config  # these methods train their own tensors only
        return []

    _default_parameters: ClassVar[Mapping[str, frozenset[str]]] = _NO_PARAMETERS
