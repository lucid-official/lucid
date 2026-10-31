"""Pieces the layer and tuner families share: spec views, dtype, additive weight updates, ``fan_in_fan_out``."""

import warnings
from abc import abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Protocol, Self, overload, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import DTypeLike
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._base import _BaseTunerLayer

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


class _AdditiveWeight(_BaseTunerLayer):
    """A layer whose update is a weight-shaped change added to the base weight.

    A subclass gives ``get_delta_weight``, the change.  Merging adds it to the
    base weight in place and unmerging subtracts it.
    """

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        return {_WEIGHT: self.weight}

    @override
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        weight = self.weight
        return {_WEIGHT: (weight + self.get_delta_weight(adapter)).to(weight.dtype)}

    @override
    def _fold(self, adapter: str) -> None:
        self.weight.copy_(self._merged_values(adapter)[_WEIGHT])

    @override
    def _unfold(self) -> None:
        weight = self.weight
        for adapter in reversed(self.merged_adapters):
            weight.copy_((weight - self.get_delta_weight(adapter)).to(weight.dtype))


class _ResidualUpdate(_AdditiveWeight):
    """An additive layer whose forward pass is the base output plus each active adapter's contribution.

    A subclass keeps its per-adapter scalars in ``_specs`` (an adapter absent
    from it is not applied) and gives ``_contribution``, the update applied to
    an input.
    """

    _specs: Mapping[str, tuple[object, ...]]

    @abstractmethod
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        """The adapter's addition to the base output for the input ``x``."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.base_layer(x)
        if self.disable_adapters or self.merged:
            return result
        out = result
        for adapter in self.active_adapters:
            if adapter in self._specs:  # an adapter aimed at other layers: this one runs its base
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
