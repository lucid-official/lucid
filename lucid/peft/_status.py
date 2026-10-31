r"""What a model's adapters are doing: enabled, active, merged, trainable and where they live.

``TunerLayerStatus`` reports one adapter slot (a tuner layer or a
``modules_to_save`` module); ``TunerModelStatus`` folds them into one record
for the model.  Where the layers disagree — some enabled and some not, one
adapter trainable in one layer and frozen in another — a model-level field
reads ``"irregular"``: the model is in a state no entry point produces, and
it may not compute what the caller expects.

The classes are plain frozen records; the functions that build them
(``get_layer_status`` / ``get_model_status``) live with ``PeftModel``.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import lucid.nn as nn
from lucid.nn.parameter import Parameter
from lucid.peft._slots import _AdapterSlot, _adapter_slots

#: What a model-level field reads when its layers do not agree.
IRREGULAR: Final = "irregular"

type Irregular = Literal["irregular"]
type GradState = bool | Irregular


def _grad_state(params: Sequence[Parameter]) -> GradState:
    """``True`` if all of ``params`` require grad, ``False`` if none, else ``"irregular"``."""
    flags = {param.requires_grad for param in params}
    if flags <= {True}:
        return True  # also an adapter with no parameter here: nothing is frozen
    return False if flags == {False} else IRREGULAR


def _consensus[T](values: Sequence[T]) -> T | Irregular:
    """The one value all of ``values`` equal, or ``"irregular"`` if they differ."""
    first = values[0]
    return first if all(value == first for value in values) else IRREGULAR


def _parameter_counts(module: nn.Module) -> tuple[int, int]:
    """``(trainable, total)`` parameter element counts of ``module``, each shared parameter once."""
    trainable = total = 0
    for _, param in module.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


@dataclass(frozen=True, slots=True)
class TunerLayerStatus:
    """The state of one adapter layer.

    Attributes
    ----------
    name : str
        The layer's dotted name under the model, e.g. ``"model.encoder.q_proj"``.
    module_type : str
        The layer's class as it prints, e.g. ``"lora.Linear"``.
    enabled : bool
        Whether the layer applies adapters (it runs the base alone when not).
    active_adapters : list of str
        The adapters the forward pass applies.
    merged_adapters : list of str
        The adapters folded into the base weight.
    requires_grad : dict
        Per adapter, whether all its parameters here train (``True``), none
        do (``False``) or some do (``"irregular"``).
    available_adapters : list of str
        Every adapter the layer holds, sorted.
    devices : dict
        Per adapter, the device types its parameters are on, sorted.
    """

    name: str
    module_type: str
    enabled: bool
    active_adapters: list[str]
    merged_adapters: list[str]
    requires_grad: dict[str, GradState]
    available_adapters: list[str]
    devices: dict[str, list[str]]

    @classmethod
    def of(cls, name: str, slot: _AdapterSlot) -> TunerLayerStatus:
        """The status of ``slot``, found at ``name``."""
        state = slot._state()
        available = sorted(slot._adapter_names())
        parameters = {adapter: slot._adapter_parameters(adapter) for adapter in available}
        return cls(
            name=name,
            module_type=repr(slot).partition("(")[0],
            enabled=state.enabled,
            active_adapters=state.active,
            merged_adapters=state.merged,
            requires_grad={adapter: _grad_state(params) for adapter, params in parameters.items()},
            available_adapters=available,
            devices={
                adapter: sorted({param.device.type for param in params})
                for adapter, params in parameters.items()
            },
        )

    @classmethod
    def all_of(cls, root: nn.Module) -> list[TunerLayerStatus]:
        """The status of every adapter slot under ``root``, in module order.

        Raises
        ------
        ValueError
            ``root`` holds no adapter layer.
        """
        found = [cls.of(name, slot) for name, slot in _adapter_slots(root)]
        if not found:
            raise ValueError(
                "No adapter layers found in the model, please ensure that it's a PEFT model or "
                "that you have PEFT adapters injected in the model."
            )
        return found


def _merged_across(layers: Sequence[TunerLayerStatus]) -> list[str] | Irregular:
    """The adapters merged on some layer, or ``"irregular"`` if one of them is unmerged on another that holds it."""
    merged = {name for layer in layers for name in layer.merged_adapters}
    for layer in layers:
        # An adapter merged somewhere that is still a separate update here.
        if (set(layer.available_adapters) - set(layer.merged_adapters)) & merged:
            return IRREGULAR
    return sorted(merged)


@dataclass(frozen=True, slots=True)
class TunerModelStatus:
    """The state of the adapters of a whole model, folded from its layers.

    Attributes
    ----------
    base_model_type : str
        The class name of the model that was adapted (``"other"`` for a
        model not wrapped in a ``PeftModel``).
    adapter_model_type : str
        The class name of the tuner (``"LoraModel"``), ``"None"`` when unwrapped.
    peft_types : dict
        Each adapter's method by name, e.g. ``{"default": "LORA"}``.
    trainable_params, total_params : int
        Parameter element counts, each shared parameter once.
    num_adapter_layers : int
        How many adapter layers the model holds.
    enabled : bool or "irregular"
        Whether every layer applies adapters.
    active_adapters : list of str or "irregular"
        The adapters in use, if every layer agrees.
    merged_adapters : list of str or "irregular"
        The adapters merged into the base, unless one is merged on a layer
        and not on another that holds it.
    requires_grad : dict
        Per adapter, whether its parameters train in every layer (``True``),
        in none (``False``) or in some (``"irregular"``).
    available_adapters : list of str
        Every adapter any layer holds, sorted.
    devices : dict
        Per adapter, the device types its parameters are on, sorted.
    """

    base_model_type: str
    adapter_model_type: str
    peft_types: dict[str, str]
    trainable_params: int
    total_params: int
    num_adapter_layers: int
    enabled: bool | Irregular
    active_adapters: list[str] | Irregular
    merged_adapters: list[str] | Irregular
    requires_grad: dict[str, GradState]
    available_adapters: list[str]
    devices: dict[str, list[str]]

    @classmethod
    def of(
        cls,
        layers: Sequence[TunerLayerStatus],
        *,
        base_model_type: str,
        adapter_model_type: str,
        peft_types: dict[str, str],
        parameter_counts: tuple[int, int],
    ) -> TunerModelStatus:
        """The model's status from its ``layers`` (at least one) and what the wrapper knows."""
        grads: dict[str, list[GradState]] = defaultdict(list)
        devices: dict[str, set[str]] = defaultdict(set)
        for layer in layers:
            for adapter, state in layer.requires_grad.items():
                grads[adapter].append(state)
            for adapter, kinds in layer.devices.items():
                devices[adapter].update(kinds)
        trainable, total = parameter_counts
        return cls(
            base_model_type=base_model_type,
            adapter_model_type=adapter_model_type,
            peft_types=peft_types,
            trainable_params=trainable,
            total_params=total,
            num_adapter_layers=len(layers),
            enabled=_consensus([layer.enabled for layer in layers]),
            active_adapters=_consensus([layer.active_adapters for layer in layers]),
            merged_adapters=_merged_across(layers),
            requires_grad={adapter: _consensus(states) for adapter, states in grads.items()},
            available_adapters=sorted({a for layer in layers for a in layer.available_adapters}),
            devices={adapter: sorted(kinds) for adapter, kinds in devices.items()},
        )
