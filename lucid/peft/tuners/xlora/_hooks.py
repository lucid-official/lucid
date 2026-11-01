"""The forward hook that makes a LoRA layer an X-LoRA layer (private)."""

from collections.abc import Mapping
from typing import Protocol

from lucid._tensor.tensor import Tensor
from lucid._types import _ModuleOutput
from lucid.peft.tuners.lora import Linear
from lucid.peft.tuners.xlora._classifier import _layer_scalings
from lucid.peft.tuners.xlora._config import XLoraConfig


class _Mixer(Protocol):
    """What a layer's hook asks of the tuner it belongs to."""

    #: The experts' adapter names, in the order of the classifier's outputs.
    experts: list[str]
    #: Whether X-LoRA is switched off (the layers then run their base alone).
    disabled: bool
    #: The scalings the current call mixes with, ``None`` outside a call.
    scalings: Tensor | None

    @property
    def xlora_config(self) -> XLoraConfig:
        """The stored config, whose ``global_scaling_weight`` and ``top_k_lora`` are read each call."""
        ...


class _ScaledExperts:
    r"""Adds to a LoRA layer's base output each expert's update on the input scaled by that expert's weight.

    The layer itself is kept at its base module (its adapters are switched off), so the output
    this hook sees is the base layer's: :math:`y = f(x) + g\sum_k u_k(s_k \odot x)`, with
    :math:`s_k` the weight of expert :math:`k` for each token, :math:`u_k` its (scaled) low-rank
    update and :math:`g` the config's ``global_scaling_weight``.  An expert the layer holds no
    adapter for contributes nothing.
    """

    __slots__ = ("_layer", "_mixer")

    def __init__(self, mixer: _Mixer, layer_number: int) -> None:
        self._mixer = mixer
        self._layer = layer_number

    def __call__(
        self,
        module: object,
        args: tuple[Tensor, ...],
        kwargs: Mapping[str, object],
        output: _ModuleOutput,
    ) -> _ModuleOutput:
        """``output`` with the experts' scaled updates of the call's input added."""
        mixer = self._mixer
        scalings = mixer.scalings
        if scalings is None or mixer.disabled or not isinstance(output, Tensor):
            return output
        assert isinstance(module, Linear)  # the tuner hooks nothing else
        x = args[0]
        if x.ndim != 3:
            raise ValueError(
                "X-LoRA weighs each token of an adapted layer's input, "
                f"which must be (batch, tokens, features); got {tuple(x.shape)}"
            )
        config = mixer.xlora_config
        weights = _layer_scalings(config, scalings, self._layer)
        extras = module._extras_of(args, kwargs)
        held = module._adapter_names()
        out = output
        for k, name in enumerate(mixer.experts):
            if name not in held:
                continue
            scaled = x * weights[:, :, k : k + 1].to(x.dtype)
            update = module._lowrank(name, module._dropped(name, scaled), extras)
            out = out + update.to(output.dtype) * config.global_scaling_weight
        return out
