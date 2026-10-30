"""LoRA layers that change the weights the base layer reads, not its output.

A module that reads a raw parameter inside its own forward (an attention
module's fused ``in_proj_weight``, a mixture-of-experts weight stack) has
no ``nn.Linear`` to put a low-rank branch beside.  Its adapter is a change
of the weight instead: for the duration of the forward call the layer
makes the base module read :math:`W + \\sum_a s_a B_a A_a` in place of
:math:`W`, so the product reaches the output and the gradient reaches the
factors through the sum.  The module's registered parameter is not touched.
"""

from abc import abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._forwarding import _as_module, _ForwardsAttributes
from lucid.peft.tuners.lora._layer import LoraLayer


@contextmanager
def _shadowed(holder: nn.Module, values: Mapping[str, Tensor]) -> Iterator[None]:
    """Make ``holder.<name>`` read ``values[name]`` inside the block.

    The value goes into the instance ``__dict__``, which attribute lookup
    consults before the parameter registry: the registered ``Parameter``
    stays where it is (its identity, ``requires_grad`` and place in
    ``state_dict`` untouched, and no module-tree version bumped — a
    compiled model is not made to recompile by every forward), and the
    shadow is gone when the block exits, however it exits.
    """
    entered: list[str] = []
    try:
        for name, value in values.items():
            object.__setattr__(holder, name, value)
            entered.append(name)
        yield
    finally:
        for name in entered:
            del holder.__dict__[name]


class _WeightSpace[Out](LoraLayer, _ForwardsAttributes):
    r"""A LoRA layer whose forward runs the base module on shifted weights.

    A subclass names the parameters the adapter changes (``_weights``,
    by their attribute name on the module that reads them) and the change
    one adapter makes to each (``_delta_for``).  The forward pass sums the
    active adapters' changes into each weight, in the weight's dtype, and
    calls the base module with those in place; merging writes the same sums
    into the registered parameters.  ``Out`` is what the base module's
    forward returns.  The base module's public attributes read through the
    layer (``embed_dim``, ``num_heads``, the other parameters …), since it
    stands where the module was.
    """

    @override
    def _wrapped(self) -> nn.Module | None:
        modules = self.__dict__.get("_modules")
        return _as_module(modules.get("base_layer")) if modules else None

    @abstractmethod
    def _weights(self) -> dict[str, Parameter]:
        """The base parameters the adapter changes, by attribute name on the module that reads them."""

    @abstractmethod
    def _delta_for(self, name: str, adapter: str) -> Tensor:
        """The change ``adapter`` makes to the weight ``name``, shaped like it."""

    def _path(self, name: str) -> str:
        return f"{self._base_path()}.{name}"

    @override
    def _anchor(self) -> Tensor:
        return next(iter(self._weights().values()))

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        return {self._path(name): param for name, param in self._weights().items()}

    @override
    def _deltas(self, adapter: str) -> dict[str, Tensor]:
        return {self._path(name): self._delta_for(name, adapter) for name in self._weights()}

    @override
    def forward(self, *args: object, **kwargs: object) -> Out:
        result: Out
        applied = [name for name in self.active_adapters if name in self._specs]
        if self.disable_adapters or self.merged or not applied:
            result = self.base_layer(*args, **kwargs)
            return result
        shifted: dict[str, Tensor] = {}
        for name, param in self._weights().items():
            value: Tensor = param
            for adapter in applied:
                value = value + self._delta_for(name, adapter)
            shifted[name] = value if value.dtype == param.dtype else value.to(param.dtype)
        with _shadowed(self.get_base_layer(), shifted):
            result = self.base_layer(*args, **kwargs)
        return result
