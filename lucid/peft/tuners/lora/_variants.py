r"""What an adapter does with its low-rank update: plain LoRA, or DoRA.

A *variant* is a strategy class a layer delegates to, one per adapter
(``_AdapterSpec.variant`` names it, so it is part of what ``lucid.compile``
keys a trace on).  Each registers itself under its name as it is defined
(``variant=`` in the class statement); the layers ask it three things and
never branch on which one it is:

* ``contribution`` — what the adapter adds to the base output of a forward
  pass;
* ``shift`` — how the adapter changes the base weight when it is merged,
  and the row gain to remember so the merge can be undone;
* ``attach`` — the state the variant keeps beside the factors.

DoRA (Liu et al., 2024, arXiv:2402.09353) splits a weight into a
magnitude and a direction: the adapted weight is
:math:`m\,(W + s\,BA)/\lVert W + s\,BA\rVert_c`, the norm taken per output
unit and treated as a constant for the gradient (section 4.3 of the paper),
with a trainable magnitude vector :math:`m` per adapter, initialised to
:math:`\lVert W + s\,BA\rVert_c`.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Final, override

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter

if TYPE_CHECKING:
    from lucid.peft.tuners.lora._layer import LoraLayer, _Residual

#: The per-adapter table DoRA keeps its magnitude vectors in (the reference's name).
MAGNITUDE_TABLE: Final = "lora_magnitude_vector"


class _Magnitude(nn.Module):
    r"""DoRA's trainable magnitude vector for one adapter: ``weight``, one value per output unit.

    ``....lora_magnitude_vector.<adapter>.weight`` in a state dict, as in
    the reference.  While the adapter is merged, ``merged_gain`` is the
    row gain :math:`m/\lVert\cdot\rVert` the merge used — what undoing
    the merge needs, since the norm it divided by is of the unmerged weight.
    """

    def __init__(self, magnitude: Tensor) -> None:
        super().__init__()
        self.weight = Parameter(magnitude)
        self.merged_gain: Tensor | None = None


class _Variant(ABC):
    """One way an adapter turns its low-rank update into a change of the layer's output and weight."""

    __slots__ = ()

    _registry: ClassVar[dict[str, _Variant]] = {}
    #: The registered name, and how a refusal names the feature.
    name: ClassVar[str]
    label: ClassVar[str]

    @override
    def __init_subclass__(cls, *, variant: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if variant is not None:
            cls.name = variant
            cls._registry[variant] = cls()

    @classmethod
    def named(cls, name: str) -> _Variant:
        """The variant registered as ``name`` (it is what ``_AdapterSpec.variant`` holds)."""
        return cls._registry[name]

    @classmethod
    def requested(cls, *, use_dora: bool) -> _Variant:
        """The variant the ``use_dora`` flag asks for."""
        return cls.named("dora" if use_dora else "lora")

    def check(self, layer: LoraLayer) -> None:
        """Raise ``ValueError`` if ``layer`` cannot host this variant."""
        if self.name not in layer._variant_names:
            raise ValueError(
                f"{type(layer).__name__} does not support {self.label}; "
                f"it takes {sorted(layer._variant_names)}"
            )

    def attach(self, layer: _Residual, adapter: str) -> None:
        """Create the state this variant keeps for ``adapter``, whose factors are initialised."""
        del layer, adapter  # plain LoRA keeps nothing

    @abstractmethod
    def contribution(self, layer: _Residual, adapter: str, x: Tensor, result: Tensor) -> Tensor:
        """What ``adapter`` adds to the base output ``result`` for the input ``x``."""

    @abstractmethod
    def shift(
        self, layer: _Residual, adapter: str, rows: Tensor, delta: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """``(change, gain)``: what merging adds to the weight rows ``rows``, and the row gain.

        ``delta`` is the adapter's update in the same row layout.  ``gain``
        is ``None`` when the change does not depend on the weight it is
        added to.
        """

    def remember(self, layer: _Residual, adapter: str, gain: Tensor | None) -> None:
        """Keep the gain a merge used, until the adapter is unmerged."""
        del layer, adapter, gain

    def recall(self, layer: _Residual, adapter: str) -> Tensor | None:
        """The gain a merged ``adapter`` used, or ``None`` for one whose merge has none."""
        del layer, adapter
        return None


class _Lora(_Variant, variant="lora"):
    r"""Plain LoRA: the update :math:`s\,BA\,x` is added, and :math:`s\,BA` merges into the weight."""

    __slots__ = ()
    label = "plain LoRA"

    @override
    def contribution(self, layer: _Residual, adapter: str, x: Tensor, result: Tensor) -> Tensor:
        return layer._lowrank(adapter, layer._dropped(adapter, x))

    @override
    def shift(
        self, layer: _Residual, adapter: str, rows: Tensor, delta: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        return delta, None


def _rows_view(vector: Tensor, rows: Tensor) -> Tensor:
    """``vector`` (one value per row) shaped to scale ``rows`` along its first axis."""
    return vector.reshape(-1, *([1] * (rows.ndim - 1)))


class _Dora(_Variant, variant="dora"):
    r"""DoRA: :math:`y = f(x) + (g - 1)\,(f(x) - b) + g\,s\,BA\,x` with :math:`g = m / \lVert W + sBA\rVert`.

    The first term of the sum is the base output; the rest turns
    :math:`W + sBA` into :math:`g\,(W + sBA)` row by row.  Merged, the
    weight is :math:`W + (g - 1)\,W + g\,sBA = g\,(W + sBA)` — the
    forward pass written as one weight, which is also what the reference
    merges for a single adapter.  The norm is read in the adapter's dtype
    and carries no gradient.
    """

    __slots__ = ()
    label = "DoRA (use_dora=True)"

    @override
    def check(self, layer: LoraLayer) -> None:
        super().check(layer)
        layer._check_weight_norm()

    @staticmethod
    def weight_norm(layer: _Residual, adapter: str, rows: Tensor | None = None) -> Tensor:
        r""":math:`\lVert W + s\,BA\rVert` per output unit, detached.

        ``rows`` is the weight in row layout to take :math:`W` from; the
        layer's own by default.
        """
        with lucid.no_grad():
            delta = layer._rows(layer.get_delta_weight(adapter))
            weight = layer._rows(layer.weight) if rows is None else rows
            total = (weight if weight.dtype == delta.dtype else weight.to(delta.dtype)) + delta
            return linalg.norm(total.reshape(total.shape[0], -1), dim=1)

    @override
    def attach(self, layer: _Residual, adapter: str) -> None:
        magnitude = self.weight_norm(layer, adapter)
        table: nn.ModuleDict[_Magnitude] = getattr(layer, MAGNITUDE_TABLE)
        table[adapter] = _Magnitude(magnitude)

    @staticmethod
    def _magnitude(layer: _Residual, adapter: str) -> _Magnitude:
        table: nn.ModuleDict[_Magnitude] = getattr(layer, MAGNITUDE_TABLE)
        return table[adapter]

    @override
    def contribution(self, layer: _Residual, adapter: str, x: Tensor, result: Tensor) -> Tensor:
        dropped = layer._dropped(adapter, x)
        # Dropout on the update's input also reaches the base term, as in the reference.
        seen = layer.base_layer(dropped) if layer._applies_dropout(adapter) else result
        gain = layer._output_vector(
            self._magnitude(layer, adapter).weight / self.weight_norm(layer, adapter)
        )
        return (gain - 1) * layer._without_bias(seen) + gain * layer._lowrank(adapter, dropped)

    @override
    def shift(
        self, layer: _Residual, adapter: str, rows: Tensor, delta: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        original = layer._original_rows(rows)
        with lucid.no_grad():
            norm = self.weight_norm(layer, adapter, original)
            gain = self._magnitude(layer, adapter).weight.detach() / norm
        column = _rows_view(gain, original)
        return (column - 1) * original + column * delta, gain

    @override
    def remember(self, layer: _Residual, adapter: str, gain: Tensor | None) -> None:
        self._magnitude(layer, adapter).merged_gain = gain

    @override
    def recall(self, layer: _Residual, adapter: str) -> Tensor | None:
        return self._magnitude(layer, adapter).merged_gain


__all__ = ["MAGNITUDE_TABLE", "_Dora", "_Lora", "_Magnitude", "_Variant", "_rows_view"]
