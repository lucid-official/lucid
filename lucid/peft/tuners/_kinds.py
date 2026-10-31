"""The kinds of module a LyCORIS adapter (LoHa, LoKr) can wrap, and what each means to the adapter.

An adapter of that family is a weight-shaped update :math:`\\Delta W` that
the layer applies to the input *as the base layer would*: a matrix product
for ``nn.Linear``, a convolution with the base's stride, padding, dilation
and groups for ``nn.Conv1d`` / ``nn.Conv2d``.  A ``_Kind`` is the strategy
for one module type — its geometry (what the update's factors must add up
to) and how to apply a weight to an input — registered under that type as it
is defined, so the layers branch on nothing and a new kind is one subclass.
"""

import math
from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.functional.conv import _ConvArgs, _convnd


class _Geometry(NamedTuple):
    """The size of a base weight, ``(out, in_per_group, *kernel)``; ``kernel`` is empty for a linear layer."""

    out: int
    inn: int
    kernel: tuple[int, ...]

    @property
    def weight_shape(self) -> tuple[int, ...]:
        """The base weight's shape."""
        return (self.out, self.inn, *self.kernel)

    @property
    def flat_in(self) -> int:
        """Input units per output unit: ``in_per_group`` times the kernel's volume."""
        return self.inn * math.prod(self.kernel)

    @property
    def has_kernel(self) -> bool:
        """Whether the weight has spatial dimensions."""
        return bool(self.kernel)

    @property
    def pointwise(self) -> bool:
        """Whether every spatial extent is 1 (a convolution that is a matrix product)."""
        return all(k == 1 for k in self.kernel)


def _expect[T: nn.Module](kind: type[T], module: nn.Module) -> T:
    if not isinstance(module, kind):
        raise TypeError(f"expected an nn.{kind.__name__}, got {type(module).__name__}")
    return module


class _Kind(ABC):
    """How an adapter of the LyCORIS family reads and applies one module type."""

    __slots__ = ()

    _kinds: ClassVar[dict[type[nn.Module], _Kind]] = {}

    @override
    def __init_subclass__(cls, *, base_type: type[nn.Module] | None = None) -> None:
        super().__init_subclass__()
        if base_type is not None:
            _Kind._kinds[base_type] = cls()

    @staticmethod
    def of(base_type: type[nn.Module]) -> _Kind:
        """The kind registered for ``base_type``."""
        try:
            return _Kind._kinds[base_type]
        except KeyError:
            raise TypeError(f"no LyCORIS kind for {base_type.__name__}") from None

    @abstractmethod
    def geometry(self, base: nn.Module) -> _Geometry:
        """The size of ``base``'s weight."""

    @abstractmethod
    def apply(self, base: nn.Module, x: Tensor, weight: Tensor) -> Tensor:
        """``x`` through ``weight`` the way ``base`` applies its own (no bias)."""


class _LinearKind(_Kind, base_type=nn.Linear):
    @override
    def geometry(self, base: nn.Module) -> _Geometry:
        layer = _expect(nn.Linear, base)
        return _Geometry(layer.out_features, layer.in_features, ())

    @override
    def apply(self, base: nn.Module, x: Tensor, weight: Tensor) -> Tensor:
        del base  # a matrix product has no geometry of its own
        return x @ weight.mT


class _ConvKind(_Kind):
    """The shared half of the convolutions: the base's geometry and its convolution arguments."""

    __slots__ = ()

    @staticmethod
    @abstractmethod
    def _conv(base: nn.Module) -> nn.Conv1d | nn.Conv2d:
        """``base`` as the convolution class this kind wraps."""

    @staticmethod
    def _ints(value: int | tuple[int, ...]) -> tuple[int, ...]:
        return (value,) if isinstance(value, int) else value

    @override
    def geometry(self, base: nn.Module) -> _Geometry:
        conv = self._conv(base)
        return _Geometry(
            conv.out_channels, conv.in_channels // conv.groups, self._ints(conv.kernel_size)
        )

    @override
    def apply(self, base: nn.Module, x: Tensor, weight: Tensor) -> Tensor:
        conv = self._conv(base)
        args = _ConvArgs(conv.stride, conv._padding_str or conv.padding, conv.dilation, conv.groups)
        return _convnd(x, weight, None, args, weight.ndim - 2, conv.padding_mode)


class _Conv1dKind(_ConvKind, base_type=nn.Conv1d):
    @staticmethod
    @override
    def _conv(base: nn.Module) -> nn.Conv1d:
        return _expect(nn.Conv1d, base)


class _Conv2dKind(_ConvKind, base_type=nn.Conv2d):
    @staticmethod
    @override
    def _conv(base: nn.Module) -> nn.Conv2d:
        return _expect(nn.Conv2d, base)


def _tucker(core: Tensor, rows: Tensor, columns: Tensor) -> Tensor:
    r"""Contract a ``(r, r, *k)`` core with ``(r, p)`` and ``(r, q)`` factors into a ``(p, q, *k)`` kernel.

    :math:`K_{pqkl} = \sum_{ij} T_{ijkl}\,U_{ip}\,V_{jq}` — the "effective
    convolution" of FedPara (Proposition 3): a small core carries the
    spatial structure and two matrices carry the channels.  Written as two
    matrix products, so it needs nothing beyond ``matmul``.
    """
    r = core.shape[0]
    spatial = tuple(core.shape[2:])
    taken = rows.mT @ core.reshape(r, -1)  # (p, r * prod(spatial))
    taken = taken.reshape(rows.shape[1], r, *spatial)  # (p, j, *spatial)
    moved = taken.movedim(1, -1)  # (p, *spatial, j)
    return (moved @ columns).movedim(-1, 1)  # (p, q, *spatial)


# The kinds register themselves as they are defined; naming them here is how the dead-code
# gate sees that they are used.
__all__ = ["_Conv1dKind", "_Conv2dKind", "_Geometry", "_Kind", "_LinearKind", "_tucker"]
