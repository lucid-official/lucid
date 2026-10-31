r"""Where a base layer keeps its weight: a plain parameter, or one a weight norm derives.

A merge reads the base weight and writes it back changed.  For most layers
the weight is a ``Parameter`` and both are one line, but a layer under
:func:`lucid.nn.utils.weight_norm` has no ``weight`` parameter: it holds a
direction ``weight_v`` and a magnitude ``weight_g``, and its ``weight`` is
recomputed from them by a hook before every forward.  A *site* is the
strategy that knows the difference — ``read`` the weight the layer computes
with, ``write`` a new one, and name the parameters a write touches — so a
merge never branches on how the weight is stored.  Each site registers
itself as it is defined, and :meth:`_WeightSite.of` asks them which holds a
given module.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.nn.utils.weight_norm import WeightNorm, _compute_weight, _norm_along_dim

_WEIGHT: Final = "weight"
_MAGNITUDE: Final = "weight_g"
_DIRECTION: Final = "weight_v"


class _WeightSite(ABC):
    """The base weight of one module, however the module stores it."""

    __slots__ = ("module",)

    _kinds: ClassVar[list[type[_WeightSite]]] = []

    def __init__(self, module: nn.Module) -> None:
        self.module = module

    @override
    def __init_subclass__(cls, *, site: bool = False, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if site:
            cls._kinds.append(cls)

    @classmethod
    @abstractmethod
    def holds(cls, module: nn.Module) -> bool:
        """Whether ``module`` stores its weight the way this site reads it."""

    @classmethod
    def of(cls, module: nn.Module) -> _WeightSite:
        """The site of ``module``'s weight; ``AttributeError`` if it keeps none."""
        for kind in cls._kinds:
            if kind.holds(module):
                return kind(module)
        raise AttributeError(f"{type(module).__name__} has no weight parameter")

    @abstractmethod
    def parameters(self) -> dict[str, Parameter]:
        """The parameters a ``write`` changes, by their name under the module."""

    @abstractmethod
    def read(self) -> Tensor:
        """The weight the module computes with now."""

    @abstractmethod
    def write(self, value: Tensor) -> None:
        """Make the module compute with ``value`` as its weight, in place and without autograd."""


class _Stored(_WeightSite, site=True):
    """A weight held as a ``Parameter``: read it, and write into it."""

    __slots__ = ()

    @classmethod
    @override
    def holds(cls, module: nn.Module) -> bool:
        return isinstance(getattr(module, _WEIGHT, None), Parameter)

    def _parameter(self) -> Parameter:
        weight = getattr(self.module, _WEIGHT)
        assert isinstance(weight, Parameter)
        return weight

    @override
    def parameters(self) -> dict[str, Parameter]:
        return {_WEIGHT: self._parameter()}

    @override
    def read(self) -> Tensor:
        return self._parameter()

    @override
    def write(self, value: Tensor) -> None:
        self._parameter().copy_(value)


class _Normalised(_WeightSite, site=True):
    r"""A weight a weight norm derives: :math:`W = g\,v/\lVert v\rVert` along ``dim``.

    A write finds the :math:`(g, v)` that derive the new weight :math:`W'`:
    :math:`g' = \lVert W'\rVert` and :math:`v'` along :math:`W'`, kept at the
    norm :math:`v` had, so that writing back the weight a merge started from
    restores :math:`g` and :math:`v` themselves (up to rounding) and not
    merely a pair that computes the same weight.  A slice whose new weight
    is zero keeps its :math:`v`, and :math:`g' = 0` zeroes it.
    """

    __slots__ = ()

    @classmethod
    @override
    def holds(cls, module: nn.Module) -> bool:
        return WeightNorm.find(module, _WEIGHT) is not None and isinstance(
            getattr(module, _MAGNITUDE, None), Parameter
        )

    def _pair(self) -> tuple[Parameter, Parameter]:
        magnitude, direction = getattr(self.module, _MAGNITUDE), getattr(self.module, _DIRECTION)
        assert isinstance(magnitude, Parameter) and isinstance(direction, Parameter)
        return magnitude, direction

    def _dim(self) -> int:
        hook = WeightNorm.find(self.module, _WEIGHT)
        assert hook is not None
        return hook.dim

    @override
    def parameters(self) -> dict[str, Parameter]:
        magnitude, direction = self._pair()
        return {_MAGNITUDE: magnitude, _DIRECTION: direction}

    @override
    def read(self) -> Tensor:
        magnitude, direction = self._pair()
        return _compute_weight(magnitude, direction, self._dim())

    @override
    def write(self, value: Tensor) -> None:
        magnitude, direction = self._pair()
        dim = self._dim()
        value = value if value.dtype == direction.dtype else value.to(direction.dtype)
        norm = _norm_along_dim(value, dim)
        kept = _norm_along_dim(direction, dim)
        positive = norm > 0
        safe = lucid.where(positive, norm, lucid.ones_like(norm))
        direction.copy_(lucid.where(positive, value * (kept / safe), direction))
        magnitude.copy_(norm)
        # The hook that recomputes the weight runs only on a forward: refresh what a read sees.
        object.__setattr__(self.module, _WEIGHT, self.read())
