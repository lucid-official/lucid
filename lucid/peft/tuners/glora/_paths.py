r"""The support tensors of GLoRA: each is a low-rank product, a vector, a scalar or nothing.

GLoRA (Chavan et al., 2023, arXiv:2306.07967, section 2.2, Eq. 11) corrects a layer with five
support tensors :math:`A, B, C, D, E`, each drawn from its own search space.  A tensor is a
``_Path``: a module whose call gives the tensor, of one of four kinds registered as they are
defined (``kind=``), so a layer builds the five from the names the config gives and knows nothing
of the kinds.  The parameters carry the reference's names (``Xd`` and ``Xu`` of a low-rank
product, ``X`` of a vector or a scalar, the buffer ``zeros`` of an absent tensor), which an
adapter file keeps.
"""

import math
from abc import ABC, abstractmethod
from typing import ClassVar, override

import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._delta import _Home


class _Path(nn.Module, ABC):
    """One support tensor: a module that, called, gives the tensor of a given shape."""

    _kinds: ClassVar[dict[str, type[_Path]]] = {}

    @override
    def __init_subclass__(cls, *, kind: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if kind is not None:
            _Path._kinds[kind] = cls

    @staticmethod
    def build(
        kind: str, shape: tuple[int, ...], *, column: bool, r: int, home: _Home, init_weights: bool
    ) -> _Path:
        """The support tensor of ``kind`` standing for a ``shape`` tensor.

        ``column`` says a vector is a ``(m, 1)`` column (it multiplies rows of a matrix) rather than a
        flat ``(m,)`` one (it adds to a bias); ``init_weights`` the identity start.
        """
        return _Path._kinds[kind].create(shape, column, r, home, init_weights)

    @classmethod
    @abstractmethod
    def create(
        cls, shape: tuple[int, ...], column: bool, r: int, home: _Home, init_weights: bool
    ) -> _Path:
        """Allocate the path and give it its starting values."""

    @override
    @abstractmethod
    def forward(self) -> Tensor:
        """The tensor this path stands for."""


class _LoraPath(_Path, kind="lora"):
    r""":math:`X_d X_u`, with :math:`X_d \in \mathbb{R}^{m \times r}` and :math:`X_u \in \mathbb{R}^{r \times n}`."""

    def __init__(self, down: Tensor, up: Tensor) -> None:
        super().__init__()
        self.Xd = Parameter(down)
        self.Xu = Parameter(up)

    @override
    @classmethod
    def create(
        cls, shape: tuple[int, ...], column: bool, r: int, home: _Home, init_weights: bool
    ) -> _Path:
        del column  # a product is a matrix whatever the slot
        rows, cols = shape
        down, up = home.zeros(rows, r), home.zeros(r, cols)
        init.kaiming_uniform_(up, a=math.sqrt(5))
        if not init_weights:
            init.kaiming_uniform_(down, a=math.sqrt(5))
        return cls(down, up)

    @override
    def forward(self) -> Tensor:
        product: Tensor = self.Xd @ self.Xu
        return product


class _VectorPath(_Path, kind="vector"):
    """One value per row: a ``(m, 1)`` column, or a flat ``(m,)`` vector."""

    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.X = Parameter(values)

    @override
    @classmethod
    def create(
        cls, shape: tuple[int, ...], column: bool, r: int, home: _Home, init_weights: bool
    ) -> _Path:
        del r  # a vector has no rank
        rows = shape[0]
        values = home.zeros(rows, 1) if column else home.zeros(1, rows)
        if not init_weights:
            init.kaiming_uniform_(values, a=math.sqrt(5))
        return cls(values if column else values.reshape(rows))

    @override
    def forward(self) -> Tensor:
        return self.X


class _ScalarPath(_Path, kind="constant"):
    """One value for the whole tensor."""

    def __init__(self, value: Tensor) -> None:
        super().__init__()
        self.X = Parameter(value)

    @override
    @classmethod
    def create(
        cls, shape: tuple[int, ...], column: bool, r: int, home: _Home, init_weights: bool
    ) -> _Path:
        del shape, column, r  # one value, wherever it goes
        value = home.zeros(1)
        if not init_weights:
            init.normal_(value, mean=0.0, std=0.02)
        return cls(value)

    @override
    def forward(self) -> Tensor:
        return self.X[0]


class _AbsentPath(_Path, kind="none"):
    """Nothing: zeros of the tensor's shape, held as a buffer (which the reference's file holds too)."""

    zeros: Tensor

    def __init__(self, zeros: Tensor) -> None:
        super().__init__()
        self.register_buffer("zeros", zeros)

    @override
    @classmethod
    def create(
        cls, shape: tuple[int, ...], column: bool, r: int, home: _Home, init_weights: bool
    ) -> _Path:
        del column, r, init_weights  # nothing to start
        return cls(home.zeros(*shape))

    @override
    def forward(self) -> Tensor:
        return self.zeros


# The kinds register themselves as they are defined; naming them here is how the dead-code gate
# sees that they are used.
__all__ = ["_AbsentPath", "_LoraPath", "_Path", "_ScalarPath", "_VectorPath"]
