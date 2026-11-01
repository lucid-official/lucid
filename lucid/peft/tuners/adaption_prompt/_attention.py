"""How an adaption prompt reads one family's attention module (private).

An adaption prompt needs three things of the attention it wraps: the keys and values of the
learned prompt (the module's own projections applied to the prompt), the queries of the layer's
input (the module does not return them), and where the result goes back into the layer's output.
Families differ in all three, so each is a strategy, registered by the ``model_type`` strings it
serves (``class _FusedQKV(_AttentionKind, model_types=("gpt", "gpt2"))``).  The layer asks the
strategy; neither the layer nor the tuner names a family.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._target_defaults import _model_type


class _AttentionKind(ABC):
    """One family's attention module, as an adaption prompt needs it."""

    _kinds: ClassVar[dict[str, type[_AttentionKind]]] = {}
    #: The name of the attention module inside a decoder block.
    target_module: ClassVar[str]

    @override
    def __init_subclass__(cls, *, model_types: tuple[str, ...] = (), **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for model_type in model_types:
            _AttentionKind._kinds[model_type] = cls

    @staticmethod
    def for_model(model: nn.Module) -> type[_AttentionKind]:
        """The strategy for ``model``'s type; ``ValueError`` naming the supported types if it has none."""
        model_type = _model_type(model)
        kind = _AttentionKind._kinds.get(model_type)
        if kind is None:
            raise ValueError(
                f"Unsupported model type for adaption prompt: '{model_type}' "
                f"(the decoders it supports are {', '.join(sorted(_AttentionKind._kinds))})."
            )
        return kind

    @staticmethod
    def of(attention: nn.Module) -> type[_AttentionKind]:
        """The strategy whose family ``attention`` is; ``TypeError`` if it is none of them."""
        for kind in dict.fromkeys(_AttentionKind._kinds.values()):
            if kind.matches(attention):
                return kind
        raise TypeError(
            f"{type(attention).__name__} is not an attention module an adaption prompt can wrap"
        )

    @staticmethod
    @abstractmethod
    def matches(attention: nn.Module) -> bool:
        """Whether ``attention`` is a module of this family."""

    @staticmethod
    @abstractmethod
    def anchor(attention: nn.Module) -> Parameter:
        """A parameter of ``attention`` that gives the prompt its device and dtype."""

    @staticmethod
    @abstractmethod
    def width(attention: nn.Module) -> int:
        """The width of the model, which the prompt vectors share."""

    @staticmethod
    @abstractmethod
    def heads(attention: nn.Module) -> tuple[int, int]:
        """``(number of heads, head size)``."""

    @staticmethod
    @abstractmethod
    def keys_values(attention: nn.Module, prompt: Tensor) -> tuple[Tensor, Tensor]:
        """The prompt ``(1, length, width)`` through the module's key and value projections."""

    @staticmethod
    @abstractmethod
    def queries(attention: nn.Module, hidden: Tensor) -> Tensor:
        """The module's queries of ``hidden``, ``(batch, tokens, width)``, before heads are split."""


class _FusedQKV(_AttentionKind, model_types=("gpt", "gpt2")):
    """A decoder whose attention has one ``c_attn`` projecting to queries, keys and values at once."""

    target_module = "attn"

    @staticmethod
    @override
    def matches(attention: nn.Module) -> bool:
        return (
            isinstance(getattr(attention, "c_attn", None), nn.Linear)
            and isinstance(getattr(attention, "num_heads", None), int)
            and isinstance(getattr(attention, "head_dim", None), int)
        )

    @staticmethod
    def _c_attn(attention: nn.Module) -> nn.Linear:
        projection = attention.c_attn
        assert isinstance(projection, nn.Linear)  # ``matches`` checked it
        return projection

    @staticmethod
    @override
    def anchor(attention: nn.Module) -> Parameter:
        return _FusedQKV._c_attn(attention).weight

    @staticmethod
    @override
    def width(attention: nn.Module) -> int:
        return _FusedQKV._c_attn(attention).in_features

    @staticmethod
    @override
    def heads(attention: nn.Module) -> tuple[int, int]:
        num_heads, head_dim = attention.num_heads, attention.head_dim
        assert isinstance(num_heads, int) and isinstance(head_dim, int)  # ``matches`` checked it
        return num_heads, head_dim

    @staticmethod
    @override
    def keys_values(attention: nn.Module, prompt: Tensor) -> tuple[Tensor, Tensor]:
        _, key, value = _FusedQKV._c_attn(attention)(prompt).chunk(3, dim=-1)
        return key, value

    @staticmethod
    @override
    def queries(attention: nn.Module, hidden: Tensor) -> Tensor:
        query, _, _ = _FusedQKV._c_attn(attention)(hidden).chunk(3, dim=-1)
        return query
