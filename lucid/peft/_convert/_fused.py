"""Shared fused-projection mechanics for QKV ``nn.Linear`` and ``nn.MultiheadAttention`` converters."""

import dataclasses
import math
import re
from collections.abc import Callable
from typing import ClassVar, Final, NamedTuple, override

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType
from lucid.peft._convert._base import (
    MODEL_PREFIX,
    AdapterConverter,
    AdapterFile,
    ModulePair,
    split_key,
)
from lucid.peft._targeting import _pattern_value
from lucid.peft.tuners._layer import _BaseTunerLayer
from lucid.peft.tuners.lora._config import LoraConfig

_PROJECTIONS: Final = ("query", "key", "value", "out_proj")
_ROLE_INDEX: Final = {
    "query": 0,
    "key": 1,
    "value": 2,
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "out_proj": 3,
    "o_proj": 3,
}
_FACTOR_A: Final = ".lora_A.weight"
_FACTOR_B: Final = ".lora_B.weight"
_BIAS: Final = ".lora_B.bias"
_MAGNITUDE: Final = ".lora_magnitude_vector.weight"
_QKV_SUPPORTED: Final = frozenset({_FACTOR_A, _FACTOR_B, _BIAS, _MAGNITUDE})
_MHA_SUPPORTED: Final = frozenset({_FACTOR_A, _FACTOR_B})


class _Projection(NamedTuple):
    """One reference projection's adapter tensors, by the key part after the module."""

    reference: str
    tensors: dict[str, Tensor]


def _scaling(alpha: float, rank: int, rslora: bool) -> float:
    return alpha / (math.sqrt(rank) if rslora else rank)


def _number(value: float) -> float:
    """``value`` as an int when it is one to rounding."""
    rounded = round(value)
    return rounded if math.isclose(value, rounded, rel_tol=1e-12, abs_tol=0.0) else value


class _FusedLayer:
    """The adapters of one block's projections, as the tensors of its fused zoo layer.

    ``parts`` maps a projection's index (0, 1, 2 for query, key, value; 3 for an
    ``nn.MultiheadAttention`` output projection) to its reference tensors; an index
    it lacks is a projection the adapter left alone.
    """

    def __init__(self, parts: dict[int, _Projection], config: LoraConfig) -> None:
        for index, part in parts.items():
            if _FACTOR_A not in part.tensors or _FACTOR_B not in part.tensors:
                raise ValueError(
                    f"{part.reference!r} ({_PROJECTIONS[index]}) has no lora_A / lora_B pair"
                )
        self.parts = parts
        self.use_rslora = config.use_rslora
        ranks = {i: p.tensors[_FACTOR_A].shape[0] for i, p in parts.items()}
        qkv_ranks = [r for i, r in ranks.items() if i < 3]
        #: Every Q/K/V projection's block is this many rows of the fused ``lora_A``.
        self.block = max(qkv_ranks, default=0)
        self._out_rank = ranks.get(3, 0)
        first = parts[min(parts)].tensors
        self.sample = first[_FACTOR_A]
        self.features, self.width = self.sample.shape[1], first[_FACTOR_B].shape[0]
        self.scale = {
            i: _scaling(
                _pattern_value(config.alpha_pattern, p.reference, config.lora_alpha),
                ranks[i],
                config.use_rslora,
            )
            for i, p in parts.items()
        }
        #: The fused layer's scaling: the first projection's, so the common case rescales nothing.
        self.fused_scale = self.scale[min(parts)]
        for part in parts.values():
            a, b = part.tensors[_FACTOR_A], part.tensors[_FACTOR_B]
            if a.shape[1] != self.features or b.shape[0] != self.width:
                raise ValueError(f"{part.reference!r} does not match its sibling projections")

    @property
    def rank(self) -> int:
        return max(3 * self.block, self._out_rank)

    @property
    def alpha(self) -> float:
        """The ``alpha`` that gives the fused layer ``fused_scale`` at its rank."""
        return _number(self.fused_scale * (math.sqrt(self.rank) if self.use_rslora else self.rank))

    def _zeros(self, *shape: int) -> Tensor:
        return lucid.zeros(shape, dtype=self.sample.dtype, device=self.sample.device)

    def _padded(self, tensor: Tensor, axis: int, target: int | None = None) -> Tensor:
        """``tensor`` with zeros after it along ``axis`` up to ``target`` (default ``block``)."""
        short = (self.block if target is None else target) - tensor.shape[axis]
        if short == 0:
            return tensor
        gap = list(tensor.shape)
        gap[axis] = short
        return lucid.cat([tensor, self._zeros(*gap)], dim=axis)

    def _ratio(self, index: int) -> float:
        return self.scale[index] / self.fused_scale

    def _scaled_b(self, index: int) -> Tensor:
        b = self.parts[index].tensors[_FACTOR_B]
        return b * self._ratio(index) if self._ratio(index) != 1.0 else b

    def factor_a(self) -> Tensor:
        """The three Q/K/V ``A`` stacked along the rank, padded to ``rank``."""
        if self.block == 0:
            return self._zeros(self.rank, self.features)
        stacked = lucid.cat(
            [
                (
                    self._padded(self.parts[i].tensors[_FACTOR_A], 0)
                    if i in self.parts
                    else self._zeros(self.block, self.features)
                )
                for i in range(3)
            ],
            dim=0,
        )
        return self._padded(stacked, 0, self.rank)

    def factor_b(self) -> Tensor:
        """The three Q/K/V ``B`` on the diagonal, each carrying its scaling relative to the fused one."""
        if self.block == 0:
            return self._zeros(3 * self.width, self.rank)
        rows = []
        for i in range(3):
            if i not in self.parts:
                rows.append(self._zeros(self.width, 3 * self.block))
                continue
            before = self._zeros(self.width, i * self.block)
            after = self._zeros(self.width, (2 - i) * self.block)
            rows.append(lucid.cat([before, self._padded(self._scaled_b(i), 1), after], dim=1))
        return self._padded(lucid.cat(rows, dim=0), 1, self.rank)

    def out_factor_a(self) -> Tensor:
        """The output projection's ``A`` padded to ``rank``, or zeros if unadapted."""
        if 3 not in self.parts:
            return self._zeros(self.rank, self.features)
        return self._padded(self.parts[3].tensors[_FACTOR_A], 0, self.rank)

    def out_factor_b(self) -> Tensor:
        """The output projection's ``B`` scaled and padded to ``rank``, or zeros if unadapted."""
        if 3 not in self.parts:
            return self._zeros(self.width, self.rank)
        return self._padded(self._scaled_b(3), 1, self.rank)

    def bias(self) -> Tensor | None:
        """The ``lora_bias`` of ``B`` stacked (zeros for a projection without one), or ``None``."""
        if not any(_BIAS in p.tensors for p in self.parts.values()):
            return None
        pieces = []
        for i in range(3):
            held = self.parts.get(i)
            if held is None or _BIAS not in held.tensors:
                pieces.append(self._zeros(self.width))
            else:
                bias = held.tensors[_BIAS]
                pieces.append(bias * self._ratio(i) if self._ratio(i) != 1.0 else bias)
        return lucid.cat(pieces)

    def magnitude(self, unadapted: Callable[[int], Tensor]) -> Tensor | None:
        """The DoRA magnitudes stacked, or ``None``; ``unadapted(i)`` is a left-alone projection's."""
        if not any(_MAGNITUDE in p.tensors for p in self.parts.values()):
            return None
        pieces = []
        for i in range(3):
            held = self.parts.get(i)
            if held is None or _MAGNITUDE not in held.tensors:
                pieces.append(unadapted(i).to(dtype=self.sample.dtype, device=self.sample.device))
            else:
                pieces.append(held.tensors[_MAGNITUDE])
        return lucid.cat(pieces)


class _FusedQKVConverter(AdapterConverter):
    """Template for converters that fuse separate Q/K/V projections into a single zoo ``qkv``."""

    _FUSED: ClassVar[re.Pattern[str]]
    _FUSED_TARGET: ClassVar[str] = ""
    _COMPILED: ClassVar[tuple[tuple[re.Pattern[str], str], ...]] = ()
    _SUPPORTED: ClassVar[frozenset[str]] = _QKV_SUPPORTED
    _FUSED_DESC: ClassVar[str] = (
        "only the LoRA factors, lora_bias and the DoRA magnitude of a "
        "query / key / value projection can be fused into the zoo's qkv"
    )

    def _fused_name(self, match: re.Match[str]) -> str:
        return match.expand(self._FUSED_TARGET)

    @override
    def map_module(self, module: str) -> str | None:
        fused = self._FUSED.fullmatch(module)
        if fused is not None:
            return self._fused_name(fused)
        for pattern, target in self._COMPILED:
            if pattern.fullmatch(module):
                return pattern.sub(target, module)
        return None

    def _gathered(
        self, adapter: AdapterFile
    ) -> tuple[dict[str, dict[int, _Projection]], dict[str, Tensor]]:
        """The projections' tensors grouped by the zoo module they fuse into, and every other tensor."""
        groups: dict[str, dict[int, _Projection]] = {}
        others: dict[str, Tensor] = {}
        for key, tensor in adapter.tensors.items():
            module, rest = split_key(key.removeprefix(MODEL_PREFIX))
            fused = self._FUSED.fullmatch(module) if key.startswith(MODEL_PREFIX) else None
            if fused is None:
                others[key] = tensor
                continue
            self._lora(adapter.config)
            if rest not in self._SUPPORTED:
                raise NotImplementedError(f"{key!r}: {self._FUSED_DESC}")
            role = _ROLE_INDEX[fused["role"]]
            projections = groups.setdefault(self._fused_name(fused), {})
            projections.setdefault(role, _Projection(module, {})).tensors[rest] = tensor
        return groups, others

    @override
    def convert(self, adapter: AdapterFile) -> AdapterFile:
        config = adapter.config
        groups, others = self._gathered(adapter)
        pairs = self._pairs(others) + [
            ModulePair(part.reference, name, adapted=True)
            for name, parts in groups.items()
            for part in parts.values()
        ]
        tensors = self._renamed(others)
        layers = {name: _FusedLayer(parts, self._lora(config)) for name, parts in groups.items()}
        for name, layer in layers.items():
            self._write(name, layer, tensors)
        converted = self._config(config, pairs, frozenset(layers))
        if layers and isinstance(converted, LoraConfig):
            converted = dataclasses.replace(
                converted,
                rank_pattern={**converted.rank_pattern, **{n: ly.rank for n, ly in layers.items()}},
                alpha_pattern={
                    **converted.alpha_pattern,
                    **{n: ly.alpha for n, ly in layers.items()},
                },
            )
        return AdapterFile(converted, tensors)

    @classmethod
    def _lora(cls, config: PeftConfig) -> LoraConfig:
        """``config`` as a LoRA config: the only method whose factors fuse exactly."""
        if config.peft_type != PeftType.LORA or not isinstance(config, LoraConfig):
            raise NotImplementedError(
                f"{config.peft_type} adapters on attention projections cannot be converted for "
                f"{cls.model_type or 'this model'}: the zoo fuses the projections, and only "
                "LoRA's factors fuse exactly"
            )
        return config

    def _write(self, name: str, layer: _FusedLayer, out: dict[str, Tensor]) -> None:
        """The fused layer ``name``'s tensors, under the zoo's keys."""
        base = f"{MODEL_PREFIX}{name}"
        out[base + _FACTOR_A] = layer.factor_a()
        out[base + _FACTOR_B] = layer.factor_b()
        bias = layer.bias()
        if bias is not None:
            out[base + _BIAS] = bias
        magnitude = layer.magnitude(lambda i: self._base_row_norms(name, i, layer.width))
        if magnitude is not None:
            out[base + _MAGNITUDE] = magnitude

    def _base_row_norms(self, name: str, index: int, width: int) -> Tensor:
        """The row norms of one projection of the base ``qkv`` weight: a DoRA magnitude that leaves it as it is."""
        model = self.context.model
        if model is None:
            raise NotImplementedError(
                f"{name}: DoRA on some of query / key / value needs the unadapted projection's "
                "weight norm as its magnitude, so the adapter must be loaded into the model "
                "(PeftModel.from_pretrained) rather than converted on files"
            )
        layer = dict(model.named_modules())[name]
        if isinstance(layer, _BaseTunerLayer):
            base = layer._root()  # the module beneath every layer, as methods may be stacked
        else:
            base = layer.get_base_layer() if hasattr(layer, "get_base_layer") else layer
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is {type(base).__name__}, not the zoo's fused nn.Linear")
        rows = base.weight[index * width : (index + 1) * width]
        with lucid.no_grad():
            return linalg.norm(rows, dim=1)


class _FusedMhaConverter(_FusedQKVConverter):
    """Template for converters that fuse Q/K/V and output projections into ``nn.MultiheadAttention``."""

    _SUPPORTED = _MHA_SUPPORTED
    _FUSED_DESC = (
        "only the LoRA factors (lora_A / lora_B) of an attention projection can be "
        "fused into the zoo's MultiheadAttention"
    )

    @override
    def _write(self, name: str, layer: _FusedLayer, out: dict[str, Tensor]) -> None:
        base = f"{MODEL_PREFIX}{name}"
        out[base + _FACTOR_A] = layer.factor_a()
        out[base + _FACTOR_B] = layer.factor_b()
        out[base + ".base_layer.out_proj" + _FACTOR_A] = layer.out_factor_a()
        out[base + ".base_layer.out_proj" + _FACTOR_B] = layer.out_factor_b()
