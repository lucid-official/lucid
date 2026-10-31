r"""ViT: separate ``query`` / ``key`` / ``value`` adapters → the zoo's one fused ``qkv``.

The reference library's ViT has three projections per attention block and an
adapter on each; the zoo's has one ``qkv`` ``nn.Linear`` whose output is the
three stacked.  A LoRA update :math:`s_i B_i A_i x` on each projection is the
update of one fused layer exactly, with :math:`r_g = \max_i r_i`:

.. math::

    A_f = \begin{bmatrix}A_q\\A_k\\A_v\end{bmatrix},\qquad
    B_f = \mathrm{blockdiag}(B_q, B_k, B_v)\ (\text{each }s_i/s_f\text{-scaled}),\qquad
    B_f A_f x = \begin{bmatrix}B_q A_q x\\B_k A_k x\\B_v A_v x\end{bmatrix}

a rank-:math:`3 r_g` layer.  A projection the adapter did not target gets zero
blocks; a rank smaller than :math:`r_g` is zero-padded.  The fused layer's
scaling :math:`s_f` is that of the first projection present, so in the usual
case (one :math:`\alpha`, one :math:`r`) no tensor is rescaled at all; where the
projections' scalings differ, the ratio :math:`s_i / s_f` is folded into
:math:`B_i`.  The layer's rank and :math:`\alpha` go into ``rank_pattern`` /
``alpha_pattern`` under its own name, so the other modules keep the config's.

The rest of the layout is renamed: ``encoder.layer.N.attention.output.dense`` →
``blocks.N.attn.proj``, ``intermediate.dense`` / ``output.dense`` →
``mlp.fc1`` / ``mlp.fc2``, ``layernorm_before`` / ``layernorm_after`` →
``norm1`` / ``norm2``, the head ``classifier`` → ``head``.

Beside the factors, a ``lora_bias`` bias is stacked (zeros for an absent
projection) and a DoRA magnitude is stacked — both are per output row, so
stacking is exact.  An absent projection's DoRA magnitude is its base
weight's row norm (the unadapted layer), so that case needs the model.  Only
LoRA is fused: another method refuses, naming itself.
"""

import dataclasses
import math
import re
from collections.abc import Callable
from typing import Final, NamedTuple, override

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
from lucid.peft.tuners.lora._config import LoraConfig

_PROJECTIONS: Final = ("query", "key", "value")
_FUSED: Final = re.compile(
    r"(?:vit\.)?encoder\.layer\.(?P<n>\d+)\.attention\.attention\.(?P<role>query|key|value)"
)
_FACTOR_A: Final = ".lora_A.weight"
_FACTOR_B: Final = ".lora_B.weight"
_BIAS: Final = ".lora_B.bias"
_MAGNITUDE: Final = ".lora_magnitude_vector.weight"
_SUPPORTED: Final = frozenset({_FACTOR_A, _FACTOR_B, _BIAS, _MAGNITUDE})

_RENAMES: Final = (
    (r"(?:vit\.)?encoder\.layer\.(\d+)\.attention\.output\.dense", r"blocks.\1.attn.proj"),
    (r"(?:vit\.)?encoder\.layer\.(\d+)\.intermediate\.dense", r"blocks.\1.mlp.fc1"),
    (r"(?:vit\.)?encoder\.layer\.(\d+)\.output\.dense", r"blocks.\1.mlp.fc2"),
    (r"(?:vit\.)?encoder\.layer\.(\d+)\.layernorm_before", r"blocks.\1.norm1"),
    (r"(?:vit\.)?encoder\.layer\.(\d+)\.layernorm_after", r"blocks.\1.norm2"),
    (r"(?:vit\.)?layernorm", "norm"),
    (r"(?:vit\.)?embeddings\.patch_embeddings\.projection", "patch_embed.proj"),
    (r"classifier", "head"),
)
_COMPILED: Final = tuple((re.compile(pattern), target) for pattern, target in _RENAMES)


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
    """The adapters of one block's ``query`` / ``key`` / ``value``, as the tensors of its one ``qkv``.

    ``parts`` maps a projection's index (0, 1, 2) to its reference tensors; an
    index it lacks is a projection the adapter left alone.
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
        #: Every projection's block is this many rows of the fused ``lora_A``.
        self.block = max(ranks.values())
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
        return 3 * self.block

    @property
    def alpha(self) -> float:
        """The ``alpha`` that gives the fused layer ``fused_scale`` at its rank."""
        return _number(self.fused_scale * (math.sqrt(self.rank) if self.use_rslora else self.rank))

    def _zeros(self, *shape: int) -> Tensor:
        return lucid.zeros(shape, dtype=self.sample.dtype, device=self.sample.device)

    def _padded(self, tensor: Tensor, axis: int) -> Tensor:
        """``tensor`` with zeros after it along ``axis`` up to ``block``."""
        short = self.block - tensor.shape[axis]
        if short == 0:
            return tensor
        gap = list(tensor.shape)
        gap[axis] = short
        return lucid.cat([tensor, self._zeros(*gap)], dim=axis)

    def _ratio(self, index: int) -> float:
        return self.scale[index] / self.fused_scale

    def factor_a(self) -> Tensor:
        """The three ``A`` stacked along the rank."""
        return lucid.cat(
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

    def factor_b(self) -> Tensor:
        """The three ``B`` on the diagonal, each carrying its scaling relative to the fused one."""
        rows = []
        for i in range(3):
            if i not in self.parts:
                rows.append(self._zeros(self.width, self.rank))
                continue
            b = self.parts[i].tensors[_FACTOR_B]
            if self._ratio(i) != 1.0:
                b = b * self._ratio(i)
            before = self._zeros(self.width, i * self.block)
            after = self._zeros(self.width, (2 - i) * self.block)
            rows.append(lucid.cat([before, self._padded(b, 1), after], dim=1))
        return lucid.cat(rows, dim=0)

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
        return lucid.cat(
            [
                self.parts[i].tensors[_MAGNITUDE] if i in self.parts else unadapted(i)
                for i in range(3)
            ]
        )


class ViTConverter(AdapterConverter, model_type="vit"):
    """The reference library's ViT adapters: renamed, and the three projections fused into ``qkv``."""

    reference_classes = frozenset({"ViTModel", "ViTForImageClassification"})

    @override
    def map_module(self, module: str) -> str | None:
        fused = _FUSED.fullmatch(module)
        if fused is not None:
            return f"blocks.{fused['n']}.attn.qkv"
        for pattern, target in _COMPILED:
            if pattern.fullmatch(module):
                return pattern.sub(target, module)
        return None

    def _gathered(
        self, adapter: AdapterFile
    ) -> tuple[dict[str, dict[int, _Projection]], dict[str, Tensor]]:
        """The projections' tensors grouped by the ``qkv`` they fuse into, and every other tensor."""
        groups: dict[str, dict[int, _Projection]] = {}
        others: dict[str, Tensor] = {}
        for key, tensor in adapter.tensors.items():
            module, rest = split_key(key.removeprefix(MODEL_PREFIX))
            fused = _FUSED.fullmatch(module) if key.startswith(MODEL_PREFIX) else None
            if fused is None:
                others[key] = tensor
                continue
            self._lora(adapter.config)  # a method other than LoRA names itself first
            if rest not in _SUPPORTED:
                raise NotImplementedError(
                    f"{key!r}: only the LoRA factors, lora_bias and the DoRA magnitude of a "
                    "query / key / value projection can be fused into the zoo's qkv"
                )
            role = _PROJECTIONS.index(fused["role"])
            projections = groups.setdefault(f"blocks.{fused['n']}.attn.qkv", {})
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

    @staticmethod
    def _lora(config: PeftConfig) -> LoraConfig:
        """``config`` as a LoRA config: the only method whose factors fuse exactly."""
        if config.peft_type != PeftType.LORA or not isinstance(config, LoraConfig):
            raise NotImplementedError(
                f"{config.peft_type} adapters on query / key / value cannot be converted: "
                "the zoo's ViT fuses the three into one qkv layer, and only LoRA's "
                "factors fuse exactly"
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
        base = layer.get_base_layer() if hasattr(layer, "get_base_layer") else layer
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is {type(base).__name__}, not the zoo's fused nn.Linear")
        rows = base.weight[index * width : (index + 1) * width]
        with lucid.no_grad():
            return linalg.norm(rows, dim=1)
