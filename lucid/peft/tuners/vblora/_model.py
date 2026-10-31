"""``VBLoRAModel``: the VB-LoRA half of a tuner — the vector bank its layers share, and how a file is written.

Everything that is not VB-LoRA — planning, applying, freezing, switching,
merging, unloading — is ``_BaseTuner``'s, in ``lucid.peft.tuners._base``; the
bank the layers share and where it goes in the adapter file are
``lucid.peft.tuners._shared``'s.
"""

from collections.abc import Mapping
from typing import ClassVar, Final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid._types import DTypeLike
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft.tuners._base import _Target, _tuner_layers
from lucid.peft.tuners._files import _FileContext
from lucid.peft.tuners._shared import _base_linear, _SharedTablesTuner
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.vblora._config import VBLoRAConfig
from lucid.peft.tuners.vblora._defaults import VBLORA_DEFAULT_TARGET_MODULES
from lucid.peft.tuners.vblora._layer import VBLoRALayer, _new_bank

_LOGITS: Final = ("vblora_logits_A", "vblora_logits_B")
_INDICES: Final = "_topk_indices"
_WEIGHTS: Final = "_topk_weights"


def _as_vblora(config: PeftConfig) -> VBLoRAConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, VBLoRAConfig)
    return config


def _index_dtype(num_vectors: int) -> DTypeLike:
    """The narrowest integer dtype that holds an index into a bank of ``num_vectors`` (the reference also uses 8 bits below 256)."""
    if num_vectors < 2**15:
        return lucid.int16
    return lucid.int32 if num_vectors < 2**31 else lucid.int64


class VBLoRAModel(_SharedTablesTuner, _ConfigTuner, peft_type=PeftType.VBLORA):
    r"""A model with VB-LoRA layers put into it, the vector bank they share, and the tools to take them out.

    Construction replaces each ``nn.Linear`` ``config`` selects with a VB-LoRA
    layer around it and freezes every other parameter.  The model is changed in
    place and kept as ``model``; a public name this wrapper lacks is read from
    it.  ``config`` itself is not changed: ``peft_config`` holds a copy with the
    default ``target_modules`` spelled out.

    The vector bank is one ``(num_vectors, vector_length)`` parameter per adapter,
    held here as ``vblora_vector_bank`` (the reference's attribute) and trained
    with the layers' logits.  Every layer reads it from there; the first layer
    holds it in its module tree, so a state dict lists it once and the adapter
    file holds one bank.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : VBLoRAConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``vblora_logits_*`` and in the bank.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is not an ``nn.Linear``,
        a match's sizes are not multiples of ``vector_length``, or
        ``modules_to_save`` matches nothing.
    NotImplementedError
        The model already holds adapter layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.vblora import VBLoRAConfig, VBLoRAModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    >>> config = VBLoRAConfig(r=2, num_vectors=6, vector_length=4, target_modules=["0", "2"])
    >>> vblora = VBLoRAModel(net, config)
    >>> tuple(vblora.vblora_vector_bank["default"].shape)
    (6, 4)
    >>> net[0].base_layer.weight.requires_grad, vblora.vblora_vector_bank["default"].requires_grad
    (False, True)
    >>> vblora.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = VBLoRALayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = VBLoRAConfig
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = VBLORA_DEFAULT_TARGET_MODULES
    _table_names: ClassVar[tuple[str, ...]] = ("vblora_vector_bank",)

    def __init__(
        self, model: nn.Module, config: VBLoRAConfig, adapter_name: str = "default"
    ) -> None:
        super().__init__(model, config, adapter_name)

    @property
    def vblora_vector_bank(self) -> nn.ParameterDict:
        """The shared vector banks, one ``(num_vectors, vector_length)`` parameter per adapter."""
        return self._shared.parameters_of("vblora_vector_bank")

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = _as_vblora(super()._prepare_config(config, model))
        _reset_fan_in_fan_out(stored)
        return stored

    # ── the shared bank ───────────────────────────────────────────────────

    @override
    @classmethod
    def _new_tables(cls, config: _TargetedConfig) -> dict[str, nn.Module]:
        del config  # the bank is a parameter table whatever the config
        return {"vblora_vector_bank": nn.ParameterDict()}

    @override
    def _provide(self, adapter_name: str, config: _TargetedConfig, targets: list[_Target]) -> None:
        """Enter the bank ``adapter_name`` reads: one per adapter, drawn here and shared by the layers."""
        anchor = _base_linear(targets[0].module).weight
        self.vblora_vector_bank[adapter_name] = _new_bank(_as_vblora(config), anchor)

    # ── the adapter file ──────────────────────────────────────────────────

    @override
    @classmethod
    def _to_file(cls, context: _FileContext, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """With ``save_only_topk_weights``, each logits tensor becomes its top-``k`` indices and all but the last softmax weight."""
        config = _as_vblora(context.config)
        if not config.save_only_topk_weights:
            return state
        kept: dict[str, Tensor] = {}
        for key, value in state.items():
            if key.rpartition(".")[2] not in _LOGITS:
                kept[key] = value
                continue
            top, indices = value.topk(config.topk, dim=-1)
            kept[key + _INDICES] = indices.to(dtype=_index_dtype(config.num_vectors))
            kept[key + _WEIGHTS] = F.softmax(top, dim=-1)[..., :-1].contiguous()
        return kept

    @override
    @classmethod
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Rebuild the logits of a ``save_only_topk_weights`` file (``-inf`` off the top ``k``), then read the bank."""
        num_vectors = _as_vblora(context.config).num_vectors
        rebuilt: dict[str, Tensor] = {}
        for key, value in state.items():
            if not key.endswith(_INDICES):
                if not key.endswith(_WEIGHTS):
                    rebuilt[key] = value
                continue
            base = key.removesuffix(_INDICES)
            if base + _WEIGHTS not in state:
                raise ValueError(f"{key!r} has no {base + _WEIGHTS!r} beside it")
            kept = state[base + _WEIGHTS]
            # Only the first k - 1 weights are saved; the last is what is left of 1.
            weights = lucid.cat([kept, 1.0 - kept.sum(-1, keepdim=True)], dim=-1).clamp(min=0.0)
            empty = lucid.full(
                (*weights.shape[:-1], num_vectors),
                float("-inf"),
                device=weights.device,
                dtype=weights.dtype,
            )
            rebuilt[base] = empty.scatter(-1, value.to(dtype=lucid.int64), weights.log())
        return super()._from_file(context, rebuilt)

    # ── the trainable count ───────────────────────────────────────────────

    def get_nb_savable_parameters(self, adapter: str = "default") -> tuple[int, int]:
        r"""Count the VB-LoRA parameters a file holds, and the other trainable ones.

        With ``save_only_topk_weights`` the logits are counted as what is saved
        instead: the top-``k`` weights (all but one per sub-vector) and indices
        (at a quarter, half or the whole of a float32 each, by how many bits an
        index takes).

        Parameters
        ----------
        adapter : str, default="default"
            The adapter whose config says what is saved.

        Returns
        -------
        tuple of int
            ``(VB-LoRA parameters, other trainable parameters)``, float32-equivalent.
        """
        config = _as_vblora(self.peft_config[adapter])
        logit_params = [
            p
            for _, layer in _tuner_layers(self.model)
            if isinstance(layer, VBLoRALayer)
            for table in (layer.vblora_logits_A, layer.vblora_logits_B)
            for p in table.values()
        ]
        bank_params = list(self.vblora_vector_bank.values())
        logits = sum(int(p.numel()) for p in logit_params)
        bank = sum(int(p.numel()) for p in bank_params)
        counted = {id(p) for p in (*logit_params, *bank_params)}
        other = sum(
            int(p.numel())
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in counted
        )
        if not config.save_only_topk_weights:
            return bank + logits, other
        if config.num_vectors < 2**8:
            factor = 0.25
        elif config.num_vectors < 2**15:
            factor = 0.5
        else:
            factor = 1.0 if config.num_vectors < 2**31 else 2.0
        weights = logits / config.num_vectors * (config.topk - 1)
        indices = logits / config.num_vectors * config.topk * factor
        return int(bank + weights + indices), other

    def print_savable_parameters(self) -> None:
        """Print the VB-LoRA parameters a file holds and the total it holds, float32-equivalent."""
        vblora, other = self.get_nb_savable_parameters()
        print(
            f"VB-LoRA params to-be-saved (float32-equivalent): {vblora:,d} "
            f"|| total params to-be-saved: {(vblora + other):,d}"
        )
