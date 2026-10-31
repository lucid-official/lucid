"""``RandLoraModel``: the RandLoRA half of a tuner — the bases its layers share; the rest is the generic tuner's.

Everything that is not RandLoRA — planning, applying, freezing, switching,
merging, unloading — is ``_BaseTuner``'s, in ``lucid.peft.tuners._base``; the
tables the layers share and the file they are written to are
``lucid.peft.tuners._shared``'s.
"""

from collections.abc import Mapping
from typing import ClassVar, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft.tuners._base import _BaseTunerLayer, _Target
from lucid.peft.tuners._shared import _base_linear, _BufferDict, _ProjectionTuner
from lucid.peft.tuners._support import _adapter_dtype, _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.randlora._bases import _Bases, _num_bases
from lucid.peft.tuners.randlora._config import RandLoraConfig
from lucid.peft.tuners.randlora._layer import RandLoraLayer, _sparsity, _RandLoraSpec


def _as_randlora(config: PeftConfig) -> RandLoraConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, RandLoraConfig)
    return config


class RandLoraModel(_ProjectionTuner, _ConfigTuner, peft_type=PeftType.RANDLORA):
    r"""A model with RandLoRA layers put into it, the random bases they share, and the tools to take them out.

    Construction replaces each ``nn.Linear`` ``config`` selects with a RandLoRA
    layer around it and freezes every other parameter.  The model is changed in
    place and kept as ``model``; a public name this wrapper lacks is read from
    it.  ``config`` itself is not changed: ``peft_config`` holds a copy with the
    default ``target_modules`` spelled out.

    The two random bases :math:`A` and :math:`B` are drawn once, sized for the
    largest targeted layer, and held here as ``randlora_A`` / ``randlora_B``
    (tables of one tensor per adapter, aliased when adapters share them) — the
    reference's attributes.  Every layer reads them from there; the first layer
    holds them in its module tree, so a state dict lists each once and the adapter
    file holds one pair.  Adapters added later must fit them (the same
    ``projection_prng_key``, ``save_projection`` and ``r``, layer sizes no larger).

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : RandLoraConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``randlora_lambda`` / ``randlora_gamma``.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is not an ``nn.Linear``,
        ``modules_to_save`` matches nothing, or an adapter does not fit the bases
        of the first.
    NotImplementedError
        The model already holds adapter layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.randlora import RandLoraConfig, RandLoraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> randlora = RandLoraModel(net, RandLoraConfig(r=2, target_modules=["0", "2"]))
    >>> tuple(randlora.randlora_A["default"].shape), tuple(randlora.randlora_B["default"].shape)
    ((2, 1, 8), (8, 4, 2))
    >>> net[0].base_layer.weight.requires_grad, net[0].randlora_lambda["default"].requires_grad
    (False, True)
    >>> randlora.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = RandLoraLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = RandLoraConfig
    _table_names: ClassVar[tuple[str, ...]] = ("randlora_A", "randlora_B")

    def __init__(
        self, model: nn.Module, config: RandLoraConfig, adapter_name: str = "default"
    ) -> None:
        super().__init__(model, config, adapter_name)

    @property
    def randlora_A(self) -> _BufferDict:
        """The shared ``A`` bases, one entry per adapter (the reference's attribute)."""
        return self._shared.buffers("randlora_A")

    @property
    def randlora_B(self) -> _BufferDict:
        """The shared ``B`` bases, one entry per adapter (the reference's attribute)."""
        return self._shared.buffers("randlora_B")

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = _as_randlora(super()._prepare_config(config, model))
        _reset_fan_in_fan_out(stored)
        return stored

    # ── the shared bases ──────────────────────────────────────────────────

    @override
    @classmethod
    def _new_tables(cls, config: _TargetedConfig) -> dict[str, nn.Module]:
        persistent = _as_randlora(config).save_projection
        return {"randlora_A": _BufferDict(persistent), "randlora_B": _BufferDict(persistent)}

    @override
    @classmethod
    def _saves_projection(cls, config: PeftConfig) -> bool:
        return _as_randlora(config).save_projection

    @override
    @classmethod
    def _prng_key(cls, config: PeftConfig) -> int:
        return _as_randlora(config).projection_prng_key

    @override
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        super()._check_new_adapter_config(config)
        randlora = _as_randlora(config)
        for held in self.peft_config.values():
            if _as_randlora(held).r != randlora.r:
                raise ValueError(
                    "RandLoRA adapters of one model share their random bases, so they share a "
                    f"rank: got r={randlora.r} but a previous config had r={_as_randlora(held).r}."
                )

    @override
    def _provide(self, adapter_name: str, config: _TargetedConfig, targets: list[_Target]) -> None:
        """Enter the bases ``adapter_name`` reads: drawn for the first adapter, the first's shared with the others.

        Sized for the largest of ``targets`` in each dimension.  A later adapter must
        fit what the first drew.
        """
        randlora = _as_randlora(config)
        linears = [_base_linear(target.module) for target in targets]
        out_size = max(linear.out_features for linear in linears)
        in_size = max(linear.in_features for linear in linears)
        max_dim, min_dim = max(out_size, in_size), min(out_size, in_size)
        if len(self.randlora_A) == 0:
            anchor = linears[0].weight
            spec = _RandLoraSpec(
                randlora.r, 1.0, randlora.projection_prng_key, randlora.sparse, randlora.very_sparse
            )
            self.randlora_A[adapter_name], self.randlora_B[adapter_name] = _Bases.draw(
                randlora.projection_prng_key,
                randlora.r,
                max_dim,
                min_dim,
                _sparsity(spec, min_dim),
                anchor.device,
                _adapter_dtype(anchor),
            )
            return
        held = next(iter(self.randlora_A))
        basis_a, basis_b = self.randlora_A[held], self.randlora_B[held]
        too_small = (
            "{} has a size of {} but {} or greater is required; this probably happened because "
            "an additional RandLora adapter was added after the first one with incompatible "
            "shapes."
        )
        for label, size, need in (
            ("randlora_B", basis_b.shape[0], max_dim),
            ("randlora_A", basis_a.shape[-1], min_dim),
        ):
            if size < need:
                raise ValueError(too_small.format(label, size, need))
        self.randlora_A[adapter_name], self.randlora_B[adapter_name] = basis_a, basis_b

    @override
    @classmethod
    def _covers(
        cls, config: PeftConfig, loaded: Mapping[str, Tensor], layers: list[_BaseTunerLayer]
    ) -> bool:
        """Whether the file's bases are rank-``r`` and big enough to be sliced for every one of ``layers``."""
        basis_a, basis_b = loaded["randlora_A"], loaded["randlora_B"]
        r = _as_randlora(config).r
        if basis_a.ndim != 3 or basis_b.ndim != 3 or basis_a.shape[0] != r or basis_b.shape[2] != r:
            return False
        for layer in layers:
            assert isinstance(layer, RandLoraLayer)
            low = min(layer.in_features, layer.out_features)
            high = max(layer.in_features, layer.out_features)
            if (
                basis_a.shape[-1] < low
                or basis_b.shape[0] < high
                or basis_b.shape[1] < _num_bases(low, r)
            ):
                return False
        return True
