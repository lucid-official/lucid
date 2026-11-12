"""``UniLoraModel``: the Uni-LoRA tuner — the shared vector, the projection it is read through, and the adapter file."""

from typing import ClassVar, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft.tuners._delta import _Home
from lucid.peft.tuners._files import _FileContext
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners._plan import _Target
from lucid.peft.tuners._shared import _base_linear, _SettlingTuner, _SharedTablesTuner
from lucid.peft.tuners._support import _adapter_dtype, _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.unilora._config import UniLoraConfig
from lucid.peft.tuners.unilora._indices import _assign, _generate_index
from lucid.peft.tuners.unilora._layer import _PROJECTION, UniLoraLayer, _new_theta_d


def _as_unilora(config: PeftConfig) -> UniLoraConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, UniLoraConfig)
    return config


class UniLoraModel(_SettlingTuner, _SharedTablesTuner, _ConfigTuner, peft_type=PeftType.UNILORA):
    r"""A model with Uni-LoRA layers put into it, the vector they all read, and the tools to take them out.

    Construction replaces each ``nn.Linear`` ``config`` selects with a Uni-LoRA layer around it and
    freezes every other parameter.  The model is changed in place and kept as ``model``; a public
    name this wrapper lacks is read from it.  ``config`` itself is not changed: ``peft_config`` holds
    a copy with the default ``target_modules`` spelled out.

    The adapter of the whole model is one trainable vector of ``theta_d_length`` entries per
    adapter, held here as ``unilora_theta_d`` (the reference's attribute).  The LoRA factors of
    every layer are read out of it through a fixed projection, drawn from ``proj_seed`` over all the
    layers of the adapter at once (:meth:`generate_index`).  The first layer holds the vector in its
    module tree, so a state dict lists it once and the adapter file holds one vector, with the
    indices and scales of each layer besides when ``save_indices`` is set.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : UniLoraConfig
        What to adapt and how.
    adapter_name : str
        Name of the adapter, the key in ``unilora_theta_d``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is not an ``nn.Linear`` or
        ``modules_to_save`` matches nothing.
    NotImplementedError
        The model already holds adapter layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import UniLoraConfig, UniLoraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> unilora = UniLoraModel(net, UniLoraConfig(r=2, theta_d_length=16, target_modules=["0", "2"]), "default")
    >>> tuple(unilora.unilora_theta_d["default"].shape)
    (16,)
    >>> net[0].base_layer.weight.requires_grad, unilora.unilora_theta_d["default"].requires_grad
    (False, True)
    >>> unilora.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = UniLoraLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = UniLoraConfig
    _table_names: ClassVar[tuple[str, ...]] = ("unilora_theta_d",)

    @override
    def __init__(
        self,
        model: nn.Module,
        config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        """Adapt ``model`` in place with ``config``, as adapter ``adapter_name``; the reference names the config ``config``."""
        super().__init__(model, config, adapter_name, low_cpu_mem_usage, state_dict)

    @property
    def unilora_theta_d(self) -> nn.ParameterDict:
        """The shared vectors, one ``(theta_d_length,)`` parameter per adapter (the reference's attribute)."""
        return self._shared.parameters_of("unilora_theta_d")

    @staticmethod
    def generate_index(lora_param_count: int, theta_d_length: int, proj_seed: int) -> Tensor:
        """The slot of the shared vector each of ``lora_param_count`` LoRA parameters reads.

        Balanced — each slot is read ``floor(D / d)`` or ``ceil(D / d)`` times — and shuffled by
        ``proj_seed``, the same on every device.  The values differ from the reference's, whose
        shuffle comes from a generator this library does not have: an adapter file written there
        reads its indices back only with ``save_indices=True``.

        Parameters
        ----------
        lora_param_count : int
            The number of LoRA parameters of the whole model, non-negative.
        theta_d_length : int
            The length of the shared vector, positive.
        proj_seed : int
            Seed of the shuffle.

        Returns
        -------
        Tensor
            ``(lora_param_count,)`` ``int64`` indices.

        Raises
        ------
        ValueError
            ``lora_param_count`` is negative or ``theta_d_length`` is not positive.
        """
        return _generate_index(lora_param_count, theta_d_length, proj_seed)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = _as_unilora(super()._prepare_config(config, model))
        _reset_fan_in_fan_out(stored)
        return stored

    # ── the shared vector ─────────────────────────────────────────────────

    @override
    @classmethod
    def _new_tables(cls, config: _TargetedConfig) -> dict[str, nn.Module]:
        del config  # the vector is a parameter table whatever the config
        return {"unilora_theta_d": nn.ParameterDict()}

    @override
    def _provide(self, adapter_name: str, config: _TargetedConfig, targets: list[_Target]) -> None:
        """Enter the vector ``adapter_name`` reads: one per adapter, drawn here and shared by the layers."""
        anchor = _base_linear(targets[0].module).weight
        unilora = _as_unilora(config)
        self.unilora_theta_d[adapter_name] = _new_theta_d(
            unilora, _Home(anchor.device, _adapter_dtype(anchor))
        )

    @override
    def _settle(self, config: _TargetedConfig, adapter_name: str, targets: list[_Target]) -> None:
        del targets  # the layers that hold the adapter are what it reads
        unilora = _as_unilora(config)
        # The reference's order: the layers as the model lists them.
        layers = [
            layer
            for _, layer in _tuner_layers(self.model)
            if isinstance(layer, UniLoraLayer) and adapter_name in layer._adapter_names()
        ]
        drawn = _assign(
            [layer.factor_shapes(adapter_name) for layer in layers],
            unilora.theta_d_length,
            unilora.proj_seed,
        )
        for layer, assignment in zip(layers, drawn, strict=True):
            layer.bind_projection(adapter_name, assignment)

    # ── the adapter file ──────────────────────────────────────────────────

    @override
    @classmethod
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        """The vector, as the file's one model-wide tensor, and each layer's projection when the adapter saves it (``save_indices``)."""
        keys = super()._shared_keys(context)
        for path, layer in context.layers:
            keys.update(context.layer_buffer_keys(path, layer, *_PROJECTION))
        return keys
