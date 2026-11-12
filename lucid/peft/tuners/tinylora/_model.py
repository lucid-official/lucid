"""``TinyLoraModel``: the TinyLoRA tuner — the vectors its layers share, which module reads which, and the adapter file."""

import warnings
from collections.abc import Mapping
from typing import ClassVar, override

from lucid._tensor.tensor import Tensor
from lucid.nn import Module, ModuleDict, ParameterDict
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft.tuners._files import _FileContext
from lucid.peft.tuners._plan import _Target
from lucid.peft.tuners._shared import _FILE_PREFIX, _SettlingTuner, _SharedTablesTuner
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.tinylora._config import TinyLoraConfig
from lucid.peft.tuners.tinylora._layer import TinyLoraLayer, _new_vector

#: Where the reference's file keeps the trained vectors: one tensor per group, ``base_model.tinylora_v.<group>``.
_VECTORS = f"{_FILE_PREFIX}tinylora_v."


def _as_tinylora(config: PeftConfig) -> TinyLoraConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, TinyLoraConfig)
    return config


def _groups(count: int, weight_tying: float) -> tuple[int, int]:
    """``(number of groups, modules per group)`` for ``count`` modules whose tying is ``weight_tying``."""
    groups = max(1, round(count * (1.0 - weight_tying)))
    return groups, max(1, count // groups)


class TinyLoraModel(_SettlingTuner, _SharedTablesTuner, _ConfigTuner, peft_type=PeftType.TINYLORA):
    r"""A model with TinyLoRA layers put into it, the vectors they train, and the tools to take them out.

    Construction replaces each ``nn.Linear`` or ``nn.Embedding`` ``config`` selects with a TinyLoRA layer
    around it and freezes every other parameter.  The model is changed in place and kept as ``model``;
    a public name this wrapper lacks is read from it.  ``config`` itself is not changed:
    ``peft_config`` holds a copy with the default ``target_modules`` spelled out.

    The only trained tensors are the vectors of ``u`` numbers, held here as ``tinylora_v`` (the
    reference's attribute): per adapter, one per *group* of modules.  The adapter's targets, in the
    order the model lists them, are cut into ``max(1, round(n (1 - weight_tying)))`` consecutive groups
    and a group's modules read one vector; at ``weight_tying=1.0`` the whole model reads one.  The first
    layer holds the vectors in its module tree, so a state dict lists each once and the adapter file
    holds ``base_model.tinylora_v.<group>`` beside each layer's frozen ``tinylora_A`` and ``tinylora_B``
    (and ``tinylora_P`` when ``save_projection`` is set).

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : TinyLoraConfig
        What to adapt and how.
    adapter_name : str
        Name of the adapter, the key in ``tinylora_v``.
    low_cpu_mem_usage : bool, default=False
        As in :class:`~lucid.peft.LoraModel`.
    **kwargs
        ``state_dict``, as in :class:`~lucid.peft.LoraModel`; the reference's constructor takes its
        options among the keywords.

    Raises
    ------
    TypeError
        A keyword other than ``state_dict`` is given.
    ValueError
        No module matches ``target_modules``, a match is not an ``nn.Linear`` or ``nn.Embedding``, or
        ``modules_to_save`` matches nothing.
    NotImplementedError
        Layers of another method are already in the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import TinyLoraConfig, TinyLoraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> config = TinyLoraConfig(r=2, u=3, weight_tying=1.0, target_modules=["0", "2"])
    >>> tiny = TinyLoraModel(net, config, "default")
    >>> {k: tuple(v.shape) for k, v in tiny.tinylora_v["default"].items()}
    {'0': (3,)}
    >>> tiny.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = TinyLoraLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = TinyLoraConfig
    _table_names: ClassVar[tuple[str, ...]] = ("tinylora_v",)

    @override
    def __init__(
        self,
        model: Module,
        config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        **kwargs: object,
    ) -> None:
        """Adapt ``model`` in place with ``config``, as adapter ``adapter_name``; the reference names the config ``config`` and takes its other options as keywords."""
        state_dict = kwargs.pop("state_dict", None)
        if kwargs:
            raise TypeError(f"TinyLoraModel got unexpected keyword arguments {sorted(kwargs)}")
        if state_dict is not None and not isinstance(state_dict, dict):
            raise TypeError(
                f"state_dict must be a dict of tensors, got {type(state_dict).__name__}"
            )
        super().__init__(model, config, adapter_name, low_cpu_mem_usage, state_dict)

    @property
    def tinylora_v(self) -> ModuleDict[Module]:
        """The trained vectors: per adapter, a ``ParameterDict`` of one ``(u,)`` parameter per group of modules."""
        table = self._shared.tables["tinylora_v"]
        assert isinstance(table, ModuleDict)
        return table

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: Module) -> _TargetedConfig:
        stored = _as_tinylora(super()._prepare_config(config, model))
        _reset_fan_in_fan_out(stored)
        return stored

    # ── the shared vectors ────────────────────────────────────────────────

    @override
    @classmethod
    def _new_tables(cls, config: _TargetedConfig) -> dict[str, Module]:
        del config  # the vectors are a table of parameters whatever the config
        return {"tinylora_v": ModuleDict()}

    @override
    def _provide(self, adapter_name: str, config: _TargetedConfig, targets: list[_Target]) -> None:
        """Open ``adapter_name``'s table of vectors; ``_attach_adapter`` fills it, once every layer exists."""
        del config, targets
        self.tinylora_v[adapter_name] = ParameterDict()

    @override
    def _drop(self, adapter_name: str) -> None:
        if adapter_name in self.tinylora_v:
            del self.tinylora_v[adapter_name]

    @override
    def _settle(self, config: _TargetedConfig, adapter_name: str, targets: list[_Target]) -> None:
        tiny = _as_tinylora(config)
        groups, size = _groups(len(targets), tiny.weight_tying)
        vectors = self.tinylora_v[adapter_name]
        assert isinstance(vectors, ParameterDict)
        for index, target in enumerate(targets):
            layer = self.model.get_submodule(target.key)
            assert isinstance(layer, TinyLoraLayer)
            group = str(min(index // size, groups - 1))
            if group not in vectors:
                vectors[group] = _new_vector(tiny, layer._home())
            layer.place(adapter_name, index, group)

    @override
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        saved = sorted(
            {_as_tinylora(c).save_projection for c in [*self.peft_config.values(), config]}
        )
        if len(saved) > 1:
            raise ValueError(
                "TinyLoRA projection tensors must be saved for all adapters or none, but got "
                f"multiple different values: {saved}"
            )

    # ── the adapter file ──────────────────────────────────────────────────

    @override
    @classmethod
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        """Each group's vector as ``base_model.tinylora_v.<group>``, and each layer's frozen factors (and projection, when saved)."""
        config = _as_tinylora(context.config)
        keys: dict[str, str] = {}
        owner_found = False
        for path, layer in context.layers:
            if (
                not isinstance(layer, TinyLoraLayer)
                or context.adapter not in layer._adapter_names()
            ):
                continue
            where = f"{context.prefix}{path}"
            held = getattr(layer, "tinylora_v", None)
            if held is not None and not owner_found and context.adapter in held:
                owner_found = True
                for group in held[context.adapter]:
                    keys[f"{where}.tinylora_v.{context.adapter}.{group}"] = f"{_VECTORS}{group}"
            names = ("tinylora_A", "tinylora_B") + (
                ("tinylora_P",) if config.save_projection else ()
            )
            keys.update({f"{where}.{name}.{context.adapter}": f"{where}.{name}" for name in names})
        return keys

    @override
    @classmethod
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Require the trained vectors; settle the projection as the config says, warning when it is regenerated or ignored."""
        if not any(key.startswith(_VECTORS) for key in state):
            raise ValueError(
                "Specified to load tinylora_v from state dictionary however they were not present!"
            )
        projected = {key for key in state if key.rpartition(".")[2] == "tinylora_P"}
        if _as_tinylora(context.config).save_projection:
            if not projected:
                warnings.warn(
                    "Specified to load tinylora_P from state dictionary however it was not "
                    "present! Projection tensors will be regenerated from the projection_seed.",
                    stacklevel=4,
                )
            return state
        if projected:
            warnings.warn(
                "Specified to not load tinylora_P from state dictionary however they are present "
                "in state dictionary! Consider using them to ensure checkpoint loading is correct "
                "on all platforms using `peft_config.save_projection = True`",
                stacklevel=4,
            )
        else:
            warnings.warn(
                "Specified to not load tinylora_P from state dictionary. This means we will be "
                "relying on PRNG initialisation to restore these projections using "
                "`config.projection_seed`, which may not be accurate on all system configurations.",
                stacklevel=4,
            )
        return {key: value for key, value in state.items() if key not in projected}
