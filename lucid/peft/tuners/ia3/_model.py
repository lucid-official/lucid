"""``IA3Model``: the (IA)^3 half of a tuner — which layers are feed-forward, and weighted adapters.

Everything that is not (IA)^3 — planning, applying, freezing, switching,
merging, unloading — is ``_BaseTuner``'s, in ``lucid.peft.tuners._base``.
"""

import dataclasses
import re
import warnings
from collections.abc import Mapping
from typing import ClassVar, override

import lucid
import lucid.nn as nn
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft._targeting import _suffix_match
from lucid.peft.tuners._base import _BaseTuner, _Target, _tuner_layers
from lucid.peft.tuners.ia3._config import IA3Config
from lucid.peft.tuners.ia3._defaults import IA3_DEFAULT_TARGET_MODULES, _default_feedforward_modules
from lucid.peft.tuners.ia3._layer import IA3Layer


def _as_ia3(config: PeftConfig) -> IA3Config:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, IA3Config)
    return config


def _is_feedforward(config: IA3Config, key: str) -> bool:
    """Whether the module named ``key`` scales its input under ``config``."""
    names = config.feedforward_modules
    if isinstance(names, str):
        return re.fullmatch(names, key) is not None
    return names is not None and _suffix_match(key, names)


class IA3Model(_BaseTuner, peft_type=PeftType.IA3):
    r"""A model with (IA)^3 layers put into it, and the tools to take them out.

    Construction replaces each module ``config`` selects with an (IA)^3 layer
    around it — :class:`~lucid.peft.tuners.ia3.Linear`, ``Conv2d`` or
    ``Conv3d``, by the module's type — an output-side one, or a feed-forward
    one for a module ``feedforward_modules`` names, and freezes every other
    parameter.  The model is changed in place and kept as ``model``; a public
    name this wrapper lacks is read from it.  ``config`` itself is not
    changed: ``peft_config`` holds a copy with the default ``target_modules`` /
    ``feedforward_modules`` and ``"all-linear"`` spelled out.

    The model can hold any number of adapters, each under its own name
    (``add_adapter``, ``set_adapter``, ``delete_adapter``);
    ``add_weighted_adapter`` makes one out of others by a weighted sum of
    their vectors.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : IA3Config
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``ia3_l``.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is a module type with
        no (IA)^3 layer, ``modules_to_save`` matches nothing, or the model type
        has no default ``target_modules`` / ``feedforward_modules`` and the
        config names none.
    NotImplementedError
        The model already holds adapter layers (wrap it once; add
        adapters with ``add_adapter``).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import IA3Config, IA3Model
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> ia3 = IA3Model(net, IA3Config(target_modules=["0", "2"], feedforward_modules=["2"]))
    >>> ia3.targeted_module_names
    ['0', '2']
    >>> tuple(net[0].ia3_l["default"].shape), tuple(net[2].ia3_l["default"].shape)
    ((8, 1), (1, 8))
    >>> ia3.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = IA3Layer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = IA3Config
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = IA3_DEFAULT_TARGET_MODULES

    def __init__(self, model: nn.Module, config: IA3Config, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = _as_ia3(super()._prepare_config(config, model))
        if stored.fan_in_fan_out:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `nn.Linear`. "
                "Setting fan_in_fan_out to False.",
                stacklevel=5,
            )
            stored.fan_in_fan_out = False
        if stored.feedforward_modules is None:
            stored.feedforward_modules = set(_default_feedforward_modules(model))
        return stored

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        layer = target.layer
        # Planning took it from ``IA3Layer``'s table.
        assert layer is not None and issubclass(layer, IA3Layer)
        ia3 = _as_ia3(config)
        return layer(
            target.module, adapter_name, ia3, is_feedforward=_is_feedforward(ia3, target.key)
        )

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        layer = target.module
        assert isinstance(layer, IA3Layer)
        ia3 = _as_ia3(config)
        if layer.is_feedforward != _is_feedforward(ia3, target.key):
            raise ValueError(
                f"{target.key!r} is {'' if layer.is_feedforward else 'not '}a feed-forward module "
                "for the adapters it already holds but the new config says otherwise; a layer "
                "scales either its input or its output"
            )
        layer.update_layer(adapter_name, ia3)

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        del model, config  # an (IA)^3 adapter trains its vectors only
        return []

    # ── weighted adapters ─────────────────────────────────────────────────

    def add_weighted_adapter(
        self, adapters: list[str], weights: list[float], adapter_name: str
    ) -> None:
        r"""Add an adapter whose vectors are the weighted sum of other adapters' vectors.

        The new adapter is frozen and inactive, like any added adapter.  It
        targets the union of the sources' modules (and feed-forward modules);
        a layer one source does not reach sums the sources it has.

        Parameters
        ----------
        adapters : list of str
            Names of the adapters to combine.
        weights : list of float
            One weight per adapter: :math:`\ell = \sum_i w_i \ell_i`.
        adapter_name : str
            Name of the new adapter.

        Raises
        ------
        ValueError
            The name is taken, a source does not exist, the weights do not
            match the sources, or two train one ``modules_to_save`` module.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> from lucid.peft import IA3Config, IA3Model
        >>> net = nn.Sequential(nn.Linear(6, 5))
        >>> config = IA3Config(target_modules=["0"], feedforward_modules=[], init_ia3_weights=False)
        >>> ia3 = IA3Model(net, config, adapter_name="a")
        >>> ia3.add_adapter("b", config)
        >>> ia3.add_weighted_adapter(["a", "b"], [0.5, 0.5], "mix")
        >>> sorted(ia3.peft_config)
        ['a', 'b', 'mix']
        """
        adapter_name = self._check_weighted_sources(adapters, weights, adapter_name)
        # Exact names of the modules the sources reach: no regex-versus-names mix to reconcile,
        # and a module keeps the side (input or output) its layer already scales.
        reached = [
            (key, layer)
            for key, layer in _tuner_layers(self.model)
            if isinstance(layer, IA3Layer) and any(name in layer.ia3_l for name in adapters)
        ]
        union = dataclasses.replace(
            _as_ia3(self.peft_config[adapters[0]]),
            target_modules={key for key, _ in reached},
            feedforward_modules={key for key, layer in reached if layer.is_feedforward},
            exclude_modules=None,
            layers_to_transform=None,
            layers_pattern=None,
        )
        self._add_combined_adapter(
            adapter_name, union, lambda: self._sum_vectors(adapter_name, adapters, weights)
        )

    def _sum_vectors(self, adapter_name: str, adapters: list[str], weights: list[float]) -> None:
        """Write into every layer the new adapter reaches the weighted sum of the sources' vectors."""
        with lucid.no_grad():
            for _, layer in _tuner_layers(self.model):
                if not isinstance(layer, IA3Layer) or adapter_name not in layer.ia3_l:
                    continue
                total = lucid.zeros_like(layer._vector(adapter_name))
                for source, weight in zip(adapters, weights, strict=True):
                    if source in layer.ia3_l:
                        total = total + layer._vector(source) * weight
                layer._vector(adapter_name).copy_(total)
