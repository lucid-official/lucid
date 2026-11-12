"""``TrainableTokensModel``: the trainable-tokens tuner — its config and layer family, and the table it targets by default."""

import copy
from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft._refusals import NO_STATE_DICT_MATCHING, refuse_value, reject_unexpected
from lucid.peft._targeting import _parameter_paths, _shared_outside
from lucid.peft.tuners._plan import _Placement, _Target
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.trainable_tokens._config import TrainableTokensConfig
from lucid.peft.tuners.trainable_tokens._layer import TrainableTokensLayer

#: The module name an embedding is looked for under when the model names none.
_FALLBACK_NAME = "embed_tokens"


def _input_embedding_name(model: nn.Module) -> str:
    """The path of ``model``'s input embedding, found by the model's ``get_input_embeddings``; else ``embed_tokens``."""
    table = getattr(model, "get_input_embeddings", lambda: None)()
    for name, module in model.named_modules():
        if name and module is table:
            return name
    return _FALLBACK_NAME


class TrainableTokensModel(_ConfigTuner, peft_type=PeftType.TRAINABLE_TOKENS):
    r"""A model whose embedding table has a trainable copy of some rows, and the tools to take it out again.

    The module ``config`` selects (the model's input embedding by default) becomes an
    :class:`~lucid.peft.tuners.trainable_tokens.Embedding` around it; everything else is
    frozen.  The model is edited in place (reachable as ``model``; public names this wrapper
    lacks are read from it) and ``peft_config`` keeps ``config`` with the default target
    spelled out.  Adapters coexist under names (``add_adapter``, ``set_adapter``,
    ``delete_adapter``); those active at once must name different tokens.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    peft_config : TrainableTokensConfig
        Which tokens, and which table.
    adapter_name : str
        Name of the adapter, the key in every ``trainable_tokens_delta``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` selects nothing or a module that is not an embedding or
        ``nn.Linear``, ``token_indices`` is empty or outside the table.
    NotImplementedError
        Layers of another method are already in the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import TrainableTokensConfig, TrainableTokensModel
    >>> net = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 2))
    >>> config = TrainableTokensConfig(token_indices=[1, 2], target_modules=["0"])
    >>> tokens = TrainableTokensModel(net, config, "default")
    >>> tuple(net[0].trainable_tokens_delta["default"].shape)
    (2, 4)
    >>> net[0].base_layer.weight.requires_grad, net[0].trainable_tokens_delta["default"].requires_grad
    (False, True)
    >>> tokens.unload() is net, type(net[0]) is nn.Embedding
    (True, True)
    """

    _layer_types = TrainableTokensLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = TrainableTokensConfig

    @override
    def inject_adapter(  # type: ignore[override]  # reference parity: its override takes **kwargs where the base takes state_dict
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        **kwargs: object,
    ) -> None:
        """Build the layers of adapter ``adapter_name`` into ``model``, as :meth:`_BaseTuner.inject_adapter` does.

        The reference's override takes the base's ``state_dict`` among ``**kwargs`` and, after
        injecting, ties the adapters of the modules that share the embedding's weight to the
        embedding's.  Here ``state_dict`` is the only keyword and is refused unless ``None``
        (as in the base), and a module sharing the embedding's weight is refused while the
        layers are built, so there is nothing to tie afterwards.

        Parameters
        ----------
        model : nn.Module
            The model to adapt; this tuner's own ``model``.
        adapter_name : str
            Name of the adapter, with its config in ``peft_config``.
        autocast_adapter_dtype : bool, default=True
            ``False`` raises ``NotImplementedError``.
        low_cpu_mem_usage : bool, default=False
            ``True`` raises ``NotImplementedError``.
        **kwargs : object
            ``state_dict`` alone, which must be ``None``.

        Raises
        ------
        TypeError
            A keyword other than ``state_dict`` is given.
        NotImplementedError
            ``state_dict``, ``autocast_adapter_dtype`` or ``low_cpu_mem_usage`` holds a value the
            library cannot honour, or the target shares its weight with another module.
        ValueError
            As :meth:`_BaseTuner.inject_adapter`.
        """
        where = f"{type(self).__name__}.inject_adapter"
        state_dict = kwargs.pop("state_dict", None)
        reject_unexpected(where, kwargs)
        refuse_value(where, "state_dict", state_dict, None, NO_STATE_DICT_MATCHING)
        super().inject_adapter(model, adapter_name, autocast_adapter_dtype, low_cpu_mem_usage)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = copy.deepcopy(config)
        if stored.target_modules is None:
            stored.target_modules = {_input_embedding_name(model)}
        return stored

    @override
    def _new_slots(
        self, config: _TargetedConfig, adapter_name: str, targets: list[_Target], saved: list[str]
    ) -> list[_Placement]:
        placements = super()._new_slots(config, adapter_name, targets, saved)
        paths = _parameter_paths(self.model)
        for placement in placements:
            if isinstance(placement.slot, TrainableTokensLayer):
                tied = _shared_outside(paths, placement.key, placement.slot.get_base_layer())
                if tied:
                    raise NotImplementedError(
                        f"{placement.key} shares its weight with {', '.join(tied)}: trained rows "
                        "would reach the embedding but not that module (the reference ties them "
                        "with a second adapter, which lucid.peft does not).  Untie the weight "
                        "first (tie_word_embeddings=False)."
                    )
        return placements
