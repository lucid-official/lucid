"""``FrodModel``: the FRoD tuner — the basis its layers share, how it is made for each adapter, and how it is saved.

Everything that is not FRoD — planning, applying, freezing, switching, merging, unloading — is
``_BaseTuner``'s, in ``lucid.peft.tuners._tuner``.
"""

import warnings
from collections.abc import Mapping
from typing import Final, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType
from lucid.peft.tuners._files import _FileContext, _FileFormat
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners._plan import _Injected, _Placement, _Target
from lucid.peft.tuners._shared import _base_linear
from lucid.peft.tuners._support import _adapter_dtype, _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.frod._basis import (
    _category_from_key,
    _draw_pattern,
    _grouped_weights,
    _projection_from_weights,
)
from lucid.peft.tuners.frod._config import FrodConfig
from lucid.peft.tuners.frod._layer import FrodLayer
from lucid.peft.tuners.frod._tables import NAMES, _Projections

#: Where the reference's adapter file keeps a table: ``base_model.<table>.<category>``, one tensor per category.
_FILE_PREFIX: Final = "base_model."


def _as_frod(config: PeftConfig) -> FrodConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, FrodConfig)
    return config


def _file_key(table: str, category: str) -> str:
    return f"{_FILE_PREFIX}{table}.{category}"


class FrodModel(_FileFormat, _ConfigTuner[FrodConfig], peft_type=PeftType.FROD):
    r"""A network whose linear layers are re-expressed over a basis they share, and the tools to undo that.

    Each ``nn.Linear`` that ``config`` selects is replaced by a :class:`~lucid.peft.tuners.frod.Linear` whose
    weight is :math:`U(\Sigma + S)V^\top`, with :math:`V` one basis for every layer of the same role in the
    network (the query projections of all blocks, say), :math:`U` frozen, and :math:`\Sigma` and the few
    off-diagonal entries of :math:`S` trained; every other parameter is frozen.  The network is edited in
    place and stays reachable as ``model``.  Adapters added later (``add_adapter``, ``set_adapter``,
    ``delete_adapter``) each make a basis of their own from the same key, and the active ones add their
    changes to the weight.

    The bases are held per category, as the reference's ``frod_V``, ``frod_s_indices`` and ``frod_s_size``
    tables of containers with one tensor per adapter; the first layer holds them in its module tree, so a
    state dict lists each once and the adapter file one tensor per category and table (``save_projection``).

    Parameters
    ----------
    model : nn.Module
        The network to adapt.
    peft_config : FrodConfig
        The sparse rate, the key the positions are drawn from, and which modules.
    adapter_name : str
        What the first adapter is called: the key of every ``frod_lambda_l`` entry.
    low_cpu_mem_usage, state_dict
        Options of the reference this library cannot honour; a value other than the default raises, as
        for :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``; a match is not an ``nn.Linear``; the layers of one category
        differ in input width; adapters disagree on ``projection_prng_key`` or ``save_projection``; or
        ``modules_to_save`` matches nothing.
    NotImplementedError
        The network already holds layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import FrodConfig, FrodModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> frod = FrodModel(net, FrodConfig(sparse_rate=0.1, target_modules=["0"]), "default")
    >>> frod.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].frod_lambda_l["default"].requires_grad
    (False, True)
    >>> frod.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = FrodLayer._layer_types

    # ── the adapters ──────────────────────────────────────────────────────

    @override
    def _inject(self, adapter_name: str, config: FrodConfig) -> _Injected:
        if "_projections" not in self.__dict__:
            # A model adapted before keeps the tables its layers read: the new adapter's go beside them.
            held = [
                layer._projections
                for _, layer in _tuner_layers(self.model)
                if isinstance(layer, FrodLayer) and layer._projections is not None
            ]
            self._projections = held[0] if held else _Projections()
        return super()._inject(adapter_name, config)

    @override
    @classmethod
    def _prepare_config(cls, config: FrodConfig, model: nn.Module) -> FrodConfig:
        stored = super()._prepare_config(config, model)
        _reset_fan_in_fan_out(stored)
        return stored

    @override
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        frod = _as_frod(config)
        for held in self.peft_config.values():
            if _as_frod(held).projection_prng_key != frod.projection_prng_key:
                raise ValueError(
                    "FRoD projection initialization key must be the same for all adapters. Got "
                    f"{frod.projection_prng_key=} but previous config had "
                    f"{_as_frod(held).projection_prng_key}."
                )
        saved = sorted({_as_frod(c).save_projection for c in [*self.peft_config.values(), frod]})
        if len(saved) > 1:
            raise ValueError(
                "FRoD projection weights must be saved for all adapters or none, but got "
                f"multiple different values: {saved}"
            )

    @override
    @classmethod
    def _create_new_module(
        cls, config: FrodConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        layer = target.layer
        # Planning took it from ``FrodLayer``'s table.
        assert layer is not None and issubclass(layer, FrodLayer)
        # The basis the layer reads is the tuner's, made before and bound after: nothing is derived yet.
        return layer(target.module, adapter_name, config, deferred=True)

    # ── the shared bases ──────────────────────────────────────────────────

    def _provide(self, adapter_name: str, config: FrodConfig, targets: list[_Target]) -> None:
        """Make ``adapter_name``'s basis and positions for every category among ``targets``, and enter them.

        The category's weights are stacked block by block (``_grouped_weights``) and the basis is made
        from the stack; the positions are drawn from ``projection_prng_key``.
        """
        weights = [(target.key, _base_linear(target.module).weight) for target in targets]
        example = {_category_from_key(key): weight for key, weight in weights}
        for category, matrices in _grouped_weights(weights).items():
            widths = {int(matrix.shape[1]) for matrix in matrices}
            if len(widths) > 1:
                raise ValueError(
                    f"FRoD shares one basis among the layers of a category, but those of "
                    f"{category!r} take inputs of widths {sorted(widths)}"
                )
            anchor = example[category]
            width = widths.pop()
            basis = _projection_from_weights(matrices, config.regularization_alpha)
            indices, size = _draw_pattern(
                width, config.sparse_rate, config.projection_prng_key, anchor.device
            )
            self._projections.put(
                category,
                adapter_name,
                (basis.to(device=anchor.device, dtype=_adapter_dtype(anchor)), indices, size),
                persistent=config.save_projection,
            )

    @override
    def _new_slots(
        self, config: FrodConfig, adapter_name: str, targets: list[_Target], saved: list[str]
    ) -> list[_Placement]:
        self._provide(adapter_name, config, targets)
        try:
            placements = super()._new_slots(config, adapter_name, targets, saved)
            for placement in placements:
                if isinstance(placement.slot, FrodLayer):
                    placement.slot.bind(self._projections, _category_from_key(placement.key))
        except BaseException:
            self._projections.drop(adapter_name)
            raise
        return placements

    @override
    def _withdraw(self, adapter_name: str, placements: list[_Placement]) -> None:
        super()._withdraw(adapter_name, placements)
        self._projections.drop(adapter_name)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name`` from the model, and its entries in the shared tables."""
        super().delete_adapter(adapter_name)
        self._projections.drop(adapter_name)

    # ── the adapter file ──────────────────────────────────────────────────

    @override
    @classmethod
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        """The bases, as the file's tensors per category and table; none when the adapter does not save them."""
        if not _as_frod(context.config).save_projection:
            return {}
        for path, layer in context.layers:
            if not (isinstance(layer, FrodLayer) and layer.holds_tables()):
                continue
            projections = layer._bound()
            keys: dict[str, str] = {}
            for category in projections.categories():
                if not projections.has(category, context.adapter):
                    continue
                if not projections.persistent(category):
                    raise ValueError(
                        "Model was initialised to not save the FRoD projections but config now "
                        "specifies to save projection! Set `config.save_projection` to `False`."
                    )
                for name in NAMES:
                    memory = f"{context.prefix}{path}.{name}.{category}.{context.adapter}"
                    keys[memory] = _file_key(name, category)
            return keys
        return {}

    @classmethod
    def _adopt(cls, context: _FileContext, tables: Mapping[str, Tensor]) -> None:
        """Make the bases of the file the model's, and the layers' frozen factors from them.

        A layer's frozen factor is derived from its weight and the basis, so a basis that is not the
        one the layer was made with (the file's, drawn on another machine) needs the factor again.
        """
        owners = [
            layer
            for _, layer in context.layers
            if isinstance(layer, FrodLayer) and layer.holds_tables()
        ]
        if not owners:
            return
        changed = cls._replace_bases(owners[0]._bound(), context.adapter, tables)
        for _, layer in context.layers:
            if (
                isinstance(layer, FrodLayer)
                and layer._category in changed
                and context.adapter in layer._adapter_names()
            ):
                layer._restart(context.adapter)

    @staticmethod
    def _replace_bases(
        projections: _Projections, adapter: str, tables: Mapping[str, Tensor]
    ) -> set[str]:
        """Put the file's ``(V, indices, size)`` of every category it has in ``adapter``'s entries; the categories changed."""
        changed: set[str] = set()
        for category in projections.categories():
            files = [tables.get(_file_key(name, category)) for name in NAMES]
            if not projections.has(category, adapter) or None in files:
                continue
            held = projections.entry(category, adapter)
            v, indices, size = (
                file.to(device=current.device, dtype=current.dtype)
                for file, current in zip(files, held, strict=True)
                if file is not None
            )
            projections.put(category, adapter, (v, indices, size), projections.persistent(category))
            changed.add(category)
        return changed

    @override
    @classmethod
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Require the bases when the adapter saves them and make them the model's; drop them (with a warning) when it does not."""
        tables = {
            key: value
            for key, value in state.items()
            if any(key.startswith(f"{_FILE_PREFIX}{name}.") for name in NAMES)
        }
        if _as_frod(context.config).save_projection:
            if not tables:
                raise ValueError(
                    "Specified to load FRoD projection tensors from state dictionary however they "
                    "were not present. If this checkpoint was saved with `save_projection=False`, "
                    "set `peft_config.save_projection` to `False` before loading so the projections "
                    "are regenerated from the base model weights. Otherwise, re-save the adapter "
                    "with `save_projection=True` to include these tensors."
                )
            cls._adopt(context, tables)
            return state
        if tables:
            warnings.warn(
                "Specified to not load FRoD projection tensors from state dictionary however they "
                "are present. Consider using them to ensure checkpoint loading is correct by "
                "setting `peft_config.save_projection = True`.",
                stacklevel=4,
            )
        return {key: value for key, value in state.items() if key not in tables}
