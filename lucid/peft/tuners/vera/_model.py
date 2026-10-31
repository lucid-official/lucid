"""``VeraModel``: the VeRA half of a tuner — the projections its layers share, and how they are saved.

Everything that is not VeRA — planning, applying, freezing, switching,
merging, unloading — is ``_BaseTuner``'s, in ``lucid.peft.tuners._base``.
"""

import warnings
from collections.abc import Mapping
from typing import ClassVar, Final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft.tuners._base import (
    _NO_PARAMETERS,
    _BaseTuner,
    _BaseTunerLayer,
    _Placement,
    _Target,
)
from lucid.peft.tuners._files import _FileContext, _FileFormat
from lucid.peft.tuners._support import _adapter_dtype, _reset_fan_in_fan_out
from lucid.peft.tuners.vera._config import VeraConfig
from lucid.peft.tuners.vera._defaults import VERA_DEFAULT_TARGET_MODULES
from lucid.peft.tuners.vera._layer import VeraLayer
from lucid.peft.tuners.vera._projection import _BufferDict, _Projection, _Shared

#: Where the reference's adapter file keeps the shared projections: one tensor each, for the whole model.
_FILE_A: Final = "base_model.vera_A"
_FILE_B: Final = "base_model.vera_B"


def _as_vera(config: PeftConfig) -> VeraConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, VeraConfig)
    return config


def _linear_of(module: nn.Module) -> nn.Linear:
    """The ``nn.Linear`` a target is: itself, or what the layer already around it wraps."""
    base = module.get_base_layer() if isinstance(module, _BaseTunerLayer) else module
    # Planning has refused every other module type.
    assert isinstance(base, nn.Linear)
    return base


def _covers(projection: tuple[Tensor, Tensor], rank: int, layers: list[VeraLayer]) -> bool:
    """Whether ``(A, B)`` are rank-``rank`` matrices big enough to be sliced for every one of ``layers``."""
    vera_a, vera_b = projection
    return (
        vera_a.ndim == 2
        and vera_b.ndim == 2
        and vera_a.shape[0] == rank
        and vera_b.shape[1] == rank
        and all(
            vera_a.shape[1] >= layer.in_features and vera_b.shape[0] >= layer.out_features
            for layer in layers
        )
    )


class VeraModel(_FileFormat, _BaseTuner, peft_type=PeftType.VERA):
    r"""A model with VeRA layers put into it, the projections they share, and the tools to take them out.

    Construction replaces each ``nn.Linear`` ``config`` selects with a VeRA
    layer around it and freezes every other parameter.  The model is changed in
    place and kept as ``model``; a public name this wrapper lacks is read from
    it.  ``config`` itself is not changed: ``peft_config`` holds a copy with the
    default ``target_modules`` spelled out.

    The two random projections :math:`A` and :math:`B` are drawn once, sized for
    the largest targeted layer, and held here as ``vera_A`` / ``vera_B``
    (containers of one tensor per adapter, aliased when adapters share them) —
    the reference's attributes.  Every layer reads them from there; the first
    layer holds them in its module tree, so a state dict lists each once
    (``<layer>.vera_A.<adapter>``) and the adapter file holds one pair.
    Adapters added later must fit them (the same ``projection_prng_key`` and
    ``save_projection``, a rank and layer sizes no larger).

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : VeraConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``vera_lambda_*`` table.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is not an ``nn.Linear``,
        ``modules_to_save`` matches nothing, or an adapter does not fit the
        projections of the first.
    NotImplementedError
        The model already holds adapter layers (wrap it once; add adapters
        with ``add_adapter``).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.vera import VeraConfig, VeraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> vera = VeraModel(net, VeraConfig(r=4, target_modules=["0", "2"]))
    >>> tuple(vera.vera_A["default"].shape), tuple(vera.vera_B["default"].shape)
    ((4, 8), (8, 4))
    >>> net[0].base_layer.weight.requires_grad, net[0].vera_lambda_d["default"].requires_grad
    (False, True)
    >>> vera.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = VeraLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = VeraConfig
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = VERA_DEFAULT_TARGET_MODULES
    _default_parameters: ClassVar[Mapping[str, frozenset[str]]] = _NO_PARAMETERS

    def __init__(self, model: nn.Module, config: VeraConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    @property
    def vera_A(self) -> _BufferDict:
        """The shared ``A`` projections, one entry per adapter (the reference's attribute)."""
        return self._projections.vera_a

    @property
    def vera_B(self) -> _BufferDict:
        """The shared ``B`` projections, one entry per adapter (the reference's attribute)."""
        return self._projections.vera_b

    # ── the adapters ──────────────────────────────────────────────────────

    @override
    def _inject(self, adapter_name: str, config: _TargetedConfig) -> None:
        if "_projections" not in self.__dict__:
            self._projections = _Shared(_as_vera(config).save_projection)
        super()._inject(adapter_name, config)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = _as_vera(super()._prepare_config(config, model))
        _reset_fan_in_fan_out(stored)
        return stored

    @override
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        vera = _as_vera(config)
        for held in self.peft_config.values():
            if _as_vera(held).projection_prng_key != vera.projection_prng_key:
                raise ValueError(
                    "Vera PRNG initialisation key must be the same for all adapters. Got "
                    f"{vera.projection_prng_key=} but previous config had "
                    f"{_as_vera(held).projection_prng_key}."
                )
        saved = sorted({_as_vera(c).save_projection for c in [*self.peft_config.values(), vera]})
        if len(saved) > 1:
            raise ValueError(
                "VeRA projection weights must be saved for all adapters or none, but got "
                f"multiple different values: {saved}"
            )

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        layer = target.layer
        # Planning took it from ``VeraLayer``'s table.
        assert layer is not None and issubclass(layer, VeraLayer)
        vera = _as_vera(config)
        return layer(target.module, adapter_name, vera, r=vera.r)

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        layer = target.module
        assert isinstance(layer, VeraLayer)
        vera = _as_vera(config)
        layer.update_layer(adapter_name, vera.r, vera)

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        del model, config  # a VeRA adapter trains its vectors only
        return []

    # ── the shared projections ────────────────────────────────────────────

    def _provide(self, adapter_name: str, config: VeraConfig, targets: list[_Target]) -> None:
        """Enter the projections ``adapter_name`` reads: drawn for the first adapter, the first's shared with the others.

        Sized for the largest of ``targets``.  A later adapter must fit what
        the first drew — layer sizes and rank no larger.
        """
        linears = [_linear_of(target.module) for target in targets]
        out_size = max(linear.out_features for linear in linears)
        in_size = max(linear.in_features for linear in linears)
        if len(self.vera_A) == 0:
            anchor = linears[0].weight
            self.vera_A[adapter_name], self.vera_B[adapter_name] = _Projection.draw(
                config.projection_prng_key,
                config.r,
                out_size,
                in_size,
                anchor.device,
                _adapter_dtype(anchor),
            )
            return
        # Any held adapter's: they are all the same tensors, drawn from one key.
        held = next(iter(self.vera_A))
        vera_a, vera_b = self.vera_A[held], self.vera_B[held]
        too_small = "{} has a size of {} but {} or greater is required; this probably happened because an additional VeRA adapter was added after the first one with incompatible shapes."
        for label, size, need in (
            ("vera_A", vera_a.shape[1], in_size),
            ("vera_B", vera_b.shape[0], out_size),
        ):
            if size < need:
                raise ValueError(too_small.format(label, size, need))
        if vera_a.shape[0] < config.r:
            raise ValueError(
                f"vera_A has a size of {vera_a.shape[0]} but {config.r} or greater is required; "
                "this probably happened because an additional VeRA adapter with a higher rank "
                "was added after the first one; loading the adapters in reverse order may solve "
                "this."
            )
        self.vera_A[adapter_name], self.vera_B[adapter_name] = vera_a, vera_b

    def _drop_projection(self, adapter_name: str) -> None:
        for held in (self.vera_A, self.vera_B):
            if adapter_name in held:
                del held[adapter_name]

    @override
    def _new_slots(
        self, config: _TargetedConfig, adapter_name: str, targets: list[_Target], saved: list[str]
    ) -> list[_Placement]:
        self._provide(adapter_name, _as_vera(config), targets)
        try:
            placements = super()._new_slots(config, adapter_name, targets, saved)
        except BaseException:
            self._drop_projection(adapter_name)
            raise
        for placement in placements:
            if isinstance(placement.slot, VeraLayer):
                placement.slot.bind(self._projections)
        return placements

    @override
    def _withdraw(self, adapter_name: str, placements: list[_Placement]) -> None:
        super()._withdraw(adapter_name, placements)
        self._drop_projection(adapter_name)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        self._drop_projection(adapter_name)

    # ── the adapter file ──────────────────────────────────────────────────

    @override
    @classmethod
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        """The projections, as the file's two model-wide tensors; none when the adapter does not save them."""
        config = _as_vera(context.config)
        if not config.save_projection:
            return {}
        for path, layer in context.layers:
            if isinstance(layer, VeraLayer) and context.adapter in layer.vera_A:
                # The layer holding the containers in its tree: the others only read them.
                if not layer.vera_A.persistent:
                    raise ValueError(
                        "Model was initialised to not save vera_A and vera_B but config now "
                        "specifies to save projection! Set `config.save_projection` to `False`."
                    )
                where = f"{context.prefix}{path}"
                return {
                    f"{where}.vera_A.{context.adapter}": _FILE_A,
                    f"{where}.vera_B.{context.adapter}": _FILE_B,
                }
        return {}

    @classmethod
    def _fit_projection(cls, context: _FileContext, state: Mapping[str, Tensor]) -> None:
        """Give the adapter's projections the size the file's have, if every layer still fits them.

        The file holds the projections drawn for the largest layer of the model
        that saved it; a model that adapts fewer layers drew smaller ones.  The
        file's win, as long as they have the rank and cover every layer — else
        nothing is changed and the loader refuses the shapes.
        """
        layers = [
            layer
            for _, layer in context.layers
            if isinstance(layer, VeraLayer) and context.adapter in layer._specs
        ]
        held = layers[0]._containers
        loaded = state[_FILE_A], state[_FILE_B]
        if context.adapter not in held[0] or not _covers(
            loaded, _as_vera(context.config).r, layers
        ):
            return
        for container, tensor in zip(held, loaded, strict=True):
            current = container[context.adapter]
            if tuple(current.shape) != tuple(tensor.shape):
                container[context.adapter] = lucid.zeros(
                    *tensor.shape, device=current.device, dtype=current.dtype
                )

    @override
    @classmethod
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Require the projections when the adapter saves them; ignore them (with a warning) when it does not."""
        present = _FILE_A in state and _FILE_B in state
        if _as_vera(context.config).save_projection:
            if not present:
                raise ValueError(
                    "Specified to load vera_A and vera_B from state dictionary however they "
                    "were not present!"
                )
            cls._fit_projection(context, state)
            return state
        if present:
            warnings.warn(
                "Specified to not load vera_A and vera_B from state dictionary however they are "
                "present in state dictionary! Consider using them to ensure checkpoint loading "
                "is correct on all platforms using `peft_config.save_projection = True`",
                stacklevel=4,
            )
        else:
            warnings.warn(
                "Specified to not load vera_A and vera_B from state dictionary. This means we "
                "will be relying on PRNG initialisation to restore these projections using "
                "`config.projection_prng_key`, which may not be accurate on all system "
                "configurations.",
                stacklevel=4,
            )
        return {key: value for key, value in state.items() if key not in (_FILE_A, _FILE_B)}
