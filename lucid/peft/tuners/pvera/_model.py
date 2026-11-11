"""``PveraModel``: the PVeRA tuner — the random projections its layers share, and how they are saved."""

from collections.abc import Mapping
from typing import ClassVar, override

from lucid._tensor.tensor import Tensor
from lucid.nn import Module
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft.tuners._layer import _BaseTunerLayer
from lucid.peft.tuners._plan import _Target
from lucid.peft.tuners._shared import _base_linear, _BufferDict, _ProjectionTuner
from lucid.peft.tuners._support import _adapter_dtype, _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.pvera._config import PveraConfig
from lucid.peft.tuners.pvera._layer import PveraLayer
from lucid.peft.tuners.pvera._projection import _Projection


def _as_pvera(config: PeftConfig) -> PveraConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, PveraConfig)
    return config


class PveraModel(_ProjectionTuner, _ConfigTuner, peft_type=PeftType.PVERA):
    r"""A model with PVeRA layers put into it, the projections they share, and the tools to take them out.

    Construction replaces each ``nn.Linear`` ``config`` selects with a PVeRA layer around it and
    freezes every other parameter.  The model is changed in place and kept as ``model``; a public
    name this wrapper lacks is read from it.  ``config`` itself is not changed: ``peft_config``
    holds a copy with the default ``target_modules`` spelled out.

    The two random projections :math:`A` (``(2r, d_in)``) and :math:`B` (``(d_out, r)``) are drawn
    once, sized for the largest targeted layer, and held here as ``pvera_A`` / ``pvera_B`` (tables of
    one tensor per adapter, aliased when adapters share them) — the reference's attributes.  Every
    layer reads them from there; the first layer holds them in its module tree, so a state dict lists
    each once and the adapter file holds one pair.  Adapters added later must fit them (the same
    ``projection_prng_key`` and ``save_projection``, a rank and layer sizes no larger).

    Parameters
    ----------
    model : Module
        The model to adapt.
    peft_config : PveraConfig
        What to adapt and how.
    adapter_name : str
        Name of the adapter, the key in every ``pvera_lambda_b`` / ``pvera_lambda_d``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` selects nothing or something that is not an ``nn.Linear``,
        ``modules_to_save`` names nothing, or a later adapter outgrows the projections of the first.
    NotImplementedError
        Layers of another method are already in the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import PveraConfig, PveraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> pvera = PveraModel(net, PveraConfig(r=4, target_modules=["0", "2"]), "default")
    >>> tuple(pvera.pvera_A["default"].shape), tuple(pvera.pvera_B["default"].shape)
    ((8, 8), (8, 4))
    >>> net[0].base_layer.weight.requires_grad, net[0].pvera_lambda_d["default"].requires_grad
    (False, True)
    >>> pvera.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = PveraLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = PveraConfig
    _table_names: ClassVar[tuple[str, ...]] = ("pvera_A", "pvera_B")

    @property
    def pvera_A(self) -> _BufferDict:
        """The shared ``A`` projections, one entry per adapter (the reference's attribute)."""
        return self._shared.buffers("pvera_A")

    @property
    def pvera_B(self) -> _BufferDict:
        """The shared ``B`` projections, one entry per adapter (the reference's attribute)."""
        return self._shared.buffers("pvera_B")

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: Module) -> _TargetedConfig:
        stored = _as_pvera(super()._prepare_config(config, model))
        _reset_fan_in_fan_out(stored)
        return stored

    # ── the shared projections ────────────────────────────────────────────

    @override
    @classmethod
    def _new_tables(cls, config: _TargetedConfig) -> dict[str, Module]:
        persistent = _as_pvera(config).save_projection
        return {"pvera_A": _BufferDict(persistent), "pvera_B": _BufferDict(persistent)}

    @override
    @classmethod
    def _saves_projection(cls, config: PeftConfig) -> bool:
        return _as_pvera(config).save_projection

    @override
    @classmethod
    def _prng_key(cls, config: PeftConfig) -> int:
        return _as_pvera(config).projection_prng_key

    @override
    def _provide(self, adapter_name: str, config: _TargetedConfig, targets: list[_Target]) -> None:
        """Enter the projections ``adapter_name`` reads: drawn for the first adapter, the first's shared with the others.

        Sized for the largest of ``targets``.  A later adapter must fit what the first drew — layer
        sizes and rank no larger.
        """
        pvera = _as_pvera(config)
        linears = [_base_linear(target.module) for target in targets]
        out_size = max(linear.out_features for linear in linears)
        in_size = max(linear.in_features for linear in linears)
        if len(self.pvera_A) == 0:
            anchor = linears[0].weight
            self.pvera_A[adapter_name], self.pvera_B[adapter_name] = _Projection.draw(
                pvera.projection_prng_key,
                pvera.r,
                out_size,
                in_size,
                anchor.device,
                _adapter_dtype(anchor),
            )
            return
        # Any held adapter's: they are all the same tensors, drawn from one key.
        held = next(iter(self.pvera_A))
        proj_a, proj_b = self.pvera_A[held], self.pvera_B[held]
        too_small = (
            "{} has a size of {} but {} or greater is required; this probably happened because an "
            "additional PVeRA adapter was added after the first one with incompatible shapes."
        )
        for label, size, need in (
            ("pvera_A", proj_a.shape[1], in_size),
            ("pvera_B", proj_b.shape[0], out_size),
        ):
            if size < need:
                raise ValueError(too_small.format(label, size, need))
        if proj_b.shape[1] < pvera.r:
            raise ValueError(
                f"pvera_B has a size of {proj_b.shape[1]} but {pvera.r} or greater is required; "
                "this probably happened because an additional PVeRA adapter with a higher rank was "
                "added after the first one; loading the adapters in reverse order may solve this."
            )
        self.pvera_A[adapter_name], self.pvera_B[adapter_name] = proj_a, proj_b

    @override
    @classmethod
    def _covers(
        cls, config: PeftConfig, loaded: Mapping[str, Tensor], layers: list[_BaseTunerLayer]
    ) -> bool:
        """Whether the file's projections have rank ``r`` and are big enough to be sliced for every one of ``layers``."""
        proj_a, proj_b = loaded["pvera_A"], loaded["pvera_B"]
        r = _as_pvera(config).r
        if proj_a.ndim != 2 or proj_b.ndim != 2 or proj_a.shape[0] != 2 * r or proj_b.shape[1] != r:
            return False
        return all(
            isinstance(layer, PveraLayer)
            and proj_a.shape[1] >= layer.in_features
            and proj_b.shape[0] >= layer.out_features
            for layer in layers
        )
