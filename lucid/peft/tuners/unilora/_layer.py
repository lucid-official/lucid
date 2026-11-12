r"""The Uni-LoRA layers: a frozen base module beside LoRA factors read out of one shared vector.

Uni-LoRA (Li et al., 2025, arXiv:2506.00799, Sec. 3.2) writes every LoRA factor of every adapted layer
as a read of one vector :math:`\theta_d`: :math:`A_{ij} = s_{ij}\,\theta_d[a_{ij}]` and
:math:`B_{ij} = s'_{ij}\,\theta_d[b_{ij}]`, with a fixed index :math:`a_{ij}` and scale :math:`s_{ij}`
for every entry.  The layer adds :math:`B A\,x`; there is no :math:`\alpha / r`.

``UniLoraLayer`` is the family root.  A layer holds, per adapter, the four tensors that say where its
factors read the vector — ``unilora_indices_A`` (``(r, in)``), ``unilora_indices_B`` (``(out, r)``) and
their scales ``unilora_scales_A`` / ``unilora_scales_B`` — and reads the vector itself, ``unilora_theta_d``
(``(d,)``, one per adapter), from the table its tuner shares between all its layers (``bind``).  The
four tensors are saved only when the adapter's ``config.save_indices`` is set (each adapter says for
itself): the seed draws them again.

A layer made on its own, outside a tuner, holds a vector of its own and draws the indices of its own
two factors the first time it needs them.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn import Parameter
from lucid.peft.tuners._delta import _DeltaLayer, _Home
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._shared import _BufferDict, _TrainedSharedLayer
from lucid.peft.tuners._support import _LinearSized, _ResidualUpdate, _SpecField
from lucid.peft.tuners.unilora._config import UniLoraConfig
from lucid.peft.tuners.unilora._indices import _Assignment, _assign

#: The per-layer tables of a layer's projection, in the order ``_Assignment`` lists its tensors.
_PROJECTION: tuple[str, ...] = (
    "unilora_indices_A",
    "unilora_indices_B",
    "unilora_scales_A",
    "unilora_scales_B",
)


class _UniLoraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int
    theta_d_length: int
    proj_seed: int
    save_indices: bool


def _new_theta_d(config: UniLoraConfig, home: _Home) -> Parameter:
    """The shared vector of ``config.theta_d_length`` entries: uniform in the bound, or standard normal when ``init_weights`` is off."""
    start = home.zeros(config.theta_d_length)
    if config.init_weights:
        start.uniform_(-config.init_theta_d_bound, config.init_theta_d_bound)
    else:
        start = lucid.randn(config.theta_d_length, device=home.device, dtype=home.dtype)
    return Parameter(start)


class UniLoraLayer(
    _TrainedSharedLayer, _DeltaLayer[UniLoraConfig], _LinearSized, _ResidualUpdate, family=True
):
    r"""Adapter bookkeeping and the vector-to-factors map shared by the Uni-LoRA layers.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : UniLoraConfig
        The adapter's config.

    Raises
    ------
    TypeError
        ``base_layer`` is not a module this layer class wraps.
    ValueError
        ``config.fan_in_fan_out`` is set, or ``base_layer`` is lazy and has not run yet.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ()
    _shared_names: ClassVar[tuple[str, ...]] = ("unilora_theta_d",)
    _dropout_table = "unilora_dropout"
    _repr_prefix = "unilora."
    unilora_dropout: nn.ModuleDict[nn.Module]
    unilora_theta_d: nn.ParameterDict
    unilora_indices_A: _BufferDict
    unilora_indices_B: _BufferDict
    unilora_scales_A: _BufferDict
    unilora_scales_B: _BufferDict

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Rank of each adapter's factors, by name (read-only).")

    @override
    def _prepare(self, config: UniLoraConfig) -> None:
        if config.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        super()._prepare(config)
        self._hold({"unilora_theta_d": nn.ParameterDict()})
        for name in _PROJECTION:
            setattr(self, name, _BufferDict(persistent=config.save_indices))
        # The projection of a layer used alone: plain state, not part of ``state_dict``.
        self._unbound: dict[str, _Assignment] = {}

    @override
    def _enter(self, adapter_name: str, config: UniLoraConfig) -> None:
        self._specs[adapter_name] = _UniLoraSpec(
            config.r, config.theta_d_length, config.proj_seed, config.save_indices
        )
        self._enter_dropout(adapter_name, config.unilora_dropout)
        if self._bound is None:
            # A layer used alone holds the vector it reads; a tuner's layers read the tuner's.
            self.unilora_theta_d[adapter_name] = _new_theta_d(config, self._home())

    @override
    def _adapter_names(self) -> list[str]:
        return list(self._specs)

    @override
    def _rebound(self) -> None:
        self._unbound.clear()

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its projection, its dropout and its recorded settings."""
        super().delete_adapter(adapter_name)
        for name in _PROJECTION:
            held = getattr(self, name)
            if adapter_name in held:
                del held[adapter_name]
        self._unbound.pop(adapter_name, None)

    # ── the projection ────────────────────────────────────────────────────

    def _spec(self, adapter: str) -> _UniLoraSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _UniLoraSpec)  # ``_enter`` is the only writer
        return spec

    def factor_shapes(self, adapter: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """The shapes of ``A`` and ``B``: ``(r, in)`` and ``(out, r)``.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        tuple of tuple of int
            The two shapes.
        """
        r = self._spec(adapter).r
        return (r, self.in_features), (self.out_features, r)

    def bind_projection(self, adapter: str, assignment: _Assignment) -> None:
        """Hold the projection the tuner drew for this layer's factors, on the base weight's device.

        The tensors are saved with the adapter when its ``save_indices`` is set, whatever the adapters
        already in the layer chose.

        Parameters
        ----------
        adapter : str
            The adapter's name.
        assignment : _Assignment
            The indices and scales of ``A`` and ``B``.
        """
        device = self.weight.device
        saved = self._spec(adapter).save_indices
        for name, tensor in zip(_PROJECTION, assignment, strict=True):
            getattr(self, name).register(adapter, tensor.to(device=device), saved)

    def _assignment(self, adapter: str) -> _Assignment:
        """The projection ``adapter`` reads: the one the tuner bound, or one drawn for this layer alone.

        A layer used without a tuner draws it from its own seed and keeps it out of its state: a
        forward pass must not change what ``state_dict`` holds.
        """
        if adapter in self.unilora_indices_A:
            return _Assignment(*(getattr(self, name)[adapter] for name in _PROJECTION))
        drawn = self._unbound.get(adapter)
        if drawn is None:
            spec = self._spec(adapter)
            (drawn,) = _assign([self.factor_shapes(adapter)], spec.theta_d_length, spec.proj_seed)
            self._unbound[adapter] = drawn
        return drawn

    def _theta_d(self, adapter: str) -> Parameter:
        return _parameter_at(self._shared_table("unilora_theta_d"), adapter)

    def _shared_table(self, name: str) -> nn.ParameterDict:
        table = self._table(name)
        assert isinstance(table, nn.ParameterDict)
        return table

    def _factors(self, adapter: str) -> tuple[Tensor, Tensor]:
        r"""The pair :math:`(A, B)` ``adapter`` reads: ``(r, in)`` and ``(out, r)``, read out of the vector."""
        theta = self._theta_d(adapter)
        index_a, index_b, scale_a, scale_b = (
            t.to(device=theta.device) for t in self._assignment(adapter)
        )
        return (theta[index_a] * scale_a.to(theta.dtype), theta[index_b] * scale_b.to(theta.dtype))

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""The dense update of ``adapter``: :math:`B A`, with :math:`A` and :math:`B` read out of the vector.

        Parameters
        ----------
        adapter : str
            Name of the adapter whose update is wanted.

        Returns
        -------
        Tensor
            ``(out, in)``, in the dtype of the vector.
        """
        factor_a, factor_b = self._factors(adapter)
        return factor_b @ factor_a

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        factor_a, factor_b = self._factors(adapter)
        dropped = self._dropped(adapter, x)
        dropped = dropped if dropped.dtype == factor_a.dtype else dropped.to(factor_a.dtype)
        return (dropped @ factor_a.mT) @ factor_b.mT


class Linear(UniLoraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a Uni-LoRA update: :math:`y = W x + b + B A\,\mathrm{drop}(x)`, :math:`A, B` read out of one vector.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`UniLoraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import UniLoraConfig
    >>> from lucid.peft.tuners.unilora import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", UniLoraConfig(r=2, theta_d_length=16, target_modules=["x"]))
    >>> tuple(layer.unilora_theta_d["default"].shape)
    (16,)
    >>> tuple(layer.get_delta_weight("default").shape)
    (4, 8)
    """

    base_layer: nn.Linear
