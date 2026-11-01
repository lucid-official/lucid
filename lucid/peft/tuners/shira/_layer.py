r"""The SHiRA layers: a frozen base module beside the values of a fixed sparse subset of its weight.

``ShiraLayer`` is the family root.  An adapter holds ``shira_weight`` (``(K,)``, the
trained values, a ``ParameterDict``) and ``shira_indices`` (``(2, K)`` int32 row and column
of each value, a buffer table); a state dict reads ``....shira_weight.default`` and an adapter
file ``....shira_weight`` and ``....shira_indices``, the reference's keys.  The update is the
dense ``d_out x d_in`` matrix with the values scattered at the indices, times the adapter's scale.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid.nn.parameter import Parameter
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._base import _parameter_at
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._shared import _BufferDict
from lucid.peft.tuners._support import (
    _adapter_dtype,
    _config_as,
    _ConfigLayer,
    _LinearSized,
    _ResidualUpdate,
    _SpecField,
)
from lucid.peft.tuners.shira._config import ShiraConfig


class _ShiraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the scale
    is baked into the graph.
    """

    r: int
    scaling: float


class ShiraLayer(_ConfigLayer, _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the values-to-weight map shared by the SHiRA layers.

    Per adapter: ``shira_weight`` (the trained values), ``shira_indices`` (where they go) and
    the read-only mappings ``r`` and ``scaling``.  The forward pass adds each active adapter's
    sparse update applied to the input; merging adds it to the weight.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` to adapt, kept as ``base_layer`` with its parameters shared.
    adapter_name : str
        The first adapter's name.
    config : ShiraConfig
        Its ``mask_fn`` picks the entries; ``r`` sizes them.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        The mask does not have ``r (d_out + d_in)`` ones, ``config.fan_in_fan_out`` is set
        (the layer's weight is stored ``(out, in)``), no ``mask_fn`` is set, or ``base_layer``
        builds parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("shira_weight",)
    _repr_prefix = "shira."
    shira_weight: nn.ParameterDict
    shira_indices: _BufferDict

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Rank each adapter's entry count is sized after, by name (read-only).")
    scaling = _SpecField[float]("Scale of each adapter's update, by name (read-only).")

    @override
    def _prepare(self, config: _TargetedConfig) -> None:
        cfg = _config_as(ShiraConfig, config)
        if cfg.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        self.shira_indices = _BufferDict(persistent=True)

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(ShiraConfig, config)
        if cfg.mask_fn is None:
            raise ValueError(
                f"mask_type={cfg.mask_type!r} needs a mask function: set `config.mask_fn`"
            )
        options = {"random_seed": cfg.random_seed} if cfg.mask_type == "random" else {}
        mask = cfg.mask_fn(self.base_layer, cfg.r, **options)
        positions = mask.nonzero()  # (K, 2), row-major
        count = cfg.r * (self.in_features + self.out_features)
        if positions.shape[0] != count:
            raise ValueError(
                f"The SHiRA indices and weights are not the same dimensions for adapter "
                f"{adapter_name} in layer {self.base_layer}: the mask holds {positions.shape[0]} "
                f"entries, r={cfg.r} needs {count}"
            )
        anchor = self.weight
        shape = (count,)
        dtype = _adapter_dtype(anchor)
        values = (
            lucid.zeros(*shape, device=anchor.device, dtype=dtype)
            if cfg.init_weights
            else lucid.randn(*shape, device=anchor.device, dtype=dtype)
        )
        self._specs[adapter_name] = _ShiraSpec(cfg.r, 1.0)
        self.shira_weight[adapter_name] = Parameter(values)
        self.shira_indices[adapter_name] = positions.mT.to(dtype=lucid.int32)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its values and its mask indices."""
        super().delete_adapter(adapter_name)
        if adapter_name in self.shira_indices:
            del self.shira_indices[adapter_name]

    def set_scale(self, adapter: str, scale: float) -> None:
        """Scale ``adapter``'s update by ``scale`` (the reference keeps 1 while training; any non-negative value is for inference).

        An adapter this layer does not hold is ignored.

        Parameters
        ----------
        adapter : str
            The adapter's name.
        scale : float
            The new scale.
        """
        spec = self._specs.get(adapter)
        if spec is not None:
            assert isinstance(spec, _ShiraSpec)
            self._specs[adapter] = spec._replace(scaling=scale)

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return the sparse update :math:`s\,\Delta` as a dense matrix shaped like the base weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Zero but at the adapter's indices, in the values' dtype.
        """
        spec = self._specs[adapter]
        assert isinstance(spec, _ShiraSpec)
        values = _parameter_at(self.shira_weight, adapter)
        where = self.shira_indices[adapter].to(device=values.device, dtype=lucid.int64)
        dense = lucid.zeros(
            self.out_features, self.in_features, device=values.device, dtype=values.dtype
        )
        return dense.index_put((where[0], where[1]), values * spec.scaling)

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        delta = self.get_delta_weight(adapter)
        return x.to(delta.dtype) @ delta.mT


class Linear(ShiraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a SHiRA update: :math:`y = (W + s\,\Delta)\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`ShiraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.shira import Linear, ShiraConfig
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", ShiraConfig(r=1, random_seed=3, target_modules=["x"]))
    >>> tuple(layer.shira_weight["default"].shape), tuple(layer.shira_indices["default"].shape)
    ((12,), (2, 12))
    >>> x = lucid.randn(2, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the values start at zero
    True
    """

    base_layer: nn.Linear
