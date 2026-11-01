r"""The MiSS layers: a frozen base module beside one trainable block whose shape the variant sets.

``MissLayer`` is the family root.  An adapter holds one parameter, ``miss_block``
(``(r, d_out)`` for the default variant; see ``lucid.peft.tuners.miss._modes``), so a
state dict reads ``....miss_block.default`` and an adapter file ``....miss_block``, the
reference's key.  The update is a weight-space rewrite (``_WeightRewrite``): merging
writes the rewritten weight into the base, and an unmerged layer computes either the
addition the shards make to the output (the balance and mini variants) or the rewritten
weight (the bat variant), which are the same function as the merged layer.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._base import _parameter_at
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._support import (
    _adapter_dtype,
    _config_as,
    _ConfigLayer,
    _LinearSized,
    _SpecField,
    _WeightRewrite,
)
from lucid.peft.tuners.miss._config import MissConfig
from lucid.peft.tuners.miss._modes import _Mode


class _MissSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameter.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    miss_r: int
    miss_mini_r: int
    mode: str


class MissLayer(_ConfigLayer, _LinearSized, _WeightRewrite, family=True):
    r"""Adapter bookkeeping and the block-to-weight map shared by the MiSS layers.

    Per adapter: ``miss_block``, one parameter whose shape the variant sets, and the
    read-only mappings ``miss_r`` and ``miss_mini_r``.  ``get_delta_weight`` is the change
    merging the adapter makes to the current base weight.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : MissConfig
        The adapter's config.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``config.r`` exceeds the layer's input size, the variant needs a size the layer
        lacks (``bat``: both sizes divisible by ``r``; ``mini``: outputs divisible by
        ``mini_r``), or ``base_layer`` builds parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("miss_block",)
    _repr_prefix = "miss."
    miss_block: nn.ParameterDict
    miss_dropout: nn.ModuleDict[nn.Module]

    miss_r = _SpecField[int]("Shard width of each adapter, by name (read-only).")
    miss_mini_r = _SpecField[int](
        "Output-axis rank of each adapter's mini block, by name (read-only)."
    )

    @override
    def _prepare(self, config: _TargetedConfig) -> None:
        del config  # nothing here depends on it
        self.miss_dropout = nn.ModuleDict()

    @property
    def _blocks(self) -> nn.ParameterDict:
        """The per-adapter blocks, under whatever name the method gives them (``miss_block``; Bone's ``bone_block``)."""
        table = getattr(self, type(self).adapter_layer_names[0])
        assert isinstance(table, nn.ParameterDict)
        return table

    def _mode(self, adapter: str) -> _Mode:
        spec = self._specs[adapter]
        assert isinstance(spec, _MissSpec)
        return _Mode.named(spec.mode)

    @classmethod
    def _miss_config(cls, config: _TargetedConfig) -> MissConfig:
        """``config`` as the MiSS config the layer reads (a successor method's layers translate their own here)."""
        return _config_as(MissConfig, config)

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = self._miss_config(config)
        mode = _Mode.named(_Mode.of(cfg.init_weights))
        mode.check(self.in_features, self.out_features, cfg.r, cfg.mini_r)
        anchor = self.weight
        shape = mode.block_shape(self.out_features, cfg.r, cfg.mini_r)
        block = lucid.zeros(*shape, device=anchor.device, dtype=_adapter_dtype(anchor))
        if cfg.init_weights is False:  # the balance variant's random start, for tests
            init.kaiming_uniform_(block, a=5**0.5)
        dropout: nn.Module = (
            nn.Dropout(p=cfg.miss_dropout) if cfg.miss_dropout > 0.0 else nn.Identity()
        )
        self._specs[adapter_name] = _MissSpec(cfg.r, cfg.mini_r, _Mode.of(cfg.init_weights))
        self._blocks[adapter_name] = Parameter(block)
        self.miss_dropout[adapter_name] = dropout

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its shard matrix and its dropout."""
        super().delete_adapter(adapter_name)
        if adapter_name in self.miss_dropout:
            del self.miss_dropout[adapter_name]

    # ── the weight an adapter makes ───────────────────────────────────────

    @override
    def _rewritten(self, weight: Tensor, adapter: str) -> Tensor:
        return self._mode(adapter).rewritten(weight, _parameter_at(self._blocks, adapter))

    @override
    def _reverted(self, weight: Tensor, adapter: str) -> Tensor:
        return self._mode(adapter).reverted(weight, _parameter_at(self._blocks, adapter))

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return the change merging ``adapter`` makes to the current base weight.

        Weight-independent for the balance and mini variants (the shard tiling); for ``bat``
        it is :math:`W S + S` blockwise, so it depends on the weight it is merged into.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the block's dtype.
        """
        return self._mode(adapter).delta(self.weight, _parameter_at(self._blocks, adapter))

    # ── the forward pass ──────────────────────────────────────────────────

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Run the layer with the active adapters' shards applied.

        The default and ``mini`` variants add their contribution to the base output; ``bat``
        rewrites the weight and the layer is computed from that.  The base layer alone
        when adapters are disabled, merged or none applies.
        """
        adapters = [name for name in self.active_adapters if name in self._specs]
        base = self.base_layer
        if self.disable_adapters or self.merged or not adapters:
            result: Tensor = base(x)
            return result
        if all(self._mode(name).additive for name in adapters):
            result = base(x)
            out = result
            for name in adapters:
                dropped: Tensor = self.miss_dropout[name](x)
                out = out + self._mode(name).contribution(
                    dropped, _parameter_at(self._blocks, name), self.out_features
                )
            return out if out.dtype == result.dtype else out.to(result.dtype)
        weight: Tensor = self.weight
        for name in adapters:
            weight = self._rewritten(weight, name)
        bias = self.bias
        out = F.linear(x.to(weight.dtype), weight, None if bias is None else bias.to(weight.dtype))
        return out if out.dtype == x.dtype else out.to(x.dtype)


class Linear(MissLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a MiSS update: :math:`y = (W + \Delta W)\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`MissLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import MissConfig
    >>> from lucid.peft.tuners.miss import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", MissConfig(r=2, target_modules=["x"]))
    >>> tuple(layer.miss_block["default"].shape)
    (2, 4)
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the block starts at zero
    True
    """

    base_layer: nn.Linear
