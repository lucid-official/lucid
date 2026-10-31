"""The LN-tuning layer: a frozen module beside a trainable copy of itself.

An adapter *is* a copy of the wrapped module (``ln_tuning_layers[name]``): it
starts with the base's values and trains while the base stays frozen.  Only one
adapter can apply at a time — a copy replaces the module rather than adding to
it — and only one can be merged.  Merging exchanges the values of the base and
the trained copy in place (so the base keeps its ``Parameter`` objects, as every
tuner promises); the copy then holds the original values, and unmerging
exchanges them back.  A state dict reads ``....ln_tuning_layers.default.weight``
and an adapter file ``....ln_tuning_layers.weight``, the reference's key.
"""

import copy
import warnings
from typing import ClassVar, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._support import _config_as, _ConfigLayer
from lucid.peft.tuners.ln_tuning._config import LNTuningConfig

_BASE = "base_layer."


class LNTuningLayer(_ConfigLayer, base_type=nn.Module):
    """A module beside a trainable copy of itself, one copy per adapter.

    The layer applies the copy of the one active adapter it holds; with none it
    runs the base.  It wraps any module that takes and returns one tensor.

    Parameters
    ----------
    base_layer : nn.Module
        The module to tune; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : LNTuningConfig
        The adapter's config.

    Raises
    ------
    ValueError
        ``base_layer`` builds parameters on its first input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LNTuningConfig
    >>> from lucid.peft.tuners.ln_tuning import LNTuningLayer
    >>> base = nn.LayerNorm(4)
    >>> layer = LNTuningLayer(base, "default", LNTuningConfig(target_modules=["x"]))
    >>> x = lucid.randn(3, 4)
    >>> bool((layer(x) == base(x)).all().item())   # the copy starts as the base
    True
    >>> layer.ln_tuning_layers["default"].weight is base.weight
    False
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("ln_tuning_layers",)
    _repr_prefix = "ln_tuning."
    ln_tuning_layers: nn.ModuleDict

    @override
    def _new_table(self) -> nn.ParameterDict | nn.ModuleDict[nn.Module]:
        return nn.ModuleDict()

    def _original(self) -> nn.Module:
        """The module that holds the base's own values: the base, or — merged — the copy it was exchanged with."""
        return self.ln_tuning_layers[self.merged_adapters[0]] if self.merged else self.base_layer

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        _config_as(LNTuningConfig, config)
        self.ln_tuning_layers[adapter_name] = copy.deepcopy(self._original())

    # ── merging: exchange the values of the base and the copy ─────────────

    def _tensors(self, module: nn.Module) -> list[Tensor]:
        return [*module.parameters(), *module.buffers()]

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        """Return the tuned weight minus the original one, shaped like the base weight.

        Raises
        ------
        AttributeError
            The wrapped module has no ``weight``.
        """
        tuned = self.ln_tuning_layers[adapter]
        original = self._original()
        if adapter in self.merged_adapters:  # the exchange put the tuned values in the base
            tuned, original = original, tuned
        weight: Tensor = tuned.weight - original.weight
        return weight

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        return {f"{_BASE}{name}": p for name, p in self.base_layer.named_parameters()}

    @override
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        tuned = self.ln_tuning_layers[adapter]
        return {f"{_BASE}{name}": p for name, p in tuned.named_parameters()}

    def _exchange(self, adapter: str) -> None:
        """Swap the values of the base and of ``adapter``'s copy, in place."""
        held = self.ln_tuning_layers[adapter]
        for base, other in zip(self._tensors(self.base_layer), self._tensors(held), strict=True):
            spare = base.clone()
            base.copy_(other)
            other.copy_(spare)

    @override
    def _fold(self, adapter: str) -> None:
        self._exchange(adapter)

    @override
    def _unfold(self) -> None:
        self._exchange(self.merged_adapters[-1])

    @override
    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """Make an adapter's copy the layer's values, in place.

        One adapter can be merged at a time.  Asking for another while one is
        merged unmerges the first, with a warning.

        Parameters
        ----------
        safe_merge : bool, default=False
            Check the tuned values for NaN / inf first and leave the base untouched if there are any.
        adapter_names : list of str, optional
            The adapter to merge; the active one this layer holds by default.

        Raises
        ------
        ValueError
            More than one adapter is asked for, or ``safe_merge`` found a non-finite value.
        RuntimeError
            The adapter is disabled.
        """
        pending = self._pending_merges(adapter_names)
        if len(pending) > 1:
            raise ValueError(
                f"Trying to merge {len(pending)} adapters, but LN tuning does not allow merging "
                "more than one adapter at a time"
            )
        if pending and self.merged:
            warnings.warn(
                f"Already merged with {set(self.merged_adapters)}. Unmerging first.", stacklevel=2
            )
            self.unmerge()
        super().merge(safe_merge, adapter_names)

    # ── the forward pass ──────────────────────────────────────────────────

    @override
    def forward(self, x: Tensor) -> Tensor:
        held = [name for name in self.active_adapters if name in self.ln_tuning_layers]
        if self.disable_adapters or self.merged or not held:
            result: Tensor = self.base_layer(x)
            return result
        if len(held) != 1:
            raise ValueError(
                f"Trying to run forward with {len(held)} active adapters, but LN tuning does "
                "not allow inference with more than one adapter at a time"
            )
        tuned: Tensor = self.ln_tuning_layers[held[0]](x)
        return tuned

