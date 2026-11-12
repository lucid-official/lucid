r"""A layer whose update depends on the weights it is merged into: merging keeps the base's own values, unmerging writes them back.

HiRA (:math:`W \odot BA`), GLoRA (:math:`W A + B`, and a bias that reads :math:`W`), DEFT
(:math:`(I - P_{proj})\,W + Q_P R`) and PEANuT (a network that reads :math:`W`) all make an update
:math:`\Delta_a(W)` that is a function of the base weight.  An update added to a weight that already
holds another update's would read the wrong :math:`W`, and an unmerge that subtracts
:math:`\Delta_a(W)` cannot recover :math:`W` from :math:`W + \Delta_a(W)` bit for bit (for GLoRA it
divides, for DEFT it is not invertible).  So the first merge stashes the base values, every update is
computed from the stash, and an unmerge copies the stash back: the merged weight is
:math:`W_0 + \sum_a \Delta_a(W_0)`, the sum the forward pass computes, and the unmerged one is
:math:`W_0` exactly.  The stash is a copy of each merge target while the layer is merged.

The stash is made of non-persistent buffers, so it follows the layer through ``.to()`` and
``deepcopy`` and is never saved: an adapter file holds the adapter, not the base.
"""

from abc import abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Final, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._layer import _dotted
from lucid.peft.tuners._shared import _BufferDict
from lucid.peft.tuners._support import _ResidualUpdate

_BASE: Final = "base_layer"
_WEIGHT: Final = _dotted(_BASE, "weight")


def _leaf(key: str) -> str:
    """The attribute a merge target key names: ``"base_layer.weight"`` is ``"weight"``."""
    return key.rpartition(".")[2]


class _PristineMerge(_ResidualUpdate):
    r"""A residual layer whose adapters change the base by an amount that reads the base.

    The forward pass is the base output plus every applied adapter's ``_contribution``
    (``_ResidualUpdate``); a layer merged for only some of its adapters applies the rest, each
    computed from the weights as they were before the merge (``_pristine_weight``), so the sum
    is the one the unmerged layer makes.  A subclass gives ``_changes`` —
    what an adapter adds to each merge target, computed from the base values it is handed — and
    this class makes ``get_delta_weight`` (the change to the weight), ``merge`` and ``unmerge`` of
    it, in place of the additive merge ``_ResidualUpdate`` brings.

    ``_changes`` always reads the values the base had before the first merge (the stash while the
    layer is merged, the live parameters otherwise), so merging several adapters, in any order, and
    unmerging them are the same sum and its undoing.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    """

    #: The parameters of ``base_layer`` an update changes, by attribute name; one the module lacks (no bias) is skipped.
    _merged_names: ClassVar[tuple[str, ...]] = ("weight",)

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self._kept = _BufferDict(persistent=False)

    # ── what a method supplies ────────────────────────────────────────────

    @abstractmethod
    def _changes(self, adapter: str, base: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """What ``adapter`` adds to each merge target (keyed as ``_merge_targets`` is), computed from the values ``base`` holds for them."""

    # ── the stash ─────────────────────────────────────────────────────────

    def _pristine(self) -> dict[str, Tensor]:
        """The base values the updates are functions of: the ones stashed while merged, else the live ones."""
        values: dict[str, Tensor] = {}
        for key, param in self._merge_targets().items():
            values[key] = self._kept[_leaf(key)] if len(self._kept) else param
        return values

    def _pristine_weight(self) -> Tensor:
        """The base weight an update is a function of, as it was before any merge: what a contribution reads."""
        return self._pristine()[_WEIGHT]

    def _stash(self) -> None:
        """Keep a copy of every merge target (once: a second merge adds to the same stash)."""
        if not len(self._kept):
            for key, param in self._merge_targets().items():
                self._kept[_leaf(key)] = param.detach().clone()

    # ── the layer ─────────────────────────────────────────────────────────

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        """The change ``adapter`` makes to the base weight, computed from the weight as it was before any merge.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the adapter's dtype.
        """
        return self._changes(adapter, self._pristine())[_WEIGHT]

    @override
    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        """Always ``False``: the update is not a fixed matrix, it reads the weight it is added to."""
        del adapter_name
        return False

    # ── merging ───────────────────────────────────────────────────────────

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        base = self._root()
        targets: dict[str, Parameter] = {}
        for name in self._merged_names:
            held = getattr(base, name, None)
            if isinstance(held, Parameter):
                targets[self._base_key(name)] = held
        return targets

    @override
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        changes = self._changes(adapter, self._pristine())
        return {key: (p + changes[key]).to(p.dtype) for key, p in self._merge_targets().items()}

    @override
    def _fold(self, adapter: str) -> None:
        values = self._merged_values(adapter)
        self._stash()
        for key, param in self._merge_targets().items():
            param.copy_(values[key])

    @override
    def _unfold(self) -> None:
        if not len(self._kept):
            raise RuntimeError(
                "this layer is merged but holds no copy of the weights it was merged into; "
                "its merge cannot be undone"
            )
        for key, param in self._merge_targets().items():
            param.copy_(self._kept[_leaf(key)])
        for name in list(self._kept):
            del self._kept[name]
