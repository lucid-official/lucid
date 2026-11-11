r"""The AdaMSS layers: a frozen linear module and an independent low-rank update inside each subspace of its rows.

AdaMSS (Zheng et al., 2025, NeurIPS, sections 3.2-3.3) clusters the rows of the weight into
:math:`K` subspaces and learns a low-rank update in each.  With the weight joined to its bias as one
more column, :math:`\tilde W = [W \mid b]`, and its leading right singular directions
:math:`V_r^\top` (``adamss_newB``, frozen), subspace :math:`k` holds the rows :math:`S_k` and two
trained factors, :math:`A_k` (``r_k x r``, starting orthonormal) and :math:`B_k` (``n_k x r_k``,
starting at zero), and the adapted layer is

.. math:: y = W x + b + \Pi\,\big[B_1 A_1; \dots; B_K A_K\big]\,V_r^\top\,[x; 1],

the blocks :math:`B_k A_k V_r^\top` in cluster order and :math:`\Pi` returning each row to its place
(the reference's ``scatter_index``).  The update's last column, the one that multiplies the constant
input, is a change of the bias: merging adds the first ``in`` columns to the weight and that column
to the bias, which an unmerge takes back bit for bit (``_PristineMerge`` keeps the stash).

``AdamssLayer`` is the family root.  An adapter holds ``adamss_A`` and ``adamss_B``, each a list of
:math:`K` parameters under the reference's keys (``adamss_A.0`` ... ``adamss_B.0`` ...), the frozen
projection ``adamss_newB`` (saved, as the reference saves it) and the row permutation
``adamss_rows`` (rebuilt from the base weight, never saved), and the read-only mappings ``r`` and
``num_subspaces``.
"""

from collections.abc import Mapping
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._layer import _dotted
from lucid.peft.tuners._shared import _BufferDict
from lucid.peft.tuners._stash import _PristineMerge
from lucid.peft.tuners._support import _LinearSized, _SpecField
from lucid.peft.tuners.adamss._config import AdamssConfig
from lucid.peft.tuners.adamss._subspaces import _split

_WEIGHT = _dotted("base_layer", "weight")
_BIAS = _dotted("base_layer", "bias")

#: The scale of the random ``B`` that ``init_weights=None`` draws, the reference's.
_RANDOM_START = 0.01


class _AdamssSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the rank and the
    number of subspaces shape the graph.
    """

    r: int
    num_subspaces: int


def _parameters(table: nn.ModuleDict[nn.Module], adapter: str) -> list[Parameter]:
    """The parameters ``table`` holds for ``adapter``, one per subspace, in subspace order."""
    held = table[adapter]
    assert isinstance(held, nn.ParameterList)
    return list(held.parameters())


class AdamssLayer(_DeltaLayer[AdamssConfig], _LinearSized, _PristineMerge, family=True):
    r"""Adapter bookkeeping and the multi-subspace update shared by the AdaMSS layers.

    Per adapter: ``adamss_A`` and ``adamss_B`` (one parameter per subspace each), the buffers
    ``adamss_newB`` and ``adamss_rows``, and the read-only mappings ``r`` and ``num_subspaces``.
    ``get_delta_weight`` and ``get_delta_bias`` are the weight and the bias parts of
    :math:`\Pi\,[B_k A_k]\,V_r^\top`; the forward pass applies it to the input without forming it.

    The split into subspaces is computed from the weight the layer holds when the adapter is made
    (the weight as it was before any merge), so building the same adapter on the same weights gives
    the same subspaces, which is what lets a saved adapter be loaded into a fresh model.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose rows are split; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : AdamssConfig
        The rank, the number of subspaces and their rank, the initialisation and the seed.

    Raises
    ------
    TypeError
        ``base_layer`` is some other module type.
    ValueError
        ``base_layer`` has parameters that only its first input will create, or its rows are too
        alike to fall into ``num_subspaces`` distinct clusters.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("adamss_A", "adamss_B")
    _merged_names: ClassVar[tuple[str, ...]] = ("weight", "bias")
    _repr_prefix = "adamss."
    adamss_A: nn.ModuleDict[nn.Module]
    adamss_B: nn.ModuleDict[nn.Module]
    adamss_newB: _BufferDict
    adamss_rows: _BufferDict

    r = _SpecField[int]("Total rank of the SVD of each adapter, by name (read-only).")
    num_subspaces = _SpecField[int]("Number of subspaces of each adapter, by name (read-only).")

    @override
    def _new_table(self) -> nn.ModuleDict[nn.Module]:
        return nn.ModuleDict()

    @override
    def _prepare(self, config: AdamssConfig) -> None:
        super()._prepare(config)
        self.adamss_newB = _BufferDict(persistent=True)
        self.adamss_rows = _BufferDict(persistent=False)

    @override
    def _enter(self, adapter_name: str, config: AdamssConfig) -> None:
        home = self._home()
        base = self._pristine()
        pieces = [base[_WEIGHT]] + ([base[_BIAS].reshape(-1, 1)] if _BIAS in base else [])
        joined = lucid.cat(pieces, dim=1).detach().to("cpu").to(lucid.float32)
        split = _split(joined, config)
        drawn = lucid.Generator(config.random_seed, device="cpu")
        factors: list[Parameter] = []
        updates: list[Parameter] = []
        for members, start in zip(split.members, split.starts, strict=True):
            factors.append(Parameter(start.to(home.device).to(home.dtype)))
            shape = (int(members.shape[0]), int(start.shape[0]))
            if config.init_weights is None:
                updates.append(
                    Parameter(
                        (_RANDOM_START * lucid.randn(*shape, generator=drawn, device="cpu"))
                        .to(home.device)
                        .to(home.dtype)
                    )
                )
            else:
                updates.append(Parameter(home.zeros(*shape)))
        self.adamss_A[adapter_name] = nn.ParameterList(factors)
        self.adamss_B[adapter_name] = nn.ParameterList(updates)
        self.adamss_newB[adapter_name] = split.projection.to(home.device).to(home.dtype)
        self.adamss_rows[adapter_name] = split.gather.to(home.device)
        self._specs[adapter_name] = _AdamssSpec(int(split.projection.shape[0]), len(factors))

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its factors, projection, row order and recorded settings."""
        super().delete_adapter(adapter_name)
        for buffers in (self.adamss_newB, self.adamss_rows):
            if adapter_name in buffers:
                del buffers[adapter_name]

    @property
    def scatter_index(self) -> Mapping[str, Tensor]:
        """For each adapter, the weight row each stacked row of the subspaces belongs to (read-only)."""
        return {name: self.adamss_rows[name].argsort() for name in self._adapter_names()}

    # ── the update ────────────────────────────────────────────────────────

    def _stacked(self, adapter: str) -> Tensor:
        """The subspaces' blocks :math:`B_k A_k` stacked in cluster order, ``(out, r)``."""
        blocks = [
            b @ a
            for a, b in zip(
                _parameters(self.adamss_A, adapter),
                _parameters(self.adamss_B, adapter),
                strict=True,
            )
        ]
        return lucid.cat(blocks, dim=0)

    def _joined_delta(self, adapter: str) -> Tensor:
        """The update of the weight joined to its bias, :math:`\\Pi\\,[B_k A_k]\\,V_r^\\top`: ``(out, in [+ 1])``."""
        placed = self._stacked(adapter) @ self.adamss_newB[adapter]
        delta: Tensor = placed.index_select(0, self.adamss_rows[adapter])
        return delta

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return the change ``adapter`` makes to the base weight: the first ``in`` columns of :math:`\Pi\,[B_k A_k]\,V_r^\top`.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the adapter's dtype.
        """
        delta = self._joined_delta(adapter)
        return delta[:, : self.in_features] if self.bias is not None else delta

    def get_delta_bias(self, adapter: str) -> Tensor:
        r"""Return the change ``adapter`` makes to the base bias: the last column of :math:`\Pi\,[B_k A_k]\,V_r^\top`.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(out,)``, in the adapter's dtype; zeros when the base layer has no bias.
        """
        delta = self._joined_delta(adapter)
        if self.bias is None:
            return lucid.zeros(self.out_features, device=delta.device, dtype=delta.dtype)
        return delta[:, self.in_features]

    @override
    def _changes(self, adapter: str, base: Mapping[str, Tensor]) -> dict[str, Tensor]:
        del base  # the update reads the adapter's tensors, not the weight it is added to
        delta = self._joined_delta(adapter)
        if self.bias is None:
            return {_WEIGHT: delta}
        return {_WEIGHT: delta[:, : self.in_features], _BIAS: delta[:, self.in_features]}

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        projection = self.adamss_newB[adapter]
        wide = x.to(projection.dtype)
        # ``[x; 1]`` times the projection: the bias column of the projection takes the place of the 1.
        reduced = wide @ projection[:, : self.in_features].mT
        if self.bias is not None:
            reduced = reduced + projection[:, self.in_features]
        blocks = [
            (reduced @ a.mT) @ b.mT
            for a, b in zip(
                _parameters(self.adamss_A, adapter),
                _parameters(self.adamss_B, adapter),
                strict=True,
            )
        ]
        placed: Tensor = lucid.cat(blocks, dim=-1).index_select(-1, self.adamss_rows[adapter])
        return placed


class Linear(AdamssLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` whose rows are split into subspaces, each with its own low-rank update.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`AdamssLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import AdamssConfig
    >>> from lucid.peft.tuners.adamss import Linear
    >>> base = nn.Linear(8, 6)
    >>> layer = Linear(base, "default", AdamssConfig(r=4, num_subspaces=2, target_modules=["x"]))
    >>> layer.num_subspaces["default"], sum(b.shape[0] for b in layer.adamss_B["default"].parameters())
    (2, 6)
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # every B starts at zero
    True
    """

    base_layer: nn.Linear
