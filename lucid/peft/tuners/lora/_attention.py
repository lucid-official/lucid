"""LoRA on ``nn.MultiheadAttention``: its input projection(s) and the output projection."""

import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners.lora._layer import _Factors, _Place, _weight_of
from lucid.peft.tuners.lora._weight_space import _WeightSpace

type _Slots = Mapping[str, tuple[str, str]]


def _slots(**tables: tuple[str, str]) -> _Slots:
    """The weights an attention layer adapts, in order, each with its ``(A, B)`` factor tables."""
    return MappingProxyType(tables)


def _table_names(slots: _Slots) -> tuple[str, ...]:
    return tuple(name for pair in slots.values() for name in pair)


def _disk_tables(slots: _Slots) -> Mapping[str, str]:
    """Where an adapter file keeps a table the reference has no module for.

    The reference nests a LoRA layer under the attention's ``out_proj``; Lucid's flat layout has
    no module there, so each table named ``<stem>_lora_A`` is keyed as ``base_layer.<stem>.lora_A``.
    The input projection's own tables (``lora_A`` / ``lora_B``) are keyed as they are.
    """
    keyed: dict[str, str] = {}
    for table_a, table_b in slots.values():
        stem = table_a.removesuffix("_lora_A")
        if stem != table_a:
            keyed[table_a] = f"base_layer.{stem}.lora_A"
            keyed[table_b] = f"base_layer.{stem}.lora_B"
    return MappingProxyType(keyed)


class MultiheadAttention(
    _WeightSpace[tuple[Tensor, Tensor | None]], base_type=nn.MultiheadAttention
):
    r"""An ``nn.MultiheadAttention`` with LoRA on its input and output projections.

    The attention holds its projections as raw parameters and reads them
    inside its forward, so the adapter is a change of those weights (see
    ``_WeightSpace``).  There is no way to adapt one projection alone: each
    adapter has a pair of factors for every weight.

    * With one fused input projection (``kdim`` and ``vdim`` equal to
      ``embed_dim``, no grouped key/value heads) the weights are
      ``in_proj_weight`` :math:`(3E, E)` — ``lora_A`` :math:`(r, E)`,
      ``lora_B`` :math:`(3E, r)` — and ``out_proj_weight`` :math:`(E, E)` —
      ``out_proj_lora_A`` :math:`(r, E)`, ``out_proj_lora_B`` :math:`(E, r)`.
    * With separate projections (``kdim`` or ``vdim`` apart from
      ``embed_dim``, as in a cross-attention over a context of another
      width, or grouped key/value heads) the query, key and value weights
      each have a pair: ``q_proj_lora_*`` :math:`(r, E)` / :math:`(E, r)`,
      ``k_proj_lora_*`` :math:`(r, E_k)` / :math:`(E_{kv}, r)` and
      ``v_proj_lora_*`` likewise, beside the output projection's.  The reference
      refuses this layout; constructing the layer on such an attention
      returns the layout's own subclass, which is still a
      ``MultiheadAttention``.

    In an adapter file the tables that the reference would nest under a
    module are keyed as it keys them, ``....base_layer.out_proj.lora_A.weight``.
    The attention's public attributes (``embed_dim``, ``num_heads``,
    ``batch_first``, ``in_proj_weight`` …) read through to the wrapped
    module.  ``lora_dropout`` is not applied: the update is folded into
    the weights, with no activations to drop.

    Parameters
    ----------
    base_layer : nn.MultiheadAttention
        The attention to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in every factor table.
    r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
        The adapter's hyper-parameters, as ``LoraLayer.update_layer`` takes
        them; ``r`` must be positive.  ``lora_dropout`` is accepted but the
        update is folded into the weights, so there is nothing to drop
        (a nonzero value warns).  ``use_dora`` and ``lora_bias`` are refused
        (the reference does not support DoRA here either), as are the
        initialisations that factorise a weight.

    Raises
    ------
    ValueError
        ``r`` is not positive.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.lora import MultiheadAttention
    >>> base = nn.MultiheadAttention(8, 2, batch_first=True)
    >>> layer = MultiheadAttention(base, "default", r=2, lora_alpha=4)
    >>> tuple(layer.lora_B["default"].weight.shape), layer.embed_dim
    ((24, 2), 8)
    >>> x = lucid.randn(1, 5, 8)
    >>> bool((layer(x, x, x)[0] == base(x, x, x)[0]).all().item())   # every B starts at zero
    True
    >>> cross = nn.MultiheadAttention(8, 2, kdim=6, vdim=6, batch_first=True)
    >>> tuple(MultiheadAttention(cross, "default", r=2).k_proj_lora_A["default"].weight.shape)
    (2, 6)
    """

    #: The weights adapted, by attribute name on the attention, each with its ``(A, B)`` tables.
    _SLOTS: ClassVar[_Slots] = _slots(
        in_proj_weight=("lora_A", "lora_B"),
        out_proj_weight=("out_proj_lora_A", "out_proj_lora_B"),
    )
    adapter_layer_names = _table_names(_SLOTS)
    _DISK: ClassVar[Mapping[str, str]] = _disk_tables(_SLOTS)

    base_layer: nn.MultiheadAttention
    lora_A: nn.ModuleDict[nn.Linear]
    lora_B: nn.ModuleDict[nn.Linear]
    out_proj_lora_A: nn.ModuleDict[nn.Linear]
    out_proj_lora_B: nn.ModuleDict[nn.Linear]

    def __new__(cls, *args: object, **kwargs: object) -> MultiheadAttention:
        # Asked for by the registry as ``MultiheadAttention``: build the layout the attention
        # needs, a subclass saying which attentions it takes (``holds``).
        base = args[0] if args else kwargs.get("base_layer")
        if cls is MultiheadAttention and isinstance(base, nn.MultiheadAttention):
            cls = next((k for k in cls.__subclasses__() if k.holds(base)), cls)
        return super().__new__(cls)

    @classmethod
    def holds(cls, base: nn.MultiheadAttention) -> bool:
        """Whether this layout adapts ``base``: the one with a fused input projection."""
        return base._qkv_same_embed_dim

    @override
    def _setup(self) -> None:
        if not type(self).holds(self.base_layer):
            raise ValueError(
                f"{type(self).__name__} does not adapt an attention with "
                f"kdim={self.base_layer.kdim}, vdim={self.base_layer.vdim}, "
                f"num_kv_heads={self.base_layer.num_kv_heads}"
            )
        self._make_tables()

    def _make_tables(self) -> None:
        """Create the empty factor tables (the layout's, one ``(A, B)`` pair per weight)."""
        for table in self.adapter_layer_names:
            setattr(self, table, nn.ModuleDict())

    @override
    def _disk_table(self, table_name: str) -> str:
        return self._DISK.get(table_name, table_name)

    @override
    def _check_dropout(self, p: float) -> None:
        if p > 0:
            warnings.warn(
                "lora_dropout is not applied by lora.MultiheadAttention: its update is folded "
                "into the projection weights",
                stacklevel=2,
            )

    @override
    def _weights(self) -> dict[str, Parameter]:
        found: dict[str, Parameter] = {}
        for name in self._SLOTS:
            weight = getattr(self.base_layer, name)
            if not isinstance(weight, Parameter):
                raise TypeError(f"the attention has no {name} to adapt")
            found[name] = weight
        return found

    @override
    def _build_factors(self, r: int, place: _Place) -> _Factors:
        return tuple(
            factor
            for weight in self._weights().values()
            for factor in (place.linear(weight.shape[1], r), place.linear(r, weight.shape[0]))
        )

    def _factor(self, table: str, adapter: str) -> Parameter:
        found = getattr(self, table)
        if not isinstance(found, nn.ModuleDict):
            raise TypeError(f"{table} is not a factor table")
        return _weight_of(found[adapter])

    @override
    def _delta_for(self, name: str, adapter: str) -> Tensor:
        table_a, table_b = self._SLOTS[name]
        lora_a, lora_b = self._factor(table_a, adapter), self._factor(table_b, adapter)
        return (lora_b @ lora_a) * self._specs[adapter].scaling

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`s\,BA` for the first input projection: the fused one, ``(3E, E)``.

        The other weights' updates are ``s * B @ A`` of their own tables (the
        output projection's is ``s * out_proj_lora_B @ out_proj_lora_A``); a
        merge writes all of them.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(3 * embed_dim, embed_dim)`` with a fused input projection, the
            query projection's ``(embed_dim, embed_dim)`` with separate ones,
            in the adapter's dtype.
        """
        return self._delta_for(next(iter(self._SLOTS)), adapter)


class _SeparateMultiheadAttention(MultiheadAttention):
    """The layout of an attention with separate query, key and value projections.

    ``kdim`` or ``vdim`` apart from ``embed_dim`` (a cross-attention over a
    context of another width), or grouped key/value heads: the weights are
    ``q_proj_weight`` :math:`(E, E)`, ``k_proj_weight`` :math:`(E_{kv}, E_k)`,
    ``v_proj_weight`` :math:`(E_{kv}, E_v)` and ``out_proj_weight``, each with its own
    pair of factors.
    """

    _SLOTS: ClassVar[_Slots] = _slots(
        q_proj_weight=("q_proj_lora_A", "q_proj_lora_B"),
        k_proj_weight=("k_proj_lora_A", "k_proj_lora_B"),
        v_proj_weight=("v_proj_lora_A", "v_proj_lora_B"),
        out_proj_weight=("out_proj_lora_A", "out_proj_lora_B"),
    )
    adapter_layer_names = _table_names(_SLOTS)
    _DISK: ClassVar[Mapping[str, str]] = _disk_tables(_SLOTS)

    @classmethod
    @override
    def holds(cls, base: nn.MultiheadAttention) -> bool:
        return not base._qkv_same_embed_dim
