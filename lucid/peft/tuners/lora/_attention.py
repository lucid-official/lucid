"""LoRA on ``nn.MultiheadAttention``: the fused input projection and the output projection."""

import warnings
from typing import Final, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners.lora._layer import _Factors, _Place
from lucid.peft.tuners.lora._weight_space import _WeightSpace

#: Where an adapter file keeps what this layer holds in ``out_proj_lora_*``: the reference nests
#: a second LoRA layer under the attention's ``out_proj``, which Lucid's flat layout has no module for.
_DISK_TABLES: Final = {
    "out_proj_lora_A": "base_layer.out_proj.lora_A",
    "out_proj_lora_B": "base_layer.out_proj.lora_B",
}

_IN_PROJ: Final = "in_proj_weight"
_OUT_PROJ: Final = "out_proj_weight"


class MultiheadAttention(
    _WeightSpace[tuple[Tensor, Tensor | None]], base_type=nn.MultiheadAttention
):
    r"""An ``nn.MultiheadAttention`` with LoRA on both its projections.

    The attention holds its projections as raw parameters — the fused
    ``in_proj_weight`` :math:`(3E, E)` and ``out_proj_weight``
    :math:`(E, E)` — and reads them inside its forward, so the adapter is a
    change of those weights (see ``_WeightSpace``): each adapter has one
    pair for the input projection (``lora_A`` :math:`(r, E)`, ``lora_B``
    :math:`(3E, r)`) and one for the output projection (``out_proj_lora_A``
    :math:`(r, E)`, ``out_proj_lora_B`` :math:`(E, r)`).  There is no way
    to adapt one of them alone.  In an adapter file the second pair is
    keyed as the reference keys it, ``....base_layer.out_proj.lora_A.weight``.

    The attention's public attributes (``embed_dim``, ``num_heads``,
    ``batch_first``, ``in_proj_weight`` …) read through to the wrapped
    module.  Only attention with one fused input projection (``kdim`` and
    ``vdim`` equal to ``embed_dim``, no grouped key/value heads) is
    supported.  ``lora_dropout`` is not applied: the update is folded into
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
        ``r`` is not positive, or the attention has separate query, key and
        value projections.

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
    """

    adapter_layer_names = ("lora_A", "lora_B", "out_proj_lora_A", "out_proj_lora_B")
    base_layer: nn.MultiheadAttention

    @override
    def _setup(self) -> None:
        base = self.base_layer
        if not base._qkv_same_embed_dim:
            raise ValueError(
                "Only same embed for query/key/value is supported as of now for "
                "lora.MultiheadAttention (kdim and vdim equal to embed_dim, no grouped "
                "key/value heads)."
            )
        self.lora_A: nn.ModuleDict[nn.Linear] = nn.ModuleDict()
        self.lora_B: nn.ModuleDict[nn.Linear] = nn.ModuleDict()
        self.out_proj_lora_A: nn.ModuleDict[nn.Linear] = nn.ModuleDict()
        self.out_proj_lora_B: nn.ModuleDict[nn.Linear] = nn.ModuleDict()

    @override
    def _disk_table(self, table_name: str) -> str:
        return _DISK_TABLES.get(table_name, table_name)

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
        base = self.base_layer
        if base.in_proj_weight is None:
            raise ValueError("the attention has no fused in_proj_weight to adapt")
        return {_IN_PROJ: base.in_proj_weight, _OUT_PROJ: base.out_proj_weight}

    @override
    def _build_factors(self, r: int, place: _Place) -> _Factors:
        e = self.base_layer.embed_dim
        return (place.linear(e, r), place.linear(r, 3 * e), place.linear(e, r), place.linear(r, e))

    @override
    def _delta_for(self, name: str, adapter: str) -> Tensor:
        if name == _IN_PROJ:
            lora_a, lora_b = self.lora_A[adapter].weight, self.lora_B[adapter].weight
        else:
            lora_a = self.out_proj_lora_A[adapter].weight
            lora_b = self.out_proj_lora_B[adapter].weight
        return (lora_b @ lora_a) * self._specs[adapter].scaling

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`s\,BA` for the fused input projection, ``(3E, E)``.

        The output projection's update is
        ``s * out_proj_lora_B @ out_proj_lora_A``; a merge writes both.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(3 * embed_dim, embed_dim)``, in the adapter's dtype.
        """
        return self._delta_for(_IN_PROJ, adapter)
