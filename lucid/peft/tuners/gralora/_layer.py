r"""The GraLoRA layers: a frozen base module beside a grid of small low-rank updates that exchange information.

GraLoRA (Jung et al., 2025, arXiv:2505.20355, section 3.1) cuts the weight into a ``k x k`` grid
of blocks and gives each its own low-rank pair of rank ``r / k`` (Eq. 5): the block that maps input
block :math:`n` to output block :math:`j` is :math:`B_{j,n} A_{j,n}^\top`.  The pairs are not
independent parameters: input block :math:`n` owns one matrix :math:`A_n \in \mathbb{R}^{N/k \times r}`
whose ``k`` runs of ``r / k`` columns are the :math:`A_{j,n}`, and output block :math:`j` owns one
:math:`B_j \in \mathbb{R}^{r \times M/k}` whose runs of rows are the :math:`B_{j,n}^\top`.  The
parameter count is a rank-``r`` LoRA's and the update has rank up to ``k r`` (section 3.2).  With
``hybrid_r > 0`` a plain LoRA of that rank is added beside the grid (section 3.5, Hybrid GraLoRA),
and the sum is scaled by :math:`\alpha / (r + \text{hybrid\_r})`.

``GraloraLayer`` is the family root.  An adapter holds ``gralora_A`` (``(k, in/k, r)``),
``gralora_B`` (``(k, r, out/k)``) and, with a hybrid part, the linear pair
``gralora_A_general`` / ``gralora_B_general``, the reference's keys.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._delta import _DeltaLayer, _Home
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._support import _LinearSized, _ResidualUpdate, _SpecField
from lucid.peft.tuners.gralora._config import GraloraConfig


class _GraloraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the scale is
    baked into the graph.
    """

    r: int
    alpha: float
    gralora_k: int
    hybrid_r: int
    scaling: float


def _uniform(home: _Home, shape: tuple[int, ...], fan_in: int) -> Tensor:
    """A tensor of ``shape`` drawn from the uniform distribution ``Linear`` starts with: ``(-1, 1) / sqrt(fan_in)``."""
    draw = lucid.rand(*shape, device=home.device, dtype=home.dtype)
    return (draw * 2.0 - 1.0) * fan_in**-0.5


class GraloraLayer(_DeltaLayer[GraloraConfig], _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the block-grid update shared by the GraLoRA layers.

    Per adapter: ``gralora_A``, ``gralora_B``, the hybrid pair (modules, or ``nn.Identity`` when
    ``hybrid_r`` is zero), a dropout of the input, and the read-only mappings ``r``, ``alpha``,
    ``gralora_k``, ``hybrid_r`` and ``scaling``.  The forward pass computes the update in its
    factored form, two batched products per adapter and a regrouping of the middle, at the cost of
    a LoRA of rank ``r``; ``get_delta_weight`` is the dense matrix the same computation gives the
    identity.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` to cut into blocks; its parameters stay shared.
    adapter_name : str
        Name of the grid made at once.
    config : GraloraConfig
        ``r``, ``hybrid_r``, ``alpha``, ``gralora_dropout``, ``gralora_k`` and ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is another module type.
    ValueError
        ``config.fan_in_fan_out`` is set (the layer's weight is stored ``(out, in)``), ``gralora_k``
        does not divide the layer's input and output sizes, or ``base_layer`` builds its
        parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = (
        "gralora_A",
        "gralora_B",
        "gralora_A_general",
        "gralora_B_general",
    )
    _dropout_table = "gralora_dropout"
    _repr_prefix = "gralora."
    gralora_A: nn.ParameterDict
    gralora_B: nn.ParameterDict
    gralora_A_general: nn.ModuleDict[nn.Module]
    gralora_B_general: nn.ModuleDict[nn.Module]
    gralora_dropout: nn.ModuleDict[nn.Module]

    r = _SpecField[int]("Rank of the grid of each adapter, by name (read-only).")
    alpha = _SpecField[float]("Scaling numerator of each adapter, by name (read-only).")
    gralora_k = _SpecField[int](
        "Blocks along each side of each adapter's grid, by name (read-only)."
    )
    hybrid_r = _SpecField[int]("Rank of the plain LoRA beside each adapter's grid (read-only).")
    scaling = _SpecField[float]("Scale of each adapter's update, by name (read-only).")

    @override
    def _prepare(self, config: GraloraConfig) -> None:
        if config.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        super()._prepare(config)
        # The hybrid pair are modules, unlike the grid's bare parameters.
        self.gralora_A_general = nn.ModuleDict()
        self.gralora_B_general = nn.ModuleDict()

    def _spec(self, adapter: str) -> _GraloraSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _GraloraSpec)
        return spec

    @override
    def _enter(self, adapter_name: str, config: GraloraConfig) -> None:
        k, r, hybrid = config.gralora_k, config.r, config.hybrid_r
        if self.in_features % k:
            raise ValueError(
                f"in_features should be divisible by gralora_k, but got {self.in_features} and {k}"
            )
        if self.out_features % k:
            raise ValueError(
                f"out_features should be divisible by gralora_k, but got {self.out_features} and {k}"
            )
        home = self._home()
        sub_in, sub_out = self.in_features // k, self.out_features // k
        grid_a = _uniform(home, (k, sub_in, r), sub_in)
        grid_b = (
            _uniform(home, (k, r, sub_out), r)
            if not config.init_weights
            else home.zeros(k, r, sub_out)
        )
        general_a: nn.Module = nn.Identity()
        general_b: nn.Module = nn.Identity()
        if hybrid > 0:
            general_a, general_b = (
                home.linear(self.in_features, hybrid),
                home.linear(hybrid, self.out_features),
            )
            init.kaiming_uniform_(general_a.weight, a=5**0.5)
            if config.init_weights:
                init.zeros_(general_b.weight)
            else:
                init.kaiming_uniform_(general_b.weight, a=5**0.5)
        self._specs[adapter_name] = _GraloraSpec(
            r, config.alpha, k, hybrid, config.alpha / (r + hybrid)
        )
        self.gralora_A[adapter_name] = Parameter(grid_a)
        self.gralora_B[adapter_name] = Parameter(grid_b)
        self.gralora_A_general[adapter_name] = general_a
        self.gralora_B_general[adapter_name] = general_b
        self._enter_dropout(adapter_name, config.gralora_dropout)

    def _block_update(self, adapter: str, x: Tensor) -> Tensor:
        r"""The unscaled update of ``adapter`` applied to ``x``, the grid's and the hybrid LoRA's.

        The block of ``x`` the ``n``-th input block reads goes through :math:`A_n`; its result,
        ``r`` wide, is cut into ``k`` runs of ``r / k``, run :math:`j` going to output block
        :math:`j`, which concatenates the runs it receives (one per input block) and goes through
        :math:`B_j`.
        """
        spec = self._spec(adapter)
        k, rank = spec.gralora_k, spec.r
        grid_a = _parameter_at(self.gralora_A, adapter)
        grid_b = _parameter_at(self.gralora_B, adapter)
        x = x.to(grid_a.dtype)
        rows = x.reshape(-1, self.in_features)  # (M, in)
        blocks = rows.reshape(rows.shape[0], k, self.in_features // k).transpose(
            0, 1
        )  # (k, M, in/k)
        mixed = (blocks @ grid_a).reshape(k, rows.shape[0], k, rank // k)  # (n, M, j, r/k)
        regrouped = mixed.permute(2, 1, 0, 3).reshape(k, rows.shape[0], rank)  # (j, M, n * r/k)
        grid = (regrouped @ grid_b).transpose(0, 1).reshape(rows.shape[0], self.out_features)
        if spec.hybrid_r > 0:
            grid = grid + self.gralora_B_general[adapter](self.gralora_A_general[adapter](rows))
        out: Tensor = grid.reshape(*x.shape[:-1], self.out_features)
        return out

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return the dense update, :math:`s\,(R_{\text{grid}} + B_{\text{hybrid}} A_{\text{hybrid}})`.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the factors' dtype.
        """
        grid_a = _parameter_at(self.gralora_A, adapter)
        probe = lucid.eye(self.in_features, device=grid_a.device, dtype=grid_a.dtype)
        delta: Tensor = self._block_update(adapter, probe).mT * self._spec(adapter).scaling
        return delta

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        update: Tensor = (
            self._block_update(adapter, self._dropped(adapter, x)) * self._spec(adapter).scaling
        )
        return update


class Linear(GraloraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a GraLoRA update: :math:`y = (W + \Delta W)\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`GraloraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import GraloraConfig
    >>> from lucid.peft.tuners.gralora import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", GraloraConfig(r=4, gralora_k=2, target_modules=["x"]))
    >>> tuple(layer.gralora_A["default"].shape), tuple(layer.gralora_B["default"].shape)
    ((2, 4, 4), (2, 4, 2))
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # B starts at zero
    True
    """

    base_layer: nn.Linear
