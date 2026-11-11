r"""The PSOFT layers: a frozen linear module beside a rotation of its principal subspace.

PSOFT (Wu et al., 2025, arXiv:2505.11235, section 4) takes the top-:math:`r` singular triplets of the
weight as two frozen factors, trains an :math:`r \times r` matrix
:math:`C = \operatorname{diag}(\beta)\,R\,\operatorname{diag}(\alpha)` between them (Eq. 8: :math:`R`
orthogonal through the Cayley parameterisation, the vectors starting at one), and leaves the rest of
the weight alone.  The paper writes the weight :math:`d \times n` and the layer :math:`h = W^\top x`;
in the ``out x in`` layout of ``nn.Linear`` (which the reference also uses) that is

.. math:: W' = W_{res} + U_r \Sigma_r\,C\,V_r^\top, \qquad \Delta W = U_r \Sigma_r\,(C - I)\,V_r^\top ,

the singular values on the output side of :math:`C` (Eq. 6 transposed).  With :math:`R = I` and the
vectors at one the update is zero, so the adapted layer starts as the base.  The factors are computed
once, from the weight at the time the adapter was made, and kept in non-persistent buffers (an adapter
file holds the trained matrix only; the factors come back from the base weights), so
:math:`\Delta W = s\,B\,(C - I)\,A` of those and merging adds it.

``PsoftLayer`` is the family root.  An adapter holds ``psoft_R`` (an :class:`OrthLayer`: ``weight``,
``vector_b``, ``vector_a``), the reference's keys, the buffers ``_psoft_A_cache`` and
``_psoft_B_cache``, a dropout of the input, and the read-only mappings of the scalars.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._shared import _BufferDict
from lucid.peft.tuners._support import _LinearSized, _ResidualUpdate, _SpecField
from lucid.peft.tuners.psoft._config import PsoftConfig
from lucid.peft.tuners.psoft._orth import OrthLayer
from lucid.peft.tuners.psoft._subspace import _principal_factors


class _PsoftSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the rank and the
    scale shape the graph.
    """

    r: int
    psoft_alpha: float
    scaling: float
    ab_svd_init: str
    psoft_svd: str
    psoft_svd_lowrank_niter: int
    random_seed: int


class PsoftLayer(_DeltaLayer[PsoftConfig], _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the principal-subspace update shared by the PSOFT layers.

    Per adapter: ``psoft_R``, the frozen factors in ``_psoft_A_cache`` and ``_psoft_B_cache``, a
    dropout of the input, and the read-only mappings ``r``, ``psoft_alpha``, ``scaling``,
    ``ab_svd_init``, ``psoft_svd``, ``psoft_svd_lowrank_niter`` and ``random_seed``.
    ``get_delta_weight`` is :math:`\Delta W = s\,B\,(C - I)\,A`; the forward pass applies it to the
    input as :math:`s\,(x A^\top C^\top - x A^\top)\,B^\top`, never forming the ``out x in`` matrix.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose principal subspace is rotated; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : PsoftConfig
        The rank, scale, SVD options, the options of the rotation and ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is some other module type.
    ValueError
        ``base_layer`` has parameters that only its first input will create, or ``r`` exceeds the
        smaller side of its weight.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("psoft_R",)
    _dropout_table = "psoft_dropout"
    _repr_prefix = "psoft."
    psoft_R: nn.ModuleDict[nn.Module]
    psoft_dropout: nn.ModuleDict[nn.Module]
    _psoft_A_cache: _BufferDict
    _psoft_B_cache: _BufferDict

    r = _SpecField[int]("Rank of the principal subspace of each adapter, by name (read-only).")
    psoft_alpha = _SpecField[float]("The alpha of each adapter, by name (read-only).")
    scaling = _SpecField[float](
        "Scale ``alpha / r`` of each adapter's update, by name (read-only)."
    )
    ab_svd_init = _SpecField[str]("How each adapter's factors were built, by name (read-only).")
    psoft_svd = _SpecField[str]("The SVD backend of each adapter, by name (read-only).")
    psoft_svd_lowrank_niter = _SpecField[int](
        "Power iterations of each adapter's low-rank SVD, by name (read-only)."
    )
    random_seed = _SpecField[int]("Seed of each adapter's low-rank SVD, by name (read-only).")

    @override
    def _new_table(self) -> nn.ModuleDict[nn.Module]:
        return nn.ModuleDict()

    @override
    def _prepare(self, config: PsoftConfig) -> None:
        super()._prepare(config)
        self._psoft_A_cache = _BufferDict(persistent=False)
        self._psoft_B_cache = _BufferDict(persistent=False)

    @override
    def _enter(self, adapter_name: str, config: PsoftConfig) -> None:
        shortest = min(self.in_features, self.out_features)
        if config.r > shortest:
            raise ValueError(
                f"`r` ({config.r}) must be less than or equal to min(in_features, out_features) "
                f"({self.in_features}, {self.out_features})"
            )
        home = self._home()
        scaling = config.psoft_alpha / config.r
        factors = _principal_factors(self.weight.to(home.dtype), config, scaling)
        self._specs[adapter_name] = _PsoftSpec(
            config.r,
            config.psoft_alpha,
            scaling,
            config.ab_svd_init,
            config.psoft_svd,
            config.psoft_svd_lowrank_niter,
            config.random_seed,
        )
        self.psoft_R[adapter_name] = OrthLayer(
            config.r,
            config.psoft_orth,
            config.psoft_mag_b,
            config.psoft_mag_a,
            config.use_cayley_neumann,
            config.num_cayley_neumann_terms,
            config.cayley_neumann_eps,
            config.init_weights,
            device=home.device,
            dtype=home.dtype,
        )
        self._psoft_A_cache[adapter_name], self._psoft_B_cache[adapter_name] = factors
        self._enter_dropout(adapter_name, config.psoft_dropout)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its matrix, factors, recorded settings and dropout."""
        super().delete_adapter(adapter_name)
        for cache in (self._psoft_A_cache, self._psoft_B_cache):
            if adapter_name in cache:
                del cache[adapter_name]

    # ── the update ────────────────────────────────────────────────────────

    def _spec(self, adapter: str) -> _PsoftSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _PsoftSpec)
        return spec

    def _orth(self, adapter: str) -> OrthLayer:
        orth = self.psoft_R[adapter]
        assert isinstance(orth, OrthLayer)
        return orth

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`\Delta W = s\,B\,(C - I)\,A`, the change ``adapter`` makes to the weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the factors' dtype.
        """
        matrix = self._orth(adapter).get_matrix()
        identity = lucid.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        delta: Tensor = (
            self._psoft_B_cache[adapter] @ (matrix - identity) @ self._psoft_A_cache[adapter]
        ) * self._spec(adapter).scaling
        return delta

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        factor_a, factor_b = self._psoft_A_cache[adapter], self._psoft_B_cache[adapter]
        projected = self._dropped(adapter, x).to(factor_a.dtype) @ factor_a.mT
        turned = self._orth(adapter)(projected)
        out: Tensor = ((turned - projected) @ factor_b.mT) * self._spec(adapter).scaling
        return out


class Linear(PsoftLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a PSOFT update: :math:`y = (W + U_r\Sigma_r (C - I) V_r^\top)\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`PsoftLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import PsoftConfig
    >>> from lucid.peft.tuners.psoft import Linear
    >>> base = nn.Linear(8, 6)
    >>> layer = Linear(base, "default", PsoftConfig(r=3, target_modules=["x"]))
    >>> sorted(layer.psoft_R["default"].state_dict()), tuple(layer.psoft_R["default"].weight.shape)
    (['vector_a', 'vector_b', 'weight'], (3,))
    >>> x = lucid.randn(4, 8)
    >>> bool((layer(x) == base(x)).all().item())   # R = I and the vectors are one
    True
    """

    base_layer: nn.Linear
