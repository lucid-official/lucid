r"""The AdaLoRA layers: a frozen base module with an SVD-parameterised update beside it.

An adapter is three tensors, the reference's names for them —
``lora_A`` (:math:`Q`, ``(r, in)``: the right singular vectors), ``lora_E``
(:math:`\lambda`, ``(r, 1)``: the singular values) and ``lora_B`` (:math:`P`,
``(out, r)``: the left singular vectors) — and the current rank ``ranknum``,
a frozen scalar that keeps the scale :math:`\alpha / \mathrm{ranknum}` the
one training began with after triplets are pruned away.  Each of the ``r``
rows of ``lora_A``, entries of ``lora_E`` and columns of ``lora_B`` is a
*triplet* the allocator (``lucid.peft.tuners.adalora.RankAllocator``) can
mask or drop; ``update = P diag(E) Q * alpha / ranknum``.
"""

from typing import ClassVar, Final, NamedTuple, override

import lucid
import lucid.linalg
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._base import _check_adapter_name, _parameter_at
from lucid.peft.tuners._support import _adapter_dtype, _ResidualUpdate, _SpecField
from lucid.peft.tuners._kinds import _Geometry, _Kind
from lucid.peft.tuners.adalora._config import AdaLoraConfig

#: Added to ``ranknum`` before dividing by it, so a layer pruned to rank 0 divides by a number.
_RANKNUM_EPS: Final = 1e-5
#: The standard deviation the singular vectors start with.
_VECTOR_STD: Final = 0.02


class _AdaSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameters.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes
    it: the scale is baked into the graph, and so is the current rank.
    """

    r: int
    lora_alpha: float
    scaling: float


def _departure(gram: Tensor) -> Tensor:
    """The Frobenius norm of ``gram - I``: how far a Gram matrix is from the identity."""
    identity = lucid.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return lucid.linalg.norm(gram - identity, ord="fro")


class AdaLoraLayer(_ResidualUpdate, family=True):
    r"""Adapter bookkeeping shared by the AdaLoRA layers.

    Holds the wrapped module as ``base_layer`` — the very module, so its
    parameters stay the same ``Parameter`` objects — and, per adapter name,
    the three tensors in ``lora_A`` / ``lora_B`` / ``lora_E`` and the
    frozen ``ranknum`` (``ParameterDict``\ s, so a state dict reads
    ``....lora_E.default`` and an adapter file ``....lora_E``).  The forward
    pass adds the update of every active adapter this layer holds.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt.
    adapter_name : str
        Name of the adapter created at once (``update_layer``).
    config : AdaLoraConfig
        The config; dropout and ``init_lora_weights`` are read.
    r : int, default=0
        Initial rank — the number of singular triplets; 0 is allowed, as in
        the reference (a layer the allocator pruned away entirely).
    lora_alpha : float, default=1
        Numerator of the scale; 0 or below takes ``r``.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``r`` is negative, or ``base_layer`` builds parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("lora_A", "lora_B", "lora_E")

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Current rank of each adapter, by name (read-only).")
    lora_alpha = _SpecField[float](
        "The numerator :math:`\\alpha` of each adapter's scale, by name (read-only)."
    )
    scaling = _SpecField[float](
        "``lora_alpha`` of each adapter, or its initial rank where that is 0 (read-only)."
    )

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: AdaLoraConfig,
        r: int = 0,
        lora_alpha: float = 1,
    ) -> None:
        super().__init__(base_layer)
        self._specs: dict[str, _AdaSpec] = {}
        self.lora_A: nn.ParameterDict = nn.ParameterDict()
        self.lora_B: nn.ParameterDict = nn.ParameterDict()
        self.lora_E: nn.ParameterDict = nn.ParameterDict()
        # The initial rank, frozen — not in ``adapter_layer_names``: it trains nothing and is not saved.
        self.ranknum: nn.ParameterDict = nn.ParameterDict()
        self.lora_dropout: nn.ModuleDict[nn.Module] = nn.ModuleDict()
        self.update_layer(adapter_name, r, lora_alpha, config)

    # ── the layer ─────────────────────────────────────────────────────────

    @property
    def _geometry(self) -> _Geometry:
        return _Kind.of(type(self.base_layer)).geometry(self.base_layer)

    @property
    def in_features(self) -> int:
        """Inputs each output unit reads: the second dimension of ``lora_A``."""
        return self._geometry.flat_in

    @property
    def out_features(self) -> int:
        """Output units of the base layer: the first dimension of ``lora_B``."""
        return self._geometry.out

    @override
    def __repr__(self) -> str:
        return "adalora." + super().__repr__()

    @override
    def extra_repr(self) -> str:
        return (
            f"adapter={self.active_adapter!r}, r={dict(self.r)}, "
            f"merged={self.merged}, disabled={self.disable_adapters}"
        )

    @override
    def _adapter_names(self) -> list[str]:
        return list(self._specs)

    # ── creating, resizing and removing an adapter ───────────────────────

    def update_layer(
        self, adapter_name: str, r: int, lora_alpha: float, config: AdaLoraConfig
    ) -> None:
        r"""Create adapter ``adapter_name`` with ``r`` singular triplets.

        The tensors live on the base weight's device, in its dtype — float32
        for a half-precision base.  With ``init_lora_weights`` the singular
        values start at zero, so the adapted layer starts as the base.  A
        layer that holds an adapter already keeps applying the adapters it
        applied: the new one is inactive until ``set_adapter`` names it.

        Parameters
        ----------
        adapter_name : str
            Key of the adapter in each table.
        r : int
            Initial rank; 0 or more.
        lora_alpha : float
            Numerator of the scale; 0 or below takes ``r``.
        config : AdaLoraConfig
            Read for ``lora_dropout`` and ``init_lora_weights``.

        Raises
        ------
        ValueError
            The layer already holds an adapter of this name, or ``r`` is negative.
        """
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self._specs:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        if r < 0:
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")
        anchor, geometry = self.weight, self._geometry
        device, dtype = anchor.device, _adapter_dtype(anchor)
        # Built first, entered together: an allocation that fails leaves no half adapter.
        singular = Parameter(lucid.randn(r, 1, device=device, dtype=dtype))
        left = Parameter(lucid.randn(geometry.out, r, device=device, dtype=dtype))
        right = Parameter(lucid.randn(r, geometry.flat_in, device=device, dtype=dtype))
        rank = Parameter(
            lucid.full((1,), float(r), device=device, dtype=dtype), requires_grad=False
        )
        dropout: nn.Module = (
            nn.Dropout(p=config.lora_dropout) if config.lora_dropout > 0.0 else nn.Identity()
        )
        if config.init_lora_weights:
            init.zeros_(singular)
            init.normal_(right, mean=0.0, std=_VECTOR_STD)
            init.normal_(left, mean=0.0, std=_VECTOR_STD)
        self._specs[adapter_name] = _AdaSpec(
            r, lora_alpha, lora_alpha if lora_alpha > 0 else float(r)
        )
        self.lora_E[adapter_name] = singular
        self.lora_B[adapter_name] = left
        self.lora_A[adapter_name] = right
        self.ranknum[adapter_name] = rank
        self.lora_dropout[adapter_name] = dropout
        if not self.active_adapters:
            self.active_adapter = adapter_name

    def _replace_triplets(
        self, adapter: str, singular: Tensor, left: Tensor, right: Tensor
    ) -> None:
        """Swap ``adapter``'s three tensors for new ones of another rank, keeping ``requires_grad`` and ``ranknum``."""
        trains = _parameter_at(self.lora_E, adapter).requires_grad
        self.lora_E[adapter] = Parameter(singular, requires_grad=trains)
        self.lora_B[adapter] = Parameter(left, requires_grad=trains)
        self.lora_A[adapter] = Parameter(right, requires_grad=trains)
        self._specs[adapter] = self._specs[adapter]._replace(r=singular.shape[0])

    def prune(self, adapter: str, kept: list[int]) -> None:
        """Keep only the singular triplets at the indices ``kept`` (in order); the scale does not change.

        ``ranknum`` stays what it was, so the update of the kept triplets is the
        one they made before the others were dropped.

        Parameters
        ----------
        adapter : str
            An adapter this layer holds.
        kept : list of int
            Indices of the triplets to keep.

        Raises
        ------
        ValueError
            The layer holds no such adapter.
        IndexError
            An index is outside the adapter's rank.
        """
        self._require(adapter)
        rank = self._specs[adapter].r
        if any(not 0 <= i < rank for i in kept):
            raise IndexError(f"triplet indices {kept} are not all in [0, {rank})")
        index = lucid.tensor(kept, dtype=lucid.int64, device=self.weight.device)
        with lucid.no_grad():
            self._replace_triplets(
                adapter,
                _parameter_at(self.lora_E, adapter).index_select(0, index),
                _parameter_at(self.lora_B, adapter).index_select(1, index),
                _parameter_at(self.lora_A, adapter).index_select(0, index),
            )

    def resize(self, adapter: str, r: int) -> None:
        """Give ``adapter`` rank ``r``: the first triplets stay, extra ones start at zero; the scale does not change.

        Parameters
        ----------
        adapter : str
            An adapter this layer holds.
        r : int
            The new rank, 0 or more.

        Raises
        ------
        ValueError
            The layer holds no such adapter.
        """
        self._require(adapter)
        rank = self._specs[adapter].r
        if r <= rank:
            self.prune(adapter, list(range(r)))
            return
        extra = r - rank
        with lucid.no_grad():
            singular, left, right = (
                _parameter_at(table, adapter) for table in (self.lora_E, self.lora_B, self.lora_A)
            )
            device, dtype = singular.device, singular.dtype
            self._replace_triplets(
                adapter,
                lucid.cat([singular, lucid.zeros(extra, 1, device=device, dtype=dtype)], dim=0),
                lucid.cat(
                    [left, lucid.zeros(left.shape[0], extra, device=device, dtype=dtype)], dim=1
                ),
                lucid.cat(
                    [right, lucid.zeros(extra, right.shape[1], device=device, dtype=dtype)], dim=0
                ),
            )

    def _require(self, adapter: str) -> None:
        if adapter not in self._specs:
            raise ValueError(
                f"this layer holds no adapter {adapter!r}; it holds {list(self._specs)}"
            )

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        for table in (self.ranknum, self.lora_dropout):
            if adapter_name in table:
                del table[adapter_name]
        self._specs.pop(adapter_name, None)

    # ── the update ────────────────────────────────────────────────────────

    def _scale(self, adapter: str) -> Tensor:
        """``scaling / (ranknum + eps)``, the factor every update is multiplied by."""
        return self._specs[adapter].scaling / (_parameter_at(self.ranknum, adapter) + _RANKNUM_EPS)

    def _singular_rows(self, adapter: str) -> Tensor:
        """``diag(E) Q``: the rows of ``lora_A`` weighted by the singular values."""
        return _parameter_at(self.lora_A, adapter) * _parameter_at(self.lora_E, adapter)

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`P\,\mathrm{diag}(\lambda)\,Q \cdot \alpha / \mathrm{ranknum}`, shaped like the base weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            The change merging ``adapter`` makes to the base weight, in the adapter's dtype.
        """
        delta = _parameter_at(self.lora_B, adapter) @ self._singular_rows(adapter)
        return (delta * self._scale(adapter)).reshape(*self.weight.shape)

    def _adapter_input(self, adapter: str, x: Tensor) -> Tensor:
        """``x`` through the adapter's dropout, in the adapter's dtype."""
        dropped: Tensor = self.lora_dropout[adapter](x)
        dtype = _parameter_at(self.lora_A, adapter).dtype
        return dropped if dropped.dtype == dtype else dropped.to(dtype)

    def orthogonality(self, adapter: str) -> list[Tensor]:
        r"""How far :math:`QQ^\top` and :math:`P^\top P` are from the identity, as two Frobenius norms.

        What the penalty on the loss sums; it keeps the singular vectors
        orthonormal, so that the values in ``lora_E`` are the update's real
        singular values.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        list of Tensor
            ``[||Q Q^T - I||_F, ||P^T P - I||_F]``, scalars that carry gradient.
        """
        right = _parameter_at(self.lora_A, adapter)
        left = _parameter_at(self.lora_B, adapter)
        return [_departure(right @ right.mT), _departure(left.mT @ left)]


class SVDLinear(AdaLoraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with an SVD-parameterised update: :math:`y = W x + b + \tfrac{\alpha}{\mathrm{ranknum}}\,P\,\mathrm{diag}(\lambda)\,Q\,\mathrm{drop}(x)`.

    The update is applied as three small products, never as an ``(out, in)``
    matrix, so a layer pruned to a few triplets costs a few triplets.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in every table.
    config : AdaLoraConfig
        The adapter's config.
    r : int, default=0
        Initial rank.
    lora_alpha : float, default=1
        Numerator of the scale.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.adalora import AdaLoraConfig
    >>> from lucid.peft.tuners.adalora import SVDLinear
    >>> base = nn.Linear(8, 4)
    >>> layer = SVDLinear(base, "default", AdaLoraConfig(total_step=10), r=3, lora_alpha=6)
    >>> tuple(layer.lora_A["default"].shape), tuple(layer.lora_E["default"].shape)
    ((3, 8), (3, 1))
    >>> x = lucid.randn(2, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the singular values start at zero
    True
    """

    base_layer: nn.Linear

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        dropped = self._adapter_input(adapter, x)
        left = _parameter_at(self.lora_B, adapter)
        return (dropped @ self._singular_rows(adapter).mT @ left.mT) * self._scale(adapter)


class SVDConv2d(AdaLoraLayer, base_type=nn.Conv2d):
    r"""An ``nn.Conv2d`` whose update is an SVD-parameterised kernel, applied as the base applies its own.

    ``lora_A`` is ``(r, in_channels / groups * kH * kW)`` and the product
    :math:`P\,\mathrm{diag}(\lambda)\,Q` is reshaped to the base kernel; the
    stride, padding, dilation and groups are the base's.

    Parameters
    ----------
    base_layer : nn.Conv2d
        The layer to adapt; the other parameters are those of :class:`SVDLinear`.
    """

    base_layer: nn.Conv2d

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        kernel = self.get_delta_weight(adapter)
        return _Kind.of(nn.Conv2d).apply(self.base_layer, self._adapter_input(adapter, x), kernel)
