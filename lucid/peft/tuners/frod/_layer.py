r"""The FRoD layers: a frozen linear module beside a full-rank update from a sparse trainable matrix over a shared basis.

FRoD (Wan et al., 2025, arXiv:2512.23485) writes every weight of a category as :math:`W_i = U_i\Sigma_i V^\top`
over one shared basis :math:`V` (section "Hierarchical Joint Decomposition", Eq. 6-7, built in ``_basis``)
and trains the middle, :math:`\Sigma_i + S_i`, with :math:`S_i` a sparse matrix of a few off-diagonal entries
(Eq. 8): the adapted weight is :math:`W_i' = U_i(\Sigma_i + S_i)V^\top`, a full-rank change of a small number of
parameters.  At the start :math:`S_i = 0` and :math:`W_i' = W_i`, up to the rounding of the decomposition.  The
adapted weight *replaces* the base one: the layer computes
:math:`x\,W'^\top + b`, which merging writes as the base weight plus :math:`W' - W`, the base weight taken from
before any merge (``_PristineMerge``), so unmerging gives it back bit for bit.

``FrodLayer`` is the family root.  An adapter holds ``frod_lambda_l`` (:math:`\operatorname{diag}\Sigma_i`, ``(in,)``)
and ``frod_lambda_s_values`` (the entries of :math:`S_i`, ``(nnz,)``), the reference's keys, the frozen left
factor in the buffer ``frod_U`` (``(out, in)``, not in a file: it is made again from the base weight and the
basis), a dropout of the input and the read-only mapping ``r``.  The basis, the positions of the entries and
the size are the tuner's (``_tables``); a layer is bound to the category it reads.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._shared import _BufferDict
from lucid.peft.tuners._stash import _WEIGHT, _PristineMerge
from lucid.peft.tuners._support import _LinearSized, _SpecField
from lucid.peft.tuners.frod._basis import (
    _draw_pattern,
    _projection_from_weights,
    _sparse_count,
    _Sparse,
    _Start,
)
from lucid.peft.tuners.frod._config import FrodConfig
from lucid.peft.tuners.frod._tables import NAMES, _Projections
from collections.abc import Mapping

#: The category a layer used on its own files its basis under.
_ALONE = "weight"
#: The standard deviation of the ``init_weights=False`` start, the reference's.
_RANDOM_STD = 0.1


class _FrodSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int
    init_weights: bool


class FrodLayer(_DeltaLayer[FrodConfig], _LinearSized, _PristineMerge, family=True):
    r"""Adapter bookkeeping and the sparse-rotation update shared by the FRoD layers.

    Per adapter: ``frod_lambda_l``, ``frod_lambda_s_values``, the frozen ``frod_U``, a dropout of the input
    and the read-only mapping ``r`` (the output width, the reference's value).  ``get_delta_weight`` is
    :math:`U(\Sigma + S)V^\top - W`, with :math:`W` the weight from before any merge; the forward pass
    applies it to the input without forming it.  A layer built by a tuner is *bound* to the tuner's tables
    (``bind``) before it is used; one built on its own makes a basis of its own from its weight.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose weight is re-expressed; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : FrodConfig
        ``sparse_rate``, ``projection_prng_key``, ``regularization_alpha``, ``frod_dropout``, ``init_weights``.
    deferred : bool, default=False
        The tuner will bind the layer to its tables and start the adapter; nothing is derived yet.

    Raises
    ------
    TypeError
        ``base_layer`` is not an ``nn.Linear``.
    ValueError
        ``base_layer`` is lazy and has not run yet.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("frod_lambda_l", "frod_lambda_s_values")
    _dropout_table = "frod_dropout"
    _repr_prefix = "frod."
    frod_lambda_l: nn.ParameterDict
    frod_lambda_s_values: nn.ParameterDict
    frod_dropout: nn.ModuleDict[nn.Module]
    frod_U: _BufferDict

    r = _SpecField[int]("Output width of the weight of each adapter, by name (read-only).")

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: FrodConfig,
        *,
        deferred: bool = False,
    ) -> None:
        """Wrap ``base_layer`` and make adapter ``adapter_name`` on it; ``deferred`` leaves the start to the tuner."""
        self._awaiting = deferred
        self._projections: _Projections | None = None
        self._category = ""
        super().__init__(base_layer, adapter_name, config)

    @override
    def _prepare(self, config: FrodConfig) -> None:
        super()._prepare(config)
        self.frod_U = _BufferDict(persistent=False)

    @override
    def _enter(self, adapter_name: str, config: FrodConfig) -> None:
        home = self._home()
        self._enter_dropout(adapter_name, config.frod_dropout)
        self._specs[adapter_name] = _FrodSpec(self.out_features, config.init_weights)
        self.frod_lambda_l[adapter_name] = Parameter(home.zeros(self.in_features))
        self.frod_lambda_s_values[adapter_name] = Parameter(
            home.zeros(_sparse_count(self.in_features, config.sparse_rate))
        )
        self.frod_U[adapter_name] = home.zeros(self.out_features, self.in_features)
        if self._projections is not None:
            self._start(adapter_name)
        elif not self._awaiting:
            self._start_alone(adapter_name, config)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its trained tensors, frozen factor, recorded settings and dropout."""
        super().delete_adapter(adapter_name)
        if adapter_name in self.frod_U:
            del self.frod_U[adapter_name]

    # ── the shared tables ─────────────────────────────────────────────────

    def holds_tables(self) -> bool:
        """Whether this layer is the one that holds the shared tables in its module tree."""
        return NAMES[0] in self._modules

    def bind(self, projections: _Projections, category: str) -> None:
        """Read the basis of ``category`` from ``projections`` from now on, and start the adapters the layer holds.

        The first layer bound holds the tables in its tree.

        Parameters
        ----------
        projections : _Projections
            The tuner's tables of bases, positions and sizes.
        category : str
            The category this layer's basis is filed under.
        """
        self._projections = projections
        self._category = category
        self._awaiting = False
        projections.shared.attach(self)
        for adapter in self._adapter_names():
            if projections.has(category, adapter):
                self._start(adapter)

    def _bound(self) -> _Projections:
        assert self._projections is not None  # a bound layer
        return self._projections

    def _start_alone(self, adapter: str, config: FrodConfig) -> None:
        """Make a basis of the layer's own weight, for a layer built outside a tuner, and start from it."""
        home = self._home()
        weight = self.weight.detach().to("cpu").to(lucid.float32)
        basis = _projection_from_weights([weight], config.regularization_alpha)
        indices, size = _draw_pattern(
            self.in_features, config.sparse_rate, config.projection_prng_key, home.device
        )
        projections = _Projections()
        projections.put(
            _ALONE,
            adapter,
            (basis.to(device=home.device, dtype=home.dtype), indices, size),
            persistent=config.save_projection,
        )
        self.bind(projections, _ALONE)

    def _start(self, adapter: str) -> None:
        """Derive the frozen factor and the initial norms from the weight and the basis; a random start if asked."""
        self._derive(adapter)
        if not self._spec(adapter).init_weights:
            with lucid.no_grad():
                entries = _parameter_at(self.frod_lambda_s_values, adapter)
                init.normal_(entries, mean=0.0, std=_RANDOM_STD)
                norms = _parameter_at(self.frod_lambda_l, adapter)
                norms.copy_(
                    norms
                    + _RANDOM_STD
                    * lucid.randn(*norms.shape, device=norms.device, dtype=norms.dtype)
                )

    def _derive(self, adapter: str) -> None:
        """Make ``frod_U`` and the initial ``frod_lambda_l`` from the base weight and ``adapter``'s basis.

        The weight is the one before any merge, which a layer that holds merged adapters
        keeps in its stash: ``_restart`` runs after a file is loaded, outside the tuner's own
        start, with the layer as it is.
        """
        v, _, _ = self._bound().entry(self._category, adapter)
        home = self._home()
        start = _Start.of(self._pristine_weight(), v, home.dtype, home.device)
        self.frod_U[adapter] = start.u
        with lucid.no_grad():
            _parameter_at(self.frod_lambda_l, adapter).copy_(start.norms)

    def _restart(self, adapter: str) -> None:
        """The basis of ``adapter`` was replaced (by a file's): derive ``frod_U`` again, and size the entries to its positions."""
        self._derive(adapter)
        _, indices, _ = self._bound().entry(self._category, adapter)
        entries = _parameter_at(self.frod_lambda_s_values, adapter)
        if int(entries.shape[0]) != int(indices.shape[1]):
            home = self._home()
            self.frod_lambda_s_values[adapter] = Parameter(
                home.zeros(int(indices.shape[1])), requires_grad=entries.requires_grad
            )

    # ── the update ────────────────────────────────────────────────────────

    def _spec(self, adapter: str) -> _FrodSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _FrodSpec)
        return spec

    def _basis(self, adapter: str) -> tuple[Tensor, _Sparse]:
        v, indices, _ = self._bound().entry(self._category, adapter)
        # The basis is square, so its shape is the side of ``S``: no value is read back from the device.
        return v, _Sparse.of(indices, int(v.shape[0]))

    def _adapted(self, adapter: str, h: Tensor) -> Tensor:
        r""":math:`h\,W'^\top = h V (S + \Sigma)^\top U^\top` for ``h`` of width ``in``, in the adapter's dtype."""
        v, sparse = self._basis(adapter)
        flat = h.to(v.dtype).reshape(-1, self.in_features)
        z = flat @ v
        inner = sparse.apply(
            z, _parameter_at(self.frod_lambda_s_values, adapter)
        ) + z * _parameter_at(self.frod_lambda_l, adapter)
        out: Tensor = inner @ self.frod_U[adapter].mT
        return out.reshape(*h.shape[:-1], self.out_features)

    def _shift(self, adapter: str, x: Tensor, base_out: Tensor | None) -> Tensor:
        r""":math:`h\,(W' - W)^\top` for :math:`h` the input through the adapter's dropout.

        ``base_out`` is the base layer's output without its bias, if the caller has it: :math:`x\,W^\top` is
        then not computed again (it is when dropout changes the input, and where the layer is
        merged, whose output holds the merged adapters' change too).  :math:`W` is the weight as it was
        before any merge.
        """
        h = self._dropped(adapter, x)
        adapted = self._adapted(adapter, h)
        if base_out is not None and h is x:
            return adapted - base_out.to(adapted.dtype)
        return adapted - h.to(adapted.dtype) @ self._pristine_weight().to(adapted.dtype).mT

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        return self._shift(adapter, x, None)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """The base layer's output plus every active adapter's change of it, as the adapted weight makes it.

        The base output alone when adapters are disabled or every active adapter is merged; the sum
        is cast back to the base output's dtype.
        """
        result: Tensor = self.base_layer(x)
        if self.disable_adapters:
            return result
        adapters = self._applied(self._adapter_names())
        if not adapters:
            return result
        base_out: Tensor | None = None
        if not self.merged:
            bias = self.bias
            base_out = result if bias is None else result - bias
        out = result
        for adapter in adapters:
            out = out + self._shift(adapter, x, base_out)
        return out if out.dtype == result.dtype else out.to(result.dtype)

    @override
    def _changes(self, adapter: str, base: Mapping[str, Tensor]) -> dict[str, Tensor]:
        v, sparse = self._basis(adapter)
        norms = _parameter_at(self.frod_lambda_l, adapter)
        middle = (
            sparse.dense(_parameter_at(self.frod_lambda_s_values, adapter))
            + lucid.eye(self.in_features, dtype=norms.dtype, device=norms.device) * norms
        )
        adapted: Tensor = self.frod_U[adapter] @ middle @ v.mT
        return {_WEIGHT: adapted - base[_WEIGHT].to(adapted.dtype)}


class Linear(FrodLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` whose weight is re-expressed as :math:`U(\Sigma + S)V^\top`: :math:`y = x\,W'^\top + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`FrodLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import FrodConfig
    >>> from lucid.peft.tuners.frod import Linear
    >>> base = nn.Linear(6, 4)
    >>> layer = Linear(base, "default", FrodConfig(sparse_rate=0.5, target_modules=["x"]))
    >>> tuple(layer.frod_lambda_l["default"].shape), tuple(layer.frod_lambda_s_values["default"].shape)
    ((6,), (18,))
    >>> x = lucid.randn(3, 6)
    >>> bool(lucid.allclose(layer(x), base(x), atol=1e-5))   # S starts at zero: W' = W up to rounding
    True
    """

    base_layer: nn.Linear
