"""The LoRA layers: a frozen base module with a low-rank update beside it.

``LoraLayer`` is the family root.  It keeps what every LoRA layer keeps per
adapter (rank, alpha, scale, dropout, the factor tables), creates and
initialises adapters (``update_layer``), and merges them into the base
weights.  What differs between layer types is a handful of hooks: the
shape of the factors (``_build_factors``), the update an adapter makes
(``get_delta_weight`` and, for the layers that add it to the output,
``_update``) and which base tensors a merge writes (``_merge_targets``).
``_Residual`` is the layers that add the update to the base output
(``Linear``, the convolutions, ``Embedding``); the ones that instead shift
the base weights for the duration of its forward are in ``_weight_space``.
"""

import math
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, Final, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.parameter import Parameter
from lucid.peft._config import _check_dropout, _check_init, _check_number, _check_rank
from lucid.peft.tuners._base import _BaseTunerLayer, _check_adapter_name, _entry_parameters

_HALF: Final = (lucid.float16, lucid.bfloat16)


class _AdapterSpec(NamedTuple):
    """What a layer remembers of one adapter besides its modules: rank, alpha and scale.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes
    it: the scale is baked into the graph.
    """

    r: int
    lora_alpha: float
    scaling: float


type _Factors = tuple[nn.Module | Parameter, ...]


class _Place(NamedTuple):
    """Where and in what dtype an adapter's factors are allocated: the base tensor's, widened if half."""

    device: DeviceLike
    dtype: DTypeLike

    def linear(self, n_in: int, n_out: int) -> nn.Linear:
        """A bias-free ``nn.Linear`` here."""
        return nn.Linear(n_in, n_out, bias=False, device=self.device, dtype=self.dtype)


def _enter(
    table: nn.ModuleDict[nn.Module] | nn.ParameterDict, name: str, factor: nn.Module | Parameter
) -> None:
    """Put ``factor`` in ``table`` under ``name``: a module in a ``ModuleDict``, a parameter in a ``ParameterDict``."""
    if isinstance(factor, Parameter):
        if not isinstance(table, nn.ParameterDict):
            raise TypeError(f"a {type(table).__name__} cannot hold the parameter {name!r}")
        table[name] = factor
    elif isinstance(factor, nn.Module):
        if not isinstance(table, nn.ModuleDict):
            raise TypeError(f"a {type(table).__name__} cannot hold the module {name!r}")
        table[name] = factor
    else:
        raise TypeError(f"{type(factor).__name__} is not an adapter factor")


class LoraLayer(_BaseTunerLayer):
    r"""Adapter bookkeeping shared by the LoRA layers.

    Holds the wrapped layer as ``base_layer`` — the very module, so its
    parameters stay the same ``Parameter`` objects (an optimizer, a tied
    head and Core ML's identity-keyed weight sharing all keep seeing
    them) — and, per adapter name, the rank ``r``, ``lora_alpha``,
    ``scaling`` and the factor tables (``lora_A`` / ``lora_B`` and the
    ``lora_dropout`` modules in ``ModuleDict``\ s; ``lora_embedding_A`` /
    ``lora_embedding_B`` in ``ParameterDict``\ s for an embedding), so a
    state dict reads ``....lora_A.default.weight`` as the reference's
    does.  A layer holds any number of adapters, one entry per name in
    those tables and in ``_specs`` (rank, alpha and scale, read as the
    ``r``, ``lora_alpha`` and ``scaling`` mappings), and the forward pass
    applies the ones named in ``active_adapter`` that it holds.  Merging
    folds an update into the base weights in place; each LoRA layer type
    supplies the update (``get_delta_weight``) and the forward pass.

    What the forward pass branches on — ``active_adapter``,
    ``disable_adapters``, ``merged`` — are plain public attributes, not
    properties as in the reference: ``lucid.compile`` keys an executable
    on each module's scalar attributes, so state kept anywhere else would
    let a compiled model replay the branch of a state it has left
    (LCD-703).  They change only through ``update_layer``,
    ``set_adapter``, ``delete_adapter``, ``enable_adapters``, ``merge``
    and ``unmerge``.

    Parameters
    ----------
    base_layer : nn.Module
        The layer to adapt.
    adapter_name : str
        Name of the adapter created at once (``update_layer``).
    r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora
        The adapter's hyper-parameters, as ``update_layer`` takes them.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``base_layer`` has parameters it builds on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("lora_A", "lora_B")

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: float = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        use_dora: bool = False,
    ) -> None:
        super().__init__(base_layer)
        self._specs: dict[str, _AdapterSpec] = {}
        self.lora_dropout: nn.ModuleDict[nn.Dropout | nn.Identity] = nn.ModuleDict()
        self._setup()
        self.update_layer(
            adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora
        )

    # ── what a layer type supplies ────────────────────────────────────────

    @abstractmethod
    def _setup(self) -> None:
        """Create the empty factor tables and read the base layer's sizes the factors are shaped by."""

    @abstractmethod
    def _anchor(self) -> Tensor:
        """The base tensor whose device and dtype an adapter's factors follow."""

    def _check_rank_fits(self, r: int) -> None:
        """Raise ``ValueError`` if rank ``r`` cannot be built on this base layer."""

    def _check_dropout(self, p: float) -> None:
        """Refuse, or warn of, a dropout probability ``p`` this layer type cannot apply."""

    @abstractmethod
    def _build_factors(self, r: int, place: _Place) -> _Factors:
        """One adapter's new factors, one per table of ``adapter_layer_names``, in that order."""

    @abstractmethod
    def _deltas(self, adapter: str) -> dict[str, Tensor]:
        """The update to each base tensor a merge writes, keyed as ``_merge_targets``."""

    def _factor_pairs(self, adapter: str) -> list[tuple[Parameter, Parameter]]:
        r"""``(A, B)`` of each low-rank pair ``adapter`` holds: the update is :math:`s\,BA`.

        The tables come in ``(A, B)`` pairs, each entry one weight.
        """
        tables = self._tables()
        pairs = []
        for lora_a, lora_b in zip(tables[0::2], tables[1::2], strict=True):
            (a,) = _entry_parameters(lora_a[adapter])
            (b,) = _entry_parameters(lora_b[adapter])
            pairs.append((a, b))
        return pairs

    # ── per-adapter state ─────────────────────────────────────────────────

    @property
    def r(self) -> Mapping[str, int]:
        """Rank of each adapter, by name (read-only)."""
        return MappingProxyType({name: spec.r for name, spec in self._specs.items()})

    @property
    def lora_alpha(self) -> Mapping[str, float]:
        """:math:`\\alpha` of each adapter, by name (read-only)."""
        return MappingProxyType({name: spec.lora_alpha for name, spec in self._specs.items()})

    @property
    def scaling(self) -> Mapping[str, float]:
        """The scale :math:`s` each adapter's update is multiplied by, by name (read-only)."""
        return MappingProxyType({name: spec.scaling for name, spec in self._specs.items()})

    @property
    def weight(self) -> Parameter:
        """The base layer's weight (unchanged unless an adapter is merged into it)."""
        weight = self.get_base_layer().weight
        if isinstance(weight, Parameter):
            return weight
        raise AttributeError(f"{type(self.get_base_layer()).__name__} has no weight parameter")

    @property
    def bias(self) -> Parameter | None:
        """The base layer's bias, or ``None``."""
        bias = getattr(self.get_base_layer(), "bias", None)
        return bias if isinstance(bias, Parameter) else None

    @override
    def get_base_layer(self) -> nn.Module:
        """Return the wrapped module, whose parameters are the base model's own."""
        return self.base_layer

    def _base_path(self) -> str:
        """The dotted path from this layer to the module holding the base parameters."""
        return "base_layer"

    @override
    def __repr__(self) -> str:
        return "lora." + super().__repr__()

    @override
    def extra_repr(self) -> str:
        held = [name for name in self.active_adapters if name in self._specs]
        shown = {name: self._specs[name] for name in held}
        if isinstance(self.active_adapter, str) and held:
            spec = shown[held[0]]
            detail = f"r={spec.r}, lora_alpha={spec.lora_alpha}"
        else:
            detail = f"r={ {name: spec.r for name, spec in shown.items()} }"
        return (
            f"adapter={self.active_adapter!r}, {detail}, "
            f"merged={self.merged}, disabled={self.disable_adapters}"
        )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: bool | str,
        use_rslora: bool,
        use_dora: bool = False,
    ) -> None:
        r"""Create adapter ``adapter_name``: its factors, dropout and scale.

        The adapter lives on the base weight's device.  Its dtype is the
        base's, except that a float16 / bfloat16 base gets a float32
        adapter — the reference's default, which keeps the low-rank update
        from underflowing in training.  A layer that holds an adapter already
        keeps applying the adapters it applied: the new one is inactive until
        ``set_adapter`` names it.  A layer with none starts applying this one.

        Parameters
        ----------
        adapter_name : str
            Key of the adapter in each table.
        r : int
            Rank; positive.
        lora_alpha : float
            Numerator of the scale.
        lora_dropout : float
            Dropout on the update's input, in ``[0, 1)``.
        init_lora_weights : bool or str
            ``True``, ``False`` or ``"gaussian"`` (see ``LoraConfig``).
        use_rslora : bool
            Scale by :math:`\alpha/\sqrt r` instead of :math:`\alpha/r`.
        use_dora : bool, default=False
            DoRA; ``True`` raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            ``use_dora=True``.
        ValueError
            The layer already holds an adapter of this name, or the rank
            does not fit the base layer.
        """
        adapter_name = _check_adapter_name(adapter_name)
        r = _check_rank("r", r)
        lora_alpha = _check_number("lora_alpha", lora_alpha)
        init_lora_weights = _check_init("init_lora_weights", init_lora_weights)
        p = _check_dropout("lora_dropout", lora_dropout)
        if use_dora:
            raise NotImplementedError("use_dora=True: DoRA is not implemented in lucid.peft v1")
        if adapter_name in self._specs:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        self._check_rank_fits(r)
        self._check_dropout(p)
        anchor = self._anchor()
        dtype = lucid.float32 if anchor.dtype in _HALF else anchor.dtype
        # Built first, entered together: an allocation that fails leaves no half adapter.
        factors = self._build_factors(r, _Place(anchor.device, dtype))
        self._specs[adapter_name] = _AdapterSpec(
            r, lora_alpha, lora_alpha / (math.sqrt(r) if use_rslora else r)
        )
        self.lora_dropout[adapter_name] = nn.Dropout(p=p) if p > 0 else nn.Identity()
        for table, factor in zip(self._tables(), factors, strict=True):
            _enter(table, adapter_name, factor)
        self.reset_lora_parameters(adapter_name, init_lora_weights)
        if not self.active_adapters:
            self.active_adapter = adapter_name

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: bool | str) -> None:
        r"""Initialise adapter ``adapter_name`` the way ``init_lora_weights`` asks.

        ``True``: every :math:`A` Kaiming-uniform with ``a = sqrt(5)`` (what
        ``nn.Linear`` uses) and every :math:`B = 0`, so the update starts at
        zero.  ``"gaussian"``: :math:`A \sim \mathcal N(0, 1/r^2)`,
        :math:`B = 0`.  ``False``: leave both as the layers made them.

        Parameters
        ----------
        adapter_name : str
            The adapter to initialise.
        init_lora_weights : bool or str
            ``True``, ``False`` or ``"gaussian"``.
        """
        init_lora_weights = _check_init("init_lora_weights", init_lora_weights)
        if init_lora_weights is False:
            return
        for lora_a, lora_b in self._factor_pairs(adapter_name):
            if init_lora_weights == "gaussian":
                init.normal_(lora_a, std=1 / self._specs[adapter_name].r)
            else:
                init.kaiming_uniform_(lora_a, a=math.sqrt(5))
            init.zeros_(lora_b)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        self._specs.pop(adapter_name, None)
        # Not in ``adapter_layer_names`` — it holds no parameters to train or save.
        if adapter_name in self.lora_dropout:
            del self.lora_dropout[adapter_name]

    # ── merging ───────────────────────────────────────────────────────────

    def _shifted(self, adapter: str, sign: float) -> dict[str, Tensor]:
        """Each merge target plus ``sign`` times ``adapter``'s update, in the target's dtype.

        Summed in the wider dtype, so a bfloat16 weight is rounded once.
        """
        deltas = self._deltas(adapter)
        shifted: dict[str, Tensor] = {}
        for key, param in self._merge_targets().items():
            value = param + deltas[key] * sign
            shifted[key] = value if value.dtype == param.dtype else value.to(param.dtype)
        return shifted

    @override
    def _check_finite_merge(self, adapter: str) -> None:
        with lucid.no_grad():
            finite = all(
                bool(lucid.isfinite(value).all().item())
                for value in self._shifted(adapter, 1.0).values()
            )
        if not finite:
            raise ValueError(
                f"NaNs detected in the merged weights. The adapter {adapter} seems to be broken"
            )

    @override
    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        r"""Fold the update into the base weights, in place.

        The weights keep their identity — they are written, not replaced —
        so whatever holds them (an optimizer, a tied embedding) sees the
        merged values.  That is also why a weight shared with another
        module must not be merged into; ``LoraModel.merge_adapter`` and
        ``merge_and_unload`` refuse it, as this layer alone cannot see the
        rest of the model.

        Parameters
        ----------
        safe_merge : bool, default=False
            Check the merged weights for NaN / inf first and leave the base
            untouched if there are any.
        adapter_names : list of str, optional
            Adapters to merge; the active ones this layer holds by default.

        Raises
        ------
        ValueError
            An adapter name this layer does not hold, or ``safe_merge``
            found a non-finite merged weight.
        RuntimeError
            The adapter is disabled.
        NotImplementedError
            This layer's update cannot be written into its base weights.
        """
        pending = self._pending_merges(adapter_names)
        if not pending:
            warnings.warn(f"Already merged: {self.merged_adapters}. Nothing to do.", stacklevel=2)
        if safe_merge:
            for adapter in pending:
                self._check_finite_merge(adapter)
        with lucid.no_grad():
            targets = self._merge_targets()
            for adapter in pending:
                for key, value in self._shifted(adapter, 1.0).items():
                    targets[key].copy_(value)
                self.merged_adapters.append(adapter)
                self.merged = True

    @override
    def unmerge(self) -> None:
        """Take every merged update back out of the base weights, in place.

        The result equals the original weights up to the rounding of the
        round trip.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.", stacklevel=2)
            return
        with lucid.no_grad():
            targets = self._merge_targets()
            while self.merged_adapters:
                adapter = self.merged_adapters.pop()
                for key, value in self._shifted(adapter, -1.0).items():
                    targets[key].copy_(value)
            self.merged = False


class _Residual(LoraLayer):
    r"""A LoRA layer whose update is added to the base layer's output: :math:`y = f(x) + \sum_a u_a(x)`.

    A subclass supplies the update an adapter makes from the layer's input
    (``_update``) and the same update as a change of the base weight
    (``get_delta_weight``), which is what merging writes.
    """

    @abstractmethod
    def _update(self, adapter: str, x: Tensor) -> Tensor:
        """What ``adapter`` adds to the base output for the input ``x``."""

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        return {"base_layer.weight": self.weight}

    @override
    def _deltas(self, adapter: str) -> dict[str, Tensor]:
        return {"base_layer.weight": self.get_delta_weight(adapter)}

    @override
    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.base_layer(x)
        if self.disable_adapters or self.merged:
            return result
        out = result
        for adapter in self.active_adapters:
            if adapter not in self._specs:
                continue  # an adapter aimed at other layers: this one runs its base for it
            out = out + self._update(adapter, x)
        return out if out.dtype == result.dtype else out.to(result.dtype)


class _Factored[Factor: nn.Module](_Residual):
    r"""A residual layer whose factors are two modules in a row: :math:`u(x) = s\,B(A(\mathrm{drop}(x)))`.

    ``lora_A`` and ``lora_B`` hold one ``Factor`` per adapter (an ``nn.Linear``,
    a convolution); a subclass builds them and says how they merge.
    """

    lora_A: nn.ModuleDict[Factor]
    lora_B: nn.ModuleDict[Factor]

    @override
    def _update(self, adapter: str, x: Tensor) -> Tensor:
        lora_a = self.lora_A[adapter]
        dtype = _entry_parameters(lora_a)[0].dtype
        dropped = self.lora_dropout[adapter](x if x.dtype == dtype else x.to(dtype))
        update: Tensor = self.lora_B[adapter](lora_a(dropped))
        return update * self._specs[adapter].scaling


class Linear(_Factored[nn.Linear], base_type=nn.Linear):
    r"""An ``nn.Linear`` with a LoRA update: :math:`y = W x + b + s\,B A\,\mathrm{drop}(x)`.

    The base layer runs as it is and stays frozen; the update is computed
    in the adapter's dtype and added in the wider of the two dtypes before
    the result is cast back to the base output's dtype.  ``merge`` folds
    :math:`s B A` into :math:`W` in place, so a merged layer costs what
    the base costs; ``unmerge`` takes it out again.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in ``lora_A`` / ``lora_B``.
    r : int, default=0
        Rank.  The default is the reference's and is refused: pass a
        positive rank.
    lora_alpha : float, default=1
        Numerator of the scale.
    lora_dropout : float, default=0.0
        Dropout on the update's input.
    fan_in_fan_out : bool, default=False
        Must be ``False``: ``nn.Linear`` stores its weight ``(out, in)``.
    is_target_conv_1d_layer : bool, default=False
        Must be ``False``; kept so positional calls line up with the
        reference, whose fused-QKV ``Conv1D`` base v1 does not have.
    init_lora_weights : bool or str, default=True
        See ``LoraConfig``.
    use_rslora : bool, default=False
        Scale by :math:`\alpha/\sqrt r`.
    use_dora : bool, default=False
        DoRA; ``True`` raises.

    Raises
    ------
    ValueError
        ``r`` is not positive, or ``fan_in_fan_out`` or
        ``is_target_conv_1d_layer`` is ``True``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.lora import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", r=2, lora_alpha=4)
    >>> layer.base_layer is base, layer.scaling["default"]
    (True, 2.0)
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # B starts at zero
    True
    """

    base_layer: nn.Linear

    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        r: int = 0,
        lora_alpha: float = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        use_dora: bool = False,
    ) -> None:
        if fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); nn.Linear stores (out, in)"
            )
        if is_target_conv_1d_layer:
            raise ValueError(
                "is_target_conv_1d_layer=True describes a Conv1D base; this layer wraps nn.Linear"
            )
        super().__init__(
            base_layer,
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            use_rslora,
            use_dora,
        )

    @override
    def _setup(self) -> None:
        self.in_features = self.base_layer.in_features
        self.out_features = self.base_layer.out_features
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

    @override
    def _anchor(self) -> Tensor:
        return self.base_layer.weight

    @override
    def _build_factors(self, r: int, place: _Place) -> _Factors:
        return (place.linear(self.in_features, r), place.linear(r, self.out_features))

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`s\,B A`, the update ``adapter`` adds to the weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(out_features, in_features)``, in the adapter's dtype.
        """
        scaling = self._specs[adapter].scaling
        return (self.lora_B[adapter].weight @ self.lora_A[adapter].weight) * scaling
